import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_vertexai import ChatVertexAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph_supervisor import create_supervisor

load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp")
DB_SOCKET_DIR = os.getenv("DB_SOCKET_DIR")
DB_INSTANCE = os.getenv("DB_INSTANCE")
DB_PASSWORD = os.getenv("DB_PASSWORD")

if DB_SOCKET_DIR:
    DB_URI = f"postgresql://postgres:{DB_PASSWORD}@/orgagent?host={DB_SOCKET_DIR}/{DB_INSTANCE}"
else:
    DB_URI = f"postgresql://postgres:{DB_PASSWORD}@127.0.0.1:5432/orgagent?sslmode=disable"

HANDOFF_PHRASES = ("Transferring back", "Successfully transferred")


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content else ""


state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = MultiServerMCPClient({"orgagent": {"transport": "streamable_http", "url": MCP_URL}})
    tools = await client.get_tools()
    tools_by_name = {t.name: t for t in tools}

    llm = ChatVertexAI(model="gemini-2.5-flash", project=PROJECT_ID)

    agente_conocimiento = create_react_agent(
        llm, [tools_by_name["buscar_documentos"]], name="agente_conocimiento",
        prompt=(
            "Eres el agente de conocimiento. Para CUALQUIER pregunta que recibas, "
            "SIEMPRE debes llamar primero a la herramienta buscar_documentos antes de responder, "
            "sin excepcion, aunque creas saber la respuesta. Nunca respondas sin haberla llamado. "
            "Responde usando solo la informacion que te devuelva la herramienta."
        ),
    )
    agente_datos = create_react_agent(
        llm, [tools_by_name["consultar_usuario"]], name="agente_datos",
        prompt=(
            "Eres el agente de datos. Para CUALQUIER pregunta sobre un usuario, "
            "SIEMPRE debes llamar primero a la herramienta consultar_usuario antes de responder, "
            "sin excepcion. Nunca respondas sin haberla llamado."
        ),
    )
    agente_acciones = create_react_agent(
        llm, [tools_by_name["listar_usuarios_inactivos"], tools_by_name["crear_recordatorio"]], name="agente_acciones",
        prompt=(
            "Eres el agente de acciones proactivas de OrgAgent. "
            "Usa listar_usuarios_inactivos con minutos=0 para encontrar usuarios inactivos. "
            "Para CADA usuario que encuentres, llama a crear_recordatorio con su email y un mensaje "
            "breve y amable invitandolo a confirmar su participacion en el proximo evento de fin de semana. "
            "Si no hay usuarios inactivos, no hagas nada y responde que no hay recordatorios que enviar."
        ),
    )

    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        supervisor = create_supervisor(
            [agente_conocimiento, agente_datos], model=llm,
            prompt=(
                "Eres el supervisor de OrgAgent. Coordinas dos agentes: "
                "agente_conocimiento (preguntas sobre documentos/FAQs de la organizacion) "
                "y agente_datos (consultas sobre usuarios/voluntarios). "
                "Decide a cual delegar segun la pregunta del usuario."
            ),
        ).compile(checkpointer=checkpointer)

        state["supervisor"] = supervisor
        state["agente_acciones"] = agente_acciones
        yield


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


@app.post("/chat")
async def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = await state["supervisor"].ainvoke({"messages": [("user", req.message)]}, config=config)
    respuesta = None
    for m in reversed(result["messages"]):
        if type(m).__name__ == "ToolMessage":
            continue
        text = extract_text(getattr(m, "content", None))
        if not text or text.strip().startswith(HANDOFF_PHRASES):
            continue
        respuesta = text
        break
    return {"respuesta": respuesta}


@app.post("/revisar-inactivos")
async def revisar_inactivos():
    result = await state["agente_acciones"].ainvoke(
        {"messages": [("user", "Revisa usuarios inactivos y crea los recordatorios necesarios.")]}
    )
    return {"resultado": extract_text(result["messages"][-1].content)}


@app.get("/")
async def root():
    return {"status": "ok"}