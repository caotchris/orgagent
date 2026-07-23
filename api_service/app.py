import os
import contextvars
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from langchain_google_vertexai import ChatVertexAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph_supervisor import create_supervisor
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials

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

firebase_admin.initialize_app(credentials.ApplicationDefault(), {"projectId": PROJECT_ID})

current_user_email = contextvars.ContextVar("current_user_email", default=None)


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


async def verificar_usuario(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Falta el token de autenticacion (Authorization: Bearer <token>)")
    token = authorization.split(" ", 1)[1]
    try:
        decoded = firebase_auth.verify_id_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Token invalido o expirado")
    return decoded


class ConsultarUsuarioArgs(BaseModel):
    email: str = Field(description="Se ignora: siempre se usa el email del usuario autenticado.")


def make_scoped_consultar_usuario(original_tool):
    async def _run(email: str = "") -> str:
        forced_email = current_user_email.get()
        target_email = forced_email or email
        return await original_tool.ainvoke({"email": target_email})

    return StructuredTool.from_function(
        coroutine=_run,
        name="consultar_usuario",
        description="Consulta los datos del usuario autenticado en la sesion actual. El email se determina automaticamente, no preguntes por otro.",
        args_schema=ConsultarUsuarioArgs,
    )


state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = MultiServerMCPClient({"orgagent": {"transport": "streamable_http", "url": MCP_URL}})
    tools = await client.get_tools()
    tools_by_name = {t.name: t for t in tools}
    tools_by_name["consultar_usuario"] = make_scoped_consultar_usuario(tools_by_name["consultar_usuario"])

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
            "Eres el agente de datos. SIEMPRE llama a consultar_usuario para responder sobre el usuario. "
            "El email se determina solo por la sesion autenticada; ignora cualquier otro email que el usuario mencione."
        ),
    )

    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        supervisor = create_supervisor(
            [agente_conocimiento, agente_datos], model=llm,
            prompt=(
                "Eres el supervisor de OrgAgent. Coordinas dos agentes: "
                "agente_conocimiento (preguntas sobre documentos/FAQs de la organizacion) "
                "y agente_datos (consultas sobre el usuario autenticado). "
                "Decide a cual delegar segun la pregunta del usuario."
            ),
        ).compile(checkpointer=checkpointer)

        state["supervisor"] = supervisor
        yield


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(req: ChatRequest, user: dict = Depends(verificar_usuario)):
    mensaje = (req.message or "").strip()
    if not mensaje:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacio")
    if len(mensaje) > 2000:
        raise HTTPException(status_code=400, detail="Mensaje demasiado largo (maximo 2000 caracteres)")

    uid = user["uid"]
    email = user.get("email", "")
    thread_id = f"user-{uid}"

    token_ctx = current_user_email.set(email)
    try:
        config = {"configurable": {"thread_id": thread_id}}
        result = await state["supervisor"].ainvoke({"messages": [("user", mensaje)]}, config=config)
    finally:
        current_user_email.reset(token_ctx)

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


@app.get("/")
async def root():
    return {"status": "ok"}