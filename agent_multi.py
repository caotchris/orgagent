import asyncio
import os
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph_supervisor import create_supervisor

load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
DB_URI = f"postgresql://postgres:{os.getenv('DB_PASSWORD')}@127.0.0.1:5432/orgagent?sslmode=disable"

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


async def main():
    client = MultiServerMCPClient(
        {
            "orgagent": {
                "transport": "streamable_http",
                "url": "https://orgagent-mcp-514997940160.us-central1.run.app/mcp",
            }
        }
    )
    tools = await client.get_tools()
    tools_by_name = {t.name: t for t in tools}

    llm = ChatVertexAI(model="gemini-2.5-flash", project=PROJECT_ID)

    agente_conocimiento = create_react_agent(
        llm,
        [tools_by_name["buscar_documentos"]],
        name="agente_conocimiento",
        prompt=(
            "Eres el agente de conocimiento. Para CUALQUIER pregunta que recibas, "
            "SIEMPRE debes llamar primero a la herramienta buscar_documentos antes de responder, "
            "sin excepcion, aunque creas saber la respuesta. Nunca respondas sin haberla llamado. "
            "Responde usando solo la informacion que te devuelva la herramienta."
        ),
    )

    agente_datos = create_react_agent(
        llm,
        [tools_by_name["consultar_usuario"]],
        name="agente_datos",
        prompt=(
            "Eres el agente de datos. Para CUALQUIER pregunta sobre un usuario, "
            "SIEMPRE debes llamar primero a la herramienta consultar_usuario antes de responder, "
            "sin excepcion. Nunca respondas sin haberla llamado."
        ),
    )

    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()

        supervisor = create_supervisor(
            [agente_conocimiento, agente_datos],
            model=llm,
            prompt=(
                "Eres el supervisor de OrgAgent. Coordinas dos agentes: "
                "agente_conocimiento (preguntas sobre documentos/FAQs de la organizacion) "
                "y agente_datos (consultas sobre usuarios/voluntarios). "
                "Decide a cual delegar segun la pregunta del usuario."
            ),
        ).compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "demo-cloudrun"}}

        print("Supervisor multi-agente listo (memoria persistente). Escribe 'salir' para terminar.\n")
        while True:
            pregunta = input("Tu: ")
            if pregunta.lower() == "salir":
                break

            result = await supervisor.ainvoke({"messages": [("user", pregunta)]}, config=config)

            respuesta = None
            for m in reversed(result["messages"]):
                if type(m).__name__ == "ToolMessage":
                    continue
                text = extract_text(getattr(m, "content", None))
                if not text or text.strip().startswith(HANDOFF_PHRASES):
                    continue
                respuesta = text
                break

            print(f"Agente: {respuesta}\n")


if __name__ == "__main__":
    import selectors
    asyncio.run(main(), loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()))