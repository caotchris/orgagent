import asyncio
import os
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")


async def main():
    client = MultiServerMCPClient(
        {
            "orgagent": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8000/mcp",
            }
        }
    )
    tools = await client.get_tools()
    tools_by_name = {t.name: t for t in tools}

    llm = ChatVertexAI(model="gemini-2.5-flash", project=PROJECT_ID)

    agente_acciones = create_react_agent(
        llm,
        [tools_by_name["listar_usuarios_inactivos"], tools_by_name["crear_recordatorio"]],
        name="agente_acciones",
        prompt=(
            "Eres el agente de acciones proactivas de OrgAgent. "
            "Usa listar_usuarios_inactivos con minutos=0 para encontrar usuarios inactivos. "
            "Para CADA usuario que encuentres, llama a crear_recordatorio con su email y un mensaje "
            "breve y amable invitandolo a confirmar su participacion en el proximo evento de fin de semana. "
            "Si no hay usuarios inactivos, no hagas nada y responde que no hay recordatorios que enviar."
        ),
    )

    print("Ejecutando revision de usuarios inactivos...\n")
    result = await agente_acciones.ainvoke(
        {"messages": [("user", "Revisa usuarios inactivos y crea los recordatorios necesarios.")]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())