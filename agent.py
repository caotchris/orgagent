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
                "url": "https://orgagent-mcp-514997940160.us-central1.run.app/mcp",
            }
        }
    )
    tools = await client.get_tools()

    llm = ChatVertexAI(model="gemini-2.5-flash", project=PROJECT_ID)
    agent = create_react_agent(llm, tools)

    print("Agente listo. Escribe 'salir' para terminar.\n")
    while True:
        pregunta = input("Tu: ")
        if pregunta.lower() == "salir":
            break
        result = await agent.ainvoke({"messages": [("user", pregunta)]})
        respuesta = result["messages"][-1].content
        print(f"Agente: {respuesta}\n")


if __name__ == "__main__":
    asyncio.run(main())