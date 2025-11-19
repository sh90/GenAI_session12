import asyncio

from autogen_agentchat.ui import Console
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Assistant Agent
# AssistantAgent is a built-in agent that uses a language model and has the ability to use tools.

load_dotenv()

async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    agent = AssistantAgent(
        "assistant",
        model_client=model_client)

    print(await agent.run(task="Say 'Hello World!'"))
    #print(agent.run_stream(task="Say 'Hello World!'"))
    await Console(agent.run_stream(task="What is the weather in New York?"))
    print(await Console(agent.run_stream(task="What is the weather in New York?")))
    await model_client.close()

asyncio.run(main())