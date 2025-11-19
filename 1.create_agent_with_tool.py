# Example taken from autogen website
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()
# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
)

# Define a simple function tool that the agent can use.
# For this example, we use a fake weather tool for demonstration purposes.
# async def marks the function as asynchronous (a coroutine).
#
# Calling it does not run it immediately; it returns a coroutine object that must be awaited.
#
# Even if the body is trivial now, making a tool async is future-proof: you can later do non-blocking I/O (HTTP requests, DB calls) without changing the agent code.
#
# In AutoGen, tools can be async so the agent can call multiple tools without blocking the event loop.

async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 73 degrees and Sunny."


# Define an AssistantAgent with the model, tool, system message, and reflection enabled.
# The system message instructs the agent via natural language.
agent = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    tools=[get_weather],
    system_message="You are a helpful assistant.",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)


# Run the agent and stream the messages to the console.
# await says: yield control to the event loop until this coroutine completes.
#
# Two awaited things here:
#
# Console(agent.run_stream(...)) — run_stream returns an async stream of model tokens/messages; Console(...) consumes/prints them as they arrive, so you see streaming output in real time.
#
# model_client.close() — shuts down the OpenAI client cleanly; it’s async, so we await it.
#
# If you forget await, nothing runs (you’ll just hold a coroutine object), or you’ll get warnings like “coroutine was never awaited”.
async def main() -> None:
    await Console(agent.run_stream(task="What is the weather in New York?"))
    # Close the connection to the model client.
    await model_client.close()


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
#await main()
asyncio.run(main())