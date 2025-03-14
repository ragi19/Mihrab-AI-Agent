"""
Example demonstrating the use of the agent factory
"""

import asyncio
import os

from mihrabai.config import config
from mihrabai.core.message import Message, MessageRole
from mihrabai.factory import create_agent
from mihrabai.runtime.runner import AgentRunner


async def main():
    # Configure providers
    config.set_provider_config(
        "openai",
        {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "default_model": "gpt-3.5-turbo",
            "default_parameters": {"temperature": 0.7, "max_tokens": 1000},
        },
    )

    # Create a chat agent using the factory
    agent = await create_agent(
        provider_name="openai",
        system_message="You are a knowledgeable assistant specialized in technology.",
        max_history_tokens=2000,
    )

    # Create runner
    runner = AgentRunner(agent=agent)

    # Run a conversation
    messages = [
        Message(
            role=MessageRole.USER,
            content="What are the key differences between REST and GraphQL?",
        )
    ]

    responses = await runner.run_conversation(messages)
    for response in responses:
        print(f"Assistant: {response.content}")

    # Create another agent with different configuration
    agent2 = await create_agent(
        agent_type="basic",  # Create a basic agent without chat features
        provider_name="openai",
        model_name="gpt-4",  # Specify a different model
        model_parameters={
            "temperature": 0.9,  # Override default temperature
            "max_tokens": 500,
        },
    )

    runner2 = AgentRunner(agent=agent2)
    responses = await runner2.run_conversation(messages)
    for response in responses:
        print(f"\nGPT-4 Assistant: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
