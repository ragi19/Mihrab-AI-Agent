"""
Basic example of using a chat agent with different providers
"""

import asyncio
import os

from dotenv import load_dotenv

from llm_agents import Message, MessageRole, create_agent

# Load environment variables from .env file
load_dotenv()


async def run_basic_chat(provider_name, model_name, user_message):
    """Run a basic chat interaction with the specified provider and model"""
    print(f"\n=== Running chat with {provider_name.upper()} ({model_name}) ===")

    # Create an agent with the specified provider and model
    agent = await create_agent(
        provider_name=provider_name,
        model_name=model_name,
        system_message="You are a helpful AI assistant that provides concise and accurate responses.",
    )

    # Create a user message
    message = Message(role=MessageRole.USER, content=user_message)

    # Process the message and get a response
    response = await agent.process_message(message)

    # Print the response
    print(f"User: {user_message}")
    print(f"Assistant: {response.content}")

    return response


async def main():
    """Run examples with different providers"""
    # Example user message
    user_message = "Explain the concept of machine learning in simple terms."

    # Run with different providers based on available API keys
    if os.getenv("OPENAI_API_KEY"):
        await run_basic_chat("openai", "gpt-3.5-turbo", user_message)

    if os.getenv("ANTHROPIC_API_KEY"):
        await run_basic_chat("anthropic", "claude-instant-1", user_message)

    if os.getenv("GROQ_API_KEY"):
        await run_basic_chat("groq", "llama3-8b-8192", user_message)

    # If no API keys are available, use a fallback message
    if not any(
        [
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("GROQ_API_KEY"),
        ]
    ):
        print(
            "\nNo API keys found. Please set at least one of the following environment variables:"
        )
        print("- OPENAI_API_KEY")
        print("- ANTHROPIC_API_KEY")
        print("- GROQ_API_KEY")


if __name__ == "__main__":
    asyncio.run(main())
