"""
Example of using streaming capabilities with different providers
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

from llm_agents import Message, MessageRole, create_agent
from llm_agents.models.base import ModelCapability

# Load environment variables from .env file
load_dotenv()


async def run_streaming_chat(provider_name, model_name, user_message):
    """Run a streaming chat interaction with the specified provider and model"""
    print(
        f"\n=== Running streaming chat with {provider_name.upper()} ({model_name}) ==="
    )

    # Create an agent with the specified provider and model
    agent = await create_agent(
        provider_name=provider_name,
        model_name=model_name,
        system_message="You are a helpful AI assistant that provides detailed and informative responses.",
        required_capabilities={
            ModelCapability.STREAMING
        },  # Ensure model supports streaming
    )

    # Create a user message
    message = Message(role=MessageRole.USER, content=user_message)

    # Process the message and stream the response
    print(f"User: {user_message}")
    print("Assistant: ", end="", flush=True)

    async for chunk in agent.stream_message(message):
        # Print each chunk as it arrives
        print(chunk.content, end="", flush=True)

    print("\n")  # Add newline after streaming completes


async def main():
    """Run streaming examples with different providers"""
    # Example user message that requires a longer response
    user_message = "Write a short story about an AI that becomes sentient."

    # Run with different providers based on available API keys
    if os.getenv("OPENAI_API_KEY"):
        await run_streaming_chat("openai", "gpt-3.5-turbo", user_message)

    if os.getenv("ANTHROPIC_API_KEY"):
        await run_streaming_chat("anthropic", "claude-instant-1", user_message)

    if os.getenv("GROQ_API_KEY"):
        await run_streaming_chat("groq", "llama3-8b-8192", user_message)

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
