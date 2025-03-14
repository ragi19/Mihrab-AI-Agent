"""
Advanced sample usage of Groq provider demonstrating streaming responses
with enhanced conversation memory and parameter control
"""

import asyncio
import os

from llm_agents.core.message import Message, MessageRole
from llm_agents.models.providers.groq import GroqProvider


async def main():
    # Get API key from environment variable
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Please set GROQ_API_KEY environment variable")
        return

    try:
        # Initialize provider
        provider = GroqProvider(api_key=api_key)

        # Create model with advanced parameters
        model = await provider.create_model(
            "llama-3.3-70b-versatile", temperature=0.7, top_p=0.95, max_tokens=1024
        )

        # Initialize conversation with system prompt
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="You are a quantum computing expert. Provide detailed, technical responses.",
            )
        ]

        print("\n=== Advanced Groq Streaming Conversation ===")
        print("Type 'quit' to exit the conversation")

        while True:
            # Get user input
            user_input = input("\nYour question: ")
            if user_input.lower() == "quit":
                break

            # Add user message to conversation
            messages.append(Message(role=MessageRole.USER, content=user_input))

            # Stream the response
            print("\nAssistant: ", end="", flush=True)
            full_response = ""
            async for chunk in model.stream_response(messages):
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
            print()

            # Add the assistant's response to the conversation history
            messages.append(Message(role=MessageRole.ASSISTANT, content=full_response))

            # Display token usage information
            total_tokens = await model.count_tokens(
                "".join(m.content for m in messages)
            )
            print(f"\n(Current conversation using approximately {total_tokens} tokens)")

            # Simple memory management - keep conversation within reasonable size
            if len(messages) > 10:  # If conversation gets too long
                # Keep system message and most recent exchanges
                messages = [messages[0]] + messages[-9:]
                print("(Trimmed conversation history to maintain context window)")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
