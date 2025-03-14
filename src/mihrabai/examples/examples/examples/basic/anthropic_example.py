"""
Basic example using Anthropic models
"""

import asyncio
import os

from mihrabai.core.agent import Agent
from mihrabai.core.message import Message, MessageRole
from mihrabai.models.providers.anthropic import AnthropicProvider
from mihrabai.runtime.runner import AgentRunner


class SimpleAgent(Agent):
    async def process_message(self, message: Message) -> Message:
        return await self.model.generate_response(self.conversation_history + [message])


async def main():
    # Initialize Anthropic provider with API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    provider = AnthropicProvider(api_key=api_key)

    # Create model instance
    model = await provider.create_model(
        model_name="claude-3-opus-20240229", parameters={"temperature": 0.7}
    )

    # Create agent
    agent = SimpleAgent(model=model)

    # Create runner
    runner = AgentRunner(agent=agent)

    # Run a conversation
    messages = [
        Message(
            role=MessageRole.SYSTEM, content="You are Claude, a helpful AI assistant."
        ),
        Message(
            role=MessageRole.USER,
            content="What are the main challenges in artificial intelligence today?",
        ),
    ]

    responses = await runner.run_conversation(messages)
    for response in responses:
        print(f"Assistant: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
