"""
Basic example using Groq models
"""

import asyncio
import os

from mihrabai.core.agent import Agent
from mihrabai.core.message import Message, MessageRole
from mihrabai.models.providers.groq import GroqProvider
from mihrabai.runtime.runner import AgentRunner


class SimpleAgent(Agent):
    async def process_message(self, message: Message) -> Message:
        return await self.model.generate_response(self.conversation_history + [message])


async def main():
    # Initialize Groq provider with API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")

    provider = GroqProvider(api_key=api_key)

    # Create model instance
    model = await provider.create_model(
        model_name="llama2-70b-4096", parameters={"temperature": 0.7}
    )

    # Create agent
    agent = SimpleAgent(model=model)

    # Create runner
    runner = AgentRunner(agent=agent)

    # Run a conversation
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(
            role=MessageRole.USER, content="Explain quantum computing in simple terms."
        ),
    ]

    responses = await runner.run_conversation(messages)
    for response in responses:
        print(f"Assistant: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
