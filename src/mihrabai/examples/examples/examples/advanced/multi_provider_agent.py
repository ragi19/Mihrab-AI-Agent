"""
Advanced example demonstrating using multiple providers simultaneously
"""

import asyncio
import os
from typing import Dict, List

from mihrabai.core.agent import Agent, SimpleAgent
from mihrabai.core.message import Message, MessageRole
from mihrabai.models.providers.anthropic import AnthropicProvider
from mihrabai.models.providers.groq import GroqProvider
from mihrabai.models.providers.openai import OpenAIProvider
from mihrabai.runtime.context import RuntimeContext
from mihrabai.runtime.runner import AgentRunner
from mihrabai.utils.async_utils import gather_with_concurrency


class MultiProviderAgent(Agent):
    def __init__(self, providers: Dict[str, Agent]):
        self.providers = providers
        # Use first provider's model as base model
        super().__init__(model=next(iter(providers.values())).model)

    async def process_message(self, message: Message) -> Message:
        # Get responses from all providers
        tasks = [agent.process_message(message) for agent in self.providers.values()]

        # Gather responses with concurrency limit
        responses = await gather_with_concurrency(
            limit=3, tasks=tasks  # Limit concurrent API calls
        )

        # Combine responses into a summary
        combined_content = "Responses from different providers:\n\n"
        for provider_name, response in zip(self.providers.keys(), responses):
            combined_content += f"{provider_name.upper()}:\n{response.content}\n\n"

        return Message(role=MessageRole.ASSISTANT, content=combined_content)


async def main():
    # Initialize providers
    providers = {}

    if api_key := os.getenv("OPENAI_API_KEY"):
        provider = OpenAIProvider(api_key=api_key)
        model = await provider.create_model("gpt-3.5-turbo")
        providers["openai"] = SimpleAgent(model=model)

    if api_key := os.getenv("ANTHROPIC_API_KEY"):
        provider = AnthropicProvider(api_key=api_key)
        model = await provider.create_model("claude-3-opus-20240229")
        providers["anthropic"] = SimpleAgent(model=model)

    if api_key := os.getenv("GROQ_API_KEY"):
        provider = GroqProvider(api_key=api_key)
        model = await provider.create_model("llama2-70b-4096")
        providers["groq"] = SimpleAgent(model=model)

    if len(providers) < 2:
        raise ValueError("At least two provider API keys are required")

    # Create multi-provider agent
    agent = MultiProviderAgent(providers=providers)

    # Create runner with context
    context = RuntimeContext()
    runner = AgentRunner(agent=agent, context=context)

    # Run example conversation getting responses from all providers
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful AI assistant."),
        Message(role=MessageRole.USER, content="What are the three laws of robotics?"),
        Message(
            role=MessageRole.USER,
            content="How might these laws evolve with modern AI development?",
        ),
    ]

    responses = await runner.run_conversation(messages)
    for response in responses:
        print(response.content)
        print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())
