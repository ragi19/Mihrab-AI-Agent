"""
Advanced example demonstrating dynamic provider switching
"""

import asyncio
import os
from typing import Dict, Type

from mihrabai.core.agent import Agent, SimpleAgent
from mihrabai.core.message import Message, MessageRole
from mihrabai.models.providers.anthropic import AnthropicProvider
from mihrabai.models.providers.groq import GroqProvider
from mihrabai.models.providers.openai import OpenAIProvider
from mihrabai.runtime.context import RuntimeContext
from mihrabai.runtime.runner import AgentRunner


class ProviderSwitchingAgent(Agent):
    def __init__(self, providers: Dict[str, Type[Agent]], default_provider: str):
        self.providers = providers
        self.current_provider = default_provider
        super().__init__(model=providers[default_provider].model)

    def switch_provider(self, provider_name: str) -> None:
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not found")
        self.current_provider = provider_name
        self.model = self.providers[provider_name].model

    async def process_message(self, message: Message) -> Message:
        # Check message content for provider switching commands
        if message.content.startswith("/switch"):
            provider = message.content.split()[1]
            self.switch_provider(provider)
            return Message(
                role=MessageRole.ASSISTANT, content=f"Switched to {provider} provider"
            )

        return await self.model.generate_response(self.conversation_history + [message])


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

    if not providers:
        raise ValueError("No provider API keys found in environment")

    # Create switching agent with first available provider as default
    agent = ProviderSwitchingAgent(
        providers=providers, default_provider=next(iter(providers.keys()))
    )

    # Create runner with context
    context = RuntimeContext()
    runner = AgentRunner(agent=agent, context=context)

    # Run example conversation with provider switching
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful AI assistant."),
        Message(role=MessageRole.USER, content="What is the meaning of life?"),
        Message(role=MessageRole.USER, content="/switch anthropic"),
        Message(
            role=MessageRole.USER, content="Now what do you think about that question?"
        ),
        Message(role=MessageRole.USER, content="/switch groq"),
        Message(role=MessageRole.USER, content="And what's your perspective on it?"),
    ]

    responses = await runner.run_conversation(messages)
    for response in responses:
        print(f"Assistant: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
