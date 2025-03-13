"""
Example implementation of a custom provider
"""

import asyncio
from typing import List, Optional, Any
from llm_agents.core.message import Message, MessageRole
from llm_agents.core.agent import Agent
from llm_agents.models.base import BaseModel
from llm_agents.runtime.runner import AgentRunner
from llm_agents.core.types import ModelParameters


# Example custom API client
class ExampleAPIClient:
    async def generate(self, messages: List[dict], **kwargs) -> str:
        # Simulate API call
        await asyncio.sleep(0.5)
        return "Response from custom model"

    def count_tokens(self, text: str) -> int:
        # Simple word-based token counting
        return len(text.split())


# Custom provider implementation
class ExampleProvider:
    def __init__(self, api_key: str):
        self.client = ExampleAPIClient()

    async def create_model(
        self, model_name: str, parameters: Optional[ModelParameters] = None
    ) -> BaseModel:
        return ExampleModel(
            model_name=model_name, client=self.client, parameters=parameters
        )


# Custom model implementation
class ExampleModel(BaseModel):
    def __init__(
        self, model_name: str, client: Any, parameters: Optional[ModelParameters] = None
    ):
        super().__init__(model_name, parameters)
        self.client = client

    async def generate_response(self, messages: List[Message]) -> Message:
        response = await self.client.generate(
            messages=[msg.to_dict() for msg in messages], **self.parameters
        )
        return Message(role=MessageRole.ASSISTANT, content=response)

    async def count_tokens(self, text: str) -> int:
        return self.client.count_tokens(text)


# Example usage
class SimpleAgent(Agent):
    async def process_message(self, message: Message) -> Message:
        return await self.model.generate_response(self.conversation_history + [message])


async def main():
    # Initialize custom provider
    provider = ExampleProvider(api_key="example-key")

    # Create model instance
    model = await provider.create_model(
        model_name="example-model", parameters={"temperature": 0.7}
    )

    # Create agent
    agent = SimpleAgent(model=model)

    # Create runner
    runner = AgentRunner(agent=agent)

    # Run a conversation
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Hello from custom model!"),
    ]

    responses = await runner.run_conversation(messages)
    for response in responses:
        print(f"Assistant: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
