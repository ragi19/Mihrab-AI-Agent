"""
Anthropic model implementations
"""

from typing import Dict, List, Optional

from anthropic import AsyncAnthropic

from ....core.message import Message, MessageRole
from ....core.types import ModelParameters
from ...base import BaseModel


class ClaudeModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        client: AsyncAnthropic,
        parameters: Optional[ModelParameters] = None,
    ):
        super().__init__(model_name, parameters)
        self.client = client

    async def generate_response(self, messages: List[Message]) -> Message:
        """Generate a response using the Anthropic API"""
        # Convert messages to Anthropic format
        prompt = self._convert_messages_to_prompt(messages)

        response = await self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **self.parameters,
        )

        return Message(role=MessageRole.ASSISTANT, content=response.content[0].text)

    async def _generate(self, messages: List[Message]) -> Message:
        """Internal method for model-specific response generation"""
        # Convert messages to Anthropic format
        prompt = self._convert_messages_to_prompt(messages)

        response = await self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **self.parameters,
        )

        return Message(role=MessageRole.ASSISTANT, content=response.content[0].text)

    async def _prepare_messages(self, messages: List[Message]) -> List[Dict]:
        """Convert messages to Anthropic format"""
        return [{"role": "user", "content": self._convert_messages_to_prompt(messages)}]

    async def _make_api_call(self, messages: List[Dict]) -> Message:
        """Make API call to Anthropic"""
        response = await self.client.messages.create(
            model=self.model_name, messages=messages, **self.parameters
        )
        return Message(role=MessageRole.ASSISTANT, content=response.content[0].text)

    async def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        import tiktoken

        encoding = tiktoken.encoding_for_model(
            "gpt-4"
        )  # Use GPT-4 encoding as a reasonable approximation
        return len(encoding.encode(text))

    def _convert_messages_to_prompt(self, messages: List[Message]) -> str:
        """Convert message list to Anthropic's expected format"""
        converted = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                converted.append(f"System: {msg.content}")
            elif msg.role == MessageRole.USER:
                converted.append(f"Human: {msg.content}")
            elif msg.role == MessageRole.ASSISTANT:
                converted.append(f"Assistant: {msg.content}")
        return "\n\n".join(converted)
