"""
OpenAI model implementations
"""

from typing import Any, AsyncIterator, Dict, List, Optional, Set

from openai import AsyncOpenAI

from ....core.message import Message, MessageRole
from ....core.types import ModelParameters
from ...base import BaseModel, ModelCapability


class GPTModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        client: AsyncOpenAI,
        parameters: Optional[ModelParameters] = None,
    ):
        super().__init__(model_name, parameters)
        self.client = client
        self._capabilities = {
            ModelCapability.CHAT,
            ModelCapability.STREAM,
            ModelCapability.FUNCTION_CALLING,
        }

    @property
    def capabilities(self) -> Set[str]:
        """Get the capabilities of this model"""
        return self._capabilities

    async def generate(self, messages: List[Message], **kwargs: Any) -> Message:
        """Generate a response from the model"""
        return await self.generate_response(messages)

    async def generate_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> AsyncIterator[Message]:
        """Stream a response from the model"""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[msg.to_dict() for msg in messages],
            stream=True,
            **self.parameters,
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield Message(
                    role=MessageRole.ASSISTANT, content=chunk.choices[0].delta.content
                )

    async def generate_response(self, messages: List[Message]) -> Message:
        """Generate a response using the OpenAI API"""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[msg.to_dict() for msg in messages],
            **self.parameters,
        )
        return Message(
            role=MessageRole.ASSISTANT, content=response.choices[0].message.content
        )

    async def _generate(self, messages: List[Message]) -> Message:
        """Internal method for model-specific response generation"""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[msg.to_dict() for msg in messages],
            **self.parameters,
        )
        return Message(
            role=MessageRole.ASSISTANT, content=response.choices[0].message.content
        )

    async def _prepare_messages(self, messages: List[Message]) -> List[Dict]:
        """Convert messages to OpenAI format"""
        return [msg.to_dict() for msg in messages]

    async def _make_api_call(self, messages: List[Dict]) -> Message:
        """Make API call to OpenAI"""
        response = await self.client.chat.completions.create(
            model=self.model_name, messages=messages, **self.parameters
        )
        return Message(
            role=MessageRole.ASSISTANT, content=response.choices[0].message.content
        )

    async def count_tokens(self, text: str) -> int:
        """Count tokens in text - simplified version without tiktoken"""
        # Simple approximation instead of using tiktoken
        # Roughly 4 characters per token for English text
        return len(text) // 4
