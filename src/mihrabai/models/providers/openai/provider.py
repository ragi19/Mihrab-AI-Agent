"""Provider implementation for OpenAI models."""

from typing import Dict, List, Optional

import openai

from mihrabai.core.message import Message
from mihrabai.core.types import ModelParameters
from mihrabai.models.base import BaseModel
from mihrabai.models.provider_registry import register_provider
from mihrabai.utils.tracing import Span, TraceProvider


class OpenAIProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def create_model(
        self, model_name: str, parameters: Optional[ModelParameters] = None
    ) -> BaseModel:
        """Create a new OpenAI model instance"""
        from .models import GPTModel

        return GPTModel(
            model_name=model_name, client=self.client, parameters=parameters
        )


class OpenAIModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        trace_provider: Optional[TraceProvider] = None,
    ):
        super().__init__(model_name, trace_provider)
        self.client = openai.OpenAI(api_key=api_key)

    async def generate_response(self, messages: List[Message]) -> Message:
        return await self._wrapped_generate(messages)

    async def _generate(self, messages: List[Message]) -> Message:
        """Internal method for model-specific response generation"""
        prepared = await self._prepare_messages(messages)
        return await self._make_api_call(prepared)

    async def _prepare_messages(self, messages: List[Message]) -> List[Dict]:
        """Convert messages to OpenAI format"""
        with self._create_span(
            "format_messages", {"message_count": len(messages)}
        ) as span:
            formatted = [{"role": msg.role, "content": msg.content} for msg in messages]
            span.metadata["formatted_count"] = len(formatted)
            return formatted

    async def _make_api_call(self, messages: List[Dict]) -> Message:
        """Make API call to OpenAI"""
        completion = await self.client.chat.completions.create(
            model=self.model_name, messages=messages
        )
        return Message(role="assistant", content=completion.choices[0].message.content)

    async def count_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken"""
        import tiktoken

        with self._create_span("count_tokens", {"text_length": len(text)}) as span:
            encoding = tiktoken.encoding_for_model(self.model_name)
            token_count = len(encoding.encode(text))
            span.metadata["token_count"] = token_count
            return token_count
