"""
Tests for provider streaming capabilities
"""

from typing import Final
from unittest.mock import AsyncMock, patch

import pytest

from llm_agents.core.message import Message, MessageRole
from llm_agents.models import ModelCapability
from llm_agents.models.providers.anthropic import ClaudeProvider
from llm_agents.models.providers.groq import GroqProvider

STREAM: Final[str] = "streaming"  # Alias for STREAMING


@pytest.mark.asyncio
async def test_claude_streaming():
    """Test Claude model streaming capability"""
    with patch("anthropic.AsyncAnthropic") as mock_client:
        # Mock streaming response
        async def mock_stream():
            responses = [
                AsyncMock(content=[AsyncMock(text="Hello")]),
                AsyncMock(content=[AsyncMock(text=" world")]),
                AsyncMock(content=[AsyncMock(text="!")]),
            ]
            for response in responses:
                yield response

        mock_client.return_value.messages.create = AsyncMock(return_value=mock_stream())

        provider = ClaudeProvider(api_key="test-key")
        model = await provider.create_model("claude-3-opus-20240229")

        assert STREAM in model.model_info.capabilities

        message = Message(role=MessageRole.USER, content="Test message")
        chunks = []

        async for chunk in model.generate_stream([message]):
            assert isinstance(chunk, Message)
            assert chunk.role == MessageRole.ASSISTANT
            chunks.append(chunk.content)

        assert "".join(chunks) == "Hello world!"


@pytest.mark.asyncio
async def test_groq_streaming():
    """Test Groq model streaming capability"""
    with patch("groq.AsyncGroq") as mock_client:
        # Mock streaming response
        async def mock_stream():
            responses = [
                AsyncMock(choices=[AsyncMock(delta=AsyncMock(content="Hello"))]),
                AsyncMock(choices=[AsyncMock(delta=AsyncMock(content=" world"))]),
                AsyncMock(choices=[AsyncMock(delta=AsyncMock(content="!"))]),
            ]
            for response in responses:
                yield response

        # Create a mock for the chat.completions.create method
        mock_completions = AsyncMock()
        mock_completions.create = AsyncMock(return_value=mock_stream())

        # Set up the mock client
        mock_client.return_value = AsyncMock()
        mock_client.return_value.chat = AsyncMock()
        mock_client.return_value.chat.completions = mock_completions

        provider = GroqProvider(api_key="test-key")
        model = await provider.create_model("llama2-70b-4096")

        assert STREAM in model.model_info.capabilities

        message = Message(role=MessageRole.USER, content="Test message")
        chunks = []

        async for chunk in model.generate_stream([message]):
            assert isinstance(chunk, Message)
            assert chunk.role == MessageRole.ASSISTANT
            chunks.append(chunk.content)

        assert "".join(chunks) == "Hello world!"


@pytest.mark.asyncio
async def test_stream_error_handling():
    """Test error handling in streaming responses"""
    with patch("anthropic.AsyncAnthropic") as mock_client:

        async def mock_stream():
            yield AsyncMock(content=[AsyncMock(text="Hello")])
            raise Exception("Stream error")

        mock_client.return_value.messages.create = AsyncMock(return_value=mock_stream())

        provider = ClaudeProvider(api_key="test-key")
        model = await provider.create_model("claude-3-opus-20240229")
        message = Message(role=MessageRole.USER, content="Test message")

        with pytest.raises(Exception, match="Stream error"):
            async for _ in model.generate_stream([message]):
                pass
