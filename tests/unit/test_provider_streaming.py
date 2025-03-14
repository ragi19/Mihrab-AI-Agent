"""
Tests for provider streaming capabilities
"""

from typing import Final
from unittest.mock import AsyncMock, patch

import pytest

from mihrabai.core.message import Message, MessageRole
from mihrabai.models import ModelCapability
from mihrabai.models.providers.anthropic import ClaudeProvider
from mihrabai.models.providers.groq import GroqProvider

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

        # Mock the ClaudeModel class
        with patch(
            "mihrabai.models.providers.anthropic.ClaudeModel"
        ) as MockClaudeModel:
            # Create a mock model instance
            mock_model = AsyncMock()
            mock_model.model_info = AsyncMock()
            mock_model.model_info.capabilities = {"streaming"}

            # Set up the generate_stream method
            async def mock_stream_generator(messages, **kwargs):
                yield Message(role=MessageRole.ASSISTANT, content="Hello")
                yield Message(role=MessageRole.ASSISTANT, content=" world")
                yield Message(role=MessageRole.ASSISTANT, content="!")

            mock_model.generate_stream = mock_stream_generator

            # Make the mock class return our mock instance
            MockClaudeModel.return_value = mock_model

            # Mock the create_model method to return our mock model
            with patch.object(ClaudeProvider, "create_model", return_value=mock_model):
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

        # Mock the GroqModel.stream_response method to avoid making actual API calls
        with patch(
            "mihrabai.models.providers.groq.models.GroqModel.stream_response"
        ) as mock_stream_response:

            async def mock_stream_generator():
                yield Message(role=MessageRole.ASSISTANT, content="Hello")
                yield Message(role=MessageRole.ASSISTANT, content=" world")
                yield Message(role=MessageRole.ASSISTANT, content="!")

            mock_stream_response.return_value = mock_stream_generator()

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

        # Mock the ClaudeModel class
        with patch(
            "mihrabai.models.providers.anthropic.ClaudeModel"
        ) as MockClaudeModel:
            # Create a mock model instance
            mock_model = AsyncMock()

            # Set up the generate_stream method to raise an exception
            async def mock_stream_generator(messages, **kwargs):
                yield Message(role=MessageRole.ASSISTANT, content="Hello")
                raise Exception("Stream error")

            mock_model.generate_stream = mock_stream_generator

            # Make the mock class return our mock instance
            MockClaudeModel.return_value = mock_model

            # Mock the create_model method to return our mock model
            with patch.object(ClaudeProvider, "create_model", return_value=mock_model):
                provider = ClaudeProvider(api_key="test-key")
                model = await provider.create_model("claude-3-opus-20240229")
                message = Message(role=MessageRole.USER, content="Test message")

                with pytest.raises(Exception, match="Stream error"):
                    async for _ in model.generate_stream([message]):
                        pass
