"""
Integration tests for multi-provider functionality
"""

import pytest
from unittest.mock import AsyncMock, patch
from llm_agents.core.message import Message, MessageRole
from llm_agents.models.providers.openai import OpenAIProvider
from llm_agents.models.providers.anthropic import AnthropicProvider
from llm_agents.models.providers.groq import GroqProvider
from llm_agents.examples.advanced.multi_provider_agent import MultiProviderAgent
from llm_agents.runtime.runner import AgentRunner
from llm_agents.core.agent import Agent
from llm_agents.models import (
    ModelCapability,
    ProviderRegistry,
    create_model,
    ModelCreationError,
)
from llm_agents.models.multi_provider import MultiProviderModel
from llm_agents.models.provider_stats import ProviderStatsManager


class SimpleAgent(Agent):
    async def process_message(self, message: Message) -> Message:
        return await self.model.generate_response(self.conversation_history + [message])


@pytest.fixture
def stats_manager():
    """Stats manager fixture"""
    manager = ProviderStatsManager()
    manager._stats.clear()
    return manager


@pytest.mark.asyncio
async def test_multi_provider_agent():
    """Test MultiProviderAgent with multiple providers"""
    # Mock responses for each provider
    responses = {
        "openai": "OpenAI response",
        "anthropic": "Anthropic response",
        "groq": "Groq response",
    }

    providers = {}

    # Mock OpenAI
    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_openai_response = AsyncMock()
        mock_openai_response.choices = [
            AsyncMock(message=AsyncMock(content=responses["openai"]))
        ]
        mock_openai.return_value.chat.completions.create = AsyncMock(
            return_value=mock_openai_response
        )
        provider = OpenAIProvider(api_key="test-key")
        model = await provider.create_model("gpt-3.5-turbo")
        providers["openai"] = SimpleAgent(model=model)

    # Mock Anthropic
    with patch("anthropic.AsyncAnthropic") as mock_anthropic:
        mock_anthropic_response = AsyncMock()
        mock_anthropic_response.content = [AsyncMock(text=responses["anthropic"])]
        mock_anthropic.return_value.messages.create = AsyncMock(
            return_value=mock_anthropic_response
        )
        provider = AnthropicProvider(api_key="test-key")
        model = await provider.create_model("claude-3-opus-20240229")
        providers["anthropic"] = SimpleAgent(model=model)

    # Mock Groq
    with patch("groq.AsyncGroq") as mock_groq:
        mock_groq_response = AsyncMock()
        mock_groq_response.choices = [
            AsyncMock(message=AsyncMock(content=responses["groq"]))
        ]
        mock_groq.return_value.chat.completions.create = AsyncMock(
            return_value=mock_groq_response
        )
        provider = GroqProvider(api_key="test-key")
        model = await provider.create_model("llama2-70b-4096")
        providers["groq"] = SimpleAgent(model=model)

    # Create multi-provider agent
    agent = MultiProviderAgent(providers=providers)
    runner = AgentRunner(agent=agent)

    # Test conversation
    message = Message(role=MessageRole.USER, content="Test message")
    response = await runner.run(message)

    # Verify response includes all provider responses
    for provider_name, expected_response in responses.items():
        assert provider_name.upper() in response.content
        assert expected_response in response.content


@pytest.mark.asyncio
async def test_multi_provider_error_handling():
    """Test MultiProviderAgent error handling when a provider fails"""
    providers = {}

    # Mock successful provider
    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_openai_response = AsyncMock()
        mock_openai_response.choices = [
            AsyncMock(message=AsyncMock(content="Success response"))
        ]
        mock_openai.return_value.chat.completions.create = AsyncMock(
            return_value=mock_openai_response
        )
        provider = OpenAIProvider(api_key="test-key")
        model = await provider.create_model("gpt-3.5-turbo")
        providers["openai"] = SimpleAgent(model=model)

    # Mock failing provider
    with patch("anthropic.AsyncAnthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        provider = AnthropicProvider(api_key="test-key")
        model = await provider.create_model("claude-3-opus-20240229")
        providers["anthropic"] = SimpleAgent(model=model)

    # Create multi-provider agent
    agent = MultiProviderAgent(providers=providers)
    runner = AgentRunner(agent=agent)

    # Test conversation
    message = Message(role=MessageRole.USER, content="Test message")
    response = await runner.run(message)

    # Verify successful provider response is included
    assert "OPENAI" in response.content
    assert "Success response" in response.content


@pytest.mark.asyncio
async def test_multi_provider_fallback():
    """Test multi-provider fallback functionality"""
    with (
        patch("anthropic.AsyncAnthropic") as anthropic_client,
        patch("groq.AsyncGroq") as groq_client,
    ):

        # Setup primary provider to fail
        anthropic_client.return_value.messages.create = AsyncMock(
            side_effect=Exception("Rate limit exceeded")
        )

        # Setup fallback provider
        groq_response = AsyncMock()
        groq_response.choices = [
            AsyncMock(message=AsyncMock(content="Fallback response"))
        ]
        groq_client.return_value.chat.completions.create = AsyncMock(
            return_value=groq_response
        )

        # Create multi-provider model
        model = await MultiProviderModel.create(
            primary_model="claude-3-opus-20240229",
            fallback_models=["llama2-70b-4096"],
            required_capabilities={ModelCapability.CHAT},
        )

        # Test fallback
        message = Message(role=MessageRole.USER, content="Test message")
        response = await model.generate_response([message])

        assert response.content == "Fallback response"


@pytest.mark.asyncio
async def test_multi_provider_capability_matching():
    """Test capability-based provider selection"""
    providers = {
        # Provider with basic capabilities
        "basic": {"capabilities": {ModelCapability.CHAT}, "response": "Basic response"},
        # Provider with advanced capabilities
        "advanced": {
            "capabilities": {
                ModelCapability.CHAT,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.STREAM,
            },
            "response": "Advanced response",
        },
    }

    model = await MultiProviderModel.create(
        primary_model="claude-3-opus-20240229",
        fallback_models=["llama2-70b-4096"],
        required_capabilities={ModelCapability.CHAT, ModelCapability.FUNCTION_CALLING},
    )

    # Should select provider with required capabilities
    assert model.current_provider == "anthropic"


@pytest.mark.asyncio
async def test_multi_provider_cost_optimization(stats_manager):
    """Test cost-based provider selection"""
    # Record usage stats
    stats_manager.record_request(
        provider="anthropic",
        model="claude-3-opus-20240229",
        prompt_tokens=1000,
        completion_tokens=500,
        cost=0.015,
    )
    stats_manager.record_request(
        provider="groq",
        model="llama2-70b-4096",
        prompt_tokens=1000,
        completion_tokens=500,
        cost=0.0007,
    )

    # Create cost-optimized model
    model = await MultiProviderModel.create(
        primary_model="llama2-70b-4096",  # Should pick cheaper model
        fallback_models=["claude-3-opus-20240229"],
        required_capabilities={ModelCapability.CHAT},
        optimize_for="cost",
    )

    assert model.current_provider == "groq"


@pytest.mark.asyncio
async def test_multi_provider_error_handling():
    """Test error handling across providers"""
    with (
        patch("anthropic.AsyncAnthropic") as anthropic_client,
        patch("groq.AsyncGroq") as groq_client,
    ):

        # Make all providers fail
        anthropic_client.return_value.messages.create = AsyncMock(
            side_effect=Exception("Primary provider error")
        )
        groq_client.return_value.chat.completions.create = AsyncMock(
            side_effect=Exception("Fallback provider error")
        )

        model = await MultiProviderModel.create(
            primary_model="claude-3-opus-20240229",
            fallback_models=["llama2-70b-4096"],
            required_capabilities={ModelCapability.CHAT},
        )

        message = Message(role=MessageRole.USER, content="Test message")

        # Should raise error when all providers fail
        with pytest.raises(ModelCreationError):
            await model.generate_response([message])


@pytest.mark.asyncio
async def test_multi_provider_streaming():
    """Test streaming across providers"""
    with patch("anthropic.AsyncAnthropic") as anthropic_client:
        # Mock streaming response
        async def mock_stream():
            responses = [
                AsyncMock(content=[AsyncMock(text="Hello")]),
                AsyncMock(content=[AsyncMock(text=" world")]),
            ]
            for response in responses:
                yield response

        anthropic_client.return_value.messages.create = AsyncMock(
            return_value=mock_stream()
        )

        model = await MultiProviderModel.create(
            primary_model="claude-3-opus-20240229",
            fallback_models=[],
            required_capabilities={ModelCapability.CHAT, ModelCapability.STREAM},
        )

        message = Message(role=MessageRole.USER, content="Test message")
        chunks = []

        async for chunk in model.generate_stream([message]):
            chunks.append(chunk.content)

        assert "".join(chunks) == "Hello world"


def test_multi_provider_stats_tracking(stats_manager):
    """Test statistics tracking for multi-provider usage"""
    # Record mixed success/failure stats
    stats_manager.record_request(
        provider="anthropic",
        model="claude-3-opus-20240229",
        prompt_tokens=100,
        completion_tokens=50,
        cost=0.0015,
    )
    stats_manager.record_error(
        provider="anthropic",
        model="claude-3-opus-20240229",
        error="Rate limit",
        is_rate_limit=True,
    )
    stats_manager.record_request(
        provider="groq",
        model="llama2-70b-4096",
        prompt_tokens=100,
        completion_tokens=50,
        cost=0.0007,
    )

    # Generate report
    report = stats_manager.get_usage_report()

    assert report["total_requests"] == 2
    assert report["total_errors"] == 1
    assert report["providers"]["anthropic"]["rate_limit_hits"] == 1
    assert report["total_cost"] == 0.0022  # Sum of both providers
