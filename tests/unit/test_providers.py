"""
Unit tests for provider implementations
"""

from unittest.mock import AsyncMock, patch

import pytest

from llm_agents.core.message import Message, MessageRole
from llm_agents.models.providers.anthropic import AnthropicProvider
from llm_agents.models.providers.groq import GroqProvider
from llm_agents.models.providers.openai import OpenAIProvider


@pytest.mark.asyncio
async def test_openai_provider():
    """Test OpenAI provider implementation"""
    with patch("openai.AsyncOpenAI") as mock_client:
        # Mock response
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock(message=AsyncMock(content="Test response"))]
        mock_client.return_value.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        provider = OpenAIProvider(api_key="test-key")
        model = await provider.create_model("gpt-3.5-turbo")

        message = Message(role=MessageRole.USER, content="Hello")
        response = await model.generate_response([message])

        assert response.role == MessageRole.ASSISTANT
        assert response.content == "Test response"


@pytest.mark.asyncio
async def test_anthropic_provider():
    """Test Anthropic provider implementation"""
    with patch("anthropic.AsyncAnthropic") as mock_client:
        # Mock response
        mock_response = AsyncMock()
        mock_response.content = [AsyncMock(text="Test response")]
        mock_client.return_value.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicProvider(api_key="test-key")
        model = await provider.create_model("claude-3-opus-20240229")

        message = Message(role=MessageRole.USER, content="Hello")
        response = await model.generate_response([message])

        assert response.role == MessageRole.ASSISTANT
        assert response.content == "Test response"


@pytest.mark.asyncio
async def test_groq_provider():
    """Test Groq provider implementation"""
    with patch("groq.AsyncGroq") as mock_client:
        # Mock response
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock(message=AsyncMock(content="Test response"))]
        mock_client.return_value.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        provider = GroqProvider(api_key="test-key")
        model = await provider.create_model("llama2-70b-4096")

        message = Message(role=MessageRole.USER, content="Hello")
        response = await model.generate_response([message])

        assert response.role == MessageRole.ASSISTANT
        assert response.content == "Test response"


@pytest.mark.asyncio
async def test_token_counting():
    """Test token counting across providers"""
    providers = [
        (OpenAIProvider, "gpt-3.5-turbo", "openai.AsyncOpenAI"),
        (AnthropicProvider, "claude-3-opus-20240229", "anthropic.AsyncAnthropic"),
        (GroqProvider, "llama2-70b-4096", "groq.AsyncGroq"),
    ]

    text = "This is a test message"

    for Provider, model_name, client_path in providers:
        with patch(client_path) as mock_client:
            provider = Provider(api_key="test-key")
            provider.client = mock_client.return_value
            model = await provider.create_model(model_name)
            token_count = await model.count_tokens(text)

            assert isinstance(token_count, int)
            assert token_count > 0


"""
Tests for provider framework functionality
"""
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from llm_agents.models import (
    BaseModel,
    BaseProvider,
    ModelCapability,
    ModelInfo,
    ProviderError,
    ProviderRegistry,
)
from llm_agents.models.provider_discovery import ProviderDiscovery
from llm_agents.models.provider_registry import ProviderInfo
from llm_agents.models.provider_stats import ProviderStatsManager


class MockProvider(BaseProvider):
    """Mock provider for testing"""

    SUPPORTED_MODELS = {
        "test-model": ModelInfo(
            id="test-model",
            name="test-model",
            provider="mock",
            capabilities={ModelCapability.CHAT, ModelCapability.COMPLETION},
            max_tokens=1000,
            context_window=1000,
        )
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def create_model(self, model_name: str) -> BaseModel:
        """Create a mock model instance"""
        # This is just a stub for testing
        from unittest.mock import AsyncMock

        mock_model = AsyncMock(spec=BaseModel)
        mock_model.model_name = model_name
        mock_model.model_info = self.SUPPORTED_MODELS.get(model_name)

        # Add initialize method
        async def initialize():
            return None

        mock_model.initialize = initialize
        return mock_model


@pytest.fixture
def provider_registry():
    """Provider registry fixture"""
    ProviderRegistry._providers.clear()
    ProviderRegistry._provider_info.clear()
    ProviderRegistry._initialized_providers.clear()
    return ProviderRegistry


@pytest.fixture
def stats_manager():
    """Stats manager fixture"""
    manager = ProviderStatsManager()
    manager._stats.clear()
    return manager


def test_provider_registration(provider_registry):
    """Test provider registration"""
    provider = MockProvider()
    model_info = provider.SUPPORTED_MODELS["test-model"]

    # Create ProviderInfo from model info
    provider_info = ProviderInfo(
        name="mock",
        supported_models={"test-model"},
        features={ModelCapability.CHAT, ModelCapability.COMPLETION},
        requires_api_key=True,
    )

    provider_registry.register_provider("mock", MockProvider, provider_info)

    assert "mock" in provider_registry.list_providers()
    assert provider_registry.get_provider("mock") == MockProvider
    assert provider_registry.get_provider_info("mock") == provider_info


def test_provider_model_support(provider_registry):
    """Test provider model support checks"""
    provider = MockProvider()
    model_info = provider.SUPPORTED_MODELS["test-model"]

    # Create ProviderInfo from model info
    provider_info = ProviderInfo(
        name="mock",
        supported_models={"test-model"},
        features={ModelCapability.CHAT, ModelCapability.COMPLETION},
        requires_api_key=True,
    )

    provider_registry.register_provider("mock", MockProvider, provider_info)

    assert "test-model" in provider_registry.list_models_for_provider("mock")
    assert provider_registry.find_provider_for_model("test-model") == "mock"
    assert provider_registry.find_provider_for_model("nonexistent") is None


@pytest.mark.asyncio
async def test_provider_model_creation(provider_registry):
    """Test model creation through provider"""
    provider = MockProvider(api_key="test-key")
    model_info = provider.SUPPORTED_MODELS["test-model"]

    # Create ProviderInfo from model info
    provider_info = ProviderInfo(
        name="mock",
        supported_models={"test-model"},
        features={ModelCapability.CHAT, ModelCapability.COMPLETION},
        requires_api_key=True,
    )

    provider_registry.register_provider("mock", MockProvider, provider_info)

    model = await provider_registry.create_model(
        provider_name="mock", model_name="test-model", api_key="test-key"
    )

    assert model is not None


def test_provider_config_validation(provider_registry):
    """Test provider configuration validation"""
    provider = MockProvider()
    model_info = provider.SUPPORTED_MODELS["test-model"]

    # Create ProviderInfo from model info
    provider_info = ProviderInfo(
        name="mock",
        supported_models={"test-model"},
        features={ModelCapability.CHAT, ModelCapability.COMPLETION},
        requires_api_key=True,
    )

    provider_registry.register_provider("mock", MockProvider, provider_info)

    # Should pass with API key
    provider_registry.validate_provider_config("mock", {"api_key": "test"})

    # Should fail without API key - this is expected to raise ProviderError
    with pytest.raises(ProviderError, match="requires an API key"):
        provider_registry.validate_provider_config("mock", {})


def test_stats_tracking(stats_manager):
    """Test provider statistics tracking"""
    stats_manager.record_request(
        provider="mock",
        model="test-model",
        prompt_tokens=100,
        completion_tokens=50,
        cost=0.0015,
    )

    stats = stats_manager.get_provider_stats("mock")
    assert stats is not None
    assert stats.successes == 1
    assert stats.total_tokens == 150
    assert stats.total_cost == 0.0015


def test_error_tracking(stats_manager):
    """Test provider error tracking"""
    stats_manager.record_error(
        provider="mock", model="test-model", error="Test error", is_rate_limit=True
    )

    stats = stats_manager.get_provider_stats("mock")
    assert stats is not None
    assert stats.failures == 1


def test_usage_report(stats_manager):
    """Test usage report generation"""
    # Record some usage
    stats_manager.record_request(
        provider="mock",
        model="test-model",
        prompt_tokens=100,
        completion_tokens=50,
        cost=0.0015,
    )
    stats_manager.record_error(provider="mock", model="test-model", error="Test error")

    # Get report
    report = stats_manager.get_usage_report()

    assert report["total_cost"] == 0.0015
    assert report["total_tokens"] == 150
    assert report["total_requests"] == 2  # One request and one error


@pytest.mark.asyncio
async def test_provider_discovery():
    """Test provider discovery"""
    providers = ProviderDiscovery.discover_providers()

    # Should find our built-in providers
    assert "anthropic" in providers
    assert "groq" in providers
    # grok is not a provider in the current implementation
    # assert "grok" in providers
