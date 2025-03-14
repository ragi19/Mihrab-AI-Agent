"""
Anthropic provider implementation
"""

from typing import Any, Dict, Set

from anthropic import AsyncAnthropic

from ...types import ModelCapability, ModelInfo
from ..base import BaseProvider, ProviderError
from .models import ClaudeModel


class AnthropicProvider(BaseProvider):
    """Provider implementation for Anthropic Claude models"""

    # Supported models and their capabilities
    SUPPORTED_MODELS = {
        "claude-3-opus-20240229": ModelInfo(
            name="claude-3-opus-20240229",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.STREAM,
            },
            max_tokens=200000,
            context_window=200000,
            supports_streaming=True,
            supports_functions=True,
            pricing_per_1k_tokens=0.015,
        ),
        "claude-3-sonnet-20240229": ModelInfo(
            name="claude-3-sonnet-20240229",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.STREAM,
            },
            max_tokens=200000,
            context_window=200000,
            supports_streaming=True,
            supports_functions=True,
            pricing_per_1k_tokens=0.003,
        ),
        "claude-2.1": ModelInfo(
            name="claude-2.1",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
            },
            max_tokens=100000,
            context_window=100000,
            supports_streaming=True,
            pricing_per_1k_tokens=0.008,
        ),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize Anthropic client
        if "api_key" not in kwargs:
            raise ProviderError("Anthropic API key required")

        self.client = AsyncAnthropic(api_key=kwargs["api_key"])

        # Register supported models
        for model_name, model_info in self.SUPPORTED_MODELS.items():
            self.register_model(model_name, ClaudeModel, model_info)

    async def create_model(self, model_name: str) -> ClaudeModel:
        """Create a Claude model instance"""
        if model_name not in self._models:
            raise ProviderError(f"Model {model_name} not supported")

        model = ClaudeModel(
            model_name=model_name,
            client=self.client,
            parameters=self.get_default_parameters(model_name),
        )

        return model

    def register_model(
        self, model_name: str, model_class: type, model_info: ModelInfo
    ) -> None:
        """Register a model with this provider"""
        self._models[model_name] = model_class
        self._model_info[model_name] = model_info
        self.logger.info(f"Registered Anthropic model: {model_name}")

    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for model initialization"""
        return {
            "client": self.client,
            "parameters": self._config.get("default_parameters", {}),
        }

    @classmethod
    def get_required_config_keys(cls) -> Set[str]:
        """Get required configuration keys"""
        return {"api_key"}

    async def validate_api_key(self) -> bool:
        """Validate the Anthropic API key"""
        try:
            # Try to list models as a validation test
            await self.client.messages.create(
                model="claude-instant-1.2",
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}],
            )
            return True
        except Exception as e:
            raise ProviderError(f"Invalid Anthropic API key: {e}")


# For backward compatibility
ClaudeProvider = AnthropicProvider

# Export the provider
__all__ = ["AnthropicProvider", "ClaudeProvider", "ClaudeModel"]
