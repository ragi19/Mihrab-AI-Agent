"""
Groq provider implementation
"""

from typing import Any, Dict, Set, Type

from groq import AsyncGroq

from ...base import ModelInfo
from ...provider_registry import ProviderInfo, ProviderRegistry
from ...types import ModelCapability
from ..base import BaseProvider, ProviderError
from .models import GroqModel


class GroqProvider(BaseProvider):
    """Provider implementation for Groq models"""

    # Supported models and their capabilities
    SUPPORTED_MODELS = {
        "llama2-70b-4096": ModelInfo(
            id="llama2-70b-4096",
            name="llama2-70b-4096",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=4096,
            max_tokens=4096,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0007,
            },
        ),
        "llama3-70b-8192": ModelInfo(
            id="llama3-70b-8192",
            name="llama3-70b-8192",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=8192,
            max_tokens=8192,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0007,
            },
        ),
        "llama3-8b-8192": ModelInfo(
            id="llama3-8b-8192",
            name="llama3-8b-8192",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=8192,
            max_tokens=8192,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0003,
            },
        ),
        "mixtral-8x7b-32768": ModelInfo(
            id="mixtral-8x7b-32768",
            name="mixtral-8x7b-32768",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=32768,
            max_tokens=32768,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0009,
            },
        ),
        "gemma2-9b-it": ModelInfo(
            id="gemma2-9b-it",
            name="gemma2-9b-it",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=8192,
            max_tokens=8192,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0003,
            },
        ),
        "gemma2-27b-it": ModelInfo(
            id="gemma2-27b-it",
            name="gemma2-27b-it",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=8192,
            max_tokens=8192,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0007,
            },
        ),
        "llama-3.1-8b-instant": ModelInfo(
            id="llama-3.1-8b-instant",
            name="llama-3.1-8b-instant",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=128000,
            max_tokens=128000,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0003,
            },
        ),
        "llama-3.1-70b-instant": ModelInfo(
            id="llama-3.1-70b-instant",
            name="llama-3.1-70b-instant",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=128000,
            max_tokens=128000,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0007,
            },
        ),
        "llama-3.2-11b-vision-preview": ModelInfo(
            id="llama-3.2-11b-vision-preview",
            name="llama-3.2-11b-vision-preview",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
                "IMAGE_INPUT",
            },
            context_window=128000,
            max_tokens=128000,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0003,
                "supports_vision": True,
            },
        ),
        "llama-3.2-90b-vision-preview": ModelInfo(
            id="llama-3.2-90b-vision-preview",
            name="llama-3.2-90b-vision-preview",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
                "IMAGE_INPUT",
            },
            context_window=128000,
            max_tokens=128000,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0007,
                "supports_vision": True,
            },
        ),
        "llama-3.3-70b-versatile": ModelInfo(
            id="llama-3.3-70b-versatile",
            name="llama-3.3-70b-versatile",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=128000,
            max_tokens=128000,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0007,
            },
        ),
        "llama-3.3-8b-versatile": ModelInfo(
            id="llama-3.3-8b-versatile",
            name="llama-3.3-8b-versatile",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=128000,
            max_tokens=128000,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0003,
            },
        ),
        # Additional models from the example
        "gemma-7b-it": ModelInfo(
            id="gemma-7b-it",
            name="gemma-7b-it",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=8192,
            max_tokens=8192,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0003,
            },
        ),
        "mixtral-8x7b-instruct": ModelInfo(
            id="mixtral-8x7b-instruct",
            name="mixtral-8x7b-instruct",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=32768,
            max_tokens=32768,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0007,
            },
        ),
        "deepseek-r1-distill-llama-70b": ModelInfo(
            id="deepseek-r1-distill-llama-70b",
            name="deepseek-r1-distill-llama-70b",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=8192,
            max_tokens=8192,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0007,
            },
        ),
        "deepseek-r1-distill-qwen-32b": ModelInfo(
            id="deepseek-r1-distill-qwen-32b",
            name="deepseek-r1-distill-qwen-32b",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=8192,
            max_tokens=8192,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0005,
            },
        ),
        "distil-whisper-large-v3-en": ModelInfo(
            id="distil-whisper-large-v3-en",
            name="distil-whisper-large-v3-en",
            provider="groq",
            capabilities={
                "AUDIO_TRANSCRIPTION",
            },
            context_window=0,
            max_tokens=0,
            metadata={
                "supports_streaming": False,
                "pricing_per_1k_tokens": 0.0001,
            },
        ),
        "whisper-large-v3": ModelInfo(
            id="whisper-large-v3",
            name="whisper-large-v3",
            provider="groq",
            capabilities={
                "AUDIO_TRANSCRIPTION",
            },
            context_window=0,
            max_tokens=0,
            metadata={
                "supports_streaming": False,
                "pricing_per_1k_tokens": 0.0001,
            },
        ),
        "llama-3.2-1b-preview": ModelInfo(
            id="llama-3.2-1b-preview",
            name="llama-3.2-1b-preview",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=128000,
            max_tokens=128000,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0001,
            },
        ),
        "llama-3.2-3b-preview": ModelInfo(
            id="llama-3.2-3b-preview",
            name="llama-3.2-3b-preview",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=128000,
            max_tokens=128000,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0002,
            },
        ),
        "llama-3.3-70b-specdec": ModelInfo(
            id="llama-3.3-70b-specdec",
            name="llama-3.3-70b-specdec",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=128000,
            max_tokens=128000,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0007,
            },
        ),
        "llama-3.3-8b-specdec": ModelInfo(
            id="llama-3.3-8b-specdec",
            name="llama-3.3-8b-specdec",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=128000,
            max_tokens=128000,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0003,
            },
        ),
        "llama-guard-3-8b": ModelInfo(
            id="llama-guard-3-8b",
            name="llama-guard-3-8b",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=8192,
            max_tokens=4096,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0003,
            },
        ),
        "mistral-saba-24b": ModelInfo(
            id="mistral-saba-24b",
            name="mistral-saba-24b",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=8192,
            max_tokens=4096,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0005,
            },
        ),
        "qwen-2.5-32b": ModelInfo(
            id="qwen-2.5-32b",
            name="qwen-2.5-32b",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=32768,
            max_tokens=4096,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0007,
            },
        ),
        "qwen-2.5-coder-32b": ModelInfo(
            id="qwen-2.5-coder-32b",
            name="qwen-2.5-coder-32b",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=32768,
            max_tokens=4096,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0007,
            },
        ),
        "qwen-qwq-32b": ModelInfo(
            id="qwen-qwq-32b",
            name="qwen-qwq-32b",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=32768,
            max_tokens=4096,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0007,
            },
        ),
        "deepseek-r1-distill-llama-7b": ModelInfo(
            id="deepseek-r1-distill-llama-7b",
            name="deepseek-r1-distill-llama-7b",
            provider="groq",
            capabilities={
                "CHAT",
                "COMPLETION",
                "SYSTEM_MESSAGES",
                "TOKEN_COUNTING",
                "STREAM",
            },
            context_window=8192,
            max_tokens=8192,
            metadata={
                "supports_streaming": True,
                "pricing_per_1k_tokens": 0.0003,
            },
        ),
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Initialize Groq client
        if "api_key" not in kwargs:
            raise ProviderError("Groq API key required")

        self.client = AsyncGroq(api_key=kwargs["api_key"])

        # Register supported models
        for model_name, model_info in self.SUPPORTED_MODELS.items():
            self.register_model(model_name, GroqModel, model_info)

    async def create_model(self, model_name: str) -> GroqModel:
        """Create a Groq model instance"""
        if model_name not in self._models:
            raise ProviderError(
                f"Model {model_name} not supported by Groq provider. Supported models: {', '.join(self._models.keys())}"
            )

        model = GroqModel(
            model_name=model_name,
            client=self.client,
            parameters=self.get_default_parameters(model_name),
        )

        return model

    def register_model(
        self, model_name: str, model_class: Type[GroqModel], model_info: ModelInfo
    ) -> None:
        """Register a model with this provider"""
        super().register_model(model_name, model_class, model_info)

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
        """Validate the Groq API key"""
        try:
            # Try to list models as a validation test
            await self.client.chat.completions.create(
                model="llama3-8b-8192",  # Use a model that's likely to be available
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}],
            )
            return True
        except Exception as e:
            raise ProviderError(f"Invalid Groq API key: {e}")


# Register provider with registry
def register_provider() -> None:
    """Register Groq provider with registry"""
    provider_info = ProviderInfo(
        name="groq",
        supported_models=set(GroqProvider.SUPPORTED_MODELS.keys()),
        features={"chat", "streaming", "model_selection", "system_messages"},
        requires_api_key=True,
        default_model="llama3-70b-8192",  # Updated to newer model
    )

    ProviderRegistry.register_provider(
        name="groq", provider_class=GroqProvider, provider_info=provider_info
    )


# Auto-register when module is imported
register_provider()

__all__ = ["GroqProvider", "GroqModel"]
