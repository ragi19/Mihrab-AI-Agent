"""
Groq provider implementation
"""
from typing import Dict, Any, Set
from groq import AsyncGroq

from ..base import BaseProvider, ProviderError
from ...types import ModelInfo, ModelCapability
from .models import GroqModel
from ...provider_registry import ProviderRegistry, ProviderInfo

class GroqProvider(BaseProvider):
    """Provider implementation for Groq models"""
    
    # Supported models and their capabilities
    SUPPORTED_MODELS = {
        "llama2-70b-4096": ModelInfo(
            name="llama2-70b-4096",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM
            },
            max_tokens=4096,
            context_window=4096,
            supports_streaming=True,
            pricing_per_1k_tokens=0.0007,
            provider_name="groq"
        ),
        "llama3-70b-8192": ModelInfo(
            name="llama3-70b-8192",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM
            },
            max_tokens=8192,
            context_window=8192,
            supports_streaming=True,
            pricing_per_1k_tokens=0.0007,
            provider_name="groq"
        ),
        "llama3-8b-8192": ModelInfo(
            name="llama3-8b-8192",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM
            },
            max_tokens=8192,
            context_window=8192,
            supports_streaming=True,
            pricing_per_1k_tokens=0.0003,
            provider_name="groq"
        ),
        "mixtral-8x7b-32768": ModelInfo(
            name="mixtral-8x7b-32768",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM
            },
            max_tokens=32768,
            context_window=32768,
            supports_streaming=True,
            pricing_per_1k_tokens=0.0007,
            provider_name="groq"
        ),
        "gemma2-9b-it": ModelInfo(
            name="gemma2-9b-it",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM
            },
            max_tokens=8192,
            context_window=8192,
            supports_streaming=True,
            pricing_per_1k_tokens=0.0003,
            provider_name="groq"
        ),
        "llama-3.1-8b-instant": ModelInfo(
            name="llama-3.1-8b-instant",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM
            },
            max_tokens=8192,
            context_window=8192,
            supports_streaming=True,
            pricing_per_1k_tokens=0.0003,
            provider_name="groq"
        ),
        "llama-3.2-11b-vision-preview": ModelInfo(
            name="llama-3.2-11b-vision-preview",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM,
                ModelCapability.IMAGE_INPUT
            },
            max_tokens=4096,
            context_window=8192,
            supports_streaming=True,
            pricing_per_1k_tokens=0.0005,
            provider_name="groq"
        ),
        "llama-3.2-90b-vision-preview": ModelInfo(
            name="llama-3.2-90b-vision-preview",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM,
                ModelCapability.IMAGE_INPUT
            },
            max_tokens=4096,
            context_window=8192,
            supports_streaming=True,
            pricing_per_1k_tokens=0.001,
            provider_name="groq"
        ),
        "llama-3.3-70b-versatile": ModelInfo(
            name="llama-3.3-70b-versatile",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM
            },
            max_tokens=8192,
            context_window=8192,
            supports_streaming=True,
            pricing_per_1k_tokens=0.001,
            provider_name="groq"
        ),
        # Additional models from the example
        "deepseek-r1-distill-llama-70b": ModelInfo(
            name="deepseek-r1-distill-llama-70b",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM
            },
            max_tokens=4096,
            context_window=32768,
            supports_streaming=True,
            pricing_per_1k_tokens=0.0007,
            provider_name="groq"
        ),
        "deepseek-r1-distill-qwen-32b": ModelInfo(
            name="deepseek-r1-distill-qwen-32b",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM
            },
            max_tokens=4096,
            context_window=32768,
            supports_streaming=True,
            pricing_per_1k_tokens=0.0007,
            provider_name="groq"
        ),
        "distil-whisper-large-v3-en": ModelInfo(
            name="distil-whisper-large-v3-en",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM
            },
            max_tokens=4096,
            context_window=8192,
            supports_streaming=True,
            pricing_per_1k_tokens=0.0003,
            provider_name="groq"
        ),
        "llama-3.2-1b-preview": ModelInfo(
            name="llama-3.2-1b-preview",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM
            },
            max_tokens=4096,
            context_window=8192,
            supports_streaming=True,
            pricing_per_1k_tokens=0.0001,
            provider_name="groq"
        ),
        "llama-3.2-3b-preview": ModelInfo(
            name="llama-3.2-3b-preview",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM
            },
            max_tokens=4096,
            context_window=8192,
            supports_streaming=True,
            pricing_per_1k_tokens=0.0002,
            provider_name="groq"
        ),
        "llama-3.3-70b-specdec": ModelInfo(
            name="llama-3.3-70b-specdec",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM
            },
            max_tokens=4096,
            context_window=8192,
            supports_streaming=True,
            pricing_per_1k_tokens=0.001,
            provider_name="groq"
        ),
        "llama-guard-3-8b": ModelInfo(
            name="llama-guard-3-8b",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM
            },
            max_tokens=4096,
            context_window=8192,
            supports_streaming=True,
            pricing_per_1k_tokens=0.0003,
            provider_name="groq"
        ),
        "mistral-saba-24b": ModelInfo(
            name="mistral-saba-24b",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM
            },
            max_tokens=4096,
            context_window=8192,
            supports_streaming=True,
            pricing_per_1k_tokens=0.0005,
            provider_name="groq"
        ),
        "qwen-2.5-32b": ModelInfo(
            name="qwen-2.5-32b",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM
            },
            max_tokens=4096,
            context_window=32768,
            supports_streaming=True,
            pricing_per_1k_tokens=0.0007,
            provider_name="groq"
        ),
        "qwen-2.5-coder-32b": ModelInfo(
            name="qwen-2.5-coder-32b",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM
            },
            max_tokens=4096,
            context_window=32768,
            supports_streaming=True,
            pricing_per_1k_tokens=0.0007,
            provider_name="groq"
        ),
        "qwen-qwq-32b": ModelInfo(
            name="qwen-qwq-32b",
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.COMPLETION,
                ModelCapability.SYSTEM_MESSAGES,
                ModelCapability.TOKEN_COUNTING,
                ModelCapability.STREAM
            },
            max_tokens=4096,
            context_window=32768,
            supports_streaming=True,
            pricing_per_1k_tokens=0.0007,
            provider_name="groq"
        )
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize Groq client
        if "api_key" not in kwargs:
            raise ProviderError("Groq API key required")
        
        self.client = AsyncGroq(
            api_key=kwargs["api_key"]
        )
        
        # Register supported models
        for model_name, model_info in self.SUPPORTED_MODELS.items():
            self.register_model(model_name, GroqModel, model_info)
    
    async def create_model(self, model_name: str) -> GroqModel:
        """Create a Groq model instance"""
        if model_name not in self._models:
            raise ProviderError(f"Model {model_name} not supported by Groq provider. Supported models: {', '.join(self._models.keys())}")
            
        model = GroqModel(
            model_name=model_name,
            client=self.client,
            parameters=self.get_default_parameters(model_name)
        )
        
        return model
    
    def register_model(self, model_name: str, model_class: type, model_info: ModelInfo) -> None:
        """Register a model with this provider"""
        self._models[model_name] = model_class
        self._model_info[model_name] = model_info
        self.logger.info(f"Registered Groq model: {model_name}")
    
    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for model initialization"""
        return {
            "client": self.client,
            "parameters": self._config.get("default_parameters", {})
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
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception as e:
            raise ProviderError(f"Invalid Groq API key: {e}")

# Register provider with registry
def register_provider():
    """Register Groq provider with registry"""
    provider_info = ProviderInfo(
        name="groq",
        supported_models=set(GroqProvider.SUPPORTED_MODELS.keys()),
        features={
            "chat", 
            "streaming", 
            "model_selection",
            "system_messages"
        },
        requires_api_key=True,
        default_model="llama3-70b-8192"  # Updated to newer model
    )
    
    ProviderRegistry.register_provider(
        name="groq",
        provider_class=GroqProvider,
        provider_info=provider_info
    )

# Auto-register when module is imported
register_provider()

__all__ = ['GroqProvider', 'GroqModel']