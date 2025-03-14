"""
Model configuration and capabilities management
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Type, Union, cast

from pydantic import BaseModel, Field, validator


class ModelCapability(Enum):
    """Capabilities that a model might support"""

    TEXT_GENERATION = auto()
    CHAT_COMPLETION = auto()
    FUNCTION_CALLING = auto()
    EMBEDDINGS = auto()
    IMAGE_GENERATION = auto()
    IMAGE_UNDERSTANDING = auto()
    AUDIO_TRANSCRIPTION = auto()
    CODE_GENERATION = auto()
    CODE_COMPLETION = auto()
    TOOL_USE = auto()
    STRUCTURED_OUTPUT = auto()
    MULTI_MODAL = auto()

    def __str__(self) -> str:
        return self.name.lower()


class ModelConfig(BaseModel):
    """Configuration for a specific model"""

    model_name: str
    provider_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    capabilities: Set[ModelCapability] = Field(default_factory=set)
    context_window: Optional[int] = None
    max_tokens: Optional[int] = None
    supports_streaming: bool = False
    supports_vision: bool = False
    supports_functions: bool = False
    supports_tools: bool = False
    token_limit: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

    @validator("capabilities", pre=True)
    def parse_capabilities(cls, v: Any) -> Set[ModelCapability]:
        """Convert capability strings to enum values"""
        if isinstance(v, set):
            return v
        if isinstance(v, list):
            result: Set[ModelCapability] = set()
            for item in v:
                if isinstance(item, str):
                    try:
                        # Try to match by name (case-insensitive)
                        result.add(ModelCapability[item.upper()])
                    except KeyError:
                        # If that fails, try to find a capability with matching lowercase name
                        for cap in ModelCapability:
                            if str(cap) == item.lower():
                                result.add(cap)
                                break
                elif isinstance(item, ModelCapability):
                    result.add(item)
            return result
        return set()


class ProviderConfig(BaseModel):
    """Configuration for a model provider"""

    name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    organization_id: Optional[str] = None
    default_parameters: Dict[str, Any] = Field(default_factory=dict)
    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    auth_method: str = "api_key"
    timeout: float = 60.0
    max_retries: int = 3
    additional_config: Dict[str, Any] = Field(default_factory=dict)

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        return self.models.get(model_name)

    def supports_model(self, model_name: str) -> bool:
        """Check if this provider supports a specific model"""
        return model_name in self.models

    def get_auth_params(self) -> Dict[str, Any]:
        """Get authentication parameters for this provider"""
        auth_params = {}
        if self.api_key:
            auth_params["api_key"] = self.api_key
        if self.api_base:
            auth_params["api_base"] = self.api_base
        if self.organization_id:
            auth_params["organization_id"] = self.organization_id
        return auth_params


class ModelRegistry:
    """Registry for model configurations"""

    _provider_configs: Dict[str, ProviderConfig] = {}
    _model_to_provider: Dict[str, str] = {}

    @classmethod
    def register_provider(cls, provider_config: ProviderConfig) -> None:
        """Register a provider configuration"""
        cls._provider_configs[provider_config.name] = provider_config

        # Update model to provider mapping
        for model_name in provider_config.models:
            cls._model_to_provider[model_name] = provider_config.name

    @classmethod
    def get_provider_config(cls, provider_name: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider"""
        return cls._provider_configs.get(provider_name)

    @classmethod
    def get_model_config(
        cls, model_name: str, provider_name: Optional[str] = None
    ) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        if provider_name:
            # Get from specific provider
            provider_config = cls.get_provider_config(provider_name)
            if provider_config:
                return provider_config.get_model_config(model_name)
        else:
            # Try to find provider for this model
            provider_name = cls._model_to_provider.get(model_name)
            if provider_name:
                provider_config = cls.get_provider_config(provider_name)
                if provider_config:
                    return provider_config.get_model_config(model_name)
        return None

    @classmethod
    def find_provider_for_model(cls, model_name: str) -> Optional[str]:
        """Find a provider that supports a specific model"""
        return cls._model_to_provider.get(model_name)

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered providers"""
        return list(cls._provider_configs.keys())

    @classmethod
    def list_models(cls, provider_name: Optional[str] = None) -> List[str]:
        """List all registered models, optionally filtered by provider"""
        if provider_name:
            provider_config = cls.get_provider_config(provider_name)
            if provider_config:
                return list(provider_config.models.keys())
            return []
        return list(cls._model_to_provider.keys())

    @classmethod
    def get_models_with_capability(cls, capability: ModelCapability) -> List[str]:
        """Get models that support a specific capability"""
        result = []
        for provider_name, provider_config in cls._provider_configs.items():
            for model_name, model_config in provider_config.models.items():
                if capability in model_config.capabilities:
                    result.append(model_name)
        return result


# Default model configurations for common providers
DEFAULT_OPENAI_MODELS = {
    "gpt-4": ModelConfig(
        model_name="gpt-4",
        provider_name="openai",
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT_COMPLETION,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.TOOL_USE,
        },
        context_window=8192,
        supports_streaming=True,
        supports_functions=True,
        supports_tools=True,
    ),
    "gpt-4-turbo": ModelConfig(
        model_name="gpt-4-turbo",
        provider_name="openai",
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT_COMPLETION,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.TOOL_USE,
        },
        context_window=128000,
        supports_streaming=True,
        supports_functions=True,
        supports_tools=True,
    ),
    "gpt-3.5-turbo": ModelConfig(
        model_name="gpt-3.5-turbo",
        provider_name="openai",
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT_COMPLETION,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.TOOL_USE,
        },
        context_window=16385,
        supports_streaming=True,
        supports_functions=True,
        supports_tools=True,
    ),
}

DEFAULT_ANTHROPIC_MODELS = {
    "claude-3-opus": ModelConfig(
        model_name="claude-3-opus",
        provider_name="anthropic",
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT_COMPLETION,
            ModelCapability.TOOL_USE,
            ModelCapability.MULTI_MODAL,
        },
        context_window=200000,
        supports_streaming=True,
        supports_vision=True,
        supports_tools=True,
    ),
    "claude-3-sonnet": ModelConfig(
        model_name="claude-3-sonnet",
        provider_name="anthropic",
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT_COMPLETION,
            ModelCapability.TOOL_USE,
            ModelCapability.MULTI_MODAL,
        },
        context_window=200000,
        supports_streaming=True,
        supports_vision=True,
        supports_tools=True,
    ),
    "claude-3-haiku": ModelConfig(
        model_name="claude-3-haiku",
        provider_name="anthropic",
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT_COMPLETION,
            ModelCapability.TOOL_USE,
            ModelCapability.MULTI_MODAL,
        },
        context_window=200000,
        supports_streaming=True,
        supports_vision=True,
        supports_tools=True,
    ),
}

DEFAULT_GROQ_MODELS = {
    "llama3-70b-8192": ModelConfig(
        model_name="llama3-70b-8192",
        provider_name="groq",
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT_COMPLETION,
        },
        context_window=8192,
        supports_streaming=True,
    ),
    "mixtral-8x7b-32768": ModelConfig(
        model_name="mixtral-8x7b-32768",
        provider_name="groq",
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT_COMPLETION,
        },
        context_window=32768,
        supports_streaming=True,
    ),
    "gemma-7b-it": ModelConfig(
        model_name="gemma-7b-it",
        provider_name="groq",
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT_COMPLETION,
        },
        context_window=8192,
        supports_streaming=True,
    ),
}

# Register default providers
DEFAULT_PROVIDERS = [
    ProviderConfig(
        name="openai",
        models=DEFAULT_OPENAI_MODELS,
        default_parameters={
            "temperature": 0.7,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    ),
    ProviderConfig(
        name="anthropic",
        models=DEFAULT_ANTHROPIC_MODELS,
        default_parameters={"temperature": 0.7, "top_p": 0.9, "max_tokens": 1024},
    ),
    ProviderConfig(
        name="groq",
        models=DEFAULT_GROQ_MODELS,
        default_parameters={"temperature": 0.7, "top_p": 1.0, "max_tokens": 1024},
    ),
]

# Register default providers
for provider_config in DEFAULT_PROVIDERS:
    ModelRegistry.register_provider(provider_config)
