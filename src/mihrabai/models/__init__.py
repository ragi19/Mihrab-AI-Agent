"""
Models package exports
"""

from .base import BaseModel, ModelError, ModelInfo, TokenLimitError
from .factory import ModelFactory, create_model
from .provider_discovery import ProviderDiscovery
from .provider_registry import (
    ModelCreationError,
    ProviderInfo,
    ProviderRegistry,
    register_provider,
)
from .providers.base import BaseProvider, ProviderError
from .registry import ModelRegistry
from .types import ModelCapability, ModelConfig, TokenCount


# Add convenience functions for tests
def get_provider(name):
    """Get a provider by name"""
    return ProviderRegistry.get_provider(name)


def list_providers():
    """List all registered providers"""
    return ProviderRegistry.list_providers()


def list_models(provider_name=None):
    """List all models, optionally filtered by provider"""
    if provider_name:
        return ProviderRegistry.list_models_for_provider(provider_name)
    return ModelRegistry.list_models()


def register_model(provider_name, model_config):
    """Register a model configuration"""
    return ProviderRegistry.register_model_config(provider_name, model_config)


__all__ = [
    "BaseModel",
    "BaseProvider",
    "ModelCapability",
    "ModelInfo",
    "ModelError",
    "ProviderError",
    "TokenLimitError",
    "TokenCount",
    "ModelConfig",
    "ModelRegistry",
    "ProviderDiscovery",
    "ProviderRegistry",
    "ProviderInfo",
    "ModelCreationError",
    "create_model",
    "ModelFactory",
    # Added convenience functions
    "get_provider",
    "list_providers",
    "list_models",
    "register_model",
    "register_provider",
]
