"""
Model registry for managing model configurations
"""

from typing import Dict, Optional, Type

from ..utils.logging import get_logger
from .types import ModelConfig

logger = get_logger("models.registry")


class ModelRegistry:
    """Central registry for model configurations"""

    _models: Dict[str, ModelConfig] = {}
    _providers: Dict[str, Type] = {}

    @classmethod
    def register(
        cls, model_name: str, config: ModelConfig, provider_cls: Optional[Type] = None
    ) -> None:
        """Register a model configuration"""
        cls._models[model_name] = config
        if provider_cls:
            cls._providers[model_name] = provider_cls
        logger.info(f"Registered model: {model_name}")

    @classmethod
    def get_config(cls, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a model"""
        return cls._models.get(model_name)

    @classmethod
    def get_provider(cls, model_name: str) -> Optional[Type]:
        """Get provider class for a model"""
        return cls._providers.get(model_name)

    @classmethod
    def list_models(cls) -> Dict[str, ModelConfig]:
        """Get all registered models"""
        return cls._models.copy()

    @classmethod
    def clear(cls) -> None:
        """Clear all registered models"""
        cls._models.clear()
        cls._providers.clear()
        logger.info("Cleared model registry")
