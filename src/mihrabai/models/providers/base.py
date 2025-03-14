"""
Base provider interface definition
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Type, cast

from ...core.message import Message
from ...core.types import ModelParameters
from ...utils.logging import get_logger
from ..base import BaseModel, ModelError, ModelInfo
from ..config import ModelCapability as ConfigModelCapability
from ..config import ModelConfig


class ProviderError(ModelError):
    """Base class for provider-related errors"""

    pass


class BaseProvider(ABC):
    """Base interface for model providers"""

    # Supported models and their capabilities - should be overridden by subclasses
    SUPPORTED_MODELS: Dict[str, ModelInfo] = {}

    def __init__(self, **kwargs: Any) -> None:
        """Initialize provider with configuration"""
        self._config = kwargs
        self._models: Dict[str, Type[BaseModel]] = {}
        self._model_info: Dict[str, ModelInfo] = {}
        self.logger = get_logger(f"models.providers.{self.__class__.__name__.lower()}")

    @abstractmethod
    async def create_model(self, model_name: str) -> BaseModel:
        """Create a model instance"""
        pass

    def register_model(
        self, model_name: str, model_class: Type[BaseModel], model_info: ModelInfo
    ) -> None:
        """Register a model with this provider"""
        self._models[model_name] = model_class
        self._model_info[model_name] = model_info
        self.logger.info(f"Registered model: {model_name}")

    def get_default_parameters(self, model_name: str) -> Dict[str, Any]:
        """Get default parameters for a model"""
        default_params = self._config.get("default_parameters", {})
        if not isinstance(default_params, dict):
            return {}
        return default_params

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        if model_name not in self._model_info:
            return None

        model_info = self._model_info[model_name]
        # Convert string capabilities to ModelCapability enum values
        capabilities = set()
        for cap in model_info.capabilities:
            try:
                # Try to convert string capability to enum
                capabilities.add(ConfigModelCapability[cap.upper()])
            except (KeyError, AttributeError):
                # Skip capabilities that don't match the enum
                pass

        return ModelConfig(
            model_name=model_name,
            provider_name=self.__class__.__name__.replace("Provider", "").lower(),
            capabilities=capabilities,
            context_window=model_info.context_window,
            max_tokens=model_info.max_tokens,
            supports_streaming="STREAMING"
            in [str(c).upper() for c in model_info.capabilities],
            supports_functions="FUNCTION_CALLING"
            in [str(c).upper() for c in model_info.capabilities],
        )

    @classmethod
    def register_models(cls) -> Dict[str, ModelConfig]:
        """Register supported models and their configurations"""
        configs = {}
        for model_name, model_info in cls.SUPPORTED_MODELS.items():
            # Convert string capabilities to ModelCapability enum values
            capabilities = set()
            for cap in model_info.capabilities:
                try:
                    # Try to convert string capability to enum
                    capabilities.add(ConfigModelCapability[cap.upper()])
                except (KeyError, AttributeError):
                    # Skip capabilities that don't match the enum
                    pass

            config = ModelConfig(
                model_name=model_name,
                provider_name=cls.__name__.replace("Provider", "").lower(),
                capabilities=capabilities,
                context_window=model_info.context_window,
                max_tokens=model_info.max_tokens,
                supports_streaming="STREAMING"
                in [str(c).upper() for c in model_info.capabilities],
                supports_functions="FUNCTION_CALLING"
                in [str(c).upper() for c in model_info.capabilities],
            )
            configs[model_name] = config
        return configs

    @property
    def supported_models(self) -> Dict[str, ModelConfig]:
        """Get dictionary of supported models"""
        return {
            model_name: config
            for model_name in self._models
            if (config := self.get_model_config(model_name)) is not None
        }

    @classmethod
    def get_required_config_keys(cls) -> Set[str]:
        """Get required configuration keys"""
        return {"api_key"}

    async def validate_api_key(self) -> bool:
        """Validate the provider API key"""
        # Default implementation - subclasses should override this
        return True
