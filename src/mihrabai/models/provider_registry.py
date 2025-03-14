"""
Registry for managing and accessing provider implementations
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from ..utils.logging import get_logger
from .base import BaseModel, ModelError
from .config import ModelConfig, ModelRegistry, ProviderConfig
from .types import ModelInfo

logger = get_logger("models.registry")


@dataclass
class ProviderInfo:
    """Information about a provider's capabilities"""

    name: str
    supported_models: Set[str]
    features: Set[str]
    requires_api_key: bool = True
    default_model: Optional[str] = None
    documentation_url: Optional[str] = None
    rate_limits: Optional[Dict[str, Any]] = None


class ProviderError(ModelError):
    """Base class for provider-related errors"""

    pass


class ModelCreationError(ProviderError):
    """Exception raised when model creation fails"""

    pass


class ProviderRegistry:
    """Registry for managing model providers"""

    _providers: Dict[str, Type[Any]] = {}
    _provider_info: Dict[str, ProviderInfo] = {}
    _initialized_providers: Dict[str, Any] = {}
    _provider_factories: Dict[str, Callable] = {}

    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: Type[Any],
        provider_info: ProviderInfo,
        provider_factory: Optional[Callable] = None,
    ) -> None:
        """Register a new provider implementation

        Args:
            name: Provider identifier
            provider_class: Provider class implementation
            provider_info: Provider capability information
            provider_factory: Optional factory function to create provider instances
        """
        cls._providers[name] = provider_class
        cls._provider_info[name] = provider_info

        if provider_factory:
            cls._provider_factories[name] = provider_factory

        # Register with ModelRegistry if not already registered
        if not ModelRegistry.get_provider_config(name):
            provider_config = ProviderConfig(
                name=name,
                models={
                    model_name: ModelConfig(
                        model_name=model_name,
                        provider_name=name,
                        # Default capabilities - providers should update these
                        capabilities=set(),
                    )
                    for model_name in provider_info.supported_models
                },
            )
            ModelRegistry.register_provider(provider_config)

        logger.info(
            f"Registered provider: {name} supporting {len(provider_info.supported_models)} models"
        )

    @classmethod
    def get_provider(cls, name: str) -> Optional[Type[Any]]:
        """Get a provider implementation by name"""
        provider = cls._providers.get(name)
        if provider:
            logger.debug(f"Retrieved provider: {name}")
        else:
            logger.warning(f"Provider not found: {name}")
        return provider

    @classmethod
    def get_provider_info(cls, name: str) -> Optional[ProviderInfo]:
        """Get information about a provider's capabilities"""
        return cls._provider_info.get(name)

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider names"""
        providers = list(cls._providers.keys())
        logger.debug(f"Available providers: {providers}")
        return providers

    @classmethod
    def list_models_for_provider(cls, provider_name: str) -> Set[str]:
        """List all models supported by a provider"""
        provider_info = cls.get_provider_info(provider_name)
        if provider_info:
            return provider_info.supported_models
        return set()

    @classmethod
    def find_provider_for_model(cls, model_name: str) -> Optional[str]:
        """Find which provider supports a given model"""
        # First check ModelRegistry for more detailed configuration
        provider_name = ModelRegistry.find_provider_for_model(model_name)
        if provider_name:
            return provider_name

        # Fall back to provider info
        for provider_name, info in cls._provider_info.items():
            if model_name in info.supported_models:
                return provider_name
        return None

    @classmethod
    async def create_model(
        cls, provider_name: str, model_name: str, **provider_kwargs
    ) -> BaseModel:
        """Create a model instance from a provider

        Args:
            provider_name: Name of the provider to use
            model_name: Name of the model to create
            **provider_kwargs: Additional arguments for provider initialization

        Returns:
            An instance of the requested model

        Raises:
            ProviderError: If provider/model creation fails
        """
        logger.info(f"Creating model {model_name} with provider {provider_name}")
        logger.debug(f"Provider kwargs: {provider_kwargs}")

        # Validate provider exists
        provider_class = cls.get_provider(provider_name)
        if not provider_class:
            raise ProviderError(f"Provider {provider_name} not found")

        # Check if model is supported
        model_config = ModelRegistry.get_model_config(model_name, provider_name)
        provider_info = cls.get_provider_info(provider_name)

        if model_config:
            # Use model config from registry
            pass
        elif not provider_info or model_name not in provider_info.supported_models:
            raise ProviderError(
                f"Model {model_name} not supported by provider {provider_name}"
            )

        try:
            # Get or create provider instance
            if provider_name not in cls._initialized_providers:
                # Use factory if available
                if provider_name in cls._provider_factories:
                    provider = cls._provider_factories[provider_name](**provider_kwargs)
                else:
                    provider = provider_class(**provider_kwargs)
                cls._initialized_providers[provider_name] = provider
            else:
                provider = cls._initialized_providers[provider_name]

            # Create and initialize model
            model = await provider.create_model(model_name)
            await model.initialize()

            # Update model config in registry if needed
            if model.model_info and model_config:
                updated_config = model_config.copy()
                if hasattr(model.model_info, "capabilities"):
                    updated_config.capabilities = model.model_info.capabilities
                if hasattr(model.model_info, "context_window"):
                    updated_config.context_window = model.model_info.context_window

                provider_config = ModelRegistry.get_provider_config(provider_name)
                if provider_config:
                    provider_config.models[model_name] = updated_config

            logger.info(f"Successfully created model {model_name}")
            return model

        except Exception as e:
            logger.error(f"Failed to create model: {e}", exc_info=True)
            raise ProviderError(f"Failed to create model: {e}")

    @classmethod
    def validate_provider_config(
        cls, provider_name: str, config: Dict[str, Any]
    ) -> None:
        """Validate provider configuration

        Args:
            provider_name: Name of provider to validate
            config: Provider configuration to validate

        Raises:
            ProviderError: If configuration is invalid
        """
        provider_info = cls.get_provider_info(provider_name)
        if not provider_info:
            raise ProviderError(f"Provider {provider_name} not found")

        # Check required API key
        if provider_info.requires_api_key and "api_key" not in config:
            raise ProviderError(f"Provider {provider_name} requires an API key")

        # Let provider class validate the rest
        provider_class = cls.get_provider(provider_name)
        if hasattr(provider_class, "validate_config"):
            try:
                provider_class.validate_config(config)
            except Exception as e:
                raise ProviderError(f"Invalid provider configuration: {e}")

    @classmethod
    def clear_provider_cache(cls) -> None:
        """Clear cached provider instances"""
        cls._initialized_providers.clear()
        logger.info("Cleared provider cache")

    @classmethod
    def register_model_config(
        cls, provider_name: str, model_config: Union[ModelConfig, Dict[str, Any]]
    ) -> None:
        """Register or update a model configuration

        Args:
            provider_name: Name of the provider
            model_config: Model configuration to register

        Raises:
            ProviderError: If provider not found
        """
        if provider_name not in cls._providers:
            raise ProviderError(f"Provider {provider_name} not found")

        # Convert dict to ModelConfig if needed
        if isinstance(model_config, dict):
            if "model_name" not in model_config:
                raise ProviderError("Model configuration missing 'model_name'")

            model_name = model_config["model_name"]
            config = ModelConfig(
                model_name=model_name,
                provider_name=provider_name,
                **{k: v for k, v in model_config.items() if k != "model_name"},
            )
        else:
            config = model_config
            model_name = config.model_name

        # Update provider info
        provider_info = cls._provider_info.get(provider_name)
        if provider_info:
            provider_info.supported_models.add(model_name)

        # Update model registry
        provider_config = ModelRegistry.get_provider_config(provider_name)
        if provider_config:
            provider_config.models[model_name] = config
        else:
            # Create new provider config
            provider_config = ProviderConfig(
                name=provider_name, models={model_name: config}
            )
            ModelRegistry.register_provider(provider_config)

        logger.info(
            f"Registered model configuration: {model_name} for provider {provider_name}"
        )


# Add module-level function that redirects to the class method
def register_provider(
    name: str,
    provider_class: Type[Any],
    provider_info: ProviderInfo,
    provider_factory: Optional[Callable] = None,
) -> None:
    """
    Module-level function that redirects to ProviderRegistry.register_provider

    Args:
        name: Provider identifier
        provider_class: Provider class implementation
        provider_info: Provider capability information
        provider_factory: Optional factory function to create provider instances
    """
    ProviderRegistry.register_provider(
        name=name,
        provider_class=provider_class,
        provider_info=provider_info,
        provider_factory=provider_factory,
    )
