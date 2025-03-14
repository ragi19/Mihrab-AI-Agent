"""
Factory functions for creating model instances
"""

from typing import Any, Dict, Optional, Set, Type, Union

from ..config import config
from ..utils.logging import get_logger
from .base import BaseModel, ModelCapability, ModelError
from .config import ModelConfig, ModelRegistry
from .provider_registry import ProviderError, ProviderRegistry

logger = get_logger("models.factory")


class ModelCreationError(ModelError):
    """Raised when model creation fails"""

    pass


class ModelFactory:
    """Factory for creating model instances"""

    @classmethod
    async def create_model(
        cls,
        model_name: str,
        provider_name: Optional[str] = None,
        provider_kwargs: Optional[Dict[str, Any]] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        required_capabilities: Optional[Set[ModelCapability]] = None,
        model_config: Optional[Union[ModelConfig, Dict[str, Any]]] = None,
    ) -> BaseModel:
        """Create a model instance with the specified configuration

        Args:
            model_name: Name of the model to create
            provider_name: Name of the provider to use (default: from config or auto-detect)
            provider_kwargs: Additional arguments for provider initialization
            model_parameters: Model-specific parameters
            required_capabilities: Set of capabilities the model must support
            model_config: Optional model configuration to use or register

        Returns:
            Configured model instance

        Raises:
            ModelCreationError: If model creation fails
            ProviderError: If provider configuration is invalid
        """
        return await create_model(
            model_name=model_name,
            provider_name=provider_name,
            provider_kwargs=provider_kwargs,
            model_parameters=model_parameters,
            required_capabilities=required_capabilities,
            model_config=model_config,
        )

    async def create(
        self,
        model_name: str,
        provider_name: Optional[str] = None,
        required_capabilities: Optional[Set[ModelCapability]] = None,
        trace_provider=None,
    ) -> BaseModel:
        """Create a model instance

        Args:
            model_name: Name of the model to create
            provider_name: Name of the provider to use (default: auto-detect)
            required_capabilities: Set of required capabilities for the model
            trace_provider: Optional trace provider for monitoring

        Returns:
            Configured model instance
        """
        return await create_model(
            model_name=model_name,
            provider_name=provider_name,
            required_capabilities=required_capabilities,
        )


async def create_model(
    model_name: str,
    provider_name: Optional[str] = None,
    provider_kwargs: Optional[Dict[str, Any]] = None,
    model_parameters: Optional[Dict[str, Any]] = None,
    required_capabilities: Optional[Set[ModelCapability]] = None,
    model_config: Optional[Union[ModelConfig, Dict[str, Any]]] = None,
) -> BaseModel:
    """Create a model instance with the specified configuration

    Args:
        model_name: Name of the model to create
        provider_name: Name of the provider to use (default: from config or auto-detect)
        provider_kwargs: Additional arguments for provider initialization
        model_parameters: Model-specific parameters
        required_capabilities: Set of capabilities the model must support
        model_config: Optional model configuration to use or register

    Returns:
        Configured model instance

    Raises:
        ModelCreationError: If model creation fails
        ProviderError: If provider configuration is invalid
    """
    try:
        # Auto-detect provider if not specified
        if not provider_name:
            provider_name = _resolve_provider(model_name)
            logger.info(f"Auto-detected provider: {provider_name}")

        # Register model config if provided
        if model_config and provider_name:
            if isinstance(model_config, dict):
                # Ensure model_name is set
                model_config_dict = dict(model_config)
                model_config_dict["model_name"] = model_name
                ProviderRegistry.register_model_config(provider_name, model_config_dict)
            else:
                # Ensure model_name and provider_name are set correctly
                if model_config.model_name != model_name:
                    model_config = ModelConfig(
                        model_name=model_name,
                        provider_name=provider_name,
                        **model_config.dict(exclude={"model_name", "provider_name"}),
                    )
                elif model_config.provider_name != provider_name:
                    model_config = ModelConfig(
                        model_name=model_name,
                        provider_name=provider_name,
                        **model_config.dict(exclude={"provider_name"}),
                    )
                ProviderRegistry.register_model_config(provider_name, model_config)

        # Get provider configuration
        provider_config = _get_provider_config(provider_name)

        # Validate provider configuration
        ProviderRegistry.validate_provider_config(provider_name, provider_config)

        # Merge provider kwargs with config
        final_provider_kwargs = provider_config.copy()
        if provider_kwargs:
            final_provider_kwargs.update(provider_kwargs)

        # Create model instance
        model = await ProviderRegistry.create_model(
            provider_name=provider_name, model_name=model_name, **final_provider_kwargs
        )

        # Validate required capabilities
        if required_capabilities:
            _validate_capabilities(model, required_capabilities)

        # Set model parameters
        _configure_model_parameters(model, model_parameters, provider_config)

        return model

    except (ModelError, ProviderError) as e:
        # Re-raise known errors
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating model: {e}", exc_info=True)
        raise ModelCreationError(f"Failed to create model: {e}")


def _resolve_provider(model_name: str) -> str:
    """Resolve which provider to use for a model

    Args:
        model_name: Name of the model

    Returns:
        Provider name to use

    Raises:
        ModelCreationError: If no suitable provider found
    """
    # First check ModelRegistry for provider
    provider_name = ModelRegistry.find_provider_for_model(model_name)
    if provider_name:
        return provider_name

    # Then try ProviderRegistry
    provider_name = ProviderRegistry.find_provider_for_model(model_name)
    if provider_name:
        return provider_name

    # Check if model name starts with provider prefix
    if model_name.startswith("llama") or model_name.startswith("mixtral"):
        return "groq"
    elif model_name.startswith("gpt"):
        return "openai"
    elif model_name.startswith("claude"):
        return "anthropic"

    # Fall back to default provider from config
    default_provider = config.get_default_provider()
    if default_provider:
        return default_provider

    raise ModelCreationError(f"Could not determine provider for model {model_name}")


def _get_provider_config(provider_name: str) -> Dict[str, Any]:
    """Get configuration for a provider

    Args:
        provider_name: Name of the provider

    Returns:
        Provider configuration

    Raises:
        ModelCreationError: If provider config not found
    """
    try:
        # Try to get from ModelRegistry first
        provider_config = ModelRegistry.get_provider_config(provider_name)
        if provider_config:
            # Convert to dict for compatibility
            config_dict = provider_config.dict()
            # Extract auth params
            auth_params = provider_config.get_auth_params()
            if auth_params:
                config_dict.update(auth_params)
            return config_dict

        # Fall back to config
        return config.get_provider_config(provider_name)
    except Exception as e:
        raise ModelCreationError(f"Invalid provider configuration: {e}")


def _validate_capabilities(model: BaseModel, required: Set[ModelCapability]) -> None:
    """Validate that a model supports required capabilities

    Args:
        model: Model instance to validate
        required: Set of required capabilities

    Raises:
        ModelCreationError: If model doesn't support required capabilities
    """
    # First check model_info on the instance
    if hasattr(model, "model_info") and model.model_info:
        if hasattr(model.model_info, "capabilities"):
            missing = required - model.model_info.capabilities
            if missing:
                raise ModelCreationError(
                    f"Model {model.model_name} missing required capabilities: {missing}"
                )
            return

    # Fall back to ModelRegistry
    model_config = ModelRegistry.get_model_config(model.model_name, model.provider_name)
    if model_config and model_config.capabilities:
        missing = required - set(model_config.capabilities)
        if missing:
            raise ModelCreationError(
                f"Model {model.model_name} missing required capabilities: {missing}"
            )
        return

    # If we can't verify capabilities, log a warning
    logger.warning(
        f"Cannot verify capabilities for model {model.model_name}. "
        f"Required: {required}"
    )


def _configure_model_parameters(
    model: BaseModel,
    parameters: Optional[Dict[str, Any]],
    provider_config: Dict[str, Any],
) -> None:
    """Configure model parameters

    Args:
        model: Model instance to configure
        parameters: Model-specific parameters to set
        provider_config: Provider configuration

    Raises:
        ModelError: If parameter validation fails
    """
    if parameters:
        model.update_parameters(parameters)
    elif "default_parameters" in provider_config:
        model.update_parameters(provider_config["default_parameters"])

    # Check if we have model-specific defaults in ModelRegistry
    else:
        model_config = ModelRegistry.get_model_config(
            model.model_name, model.provider_name
        )
        if model_config and model_config.parameters:
            model.update_parameters(model_config.parameters)


async def create_model_from_config(model_config: ModelConfig) -> BaseModel:
    """Create a model instance from a ModelConfig

    Args:
        model_config: Configuration for the model

    Returns:
        Configured model instance

    Raises:
        ModelCreationError: If model creation fails
    """
    return await create_model(
        model_name=model_config.model_name,
        provider_name=model_config.provider_name,
        model_parameters=model_config.parameters,
        model_config=model_config,
    )
