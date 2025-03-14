"""
Model adapter for providing a consistent interface across different model providers
"""

import asyncio
import inspect
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, AsyncIterator, Dict, List, NoReturn, Optional, Set, Type, Union

from ..core.message import Message
from ..core.types import ModelParameters
from ..utils.logging import get_logger
from .base import BaseModel, ModelCapability, ModelError, ModelInfo
from .config import ModelConfig

logger = get_logger("models.adapter")


class ModelAdapter(ABC):
    """Base adapter for model providers

    This class provides a consistent interface for interacting with different
    model providers, handling provider-specific implementation details.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the adapter

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger(f"adapter.{self.__class__.__name__}")
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the adapter

        This should be called before using the adapter to ensure
        any necessary setup is performed.

        Raises:
            ModelError: If initialization fails
        """
        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        """Check if adapter is initialized"""
        return self._initialized

    @abstractmethod
    async def get_available_models(self) -> List[str]:
        """Get list of available models from this provider

        Returns:
            List of model identifiers
        """
        raise NotImplementedError()

    @abstractmethod
    async def get_model_info(self, model_id: str) -> ModelInfo:
        """Get information about a specific model

        Args:
            model_id: Model identifier

        Returns:
            ModelInfo object containing model details

        Raises:
            ModelError: If model info cannot be retrieved
        """
        raise NotImplementedError()

    @abstractmethod
    async def validate_model(self, model_id: str, capabilities: Set[str]) -> bool:
        """Validate that a model exists and supports required capabilities

        Args:
            model_id: Model identifier
            capabilities: Set of required capabilities

        Returns:
            True if model is valid, False otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    async def create_model(self, model_id: str, config: ModelConfig) -> BaseModel:
        """Create a model instance

        Args:
            model_id: Model identifier
            config: Model configuration

        Returns:
            Model instance

        Raises:
            ModelError: If model cannot be created
        """
        raise NotImplementedError()

    @abstractmethod
    async def chat(
        self,
        model: BaseModel,
        messages: List[Message],
        parameters: Optional[ModelParameters] = None,
    ) -> Message:
        """Generate a chat response

        Args:
            model: Model instance
            messages: List of chat messages
            parameters: Optional model parameters

        Returns:
            Response message

        Raises:
            ModelError: If chat fails
        """
        raise NotImplementedError()

    @abstractmethod
    async def stream_chat(
        self,
        model: BaseModel,
        messages: List[Message],
        parameters: Optional[ModelParameters] = None,
    ) -> AsyncIterator[Message]:
        """Stream a chat response

        Args:
            model: Model instance
            messages: List of chat messages
            parameters: Optional model parameters

        Returns:
            AsyncIterator yielding response message chunks

        Raises:
            ModelError: If chat streaming fails
        """
        raise NotImplementedError()

    @abstractmethod
    async def complete(
        self,
        model: BaseModel,
        prompt: str,
        parameters: Optional[ModelParameters] = None,
    ) -> str:
        """Generate a completion response

        Args:
            model: Model instance
            prompt: Input prompt
            parameters: Optional model parameters

        Returns:
            Completion text

        Raises:
            ModelError: If completion fails
        """
        raise NotImplementedError()

    @abstractmethod
    async def stream_complete(
        self,
        model: BaseModel,
        prompt: str,
        parameters: Optional[ModelParameters] = None,
    ) -> AsyncIterator[str]:
        """Stream a completion response

        Args:
            model: Model instance
            prompt: Input prompt
            parameters: Optional model parameters

        Returns:
            AsyncIterator yielding completion text chunks

        Raises:
            ModelError: If completion streaming fails
        """
        raise NotImplementedError()

    @abstractmethod
    async def embed(
        self,
        model: BaseModel,
        text: Union[str, List[str]],
        parameters: Optional[ModelParameters] = None,
    ) -> List[List[float]]:
        """Generate embeddings

        Args:
            model: Model instance
            text: Input text or list of texts
            parameters: Optional model parameters

        Returns:
            List of embedding vectors

        Raises:
            ModelError: If embedding fails
        """
        raise NotImplementedError()

    @abstractmethod
    async def close(self) -> None:
        """Close the adapter and cleanup resources"""
        raise NotImplementedError()

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """Set a configuration value

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value

    @abstractmethod
    def supports_model(self, model_name: str) -> bool:
        """Check if this adapter supports a specific model

        Args:
            model_name: Name of the model

        Returns:
            True if model is supported, False otherwise
        """
        pass

    @abstractmethod
    def get_model_capabilities(self, model_name: str) -> Set[ModelCapability]:
        """Get capabilities for a specific model

        Args:
            model_name: Name of the model

        Returns:
            Set of capabilities

        Raises:
            ModelError: If model is not supported
        """
        pass

    @classmethod
    def register_models(cls, models: Dict[str, ModelConfig]) -> None:
        """Register models with the adapter

        This is a hook for adapter implementations to register
        their supported models with the model registry.

        Args:
            models: Dictionary mapping model names to configurations
        """
        # This is a hook for adapter implementations
        pass

    def require_init(self, func: Any) -> Any:
        """Decorator to ensure adapter is initialized before method call

        Args:
            func: Function to wrap

        Returns:
            Wrapped function that checks initialization

        Raises:
            ModelError: If adapter is not initialized
        """
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(self, *args: Any, **kwargs: Any) -> Any:
                if not self.is_initialized:
                    raise ModelError("Adapter not initialized")
                return await func(self, *args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(self, *args: Any, **kwargs: Any) -> Any:
                if not self.is_initialized:
                    raise ModelError("Adapter not initialized")
                return func(self, *args, **kwargs)

            return sync_wrapper

    def __repr__(self) -> str:
        """Get string representation of adapter

        Returns:
            String representation
        """
        return f"{self.__class__.__name__}(config={self.config})"


class CachedModelAdapter(ModelAdapter):
    """Model adapter with caching support

    This adapter wraps another adapter and caches model instances
    to avoid recreating them unnecessarily.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the cached adapter

        Args:
            adapter: Base adapter to wrap
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.adapter = adapter
        self.models: Dict[str, BaseModel] = {}

    async def initialize(self) -> None:
        """Initialize the adapter

        This initializes the underlying adapter.

        Raises:
            ModelError: If initialization fails
        """
        await self.adapter.initialize()
        self._initialized = True

    async def get_available_models(self) -> List[str]:
        """Get list of available models

        Returns:
            List of model identifiers
        """
        return await self.adapter.get_available_models()

    async def get_model_info(self, model_id: str) -> ModelInfo:
        """Get information about a specific model

        Args:
            model_id: Model identifier

        Returns:
            ModelInfo object containing model details

        Raises:
            ModelError: If model info cannot be retrieved
        """
        return await self.adapter.get_model_info(model_id)

    async def validate_model(self, model_id: str, capabilities: Set[str]) -> bool:
        """Validate that a model exists and supports required capabilities

        Args:
            model_id: Model identifier
            capabilities: Set of required capabilities

        Returns:
            True if model is valid, False otherwise
        """
        return await self.adapter.validate_model(model_id, capabilities)

    async def create_model(self, model_id: str, config: ModelConfig) -> BaseModel:
        """Create or get a cached model instance

        Args:
            model_id: Model identifier
            config: Model configuration

        Returns:
            Model instance

        Raises:
            ModelError: If model cannot be created
        """
        if model_id not in self.models:
            self.models[model_id] = await self.adapter.create_model(model_id, config)
        return self.models[model_id]

    async def chat(
        self,
        model: BaseModel,
        messages: List[Message],
        parameters: Optional[ModelParameters] = None,
    ) -> Message:
        """Generate a chat response

        Args:
            model: Model instance
            messages: List of chat messages
            parameters: Optional model parameters

        Returns:
            Response message

        Raises:
            ModelError: If chat fails
        """
        return await self.adapter.chat(model, messages, parameters)

    async def stream_chat(
        self,
        model: BaseModel,
        messages: List[Message],
        parameters: Optional[ModelParameters] = None,
    ) -> AsyncIterator[Message]:
        """Stream a chat response

        Args:
            model: Model instance
            messages: List of chat messages
            parameters: Optional model parameters

        Returns:
            AsyncIterator yielding response message chunks

        Raises:
            ModelError: If chat streaming fails
        """
        async for chunk in self.adapter.stream_chat(model, messages, parameters):
            yield chunk

    async def complete(
        self,
        model: BaseModel,
        prompt: str,
        parameters: Optional[ModelParameters] = None,
    ) -> str:
        """Generate a completion response

        Args:
            model: Model instance
            prompt: Input prompt
            parameters: Optional model parameters

        Returns:
            Completion text

        Raises:
            ModelError: If completion fails
        """
        return await self.adapter.complete(model, prompt, parameters)

    async def stream_complete(
        self,
        model: BaseModel,
        prompt: str,
        parameters: Optional[ModelParameters] = None,
    ) -> AsyncIterator[str]:
        """Stream a completion response

        Args:
            model: Model instance
            prompt: Input prompt
            parameters: Optional model parameters

        Returns:
            AsyncIterator yielding completion text chunks

        Raises:
            ModelError: If completion streaming fails
        """
        async for chunk in self.adapter.stream_complete(model, prompt, parameters):
            yield chunk

    async def embed(
        self,
        model: BaseModel,
        text: Union[str, List[str]],
        parameters: Optional[ModelParameters] = None,
    ) -> List[List[float]]:
        """Generate embeddings

        Args:
            model: Model instance
            text: Input text or list of texts
            parameters: Optional model parameters

        Returns:
            List of embedding vectors

        Raises:
            ModelError: If embedding fails
        """
        return await self.adapter.embed(model, text, parameters)

    async def close(self) -> None:
        """Close the adapter and cleanup resources

        This closes all cached models and the underlying adapter.
        """
        for model in self.models.values():
            await model.close()
        self.models.clear()
        await self.adapter.close()


class AdapterRegistry:
    """Registry for model adapters"""

    _adapters: Dict[str, Type[ModelAdapter]] = {}
    _initialized_adapters: Dict[str, ModelAdapter] = {}

    @classmethod
    def register_adapter(cls, name: str, adapter_class: Type[ModelAdapter]) -> None:
        """Register a new adapter

        Args:
            name: Adapter identifier
            adapter_class: Adapter class
        """
        cls._adapters[name] = adapter_class
        logger.info(f"Registered adapter: {name}")

    @classmethod
    def get_adapter_class(cls, name: str) -> Optional[Type[ModelAdapter]]:
        """Get an adapter class by name

        Args:
            name: Name of the adapter

        Returns:
            Adapter class if found, None otherwise
        """
        return cls._adapters.get(name)

    @classmethod
    def get_adapter(
        cls, name: str, config: Optional[Dict[str, Any]] = None
    ) -> Optional[ModelAdapter]:
        """Get or create an adapter instance

        Args:
            name: Name of the adapter
            config: Optional configuration for the adapter

        Returns:
            Adapter instance if found, None otherwise
        """
        if name not in cls._adapters:
            return None

        # Create new instance if not cached or config provided
        if name not in cls._initialized_adapters or config is not None:
            adapter_class = cls._adapters[name]
            adapter = adapter_class(config)
            cls._initialized_adapters[name] = adapter
            return adapter

        return cls._initialized_adapters[name]

    @classmethod
    def list_adapters(cls) -> List[str]:
        """List all registered adapters

        Returns:
            List of adapter names
        """
        return list(cls._adapters.keys())

    @classmethod
    def clear_adapters(cls) -> None:
        """Clear all initialized adapters"""
        cls._initialized_adapters.clear()
        logger.info("Cleared adapter cache")


def requires_initialization(func):
    """Decorator to ensure adapter is initialized before method call"""

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not self.is_initialized:
            await self.initialize()
        return await func(self, *args, **kwargs)

    return wrapper
