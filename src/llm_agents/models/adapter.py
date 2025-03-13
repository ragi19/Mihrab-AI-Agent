"""
Model adapter for providing a consistent interface across different model providers
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Set, Type, Union, AsyncIterator
import asyncio
import inspect
from functools import wraps

from ..core.message import Message
from ..core.types import ModelParameters
from ..utils.logging import get_logger
from .base import BaseModel, ModelCapability, ModelInfo, ModelError
from .config import ModelConfig

logger = get_logger("models.adapter")

class ModelAdapter(ABC):
    """Base adapter for model providers
    
    This class provides a consistent interface for interacting with different
    model providers, handling provider-specific implementation details.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
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
        pass
    
    @abstractmethod
    async def get_model_info(self, model_name: str) -> ModelInfo:
        """Get information about a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information
            
        Raises:
            ModelError: If model information cannot be retrieved
        """
        pass
    
    @abstractmethod
    async def create_model(self, model_name: str, **kwargs) -> BaseModel:
        """Create a model instance
        
        Args:
            model_name: Name of the model to create
            **kwargs: Additional model-specific parameters
            
        Returns:
            Initialized model instance
            
        Raises:
            ModelError: If model creation fails
        """
        pass
    
    @abstractmethod
    async def generate(
        self, 
        model_name: str, 
        messages: List[Message],
        **kwargs
    ) -> Message:
        """Generate a response using the specified model
        
        Args:
            model_name: Name of the model to use
            messages: List of conversation messages
            **kwargs: Additional generation parameters
            
        Returns:
            Model's response message
            
        Raises:
            ModelError: If generation fails
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        model_name: str,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[Message]:
        """Generate a streaming response using the specified model
        
        Args:
            model_name: Name of the model to use
            messages: List of conversation messages
            **kwargs: Additional generation parameters
            
        Yields:
            Partial response messages
            
        Raises:
            ModelError: If generation fails
        """
        pass
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
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


class CachedModelAdapter(ModelAdapter):
    """Adapter that caches model instances and information"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the adapter
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self._model_cache: Dict[str, BaseModel] = {}
        self._model_info_cache: Dict[str, ModelInfo] = {}
        
    async def get_model(self, model_name: str, **kwargs) -> BaseModel:
        """Get a cached model instance or create a new one
        
        Args:
            model_name: Name of the model
            **kwargs: Additional model parameters
            
        Returns:
            Model instance
            
        Raises:
            ModelError: If model creation fails
        """
        cache_key = f"{model_name}:{hash(frozenset(kwargs.items()))}"
        
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        model = await self.create_model(model_name, **kwargs)
        self._model_cache[cache_key] = model
        return model
    
    async def get_model_info(self, model_name: str) -> ModelInfo:
        """Get cached model information or fetch it
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information
            
        Raises:
            ModelError: If model information cannot be retrieved
        """
        if model_name in self._model_info_cache:
            return self._model_info_cache[model_name]
        
        # Implementation should override this to fetch info
        # This is just a placeholder
        raise NotImplementedError("Subclasses must implement get_model_info")
    
    def clear_cache(self) -> None:
        """Clear all cached models and information"""
        self._model_cache.clear()
        self._model_info_cache.clear()
        self.logger.info("Cleared model cache")
    
    def remove_from_cache(self, model_name: str) -> None:
        """Remove a specific model from cache
        
        Args:
            model_name: Name of the model to remove
        """
        # Remove model instances
        keys_to_remove = []
        for key in self._model_cache:
            if key.startswith(f"{model_name}:"):
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self._model_cache[key]
            
        # Remove model info
        if model_name in self._model_info_cache:
            del self._model_info_cache[model_name]
            
        self.logger.info(f"Removed model {model_name} from cache")


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
    def get_adapter(cls, name: str, config: Optional[Dict[str, Any]] = None) -> Optional[ModelAdapter]:
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
