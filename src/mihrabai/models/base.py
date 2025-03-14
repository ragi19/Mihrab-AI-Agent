"""
Base model definitions for LLM Agents
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Final, List, Optional, Set, Union

from ..core.message import Message
from ..core.types import ModelParameters


class ModelCapability:
    """Capabilities that a model may support"""

    CHAT: Final[str] = "chat"
    COMPLETION: Final[str] = "completion"
    EMBEDDING: Final[str] = "embedding"
    FUNCTION_CALLING: Final[str] = "function_calling"
    VISION: Final[str] = "vision"
    STREAMING: Final[str] = "streaming"
    STREAM: Final[str] = "streaming"  # Alias for STREAMING
    IMAGE_INPUT: Final[str] = "image_input"


class ModelInfo:
    """Information about a model"""

    def __init__(
        self,
        id: str,
        name: str,
        provider: str,
        capabilities: Set[str],
        context_window: int,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.name = name
        self.provider = provider
        self.capabilities = capabilities
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.metadata = metadata or {}


class ModelError(Exception):
    """Base exception for model-related errors"""

    pass


class TokenLimitError(ModelError):
    """Exception raised when token limits are exceeded"""

    pass


class BaseModel(ABC):
    """Base class for all models

    This abstract class defines the interface that all models must implement.
    """

    def __init__(self, model_id: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the model

        Args:
            model_id: The identifier for this model
            config: Optional configuration for the model
        """
        self.model_id = model_id
        self.config = config or {}
        self._capabilities = {ModelCapability.CHAT, ModelCapability.STREAMING}

    @property
    @abstractmethod
    def capabilities(self) -> Set[str]:
        """Get the capabilities of this model"""
        pass

    @abstractmethod
    async def generate(self, messages: List[Message], **kwargs: Any) -> Message:
        """Generate a response from the model

        Args:
            messages: The conversation history
            **kwargs: Additional parameters to pass to the model

        Returns:
            The model's response message

        Raises:
            ModelError: If generation fails
        """
        pass

    @abstractmethod
    async def generate_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> AsyncIterator[Message]:
        """Stream a response from the model

        Args:
            messages: The conversation history
            **kwargs: Additional parameters to pass to the model

        Returns:
            An async iterator of message chunks

        Raises:
            ModelError: If generation fails
        """
        pass

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value

        Args:
            key: The configuration key
            default: Default value if key is not found

        Returns:
            The configuration value or default
        """
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """Set a configuration value

        Args:
            key: The configuration key
            value: The value to set
        """
        self.config[key] = value
