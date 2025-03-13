"""
Core type definitions used throughout the framework
"""

from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union

# Type aliases for common patterns
JSON = Dict[str, Any]
AsyncCallable = Callable[..., Awaitable[Any]]
ModelParameters = Dict[str, Union[str, int, float, bool]]

# Generic type variables
T = TypeVar("T")
R = TypeVar("R")


class ModelResponse:
    """Response from a model"""

    def __init__(
        self,
        content: str,
        model_id: str,
        provider_id: str,
        usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        function_calls: Optional[List[Dict[str, Any]]] = None,
    ):
        self.content = content
        self.model_id = model_id
        self.provider_id = provider_id
        self.usage = usage or {}
        self.metadata = metadata or {}
        self.function_calls = function_calls or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "usage": self.usage,
            "metadata": self.metadata,
            "function_calls": self.function_calls,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelResponse":
        """Create from dictionary"""
        return cls(
            content=data.get("content", ""),
            model_id=data.get("model_id", ""),
            provider_id=data.get("provider_id", ""),
            usage=data.get("usage"),
            metadata=data.get("metadata"),
            function_calls=data.get("function_calls"),
        )
