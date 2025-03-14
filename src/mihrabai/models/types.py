"""
Core type definitions for model interfaces
"""

from enum import IntEnum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel


class ModelCapability(IntEnum):
    """Model capability flags"""

    COMPLETION = 1  # Basic text completion
    CHAT = 2  # Chat message handling
    STREAM = 3  # Streaming responses
    FUNCTION_CALLING = 4  # Function calling
    SYSTEM_MESSAGES = 5  # System messages/prompts
    TOKEN_COUNTING = 6  # Token counting
    EMBEDDINGS = 7  # Text embeddings
    TOOL_USE = 8  # Tool usage
    IMAGE_INPUT = 9  # Image input handling
    IMAGE_OUTPUT = 10  # Image generation
    CODE = 11  # Code generation/analysis
    RAG = 12  # Retrieval-augmented generation
    MULTI_MODAL = 13  # Multi-modal capabilities


class ModelConfig(BaseModel):
    """Model configuration and capabilities"""

    name: str
    provider: str
    capabilities: List[ModelCapability]
    max_tokens: Optional[int] = None
    token_limit: Optional[int] = None
    default_parameters: Dict[str, Any] = {}
    description: Optional[str] = None
    version: Optional[str] = None


class TokenCount(BaseModel):
    """Token usage statistics"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelInfo(BaseModel):
    """Model information"""

    name: str
    capabilities: Set[ModelCapability]
    max_tokens: int
    context_window: int
    supports_streaming: bool = False
    supports_functions: bool = False
    pricing_per_1k_tokens: Optional[float] = None
    provider_name: Optional[str] = None
