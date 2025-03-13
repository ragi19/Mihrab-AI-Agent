"""
LLM Agents - A framework for building and running LLM-powered agents
"""

__version__ = "0.1.0"

from .core import Agent, ChatAgent, Message, MessageRole
from .models import BaseModel, ModelRegistry, ProviderRegistry
from .models.provider_registry import ProviderRegistry as _ProviderRegistry

# Import create_model from provider_registry
create_model = _ProviderRegistry.create_model
from .config import config
from .factory import create_agent
from .handoff import HandoffAgent, HandoffConfig, HandoffInputData
from .runtime import AgentRunner, CoordinatedAgentGroup, RuntimeContext
from .tools import BaseTool, ToolRegistry

__all__ = [
    # Core
    "Agent",
    "ChatAgent",
    "Message",
    "MessageRole",
    # Models
    "BaseModel",
    "ProviderRegistry",
    "ModelRegistry",
    "create_model",
    # Runtime
    "RuntimeContext",
    "AgentRunner",
    "CoordinatedAgentGroup",
    # Tools
    "BaseTool",
    "ToolRegistry",
    # Factory
    "create_agent",
    # Config
    "config",
    # Handoff
    "HandoffAgent",
    "HandoffConfig",
    "HandoffInputData",
]
