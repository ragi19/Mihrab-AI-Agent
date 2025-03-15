"""
LLM Agents - A framework for building and running LLM-powered agents
"""

__version__ = "0.2.0"

from .core import Agent, ChatAgent, Message, MessageRole
from .core import MemoryEnabledTaskAgent as MemoryTaskAgent, TaskAgent
from .models import BaseModel, ModelRegistry, ProviderRegistry
from .models.provider_registry import ProviderRegistry as _ProviderRegistry

# Import create_model from provider_registry
create_model = _ProviderRegistry.create_model
from .config import config
from .factory import create_agent, create_memory_task_agent, create_task_agent
from .handoff import HandoffAgent, HandoffConfig, HandoffInputData
from .runtime import AgentRunner, AgentCoordinator, RuntimeContext
from .tools import BaseTool, ToolRegistry

__all__ = [
    # Core
    "Agent",
    "ChatAgent",
    "TaskAgent",
    "MemoryTaskAgent",
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
    "AgentCoordinator",
    # Tools
    "BaseTool",
    "ToolRegistry",
    # Factory
    "create_agent",
    "create_task_agent",
    "create_memory_task_agent",
    # Config
    "config",
    # Handoff
    "HandoffAgent",
    "HandoffConfig",
    "HandoffInputData",
]
