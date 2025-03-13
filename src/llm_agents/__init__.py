"""
LLM Agents - A framework for building and running LLM-powered agents
"""

__version__ = "0.1.0"

from .core import (
    Agent,
    ChatAgent,
    Message,
    MessageRole
)
from .models import (
    BaseModel,
    ProviderRegistry,
    ModelRegistry
)
from .models.provider_registry import ProviderRegistry as _ProviderRegistry
# Import create_model from provider_registry
create_model = _ProviderRegistry.create_model
from .runtime import (
    RuntimeContext,
    AgentRunner,
    CoordinatedAgentGroup
)
from .tools import (
    BaseTool,
    ToolRegistry
)
from .factory import create_agent
from .config import config
from .handoff import (
    HandoffAgent,
    HandoffConfig,
    HandoffInputData
)

__all__ = [
    # Core
    'Agent',
    'ChatAgent',
    'Message',
    'MessageRole',
    
    # Models
    'BaseModel',
    'ProviderRegistry',
    'ModelRegistry',
    'create_model',
    
    # Runtime
    'RuntimeContext',
    'AgentRunner',
    'CoordinatedAgentGroup',
    
    # Tools
    'BaseTool',
    'ToolRegistry',
    
    # Factory
    'create_agent',
    
    # Config
    'config',
    
    # Handoff
    'HandoffAgent',
    'HandoffConfig',
    'HandoffInputData'
]