"""
Mihrab AI Agent Framework

A flexible framework for building and deploying LLM-powered agents with multiple provider support,
inspired by the mihrab that guides prayer in a mosque.
"""

__version__ = "0.2.0"

# Core components
from mihrabai.core.message import Message, MessageRole
from mihrabai.core.agent import Agent
from mihrabai.core.task_agent import TaskAgent
from mihrabai.core.memory_task_agent import MemoryTaskAgent

# Factory functions
from mihrabai.factory import create_task_agent, create_memory_agent

# Handoff system
from mihrabai.handoff.agent import HandoffAgent
from mihrabai.handoff.config import HandoffConfig
from mihrabai.handoff.input_data import HandoffInputData

# Make commonly used components available at the top level
__all__ = [
    "Agent",
    "TaskAgent",
    "MemoryTaskAgent",
    "Message",
    "MessageRole",
    "create_task_agent",
    "create_memory_agent",
    "HandoffAgent",
    "HandoffConfig",
    "HandoffInputData",
]
