"""
Configuration classes for the handoff system
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..core.message import Message
from .types import HandoffInputData

if TYPE_CHECKING:
    from .agent import HandoffAgent


class HandoffConfig:
    """
    Configuration for agent handoffs

    This class defines when and how to transfer control between agents,
    including the target agent, condition function, and input filter.
    """

    def __init__(
        self,
        name: str,
        description: str,
        target_agent: "HandoffAgent",
        condition: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        input_filter: Optional[Callable[[HandoffInputData], HandoffInputData]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize handoff configuration

        Args:
            name: Name of the handoff configuration
            description: Description of when this handoff should be used
            target_agent: The agent to hand off to
            condition: Optional function to determine if handoff should occur
            input_filter: Optional function to filter conversation history
            metadata: Optional metadata to include with the handoff
        """
        self.name = name
        self.description = description
        self.target_agent = target_agent
        self.condition = condition  # Custom condition function
        self.input_filter = input_filter  # Custom filter function
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary

        Returns:
            Dictionary representation of the handoff configuration
        """
        return {
            "name": self.name,
            "description": self.description,
            "target_agent": self.target_agent.name,
            "has_condition": self.condition is not None,
            "has_input_filter": self.input_filter is not None,
            "metadata": self.metadata,
        }
