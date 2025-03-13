"""
Configuration classes for the handoff system
"""

from typing import Dict, Any, List, Optional, Callable, TYPE_CHECKING

from ..core.message import Message

if TYPE_CHECKING:
    from .agent import HandoffAgent


class HandoffInputData:
    """
    Data structure for handoff inputs

    This class encapsulates the data passed between agents during a handoff,
    including conversation history, system message, and metadata.
    """

    def __init__(
        self,
        conversation_history: List[Message],
        system_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_agent: Optional[str] = None,
        handoff_chain: Optional[List[str]] = None,
    ):
        """
        Initialize handoff input data

        Args:
            conversation_history: List of messages in the conversation
            system_message: Optional system message to override the target agent's default
            metadata: Optional metadata to pass to the target agent
            source_agent: Optional name of the source agent initiating the handoff
            handoff_chain: Optional list of agent names in the handoff chain
        """
        self.conversation_history = conversation_history
        self.system_message = system_message
        self.metadata = metadata or {}
        self.source_agent = source_agent
        self.handoff_chain = handoff_chain or []

    def add_to_chain(self, agent_name: str) -> None:
        """
        Add an agent to the handoff chain

        Args:
            agent_name: Name of the agent to add to the chain
        """
        if agent_name not in self.handoff_chain:
            self.handoff_chain.append(agent_name)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary

        Returns:
            Dictionary representation of the handoff input data
        """
        return {
            "conversation_history": [
                msg.to_dict() for msg in self.conversation_history
            ],
            "system_message": self.system_message,
            "metadata": self.metadata,
            "source_agent": self.source_agent,
            "handoff_chain": self.handoff_chain,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HandoffInputData":
        """
        Create from dictionary

        Args:
            data: Dictionary representation of handoff input data

        Returns:
            HandoffInputData instance
        """
        # Convert dictionary messages to Message objects
        conversation_history = []
        for msg_dict in data.get("conversation_history", []):
            msg = Message.from_dict(msg_dict)
            conversation_history.append(msg)

        return cls(
            conversation_history=conversation_history,
            system_message=data.get("system_message"),
            metadata=data.get("metadata", {}),
            source_agent=data.get("source_agent"),
            handoff_chain=data.get("handoff_chain", []),
        )


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
