"""
Type definitions for the handoff system
"""

from typing import Any, Dict, List, Optional

from ..core.message import Message


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


class HandoffOutputData:
    """
    Data structure for handoff outputs

    This class encapsulates the data returned from a handoff operation,
    including the response, updated conversation history, and metadata.
    """

    def __init__(
        self,
        response: str,
        conversation_history: List[Message],
        metadata: Optional[Dict[str, Any]] = None,
        handling_agent: Optional[str] = None,
    ):
        """
        Initialize handoff output data

        Args:
            response: The response text from the target agent
            conversation_history: Updated conversation history after handoff
            metadata: Optional metadata returned from the handoff
            handling_agent: Name of the agent that handled the request
        """
        self.response = response
        self.conversation_history = conversation_history
        self.metadata = metadata or {}
        self.handling_agent = handling_agent

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary

        Returns:
            Dictionary representation of the handoff output data
        """
        return {
            "response": self.response,
            "conversation_history": [
                msg.to_dict() for msg in self.conversation_history
            ],
            "metadata": self.metadata,
            "handling_agent": self.handling_agent,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HandoffOutputData":
        """
        Create from dictionary

        Args:
            data: Dictionary representation of handoff output data

        Returns:
            HandoffOutputData instance
        """
        # Convert dictionary messages to Message objects
        conversation_history = []
        for msg_dict in data.get("conversation_history", []):
            msg = Message.from_dict(msg_dict)
            conversation_history.append(msg)

        return cls(
            response=data.get("response", ""),
            conversation_history=conversation_history,
            metadata=data.get("metadata", {}),
            handling_agent=data.get("handling_agent"),
        ) 