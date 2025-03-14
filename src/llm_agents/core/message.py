"""
Message data structures for agent communication
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


@dataclass
class Message:
    role: MessageRole
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format"""
        result: Dict[str, Any] = {"role": self.role.value, "content": self.content}

        if self.metadata:
            result["metadata"] = self.metadata

        # Include timestamp in ISO format
        result["timestamp"] = self.timestamp.isoformat()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create a Message from a dictionary"""
        role = MessageRole(data["role"])
        content = data["content"]
        metadata = data.get("metadata")

        # Parse timestamp if present
        timestamp = None
        if "timestamp" in data:
            timestamp = datetime.fromisoformat(data["timestamp"])

        # Create message with or without timestamp
        if timestamp:
            return cls(
                role=role, content=content, metadata=metadata, timestamp=timestamp
            )
        else:
            return cls(role=role, content=content, metadata=metadata)
