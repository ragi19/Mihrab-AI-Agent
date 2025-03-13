"""
Message data structures for agent communication
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum
from datetime import datetime

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
        result = {
            "role": self.role.value,
            "content": self.content
        }
        
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result