"""
Common interfaces and base classes to avoid circular imports
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

from ..core.message import Message


@runtime_checkable
class AgentInterface(Protocol):
    """
    Protocol defining the interface for all agent types
    """
    name: str
    
    async def process(
        self,
        message: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """
        Process a message and return a response
        
        Args:
            message: The message to process
            session_id: Optional session identifier
            context: Optional context information
            
        Returns:
            Either a response string or a tuple of (response, context)
        """
        ...
        
    async def should_handle(self, message: Union[str, Message]) -> bool:
        """
        Determine if this agent should handle the given message
        
        Args:
            message: The message to check
            
        Returns:
            True if this agent should handle the message, False otherwise
        """
        ...
