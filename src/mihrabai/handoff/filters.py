"""
Filter functions for handoffs
"""

from typing import List, Dict, Any, Optional, Callable
import re

from ..core.message import Message, MessageRole
from .config import HandoffInputData


def preserve_user_messages_only(input_data: HandoffInputData) -> HandoffInputData:
    """
    Filter that keeps only user messages in the conversation history

    This filter is useful when you want to preserve only the user's queries
    and discard assistant responses when transferring to another agent.

    Args:
        input_data: The handoff input data to filter

    Returns:
        Filtered handoff input data
    """
    return HandoffInputData(
        conversation_history=[
            msg
            for msg in input_data.conversation_history
            if msg.role == MessageRole.USER
        ],
        system_message=input_data.system_message,
        metadata=input_data.metadata,
        source_agent=input_data.source_agent,
        handoff_chain=input_data.handoff_chain,
    )


def summarize_previous_responses(input_data: HandoffInputData) -> HandoffInputData:
    """
    Filter that summarizes previous assistant responses into a single context message

    This filter is useful when you want to preserve the essence of the conversation
    without transferring the entire conversation history.

    Args:
        input_data: The handoff input data to filter

    Returns:
        Filtered handoff input data with summarized assistant responses
    """
    user_messages = [
        msg for msg in input_data.conversation_history if msg.role == MessageRole.USER
    ]

    # Create a summary of assistant responses
    assistant_messages = [
        msg
        for msg in input_data.conversation_history
        if msg.role == MessageRole.ASSISTANT
    ]
    if assistant_messages:
        summary = "Previous assistant responses summary:\n"
        for i, msg in enumerate(assistant_messages):
            summary += f"- Response {i+1}: {msg.content[:100]}...\n"

        # Add a system message with the summary
        new_system = (
            input_data.system_message + "\n\n" + summary
            if input_data.system_message
            else summary
        )
    else:
        new_system = input_data.system_message

    return HandoffInputData(
        conversation_history=user_messages,
        system_message=new_system,
        metadata=input_data.metadata,
        source_agent=input_data.source_agent,
        handoff_chain=input_data.handoff_chain,
    )


def remove_sensitive_information(input_data: HandoffInputData) -> HandoffInputData:
    """
    Filter that removes sensitive information from the conversation history

    This filter is useful when transferring to an agent that shouldn't have
    access to sensitive information like credit card numbers, passwords, etc.

    Args:
        input_data: The handoff input data to filter

    Returns:
        Filtered handoff input data with sensitive information removed
    """
    import re

    # Patterns for sensitive information
    patterns = {
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+\d{1,2}\s?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "password": r"\b(?:password|pwd|passcode|pin)[\s:=]+\S+\b",
    }

    # Function to redact sensitive information
    def redact_text(text: str) -> str:
        for info_type, pattern in patterns.items():
            text = re.sub(pattern, f"[REDACTED {info_type.upper()}]", text)
        return text

    # Filter conversation history
    filtered_history = []
    for msg in input_data.conversation_history:
        filtered_content = redact_text(msg.content)
        filtered_msg = Message(role=msg.role, content=filtered_content)
        filtered_history.append(filtered_msg)

    # Filter system message if present
    filtered_system = (
        redact_text(input_data.system_message) if input_data.system_message else None
    )

    # Filter metadata
    filtered_metadata = input_data.metadata.copy()
    for key, value in input_data.metadata.items():
        if isinstance(value, str):
            filtered_metadata[key] = redact_text(value)

    return HandoffInputData(
        conversation_history=filtered_history,
        system_message=filtered_system,
        metadata=filtered_metadata,
        source_agent=input_data.source_agent,
        handoff_chain=input_data.handoff_chain,
    )


def preserve_context(
    max_messages: int = 5
) -> Callable[[HandoffInputData], HandoffInputData]:
    """
    Create a filter that preserves the most recent messages for context

    This filter is useful when you want to preserve some context
    but don't want to transfer the entire conversation history.

    Args:
        max_messages: Maximum number of messages to preserve

    Returns:
        A filter function that preserves context
    """
    def filter_function(input_data: HandoffInputData) -> HandoffInputData:
        """
        Filter that preserves the most recent messages for context

        Args:
            input_data: The handoff input data to filter

        Returns:
            Filtered handoff input data with preserved context
        """
        # If we have fewer messages than the max, return all of them
        if len(input_data.conversation_history) <= max_messages:
            return input_data

        # Otherwise, keep only the most recent messages
        preserved_history = input_data.conversation_history[-max_messages:]

        return HandoffInputData(
            conversation_history=preserved_history,
            system_message=input_data.system_message,
            metadata=input_data.metadata,
            source_agent=input_data.source_agent,
            handoff_chain=input_data.handoff_chain,
        )
    
    return filter_function


def extract_key_information(
    input_data: HandoffInputData, 
    key_patterns: Dict[str, str]
) -> HandoffInputData:
    """
    Filter that extracts key information from conversation history

    This filter identifies and extracts important information from the conversation
    based on provided patterns, and adds it to the metadata for the target agent.

    Args:
        input_data: The handoff input data to filter
        key_patterns: Dictionary mapping metadata keys to regex patterns

    Returns:
        Filtered handoff input data with extracted information in metadata
    """
    # Create a copy of the metadata
    enhanced_metadata = input_data.metadata.copy()
    
    # Extract information from all messages
    for msg in input_data.conversation_history:
        for key, pattern in key_patterns.items():
            matches = re.findall(pattern, msg.content)
            if matches:
                # Store the first match in metadata
                enhanced_metadata[key] = matches[0]
    
    return HandoffInputData(
        conversation_history=input_data.conversation_history,
        system_message=input_data.system_message,
        metadata=enhanced_metadata,
        source_agent=input_data.source_agent,
        handoff_chain=input_data.handoff_chain,
    )


def prioritize_messages(
    input_data: HandoffInputData,
    priority_keywords: List[str],
) -> HandoffInputData:
    """
    Filter that reorders messages to prioritize important ones

    This filter moves messages containing priority keywords to the front
    of the conversation history, ensuring they get more attention.

    Args:
        input_data: The handoff input data to filter
        priority_keywords: List of keywords indicating priority

    Returns:
        Filtered handoff input data with reordered messages
    """
    # Separate messages into priority and non-priority
    priority_messages = []
    normal_messages = []
    
    for msg in input_data.conversation_history:
        # Check if message contains any priority keywords
        if any(kw.lower() in msg.content.lower() for kw in priority_keywords):
            priority_messages.append(msg)
        else:
            normal_messages.append(msg)
    
    # Combine priority messages first, then normal messages
    reordered_history = priority_messages + normal_messages
    
    return HandoffInputData(
        conversation_history=reordered_history,
        system_message=input_data.system_message,
        metadata=input_data.metadata,
        source_agent=input_data.source_agent,
        handoff_chain=input_data.handoff_chain,
    )


def add_handoff_context(
    context_generator: Callable[[HandoffInputData], str],
) -> Callable[[HandoffInputData], HandoffInputData]:
    """
    Create a filter that adds contextual information about the handoff

    This filter adds a system message with context about why the handoff
    occurred and what the target agent should focus on.

    Args:
        context_generator: Function that generates context from input data

    Returns:
        A filter function that adds handoff context
    """
    def filter_function(input_data: HandoffInputData) -> HandoffInputData:
        # Generate context information
        context_info = context_generator(input_data)
        
        # Add context to system message
        enhanced_system = (
            input_data.system_message + "\n\n" + context_info
            if input_data.system_message
            else context_info
        )
        
        return HandoffInputData(
            conversation_history=input_data.conversation_history,
            system_message=enhanced_system,
            metadata=input_data.metadata,
            source_agent=input_data.source_agent,
            handoff_chain=input_data.handoff_chain,
        )
    
    return filter_function


def transform_message_format(
    input_data: HandoffInputData,
    format_transformer: Callable[[str], str],
) -> HandoffInputData:
    """
    Filter that transforms message format for the target agent

    This filter applies a transformation function to each message,
    which can be useful for adapting content to different agent capabilities.

    Args:
        input_data: The handoff input data to filter
        format_transformer: Function that transforms message content

    Returns:
        Filtered handoff input data with transformed messages
    """
    # Transform each message
    transformed_history = []
    for msg in input_data.conversation_history:
        transformed_content = format_transformer(msg.content)
        transformed_msg = Message(role=msg.role, content=transformed_content)
        transformed_history.append(transformed_msg)
    
    return HandoffInputData(
        conversation_history=transformed_history,
        system_message=input_data.system_message,
        metadata=input_data.metadata,
        source_agent=input_data.source_agent,
        handoff_chain=input_data.handoff_chain,
    )


def merge_related_messages(
    input_data: HandoffInputData,
    time_threshold: int = 60,  # seconds
) -> HandoffInputData:
    """
    Filter that merges related messages from the same user

    This filter combines consecutive messages from the same user that
    were sent within a short time period, reducing fragmentation.

    Args:
        input_data: The handoff input data to filter
        time_threshold: Maximum time difference (in seconds) to merge messages

    Returns:
        Filtered handoff input data with merged messages
    """
    if not input_data.conversation_history:
        return input_data
        
    # Initialize merged history with the first message
    merged_history = [input_data.conversation_history[0]]
    
    # Merge consecutive messages from the same user
    for i in range(1, len(input_data.conversation_history)):
        current_msg = input_data.conversation_history[i]
        prev_msg = merged_history[-1]
        
        # Check if messages are from the same user and within time threshold
        same_user = current_msg.role == prev_msg.role
        
        # If timestamps are available, check time difference
        time_close = True
        if hasattr(current_msg, 'timestamp') and hasattr(prev_msg, 'timestamp'):
            time_diff = abs(current_msg.timestamp - prev_msg.timestamp)
            time_close = time_diff <= time_threshold
        
        if same_user and time_close:
            # Merge with previous message
            merged_content = prev_msg.content + "\n\n" + current_msg.content
            merged_msg = Message(role=prev_msg.role, content=merged_content)
            
            # Replace the previous message with the merged one
            merged_history[-1] = merged_msg
        else:
            # Add as a new message
            merged_history.append(current_msg)
    
    return HandoffInputData(
        conversation_history=merged_history,
        system_message=input_data.system_message,
        metadata=input_data.metadata,
        source_agent=input_data.source_agent,
        handoff_chain=input_data.handoff_chain,
    )


def add_feedback_loop(
    input_data: HandoffInputData,
    feedback_message: str = "Please provide feedback on this response to improve future handoffs.",
) -> HandoffInputData:
    """
    Filter that adds a feedback request to the conversation

    This filter adds a system message requesting feedback from the user
    about the handoff, which can be used to improve future handoffs.

    Args:
        input_data: The handoff input data to filter
        feedback_message: Message requesting feedback

    Returns:
        Filtered handoff input data with feedback request
    """
    # Add a system message requesting feedback
    feedback_msg = Message(role=MessageRole.SYSTEM, content=feedback_message)
    
    # Add to the end of conversation history
    updated_history = input_data.conversation_history.copy()
    updated_history.append(feedback_msg)
    
    # Add metadata flag indicating feedback is requested
    updated_metadata = input_data.metadata.copy()
    updated_metadata["feedback_requested"] = True
    
    return HandoffInputData(
        conversation_history=updated_history,
        system_message=input_data.system_message,
        metadata=updated_metadata,
        source_agent=input_data.source_agent,
        handoff_chain=input_data.handoff_chain,
    )
