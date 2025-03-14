"""
Filter functions for handoffs
"""

from typing import List

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
    input_data: HandoffInputData, max_messages: int = 5
) -> HandoffInputData:
    """
    Filter that preserves the most recent messages for context

    This filter is useful when you want to preserve some context
    but don't want to transfer the entire conversation history.

    Args:
        input_data: The handoff input data to filter
        max_messages: Maximum number of messages to preserve

    Returns:
        Filtered handoff input data with limited context
    """
    # Get the most recent messages
    recent_messages = (
        input_data.conversation_history[-max_messages:]
        if input_data.conversation_history
        else []
    )

    return HandoffInputData(
        conversation_history=recent_messages,
        system_message=input_data.system_message,
        metadata=input_data.metadata,
        source_agent=input_data.source_agent,
        handoff_chain=input_data.handoff_chain,
    )
