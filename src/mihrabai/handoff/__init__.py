"""
Handoff module for multi-agent systems

This module provides components for building multi-agent systems with handoff capabilities,
allowing agents to transfer control to other agents based on user needs or conversation context.
"""

from .agent import HandoffAgent
from .conditions import (
    complexity_based_condition,
    keyword_based_condition,
    sentiment_based_condition,
    topic_based_condition,
)
from .config import HandoffConfig, HandoffInputData
from .filters import (
    preserve_context,
    preserve_user_messages_only,
    remove_sensitive_information,
    summarize_previous_responses,
)

__all__ = [
    "HandoffAgent",
    "HandoffConfig",
    "HandoffInputData",
    # Filters
    "preserve_user_messages_only",
    "summarize_previous_responses",
    "remove_sensitive_information",
    "preserve_context",
    # Conditions
    "keyword_based_condition",
    "complexity_based_condition",
    "sentiment_based_condition",
    "topic_based_condition",
]
