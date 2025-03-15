"""
Handoff module for Mihrab AI Agent.

This module provides functionality for handling agent handoffs.
"""

from .agent import HandoffAgent
from .config import HandoffConfig
from .types import HandoffInputData, HandoffOutputData

# Import conditions
from .conditions import (
    keyword_based_condition,
    complexity_based_condition,
    sentiment_based_condition,
    topic_based_condition,
    conversation_history_condition,
    intent_change_condition,
    expertise_boundary_condition,
    multi_agent_collaboration_condition,
    context_aware_condition,
)

# Import filters
from .filters import (
    preserve_user_messages_only,
    summarize_previous_responses,
    remove_sensitive_information,
    preserve_context,
    extract_key_information,
    prioritize_messages,
    add_handoff_context,
    transform_message_format,
    merge_related_messages,
    add_feedback_loop,
)

# Import model selection
from .model_selection import (
    ModelSelectionStrategy,
    PrioritizedModelStrategy,
    CapabilityBasedStrategy,
    RoundRobinStrategy,
    ModelSelectionManager,
    select_model,
    create_model_config,
)

__all__ = [
    # Handoff core
    "HandoffAgent",
    "HandoffConfig",
    "HandoffInputData",
    "HandoffOutputData",
    
    # Conditions
    "keyword_based_condition",
    "complexity_based_condition",
    "sentiment_based_condition",
    "topic_based_condition",
    "conversation_history_condition",
    "intent_change_condition",
    "expertise_boundary_condition",
    "multi_agent_collaboration_condition",
    "context_aware_condition",
    
    # Filters
    "preserve_user_messages_only",
    "summarize_previous_responses",
    "remove_sensitive_information",
    "preserve_context",
    "extract_key_information",
    "prioritize_messages",
    "add_handoff_context",
    "transform_message_format",
    "merge_related_messages",
    "add_feedback_loop",
    
    # Model Selection
    "ModelSelectionStrategy",
    "PrioritizedModelStrategy",
    "CapabilityBasedStrategy",
    "RoundRobinStrategy",
    "ModelSelectionManager",
    "select_model",
    "create_model_config",
]
