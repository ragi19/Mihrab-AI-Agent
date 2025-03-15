"""
Condition functions for handoffs
"""

from typing import Any, Callable, Dict, List, Optional
import re


def keyword_based_condition(
    keywords: List[str],
) -> Callable[[str, Dict[str, Any]], bool]:
    """
    Create a condition function based on keywords

    This function returns a condition function that checks if any of the
    specified keywords are present in the message.

    Args:
        keywords: List of keywords to check for

    Returns:
        Condition function that returns True if any keyword is found
    """

    def condition(message: str, context: Dict[str, Any]) -> bool:
        return any(kw.lower() in message.lower() for kw in keywords)

    return condition


def complexity_based_condition(
    threshold: int = 20, technical_terms: Optional[List[str]] = None
) -> Callable[[str, Dict[str, Any]], bool]:
    """
    Create a condition function based on message complexity

    This function returns a condition function that checks if the message
    exceeds a certain complexity threshold, based on word count and
    the presence of technical terms.

    Args:
        threshold: Word count threshold
        technical_terms: List of technical terms to check for

    Returns:
        Condition function that returns True if message is complex
    """
    technical_terms = technical_terms or [
        "api",
        "database",
        "server",
        "code",
        "programming",
        "algorithm",
        "network",
        "configuration",
        "integration",
        "deployment",
    ]

    def condition(message: str, context: Dict[str, Any]) -> bool:
        # Count technical terms
        term_count = sum(1 for term in technical_terms if term in message.lower())

        # Check message length as a proxy for complexity
        is_complex = term_count >= 2 or len(message.split()) > threshold

        return is_complex

    return condition


def sentiment_based_condition(
    negative_terms: Optional[List[str]] = None,
) -> Callable[[str, Dict[str, Any]], bool]:
    """
    Create a condition function based on message sentiment

    This function returns a condition function that checks if the message
    contains negative sentiment, based on the presence of negative terms.

    Args:
        negative_terms: List of negative terms to check for

    Returns:
        Condition function that returns True if message has negative sentiment
    """
    negative_terms = negative_terms or [
        "angry",
        "frustrated",
        "disappointed",
        "unhappy",
        "terrible",
        "awful",
        "horrible",
        "bad",
        "worst",
        "complaint",
        "issue",
        "problem",
    ]

    def condition(message: str, context: Dict[str, Any]) -> bool:
        # Count negative terms
        negative_count = sum(1 for term in negative_terms if term in message.lower())

        # Check if message has negative sentiment
        return negative_count >= 2

    return condition


def topic_based_condition(
    topics: Dict[str, List[str]],
) -> Callable[[str, Dict[str, Any]], bool]:
    """
    Create a condition function based on message topic

    This function returns a condition function that checks if the message
    is related to a specific topic, based on the presence of topic-related terms.

    Args:
        topics: Dictionary mapping topic names to lists of related terms

    Returns:
        Condition function that returns True if message is related to the topic
    """

    def condition(message: str, context: Dict[str, Any]) -> bool:
        message_lower = message.lower()

        # Check if message contains terms related to the topic
        for topic, terms in topics.items():
            if context.get("target_topic") == topic:
                term_count = sum(1 for term in terms if term in message_lower)
                if term_count >= 2:
                    return True

        return False

    return condition


def conversation_history_condition(
    threshold: int = 3,
    max_history: int = 5,
) -> Callable[[str, Dict[str, Any]], bool]:
    """
    Create a condition function based on conversation history

    This function returns a condition function that checks if the conversation
    has reached a certain length or complexity, indicating it might benefit
    from a handoff to a specialist.

    Args:
        threshold: Number of back-and-forth exchanges to trigger handoff
        max_history: Maximum number of messages to consider

    Returns:
        Condition function that returns True if conversation is complex
    """

    def condition(message: str, context: Dict[str, Any]) -> bool:
        # Get conversation history from context
        history = context.get("conversation_history", [])
        
        # Only consider recent history
        recent_history = history[-max_history:] if len(history) > max_history else history
        
        # Count exchanges (pairs of user and assistant messages)
        exchanges = len(recent_history) // 2
        
        # Check if we've had enough back-and-forth to warrant a handoff
        return exchanges >= threshold

    return condition


def intent_change_condition() -> Callable[[str, Dict[str, Any]], bool]:
    """
    Create a condition function that detects changes in user intent

    This function returns a condition function that checks if the user's intent
    has changed significantly from previous messages, indicating a potential
    need for handoff to a different specialist.

    Returns:
        Condition function that returns True if user intent has changed
    """

    def condition(message: str, context: Dict[str, Any]) -> bool:
        # Get conversation history from context
        history = context.get("conversation_history", [])
        
        # Need at least one previous user message to compare
        if len(history) < 2:
            return False
            
        # Get the previous user message
        prev_user_messages = [msg for msg in history if msg.get("role") == "user"]
        if not prev_user_messages:
            return False
            
        prev_message = prev_user_messages[-1].get("content", "")
        
        # Define intent categories and their keywords
        intent_categories = {
            "question": ["what", "how", "why", "when", "where", "who", "?"],
            "request": ["can you", "please", "could you", "would you"],
            "complaint": ["problem", "issue", "not working", "doesn't work", "failed"],
            "information": ["fyi", "just so you know", "for your information"],
            "gratitude": ["thanks", "thank you", "appreciate"],
        }
        
        # Determine previous intent
        prev_intent = None
        prev_score = 0
        for intent, keywords in intent_categories.items():
            score = sum(1 for kw in keywords if kw.lower() in prev_message.lower())
            if score > prev_score:
                prev_intent = intent
                prev_score = score
                
        # Determine current intent
        current_intent = None
        current_score = 0
        for intent, keywords in intent_categories.items():
            score = sum(1 for kw in keywords if kw.lower() in message.lower())
            if score > current_score:
                current_intent = intent
                current_score = score
                
        # Check if intent has changed
        return prev_intent != current_intent and prev_intent is not None and current_intent is not None

    return condition


def expertise_boundary_condition(
    expertise_areas: Dict[str, List[str]],
    confidence_threshold: float = 0.7,
) -> Callable[[str, Dict[str, Any]], bool]:
    """
    Create a condition function that detects when a query falls outside an agent's expertise

    This function returns a condition function that checks if the message
    contains topics outside the agent's expertise areas, indicating a need
    for handoff to a more appropriate specialist.

    Args:
        expertise_areas: Dictionary mapping expertise areas to related terms
        confidence_threshold: Threshold for confidence in handling the query

    Returns:
        Condition function that returns True if query is outside expertise
    """

    def condition(message: str, context: Dict[str, Any]) -> bool:
        message_lower = message.lower()
        
        # Get the agent's expertise areas
        agent_expertise = context.get("agent_expertise", [])
        
        # Calculate confidence for each expertise area
        confidence_scores = {}
        for area, terms in expertise_areas.items():
            # Skip areas not in agent's expertise
            if area not in agent_expertise:
                continue
                
            # Count matching terms
            matches = sum(1 for term in terms if term in message_lower)
            
            # Calculate confidence score (simple version)
            total_terms = len(terms)
            confidence = matches / total_terms if total_terms > 0 else 0
            
            confidence_scores[area] = confidence
            
        # Get the highest confidence score
        max_confidence = max(confidence_scores.values()) if confidence_scores else 0
        
        # Check if confidence is below threshold
        return max_confidence < confidence_threshold

    return condition


def multi_agent_collaboration_condition(
    task_complexity_threshold: int = 3,
) -> Callable[[str, Dict[str, Any]], bool]:
    """
    Create a condition function that detects when a task requires multiple agents

    This function returns a condition function that checks if the message
    contains a complex task that would benefit from collaboration between
    multiple specialized agents.

    Args:
        task_complexity_threshold: Threshold for task complexity

    Returns:
        Condition function that returns True if task requires collaboration
    """

    def condition(message: str, context: Dict[str, Any]) -> bool:
        message_lower = message.lower()
        
        # Define indicators of multi-agent tasks
        multi_agent_indicators = [
            # Tasks requiring multiple steps or domains
            "and then", "after that", "followed by",
            # Tasks with multiple distinct parts
            "first", "second", "finally",
            # Tasks spanning multiple domains
            "both", "as well as", "in addition to",
        ]
        
        # Count indicators
        indicator_count = sum(1 for indicator in multi_agent_indicators if indicator in message_lower)
        
        # Check for explicit requests for collaboration
        collaboration_requests = [
            "work together", "collaborate", "need both", "need multiple",
        ]
        has_collaboration_request = any(req in message_lower for req in collaboration_requests)
        
        # Check if task complexity exceeds threshold or explicit collaboration is requested
        return indicator_count >= task_complexity_threshold or has_collaboration_request

    return condition


def context_aware_condition(
    context_keys: List[str],
    required_keys_count: int = 1,
) -> Callable[[str, Dict[str, Any]], bool]:
    """
    Create a condition function that makes decisions based on conversation context

    This function returns a condition function that checks if specific context
    information is available and should influence the handoff decision.

    Args:
        context_keys: List of context keys to check for
        required_keys_count: Minimum number of required keys to trigger handoff

    Returns:
        Condition function that returns True if context suggests handoff
    """

    def condition(message: str, context: Dict[str, Any]) -> bool:
        # Count how many required context keys are present
        present_keys = sum(1 for key in context_keys if key in context)
        
        # Check if we have enough context information to make a decision
        return present_keys >= required_keys_count

    return condition
