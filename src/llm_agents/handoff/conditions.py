"""
Condition functions for handoffs
"""

from typing import Any, Callable, Dict, List, Optional


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
