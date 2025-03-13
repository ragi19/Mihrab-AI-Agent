"""
Development tools for debugging, profiling and benchmarking
"""

import asyncio
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from ..core.agent import Agent
from ..core.message import Message
from .logging import get_logger

logger = get_logger("utils.dev_tools")


class AgentProfiler:
    """Profile agent performance and resource usage"""

    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "request_times": [],
            "token_counts": [],
        }

    async def profile_agent(
        self,
        agent: Agent,
        messages: List[Message],
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Dict[str, Any]:
        """Profile agent performance processing messages

        Args:
            agent: Agent to profile
            messages: Messages to process
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries

        Returns:
            Profiling metrics
        """
        logger.info(f"Starting agent profiling with {len(messages)} messages")

        for message in messages:
            attempt = 0
            while attempt < max_retries:
                try:
                    start_time = time.time()
                    response = await agent.process_message(message)
                    elapsed = time.time() - start_time

                    # Update metrics
                    self.metrics["total_requests"] += 1
                    self.metrics["total_time"] += elapsed
                    self.metrics["request_times"].append(elapsed)

                    # Count tokens if supported
                    if hasattr(agent.model, "count_tokens"):
                        message_tokens = await agent.model.count_tokens(message.content)
                        response_tokens = await agent.model.count_tokens(
                            response.content
                        )
                        total_tokens = message_tokens + response_tokens
                        self.metrics["total_tokens"] += total_tokens
                        self.metrics["token_counts"].append(total_tokens)

                    logger.debug(
                        f"Message processed in {elapsed:.2f}s "
                        f"(tokens: {total_tokens if 'total_tokens' in locals() else 'N/A'})"
                    )
                    break

                except Exception as e:
                    attempt += 1
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt} failed, retrying in {retry_delay}s: {e}"
                        )
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(f"Max retries exceeded: {e}")
                        raise

        # Calculate summary statistics
        if self.metrics["request_times"]:
            self.metrics["avg_request_time"] = self.metrics["total_time"] / len(
                self.metrics["request_times"]
            )
            self.metrics["min_request_time"] = min(self.metrics["request_times"])
            self.metrics["max_request_time"] = max(self.metrics["request_times"])

        if self.metrics["token_counts"]:
            self.metrics["avg_tokens"] = self.metrics["total_tokens"] / len(
                self.metrics["token_counts"]
            )
            self.metrics["min_tokens"] = min(self.metrics["token_counts"])
            self.metrics["max_tokens"] = max(self.metrics["token_counts"])

        logger.info("Profiling completed")
        return self.metrics


def async_retry(
    max_retries: int = 3, retry_delay: float = 1.0, exceptions: tuple = (Exception,)
) -> Callable:
    """Decorator for retrying async functions

    Args:
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries in seconds
        exceptions: Exceptions to catch and retry

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed, "
                            f"retrying in {retry_delay}s: {e}"
                        )
                        await asyncio.sleep(retry_delay)

            logger.error(f"Max retries exceeded: {last_error}")
            raise last_error

        return wrapper

    return decorator
