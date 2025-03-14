"""
Error recovery and retry mechanisms for agent execution
"""

import asyncio
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from ..core.message import Message
from ..utils.logging import get_logger

T = TypeVar("T")
logger = get_logger("runtime.recovery")


class RetryConfig:
    """Configuration for retry behavior"""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        retry_exceptions: Optional[List[Type[Exception]]] = None,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.retry_exceptions = retry_exceptions or [Exception]

    def should_retry(self, exception: Exception) -> bool:
        """Determine if the given exception should trigger a retry"""
        return any(
            isinstance(exception, exc_type) for exc_type in self.retry_exceptions
        )

    def get_delay(self, attempt: int) -> float:
        """Calculate the delay for a given retry attempt"""
        delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
        return min(delay, self.max_delay)


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator for adding retry behavior to async functions

    Args:
        config: Retry configuration, or None to use defaults
    """
    retry_config = config or RetryConfig()

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0

            while True:
                attempt += 1
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if (
                        attempt > retry_config.max_retries
                        or not retry_config.should_retry(e)
                    ):
                        raise

                    delay = retry_config.get_delay(attempt)
                    logger.warning(
                        f"Retry {attempt}/{retry_config.max_retries} "
                        f"after error: {e}. Waiting {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)

        return wrapper

    return decorator


class CircuitBreaker:
    """Circuit breaker pattern implementation to prevent cascading failures"""

    CLOSED = "closed"  # Normal operation, requests go through
    OPEN = "open"  # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service is back online

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        self.logger = get_logger("runtime.circuit_breaker")

    def record_success(self) -> None:
        """Record a successful operation"""
        if self.state == self.HALF_OPEN:
            self.logger.info("Success in half-open state, closing circuit")
            self.state = self.CLOSED

        self.failure_count = 0
        self.half_open_calls = 0

    def record_failure(self) -> None:
        """Record a failed operation"""
        self.last_failure_time = time.time()

        if self.state == self.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.logger.warning(
                    f"Failure threshold reached ({self.failure_count}), opening circuit"
                )
                self.state = self.OPEN

        elif self.state == self.HALF_OPEN:
            self.logger.info("Failure in half-open state, reopening circuit")
            self.state = self.OPEN

    def allow_request(self) -> bool:
        """Check if a request should be allowed through"""
        if self.state == self.CLOSED:
            return True

        elif self.state == self.OPEN:
            # Check if recovery timeout has elapsed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.logger.info(
                    f"Recovery timeout elapsed ({self.recovery_timeout}s), "
                    f"moving to half-open state"
                )
                self.state = self.HALF_OPEN
                self.half_open_calls = 0
                return self.allow_request()
            return False

        elif self.state == self.HALF_OPEN:
            # Allow limited number of test requests
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False

        return False


def with_circuit_breaker(circuit_breaker: CircuitBreaker):
    """Decorator for adding circuit breaker behavior to async functions

    Args:
        circuit_breaker: Circuit breaker instance to use
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not circuit_breaker.allow_request():
                logger.warning(
                    f"Circuit breaker open, failing fast for {func.__name__}"
                )
                raise RuntimeError(
                    f"Circuit breaker open: Too many failures for {func.__name__}"
                )

            try:
                result = await func(*args, **kwargs)
                circuit_breaker.record_success()
                return result

            except Exception as e:
                circuit_breaker.record_failure()
                raise

        return wrapper

    return decorator


class RecoveryStrategy:
    """Base class for recovery strategies"""

    async def recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt to recover from an error

        Args:
            error: The exception that occurred
            context: Contextual information about the error

        Returns:
            True if recovery was successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement recover")


class FallbackStrategy(RecoveryStrategy):
    """Recovery strategy that falls back to an alternative implementation"""

    def __init__(self, fallback_fn: Callable[..., Any]):
        self.fallback_fn = fallback_fn
        self.logger = get_logger("recovery.fallback")

    async def recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover by calling the fallback function"""
        self.logger.info(f"Attempting fallback recovery for {error}")
        try:
            if asyncio.iscoroutinefunction(self.fallback_fn):
                await self.fallback_fn(**context)
            else:
                self.fallback_fn(**context)
            return True
        except Exception as e:
            self.logger.error(f"Fallback recovery failed: {e}", exc_info=True)
            return False


class ModelSwitchStrategy(RecoveryStrategy):
    """Recovery strategy that switches to a different model"""

    def __init__(self, fallback_model_provider: str, fallback_model_name: str):
        from ..models import create_model

        self.fallback_model_provider = fallback_model_provider
        self.fallback_model_name = fallback_model_name
        self.create_model = create_model
        self.logger = get_logger("recovery.model_switch")

    async def recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover by switching to a fallback model"""
        self.logger.info(
            f"Attempting model switch recovery from {error} "
            f"to {self.fallback_model_provider}/{self.fallback_model_name}"
        )

        try:
            # Get the agent from context
            agent = context.get("agent")
            if not agent:
                self.logger.error("No agent found in context")
                return False

            # Create fallback model
            fallback_model = await self.create_model(
                self.fallback_model_provider, self.fallback_model_name
            )

            # Replace the model in the agent
            original_model = agent.model
            agent.model = fallback_model

            self.logger.info(
                f"Switched model from {original_model.model_name} to {fallback_model.model_name}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Model switch recovery failed: {e}", exc_info=True)
            return False


class MessageRetryStrategy(RecoveryStrategy):
    """Recovery strategy that retries with a modified message"""

    def __init__(
        self, message_transformer: Callable[[Message], Message], max_retries: int = 3
    ):
        self.message_transformer = message_transformer
        self.max_retries = max_retries
        self.logger = get_logger("recovery.message_retry")

    async def recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover by retrying with a transformed message"""
        self.logger.info(f"Attempting message retry recovery for {error}")

        try:
            # Get agent and message from context
            agent = context.get("agent")
            message = context.get("message")

            if not agent or not message:
                self.logger.error("Missing agent or message in context")
                return False

            # Transform message and retry
            for attempt in range(1, self.max_retries + 1):
                try:
                    transformed_message = self.message_transformer(message)
                    self.logger.info(
                        f"Retry attempt {attempt} with transformed message"
                    )

                    # Process the transformed message
                    await agent.process_message(transformed_message)
                    return True

                except Exception as retry_error:
                    self.logger.warning(
                        f"Retry attempt {attempt} failed: {retry_error}"
                    )

            return False

        except Exception as e:
            self.logger.error(f"Message retry recovery failed: {e}", exc_info=True)
            return False


class CompositeRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy that combines multiple strategies"""

    def __init__(self, strategies: List[RecoveryStrategy]):
        self.strategies = strategies
        self.logger = get_logger("recovery.composite")

    async def recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Try each recovery strategy in sequence"""
        for i, strategy in enumerate(self.strategies):
            self.logger.debug(f"Trying recovery strategy {i+1}/{len(self.strategies)}")
            try:
                if await strategy.recover(error, context):
                    self.logger.info(f"Recovery strategy {i+1} succeeded")
                    return True
            except Exception as e:
                self.logger.warning(f"Recovery strategy {i+1} failed with error: {e}")

        self.logger.error("All recovery strategies failed")
        return False


class RecoveryManager:
    """Manager for applying recovery strategies to errors"""

    def __init__(self):
        self.strategies: Dict[Type[Exception], RecoveryStrategy] = {}
        self.default_strategy: Optional[RecoveryStrategy] = None
        self.logger = get_logger("recovery.manager")

    def register_strategy(
        self, exception_type: Type[Exception], strategy: RecoveryStrategy
    ) -> None:
        """Register a recovery strategy for a specific exception type"""
        self.strategies[exception_type] = strategy
        self.logger.info(f"Registered recovery strategy for {exception_type.__name__}")

    def set_default_strategy(self, strategy: RecoveryStrategy) -> None:
        """Set the default recovery strategy"""
        self.default_strategy = strategy
        self.logger.info("Set default recovery strategy")

    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle an error using the appropriate recovery strategy

        Args:
            error: The exception that occurred
            context: Contextual information about the error

        Returns:
            True if recovery was successful, False otherwise
        """
        # Find the most specific matching strategy
        for exc_type, strategy in self.strategies.items():
            if isinstance(error, exc_type):
                self.logger.info(f"Using recovery strategy for {exc_type.__name__}")
                return await strategy.recover(error, context)

        # Fall back to default strategy if available
        if self.default_strategy:
            self.logger.info(
                f"Using default recovery strategy for {type(error).__name__}"
            )
            return await self.default_strategy.recover(error, context)

        self.logger.warning(f"No recovery strategy found for {type(error).__name__}")
        return False


def recoverable(recovery_manager: RecoveryManager = None):
    """Decorator for making a function recoverable

    Args:
        recovery_manager: Recovery manager to use, or None to create a new one
    """
    manager = recovery_manager or RecoveryManager()

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Create context with available information
                context = {
                    "function": func.__name__,
                    "args": args,
                    "kwargs": kwargs,
                    "error": e,
                }

                # Try to recover
                if await manager.handle_error(e, context):
                    # If recovery succeeded, try again
                    return await func(*args, **kwargs)

                # If recovery failed, re-raise the original error
                raise

        return wrapper

    return decorator


class ConversationRecovery:
    """Handles recovery of conversation state after errors"""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        fallback_message: str = "I apologize, but I encountered an issue processing your request.",
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.fallback_message = fallback_message
        self.recovery_strategies: List[RecoveryStrategy] = []

    def add_strategy(self, strategy: "RecoveryStrategy") -> None:
        """Add a recovery strategy

        Args:
            strategy: Strategy to add
        """
        self.recovery_strategies.append(strategy)

    async def recover(
        self, error: Exception, context: Dict[str, Any]
    ) -> Optional[Message]:
        """Attempt to recover from an error

        Args:
            error: The exception that occurred
            context: Context information about the conversation

        Returns:
            Recovery message if successful, None if recovery failed
        """
        logger.warning(f"Attempting conversation recovery after error: {error}")

        # Try each strategy in order
        for strategy in self.recovery_strategies:
            try:
                success = await strategy.recover(error, context)
                if success:
                    logger.info(
                        f"Recovery successful using {strategy.__class__.__name__}"
                    )

                    # If context contains a response, return it
                    if "response" in context:
                        return context["response"]

                    # Otherwise create a generic success message
                    return Message(
                        role="assistant",
                        content="I've resolved the issue and can continue our conversation.",
                    )
            except Exception as e:
                logger.error(
                    f"Recovery strategy {strategy.__class__.__name__} failed: {e}"
                )

        # If all strategies fail, return fallback message
        logger.error("All recovery strategies failed")
        return Message(role="assistant", content=self.fallback_message)

    async def retry_with_backoff(
        self, func: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """Retry a function with exponential backoff

        Args:
            func: Function to retry
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the function if successful

        Raises:
            Exception: If all retries fail
        """
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                attempt += 1
                last_error = e

                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"Retry {attempt}/{self.max_retries} "
                        f"after error: {e}. Waiting {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)

        # If we get here, all retries failed
        if last_error:
            raise last_error
        raise RuntimeError("All retries failed")
