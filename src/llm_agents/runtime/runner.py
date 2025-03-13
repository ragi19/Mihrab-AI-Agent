"""
Agent runner implementation for managing agent execution
"""

import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..core.agent import Agent
from ..core.message import Message
from ..utils.logging import get_logger
from ..utils.tracing import Trace, TraceProvider
from .context import RuntimeContext


class AgentRunner:
    """Runner for managing agent execution and conversation flow"""

    def __init__(
        self,
        agent: Agent,
        context: Optional[RuntimeContext] = None,
        trace_provider: Optional[TraceProvider] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        error_handler: Optional[
            Callable[[Exception, Message, int], Awaitable[Optional[Message]]]
        ] = None,
    ):
        self.agent = agent
        self.context = context or RuntimeContext()
        self.context.max_retries = max_retries
        self.trace_provider = trace_provider or TraceProvider()
        self.retry_delay = retry_delay
        self.error_handler = error_handler
        self.logger = get_logger("runtime.runner")
        self._current_trace: Optional[Trace] = None
        self.logger.info(
            f"Initialized runner with agent type: {agent.__class__.__name__}"
        )

    async def run(self, message: Message) -> Message:
        """Run a single message through the agent with retry logic

        Args:
            message: Message to process

        Returns:
            Agent's response message
        """
        self.logger.debug(f"Processing message: {message}")
        self._current_trace = self.trace_provider.create_trace(
            name="agent_runner.run", group_id=str(id(self))
        )
        self._current_trace.start()

        try:
            # Set execution context
            self.context.start_processing()
            self.logger.debug("Started message processing")

            # Process message with retry logic
            retries = 0
            last_error = None

            while retries <= self.context.max_retries:
                try:
                    # Process message
                    response = await self.agent.process_message(message)
                    self.logger.debug(f"Generated response: {response}")
                    return response

                except Exception as e:
                    retries += 1
                    last_error = e
                    self.logger.warning(
                        f"Error processing message (attempt {retries}/{self.context.max_retries + 1}): {e}"
                    )

                    # Record error in context
                    self.context.record_error(e)

                    # Try custom error handler if provided
                    if self.error_handler:
                        try:
                            custom_response = await self.error_handler(
                                e, message, retries
                            )
                            if custom_response:
                                return custom_response
                        except Exception as handler_error:
                            self.logger.error(f"Error handler failed: {handler_error}")

                    # If we've reached max retries, raise the error
                    if retries > self.context.max_retries:
                        break

                    # Wait before retry
                    await asyncio.sleep(self.retry_delay * retries)

            # If we've exhausted retries, raise the last error
            if last_error:
                self.logger.error(f"Max retries reached. Last error: {last_error}")
                raise last_error

            # This shouldn't be reached, but just in case
            raise RuntimeError("Unknown error in agent execution")

        except Exception as e:
            self.logger.error(f"Error processing message: {e}", exc_info=True)
            self.trace_provider.end_trace(self._current_trace, error=e)
            raise

        finally:
            # Clear execution context
            self.context.end_processing()
            self.logger.debug("Ended message processing")
            if self._current_trace:
                self.trace_provider.end_trace(self._current_trace)
            self._current_trace = None

    async def run_conversation(self, messages: List[Message]) -> List[Message]:
        """Run a sequence of messages through the agent

        Args:
            messages: List of messages to process

        Returns:
            List of agent response messages
        """
        self.logger.info(f"Starting conversation with {len(messages)} messages")
        self._current_trace = self.trace_provider.create_trace(
            name="agent_runner.conversation", group_id=str(id(self))
        )
        self._current_trace.start()

        responses = []
        success_count = 0

        try:
            for i, message in enumerate(messages):
                try:
                    # Create child context for this message
                    msg_context = self.context.create_child(f"msg_{i}")

                    # Run message with the agent
                    response = await self.run(message)
                    responses.append(response)
                    success_count += 1

                except Exception as e:
                    self.logger.error(
                        f"Error in conversation at message {i} after {success_count} successes: {e}",
                        exc_info=True,
                    )

                    # Try custom error handler for conversation-level recovery
                    if self.error_handler:
                        try:
                            custom_response = await self.error_handler(e, message, i)
                            if custom_response:
                                responses.append(custom_response)
                                continue
                        except Exception as handler_error:
                            self.logger.error(f"Error handler failed: {handler_error}")

                    # Re-raise the error if we can't handle it
                    raise

            self.logger.info(f"Completed conversation with {len(responses)} responses")
            return responses

        except Exception as e:
            self.trace_provider.end_trace(self._current_trace, error=e)
            raise

        finally:
            if self._current_trace:
                self.trace_provider.end_trace(self._current_trace)
            self._current_trace = None

    def set_error_handler(
        self, handler: Callable[[Exception, Message, int], Awaitable[Optional[Message]]]
    ) -> None:
        """Set a custom error handler for failed message processing

        Args:
            handler: Async function that takes (exception, message, attempt_number)
                    and returns an optional response message
        """
        self.error_handler = handler
        self.logger.info("Set custom error handler")

    async def default_error_handler(
        self, error: Exception, message: Message, attempt: int
    ) -> Optional[Message]:
        """Default error handler implementation that can be extended

        Args:
            error: The exception that occurred
            message: The message that was being processed
            attempt: Which attempt number this is (1-based)

        Returns:
            Optional response message if error can be handled, None otherwise
        """
        # By default, return None to allow the normal retry logic to proceed
        return None
