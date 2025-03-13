"""
Chat-optimized agent implementation
"""

from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..models.base import BaseModel
from ..utils.logging import get_logger
from ..utils.tokenization import count_tokens, truncate_text
from ..utils.tracing import TraceProvider
from .agent import Agent
from .message import Message, MessageRole


class ChatAgent(Agent):
    """Agent specialized for chat-based interactions"""

    def __init__(
        self,
        model: BaseModel,
        system_message: str = "You are a helpful AI assistant.",
        max_history_tokens: Optional[int] = 4000,
        max_history_messages: Optional[int] = None,
        memory_window_size: int = 10,
        trace_provider: Optional[TraceProvider] = None,
    ):
        super().__init__(model=model, trace_provider=trace_provider)
        self.system_message = system_message
        self.max_history_tokens = max_history_tokens
        self.max_history_messages = max_history_messages
        self.memory_window_size = memory_window_size
        self._history_summarizer: Optional[
            Callable[[List[Message]], Awaitable[str]]
        ] = None
        self._message_metadata: Dict[str, Dict[str, Any]] = {}
        self._conversation_id: Optional[str] = None

        self.logger.info(
            f"Initialized chat agent with max_history_tokens={max_history_tokens}, "
            f"max_history_messages={max_history_messages}, "
            f"memory_window_size={memory_window_size}"
        )
        self.add_to_history(Message(role=MessageRole.SYSTEM, content=system_message))

    async def process_message(self, message: Message) -> Message:
        """Process a chat message, managing history and token limits"""
        self.logger.debug(f"Processing chat message: {message}")

        # Set up tracing
        self._current_trace = self._trace_provider.create_trace(
            name="chat_agent.process_message",
            group_id=self._conversation_id or str(id(self)),
        )
        self._current_trace.start()

        try:
            # Apply pre-processing hooks
            processed_message = message
            for hook in self._pre_process_hooks:
                processed_message = await hook(processed_message)

            # Add message metadata
            self._track_message_metadata(processed_message)

            # Add new message to history
            self.add_to_history(processed_message)

            # Manage history size
            await self._manage_history()

            # Generate response using model
            self.logger.debug("Generating model response")
            with self._create_span(
                "generate_response",
                {
                    "model": self.model.model_name,
                    "history_size": len(self.conversation_history),
                },
            ) as span:
                response = await self.model.generate_response(self.conversation_history)

                # Track token usage
                prompt_tokens = await self.model.count_tokens(
                    "\n".join([m.content for m in self.conversation_history])
                )
                completion_tokens = await self.model.count_tokens(response.content)
                self.update_token_usage(prompt_tokens, completion_tokens)

                span.metadata.update(
                    {
                        "response_length": len(response.content),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    }
                )

            # Track response metadata
            self._track_message_metadata(response)

            # Add response to history
            self.add_to_history(response)

            # Apply post-processing hooks
            processed_response = response
            for hook in self._post_process_hooks:
                processed_response = await hook(processed_message, processed_response)

            return processed_response

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            self._trace_provider.end_trace(self._current_trace, error=e)
            raise
        finally:
            if self._current_trace:
                self._trace_provider.end_trace(self._current_trace)
            self._current_trace = None

    async def _process_message_internal(self, message: Message) -> Message:
        """Process a chat message with tracing - this is now handled in process_message"""
        # This is kept for backward compatibility
        with self._create_span(
            "prepare_messages", {"history_length": len(self.conversation_history)}
        ):
            messages = self.conversation_history.copy()

        with self._create_span(
            "generate_response", {"model": self.model.model_name}
        ) as span:
            response = await self.model.generate_response(messages)
            span.metadata = {"response": response.dict()}

        return response

    async def _manage_history(self) -> None:
        """Manage conversation history according to constraints"""
        # Check if we need to manage by token count
        if self.max_history_tokens:
            await self._truncate_history_by_tokens()

        # Check if we need to manage by message count
        if (
            self.max_history_messages
            and len(self.conversation_history) > self.max_history_messages
        ):
            await self._truncate_history_by_messages()

        # If we have a history summarizer, use it for long conversations
        if (
            self._history_summarizer
            and len(self.conversation_history) > self.memory_window_size * 2
        ):
            await self._summarize_old_history()

    async def _truncate_history_by_tokens(self) -> None:
        """Truncate conversation history to fit within token limit"""
        total_tokens = 0
        truncated_history: List[Message] = []

        # Always keep system message
        system_messages = [
            msg for msg in self.conversation_history if msg.role == MessageRole.SYSTEM
        ]
        other_messages = [
            msg for msg in self.conversation_history if msg.role != MessageRole.SYSTEM
        ]

        # Calculate system message tokens
        for msg in system_messages:
            tokens = await self.model.count_tokens(msg.content)
            total_tokens += tokens
            self.logger.debug(f"System message tokens: {tokens}")

        truncated_history.extend(system_messages)

        # Add most recent messages that fit within limit
        for msg in reversed(other_messages):
            msg_tokens = await self.model.count_tokens(msg.content)
            if total_tokens + msg_tokens <= self.max_history_tokens:
                truncated_history.insert(len(system_messages), msg)
                total_tokens += msg_tokens
                self.logger.debug(f"Added message with {msg_tokens} tokens")
            else:
                self.logger.debug(
                    f"Skipping message with {msg_tokens} tokens (would exceed limit)"
                )
                break

        original_count = len(self.conversation_history)
        self.conversation_history = truncated_history
        new_count = len(self.conversation_history)

        if original_count != new_count:
            self.logger.info(
                f"Truncated history from {original_count} to {new_count} messages "
                f"({total_tokens} tokens)"
            )

    async def _truncate_history_by_messages(self) -> None:
        """Truncate history to keep only the most recent messages"""
        # Always keep system messages
        system_messages = [
            msg for msg in self.conversation_history if msg.role == MessageRole.SYSTEM
        ]
        other_messages = [
            msg for msg in self.conversation_history if msg.role != MessageRole.SYSTEM
        ]

        # Calculate how many recent messages to keep
        keep_count = self.max_history_messages - len(system_messages)
        if keep_count <= 0:
            keep_count = 1  # At least keep one non-system message

        # Keep only the most recent messages
        truncated_history = system_messages + other_messages[-keep_count:]

        original_count = len(self.conversation_history)
        self.conversation_history = truncated_history
        new_count = len(self.conversation_history)

        self.logger.info(
            f"Truncated history from {original_count} to {new_count} messages "
            f"(max_history_messages={self.max_history_messages})"
        )

    async def _summarize_old_history(self) -> None:
        """Summarize older history messages to create a compact representation"""
        if not self._history_summarizer:
            return

        # Keep system messages separate
        system_messages = [
            msg for msg in self.conversation_history if msg.role == MessageRole.SYSTEM
        ]
        other_messages = [
            msg for msg in self.conversation_history if msg.role != MessageRole.SYSTEM
        ]

        # If we don't have enough messages to summarize, skip
        if len(other_messages) <= self.memory_window_size:
            return

        # Split messages into recent and older groups
        recent_messages = other_messages[-self.memory_window_size :]
        older_messages = other_messages[: -self.memory_window_size]

        try:
            # Summarize older messages
            summary = await self._history_summarizer(older_messages)

            # Create a new system message with the summary
            summary_msg = Message(
                role=MessageRole.SYSTEM,
                content=f"Previous conversation summary: {summary}",
            )

            # Update history with summarized version
            self.conversation_history = (
                system_messages + [summary_msg] + recent_messages
            )
            self.logger.info(
                f"Summarized {len(older_messages)} older messages into a system message"
            )
        except Exception as e:
            self.logger.warning(f"Failed to summarize history: {e}")

    def _track_message_metadata(self, message: Message) -> None:
        """Track metadata for a message"""
        message_id = str(id(message))
        self._message_metadata[message_id] = {
            "timestamp": message.timestamp,
            "length": len(message.content),
            "role": message.role.value,
        }

    def update_system_message(self, new_message: str) -> None:
        """Update the system message and reset conversation"""
        self.logger.info(f"Updating system message: {new_message}")
        self.system_message = new_message
        self.clear_history()
        self.add_to_history(Message(role=MessageRole.SYSTEM, content=new_message))

    def set_history_summarizer(
        self, summarizer: Callable[[List[Message]], Awaitable[str]]
    ) -> None:
        """Set a function to summarize conversation history"""
        self._history_summarizer = summarizer
        self.logger.info("Set history summarizer function")

    def set_conversation_id(self, conversation_id: str) -> None:
        """Set conversation ID for tracking purposes"""
        self._conversation_id = conversation_id
        self.logger.info(f"Set conversation ID: {conversation_id}")

    def get_message_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all processed messages"""
        return self._message_metadata.copy()

    def set_max_history(
        self, max_tokens: Optional[int] = None, max_messages: Optional[int] = None
    ) -> None:
        """Configure history management parameters"""
        if max_tokens is not None:
            self.max_history_tokens = max_tokens
        if max_messages is not None:
            self.max_history_messages = max_messages
        self.logger.info(
            f"Updated max_history_tokens={self.max_history_tokens}, "
            f"max_history_messages={self.max_history_messages}"
        )
