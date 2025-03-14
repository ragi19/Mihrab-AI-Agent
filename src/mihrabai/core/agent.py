"""
Base agent class providing core functionality for LLM-powered agents
"""

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, TypeVar

from ..models.base import BaseModel
from ..utils.logging import get_logger
from ..utils.tracing import Span, Trace, TraceProvider
from .message import Message

T = TypeVar("T")


class Agent(ABC):
    def __init__(
        self, model: BaseModel, trace_provider: Optional[TraceProvider] = None
    ):
        self.model = model
        self.conversation_history: List[Message] = []
        self.logger = get_logger(f"agent.{self.__class__.__name__}")
        self.logger.info(f"Initialized agent with model: {model.model_name}")
        self._trace_provider = trace_provider or TraceProvider()
        self._current_trace: Optional[Trace] = None
        self._pre_process_hooks: List[Callable[[Message], Awaitable[Message]]] = []
        self._post_process_hooks: List[
            Callable[[Message, Message], Awaitable[Message]]
        ] = []
        self._token_usage: Dict[str, int] = {"prompt": 0, "completion": 0, "total": 0}

    @abstractmethod
    async def process_message(self, message: Message) -> Message:
        """Process an incoming message and generate a response"""
        self.logger.debug(f"Processing message: {message}")
        # Create trace for this interaction
        self._current_trace = self._trace_provider.create_trace(
            name="agent.process_message", group_id=str(id(self))
        )
        self._current_trace.start()

        try:
            # Apply pre-processing hooks
            processed_message = message
            for hook in self._pre_process_hooks:
                processed_message = await hook(processed_message)

            # Create span for message processing
            with self._create_span(
                "process_message", {"message": processed_message.dict()}
            ) as span:
                response = await self._process_message_internal(processed_message)
                span.metadata = {"response": response.dict()}

                # Apply post-processing hooks
                processed_response = response
                for hook in self._post_process_hooks:
                    processed_response = await hook(
                        processed_message, processed_response
                    )

                return processed_response

        except Exception as e:
            self._trace_provider.end_trace(self._current_trace, error=e)
            raise
        finally:
            if self._current_trace:
                self._trace_provider.end_trace(self._current_trace)
            self._current_trace = None

    async def _process_message_internal(self, message: Message) -> Message:
        """Internal message processing logic"""
        raise NotImplementedError

    def _create_span(self, name: str, data: Dict[str, Any]) -> Span:
        """Create a new span in the current trace"""
        if not self._current_trace:
            raise RuntimeError("No active trace")
        return self._trace_provider.create_span(name, data)

    def add_to_history(self, message: Message) -> None:
        """Add a message to the conversation history"""
        self.conversation_history.append(message)
        self.logger.debug(f"Added message to history: {message}")

    def get_history(self) -> List[Message]:
        """Get the full conversation history"""
        return self.conversation_history

    def clear_history(self) -> None:
        """Clear the conversation history"""
        self.conversation_history = []
        self.logger.debug("Cleared conversation history")

    async def add_pre_process_hook(
        self, hook: Callable[[Message], Awaitable[Message]]
    ) -> None:
        """Add a pre-processing hook for messages"""
        self._pre_process_hooks.append(hook)
        self.logger.debug(f"Added pre-processing hook: {hook.__name__}")

    async def add_post_process_hook(
        self, hook: Callable[[Message, Message], Awaitable[Message]]
    ) -> None:
        """Add a post-processing hook for responses"""
        self._post_process_hooks.append(hook)
        self.logger.debug(f"Added post-processing hook: {hook.__name__}")

    def get_token_usage(self) -> Dict[str, int]:
        """Get token usage statistics for this agent"""
        return self._token_usage.copy()

    def update_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Update token usage statistics"""
        self._token_usage["prompt"] += prompt_tokens
        self._token_usage["completion"] += completion_tokens
        self._token_usage["total"] = (
            self._token_usage["prompt"] + self._token_usage["completion"]
        )

    def set_model(self, model: BaseModel) -> None:
        """Change the model used by this agent"""
        self.logger.info(
            f"Changing model from {self.model.model_name} to {model.model_name}"
        )
        self.model = model

    def get_capabilities(self) -> List[str]:
        """Get a list of this agent's capabilities"""
        return ["text_generation"]

    def get_description(self) -> str:
        """Get a description of this agent"""
        return f"Base agent using {self.model.model_name}"
