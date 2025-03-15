"""
Task-specialized agent implementation with advanced task management
"""

import contextlib
from typing import Any, Callable, ContextManager, Dict, List, Optional, Union

from ..models.base import BaseModel
from ..utils.logging import get_logger
from ..utils.tracing import Span, TraceProvider
from .agent import Agent
from .message import Message, MessageRole


class ToolConfig:
    """Configuration for a tool that can be used by the agent"""

    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Optional[Dict[str, Any]] = None,
        required_params: Optional[List[str]] = None,
        is_async: bool = False,
    ):
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters or {}
        self.required_params = required_params or []
        self.is_async = is_async

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for model function calling"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class TaskAgent(Agent):
    """Agent specialized for task-based operations"""

    def __init__(
        self,
        model: BaseModel,
        system_message: str = "You are a task-focused AI assistant.",
        max_steps: Optional[int] = 10,
        max_history_tokens: Optional[int] = 4000,
        tools: Optional[List[ToolConfig]] = None,
        trace_provider: Optional[TraceProvider] = None,
    ):
        super().__init__(model=model, trace_provider=trace_provider)
        self.system_message = system_message
        self.max_steps = max_steps
        self.max_history_tokens = max_history_tokens
        self.current_step = 0
        self.task_state: Dict[str, Any] = {}
        self.tools = tools or []
        self.logger = get_logger("agent.task")
        self.add_to_history(Message(role=MessageRole.SYSTEM, content=system_message))

    def _safe_create_span(self, name: str, data: Dict[str, Any]) -> ContextManager:
        """Create a tracing span with safe fallback for tests"""
        try:
            # Try to create normal span
            span = self._create_span(name, data)
            # Check if span supports context manager protocol
            if hasattr(span, "__enter__") and hasattr(span, "__exit__"):
                return span
            else:
                # Wrap span in a dummy context manager
                self.logger.debug(f"Wrapping span in dummy context manager for {name}")
                return contextlib.nullcontext(span)
        except (RuntimeError, AttributeError, TypeError):
            # Fallback for tests - use dummy context manager
            self.logger.debug(f"Creating dummy span for {name} in testing environment")
            return contextlib.nullcontext()

    async def process_message(self, message: Message) -> Message:
        """Process a task-oriented message with step tracking"""
        self.logger.debug(f"Processing task message: {message}")

        # Initialize trace for this interaction if not already set
        if not self._current_trace:
            self._current_trace = self._trace_provider.create_trace(
                name="task.process_message", group_id=str(id(self))
            )
            self._current_trace.start()

        with self._safe_create_span(
            "task_execution", {"step": self.current_step, "max_steps": self.max_steps}
        ) as span:
            try:
                if self.max_steps and self.current_step >= self.max_steps:
                    return Message(
                        role=MessageRole.ASSISTANT,
                        content="Task exceeded maximum allowed steps.",
                    )

                # Add task context to message
                enriched_message = self._enrich_message(message)
                self.add_to_history(enriched_message)

                # Truncate history if needed
                if self.max_history_tokens:
                    await self._truncate_history()

                # Generate and process response
                response = await self._process_message_internal(enriched_message)
                self.current_step += 1

                # Update task state based on response
                self._update_task_state(response)

                # Set span metadata if span is valid
                if isinstance(span, Span):
                    span.set_metadata(
                        {
                            "task_state": self.task_state,
                            "response_length": len(response.content),
                        }
                    )

                return response

            except Exception as e:
                self.logger.error(f"Task execution failed: {e}")
                if isinstance(span, Span):
                    span.set_error(e)
                raise
            finally:
                # End trace if we created it
                if self._current_trace:
                    self._trace_provider.end_trace(self._current_trace)
                    self._current_trace = None

    async def _process_message_internal(self, message: Message) -> Message:
        """Internal implementation for processing messages"""
        return await self.model.generate_response(self.conversation_history)

    def _enrich_message(self, message: Message) -> Message:
        """Add task context and state to message"""
        context = f"\nCurrent step: {self.current_step}"
        if self.task_state:
            context += f"\nTask state: {self.task_state}"

        # Create metadata dict with task state
        metadata = {"task_state": self.task_state.copy()}
        # Add original metadata if it exists
        if message.metadata is not None:
            metadata.update(message.metadata)

        return Message(
            role=message.role,
            content=f"{message.content}\n{context}",
            metadata=metadata,
        )

    def _update_task_state(self, response: Message) -> None:
        """Update task state based on response"""
        # Extract any state updates from response metadata
        if hasattr(response, "metadata") and response.metadata:
            if "task_updates" in response.metadata:
                self.task_state.update(response.metadata["task_updates"])

            # Update completion status
            if "task_completed" in response.metadata:
                self.task_state["completed"] = response.metadata["task_completed"]

    async def _truncate_history(self) -> None:
        """Truncate conversation history while preserving task context"""
        total_tokens = 0
        truncated_history: List[Message] = []

        # Always keep system message
        system_messages = [
            msg for msg in self.conversation_history if msg.role == MessageRole.SYSTEM
        ]
        task_messages = [
            msg for msg in self.conversation_history if msg.role != MessageRole.SYSTEM
        ]

        # Calculate system message tokens
        for msg in system_messages:
            tokens = await self.model.count_tokens(msg.content)
            total_tokens += tokens

        truncated_history.extend(system_messages)

        # Add most recent task messages that fit within limit
        for msg in reversed(task_messages):
            msg_tokens = await self.model.count_tokens(msg.content)
            if total_tokens + msg_tokens <= self.max_history_tokens:
                truncated_history.insert(len(system_messages), msg)
                total_tokens += msg_tokens
            else:
                break

        self.conversation_history = truncated_history

    def reset_task(self) -> None:
        """Reset task state and counter"""
        self.current_step = 0
        self.task_state.clear()
        self.clear_history()
        self.add_to_history(
            Message(role=MessageRole.SYSTEM, content=self.system_message)
        )

    def get_task_state(self) -> Dict[str, Any]:
        """Get current task state"""
        return {
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "state": self.task_state.copy(),
            "completed": self.task_state.get("completed", False),
        }
