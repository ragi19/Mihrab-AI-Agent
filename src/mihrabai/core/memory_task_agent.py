"""
Memory-enabled task agent with enhanced contextual memory
"""

import asyncio
import json
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Union

from ..models.base import BaseModel
from ..utils.logging import get_logger
from ..utils.tracing import TraceProvider
from .agent import Agent
from .memory import Memory, MemoryEntry
from .message import Message, MessageRole
from .task_agent import TaskAgent, ToolConfig


class MemoryEnabledTaskAgent(TaskAgent):
    """Task agent with long-term memory capabilities

    This agent extends TaskAgent with:
    - Long-term memory for context persistence
    - Memory search and retrieval
    - Automatic context augmentation
    - Memory summarization
    """

    def __init__(
        self,
        model: BaseModel,
        system_message: str = "You are a helpful AI assistant that can use tools.",
        tools: Optional[List[ToolConfig]] = None,
        memory: Optional[Memory] = None,
        max_memory_items: int = 50,
        memory_retrieval_count: int = 5,
        automatic_memory: bool = True,
        memory_threshold_score: float = 0.7,
        trace_provider: Optional[TraceProvider] = None,
    ):
        super().__init__(
            model=model,
            system_message=system_message,
            tools=tools or [],
            trace_provider=trace_provider,
        )
        self.memory = memory or Memory()
        self.max_memory_items = max_memory_items
        self.memory_retrieval_count = memory_retrieval_count
        self.automatic_memory = automatic_memory
        self.memory_threshold_score = memory_threshold_score
        self._last_memory_access = 0
        self._memory_access_interval = 2  # seconds
        self.logger = get_logger("agent.memory_task")
        self.logger.info("Initialized memory-enabled task agent")

    async def process_message(self, message: Message) -> Message:
        """Process a message with memory integration

        If automatic memory is enabled, relevant memories will be retrieved and
        added to the context automatically.
        """
        self._current_trace = self._trace_provider.create_trace(
            name="memory_task_agent.process_message", group_id=str(id(self))
        )
        self._current_trace.start()

        try:
            # Add message to history
            self.add_to_history(message)

            if self.automatic_memory:
                # Retrieve relevant memories
                await self._augment_with_memories(message.content)

            # Process message (potentially using tools)
            response = await self._process_with_tools(message)

            # Add response to history
            self.add_to_history(response)

            # Store important exchanges in memory
            await self._store_in_memory(message, response)

            return response

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            self._trace_provider.end_trace(self._current_trace, error=e)
            raise
        finally:
            if self._current_trace:
                self._trace_provider.end_trace(self._current_trace)
            self._current_trace = None

    async def _process_with_tools(self, message: Message) -> Message:
        """Process a message with tool support

        This method handles the interaction with the model, including:
        - Generating responses
        - Detecting and executing tool calls
        - Handling tool results

        Args:
            message: The user message to process

        Returns:
            The final response message
        """
        span = self._create_span("process_with_tools", {})
        span.start()

        try:
            # Get available tools for the model
            tool_configs = [tool.get_schema() for tool in self.tools]

            # Generate response with potential tool calls
            response = await self.model.generate_response(
                self.conversation_history, tools=tool_configs if tool_configs else None
            )

            # Check if the response contains tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                span.set_metadata(
                    {"has_tool_calls": True, "tool_count": len(response.tool_calls)}
                )

                # Process each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name")
                    tool_params = tool_call.get("parameters", {})

                    # Find the matching tool
                    matching_tools = [t for t in self.tools if t.name == tool_name]
                    if not matching_tools:
                        self.logger.warning(f"Tool not found: {tool_name}")
                        continue

                    tool = matching_tools[0]

                    # Execute the tool
                    try:
                        self.logger.debug(
                            f"Executing tool: {tool_name} with params: {tool_params}"
                        )

                        if tool.is_async:
                            result = await tool.function(tool_params)
                        else:
                            result = tool.function(tool_params)

                        # Add tool result to conversation
                        tool_result_msg = Message(
                            role=MessageRole.TOOL,
                            content=str(result),
                            metadata={
                                "tool_name": tool_name,
                                "tool_params": tool_params,
                                "tool_result": result,
                            },
                        )
                        self.add_to_history(tool_result_msg)

                    except Exception as e:
                        self.logger.error(f"Tool execution failed: {e}")
                        error_msg = Message(
                            role=MessageRole.TOOL,
                            content=f"Error executing tool {tool_name}: {str(e)}",
                            metadata={
                                "tool_name": tool_name,
                                "tool_params": tool_params,
                                "error": str(e),
                            },
                        )
                        self.add_to_history(error_msg)

                # Generate final response after tool execution
                final_response = await self.model.generate_response(
                    self.conversation_history
                )
                return final_response

            # If no tool calls, return the original response
            return response

        except Exception as e:
            self.logger.error(f"Error in _process_with_tools: {e}")
            span.set_error(e)
            raise
        finally:
            span.end()

    async def _augment_with_memories(self, query: str) -> None:
        """Augment the conversation with relevant memories

        This retrieves relevant memories and adds them as a system message
        to provide additional context to the model.
        """
        span = self._create_span("retrieve_memories", {"query": query})
        span.start()

        try:
            # Rate limit memory access
            current_time = time.time()
            if current_time - self._last_memory_access < self._memory_access_interval:
                self.logger.debug("Skipping memory retrieval due to rate limiting")
                return

            self._last_memory_access = current_time

            # Retrieve relevant memories
            retrieved_memories = await self.memory.search(
                query=query,
                k=self.memory_retrieval_count,
                threshold=self.memory_threshold_score,
            )

            if not retrieved_memories:
                self.logger.debug("No relevant memories found")
                return

            span.set_metadata({"memory_count": len(retrieved_memories)})

            # Add memories to context
            memory_context = self._format_memories_as_context(retrieved_memories)

            # Check if we already have a memory context message
            for i, msg in enumerate(self.conversation_history):
                if (
                    msg.role == MessageRole.SYSTEM
                    and "RELEVANT MEMORIES:" in msg.content
                ):
                    # Update existing memory context
                    self.conversation_history[i] = Message(
                        role=MessageRole.SYSTEM, content=memory_context
                    )
                    self.logger.debug("Updated memory context")
                    return

            # Add new memory context
            self.conversation_history.append(
                Message(role=MessageRole.SYSTEM, content=memory_context)
            )
            self.logger.debug(
                f"Added memory context with {len(retrieved_memories)} memories"
            )
        finally:
            span.end()

    def _format_memories_as_context(self, memories: List[MemoryEntry]) -> str:
        """Format retrieved memories as context for the model"""
        memory_texts = []

        for i, entry in enumerate(memories):
            importance = (
                f"(importance: {entry.importance:.2f})" if entry.importance else ""
            )
            timestamp = (
                entry.timestamp.strftime("%Y-%m-%d")
                if entry.timestamp
                else "unknown_date"
            )
            memory_texts.append(f"{i+1}. [{timestamp}] {importance} {entry.content}")

        return "RELEVANT MEMORIES:\n" + "\n".join(memory_texts)

    async def _store_in_memory(self, user_message: Message, response: Message) -> None:
        """Store important exchanges in memory

        This evaluates the importance of the exchange and stores it in
        memory if it meets certain criteria.
        """
        span = self._create_span("store_memory", {})
        span.start()

        try:
            # Check if we need to prune memory
            if len(self.memory) >= self.max_memory_items:
                await self._prune_memory()

            # Store user message
            message_importance = await self._assess_importance(user_message.content)
            if message_importance > 0.5:  # Only store important messages
                await self.memory.add(
                    content=user_message.content,
                    metadata={
                        "role": user_message.role.value,
                        "timestamp": user_message.timestamp,
                        "importance": message_importance,
                    },
                    importance=message_importance,
                )
                span.set_metadata({"stored_user_message": True})

            # Store response if it contains factual information
            response_importance = await self._assess_importance(response.content)
            if response_importance > 0.5:
                await self.memory.add(
                    content=response.content,
                    metadata={
                        "role": response.role.value,
                        "timestamp": response.timestamp,
                        "importance": response_importance,
                    },
                    importance=response_importance,
                )
                span.set_metadata({"stored_response": True})
        finally:
            span.end()

    async def _assess_importance(self, text: str) -> float:
        """Assess the importance of a text for memory storage

        Higher values indicate more important information that should be
        remembered for longer. This is a simple heuristic and could be
        improved with a more sophisticated model.
        """
        # Basic heuristic based on text length and information density
        # In a real implementation, this could use a model to evaluate importance

        # Simple length-based heuristic
        if len(text) < 20:
            return 0.3

        # Check for likely important content
        importance_markers = [
            "important",
            "remember",
            "don't forget",
            "key",
            "critical",
            "password",
            "account",
            "number",
            "address",
            "name",
            "date",
            "schedule",
            "appointment",
            "meeting",
        ]

        text_lower = text.lower()
        importance = 0.5  # Base importance

        for marker in importance_markers:
            if marker in text_lower:
                importance += 0.1

        # Cap at 1.0
        return min(importance, 1.0)

    async def _prune_memory(self) -> None:
        """Prune memory to stay within size limits"""
        self.logger.debug("Pruning memory to stay within limits")

        # Remove least important memories
        await self.memory.prune(
            max_entries=int(self.max_memory_items * 0.9)  # Keep 90%
        )

    async def recall(self, query: str, k: int = 5) -> List[MemoryEntry]:
        """Explicitly recall memories related to a query

        Args:
            query: Search query
            k: Maximum number of memories to retrieve

        Returns:
            List of relevant memory entries
        """
        return await self.memory.search(query, k=k)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent's memory"""
        return self.memory.get_memory_stats()

    async def summarize_memory(self, query: Optional[str] = None) -> str:
        """Generate a summary of memories

        Args:
            query: Optional query to filter memories before summarizing

        Returns:
            Text summary of relevant memories
        """
        # If query provided, get relevant memories, otherwise use all
        if query:
            memories = await self.memory.search(query, k=10)
        else:
            memories = self.memory.entries[:10]  # Use latest memories

        if not memories:
            return "No relevant memories found."

        # Format memories for summarization
        memory_text = "\n".join(f"- {entry.content}" for entry in memories)

        # Create a system message for summarization
        system_message = (
            "Summarize the following memory entries into a concise paragraph. "
            "Focus on key information, facts, and preferences."
        )

        # Use the model to generate a summary
        messages = [
            Message(role=MessageRole.SYSTEM, content=system_message),
            Message(role=MessageRole.USER, content=memory_text),
        ]

        response = await self.model.generate_response(messages)
        return response.content

    def set_automatic_memory(self, enabled: bool) -> None:
        """Enable or disable automatic memory retrieval"""
        self.automatic_memory = enabled
        self.logger.info(f"Set automatic memory to: {enabled}")

    def set_memory_threshold(self, threshold: float) -> None:
        """Set the relevance threshold for memory retrieval

        Args:
            threshold: Value between 0.0 and 1.0
        """
        self.memory_threshold_score = max(0.0, min(1.0, threshold))
        self.logger.info(f"Set memory threshold to: {threshold}")

    def clear_memory(self) -> None:
        """Clear all stored memories"""
        self.memory.clear()
        self.logger.info("Cleared all memories")

    def export_memory(self) -> List[Dict[str, Any]]:
        """Export all memories as serializable objects"""
        return [
            {
                "content": entry.content,
                "metadata": entry.metadata,
                "embedding": (
                    entry.embedding.tolist()
                    if hasattr(entry, "embedding") and entry.embedding is not None
                    else None
                ),
                "importance": entry.importance,
                "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
            }
            for entry in self.memory.entries
        ]

    async def import_memory(self, memories: List[Dict[str, Any]]) -> int:
        """Import memories from serialized data

        Args:
            memories: List of serialized memory entries

        Returns:
            Number of memories imported
        """
        count = 0
        for entry in memories:
            try:
                await self.memory.add(
                    content=entry["content"],
                    metadata=entry.get("metadata", {}),
                    importance=entry.get("importance", 0.5),
                )
                count += 1
            except Exception as e:
                self.logger.warning(f"Failed to import memory: {e}")

        self.logger.info(f"Imported {count} memories")
        return count
