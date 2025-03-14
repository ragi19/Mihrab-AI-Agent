"""
Memory-enabled agent runner with enhanced conversation management
"""

import asyncio
import contextlib
import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from ..core.agent import Agent
from ..core.memory_task_agent import MemoryEnabledTaskAgent
from ..core.message import Message, MessageRole
from ..utils.logging import get_logger
from ..utils.tracing import Trace, TraceProvider
from .context import RuntimeContext
from .recovery import ConversationRecovery
from .runner import AgentRunner

logger = get_logger("runtime.memory_runner")


# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)


class MemoryAgentRunner(AgentRunner):
    """Enhanced runner for memory-enabled agents

    This class extends the base AgentRunner with:
    - Memory persistence and recovery
    - Managed memory export/import
    - Context summarization
    - Memory search functionality
    """

    def __init__(
        self,
        agent: MemoryEnabledTaskAgent,
        context: Optional[RuntimeContext] = None,
        trace_provider: Optional[TraceProvider] = None,
        recovery: Optional[ConversationRecovery] = None,
        memory_persistence_path: Optional[str] = None,
        auto_save_interval_seconds: int = 300,  # 5 minutes
    ):
        super().__init__(agent, context, trace_provider, recovery)

        if not isinstance(agent, MemoryEnabledTaskAgent):
            raise TypeError("MemoryAgentRunner requires a MemoryEnabledTaskAgent")

        self.memory_agent = agent
        self.memory_persistence_path = memory_persistence_path or os.path.expanduser(
            "~/.mihrabai/memories/"
        )

        # Create persistence directory if it doesn't exist
        os.makedirs(self.memory_persistence_path, exist_ok=True)

        self.auto_save_interval = auto_save_interval_seconds
        self._last_save_time = time.time()
        self._current_session_id = None

        logger.info(
            f"Initialized MemoryAgentRunner with persistence path: {self.memory_persistence_path}"
        )

    def _create_span(self, name: str, data: Dict[str, Any]):
        """Create a new span in the current trace"""
        if not self._current_trace:
            return contextlib.nullcontext()
        return self.trace_provider.create_span(name, data)

    async def run(
        self, message: Union[str, Message], session_id: Optional[str] = None
    ) -> Message:
        """Process a message and return the response

        Args:
            message: Message to process
            session_id: Optional session ID for memory loading/saving

        Returns:
            Agent's response message
        """
        # Convert to Message if needed
        if isinstance(message, str):
            message = Message(role=MessageRole.USER, content=message)

        # Load session memories if provided
        if session_id:
            await self.load_memory(session_id)

        # Process message
        self._current_trace = self.trace_provider.create_trace(
            name="memory_runner.run", group_id=session_id or str(id(self))
        )
        self._current_trace.start()

        try:
            # Process the message with agent
            response = await self.agent.process_message(message)

            # Handle auto-save if needed
            current_time = datetime.datetime.now().timestamp()
            if (
                session_id
                and current_time - self._last_save_time > self.auto_save_interval
            ):
                await self.save_memory(session_id)
                self._last_save_time = current_time

            return response

        except Exception as e:
            self.logger.error(f"Error running agent: {e}")
            self.trace_provider.end_trace(self._current_trace, error=e)

            # Try recovery if available
            if hasattr(self, "recovery") and self.recovery:
                try:
                    return await self.recovery.recover(
                        agent=self.agent,
                        context=self.context,
                        error=str(e),
                        message=message,
                    )
                except Exception as recovery_error:
                    self.logger.error(f"Recovery failed: {recovery_error}")

            # Raise original error if recovery failed or not available
            raise
        finally:
            if self._current_trace:
                self.trace_provider.end_trace(self._current_trace)
            self._current_trace = None

    async def save_memory(self, session_id: str) -> None:
        """Save agent's memory to disk

        Args:
            session_id: Unique session identifier
        """
        with self._create_span("save_memory", {"session_id": session_id}):
            try:
                # Create directory if it doesn't exist
                memory_dir = Path(self.memory_persistence_path)
                memory_dir.mkdir(parents=True, exist_ok=True)

                # Export memory
                memories = self.memory_agent.export_memory()

                # Save to file
                memory_file = memory_dir / f"{session_id}.json"
                with open(memory_file, "w") as f:
                    json.dump(
                        {
                            "session_id": session_id,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "memories": memories,
                        },
                        f,
                        indent=2,
                        cls=DateTimeEncoder,
                    )

                self.logger.info(f"Saved {len(memories)} memories to {memory_file}")
                self._last_save_time = datetime.datetime.now().timestamp()

            except Exception as e:
                self.logger.error(f"Error saving memory: {e}")
                raise

    async def load_memory(self, session_id: str) -> int:
        """Load agent's memory from disk

        Args:
            session_id: Unique session identifier

        Returns:
            Number of memories loaded
        """
        with self._create_span("load_memory", {"session_id": session_id}):
            try:
                # Check if memory file exists
                memory_file = Path(self.memory_persistence_path) / f"{session_id}.json"
                if not memory_file.exists():
                    self.logger.info(f"No memory file found for session {session_id}")
                    return 0

                # Load from file
                with open(memory_file, "r") as f:
                    data = json.load(f)

                # Import memories
                count = await self.memory_agent.import_memory(data["memories"])
                self.logger.info(f"Loaded {count} memories for session {session_id}")

                return count

            except Exception as e:
                self.logger.error(f"Error loading memory: {e}")
                return 0

    async def search_memory(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search agent's memory

        Args:
            query: Search query
            k: Maximum number of results

        Returns:
            List of memory entries
        """
        with self._create_span("search_memory", {"query": query, "limit": k}):
            memories = await self.memory_agent.recall(query, k)

            # Convert to serializable format
            results = []
            for entry in memories:
                results.append(
                    {
                        "content": entry.content,
                        "importance": entry.importance,
                        "timestamp": (
                            entry.timestamp.isoformat() if entry.timestamp else None
                        ),
                        "metadata": entry.metadata,
                    }
                )

            return results

    async def summarize_conversation(self) -> str:
        """Generate a summary of the current conversation"""
        with self._create_span("summarize_conversation", {}):
            return await self.memory_agent.summarize_memory()

    async def clear_conversation(self) -> None:
        """Clear conversation history and memory"""
        self.agent.clear_history()
        self.memory_agent.clear_memory()
        self.logger.info("Cleared conversation history and memory")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent's memory"""
        return self.memory_agent.get_memory_stats()

    def set_memory_settings(
        self,
        automatic_memory: Optional[bool] = None,
        threshold: Optional[float] = None,
        max_items: Optional[int] = None,
        retrieval_count: Optional[int] = None,
    ) -> None:
        """Update memory settings

        Args:
            automatic_memory: Enable/disable automatic memory integration
            threshold: Memory relevance threshold
            max_items: Maximum number of memories to store
            retrieval_count: Number of memories to retrieve
        """
        if automatic_memory is not None:
            self.memory_agent.set_automatic_memory(automatic_memory)

        if threshold is not None:
            self.memory_agent.set_memory_threshold(threshold)

        if max_items is not None:
            self.memory_agent.max_memory_items = max_items

        if retrieval_count is not None:
            self.memory_agent.memory_retrieval_count = retrieval_count

        self.logger.info("Updated memory settings")
