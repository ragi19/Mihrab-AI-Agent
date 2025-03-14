"""
Advanced memory management system for LLM agents
"""

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

from ..core.message import Message, MessageRole
from ..utils.logging import get_logger

T = TypeVar("T")


@dataclass
class MemoryEntry(Generic[T]):
    """Memory entry for storing information in memory"""

    content: T
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory entry to dictionary"""
        content_dict: Any

        # Handle different content types
        if isinstance(self.content, Message):
            # Convert Message to dict manually
            content_dict = {
                "role": self.content.role.value,  # Access enum value
                "content": self.content.content,
                "metadata": self.content.metadata if self.content.metadata else {},
            }
        else:
            # Use content as is for other types
            content_dict = self.content

        return {
            "content": content_dict,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry[Any]":
        """Create memory entry from dictionary

        This method returns a MemoryEntry with Any as the type parameter
        since we can't know the exact type at runtime.
        """
        content_data = data["content"]

        # Handle the content based on its type
        if isinstance(content_data, dict) and "role" in content_data:
            # Reconstruct Message object if the content was a Message
            content: Any = Message(
                role=MessageRole(content_data["role"]),
                content=content_data["content"],
                metadata=content_data.get("metadata", {}),
            )
        else:
            content = content_data

        # Create a new MemoryEntry with Any type
        return MemoryEntry[Any](
            content=content,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            importance=data["importance"],
            metadata=data["metadata"],
        )


class Memory:
    """Advanced memory management system"""

    def __init__(
        self,
        working_memory_size: int = 10,
        long_term_memory_size: Optional[int] = None,
        importance_threshold: float = 0.5,
    ) -> None:
        self.working_memory: deque[MemoryEntry[Any]] = deque(maxlen=working_memory_size)
        self.long_term_memory: List[MemoryEntry[Any]] = []
        self.semantic_memory: Dict[str, MemoryEntry[Any]] = {}
        self.episodic_memory: List[MemoryEntry[Any]] = []
        self.long_term_memory_size = long_term_memory_size
        self.importance_threshold = importance_threshold
        self.logger = get_logger("memory")

    def add_to_working_memory(
        self,
        item: Any,
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add item to working memory"""
        entry = MemoryEntry(
            content=item, importance=importance, metadata=metadata or {}
        )
        self.working_memory.append(entry)
        self.logger.debug(f"Added item to working memory")

        # Consider moving to long-term memory if important
        if importance >= self.importance_threshold:
            self._consolidate_to_long_term(entry)

    def add_to_semantic_memory(
        self, key: str, item: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add item to semantic memory with a key"""
        entry = MemoryEntry(content=item, metadata=metadata or {})
        self.semantic_memory[key] = entry
        self.logger.debug(f"Added item to semantic memory with key '{key}'")

    def add_to_episodic_memory(
        self, item: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add item to episodic memory"""
        entry = MemoryEntry(content=item, metadata=metadata or {})
        self.episodic_memory.append(entry)
        self.logger.debug(f"Added item to episodic memory")

    def get_from_working_memory(self) -> List[Any]:
        """Get all items from working memory"""
        return [entry.content for entry in self.working_memory]

    def get_from_semantic_memory(self, key: str) -> Optional[Any]:
        """Get item from semantic memory by key"""
        if key in self.semantic_memory:
            return self.semantic_memory[key].content
        return None

    def get_recent_episodes(self, n: int = 5) -> List[Any]:
        """Get n most recent episodic memories"""
        return [entry.content for entry in self.episodic_memory[-n:]]

    def search_long_term_memory(
        self, query: str, limit: int = 5, min_importance: float = 0.0
    ) -> List[Any]:
        """Search long-term memory with basic relevance filtering"""
        # Filter entries by importance
        relevant_entries = [
            entry
            for entry in self.long_term_memory
            if entry.importance >= min_importance
        ]

        # Sort entries by importance and recency
        relevant_entries.sort(
            key=lambda entry: (entry.importance, entry.timestamp), reverse=True
        )

        # Return only the content of filtered entries
        return [entry.content for entry in relevant_entries[:limit]]

    def _consolidate_to_long_term(self, entry: MemoryEntry[Any]) -> None:
        """Move important memories to long-term storage"""
        self.long_term_memory.append(entry)
        self.logger.debug(f"Consolidated memory to long-term storage")

        # Maintain long-term memory size limit
        if (
            self.long_term_memory_size
            and len(self.long_term_memory) > self.long_term_memory_size
        ):
            # Remove least important memories
            self.long_term_memory.sort(key=lambda x: x.importance)
            excess = len(self.long_term_memory) - self.long_term_memory_size
            self.long_term_memory = self.long_term_memory[excess:]

    def summarize_working_memory(self) -> str:
        """Generate a summary of working memory contents"""
        summary = []
        for entry in self.working_memory:
            summary.append(
                f"- {str(entry.content)} (importance: {entry.importance:.2f})"
            )
        return "\n".join(summary)

    def clear_working_memory(self) -> None:
        """Clear working memory"""
        self.working_memory.clear()
        self.logger.debug("Cleared working memory")

    def save_state(self, file_path: str) -> None:
        """Save memory state to file"""
        state = {
            "working_memory": [entry.to_dict() for entry in self.working_memory],
            "long_term_memory": [entry.to_dict() for entry in self.long_term_memory],
            "semantic_memory": {
                k: v.to_dict() for k, v in self.semantic_memory.items()
            },
            "episodic_memory": [entry.to_dict() for entry in self.episodic_memory],
        }
        with open(file_path, "w") as f:
            json.dump(state, f, indent=2)
        self.logger.info(f"Saved memory state to {file_path}")

    def load_state(self, file_path: str) -> None:
        """Load memory state from file"""
        with open(file_path, "r") as f:
            state = json.load(f)

        self.working_memory = deque(
            [MemoryEntry.from_dict(entry) for entry in state["working_memory"]],
            maxlen=self.working_memory.maxlen,
        )
        self.long_term_memory = [
            MemoryEntry.from_dict(entry) for entry in state["long_term_memory"]
        ]
        self.semantic_memory = {
            k: MemoryEntry.from_dict(v) for k, v in state["semantic_memory"].items()
        }
        self.episodic_memory = [
            MemoryEntry.from_dict(entry) for entry in state["episodic_memory"]
        ]
        self.logger.info(f"Loaded memory state from {file_path}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        return {
            "working_memory_size": len(self.working_memory),
            "long_term_memory_size": len(self.long_term_memory),
            "semantic_memory_size": len(self.semantic_memory),
            "episodic_memory_size": len(self.episodic_memory),
            "total_memories": (
                len(self.working_memory)
                + len(self.long_term_memory)
                + len(self.semantic_memory)
                + len(self.episodic_memory)
            ),
        }

    async def search(
        self, query: str, k: int = 5, threshold: float = 0.0
    ) -> List[MemoryEntry]:
        """Search for memories related to a query

        Args:
            query: Search query
            k: Maximum number of results to return
            threshold: Minimum relevance score threshold

        Returns:
            List of relevant memory entries
        """
        # For now, implement a simple search based on string matching
        # In a real implementation, this would use embeddings and semantic search

        # Combine all memories
        all_memories = list(self.working_memory) + self.long_term_memory

        # Simple relevance scoring based on substring matching
        scored_memories = []
        query_lower = query.lower()

        for entry in all_memories:
            if isinstance(entry.content, str):
                content_lower = entry.content.lower()
                # Simple relevance score based on substring presence
                if query_lower in content_lower:
                    # Higher score for exact matches
                    score = 0.8
                else:
                    # Check for partial word matches
                    words = query_lower.split()
                    matches = sum(1 for word in words if word in content_lower)
                    score = 0.5 * (matches / len(words)) if words else 0

                if score >= threshold:
                    scored_memories.append((entry, score))

        # Sort by score (descending)
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # Return top k entries
        return [entry for entry, _ in scored_memories[:k]]

    async def add(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
    ) -> None:
        """Add an item to memory

        Args:
            content: Content to store
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)
        """
        entry = MemoryEntry(
            content=content, metadata=metadata or {}, importance=importance
        )

        # Add to working memory
        self.working_memory.append(entry)

        # Also add to long-term if important enough
        if importance >= self.importance_threshold:
            self.long_term_memory.append(entry)

        # Prune if needed
        if (
            self.long_term_memory_size is not None
            and len(self.long_term_memory) > self.long_term_memory_size
        ):
            self._prune_long_term_memory()

    async def prune(self, max_entries: int) -> None:
        """Prune memory to stay within size limits

        Args:
            max_entries: Maximum number of entries to keep
        """
        if len(self.long_term_memory) <= max_entries:
            return

        # Sort by importance (ascending)
        self.long_term_memory.sort(key=lambda x: x.importance)

        # Keep only the most important entries
        self.long_term_memory = self.long_term_memory[-max_entries:]

    def _prune_long_term_memory(self) -> None:
        """Prune long-term memory to stay within size limits"""
        # Sort by importance (ascending)
        self.long_term_memory.sort(key=lambda x: x.importance)

        # Remove least important memories
        if self.long_term_memory_size is not None:
            excess = len(self.long_term_memory) - self.long_term_memory_size
            if excess > 0:
                self.long_term_memory = self.long_term_memory[excess:]

    def clear(self) -> None:
        """Clear all memories"""
        self.working_memory.clear()
        self.long_term_memory.clear()
        self.semantic_memory.clear()
        self.episodic_memory.clear()

    @property
    def entries(self) -> List[MemoryEntry]:
        """Get all memory entries"""
        return list(self.working_memory) + self.long_term_memory

    def __len__(self) -> int:
        """Get the total number of memory entries"""
        return (
            len(self.working_memory)
            + len(self.long_term_memory)
            + len(self.semantic_memory)
            + len(self.episodic_memory)
        )
