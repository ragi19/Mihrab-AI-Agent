"""
Unit tests for memory functionality.
"""
from datetime import datetime, timedelta

import pytest

from mihrabai.core.memory import Memory, MemoryEntry
from mihrabai.core.message import Message, MessageRole


@pytest.fixture
def memory():
    return Memory(
        working_memory_size=5, long_term_memory_size=10, importance_threshold=0.7
    )


def test_memory_initialization(memory):
    """Test memory system initialization"""
    assert len(memory.working_memory) == 0
    assert len(memory.long_term_memory) == 0
    assert len(memory.semantic_memory) == 0
    assert len(memory.episodic_memory) == 0


def test_working_memory_operations(memory):
    """Test working memory functionality"""
    # Add items
    memory.add_to_working_memory("item1", importance=0.5)
    memory.add_to_working_memory("item2", importance=0.8)

    # Check contents
    items = memory.get_from_working_memory()
    assert len(items) == 2
    assert "item1" in items
    assert "item2" in items

    # Test size limit
    for i in range(5):
        memory.add_to_working_memory(f"extra{i}")
    assert len(memory.working_memory) == 5


def test_semantic_memory_operations(memory):
    """Test semantic memory functionality"""
    # Add and retrieve items
    memory.add_to_semantic_memory("key1", "value1")
    memory.add_to_semantic_memory("key2", {"data": "value2"})

    assert memory.get_from_semantic_memory("key1") == "value1"
    assert memory.get_from_semantic_memory("key2")["data"] == "value2"
    assert memory.get_from_semantic_memory("nonexistent") is None


def test_episodic_memory_operations(memory):
    """Test episodic memory functionality"""
    # Add episodes
    for i in range(3):
        memory.add_to_episodic_memory(f"event{i}")

    # Get recent episodes
    recent = memory.get_recent_episodes(n=2)
    assert len(recent) == 2
    assert recent == ["event1", "event2"]


def test_long_term_memory_consolidation(memory):
    """Test memory consolidation to long-term storage"""
    # Add important items that should be consolidated
    memory.add_to_working_memory("important1", importance=0.8)
    memory.add_to_working_memory("important2", importance=0.9)
    memory.add_to_working_memory("unimportant", importance=0.3)

    # Check long-term memory - filter by importance
    relevant = memory.search_long_term_memory("", min_importance=0.7)
    assert len(relevant) == 2
    assert "important1" in relevant
    assert "important2" in relevant


def test_memory_state_persistence(memory, tmp_path):
    """Test memory state save and load"""
    # Add various types of memories
    memory.add_to_working_memory("work_item")
    memory.add_to_semantic_memory("sem_key", "sem_value")
    memory.add_to_episodic_memory("episode")

    # Save state
    file_path = tmp_path / "memory_state.json"
    memory.save_state(str(file_path))

    # Create new memory instance and load state
    new_memory = Memory()
    new_memory.load_state(str(file_path))

    # Verify state
    assert len(new_memory.working_memory) == 1
    assert new_memory.get_from_semantic_memory("sem_key") == "sem_value"
    assert len(new_memory.episodic_memory) == 1


def test_memory_with_messages():
    """Test memory operations with Message objects"""
    memory = Memory()

    # Create test messages
    message1 = Message(role=MessageRole.USER, content="Test message 1")
    message2 = Message(role=MessageRole.ASSISTANT, content="Test message 2")

    # Test working memory
    memory.add_to_working_memory(message1)
    memory.add_to_working_memory(message2)
    items = memory.get_from_working_memory()
    assert len(items) == 2
    assert all(isinstance(item, Message) for item in items)

    # Test semantic memory
    memory.add_to_semantic_memory("msg_key", message1)
    retrieved = memory.get_from_semantic_memory("msg_key")
    assert isinstance(retrieved, Message)
    assert retrieved.content == "Test message 1"


def test_memory_stats(memory):
    """Test memory statistics"""
    # Add various memories
    memory.add_to_working_memory("work1")
    memory.add_to_semantic_memory("key1", "value1")
    memory.add_to_episodic_memory("episode1")
    memory.add_to_working_memory("important", importance=0.8)

    # Get stats
    stats = memory.get_memory_stats()
    assert stats["working_memory_size"] == 2
    assert stats["semantic_memory_size"] == 1
    assert stats["episodic_memory_size"] == 1
    assert stats["total_memories"] > 0
