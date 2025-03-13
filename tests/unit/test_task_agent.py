"""
Unit tests for task agent implementations
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, create_autospec, MagicMock
from llm_agents.core.task_agent import TaskAgent
from llm_agents.core.memory_task_agent import MemoryEnabledTaskAgent
from llm_agents.core.message import Message, MessageRole
from llm_agents.models.base import BaseModel
from llm_agents.utils.tracing import TraceProvider, Trace, Span
import contextlib

class MockContextManager:
    def __init__(self):
        self.set_metadata = Mock()
        self.set_error = Mock()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

@pytest.fixture
def mock_model():
    model = Mock(spec=BaseModel)
    model.model_name = "test-model"
    model.generate_response = AsyncMock()
    model.count_tokens = AsyncMock(return_value=10)
    return model

@pytest.fixture
def mock_span():
    return MockContextManager()

@pytest.fixture
def mock_trace():
    trace = Mock(spec=Trace)
    trace.start = Mock()
    return trace

@pytest.fixture
def mock_trace_provider(mock_span, mock_trace):
    provider = Mock(spec=TraceProvider)
    provider.create_trace = Mock(return_value=mock_trace)
    provider.create_span = Mock(return_value=mock_span)
    provider.end_trace = Mock()
    return provider

@pytest.fixture
def task_agent(mock_model, mock_trace_provider):
    agent = TaskAgent(
        model=mock_model,
        max_steps=5,
        max_history_tokens=1000,
        trace_provider=mock_trace_provider
    )
    return agent

@pytest.fixture
def memory_task_agent(mock_model, mock_trace_provider):
    agent = MemoryEnabledTaskAgent(
        model=mock_model,
        max_steps=5,
        max_history_tokens=1000,
        working_memory_size=5,
        trace_provider=mock_trace_provider
    )
    return agent

@pytest.mark.asyncio
async def test_task_agent_initialization(task_agent):
    """Test task agent initialization"""
    assert task_agent.current_step == 0
    assert task_agent.max_steps == 5
    assert len(task_agent.conversation_history) == 1  # System message
    assert task_agent.conversation_history[0].role == MessageRole.SYSTEM

@pytest.mark.asyncio
async def test_task_agent_step_tracking(task_agent):
    """Test task step counting and limits"""
    message = Message(role=MessageRole.USER, content="Test task")
    task_agent.model.generate_response.return_value = Message(
        role=MessageRole.ASSISTANT,
        content="Test response"
    )
    
    # Process messages up to step limit
    for _ in range(task_agent.max_steps):
        response = await task_agent.process_message(message)
        assert response.role == MessageRole.ASSISTANT
    
    # Verify step limit
    response = await task_agent.process_message(message)
    assert "exceeded maximum allowed steps" in response.content
    
    # Verify trace provider was used
    assert task_agent._trace_provider.create_trace.called

@pytest.mark.asyncio
async def test_task_agent_state_management(task_agent):
    """Test task state management"""
    message = Message(
        role=MessageRole.USER,
        content="Test task",
        metadata={"task_updates": {"progress": 50}}
    )
    
    task_agent.model.generate_response.return_value = Message(
        role=MessageRole.ASSISTANT,
        content="Progress updated",
        metadata={"task_updates": {"progress": 75}}
    )
    
    await task_agent.process_message(message)
    state = task_agent.get_task_state()
    assert state["current_step"] == 1
    assert state["state"].get("progress") == 75

@pytest.mark.asyncio
async def test_memory_task_agent_integration(memory_task_agent):
    """Test memory-enabled task agent functionality"""
    # Setup test message
    message = Message(
        role=MessageRole.USER,
        content="Test memory task",
        metadata={"importance": 0.8}
    )
    
    memory_task_agent.model.generate_response.return_value = Message(
        role=MessageRole.ASSISTANT,
        content="Test response",
        metadata={"task_updates": {"status": "completed"}}
    )
    
    # Process message
    response = await memory_task_agent.process_message(message)
    
    # Verify memory integration
    memory_stats = memory_task_agent.get_memory_stats()
    assert memory_stats["memory"]["working_memory_size"] > 0
    assert memory_stats["memory"]["episodic_memory_size"] > 0

@pytest.mark.asyncio
async def test_memory_task_agent_context_enrichment(memory_task_agent):
    """Test context enrichment with memories"""
    # Setup response
    memory_task_agent.model.generate_response.return_value = Message(
        role=MessageRole.ASSISTANT,
        content="Contextual response"
    )
    
    # Add some initial memories
    memory_task_agent.memory.add_to_working_memory("Previous context")
    memory_task_agent.memory.add_to_episodic_memory("Previous episode")
    
    # Process message
    message = Message(role=MessageRole.USER, content="Test with context")
    response = await memory_task_agent.process_message(message)
    
    # Extract the calls to ensure enriched context is passed
    assert memory_task_agent.model.generate_response.called
    # Verify that memory context was included
    calls = memory_task_agent.model.generate_response.call_args_list
    assert len(calls) > 0
    call_args = calls[0][0][0]  # Get first positional arg of first call
    assert any(isinstance(msg, Message) and "Memory Context" in msg.content for msg in call_args)

@pytest.mark.asyncio
@patch('json.dump')  # Mock json.dump to avoid actual file operations
@patch('json.load')
@patch('builtins.open', create=True)
async def test_memory_task_agent_state_persistence(mock_open, mock_json_load, mock_json_dump, memory_task_agent, tmp_path):
    """Test state persistence for memory-enabled task agent"""
    # Setup initial state
    message = Message(
        role=MessageRole.USER,
        content="Test persistence",
        metadata={"task_updates": {"status": "in_progress"}}
    )
    
    memory_task_agent.model.generate_response.return_value = Message(
        role=MessageRole.ASSISTANT,
        content="State saved"
    )
    
    # Setup mocks
    mock_json_load.return_value = {
        "working_memory": [],
        "long_term_memory": [],
        "semantic_memory": {},
        "episodic_memory": [{"content": "test", "timestamp": "2023-01-01T00:00:00", "importance": 1.0, "metadata": {}}]
    }
    
    await memory_task_agent.process_message(message)
    
    # Save state
    state_file = str(tmp_path / "agent_state.json")
    memory_task_agent.save_state(state_file)
    
    # Load state in a new agent
    new_agent = MemoryEnabledTaskAgent(memory_task_agent.model)
    new_agent.load_state(state_file)
    
    # Verify json operations were called
    assert mock_json_dump.called
    assert mock_json_load.called

def test_task_agent_reset(task_agent):
    """Test task agent reset functionality"""
    task_agent.current_step = 3
    task_agent.task_state = {"progress": 50}
    
    task_agent.reset_task()
    
    assert task_agent.current_step == 0
    assert task_agent.task_state == {}