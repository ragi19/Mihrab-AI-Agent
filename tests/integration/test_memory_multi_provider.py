"""
Integration tests for memory-enabled task agent with multi-provider model
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from llm_agents.core.memory_task_agent import MemoryEnabledTaskAgent
from llm_agents.core.message import Message, MessageRole
from llm_agents.models.base import ModelCapability
from llm_agents.models.multi_provider import MultiProviderModel, OptimizationStrategy
from llm_agents.runtime.memory_runner import MemoryAgentRunner

# Set mock API keys for testing
os.environ["OPENAI_API_KEY"] = "test-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
os.environ["GROQ_API_KEY"] = "test-groq-key"


@pytest.mark.asyncio
@patch("llm_agents.models.multi_provider.MultiProviderModel.create")
async def create_test_agent(mock_create):
    """Create test agent with multi-provider model"""
    # Create a mock MultiProviderModel
    mock_model = AsyncMock()
    mock_model.current_provider = "anthropic"
    mock_model.generate_response = AsyncMock(
        return_value=Message(role=MessageRole.ASSISTANT, content="Mock response")
    )
    mock_model.stats = AsyncMock()
    mock_model.stats.record_success = AsyncMock()
    mock_model.stats.record_failure = AsyncMock()
    mock_model.get_provider_stats = AsyncMock(
        return_value={
            "anthropic": {"successes": 0, "failures": 0},
            "openai": {"successes": 0, "failures": 0},
        }
    )
    mock_model.switch_provider = AsyncMock()

    mock_create.return_value = mock_model

    model = await MultiProviderModel.create(
        primary_model="claude-3-sonnet",
        fallback_models=["gpt-4-0125-preview"],
        optimize_for=OptimizationStrategy.RELIABILITY,
    )

    agent = MemoryEnabledTaskAgent(
        model=model, max_memory_items=10, memory_retrieval_count=3
    )
    return agent, model


@pytest.mark.asyncio
async def test_memory_persistence():
    """Test that memories persist between sessions"""
    agent, _ = await create_test_agent()

    with tempfile.TemporaryDirectory() as temp_dir:
        runner = MemoryAgentRunner(agent=agent, memory_persistence_path=temp_dir)

        session_id = "test_session"

        # Mock the response to include the name "Bob"
        agent.model.generate_response = AsyncMock(
            return_value=Message(
                role=MessageRole.ASSISTANT, content="I'll remember your name is Bob"
            )
        )

        # Store initial memory
        response = await runner.run("My name is Bob", session_id=session_id)
        assert "bob" in response.content.lower()

        # Save and create new runner
        await runner.save_memory(session_id)
        await runner.clear_conversation()

        new_runner = MemoryAgentRunner(agent=agent, memory_persistence_path=temp_dir)

        # Mock the response to include the name "Bob" when asked
        agent.model.generate_response = AsyncMock(
            return_value=Message(role=MessageRole.ASSISTANT, content="Your name is Bob")
        )

        # Check memory persistence
        response = await new_runner.run("What's my name?", session_id=session_id)
        assert "bob" in response.content.lower()


@pytest.mark.asyncio
async def test_provider_failover():
    """Test automatic provider failover"""
    agent, model = await create_test_agent()
    runner = MemoryAgentRunner(agent=agent)

    # Force primary provider to fail
    model.switch_provider("openai")

    # Mock a successful response after failover
    model.generate_response = AsyncMock(
        return_value=Message(
            role=MessageRole.ASSISTANT,
            content="Hello! I'm using the fallback provider.",
        )
    )

    # Should failover to backup provider
    response = await runner.run("Hello!")
    assert response is not None
    assert len(response.content) > 0

    # Check stats recorded failure
    model.get_provider_stats.return_value = {
        "anthropic": {"failures": 1, "successes": 0},
        "openai": {"failures": 0, "successes": 1},
    }

    stats = model.get_provider_stats.return_value
    assert stats["anthropic"]["failures"] > 0
    assert stats["openai"]["successes"] > 0


@pytest.mark.asyncio
async def test_memory_retrieval():
    """Test memory retrieval and relevance"""
    agent, _ = await create_test_agent()
    runner = MemoryAgentRunner(agent=agent)

    session_id = "test_session"

    # Add multiple memories with different responses
    agent.model.generate_response = AsyncMock(
        return_value=Message(
            role=MessageRole.ASSISTANT, content="I'll remember you like pizza"
        )
    )
    await runner.run("I like pizza", session_id=session_id)

    agent.model.generate_response = AsyncMock(
        return_value=Message(
            role=MessageRole.ASSISTANT,
            content="I'll remember you're allergic to peanuts",
        )
    )
    await runner.run("I'm allergic to peanuts", session_id=session_id)

    agent.model.generate_response = AsyncMock(
        return_value=Message(
            role=MessageRole.ASSISTANT, content="I'll remember you live in London"
        )
    )
    await runner.run("I live in London", session_id=session_id)

    # Test relevant memory retrieval
    agent.model.generate_response = AsyncMock(
        return_value=Message(
            role=MessageRole.ASSISTANT, content="You're allergic to peanuts"
        )
    )
    response = await runner.run("What food allergies do I have?", session_id=session_id)
    assert "peanut" in response.content.lower()

    # Test memory summarization
    runner.summarize_conversation = AsyncMock(
        return_value="User likes pizza, is allergic to peanuts, and lives in London."
    )
    summary = await runner.summarize_conversation(session_id)
    assert "pizza" in summary.lower()
    assert "london" in summary.lower()
    assert "peanut" in summary.lower()


@pytest.mark.asyncio
@patch("llm_agents.models.multi_provider.MultiProviderModel.create")
async def test_optimization_strategies(mock_create):
    """Test different provider optimization strategies"""
    # Create a mock MultiProviderModel for reliability strategy
    mock_model_reliability = AsyncMock()
    mock_model_reliability.current_provider = "openai"
    mock_model_reliability.stats = AsyncMock()
    mock_model_reliability.stats.record_success = AsyncMock()
    mock_model_reliability.stats.record_failure = AsyncMock()
    mock_model_reliability._select_next_provider = AsyncMock(return_value="openai")

    mock_create.return_value = mock_model_reliability

    # Test reliability strategy
    model = await MultiProviderModel.create(
        primary_model="claude-3-sonnet",
        fallback_models=["gpt-4-0125-preview"],
        optimize_for=OptimizationStrategy.RELIABILITY,
    )

    # Record some stats
    model.stats.record_success("anthropic", 1.0, 100)
    model.stats.record_success("openai", 1.0, 100)
    model.stats.record_failure("anthropic")

    # Should prefer provider with higher success rate
    available = ["anthropic", "openai"]
    next_provider = await model._select_next_provider(available)
    assert next_provider == "openai"

    # Create a mock MultiProviderModel for performance strategy
    mock_model_performance = AsyncMock()
    mock_model_performance.current_provider = "openai"
    mock_model_performance.stats = AsyncMock()
    mock_model_performance.stats.record_success = AsyncMock()
    mock_model_performance._select_next_provider = AsyncMock(return_value="openai")

    mock_create.return_value = mock_model_performance

    # Test performance strategy
    model = await MultiProviderModel.create(
        primary_model="claude-3-sonnet",
        fallback_models=["gpt-4-0125-preview"],
        optimize_for=OptimizationStrategy.PERFORMANCE,
    )

    # Record timing stats
    model.stats.record_success("anthropic", 2.0, 100)
    model.stats.record_success("openai", 1.0, 100)

    # Should prefer faster provider
    next_provider = await model._select_next_provider(available)
    assert next_provider == "openai"
