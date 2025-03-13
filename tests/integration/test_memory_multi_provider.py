"""
Integration tests for memory-enabled task agent with multi-provider model
"""

import pytest
import asyncio
from pathlib import Path
import tempfile

from llm_agents.core.memory_task_agent import MemoryEnabledTaskAgent
from llm_agents.models.multi_provider import MultiProviderModel, OptimizationStrategy
from llm_agents.runtime.memory_runner import MemoryAgentRunner
from llm_agents.models.base import ModelCapability
from llm_agents.core.message import Message, MessageRole


async def create_test_agent():
    """Create test agent with multi-provider model"""
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

        # Store initial memory
        response = await runner.run("My name is Bob", session_id=session_id)
        assert "Bob" in response.content.lower()

        # Save and create new runner
        await runner.save_memory(session_id)
        await runner.clear_conversation()

        new_runner = MemoryAgentRunner(agent=agent, memory_persistence_path=temp_dir)

        # Check memory persistence
        response = await new_runner.run("What's my name?", session_id=session_id)
        assert "Bob" in response.content.lower()


@pytest.mark.asyncio
async def test_provider_failover():
    """Test automatic provider failover"""
    agent, model = await create_test_agent()
    runner = MemoryAgentRunner(agent=agent)

    # Force primary provider to fail
    model.switch_provider("openai")

    # Should failover to backup provider
    response = await runner.run("Hello!")
    assert response is not None
    assert len(response.content) > 0

    # Check stats recorded failure
    stats = model.get_provider_stats()
    assert stats["anthropic"]["failures"] > 0
    assert stats["openai"]["successes"] > 0


@pytest.mark.asyncio
async def test_memory_retrieval():
    """Test memory retrieval and relevance"""
    agent, _ = await create_test_agent()
    runner = MemoryAgentRunner(agent=agent)

    session_id = "test_session"

    # Add multiple memories
    await runner.run("I like pizza", session_id=session_id)
    await runner.run("I'm allergic to peanuts", session_id=session_id)
    await runner.run("I live in London", session_id=session_id)

    # Test relevant memory retrieval
    response = await runner.run("What food allergies do I have?", session_id=session_id)
    assert "peanut" in response.content.lower()

    # Test memory summarization
    summary = await runner.summarize_conversation(session_id)
    assert "pizza" in summary.lower()
    assert "london" in summary.lower()
    assert "peanut" in summary.lower()


@pytest.mark.asyncio
async def test_optimization_strategies():
    """Test different provider optimization strategies"""
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
