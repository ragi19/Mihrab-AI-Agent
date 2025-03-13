"""
Example of using memory-enabled task agent with multi-provider model
"""
import asyncio
import os
from pathlib import Path

from llm_agents.core.memory_task_agent import MemoryEnabledTaskAgent
from llm_agents.models.multi_provider import MultiProviderModel, OptimizationStrategy
from llm_agents.runtime.memory_runner import MemoryAgentRunner
from llm_agents.models.base import ModelCapability
from llm_agents.core.message import Message, MessageRole
from llm_agents.tools.standard import CalculatorTool
from llm_agents.utils.logging import get_logger, configure_logging

# Configure logging
configure_logging()
logger = get_logger(__name__)

async def main():
    # Initialize multi-provider model
    model = await MultiProviderModel.create(
        primary_model="claude-3-sonnet",  # Anthropic primary
        fallback_models=[
            "gpt-4-0125-preview",  # OpenAI fallback
            "mixtral-8x7b-instruct"  # Groq fallback
        ],
        required_capabilities={
            ModelCapability.CHAT,
            ModelCapability.SYSTEM_MESSAGES
        },
        optimize_for=OptimizationStrategy.RELIABILITY
    )
    
    # Create memory-enabled agent
    agent = MemoryEnabledTaskAgent(
        model=model,
        tools=[CalculatorTool()],
        max_memory_items=100,
        memory_retrieval_count=5,
        automatic_memory=True
    )
    
    # Create runner with memory persistence
    memory_dir = Path("./memories")
    memory_dir.mkdir(exist_ok=True)
    
    runner = MemoryAgentRunner(
        agent=agent,
        memory_persistence_path=str(memory_dir),
        auto_save_interval_seconds=60
    )
    
    # Example conversation with memory
    session_id = "example_session"
    
    # Initial interaction storing memory
    response = await runner.run(
        "My name is Alice and I'm a software engineer working on ML projects",
        session_id=session_id
    )
    logger.info(f"Response: {response}")
    
    # Ask about remembered information
    response = await runner.run(
        "What do you remember about me and my work?",
        session_id=session_id
    )
    logger.info(f"Response: {response}")
    
    # Add more context
    response = await runner.run(
        "I'm currently working on a project using PyTorch and transformers",
        session_id=session_id
    )
    logger.info(f"Response: {response}")
    
    # Test provider failover by simulating error
    model.switch_provider("openai")  # Switch to fallback
    
    response = await runner.run(
        "Can you summarize what you know about my ML work?",
        session_id=session_id
    )
    logger.info(f"Response: {response}")
    
    # Check provider stats
    stats = model.get_provider_stats()
    logger.info("Provider Statistics:")
    for provider, metrics in stats.items():
        logger.info(f"{provider}: {metrics}")
        
    # Save final state
    await runner.save_memory(session_id)
    
    # Clean up
    await runner.clear_conversation()

if __name__ == "__main__":
    asyncio.run(main())