"""
Example demonstrating logging and profiling capabilities
"""

import asyncio
from pathlib import Path

from mihrabai import Message, MessageRole, config, create_agent
from mihrabai.utils.dev_tools import AgentProfiler


async def main():
    # Configure logging to file
    config.set_logging_config(
        {
            "level": "DEBUG",
            "file": "agent_debug.log",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        }
    )

    # Create an agent
    agent = await create_agent(
        "openai",
        "gpt-3.5-turbo",
        system_message="You are a helpful AI assistant that provides concise responses.",
    )

    # Create test messages
    messages = [
        Message(role=MessageRole.USER, content="What is the capital of France?"),
        Message(role=MessageRole.USER, content="What is the population of Paris?"),
        Message(
            role=MessageRole.USER, content="What are some famous landmarks in Paris?"
        ),
    ]

    # Profile agent performance
    profiler = AgentProfiler()
    metrics = await profiler.profile_agent(agent, messages)

    # Print profiling results
    print("\nAgent Performance Metrics:")
    print(f"Total Requests: {metrics['total_requests']}")
    print(f"Total Time: {metrics['total_time']:.2f}s")
    print(f"Average Request Time: {metrics['avg_request_time']:.2f}s")

    if "total_tokens" in metrics:
        print(f"\nToken Usage:")
        print(f"Total Tokens: {metrics['total_tokens']}")
        print(f"Average Tokens per Request: {metrics['avg_tokens']:.1f}")

    print("\nCheck agent_debug.log for detailed logging information")


if __name__ == "__main__":
    asyncio.run(main())
