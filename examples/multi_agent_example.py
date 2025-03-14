"""
Basic Multi-Agent Example

This example demonstrates a simple system with three agents (Analyzer, Researcher, and
Synthesizer) working together to process user queries.
"""
import asyncio

from mihrabai import create_agent
from mihrabai.core.message import Message, MessageRole
from mihrabai.runtime.coordinator import AgentCoordinator


async def main():
    # Create specialized agents
    analyzer = await create_agent(
        "openai",
        "gpt-3.5-turbo",
        system_message="You are an Analyzer that breaks down complex queries into key components."
    )
    
    researcher = await create_agent(
        "openai",
        "gpt-3.5-turbo",
        system_message="You are a Researcher that finds relevant information for queries."
    )
    
    synthesizer = await create_agent(
        "openai",
        "gpt-3.5-turbo",
        system_message="You are a Synthesizer that combines information into coherent responses."
    )
    
    # Create coordinator
    coordinator = AgentCoordinator([analyzer, researcher, synthesizer])
    
    # Example query
    query = "What are the major developments in quantum computing in 2023?"
    
    # Process through agent pipeline
    result = await coordinator.process_query(
        Message(role=MessageRole.USER, content=query)
    )
    
    print(f"Final Response:\n{result.content}")

if __name__ == "__main__":
    asyncio.run(main())