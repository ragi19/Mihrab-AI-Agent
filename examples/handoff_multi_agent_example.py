"""
Handoff Multi-Agent Example

This example demonstrates a multi-agent system with handoff capabilities.
"""
import asyncio
from mihrabai import create_agent
from mihrabai.core.message import Message, MessageRole
from mihrabai.handoff import HandoffAgent, HandoffCondition
from mihrabai.runtime.coordinator import AgentCoordinator

async def main():
    # Create specialized agents
    math_agent = await create_agent(
        "openai",
        "gpt-4",
        system_message="You are a mathematics specialist that handles complex calculations and math problems."
    )
    
    code_agent = await create_agent(
        "openai",
        "gpt-4",
        system_message="You are a coding specialist that helps with programming questions and code reviews."
    )
    
    general_agent = await create_agent(
        "openai",
        "gpt-3.5-turbo",
        system_message="You are a general-purpose assistant that handles common queries and routes specialized questions."
    )
    
    # Create handoff conditions
    math_condition = HandoffCondition(
        "math",
        "Query requires mathematical calculations or mathematical theory.",
        ["calculate", "solve", "equation", "mathematics", "algebra", "calculus"]
    )
    
    code_condition = HandoffCondition(
        "code",
        "Query involves programming or code-related questions.",
        ["code", "programming", "function", "bug", "error", "algorithm"]
    )
    
    # Create handoff-enabled agents
    math_handoff = HandoffAgent(math_agent, [math_condition])
    code_handoff = HandoffAgent(code_agent, [code_condition])
    general_handoff = HandoffAgent(general_agent, [])
    
    # Create coordinator with handoff chain
    coordinator = AgentCoordinator(
        [general_handoff, math_handoff, code_handoff],
        enable_handoff=True
    )
    
    # Example queries
    queries = [
        "What's the square root of 256?",
        "How do I write a recursive function in Python?",
        "What's the weather like in London?"
    ]
    
    # Process queries
    for query in queries:
        print(f"\nQuery: {query}")
        result = await coordinator.process_query(
            Message(role=MessageRole.USER, content=query)
        )
        print(f"Response: {result.content}")

if __name__ == "__main__":
    asyncio.run(main())