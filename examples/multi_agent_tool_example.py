"""
Multi-Agent Tool Example

This example demonstrates multiple agents using shared tools.
"""
import asyncio

from mihrabai import create_agent
from mihrabai.core.message import Message, MessageRole
from mihrabai.runtime.coordinator import AgentCoordinator
from mihrabai.tools.standard import CalculatorTool, DateTimeTool


async def main():
    # Initialize tools
    calculator = CalculatorTool()
    datetime_tool = DateTimeTool()
    
    # Create an agent with tool access
    tool_agent = await create_agent(
        "openai",
        "gpt-4",
        system_message="You are a helpful assistant with access to tools for calculations and date/time operations.",
        tools=[calculator, datetime_tool]
    )
    
    # Create result processor agent
    processor = await create_agent(
        "openai",
        "gpt-3.5-turbo",
        system_message="You format and explain tool results in a clear, user-friendly way."
    )
    
    # Create coordinator
    coordinator = AgentCoordinator([tool_agent, processor])
    
    # Example query using tools
    query = "What will be the total cost of a 12.5% annual investment of $1000 in 3 years, and what date will that be?"
    
    # Process query
    result = await coordinator.process_query(
        Message(role=MessageRole.USER, content=query)
    )
    
    print(f"Final Response:\n{result.content}")

if __name__ == "__main__":
    asyncio.run(main())