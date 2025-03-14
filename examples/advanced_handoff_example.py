"""
Advanced Handoff Example

This example demonstrates advanced agent handoff patterns with specialized agents.
"""
import asyncio
from typing import List

from mihrabai import create_agent
from mihrabai.core.message import Message, MessageRole
from mihrabai.handoff import (
    ContextPreservingHandoff,
    HandoffAgent,
    HandoffCondition,
    HandoffCoordinator,
)
from mihrabai.handoff.conditions import ComplexityThreshold, TopicDetection
from mihrabai.runtime.coordinator import AgentCoordinator
from mihrabai.tools.base import BaseTool
from mihrabai.tools.standard import CalculatorTool, DateTimeTool


class SpecializedHandoffAgent(HandoffAgent):
    def __init__(self, agent, conditions: List[HandoffCondition], expertise_areas: List[str]):
        super().__init__(agent, conditions)
        self.expertise_areas = expertise_areas
        
    async def should_handle(self, message: Message) -> bool:
        # Check if query matches expertise areas
        return any(area.lower() in message.content.lower() 
                  for area in self.expertise_areas)

async def main():
    # Create tools
    calculator = CalculatorTool()
    datetime_tool = DateTimeTool()
    
    # Create specialized agents
    finance_agent = await create_agent(
        "openai",
        "gpt-4",
        system_message="You are a financial advisor specializing in investment analysis.",
        tools=[calculator]
    )
    
    planning_agent = await create_agent(
        "openai",
        "gpt-4",
        system_message="You are a planning specialist that helps with scheduling and timing.",
        tools=[datetime_tool]
    )
    
    research_agent = await create_agent(
        "openai",
        "gpt-4",
        system_message="You are a research specialist that provides detailed information."
    )
    
    coordinator_agent = await create_agent(
        "openai",
        "gpt-3.5-turbo",
        system_message="You coordinate between specialists and maintain conversation context."
    )
    
    # Create specialized handoff agents with context preservation
    finance_handoff = SpecializedHandoffAgent(
        finance_agent,
        [HandoffCondition("finance", "Query involves financial calculations or advice.",
                         ["invest", "money", "financial", "cost", "price"])],
        ["investment", "finance", "money", "budget"]
    )
    
    planning_handoff = SpecializedHandoffAgent(
        planning_agent,
        [HandoffCondition("planning", "Query involves scheduling or timing.",
                         ["when", "schedule", "timing", "plan"])],
        ["planning", "schedule", "timing", "calendar"]
    )
    
    research_handoff = SpecializedHandoffAgent(
        research_agent,
        [HandoffCondition("research", "Query requires detailed information or analysis.",
                         ["explain", "details", "information", "analysis"])],
        ["research", "analysis", "information", "explanation"]
    )
    
    # Create context-preserving coordinator
    coordinator = AgentCoordinator(
        [
            ContextPreservingHandoff(coordinator_agent),
            finance_handoff,
            planning_handoff,
            research_handoff
        ],
        enable_handoff=True,
        preserve_context=True
    )
    
    # Example complex query that requires multiple specialists
    query = """I have $10,000 to invest. Can you help me understand:
    1. What investment options are available?
    2. When would be the best time to invest?
    3. What are the potential returns and risks?"""
    
    # Process multi-part query
    print("Query:", query)
    result = await coordinator.process_query(
        Message(role=MessageRole.USER, content=query)
    )
    print("\nResponse:", result.content)
    
    # Follow-up query demonstrating context preservation
    follow_up = "Given those options, what would be the best investment strategy for a 5-year term?"
    print("\nFollow-up:", follow_up)
    result = await coordinator.process_query(
        Message(role=MessageRole.USER, content=follow_up)
    )
    print("\nResponse:", result.content)

if __name__ == "__main__":
    asyncio.run(main())