#!/usr/bin/env python
"""
Advanced Handoff System Example

This example demonstrates sophisticated handoff patterns including:
1. Conditional handoffs based on message content analysis
2. Context preservation during handoffs
3. Chain handoffs (A -> B -> C)
4. Specialized agents with different capabilities
"""

import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from llm_agents.core.memory import Memory
from llm_agents.core.memory_task_agent import MemoryEnabledTaskAgent
from llm_agents.core.message import Message, MessageRole
from llm_agents.core.task_agent import ToolConfig

# Import handoff components
from llm_agents.handoff import (
    HandoffAgent,
    HandoffConfig,
    HandoffInputData,
    preserve_context,
    preserve_user_messages_only,
    remove_sensitive_information,
    summarize_previous_responses,
)
from llm_agents.models.base import ModelCapability

# Import required modules
from llm_agents.models.multi_provider import MultiProviderModel, OptimizationStrategy
from llm_agents.runtime.memory_runner import MemoryAgentRunner
from llm_agents.tools.standard import CalculatorTool, DateTimeTool


# Handoff conditions
def technical_complexity_condition(message: str, context: Dict[str, Any]) -> bool:
    """Check if message requires technical expertise"""
    technical_terms = [
        "code",
        "error",
        "bug",
        "programming",
        "database",
        "api",
        "server",
        "network",
        "configuration",
        "system",
    ]
    term_count = sum(1 for term in technical_terms if term.lower() in message.lower())
    return term_count >= 2


def needs_research_condition(message: str, context: Dict[str, Any]) -> bool:
    """Check if message requires research"""
    research_terms = ["research", "study", "analysis", "investigate", "compare"]
    return any(term in message.lower() for term in research_terms)


def needs_calculation_condition(message: str, context: Dict[str, Any]) -> bool:
    """Check if message requires calculations"""
    calc_terms = ["calculate", "compute", "sum", "average", "percentage"]
    return any(term in message.lower() for term in calc_terms)


class AdvancedHandoffSystem:
    """Advanced multi-agent system with sophisticated handoff capabilities"""

    def __init__(self):
        self.agents = {}
        self.model = None

    async def initialize(self):
        """Initialize the advanced handoff system"""
        # Set up the model
        self.model = await MultiProviderModel.create(
            primary_model="llama3-70b-8192",
            fallback_models=[],
            required_capabilities={
                ModelCapability.CHAT,
                ModelCapability.FUNCTION_CALLING,
            },
            optimize_for=OptimizationStrategy.PERFORMANCE,
        )

        # Create specialized agents

        # Technical Agent
        self.agents["technical"] = HandoffAgent(
            name="Technical",
            system_prompt=(
                "You are a technical expert specializing in software, hardware, and system issues. "
                "Provide detailed technical solutions and explanations."
            ),
            tools=[CalculatorTool()],
            expertise=["programming", "debugging", "system configuration"],
            can_defer=True,
        )

        # Research Agent
        self.agents["research"] = HandoffAgent(
            name="Research",
            system_prompt=(
                "You are a research specialist focusing on gathering and analyzing information. "
                "Provide well-researched, factual responses with citations when possible."
            ),
            expertise=["research", "analysis", "investigation"],
            can_defer=True,
        )

        # Calculator Agent
        self.agents["calculator"] = HandoffAgent(
            name="Calculator",
            system_prompt=(
                "You are a mathematical analysis specialist. Focus on numerical calculations "
                "and mathematical problem-solving."
            ),
            tools=[CalculatorTool()],
            expertise=["calculations", "math", "statistics"],
            can_defer=False,
        )

        # General Agent (with handoff capabilities)
        self.agents["general"] = HandoffAgent(
            name="General",
            system_prompt=(
                "You are a general-purpose assistant that can handle basic queries and coordinate "
                "with specialized agents when needed. Always be helpful and clear in your responses."
            ),
            handoffs=[
                HandoffConfig(
                    name="technical_handoff",
                    description="Transfer to technical agent for complex technical issues",
                    target_agent=self.agents["technical"],
                    condition=technical_complexity_condition,
                    input_filter=preserve_context,
                ),
                HandoffConfig(
                    name="research_handoff",
                    description="Transfer to research agent for research-heavy queries",
                    target_agent=self.agents["research"],
                    condition=needs_research_condition,
                    input_filter=summarize_previous_responses,
                ),
                HandoffConfig(
                    name="calculator_handoff",
                    description="Transfer to calculator agent for mathematical queries",
                    target_agent=self.agents["calculator"],
                    condition=needs_calculation_condition,
                    input_filter=preserve_user_messages_only,
                ),
            ],
        )

        # Initialize all agents
        for agent in self.agents.values():
            await agent.initialize(self.model)
            logger.info(f"Initialized agent: {agent.name}")

    async def process_query(
        self, query: str, session_id: str = "default_session"
    ) -> Dict[str, Any]:
        """Process a query through the advanced handoff system"""
        try:
            # Start with the general agent
            response, metadata = await self.agents["general"].process(
                message=query, session_id=session_id, context={"initial_query": query}
            )

            # Return results including handoff information
            result = {
                "response": response,
                "final_agent": metadata.get("final_agent", "general"),
                "handoff_chain": metadata.get("handoff_chain", []),
                "expertise_used": metadata.get("expertise_used", []),
            }

            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"Error processing your request: {str(e)}",
                "final_agent": "general",
                "handoff_chain": [],
                "expertise_used": [],
            }


async def main():
    """Main function to run the advanced handoff example"""
    try:
        # Check for API key
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("\nError: GROQ_API_KEY environment variable not found!")
            print("Please set your Groq API key using:")
            print("$env:GROQ_API_KEY = 'your-api-key'")
            return

        print("\nInitializing advanced handoff system...")

        # Initialize the system
        system = AdvancedHandoffSystem()
        await system.initialize()

        # Test queries demonstrating different handoff scenarios
        test_queries = [
            "Can you help debug this Python code that's giving a TypeError?",  # Technical
            "Research the impact of quantum computing on cryptography",  # Research
            "Calculate the compound interest on $1000 at 5% for 3 years",  # Calculator
            "What's the weather like today?",  # General
            "Analyze this code and calculate its time complexity",  # Technical + Calculator
        ]

        print("\nProcessing test queries...")
        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"QUERY: {query}")
            print(f"{'='*80}")

            result = await system.process_query(query)

            print(f"\nResponse from {result['final_agent']} Agent:")
            print(f"{'='*40}")
            print(result["response"])

            if result["handoff_chain"]:
                print(f"\nHandoff Chain: {' -> '.join(result['handoff_chain'])}")

            if result["expertise_used"]:
                print(f"Expertise Used: {', '.join(result['expertise_used'])}")

            print(f"{'='*80}")

    except Exception as e:
        print(f"\nError in main: {str(e)}")
        logger.exception("Detailed error information:")


if __name__ == "__main__":
    asyncio.run(main())
