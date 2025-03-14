#!/usr/bin/env python
"""
Example demonstrating a multi-agent system with handoff capabilities.

This example shows:
1. Creating specialized agents for different domains
2. Configuring handoff rules between agents
3. Processing queries through a general agent that can hand off to specialists
4. Maintaining context between handoffs
"""

import asyncio
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from llm_agents.core.memory import Memory
from llm_agents.core.memory_task_agent import MemoryEnabledTaskAgent
from llm_agents.core.message import Message, MessageRole
from llm_agents.models.base import ModelCapability

# Import required modules
from llm_agents.models.multi_provider import MultiProviderModel, OptimizationStrategy
from llm_agents.runtime.memory_runner import MemoryAgentRunner
from llm_agents.tools.standard import CalculatorTool, DateTimeTool

print("Starting handoff multi-agent example...")


class HandoffConfig:
    """Configuration for agent handoffs"""

    def __init__(
        self,
        name: str,
        description: str,
        target_agent: "HandoffAgent",
        input_filter: Optional[Callable] = None,
    ):
        self.name = name
        self.description = description
        self.target_agent = target_agent
        self.input_filter = input_filter
        print(f"Created HandoffConfig: {name} -> {target_agent.name}")


class HandoffInputData:
    """Data structure for handoff inputs"""

    def __init__(
        self,
        conversation_history: List[Message],
        system_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.conversation_history = conversation_history
        self.system_message = system_message
        self.metadata = metadata or {}


class HandoffAgent:
    """Agent with handoff capabilities"""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: Optional[List[Any]] = None,
        handoffs: Optional[List[HandoffConfig]] = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.handoffs = handoffs or []
        self.agent = None
        self.runner = None
        print(f"Created HandoffAgent: {name}")

    async def initialize(self, model: MultiProviderModel):
        """Initialize the agent with a model"""
        print(f"Initializing agent: {self.name}")
        try:
            # Create memory
            memory = Memory(working_memory_size=10, long_term_memory_size=50)

            # Create agent
            self.agent = MemoryEnabledTaskAgent(
                model=model,
                memory=memory,
                system_message=self.system_prompt,
                tools=self.tools,
            )

            # Create runner
            self.runner = MemoryAgentRunner(
                agent=self.agent, memory_persistence_path="./memories"
            )

            print(f"Successfully initialized agent: {self.name}")
            logger.info(f"Initialized agent: {self.name}")
        except Exception as e:
            print(f"Error initializing agent {self.name}: {str(e)}")
            logger.error(
                f"Error initializing agent {self.name}: {str(e)}", exc_info=True
            )
            raise

    async def process(self, message: str, session_id: str = None) -> str:
        """Process a message with potential handoffs"""
        print(f"Agent {self.name} processing message: {message[:50]}...")

        # Check if we should hand off to a specialized agent
        for handoff in self.handoffs:
            if self._should_handoff(message, handoff):
                print(f"Handing off from {self.name} to {handoff.target_agent.name}")
                logger.info(f"Handing off to {handoff.target_agent.name}")
                return await handoff.target_agent.process(
                    message, session_id=session_id
                )

        # Process with current agent if no handoff needed
        print(f"No handoff needed, {self.name} processing directly")
        try:
            response = await self.runner.run(message, session_id=session_id)
            print(f"Agent {self.name} response: {response.content[:50]}...")
            return response.content
        except Exception as e:
            print(f"Error in agent {self.name} processing: {str(e)}")
            logger.error(
                f"Error in agent {self.name} processing: {str(e)}", exc_info=True
            )
            return f"Error processing your request: {str(e)}"

    def _should_handoff(self, message: str, handoff: HandoffConfig) -> bool:
        """Determine if message should be handed off to another agent"""
        # Simple keyword-based handoff logic
        keywords = {
            "technical": [
                "error",
                "bug",
                "broken",
                "not working",
                "help with",
                "how to",
                "fix",
            ],
            "billing": [
                "payment",
                "invoice",
                "charge",
                "refund",
                "price",
                "cost",
                "subscription",
            ],
            "scheduling": [
                "schedule",
                "appointment",
                "book",
                "meeting",
                "when",
                "time",
                "calendar",
            ],
        }

        if handoff.name in keywords:
            return any(kw in message.lower() for kw in keywords[handoff.name])
        return False


# Define a simple filter function to demonstrate handoff filtering
def remove_all_tools(input_data: HandoffInputData) -> HandoffInputData:
    """Filter that removes all tool-related messages from the conversation history"""
    return HandoffInputData(
        conversation_history=[
            msg
            for msg in input_data.conversation_history
            if not (msg.role == MessageRole.FUNCTION or msg.role == MessageRole.TOOL)
        ],
        system_message=input_data.system_message,
        metadata=input_data.metadata,
    )


class HandoffMultiAgentSystem:
    """Multi-agent system with handoff capabilities"""

    def __init__(self):
        self.agents = {}
        self.model = None
        print("Created HandoffMultiAgentSystem")

    async def initialize(self):
        """Initialize the handoff-enabled multi-agent system"""
        try:
            print("Checking for GROQ_API_KEY...")
            # Check for API key
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("Please set the GROQ_API_KEY environment variable")
            print("GROQ_API_KEY found.")

            print("Setting up the model...")
            # Set up the model
            self.model = await MultiProviderModel.create(
                primary_model="llama-3.1-8b-instant",  # Using a model that supports function calling
                fallback_models=[],
                required_capabilities={ModelCapability.CHAT},
                optimize_for=OptimizationStrategy.PERFORMANCE,
            )
            logger.info(
                f"Created multi-provider model with primary: {self.model.current_provider}"
            )
            print(f"Model setup complete with provider: {self.model.current_provider}")

            # Initialize tools
            calculator_tool = CalculatorTool()
            datetime_tool = DateTimeTool()

            print("Creating specialized agents...")
            # Create technical agent with calculator tool
            self.agents["technical"] = HandoffAgent(
                name="Technical",
                system_prompt=(
                    "You are a technical support agent who helps users with computer and software issues. "
                    "Provide clear, step-by-step solutions. Ask clarifying questions when needed. "
                    "You have access to a calculator tool to help with technical calculations."
                ),
                tools=[calculator_tool],
            )

            # Create billing agent
            self.agents["billing"] = HandoffAgent(
                name="Billing",
                system_prompt=(
                    "You are a billing support agent who helps with payment and invoice issues. "
                    "Be clear about costs and policies. Maintain confidentiality."
                ),
            )

            # Create scheduling agent with datetime tool
            self.agents["scheduling"] = HandoffAgent(
                name="Scheduling",
                system_prompt=(
                    "You are a scheduling assistant who helps users book appointments and meetings. "
                    "Be efficient and clear about availability and timing. "
                    "You have access to a datetime tool to help with scheduling."
                ),
                tools=[datetime_tool],
            )

            # Initialize specialized agents
            print("Initializing specialized agents...")
            for agent_name, agent in self.agents.items():
                print(f"Initializing {agent_name} agent...")
                await agent.initialize(self.model)

            # Create general agent with handoff capabilities to all specialized agents
            print("Creating general agent with handoffs...")
            self.agents["general"] = HandoffAgent(
                name="General",
                system_prompt=(
                    "You are a general customer service agent who can help with a wide range of inquiries. "
                    "For technical issues, transfer to the Technical agent. "
                    "For billing questions, transfer to the Billing agent. "
                    "For scheduling needs, transfer to the Scheduling agent. "
                    "Handle general inquiries yourself. Be friendly and helpful."
                ),
                handoffs=[
                    HandoffConfig(
                        name="technical",
                        description="the user has a technical issue or question",
                        target_agent=self.agents["technical"],
                        input_filter=remove_all_tools,
                    ),
                    HandoffConfig(
                        name="billing",
                        description="the user has a billing inquiry or payment issue",
                        target_agent=self.agents["billing"],
                    ),
                    HandoffConfig(
                        name="scheduling",
                        description="the user needs help with scheduling or calendar management",
                        target_agent=self.agents["scheduling"],
                    ),
                ],
            )

            print("Initializing general agent...")
            await self.agents["general"].initialize(self.model)
            print("All agents initialized successfully")

        except Exception as e:
            print(f"Error in handoff multi-agent system: {str(e)}")
            logger.exception("Detailed error information:")
            raise

    async def process_query(self, query: str) -> str:
        """Process a query through the general agent with potential handoffs"""
        print(f"Processing query: {query}")
        logger.info("Processing query through handoff-enabled multi-agent system...")
        return await self.agents["general"].process(query, session_id="handoff_demo")


async def main():
    """Main function to run the handoff-enabled multi-agent system"""
    try:
        # Check for GROQ_API_KEY first
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("\nError: GROQ_API_KEY environment variable not found!")
            print("Please set your Groq API key using:")
            print("$env:GROQ_API_KEY = 'your-api-key'")
            return

        print("\nInitializing handoff-enabled multi-agent system...")

        # Initialize the system
        system = HandoffMultiAgentSystem()
        await system.initialize()

        # Test queries
        test_queries = [
            "How do I fix my computer that keeps crashing?",  # Technical query
            "I need to schedule a meeting for next week",  # Scheduling query
            "Can you help me with my billing issue?",  # Billing query
            "Tell me about your services",  # General query
        ]

        print("\nProcessing test queries...")
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"QUERY: {query}")
            print(f"{'='*50}")
            response = await system.process_query(query)
            print(f"\nRESPONSE: {response}")
            print(f"{'='*50}")

    except Exception as e:
        print(f"\nError in main: {str(e)}")
        logger.exception("Detailed error information:")


if __name__ == "__main__":
    asyncio.run(main())
