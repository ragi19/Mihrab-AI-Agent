#!/usr/bin/env python
"""
Example demonstrating a handoff-enabled multi-agent system where agents can transfer control
to specialized agents based on the query type.

This example shows:
1. Setting up a general agent that handles initial queries
2. Creating specialized agents for technical, billing, and scheduling queries
3. Implementing handoffs between agents based on query content
4. Managing conversation context during handoffs
"""

import asyncio
import os
import logging
import sys
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# Import required modules
from llm_agents.models.multi_provider import MultiProviderModel, OptimizationStrategy
from llm_agents.models.base import ModelCapability
from llm_agents.core.memory import Memory
from llm_agents.core.memory_task_agent import MemoryEnabledTaskAgent
from llm_agents.runtime.memory_runner import MemoryAgentRunner
from llm_agents.core.message import Message, MessageRole

print("Starting handoff multi-agent example...")


class HandoffConfig:
    """Configuration for agent handoffs"""

    def __init__(
        self,
        name: str,
        description: str,
        target_agent: "HandoffAgent",
        input_filter: Optional[callable] = None,
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
                return await handoff.target_agent.process(message, session_id)

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
            ],
            "billing": ["payment", "invoice", "charge", "refund", "price", "cost"],
            "scheduling": [
                "schedule",
                "appointment",
                "book",
                "meeting",
                "when",
                "time",
            ],
        }

        if handoff.name in keywords:
            matches = [kw for kw in keywords[handoff.name] if kw in message.lower()]
            if matches:
                print(f"Handoff match for {handoff.name}: {matches}")
                return True
        return False


def remove_all_tools(input_data: HandoffInputData) -> HandoffInputData:
    """Remove all tools from the conversation history"""
    return HandoffInputData(
        conversation_history=[
            msg
            for msg in input_data.conversation_history
            if not (hasattr(msg, "tool_calls") and msg.tool_calls)
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
            # Set up the model - using a simpler model that's more likely to be available
            self.model = await MultiProviderModel.create(
                primary_model="deepseek-r1-distill-llama-70b",
                fallback_models=[],
                required_capabilities={ModelCapability.CHAT},
                optimize_for=OptimizationStrategy.PERFORMANCE,
            )
            logger.info(
                f"Created multi-provider model with primary: {self.model.current_provider}"
            )
            print(f"Model setup complete with provider: {self.model.current_provider}")

            print("Creating specialized agents...")
            # Create specialized agents
            self.agents["technical"] = HandoffAgent(
                name="Technical",
                system_prompt=(
                    "You are a technical support agent who helps users with computer and software issues. "
                    "Provide clear, step-by-step solutions. Ask clarifying questions when needed."
                ),
            )

            self.agents["billing"] = HandoffAgent(
                name="Billing",
                system_prompt=(
                    "You are a billing support agent who helps with payment and invoice issues. "
                    "Be clear about costs and policies. Maintain confidentiality."
                ),
            )

            self.agents["scheduling"] = HandoffAgent(
                name="Scheduling",
                system_prompt=(
                    "You are a scheduling assistant who helps users book appointments and meetings. "
                    "Be efficient and clear about availability and timing."
                ),
            )

            # Initialize specialized agents
            print("Initializing specialized agents...")
            for agent_name, agent in self.agents.items():
                print(f"Initializing {agent_name} agent...")
                await agent.initialize(self.model)

            # Create general agent with handoff capabilities
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
                        description="the user has a technical issue or question about computer systems",
                        target_agent=self.agents["technical"],
                        input_filter=remove_all_tools,
                    ),
                    HandoffConfig(
                        name="billing",
                        description="the user has a billing inquiry or payment issue",
                        target_agent=self.agents["billing"],
                        input_filter=remove_all_tools,
                    ),
                    HandoffConfig(
                        name="scheduling",
                        description="the user needs help with scheduling or calendar management",
                        target_agent=self.agents["scheduling"],
                        input_filter=remove_all_tools,
                    ),
                ],
            )

            print("Initializing general agent...")
            await self.agents["general"].initialize(self.model)
            print("All agents initialized successfully")

        except Exception as e:
            print(f"\nError in handoff-enabled multi-agent system: {str(e)}")
            logger.exception("Detailed error information:")
            raise

    async def process_query(self, query: str) -> str:
        """Process a query through the general agent with potential handoffs"""
        print(f"\nProcessing query: {query}")
        logger.info("\nProcessing query through handoff-enabled multi-agent system...")
        try:
            response = await self.agents["general"].process(
                query, session_id="handoff_session"
            )
            print(f"Final response: {response[:50]}...")
            return response
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return f"Error processing your request: {str(e)}"


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
