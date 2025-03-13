#!/usr/bin/env python
"""
Example script demonstrating how to use the multi-agent system.
"""
import asyncio
import os
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import required modules
from llm_agents.models.multi_provider import MultiProviderModel, OptimizationStrategy
from llm_agents.models.base import ModelCapability
from llm_agents.core.memory import Memory
from llm_agents.core.memory_task_agent import MemoryEnabledTaskAgent
from llm_agents.runtime.memory_runner import MemoryAgentRunner
from llm_agents.core.message import Message, MessageRole


class Agent:
    """Simple agent wrapper for the multi-agent system"""

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.agent = None
        self.runner = None

    async def initialize(self, model: MultiProviderModel):
        """Initialize the agent with a model"""
        # Create memory
        memory = Memory(working_memory_size=10, long_term_memory_size=50)

        # Create agent
        self.agent = MemoryEnabledTaskAgent(
            model=model, memory=memory, system_message=self.system_prompt
        )

        # Create runner
        self.runner = MemoryAgentRunner(
            agent=self.agent, memory_persistence_path="./memories"
        )

        logger.info(f"Initialized agent: {self.name}")

    async def process(self, message: str, session_id: str = None) -> str:
        """Process a message and return the response"""
        response = await self.runner.run(message, session_id=session_id)
        return response.content


class MultiAgentSystem:
    """Simple multi-agent system"""

    def __init__(self):
        self.agents = {}
        self.model = None

    async def initialize(self):
        """Initialize the multi-agent system"""
        # Check for API key
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Please set the GROQ_API_KEY environment variable")

        # Set up the model
        self.model = await MultiProviderModel.create(
            primary_model="llama3-70b-8192",  # Groq's Llama 3 model
            fallback_models=[],
            required_capabilities={ModelCapability.CHAT},
            optimize_for=OptimizationStrategy.PERFORMANCE,
        )
        logger.info(
            f"Created multi-provider model with primary: {self.model.current_provider}"
        )

        # Create analyzer agent
        self.agents["analyzer"] = Agent(
            name="Analyzer",
            system_prompt=(
                "You are an analytical agent. Break down complex questions into key components "
                "that need investigation. Be concise and precise."
            ),
        )

        # Create researcher agent
        self.agents["researcher"] = Agent(
            name="Researcher",
            system_prompt=(
                "You are a research agent. Focus on providing key facts and information about "
                "the specified components. Be concise and informative."
            ),
        )

        # Create synthesizer agent
        self.agents["synthesizer"] = Agent(
            name="Synthesizer",
            system_prompt=(
                "You are a synthesis agent. Combine the provided analysis and research into a clear, "
                "coherent response. Be concise yet comprehensive."
            ),
        )

        # Initialize all agents
        for agent in self.agents.values():
            await agent.initialize(self.model)

    async def process_query(self, query: str) -> Dict[str, str]:
        """Process a query through all agents in sequence"""
        logger.info("\nProcessing query through multi-agent system...")

        # Step 1: Analyzer breaks down the question
        analysis = await self.agents["analyzer"].process(
            f"Break down this question into key components: {query}",
            session_id="analyzer_session",
        )
        logger.info("\nAnalysis complete.")

        # Step 2: Researcher investigates each component
        research = await self.agents["researcher"].process(
            f"Research these components: {analysis}", session_id="researcher_session"
        )
        logger.info("\nResearch complete.")

        # Step 3: Synthesizer combines everything
        synthesis = await self.agents["synthesizer"].process(
            f"Create a concise response using:\nAnalysis: {analysis}\nResearch: {research}\nQuestion: {query}",
            session_id="synthesizer_session",
        )
        logger.info("\nSynthesis complete.")

        return {"analysis": analysis, "research": research, "synthesis": synthesis}


async def main():
    """Main function to run the multi-agent system"""
    try:
        # Initialize multi-agent system
        system = MultiAgentSystem()
        await system.initialize()

        # Process user queries
        while True:
            # Get user query
            query = input("\nEnter your question (or 'exit' to quit): ")

            # Check if user wants to exit
            if query.lower() == "exit":
                break

            # Process the query
            results = await system.process_query(query)

            # Display results
            print("\n" + "=" * 80)
            print("ANALYZER'S BREAKDOWN:")
            print("-" * 80)
            print(results["analysis"])

            print("\n" + "=" * 80)
            print("RESEARCHER'S FINDINGS:")
            print("-" * 80)
            print(results["research"])

            print("\n" + "=" * 80)
            print("SYNTHESIZER'S FINAL RESPONSE:")
            print("-" * 80)
            print(results["synthesis"])
            print("=" * 80)

    except Exception as e:
        logger.error(f"Error in multi-agent system: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
