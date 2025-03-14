"""
Advanced sample usage of Groq provider demonstrating multi-agent system
"""

import asyncio
import json
import os
from typing import Dict, List

from llm_agents.core.message import Message, MessageRole
from llm_agents.core.types import ModelParameters
from llm_agents.models.providers.groq import GroqProvider
from llm_agents.utils.async_utils import gather_with_concurrency


class GroqAgent:
    def __init__(
        self, api_key: str, role: str, system_prompt: str, max_history: int = 3
    ):
        self.provider = GroqProvider(api_key=api_key)
        self.role = role
        self.conversation_history: List[Message] = []
        self.system_prompt = system_prompt
        self.max_history = max_history

    async def initialize_model(self):
        """Initialize the model with specific parameters"""
        parameters = {
            "temperature": 0.7,
            "max_tokens": 800,  # Reduced from 1000
            "top_p": 0.95,
            "stream": False,
        }

        self.model = await self.provider.create_model(
            "llama-3.3-70b-versatile", parameters=parameters
        )

        # Add system prompt to conversation history
        system_message = Message(role=MessageRole.SYSTEM, content=self.system_prompt)
        self.conversation_history.append(system_message)

    def _truncate_history(self):
        """Keep only the most recent messages within the max_history limit"""
        if (
            len(self.conversation_history) > self.max_history + 1
        ):  # +1 for system message
            # Always keep system message
            self.conversation_history = [
                self.conversation_history[0],  # System message
                *self.conversation_history[-(self.max_history) :],  # Recent messages
            ]

    def clear_history(self):
        """Clear conversation history except system message"""
        self.conversation_history = [
            self.conversation_history[0]
        ]  # Keep only system message

    async def process_message(self, message: str) -> str:
        """Process a message and return a response"""
        # Clear history except system message to prevent token accumulation
        self.clear_history()

        user_message = Message(role=MessageRole.USER, content=message)
        self.conversation_history.append(user_message)

        response = await self.model.generate_response(self.conversation_history)
        self.conversation_history.append(response)

        return response.content


class MultiAgentSystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.agents: Dict[str, GroqAgent] = {}

    async def initialize_agents(self):
        """Initialize all agents in the system"""
        # Create analyzer agent with focused prompt
        self.agents["analyzer"] = GroqAgent(
            self.api_key,
            "analyzer",
            """You are an analytical agent. Break down complex questions into 2-3 key components 
            that need investigation. Be concise and precise.""",
        )

        # Create researcher agent with focused prompt
        self.agents["researcher"] = GroqAgent(
            self.api_key,
            "researcher",
            """You are a research agent. Focus on providing key facts and information about the specified components. 
            Be concise and informative.""",
        )

        # Create synthesizer agent with focused prompt
        self.agents["synthesizer"] = GroqAgent(
            self.api_key,
            "synthesizer",
            """You are a synthesis agent. Combine the provided analysis and research into a clear, 
            coherent response. Be concise yet comprehensive.""",
        )

        # Initialize all agents
        for agent in self.agents.values():
            await agent.initialize_model()

    async def process_query(self, query: str) -> Dict[str, str]:
        """Process a query through all agents in sequence"""
        print("\nProcessing query through multi-agent system...")

        # Step 1: Analyzer breaks down the question
        analysis = await self.agents["analyzer"].process_message(
            f"Break down this question into key components: {query}"
        )
        print("\nAnalysis complete.")

        # Step 2: Researcher investigates each component
        research = await self.agents["researcher"].process_message(
            f"Research these components: {analysis}"
        )
        print("\nResearch complete.")

        # Step 3: Synthesizer combines everything
        synthesis = await self.agents["synthesizer"].process_message(
            f"Create a concise response using:\nAnalysis: {analysis}\nResearch: {research}\nQuestion: {query}"
        )
        print("\nSynthesis complete.")

        return {"analysis": analysis, "research": research, "synthesis": synthesis}


async def main():
    # Get API key from environment variable
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Please set GROQ_API_KEY environment variable")
        return

    try:
        # Initialize multi-agent system
        system = MultiAgentSystem(api_key)
        await system.initialize_agents()

        # Example queries to demonstrate multi-agent cooperation
        queries = [
            "What are the potential implications of quantum computing on cybersecurity?",
            "How might artificial intelligence impact the future of healthcare delivery?",
        ]

        for query in queries:
            print(f"\n\n=== Processing Query: {query} ===")
            results = await system.process_query(query)

            print("\nAnalyzer's Breakdown:")
            print(results["analysis"])

            print("\nResearcher's Findings:")
            print(results["research"])

            print("\nSynthesizer's Final Response:")
            print(results["synthesis"])

            print("\n" + "=" * 80)

    except Exception as e:
        print(f"Error in demo: {e}")


if __name__ == "__main__":
    asyncio.run(main())
