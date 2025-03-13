#!/usr/bin/env python
"""
Advanced example demonstrating how to create custom tools and integrate them into the multi-agent system.
This example shows how to build a custom Wikipedia search tool and use it with the multi-agent system.
"""
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from llm_agents.core.memory import Memory
from llm_agents.core.memory_task_agent import MemoryEnabledTaskAgent
from llm_agents.core.message import Message, MessageRole
from llm_agents.core.task_agent import ToolConfig
from llm_agents.models.base import ModelCapability

# Import required modules
from llm_agents.models.multi_provider import MultiProviderModel, OptimizationStrategy
from llm_agents.runtime.memory_runner import MemoryAgentRunner
from llm_agents.tools.base import BaseTool
from llm_agents.tools.standard import CalculatorTool


class WikipediaSearchTool(BaseTool):
    """Custom tool for searching Wikipedia"""

    def __init__(self):
        super().__init__(
            name="wikipedia_search",
            description="Search Wikipedia for information on a given topic",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for Wikipedia",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 1)",
                    },
                },
                "required": ["query"],
            },
        )

    async def _execute(self, params: Dict[str, Any]) -> str:
        """Execute the Wikipedia search"""
        query = params.get("query")
        limit = params.get("limit", 1)

        try:
            # Use Wikipedia API to search for the query
            async with aiohttp.ClientSession() as session:
                # First, search for the page
                search_url = "https://en.wikipedia.org/w/api.php"
                search_params = {
                    "action": "query",
                    "format": "json",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": limit,
                }

                async with session.get(search_url, params=search_params) as response:
                    if response.status != 200:
                        return f"Error: Failed to search Wikipedia. Status code: {response.status}"

                    search_data = await response.json()
                    search_results = search_data.get("query", {}).get("search", [])

                    if not search_results:
                        return f"No Wikipedia articles found for '{query}'."

                    # Get the first result's page ID
                    page_id = search_results[0]["pageid"]

                    # Now get the content for this page
                    content_url = "https://en.wikipedia.org/w/api.php"
                    content_params = {
                        "action": "query",
                        "format": "json",
                        "prop": "extracts",
                        "pageids": page_id,
                        "exintro": 1,  # Only get the introduction
                        "explaintext": 1,  # Get plain text
                    }

                    async with session.get(
                        content_url, params=content_params
                    ) as content_response:
                        if content_response.status != 200:
                            return f"Error: Failed to get Wikipedia content. Status code: {content_response.status}"

                        content_data = await content_response.json()
                        page_content = (
                            content_data.get("query", {})
                            .get("pages", {})
                            .get(str(page_id), {})
                            .get("extract", "")
                        )

                        if not page_content:
                            return f"Found Wikipedia article for '{query}' but couldn't retrieve content."

                        title = search_results[0]["title"]
                        return f"Wikipedia article: {title}\n\n{page_content}"

        except Exception as e:
            logger.error(f"Error in Wikipedia search: {e}")
            return f"Error searching Wikipedia: {str(e)}"


class CustomToolAgent:
    """Agent wrapper with custom tool capabilities for the multi-agent system"""

    def __init__(
        self, name: str, system_prompt: str, tools: Optional[List[ToolConfig]] = None
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.agent = None
        self.runner = None

    async def initialize(self, model: MultiProviderModel):
        """Initialize the agent with a model and tools"""
        # Create memory
        memory = Memory(working_memory_size=10, long_term_memory_size=50)

        # Create agent with tools
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

        logger.info(f"Initialized agent: {self.name} with {len(self.tools)} tools")

    async def process(self, message: str, session_id: str = None) -> str:
        """Process a message and return the response"""
        response = await self.runner.run(message, session_id=session_id)
        return response.content


class AdvancedMultiAgentSystem:
    """Advanced multi-agent system with custom tools"""

    def __init__(self):
        self.agents = {}
        self.model = None

    async def initialize(self):
        """Initialize the multi-agent system with custom tools"""
        # Check for API key
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Please set the GROQ_API_KEY environment variable")

        # Set up the model
        self.model = await MultiProviderModel.create(
            primary_model="llama3-70b-8192",  # Groq's Llama 3 model
            fallback_models=[],
            required_capabilities={
                ModelCapability.CHAT,
                ModelCapability.FUNCTION_CALLING,
            },
            optimize_for=OptimizationStrategy.PERFORMANCE,
        )
        logger.info(
            f"Created multi-provider model with primary: {self.model.current_provider}"
        )

        # Initialize tools
        calculator_tool = CalculatorTool()
        wikipedia_tool = WikipediaSearchTool()

        # Create analyzer agent (with calculator tool)
        self.agents["analyzer"] = CustomToolAgent(
            name="Analyzer",
            system_prompt=(
                "You are an analytical agent with calculation capabilities. "
                "Break down complex questions into key components that need investigation. "
                "Use the calculator tool when numerical analysis is needed. "
                "Be concise and precise."
            ),
            tools=[calculator_tool],
        )

        # Create researcher agent (with Wikipedia tool)
        self.agents["researcher"] = CustomToolAgent(
            name="Researcher",
            system_prompt=(
                "You are a research agent with Wikipedia search capabilities. "
                "Focus on providing key facts and information about the specified components. "
                "Use the Wikipedia search tool to find relevant information. "
                "Be concise and informative."
            ),
            tools=[wikipedia_tool],
        )

        # Create synthesizer agent (with both tools)
        self.agents["synthesizer"] = CustomToolAgent(
            name="Synthesizer",
            system_prompt=(
                "You are a synthesis agent with calculation and Wikipedia search capabilities. "
                "Combine the provided analysis and research into a clear, coherent response. "
                "Use tools when needed to verify or enhance your response. "
                "Be concise yet comprehensive."
            ),
            tools=[calculator_tool, wikipedia_tool],
        )

        # Initialize all agents
        for agent in self.agents.values():
            await agent.initialize(self.model)

    async def process_query(self, query: str) -> Dict[str, str]:
        """Process a query through all agents in sequence"""
        logger.info("\nProcessing query through advanced multi-agent system...")

        # Step 1: Analyzer breaks down the question (can use calculator)
        analysis = await self.agents["analyzer"].process(
            f"Break down this question into key components. Use calculator if needed: {query}",
            session_id="analyzer_session",
        )
        logger.info("\nAnalysis complete.")

        # Step 2: Researcher investigates each component (can use Wikipedia)
        research = await self.agents["researcher"].process(
            f"Research these components. Use Wikipedia search tool to find relevant information: {analysis}",
            session_id="researcher_session",
        )
        logger.info("\nResearch complete.")

        # Step 3: Synthesizer combines everything (can use both tools)
        synthesis = await self.agents["synthesizer"].process(
            f"Create a concise response using:\nAnalysis: {analysis}\nResearch: {research}\nQuestion: {query}\n"
            f"Use calculator or Wikipedia search tools if needed to verify or enhance your response.",
            session_id="synthesizer_session",
        )
        logger.info("\nSynthesis complete.")

        return {"analysis": analysis, "research": research, "synthesis": synthesis}


async def main():
    """Main function to run the advanced multi-agent system"""
    try:
        # Initialize advanced multi-agent system
        system = AdvancedMultiAgentSystem()
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
            print("ANALYZER'S BREAKDOWN (with calculator):")
            print("-" * 80)
            print(results["analysis"])

            print("\n" + "=" * 80)
            print("RESEARCHER'S FINDINGS (with Wikipedia):")
            print("-" * 80)
            print(results["research"])

            print("\n" + "=" * 80)
            print("SYNTHESIZER'S FINAL RESPONSE (with all tools):")
            print("-" * 80)
            print(results["synthesis"])
            print("=" * 80)

    except Exception as e:
        logger.error(f"Error in advanced multi-agent system: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
