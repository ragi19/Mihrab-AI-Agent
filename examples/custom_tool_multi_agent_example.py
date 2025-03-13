"""
Custom Tool Multi-Agent Example

Shows how to create and use a custom Wikipedia search tool in a multi-agent system.
"""
import asyncio
import json
import aiohttp
from llm_agents import create_agent
from llm_agents.core.message import Message, MessageRole
from llm_agents.tools import BaseTool
from llm_agents.runtime.coordinator import AgentCoordinator

class WikipediaSearchTool(BaseTool):
    """Custom tool for searching Wikipedia articles."""
    
    async def _execute(self, parameters):
        async with aiohttp.ClientSession() as session:
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": parameters["query"],
                "srlimit": 3
            }
            async with session.get(
                "https://en.wikipedia.org/w/api.php",
                params=params
            ) as response:
                data = await response.json()
                return {
                    "results": [
                        {
                            "title": result["title"],
                            "snippet": result["snippet"]
                        }
                        for result in data["query"]["search"]
                    ]
                }
    
    def _get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for Wikipedia"
                }
            },
            "required": ["query"]
        }

async def main():
    # Create wiki search tool
    wiki_tool = WikipediaSearchTool()
    
    # Create research agent with wiki tool
    researcher = await create_agent(
        "openai",
        "gpt-4",
        system_message="You are a research assistant that uses Wikipedia to find information.",
        tools=[wiki_tool]
    )
    
    # Create summarizer agent
    summarizer = await create_agent(
        "openai",
        "gpt-3.5-turbo",
        system_message="You create concise summaries from research findings."
    )
    
    # Create coordinator
    coordinator = AgentCoordinator([researcher, summarizer])
    
    # Example query
    query = "What is the history of the Python programming language?"
    
    # Process query
    result = await coordinator.process_query(
        Message(role=MessageRole.USER, content=query)
    )
    
    print(f"Final Response:\n{result.content}")

if __name__ == "__main__":
    asyncio.run(main())