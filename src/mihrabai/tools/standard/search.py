"""
Web search tools
"""

import json
from typing import Any, Dict, List, Optional

import aiohttp

from ...utils.logging import get_logger
from ..base import BaseTool

logger = get_logger("tools.search")


class WebSearchTool(BaseTool):
    """Tool for performing web searches using Serper API"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="web_search", description="Search the web for information"
        )
        self._parameters = {
            "query": {"type": "string", "description": "The search query"},
            "num_results": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 5,
            },
        }
        self._required_params = ["query"]
        self.api_key = api_key
        logger.info("Initialized tool: web_search")

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": self._parameters,
            "required": self._required_params,
        }

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        self.logger.debug(f"Executing web search tool with parameters: {parameters}")

        try:
            # Validate parameters
            self._validate_parameters(parameters)

            # Execute tool-specific logic
            result = await self._execute(**parameters)
            self.logger.debug(f"Web search execution result: {result}")

            return result
        except Exception as e:
            self.logger.error(f"Web search execution failed: {e}", exc_info=True)
            return {"error": str(e)}

    async def _execute(
        self, query: str, num_results: int = 5, **kwargs
    ) -> Dict[str, Any]:
        """Execute the web search tool

        Args:
            query: The search query
            num_results: Number of results to return

        Returns:
            Dictionary with search results

        Raises:
            ValueError: If API key is missing or API request fails
        """
        if not self.api_key:
            return {
                "error": "API key not provided. Please set SERPER_API_KEY environment variable."
            }

        try:
            # Limit number of results
            num_results = min(max(1, num_results), 10)

            # Make API request to Serper
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://google.serper.dev/search",
                    headers={
                        "X-API-KEY": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json={"q": query, "num": num_results},
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Search API error: {error_text}")
                        return {
                            "error": f"Search API error: {response.status}",
                            "query": query,
                        }

                    data = await response.json()

            # Process and format results
            results = self._process_results(data, num_results)

            return {"query": query, "results": results, "num_results": len(results)}

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {"error": f"Web search error: {str(e)}", "query": query}

    def _process_results(
        self, data: Dict[str, Any], limit: int
    ) -> List[Dict[str, Any]]:
        """Process and format search results

        Args:
            data: Raw API response data
            limit: Maximum number of results to include

        Returns:
            List of formatted search results
        """
        results = []

        # Process organic search results
        if "organic" in data:
            for item in data["organic"][:limit]:
                result = {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
                results.append(result)

        # Add knowledge graph if available
        if "knowledgeGraph" in data and len(results) < limit:
            kg = data["knowledgeGraph"]
            if "title" in kg:
                result = {
                    "title": kg.get("title", ""),
                    "description": kg.get("description", ""),
                    "type": "knowledge_graph",
                }
                results.append(result)

        return results[:limit]
