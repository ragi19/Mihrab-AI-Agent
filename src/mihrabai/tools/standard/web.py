"""
Standard web tools for HTTP requests and web interactions
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

import aiohttp

from ...core.types import JSON
from ...utils.logging import get_logger
from ..base import BaseTool

logger = get_logger("tools.web")


class HTTPRequestTool(BaseTool):
    """Tool for making HTTP requests"""

    def __init__(self):
        super().__init__(
            name="http_request", description="Make HTTP requests to web endpoints"
        )
        self._parameters = {
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                "default": "GET",
            },
            "url": {"type": "string", "description": "The URL to make the request to"},
            "headers": {"type": "object", "description": "Request headers"},
            "data": {"type": "object", "description": "Request body data"},
            "params": {"type": "object", "description": "URL query parameters"},
            "timeout": {
                "type": "number",
                "description": "Request timeout in seconds",
                "default": 30,
            },
        }
        self._required_params = ["url"]
        logger.info("Initialized tool: http_request")

    async def execute(self, parameters: Dict[str, Any]) -> JSON:
        """Execute the tool with given parameters"""
        logger.debug(f"Executing tool {self.name} with parameters: {parameters}")

        try:
            # Validate parameters
            self._validate_parameters(parameters)

            # Execute tool-specific logic
            result = await self._execute(**parameters)
            logger.debug(f"Tool execution result: {result}")

            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            raise

    async def _execute(
        self,
        url: str,
        method: str = "GET",
        headers: Dict[str, str] = None,
        data: Any = None,
        params: Dict[str, str] = None,
        timeout: float = 30,
        **kwargs,
    ) -> JSON:
        """Execute an HTTP request"""
        if headers is None:
            headers = {}

        # Add a user agent if not provided
        if "User-Agent" not in headers:
            headers["User-Agent"] = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method,
                    url,
                    headers=headers,
                    json=data if method != "GET" else None,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    # Try to parse as JSON first
                    try:
                        body = await response.json()
                    except:
                        body = await response.text()

                    return {
                        "status": response.status,
                        "success": 200 <= response.status < 300,
                        "headers": dict(response.headers),
                        "body": body,
                        "url": str(response.url),
                    }
        except aiohttp.ClientError as e:
            return {
                "error": f"HTTP request failed: {str(e)}",
                "status": 0,
                "success": False,
            }

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": self._parameters,
            "required": self._required_params,
        }


class WebScraperTool(BaseTool):
    """Tool for scraping content from web pages"""

    def __init__(self):
        super().__init__(
            name="web_scraper", description="Extract content from web pages"
        )
        self._parameters = {
            "url": {"type": "string", "description": "URL of the web page to scrape"},
            "selectors": {
                "type": "array",
                "description": "CSS selectors to extract specific elements",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "selector": {"type": "string"},
                        "attribute": {"type": "string", "default": "text"},
                        "multiple": {"type": "boolean", "default": False},
                    },
                    "required": ["name", "selector"],
                },
            },
            "extract_text": {
                "type": "boolean",
                "description": "Extract main text content from the page",
                "default": True,
            },
            "extract_links": {
                "type": "boolean",
                "description": "Extract links from the page",
                "default": False,
            },
            "extract_images": {
                "type": "boolean",
                "description": "Extract image URLs from the page",
                "default": False,
            },
            "extract_metadata": {
                "type": "boolean",
                "description": "Extract metadata (title, description, etc.)",
                "default": True,
            },
            "headers": {
                "type": "object",
                "description": "Custom HTTP headers for the request",
            },
        }
        self._required_params = ["url"]
        logger.info("Initialized tool: web_scraper")

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": self._parameters,
            "required": self._required_params,
        }

    async def execute(self, parameters: Dict[str, Any]) -> JSON:
        """Execute the tool with given parameters"""
        logger.debug(f"Executing tool {self.name} with parameters: {parameters}")

        try:
            # Validate parameters
            self._validate_parameters(parameters)

            # Execute tool-specific logic
            result = await self._execute(**parameters)
            logger.debug(f"Tool execution result: {result}")

            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            raise

    async def _execute(
        self,
        url: str,
        selectors: List[Dict[str, Any]] = None,
        extract_text: bool = True,
        extract_links: bool = False,
        extract_images: bool = False,
        extract_metadata: bool = True,
        headers: Dict[str, str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute the web scraper tool

        Args:
            url: URL of the web page to scrape
            selectors: CSS selectors to extract specific elements
            extract_text: Extract main text content from the page
            extract_links: Extract links from the page
            extract_images: Extract image URLs from the page
            extract_metadata: Extract metadata (title, description, etc.)
            headers: Custom HTTP headers for the request

        Returns:
            Dictionary with scraped content
        """
        try:
            # Import BeautifulSoup
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                return {
                    "error": "BeautifulSoup is required for web scraping. Install it with 'pip install beautifulsoup4'"
                }

            # Make HTTP request to get the page content
            if headers is None:
                headers = {}

            # Add a user agent if not provided
            if "User-Agent" not in headers:
                headers["User-Agent"] = (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                )

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        return {
                            "error": f"Failed to fetch URL: HTTP {response.status}",
                            "url": url,
                        }

                    html_content = await response.text()

            # Parse HTML
            soup = BeautifulSoup(html_content, "html.parser")
            result = {"url": url}

            # Extract metadata
            if extract_metadata:
                metadata = {}

                # Title
                if soup.title:
                    metadata["title"] = soup.title.string

                # Meta tags
                for meta in soup.find_all("meta"):
                    name = meta.get("name") or meta.get("property")
                    if name and meta.get("content"):
                        metadata[name] = meta.get("content")

                result["metadata"] = metadata

            # Extract main text content
            if extract_text:
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()

                # Get text
                text = soup.get_text(separator=" ", strip=True)

                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (
                    phrase.strip() for line in lines for phrase in line.split("  ")
                )
                text = " ".join(chunk for chunk in chunks if chunk)

                result["text"] = text

            # Extract links
            if extract_links:
                links = []
                for link in soup.find_all("a"):
                    href = link.get("href")
                    if href:
                        # Convert relative URLs to absolute
                        if not href.startswith(("http://", "https://")):
                            href = urljoin(url, href)

                        links.append(
                            {
                                "url": href,
                                "text": link.get_text(strip=True),
                                "title": link.get("title"),
                            }
                        )

                result["links"] = links

            # Extract images
            if extract_images:
                images = []
                for img in soup.find_all("img"):
                    src = img.get("src")
                    if src:
                        # Convert relative URLs to absolute
                        if not src.startswith(("http://", "https://")):
                            src = urljoin(url, src)

                        images.append(
                            {
                                "url": src,
                                "alt": img.get("alt"),
                                "title": img.get("title"),
                            }
                        )

                result["images"] = images

            # Extract content using selectors
            if selectors:
                extracted = {}

                for selector_info in selectors:
                    name = selector_info.get("name")
                    selector = selector_info.get("selector")
                    attribute = selector_info.get("attribute", "text")
                    multiple = selector_info.get("multiple", False)

                    if not name or not selector:
                        continue

                    elements = soup.select(selector)

                    if multiple:
                        values = []
                        for element in elements:
                            if attribute == "text":
                                values.append(element.get_text(strip=True))
                            elif attribute == "html":
                                values.append(str(element))
                            else:
                                values.append(element.get(attribute))
                        extracted[name] = values
                    else:
                        if elements:
                            element = elements[0]
                            if attribute == "text":
                                extracted[name] = element.get_text(strip=True)
                            elif attribute == "html":
                                extracted[name] = str(element)
                            else:
                                extracted[name] = element.get(attribute)

                result["extracted"] = extracted

            return result
        except Exception as e:
            logger.error(f"Web scraping error: {e}")
            return {"error": f"Web scraping error: {str(e)}", "url": url}
