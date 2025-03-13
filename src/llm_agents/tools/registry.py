"""
Registry for managing and accessing tool implementations
"""

from typing import Any, Dict, List, Optional, Type

from ..utils.logging import get_logger
from .base import BaseTool

logger = get_logger("tools.registry")


class ToolRegistry:
    _tools: Dict[str, Type[BaseTool]] = {}

    @classmethod
    def register(cls, name: str, tool_class: Type[BaseTool]) -> None:
        """Register a new tool implementation

        Args:
            name: Name of the tool
            tool_class: Tool class implementation
        """
        cls._tools[name] = tool_class
        logger.info(f"Registered tool: {name}")

    @classmethod
    def get_tool(cls, name: str) -> Optional[Type[BaseTool]]:
        """Get a tool implementation by name

        Args:
            name: Name of the tool to get

        Returns:
            Tool class if found, None otherwise
        """
        tool = cls._tools.get(name)
        if tool:
            logger.debug(f"Retrieved tool: {name}")
        else:
            logger.warning(f"Tool not found: {name}")
        return tool

    @classmethod
    def list_tools(cls) -> list[str]:
        """List all registered tool names

        Returns:
            List of registered tool names
        """
        tools = list(cls._tools.keys())
        logger.debug(f"Available tools: {tools}")
        return tools

    @classmethod
    def create_tool(cls, name: str, **kwargs: Any) -> BaseTool:
        """Create a new tool instance

        Args:
            name: Name of the tool to create
            **kwargs: Additional arguments for tool initialization

        Returns:
            Initialized tool instance

        Raises:
            KeyError: If tool is not found
            ValueError: If tool creation fails
        """
        logger.info(f"Creating tool: {name}")
        logger.debug(f"Tool kwargs: {kwargs}")

        tool_class = cls.get_tool(name)
        if not tool_class:
            logger.error(f"Tool {name} not found")
            raise KeyError(f"Tool {name} not found")

        try:
            tool = tool_class(**kwargs)
            logger.info(f"Successfully created tool: {name}")
            return tool
        except Exception as e:
            logger.error(f"Failed to create tool: {e}", exc_info=True)
            raise ValueError(f"Failed to create tool: {e}")

    @classmethod
    def get_schemas(cls) -> List[Dict]:
        """Get JSON schemas for all registered tools"""
        schemas = []
        for name, tool_class in cls._tools.items():
            tool = tool_class(name=name, description=tool_class.__doc__ or "")
            schemas.append(tool.get_schema())
        return schemas
