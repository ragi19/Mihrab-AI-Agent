"""
Tools package exports and automatic registration
"""

from .base import BaseTool
from .registry import ToolRegistry
from .standard import (
    CSVParserTool,
    FileReaderTool,
    FileWriterTool,
    HTTPRequestTool,
    JSONParserTool,
    ShellCommandTool,
)

# Register standard tools
standard_tools = [
    HTTPRequestTool,
    JSONParserTool,
    CSVParserTool,
    FileReaderTool,
    FileWriterTool,
    ShellCommandTool,
]

for tool_class in standard_tools:
    tool = tool_class()
    ToolRegistry.register(tool.name, tool_class)

__all__ = [
    "BaseTool",
    "ToolRegistry",
    "HTTPRequestTool",
    "JSONParserTool",
    "CSVParserTool",
    "FileReaderTool",
    "FileWriterTool",
    "ShellCommandTool",
]
