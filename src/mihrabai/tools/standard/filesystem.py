"""
File system tools for reading and writing files
"""

import json
import os
from typing import Any, Dict

from ...core.types import JSON
from ..base import BaseTool


class FileReaderTool(BaseTool):
    """Tool for reading file contents"""

    def __init__(self):
        super().__init__(name="file_reader", description="Read contents from a file")

    async def execute(self, parameters: Dict[str, Any]) -> JSON:
        """Execute the tool with given parameters"""
        self.logger.debug(f"Executing tool {self.name} with parameters: {parameters}")

        try:
            # Validate parameters
            self._validate_parameters(parameters)

            # Execute tool-specific logic
            result = await self._execute(parameters)
            self.logger.debug(f"Tool execution result: {result}")

            return result
        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}", exc_info=True)
            raise

    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        """Read file contents"""
        path = parameters["path"]
        encoding = parameters.get("encoding", "utf-8")

        try:
            with open(path, "r", encoding=encoding) as f:
                content = f.read()
            return {"success": True, "content": content}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
                "encoding": {
                    "type": "string",
                    "description": "File encoding",
                    "default": "utf-8",
                },
            },
            "required": ["path"],
        }


class FileWriterTool(BaseTool):
    """Tool for writing file contents"""

    def __init__(self):
        super().__init__(name="file_writer", description="Write contents to a file")

    async def execute(self, parameters: Dict[str, Any]) -> JSON:
        """Execute the tool with given parameters"""
        self.logger.debug(f"Executing tool {self.name} with parameters: {parameters}")

        try:
            # Validate parameters
            self._validate_parameters(parameters)

            # Execute tool-specific logic
            result = await self._execute(parameters)
            self.logger.debug(f"Tool execution result: {result}")

            return result
        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}", exc_info=True)
            raise

    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        """Write file contents"""
        path = parameters["path"]
        content = parameters["content"]
        mode = parameters.get("mode", "w")
        encoding = parameters.get("encoding", "utf-8")

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, mode, encoding=encoding) as f:
                f.write(content)
            return {"success": True, "path": path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
                "mode": {
                    "type": "string",
                    "description": "File write mode (w for write, a for append)",
                    "enum": ["w", "a"],
                    "default": "w",
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding",
                    "default": "utf-8",
                },
            },
            "required": ["path", "content"],
        }
