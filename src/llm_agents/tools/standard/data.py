"""
Standard data tools for data processing and manipulation
"""

import csv
import json
from typing import Any, Dict, List

from ...core.types import JSON
from ..base import BaseTool


class JSONParserTool(BaseTool):
    """Tool for parsing and validating JSON data"""

    def __init__(self):
        super().__init__(name="json_parser", description="Parse and validate JSON data")

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
        """Parse JSON data"""
        data = parameters["data"]
        try:
            if isinstance(data, str):
                parsed = json.loads(data)
            else:
                parsed = data
            return {"success": True, "result": parsed}
        except json.JSONDecodeError as e:
            return {"success": False, "error": str(e)}

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": ["string", "object"],
                    "description": "JSON data to parse",
                }
            },
            "required": ["data"],
        }


class CSVParserTool(BaseTool):
    """Tool for parsing CSV data"""

    def __init__(self):
        super().__init__(
            name="csv_parser", description="Parse CSV data into structured format"
        )

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
        """Parse CSV data"""
        data = parameters["data"]
        delimiter = parameters.get("delimiter", ",")

        try:
            lines = data.split("\n")
            reader = csv.DictReader(lines, delimiter=delimiter)
            rows = list(reader)
            return {"success": True, "result": rows}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "CSV data to parse"},
                "delimiter": {
                    "type": "string",
                    "description": "CSV delimiter character",
                    "default": ",",
                },
            },
            "required": ["data"],
        }
