"""
Base tool class and interfaces
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..core.types import JSON
from ..utils.logging import get_logger


class BaseTool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = get_logger(f"tools.{name}")
        self.logger.info(f"Initialized tool: {name}")

    @abstractmethod
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

    @abstractmethod
    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        """Internal method for tool-specific execution logic"""
        raise NotImplementedError("Subclasses must implement _execute")

    def _validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate tool parameters against schema"""
        schema = self._get_parameters_schema()
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check required parameters
        for param in required:
            if param not in parameters:
                self.logger.error(f"Missing required parameter: {param}")
                raise ValueError(f"Missing required parameter: {param}")

        # Check parameter types
        for param, value in parameters.items():
            if param not in properties:
                self.logger.warning(f"Unknown parameter: {param}")
                continue

            param_type = properties[param]["type"]
            if not self._check_type(value, param_type):
                self.logger.error(
                    f"Invalid type for parameter {param}. "
                    f"Expected {param_type}, got {type(value)}"
                )
                raise ValueError(
                    f"Invalid type for parameter {param}. "
                    f"Expected {param_type}, got {type(value)}"
                )

    @abstractmethod
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters_schema(),
        }

    @staticmethod
    def _check_type(value: Any, expected_type: str) -> bool:
        """Check if a value matches the expected JSON schema type"""
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "number":
            return isinstance(value, (int, float))
        elif expected_type == "integer":
            return isinstance(value, int)
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        return False
