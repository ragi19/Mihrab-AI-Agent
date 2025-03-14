"""
Utility tools for common operations
"""

import datetime
import math
import re
from typing import Any, Dict, Optional

from ...utils.logging import get_logger
from ..base import BaseTool

logger = get_logger("tools.utility")


class CalculatorTool(BaseTool):
    """Tool for performing mathematical calculations"""

    def __init__(self):
        super().__init__(
            name="calculator", description="Perform mathematical calculations"
        )
        self._parameters = {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate",
            }
        }
        self._required_params = ["expression"]
        logger.info("Initialized tool: calculator")

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": self._parameters,
            "required": self._required_params,
        }

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        self.logger.debug(f"Executing calculator tool with parameters: {parameters}")

        try:
            # Validate parameters
            self._validate_parameters(parameters)

            # Execute tool-specific logic
            result = await self._execute(**parameters)
            self.logger.debug(f"Calculator execution result: {result}")

            return result
        except Exception as e:
            self.logger.error(f"Calculator execution failed: {e}", exc_info=True)
            return {"error": str(e)}

    async def _execute(self, expression: str, **kwargs) -> Dict[str, Any]:
        """Execute the calculator tool

        Args:
            expression: The mathematical expression to evaluate

        Returns:
            Dictionary with result

        Raises:
            ValueError: If expression is invalid or unsafe
        """
        try:
            # Sanitize the expression
            sanitized = self._sanitize_expression(expression)

            # Evaluate the expression
            result = eval(
                sanitized,
                {"__builtins__": None},
                {
                    "abs": abs,
                    "round": round,
                    "max": max,
                    "min": min,
                    "sum": sum,
                    "pow": pow,
                    "math": math,
                },
            )

            return {"result": result, "expression": expression}
        except Exception as e:
            logger.error(f"Calculator error: {e}")
            return {"error": f"Calculation error: {str(e)}", "expression": expression}

    def _sanitize_expression(self, expression: str) -> str:
        """Sanitize a mathematical expression to prevent code execution

        Args:
            expression: The expression to sanitize

        Returns:
            Sanitized expression

        Raises:
            ValueError: If expression contains unsafe patterns
        """
        # Remove any whitespace
        expression = expression.strip()

        # Check for unsafe patterns
        unsafe_patterns = [
            r"import",
            r"exec",
            r"eval",
            r"compile",
            r"__",
            r"globals",
            r"locals",
            r"getattr",
            r"setattr",
            r"open",
            r"file",
            r"os\.",
            r"sys\.",
        ]

        for pattern in unsafe_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                raise ValueError(f"Expression contains unsafe pattern: {pattern}")

        # Replace common math functions with math.function
        math_funcs = [
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "atan2",
            "sinh",
            "cosh",
            "tanh",
            "exp",
            "log",
            "log10",
            "sqrt",
            "ceil",
            "floor",
            "degrees",
            "radians",
        ]

        for func in math_funcs:
            expression = re.sub(r"\b" + func + r"\(", "math." + func + "(", expression)

        return expression


class DateTimeTool(BaseTool):
    """Tool for date and time operations"""

    def __init__(self):
        super().__init__(
            name="datetime",
            description="Get current date and time information or perform date calculations",
        )
        self._parameters = {
            "operation": {
                "type": "string",
                "description": "The operation to perform (current, format, add, subtract)",
                "enum": ["current", "format", "add", "subtract"],
            },
            "format": {
                "type": "string",
                "description": "Format string for date/time (e.g. '%Y-%m-%d %H:%M:%S')",
                "default": "%Y-%m-%d %H:%M:%S",
            },
            "date": {
                "type": "string",
                "description": "Date string in ISO format (YYYY-MM-DD) for operations",
            },
            "days": {
                "type": "integer",
                "description": "Number of days for add/subtract operations",
            },
            "hours": {
                "type": "integer",
                "description": "Number of hours for add/subtract operations",
            },
            "minutes": {
                "type": "integer",
                "description": "Number of minutes for add/subtract operations",
            },
        }
        self._required_params = ["operation"]
        logger.info("Initialized tool: datetime")

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": self._parameters,
            "required": self._required_params,
        }

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        self.logger.debug(f"Executing datetime tool with parameters: {parameters}")

        try:
            # Validate parameters
            self._validate_parameters(parameters)

            # Execute tool-specific logic
            result = await self._execute(**parameters)
            self.logger.debug(f"DateTime execution result: {result}")

            return result
        except Exception as e:
            self.logger.error(f"DateTime execution failed: {e}", exc_info=True)
            return {"error": str(e)}

    async def _execute(
        self,
        operation: str,
        format: str = "%Y-%m-%d %H:%M:%S",
        date: Optional[str] = None,
        days: Optional[int] = None,
        hours: Optional[int] = None,
        minutes: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute the datetime tool

        Args:
            operation: The operation to perform
            format: Format string for date/time
            date: Date string for operations
            days: Number of days for add/subtract
            hours: Number of hours for add/subtract
            minutes: Number of minutes for add/subtract

        Returns:
            Dictionary with result
        """
        try:
            now = datetime.datetime.now()

            if operation == "current":
                return {
                    "current_date": now.strftime("%Y-%m-%d"),
                    "current_time": now.strftime("%H:%M:%S"),
                    "current_datetime": now.strftime(format),
                    "day_of_week": now.strftime("%A"),
                    "timestamp": now.timestamp(),
                    "timezone": datetime.datetime.now().astimezone().tzname(),
                }

            elif operation == "format":
                if date:
                    dt = datetime.datetime.fromisoformat(date)
                else:
                    dt = now

                return {"formatted_date": dt.strftime(format)}

            elif operation in ["add", "subtract"]:
                if date:
                    dt = datetime.datetime.fromisoformat(date)
                else:
                    dt = now

                delta = datetime.timedelta(
                    days=days or 0, hours=hours or 0, minutes=minutes or 0
                )

                if operation == "add":
                    result = dt + delta
                else:
                    result = dt - delta

                return {
                    "original_date": dt.strftime(format),
                    "result_date": result.strftime(format),
                    "operation": operation,
                    "days": days,
                    "hours": hours,
                    "minutes": minutes,
                }

            else:
                return {"error": f"Unknown operation: {operation}"}

        except Exception as e:
            logger.error(f"DateTime error: {e}")
            return {"error": f"DateTime error: {str(e)}"}
