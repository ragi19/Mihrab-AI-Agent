"""
Shell command execution tools
"""

import asyncio
import shlex
from typing import Any, Dict

from ...core.types import JSON
from ..base import BaseTool


class ShellCommandTool(BaseTool):
    """Tool for executing shell commands"""

    def __init__(self):
        super().__init__(
            name="shell_command", description="Execute shell commands safely"
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
        """Execute a shell command"""
        command = parameters["command"]
        timeout = parameters.get("timeout", 60)
        cwd = parameters.get("cwd")

        try:
            # Split command into arguments safely
            args = shlex.split(command)

            # Create process
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )

                return {
                    "success": True,
                    "exit_code": process.returncode,
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                }
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "timeout": {
                    "type": "number",
                    "description": "Command timeout in seconds",
                    "default": 60,
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for command execution",
                },
            },
            "required": ["command"],
        }
