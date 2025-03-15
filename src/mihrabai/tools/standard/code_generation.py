"""
Code generation and execution tools for agents
"""

import json
import os
import tempfile
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union
import asyncio
from pathlib import Path
import uuid
import shutil

from ..base import BaseTool


class CodeGenerationTool(BaseTool):
    """Tool for generating and executing code"""

    def __init__(self, workspace_dir: str = "./code_workspace"):
        super().__init__(
            name="code_generation",
            description="Generate and execute code in various programming languages",
        )
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
    
    async def _execute(self, parameters: Dict[str, Any]) -> str:
        """Execute the code generation tool with the given parameters"""
        action = parameters.get("action", "generate")
        
        if action == "generate":
            return await self._generate_code(
                code=parameters.get("code", ""),
                language=parameters.get("language", "python"),
                filename=parameters.get("filename"),
                description=parameters.get("description", ""),
            )
        elif action == "execute":
            return await self._execute_code(
                code=parameters.get("code"),
                language=parameters.get("language", "python"),
                args=parameters.get("args", []),
                timeout=parameters.get("timeout", 10),
            )
        elif action == "save_file":
            return await self._save_file(
                filename=parameters.get("filename"),
                content=parameters.get("content", ""),
                overwrite=parameters.get("overwrite", False),
            )
        elif action == "read_file":
            return await self._read_file(
                filename=parameters.get("filename"),
            )
        elif action == "list_files":
            return await self._list_files(
                directory=parameters.get("directory", ""),
            )
        else:
            return f"Unknown action: {action}"
    
    async def _generate_code(
        self,
        code: str,
        language: str = "python",
        filename: Optional[str] = None,
        description: str = "",
    ) -> str:
        """Generate code and optionally save it to a file"""
        if not code:
            return "No code provided to generate"
        
        # Generate a filename if not provided
        if not filename:
            extension = self._get_file_extension(language)
            filename = f"generated_{uuid.uuid4().hex[:8]}{extension}"
        
        # Ensure the filename has the correct extension
        if not self._has_correct_extension(filename, language):
            extension = self._get_file_extension(language)
            filename = f"{filename}{extension}"
        
        # Create the file path
        file_path = self.workspace_dir / filename
        
        # Save the code to the file
        with open(file_path, "w") as f:
            f.write(code)
        
        # Create a metadata file if description is provided
        if description:
            metadata_path = self.workspace_dir / f"{filename}.meta.json"
            with open(metadata_path, "w") as f:
                json.dump({
                    "description": description,
                    "language": language,
                    "created_at": datetime.now().isoformat(),
                }, f, indent=2)
        
        return f"Generated code saved to {filename}"
    
    async def _execute_code(
        self,
        code: Optional[str] = None,
        language: str = "python",
        args: List[str] = [],
        timeout: int = 10,
    ) -> str:
        """Execute code in the specified language"""
        if not code and not (language.lower() == "python" and args):
            return "No code or script provided to execute"
        
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(
            suffix=self._get_file_extension(language),
            delete=False,
            mode="w",
            dir=self.workspace_dir
        ) as temp_file:
            if code:
                temp_file.write(code)
                temp_file_path = temp_file.name
            else:
                # If no code is provided but args are, assume the first arg is a script path
                temp_file_path = None
        
        try:
            # Prepare the command based on the language
            if language.lower() == "python":
                if temp_file_path:
                    cmd = [sys.executable, temp_file_path] + args
                else:
                    # If no code is provided, assume the first arg is a script path
                    script_path = args[0]
                    if not os.path.isabs(script_path):
                        script_path = str(self.workspace_dir / script_path)
                    cmd = [sys.executable, script_path] + args[1:]
            elif language.lower() == "javascript" or language.lower() == "node":
                cmd = ["node", temp_file_path] + args
            elif language.lower() == "shell" or language.lower() == "bash":
                cmd = ["bash", temp_file_path] + args
            else:
                if temp_file_path:
                    os.unlink(temp_file_path)
                return f"Unsupported language: {language}"
            
            # Execute the command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_dir)
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                stdout_str = stdout.decode("utf-8")
                stderr_str = stderr.decode("utf-8")
                
                result = f"Execution completed with exit code: {process.returncode}\n\n"
                
                if stdout_str:
                    result += f"STDOUT:\n{stdout_str}\n\n"
                
                if stderr_str:
                    result += f"STDERR:\n{stderr_str}\n"
                
                return result
            except asyncio.TimeoutError:
                process.kill()
                return f"Execution timed out after {timeout} seconds"
        finally:
            # Clean up the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    async def _save_file(
        self,
        filename: str,
        content: str,
        overwrite: bool = False,
    ) -> str:
        """Save content to a file in the workspace"""
        if not filename:
            return "No filename provided"
        
        # Create the file path
        file_path = self.workspace_dir / filename
        
        # Check if the file already exists
        if file_path.exists() and not overwrite:
            return f"File {filename} already exists. Use overwrite=True to replace it."
        
        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the content to the file
        with open(file_path, "w") as f:
            f.write(content)
        
        return f"File saved to {filename}"
    
    async def _read_file(
        self,
        filename: str,
    ) -> str:
        """Read content from a file in the workspace"""
        if not filename:
            return "No filename provided"
        
        # Create the file path
        file_path = self.workspace_dir / filename
        
        # Check if the file exists
        if not file_path.exists():
            return f"File {filename} not found"
        
        # Read the file content
        try:
            with open(file_path, "r") as f:
                content = f.read()
            
            # Get file extension
            extension = file_path.suffix.lower()
            
            # Format the response
            result = f"Content of {filename}:\n\n"
            
            # Add code formatting for recognized file types
            if extension in [".py", ".js", ".java", ".c", ".cpp", ".h", ".cs", ".php", ".rb", ".go", ".ts", ".sh"]:
                result += f"```{extension[1:]}\n{content}\n```"
            else:
                result += content
            
            return result
        except Exception as e:
            return f"Error reading file {filename}: {str(e)}"
    
    async def _list_files(
        self,
        directory: str = "",
    ) -> str:
        """List files in the workspace or a subdirectory"""
        # Create the directory path
        dir_path = self.workspace_dir
        if directory:
            dir_path = dir_path / directory
        
        # Check if the directory exists
        if not dir_path.exists() or not dir_path.is_dir():
            return f"Directory {directory or '.'} not found"
        
        # List files and directories
        files = []
        directories = []
        
        for item in dir_path.iterdir():
            if item.is_file():
                # Skip metadata files
                if item.name.endswith(".meta.json"):
                    continue
                
                files.append({
                    "name": item.name,
                    "size": item.stat().st_size,
                    "modified": item.stat().st_mtime,
                })
            elif item.is_dir():
                directories.append({
                    "name": item.name,
                    "items": len(list(item.iterdir())),
                })
        
        # Sort files and directories by name
        files.sort(key=lambda x: x["name"])
        directories.sort(key=lambda x: x["name"])
        
        # Format the response
        result = f"Contents of {directory or '.'}:\n\n"
        
        if directories:
            result += "Directories:\n"
            for d in directories:
                result += f"- {d['name']}/ ({d['items']} items)\n"
            result += "\n"
        
        if files:
            result += "Files:\n"
            for f in files:
                size_str = self._format_size(f["size"])
                result += f"- {f['name']} ({size_str})\n"
        
        if not files and not directories:
            result += "Directory is empty"
        
        return result
    
    def _get_file_extension(self, language: str) -> str:
        """Get the file extension for a language"""
        language = language.lower()
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "node": ".js",
            "typescript": ".ts",
            "java": ".java",
            "c": ".c",
            "cpp": ".cpp",
            "c++": ".cpp",
            "csharp": ".cs",
            "c#": ".cs",
            "go": ".go",
            "ruby": ".rb",
            "php": ".php",
            "shell": ".sh",
            "bash": ".sh",
            "html": ".html",
            "css": ".css",
        }
        return extensions.get(language, ".txt")
    
    def _has_correct_extension(self, filename: str, language: str) -> bool:
        """Check if a filename has the correct extension for the language"""
        expected_ext = self._get_file_extension(language)
        return filename.lower().endswith(expected_ext.lower())
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in a human-readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["generate", "execute", "save_file", "read_file", "list_files"],
                    "description": "Action to perform with the code generation tool",
                },
                "code": {
                    "type": "string",
                    "description": "Code to generate or execute",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language of the code",
                },
                "filename": {
                    "type": "string",
                    "description": "Filename to save the code to or read from",
                },
                "description": {
                    "type": "string",
                    "description": "Description of the generated code",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command-line arguments for code execution",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds for code execution",
                },
                "content": {
                    "type": "string",
                    "description": "Content to save to a file",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Whether to overwrite an existing file",
                },
                "directory": {
                    "type": "string",
                    "description": "Directory to list files from",
                },
            },
            "required": ["action"],
        }
