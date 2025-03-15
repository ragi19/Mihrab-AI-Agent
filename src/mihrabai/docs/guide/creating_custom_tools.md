# Creating Custom Tools for Mihrab AI Agent

This guide explains how to create custom tools for the Mihrab AI Agent package, allowing you to extend the functionality of your agents with specialized capabilities.

## Table of Contents

1. [Understanding Tools](#understanding-tools)
2. [Tool Structure](#tool-structure)
3. [Creating a Basic Tool](#creating-a-basic-tool)
4. [Parameter Validation](#parameter-validation)
5. [Error Handling](#error-handling)
6. [Registering and Using Tools](#registering-and-using-tools)
7. [Advanced Tool Patterns](#advanced-tool-patterns)
8. [Best Practices](#best-practices)
9. [Examples](#examples)

## Understanding Tools

In Mihrab AI Agent, tools are functions that agents can use to interact with their environment. Each tool has:

- A name and description
- A set of parameters with types and descriptions
- Execution logic that performs the tool's function
- A return value that the agent can use to generate responses

Tools are defined as classes that extend the `BaseTool` abstract base class, which provides common functionality for parameter validation, error handling, and schema generation.

## Tool Structure

A tool class has the following structure:

```python
from typing import Dict, Any
from mihrabai.tools.base import BaseTool
from mihrabai.core.types import JSON

class MyCustomTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_custom_tool",
            description="A custom tool that does something useful"
        )
    
    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        # Implement your tool logic here
        result = {"status": "success", "message": "Custom tool executed"}
        return result
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "First parameter"
                },
                "param2": {
                    "type": "integer",
                    "description": "Second parameter"
                }
            },
            "required": ["param1"]
        }
```

The key components are:

1. **Initialization**: Set the tool's name and description
2. **_execute method**: Implement the tool's logic
3. **_get_parameters_schema method**: Define the parameters the tool accepts

## Creating a Basic Tool

Let's create a simple calculator tool that can perform basic arithmetic operations:

```python
from typing import Dict, Any
from mihrabai.tools.base import BaseTool
from mihrabai.core.types import JSON
from mihrabai.utils.logging import get_logger

class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Performs basic arithmetic operations"
        )
        self.logger = get_logger("tools.calculator")
    
    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        operation = parameters.get("operation")
        a = parameters.get("a")
        b = parameters.get("b")
        
        result = None
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Cannot divide by zero")
            result = a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return {
            "operation": operation,
            "a": a,
            "b": b,
            "result": result
        }
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "The first operand"
                },
                "b": {
                    "type": "number",
                    "description": "The second operand"
                }
            },
            "required": ["operation", "a", "b"]
        }
```

This tool:
1. Takes an operation name and two numbers as parameters
2. Performs the specified arithmetic operation
3. Returns the result along with the input parameters

## Parameter Validation

The `BaseTool` class automatically validates parameters against the schema you define in `_get_parameters_schema()`. This includes:

- Checking that required parameters are present
- Validating parameter types
- Ensuring parameters match any constraints (like enum values)

You can add custom validation in your `_execute` method if needed:

```python
async def _execute(self, parameters: Dict[str, Any]) -> JSON:
    # Parameters are already validated against the schema
    # But you can add custom validation logic here
    
    operation = parameters.get("operation")
    a = parameters.get("a")
    b = parameters.get("b")
    
    # Custom validation
    if operation == "divide" and b == 0:
        raise ValueError("Cannot divide by zero")
    
    # Rest of the implementation
    # ...
```

## Error Handling

Tools should handle errors gracefully and provide informative error messages. The `BaseTool` class catches exceptions and logs them, but you should still handle expected errors in your tool implementation:

```python
async def _execute(self, parameters: Dict[str, Any]) -> JSON:
    try:
        # Tool logic that might fail
        result = perform_operation(parameters)
        return {"status": "success", "result": result}
    except ValueError as e:
        # Handle expected errors
        self.logger.warning(f"Value error in calculator tool: {e}")
        return {"status": "error", "error": str(e)}
    except Exception as e:
        # Log unexpected errors and re-raise
        self.logger.error(f"Unexpected error in calculator tool: {e}", exc_info=True)
        raise
```

## Registering and Using Tools

Once you've created your tool class, you need to register it with the `ToolRegistry` and create an instance to use it:

```python
from mihrabai.tools.registry import ToolRegistry

# Register your tool
ToolRegistry.register("calculator", CalculatorTool)

# Create a tool instance
calculator_tool = ToolRegistry.create_tool("calculator")

# Add the tool to an agent
agent.add_tool(calculator_tool)
```

Now your agent can use the calculator tool when processing messages:

```python
# User asks a question that might use the calculator
message = Message(
    role=MessageRole.USER,
    content="What is 123 multiplied by 456?"
)

# Agent will use the calculator tool to find the answer
response = await agent.process_message(message)
```

## Advanced Tool Patterns

### Tools with External Dependencies

For tools that require external dependencies, initialize them in the constructor:

```python
class DatabaseTool(BaseTool):
    def __init__(self, connection_string: str):
        super().__init__(
            name="database",
            description="Queries a database"
        )
        # Initialize external dependencies
        self.db_client = DatabaseClient(connection_string)
    
    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        # Use the initialized client
        result = await self.db_client.query(parameters.get("query"))
        return {"rows": result}
    
    # ...

# Create with dependencies
db_tool = ToolRegistry.create_tool(
    "database", 
    connection_string="postgresql://user:password@localhost/db"
)
```

### Stateful Tools

Some tools might need to maintain state between calls:

```python
class CounterTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="counter",
            description="Counts things"
        )
        # Initialize state
        self.counts = {}
    
    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        key = parameters.get("key", "default")
        action = parameters.get("action")
        
        if action == "increment":
            self.counts[key] = self.counts.get(key, 0) + 1
        elif action == "reset":
            self.counts[key] = 0
        
        return {"key": key, "count": self.counts.get(key, 0)}
    
    # ...
```

### Composable Tools

You can create tools that use other tools:

```python
class CompoundTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="compound_tool",
            description="A tool that uses other tools"
        )
        # Get instances of other tools
        self.tool1 = ToolRegistry.create_tool("tool1")
        self.tool2 = ToolRegistry.create_tool("tool2")
    
    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        # Use the first tool
        result1 = await self.tool1.execute(parameters.get("tool1_params", {}))
        
        # Use the second tool with results from the first
        tool2_params = parameters.get("tool2_params", {})
        tool2_params["input_from_tool1"] = result1.get("output")
        result2 = await self.tool2.execute(tool2_params)
        
        # Combine results
        return {
            "tool1_result": result1,
            "tool2_result": result2
        }
    
    # ...
```

## Best Practices

When creating custom tools, follow these best practices:

1. **Clear Description**: Provide a clear and concise description of what your tool does
2. **Descriptive Parameters**: Make parameter names and descriptions intuitive
3. **Comprehensive Validation**: Validate all parameters and handle edge cases
4. **Informative Errors**: Return helpful error messages when things go wrong
5. **Proper Logging**: Log important events and errors
6. **Async Implementation**: Use async/await for I/O-bound operations
7. **Structured Results**: Return structured data that's easy for the agent to use
8. **Security Considerations**: Validate inputs and limit access to sensitive operations
9. **Resource Management**: Clean up resources when they're no longer needed
10. **Documentation**: Document your tool's purpose, parameters, and return values

## Examples

### Web Search Tool

```python
import aiohttp
from typing import Dict, Any
from mihrabai.tools.base import BaseTool
from mihrabai.core.types import JSON

class WebSearchTool(BaseTool):
    def __init__(self, api_key: str):
        super().__init__(
            name="web_search",
            description="Searches the web for information"
        )
        self.api_key = api_key
        self.search_url = "https://api.search.com/v1/search"
    
    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        query = parameters.get("query")
        num_results = parameters.get("num_results", 5)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.search_url,
                params={
                    "q": query,
                    "limit": num_results,
                    "api_key": self.api_key
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Search failed: {response.status} - {error_text}")
                
                data = await response.json()
                
                return {
                    "query": query,
                    "results": [
                        {
                            "title": result.get("title"),
                            "url": result.get("url"),
                            "snippet": result.get("snippet")
                        }
                        for result in data.get("results", [])
                    ]
                }
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["query"]
        }
```

### File Processing Tool

```python
import os
import json
from typing import Dict, Any
from mihrabai.tools.base import BaseTool
from mihrabai.core.types import JSON

class FileProcessorTool(BaseTool):
    def __init__(self, allowed_directories: list[str]):
        super().__init__(
            name="file_processor",
            description="Processes files in allowed directories"
        )
        self.allowed_directories = allowed_directories
    
    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        file_path = parameters.get("file_path")
        operation = parameters.get("operation")
        
        # Security check: ensure file is in an allowed directory
        if not any(file_path.startswith(directory) for directory in self.allowed_directories):
            raise ValueError(f"Access denied: {file_path} is not in an allowed directory")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
        
        if operation == "read":
            with open(file_path, "r") as f:
                content = f.read()
            return {
                "file_path": file_path,
                "content": content
            }
        elif operation == "stats":
            stats = os.stat(file_path)
            return {
                "file_path": file_path,
                "size_bytes": stats.st_size,
                "last_modified": stats.st_mtime,
                "is_directory": os.path.isdir(file_path)
            }
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to process"
                },
                "operation": {
                    "type": "string",
                    "enum": ["read", "stats"],
                    "description": "Operation to perform on the file"
                }
            },
            "required": ["file_path", "operation"]
        }
```

These examples demonstrate how to create custom tools for different use cases. You can adapt them to your specific needs or use them as inspiration for your own tools. 