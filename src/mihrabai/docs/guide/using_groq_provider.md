# Using Groq with Mihrab AI Agent

This guide explains how to use Groq models with Mihrab AI Agent. Groq is a high-performance AI inference platform that offers fast and efficient access to large language models.

## Setting Up Groq API Key

Before using Groq with Mihrab AI Agent, you need to obtain a Groq API key from [Groq's website](https://console.groq.com/).

Once you have your API key, you can set it as an environment variable:

```bash
export GROQ_API_KEY=your_groq_api_key
```

Or add it to your `.env` file:

```
GROQ_API_KEY=your_groq_api_key
```

## Basic Usage with Groq

Here's a simple example of creating a basic chat agent using Groq:

```python
import asyncio
import os
import sys
from dotenv import load_dotenv

from mihrabai import create_task_agent, Message, MessageRole

# Load environment variables
load_dotenv()

# Get Groq API key
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    print("Groq API key not found in environment variables.")
    groq_api_key = input("Please enter your Groq API key: ")
    if not groq_api_key:
        print("Error: Groq API key is required to run this example.")
        sys.exit(1)

async def main():
    # Create an agent with Groq's Llama3-70B model
    agent = await create_task_agent(
        provider_name="groq",
        model_name="llama3-70b-8192",
        system_message="You are a helpful AI assistant.",
        provider_kwargs={"api_key": groq_api_key}  # Explicitly pass the API key
    )
    
    # Create a user message
    message = Message(role=MessageRole.USER, content="Hello, who are you?")
    
    # Process the message and get a response
    response = await agent.process_message(message)
    
    # Print the response
    print(f"Agent: {response.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Important Differences When Using Groq

When using Groq with Mihrab AI Agent, there are a few important differences compared to using other providers like OpenAI:

1. **Use `create_task_agent` instead of `create_agent`**: Groq works best with the `TaskAgent` implementation.

2. **Explicitly pass the API key**: Always pass the Groq API key explicitly in the `provider_kwargs` parameter.

3. **Pass tools directly**: When using tools with Groq, pass them directly in the `tools` parameter of `create_task_agent` instead of using `add_tool`.

4. **Implement `execute` method in custom tools**: If you're creating custom tools, make sure to implement the `execute` method that calls `_execute`.

## Using Tools with Groq

Here's an example of using web tools with Groq:

```python
import asyncio
import os
import sys
from dotenv import load_dotenv

from mihrabai import create_task_agent, Message, MessageRole
from mihrabai.tools.standard.web import HTTPRequestTool, WebScraperTool

# Load environment variables
load_dotenv()

# Get Groq API key
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    print("Groq API key not found in environment variables.")
    groq_api_key = input("Please enter your Groq API key: ")
    if not groq_api_key:
        print("Error: Groq API key is required to run this example.")
        sys.exit(1)

async def main():
    # Create tool instances
    http_tool = HTTPRequestTool()
    scraper_tool = WebScraperTool()
    
    # Create an agent with Groq's Llama3-70B model
    agent = await create_task_agent(
        provider_name="groq",
        model_name="llama3-70b-8192",
        system_message="You are a helpful AI assistant that can search the web for information.",
        provider_kwargs={"api_key": groq_api_key},  # Explicitly pass the API key
        tools=[http_tool, scraper_tool]  # Pass tools directly
    )
    
    # Create a user message
    message = Message(
        role=MessageRole.USER,
        content="What are the latest headlines about artificial intelligence?"
    )
    
    # Process the message
    response = await agent.process_message(message)
    
    # Print the response
    print(f"Agent: {response.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Creating Custom Tools for Groq

When creating custom tools for use with Groq, make sure to implement the `execute` method that calls `_execute`:

```python
from typing import Dict, Any
from mihrabai.tools.base import BaseTool
from mihrabai.core.types import JSON

class MyCustomTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_custom_tool",
            description="Description of my custom tool"
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> JSON:
        """Execute the tool with the given parameters"""
        return await self._execute(parameters)
    
    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        # Implement your tool logic here
        return {"result": "Tool execution result"}
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Description of parameter 1"
                },
                "param2": {
                    "type": "number",
                    "description": "Description of parameter 2"
                }
            },
            "required": ["param1"]
        }
```

## Advanced Usage: Multi-Agent Collaboration with Groq

Here's an example of creating multiple specialized agents using Groq that collaborate on complex tasks:

```python
import asyncio
import os
import sys
from dotenv import load_dotenv

from mihrabai import create_task_agent, Message, MessageRole
from mihrabai.tools.standard.web import HTTPRequestTool, WebScraperTool
from mihrabai.tools.standard.filesystem import FileReaderTool, FileWriterTool

# Load environment variables
load_dotenv()

# Get Groq API key
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    print("Groq API key not found in environment variables.")
    groq_api_key = input("Please enter your Groq API key: ")
    if not groq_api_key:
        print("Error: Groq API key is required to run this example.")
        sys.exit(1)

async def main():
    # Create tool instances
    http_tool = HTTPRequestTool()
    scraper_tool = WebScraperTool()
    file_read_tool = FileReaderTool()
    file_write_tool = FileWriterTool()
    
    # Create specialized agents
    research_agent = await create_task_agent(
        provider_name="groq",
        model_name="llama3-70b-8192",
        system_message="You are a specialized research agent that excels at finding information.",
        provider_kwargs={"api_key": groq_api_key},
        tools=[http_tool, scraper_tool]
    )
    
    writing_agent = await create_task_agent(
        provider_name="groq",
        model_name="llama3-70b-8192",
        system_message="You are a specialized writing agent that excels at creating well-structured content.",
        provider_kwargs={"api_key": groq_api_key},
        tools=[file_read_tool, file_write_tool]
    )
    
    # Implement coordination logic
    # ...

if __name__ == "__main__":
    asyncio.run(main())
```

## Available Groq Models

Mihrab AI Agent supports the following Groq models:

- `llama3-70b-8192`: Llama 3 70B model with 8192 context window
- `llama3-8b-8192`: Llama 3 8B model with 8192 context window
- `llama2-70b-4096`: Llama 2 70B model with 4096 context window
- `mixtral-8x7b-32768`: Mixtral 8x7B model with 32768 context window
- And many more...

You can check the available models by running:

```python
from mihrabai.models.providers.groqprovider import GroqProvider

provider = GroqProvider(api_key="your_groq_api_key")
models = provider.list_models()
for model in models:
    print(model)
```

## Troubleshooting

If you encounter issues when using Groq with Mihrab AI Agent, here are some common problems and solutions:

### Invalid API Key

If you see an error about an invalid API key, make sure:
- Your Groq API key is correct
- You're passing it explicitly in the `provider_kwargs` parameter
- Your API key has not expired

### Import Errors

If you see import errors related to missing classes or modules, make sure:
- You're using the latest version of Mihrab AI Agent
- You're using the correct import paths
- You're using `create_task_agent` instead of `create_agent`

### Tool Execution Errors

If you see errors related to tool execution, make sure:
- Your custom tools implement the `execute` method
- You're passing tools directly in the `tools` parameter
- Your tool parameters match the schema

## Conclusion

Groq provides a high-performance alternative to other LLM providers, offering fast inference times and competitive model quality. By following the guidelines in this document, you can effectively use Groq models with Mihrab AI Agent for a wide range of applications.

For more examples and detailed information, check out the [examples directory](../../examples/) and the [API reference](api_reference.md). 