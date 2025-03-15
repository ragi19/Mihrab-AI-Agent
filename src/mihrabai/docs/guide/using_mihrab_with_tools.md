# Using Mihrab AI Agent with Tools

This documentation provides a comprehensive guide on how to use the Mihrab AI Agent package with various tools to create powerful AI agents that can interact with the world.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Basic Concepts](#basic-concepts)
4. [Setting Up Your Environment](#setting-up-your-environment)
5. [Creating Your First Agent](#creating-your-first-agent)
6. [Working with Tools](#working-with-tools)
   - [Web Tools](#web-tools)
   - [Filesystem Tools](#filesystem-tools)
   - [Code Generation Tools](#code-generation-tools)
   - [Text Processing Tools](#text-processing-tools)
   - [Custom Tools](#custom-tools)
7. [Multi-Agent Systems](#multi-agent-systems)
8. [Advanced Features](#advanced-features)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)

## Introduction

Mihrab AI Agent is a powerful framework for creating AI agents that can use tools to interact with their environment. The framework supports multiple LLM providers (OpenAI, Anthropic, Groq, etc.) and offers a wide range of built-in tools for web interaction, filesystem operations, code generation, text processing, and more.

Key features of Mihrab AI Agent:

- **Multi-provider support**: Use models from OpenAI, Anthropic, Groq, and more
- **Extensive tool library**: Built-in tools for various tasks
- **Multi-agent collaboration**: Create teams of specialized agents
- **Memory systems**: Implement short and long-term memory for agents
- **Handoff mechanisms**: Transfer tasks between agents
- **Error recovery**: Robust error handling and recovery

## Installation

To install the Mihrab AI Agent package:

```bash
pip install mihrab-ai-agent
```

## Basic Concepts

Before diving into the details, let's understand some basic concepts:

### Agents

An agent is an AI entity powered by a language model that can process messages, use tools, and generate responses. Agents can be specialized for specific tasks or domains.

### Tools

Tools are functions that agents can use to interact with their environment. Tools have a name, description, and parameters schema, and they return results that the agent can use to generate responses.

### Messages

Messages are the units of communication between users and agents. Each message has a role (user, assistant, system, or function) and content.

### Providers

Providers are the sources of language models that power agents. Mihrab supports multiple providers, including OpenAI, Anthropic, and Groq.

## Setting Up Your Environment

Before using Mihrab AI Agent, you need to set up your environment with the necessary API keys:

1. Create a `.env` file in your project root
2. Add your API keys for the providers you want to use:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GROQ_API_KEY=your_groq_api_key
```

3. Load the environment variables in your code:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Creating Your First Agent

Here's how to create a basic agent:

```python
import asyncio
from mihrabai import create_agent, Message, MessageRole

async def main():
    # Create an agent with OpenAI's GPT-4
    agent = await create_agent(
        provider_name="openai",
        model_name="gpt-4o",
        system_message="You are a helpful AI assistant."
    )
    
    # Create a user message
    message = Message(role=MessageRole.USER, content="Hello, who are you?")
    
    # Add the message to the agent's conversation history
    agent.add_to_history(message)
    
    # Process the message and get a response
    response = await agent.process_message(message)
    
    # Print the response
    print(f"Agent: {response.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Working with Tools

Mihrab AI Agent provides a wide range of built-in tools that agents can use to interact with their environment.

### Web Tools

Web tools allow agents to interact with the internet, make HTTP requests, and scrape web content.

```python
from mihrabai.tools.registry import ToolRegistry
from mihrabai.tools.standard.web import HTTPRequestTool, WebScraperTool

# Register web tools
ToolRegistry.register("http_request", HTTPRequestTool)
ToolRegistry.register("web_scraper", WebScraperTool)

# Create tool instances
http_tool = ToolRegistry.create_tool("http_request")
scraper_tool = ToolRegistry.create_tool("web_scraper")

# Add tools to the agent
agent.add_tool(http_tool)
agent.add_tool(scraper_tool)
```

Example usage:

```python
# User asks a question that requires web search
message = Message(
    role=MessageRole.USER,
    content="What's the current weather in New York City?"
)

# Agent will use web tools to find the information
response = await agent.process_message(message)
```

### Filesystem Tools

Filesystem tools allow agents to read, write, and list files.

```python
from mihrabai.tools.standard.filesystem import FileReadTool, FileWriteTool, FileListTool

# Register filesystem tools
ToolRegistry.register("file_read", FileReadTool)
ToolRegistry.register("file_write", FileWriteTool)
ToolRegistry.register("file_list", FileListTool)

# Create tool instances
file_read_tool = ToolRegistry.create_tool("file_read")
file_write_tool = ToolRegistry.create_tool("file_write")
file_list_tool = ToolRegistry.create_tool("file_list")

# Add tools to the agent
agent.add_tool(file_read_tool)
agent.add_tool(file_write_tool)
agent.add_tool(file_list_tool)
```

Example usage:

```python
# User asks to create a file
message = Message(
    role=MessageRole.USER,
    content="Create a file called notes.txt with a list of 3 important tasks"
)

# Agent will use filesystem tools to create the file
response = await agent.process_message(message)
```

### Code Generation Tools

Code generation tools allow agents to generate, review, and execute code.

```python
from mihrabai.tools.standard.code_generation import CodeGeneratorTool, CodeReviewTool, CodeExecutorTool

# Register code tools
ToolRegistry.register("code_generator", CodeGeneratorTool)
ToolRegistry.register("code_reviewer", CodeReviewTool)
ToolRegistry.register("code_executor", CodeExecutorTool)

# Create tool instances
generator_tool = ToolRegistry.create_tool("code_generator")
reviewer_tool = ToolRegistry.create_tool("code_reviewer")
executor_tool = ToolRegistry.create_tool("code_executor")

# Add tools to the agent
agent.add_tool(generator_tool)
agent.add_tool(reviewer_tool)
agent.add_tool(executor_tool)
```

Example usage:

```python
# User asks to generate code
message = Message(
    role=MessageRole.USER,
    content="Generate a Python script that creates a simple web server"
)

# Agent will use code generation tools to create the script
response = await agent.process_message(message)
```

### Text Processing Tools

Text processing tools allow agents to analyze, summarize, and transform text.

```python
from mihrabai.tools.standard.text_processing import TextSummarizerTool, TextAnalysisTool

# Register text processing tools
ToolRegistry.register("text_summarizer", TextSummarizerTool)
ToolRegistry.register("text_analyzer", TextAnalysisTool)

# Create tool instances
summarizer_tool = ToolRegistry.create_tool("text_summarizer")
analyzer_tool = ToolRegistry.create_tool("text_analyzer")

# Add tools to the agent
agent.add_tool(summarizer_tool)
agent.add_tool(analyzer_tool)
```

Example usage:

```python
# User asks to summarize text
message = Message(
    role=MessageRole.USER,
    content="Summarize the following article: [long article text]"
)

# Agent will use text processing tools to summarize the article
response = await agent.process_message(message)
```

### Custom Tools

You can create custom tools by extending the `BaseTool` class:

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

# Register and use your custom tool
ToolRegistry.register("my_custom_tool", MyCustomTool)
custom_tool = ToolRegistry.create_tool("my_custom_tool")
agent.add_tool(custom_tool)
```

## Multi-Agent Systems

Mihrab AI Agent allows you to create teams of specialized agents that can collaborate on complex tasks.

```python
from mihrabai import create_agent, Message, MessageRole

# Create specialized agents
web_researcher = await create_agent(
    provider_name="openai",
    model_name="gpt-4o",
    system_message="You are a web research specialist."
)

text_analyzer = await create_agent(
    provider_name="anthropic",
    model_name="claude-3-opus-20240229",
    system_message="You are a text analysis specialist."
)

# Add appropriate tools to each agent
# ...

# Create a coordinator agent
coordinator = await create_agent(
    provider_name="openai",
    model_name="gpt-4o",
    system_message="You are a coordinator that manages a team of specialist agents."
)

# Implement the coordination logic
# ...
```

For a complete example of multi-agent collaboration, see the [Multi-Agent Systems](#multi-agent-systems) section.

## Advanced Features

### Memory Systems

Mihrab AI Agent supports both short-term and long-term memory for agents:

```python
from mihrabai.tools.standard.memory import MemoryStoreTool, MemoryRetrieveTool

# Register memory tools
ToolRegistry.register("memory_store", MemoryStoreTool)
ToolRegistry.register("memory_retrieve", MemoryRetrieveTool)

# Create tool instances
memory_store_tool = ToolRegistry.create_tool("memory_store")
memory_retrieve_tool = ToolRegistry.create_tool("memory_retrieve")

# Add tools to the agent
agent.add_tool(memory_store_tool)
agent.add_tool(memory_retrieve_tool)
```

### Handoff Mechanisms

Mihrab AI Agent supports handoff mechanisms for transferring tasks between agents:

```python
from mihrabai.handoff import HandoffManager

# Create a handoff manager
handoff_manager = HandoffManager()

# Register agents with the handoff manager
handoff_manager.register_agent("general", general_agent)
handoff_manager.register_agent("code_specialist", code_specialist)
handoff_manager.register_agent("research_specialist", research_specialist)

# Configure handoff rules
# ...

# Process a message with potential handoffs
result = await handoff_manager.process_with_handoff(
    agent_id="general",
    message=user_message
)
```

## Examples

Here are some examples of using Mihrab AI Agent with tools:

### Web Search Example

```python
import asyncio
import os
from dotenv import load_dotenv
from mihrabai import create_agent, Message, MessageRole
from mihrabai.models.base import ModelCapability
from mihrabai.tools.registry import ToolRegistry
from mihrabai.tools.standard.web import HTTPRequestTool, WebScraperTool

# Load environment variables
load_dotenv()

async def main():
    # Register web tools
    ToolRegistry.register("http_request", HTTPRequestTool)
    ToolRegistry.register("web_scraper", WebScraperTool)
    
    # Create tool instances
    http_tool = ToolRegistry.create_tool("http_request")
    scraper_tool = ToolRegistry.create_tool("web_scraper")
    
    # Create an agent
    agent = await create_agent(
        provider_name="openai",
        model_name="gpt-4o",
        system_message="You are a helpful AI assistant that can search the web for information.",
        required_capabilities={ModelCapability.FUNCTION_CALLING},
    )
    
    # Add tools to the agent
    agent.add_tool(http_tool)
    agent.add_tool(scraper_tool)
    
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

### File Management Example

```python
import asyncio
import os
import tempfile
from dotenv import load_dotenv
from mihrabai import create_agent, Message, MessageRole
from mihrabai.models.base import ModelCapability
from mihrabai.tools.registry import ToolRegistry
from mihrabai.tools.standard.filesystem import FileReadTool, FileWriteTool, FileListTool

# Load environment variables
load_dotenv()

async def main():
    # Create a temporary directory for file operations
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Register filesystem tools
        ToolRegistry.register("file_read", FileReadTool)
        ToolRegistry.register("file_write", FileWriteTool)
        ToolRegistry.register("file_list", FileListTool)
        
        # Create tool instances
        file_read_tool = ToolRegistry.create_tool("file_read")
        file_write_tool = ToolRegistry.create_tool("file_write")
        file_list_tool = ToolRegistry.create_tool("file_list")
        
        # Create an agent
        agent = await create_agent(
            provider_name="openai",
            model_name="gpt-4o",
            system_message=f"You are a helpful AI assistant that can manage files in {temp_dir}.",
            required_capabilities={ModelCapability.FUNCTION_CALLING},
        )
        
        # Add tools to the agent
        agent.add_tool(file_read_tool)
        agent.add_tool(file_write_tool)
        agent.add_tool(file_list_tool)
        
        # Create a user message
        message = Message(
            role=MessageRole.USER,
            content=f"Create a file called notes.txt in {temp_dir} with a list of 3 important tasks"
        )
        
        # Process the message
        response = await agent.process_message(message)
        
        # Print the response
        print(f"Agent: {response.content}")
        
        # List files in the temporary directory
        files = os.listdir(temp_dir)
        print(f"Files in {temp_dir}: {files}")

if __name__ == "__main__":
    asyncio.run(main())
```

For more examples, see the `examples` directory in the Mihrab AI Agent package.

## Troubleshooting

### Common Issues

1. **API Key Issues**
   - Make sure you have set the correct API keys in your `.env` file
   - Check that you're loading the environment variables with `load_dotenv()`

2. **Tool Registration Issues**
   - Ensure you're registering tools before creating tool instances
   - Check that tool names match between registration and creation

3. **Model Capability Issues**
   - Make sure the model you're using supports the capabilities you need (e.g., function calling)
   - Set the required capabilities when creating the agent

### Getting Help

If you encounter issues not covered in this documentation, you can:

- Check the examples in the `examples` directory
- Look at the source code for more details
- File an issue on the GitHub repository

## Next Steps

Now that you understand the basics of using Mihrab AI Agent with tools, you can:

1. Explore the examples in the `examples` directory
2. Create your own custom tools
3. Build multi-agent systems for complex tasks
4. Experiment with different LLM providers and models

Happy building with Mihrab AI Agent! 