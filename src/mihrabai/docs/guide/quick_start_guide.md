# Mihrab AI Agent - Quick Start Guide

This quick start guide will help you get up and running with the Mihrab AI Agent package in minutes.

## Installation

Install the package using pip:

```bash
pip install mihrab-ai-agent
```

## Setting Up API Keys

Create a `.env` file in your project root with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GROQ_API_KEY=your_groq_api_key
```

## Basic Usage

Here's a simple example to create a basic chat agent:

```python
import asyncio
import os
from dotenv import load_dotenv

from mihrabai import create_agent, Message, MessageRole

# Load environment variables
load_dotenv()

async def main():
    # Create an agent with OpenAI's GPT-4
    agent = await create_agent(
        provider_name="openai",
        model_name="gpt-4o",
        system_message="You are a helpful AI assistant."
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

### Using Groq Models

You can also use Groq models for faster inference:

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

Note: When using Groq, use `create_task_agent` instead of `create_agent` and explicitly pass the API key in `provider_kwargs`. For more details, see the [Using Groq Provider](using_groq_provider.md) guide.

## Using Tools

Here's how to create an agent that can use web tools:

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
    
    # Create an agent with function calling capability
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

### Using Tools with Groq

When using tools with Groq, pass them directly in the `tools` parameter:

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

## Available Tools

Mihrab AI Agent comes with a wide range of built-in tools:

- **Web Tools**: HTTP requests, web scraping
- **Filesystem Tools**: Read, write, and list files
- **Code Generation Tools**: Generate, review, and execute code
- **Text Processing Tools**: Analyze and summarize text
- **Memory Tools**: Store and retrieve information
- **Data Analysis Tools**: Analyze and visualize data
- **Document Tools**: Create and manipulate documents
- **Utility Tools**: Various utility functions

## Multi-Agent Collaboration

You can create teams of specialized agents that collaborate on complex tasks:

```python
import asyncio
import os
from dotenv import load_dotenv

from mihrabai import create_agent, Message, MessageRole
from mihrabai.models.base import ModelCapability

# Load environment variables
load_dotenv()

async def main():
    # Create specialized agents
    web_researcher = await create_agent(
        provider_name="openai",
        model_name="gpt-4o",
        system_message="You are a web research specialist.",
        required_capabilities={ModelCapability.FUNCTION_CALLING},
    )
    
    text_analyzer = await create_agent(
        provider_name="openai",
        model_name="gpt-4o",
        system_message="You are a text analysis specialist.",
        required_capabilities={ModelCapability.FUNCTION_CALLING},
    )
    
    # Add appropriate tools to each agent
    # ...
    
    # Implement coordination logic
    # ...

if __name__ == "__main__":
    asyncio.run(main())
```

## Supported LLM Providers

Mihrab AI Agent supports multiple LLM providers:

- **OpenAI**: GPT-4, GPT-3.5, etc.
- **Anthropic**: Claude 3 Opus, Claude 3 Sonnet, etc.
- **Groq**: Llama 3, Llama 2, Mixtral, etc.
- **Local Models**: Via Ollama integration

## Next Steps

For more detailed information and examples, check out:

- [Full Documentation](using_mihrab_with_tools.md)
- [Using Groq Provider](using_groq_provider.md)
- [Examples Directory](../../examples/)
- [API Reference](api_reference.md)

Happy building with Mihrab AI Agent! 