# MihrabAI
![MihrabAI](image.webp)
A flexible and extensible framework for building AI agents powered by large language models (LLMs). Like the mihrab that guides prayer in a mosque, this framework provides direction and guidance through seamless integration with multiple LLM providers, intelligent provider fallback, and memory-enabled agents.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Multi-Provider Support**: Seamlessly integrate with OpenAI, Anthropic, Groq, and other LLM providers through a unified interface
- **Automatic Fallback**: Gracefully handle API errors with configurable fallback strategies
- **Cost Optimization**: Intelligently select providers based on cost, performance, or reliability
- **Memory-Enabled Agents**: Build agents with persistent memory for contextual conversations
- **Streaming Support**: Stream responses for real-time interaction
- **Tool Integration**: Easily extend agents with custom tools and functions
- **Provider Statistics**: Track usage, costs, and performance across providers

## New in Version 0.2.0

- **Command-Line Interface**: Interact with agents directly from the terminal
- **Memory Task Agents**: Enhanced agents with built-in memory capabilities
- **Improved Factory Functions**: Easily create specialized agents with a single function call
- **Extended Provider Support**: Better integration with more LLM providers, including Groq
- **Enhanced Documentation**: More examples and clearer usage instructions
- **Groq Integration**: Full support for Groq's high-performance LLM models, including Llama 3

## Installation

```bash
pip install mihrab-ai-agent
```

## Quick Start

### Basic Agent

```python
import asyncio
import os
from mihrabai.core.agent import SimpleAgent
from mihrabai.models import create_model
from mihrabai.runtime.runner import AgentRunner

async def main():
    # Create a model using OpenAI
    model = await create_model(
        provider_name="openai",
        model_name="gpt-3.5-turbo",
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # Create a simple agent
    agent = SimpleAgent(model=model)
    
    # Create a runner
    runner = AgentRunner(agent=agent)
    
    # Run a conversation
    response = await runner.run("Hello, who are you?")
    print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-Provider Agent

```python
import asyncio
import os
from mihrabai.models.multi_provider import MultiProviderModel, OptimizationStrategy
from mihrabai.core.agent import SimpleAgent
from mihrabai.runtime.runner import AgentRunner

async def main():
    # Create a multi-provider model
    model = await MultiProviderModel.create(
        primary_model="gpt-3.5-turbo",
        fallback_models=["claude-3-sonnet", "llama2-70b-4096"],
        required_capabilities={"chat"},
        optimize_for=OptimizationStrategy.COST,
        api_keys={
            "openai": os.environ.get("OPENAI_API_KEY"),
            "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
            "groq": os.environ.get("GROQ_API_KEY")
        }
    )
    
    # Create an agent with the multi-provider model
    agent = SimpleAgent(model=model)
    
    # Create a runner
    runner = AgentRunner(agent=agent)
    
    # Run a conversation
    response = await runner.run("What's the capital of Morocco?")
    print(f"Response from {model.current_provider}: {response.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Memory-Enabled Agent

```python
import asyncio
import os
from mihrabai.core.memory_task_agent import MemoryEnabledTaskAgent
from mihrabai.models import create_model
from mihrabai.runtime.memory_runner import MemoryAgentRunner

async def main():
    # Create a model
    model = await create_model(
        provider_name="anthropic",
        model_name="claude-3-sonnet",
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    # Create a memory-enabled agent
    agent = MemoryEnabledTaskAgent(
        model=model,
        system_message="You are a helpful assistant with memory like the ancient scholars of the House of Wisdom.",
        max_memory_items=50,
        memory_retrieval_count=5
    )
    
    # Create a memory runner
    runner = MemoryAgentRunner(
        agent=agent,
        memory_persistence_path="./manuscripts"
    )
    
    # Start a conversation with a session ID
    session_id = "scholar123"
    
    # First interaction
    await runner.run("My name is Hassan", session_id=session_id)
    
    # Save memory
    await runner.save_memory(session_id)
    
    # Later interaction (memory will be loaded automatically)
    response = await runner.run("What's my name?", session_id=session_id)
    print(response.content)  # Should remember the name "Hassan"

if __name__ == "__main__":
    asyncio.run(main())
```

## Architecture

The framework is built around a modular architecture like the geometric patterns of Islamic art:

- **Core**: Base classes for agents, messages, and memory
- **Models**: Provider integrations and model abstractions
- **Runtime**: Execution environment for agents
- **Utils**: Logging, tracing, and other utilities

## Supported Providers

- **OpenAI**: GPT-3.5, GPT-4, and other models
- **Anthropic**: Claude 3 Opus, Claude 3 Sonnet, and other models
- **Groq**: Llama 3-70B, Llama 3-8B, Llama 2-70B, Mixtral, and other models
  - High-performance inference with low latency
  - Cost-effective alternative to other providers
  - See our [Groq Integration Guide](src/mihrabai/docs/guide/using_groq_provider.md) for details
- Easily extensible to other providers

## Advanced Usage

### Custom Tools

```python
from mihrabai.core.task_agent import TaskAgent, ToolConfig
from mihrabai.core.message import Message, MessageRole

# Define a tool function
def get_weather(params):
    location = params.get("location", "Marrakech")
    return f"The weather in {location} is sunny and 85°F"

# Create a tool configuration
weather_tool = ToolConfig(
    name="get_weather",
    description="Get the current weather for a location",
    function=get_weather,
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and country, e.g. Marrakech, Morocco"
            }
        },
        "required": ["location"]
    }
)

# Create an agent with the tool
agent = TaskAgent(
    model=model,
    system_message="You are a helpful assistant that can check the weather.",
    tools=[weather_tool]
)
```

### Provider Statistics

```python
from mihrabai.models.provider_stats import ProviderStatsManager

# Create a stats manager
stats_manager = ProviderStatsManager()

# Record usage
stats_manager.record_request(
    provider="openai",
    model="gpt-4",
    prompt_tokens=500,
    completion_tokens=200,
    cost=0.01
)

# Get usage report
report = stats_manager.get_usage_report()
print(f"Total cost: ${report['total_cost']}")
print(f"Total tokens: {report['total_tokens']}")
```

## Configuration

You can configure the framework using environment variables:

```bash
# API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GROQ_API_KEY="your-groq-key"

# Logging
export LOG_LEVEL="INFO"
export LOG_FILE="logs/mihrabai.log"

# Runtime configuration
export MAX_RETRIES=3
export RETRY_DELAY=2
export REQUEST_TIMEOUT=30
```

## Documentation

Comprehensive documentation is available in the `src/mihrabai/docs` directory:

- [Quick Start Guide](src/mihrabai/docs/guide/quick_start_guide.md)
- [Using Mihrab with Tools](src/mihrabai/docs/guide/using_mihrab_with_tools.md)
- [API Reference](src/mihrabai/docs/guide/api_reference.md)
- [Creating Custom Tools](src/mihrabai/docs/guide/creating_custom_tools.md)
- [Using Groq Provider](src/mihrabai/docs/guide/using_groq_provider.md)
- [Multi-Agent Systems](src/mihrabai/docs/multi_agent_system.md)
- [Handoff System](src/mihrabai/docs/handoff_system.md)

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/mihrabai.git
cd mihrabai

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/unit/
pytest tests/integration/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all the LLM providers for their amazing models
- Inspired by LangChain, AutoGPT, and other agent frameworks
- Named after the mihrab, the niche in a mosque that indicates the direction of prayer, symbolizing guidance and direction