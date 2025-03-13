# LLM Agents Framework

A flexible framework for building and deploying LLM-powered agents with multiple provider support.

## Features

- Support for multiple LLM providers (OpenAI, Anthropic, Groq)
- Extensible tool system for custom capabilities
- Comprehensive logging and debugging support
- Built-in profiling and benchmarking tools
- Configurable conversation history management
- Automatic retry handling and error recovery
- Easy configuration management
- Advanced handoff system for multi-agent coordination

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-agents.git
cd llm-agents

# Install in development mode
python install.py
```

## Quick Start

```python
import asyncio
from llm_agents import create_agent, Message, MessageRole

async def main():
    # Create an agent
    agent = await create_agent(
        "openai",
        "gpt-3.5-turbo",
        system_message="You are a helpful AI assistant."
    )
    
    # Send a message
    response = await agent.process_message(
        Message(role=MessageRole.USER, content="Hello!")
    )
    print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

The framework supports flexible configuration through either a config file or environment variables:

```python
from llm_agents import config

# Configure through code
config.set_provider_config("openai", {
    "api_key": "your-api-key",
    "default_model": "gpt-3.5-turbo"
})

# Configure logging
config.set_logging_config({
    "level": "DEBUG",
    "file": "agent.log",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
})
```

Or generate a config file:

```bash
python -m llm_agents.scripts.generate_config --openai-key YOUR_KEY --log-level DEBUG
```

## Logging and Profiling

The framework includes comprehensive logging and profiling capabilities:

```python
from llm_agents.utils.dev_tools import AgentProfiler

# Create a profiler
profiler = AgentProfiler()

# Profile agent performance
metrics = await profiler.profile_agent(agent, messages)

# View metrics
print(f"Average response time: {metrics['avg_request_time']:.2f}s")
print(f"Total tokens used: {metrics['total_tokens']}")
```

Log files contain detailed information about:
- Agent initialization and configuration
- Message processing and token usage
- Tool execution and results
- Error handling and retries
- Performance metrics

## Provider Support

Currently supported LLM providers:
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Groq (LLaMA 3)

Adding a new provider:
1. Create provider class implementing BaseProvider
2. Add model configurations
3. Register provider with ProviderRegistry

## Tool System

Create custom tools by extending BaseTool:

```python
from llm_agents.tools import BaseTool

class CustomTool(BaseTool):
    async def _execute(self, parameters):
        # Tool implementation
        return {"result": "success"}
    
    def _get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string"}
            }
        }
```

## Example Scripts

The following example scripts demonstrate how to use the multi-agent system:

1. **Basic Multi-Agent Example**: A simple example with three agents (Analyzer, Researcher, and Synthesizer) working together to process user queries.
   ```bash
   python examples/multi_agent_example.py
   ```

2. **Tool-Enabled Multi-Agent Example**: Demonstrates how to use standard tools (Calculator and DateTime) with multi-agent systems.
   ```bash
   python examples/multi_agent_tool_example.py
   ```

3. **Custom Tool Multi-Agent Example**: Shows how to create a custom Wikipedia search tool and use it in a multi-agent system.
   ```bash
   python examples/custom_tool_multi_agent_example.py
   ```

4. **Handoff Multi-Agent Example**: Illustrates how to implement handoffs between agents, allowing specialized agents to handle specific types of queries.
   ```bash
   python examples/handoff_multi_agent_example.py
   ```

5. **Advanced Handoff Example**: Demonstrates sophisticated handoff patterns with context preservation and conditional handoffs.
   ```bash
   python examples/advanced_handoff_example.py
   ```

6. **Handoff System Tutorial**: A step-by-step tutorial showing how to build a customer support system using the handoff framework.
   ```bash
   python examples/handoff_system_tutorial.py
   ```

## Advanced Features

### Multi-Provider Model

The framework includes a multi-provider model that can automatically switch between different LLM providers based on availability, cost, or performance requirements. This ensures robustness and flexibility in your agent applications.

### Memory Management

Agents can be equipped with memory capabilities, allowing them to remember past interactions and maintain context across multiple turns of conversation.

### Tool Integration

The framework provides a simple way to integrate tools that agents can use to perform actions, such as making API calls, performing calculations, or accessing external knowledge.

### Handoff System

The handoff system enables seamless coordination between specialized agents, allowing them to transfer control based on user needs or conversation context. Key features include:

- **Specialized Agents**: Create agents that focus on specific domains or tasks
- **Handoff Rules**: Define when and how to transfer control between agents
- **Context Preservation**: Maintain relevant context during handoffs
- **Handoff Patterns**: Support for direct, chain, and hub-and-spoke handoff patterns
- **Custom Filters**: Filter conversation history during handoffs
- **Conditional Handoffs**: Define complex conditions for when to hand off
- **Expertise-Based Routing**: Route queries to agents based on their expertise

For more information, see the [handoff system documentation](documentation/handoff_system_readme.md).

## Building Multi-Agent Systems

The framework makes it easy to build sophisticated multi-agent systems:

```python
from llm_agents.handoff import HandoffAgent, HandoffConfig

# Create specialized agents
technical_agent = HandoffAgent(
    name="Technical",
    system_prompt="You are a technical support specialist...",
    expertise=["programming", "troubleshooting"]
)

billing_agent = HandoffAgent(
    name="Billing",
    system_prompt="You are a billing support specialist...",
    expertise=["payments", "subscriptions"]
)

# Create general agent with handoffs
general_agent = HandoffAgent(
    name="General",
    system_prompt="You are a general support agent...",
    handoffs=[
        HandoffConfig(
            name="technical_handoff",
            description="Handoff for technical issues",
            target_agent=technical_agent,
            condition=technical_issue_condition
        ),
        HandoffConfig(
            name="billing_handoff",
            description="Handoff for billing issues",
            target_agent=billing_agent,
            condition=billing_issue_condition
        )
    ]
)

# Process a query
response = await general_agent.process("I need help with my account")
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest -m unit
pytest -m integration
```

### Code Quality

Use the provided script to check code quality:

```bash
python scripts/format_and_check.py
```

This will:
1. Format code with black
2. Sort imports with isort
3. Run flake8 linting
4. Run mypy type checking
5. Run tests with coverage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run code quality checks
5. Submit a pull request

## License

MIT License - see LICENSE file for details 