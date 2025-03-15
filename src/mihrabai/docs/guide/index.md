# Mihrab AI Agent Documentation

Welcome to the Mihrab AI Agent documentation. This documentation provides comprehensive information on how to use the Mihrab AI Agent package to create powerful AI agents with various tools and capabilities.

## Getting Started

- [Quick Start Guide](quick_start_guide.md) - Get up and running with Mihrab AI Agent in minutes
- [Using Mihrab with Tools](using_mihrab_with_tools.md) - Comprehensive guide on using Mihrab AI Agent with tools

## Core Concepts

- [API Reference](api_reference.md) - Detailed reference of the Mihrab AI Agent API
- [Creating Custom Tools](creating_custom_tools.md) - Guide on creating custom tools for Mihrab AI Agent
- [Using Groq Provider](using_groq_provider.md) - Guide on using Groq models with Mihrab AI Agent

## Advanced Topics

- [Multi-Agent Systems](../multi_agent_system.md) - Creating teams of specialized agents
- [Handoff System](../handoff_system.md) - Transferring tasks between agents
- [Advanced Handoff System](../advanced_handoff_system.md) - Advanced handoff mechanisms
- [Memory and Multi-Provider](memory_and_multi_provider.md) - Working with memory and multiple providers
- [Tracing Documentation](tracing_documentation.md) - Tracing and debugging agent interactions

## Examples

The Mihrab AI Agent package includes a variety of examples that demonstrate how to use the package with different tools and capabilities. You can find these examples in the `mihrabai/examples` directory:

- **Basic Examples**
  - `basic_chat_agent.py` - Simple chat agent without tools
  - `streaming_chat.py` - Streaming responses from LLM models
  - `function_calling.py` - Basic function calling with LLMs

- **Tool Examples**
  - `web_tools_example.py` - Using web tools (HTTP requests, web scraping)
  - `filesystem_tools_example.py` - Using filesystem tools (read/write/list files)
  - `code_generation_example.py` - Code generation, review, and execution

- **Advanced Examples**
  - `multi_agent_collaboration.py` - Multiple specialized agents collaborating on tasks
  - `memory_task_agent_example.py` - Using memory tools for persistent agent memory
  - `advanced_memory_agent.py` - Advanced memory capabilities
  - `vision_capabilities.py` - Using vision capabilities with multimodal models
  - `handoff_example.py` - Agent-to-agent handoff for specialized tasks
  - `error_recovery.py` - Recovering from errors during task execution
  - `benchmarking.py` - Benchmarking agent performance
  - `coordinated_agents.py` - Coordinating multiple agents for complex workflows

## Additional Resources

- [GitHub Repository](https://github.com/yourusername/mihrab-ai-agent) - Source code and issues
- [PyPI Package](https://pypi.org/project/mihrab-ai-agent/) - Package on PyPI
- [Contributing Guide](../../CONTRIBUTING.md) - How to contribute to the project
- [License](../../LICENSE) - License information 