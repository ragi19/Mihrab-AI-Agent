# Mihrab AI Agent Examples

This directory contains examples demonstrating how to use the Mihrab AI Agent package for various use cases. The examples are organized into three categories:

1. **Basic Examples**: Simple examples demonstrating core functionality
2. **Tool Examples**: Examples showing how to use various tools
3. **Advanced Examples**: Complex examples demonstrating advanced features

## Prerequisites

Before running these examples, make sure you have:

1. Installed the Mihrab AI Agent package:
   ```bash
   pip install mihrab-ai-agent
   ```

2. Set up your API keys in a `.env` file:
   ```
   GROQ_API_KEY=your_groq_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

   Note: These examples primarily use Groq's LLaMA models. The memory_and_multi_provider example also uses Anthropic's Claude model.

## Basic Examples

### Basic Chat Agent
A simple chat agent without tools.
```bash
python -m src.mihrabai.examples.basic.basic_chat_agent
```

### Streaming Chat
Demonstrates streaming responses from LLM models.
```bash
python -m src.mihrabai.examples.basic.streaming_chat
```

### Function Calling
Shows basic function calling with a calculator tool.
```bash
python -m src.mihrabai.examples.basic.function_calling
```

## Tool Examples

### Web Tools
Demonstrates using web tools (HTTP requests, web scraping).
```bash
python -m src.mihrabai.examples.tools.web_tools_example
```

### Filesystem Tools
Shows how to use filesystem tools (read/write/list files).
```bash
python -m src.mihrabai.examples.tools.filesystem_tools_example
```

### Code Generation Tools
Demonstrates code generation, review, and execution tools.
```bash
python -m src.mihrabai.examples.tools.code_generation_example
```

### Custom Tools
Shows how to create and use custom tools.
```bash
python -m src.mihrabai.examples.tools.custom_tools_example
```

## Advanced Examples

### Multi-Agent Collaboration
Demonstrates how multiple specialized agents can collaborate on complex tasks.
```bash
python -m src.mihrabai.examples.advanced.multi_agent_collaboration
```

### Memory and Multi-Provider
Shows how to use memory tools and multiple LLM providers (Groq and Anthropic).
```bash
python -m src.mihrabai.examples.advanced.memory_and_multi_provider
```

### Handoff System
Demonstrates the handoff system for transferring tasks between agents.
```bash
python -m src.mihrabai.examples.advanced.handoff_example
```

## Customizing Examples

Feel free to modify these examples to suit your needs. Some ideas for customization:

- Change the LLM provider or model (examples currently use Groq's LLaMA models)
- Add or remove tools
- Modify system messages
- Create new custom tools
- Implement different agent collaboration patterns

## Troubleshooting

If you encounter issues running the examples:

1. Make sure your API keys are correctly set up in the `.env` file
2. Check that you have the latest version of the Mihrab AI Agent package
3. Verify that you have all the required dependencies installed
4. Check the Mihrab AI Agent documentation for more information

## Additional Resources

- [Mihrab AI Agent Documentation](https://github.com/yourusername/mihrab-ai-agent/docs)
- [API Reference](https://github.com/yourusername/mihrab-ai-agent/docs/api_reference.md)
- [Creating Custom Tools](https://github.com/yourusername/mihrab-ai-agent/docs/creating_custom_tools.md) 