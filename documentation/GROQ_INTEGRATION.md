# Groq API Integration for Mihrab AI Agent

This document provides information about the Groq API integration with the Mihrab AI Agent framework.

> **Note**: For a more comprehensive guide on using Groq with Mihrab AI Agent, please refer to our [Using Groq Provider](src/mihrabai/docs/guide/using_groq_provider.md) documentation.

## Overview

The Groq API integration allows you to use Groq's powerful language models within the Mihrab AI Agent framework. This integration supports:

- Basic chat completions
- Function calling
- Advanced agent handoffs
- Specialized task agents

## Setup

### Prerequisites

- A Groq API key (sign up at [https://console.groq.com](https://console.groq.com))
- Python 3.8 or higher
- Mihrab AI Agent framework

### Environment Setup

Set your Groq API key as an environment variable:

```bash
# Linux/macOS
export GROQ_API_KEY=your_api_key_here

# Windows (Command Prompt)
set GROQ_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:GROQ_API_KEY="your_api_key_here"
```

Alternatively, you can create a `.env` file in your project root:

```
GROQ_API_KEY=your_api_key_here
```

## Examples

### 1. Simple Groq Test

The `groq_test.py` example demonstrates basic interaction with the Groq API:

```bash
python groq_test.py
```

This example sends simple queries to the Groq API and displays the responses.

### 2. Basic Chat Agent

```bash
python -m src.mihrabai.examples.basic.basic_chat_agent
```

This example creates a basic chat agent using the Groq API.

### 3. Web Tools Example

```bash
python -m src.mihrabai.examples.tools.web_tools_example
```

This example demonstrates how to use web tools with the Groq API.

### 4. Function Calling Example

```bash
python -m src.mihrabai.examples.basic.function_calling
```

This example shows how to use function calling with the Groq API.

### 5. Advanced Handoff Example

```bash
python -m src.mihrabai.examples.advanced.handoff_example
```

This example creates multiple specialized agents and routes queries to the appropriate agent based on content.

## Convenience Runner

For convenience, you can use the `run_groq_example.py` script to run any of the examples:

```bash
python run_groq_example.py [your_api_key]
```

This script will prompt you to choose which example to run and set the API key for you.

## Available Models

The Groq API integration supports the following models:

- llama3-8b-8192
- llama3-70b-8192
- mixtral-8x7b-32768
- gemma-7b-it

## Troubleshooting

### Common Issues

1. **API Key Not Set**: Ensure your Groq API key is correctly set as an environment variable or passed as a parameter.

2. **Unsupported Properties**: The Groq API may not support all properties that other providers support. The integration handles this by cleaning message structures.

3. **Rate Limiting**: If you encounter rate limiting issues, try using a different model or reducing the frequency of requests.

## Implementation Details

The Groq API integration is implemented in the `GroqAdapter` class in `mihrabai/models/providers/groq/groq_provider.py`. This adapter:

1. Handles authentication with the Groq API
2. Cleans message structures to remove unsupported properties
3. Formats requests according to the Groq API specifications
4. Processes responses and extracts relevant information

## Important Implementation Notes

When using Groq with Mihrab AI Agent, keep these key differences in mind:

1. Use `create_task_agent` instead of `create_agent`
2. Pass the Groq API key explicitly in the `provider_kwargs` parameter
3. Pass tools directly in the agent creation function instead of using `add_tool`
4. Implement the `execute` method in custom tools

For more details, see the [Using Groq Provider](src/mihrabai/docs/guide/using_groq_provider.md) guide.

## Contributing

If you encounter issues or have suggestions for improving the Groq API integration, please open an issue or submit a pull request.

## Documentation

For more detailed information about using Groq with Mihrab AI Agent, please refer to:

- [Using Groq Provider](src/mihrabai/docs/guide/using_groq_provider.md) - Comprehensive guide on using Groq
- [Quick Start Guide](src/mihrabai/docs/guide/quick_start_guide.md) - Getting started with Mihrab AI Agent
- [API Reference](src/mihrabai/docs/guide/api_reference.md) - Detailed API reference

## License

This integration is provided under the same license as the Mihrab AI Agent framework.
