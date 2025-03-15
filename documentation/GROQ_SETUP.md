# Setting Up Groq API for Mihrab AI Agent

This guide provides instructions on how to set up and use the Groq API with the Mihrab AI Agent framework.

> **Note**: For a more comprehensive guide on using Groq with Mihrab AI Agent, please refer to our [Using Groq Provider](src/mihrabai/docs/guide/using_groq_provider.md) documentation.

## Prerequisites

1. A Groq API key - Sign up at [https://console.groq.com](https://console.groq.com) to get your API key
2. Python 3.8 or higher
3. Mihrab AI Agent framework

## Setting Up Your Groq API Key

There are several ways to provide your Groq API key to the Mihrab AI Agent framework:

### 1. Environment Variable (Recommended)

Set the `GROQ_API_KEY` environment variable:

```bash
# Linux/macOS
export GROQ_API_KEY=your_api_key_here

# Windows (Command Prompt)
set GROQ_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:GROQ_API_KEY="your_api_key_here"
```

### 2. .env File

Create a `.env` file in your project root with the following content:

```
GROQ_API_KEY=your_api_key_here
```

### 3. Explicit Parameter

Pass the API key explicitly when creating an agent:

```python
agent = await create_agent(
    provider_name="groq",
    model_name="llama3-70b-8192",
    provider_kwargs={"api_key": "your_api_key_here"}
)
```

## Testing Your Groq API Key

You can test your Groq API key using the provided `groq_test.py` script:

```bash
python groq_test.py your_api_key_here
```

Or if you've set the environment variable:

```bash
python groq_test.py
```

## Running the Examples

Once you have set up your Groq API key, you can run the examples:

```bash
# Basic chat agent example
python -m src.mihrabai.examples.basic.basic_chat_agent

# Web tools example
python -m src.mihrabai.examples.tools.web_tools_example

# Function calling example
python -m src.mihrabai.examples.basic.function_calling

# Advanced handoff example
python -m src.mihrabai.examples.advanced.handoff_example
```

> **Note**: When using Groq, make sure to use `create_task_agent` instead of `create_agent` and pass tools directly in the agent creation function. See the [Using Groq Provider](src/mihrabai/docs/guide/using_groq_provider.md) guide for details.

## Troubleshooting

### Invalid API Key

If you see an error like:

```
Error code: 401 - {'error': {'message': 'Invalid API Key', 'type': 'invalid_request_error', 'code': 'invalid_api_key'}}
```

This means your API key is not valid or not properly set. Make sure:

1. You have copied the correct API key from the Groq console
2. The API key is properly set in the environment variable or passed as a parameter
3. There are no extra spaces or characters in your API key

### Rate Limiting

If you encounter rate limiting issues, try:

1. Using a different model
2. Reducing the frequency of requests
3. Checking your Groq account for usage limits

## Available Groq Models

The Mihrab AI Agent framework supports the following Groq models:

- llama3-70b-8192
- llama3-8b-8192
- mixtral-8x7b-32768
- gemma-7b-it
- And many more...

## Further Help

If you continue to experience issues, please:

1. Check the Groq documentation at [https://console.groq.com/docs](https://console.groq.com/docs)
2. Refer to the [Mihrab AI Agent documentation](src/mihrabai/docs/guide/index.md), especially the [Using Groq Provider](src/mihrabai/docs/guide/using_groq_provider.md) guide
3. Contact support for further assistance 