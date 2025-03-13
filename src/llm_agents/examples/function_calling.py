"""
Example of using function calling capabilities with supported providers
"""

import asyncio
import json
import os
from datetime import datetime

from dotenv import load_dotenv

from llm_agents import Message, MessageRole, create_agent
from llm_agents.models.base import ModelCapability

# Load environment variables from .env file
load_dotenv()


# Define some example functions that the model can call
def get_current_weather(location, unit="celsius"):
    """Get the current weather in a given location"""
    # This is a mock function - in a real application, you would call a weather API
    weather_data = {
        "New York": {"temperature": 22, "condition": "Sunny"},
        "London": {"temperature": 15, "condition": "Cloudy"},
        "Tokyo": {"temperature": 28, "condition": "Rainy"},
        "Sydney": {"temperature": 25, "condition": "Clear"},
    }

    # Default to a generic response if location not found
    location_data = weather_data.get(
        location, {"temperature": 20, "condition": "Unknown"}
    )

    # Convert temperature if needed
    temp = location_data["temperature"]
    if unit.lower() == "fahrenheit":
        temp = (temp * 9 / 5) + 32

    return {
        "location": location,
        "temperature": temp,
        "unit": unit,
        "condition": location_data["condition"],
        "timestamp": datetime.now().isoformat(),
    }


def get_stock_price(symbol):
    """Get the current stock price for a given symbol"""
    # This is a mock function - in a real application, you would call a financial API
    stock_data = {"AAPL": 175.50, "MSFT": 325.75, "GOOGL": 140.25, "AMZN": 180.30}

    price = stock_data.get(symbol, 100.00)  # Default price if symbol not found

    return {
        "symbol": symbol,
        "price": price,
        "currency": "USD",
        "timestamp": datetime.now().isoformat(),
    }


# Define the available functions and their schemas
available_functions = {
    "get_current_weather": {
        "name": "get_current_weather",
        "description": "Get the current weather in a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. New York",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature",
                },
            },
            "required": ["location"],
        },
    },
    "get_stock_price": {
        "name": "get_stock_price",
        "description": "Get the current stock price for a company",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The stock symbol, e.g. AAPL for Apple",
                }
            },
            "required": ["symbol"],
        },
    },
}

# Function mapping for execution
function_map = {
    "get_current_weather": get_current_weather,
    "get_stock_price": get_stock_price,
}


async def run_function_calling_example(provider_name, model_name, user_message):
    """Run a function calling example with the specified provider and model"""
    print(
        f"\n=== Running function calling with {provider_name.upper()} ({model_name}) ==="
    )

    # Create an agent with the specified provider and model
    agent = await create_agent(
        provider_name=provider_name,
        model_name=model_name,
        system_message="You are a helpful AI assistant that can call functions to get real-time information.",
        required_capabilities={
            ModelCapability.FUNCTION_CALLING
        },  # Ensure model supports function calling
    )

    # Create a user message
    message = Message(role=MessageRole.USER, content=user_message)

    # Add the message to conversation history
    agent.add_to_history(message)

    # Process the message with function calling
    print(f"User: {user_message}")

    # Convert available_functions to the format expected by the provider
    tools = [
        {"type": "function", "function": func_def}
        for func_def in available_functions.values()
    ]

    # Generate a response with function calls
    response = await agent.model.generate(agent.conversation_history, tools=tools)

    # Check if the model wants to call a function
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(
            f"Assistant wants to call functions: {[call['name'] for call in response.tool_calls]}"
        )

        # Process each function call
        for tool_call in response.tool_calls:
            function_name = tool_call["name"]
            function_args = json.loads(tool_call["parameters"])

            print(f"Calling function: {function_name} with args: {function_args}")

            # Call the function
            if function_name in function_map:
                function_result = function_map[function_name](**function_args)

                # Add function result to conversation
                agent.add_to_history(
                    Message(
                        role=MessageRole.FUNCTION,
                        content=json.dumps(function_result),
                        name=function_name,
                    )
                )

                print(f"Function result: {json.dumps(function_result, indent=2)}")
            else:
                print(f"Function {function_name} not found")

        # Get final response after function calls
        final_response = await agent.model.generate(agent.conversation_history)
        print(f"Assistant: {final_response.content}")

    else:
        # Model didn't call a function, just return the response
        print(f"Assistant: {response.content}")


async def main():
    """Run function calling examples with different providers"""
    # Example user messages that might trigger function calls
    weather_query = "What's the weather like in Tokyo right now?"
    stock_query = "What's the current stock price of Apple?"

    # Run with different providers that support function calling
    if os.getenv("OPENAI_API_KEY"):
        await run_function_calling_example("openai", "gpt-3.5-turbo", weather_query)
        await run_function_calling_example("openai", "gpt-3.5-turbo", stock_query)

    if os.getenv("ANTHROPIC_API_KEY"):
        # Note: Check if your Anthropic model version supports function calling
        await run_function_calling_example(
            "anthropic", "claude-3-opus-20240229", weather_query
        )

    if os.getenv("GROQ_API_KEY"):
        # Llama models with function calling support
        await run_function_calling_example(
            "groq", "llama-3.1-8b-instant", weather_query
        )

    # If no API keys are available, use a fallback message
    if not any(
        [
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("GROQ_API_KEY"),
        ]
    ):
        print(
            "\nNo API keys found. Please set at least one of the following environment variables:"
        )
        print("- OPENAI_API_KEY")
        print("- ANTHROPIC_API_KEY")
        print("- GROQ_API_KEY")


if __name__ == "__main__":
    asyncio.run(main())
