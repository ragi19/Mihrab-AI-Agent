"""
Custom Tools Example

This example demonstrates how to create and use custom tools with Mihrab AI Agent.
It shows how to create a weather tool and a translation tool.
"""

import asyncio
import os
import json
import requests
from typing import Dict, Any
from dotenv import load_dotenv

from mihrabai import create_agent, Message, MessageRole
from mihrabai.models.base import ModelCapability
from mihrabai.tools.base import BaseTool
from mihrabai.core.types import JSON
from mihrabai.tools.registry import ToolRegistry

# Load environment variables
load_dotenv()

# Define a custom weather tool
class WeatherTool(BaseTool):
    def __init__(self, api_key=None):
        super().__init__(
            name="weather",
            description="Gets the current weather for a specified location"
        )
        # In a real application, you would use a real API key
        self.api_key = api_key or "demo_api_key"
    
    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        location = parameters.get("location")
        
        # In a real application, you would make a real API call
        # This is a mock implementation
        weather_data = {
            "location": location,
            "temperature": 72,
            "condition": "Sunny",
            "humidity": 45,
            "wind_speed": 10,
            "units": "imperial"
        }
        
        return weather_data
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get weather for (city name or zip code)"
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "description": "The units to use for temperature (metric or imperial)",
                    "default": "imperial"
                }
            },
            "required": ["location"]
        }

# Define a custom translation tool
class TranslationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="translate",
            description="Translates text from one language to another"
        )
    
    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        text = parameters.get("text")
        source_language = parameters.get("source_language", "auto")
        target_language = parameters.get("target_language")
        
        # In a real application, you would make a real API call
        # This is a mock implementation
        translations = {
            "en-es": {
                "hello": "hola",
                "goodbye": "adiós",
                "thank you": "gracias"
            },
            "en-fr": {
                "hello": "bonjour",
                "goodbye": "au revoir",
                "thank you": "merci"
            },
            "en-de": {
                "hello": "hallo",
                "goodbye": "auf wiedersehen",
                "thank you": "danke"
            }
        }
        
        # Simple mock translation
        translation_key = f"{source_language}-{target_language}"
        if translation_key in translations and text.lower() in translations[translation_key]:
            translated_text = translations[translation_key][text.lower()]
        else:
            translated_text = f"[Translation of '{text}' to {target_language}]"
        
        return {
            "original_text": text,
            "translated_text": translated_text,
            "source_language": source_language,
            "target_language": target_language
        }
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to translate"
                },
                "source_language": {
                    "type": "string",
                    "description": "The source language code (e.g., 'en' for English, 'auto' for auto-detect)",
                    "default": "auto"
                },
                "target_language": {
                    "type": "string",
                    "description": "The target language code (e.g., 'es' for Spanish)"
                }
            },
            "required": ["text", "target_language"]
        }

async def main():
    # Register custom tools
    ToolRegistry.register("weather", WeatherTool)
    ToolRegistry.register("translate", TranslationTool)
    
    # Create tool instances
    weather_tool = ToolRegistry.create_tool("weather")
    translate_tool = ToolRegistry.create_tool("translate")
    
    # Create an agent with function calling capability
    agent = await create_agent(
        provider_name="groq",
        model_name="llama3-70b-8192",
        system_message="""You are a helpful AI assistant that can provide weather information and translate text.
        Use the weather tool to get current weather conditions and the translation tool to translate text between languages.""",
        required_capabilities={ModelCapability.FUNCTION_CALLING},
    )
    
    # Add tools to the agent
    agent.add_tool(weather_tool)
    agent.add_tool(translate_tool)
    
    # Example 1: Get weather information
    weather_message = Message(
        role=MessageRole.USER,
        content="What's the current weather in San Francisco?"
    )
    
    print("Processing weather request...")
    weather_response = await agent.process_message(weather_message)
    print(f"Agent: {weather_response.content}\n")
    
    # Example 2: Translate text
    translate_message = Message(
        role=MessageRole.USER,
        content="Translate 'hello' to Spanish."
    )
    
    # Clear history to start fresh
    agent.clear_history()
    
    print("Processing translation request...")
    translate_response = await agent.process_message(translate_message)
    print(f"Agent: {translate_response.content}\n")
    
    # Example 3: Combined request
    combined_message = Message(
        role=MessageRole.USER,
        content="What's the weather in Paris and how do you say 'thank you' in French?"
    )
    
    # Clear history to start fresh
    agent.clear_history()
    
    print("Processing combined request...")
    combined_response = await agent.process_message(combined_message)
    print(f"Agent: {combined_response.content}")

if __name__ == "__main__":
    asyncio.run(main()) 