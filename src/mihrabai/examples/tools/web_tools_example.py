"""
Web Tools Example

This example demonstrates how to use web tools (HTTP requests, web scraping)
with Mihrab AI Agent to interact with the internet.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

from mihrabai import create_task_agent, Message, MessageRole
from mihrabai.models.base import ModelCapability
from mihrabai.tools.registry import ToolRegistry
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
    # Register web tools
    ToolRegistry.register("http_request", HTTPRequestTool)
    ToolRegistry.register("web_scraper", WebScraperTool)
    
    # Create tool instances
    http_tool = ToolRegistry.create_tool("http_request")
    scraper_tool = ToolRegistry.create_tool("web_scraper")
    
    # Create a task agent with tools
    agent = await create_task_agent(
        provider_name="groq",
        model_name="llama3-70b-8192",
        system_message="""You are a helpful AI assistant that can search the web for information.
        Use the HTTP request tool to make API calls and the web scraper tool to extract information from websites.
        Always cite your sources when providing information from the web.""",
        provider_kwargs={"api_key": groq_api_key},  # Explicitly pass the API key
        tools=[http_tool, scraper_tool]  # Pass tools directly
    )
    
    # Create a user message that requires web search
    message = Message(
        role=MessageRole.USER,
        content="What is the current weather in New York City?"
    )
    
    # Process the message
    response = await agent.process_message(message)
    
    # Print the response
    print(f"Agent: {response.content}")
    
    # Continue the conversation with a follow-up
    follow_up = Message(
        role=MessageRole.USER,
        content="Can you find the top news headlines today?"
    )
    
    # Add the previous messages to the history
    agent.add_to_history(message)
    agent.add_to_history(response)
    
    # Process the follow-up message
    response2 = await agent.process_message(follow_up)
    
    # Print the response
    print(f"Agent: {response2.content}")

if __name__ == "__main__":
    asyncio.run(main()) 