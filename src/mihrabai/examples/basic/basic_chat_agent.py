"""
Basic Chat Agent Example

This example demonstrates how to create a simple chat agent without tools.
It shows the basic usage of the Mihrab AI Agent package.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

from mihrabai import create_agent, Message, MessageRole
from mihrabai.models.provider_registry import ProviderRegistry

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
    # Create an agent with Groq's LLaMA model, explicitly passing the API key
    agent = await create_agent(
        provider_name="groq",
        model_name="llama3-70b-8192",
        system_message="You are a helpful AI assistant that provides clear and concise answers.",
        provider_kwargs={"api_key": groq_api_key}  # Explicitly pass the API key
    )
    
    # Create a user message
    message = Message(role=MessageRole.USER, content="What are the three laws of robotics?")
    
    # Process the message and get a response
    response = await agent.process_message(message)
    
    # Print the response
    print(f"Agent: {response.content}")
    
    # Continue the conversation
    follow_up = Message(role=MessageRole.USER, content="Who created these laws?")
    
    # Add the previous messages to the history
    agent.add_to_history(message)
    agent.add_to_history(response)
    
    # Process the follow-up message
    response2 = await agent.process_message(follow_up)
    
    # Print the response
    print(f"Agent: {response2.content}")

if __name__ == "__main__":
    asyncio.run(main()) 