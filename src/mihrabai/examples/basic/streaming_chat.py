"""
Streaming Chat Example

This example demonstrates how to create a chat agent that streams responses
from LLM models, providing a more interactive user experience.
"""

import asyncio
import os
from dotenv import load_dotenv

from mihrabai import create_agent, Message, MessageRole
from mihrabai.models.base import ModelCapability

# Load environment variables
load_dotenv()

async def main():
    # Create an agent with streaming capability
    agent = await create_agent(
        provider_name="groq",
        model_name="llama3-70b-8192",
        system_message="You are a helpful AI assistant that provides detailed explanations.",
        required_capabilities={ModelCapability.STREAMING},
    )
    
    # Create a user message
    message = Message(
        role=MessageRole.USER,
        content="Explain quantum computing in simple terms and give three potential applications."
    )
    
    # Define a callback function to handle streamed chunks
    async def stream_callback(chunk: str):
        print(chunk, end="", flush=True)
    
    # Process the message with streaming
    print("Agent: ", end="", flush=True)
    response = await agent.process_message(message, stream_callback=stream_callback)
    print("\n")  # Add a newline after the streaming is complete
    
    # The complete response is also available
    print(f"Complete response length: {len(response.content)} characters")

if __name__ == "__main__":
    asyncio.run(main()) 