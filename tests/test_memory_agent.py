"""
Test script for MihrabAI memory task agent
"""

import asyncio
import os
from mihrabai import Message, MessageRole, create_memory_task_agent
from mihrabai.tools.standard.filesystem import ReadFileTool, WriteFileTool
from mihrabai.tools.standard.search import WebSearchTool
from mihrabai.tools.standard.text_processing import TextSummarizerTool


async def main():
    # Set up API keys from environment variables
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("Using a placeholder value for demonstration purposes")
        os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
    
    # Create a memory-enabled task agent with tools
    agent = await create_memory_task_agent(
        provider_name="openai",
        model_name="gpt-4o",
        system_message=(
            "You are a research assistant that can remember important information "
            "across conversations. You can use tools to search the web, read and write "
            "files, and summarize text."
        ),
        tools=[
            WebSearchTool(),
            ReadFileTool(),
            WriteFileTool(),
            TextSummarizerTool(),
        ],
    )
    
    print("Memory Task Agent Test")
    print("=====================")
    print("This test demonstrates a memory task agent with standard tools.")
    print("Type 'exit' to quit the program.")
    print()
    
    # Interactive conversation
    while True:
        user_input = input("> ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # Process the message
        response = await agent.process_message(
            Message(role=MessageRole.USER, content=user_input)
        )
        
        # Display the response
        print(f"\n{response.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
