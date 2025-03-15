"""
Filesystem Tools Example

This example demonstrates how to use filesystem tools (read, write, and list files)
with Mihrab AI Agent to interact with the local filesystem.
"""

import asyncio
import os
import tempfile
from dotenv import load_dotenv

from mihrabai import create_agent, Message, MessageRole
from mihrabai.models.base import ModelCapability
from mihrabai.tools.registry import ToolRegistry
from mihrabai.tools.standard.filesystem import FileReadTool, FileWriteTool, FileListTool

# Load environment variables
load_dotenv()

async def main():
    # Create a temporary directory for our example
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Create some sample files
        with open(os.path.join(temp_dir, "sample1.txt"), "w") as f:
            f.write("This is a sample text file.\nIt contains some example content.\n")
        
        with open(os.path.join(temp_dir, "sample2.txt"), "w") as f:
            f.write("This is another sample file with different content.\n")
        
        # Register filesystem tools
        ToolRegistry.register("file_read", FileReadTool)
        ToolRegistry.register("file_write", FileWriteTool)
        ToolRegistry.register("file_list", FileListTool)
        
        # Create tool instances with the temporary directory as the base path
        file_read_tool = ToolRegistry.create_tool("file_read", base_path=temp_dir)
        file_write_tool = ToolRegistry.create_tool("file_write", base_path=temp_dir)
        file_list_tool = ToolRegistry.create_tool("file_list", base_path=temp_dir)
        
        # Create an agent with function calling capability
        agent = await create_agent(
            provider_name="groq",
            model_name="llama3-70b-8192",
            system_message="""You are a helpful AI assistant that can interact with the filesystem.
            You can read, write, and list files in the specified directory.
            Always be careful when modifying files and confirm your actions.""",
            required_capabilities={ModelCapability.FUNCTION_CALLING},
        )
        
        # Add tools to the agent
        agent.add_tool(file_read_tool)
        agent.add_tool(file_write_tool)
        agent.add_tool(file_list_tool)
        
        # Example 1: List files in the directory
        list_message = Message(
            role=MessageRole.USER,
            content=f"List all files in the directory."
        )
        
        print("Processing list files request...")
        list_response = await agent.process_message(list_message)
        print(f"Agent: {list_response.content}\n")
        
        # Example 2: Read a file
        read_message = Message(
            role=MessageRole.USER,
            content="Read the contents of sample1.txt"
        )
        
        # Add previous messages to history
        agent.add_to_history(list_message)
        agent.add_to_history(list_response)
        
        print("Processing read file request...")
        read_response = await agent.process_message(read_message)
        print(f"Agent: {read_response.content}\n")
        
        # Example 3: Write to a file
        write_message = Message(
            role=MessageRole.USER,
            content="Create a new file called notes.txt with a list of 3 important tasks"
        )
        
        # Clear history to start fresh
        agent.clear_history()
        
        print("Processing write file request...")
        write_response = await agent.process_message(write_message)
        print(f"Agent: {write_response.content}\n")
        
        # Example 4: Read the newly created file
        verify_message = Message(
            role=MessageRole.USER,
            content="Now read the contents of notes.txt to verify it was created correctly"
        )
        
        # Add previous messages to history
        agent.add_to_history(write_message)
        agent.add_to_history(write_response)
        
        print("Processing verification request...")
        verify_response = await agent.process_message(verify_message)
        print(f"Agent: {verify_response.content}")

if __name__ == "__main__":
    asyncio.run(main()) 