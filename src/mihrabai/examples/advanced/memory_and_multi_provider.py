"""
Memory and Multi-Provider Example

This example demonstrates how to use memory tools for persistent agent memory
and how to work with multiple LLM providers in the same application.
"""

import asyncio
import os
import json
import sys
import tempfile
from typing import Dict, Any, List
from dotenv import load_dotenv

from mihrabai import create_task_agent, Message, MessageRole
from mihrabai.models.base import ModelCapability
from mihrabai.tools.registry import ToolRegistry
from mihrabai.tools.base import BaseTool
from mihrabai.core.types import JSON

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

# Define a simple memory tool
class MemoryTool(BaseTool):
    def __init__(self, memory_file: str):
        super().__init__(
            name="memory",
            description="Store and retrieve information from memory"
        )
        self.memory_file = memory_file
        
        # Initialize memory file if it doesn't exist
        if not os.path.exists(memory_file):
            with open(memory_file, "w") as f:
                json.dump({}, f)
    
    async def execute(self, parameters: Dict[str, Any]) -> JSON:
        """Execute the memory tool with the given parameters"""
        return await self._execute(parameters)
    
    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        operation = parameters.get("operation")
        key = parameters.get("key")
        value = parameters.get("value", None)
        
        # Load current memory
        with open(self.memory_file, "r") as f:
            memory = json.load(f)
        
        result = None
        if operation == "store":
            memory[key] = value
            result = {"status": "success", "message": f"Stored '{value}' under key '{key}'"}
        elif operation == "retrieve":
            if key in memory:
                result = {"status": "success", "value": memory[key]}
            else:
                result = {"status": "error", "message": f"Key '{key}' not found in memory"}
        elif operation == "list":
            result = {"status": "success", "keys": list(memory.keys())}
        elif operation == "delete":
            if key in memory:
                del memory[key]
                result = {"status": "success", "message": f"Deleted key '{key}' from memory"}
            else:
                result = {"status": "error", "message": f"Key '{key}' not found in memory"}
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Save updated memory
        with open(self.memory_file, "w") as f:
            json.dump(memory, f)
        
        return result
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["store", "retrieve", "list", "delete"],
                    "description": "The operation to perform on memory"
                },
                "key": {
                    "type": "string",
                    "description": "The key to store, retrieve, or delete"
                },
                "value": {
                    "type": "string",
                    "description": "The value to store (only used with 'store' operation)"
                }
            },
            "required": ["operation", "key"]
        }

async def main():
    # Create a temporary directory for our example
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Create a memory file
        memory_file = os.path.join(temp_dir, "agent_memory.json")
        
        # Create a memory tool
        memory_tool = MemoryTool(memory_file)
        
        # Create agents with different providers
        
        # 1. Groq Agent (using TaskAgent)
        groq_agent = await create_task_agent(
            provider_name="groq",
            model_name="llama3-70b-8192",
            system_message="""You are a helpful AI assistant with memory capabilities.
            You can store information in your memory and retrieve it later.
            Use the memory tool to store important information from conversations.""",
            provider_kwargs={"api_key": groq_api_key},  # Explicitly pass the API key
            tools=[memory_tool]  # Pass tools directly
        )
        
        # 2. Anthropic Agent (if available)
        try:
            anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not anthropic_api_key:
                raise ValueError("Anthropic API key not found")
                
            anthropic_agent = await create_task_agent(
                provider_name="anthropic",
                model_name="claude-3-opus-20240229",
                system_message="""You are a helpful AI assistant with memory capabilities.
                You can store information in your memory and retrieve it later.
                Use the memory tool to store important information from conversations.""",
                provider_kwargs={"api_key": anthropic_api_key},
                tools=[memory_tool]
            )
            has_anthropic = True
        except Exception as e:
            print(f"Could not initialize Anthropic agent: {e}")
            print("Continuing with only Groq agent...")
            anthropic_agent = groq_agent  # Use Groq agent as fallback
            has_anthropic = False
        
        # Example 1: Store information with Groq agent
        store_message = Message(
            role=MessageRole.USER,
            content="My name is Alice and my favorite color is blue. Please remember this information."
        )
        
        print("Groq Agent - Storing information...")
        store_response = await groq_agent.process_message(store_message)
        print(f"Groq Agent: {store_response.content}\n")
        
        # Example 2: Retrieve information with second agent
        retrieve_message = Message(
            role=MessageRole.USER,
            content="What's my name and favorite color?"
        )
        
        agent_name = "Anthropic" if has_anthropic else "Groq (second instance)"
        print(f"{agent_name} Agent - Retrieving information...")
        retrieve_response = await anthropic_agent.process_message(retrieve_message)
        print(f"{agent_name} Agent: {retrieve_response.content}\n")
        
        # Example 3: Continue conversation with Groq agent
        follow_up_message = Message(
            role=MessageRole.USER,
            content="I also like hiking and reading science fiction books. Remember this too."
        )
        
        # Add previous messages to history
        groq_agent.add_to_history(store_message)
        groq_agent.add_to_history(store_response)
        
        print("Groq Agent - Storing additional information...")
        follow_up_response = await groq_agent.process_message(follow_up_message)
        print(f"Groq Agent: {follow_up_response.content}\n")
        
        # Example 4: Retrieve all information with second agent
        retrieve_all_message = Message(
            role=MessageRole.USER,
            content="Tell me everything you know about me."
        )
        
        print(f"{agent_name} Agent - Retrieving all information...")
        retrieve_all_response = await anthropic_agent.process_message(retrieve_all_message)
        print(f"{agent_name} Agent: {retrieve_all_response.content}\n")
        
        # Example 5: List all memory keys
        list_message = Message(
            role=MessageRole.USER,
            content="List all the information you have stored in your memory."
        )
        
        print("Groq Agent - Listing memory keys...")
        list_response = await groq_agent.process_message(list_message)
        print(f"Groq Agent: {list_response.content}")

if __name__ == "__main__":
    asyncio.run(main()) 