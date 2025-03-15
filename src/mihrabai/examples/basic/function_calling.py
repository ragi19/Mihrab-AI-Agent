"""
Function Calling Example

This example demonstrates how to use function calling with LLMs in Mihrab AI Agent.
It shows how to create a simple calculator tool and use it with an agent.
"""

import asyncio
import os
import sys
from typing import Dict, Any
from dotenv import load_dotenv

from mihrabai import create_task_agent, Message, MessageRole
from mihrabai.models.base import ModelCapability
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

# Define a simple calculator tool
class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Performs basic arithmetic operations"
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> JSON:
        """Execute the calculator tool with the given parameters"""
        return await self._execute(parameters)
    
    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        operation = parameters.get("operation")
        a = parameters.get("a", 0)
        b = parameters.get("b", 0)
        
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return {"error": "Cannot divide by zero"}
            result = a / b
        else:
            return {"error": f"Unknown operation: {operation}"}
        
        return {"result": result}
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the parameters schema for the calculator tool"""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "The first operand"
                },
                "b": {
                    "type": "number",
                    "description": "The second operand"
                }
            },
            "required": ["operation", "a", "b"]
        }

async def main():
    # Create a calculator tool instance
    calculator_tool = CalculatorTool()
    
    # Create a task agent with the calculator tool
    agent = await create_task_agent(
        provider_name="groq",
        model_name="llama3-70b-8192",
        system_message="You are a helpful AI assistant that can perform calculations.",
        provider_kwargs={"api_key": groq_api_key},  # Explicitly pass the API key
        tools=[calculator_tool]  # Pass tools directly
    )
    
    # Create a user message that requires calculation
    message = Message(
        role=MessageRole.USER,
        content="What is 25 multiplied by 16?"
    )
    
    # Process the message
    response = await agent.process_message(message)
    
    # Print the response
    print(f"Agent: {response.content}")
    
    # Continue the conversation with a more complex calculation
    follow_up = Message(
        role=MessageRole.USER,
        content="If I have 3 apples and you give me 5 more, then I eat 2, how many do I have left?"
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