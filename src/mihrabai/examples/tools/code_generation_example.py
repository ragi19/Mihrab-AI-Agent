"""
Code Generation Example

This example demonstrates how to use code generation, review, and execution tools
with Mihrab AI Agent to generate, review, and execute code.
"""

import asyncio
import os
import tempfile
from dotenv import load_dotenv

from mihrabai import create_agent, Message, MessageRole
from mihrabai.models.base import ModelCapability
from mihrabai.tools.registry import ToolRegistry
from mihrabai.tools.standard.code_generation import CodeGeneratorTool, CodeReviewTool, CodeExecutorTool

# Load environment variables
load_dotenv()

async def main():
    # Create a temporary directory for our example
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Register code tools
        ToolRegistry.register("code_generator", CodeGeneratorTool)
        ToolRegistry.register("code_reviewer", CodeReviewTool)
        ToolRegistry.register("code_executor", CodeExecutorTool)
        
        # Create tool instances with the temporary directory as the base path
        code_generator_tool = ToolRegistry.create_tool("code_generator", base_path=temp_dir)
        code_reviewer_tool = ToolRegistry.create_tool("code_reviewer")
        code_executor_tool = ToolRegistry.create_tool("code_executor", base_path=temp_dir)
        
        # Create an agent with function calling capability
        agent = await create_agent(
            provider_name="groq",
            model_name="llama3-70b-8192",
            system_message="""You are a helpful AI assistant that can generate, review, and execute code.
            You can generate code in various programming languages, review code for errors and improvements,
            and execute code to see the results. Always explain your code and the results.""",
            required_capabilities={ModelCapability.FUNCTION_CALLING},
        )
        
        # Add tools to the agent
        agent.add_tool(code_generator_tool)
        agent.add_tool(code_reviewer_tool)
        agent.add_tool(code_executor_tool)
        
        # Example 1: Generate a simple Python script
        generate_message = Message(
            role=MessageRole.USER,
            content="Generate a Python script that calculates the Fibonacci sequence up to the 10th number."
        )
        
        print("Processing code generation request...")
        generate_response = await agent.process_message(generate_message)
        print(f"Agent: {generate_response.content}\n")
        
        # Example 2: Review the generated code
        review_message = Message(
            role=MessageRole.USER,
            content="Review the code you just generated for any improvements or optimizations."
        )
        
        # Add previous messages to history
        agent.add_to_history(generate_message)
        agent.add_to_history(generate_response)
        
        print("Processing code review request...")
        review_response = await agent.process_message(review_message)
        print(f"Agent: {review_response.content}\n")
        
        # Example 3: Generate an improved version based on the review
        improve_message = Message(
            role=MessageRole.USER,
            content="Generate an improved version of the Fibonacci script based on your review."
        )
        
        # Add previous messages to history
        agent.add_to_history(review_message)
        agent.add_to_history(review_response)
        
        print("Processing code improvement request...")
        improve_response = await agent.process_message(improve_message)
        print(f"Agent: {improve_response.content}\n")
        
        # Example 4: Execute the improved code
        execute_message = Message(
            role=MessageRole.USER,
            content="Execute the improved Fibonacci script and show me the results."
        )
        
        # Add previous messages to history
        agent.add_to_history(improve_message)
        agent.add_to_history(improve_response)
        
        print("Processing code execution request...")
        execute_response = await agent.process_message(execute_message)
        print(f"Agent: {execute_response.content}")

if __name__ == "__main__":
    asyncio.run(main()) 