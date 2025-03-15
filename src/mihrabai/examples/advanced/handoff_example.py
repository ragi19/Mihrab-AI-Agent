"""
Handoff System Example

This example demonstrates how to use the handoff system to transfer tasks between agents.
It shows how to create a system where different specialized agents can hand off tasks
to each other based on their expertise.
"""

import asyncio
import os
import sys
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from mihrabai import create_task_agent, Message, MessageRole
from mihrabai.models.base import ModelCapability
from mihrabai.tools.registry import ToolRegistry
from mihrabai.tools.standard.web import HTTPRequestTool, WebScraperTool
from mihrabai.tools.standard.code_generation import CodeGenerationTool
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

# Define a wrapper for CodeGenerationTool that implements the missing execute method
class CodeGenerationToolWrapper(CodeGenerationTool):
    """Wrapper for CodeGenerationTool that implements the missing execute method"""
    
    async def execute(self, parameters: Dict[str, Any]) -> JSON:
        """Execute the code generation tool with the given parameters"""
        return await self._execute(parameters)

# Define a handoff manager class
class HandoffManager:
    def __init__(self):
        self.agents = {}
        self.conversation_history = []
    
    def register_agent(self, name: str, agent, expertise: List[str]):
        """Register an agent with the handoff manager."""
        self.agents[name] = {
            "agent": agent,
            "expertise": expertise
        }
    
    def add_to_history(self, message: Message):
        """Add a message to the conversation history."""
        self.conversation_history.append(message)
    
    def get_history(self) -> List[Message]:
        """Get the conversation history."""
        return self.conversation_history
    
    def find_agent_for_task(self, task: str) -> Optional[str]:
        """Find the most appropriate agent for a task based on expertise."""
        # In a real system, this would use more sophisticated matching
        for name, info in self.agents.items():
            for expertise in info["expertise"]:
                if expertise.lower() in task.lower():
                    return name
        return None

# Define a handoff tool
class HandoffTool(BaseTool):
    def __init__(self, handoff_manager: HandoffManager, current_agent_name: str):
        super().__init__(
            name="handoff",
            description="Hand off a task to another agent with more appropriate expertise"
        )
        self.handoff_manager = handoff_manager
        self.current_agent_name = current_agent_name
    
    async def execute(self, parameters: Dict[str, Any]) -> JSON:
        """Execute the handoff tool with the given parameters"""
        return await self._execute(parameters)
    
    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        task = parameters.get("task")
        target_agent = parameters.get("target_agent", None)
        
        # If no target agent is specified, find the most appropriate one
        if not target_agent:
            target_agent = self.handoff_manager.find_agent_for_task(task)
        
        # If no appropriate agent is found, return an error
        if not target_agent or target_agent not in self.handoff_manager.agents:
            return {
                "status": "error",
                "message": f"No appropriate agent found for task: {task}"
            }
        
        # Get the target agent
        agent_info = self.handoff_manager.agents[target_agent]
        agent = agent_info["agent"]
        
        # Create a message for the target agent
        message = Message(
            role=MessageRole.USER,
            content=f"[Handoff from {self.current_agent_name}] {task}"
        )
        
        # Add the message to the conversation history
        self.handoff_manager.add_to_history(message)
        
        # Process the message with the target agent
        response = await agent.process_message(message)
        
        # Add the response to the conversation history
        self.handoff_manager.add_to_history(response)
        
        return {
            "status": "success",
            "task": task,
            "target_agent": target_agent,
            "response": response.content
        }
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to hand off to another agent"
                },
                "target_agent": {
                    "type": "string",
                    "description": "The name of the target agent (optional, will be determined automatically if not provided)"
                }
            },
            "required": ["task"]
        }

async def main():
    # Create a handoff manager
    handoff_manager = HandoffManager()
    
    # Create tool instances
    http_tool = HTTPRequestTool()
    scraper_tool = WebScraperTool()
    code_generator_tool = CodeGenerationToolWrapper()
    
    # Create handoff tools (we'll create these first and then pass them to the agents)
    general_handoff_tool = HandoffTool(handoff_manager, "general_agent")
    research_handoff_tool = HandoffTool(handoff_manager, "research_agent")
    code_handoff_tool = HandoffTool(handoff_manager, "code_agent")
    creative_handoff_tool = HandoffTool(handoff_manager, "creative_agent")
    
    # Create specialized agents
    
    # 1. General Assistant - Handles general queries and routes to specialists
    general_agent = await create_task_agent(
        provider_name="groq",
        model_name="llama3-70b-8192",
        system_message="""You are a general assistant that can handle a wide range of tasks.
        For specialized tasks, you can hand off to other agents with more appropriate expertise.
        Available specialists:
        - research_agent: Expert in finding and summarizing information
        - code_agent: Expert in writing and debugging code
        - creative_agent: Expert in creative writing and content generation""",
        provider_kwargs={"api_key": groq_api_key},  # Explicitly pass the API key
        tools=[general_handoff_tool]  # Pass tools directly
    )
    
    # 2. Research Agent - Specializes in finding information
    research_agent = await create_task_agent(
        provider_name="groq",
        model_name="llama3-70b-8192",
        system_message="""You are a specialized research agent that excels at finding information.
        Your task is to search for relevant information on topics and provide comprehensive research results.
        Focus on finding factual information from reliable sources. Include citations when possible.""",
        provider_kwargs={"api_key": groq_api_key},  # Explicitly pass the API key
        tools=[http_tool, scraper_tool, research_handoff_tool]  # Pass tools directly
    )
    
    # 3. Code Agent - Specializes in writing and debugging code
    code_agent = await create_task_agent(
        provider_name="groq",
        model_name="llama3-70b-8192",
        system_message="""You are a specialized code agent that excels at writing and debugging code.
        Your task is to write clean, efficient, and well-documented code based on requirements.
        You can also debug existing code and suggest improvements.""",
        provider_kwargs={"api_key": groq_api_key},  # Explicitly pass the API key
        tools=[code_generator_tool, code_handoff_tool]  # Pass tools directly
    )
    
    # 4. Creative Agent - Specializes in creative writing
    creative_agent = await create_task_agent(
        provider_name="groq",
        model_name="llama3-70b-8192",
        system_message="""You are a specialized creative agent that excels at creative writing and content generation.
        Your task is to generate engaging, original content based on prompts.
        You can write stories, poems, marketing copy, and other creative content.""",
        provider_kwargs={"api_key": groq_api_key},  # Explicitly pass the API key
        tools=[creative_handoff_tool]  # Pass tools directly
    )
    
    # Register agents with the handoff manager
    handoff_manager.register_agent("general_agent", general_agent, ["general", "assistant", "help"])
    handoff_manager.register_agent("research_agent", research_agent, ["research", "information", "find", "search", "data"])
    handoff_manager.register_agent("code_agent", code_agent, ["code", "programming", "debug", "function", "algorithm"])
    handoff_manager.register_agent("creative_agent", creative_agent, ["creative", "write", "story", "poem", "content"])
    
    # Example 1: Start with a general query
    initial_message = Message(
        role=MessageRole.USER,
        content="I need help with a research project on renewable energy sources."
    )
    
    # Add the message to the conversation history
    handoff_manager.add_to_history(initial_message)
    
    # Process the message with the general agent
    print("User: I need help with a research project on renewable energy sources.")
    general_response = await general_agent.process_message(initial_message)
    
    # Add the response to the conversation history
    handoff_manager.add_to_history(general_response)
    
    print(f"General Agent: {general_response.content}\n")
    
    # Example 2: Follow up with a more specific query
    follow_up_message = Message(
        role=MessageRole.USER,
        content="Can you write a Python script to visualize renewable energy adoption trends?"
    )
    
    # Add the message to the conversation history
    handoff_manager.add_to_history(follow_up_message)
    
    # Process the message with the general agent
    print("User: Can you write a Python script to visualize renewable energy adoption trends?")
    follow_up_response = await general_agent.process_message(follow_up_message)
    
    # Add the response to the conversation history
    handoff_manager.add_to_history(follow_up_response)
    
    print(f"General Agent: {follow_up_response.content}\n")
    
    # Example 3: Ask for a creative piece
    creative_message = Message(
        role=MessageRole.USER,
        content="Write a short poem about the beauty of solar energy."
    )
    
    # Add the message to the conversation history
    handoff_manager.add_to_history(creative_message)
    
    # Process the message with the general agent
    print("User: Write a short poem about the beauty of solar energy.")
    creative_response = await general_agent.process_message(creative_message)
    
    # Add the response to the conversation history
    handoff_manager.add_to_history(creative_response)
    
    print(f"General Agent: {creative_response.content}\n")
    
    # Print the full conversation history
    print("\n--- CONVERSATION HISTORY ---\n")
    for message in handoff_manager.get_history():
        role = "User" if message.role == MessageRole.USER else "Agent"
        print(f"{role}: {message.content}\n")

if __name__ == "__main__":
    asyncio.run(main())