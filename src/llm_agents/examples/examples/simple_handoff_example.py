#!/usr/bin/env python
"""
Simple example demonstrating a basic agent using Groq.
This is a simplified version to troubleshoot the handoff system.
"""

import asyncio
import os
import logging
import sys
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Import required modules
from llm_agents.models.multi_provider import MultiProviderModel, OptimizationStrategy
from llm_agents.models.base import ModelCapability
from llm_agents.core.memory import Memory
from llm_agents.core.memory_task_agent import MemoryEnabledTaskAgent
from llm_agents.runtime.memory_runner import MemoryAgentRunner
from llm_agents.core.message import Message, MessageRole

print("Starting simple agent example...")

class SimpleAgent:
    """Basic agent implementation"""
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.agent = None
        self.runner = None
        print(f"Created SimpleAgent: {name}")
    
    async def initialize(self, model: MultiProviderModel):
        """Initialize the agent with a model"""
        print(f"Initializing agent: {self.name}")
        try:
            # Create memory
            memory = Memory(working_memory_size=10, long_term_memory_size=50)
            
            # Create agent
            self.agent = MemoryEnabledTaskAgent(
                model=model,
                memory=memory,
                system_message=self.system_prompt
            )
            
            # Create runner
            self.runner = MemoryAgentRunner(
                agent=self.agent,
                memory_persistence_path="./memories"
            )
            
            print(f"Successfully initialized agent: {self.name}")
            logger.info(f"Initialized agent: {self.name}")
        except Exception as e:
            print(f"Error initializing agent {self.name}: {str(e)}")
            logger.error(f"Error initializing agent {self.name}: {str(e)}", exc_info=True)
            raise
    
    async def process(self, message: str, session_id: str = None) -> str:
        """Process a message and return the response"""
        print(f"Agent {self.name} processing message: {message[:50]}...")
        try:
            response = await self.runner.run(message, session_id=session_id)
            print(f"Agent {self.name} response: {response.content[:50]}...")
            return response.content
        except Exception as e:
            print(f"Error in agent {self.name} processing: {str(e)}")
            logger.error(f"Error in agent {self.name} processing: {str(e)}", exc_info=True)
            return f"Error processing your request: {str(e)}"

async def main():
    """Main function to run the simple agent example"""
    try:
        # Check for GROQ_API_KEY first
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("\nError: GROQ_API_KEY environment variable not found!")
            print("Please set your Groq API key using:")
            print("$env:GROQ_API_KEY = 'your-api-key'")
            return

        print("\nInitializing simple agent system...")
        
        # Set up the model - using a simpler model that's more likely to be available
        print("Setting up the model...")
        model = await MultiProviderModel.create(
            primary_model="llama3-8b-8192",  # Smaller Llama 3 model
            fallback_models=[],
            required_capabilities={ModelCapability.CHAT},
            optimize_for=OptimizationStrategy.PERFORMANCE
        )
        print(f"Model setup complete with provider: {model.current_provider}")
        
        # Create a simple agent
        print("Creating agent...")
        agent = SimpleAgent(
            name="General",
            system_prompt="You are a helpful assistant who provides concise, accurate responses."
        )
        
        # Initialize the agent
        print("Initializing agent...")
        await agent.initialize(model)
        print("Agent initialized successfully")
        
        # Test queries
        test_queries = [
            "What is machine learning?",
            "How does a computer work?",
            "Tell me about Python programming"
        ]
        
        print("\nProcessing test queries...")
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"QUERY: {query}")
            print(f"{'='*50}")
            response = await agent.process(query, session_id="simple_session")
            print(f"\nRESPONSE: {response}")
            print(f"{'='*50}")
            
    except Exception as e:
        print(f"\nError in main: {str(e)}")
        logger.exception("Detailed error information:")

if __name__ == "__main__":
    asyncio.run(main()) 