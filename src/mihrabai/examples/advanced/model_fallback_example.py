"""
Model Selection and Fallback Example

This example demonstrates the enhanced handoff system with model selection and fallback capabilities:
1. Dynamic model selection based on availability
2. Fallback to alternative models when primary model is unavailable
3. Model switching based on capabilities
4. Resilient multi-agent system with model redundancy
"""

import asyncio
import os
import sys
import json
import time
from typing import Dict, Any, List, Optional, Set
from dotenv import load_dotenv

from mihrabai import create_task_agent, Message, MessageRole
from mihrabai.models.base import ModelCapability
from mihrabai.tools.registry import ToolRegistry
from mihrabai.tools.standard.web import HTTPRequestTool, WebScraperTool
from mihrabai.tools.standard.code_generation import CodeGenerationTool
from mihrabai.tools.base import BaseTool
from mihrabai.core.types import JSON

# Import handoff components
from mihrabai.handoff import (
    HandoffAgent, 
    HandoffConfig, 
    HandoffInputData,
    # Model Selection
    ModelSelectionStrategy,
    PrioritizedModelStrategy,
    CapabilityBasedStrategy,
    RoundRobinStrategy,
    ModelSelectionManager,
    select_model,
    create_model_config,
    # Conditions
    keyword_based_condition,
    complexity_based_condition,
    sentiment_based_condition,
    topic_based_condition,
    conversation_history_condition,
    intent_change_condition,
    expertise_boundary_condition,
    multi_agent_collaboration_condition,
    context_aware_condition,
    # Filters
    preserve_user_messages_only,
    summarize_previous_responses,
    remove_sensitive_information,
    preserve_context,
    extract_key_information,
    prioritize_messages,
    add_handoff_context,
    transform_message_format,
    merge_related_messages,
    add_feedback_loop,
)

# Load environment variables
load_dotenv()

# Get API keys
groq_api_key = os.environ.get("GROQ_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

# Create memory directory if it doesn't exist
MEMORY_DIR = os.path.join(os.path.dirname(__file__), "agent_memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

# Define a wrapper for CodeGenerationTool that implements the missing execute method
class CodeGenerationToolWrapper(CodeGenerationTool):
    """Wrapper for CodeGenerationTool that implements the missing execute method"""
    
    async def execute(self, parameters: Dict[str, Any]) -> JSON:
        """Execute the code generation tool with the given parameters"""
        return await self._execute(parameters)

# Define a feedback tool
class FeedbackTool(BaseTool):
    """Tool for collecting feedback on agent responses"""
    
    def __init__(self, feedback_handler):
        super().__init__(
            name="feedback",
            description="Provide feedback on the agent's response"
        )
        self.feedback_handler = feedback_handler
    
    async def execute(self, parameters: Dict[str, Any]) -> JSON:
        """Execute the feedback tool with the given parameters"""
        return await self._execute(parameters)
    
    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        score = parameters.get("score", 0)
        comments = parameters.get("comments", "")
        
        # Process feedback
        await self.feedback_handler(score, comments)
        
        return {
            "status": "success",
            "message": "Thank you for your feedback!"
        }
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "description": "Rating from 1-5, where 5 is excellent",
                    "minimum": 1,
                    "maximum": 5
                },
                "comments": {
                    "type": "string",
                    "description": "Optional comments about the response"
                }
            },
            "required": ["score"]
        }

# Define a context generator for handoffs
def generate_handoff_context(input_data: HandoffInputData) -> str:
    """Generate context information for a handoff."""
    source_agent = input_data.source_agent or "unknown"
    reason = input_data.metadata.get("handoff_reason", "specialized expertise needed")
    
    context = f"""
    This conversation was handed off to you from {source_agent} because: {reason}
    
    The user's query requires your specific expertise. Please focus on providing
    a helpful and accurate response based on your specialized knowledge.
    
    Handoff chain: {' -> '.join(input_data.handoff_chain)}
    """
    
    return context

async def main():
    # Create tool instances
    http_tool = HTTPRequestTool()
    scraper_tool = WebScraperTool()
    code_generator_tool = CodeGenerationToolWrapper()
    
    # Create feedback handler
    async def feedback_handler(score, comments):
        print(f"Received feedback: {score}/5 - {comments}")
    
    feedback_tool = FeedbackTool(feedback_handler)
    
    # Define model candidates for different providers
    groq_models = []
    if groq_api_key:
        groq_models = [
            create_model_config(
                model_name="llama3-70b-8192",
                provider_name="groq",
                provider_kwargs={"api_key": groq_api_key},
                priority=3,
                capabilities={
                    "CHAT", "COMPLETION", "SYSTEM_MESSAGES", "TOKEN_COUNTING", "STREAM"
                },
            ),
            create_model_config(
                model_name="llama3-8b-8192",
                provider_name="groq",
                provider_kwargs={"api_key": groq_api_key},
                priority=2,
                capabilities={
                    "CHAT", "COMPLETION", "SYSTEM_MESSAGES", "TOKEN_COUNTING", "STREAM"
                },
            ),
            create_model_config(
                model_name="mixtral-8x7b-32768",
                provider_name="groq",
                provider_kwargs={"api_key": groq_api_key},
                priority=1,
                capabilities={
                    "CHAT", "COMPLETION", "SYSTEM_MESSAGES", "TOKEN_COUNTING", "STREAM"
                },
            ),
        ]
    
    openai_models = []
    if openai_api_key:
        openai_models = [
            create_model_config(
                model_name="gpt-4o",
                provider_name="openai",
                provider_kwargs={"api_key": openai_api_key},
                priority=3,
                capabilities={
                    "CHAT", "COMPLETION", "SYSTEM_MESSAGES", "TOKEN_COUNTING", "STREAM", "FUNCTION_CALLING"
                },
            ),
            create_model_config(
                model_name="gpt-4-turbo",
                provider_name="openai",
                provider_kwargs={"api_key": openai_api_key},
                priority=2,
                capabilities={
                    "CHAT", "COMPLETION", "SYSTEM_MESSAGES", "TOKEN_COUNTING", "STREAM", "FUNCTION_CALLING"
                },
            ),
            create_model_config(
                model_name="gpt-3.5-turbo",
                provider_name="openai",
                provider_kwargs={"api_key": openai_api_key},
                priority=1,
                capabilities={
                    "CHAT", "COMPLETION", "SYSTEM_MESSAGES", "TOKEN_COUNTING", "STREAM", "FUNCTION_CALLING"
                },
            ),
        ]
    
    anthropic_models = []
    if anthropic_api_key:
        anthropic_models = [
            create_model_config(
                model_name="claude-3-opus-20240229",
                provider_name="anthropic",
                provider_kwargs={"api_key": anthropic_api_key},
                priority=3,
                capabilities={
                    "CHAT", "COMPLETION", "SYSTEM_MESSAGES", "TOKEN_COUNTING", "STREAM"
                },
            ),
            create_model_config(
                model_name="claude-3-sonnet-20240229",
                provider_name="anthropic",
                provider_kwargs={"api_key": anthropic_api_key},
                priority=2,
                capabilities={
                    "CHAT", "COMPLETION", "SYSTEM_MESSAGES", "TOKEN_COUNTING", "STREAM"
                },
            ),
            create_model_config(
                model_name="claude-3-haiku-20240307",
                provider_name="anthropic",
                provider_kwargs={"api_key": anthropic_api_key},
                priority=1,
                capabilities={
                    "CHAT", "COMPLETION", "SYSTEM_MESSAGES", "TOKEN_COUNTING", "STREAM"
                },
            ),
        ]
    
    # Combine all model candidates
    all_models = groq_models + openai_models + anthropic_models
    
    if not all_models:
        print("Error: No API keys provided. Please set at least one of GROQ_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.")
        sys.exit(1)
    
    # Create specialized agents with model selection capabilities
    
    # 1. General Assistant - Handles general queries and routes to specialists
    general_agent = HandoffAgent(
        name="general_agent",
        system_prompt="""You are a general assistant that can handle a wide range of tasks.
        For specialized tasks, you can hand off to other agents with more appropriate expertise.
        You excel at understanding user needs and routing them to the right specialist.""",
        expertise=["general", "assistant", "help", "routing"],
        can_defer=True,
        memory_path=os.path.join(MEMORY_DIR, "general_agent.json"),
        feedback_enabled=True,
        auto_discover_agents=True,
        collaboration_mode=True,
        model_candidates=all_models,
        required_model_capabilities={"CHAT", "SYSTEM_MESSAGES"},
    )
    
    # 2. Research Agent - Specializes in finding information
    research_agent = HandoffAgent(
        name="research_agent",
        system_prompt="""You are a specialized research agent that excels at finding information.
        Your task is to search for relevant information on topics and provide comprehensive research results.
        Focus on finding factual information from reliable sources. Include citations when possible.""",
        tools=[http_tool, scraper_tool],
        expertise=["research", "information", "find", "search", "data", "facts"],
        can_defer=True,
        memory_path=os.path.join(MEMORY_DIR, "research_agent.json"),
        feedback_enabled=True,
        model_candidates=all_models,
        required_model_capabilities={"CHAT", "SYSTEM_MESSAGES"},
    )
    
    # 3. Code Agent - Specializes in writing and debugging code
    code_agent = HandoffAgent(
        name="code_agent",
        system_prompt="""You are a specialized code agent that excels at writing and debugging code.
        Your task is to write clean, efficient, and well-documented code based on requirements.
        You can also debug existing code and suggest improvements.""",
        tools=[code_generator_tool],
        expertise=["code", "programming", "debug", "function", "algorithm", "development"],
        can_defer=True,
        memory_path=os.path.join(MEMORY_DIR, "code_agent.json"),
        feedback_enabled=True,
        model_candidates=all_models,
        required_model_capabilities={"CHAT", "SYSTEM_MESSAGES"},
    )
    
    # Initialize agents
    print("Initializing agents with dynamic model selection...")
    
    try:
        await general_agent.initialize()
        print(f"General agent initialized with model: {general_agent.model.model_name}")
    except Exception as e:
        print(f"Failed to initialize general agent: {e}")
        sys.exit(1)
        
    try:
        await research_agent.initialize()
        print(f"Research agent initialized with model: {research_agent.model.model_name}")
    except Exception as e:
        print(f"Failed to initialize research agent: {e}")
        sys.exit(1)
        
    try:
        await code_agent.initialize()
        print(f"Code agent initialized with model: {code_agent.model.model_name}")
    except Exception as e:
        print(f"Failed to initialize code agent: {e}")
        sys.exit(1)
    
    # Set up handoff configurations
    general_to_research = HandoffConfig(
        name="general_to_research",
        description="Handoff to research agent for information gathering",
        target_agent=research_agent,
        condition=keyword_based_condition(["research", "find", "information", "data", "search"]),
        input_filter=add_handoff_context(generate_handoff_context),
        metadata={"priority": 2}
    )
    
    general_to_code = HandoffConfig(
        name="general_to_code",
        description="Handoff to code agent for programming tasks",
        target_agent=code_agent,
        condition=keyword_based_condition(["code", "program", "function", "algorithm", "debug"]),
        input_filter=add_handoff_context(generate_handoff_context),
        metadata={"priority": 2}
    )
    
    # Add the handoffs to the general agent
    general_agent.handoffs = [
        general_to_research,
        general_to_code,
    ]
    
    # Example 1: Basic query with model information
    print("\n=== Example 1: Basic Query with Model Information ===\n")
    initial_message = Message(
        role=MessageRole.USER,
        content="What model are you using to answer my questions?"
    )
    
    print("User: What model are you using to answer my questions?")
    general_response, context = await general_agent.process(initial_message.content)
    print(f"General Agent ({general_agent.model.model_name}): {general_response}\n")
    
    # Example 2: Model switching demonstration
    print("\n=== Example 2: Model Switching Demonstration ===\n")
    print("Attempting to switch the general agent's model...")
    
    # Try to switch to a different model
    switch_success = await general_agent.switch_model()
    
    if switch_success:
        print(f"Successfully switched to model: {general_agent.model.model_name}")
        
        # Test the new model
        switch_message = Message(
            role=MessageRole.USER,
            content="What model are you using now after the switch?"
        )
        
        print("User: What model are you using now after the switch?")
        switch_response, switch_context = await general_agent.process(switch_message.content)
        print(f"General Agent ({general_agent.model.model_name}): {switch_response}\n")
    else:
        print("Failed to switch models. No alternative models available.")
    
    # Example 3: Handoff with model information
    print("\n=== Example 3: Handoff with Model Information ===\n")
    handoff_message = Message(
        role=MessageRole.USER,
        content="Can you write a Python function to calculate the Fibonacci sequence?"
    )
    
    print("User: Can you write a Python function to calculate the Fibonacci sequence?")
    handoff_response, handoff_context = await general_agent.process(handoff_message.content)
    
    # Determine which agent handled the request
    handling_agent = "General Agent"
    handling_model = general_agent.model.model_name
    
    if "handoff_metadata" in handoff_context:
        if handoff_context.get("source_agent") == "general_agent":
            if "code" in handoff_context.get("handoff_reason", "").lower():
                handling_agent = "Code Agent"
                handling_model = code_agent.model.model_name
    
    print(f"{handling_agent} ({handling_model}): {handoff_response}\n")
    
    # Example 4: Simulated model failure and fallback
    print("\n=== Example 4: Simulated Model Failure and Fallback ===\n")
    
    # Temporarily make the current model unavailable to simulate a failure
    original_model = general_agent.model
    general_agent.model = None
    
    # Create a new message
    fallback_message = Message(
        role=MessageRole.USER,
        content="What's the capital of France?"
    )
    
    print("User: What's the capital of France?")
    print("Simulating model failure and fallback...")
    
    try:
        # This should fail because the model is temporarily unavailable
        await general_agent.process(fallback_message.content)
    except Exception as e:
        print(f"Expected error: {e}")
        
        # Try to recover by switching to a different model
        recovery_success = await general_agent.switch_model()
        
        if recovery_success:
            print(f"Recovered by switching to model: {general_agent.model.model_name}")
            
            # Try again with the new model
            fallback_response, fallback_context = await general_agent.process(fallback_message.content)
            print(f"General Agent ({general_agent.model.model_name}): {fallback_response}\n")
        else:
            print("Failed to recover. No alternative models available.")
            # Restore the original model
            general_agent.model = original_model
    
    # Example 5: Model selection based on capabilities
    print("\n=== Example 5: Model Selection Based on Capabilities ===\n")
    
    # Create a specialized agent that requires function calling capability
    function_calling_models = [m for m in all_models if "FUNCTION_CALLING" in m.get("capabilities", [])]
    
    if function_calling_models:
        function_agent = HandoffAgent(
            name="function_agent",
            system_prompt="""You are a specialized agent that uses function calling capabilities.
            Your task is to use structured function calls to perform tasks.""",
            expertise=["function", "structured", "api"],
            model_candidates=function_calling_models,
            required_model_capabilities={"CHAT", "SYSTEM_MESSAGES", "FUNCTION_CALLING"},
        )
        
        try:
            await function_agent.initialize()
            print(f"Function agent initialized with model: {function_agent.model.model_name}")
            
            # Test the function-calling capable model
            function_message = Message(
                role=MessageRole.USER,
                content="What model are you using that supports function calling?"
            )
            
            print("User: What model are you using that supports function calling?")
            function_response, function_context = await function_agent.process(function_message.content)
            print(f"Function Agent ({function_agent.model.model_name}): {function_response}\n")
        except Exception as e:
            print(f"Failed to initialize function agent: {e}")
    else:
        print("No models with function calling capability available.")
    
    # Example 6: Round-Robin Strategy Demonstration
    print("\n=== Example 6: Round-Robin Strategy Demonstration ===\n")
    
    # Create a new agent that uses the round-robin strategy
    round_robin_agent = HandoffAgent(
        name="round_robin_agent",
        system_prompt="""You are an agent that uses the round-robin strategy to cycle through available models.
        This helps distribute the load and provides redundancy in case of model failures.""",
        expertise=["round_robin", "load_balancing", "redundancy"],
        model_candidates=all_models,
        required_model_capabilities={"CHAT", "SYSTEM_MESSAGES"},
    )
    
    # Register the round-robin strategy as the default for this agent
    from mihrabai.handoff.model_selection import default_manager
    default_manager.set_default_strategy("round_robin")
    
    try:
        await round_robin_agent.initialize()
        print(f"Round-robin agent initialized with model: {round_robin_agent.model.model_name}")
        
        # Test the round-robin strategy with multiple requests
        print("Testing round-robin strategy with multiple requests...")
        
        for i in range(3):
            rr_message = Message(
                role=MessageRole.USER,
                content=f"This is request {i+1}. What model are you using?"
            )
            
            print(f"User: This is request {i+1}. What model are you using?")
            
            # Force a model switch to demonstrate round-robin behavior
            await round_robin_agent.switch_model(strategy_name="round_robin")
            
            rr_response, rr_context = await round_robin_agent.process(rr_message.content)
            print(f"Round-Robin Agent ({round_robin_agent.model.model_name}): {rr_response}\n")
            
            # Small delay to make the output more readable
            await asyncio.sleep(1)
        
        # Reset the default strategy
        default_manager.set_default_strategy("prioritized")
        
    except Exception as e:
        print(f"Failed to demonstrate round-robin strategy: {e}")
        # Reset the default strategy
        default_manager.set_default_strategy("prioritized")

if __name__ == "__main__":
    asyncio.run(main()) 