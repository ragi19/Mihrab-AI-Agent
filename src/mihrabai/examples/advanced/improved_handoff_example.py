"""
Improved Handoff System Example

This example demonstrates the enhanced handoff system with advanced features:
1. Context-aware handoffs
2. Feedback loops
3. Memory persistence
4. Dynamic agent discovery
5. Multi-agent collaboration
6. Adaptive expertise learning
"""

import asyncio
import os
import sys
import json
import time
from typing import Dict, Any, List, Optional
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

# Get Groq API key
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    print("Groq API key not found in environment variables.")
    groq_api_key = input("Please enter your Groq API key: ")
    if not groq_api_key:
        print("Error: Groq API key is required to run this example.")
        sys.exit(1)

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

# Define an enhanced handoff manager
class EnhancedHandoffManager:
    def __init__(self):
        self.agents = {}
        self.conversation_history = []
        self.agent_performance = {}
        self.last_handoff_time = None
        self.feedback_history = []
    
    def register_agent(self, name: str, agent: HandoffAgent, expertise: List[str], priority: int = 0):
        """Register an agent with the handoff manager."""
        self.agents[name] = {
            "agent": agent,
            "expertise": expertise,
            "priority": priority,
            "last_active": time.time()
        }
        
        # Initialize performance tracking
        self.agent_performance[name] = {
            "handoffs_received": 0,
            "handoffs_initiated": 0,
            "successful_handoffs": 0,
            "failed_handoffs": 0,
            "average_feedback": 0.0,
            "total_feedback_count": 0
        }
        
        # Register this agent with all other agents for dynamic discovery
        for other_name, other_info in self.agents.items():
            if other_name != name:
                other_agent = other_info["agent"]
                other_agent.register_agent(name, agent, expertise)
                agent.register_agent(other_name, other_info["agent"], other_info["expertise"])
    
    def add_to_history(self, message: Message):
        """Add a message to the conversation history."""
        self.conversation_history.append({
            "role": message.role.value,
            "content": message.content,
            "timestamp": time.time()
        })
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation_history
    
    def find_agent_for_task(self, task: str) -> Optional[str]:
        """Find the most appropriate agent for a task based on expertise."""
        best_match = None
        best_score = -1
        
        for name, info in self.agents.items():
            # Calculate match score based on expertise keywords and priority
            score = 0
            for expertise in info["expertise"]:
                if expertise.lower() in task.lower():
                    score += 1
            
            # Adjust score by priority
            score = score * (1 + 0.1 * info["priority"])
            
            if score > best_score:
                best_score = score
                best_match = name
        
        # Only return if we have a reasonable match
        if best_score > 0:
            return best_match
        return None
    
    async def process_feedback(self, agent_name: str, score: int, comments: str):
        """Process feedback for an agent."""
        # Record feedback
        feedback = {
            "agent": agent_name,
            "score": score,
            "comments": comments,
            "timestamp": time.time()
        }
        self.feedback_history.append(feedback)
        
        # Update agent performance metrics
        if agent_name in self.agent_performance:
            perf = self.agent_performance[agent_name]
            total = perf["total_feedback_count"]
            current_avg = perf["average_feedback"]
            
            # Update running average
            if total > 0:
                perf["average_feedback"] = (current_avg * total + score) / (total + 1)
            else:
                perf["average_feedback"] = score
                
            perf["total_feedback_count"] += 1
        
        # Save feedback to file
        self._save_feedback()
    
    def _save_feedback(self):
        """Save feedback history to file."""
        feedback_path = os.path.join(MEMORY_DIR, "feedback_history.json")
        try:
            with open(feedback_path, "w") as f:
                json.dump(self.feedback_history, f, indent=2)
        except Exception as e:
            print(f"Error saving feedback: {e}")
    
    def get_agent_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all agents."""
        # Update with the latest metrics from each agent
        for name, info in self.agents.items():
            agent = info["agent"]
            metrics = agent.get_performance_metrics()
            
            # Update our tracking with agent's internal metrics
            self.agent_performance[name].update({
                "handoffs_initiated": metrics.get("handoffs_initiated", 0),
                "handoffs_received": metrics.get("handoffs_received", 0),
                "successful_handoffs": metrics.get("successful_handoffs", 0),
                "failed_handoffs": metrics.get("failed_handoffs", 0),
            })
        
        return self.agent_performance

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
    # Create a handoff manager
    handoff_manager = EnhancedHandoffManager()
    
    # Create tool instances
    http_tool = HTTPRequestTool()
    scraper_tool = WebScraperTool()
    code_generator_tool = CodeGenerationToolWrapper()
    
    # Create feedback handler
    async def feedback_handler(score, comments):
        current_agent = "general_agent"  # This would be tracked in a real system
        await handoff_manager.process_feedback(current_agent, score, comments)
    
    feedback_tool = FeedbackTool(feedback_handler)
    
    # Create specialized agents with enhanced capabilities
    
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
    )
    
    # 4. Creative Agent - Specializes in creative writing
    creative_agent = HandoffAgent(
        name="creative_agent",
        system_prompt="""You are a specialized creative agent that excels at creative writing and content generation.
        Your task is to generate engaging, original content based on prompts.
        You can write stories, poems, marketing copy, and other creative content.""",
        expertise=["creative", "write", "story", "poem", "content", "marketing"],
        can_defer=True,
        memory_path=os.path.join(MEMORY_DIR, "creative_agent.json"),
        feedback_enabled=True,
    )
    
    # 5. Technical Support Agent - Specializes in troubleshooting
    support_agent = HandoffAgent(
        name="support_agent",
        system_prompt="""You are a specialized technical support agent that excels at troubleshooting.
        Your task is to help users diagnose and resolve technical issues.
        You can provide step-by-step instructions and explain complex concepts in simple terms.""",
        expertise=["support", "troubleshoot", "fix", "error", "problem", "issue", "help"],
        can_defer=True,
        memory_path=os.path.join(MEMORY_DIR, "support_agent.json"),
        feedback_enabled=True,
    )
    
    # Initialize agents with models
    from mihrabai.handoff.model_selection import create_model_config, select_model
    
    # Create model configuration
    model_config = create_model_config(
        model_name="llama3-70b-8192",
        provider_name="groq",
        provider_kwargs={"api_key": groq_api_key},
        capabilities={"CHAT", "COMPLETION"}
    )
    
    # Select model
    model, _ = await select_model([model_config])
    
    await general_agent.initialize(model)
    await research_agent.initialize(model)
    await code_agent.initialize(model)
    await creative_agent.initialize(model)
    await support_agent.initialize(model)
    
    # Set up handoff configurations with advanced conditions
    
    # General agent handoffs
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
    
    general_to_creative = HandoffConfig(
        name="general_to_creative",
        description="Handoff to creative agent for content creation",
        target_agent=creative_agent,
        condition=keyword_based_condition(["write", "story", "poem", "creative", "content"]),
        input_filter=add_handoff_context(generate_handoff_context),
        metadata={"priority": 1}
    )
    
    general_to_support = HandoffConfig(
        name="general_to_support",
        description="Handoff to support agent for technical issues",
        target_agent=support_agent,
        condition=keyword_based_condition(["error", "issue", "problem", "fix", "troubleshoot"]),
        input_filter=add_handoff_context(generate_handoff_context),
        metadata={"priority": 3}
    )
    
    # Add complexity-based handoffs
    complex_technical_handoff = HandoffConfig(
        name="complex_technical",
        description="Handoff complex technical queries to code agent",
        target_agent=code_agent,
        condition=complexity_based_condition(threshold=30),
        input_filter=add_handoff_context(generate_handoff_context),
        metadata={"priority": 1}
    )
    
    # Add sentiment-based handoffs
    negative_sentiment_handoff = HandoffConfig(
        name="negative_sentiment",
        description="Handoff negative sentiment queries to support agent",
        target_agent=support_agent,
        condition=sentiment_based_condition(),
        input_filter=add_handoff_context(generate_handoff_context),
        metadata={"priority": 4}  # High priority for upset users
    )
    
    # Add conversation history based handoff
    long_conversation_handoff = HandoffConfig(
        name="long_conversation",
        description="Handoff long conversations to appropriate specialist",
        target_agent=research_agent,  # Default target, will be determined dynamically
        condition=conversation_history_condition(threshold=4),
        input_filter=preserve_context(max_messages=10),
        metadata={"priority": 1}
    )
    
    # Add the handoffs to the general agent
    general_agent.handoffs = [
        general_to_research,
        general_to_code,
        general_to_creative,
        general_to_support,
        complex_technical_handoff,
        negative_sentiment_handoff,
        long_conversation_handoff,
    ]
    
    # Add cross-referral capabilities between specialists
    
    # Research agent can hand off to code agent for data processing
    research_to_code = HandoffConfig(
        name="research_to_code",
        description="Handoff to code agent for data processing or visualization",
        target_agent=code_agent,
        condition=keyword_based_condition(["code", "script", "program", "visualize", "process"]),
        input_filter=add_handoff_context(generate_handoff_context),
    )
    research_agent.handoffs = [research_to_code]
    
    # Code agent can hand off to research agent for information
    code_to_research = HandoffConfig(
        name="code_to_research",
        description="Handoff to research agent for information gathering",
        target_agent=research_agent,
        condition=keyword_based_condition(["research", "find", "information", "documentation"]),
        input_filter=add_handoff_context(generate_handoff_context),
    )
    code_agent.handoffs = [code_to_research]
    
    # Register agents with the handoff manager
    handoff_manager.register_agent("general_agent", general_agent, 
                                  ["general", "assistant", "help"], priority=1)
    handoff_manager.register_agent("research_agent", research_agent, 
                                  ["research", "information", "find", "search", "data"], priority=2)
    handoff_manager.register_agent("code_agent", code_agent, 
                                  ["code", "programming", "debug", "function", "algorithm"], priority=2)
    handoff_manager.register_agent("creative_agent", creative_agent, 
                                  ["creative", "write", "story", "poem", "content"], priority=1)
    handoff_manager.register_agent("support_agent", support_agent, 
                                  ["support", "troubleshoot", "fix", "error", "problem"], priority=3)
    
    # Example 1: Start with a general query
    print("\n=== Example 1: Research Query ===\n")
    initial_message = Message(
        role=MessageRole.USER,
        content="I need help with a research project on renewable energy sources."
    )
    
    # Add the message to the conversation history
    handoff_manager.add_to_history(initial_message)
    
    # Process the message with the general agent
    print("User: I need help with a research project on renewable energy sources.")
    general_response, context = await general_agent.process(initial_message.content)
    
    # Create a response message
    response_message = Message(
        role=MessageRole.ASSISTANT,
        content=general_response
    )
    
    # Add the response to the conversation history
    handoff_manager.add_to_history(response_message)
    
    print(f"Agent: {general_response}\n")
    
    # Example 2: Follow up with a coding query
    print("\n=== Example 2: Coding Query ===\n")
    follow_up_message = Message(
        role=MessageRole.USER,
        content="Can you write a Python script to visualize renewable energy adoption trends?"
    )
    
    # Add the message to the conversation history
    handoff_manager.add_to_history(follow_up_message)
    
    # Process the message with the general agent
    print("User: Can you write a Python script to visualize renewable energy adoption trends?")
    follow_up_response, updated_context = await general_agent.process(
        follow_up_message.content,
        context=context
    )
    
    # Create a response message
    response_message = Message(
        role=MessageRole.ASSISTANT,
        content=follow_up_response
    )
    
    # Add the response to the conversation history
    handoff_manager.add_to_history(response_message)
    
    print(f"Agent: {follow_up_response}\n")
    
    # Example 3: Complex multi-agent collaboration
    print("\n=== Example 3: Complex Collaboration Query ===\n")
    collab_message = Message(
        role=MessageRole.USER,
        content="""I'm working on a project that needs both research and code. 
        First, I need information about climate change impacts by region, 
        and then I need a visualization tool that can show this data on an interactive map.
        Finally, I'd like some creative text to introduce the tool on our website."""
    )
    
    # Add the message to the conversation history
    handoff_manager.add_to_history(collab_message)
    
    # Process with collaboration mode
    print("User:", collab_message.content)
    collab_response, collab_context = await general_agent.process(
        collab_message.content,
        context=updated_context
    )
    
    # Create a response message
    response_message = Message(
        role=MessageRole.ASSISTANT,
        content=collab_response
    )
    
    # Add the response to the conversation history
    handoff_manager.add_to_history(response_message)
    
    print(f"Agent: {collab_response}\n")
    
    # Example 4: Sentiment-based handoff
    print("\n=== Example 4: Sentiment-Based Handoff ===\n")
    sentiment_message = Message(
        role=MessageRole.USER,
        content="""I'm really frustrated with this code. I've been trying to fix this bug for hours
        and nothing is working. The program keeps crashing when I try to load the data file.
        This is terrible and I'm about to give up on the whole project."""
    )
    
    # Add the message to the conversation history
    handoff_manager.add_to_history(sentiment_message)
    
    # Process with sentiment detection
    print("User:", sentiment_message.content)
    sentiment_response, sentiment_context = await general_agent.process(
        sentiment_message.content,
        context=collab_context
    )
    
    # Create a response message
    response_message = Message(
        role=MessageRole.ASSISTANT,
        content=sentiment_response
    )
    
    # Add the response to the conversation history
    handoff_manager.add_to_history(response_message)
    
    print(f"Agent: {sentiment_response}\n")
    
    # Print performance metrics
    print("\n=== Agent Performance Metrics ===\n")
    performance = handoff_manager.get_agent_performance()
    for agent_name, metrics in performance.items():
        print(f"{agent_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
        print()

if __name__ == "__main__":
    asyncio.run(main()) 