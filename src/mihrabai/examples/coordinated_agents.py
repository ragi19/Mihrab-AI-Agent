"""
Example of using coordinated agent groups for multi-agent workflows
"""

import asyncio
import os

from dotenv import load_dotenv

from llm_agents import CoordinatedAgentGroup, Message, MessageRole, create_agent
from llm_agents.models.base import ModelCapability

# Load environment variables from .env file
load_dotenv()


async def create_specialized_agent(provider_name, model_name, role, system_message):
    """Create a specialized agent with a specific role"""
    return await create_agent(
        provider_name=provider_name,
        model_name=model_name,
        system_message=system_message,
    )


async def run_coordinated_agents_example():
    """Run an example with coordinated agents working together"""
    print("\n=== Running Coordinated Agents Example ===")

    # Choose a provider based on available API keys
    provider_name = None
    model_name = None

    if os.getenv("OPENAI_API_KEY"):
        provider_name = "openai"
        model_name = "gpt-3.5-turbo"
    elif os.getenv("ANTHROPIC_API_KEY"):
        provider_name = "anthropic"
        model_name = "claude-instant-1"
    elif os.getenv("GROQ_API_KEY"):
        provider_name = "groq"
        model_name = "llama3-8b-8192"

    if not provider_name:
        print("No API keys found. Please set at least one provider API key.")
        return

    print(f"Using provider: {provider_name} with model: {model_name}")

    # Create specialized agents for different roles
    researcher = await create_specialized_agent(
        provider_name,
        model_name,
        "researcher",
        "You are a research specialist. Your role is to analyze questions and provide factual information. "
        "Focus on providing accurate data and relevant context. Be thorough but concise.",
    )

    creative_writer = await create_specialized_agent(
        provider_name,
        model_name,
        "creative_writer",
        "You are a creative writing specialist. Your role is to take information and craft engaging narratives. "
        "Focus on storytelling, vivid descriptions, and compelling language. Be imaginative and entertaining.",
    )

    critic = await create_specialized_agent(
        provider_name,
        model_name,
        "critic",
        "You are a critical analysis specialist. Your role is to evaluate content for accuracy, clarity, and effectiveness. "
        "Identify strengths and weaknesses, and suggest specific improvements. Be constructive but thorough.",
    )

    # Create a coordinated agent group
    agent_group = CoordinatedAgentGroup(
        agents={
            "researcher": researcher,
            "creative_writer": creative_writer,
            "critic": critic,
        },
        coordinator=None,  # Use default round-robin coordination
    )

    # Example workflow: Research → Creative Writing → Critique
    user_query = "Tell me about the history of artificial intelligence and create a short story based on it."

    print(f"\nUser Query: {user_query}")
    print("\n--- Starting Multi-Agent Workflow ---")

    # Step 1: Research phase
    print("\n[RESEARCHER AGENT]")
    research_message = Message(role=MessageRole.USER, content=user_query)
    research_response = await agent_group.process_with_agent(
        "researcher", research_message
    )
    print(f"Research findings: {research_response.content}")

    # Step 2: Creative writing phase
    print("\n[CREATIVE WRITER AGENT]")
    writing_prompt = (
        f"Based on this research, create a short story:\n{research_response.content}"
    )
    writing_message = Message(role=MessageRole.USER, content=writing_prompt)
    story_response = await agent_group.process_with_agent(
        "creative_writer", writing_message
    )
    print(f"Creative story: {story_response.content}")

    # Step 3: Critique phase
    print("\n[CRITIC AGENT]")
    critique_prompt = f"Evaluate this AI-themed story for accuracy, engagement, and quality:\n{story_response.content}"
    critique_message = Message(role=MessageRole.USER, content=critique_prompt)
    critique_response = await agent_group.process_with_agent("critic", critique_message)
    print(f"Critique: {critique_response.content}")

    print("\n--- Multi-Agent Workflow Complete ---")

    # Final output combining all agent contributions
    final_output = f"""
MULTI-AGENT COLLABORATION RESULT:

RESEARCH:
{research_response.content}

CREATIVE STORY:
{story_response.content}

CRITIQUE:
{critique_response.content}
"""

    print("\n=== Final Output ===")
    print(final_output)


async def main():
    """Run the coordinated agents example"""
    await run_coordinated_agents_example()


if __name__ == "__main__":
    asyncio.run(main())
