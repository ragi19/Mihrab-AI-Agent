"""
Example of using handoff capabilities to transfer conversations between agents
"""

import asyncio
import os

from dotenv import load_dotenv

from llm_agents import Message, MessageRole, create_agent
from llm_agents.handoff import HandoffAgent, HandoffConfig, HandoffInputData

# Load environment variables from .env file
load_dotenv()


async def run_handoff_example():
    """Run an example demonstrating agent handoff capabilities"""
    print("\n=== Running Agent Handoff Example ===")

    # Choose providers based on available API keys
    available_providers = []

    if os.getenv("OPENAI_API_KEY"):
        available_providers.append(("openai", "gpt-3.5-turbo"))

    if os.getenv("ANTHROPIC_API_KEY"):
        available_providers.append(("anthropic", "claude-instant-1"))

    if os.getenv("GROQ_API_KEY"):
        available_providers.append(("groq", "llama3-8b-8192"))

    if len(available_providers) < 2:
        print("Need at least two provider API keys for handoff example.")
        return

    # Use different providers or models for specialized agents
    general_provider = available_providers[0]
    specialist_provider = (
        available_providers[1]
        if len(available_providers) > 1
        else available_providers[0]
    )

    print(f"General Agent: {general_provider[0]} ({general_provider[1]})")
    print(f"Specialist Agent: {specialist_provider[0]} ({specialist_provider[1]})")

    # Create a general-purpose agent
    general_agent = await create_agent(
        provider_name=general_provider[0],
        model_name=general_provider[1],
        system_message="You are a helpful general-purpose AI assistant. For technical questions about programming, "
        "machine learning, or data science, you should transfer the conversation to a technical specialist. "
        "For all other topics, you can handle the conversation yourself.",
    )

    # Create a specialist agent for technical topics
    technical_specialist = await create_agent(
        provider_name=specialist_provider[0],
        model_name=specialist_provider[1],
        system_message="You are a technical specialist AI assistant with expertise in programming, "
        "machine learning, and data science. You provide detailed and accurate technical information.",
    )

    # Configure handoff between agents
    handoff_config = HandoffConfig(
        source_agent_id="general",
        target_agents={"technical": technical_specialist},
        handoff_prompt="This conversation requires specialized technical knowledge. "
        "I'll transfer you to our technical specialist.",
    )

    # Create a handoff agent wrapping the general agent
    handoff_agent = HandoffAgent(base_agent=general_agent, config=handoff_config)

    # Example conversation flow
    conversation = [
        # General question that doesn't need handoff
        "What's the weather like on Mars?",
        # Technical question that should trigger handoff
        "Can you explain how transformer neural networks work in detail?",
        # Follow-up technical question after handoff
        "How does attention mechanism differ from recurrent neural networks?",
        # General question after technical discussion
        "Thanks for the explanation! On a different note, can you recommend a good sci-fi book?",
    ]

    # Process the conversation
    current_agent_id = "general"

    for i, user_text in enumerate(conversation):
        print(f"\n--- Turn {i+1} ---")
        print(f"User: {user_text}")

        # Create message
        message = Message(role=MessageRole.USER, content=user_text)

        # Process with current agent
        if current_agent_id == "general":
            # Check if handoff is needed
            handoff_data = await handoff_agent.check_handoff_needed(message)

            if handoff_data:
                # Handoff is needed
                target_agent_id = handoff_data.target_agent_id
                print(f"\nHandoff detected! Transferring to {target_agent_id} agent...")
                print(f"Handoff reason: {handoff_data.reason}")

                # Update current agent
                current_agent_id = target_agent_id

                # Process with target agent
                response = await technical_specialist.process_message(message)
            else:
                # No handoff needed, process with general agent
                response = await general_agent.process_message(message)
        else:
            # Already using specialist, check if we should transfer back
            if "sci-fi book" in user_text.lower() or "thanks" in user_text.lower():
                print(
                    "\nDetected non-technical question, transferring back to general agent..."
                )
                current_agent_id = "general"
                response = await general_agent.process_message(message)
            else:
                # Continue with specialist
                response = await technical_specialist.process_message(message)

        # Print current agent and response
        print(f"\nCurrent Agent: {current_agent_id}")
        print(f"Assistant: {response.content}")


async def main():
    """Run the handoff example"""
    await run_handoff_example()


if __name__ == "__main__":
    asyncio.run(main())
