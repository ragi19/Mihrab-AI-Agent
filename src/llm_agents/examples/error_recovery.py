"""
Example of using error recovery features in the LLM Agents framework
"""

import asyncio
import os
import random

from dotenv import load_dotenv

from llm_agents import Message, MessageRole, create_agent
from llm_agents.models.base import ModelError, TokenLimitError
from llm_agents.runtime.recovery import AgentRecovery, ExponentialBackoff, RetryStrategy

# Load environment variables from .env file
load_dotenv()


# Custom exception for simulating errors
class SimulatedAPIError(ModelError):
    """Simulated API error for testing recovery mechanisms"""

    pass


# Mock model that randomly fails to test recovery
class UnreliableMockModel:
    """A mock model that randomly fails to test recovery mechanisms"""

    def __init__(self, real_model, failure_rate=0.5):
        self.real_model = real_model
        self.failure_rate = failure_rate
        self.failure_count = 0
        self.max_failures = 3  # Will succeed after this many failures

    async def generate(self, messages, **kwargs):
        """Randomly fail to test recovery"""
        if (
            random.random() < self.failure_rate
            and self.failure_count < self.max_failures
        ):
            self.failure_count += 1
            error_types = [
                SimulatedAPIError("Simulated API timeout"),
                SimulatedAPIError("Simulated rate limit exceeded"),
                TokenLimitError("Simulated token limit exceeded"),
            ]
            raise random.choice(error_types)

        # If we don't fail, use the real model
        return await self.real_model.generate(messages, **kwargs)

    # Pass through other methods to the real model
    def __getattr__(self, name):
        return getattr(self.real_model, name)


async def run_recovery_example():
    """Run an example demonstrating error recovery features"""
    print("\n=== Running Error Recovery Example ===")

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

    # Create a regular agent
    agent = await create_agent(
        provider_name=provider_name,
        model_name=model_name,
        system_message="You are a helpful AI assistant.",
    )

    # Replace the model with our unreliable mock
    agent.model = UnreliableMockModel(agent.model)

    # Create recovery configuration
    retry_strategy = ExponentialBackoff(initial_delay=0.5, max_delay=5.0, max_retries=5)

    recovery = AgentRecovery(
        agent=agent,
        retry_strategy=retry_strategy,
        recoverable_errors=[SimulatedAPIError, TokenLimitError],
    )

    # Example user message
    user_message = "Tell me about machine learning in 3 sentences."
    message = Message(role=MessageRole.USER, content=user_message)

    print(f"\nUser: {user_message}")
    print("\nAttempting to process message with unreliable model...")

    try:
        # Process with recovery
        response = await recovery.process_with_recovery(message)
        print(f"\nSuccessfully recovered after failures!")
        print(f"Assistant: {response.content}")
    except Exception as e:
        print(f"\nFailed to recover after multiple attempts: {e}")


async def run_fallback_example():
    """Run an example demonstrating fallback mechanisms"""
    print("\n=== Running Fallback Provider Example ===")

    # Create primary and fallback providers based on available API keys
    available_providers = []

    if os.getenv("OPENAI_API_KEY"):
        available_providers.append(("openai", "gpt-3.5-turbo"))

    if os.getenv("ANTHROPIC_API_KEY"):
        available_providers.append(("anthropic", "claude-instant-1"))

    if os.getenv("GROQ_API_KEY"):
        available_providers.append(("groq", "llama3-8b-8192"))

    if len(available_providers) < 2:
        print("Need at least two provider API keys for fallback example.")
        return

    primary_provider, fallback_provider = available_providers[:2]

    print(f"Primary: {primary_provider[0]} ({primary_provider[1]})")
    print(f"Fallback: {fallback_provider[0]} ({fallback_provider[1]})")

    # Create primary agent with unreliable model
    primary_agent = await create_agent(
        provider_name=primary_provider[0],
        model_name=primary_provider[1],
        system_message="You are a helpful AI assistant.",
    )

    # Make the primary model completely unreliable (always fails)
    primary_agent.model = UnreliableMockModel(primary_agent.model, failure_rate=1.0)
    primary_agent.model.max_failures = 10  # Always fail

    # Create fallback agent
    fallback_agent = await create_agent(
        provider_name=fallback_provider[0],
        model_name=fallback_provider[1],
        system_message="You are a helpful AI assistant.",
    )

    # Example user message
    user_message = "What are three interesting facts about dolphins?"
    message = Message(role=MessageRole.USER, content=user_message)

    print(f"\nUser: {user_message}")

    # Try primary first
    try:
        print("\nAttempting with primary provider...")
        response = await primary_agent.process_message(message)
        print(f"Primary provider succeeded!")
        print(f"Assistant: {response.content}")
    except Exception as e:
        print(f"Primary provider failed: {e}")

        # Fall back to secondary provider
        try:
            print("\nFalling back to secondary provider...")
            response = await fallback_agent.process_message(message)
            print(f"Fallback provider succeeded!")
            print(f"Assistant: {response.content}")
        except Exception as e:
            print(f"Fallback provider also failed: {e}")


async def main():
    """Run all recovery examples"""
    await run_recovery_example()
    await run_fallback_example()


if __name__ == "__main__":
    asyncio.run(main())
