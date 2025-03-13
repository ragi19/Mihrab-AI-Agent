"""
Example demonstrating the enhanced model configuration system
"""

import asyncio
import os
from typing import Any, Dict, Set

from llm_agents.core.message import Message, MessageRole
from llm_agents.models import (
    AdapterRegistry,
    BaseModel,
    ModelCapability,
    ModelConfig,
    ModelRegistry,
    ProviderConfig,
    ProviderRegistry,
    create_model,
    create_model_from_config,
)
from llm_agents.models.providers import OpenAIAdapter
from llm_agents.utils.logging import configure_logging, get_logger

# Configure logging
configure_logging(log_level="INFO")
logger = get_logger("model_config_example")


async def example_model_config():
    """Demonstrate model configuration usage"""
    logger.info("Demonstrating model configuration system")

    # Create a model configuration
    model_config = ModelConfig(
        model_name="gpt-3.5-turbo",
        provider_name="openai",
        capabilities={
            ModelCapability.CHAT,
            ModelCapability.SYSTEM_MESSAGES,
            ModelCapability.STREAM,
        },
        context_window=16385,
        max_tokens=4096,
        supports_streaming=True,
        parameters={"temperature": 0.7, "top_p": 1.0},
    )

    logger.info(f"Created model configuration: {model_config}")

    # Register the model configuration
    ModelRegistry.register(
        name="gpt-3.5-turbo",
        model_class=BaseModel,
        capabilities=model_config.capabilities,
        metadata={
            "description": "GPT-3.5 Turbo model from OpenAI",
            "version": "latest",
        },
    )

    # Find models with specific capabilities
    required_capabilities = {ModelCapability.CHAT, ModelCapability.SYSTEM_MESSAGES}
    matching_models = ModelRegistry.find_models_with_capabilities(required_capabilities)

    logger.info(f"Models with required capabilities: {matching_models}")

    # Create a provider configuration
    provider_config = ProviderConfig(
        name="openai",
        models={"gpt-3.5-turbo": model_config},
        default_parameters={"temperature": 0.7},
    )

    logger.info(f"Created provider configuration: {provider_config}")

    # Get adapter for OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables")
        logger.info("Using adapter without API key for demonstration purposes")

    adapter = AdapterRegistry.get_adapter("openai", {"api_key": api_key})

    if adapter:
        logger.info(f"Got adapter for OpenAI: {adapter}")

        # Initialize adapter if API key is available
        if api_key:
            try:
                await adapter.initialize()
                logger.info("Adapter initialized successfully")

                # Get available models
                models = await adapter.get_available_models()
                logger.info(f"Available models: {models}")

                # Create a model instance
                model = await adapter.create_model(
                    model_name="gpt-3.5-turbo", parameters={"temperature": 0.7}
                )

                logger.info(f"Created model instance: {model}")

                # Generate a response
                messages = [
                    Message(
                        role=MessageRole.SYSTEM, content="You are a helpful assistant."
                    ),
                    Message(role=MessageRole.USER, content="Hello, how are you today?"),
                ]

                logger.info("Generating response...")
                response = await model.generate_response(messages)

                logger.info(f"Response: {response.content}")

            except Exception as e:
                logger.error(f"Error using adapter: {e}")
    else:
        logger.warning("OpenAI adapter not found")

    # Create model using factory function
    try:
        # Only attempt if API key is available
        if api_key:
            logger.info("Creating model using factory function...")
            model = await create_model(
                model_name="gpt-3.5-turbo",
                provider_name="openai",
                provider_kwargs={"api_key": api_key},
                model_parameters={"temperature": 0.7},
                required_capabilities={ModelCapability.CHAT},
            )

            logger.info(f"Created model using factory: {model}")
        else:
            logger.info("Skipping factory creation example (no API key)")
    except Exception as e:
        logger.error(f"Error creating model with factory: {e}")

    # Create model from config
    try:
        # Only attempt if API key is available
        if api_key:
            logger.info("Creating model from config...")
            model = await create_model_from_config(
                model_config=model_config, provider_kwargs={"api_key": api_key}
            )

            logger.info(f"Created model from config: {model}")
        else:
            logger.info("Skipping config creation example (no API key)")
    except Exception as e:
        logger.error(f"Error creating model from config: {e}")


async def main():
    """Main entry point"""
    await example_model_config()


if __name__ == "__main__":
    asyncio.run(main())
