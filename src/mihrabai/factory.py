"""
Factory functions for creating agents and models
"""

from typing import Any, Dict, Optional

from .config import config
from .core.agent import Agent
from .core.chat_agent import ChatAgent
from .models.provider_registry import ProviderRegistry


async def create_agent(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    agent_type: str = "chat",
    provider_kwargs: Optional[Dict[str, Any]] = None,
    model_parameters: Optional[Dict[str, Any]] = None,
    system_message: Optional[str] = None,
    max_history_tokens: Optional[int] = None,
) -> Agent:
    """Create an agent with the specified configuration

    Args:
        provider_name: Name of the provider to use (default: from config)
        model_name: Name of the model to use (default: from provider config)
        agent_type: Type of agent to create ("chat" or "basic")
        provider_kwargs: Additional arguments for provider initialization
        model_parameters: Model-specific parameters
        system_message: System message for chat agents
        max_history_tokens: Maximum history tokens for chat agents

    Returns:
        Configured agent instance
    """
    # Get provider name from config if not specified
    if not provider_name:
        provider_name = config.get_default_provider()

    # Get provider configuration
    provider_config = config.get_provider_config(provider_name)

    # Use default model name if not specified
    if not model_name:
        model_name = provider_config.get("default_model")

    # Merge provider kwargs with config
    final_provider_kwargs = provider_config.copy()
    if provider_kwargs:
        final_provider_kwargs.update(provider_kwargs)

    # Create model instance
    model = await ProviderRegistry.create_model(
        provider_name=provider_name, model_name=model_name, **final_provider_kwargs
    )

    # Set model parameters
    if model_parameters:
        model.update_parameters(model_parameters)
    elif "default_parameters" in provider_config:
        model.update_parameters(provider_config["default_parameters"])

    # Create appropriate agent type
    if agent_type == "chat":
        return ChatAgent(
            model=model,
            system_message=system_message or config.default_system_message,
            max_history_tokens=max_history_tokens or config.max_history_tokens,
        )
    else:
        return Agent(model=model)
