"""
Factory functions for creating agents and models
"""

from typing import Any, Dict, List, Optional, Union

from .config import config
from .core.agent import Agent
from .core.chat_agent import ChatAgent
from .core.memory_task_agent import MemoryEnabledTaskAgent as MemoryTaskAgent
from .core.task_agent import TaskAgent
from .models.provider_registry import ProviderRegistry


async def create_agent(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    agent_type: str = "chat",
    provider_kwargs: Optional[Dict[str, Any]] = None,
    model_parameters: Optional[Dict[str, Any]] = None,
    system_message: Optional[str] = None,
    max_history_tokens: Optional[int] = None,
) -> Union[Agent, ChatAgent]:
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


async def create_task_agent(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    provider_kwargs: Optional[Dict[str, Any]] = None,
    model_parameters: Optional[Dict[str, Any]] = None,
    system_message: Optional[str] = None,
    max_history_tokens: Optional[int] = None,
    tools: Optional[List[Any]] = None,
) -> TaskAgent:
    """Create a task agent with the specified configuration

    Args:
        provider_name: Name of the provider to use (default: from config)
        model_name: Name of the model to use (default: from provider config)
        provider_kwargs: Additional arguments for provider initialization
        model_parameters: Model-specific parameters
        system_message: System message for the agent
        max_history_tokens: Maximum history tokens for the agent
        tools: List of tools to make available to the agent

    Returns:
        Configured TaskAgent instance
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

    # Create task agent
    return TaskAgent(
        model=model,
        system_message=system_message or config.default_system_message,
        max_history_tokens=max_history_tokens or config.max_history_tokens,
        tools=tools or [],
    )


async def create_memory_task_agent(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    provider_kwargs: Optional[Dict[str, Any]] = None,
    model_parameters: Optional[Dict[str, Any]] = None,
    system_message: Optional[str] = None,
    max_history_tokens: Optional[int] = None,
    tools: Optional[List[Any]] = None,
    memory_provider_name: Optional[str] = None,
    memory_model_name: Optional[str] = None,
    memory_provider_kwargs: Optional[Dict[str, Any]] = None,
    memory_model_parameters: Optional[Dict[str, Any]] = None,
) -> MemoryTaskAgent:
    """Create a memory task agent with the specified configuration

    Args:
        provider_name: Name of the provider to use (default: from config)
        model_name: Name of the model to use (default: from provider config)
        provider_kwargs: Additional arguments for provider initialization
        model_parameters: Model-specific parameters
        system_message: System message for the agent
        max_history_tokens: Maximum history tokens for the agent
        tools: List of tools to make available to the agent
        memory_provider_name: Provider name for memory operations (default: same as provider_name)
        memory_model_name: Model name for memory operations (default: same as model_name)
        memory_provider_kwargs: Provider kwargs for memory operations (default: same as provider_kwargs)
        memory_model_parameters: Model parameters for memory operations (default: same as model_parameters)

    Returns:
        Configured MemoryTaskAgent instance
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

    # Create model instance for the main agent
    model = await ProviderRegistry.create_model(
        provider_name=provider_name, model_name=model_name, **final_provider_kwargs
    )

    # Set model parameters
    if model_parameters:
        model.update_parameters(model_parameters)
    elif "default_parameters" in provider_config:
        model.update_parameters(provider_config["default_parameters"])

    # Use the same model for memory operations if not specified
    memory_provider_name = memory_provider_name or provider_name
    memory_model_name = memory_model_name or model_name
    memory_provider_kwargs = memory_provider_kwargs or provider_kwargs
    memory_model_parameters = memory_model_parameters or model_parameters

    # Get memory provider configuration
    memory_provider_config = config.get_provider_config(memory_provider_name)

    # Merge memory provider kwargs with config
    final_memory_provider_kwargs = memory_provider_config.copy()
    if memory_provider_kwargs:
        final_memory_provider_kwargs.update(memory_provider_kwargs)

    # Create model instance for memory operations
    memory_model = await ProviderRegistry.create_model(
        provider_name=memory_provider_name,
        model_name=memory_model_name,
        **final_memory_provider_kwargs
    )

    # Set memory model parameters
    if memory_model_parameters:
        memory_model.update_parameters(memory_model_parameters)
    elif "default_parameters" in memory_provider_config:
        memory_model.update_parameters(memory_provider_config["default_parameters"])

    # Create memory task agent
    return MemoryTaskAgent(
        model=model,
        memory_model=memory_model,
        system_message=system_message or config.default_system_message,
        max_history_tokens=max_history_tokens or config.max_history_tokens,
        tools=tools or [],
    )
