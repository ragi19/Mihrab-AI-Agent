"""
Command-line interface for mihrabai
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import config
from .core import Message, MessageRole
from .factory import create_agent, create_memory_task_agent, create_task_agent
from .models import ProviderRegistry
from .tools import ToolRegistry


async def chat_command(args: argparse.Namespace) -> None:
    """Run an interactive chat session with an agent"""
    # Configure the agent
    agent = await create_agent(
        provider_name=args.provider,
        model_name=args.model,
        system_message=args.system_message,
    )

    print(f"Chat session started with {agent.model.model_name}")
    print(f"Type 'exit' or 'quit' to end the session")
    print(f"Type 'clear' to clear the conversation history")
    print(f"Type 'system: <message>' to change the system message")
    print()

    while True:
        try:
            # Get user input
            user_input = input("> ")
            
            # Check for special commands
            if user_input.lower() in ["exit", "quit"]:
                break
            elif user_input.lower() == "clear":
                agent.clear_history()
                agent.add_to_history(
                    Message(role=MessageRole.SYSTEM, content=args.system_message)
                )
                print("Conversation history cleared")
                continue
            elif user_input.lower().startswith("system:"):
                system_message = user_input[7:].strip()
                # Replace the system message
                for i, msg in enumerate(agent.conversation_history):
                    if msg.role == MessageRole.SYSTEM:
                        agent.conversation_history[i] = Message(
                            role=MessageRole.SYSTEM, content=system_message
                        )
                        break
                print(f"System message updated: {system_message}")
                continue
            
            # Process the message
            user_message = Message(role=MessageRole.USER, content=user_input)
            response = await agent.process_message(user_message)
            
            # Display the response
            print(f"\n{response.content}\n")
            
        except KeyboardInterrupt:
            print("\nExiting chat session")
            break
        except Exception as e:
            print(f"Error: {e}")


async def task_command(args: argparse.Namespace) -> None:
    """Run a task agent with specified tools"""
    # Load tools
    tools = []
    if args.tools:
        for tool_name in args.tools.split(","):
            tool_name = tool_name.strip()
            if tool_name in ToolRegistry.get_registered_tools():
                tool_class = ToolRegistry.get_tool(tool_name)
                tools.append(tool_class())
            else:
                print(f"Warning: Tool '{tool_name}' not found, skipping")
    
    # Create task agent
    if args.memory:
        agent = await create_memory_task_agent(
            provider_name=args.provider,
            model_name=args.model,
            system_message=args.system_message,
            tools=tools,
        )
        print(f"Memory-enabled task agent created with {agent.model.model_name}")
    else:
        agent = await create_task_agent(
            provider_name=args.provider,
            model_name=args.model,
            system_message=args.system_message,
            tools=tools,
        )
        print(f"Task agent created with {agent.model.model_name}")
    
    print(f"Available tools: {', '.join([tool.name for tool in tools])}")
    print(f"Type 'exit' or 'quit' to end the session")
    print(f"Type 'clear' to clear the conversation history")
    print()

    while True:
        try:
            # Get user input
            user_input = input("> ")
            
            # Check for special commands
            if user_input.lower() in ["exit", "quit"]:
                break
            elif user_input.lower() == "clear":
                agent.clear_history()
                print("Conversation history cleared")
                continue
            
            # Process the message
            user_message = Message(role=MessageRole.USER, content=user_input)
            response = await agent.process_message(user_message)
            
            # Display the response
            print(f"\n{response.content}\n")
            
        except KeyboardInterrupt:
            print("\nExiting task session")
            break
        except Exception as e:
            print(f"Error: {e}")


async def config_command(args: argparse.Namespace) -> None:
    """Manage configuration settings"""
    if args.list:
        # List current configuration
        print("Current configuration:")
        print(f"Default provider: {config.get_default_provider()}")
        print("Provider configurations:")
        for provider, provider_config in config._config["providers"].items():
            print(f"  {provider}:")
            for key, value in provider_config.items():
                if key == "api_key" and value:
                    print(f"    {key}: ****")
                else:
                    print(f"    {key}: {value}")
        print("Logging configuration:")
        for key, value in config.get_logging_config().items():
            print(f"  {key}: {value}")
    
    elif args.set_provider:
        # Set default provider
        config.set_default_provider(args.set_provider)
        config.save()
        print(f"Default provider set to: {args.set_provider}")
    
    elif args.set_api_key:
        # Set API key for a provider
        if not args.provider:
            print("Error: --provider is required when setting an API key")
            return
        
        provider_config = config.get_provider_config(args.provider)
        provider_config["api_key"] = args.set_api_key
        config.set_provider_config(args.provider, provider_config)
        config.save()
        print(f"API key set for provider: {args.provider}")
    
    elif args.set_default_model:
        # Set default model for a provider
        if not args.provider:
            print("Error: --provider is required when setting a default model")
            return
        
        provider_config = config.get_provider_config(args.provider)
        provider_config["default_model"] = args.set_default_model
        config.set_provider_config(args.provider, provider_config)
        config.save()
        print(f"Default model for {args.provider} set to: {args.set_default_model}")


async def providers_command(args: argparse.Namespace) -> None:
    """List available providers and models"""
    if args.list:
        print("Available providers:")
        for provider in ProviderRegistry.get_registered_providers():
            print(f"  {provider}")
    
    elif args.models and args.provider:
        # List models for a specific provider
        try:
            provider_class = ProviderRegistry.get_provider(args.provider)
            provider_instance = provider_class()
            models = await provider_instance.list_available_models()
            
            print(f"Available models for {args.provider}:")
            for model in models:
                print(f"  {model}")
        except Exception as e:
            print(f"Error listing models for {args.provider}: {e}")


async def tools_command(args: argparse.Namespace) -> None:
    """List available tools"""
    print("Available tools:")
    for tool_name in ToolRegistry.get_registered_tools():
        tool_class = ToolRegistry.get_tool(tool_name)
        tool_instance = tool_class()
        print(f"  {tool_name}: {tool_instance.description}")


def main() -> None:
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description="mihrabai - LLM Agent Framework")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")
    chat_parser.add_argument("--provider", "-p", help="Provider to use")
    chat_parser.add_argument("--model", "-m", help="Model to use")
    chat_parser.add_argument(
        "--system-message", "-s", 
        help="System message to use",
        default="You are a helpful AI assistant."
    )
    
    # Task command
    task_parser = subparsers.add_parser("task", help="Start a task agent session")
    task_parser.add_argument("--provider", "-p", help="Provider to use")
    task_parser.add_argument("--model", "-m", help="Model to use")
    task_parser.add_argument(
        "--system-message", "-s", 
        help="System message to use",
        default="You are a helpful AI assistant that can use tools to accomplish tasks."
    )
    task_parser.add_argument(
        "--tools", "-t",
        help="Comma-separated list of tools to enable"
    )
    task_parser.add_argument(
        "--memory", "-mem",
        action="store_true",
        help="Enable memory capabilities"
    )
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List current configuration"
    )
    config_parser.add_argument(
        "--set-provider",
        help="Set default provider"
    )
    config_parser.add_argument(
        "--set-api-key",
        help="Set API key for a provider"
    )
    config_parser.add_argument(
        "--set-default-model",
        help="Set default model for a provider"
    )
    config_parser.add_argument(
        "--provider", "-p",
        help="Provider to configure"
    )
    
    # Providers command
    providers_parser = subparsers.add_parser("providers", help="List available providers")
    providers_parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available providers"
    )
    providers_parser.add_argument(
        "--models", "-m",
        action="store_true",
        help="List available models for a provider"
    )
    providers_parser.add_argument(
        "--provider", "-p",
        help="Provider to list models for"
    )
    
    # Tools command
    tools_parser = subparsers.add_parser("tools", help="List available tools")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "chat":
        asyncio.run(chat_command(args))
    elif args.command == "task":
        asyncio.run(task_command(args))
    elif args.command == "config":
        asyncio.run(config_command(args))
    elif args.command == "providers":
        asyncio.run(providers_command(args))
    elif args.command == "tools":
        asyncio.run(tools_command(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
