"""
Configuration file generator script
"""

import argparse
import os

from mihrabai.utils.config_generator import generate_config, load_environment_keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LLM Agents configuration file"
    )

    # File path
    parser.add_argument(
        "--path",
        help="Path to save the configuration file (default: ~/.mihrabai/config.json)",
    )

    # Provider settings
    parser.add_argument(
        "--openai-key",
        help="OpenAI API key (default: from OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--anthropic-key",
        help="Anthropic API key (default: from ANTHROPIC_API_KEY environment variable)",
    )
    parser.add_argument(
        "--groq-key",
        help="Groq API key (default: from GROQ_API_KEY environment variable)",
    )
    parser.add_argument(
        "--default-provider",
        choices=["openai", "anthropic", "groq"],
        help="Default provider to use",
    )

    # Model settings
    parser.add_argument(
        "--max-history-tokens",
        type=int,
        help="Maximum number of tokens to keep in conversation history",
    )

    # Logging settings
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--log-file", help="Path to log file (if not specified, logs to console only)"
    )
    parser.add_argument("--log-format", help="Custom log message format")

    args = parser.parse_args()

    # Load keys from environment if not provided
    env_keys = load_environment_keys()
    openai_key = args.openai_key or env_keys["openai_key"]
    anthropic_key = args.anthropic_key or env_keys["anthropic_key"]
    groq_key = args.groq_key or env_keys["groq_key"]

    # Build custom settings
    custom_settings = {}
    if args.default_provider:
        custom_settings["default_provider"] = args.default_provider
    if args.max_history_tokens:
        custom_settings["max_history_tokens"] = args.max_history_tokens

    # Build logging config
    logging_config = {}
    if args.log_level:
        logging_config["level"] = args.log_level
    if args.log_file:
        logging_config["file"] = args.log_file
    if args.log_format:
        logging_config["format"] = args.log_format

    # Generate configuration
    config = generate_config(
        path=args.path,
        openai_key=openai_key,
        anthropic_key=anthropic_key,
        groq_key=groq_key,
        logging_config=logging_config,
        **custom_settings,
    )

    print("\nConfiguration generated successfully!")
    if args.path:
        print(f"Configuration saved to: {args.path}")
    else:
        print("Configuration saved to: ~/.mihrabai/config.json")

    # Show provider status
    print("\nProvider Status:")
    for provider in ["openai", "anthropic", "groq"]:
        key = config["providers"][provider]["api_key"]
        status = "Configured" if key else "Not configured"
        print(f"- {provider.title()}: {status}")

    print(f"\nDefault Provider: {config['default_provider']}")

    # Show logging status
    print("\nLogging Configuration:")
    print(f"- Level: {config['logging']['level']}")
    print(f"- File: {config['logging']['file'] or 'Console only'}")


if __name__ == "__main__":
    main()
