"""
Configuration file generator utility
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Define the type for the config dictionary
ConfigDict = Dict[str, Any]

DEFAULT_CONFIG: ConfigDict = {
    "providers": {
        "openai": {
            "api_key": None,
            "default_model": "gpt-3.5-turbo",
            "default_parameters": {"temperature": 0.7, "max_tokens": 1000},
        },
        "anthropic": {
            "api_key": None,
            "default_model": "claude-3-opus-20240229",
            "default_parameters": {"temperature": 0.7, "max_tokens": 1000},
        },
        "groq": {
            "api_key": None,
            "default_model": "llama2-70b-4096",
            "default_parameters": {"temperature": 0.7, "max_tokens": 1000},
        },
    },
    "default_provider": "openai",
    "max_history_tokens": 4000,
    "default_system_message": "You are a helpful AI assistant.",
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": None,
    },
}


def generate_config(
    path: Optional[str] = None,
    openai_key: Optional[str] = None,
    anthropic_key: Optional[str] = None,
    groq_key: Optional[str] = None,
    logging_config: Optional[Dict[str, Any]] = None,
    **custom_settings: Any,
) -> ConfigDict:
    """Generate a configuration file with the provided settings

    Args:
        path: Path to save the configuration file (default: ~/.mihrabai/config.json)
        openai_key: OpenAI API key
        anthropic_key: Anthropic API key
        groq_key: Groq API key
        logging_config: Custom logging configuration
        **custom_settings: Additional custom settings to include

    Returns:
        The generated configuration dictionary
    """
    # Start with default config
    config: ConfigDict = DEFAULT_CONFIG.copy()

    # Update API keys if provided
    if openai_key:
        config["providers"]["openai"]["api_key"] = openai_key
    if anthropic_key:
        config["providers"]["anthropic"]["api_key"] = anthropic_key
    if groq_key:
        config["providers"]["groq"]["api_key"] = groq_key

    # Update logging configuration if provided
    if logging_config:
        config["logging"].update(logging_config)

    # Update with any custom settings
    config.update(custom_settings)

    # Save configuration if path is provided
    if path:
        save_path = Path(path)
    else:
        save_path = Path.home() / ".mihrabai" / "config.json"

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(config, f, indent=2)

    return config


def load_environment_keys() -> Dict[str, Optional[str]]:
    """Load API keys from environment variables"""
    return {
        "openai_key": os.getenv("OPENAI_API_KEY"),
        "anthropic_key": os.getenv("ANTHROPIC_API_KEY"),
        "groq_key": os.getenv("GROQ_API_KEY"),
    }
