"""
Configuration management for the framework
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict, cast

from .utils.logging import setup_logging


class ProviderConfig(TypedDict, total=False):
    """Type for provider configuration"""

    api_key: Optional[str]
    default_model: str
    default_parameters: Dict[str, Any]


class LoggingConfig(TypedDict, total=False):
    """Type for logging configuration"""

    level: str
    format: str
    file: Optional[str]


class Config:
    """Configuration manager"""

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {
            "providers": {},
            "default_provider": "openai",
            "max_history_tokens": 4000,
            "default_system_message": "You are a helpful AI assistant.",
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": None,
            },
        }
        self._config_path: Optional[Path] = None

    def load(self, path: Optional[str] = None) -> None:
        """Load configuration from file

        Args:
            path: Path to config file (default: ~/.mihrabai/config.json)
        """
        if path:
            config_path = Path(path)
        else:
            config_path = Path.home() / ".mihrabai" / "config.json"

        if config_path.exists():
            with open(config_path) as f:
                self._config.update(json.load(f))
            self._config_path = config_path

        # Configure logging
        setup_logging(
            level=self._config["logging"]["level"],
            log_file=self._config["logging"]["file"],
            format_string=self._config["logging"]["format"],
        )

    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file

        Args:
            path: Path to save config file (default: use loaded path)
        """
        if path:
            save_path = Path(path)
        elif self._config_path:
            save_path = self._config_path
        else:
            save_path = Path.home() / ".mihrabai" / "config.json"

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(self._config, f, indent=2)

    def get_provider_config(self, provider: str) -> ProviderConfig:
        """Get configuration for a specific provider"""
        return cast(ProviderConfig, self._config["providers"].get(provider, {}))

    def set_provider_config(self, provider: str, config: Dict[str, Any]) -> None:
        """Set configuration for a specific provider"""
        self._config["providers"][provider] = config

    def get_default_provider(self) -> str:
        """Get the default provider name"""
        return cast(str, self._config["default_provider"])

    def set_default_provider(self, provider: str) -> None:
        """Set the default provider name"""
        self._config["default_provider"] = provider

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration"""
        return cast(LoggingConfig, self._config["logging"])

    def set_logging_config(self, config: Dict[str, Any]) -> None:
        """Set logging configuration and reconfigure logging"""
        self._config["logging"].update(config)
        setup_logging(
            level=self._config["logging"]["level"],
            log_file=self._config["logging"]["file"],
            format_string=self._config["logging"]["format"],
        )

    @property
    def max_history_tokens(self) -> int:
        """Get maximum history tokens"""
        return cast(int, self._config["max_history_tokens"])

    @property
    def default_system_message(self) -> str:
        """Get default system message"""
        return cast(str, self._config["default_system_message"])


# Global configuration instance
config = Config()
# Load configuration from default location on import
config.load()
