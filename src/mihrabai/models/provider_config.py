"""
Provider configuration management
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set

from ..utils.logging import get_logger

logger = get_logger("models.provider_config")


@dataclass
class ProviderConfig:
    """Configuration for a model provider"""

    name: str
    api_key: Optional[str] = None
    default_parameters: Optional[Dict[str, Any]] = None
    rate_limits: Optional[Dict[str, Any]] = None
    env_vars: Optional[Dict[str, str]] = None


class ProviderConfigManager:
    """Manages provider configurations"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._configs: Dict[str, ProviderConfig] = {}
        self._load_configs()

    def _get_default_config_path(self) -> str:
        """Get default configuration path"""
        return os.path.expanduser("~/.mihrabai/provider_config.json")

    def _load_configs(self) -> None:
        """Load provider configurations from file"""
        if not os.path.exists(self.config_path):
            logger.info(f"No config file found at {self.config_path}")
            return

        try:
            with open(self.config_path, "r") as f:
                data = json.load(f)

            for provider_name, config in data.items():
                self._configs[provider_name] = ProviderConfig(
                    name=provider_name, **config
                )
                logger.debug(f"Loaded config for provider: {provider_name}")

        except Exception as e:
            logger.error(f"Error loading provider configs: {e}")

    def save_configs(self) -> None:
        """Save provider configurations to file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        try:
            data = {
                name: {
                    "api_key": config.api_key,
                    "default_parameters": config.default_parameters,
                    "rate_limits": config.rate_limits,
                    "env_vars": config.env_vars,
                }
                for name, config in self._configs.items()
            }

            with open(self.config_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved provider configs to {self.config_path}")

        except Exception as e:
            logger.error(f"Error saving provider configs: {e}")

    def get_config(self, provider_name: str) -> Optional[ProviderConfig]:
        """Get configuration for a provider"""
        config = self._configs.get(provider_name)

        if config and config.env_vars:
            # Update values from environment variables
            for key, env_var in config.env_vars.items():
                if env_value := os.getenv(env_var):
                    if key == "api_key":
                        config.api_key = env_value
                    elif key == "default_parameters":
                        try:
                            config.default_parameters = json.loads(env_value)
                        except:
                            pass

        return config

    def set_config(self, provider_name: str, config: ProviderConfig) -> None:
        """Set configuration for a provider"""
        self._configs[provider_name] = config
        logger.info(f"Updated config for provider: {provider_name}")

    def remove_config(self, provider_name: str) -> None:
        """Remove configuration for a provider"""
        if provider_name in self._configs:
            del self._configs[provider_name]
            logger.info(f"Removed config for provider: {provider_name}")

    def list_providers(self) -> Set[str]:
        """List all configured providers"""
        return set(self._configs.keys())

    def clear_configs(self) -> None:
        """Clear all provider configurations"""
        self._configs.clear()
        logger.info("Cleared all provider configs")
