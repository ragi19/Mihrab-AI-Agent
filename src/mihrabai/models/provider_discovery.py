"""
Provider discovery and automatic registration
"""

import importlib
import os
import pkgutil
from typing import Dict, List, Optional, Type

from ..utils.logging import get_logger
from .provider_registry import ProviderRegistry
from .providers.base import BaseProvider

logger = get_logger("models.provider_discovery")


class ProviderDiscovery:
    """Discovers and registers available model providers"""

    @classmethod
    def discover_providers(cls) -> Dict[str, Type[BaseProvider]]:
        """Discover available providers in the providers package

        Returns:
            Dictionary mapping provider names to provider classes
        """
        providers = {}

        try:
            # Get the providers package directory
            import mihrabai.models.providers as providers_pkg

            pkg_dir = os.path.dirname(providers_pkg.__file__)

            # Find all provider modules
            for _, name, is_pkg in pkgutil.iter_modules([pkg_dir]):
                if is_pkg and name not in ["base", "custom"]:
                    try:
                        # Import the provider module
                        module = importlib.import_module(
                            f"mihrabai.models.providers.{name}"
                        )

                        # Look for provider class
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (
                                isinstance(attr, type)
                                and issubclass(attr, BaseProvider)
                                and attr != BaseProvider
                            ):
                                providers[name] = attr
                                logger.debug(f"Discovered provider: {name}")
                                break

                    except Exception as e:
                        logger.warning(f"Error loading provider {name}: {e}")

            logger.info(f"Discovered {len(providers)} providers")
            return providers

        except Exception as e:
            logger.error(f"Error discovering providers: {e}")
            return {}

    @classmethod
    def register_discovered_providers(cls) -> None:
        """Discover and register all available providers"""
        providers = cls.discover_providers()

        for name, provider_class in providers.items():
            try:
                # Get provider info from the class
                info = provider_class.get_provider_info()

                # Register with the provider registry
                ProviderRegistry.register_provider(name, provider_class, info)
                logger.info(f"Registered provider: {name}")

            except Exception as e:
                logger.error(f"Error registering provider {name}: {e}")

    @classmethod
    def load_provider_configurations(cls) -> None:
        """Load configurations for all discovered providers"""
        from .provider_config import ProviderConfigManager

        config_manager = ProviderConfigManager()
        providers = cls.discover_providers()

        for name in providers:
            if config := config_manager.get_config(name):
                logger.debug(f"Loaded configuration for provider: {name}")
            else:
                logger.debug(f"No configuration found for provider: {name}")

    @classmethod
    def initialize(cls) -> None:
        """Initialize provider discovery and registration"""
        # Discover and register providers
        cls.register_discovered_providers()

        # Load provider configurations
        cls.load_provider_configurations()
