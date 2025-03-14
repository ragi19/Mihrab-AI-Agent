#!/usr/bin/env python
"""
Register Additional Groq Models

This script registers additional Groq models that may not be included in the default provider.
It can be run directly or imported and used programmatically.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import required modules
from mihrabai.models.base import ModelCapability, ModelInfo
from mihrabai.models.provider_registry import ProviderRegistry
from mihrabai.models.providers.groq.models import GroqModel

# List of available Groq models
AVAILABLE_GROQ_MODELS = [
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b",
    "distil-whisper-large-v3-en",
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
    "llama-3.3-70b-specdec",
    "llama-3.3-70b-versatile",
    "llama-guard-3-8b",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mistral-saba-24b",
    "qwen-2.5-32b",
    "qwen-2.5-coder-32b",
    "qwen-qwq-32b",
]

# Model information mapping
MODEL_INFO = {
    "deepseek-r1-distill-llama-70b": {
        "context_window": 32768,
        "max_tokens": 4096,
        "capabilities": {ModelCapability.CHAT, ModelCapability.STREAMING},
    },
    "deepseek-r1-distill-qwen-32b": {
        "context_window": 32768,
        "max_tokens": 4096,
        "capabilities": {ModelCapability.CHAT, ModelCapability.STREAMING},
    },
    "distil-whisper-large-v3-en": {
        "context_window": 8192,
        "max_tokens": 4096,
        "capabilities": {ModelCapability.CHAT, ModelCapability.STREAMING},
    },
    "llama-3.2-1b-preview": {
        "context_window": 8192,
        "max_tokens": 4096,
        "capabilities": {ModelCapability.CHAT, ModelCapability.STREAMING},
    },
    "llama-3.2-3b-preview": {
        "context_window": 8192,
        "max_tokens": 4096,
        "capabilities": {ModelCapability.CHAT, ModelCapability.STREAMING},
    },
    "llama-3.2-11b-vision-preview": {
        "context_window": 8192,
        "max_tokens": 4096,
        "capabilities": {
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.IMAGE_INPUT,
        },
    },
    "llama-3.2-90b-vision-preview": {
        "context_window": 8192,
        "max_tokens": 4096,
        "capabilities": {
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.IMAGE_INPUT,
        },
    },
    "llama-3.3-70b-specdec": {
        "context_window": 8192,
        "max_tokens": 4096,
        "capabilities": {ModelCapability.CHAT, ModelCapability.STREAMING},
    },
    "llama-guard-3-8b": {
        "context_window": 8192,
        "max_tokens": 4096,
        "capabilities": {ModelCapability.CHAT, ModelCapability.STREAMING},
    },
    "mistral-saba-24b": {
        "context_window": 8192,
        "max_tokens": 4096,
        "capabilities": {ModelCapability.CHAT, ModelCapability.STREAMING},
    },
    "qwen-2.5-32b": {
        "context_window": 32768,
        "max_tokens": 4096,
        "capabilities": {ModelCapability.CHAT, ModelCapability.STREAMING},
    },
    "qwen-2.5-coder-32b": {
        "context_window": 32768,
        "max_tokens": 4096,
        "capabilities": {ModelCapability.CHAT, ModelCapability.STREAMING},
    },
    "qwen-qwq-32b": {
        "context_window": 32768,
        "max_tokens": 4096,
        "capabilities": {ModelCapability.CHAT, ModelCapability.STREAMING},
    },
}


def register_additional_groq_models() -> int:
    """Register additional Groq models that aren't in the default provider"""
    # Get the Groq provider
    groq_provider = None
    for provider_name in ProviderRegistry.list_providers():
        if provider_name == "groq":
            groq_provider = ProviderRegistry._initialized_providers.get("groq")
            break

    if not groq_provider:
        logger.warning("Groq provider not found, cannot register additional models")
        return 0

    # Define additional models
    additional_models = {}
    for model_name, info in MODEL_INFO.items():
        additional_models[model_name] = ModelInfo(
            id=model_name,
            name=model_name,
            provider="groq",
            capabilities=info["capabilities"],
            context_window=info["context_window"],
            max_tokens=info["max_tokens"],
        )

    # Register each model with the Groq provider
    registered_count = 0
    for model_name, model_info in additional_models.items():
        if hasattr(groq_provider, "register_model"):
            groq_provider.register_model(model_name, GroqModel, model_info)
            logger.info(f"Registered additional Groq model: {model_name}")
            registered_count += 1
        else:
            logger.warning(
                f"Could not register model {model_name}, register_model method not found"
            )

    # Update the provider info in the registry
    if (
        hasattr(ProviderRegistry, "_provider_info")
        and "groq" in ProviderRegistry._provider_info
    ):
        provider_info = ProviderRegistry._provider_info["groq"]
        if hasattr(provider_info, "supported_models"):
            for model_name in additional_models:
                provider_info.supported_models.add(model_name)
            logger.info(
                f"Updated Groq provider supported models: {provider_info.supported_models}"
            )

    logger.info(f"Successfully registered {registered_count} additional Groq models")
    return registered_count


def list_available_models() -> Set[str]:
    """List all available Groq models"""
    if "groq" in ProviderRegistry._provider_info:
        supported_models = ProviderRegistry._provider_info["groq"].supported_models
        print("\nAvailable Groq models:")
        for i, model in enumerate(sorted(supported_models)):
            print(f"{i+1}. {model}")
        return set(supported_models)
    return set()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Groq Model Registration Utility")
    print("=" * 80)

    # Check for API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\nWarning: GROQ_API_KEY environment variable not found!")
        print("Please set your Groq API key using:")
        print("$env:GROQ_API_KEY = 'your-api-key'")

    # Register additional models
    print("\nRegistering additional Groq models...")
    count = register_additional_groq_models()

    if count > 0:
        print(f"\nSuccessfully registered {count} additional Groq models")

        # List all available models
        list_available_models()
    else:
        print("\nNo additional models were registered")

    print("\nDone!")
