#!/usr/bin/env python
"""
Register Additional Groq Models

This script registers additional Groq models that may not be included in the default provider.
It can be run directly or imported and used programmatically.
"""

# Re-export from the actual implementation
from mihrabai.models.providers.groq.register_groq_models import (
    AVAILABLE_GROQ_MODELS,
    MODEL_INFO,
    list_available_models,
    register_additional_groq_models,
)

if __name__ == "__main__":
    # Run the registration process
    register_additional_groq_models()
    list_available_models()
