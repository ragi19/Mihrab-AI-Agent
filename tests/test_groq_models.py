#!/usr/bin/env python
"""
Test Groq Models

This script tests that the Groq models are registered correctly.
"""

import os
import asyncio
import unittest
from typing import Set

from llm_agents.models.provider_registry import ProviderRegistry
from llm_agents.scripts.register_groq_models import register_additional_groq_models, AVAILABLE_GROQ_MODELS

class TestGroqModels(unittest.TestCase):
    """Test case for Groq models"""
    
    def setUp(self):
        """Set up the test case"""
        # Skip tests if GROQ_API_KEY is not set
        if not os.environ.get("GROQ_API_KEY"):
            self.skipTest("GROQ_API_KEY environment variable not set")
    
    def test_groq_provider_exists(self):
        """Test that the Groq provider exists"""
        providers = ProviderRegistry.list_providers()
        self.assertIn("groq", providers, "Groq provider not found in registry")
    
    def test_register_additional_models(self):
        """Test registering additional Groq models"""
        # Get initial models
        initial_models = set()
        if "groq" in ProviderRegistry._provider_info:
            initial_models = set(ProviderRegistry._provider_info["groq"].supported_models)
        
        # Register additional models
        register_additional_groq_models()
        
        # Get updated models
        updated_models = set()
        if "groq" in ProviderRegistry._provider_info:
            updated_models = set(ProviderRegistry._provider_info["groq"].supported_models)
        
        # Check that new models were added
        self.assertGreater(len(updated_models), len(initial_models), 
                          "No new models were added")
        
        # Check that all models in AVAILABLE_GROQ_MODELS are registered
        for model in AVAILABLE_GROQ_MODELS:
            self.assertIn(model, updated_models, 
                         f"Model {model} not found in registered models")
    
    def test_model_info(self):
        """Test that model info is correctly set"""
        # Register models
        register_additional_groq_models()
        
        # Check a few specific models
        test_models = [
            "deepseek-r1-distill-llama-70b",
            "llama-3.3-70b-versatile",
            "qwen-2.5-32b"
        ]
        
        for model_name in test_models:
            # Check if model exists in provider info
            self.assertIn(model_name, ProviderRegistry._provider_info["groq"].supported_models,
                         f"Model {model_name} not found in provider info")

async def async_main():
    """Run the tests asynchronously"""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGroqModels)
    
    # Run tests
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    
    # Return success/failure
    return result.wasSuccessful()

def main():
    """Main function"""
    success = asyncio.run(async_main())
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 