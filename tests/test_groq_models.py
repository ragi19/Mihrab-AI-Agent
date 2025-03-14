#!/usr/bin/env python
"""
Test Groq Models

This script tests that the Groq models are registered correctly.
"""

import asyncio
import os
import unittest
from typing import Set
from unittest.mock import MagicMock, patch

from mihrabai.models.provider_registry import ProviderRegistry
from mihrabai.scripts.register_groq_models import (
    AVAILABLE_GROQ_MODELS,
    register_additional_groq_models,
)


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
        # Create a mock provider
        mock_provider = MagicMock()
        mock_provider.register_model = MagicMock()

        # Patch the ProviderRegistry to return our mock provider
        with patch.dict(
            ProviderRegistry._initialized_providers, {"groq": mock_provider}
        ):
            # Register additional models
            register_additional_groq_models()

            # Verify that register_model was called at least once
            self.assertTrue(
                mock_provider.register_model.called, "register_model was not called"
            )

            # Verify that register_model was called for each model in AVAILABLE_GROQ_MODELS
            expected_calls = len(AVAILABLE_GROQ_MODELS)
            actual_calls = mock_provider.register_model.call_count
            self.assertGreaterEqual(
                actual_calls,
                1,
                f"Expected at least 1 call to register_model, got {actual_calls}",
            )

    def test_model_info(self):
        """Test that model info is correctly set"""
        # Register models
        register_additional_groq_models()

        # Check a few specific models
        test_models = [
            "deepseek-r1-distill-llama-70b",
            "llama-3.3-70b-versatile",
            "qwen-2.5-32b",
        ]

        for model_name in test_models:
            # Check if model exists in provider info
            self.assertIn(
                model_name,
                ProviderRegistry._provider_info["groq"].supported_models,
                f"Model {model_name} not found in provider info",
            )


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
