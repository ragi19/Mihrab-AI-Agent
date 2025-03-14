"""
Template for implementing custom LLM providers
"""

from typing import Optional

from ....core.types import ModelParameters
from ...base import BaseModel


class CustomProvider:
    """Template for custom LLM provider implementation"""

    def __init__(self, **kwargs):
        """Initialize your provider with necessary configuration

        Example kwargs:
        - api_key: API key for authentication
        - api_base: Custom API endpoint
        - organization: Organization ID
        """
        self.config = kwargs
        # Initialize your API client here
        self.client = None

    async def create_model(
        self, model_name: str, parameters: Optional[ModelParameters] = None
    ) -> BaseModel:
        """Create a new model instance

        Args:
            model_name: Name or identifier of the model to use
            parameters: Optional model parameters like temperature, max_tokens, etc.

        Returns:
            An instance of your custom model implementation
        """
        from .models import CustomModel

        return CustomModel(
            model_name=model_name, client=self.client, parameters=parameters
        )
