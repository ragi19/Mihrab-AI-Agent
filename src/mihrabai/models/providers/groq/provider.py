"""Provider implementation for Groq models."""

from typing import Dict, List, Optional

import groq

from mihrabai.core.message import Message
from mihrabai.core.types import ModelParameters
from mihrabai.models.base import BaseModel
from mihrabai.models.provider_registry import register_provider


class GroqProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = groq.AsyncGroq(api_key=api_key)

    async def create_model(
        self, model_name: str, parameters: Optional[ModelParameters] = None
    ) -> BaseModel:
        """Create a new Groq model instance"""
        from .models import GroqModel

        return GroqModel(
            model_name=model_name, client=self.client, parameters=parameters
        )


class GroqError(Exception):
    """Exception for Groq-related errors"""

    pass
