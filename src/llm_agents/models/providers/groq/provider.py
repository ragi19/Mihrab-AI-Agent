"""
Groq provider implementation
"""
from typing import Dict, List, Optional

from llm_agents.core.message import Message
from llm_agents.models.base import BaseModel
from llm_agents.utils.tracing import TraceProvider, Span
from llm_agents.core.types import ModelParameters
import groq

class GroqProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = groq.AsyncGroq(api_key=api_key)
    
    async def create_model(self, model_name: str, parameters: Optional[ModelParameters] = None) -> BaseModel:
        """Create a new Groq model instance"""
        from .models import GroqModel
        return GroqModel(model_name=model_name, client=self.client, parameters=parameters)

class GroqError(Exception):
    """Exception for Groq-related errors"""
    pass