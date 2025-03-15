import logging
from typing import Any, Dict, List, Optional, Union

from groq import APIError, Groq

from ...base import BaseModel, ModelError, TokenLimitError

logger = logging.getLogger(__name__)


class GroqError(ModelError):
    """Exception for Groq-related errors."""

    pass


class GroqAdapter:
    """Adapter for the Groq API."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = Groq(api_key=api_key)

    def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        try:
            # Create a clean messages array without any timestamp properties
            messages = [{"role": "user", "content": prompt}]
            
            # Create a clean request without any unsupported properties
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )

            return {
                "text": response.choices[0].message.content,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }
        except APIError as e:
            logger.error(f"Groq API error: {str(e)}")
            if "maximum token" in str(e).lower():
                raise TokenLimitError(str(e))
            raise GroqError(str(e))


class GroqModel(BaseModel):
    """Implementation of the Groq model."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.adapter = GroqAdapter(api_key=api_key)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
    ) -> str:
        response = self.adapter.generate(
            prompt=prompt,
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return str(response["text"])
