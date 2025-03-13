"""
Template for implementing custom model implementations
"""

from typing import Any, List, Optional

from ....core.message import Message, MessageRole
from ....core.types import ModelParameters
from ...base import BaseModel


class CustomModel(BaseModel):
    """Template for custom model implementation"""

    def __init__(
        self, model_name: str, client: Any, parameters: Optional[ModelParameters] = None
    ):
        """Initialize your model

        Args:
            model_name: Name or identifier of the model
            client: Your API client instance
            parameters: Optional model parameters
        """
        super().__init__(model_name, parameters)
        self.client = client

    async def generate_response(self, messages: List[Message]) -> Message:
        """Generate a response using your custom model

        Args:
            messages: List of messages in the conversation

        Returns:
            A message containing the model's response

        Implementation example:
        ```python
        response = await self.client.generate(
            messages=[msg.to_dict() for msg in messages],
            **self.parameters
        )
        return Message(
            role=MessageRole.ASSISTANT,
            content=response.text
        )
        ```
        """
        raise NotImplementedError("Implement generate_response for your custom model")

    async def count_tokens(self, text: str) -> int:
        """Count tokens in text using your model's tokenizer

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens in the text

        Implementation example:
        ```python
        tokenizer = self.client.get_tokenizer(self.model_name)
        return len(tokenizer.encode(text))
        ```
        """
        raise NotImplementedError("Implement count_tokens for your custom model")
