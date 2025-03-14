"""
Pytest configuration and fixtures
"""

from typing import Any, AsyncIterator, Dict, List, Set

import pytest

from mihrabai.core.message import Message, MessageRole
from mihrabai.models.base import BaseModel, ModelCapability


class MockModel(BaseModel):
    """Mock model for testing"""

    def __init__(self, responses: Dict[str, str] = None):
        super().__init__("mock-model")
        self.responses = {"default": "This is a mock response"}
        if responses:
            self.responses.update(responses)
        self._capabilities = {"chat", "streaming"}
        self.model_name = "mock-model"

    @property
    def capabilities(self) -> Set[str]:
        """Get the capabilities of this model"""
        return self._capabilities

    async def generate(self, messages: List[Message], **kwargs: Any) -> Message:
        """Generate a response from the model"""
        last_message = messages[-1].content
        response = self.responses.get(last_message, self.responses["default"])
        return Message(role=MessageRole.ASSISTANT, content=response)

    async def generate_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> AsyncIterator[Message]:
        """Stream a response from the model"""
        response = await self.generate(messages, **kwargs)
        # Split the response into chunks for streaming
        words = response.content.split()
        for i in range(0, len(words), 2):
            chunk = " ".join(words[i : i + 2])
            yield Message(role=MessageRole.ASSISTANT, content=chunk)

    async def generate_response(self, messages: List[Message]) -> Message:
        """Generate a response using mock data"""
        return await self.generate(messages)

    async def _generate(self, messages: List[Message]) -> Message:
        """Internal method for model-specific response generation"""
        last_message = messages[-1].content
        response = self.responses.get(last_message, self.responses["default"])
        return Message(role=MessageRole.ASSISTANT, content=response)

    async def _prepare_messages(self, messages: List[Message]) -> List[Dict]:
        """Convert messages to mock format"""
        return [msg.to_dict() for msg in messages]

    async def _make_api_call(self, messages: List[Dict]) -> Message:
        """Mock API call"""
        last_message = messages[-1]["content"]
        response = self.responses.get(last_message, self.responses["default"])
        return Message(role=MessageRole.ASSISTANT, content=response)

    async def count_tokens(self, text: str) -> int:
        """Mock token counting"""
        return len(text.split())


@pytest.fixture
def mock_model():
    """Fixture providing a mock model for testing"""
    return MockModel()


@pytest.fixture
def custom_mock_model():
    """Fixture providing a mock model with custom responses"""

    def _create_mock_model(responses: Dict[str, str]):
        return MockModel(responses=responses)

    return _create_mock_model
