"""
Pytest configuration and fixtures
"""

import pytest
from typing import Dict, Any, List
from llm_agents.core.message import Message, MessageRole
from llm_agents.models.base import BaseModel


class MockModel(BaseModel):
    """Mock model for testing"""

    def __init__(self, responses: Dict[str, str] = None):
        super().__init__("mock-model")
        self.responses = {"default": "This is a mock response"}
        if responses:
            self.responses.update(responses)

    async def generate_response(self, messages: List[Message]) -> Message:
        """Generate a response using mock data"""
        last_message = messages[-1].content
        response = self.responses.get(last_message, self.responses["default"])
        return Message(role=MessageRole.ASSISTANT, content=response)

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
