"""
Unit tests for the Agent base class
"""
import pytest
from llm_agents.core.agent import Agent
from llm_agents.core.message import Message, MessageRole

class SimpleTestAgent(Agent):
    async def process_message(self, message: Message) -> Message:
        return await self.model.generate_response(self.conversation_history + [message])

@pytest.mark.asyncio
async def test_agent_initialization(mock_model):
    """Test agent initialization"""
    agent = SimpleTestAgent(model=mock_model)
    assert agent.model == mock_model
    assert len(agent.conversation_history) == 0

@pytest.mark.asyncio
async def test_agent_message_processing(mock_model):
    """Test agent message processing"""
    agent = SimpleTestAgent(model=mock_model)
    message = Message(role=MessageRole.USER, content="Hello")
    
    response = await agent.process_message(message)
    assert response.role == MessageRole.ASSISTANT
    assert isinstance(response.content, str)

@pytest.mark.asyncio
async def test_agent_conversation_history(mock_model):
    """Test agent conversation history management"""
    agent = SimpleTestAgent(model=mock_model)
    message = Message(role=MessageRole.USER, content="Hello")
    
    # Add message to history
    agent.add_to_history(message)
    assert len(agent.conversation_history) == 1
    assert agent.conversation_history[0] == message
    
    # Get history
    history = agent.get_history()
    assert len(history) == 1
    assert history[0] == message
    
    # Clear history
    agent.clear_history()
    assert len(agent.conversation_history) == 0

@pytest.mark.asyncio
async def test_agent_custom_responses(custom_mock_model):
    """Test agent with custom model responses"""
    responses = {
        "Hello": "Hi there!",
        "How are you?": "I'm doing great!"
    }
    model = custom_mock_model(responses)
    agent = SimpleTestAgent(model=model)
    
    # Test specific responses
    message1 = Message(role=MessageRole.USER, content="Hello")
    response1 = await agent.process_message(message1)
    assert response1.content == "Hi there!"
    
    message2 = Message(role=MessageRole.USER, content="How are you?")
    response2 = await agent.process_message(message2)
    assert response2.content == "I'm doing great!"