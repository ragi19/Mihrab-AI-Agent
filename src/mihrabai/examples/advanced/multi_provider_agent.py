"""
Example implementation of an agent that uses multiple providers
"""

from typing import Dict

from ...core.agent import Agent
from ...core.message import Message, MessageRole
from ...utils.logging import get_logger

logger = get_logger("examples.multi_provider")


class MultiProviderAgent(Agent):
    """Agent that uses multiple providers to generate responses"""

    def __init__(self, providers: Dict[str, Agent]):
        """Initialize with a dictionary of provider name to agent mappings"""
        # Use first provider's model as base model
        super().__init__(model=next(iter(providers.values())).model)
        self.providers = providers
        self.logger = get_logger("examples.multi_provider")
        self.logger.info(f"Initialized with providers: {list(providers.keys())}")

    async def process_message(self, message: Message) -> Message:
        """Process message using all providers and aggregate responses"""
        self.logger.debug(f"Processing message with {len(self.providers)} providers")

        responses = []
        errors = []

        # Get responses from all providers
        for name, agent in self.providers.items():
            try:
                response = await agent.process_message(message)
                responses.append((name, response.content))
                self.logger.debug(f"Got response from {name}: {response.content}")
            except Exception as e:
                self.logger.error(f"Error from {name}: {e}", exc_info=True)
                errors.append((name, str(e)))

        # Build aggregated response
        content_parts = []

        # Add successful responses
        if responses:
            content_parts.append("\nSUCCESSFUL RESPONSES:")
            for name, response in responses:
                content_parts.extend([f"\n{name.upper()}:", response])

        # Add error information if any
        if errors:
            content_parts.append("\nERRORS:")
            for name, error in errors:
                content_parts.append(f"{name}: {error}")

        # Create final response
        return Message(role=MessageRole.ASSISTANT, content="\n".join(content_parts))
