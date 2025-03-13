"""
HandoffAgent implementation for multi-agent systems
"""

from typing import Dict, Any, List, Optional, Union, Tuple, Callable

from ..core.memory import Memory
from ..core.memory_task_agent import MemoryEnabledTaskAgent
from ..runtime.memory_runner import MemoryAgentRunner
from ..core.message import Message, MessageRole
from ..models.base import BaseModel
from ..utils.logging import get_logger

from .config import HandoffConfig, HandoffInputData

logger = get_logger("handoff.agent")


class HandoffAgent:
    """
    Agent with handoff capabilities

    This class extends the base agent functionality with the ability to
    transfer control to other agents based on defined rules.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: Optional[List[Any]] = None,
        handoffs: Optional[List[HandoffConfig]] = None,
        can_defer: bool = False,
        expertise: Optional[List[str]] = None,
    ):
        """
        Initialize a handoff-enabled agent

        Args:
            name: Name of the agent
            system_prompt: System prompt for the agent
            tools: Optional list of tools available to the agent
            handoffs: Optional list of handoff configurations
            can_defer: Whether this agent can defer to other agents
            expertise: Optional list of areas of expertise
        """
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.handoffs = handoffs or []
        self.agent = None
        self.runner = None
        self.can_defer = can_defer
        self.expertise = expertise or []
        self.conversation_context = {}  # Store context between conversations
        logger.info(f"Created HandoffAgent: {name}")

    async def initialize(self, model: BaseModel):
        """
        Initialize the agent with a model

        Args:
            model: The language model to use for this agent

        Raises:
            Exception: If initialization fails
        """
        logger.info(f"Initializing agent: {self.name}")
        try:
            # Create memory
            memory = Memory(working_memory_size=10, long_term_memory_size=50)

            # Create agent
            self.agent = MemoryEnabledTaskAgent(
                model=model,
                memory=memory,
                system_message=self.system_prompt,
                tools=self.tools,
            )

            # Create runner
            self.runner = MemoryAgentRunner(
                agent=self.agent, memory_persistence_path="./memories"
            )

            logger.info(f"Successfully initialized agent: {self.name}")
        except Exception as e:
            logger.error(
                f"Error initializing agent {self.name}: {str(e)}", exc_info=True
            )
            raise

    async def process(
        self,
        message: str,
        session_id: str = None,
        context: Optional[Dict[str, Any]] = None,
        handoff_chain: Optional[List[str]] = None,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """
        Process a message with potential handoffs

        Args:
            message: The user message to process
            session_id: Optional session ID for persistence
            context: Optional context from previous agents
            handoff_chain: Optional list of agents in the handoff chain

        Returns:
            Either a response string or a tuple of (response, context)
        """
        handoff_chain = handoff_chain or []
        if self.name not in handoff_chain:
            handoff_chain.append(self.name)

        logger.info(f"Agent {self.name} processing message: {message[:50]}...")
        logger.debug(f"Current handoff chain: {' -> '.join(handoff_chain)}")

        # Update conversation context
        if context:
            self.conversation_context.update(context)
            logger.debug(f"Updated context for {self.name}: {context}")

        # Check if we should hand off to a specialized agent
        for handoff in self.handoffs:
            if self._should_handoff(message, handoff, context):
                logger.info(
                    f"Handing off from {self.name} to {handoff.target_agent.name}"
                )

                # Prepare context to pass along
                handoff_context = self.conversation_context.copy()
                handoff_context.update(
                    {
                        "source_agent": self.name,
                        "handoff_reason": handoff.description,
                        "handoff_metadata": handoff.metadata,
                    }
                )

                # Process with target agent
                result = await handoff.target_agent.process(
                    message,
                    session_id=session_id,
                    context=handoff_context,
                    handoff_chain=handoff_chain.copy(),
                )

                # Handle tuple result (response and context)
                if isinstance(result, tuple):
                    response, returned_context = result
                    # Update our context with information from the target agent
                    self.conversation_context.update(returned_context)
                    return response, self.conversation_context
                return result

        # Process with current agent if no handoff needed
        logger.info(f"No handoff needed, {self.name} processing directly")
        try:
            # Add context to the message if available
            enhanced_message = message
            if self.conversation_context:
                context_str = "\n\nContext from previous agents:\n"
                for key, value in self.conversation_context.items():
                    if isinstance(value, str):
                        context_str += f"- {key}: {value}\n"
                enhanced_message = context_str + "\n" + message

            response = await self.runner.run(enhanced_message, session_id=session_id)
            logger.info(f"Agent {self.name} response: {response.content[:50]}...")

            # Extract any information to add to context
            extracted_context = self._extract_context_from_response(response.content)
            if extracted_context:
                self.conversation_context.update(extracted_context)

            return response.content, self.conversation_context
        except Exception as e:
            logger.error(
                f"Error in agent {self.name} processing: {str(e)}", exc_info=True
            )
            return f"Error processing your request: {str(e)}"

    def _should_handoff(
        self,
        message: str,
        handoff: HandoffConfig,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Determine if message should be handed off to another agent

        Args:
            message: The user message
            handoff: The handoff configuration to check
            context: Optional context from previous agents

        Returns:
            True if the message should be handed off, False otherwise
        """
        context = context or {}

        # If there's a custom condition function, use it
        if handoff.condition:
            return handoff.condition(message, context)

        # Default keyword-based handoff logic
        keywords = {
            "technical": [
                "error",
                "bug",
                "broken",
                "not working",
                "help with",
                "how to",
                "fix",
                "troubleshoot",
            ],
            "billing": [
                "payment",
                "invoice",
                "charge",
                "refund",
                "price",
                "cost",
                "subscription",
                "plan",
            ],
            "scheduling": [
                "schedule",
                "appointment",
                "book",
                "meeting",
                "when",
                "time",
                "calendar",
                "availability",
            ],
            "research": [
                "research",
                "information",
                "data",
                "analysis",
                "study",
                "investigate",
                "find out",
            ],
            "legal": [
                "legal",
                "law",
                "regulation",
                "compliance",
                "terms",
                "policy",
                "agreement",
                "contract",
            ],
            "sales": [
                "purchase",
                "buy",
                "order",
                "product",
                "service",
                "pricing",
                "discount",
                "deal",
            ],
        }

        if handoff.name in keywords:
            return any(kw in message.lower() for kw in keywords[handoff.name])
        return False

    def _extract_context_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extract structured information from the response to add to context

        Args:
            response: The agent's response

        Returns:
            Dictionary of extracted context
        """
        import re
        import json

        context = {}

        # Look for key-value pairs in the format "Key: Value"
        kv_pattern = r"([A-Za-z\s]+):\s*([^:\n]+)"
        kv_matches = re.findall(kv_pattern, response)
        for key, value in kv_matches:
            clean_key = key.strip().lower().replace(" ", "_")
            clean_value = value.strip()
            if clean_key and clean_value:
                context[clean_key] = clean_value

        # Look for JSON-like structures
        json_pattern = r"\{[\s\S]*?\}"
        json_matches = re.findall(json_pattern, response)
        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                if isinstance(data, dict):
                    context.update(data)
            except:
                pass

        return context
