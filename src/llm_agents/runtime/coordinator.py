"""
Agent coordination utilities for multi-agent workflows
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Union

from ..core.agent import Agent
from ..core.memory_task_agent import MemoryEnabledTaskAgent
from ..core.message import Message, MessageRole
from ..core.task_agent import TaskAgent
from ..utils.logging import get_logger
from ..utils.tracing import Trace, TraceProvider
from .context import RuntimeContext

logger = get_logger("runtime.coordinator")


@dataclass
class AgentConfig:
    """Configuration for an agent in a coordinated group"""

    agent: Agent
    name: str
    description: str
    capabilities: Set[str] = field(default_factory=set)
    can_initiate: bool = False
    priority: int = 0
    max_failures: int = 3

    # Runtime state
    failures: int = 0
    last_error: Optional[str] = None


@dataclass
class AgentMessage:
    """Message from one agent to another in a coordinated group"""

    from_agent: str
    to_agent: str
    message: Message
    id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


class CoordinatedAgentGroup:
    """Manages a group of agents that work together

    This class handles:
    - Message routing between agents
    - Agent coordination based on capabilities
    - Fallback mechanisms when agents fail
    - Context sharing between agents
    - Workflow orchestration
    """

    def __init__(
        self,
        context: Optional[RuntimeContext] = None,
        trace_provider: Optional[TraceProvider] = None,
        coordinator_system_message: str = "You are the coordinator that manages conversation flow between multiple specialized agents.",
    ):
        self.agents: Dict[str, AgentConfig] = {}
        self.context = context or RuntimeContext()
        self.message_history: List[AgentMessage] = []
        self._trace_provider = trace_provider or TraceProvider()
        self._current_trace: Optional[Trace] = None
        self._coordinator_system_message = coordinator_system_message
        self.logger = get_logger("runtime.coordinator")

    def add_agent(
        self,
        agent: Agent,
        name: str,
        description: str,
        capabilities: Optional[Set[str]] = None,
        can_initiate: bool = False,
        priority: int = 0,
    ) -> None:
        """Add an agent to the group

        Args:
            agent: Agent instance
            name: Unique name for the agent
            description: Description of the agent's role
            capabilities: Set of capabilities the agent provides
            can_initiate: Whether this agent can initiate interactions
            priority: Priority level (higher means higher priority)
        """
        if name in self.agents:
            raise ValueError(f"Agent '{name}' already exists in this group")

        self.agents[name] = AgentConfig(
            agent=agent,
            name=name,
            description=description,
            capabilities=capabilities or set(),
            can_initiate=can_initiate,
            priority=priority,
        )
        self.logger.info(f"Added agent '{name}' to group")

    def remove_agent(self, name: str) -> None:
        """Remove an agent from the group"""
        if name in self.agents:
            del self.agents[name]
            self.logger.info(f"Removed agent '{name}' from group")

    async def process_message(
        self, message: Message, target_agent: Optional[str] = None
    ) -> Message:
        """Process a message through the appropriate agent(s)

        Args:
            message: Input message
            target_agent: Optional target agent name

        Returns:
            Final response message
        """
        self._current_trace = self._trace_provider.create_trace(
            name="coordinator.process_message", group_id=f"coord_{id(self)}"
        )
        self._current_trace.start()

        try:
            # If target agent specified, route directly to it
            if target_agent:
                if target_agent not in self.agents:
                    raise ValueError(f"Target agent '{target_agent}' not found")
                return await self._route_to_agent(message, target_agent)

            # Otherwise, determine the appropriate agent(s) for this message
            target = await self._determine_target_agent(message)
            self.logger.info(f"Routing message to agent '{target}'")

            # Process with target agent
            return await self._route_to_agent(message, target)

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            self._trace_provider.end_trace(self._current_trace, error=e)
            raise
        finally:
            if self._current_trace:
                self._trace_provider.end_trace(self._current_trace)
            self._current_trace = None

    async def _route_to_agent(self, message: Message, agent_name: str) -> Message:
        """Route a message to a specific agent

        Args:
            message: Message to route
            agent_name: Name of the target agent

        Returns:
            Agent's response message
        """
        agent_config = self.agents[agent_name]
        agent = agent_config.agent

        try:
            # Prepare agent message
            agent_message = AgentMessage(
                from_agent="user" if message.role == MessageRole.USER else "system",
                to_agent=agent_name,
                message=message,
            )

            # Add to history
            self.message_history.append(agent_message)

            # Process with agent
            response = await agent.process_message(message)

            # Record response
            response_message = AgentMessage(
                from_agent=agent_name, to_agent="user", message=response
            )
            self.message_history.append(response_message)

            # Reset failure count on success
            agent_config.failures = 0

            return response

        except Exception as e:
            # Track failure
            agent_config.failures += 1
            agent_config.last_error = str(e)
            self.logger.error(f"Agent '{agent_name}' failed: {e}")

            # Try fallback if available
            if agent_config.failures >= agent_config.max_failures:
                fallback = await self._find_fallback_agent(agent_name)
                if fallback:
                    self.logger.info(
                        f"Using fallback agent '{fallback}' after {agent_config.failures} failures"
                    )
                    return await self._route_to_agent(message, fallback)

            raise

    async def _determine_target_agent(self, message: Message) -> str:
        """Determine the most appropriate agent for a message

        This uses a coordinator model to decide which agent should handle
        the message based on content and agent capabilities.

        Args:
            message: User message

        Returns:
            Name of the most appropriate agent
        """
        # Simple content-based routing first (fast path)
        target = await self._simple_agent_routing(message.content)
        if target:
            return target

        # Create prompt for coordinator
        agent_descriptions = []
        for name, config in self.agents.items():
            caps = ", ".join(config.capabilities) if config.capabilities else "None"
            agent_descriptions.append(
                f"- {name}: {config.description} (Capabilities: {caps})"
            )

        prompt = f"""Based on the user's message, determine which agent is best suited to respond.

Available agents:
{chr(10).join(agent_descriptions)}

User message: "{message.content}"

Respond with ONLY the name of the most appropriate agent."""

        # Create coordinator messages
        coord_messages = [
            Message(role=MessageRole.SYSTEM, content=self._coordinator_system_message),
            Message(role=MessageRole.USER, content=prompt),
        ]

        # Use primary agent's model for coordination decision
        primary_agent = self._get_primary_agent()
        response = await primary_agent.agent.model.generate_response(coord_messages)

        # Extract agent name from response
        agent_name = response.content.strip().lower()

        # Validate agent exists, default to primary if not
        if agent_name not in self.agents:
            self.logger.warning(
                f"Coordinator selected unknown agent '{agent_name}', using primary"
            )
            return primary_agent.name

        return agent_name

    async def _simple_agent_routing(self, content: str) -> Optional[str]:
        """Do simple keyword-based routing for common cases

        This provides a fast path for obvious routing decisions without
        needing to call the coordinator model.
        """
        content_lower = content.lower()

        # Check for explicit agent requests
        for name, config in self.agents.items():
            if f"@{name.lower()}" in content_lower:
                return name

        # Check simple keyword matches
        keyword_map = {
            "search": {"search", "find", "look up"},
            "code": {"code", "programming", "function", "class"},
            "math": {"calculate", "math", "equation"},
            "planning": {"plan", "schedule", "timeline"},
        }

        for agent_type, keywords in keyword_map.items():
            for keyword in keywords:
                if keyword in content_lower:
                    # Find an agent with matching capability or name
                    for name, config in self.agents.items():
                        if (
                            agent_type in config.capabilities
                            or agent_type in name.lower()
                        ):
                            return name

        return None

    async def _find_fallback_agent(self, failed_agent: str) -> Optional[str]:
        """Find a fallback agent when the primary agent fails"""
        failed_config = self.agents[failed_agent]

        # Find agent with most overlapping capabilities
        candidates = []
        for name, config in self.agents.items():
            if name == failed_agent:
                continue

            overlap = len(failed_config.capabilities.intersection(config.capabilities))
            candidates.append((name, overlap, config.priority))

        # Sort by capability overlap then priority
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)

        if candidates:
            return candidates[0][0]

        return None

    def _get_primary_agent(self) -> AgentConfig:
        """Get the primary (highest priority) agent"""
        primary = max(self.agents.values(), key=lambda a: a.priority)
        return primary

    async def broadcast(self, message: Message) -> Dict[str, Message]:
        """Broadcast a message to all agents

        Args:
            message: Message to broadcast

        Returns:
            Dictionary mapping agent names to their responses
        """
        tasks = {name: self._route_to_agent(message, name) for name in self.agents}

        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                self.logger.error(f"Agent '{name}' failed during broadcast: {e}")
                results[name] = Message(
                    role=MessageRole.ASSISTANT, content=f"Error: {str(e)}"
                )

        return results

    async def workflow(
        self, message: Message, steps: List[str], combine_responses: bool = True
    ) -> Union[Message, List[Message]]:
        """Execute a predefined workflow across multiple agents

        Args:
            message: Initial input message
            steps: List of agent names to execute in order
            combine_responses: Whether to combine responses into a single message

        Returns:
            Final response message or list of response messages
        """
        responses = []
        current_message = message

        for agent_name in steps:
            if agent_name not in self.agents:
                raise ValueError(f"Agent '{agent_name}' not found in workflow")

            try:
                response = await self._route_to_agent(current_message, agent_name)
                responses.append(response)
                current_message = response
            except Exception as e:
                self.logger.error(f"Workflow failed at step '{agent_name}': {e}")
                raise

        if combine_responses:
            # Combine all responses into a single message
            combined_content = "\n\n".join([r.content for r in responses])
            return Message(role=MessageRole.ASSISTANT, content=combined_content)

        return responses

    def get_conversation_context(self) -> Dict[str, Any]:
        """Get the current conversation context"""
        return self.context.get_all()

    def update_context(self, key: str, value: Any) -> None:
        """Update a value in the shared context"""
        self.context.set(key, value)

    def get_message_history(self) -> List[Dict[str, Any]]:
        """Get the message routing history

        Returns:
            List of message routing records
        """
        return [
            {
                "from": msg.from_agent,
                "to": msg.to_agent,
                "content": msg.message.content,
                "timestamp": msg.timestamp,
            }
            for msg in self.message_history
        ]
