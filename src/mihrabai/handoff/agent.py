"""
HandoffAgent implementation for multi-agent systems
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set
import json
import time
import asyncio

from ..common.interfaces import AgentInterface
from ..core.memory import Memory
from ..core.memory_task_agent import MemoryEnabledTaskAgent
from ..core.message import Message, MessageRole
from ..models.base import BaseModel, ModelCapability, ModelError
from ..utils.logging import get_logger
from .config import HandoffConfig, HandoffInputData
from .model_selection import select_model, create_model_config

logger = get_logger("handoff.agent")


class HandoffAgent(AgentInterface):
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
        memory_path: Optional[str] = None,
        feedback_enabled: bool = False,
        auto_discover_agents: bool = False,
        collaboration_mode: bool = False,
        model_candidates: Optional[List[Dict[str, Any]]] = None,
        required_model_capabilities: Optional[Set[str]] = None,
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
            memory_path: Optional path to store persistent memory
            feedback_enabled: Whether to collect feedback on handoffs
            auto_discover_agents: Whether to automatically discover other agents
            collaboration_mode: Whether to enable collaboration with other agents
            model_candidates: Optional list of model configurations to try
            required_model_capabilities: Optional set of required model capabilities
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
        self.memory_path = memory_path
        self.feedback_enabled = feedback_enabled
        self.auto_discover_agents = auto_discover_agents
        self.collaboration_mode = collaboration_mode
        self.handoff_registry = {}  # Registry of available agents for handoff
        self.feedback_history = []  # Store feedback on handoffs
        self.performance_metrics = {
            "handoffs_initiated": 0,
            "handoffs_received": 0,
            "successful_handoffs": 0,
            "failed_handoffs": 0,
            "average_feedback_score": 0.0,
        }
        self.last_active_time = time.time()
        self.model_candidates = model_candidates or []
        self.required_model_capabilities = required_model_capabilities or set()
        self.model = None  # Will be set during initialization
        logger.info(f"Created HandoffAgent: {name}")
        
        # Initialize persistent memory if path provided
        if self.memory_path:
            self._load_memory()

    async def initialize(self, model: Optional[BaseModel] = None):
        """
        Initialize the agent with a model
        
        Args:
            model: Optional model to use. If not provided, will try to select
                  a model from model_candidates.
        """
        # Import MemoryAgentRunner here to avoid circular import
        from ..runtime.memory_runner import MemoryAgentRunner
        from ..core.task_agent import TaskAgent
        from ..core.memory_task_agent import MemoryEnabledTaskAgent
        
        # If model is provided, use it
        if model:
            self.model = model
        # Otherwise, try to select a model from candidates
        elif self.model_candidates:
            selected_model, _ = await select_model(
                self.model_candidates,
                required_capabilities=self.required_model_capabilities,
            )
            if selected_model:
                self.model = selected_model
            else:
                logger.error(f"Failed to select a model for agent {self.name}")
                raise RuntimeError(f"No suitable model found for agent {self.name}")
        else:
            logger.error(f"No model provided and no model candidates for agent {self.name}")
            raise ValueError(f"No model provided for agent {self.name}")
        
        # Create a memory-enabled task agent
        memory = Memory()
        memory_agent = MemoryEnabledTaskAgent(
            model=self.model,
            system_message=self.system_prompt,
            tools=self.tools,
            memory=memory,
            automatic_memory=True
        )
        
        # Create a MemoryAgentRunner with the memory agent
        self.agent = MemoryAgentRunner(
            agent=memory_agent,
            memory_persistence_path=self.memory_path
        )
            
        # If auto-discover is enabled, register with other agents
        if self.auto_discover_agents:
            await self._discover_agents()
        
    async def process(
        self,
        message: str,
        session_id: str = None,
        context: Optional[Dict[str, Any]] = None,
        handoff_chain: Optional[List[str]] = None,
        feedback_data: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """
        Process a message with potential handoffs
        
        Args:
            message: The message to process
            session_id: Optional session ID for persistence
            context: Optional context from previous interactions
            handoff_chain: Optional list of previous handoffs to prevent cycles
            feedback_data: Optional feedback data from previous handoffs
            
        Returns:
            Response text, or tuple of (response text, updated context)
        """
        if not self.agent:
            raise RuntimeError(f"Agent {self.name} not initialized")
            
        # Update conversation context
        if context:
            self.conversation_context.update(context)
            logger.debug(f"Updated context for {self.name}: {context}")
            
        # Update last active time
        self.last_active_time = time.time()
            
        # Process feedback if provided
        if feedback_data and self.feedback_enabled:
            self._process_feedback(feedback_data)
            
        # Initialize handoff chain
        handoff_chain = handoff_chain or []
        if self.name in handoff_chain:
            # Prevent circular handoffs
            logger.warning(f"Circular handoff detected for {self.name}")
            return f"I apologize, but I cannot process this request due to a circular handoff chain: {' -> '.join(handoff_chain)}"
            
        # Add self to handoff chain
        current_chain = handoff_chain.copy()
        current_chain.append(self.name)
        
        # Update handoff metrics
        if len(current_chain) > 1:
            self.performance_metrics["handoffs_received"] += 1
        
        # Check if we're in collaboration mode and should involve other agents
        if self.collaboration_mode and len(current_chain) <= 1:
            collaboration_result = await self._handle_collaboration(message, session_id, context, current_chain)
            if collaboration_result:
                return collaboration_result
        
        # Check if we should hand off to a specialized agent
        if self.can_defer and self.handoffs:
            # Sort handoffs by priority if available
            sorted_handoffs = sorted(
                self.handoffs, 
                key=lambda h: h.metadata.get("priority", 0),
                reverse=True
            )
            
            for handoff in sorted_handoffs:
                if self._should_handoff(message, handoff, context):
                    logger.info(f"Handing off from {self.name} to {handoff.target_agent.name}")
                    
                    # Update handoff metrics
                    self.performance_metrics["handoffs_initiated"] += 1
                    
                    # Prepare context to pass along
                    handoff_context = self.conversation_context.copy()
                    handoff_context.update({
                        "source_agent": self.name,
                        "handoff_reason": handoff.description,
                        "handoff_metadata": handoff.metadata,
                        "handoff_time": time.time(),
                        "conversation_history": context.get("conversation_history", []) if context else [],
                    })
                    
                    try:
                        # Process with target agent
                        result = await handoff.target_agent.process(
                            message,
                            session_id=session_id,
                            context=handoff_context,
                            handoff_chain=current_chain.copy(),
                        )
                        
                        # Update successful handoff metrics
                        self.performance_metrics["successful_handoffs"] += 1
                        
                        # If feedback is enabled, request feedback
                        if self.feedback_enabled:
                            if isinstance(result, tuple):
                                response, updated_context = result
                                updated_context["feedback_requested"] = True
                                return response, updated_context
                            else:
                                return result, {"feedback_requested": True}
                        
                        return result
                    except Exception as e:
                        logger.error(f"Error in handoff to {handoff.target_agent.name}: {e}")
                        self.performance_metrics["failed_handoffs"] += 1
                        # Continue with next handoff option or process locally
                    
        # Process with current agent if no handoff needed
        logger.info(f"No handoff needed, {self.name} processing directly")
        
        try:
            # Add context to the message if available
            enhanced_message = message
            if self.conversation_context:
                context_str = "\nContext from previous interactions:\n"
                for key, value in self.conversation_context.items():
                    if key != "conversation_history" and not key.startswith("_"):
                        context_str += f"- {key}: {value}\n"
                enhanced_message = context_str + "\n" + message
            
            # Process message based on agent type
            from ..runtime.memory_runner import MemoryAgentRunner
            from ..core.task_agent import TaskAgent
            
            if isinstance(self.agent, MemoryAgentRunner):
                # Use run method for MemoryAgentRunner
                response = await self.agent.run(enhanced_message, session_id=session_id)
                response_content = response.content
            elif isinstance(self.agent, TaskAgent):
                # Use process_message for TaskAgent
                from ..core.message import Message, MessageRole
                user_message = Message(role=MessageRole.USER, content=enhanced_message)
                response = await self.agent.process_message(user_message)
                response_content = response.content
            else:
                raise TypeError(f"Unsupported agent type: {type(self.agent)}")
            
            # Update conversation history in context
            if "conversation_history" not in self.conversation_context:
                self.conversation_context["conversation_history"] = []
                
            self.conversation_context["conversation_history"].append({
                "role": "user",
                "content": message,
                "timestamp": time.time()
            })
            
            self.conversation_context["conversation_history"].append({
                "role": "assistant",
                "content": response_content,
                "timestamp": time.time()
            })
            
            # Limit history size
            max_history = 20
            if len(self.conversation_context["conversation_history"]) > max_history:
                self.conversation_context["conversation_history"] = self.conversation_context["conversation_history"][-max_history:]
            
            # Save memory if enabled
            if self.memory_path:
                self._save_memory()
            
            # Return response with updated context
            return response_content, self.conversation_context
            
        except Exception as e:
            logger.error(f"Error in agent {self.name} processing: {e}")
            raise
            
    def _should_handoff(
        self,
        message: str,
        handoff: "HandoffConfig",
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Determine if a message should be handed off based on the handoff configuration
        
        Args:
            message: The message to check
            handoff: The handoff configuration
            context: Optional context dictionary
            
        Returns:
            True if the message should be handed off, False otherwise
        """
        if handoff.condition:
            return handoff.condition(message, context or {})
        return False
        
    async def should_handle(self, message: Union[str, Message]) -> bool:
        """
        Check if this agent should handle the given message
        
        Args:
            message: The message to check
            
        Returns:
            True if this agent should handle the message, False otherwise
        """
        # Get message content
        if isinstance(message, Message):
            content = message.content
        else:
            content = message
            
        # Check expertise areas if defined
        if self.expertise:
            content_lower = content.lower()
            return any(area.lower() in content_lower for area in self.expertise)
            
        # By default, only handle if we're the target of a handoff
        return False
        
    async def _discover_agents(self):
        """
        Discover other agents available for handoff
        
        This method attempts to find other agents in the system that
        can be used for handoffs. It updates the handoff_registry with
        discovered agents.
        """
        # This is a placeholder for actual discovery logic
        # In a real implementation, this would use a registry service or similar
        logger.info(f"Agent {self.name} is discovering other agents")
        
        # For now, we'll just log that discovery is enabled
        # In a real system, this would populate self.handoff_registry
        pass
        
    def register_agent(self, agent_name: str, agent: "HandoffAgent", expertise: List[str]):
        """
        Register another agent for potential handoffs
        
        Args:
            agent_name: Name of the agent to register
            agent: The agent instance
            expertise: List of expertise areas for the agent
        """
        self.handoff_registry[agent_name] = {
            "agent": agent,
            "expertise": expertise,
            "last_active": time.time(),
        }
        
        # Create a handoff configuration for this agent
        from .conditions import keyword_based_condition
        
        handoff_config = HandoffConfig(
            name=f"auto_{agent_name}",
            description=f"Automatic handoff to {agent_name}",
            target_agent=agent,
            condition=keyword_based_condition(expertise),
            metadata={"auto_discovered": True}
        )
        
        # Add to handoffs if not already present
        if not any(h.target_agent.name == agent_name for h in self.handoffs):
            self.handoffs.append(handoff_config)
            
        logger.info(f"Agent {self.name} registered {agent_name} for handoffs")
        
    def _process_feedback(self, feedback_data: Dict[str, Any]):
        """
        Process feedback on a previous handoff
        
        Args:
            feedback_data: Dictionary containing feedback information
        """
        # Store feedback for analysis
        self.feedback_history.append({
            "timestamp": time.time(),
            "data": feedback_data,
        })
        
        # Update metrics
        if "score" in feedback_data:
            # Calculate new average
            total_feedback = len(self.feedback_history)
            current_avg = self.performance_metrics["average_feedback_score"]
            new_score = feedback_data["score"]
            
            # Update running average
            if total_feedback > 1:
                self.performance_metrics["average_feedback_score"] = (
                    (current_avg * (total_feedback - 1) + new_score) / total_feedback
                )
            else:
                self.performance_metrics["average_feedback_score"] = new_score
                
        # Save feedback to persistent storage if enabled
        if self.memory_path:
            self._save_memory()
            
        logger.info(f"Agent {self.name} processed feedback: {feedback_data}")
        
    def _save_memory(self):
        """Save agent memory to persistent storage"""
        if not self.memory_path:
            return
            
        try:
            memory_data = {
                "conversation_context": self.conversation_context,
                "performance_metrics": self.performance_metrics,
                "feedback_history": self.feedback_history,
                "last_active_time": self.last_active_time,
            }
            
            with open(self.memory_path, "w") as f:
                json.dump(memory_data, f)
                
            logger.debug(f"Agent {self.name} saved memory to {self.memory_path}")
        except Exception as e:
            logger.error(f"Error saving memory for agent {self.name}: {e}")
            
    def _load_memory(self):
        """Load agent memory from persistent storage"""
        if not self.memory_path:
            return
            
        try:
            import os
            if not os.path.exists(self.memory_path):
                logger.debug(f"No memory file found at {self.memory_path}")
                return
                
            with open(self.memory_path, "r") as f:
                memory_data = json.load(f)
                
            # Restore memory
            self.conversation_context = memory_data.get("conversation_context", {})
            self.performance_metrics = memory_data.get("performance_metrics", self.performance_metrics)
            self.feedback_history = memory_data.get("feedback_history", [])
            self.last_active_time = memory_data.get("last_active_time", time.time())
                
            logger.debug(f"Agent {self.name} loaded memory from {self.memory_path}")
        except Exception as e:
            logger.error(f"Error loading memory for agent {self.name}: {e}")
            
    async def _handle_collaboration(
        self, 
        message: str, 
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        handoff_chain: Optional[List[str]] = None,
    ) -> Optional[Union[str, Tuple[str, Dict[str, Any]]]]:
        """
        Handle collaboration with other agents
        
        In collaboration mode, the agent can work with other agents to
        solve complex problems that require multiple areas of expertise.
        
        Args:
            message: The message to process
            session_id: Optional session ID
            context: Optional context dictionary
            handoff_chain: Optional list of previous handoffs to prevent cycles
            
        Returns:
            Optional response or None if no collaboration needed
        """
        if not self.collaboration_mode or not self.handoff_registry:
            return None
            
        # Add recursion guard to prevent infinite recursion
        if context and context.get("in_collaboration_process"):
            return None
            
        # Initialize handoff chain if not provided
        handoff_chain = handoff_chain or []
            
        # Determine if this is a complex task requiring collaboration
        from .conditions import multi_agent_collaboration_condition
        
        collaboration_detector = multi_agent_collaboration_condition()
        if not collaboration_detector(message, context or {}):
            return None
            
        # Set a safe message preview that won't cause issues with logging
        message_preview = message[:50] if message else ""
        logger.info(f"Agent {self.name} initiating collaboration on: {message_preview}...")
        
        # Find relevant agents for collaboration
        relevant_agents = []
        for agent_name, info in self.handoff_registry.items():
            agent = info["agent"]
            if await agent.should_handle(message):
                relevant_agents.append(agent)
                
        if not relevant_agents:
            logger.debug(f"No relevant agents found for collaboration")
            return None
            
        # Create a collaborative context
        collab_context = (context or {}).copy()
        collab_context.update({
            "collaboration_mode": True,
            "primary_agent": self.name,
            "collaborating_agents": [agent.name for agent in relevant_agents],
            "in_collaboration_process": True,  # Mark that we're in a collaboration process
        })
        
        # Create a collaboration chain to track the flow
        collab_chain = handoff_chain.copy()
        collab_chain.append(f"{self.name}:collaboration")
        
        # Process with self first to create a plan
        plan_result = await self.process(
            f"[Collaboration planning] {message}",
            session_id=session_id,
            context=collab_context,
            handoff_chain=collab_chain,
        )
        
        # Handle both string and tuple returns
        if isinstance(plan_result, tuple):
            self_response, updated_context = plan_result
        else:
            self_response = plan_result
            updated_context = collab_context.copy()
        
        # Extract tasks for other agents
        tasks = self._extract_tasks_from_plan(self_response)
        if not tasks:
            return None
            
        # Distribute tasks to relevant agents
        results = []
        for task, agent in zip(tasks, relevant_agents[:len(tasks)]):
            task_context = updated_context.copy()
            task_context["task"] = task
            
            try:
                task_response = await agent.process(
                    f"[Collaboration task] {task}",
                    session_id=session_id,
                    context=task_context,
                    handoff_chain=collab_chain,
                )
                
                if isinstance(task_response, tuple):
                    results.append(task_response[0])
                else:
                    results.append(task_response)
            except Exception as e:
                logger.error(f"Error in collaboration with {agent.name}: {e}")
                results.append(f"Error from {agent.name}: {str(e)}")
                
        # Combine results
        combined_context = updated_context.copy()
        combined_context["collaboration_results"] = results
        
        final_result = await self.process(
            f"[Collaboration synthesis] Original query: {message}\n\nResults from collaborating agents:\n" + 
            "\n".join([f"- {agent.name}: {result[:100]}..." for agent, result in zip(relevant_agents[:len(results)], results)]),
            session_id=session_id,
            context=combined_context,
            handoff_chain=collab_chain,
        )
        
        # Handle both string and tuple returns for the final result
        if isinstance(final_result, tuple):
            return final_result
        else:
            return final_result, combined_context
        
    def _extract_tasks_from_plan(self, plan: str) -> List[str]:
        """
        Extract individual tasks from a collaboration plan
        
        Args:
            plan: The collaboration plan text
            
        Returns:
            List of individual tasks
        """
        # Simple extraction based on numbered or bulleted lists
        import re
        
        # Look for numbered tasks (1. Task description)
        numbered_tasks = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|\Z)', plan, re.DOTALL)
        
        # Look for bulleted tasks (- Task description or * Task description)
        bulleted_tasks = re.findall(r'[-*]\s*(.*?)(?=[-*]|\Z)', plan, re.DOTALL)
        
        # Combine and clean up
        all_tasks = numbered_tasks + bulleted_tasks
        cleaned_tasks = [task.strip() for task in all_tasks if task.strip()]
        
        return cleaned_tasks
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this agent
        
        Returns:
            Dictionary of performance metrics
        """
        return self.performance_metrics.copy()
        
    def get_expertise_summary(self) -> Dict[str, Any]:
        """
        Get a summary of this agent's expertise
        
        Returns:
            Dictionary summarizing agent expertise and capabilities
        """
        return {
            "name": self.name,
            "expertise": self.expertise,
            "can_defer": self.can_defer,
            "tools_count": len(self.tools),
            "handoffs_count": len(self.handoffs),
            "collaboration_enabled": self.collaboration_mode,
        }
        
    def add_model_candidate(self, model_config: Dict[str, Any]) -> None:
        """
        Add a model candidate to the agent's list of potential models
        
        Args:
            model_config: Model configuration dictionary
        """
        self.model_candidates.append(model_config)
        
    def add_model_candidates(self, model_configs: List[Dict[str, Any]]) -> None:
        """
        Add multiple model candidates to the agent's list of potential models
        
        Args:
            model_configs: List of model configuration dictionaries
        """
        self.model_candidates.extend(model_configs)
        
    async def switch_model(self, strategy_name: Optional[str] = None) -> bool:
        """
        Switch to a different model using the model selection strategy
        
        This is useful when the current model is unavailable or not performing well.
        
        Args:
            strategy_name: Optional name of the strategy to use for model selection
            
        Returns:
            True if model was successfully switched, False otherwise
        """
        if not self.model_candidates:
            logger.warning(f"No model candidates available for agent {self.name}")
            return False
            
        # Try to select a new model
        from .model_selection import select_model
        
        selected_model, _ = await select_model(
            self.model_candidates,
            strategy_name=strategy_name,
            required_capabilities=self.required_model_capabilities,
            use_cache=False,  # Don't use cache to ensure we get a different model
        )
        
        if not selected_model:
            logger.error(f"Failed to select a new model for agent {self.name}")
            return False
            
        # Update the model
        self.model = selected_model
        
        # Reinitialize the agent with the new model
        from ..runtime.memory_runner import MemoryAgentRunner
        from ..core.task_agent import TaskAgent
        from ..core.memory_task_agent import MemoryEnabledTaskAgent
        
        # Create a memory-enabled task agent
        memory = Memory()
        memory_agent = MemoryEnabledTaskAgent(
            model=self.model,
            system_message=self.system_prompt,
            tools=self.tools,
            memory=memory,
            automatic_memory=True
        )
        
        # Create a MemoryAgentRunner with the memory agent
        self.agent = MemoryAgentRunner(
            agent=memory_agent,
            memory_persistence_path=self.memory_path
        )
            
        logger.info(f"Agent {self.name} switched to model {self.model.model_name}")
        return True
