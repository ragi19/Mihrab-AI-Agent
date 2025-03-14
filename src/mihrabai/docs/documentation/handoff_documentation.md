# Extensions: Handoff Functionality Documentation

This document provides detailed information about the handoff extensions in the Agents Python library. These extensions facilitate seamless communication and coordination between different agents in a multi-agent system.

## Table of Contents
- [Overview](#overview)
- [Handoff Prompt Extension](#handoff-prompt-extension)
- [Handoff Filters Extension](#handoff-filters-extension)
- [Integration with Agents](#integration-with-agents)
- [Best Practices](#best-practices)
- [Usage Examples](#usage-examples)
- [Handoff Patterns](#handoff-patterns)
- [State Management During Handoffs](#state-management-during-handoffs)
- [Advanced Handoff Scenarios](#advanced-handoff-scenarios)
- [Best Practices for Complex Handoffs](#best-practices-for-complex-handoffs)

## Overview

The handoff extensions provide functionality to:

1. **Configure agent prompts** with appropriate handoff instructions
2. **Filter conversation history** when transferring control between agents
3. **Manage state persistence** between handoffs

These extensions enable building multi-agent systems where agents can specialize in different tasks and seamlessly transfer control between each other based on user needs or conversation context.

## Handoff Prompt Extension

The handoff prompt extension provides standardized system instructions that inform agents about their role in a multi-agent system. These instructions help agents understand when and how to use handoff functions.

### Key Components

- **`RECOMMENDED_PROMPT_PREFIX`**: A standardized system prompt that introduces the agent to the multi-agent framework
- **`prompt_with_handoff_instructions()`**: A utility function to prepend the recommended instructions to an agent's prompt

### How It Works

The prompt prefix explains to the agent:
- It's part of a multi-agent system called the Agents SDK
- The concepts of Agents and Handoffs
- How to use handoff functions (typically named `transfer_to_<agent_name>`)
- To handle transfers seamlessly without drawing attention to them

```python
# Example of adding handoff instructions to an agent prompt
from agents.extensions import handoff_prompt

agent_prompt = """
You are a customer service agent that can help with product inquiries.
If a customer asks about billing or returns, transfer them to the appropriate department.
"""

final_prompt = handoff_prompt.prompt_with_handoff_instructions(agent_prompt)
```

## Handoff Filters Extension

The handoff filters extension provides utilities to filter conversation history when transferring between agents. This helps keep conversation context relevant to each specialized agent.

### Key Components

- **`remove_all_tools()`**: Filters out all tool-related items from conversation history
- **`_remove_tools_from_items()`**: Helper function to filter tool-related items from a sequence of items
- **`_remove_tool_types_from_input()`**: Helper function to filter tool-related items based on their type

### How It Works

When transferring to a new agent, it may not be necessary or desirable to include all previous tool calls and outputs. The handoff filters allow you to:
- Remove function calls and their outputs
- Remove file search calls and results
- Remove web search calls and results
- Remove computer tool calls and outputs

This creates a cleaner context for the target agent to work with.

```python
# Example of filtering conversation history during a handoff
from agents.extensions import handoff_filters
from agents.handoffs import HandoffInputData

def custom_filter(input_data: HandoffInputData) -> HandoffInputData:
    # First remove all tools
    filtered_data = handoff_filters.remove_all_tools(input_data)
    
    # Then perform additional custom filtering if needed
    return filtered_data
```

## Integration with Agents

The handoff extensions are designed to integrate with the Agent framework:

### Agent Configuration

```python
from agents import Agent, HandoffConfig
from agents.extensions import handoff_prompt, handoff_filters

# Create a specialized agent that can receive handoffs
billing_agent = Agent(
    system_prompt=handoff_prompt.prompt_with_handoff_instructions(
        "You are a billing specialist who can help with invoices and payments."
    ),
    model="gpt-4",
)

# Create a main agent that can perform handoffs
main_agent = Agent(
    system_prompt=handoff_prompt.prompt_with_handoff_instructions(
        "You are a customer service representative. Handle general inquiries, but transfer billing questions to the billing department."
    ),
    model="gpt-4",
    handoffs=[
        HandoffConfig(
            "transfer_to_billing",
            "Transfer to billing department",
            billing_agent,
            input_filter=handoff_filters.remove_all_tools,
        )
    ]
)
```

## Best Practices

1. **Clear Agent Roles**: Define clear boundaries for each agent's responsibilities
2. **Explicit Transfer Conditions**: Provide clear instructions on when to transfer
3. **Minimal Context Transfer**: Only transfer relevant conversation history
4. **User Experience**: Ensure transfers are seamless from the user's perspective
5. **Specialized Input Filters**: Create custom filters for specific agent needs
6. **Proper Introduction**: Each agent should understand its role in the system

## Usage Examples

### Basic Multi-Agent Setup

```python
from agents import Agent, HandoffConfig
from agents.extensions import handoff_prompt, handoff_filters

# Define a technical support agent
tech_support = Agent(
    system_prompt=handoff_prompt.prompt_with_handoff_instructions(
        "You are a technical support specialist. Help users with technical problems related to our software."
    ),
    model="gpt-4",
)

# Define a sales agent
sales_agent = Agent(
    system_prompt=handoff_prompt.prompt_with_handoff_instructions(
        "You are a sales representative. Help users with pricing, licensing, and purchasing our products."
    ),
    model="gpt-4",
)

# Define a general customer service agent that can hand off to specialized agents
general_agent = Agent(
    system_prompt=handoff_prompt.prompt_with_handoff_instructions(
        "You are a customer service representative. Handle general inquiries and direct users to the appropriate department for specialized help."
    ),
    model="gpt-4",
    handoffs=[
        HandoffConfig(
            "transfer_to_tech_support",
            "Transfer to technical support",
            tech_support,
            input_filter=handoff_filters.remove_all_tools,
        ),
        HandoffConfig(
            "transfer_to_sales",
            "Transfer to sales department",
            sales_agent,
            input_filter=handoff_filters.remove_all_tools,
        ),
    ],
)

# Run the conversation
async def main():
    response = await general_agent.run("I'm having trouble with your software installation")
    # The general agent will likely transfer this to tech support
```

### Custom Filter Implementation

```python
from agents import Agent, HandoffConfig
from agents.handoffs import HandoffInputData
from agents.extensions import handoff_prompt

# Define a custom filter that keeps relevant technical context
def technical_filter(input_data: HandoffInputData) -> HandoffInputData:
    # Start with removing all tools
    from agents.extensions.handoff_filters import remove_all_tools
    filtered_data = remove_all_tools(input_data)
    
    # Additional processing could be added here
    # e.g., keep only messages related to technical issues
    
    return filtered_data

# Define the agents with custom filter
tech_agent = Agent(
    system_prompt=handoff_prompt.prompt_with_handoff_instructions(
        "You are a technical expert who can solve complex software problems."
    ),
    model="gpt-4",
)

main_agent = Agent(
    system_prompt=handoff_prompt.prompt_with_handoff_instructions(
        "You are a general assistant. Transfer technical questions to the technical expert."
    ),
    model="gpt-4",
    handoffs=[
        HandoffConfig(
            "transfer_to_technical_expert",
            "Transfer to technical expert for complex issues",
            tech_agent,
            input_filter=technical_filter,
        )
    ]
)
```

## Handoff Patterns

### Direct Handoff

The simplest form of handoff where one agent directly transfers control to another:

```python
from agents import Agent, HandoffConfig
from agents.handoffs import HandoffInputData

class CustomerServiceAgent(Agent):
    async def handle_billing_query(self, message: str) -> None:
        # Transfer to billing specialist
        await self.handoff_to("billing_specialist", message)

# Configure the handoff
billing_agent = Agent(
    system_prompt="You are a billing specialist...",
    model="gpt-4"
)

main_agent = Agent(
    system_prompt="You are a customer service agent...",
    model="gpt-4",
    handoffs=[
        HandoffConfig(
            "billing_specialist",
            "Transfer billing queries",
            billing_agent
        )
    ]
)
```

### Chain Handoff

Multiple agents in a predefined sequence:

```python
class HandoffChain:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
    
    async def process(self, message: str) -> str:
        current_message = message
        for agent in self.agents:
            response = await agent.process_message(current_message)
            current_message = response.content
        return current_message

# Example usage
chain = HandoffChain([
    intake_agent,
    analysis_agent,
    resolution_agent
])
```

### Conditional Handoff

Handoffs based on specific conditions:

```python
class ConditionalHandoff:
    def __init__(self, conditions: Dict[str, Agent]):
        self.conditions = conditions
    
    async def route_message(self, message: str) -> Agent:
        # Analyze message content
        topics = await analyze_topics(message)
        
        # Route to appropriate agent
        for topic, agent in self.conditions.items():
            if topic in topics:
                return agent
        
        # Default agent if no conditions match
        return default_agent

# Example usage
router = ConditionalHandoff({
    "billing": billing_agent,
    "technical": tech_support_agent,
    "sales": sales_agent
})
```

## State Management During Handoffs

### Conversation Context

Maintain relevant context during handoffs:

```python
@dataclass
class HandoffContext:
    conversation_history: List[Message]
    user_info: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def filter_relevant_history(self, topic: str) -> List[Message]:
        """Filter conversation history relevant to the topic"""
        return [
            msg for msg in self.conversation_history
            if is_relevant_to_topic(msg.content, topic)
        ]
    
    def add_system_context(self) -> str:
        """Create system context message"""
        return f"""Previous conversation summary:
User: {self.user_info.get('name')}
Topic: {self.metadata.get('topic')}
Key points: {self.metadata.get('key_points')}
"""
```

### State Transfer

Ensure smooth state transfer between agents:

```python
class StateTransfer:
    @staticmethod
    async def prepare_handoff_state(
        from_agent: Agent,
        to_agent: Agent,
        context: HandoffContext
    ) -> HandoffInputData:
        # Prepare conversation history
        filtered_history = context.filter_relevant_history(to_agent.specialty)
        
        # Add system context
        system_message = context.add_system_context()
        
        return HandoffInputData(
            conversation_history=filtered_history,
            system_message=system_message,
            metadata=context.metadata
        )
    
    @staticmethod
    async def transfer_state(
        from_agent: Agent,
        to_agent: Agent,
        state: HandoffInputData
    ) -> None:
        # Clear target agent's state
        await to_agent.reset_state()
        
        # Transfer relevant state
        await to_agent.load_state(state)
```

## Advanced Handoff Scenarios

### Multi-Step Handoff

Complex handoffs involving multiple agents:

```python
class MultiStepHandoff:
    def __init__(self, workflow: Dict[str, List[Agent]]):
        self.workflow = workflow
    
    async def execute_workflow(self, step: str, message: str) -> str:
        agents = self.workflow[step]
        current_message = message
        
        for agent in agents:
            response = await agent.process_message(current_message)
            current_message = response.content
            
            # Check if we need to move to next step
            if should_proceed_to_next_step(response):
                next_step = get_next_step(step)
                if next_step in self.workflow:
                    return await self.execute_workflow(next_step, current_message)
        
        return current_message

# Example usage
workflow = {
    "intake": [validation_agent, classification_agent],
    "processing": [analysis_agent, enrichment_agent],
    "resolution": [solution_agent, quality_check_agent]
}

handoff = MultiStepHandoff(workflow)
```

### Parallel Handoff

Process message with multiple agents simultaneously:

```python
class ParallelHandoff:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
    
    async def process_parallel(self, message: str) -> List[str]:
        tasks = [
            agent.process_message(message)
            for agent in self.agents
        ]
        
        responses = await asyncio.gather(*tasks)
        return [r.content for r in responses]

# Example usage
specialist_team = ParallelHandoff([
    technical_expert,
    domain_expert,
    language_expert
])
```

### Fallback Handoff

Implement fallback mechanisms for handoff failures:

```python
class FallbackHandoff:
    def __init__(
        self,
        primary_agent: Agent,
        fallback_agents: List[Agent],
        max_retries: int = 2
    ):
        self.primary_agent = primary_agent
        self.fallback_agents = fallback_agents
        self.max_retries = max_retries
    
    async def process_with_fallback(self, message: str) -> str:
        try:
            return await self.primary_agent.process_message(message)
        except Exception as e:
            self.logger.error(f"Primary agent failed: {e}")
            
            # Try fallback agents
            for agent in self.fallback_agents:
                try:
                    return await agent.process_message(message)
                except Exception as e:
                    self.logger.error(f"Fallback agent failed: {e}")
            
            # If all fallbacks fail, raise error
            raise HandoffError("All agents failed to process message")

# Example usage
handoff = FallbackHandoff(
    primary_agent=specialized_agent,
    fallback_agents=[
        general_agent,
        backup_agent
    ]
)
```

## Best Practices for Complex Handoffs

1. **State Validation**
   ```python
   class StateValidator:
       @staticmethod
       def validate_handoff_state(state: HandoffInputData) -> bool:
           # Check required fields
           if not state.conversation_history:
               return False
           
           # Validate conversation history
           for message in state.conversation_history:
               if not message.is_valid():
                   return False
           
           return True
   ```

2. **Error Recovery**
   ```python
   class HandoffErrorRecovery:
       async def recover_from_failed_handoff(
           self,
           error: Exception,
           from_agent: Agent,
           to_agent: Agent,
           message: str
       ) -> str:
           # Log error
           self.logger.error(f"Handoff failed: {error}")
           
           # Try to recover state
           if isinstance(error, StateTransferError):
               await self.recover_state(from_agent, to_agent)
           
           # Return to original agent if needed
           return await from_agent.process_message(
               f"I apologize, but I couldn't transfer you to the specialist. "
               f"Let me help you directly: {message}"
           )
   ```

3. **Monitoring and Metrics**
   ```python
   class HandoffMetrics:
       def __init__(self):
           self.successful_handoffs = 0
           self.failed_handoffs = 0
           self.average_handoff_time = 0
       
       def record_handoff(self, success: bool, duration: float) -> None:
           if success:
               self.successful_handoffs += 1
           else:
               self.failed_handoffs += 1
           
           total_handoffs = self.successful_handoffs + self.failed_handoffs
           self.average_handoff_time = (
               (self.average_handoff_time * (total_handoffs - 1) + duration)
               / total_handoffs
           )
   ```

4. **Security Checks**
   ```python
   class HandoffSecurity:
       def __init__(self):
           self.allowed_handoffs = set()
           self.blocked_patterns = []
       
       def validate_handoff(
           self,
           from_agent: Agent,
           to_agent: Agent,
           message: str
       ) -> bool:
           # Check if handoff is allowed
           handoff_key = f"{from_agent.id}->{to_agent.id}"
           if handoff_key not in self.allowed_handoffs:
               return False
           
           # Check message content
           for pattern in self.blocked_patterns:
               if pattern.search(message):
                   return False
           
           return True
   ```

These advanced patterns and practices enable building sophisticated multi-agent systems with reliable handoff mechanisms, proper state management, and robust error handling.
