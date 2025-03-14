# Handoff System for Multi-Agent Applications

The Handoff System is a powerful extension to the LLM Agents framework that enables seamless coordination and communication between specialized agents. This system allows you to build complex multi-agent applications where different agents can handle specific types of queries and transfer control between each other based on user needs.

## Table of Contents

- [Overview](#overview)
- [Key Components](#key-components)
- [Getting Started](#getting-started)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Handoff Patterns](#handoff-patterns)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Overview

The Handoff System allows you to:

1. **Create specialized agents** for different domains or tasks
2. **Define handoff rules** to determine when to transfer control
3. **Filter conversation history** during handoffs to maintain context
4. **Preserve state** between agent interactions
5. **Build complex agent networks** with various handoff patterns

This enables more natural and effective multi-agent interactions, where each agent can focus on its area of expertise while providing a seamless experience to the user.

## Key Components

The Handoff System consists of several key components:

### HandoffAgent

A specialized agent class that extends the base agent functionality with handoff capabilities. It can process messages directly or transfer control to other agents based on defined rules.

```python
class HandoffAgent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: Optional[List[Any]] = None,
        handoffs: Optional[List[HandoffConfig]] = None
    ):
        # ...
```

### HandoffConfig

Configuration for defining when and how to transfer control between agents.

```python
class HandoffConfig:
    def __init__(
        self,
        name: str,
        description: str,
        target_agent: 'HandoffAgent',
        input_filter: Optional[Callable] = None
    ):
        # ...
```

### HandoffInputData

Data structure for managing the information passed between agents during handoffs.

```python
class HandoffInputData:
    def __init__(
        self,
        conversation_history: List[Message],
        system_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        # ...
```

## Getting Started

To use the Handoff System, you need to:

1. Install the LLM Agents package
2. Import the necessary components
3. Create specialized agents with handoff capabilities
4. Configure handoff rules
5. Process user queries through the system

### Installation

```bash
pip install llm-agents
```

### Basic Setup

```python
from mihrabai.models.multi_provider import MultiProviderModel
from mihrabai.handoff import HandoffAgent, HandoffConfig, HandoffInputData

# Create a model
model = await MultiProviderModel.create(
    primary_model="your-model-name",
    required_capabilities={ModelCapability.CHAT}
)

# Create specialized agents
technical_agent = HandoffAgent(
    name="Technical",
    system_prompt="You are a technical support agent..."
)

billing_agent = HandoffAgent(
    name="Billing",
    system_prompt="You are a billing support agent..."
)

# Initialize agents
await technical_agent.initialize(model)
await billing_agent.initialize(model)

# Create general agent with handoffs
general_agent = HandoffAgent(
    name="General",
    system_prompt="You are a general customer service agent...",
    handoffs=[
        HandoffConfig(
            name="technical",
            description="technical issues",
            target_agent=technical_agent
        ),
        HandoffConfig(
            name="billing",
            description="billing inquiries",
            target_agent=billing_agent
        )
    ]
)

await general_agent.initialize(model)

# Process a query
response = await general_agent.process("I need help with my internet connection")
```

## Basic Usage

### Creating Specialized Agents

Create agents that specialize in different domains:

```python
technical_agent = HandoffAgent(
    name="Technical",
    system_prompt="You are a technical support agent who helps users with computer and software issues.",
    tools=[calculator_tool]  # Optional tools
)

billing_agent = HandoffAgent(
    name="Billing",
    system_prompt="You are a billing support agent who helps with payment and invoice issues."
)
```

### Configuring Handoff Rules

Define when and how to transfer control between agents:

```python
general_agent = HandoffAgent(
    name="General",
    system_prompt="You are a general customer service agent...",
    handoffs=[
        HandoffConfig(
            name="technical",
            description="the user has a technical issue",
            target_agent=technical_agent,
            input_filter=remove_sensitive_info  # Optional filter function
        ),
        HandoffConfig(
            name="billing",
            description="the user has a billing inquiry",
            target_agent=billing_agent
        )
    ]
)
```

### Processing Queries

Process user queries through the system:

```python
response = await general_agent.process(
    "My internet connection keeps dropping",
    session_id="user123"  # Optional session ID for persistence
)
```

## Advanced Features

### Custom Handoff Logic

You can customize when to hand off by overriding the `_should_handoff` method:

```python
def _should_handoff(self, message: str, handoff: HandoffConfig) -> bool:
    # Custom logic to determine if handoff is needed
    if handoff.name == "technical":
        return any(kw in message.lower() for kw in ["error", "broken", "not working"])
    return False
```

### Input Filters

Filter the conversation history during handoffs:

```python
def remove_sensitive_info(input_data: HandoffInputData) -> HandoffInputData:
    """Remove sensitive information before handoff"""
    filtered_history = []
    for msg in input_data.conversation_history:
        # Remove credit card numbers, etc.
        filtered_content = redact_sensitive_info(msg.content)
        filtered_msg = Message(role=msg.role, content=filtered_content)
        filtered_history.append(filtered_msg)
    
    return HandoffInputData(
        conversation_history=filtered_history,
        system_message=input_data.system_message,
        metadata=input_data.metadata
    )
```

## Handoff Patterns

The Handoff System supports various patterns:

### Direct Handoff

The simplest pattern where one agent transfers control to another.

```
User → General Agent → Technical Agent → User
```

### Chain Handoff

Multiple agents handle the query in sequence.

```
User → General Agent → Technical Agent → Billing Agent → User
```

### Hub-and-Spoke

A central agent coordinates handoffs to specialized agents.

```
                 → Technical Agent →
User → General Agent → Billing Agent → General Agent → User
                 → Scheduling Agent →
```

## Best Practices

1. **Define clear agent roles**: Each agent should have a well-defined area of expertise.
2. **Use specific handoff conditions**: Be precise about when to transfer control.
3. **Maintain context**: Use input filters to preserve relevant context during handoffs.
4. **Handle errors gracefully**: Implement error handling to recover from failed handoffs.
5. **Test thoroughly**: Test different query types to ensure handoffs work as expected.

## Examples

### Basic Handoff Example

See the [handoff_multi_agent_example.py](../examples/handoff_multi_agent_example.py) file for a complete example of a handoff-enabled multi-agent system.

### Advanced Handoff Example

For more complex handoff patterns, see the [advanced_handoff_example.py](../examples/advanced_handoff_example.py) file.

### Custom Filter Example

```python
def preserve_user_messages_only(input_data: HandoffInputData) -> HandoffInputData:
    """Filter that keeps only user messages in the conversation history"""
    return HandoffInputData(
        conversation_history=[
            msg for msg in input_data.conversation_history
            if msg.role == MessageRole.USER
        ],
        system_message=input_data.system_message,
        metadata=input_data.metadata
    )
```

For more information, see the [handoff_documentation.md](./handoff_documentation.md) file. 