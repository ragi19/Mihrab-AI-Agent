# Advanced Handoff System Documentation

This document provides a detailed explanation of the advanced handoff system implementation, which enables sophisticated interactions between multiple specialized agents.

## Table of Contents

1. [Overview](#overview)
2. [Key Components](#key-components)
3. [Handoff Patterns](#handoff-patterns)
4. [Context Preservation](#context-preservation)
5. [Custom Filters](#custom-filters)
6. [Conditional Handoffs](#conditional-handoffs)
7. [Implementation Details](#implementation-details)
8. [Example Usage](#example-usage)
9. [Best Practices](#best-practices)

## Overview

The advanced handoff system extends the basic handoff capabilities by adding:

- **Context preservation** between agent handoffs
- **Conditional handoffs** based on message content analysis
- **Chain handoffs** where agents can pass control to multiple agents in sequence
- **Custom filters** for modifying conversation history during handoffs
- **Specialized agent roles** with different capabilities and tools

This system allows for more sophisticated multi-agent interactions, enabling complex workflows where different specialized agents can collaborate to solve user queries.

## Key Components

### HandoffConfig

The `HandoffConfig` class has been enhanced with additional capabilities:

```python
class HandoffConfig:
    def __init__(
        self,
        name: str,
        description: str,
        target_agent: 'HandoffAgent',
        condition: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        input_filter: Optional[Callable[['HandoffInputData'], 'HandoffInputData']] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.description = description
        self.target_agent = target_agent
        self.condition = condition  # Custom condition function
        self.input_filter = input_filter  # Custom filter function
        self.metadata = metadata or {}
```

Key enhancements:
- **Custom condition function**: Determines when to trigger a handoff based on message content
- **Custom input filter**: Transforms conversation history during handoffs
- **Metadata**: Additional information that can be passed to the target agent

### HandoffInputData

The `HandoffInputData` class now includes:

```python
class HandoffInputData:
    def __init__(
        self,
        conversation_history: List[Message],
        system_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_agent: Optional[str] = None,
        handoff_chain: Optional[List[str]] = None
    ):
        self.conversation_history = conversation_history
        self.system_message = system_message
        self.metadata = metadata or {}
        self.source_agent = source_agent
        self.handoff_chain = handoff_chain or []
```

Key enhancements:
- **Source agent**: Tracks which agent initiated the handoff
- **Handoff chain**: Records the sequence of agents involved in processing a query

### HandoffAgent

The `HandoffAgent` class has been significantly enhanced:

```python
class HandoffAgent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: Optional[List[Any]] = None,
        handoffs: Optional[List[HandoffConfig]] = None,
        can_defer: bool = False,
        expertise: Optional[List[str]] = None
    ):
        # ... initialization ...
        self.can_defer = can_defer  # Whether this agent can defer to other agents
        self.expertise = expertise or []  # Areas of expertise
        self.conversation_context = {}  # Store context between conversations
```

Key enhancements:
- **Expertise areas**: Defines what topics the agent specializes in
- **Conversation context**: Maintains state between handoffs
- **Can defer flag**: Indicates if the agent can hand off to other agents

## Handoff Patterns

The advanced system supports several handoff patterns:

### Direct Handoff

The simplest pattern where one agent transfers control directly to another specialized agent:

```
General Agent -> Technical Agent
```

Example implementation:
```python
general_agent.handoffs = [
    HandoffConfig(
        name="technical",
        description="technical issue",
        target_agent=technical_agent
    )
]
```

### Chain Handoff

Multiple agents can be involved in sequence to handle a complex query:

```
General Agent -> Research Agent -> Calculation Agent
```

This is implemented by giving specialized agents their own handoff configurations:

```python
research_agent.handoffs = [
    HandoffConfig(
        name="calculation",
        description="calculation needed",
        target_agent=calculation_agent
    )
]
```

### Conditional Handoff

Handoffs can be triggered based on complex conditions:

```python
def technical_complexity_condition(message: str, context: Dict[str, Any]) -> bool:
    technical_terms = ["api", "database", "server", "code", "programming"]
    term_count = sum(1 for term in technical_terms if term in message.lower())
    return term_count >= 2 or len(message.split()) > 20

general_agent.handoffs = [
    HandoffConfig(
        name="technical",
        description="complex technical issue",
        target_agent=technical_agent,
        condition=technical_complexity_condition
    )
]
```

## Context Preservation

The advanced system preserves context between handoffs, allowing agents to share information:

```python
async def process(self, message: str, session_id: str = None, context: Dict[str, Any] = None):
    # Update conversation context
    if context:
        self.conversation_context.update(context)
    
    # ... process message ...
    
    # Pass context to target agent during handoff
    handoff_context = self.conversation_context.copy()
    handoff_context.update({
        "source_agent": self.name,
        "handoff_reason": handoff.description
    })
    
    result = await handoff.target_agent.process(
        message, 
        session_id=session_id,
        context=handoff_context
    )
```

This allows agents to:
1. Know which agent handed off to them
2. Understand why the handoff occurred
3. Access information gathered by previous agents

## Custom Filters

Custom filters transform the conversation history during handoffs:

### Preserve User Messages Only

```python
def preserve_user_messages_only(input_data: HandoffInputData) -> HandoffInputData:
    return HandoffInputData(
        conversation_history=[
            msg for msg in input_data.conversation_history
            if msg.role == MessageRole.USER
        ],
        system_message=input_data.system_message,
        metadata=input_data.metadata,
        source_agent=input_data.source_agent,
        handoff_chain=input_data.handoff_chain
    )
```

### Summarize Previous Responses

```python
def summarize_previous_responses(input_data: HandoffInputData) -> HandoffInputData:
    user_messages = [msg for msg in input_data.conversation_history if msg.role == MessageRole.USER]
    
    # Create a summary of assistant responses
    assistant_messages = [msg for msg in input_data.conversation_history if msg.role == MessageRole.ASSISTANT]
    if assistant_messages:
        summary = "Previous assistant responses summary:\n"
        for i, msg in enumerate(assistant_messages):
            summary += f"- Response {i+1}: {msg.content[:100]}...\n"
        
        # Add a system message with the summary
        new_system = input_data.system_message + "\n\n" + summary if input_data.system_message else summary
    else:
        new_system = input_data.system_message
    
    return HandoffInputData(
        conversation_history=user_messages,
        system_message=new_system,
        metadata=input_data.metadata,
        source_agent=input_data.source_agent,
        handoff_chain=input_data.handoff_chain
    )
```

## Conditional Handoffs

The system supports sophisticated conditions for determining when to hand off:

### Technical Complexity Detection

```python
def technical_complexity_condition(message: str, context: Dict[str, Any]) -> bool:
    technical_terms = [
        "api", "database", "server", "code", "programming", "algorithm",
        "network", "configuration", "integration", "deployment"
    ]
    
    # Count technical terms
    term_count = sum(1 for term in technical_terms if term in message.lower())
    
    # Check message length as a proxy for complexity
    is_complex = term_count >= 2 or len(message.split()) > 20
    
    return is_complex
```

### Research Need Detection

```python
def needs_research_condition(message: str, context: Dict[str, Any]) -> bool:
    research_indicators = [
        "research", "information about", "tell me about", "what is", "how does",
        "explain", "details on", "background", "history of", "compare"
    ]
    
    return any(indicator in message.lower() for indicator in research_indicators)
```

### Calculation Need Detection

```python
def needs_calculation_condition(message: str, context: Dict[str, Any]) -> bool:
    # Look for numbers and mathematical operators
    has_numbers = bool(re.search(r'\d+', message))
    has_operators = any(op in message for op in ['+', '-', '*', '/', 'plus', 'minus', 'multiply', 'divide', 'calculate'])
    
    return has_numbers and has_operators
```

## Implementation Details

### Context Extraction

The system can automatically extract structured information from agent responses:

```python
def _extract_context_from_response(self, response: str) -> Dict[str, Any]:
    context = {}
    
    # Look for JSON-like structures
    json_pattern = r'\{[\s\S]*?\}'
    json_matches = re.findall(json_pattern, response)
    for json_str in json_matches:
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                context.update(data)
        except:
            pass
    
    # Look for key-value pairs in the format "Key: Value"
    kv_pattern = r'([A-Za-z\s]+):\s*([^:\n]+)'
    kv_matches = re.findall(kv_pattern, response)
    for key, value in kv_matches:
        clean_key = key.strip().lower().replace(' ', '_')
        clean_value = value.strip()
        if clean_key and clean_value:
            context[clean_key] = clean_value
    
    return context
```

### Session Management

The system maintains session data across multiple queries:

```python
class AdvancedHandoffSystem:
    def __init__(self):
        self.agents = {}
        self.model = None
        self.session_data = {}  # Store session-specific data
    
    async def process_query(self, query: str, session_id: str = "default_session"):
        # Initialize or retrieve session data
        if session_id not in self.session_data:
            self.session_data[session_id] = {
                "queries": [],
                "handoff_chains": [],
                "context": {}
            }
        
        session = self.session_data[session_id]
        session["queries"].append(query)
        
        # ... process query ...
```

## Example Usage

Here's an example of setting up an advanced handoff system:

```python
# Create specialized agents
technical_agent = HandoffAgent(
    name="Technical",
    system_prompt="You are a technical support agent...",
    tools=[calculator_tool],
    expertise=["computers", "software", "troubleshooting"]
)

research_agent = HandoffAgent(
    name="Research",
    system_prompt="You are a research agent...",
    expertise=["research", "information gathering", "data analysis"]
)

# Set up handoff configurations for the technical agent
technical_agent.handoffs = [
    HandoffConfig(
        name="research",
        description="the query requires in-depth research",
        target_agent=research_agent,
        condition=needs_research_condition,
        input_filter=preserve_user_messages_only
    )
]

# Create general agent with handoff capabilities
general_agent = HandoffAgent(
    name="General",
    system_prompt="You are a general customer service agent...",
    can_defer=True,
    handoffs=[
        HandoffConfig(
            name="technical",
            description="technical issue",
            target_agent=technical_agent,
            condition=technical_complexity_condition,
            input_filter=preserve_user_messages_only
        ),
        # ... other handoffs ...
    ]
)

# Process a query
result = await system.process_query(
    "Can you explain how database indexing works and why it's important for performance?",
    session_id="user123"
)
```

## Best Practices

1. **Define Clear Agent Roles**: Each agent should have a well-defined area of expertise.

2. **Use Conditional Handoffs**: Implement custom conditions to determine when to hand off to specialized agents.

3. **Preserve Context**: Ensure important information is passed between agents during handoffs.

4. **Filter Conversation History**: Use custom filters to provide only relevant information to target agents.

5. **Monitor Handoff Chains**: Track the sequence of agents involved in processing a query to identify potential loops or inefficiencies.

6. **Implement Fallbacks**: Ensure the system can handle cases where no specialized agent is appropriate.

7. **Balance Specialization**: Too many specialized agents can lead to excessive handoffs, while too few can result in inadequate expertise.

8. **Test Complex Scenarios**: Test the system with queries that might trigger multiple handoffs to ensure proper functioning.

9. **Provide Transparency**: Make it clear to users when their query is being handled by a different agent.

10. **Optimize Tool Usage**: Equip agents with the tools they need for their specific domain to enhance their capabilities. 