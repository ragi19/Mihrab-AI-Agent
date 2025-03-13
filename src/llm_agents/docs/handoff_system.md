# Agent Handoff System Documentation

This document provides detailed information about the handoff system implementation in the multi-agent framework.

## Overview

The handoff system enables agents to transfer control to other specialized agents when needed. This allows for building complex multi-agent systems where each agent has a specific role and expertise, and can collaborate to handle a wide range of user queries.

## Key Components

### HandoffConfig

The `HandoffConfig` class defines the configuration for a handoff between agents:

```python
class HandoffConfig:
    """Configuration for a handoff between agents"""
    def __init__(
        self,
        name: str,
        description: str,
        target_agent: 'HandoffAgent',
        input_filter: Optional[Callable] = None
    ):
        self.name = name
        self.description = description
        self.target_agent = target_agent
        self.input_filter = input_filter
```

- **name**: The name of the handoff, used to generate the handoff function name (e.g., `transfer_to_{name}`)
- **description**: A description of when to use this handoff, included in the agent's system prompt
- **target_agent**: The agent to transfer control to
- **input_filter**: An optional function to filter the conversation history before transferring

### HandoffInputData

The `HandoffInputData` class represents the data passed during a handoff:

```python
class HandoffInputData:
    """Data passed during a handoff between agents"""
    def __init__(
        self,
        conversation_history: List[Message],
        system_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.conversation_history = conversation_history
        self.system_message = system_message
        self.metadata = metadata or {}
```

- **conversation_history**: The conversation history to transfer
- **system_message**: An optional system message to provide context to the target agent
- **metadata**: Additional metadata to pass to the target agent

### HandoffAgent

The `HandoffAgent` class extends the basic agent with handoff capabilities:

```python
class HandoffAgent:
    """Agent wrapper with handoff capabilities for the multi-agent system"""
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: Optional[List[ToolConfig]] = None,
        handoffs: Optional[List[HandoffConfig]] = None
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.handoffs = handoffs or []
        self.agent = None
        self.runner = None
        self.handoff_functions = {}
```

The `HandoffAgent` class provides methods for:
- Initializing the agent with a model, tools, and handoff functions
- Creating a system message with handoff instructions
- Creating handoff functions for each handoff configuration
- Processing messages with potential handoffs

## Handoff Process

The handoff process involves several steps:

1. **Initialization**: Each agent is initialized with its handoff configurations
2. **System Message Creation**: The agent's system message is extended with handoff instructions
3. **Handoff Function Creation**: Handoff functions are created for each handoff configuration
4. **Message Processing**: When a message is received, the agent checks if it's a handoff request
5. **Handoff Execution**: If a handoff is requested, the agent transfers control to the target agent
6. **Response Return**: The response from the target agent is returned to the user

### System Message Creation

The agent's system message is extended with handoff instructions:

```python
def _create_system_message(self) -> str:
    """Create a system message with handoff instructions"""
    base_prompt = self.system_prompt
    
    # Add handoff instructions if there are handoffs
    if self.handoffs:
        handoff_instructions = "\n\nYou are part of a multi-agent system. You can transfer control to specialized agents when needed:"
        for handoff in self.handoffs:
            handoff_instructions += f"\n- Use transfer_to_{handoff.name} when {handoff.description}"
        
        return base_prompt + handoff_instructions
    
    return base_prompt
```

### Handoff Function Creation

Handoff functions are created for each handoff configuration:

```python
def _create_handoff_function(self, handoff_config: HandoffConfig) -> Callable:
    """Create a handoff function for the given configuration"""
    async def handoff_function(message: str) -> str:
        logger.info(f"Handoff from {self.name} to {handoff_config.target_agent.name}: {message}")
        
        # Prepare handoff data
        handoff_data = HandoffInputData(
            conversation_history=self.agent.conversation_history,
            system_message=f"This conversation was transferred from {self.name}. The user asked: {message}",
            metadata={"source_agent": self.name, "original_message": message}
        )
        
        # Apply input filter if provided
        if handoff_config.input_filter:
            handoff_data = handoff_config.input_filter(handoff_data)
        
        # Process with target agent
        response = await handoff_config.target_agent.process(
            message,
            session_id=f"{self.name}_to_{handoff_config.target_agent.name}"
        )
        
        return response
    
    return handoff_function
```

### Message Processing

When a message is received, the agent checks if it's a handoff request:

```python
async def process(self, message: str, session_id: str = None) -> str:
    """Process a message and return the response"""
    # Check if this is a handoff request
    for handoff_name, handoff_info in self.handoff_functions.items():
        if handoff_name.lower() in message.lower():
            # Extract the actual message from the handoff request
            actual_message = message.replace(handoff_name, "").strip()
            return await handoff_info["function"](actual_message)
    
    # Regular processing
    response = await self.runner.run(message, session_id=session_id)
    return response.content
```

## Handoff Filters

Handoff filters allow you to control what information is passed between agents during a handoff. The framework includes a built-in filter for removing tool-related messages:

```python
def remove_all_tools(input_data: HandoffInputData) -> HandoffInputData:
    """Filter out all tool-related items from conversation history"""
    filtered_history = []
    
    for message in input_data.conversation_history:
        # Skip tool-related messages
        if hasattr(message, 'tool_calls') and message.tool_calls:
            continue
        
        # Include regular messages
        filtered_history.append(message)
    
    input_data.conversation_history = filtered_history
    return input_data
```

You can create custom filters to suit your specific needs:

```python
def custom_filter(input_data: HandoffInputData) -> HandoffInputData:
    """Custom filter for handoff data"""
    # Start with removing all tools
    filtered_data = remove_all_tools(input_data)
    
    # Additional filtering logic
    filtered_history = []
    for message in filtered_data.conversation_history:
        # Apply custom filtering logic
        if some_condition(message):
            filtered_history.append(message)
    
    filtered_data.conversation_history = filtered_history
    return filtered_data
```

## Handoff Patterns

The framework supports several handoff patterns:

### Direct Handoff

The simplest form of handoff where one agent directly transfers control to another:

```python
# Create specialized agents
technical_agent = HandoffAgent(
    name="Technical",
    system_prompt="You are a technical support specialist...",
    tools=[calculator_tool]
)

# Create a general agent with handoffs to specialized agents
general_agent = HandoffAgent(
    name="General",
    system_prompt="You are a general customer service agent...",
    handoffs=[
        HandoffConfig(
            name="technical",
            description="the user has a technical issue",
            target_agent=technical_agent,
            input_filter=remove_all_tools
        )
    ]
)
```

### Chain Handoff

Multiple agents in a predefined sequence:

```python
# Create specialized agents
intake_agent = HandoffAgent(
    name="Intake",
    system_prompt="You are an intake agent...",
    handoffs=[
        HandoffConfig(
            name="analysis",
            description="you have collected all necessary information",
            target_agent=analysis_agent,
            input_filter=remove_all_tools
        )
    ]
)

analysis_agent = HandoffAgent(
    name="Analysis",
    system_prompt="You are an analysis agent...",
    handoffs=[
        HandoffConfig(
            name="resolution",
            description="you have completed the analysis",
            target_agent=resolution_agent,
            input_filter=remove_all_tools
        )
    ]
)

resolution_agent = HandoffAgent(
    name="Resolution",
    system_prompt="You are a resolution agent..."
)
```

### Conditional Handoff

Handoffs based on specific conditions:

```python
# Create specialized agents
technical_agent = HandoffAgent(
    name="Technical",
    system_prompt="You are a technical support specialist..."
)

billing_agent = HandoffAgent(
    name="Billing",
    system_prompt="You are a billing specialist..."
)

scheduling_agent = HandoffAgent(
    name="Scheduling",
    system_prompt="You are a scheduling assistant..."
)

# Create a general agent with conditional handoffs
general_agent = HandoffAgent(
    name="General",
    system_prompt="You are a general customer service agent...",
    handoffs=[
        HandoffConfig(
            name="technical",
            description="the user has a technical issue",
            target_agent=technical_agent,
            input_filter=remove_all_tools
        ),
        HandoffConfig(
            name="billing",
            description="the user has a billing inquiry",
            target_agent=billing_agent,
            input_filter=remove_all_tools
        ),
        HandoffConfig(
            name="scheduling",
            description="the user needs help with scheduling",
            target_agent=scheduling_agent,
            input_filter=remove_all_tools
        )
    ]
)
```

## Best Practices

### Clear Agent Roles

Define clear boundaries for each agent's responsibilities:

```python
technical_agent = HandoffAgent(
    name="Technical",
    system_prompt=(
        "You are a technical support specialist with deep knowledge of computer systems. "
        "Help users with technical problems, troubleshooting, and system configurations. "
        "Be precise and thorough in your explanations."
    )
)

billing_agent = HandoffAgent(
    name="Billing",
    system_prompt=(
        "You are a billing specialist with expertise in financial matters. "
        "Help users with billing inquiries, payment issues, and subscription questions. "
        "Be clear and accurate in your responses."
    )
)
```

### Explicit Transfer Conditions

Provide clear instructions on when to transfer:

```python
general_agent = HandoffAgent(
    name="General",
    system_prompt=(
        "You are a general customer service agent who can help with a wide range of inquiries. "
        "For technical issues, transfer to the Technical agent. "
        "For billing questions, transfer to the Billing agent. "
        "For scheduling needs, transfer to the Scheduling agent. "
        "Handle general inquiries yourself. Be friendly and helpful."
    ),
    handoffs=[
        HandoffConfig(
            name="technical",
            description="the user has a technical issue or question about computer systems",
            target_agent=technical_agent
        ),
        HandoffConfig(
            name="billing",
            description="the user has a billing inquiry or payment issue",
            target_agent=billing_agent
        ),
        HandoffConfig(
            name="scheduling",
            description="the user needs help with scheduling or calendar management",
            target_agent=scheduling_agent
        )
    ]
)
```

### Minimal Context Transfer

Only transfer relevant conversation history:

```python
def technical_filter(input_data: HandoffInputData) -> HandoffInputData:
    """Filter for technical handoffs"""
    # Remove tool-related messages
    filtered_data = remove_all_tools(input_data)
    
    # Keep only messages related to technical issues
    filtered_history = []
    for message in filtered_data.conversation_history:
        if is_technical_message(message):
            filtered_history.append(message)
    
    filtered_data.conversation_history = filtered_history
    return filtered_data
```

### User Experience

Ensure transfers are seamless from the user's perspective:

```python
async def process_query(self, query: str) -> str:
    """Process a query through the general agent with potential handoffs"""
    logger.info("\nProcessing query through handoff-enabled multi-agent system...")
    
    # Start with the general agent
    response = await self.agents["general"].process(
        query,
        session_id="general_session"
    )
    
    logger.info("\nQuery processing complete.")
    return response
```

## Example Implementation

Here's a complete example of a handoff-enabled multi-agent system:

```python
class HandoffMultiAgentSystem:
    """Multi-agent system with handoff capabilities"""
    def __init__(self):
        self.agents = {}
        self.model = None
    
    async def initialize(self):
        """Initialize the multi-agent system with handoffs"""
        # Set up the model
        self.model = await MultiProviderModel.create(
            primary_model="llama3-70b-8192",
            required_capabilities={ModelCapability.CHAT, ModelCapability.FUNCTION_CALLING},
            optimize_for=OptimizationStrategy.PERFORMANCE
        )
        
        # Initialize tools
        calculator_tool = CalculatorTool()
        datetime_tool = DateTimeTool()
        
        # Create specialized agents first (without handoffs)
        self.agents["technical"] = HandoffAgent(
            name="Technical",
            system_prompt=(
                "You are a technical support specialist with deep knowledge of computer systems. "
                "Help users with technical problems, troubleshooting, and system configurations. "
                "Use the calculator tool when needed for technical calculations. "
                "Be precise and thorough in your explanations."
            ),
            tools=[calculator_tool]
        )
        
        self.agents["billing"] = HandoffAgent(
            name="Billing",
            system_prompt=(
                "You are a billing specialist with expertise in financial matters. "
                "Help users with billing inquiries, payment issues, and subscription questions. "
                "Use the calculator tool when needed for financial calculations. "
                "Be clear and accurate in your responses."
            ),
            tools=[calculator_tool]
        )
        
        self.agents["scheduling"] = HandoffAgent(
            name="Scheduling",
            system_prompt=(
                "You are a scheduling assistant with excellent time management skills. "
                "Help users with scheduling appointments, managing calendars, and planning events. "
                "Use the datetime tool to provide accurate time and date information. "
                "Be efficient and helpful in your responses."
            ),
            tools=[datetime_tool]
        )
        
        # Create the general agent with handoffs to specialized agents
        self.agents["general"] = HandoffAgent(
            name="General",
            system_prompt=(
                "You are a general customer service agent who can help with a wide range of inquiries. "
                "For technical issues, transfer to the Technical agent. "
                "For billing questions, transfer to the Billing agent. "
                "For scheduling needs, transfer to the Scheduling agent. "
                "Handle general inquiries yourself. Be friendly and helpful."
            ),
            tools=[],
            handoffs=[
                HandoffConfig(
                    name="technical",
                    description="the user has a technical issue or question about computer systems",
                    target_agent=self.agents["technical"],
                    input_filter=remove_all_tools
                ),
                HandoffConfig(
                    name="billing",
                    description="the user has a billing inquiry or payment issue",
                    target_agent=self.agents["billing"],
                    input_filter=remove_all_tools
                ),
                HandoffConfig(
                    name="scheduling",
                    description="the user needs help with scheduling or calendar management",
                    target_agent=self.agents["scheduling"],
                    input_filter=remove_all_tools
                )
            ]
        )
        
        # Initialize all agents
        for agent in self.agents.values():
            await agent.initialize(self.model)
    
    async def process_query(self, query: str) -> str:
        """Process a query through the general agent with potential handoffs"""
        logger.info("\nProcessing query through handoff-enabled multi-agent system...")
        
        # Start with the general agent
        response = await self.agents["general"].process(
            query,
            session_id="general_session"
        )
        
        logger.info("\nQuery processing complete.")
        return response
```

## Conclusion

The handoff system enables building complex multi-agent systems where agents can collaborate to handle a wide range of user queries. By defining clear agent roles, explicit transfer conditions, and appropriate context filtering, you can create a seamless user experience with specialized agents handling different aspects of the conversation. 