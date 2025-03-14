# Multi-Agent System Documentation

This document provides detailed information about the multi-agent system implementation in the framework.

## Overview

The multi-agent system is designed to break down complex tasks into smaller, more manageable components that can be handled by specialized agents. Each agent has a specific role and expertise, and they work together to provide more comprehensive and nuanced responses than a single agent could achieve alone.

## System Architecture

The system follows a pipeline architecture with three main components:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Analyzer  │────>│ Researcher  │────>│ Synthesizer │
└─────────────┘     └─────────────┘     └─────────────┘
```

1. **Analyzer Agent**: Breaks down complex questions into key components that need investigation
2. **Researcher Agent**: Gathers information about each component identified by the Analyzer
3. **Synthesizer Agent**: Combines the analysis and research into a coherent, comprehensive response

## Agent Components

Each agent in the system consists of several key components:

### 1. Memory

Each agent has its own memory system that allows it to maintain context across interactions:

```python
memory = Memory(
    working_memory_size=10,  # Number of recent items to keep in working memory
    long_term_memory_size=50  # Maximum number of items in long-term memory
)
```

The memory system includes:
- **Working Memory**: Stores recent interactions for immediate context
- **Long-Term Memory**: Stores important information for future reference
- **Memory Search**: Allows agents to search for relevant information in their memory

### 2. Task Agent

Each agent is implemented as a `MemoryEnabledTaskAgent`, which extends the base `TaskAgent` with memory capabilities:

```python
agent = MemoryEnabledTaskAgent(
    model=model,
    memory=memory,
    system_message="Agent's system message"
)
```

The task agent handles:
- Processing messages
- Managing conversation history
- Integrating with memory
- Executing tools (if available)

### 3. Agent Runner

Each agent has a runner that manages its execution:

```python
runner = MemoryAgentRunner(
    agent=agent,
    memory_persistence_path="./memories"
)
```

The runner handles:
- Running the agent
- Managing memory persistence
- Error recovery
- Session management

## Agent Roles and Prompts

### Analyzer Agent

**Role**: Break down complex questions into key components that need investigation.

**System Prompt**:
```
You are an analytical agent. Break down complex questions into key components 
that need investigation. Be concise and precise.
```

**Example Input**:
```
Break down this question into key components: What are the potential implications 
of quantum computing on cybersecurity?
```

**Example Output**:
```
Here are the key components of the question:

1. Current Cybersecurity Threats
2. Quantum Computing Capabilities
3. Cybersecurity Systems and Protocols
4. New Attack Vectors
5. Defensive Strategies
6. Timeline and Feasibility
```

### Researcher Agent

**Role**: Gather information about specific components identified by the Analyzer.

**System Prompt**:
```
You are a research agent. Focus on providing key facts and information about 
the specified components. Be concise and informative.
```

**Example Input**:
```
Research these components: [Components from Analyzer]
```

**Example Output**:
```
Here's a concise and informative research report on the specified components:

Current Cybersecurity Threats:
- Existing threats include phishing, ransomware, data breaches
- Quantum computing could accelerate password cracking

Quantum Computing Capabilities:
- Quantum parallelism enables solving complex problems faster
- Can break certain encryption algorithms
...
```

### Synthesizer Agent

**Role**: Combine the analysis and research into a coherent, comprehensive response.

**System Prompt**:
```
You are a synthesis agent. Combine the provided analysis and research into a clear, 
coherent response. Be concise yet comprehensive.
```

**Example Input**:
```
Create a concise response using:
Analysis: [Analysis from Analyzer]
Research: [Research from Researcher]
Question: [Original Question]
```

**Example Output**:
```
The potential implications of quantum computing on cybersecurity are far-reaching. 
Quantum computers can potentially exploit existing cybersecurity threats by accelerating 
password cracking and breaking certain encryption algorithms. This poses significant 
risks to current cybersecurity systems and protocols.

To mitigate these threats, defensive strategies such as quantum-resistant cryptography 
and post-quantum cryptography are being developed. While quantum computing is still in 
its early stages, it's estimated that practical quantum computers capable of breaking 
current encryption could be available within 5-10 years.
```

## Information Flow

The multi-agent system processes queries through a sequential flow:

1. **User Query**: The user submits a complex question
2. **Analysis Phase**: The Analyzer breaks down the question into key components
3. **Research Phase**: The Researcher investigates each component
4. **Synthesis Phase**: The Synthesizer combines the analysis and research into a final response
5. **Response**: The system returns the complete results, including the analysis, research, and synthesis

## Memory and Session Management

Each agent maintains its own memory, which allows it to:
- Remember previous interactions
- Maintain context across multiple queries
- Store important information for future reference

Sessions are managed using unique session IDs:
- `analyzer_session`: Session ID for the Analyzer agent
- `researcher_session`: Session ID for the Researcher agent
- `synthesizer_session`: Session ID for the Synthesizer agent

This allows each agent to maintain its own conversation history and context.

## Implementation Details

### Agent Initialization

```python
async def initialize(self, model):
    # Create memory
    memory = Memory(working_memory_size=10, long_term_memory_size=50)
    
    # Create agent
    self.agent = MemoryEnabledTaskAgent(
        model=model,
        memory=memory,
        system_message=self.system_prompt
    )
    
    # Create runner
    self.runner = MemoryAgentRunner(
        agent=self.agent,
        memory_persistence_path="./memories"
    )
```

### Query Processing

```python
async def process_query(self, query):
    # Step 1: Analyzer breaks down the question
    analysis = await self.agents["analyzer"].process(
        f"Break down this question into key components: {query}",
        session_id="analyzer_session"
    )
    
    # Step 2: Researcher investigates each component
    research = await self.agents["researcher"].process(
        f"Research these components: {analysis}",
        session_id="researcher_session"
    )
    
    # Step 3: Synthesizer combines everything
    synthesis = await self.agents["synthesizer"].process(
        f"Create a response using:\nAnalysis: {analysis}\nResearch: {research}\nQuestion: {query}",
        session_id="synthesizer_session"
    )
    
    return {
        "analysis": analysis,
        "research": research,
        "synthesis": synthesis
    }
```

## Extending the System

### Adding New Agents

You can extend the system by adding new specialized agents:

```python
# Create a new agent
fact_checker_agent = Agent(
    name="FactChecker",
    system_prompt="You are a fact-checking agent. Verify the accuracy of information."
)
await fact_checker_agent.initialize(model)

# Add to the system
self.agents["fact_checker"] = fact_checker_agent

# Modify the process_query method to include the new agent
async def process_query(self, query):
    # ... existing steps ...
    
    # Add fact-checking step
    fact_check = await self.agents["fact_checker"].process(
        f"Verify this information:\n{synthesis}",
        session_id="fact_checker_session"
    )
    
    return {
        "analysis": analysis,
        "research": research,
        "synthesis": synthesis,
        "fact_check": fact_check
    }
```

### Customizing Agent Behavior

You can customize agent behavior by modifying their system prompts:

```python
# Create a specialized researcher for medical topics
medical_researcher = Agent(
    name="MedicalResearcher",
    system_prompt=(
        "You are a medical research agent specializing in healthcare topics. "
        "Provide accurate, evidence-based information about medical conditions, "
        "treatments, and healthcare systems. Include relevant medical terminology "
        "and cite authoritative sources when possible."
    )
)
```

## Performance Considerations

- **Memory Usage**: Each agent maintains its own memory, which can consume resources. Consider adjusting memory sizes based on your needs.
- **API Costs**: Each agent makes separate API calls to the LLM provider, which can increase costs. Consider using smaller models for simpler tasks.
- **Latency**: The sequential nature of the system means that the total response time is the sum of the time taken by each agent. Consider implementing parallel processing for independent tasks.

## Best Practices

1. **Clear Agent Roles**: Define clear, non-overlapping roles for each agent
2. **Concise Prompts**: Keep system prompts concise and focused on the agent's specific role
3. **Appropriate Memory Settings**: Adjust memory settings based on the complexity of the task
4. **Error Handling**: Implement proper error handling to recover from failures
5. **Session Management**: Use consistent session IDs to maintain context across interactions

## Example Applications

- **Research Assistant**: Help researchers explore complex topics by breaking them down into manageable components
- **Educational Tool**: Explain complex concepts by analyzing them, researching details, and synthesizing explanations
- **Decision Support**: Assist in decision-making by analyzing problems, researching options, and synthesizing recommendations
- **Content Creation**: Generate high-quality content by analyzing topics, researching details, and synthesizing coherent text

## Tool Integration

The multi-agent system can be enhanced with tools that provide specialized capabilities to each agent. Tools allow agents to perform specific actions, such as calculations, web searches, or data processing.

### Standard Tools

The framework includes several standard tools:

1. **Calculator Tool**: Allows agents to perform mathematical calculations
2. **DateTime Tool**: Provides date and time information
3. **HTTP Request Tool**: Enables agents to make HTTP requests
4. **JSON Parser Tool**: Helps agents parse and manipulate JSON data
5. **CSV Parser Tool**: Assists agents in working with CSV data
6. **File Reader Tool**: Allows agents to read files
7. **File Writer Tool**: Enables agents to write to files

### Custom Tools

You can create custom tools by extending the `BaseTool` class:

```python
from mihrabai.tools.base import BaseTool

class WikipediaSearchTool(BaseTool):
    """Custom tool for searching Wikipedia"""
    
    def __init__(self):
        super().__init__(
            name="wikipedia_search",
            description="Search Wikipedia for information on a given topic",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for Wikipedia"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 1)"
                    }
                },
                "required": ["query"]
            }
        )
    
    async def _execute(self, params: Dict[str, Any]) -> str:
        """Execute the Wikipedia search"""
        query = params.get("query")
        limit = params.get("limit", 1)
        
        # Implementation details...
        # ...
        
        return "Wikipedia search results..."
```

### Adding Tools to Agents

Tools can be added to agents during initialization:

```python
# Initialize tools
calculator_tool = CalculatorTool()
wikipedia_tool = WikipediaSearchTool()

# Create agent with tools
agent = MemoryEnabledTaskAgent(
    model=model,
    memory=memory,
    system_message="Your system message here",
    tools=[calculator_tool, wikipedia_tool]
)
```

### Example: Multi-Agent System with Tools

Here's an example of how to create a multi-agent system with tools:

```python
class ToolEnabledMultiAgentSystem:
    """Multi-agent system with tool capabilities"""
    def __init__(self):
        self.agents = {}
        self.model = None
    
    async def initialize(self):
        # Set up the model
        self.model = await MultiProviderModel.create(
            primary_model="llama3-70b-8192",
            required_capabilities={ModelCapability.CHAT, ModelCapability.FUNCTION_CALLING},
            optimize_for=OptimizationStrategy.PERFORMANCE
        )
        
        # Initialize tools
        calculator_tool = CalculatorTool()
        datetime_tool = DateTimeTool()
        
        # Create analyzer agent (with calculator tool)
        self.agents["analyzer"] = Agent(
            name="Analyzer",
            system_prompt="You are an analytical agent with calculation capabilities...",
            tools=[calculator_tool]
        )
        
        # Create researcher agent (with datetime tool)
        self.agents["researcher"] = Agent(
            name="Researcher",
            system_prompt="You are a research agent with date/time capabilities...",
            tools=[datetime_tool]
        )
        
        # Create synthesizer agent (with both tools)
        self.agents["synthesizer"] = Agent(
            name="Synthesizer",
            system_prompt="You are a synthesis agent with calculation and date/time capabilities...",
            tools=[calculator_tool, datetime_tool]
        )
        
        # Initialize all agents
        for agent in self.agents.values():
            await agent.initialize(self.model)
```

## Example Scripts

The framework includes several example scripts that demonstrate how to use the multi-agent system:

1. **Basic Multi-Agent Example**: A simple example of a multi-agent system with three agents (Analyzer, Researcher, and Synthesizer).
   - File: `examples/multi_agent_example.py`

2. **Tool-Enabled Multi-Agent Example**: An example that demonstrates how to use standard tools (Calculator and DateTime) with the multi-agent system.
   - File: `examples/multi_agent_tool_example.py`

3. **Custom Tool Multi-Agent Example**: An advanced example that shows how to create a custom Wikipedia search tool and integrate it into the multi-agent system.
   - File: `examples/custom_tool_multi_agent_example.py`

4. **Handoff Multi-Agent Example**: An example that demonstrates how to implement handoffs between agents, allowing specialized agents to handle specific types of queries.
   - File: `examples/handoff_multi_agent_example.py`

To run these examples:

```bash
# Set your API key
export GROQ_API_KEY="your-api-key"

# Run the basic example
python examples/multi_agent_example.py

# Run the tool-enabled example
python examples/multi_agent_tool_example.py

# Run the custom tool example
python examples/custom_tool_multi_agent_example.py

# Run the handoff example
python examples/handoff_multi_agent_example.py
```

## Agent Handoffs

The multi-agent system supports handoffs between agents, allowing specialized agents to handle specific types of queries. This enables building more complex agent systems where each agent has a specific role and can transfer control to other agents when needed.

### Handoff Configuration

To configure handoffs between agents:

```python
# Create specialized agents
technical_agent = HandoffAgent(
    name="Technical",
    system_prompt="You are a technical support specialist...",
    tools=[calculator_tool]
)

billing_agent = HandoffAgent(
    name="Billing",
    system_prompt="You are a billing specialist...",
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
        ),
        HandoffConfig(
            name="billing",
            description="the user has a billing inquiry",
            target_agent=billing_agent,
            input_filter=remove_all_tools
        )
    ]
)
```

### Handoff Process

When a handoff occurs:

1. The source agent identifies that a query should be handled by a specialized agent
2. The source agent transfers control to the target agent
3. The target agent processes the query and returns a response
4. The response is returned to the user

### Handoff Filters

Handoff filters allow you to control what information is passed between agents during a handoff:

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

### Handoff Patterns

The framework supports several handoff patterns:

1. **Direct Handoff**: One agent directly transfers control to another agent
2. **Chain Handoff**: Multiple agents in a predefined sequence
3. **Conditional Handoff**: Handoffs based on specific conditions

### Best Practices for Handoffs

1. **Clear Agent Roles**: Define clear boundaries for each agent's responsibilities
2. **Explicit Transfer Conditions**: Provide clear instructions on when to transfer
3. **Minimal Context Transfer**: Only transfer relevant conversation history
4. **User Experience**: Ensure transfers are seamless from the user's perspective
5. **Specialized Input Filters**: Create custom filters for specific agent needs
6. **Proper Introduction**: Each agent should understand its role in the system 