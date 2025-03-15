# Mihrab AI Agent - API Reference

This document provides a detailed reference of the Mihrab AI Agent package's API.

## Table of Contents

- [Core API](#core-api)
  - [create_agent](#create_agent)
  - [Message](#message)
  - [MessageRole](#messagerole)
  - [Agent](#agent)
- [Tools API](#tools-api)
  - [BaseTool](#basetool)
  - [ToolRegistry](#toolregistry)
  - [Standard Tools](#standard-tools)
- [Models API](#models-api)
  - [ModelCapability](#modelcapability)
  - [BaseModel](#basemodel)
  - [Providers](#providers)
- [Handoff API](#handoff-api)
  - [HandoffManager](#handoffmanager)
- [Utility API](#utility-api)
  - [Logging](#logging)
  - [Configuration](#configuration)

## Core API

### create_agent

```python
async def create_agent(
    provider_name: str,
    model_name: str,
    system_message: str = None,
    required_capabilities: Set[ModelCapability] = None,
    **kwargs
) -> Agent
```

Creates a new agent with the specified provider and model.

**Parameters:**
- `provider_name` (str): The name of the provider (e.g., "openai", "anthropic", "groq")
- `model_name` (str): The name of the model to use
- `system_message` (str, optional): The system message to use for the agent
- `required_capabilities` (Set[ModelCapability], optional): The capabilities required by the agent
- `**kwargs`: Additional arguments to pass to the provider

**Returns:**
- `Agent`: A new agent instance

**Example:**
```python
agent = await create_agent(
    provider_name="openai",
    model_name="gpt-4o",
    system_message="You are a helpful AI assistant.",
    required_capabilities={ModelCapability.FUNCTION_CALLING}
)
```

### Message

```python
class Message:
    def __init__(
        self,
        role: MessageRole,
        content: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    )
```

Represents a message in a conversation.

**Parameters:**
- `role` (MessageRole): The role of the message sender
- `content` (str): The content of the message
- `name` (str, optional): The name of the function for function messages
- `metadata` (Dict[str, Any], optional): Additional metadata for the message

**Example:**
```python
message = Message(
    role=MessageRole.USER,
    content="Hello, who are you?",
    metadata={"timestamp": "2023-06-01T12:00:00Z"}
)
```

### MessageRole

```python
class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
```

Enum representing the role of a message sender.

**Values:**
- `SYSTEM`: System messages that provide instructions to the model
- `USER`: Messages from the user
- `ASSISTANT`: Messages from the assistant
- `FUNCTION`: Messages containing function results

### Agent

```python
class Agent:
    async def process_message(self, message: Message) -> Message
    def add_to_history(self, message: Message) -> None
    def add_tool(self, tool: BaseTool) -> None
    def get_tools(self) -> List[BaseTool]
    def clear_history(self) -> None
```

Represents an AI agent that can process messages and use tools.

**Methods:**
- `process_message(message)`: Processes a message and returns a response
- `add_to_history(message)`: Adds a message to the agent's conversation history
- `add_tool(tool)`: Adds a tool to the agent
- `get_tools()`: Returns the agent's tools
- `clear_history()`: Clears the agent's conversation history

**Example:**
```python
# Create a user message
message = Message(role=MessageRole.USER, content="Hello, who are you?")

# Process the message
response = await agent.process_message(message)

# Print the response
print(f"Agent: {response.content}")
```

## Tools API

### BaseTool

```python
class BaseTool(ABC):
    def __init__(self, name: str, description: str)
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> JSON
    
    @abstractmethod
    async def _execute(self, parameters: Dict[str, Any]) -> JSON
    
    @abstractmethod
    def _get_parameters_schema(self) -> Dict[str, Any]
    
    def get_schema(self) -> Dict[str, Any]
```

Base class for all tools.

**Methods:**
- `execute(parameters)`: Executes the tool with the given parameters
- `_execute(parameters)`: Internal method for tool-specific execution logic
- `_get_parameters_schema()`: Returns the JSON schema for tool parameters
- `get_schema()`: Returns the JSON schema for the tool

**Example:**
```python
class MyCustomTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_custom_tool",
            description="A custom tool that does something useful"
        )
    
    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        # Implement your tool logic here
        result = {"status": "success", "message": "Custom tool executed"}
        return result
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "First parameter"
                },
                "param2": {
                    "type": "integer",
                    "description": "Second parameter"
                }
            },
            "required": ["param1"]
        }
```

### ToolRegistry

```python
class ToolRegistry:
    @classmethod
    def register(cls, name: str, tool_class: Type[BaseTool]) -> None
    
    @classmethod
    def get_tool(cls, name: str) -> Optional[Type[BaseTool]]
    
    @classmethod
    def list_tools(cls) -> list[str]
    
    @classmethod
    def create_tool(cls, name: str, **kwargs: Any) -> BaseTool
    
    @classmethod
    def get_schemas(cls) -> List[Dict]
```

Registry for managing and accessing tool implementations.

**Methods:**
- `register(name, tool_class)`: Registers a new tool implementation
- `get_tool(name)`: Gets a tool implementation by name
- `list_tools()`: Lists all registered tool names
- `create_tool(name, **kwargs)`: Creates a new tool instance
- `get_schemas()`: Gets JSON schemas for all registered tools

**Example:**
```python
# Register a tool
ToolRegistry.register("my_custom_tool", MyCustomTool)

# Create a tool instance
tool = ToolRegistry.create_tool("my_custom_tool")

# Add the tool to an agent
agent.add_tool(tool)
```

### Standard Tools

Mihrab AI Agent provides a wide range of standard tools:

#### Web Tools

- `HTTPRequestTool`: Makes HTTP requests to web endpoints
- `WebScraperTool`: Scrapes content from web pages

#### Filesystem Tools

- `FileReadTool`: Reads the contents of a file
- `FileWriteTool`: Writes content to a file
- `FileListTool`: Lists files in a directory

#### Code Generation Tools

- `CodeGeneratorTool`: Generates code based on a description
- `CodeReviewTool`: Reviews code and provides feedback
- `CodeExecutorTool`: Executes code and returns the result

#### Text Processing Tools

- `TextSummarizerTool`: Summarizes text
- `TextAnalysisTool`: Analyzes text for sentiment, entities, etc.

#### Memory Tools

- `MemoryStoreTool`: Stores information in the agent's memory
- `MemoryRetrieveTool`: Retrieves information from the agent's memory

#### Data Analysis Tools

- `DataAnalysisTool`: Analyzes data and generates insights
- `DataVisualizationTool`: Creates visualizations from data

#### Document Tools

- `DocumentWriterTool`: Creates and writes documents
- `DocumentReaderTool`: Reads and extracts information from documents

#### Utility Tools

- `ShellTool`: Executes shell commands
- `SearchTool`: Searches for information
- `EvaluationTool`: Evaluates model outputs

## Models API

### ModelCapability

```python
class ModelCapability(Enum):
    FUNCTION_CALLING = "function_calling"
    CODE_GENERATION = "code_generation"
    VISION = "vision"
    STREAMING = "streaming"
```

Enum representing the capabilities of a model.

**Values:**
- `FUNCTION_CALLING`: The model can call functions
- `CODE_GENERATION`: The model can generate code
- `VISION`: The model can process images
- `STREAMING`: The model supports streaming responses

### BaseModel

```python
class BaseModel(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Message
    
    @abstractmethod
    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> AsyncGenerator[Message, None]
    
    @property
    @abstractmethod
    def capabilities(self) -> Set[ModelCapability]
```

Base class for all models.

**Methods:**
- `generate(messages, tools, **kwargs)`: Generates a response to the given messages
- `stream(messages, tools, **kwargs)`: Streams a response to the given messages
- `capabilities`: Returns the capabilities of the model

### Providers

Mihrab AI Agent supports multiple providers:

- `OpenAIProvider`: Provider for OpenAI models
- `AnthropicProvider`: Provider for Anthropic models
- `GroqProvider`: Provider for Groq models

## Handoff API

### HandoffManager

```python
class HandoffManager:
    def register_agent(self, agent_id: str, agent: Agent) -> None
    
    async def process_with_handoff(
        self,
        agent_id: str,
        message: Message,
        **kwargs
    ) -> Message
```

Manages handoffs between agents.

**Methods:**
- `register_agent(agent_id, agent)`: Registers an agent with the handoff manager
- `process_with_handoff(agent_id, message, **kwargs)`: Processes a message with potential handoffs

**Example:**
```python
# Create a handoff manager
handoff_manager = HandoffManager()

# Register agents with the handoff manager
handoff_manager.register_agent("general", general_agent)
handoff_manager.register_agent("code_specialist", code_specialist)

# Process a message with potential handoffs
result = await handoff_manager.process_with_handoff(
    agent_id="general",
    message=user_message
)
```

## Utility API

### Logging

```python
def get_logger(name: str) -> logging.Logger
```

Gets a logger with the given name.

**Parameters:**
- `name` (str): The name of the logger

**Returns:**
- `logging.Logger`: A logger instance

**Example:**
```python
from mihrabai.utils.logging import get_logger

logger = get_logger("my_module")
logger.info("This is an info message")
logger.error("This is an error message")
```

### Configuration

```python
def load_config(config_path: str = None) -> Dict[str, Any]
```

Loads the configuration from a file.

**Parameters:**
- `config_path` (str, optional): The path to the configuration file

**Returns:**
- `Dict[str, Any]`: The configuration

**Example:**
```python
from mihrabai.config import load_config

config = load_config()
print(f"Default model: {config['default_model']}")
``` 