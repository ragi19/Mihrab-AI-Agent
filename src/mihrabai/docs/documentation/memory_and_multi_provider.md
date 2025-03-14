# Memory and Multi-Provider Features

This guide explains how to use the memory-enabled task agent and multi-provider model features in the LLM Agents framework.

## Memory-Enabled Task Agent

The `MemoryEnabledTaskAgent` extends the base task agent with conversation memory capabilities:

```python
from mihrabai.core.memory_task_agent import MemoryEnabledTaskAgent

agent = MemoryEnabledTaskAgent(
    model=model,
    max_memory_items=100,  # Maximum memories to store
    memory_retrieval_count=5,  # Number of relevant memories to retrieve per query
    automatic_memory=True  # Automatically store memories from conversations
)
```

### Memory Features

- **Persistent Memory**: Memories can be saved and loaded between sessions using `MemoryAgentRunner`
- **Relevance-Based Retrieval**: Only the most relevant memories are retrieved for each query
- **Automatic Memory Management**: Old memories are automatically pruned when `max_memory_items` is reached
- **Memory Summarization**: Get conversation summaries with `runner.summarize_conversation()`

## Multi-Provider Model

The `MultiProviderModel` enables using multiple LLM providers with automatic failover and optimization:

```python
from mihrabai.models.multi_provider import MultiProviderModel, OptimizationStrategy

model = await MultiProviderModel.create(
    primary_model="claude-3-sonnet",  # Primary provider
    fallback_models=[  # Fallback providers in order
        "gpt-4-0125-preview",
        "mixtral-8x7b-instruct"
    ],
    required_capabilities={  # Required model capabilities
        ModelCapability.CHAT,
        ModelCapability.SYSTEM_MESSAGES
    },
    optimize_for=OptimizationStrategy.RELIABILITY  # Optimization strategy
)
```

### Features

- **Automatic Failover**: Switches to fallback providers if primary fails
- **Provider Statistics**: Tracks success rates, latency, and token usage
- **Optimization Strategies**:
  - `RELIABILITY`: Prefer providers with higher success rates
  - `PERFORMANCE`: Prefer providers with lower latency
  - `COST`: Prefer providers with lower token costs

### Provider Statistics

The `ProviderStats` class tracks:
- Success/failure counts
- Request durations
- Token usage
- Average latency
- Success rates

Access stats with:
```python
stats = model.get_provider_stats()
```

## Example Usage

See `examples/advanced/memory_multi_provider_example.py` for a complete example showing:
- Setting up memory-enabled agents
- Configuring multi-provider models
- Persisting memories between sessions
- Handling provider failover
- Accessing provider statistics

## Best Practices

1. **Memory Management**
   - Set reasonable `max_memory_items` based on your use case
   - Use `memory_retrieval_count` to control context size
   - Call `save_memory()` periodically for persistence

2. **Provider Configuration**
   - Order fallback providers by preference
   - Choose optimization strategy based on requirements
   - Monitor provider stats to optimize configuration

3. **Error Handling**
   - Handle provider switching events appropriately 
   - Implement retry logic for transient failures
   - Log provider statistics for monitoring