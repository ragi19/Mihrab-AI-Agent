# Advanced Examples for Mihrab AI Agent

This directory contains advanced examples that demonstrate sophisticated features of the Mihrab AI Agent framework.

## Model Selection and Fallback Example

The `model_fallback_example.py` file demonstrates the enhanced handoff system with dynamic model selection and fallback capabilities:

### Key Features

1. **Dynamic Model Selection**: Automatically selects the most appropriate model based on availability and required capabilities.

2. **Model Fallback Mechanisms**: Gracefully handles model failures by falling back to alternative models.

3. **Multi-Provider Support**: Works with multiple model providers (OpenAI, Anthropic, Groq) for redundancy.

4. **Capability-Based Selection**: Selects models based on specific capabilities (e.g., function calling, streaming).

5. **Priority-Based Selection**: Uses model priority to determine the preferred model when multiple options are available.

6. **Round-Robin Selection**: Cycles through available models in a round-robin fashion to distribute load and provide redundancy.

### Selection Strategies

The example demonstrates three different model selection strategies:

1. **PrioritizedModelStrategy**: Selects models based on priority, with higher priority models tried first.

2. **CapabilityBasedStrategy**: Filters models based on required capabilities and selects the best match.

3. **RoundRobinStrategy**: Cycles through available models in sequence, providing load balancing and redundancy.

### Running the Example

To run the example:

```bash
# Make sure you have the required API keys in your .env file
# OPENAI_API_KEY=your_openai_key
# ANTHROPIC_API_KEY=your_anthropic_key
# GROQ_API_KEY=your_groq_key

python -m mihrabai.examples.advanced.model_fallback_example
```

The example will:
- Initialize agents with dynamic model selection
- Demonstrate basic queries with model information
- Show model switching capabilities
- Perform handoffs between agents with different models
- Simulate model failures and demonstrate recovery
- Select models based on specific capabilities
- Demonstrate round-robin model selection for load balancing

### Requirements

- At least one API key for OpenAI, Anthropic, or Groq
- Python 3.8+
- All dependencies from the Mihrab AI Agent framework

### Example Output

The example will output information about:
- Which models were selected for each agent
- Successful model switches
- Handoffs between agents
- Recovery from simulated failures
- Model selection based on capabilities
- Round-robin model selection in action

## Other Advanced Examples

More advanced examples will be added to this directory to demonstrate other sophisticated features of the Mihrab AI Agent framework. 