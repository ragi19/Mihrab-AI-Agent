# Comprehensive Multi-Agent Tracing Framework

## 1. Introduction to Distributed Tracing

Distributed tracing is a critical observability technique for understanding the flow of operations in complex multi-agent systems. Unlike traditional logging, tracing provides a hierarchical view of interactions between components, allowing developers to visualize the entire execution path across multiple agents, services, and functions.

## 2. Core Concepts

### 2.1 Traces

A trace represents a complete workflow or transaction in a multi-agent system:
- **Trace ID**: A unique identifier that connects all spans belonging to the same operation
- **Name**: Descriptive identifier for the workflow being traced
- **Group ID**: Optional identifier for grouping related traces (e.g., by session or user)
- **Metadata**: Additional contextual information about the trace

### 2.2 Spans

A span represents a single unit of work within a trace:
- **Span ID**: Unique identifier for the span
- **Parent ID**: Reference to the parent span (creating a hierarchical structure)
- **Trace ID**: The trace this span belongs to
- **Start/End Times**: When the operation began and completed
- **Error Information**: Details about failures if they occur
- **Span Data**: Type-specific information about the operation

### 2.3 Span Data Types

Different types of spans carry specialized data relevant to their function:
- **Agent Spans**: Track agent activities with tools and capabilities
- **Function Spans**: Track function calls with inputs/outputs
- **Generation Spans**: Track AI model generations with prompts/responses
- **Response Spans**: Track structured responses
- **Handoff Spans**: Track transfers between agents
- **Custom Spans**: Track arbitrary user-defined operations
- **Guardrail Spans**: Track safety guardrail evaluations

## 3. Tracing Architecture

### 3.1 Tracing Provider

The tracing provider serves as the central component that:
- Creates and manages traces and spans
- Stores current trace/span context
- Maintains and coordinates trace processors
- Provides configuration options (enable/disable tracing)

### 3.2 Processor Interface

The processor interface defines the contract for trace handling:

```python
class TracingProcessor(abc.ABC):
    @abc.abstractmethod
    def on_trace_start(self, trace: "Trace") -> None:
        pass
        
    @abc.abstractmethod
    def on_trace_end(self, trace: "Trace") -> None:
        pass
        
    @abc.abstractmethod
    def on_span_start(self, span: "Span[Any]") -> None:
        pass
        
    @abc.abstractmethod
    def on_span_end(self, span: "Span[Any]") -> None:
        pass
        
    @abc.abstractmethod
    def shutdown(self) -> None:
        pass
        
    @abc.abstractmethod
    def force_flush(self) -> None:
        pass
```

### 3.3 Exporters

Exporters handle the final destination for trace data:

```python
class TracingExporter(abc.ABC):
    @abc.abstractmethod
    def export(self, items: list["Trace | Span[Any]"]) -> None:
        pass
```

Common implementation patterns include:
- Console exporters for local debugging
- Backend exporters for sending data to observability platforms
- File exporters for offline analysis

## 4. Implementation Patterns

### 4.1 Batch Processing

A robust tracing implementation should batch operations to minimize performance impact:

```python
class BatchTraceProcessor(TracingProcessor):
    def __init__(
        self,
        exporter: TracingExporter,
        max_queue_size: int = 8192,
        max_batch_size: int = 128,
        schedule_delay: float = 5.0,
        export_trigger_ratio: float = 0.7,
    ):
        # Initialize queue and worker thread
        
    def on_span_end(self, span: Span[Any]) -> None:
        try:
            self._queue.put_nowait(span)
        except queue.Full:
            logger.warning("Queue is full, dropping span.")
```

Key considerations:
- Thread-safe queuing mechanisms
- Background processing threads
- Configurable batch sizes and timing
- Resource management (memory constraints)

### 4.2 Error Handling and Resilience

Robust tracing systems must handle failures gracefully:
- Exponential backoff for retries
- Circuit breaking for persistent failures
- Fallback mechanisms when tracing is unavailable
- Proper resource cleanup (connection pooling)

### 4.3 Context Propagation

Maintaining trace context across asynchronous boundaries:
- Using context variables for thread-local storage
- Explicitly passing context in function calls
- Automatic context restoration in async scenarios

## 5. Practical Implementation

### 5.1 Setting Up a Tracing System

```python
# Initialize exporters and processors
exporter = ConsoleSpanExporter()  # or any other exporter
processor = BatchTraceProcessor(exporter)

# Register with the global trace provider
trace_provider = TraceProvider()
trace_provider.register_processor(processor)

# Create traces and spans
with trace_provider.create_trace("workflow_name") as trace:
    # Perform operations
    with trace_provider.create_span(span_data=FunctionSpanData(...)) as span:
        # Execute and trace function call
        result = my_function()
```

### 5.2 Custom Trace Processors

Creating custom processors allows for specialized handling:

```python
class MetricsTraceProcessor(TracingProcessor):
    def __init__(self):
        self.spans_processed = 0
        self.traces_processed = 0
        
    def on_span_end(self, span: Span[Any]) -> None:
        self.spans_processed += 1
        # Extract metrics like duration
        if span.ended_at and span.started_at:
            # Calculate and store metrics
            pass
```

### 5.3 Integrating with Observability Systems

Connecting with existing platforms through custom exporters:

```python
class PrometheusExporter(TracingExporter):
    def export(self, items: list[Trace | Span[Any]]) -> None:
        for item in items:
            if isinstance(item, Span):
                # Update Prometheus metrics based on span data
                update_counters_and_histograms(item)
```

## 6. Advanced Features

### 6.1 Sampling Strategies

To control the volume of trace data:
- Head-based sampling (random selection upfront)
- Tail-based sampling (based on outcomes)
- Priority-based sampling (based on importance)

### 6.2 Trace Correlation

Connecting traces across systems:
- Distributed trace context propagation
- Cross-service trace ID propagation
- Trace linking and hierarchies

### 6.3 Visualization and Analysis

Analyzing trace data effectively:
- Gantt chart-style timeline views
- Bottleneck identification
- Error rate analysis
- Resource consumption metrics

## 7. Best Practices

1. **Selective Tracing**: Trace important operations but avoid overwhelming the system
2. **Contextual Enrichment**: Add relevant metadata to traces for filtering and analysis
3. **Performance Considerations**: Minimize the overhead of tracing on core operations
4. **Security and Privacy**: Avoid including sensitive data in traces
5. **Standardization**: Use consistent naming and structure across your system

## 8. Conclusion

A well-designed tracing framework provides invaluable insights into multi-agent system behavior. By implementing the components described in this document, developers can gain visibility into complex workflows, diagnose issues more effectively, and optimize system performance based on empirical data rather than guesswork.

The modular architecture presented here—with clear separation between trace creation, processing, and exporting—allows for flexibility to adapt to different requirements while maintaining a consistent approach to observability across your entire multi-agent ecosystem.

# Logging and Profiling Guide

## Overview

The LLM Agents framework includes comprehensive logging and profiling capabilities to help you:
- Debug agent behavior
- Monitor performance
- Track token usage
- Analyze conversation flows
- Identify and troubleshoot errors

## Logging Configuration

### Basic Setup

Configure logging through the config system:

```python
from mihrabai import config

config.set_logging_config({
    "level": "DEBUG",  # DEBUG, INFO, WARNING, ERROR, or CRITICAL
    "file": "agent.log",  # Optional, logs to console if not specified
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
})
```

### Using the Config Generator

Use the command-line tool for quick setup:

```bash
python -m mihrabai.scripts.generate_config \
    --log-level DEBUG \
    --log-file agent.log \
    --log-format "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Log Structure

The framework uses a hierarchical logger structure:

- `agent.*` - Agent-level logs
  - Message processing
  - Conversation history management
  - Token usage
- `models.*` - Model-specific logs
  - API interactions
  - Response generation
  - Parameter updates
- `tools.*` - Tool execution logs
  - Parameter validation
  - Execution results
  - Error handling
- `runtime.*` - Runtime environment logs
  - Context management
  - Error recovery
  - Performance metrics

## Profiling Tools

### Agent Profiler

The `AgentProfiler` class provides detailed performance metrics:

```python
from mihrabai.utils.dev_tools import AgentProfiler

profiler = AgentProfiler()
metrics = await profiler.profile_agent(agent, messages)
```

Available metrics:
- `total_requests` - Number of processed messages
- `total_tokens` - Total tokens used (if supported by model)
- `total_time` - Total processing time
- `avg_request_time` - Average time per request
- `min_request_time` - Fastest request time
- `max_request_time` - Slowest request time
- `avg_tokens` - Average tokens per request
- `min_tokens` - Minimum tokens in a request
- `max_tokens` - Maximum tokens in a request

### Retry Handling

Use the `@async_retry` decorator for automatic retry logic:

```python
from mihrabai.utils.dev_tools import async_retry

@async_retry(max_retries=3, retry_delay=1.0)
async def my_function():
    # Function implementation
    pass
```

## Log Analysis

### Common Log Patterns

1. Agent Initialization:
```
INFO - agent.ChatAgent - Initialized agent with model: gpt-3.5-turbo
INFO - agent.ChatAgent - Using max_history_tokens: 4000
```

2. Message Processing:
```
DEBUG - agent.ChatAgent - Processing message: <Message role=user content="Hello">
DEBUG - agent.ChatAgent - Generating model response
INFO - agent.ChatAgent - Response generated in 0.8s
```

3. Token Usage:
```
DEBUG - agent.ChatAgent - Message tokens: 10
DEBUG - agent.ChatAgent - Response tokens: 25
DEBUG - agent.ChatAgent - Total conversation tokens: 35
```

4. History Management:
```
DEBUG - agent.ChatAgent - Current history tokens: 3500
INFO - agent.ChatAgent - Truncated history from 15 to 10 messages
```

### Error Patterns

1. API Errors:
```
WARNING - models.OpenAIProvider - API request failed, retrying in 1s
ERROR - models.OpenAIProvider - Max retries exceeded: Rate limit reached
```

2. Tool Errors:
```
WARNING - tools.WebTool - Invalid URL format, retrying with normalized URL
ERROR - tools.WebTool - Request failed: Connection timeout
```

## Best Practices

1. **Log Levels**
   - Use DEBUG for detailed troubleshooting
   - Use INFO for general operation tracking
   - Use WARNING for recoverable issues
   - Use ERROR for critical failures

2. **File Management**
   - Rotate log files regularly
   - Monitor log file size
   - Archive old logs

3. **Performance Monitoring**
   - Profile regularly in production
   - Track token usage trends
   - Monitor response times
   - Set up alerts for anomalies

4. **Debug Mode**
   ```python
   # Enable detailed debug logging temporarily
   config.set_logging_config({"level": "DEBUG"})
   
   # Run your code
   
   # Reset to normal logging
   config.set_logging_config({"level": "INFO"})
   ```

## Common Issues

1. **High Token Usage**
   - Check conversation history limits
   - Monitor system message size
   - Review tool output verbosity

2. **Slow Response Times**
   - Check API rate limits
   - Review retry configurations
   - Monitor tool execution times

3. **Memory Issues**
   - Implement log rotation
   - Clear conversation history regularly
   - Monitor file handle usage