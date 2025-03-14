# Tracing and Monitoring Documentation

## Overview

The Agents framework provides comprehensive tracing and monitoring capabilities to help you understand and debug your multi-agent systems. This documentation covers the available tracing features, monitoring tools, and best practices for observability.

## Tracing Capabilities

### Basic Tracing

```python
from agents.tracing import Tracer, TraceEvent

class AgentTracer(Tracer):
    async def record_event(self, event: TraceEvent) -> None:
        # Record basic event information
        event_data = {
            "timestamp": event.timestamp,
            "agent_id": event.agent_id,
            "event_type": event.type,
            "metadata": event.metadata
        }
        await self.store_event(event_data)
```

### Conversation Tracing

Track message flow between agents:

```python
class ConversationTracer(Tracer):
    def __init__(self):
        self.conversations = {}
    
    async def trace_message(
        self,
        from_agent: str,
        to_agent: str,
        message: str,
        metadata: Dict[str, Any]
    ) -> None:
        conversation_id = metadata.get("conversation_id")
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].append({
            "timestamp": datetime.now(),
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message": message,
            "metadata": metadata
        })
```

### Performance Tracing

Monitor agent performance metrics:

```python
class PerformanceTracer(Tracer):
    def __init__(self):
        self.response_times = defaultdict(list)
        self.token_usage = defaultdict(int)
        self.error_counts = defaultdict(int)
    
    async def record_performance(
        self,
        agent_id: str,
        response_time: float,
        tokens_used: int,
        success: bool
    ) -> None:
        self.response_times[agent_id].append(response_time)
        self.token_usage[agent_id] += tokens_used
        if not success:
            self.error_counts[agent_id] += 1
    
    def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        times = self.response_times[agent_id]
        return {
            "avg_response_time": sum(times) / len(times),
            "total_tokens": self.token_usage[agent_id],
            "error_rate": self.error_counts[agent_id] / len(times)
        }
```

## Monitoring Tools

### Real-time Monitoring

Monitor agent activity in real-time:

```python
class RealTimeMonitor:
    def __init__(self):
        self.active_agents = set()
        self.message_queue = asyncio.Queue()
    
    async def monitor_agent(self, agent_id: str) -> None:
        self.active_agents.add(agent_id)
        try:
            while True:
                event = await self.message_queue.get()
                if event["agent_id"] == agent_id:
                    await self.process_event(event)
        finally:
            self.active_agents.remove(agent_id)
    
    async def process_event(self, event: Dict[str, Any]) -> None:
        # Process and display real-time events
        print(f"Agent {event['agent_id']}: {event['type']} - {event['data']}")
```

### Health Checks

Monitor agent health and status:

```python
class HealthMonitor:
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.agent_status = {}
    
    async def check_agent_health(self, agent: Agent) -> bool:
        try:
            # Perform basic health check
            response = await agent.ping()
            self.agent_status[agent.id] = {
                "status": "healthy",
                "last_check": datetime.now(),
                "response_time": response.duration
            }
            return True
        except Exception as e:
            self.agent_status[agent.id] = {
                "status": "unhealthy",
                "last_check": datetime.now(),
                "error": str(e)
            }
            return False
    
    async def monitor_health(self, agents: List[Agent]) -> None:
        while True:
            for agent in agents:
                await self.check_agent_health(agent)
            await asyncio.sleep(self.check_interval)
```

### Resource Monitoring

Track resource usage:

```python
class ResourceMonitor:
    def __init__(self):
        self.resource_usage = defaultdict(dict)
    
    async def track_resources(self, agent_id: str) -> None:
        while True:
            usage = await self.get_resource_usage(agent_id)
            self.resource_usage[agent_id] = {
                "timestamp": datetime.now(),
                "cpu_percent": usage.cpu_percent,
                "memory_mb": usage.memory_mb,
                "active_threads": usage.active_threads
            }
            await asyncio.sleep(5)
    
    def get_agent_resources(self, agent_id: str) -> Dict[str, Any]:
        return self.resource_usage.get(agent_id, {})
```

## Observability Features

### Logging Integration

Enhanced logging capabilities:

```python
class AgentLogger:
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger("agent_logger")
        self.logger.setLevel(log_level)
        
        # Add handlers
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler("agent_logs.txt")
        
        # Add formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def log_agent_event(
        self,
        agent_id: str,
        event_type: str,
        message: str,
        level: str = "INFO"
    ) -> None:
        log_method = getattr(self.logger, level.lower())
        log_method(f"Agent {agent_id} - {event_type}: {message}")
```

### Metrics Collection

Collect and aggregate metrics:

```python
class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    async def collect_metrics(
        self,
        agent_id: str,
        metric_name: str,
        value: float
    ) -> None:
        self.metrics[f"{agent_id}_{metric_name}"].append({
            "timestamp": datetime.now(),
            "value": value
        })
    
    def get_metric_statistics(
        self,
        agent_id: str,
        metric_name: str
    ) -> Dict[str, float]:
        values = [
            m["value"]
            for m in self.metrics[f"{agent_id}_{metric_name}"]
        ]
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "count": len(values)
        }
```

### Distributed Tracing

Track requests across multiple agents:

```python
class DistributedTracer:
    def __init__(self):
        self.traces = {}
    
    async def start_trace(
        self,
        trace_id: str,
        parent_id: Optional[str] = None
    ) -> str:
        span_id = str(uuid.uuid4())
        self.traces[trace_id] = {
            "spans": [{
                "span_id": span_id,
                "parent_id": parent_id,
                "start_time": datetime.now(),
                "events": []
            }]
        }
        return span_id
    
    async def end_trace(
        self,
        trace_id: str,
        span_id: str
    ) -> None:
        for span in self.traces[trace_id]["spans"]:
            if span["span_id"] == span_id:
                span["end_time"] = datetime.now()
                break
    
    async def add_event(
        self,
        trace_id: str,
        span_id: str,
        event: Dict[str, Any]
    ) -> None:
        for span in self.traces[trace_id]["spans"]:
            if span["span_id"] == span_id:
                span["events"].append({
                    "timestamp": datetime.now(),
                    **event
                })
                break
```

## Best Practices

### 1. Structured Logging

Use consistent log formats:

```python
class StructuredLogger:
    def log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> None:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data,
            "context": context
        }
        logging.info(json.dumps(log_entry))
```

### 2. Error Tracking

Track and categorize errors:

```python
class ErrorTracker:
    def __init__(self):
        self.errors = defaultdict(list)
    
    def track_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> None:
        error_type = type(error).__name__
        self.errors[error_type].append({
            "timestamp": datetime.now(),
            "message": str(error),
            "context": context,
            "stack_trace": traceback.format_exc()
        })
    
    def get_error_summary(self) -> Dict[str, int]:
        return {
            error_type: len(errors)
            for error_type, errors in self.errors.items()
        }
```

### 3. Performance Monitoring

Monitor system performance:

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    async def monitor_performance(self) -> None:
        while True:
            metrics = {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
            
            for metric, value in metrics.items():
                self.metrics[metric].append({
                    "timestamp": datetime.now(),
                    "value": value
                })
            
            await asyncio.sleep(60)
    
    def get_performance_report(self) -> Dict[str, Any]:
        return {
            metric: {
                "current": values[-1]["value"],
                "avg": sum(v["value"] for v in values) / len(values)
            }
            for metric, values in self.metrics.items()
        }
```

### 4. Alerting

Set up alerts for critical events:

```python
class AlertManager:
    def __init__(self, alert_thresholds: Dict[str, float]):
        self.thresholds = alert_thresholds
        self.alerts = []
    
    async def check_metrics(
        self,
        metrics: Dict[str, float]
    ) -> None:
        for metric, value in metrics.items():
            if metric in self.thresholds:
                threshold = self.thresholds[metric]
                if value > threshold:
                    await self.trigger_alert(metric, value, threshold)
    
    async def trigger_alert(
        self,
        metric: str,
        value: float,
        threshold: float
    ) -> None:
        alert = {
            "timestamp": datetime.now(),
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "message": f"{metric} exceeded threshold: {value} > {threshold}"
        }
        self.alerts.append(alert)
        await self.notify_alert(alert)
    
    async def notify_alert(self, alert: Dict[str, Any]) -> None:
        # Implement notification logic (email, Slack, etc.)
        pass
```

These tracing and monitoring capabilities provide comprehensive observability for your multi-agent systems, enabling effective debugging, performance optimization, and system maintenance.

# LLM Agents Framework - Tracing System Documentation

## Overview

The LLM Agents Framework includes a sophisticated distributed tracing system designed specifically for multi-agent interactions. This system provides deep visibility into agent operations, model interactions, and system performance.

## Core Concepts

### Traces

A trace represents a complete workflow or transaction in the multi-agent system:

- **Trace ID**: Unique identifier connecting all related spans
- **Name**: Descriptive workflow identifier
- **Group ID**: Optional identifier for grouping related traces
- **Metadata**: Contextual information about the trace operation

### Spans

Spans represent individual units of work within a trace:

- **Span ID**: Unique identifier for the operation
- **Parent ID**: Reference to parent span (creating hierarchy)
- **Start/End Times**: Operation timing information
- **Status**: Success/failure and error details
- **Attributes**: Operation-specific metadata

### Span Types

Different span types carry specialized data:

| Type | Purpose | Key Data |
|------|---------|----------|
| Agent | Track agent activities | Name, tools, capabilities |
| Function | Track tool execution | Inputs, outputs, duration |
| Generation | Track model generations | Prompts, responses, tokens |
| Response | Track structured responses | Response IDs, content |
| Handoff | Track agent transitions | Source, destination, context |
| Guardrail | Track safety checks | Rules, triggers, actions |

## Implementation

### Basic Usage

```python
from mihrabai.utils.tracing import TraceProvider

# Initialize tracing
trace_provider = TraceProvider()

# Create a trace
async with trace_provider.trace("conversation") as trace:
    # Add trace-level metadata
    trace.set_metadata({
        "user_id": "123",
        "session_id": "abc",
        "conversation_id": "xyz"
    })
    
    # Create spans for operations
    async with trace.span("model_generation") as span:
        span.set_data({
            "model": "gpt-4",
            "tokens": 150,
            "latency_ms": 450
        })
        response = await model.generate()
```

### Advanced Features

#### 1. Sampling Strategies

Configure trace sampling for high-volume systems:

```python
class CustomSampler(TraceSampler):
    def should_sample(self, trace_id: str, metadata: Dict[str, Any]) -> bool:
        # Sample based on criteria
        if metadata.get("priority") == "high":
            return True
        return random.random() < 0.1  # 10% sampling
```

#### 2. Span Processors

Create custom span processors for specialized handling:

```python
class MetricsProcessor(SpanProcessor):
    async def on_span_end(self, span: Span) -> None:
        if span.type == "model_generation":
            await self.update_metrics(
                model=span.data["model"],
                tokens=span.data["tokens"],
                latency=span.data["latency_ms"]
            )
```

#### 3. Exporters

Implement custom exporters for your observability stack:

```python
class PrometheusExporter(TraceExporter):
    async def export_spans(self, spans: List[Span]) -> None:
        for span in spans:
            if span.type == "model_generation":
                GENERATION_LATENCY.observe(span.duration)
                TOKEN_COUNT.inc(span.data["tokens"])
```

## Performance Considerations

### 1. Batching

The tracing system uses efficient batching:

- In-memory span queue
- Configurable batch size and flush interval
- Automatic background flushing
- Manual flush capability for critical traces

### 2. Memory Management

Memory-efficient span storage:

- Circular buffer for span storage
- Automatic span eviction when buffer is full
- Configurable maximum spans per trace
- Efficient attribute storage

### 3. Concurrency

Thread-safe operations:

- Lock-free span creation
- Atomic span updates
- Concurrent span processing
- Async export operations

## Best Practices

### 1. Span Creation

- Create spans for significant operations
- Use meaningful span names
- Include relevant attributes
- Set appropriate span types
- Maintain proper span hierarchy

```python
async with trace.span("agent_operation") as parent:
    parent.set_type("agent")
    parent.set_data({"agent_id": "agent1"})
    
    async with parent.child("tool_execution") as child:
        child.set_type("function")
        child.set_data({"tool": "calculator"})
```

### 2. Error Handling

- Always set error status on failure
- Include error details in span data
- Maintain trace continuity during errors
- Use appropriate error types

```python
try:
    result = await risky_operation()
except Exception as e:
    span.set_status(SpanStatus.ERROR)
    span.set_data({
        "error": str(e),
        "error_type": type(e).__name__
    })
```

### 3. Context Propagation

- Properly maintain trace context
- Pass context in async operations
- Handle context in callbacks
- Restore context when needed

## Advanced Usage

### 1. Custom Span Types

```python
class SecuritySpan(Span):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = "security"
    
    def add_validation(self, rule: str, result: bool):
        self.append_data("validations", {
            "rule": rule,
            "result": result,
            "timestamp": time.time()
        })
```

### 2. Trace Analytics

```python
class TraceAnalytics:
    def analyze_trace(self, trace: Trace) -> Dict[str, Any]:
        return {
            "total_duration": trace.duration,
            "span_count": len(trace.spans),
            "error_count": sum(1 for s in trace.spans if s.status.is_error),
            "token_usage": sum(s.data.get("tokens", 0) 
                             for s in trace.spans 
                             if s.type == "model_generation")
        }
```

### 3. Trace Visualization

```python
class TraceVisualizer:
    def generate_timeline(self, trace: Trace) -> str:
        timeline = Timeline()
        for span in trace.spans:
            timeline.add_event(
                start=span.start_time,
                end=span.end_time,
                name=span.name,
                category=span.type
            )
        return timeline.render()
```

## Monitoring

### 1. Key Metrics

- Trace duration distribution
- Span count per trace
- Error rate by span type
- Token usage by model
- Latency by operation

### 2. Alerting

- Error rate thresholds
- Latency thresholds
- Token usage limits
- Trace volume anomalies

### 3. Dashboards

- Real-time trace viewer
- Performance metrics
- Error tracking
- Resource usage
- Cost analysis