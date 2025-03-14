# LLM Agents Framework - Model Implementation Guide

## Provider Architecture

The framework implements a modular provider architecture that supports multiple LLM providers while maintaining consistent interfaces and behavior.

### Core Interfaces

#### BaseModel

The foundation of all model implementations:

```python
class BaseModel(ABC):
    def __init__(self, model_name: str, trace_provider: Optional[TraceProvider] = None):
        self.model_name = model_name
        self.trace_provider = trace_provider
        self.parameters: ModelParameters = {}
    
    @abstractmethod
    async def generate_response(self, messages: List[Message]) -> Message:
        """Generate a response for the given messages"""
        pass
    
    async def _wrapped_generate(self, messages: List[Message]) -> Message:
        """Wrap generation with tracing and error handling"""
        async with self._trace_span("model_generation") as span:
            try:
                return await self._generate(messages)
            except Exception as e:
                span.set_error(e)
                raise
```

### Provider-Specific Implementations

#### 1. OpenAI Provider

```python
class OpenAIModel(BaseModel):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    async def _generate(self, messages: List[Message]) -> Message:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[msg.to_dict() for msg in messages],
            **self.parameters
        )
        return Message.from_openai_response(response)
```

#### 2. Anthropic Provider

```python
class AnthropicModel(BaseModel):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def _generate(self, messages: List[Message]) -> Message:
        prepared = self._prepare_messages(messages)
        response = await self.client.messages.create(
            model=self.model_name,
            messages=prepared,
            **self.parameters
        )
        return Message.from_anthropic_response(response)
```

#### 3. Groq Provider

```python
class GroqModel(BaseModel):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = groq.AsyncGroq(api_key=api_key)
    
    async def _generate(self, messages: List[Message]) -> Message:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[msg.to_dict() for msg in messages],
            **self.parameters
        )
        return Message.from_groq_response(response)
```

### Provider Configuration

Each provider supports flexible configuration through the config system:

```python
from mihrabai import config

# OpenAI configuration
config.set_provider_config("openai", {
    "api_key": "sk-...",
    "default_model": "gpt-4",
    "default_parameters": {
        "temperature": 0.7,
        "max_tokens": 2000
    }
})

# Anthropic configuration
config.set_provider_config("anthropic", {
    "api_key": "sk-ant-...",
    "default_model": "claude-3-opus-20240229",
    "default_parameters": {
        "temperature": 0.5,
        "max_tokens": 4000
    }
})

# Groq configuration
config.set_provider_config("groq", {
    "api_key": "gsk-...",
    "default_model": "llama2-70b-4096",
    "default_parameters": {
        "temperature": 0.8,
        "max_tokens": 4096
    }
})
```

## Model Features

### 1. Token Management

Each model implementation includes token counting and management:

```python
class BaseModel(ABC):
    async def count_tokens(self, text: str) -> int:
        """Estimate token count for the text"""
        pass
    
    async def get_prompt_tokens(self, messages: List[Message]) -> int:
        """Get total tokens in the prompt"""
        pass
    
    async def get_completion_tokens(self, response: Message) -> int:
        """Get tokens in the completion"""
        pass
```

### 2. Parameter Handling

Models support dynamic parameter updates:

```python
model.update_parameters({
    "temperature": 0.8,
    "top_p": 0.95,
    "presence_penalty": 0.2,
    "frequency_penalty": 0.2
})
```

### 3. Response Streaming

Support for streaming responses:

```python
async def stream_response(self, messages: List[Message]) -> AsyncGenerator[str, None]:
    async with self._trace_span("stream_generation"):
        async for chunk in self._stream_generate(messages):
            yield chunk.content
```

## Advanced Features

### 1. Multi-Provider Agent

Coordinate multiple providers in a single agent:

```python
class MultiProviderAgent(Agent):
    def __init__(self, providers: Dict[str, Agent]):
        self.providers = providers
        super().__init__(model=next(iter(providers.values())).model)
    
    async def process_message(self, message: Message) -> Message:
        responses = await gather_with_concurrency(
            limit=3,
            tasks=[agent.process_message(message) 
                   for agent in self.providers.values()]
        )
        return self._combine_responses(responses)
```

### 2. Provider Switching

Dynamic provider switching during execution:

```python
class ProviderSwitchingAgent(Agent):
    def __init__(self, providers: Dict[str, Agent], default_provider: str):
        self.providers = providers
        self.current_provider = default_provider
        super().__init__(model=providers[default_provider].model)
    
    def switch_provider(self, provider_name: str) -> None:
        if provider_name in self.providers:
            self.current_provider = provider_name
            self.model = self.providers[provider_name].model
```

### 3. Custom Provider Template

Template for implementing new providers:

```python
class CustomProvider:
    def __init__(self, **kwargs):
        self.config = kwargs
        self.client = initialize_client(kwargs)
    
    async def create_model(self, model_name: str,
                          parameters: Optional[ModelParameters] = None) -> BaseModel:
        return CustomModel(
            model_name=model_name,
            client=self.client,
            parameters=parameters
        )

class CustomModel(BaseModel):
    async def generate_response(self, messages: List[Message]) -> Message:
        response = await self.client.generate(
            messages=[msg.to_dict() for msg in messages],
            **self.parameters
        )
        return Message(
            role=MessageRole.ASSISTANT,
            content=response
        )
```

## Model Selection Guide

### Use Case Considerations

1. **OpenAI Models**
   - GPT-4: Complex reasoning, coding, analysis
   - GPT-3.5-Turbo: General-purpose, cost-effective
   - Use when: High accuracy needed, cost not primary concern

2. **Anthropic Models**
   - Claude-3: Long-form content, analysis, coding
   - Use when: Need large context windows, strong safety features

3. **Groq Models**
   - LLaMA-2: Open-source, customizable
   - Use when: High performance needed, want local deployment option

### Selection Criteria

Consider these factors when choosing a model:

1. **Performance Requirements**
   - Response latency
   - Token throughput
   - Concurrent request handling

2. **Cost Considerations**
   - Per-token pricing
   - Monthly volume
   - Budget constraints

3. **Feature Requirements**
   - Context window size
   - Special capabilities
   - Fine-tuning support

4. **Integration Needs**
   - API compatibility
   - Authentication methods
   - Rate limiting

## Best Practices

### 1. Error Handling

Implement robust error handling:

```python
async def _generate(self, messages: List[Message]) -> Message:
    try:
        response = await self._make_api_call(messages)
        return self._process_response(response)
    except APIError as e:
        if e.is_rate_limit():
            await self._handle_rate_limit(e)
        elif e.is_token_limit():
            await self._handle_token_limit(e)
        raise
```

### 2. Performance Optimization

Optimize model usage:

```python
class OptimizedModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._response_cache = LRUCache(maxsize=1000)
    
    async def generate_response(self, messages: List[Message]) -> Message:
        cache_key = self._get_cache_key(messages)
        if cache_key in self._response_cache:
            return self._response_cache[cache_key]
        
        response = await super().generate_response(messages)
        self._response_cache[cache_key] = response
        return response
```

### 3. Token Management

Implement efficient token management:

```python
class TokenManager:
    def __init__(self, model: BaseModel, max_tokens: int):
        self.model = model
        self.max_tokens = max_tokens
        self.used_tokens = 0
    
    async def can_process(self, messages: List[Message]) -> bool:
        estimated_tokens = await self.model.count_tokens(
            "\n".join(msg.content for msg in messages)
        )
        return (self.used_tokens + estimated_tokens) <= self.max_tokens
    
    async def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.used_tokens += prompt_tokens + completion_tokens
```

## Monitoring and Maintenance

### 1. Provider Health Checks

```python
class ProviderHealth:
    async def check_provider(self, provider: str) -> bool:
        model = await self._get_test_model(provider)
        try:
            response = await model.generate_response([
                Message(role=MessageRole.USER, content="test")
            ])
            return response is not None
        except Exception:
            return False
```

### 2. Performance Monitoring

```python
class ModelMetrics:
    def __init__(self):
        self.latencies = defaultdict(list)
        self.token_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
    
    async def record_generation(self, 
                              provider: str,
                              latency: float,
                              tokens: int,
                              success: bool) -> None:
        self.latencies[provider].append(latency)
        self.token_counts[provider] += tokens
        if not success:
            self.error_counts[provider] += 1
```

### 3. Cost Tracking

```python
class CostTracker:
    def __init__(self):
        self.costs = defaultdict(float)
        self.token_prices = {
            "gpt-4": 0.03,
            "claude-3": 0.015,
            "llama2-70b": 0.01
        }
    
    def record_usage(self, 
                    model: str,
                    prompt_tokens: int,
                    completion_tokens: int) -> None:
        price = self.token_prices.get(model, 0.0)
        cost = (prompt_tokens + completion_tokens) * price / 1000
        self.costs[model] += cost
```
