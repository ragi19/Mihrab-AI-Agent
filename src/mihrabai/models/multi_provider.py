"""
Multi-provider agent implementation for running multiple LLM providers in parallel
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from llm_agents.core.agent import Agent
from llm_agents.core.message import Message, MessageRole
from llm_agents.utils.async_utils import gather_with_concurrency
from llm_agents.utils.logging import get_logger

from ..core.types import ModelResponse
from .base import BaseModel, ModelCapability, ModelError
from .factory import ModelFactory
from .provider_stats import ProviderStats

logger = get_logger("models.multi_provider")


class OptimizationStrategy(str, Enum):
    """Strategy for selecting providers"""

    PERFORMANCE = "performance"  # Optimize for performance (latency)
    COST = "cost"  # Optimize for cost
    RELIABILITY = "reliability"  # Optimize for reliability (success rate)
    ROUND_ROBIN = "round_robin"  # Round-robin between providers
    RANDOM = "random"  # Random selection


@dataclass
class ProviderMetrics:
    """Metrics for a provider"""

    requests: int = 0
    successes: int = 0
    failures: int = 0
    tokens: int = 0
    cost: float = 0.0
    latency: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.requests == 0:
            return 1.0  # Default to perfect if no data
        return self.successes / self.requests

    @property
    def average_latency(self) -> float:
        """Calculate average latency"""
        if not self.latency:
            return 0.0
        return sum(self.latency) / len(self.latency)

    @property
    def average_cost(self) -> float:
        """Calculate average cost per request"""
        if self.requests == 0:
            return 0.0
        return self.cost / self.requests

    @property
    def average_tokens(self) -> float:
        """Calculate average tokens per request"""
        if self.requests == 0:
            return 0.0
        return self.tokens / self.requests

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "requests": self.requests,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": self.success_rate,
            "average_latency": self.average_latency,
            "average_cost": self.average_cost,
            "average_tokens": self.average_tokens,
            "total_tokens": self.tokens,
            "total_cost": self.cost,
        }


class MultiProviderModel(BaseModel):
    """Model that can use multiple providers with failover"""

    def __init__(
        self,
        models: List[BaseModel],
        selection_strategy: OptimizationStrategy = OptimizationStrategy.PERFORMANCE,
        failover_retries: int = 2,
    ):
        """Initialize the multi-provider model

        Args:
            models: List of provider models to use
            selection_strategy: Strategy for selecting providers
            failover_retries: Number of failover attempts before giving up
        """
        if not models:
            raise ValueError("At least one model must be provided")

        self.models = models
        self._primary_model = models[0]
        self._current_model = self._primary_model

        # Convert string strategy to enum
        if isinstance(selection_strategy, str):
            try:
                self.strategy = OptimizationStrategy(selection_strategy)
            except ValueError:
                logger.warning(
                    f"Invalid strategy '{selection_strategy}', using performance"
                )
                self.strategy = OptimizationStrategy.PERFORMANCE
        else:
            self.strategy = selection_strategy

        self.failover_retries = failover_retries
        self.stats = ProviderStats()
        self._current_index = 0

        # Initialize with primary model's ID
        super().__init__(model_id=self._primary_model.model_id)

    @property
    def capabilities(self) -> Set[str]:
        """Get the capabilities of this model

        Returns the intersection of capabilities from all models
        """
        if not self.models:
            return set()

        # Start with capabilities of the first model
        common_capabilities = set(self.models[0].capabilities)

        # Intersect with capabilities of other models
        for model in self.models[1:]:
            common_capabilities &= set(model.capabilities)

        return common_capabilities

    @property
    def current_provider(self) -> str:
        """Get current provider name"""
        return self._current_model.__class__.__name__

    @property
    def model_name(self) -> str:
        """Get the model name"""
        return self._current_model.model_id

    async def generate(self, messages: List[Message], **kwargs) -> Message:
        """Generate a response using the selected provider"""
        attempts = 0
        errors = []

        while attempts <= self.failover_retries:
            self._current_model = self._select_model()
            provider = self.current_provider

            try:
                self.stats.start_request(provider)
                response = await self._current_model.generate(messages, **kwargs)

                # Record success metrics
                self.stats.record_success(provider)

                return response

            except Exception as e:
                # Record failure
                self.stats.record_failure(provider)

                # Log error and try next provider
                logger.warning(f"Provider {provider} failed: {str(e)}")
                errors.append((provider, str(e)))

                attempts += 1

                # If we've tried all providers, raise the last error
                if attempts > self.failover_retries or len(self.models) <= 1:
                    raise ModelError(
                        f"All providers failed. Last error: {errors[-1][1]}"
                    )

        # This should never happen due to the check above
        raise ModelError("Unexpected error in provider selection")

    async def generate_stream(
        self, messages: List[Message], **kwargs
    ) -> AsyncIterator[Message]:
        """Stream a response from the selected provider

        Args:
            messages: The conversation history
            **kwargs: Additional parameters to pass to the model

        Yields:
            Message chunks from the model

        Raises:
            ModelError: If all providers fail
        """
        attempts = 0
        errors = []

        while attempts <= self.failover_retries:
            self._current_model = self._select_model()
            provider = self.current_provider

            try:
                self.stats.start_request(provider)

                # Stream response from the current model
                chunk_count = 0
                async for chunk in self._current_model.generate_stream(
                    messages, **kwargs
                ):
                    chunk_count += 1
                    yield chunk

                # Record success
                self.stats.record_success(provider)

                # If we got here, streaming was successful
                return

            except Exception as e:
                # Record failure
                self.stats.record_failure(provider)

                # Log error and try next provider
                logger.warning(f"Provider {provider} streaming failed: {str(e)}")
                errors.append((provider, str(e)))

                attempts += 1

                # If we've tried all providers, raise the last error
                if attempts > self.failover_retries or len(self.models) <= 1:
                    raise ModelError(
                        f"All providers failed streaming. Last error: {errors[-1][1]}"
                    )

        # This should never happen due to the check above
        raise ModelError("Unexpected error in provider selection")

    def _select_model(self) -> BaseModel:
        """Select a model based on the current strategy"""
        if self.strategy == OptimizationStrategy.ROUND_ROBIN:
            # Simple round-robin selection
            model = self.models[self._current_index]
            self._current_index = (self._current_index + 1) % len(self.models)
            return model

        elif self.strategy == OptimizationStrategy.RANDOM:
            # Random selection
            return random.choice(self.models)

        elif self.strategy == OptimizationStrategy.COST:
            # Select model with lowest cost
            best_cost = float("inf")
            best_model = self._primary_model

            for model in self.models:
                provider = model.__class__.__name__
                metrics = self.stats.get_provider_metrics(provider)

                if metrics.average_cost < best_cost:
                    best_cost = metrics.average_cost
                    best_model = model

            return best_model

        elif self.strategy == OptimizationStrategy.RELIABILITY:
            # Select model with highest success rate
            best_rate = -1
            best_model = self._primary_model

            for model in self.models:
                provider = model.__class__.__name__
                metrics = self.stats.get_provider_metrics(provider)

                if metrics.success_rate > best_rate:
                    best_rate = metrics.success_rate
                    best_model = model

            return best_model

        else:  # OptimizationStrategy.PERFORMANCE
            # Select model with best performance (success rate / latency)
            best_score = -1
            best_model = self._primary_model

            for model in self.models:
                provider = model.__class__.__name__
                metrics = self.stats.get_provider_metrics(provider)

                # Score based on success rate and latency
                score = metrics.success_rate
                if metrics.average_latency > 0:
                    score = score / metrics.average_latency

                if score > best_score:
                    best_score = score
                    best_model = model

            return best_model

    def get_provider_stats(self) -> Dict[str, Any]:
        """Get current provider statistics"""
        return self.stats.get_all_stats()

    def reset_stats(self, provider: Optional[str] = None):
        """Reset statistics for one or all providers"""
        self.stats.reset_stats(provider)

    async def generate_response(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Message:
        """Generate a response using the selected provider

        Args:
            messages: The conversation history
            tools: Optional list of tools available to the model

        Returns:
            Response message from the model

        Raises:
            ModelError: If all providers fail
        """
        attempts = 0
        errors = []

        while attempts <= self.failover_retries:
            self._current_model = self._select_model()
            provider = self.current_provider

            try:
                self.stats.start_request(provider)

                # Call the underlying model's generate_response method
                if hasattr(self._current_model, "generate_response"):
                    response = await self._current_model.generate_response(
                        messages, tools=tools
                    )
                else:
                    # Fallback to generate if generate_response is not available
                    response = await self._current_model.generate(messages)

                # Record success metrics
                self.stats.record_success(provider)

                return response

            except Exception as e:
                # Record failure
                self.stats.record_failure(provider)

                # Log error and try next provider
                logger.warning(f"Provider {provider} failed: {str(e)}")
                errors.append((provider, str(e)))

                attempts += 1

                # If we've tried all providers, raise the last error
                if attempts > self.failover_retries or len(self.models) <= 1:
                    raise ModelError(
                        f"All providers failed. Last error: {errors[-1][1]}"
                    )

        # This should never happen due to the check above
        raise ModelError("Unexpected error in provider selection")

    @classmethod
    async def create(
        cls,
        primary_model: str,
        fallback_models: List[str] = None,
        required_capabilities: Set[ModelCapability] = None,
        optimize_for: OptimizationStrategy = OptimizationStrategy.PERFORMANCE,
        trace_provider=None,
    ) -> "MultiProviderModel":
        """
        Factory method to create a multi-provider model

        Args:
            primary_model: Name of the primary model to use
            fallback_models: Names of fallback models to use
            required_capabilities: Set of required capabilities for all models
            optimize_for: Optimization strategy for provider selection
            trace_provider: Optional trace provider for monitoring

        Returns:
            MultiProviderModel instance
        """
        factory = ModelFactory()
        models = []

        # Create primary model
        try:
            primary = await factory.create(
                model_name=primary_model,
                required_capabilities=required_capabilities,
                trace_provider=trace_provider,
            )
            models.append(primary)
            logger.info(f"Created primary model: {primary.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to create primary model {primary_model}: {str(e)}")
            raise

        # Create fallback models
        if fallback_models:
            for model_name in fallback_models:
                try:
                    model = await factory.create(
                        model_name=model_name,
                        required_capabilities=required_capabilities,
                        trace_provider=trace_provider,
                    )
                    models.append(model)
                    logger.info(f"Created fallback model: {model.__class__.__name__}")
                except Exception as e:
                    logger.warning(
                        f"Failed to create fallback model {model_name}: {str(e)}"
                    )

        return cls(models=models, selection_strategy=optimize_for)
