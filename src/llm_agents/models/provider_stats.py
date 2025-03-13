"""
Provider statistics tracking for multi-provider model
"""

import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ProviderMetrics:
    successes: int = 0
    failures: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    request_times: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.0

    @property
    def average_latency(self) -> float:
        return statistics.mean(self.request_times) if self.request_times else 0.0

    @property
    def avg_tokens_per_request(self) -> float:
        total_requests = self.successes + self.failures
        return self.total_tokens / total_requests if total_requests > 0 else 0.0

    @property
    def avg_cost_per_request(self) -> float:
        total_requests = self.successes + self.failures
        return self.total_cost / total_requests if total_requests > 0 else 0.0


class ProviderStats:
    """Tracks performance statistics for LLM providers"""

    def __init__(self):
        self._stats: Dict[str, ProviderMetrics] = {}
        self._start_times: Dict[str, float] = {}

    def start_request(self, provider: str):
        """Start timing a provider request"""
        self._start_times[provider] = time.time()

    def record_success(
        self,
        provider: str,
        duration: float | None = None,
        tokens: int = 0,
        cost: float = 0.0,
    ):
        """Record a successful provider request"""
        if provider not in self._stats:
            self._stats[provider] = ProviderMetrics()

        stats = self._stats[provider]
        stats.successes += 1
        stats.total_tokens += tokens
        stats.total_cost += cost

        if duration is None and provider in self._start_times:
            duration = time.time() - self._start_times[provider]
            del self._start_times[provider]

        if duration is not None:
            stats.request_times.append(duration)

    def record_failure(self, provider: str):
        """Record a failed provider request"""
        if provider not in self._stats:
            self._stats[provider] = ProviderMetrics()

        if provider in self._start_times:
            del self._start_times[provider]

        self._stats[provider].failures += 1

    def get_provider_metrics(self, provider: str) -> ProviderMetrics:
        """Get statistics for a specific provider"""
        if provider not in self._stats:
            self._stats[provider] = ProviderMetrics()
        return self._stats[provider]

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all providers"""
        return {
            provider: {
                "successes": metrics.successes,
                "failures": metrics.failures,
                "success_rate": metrics.success_rate,
                "avg_latency": metrics.average_latency,
                "total_tokens": metrics.total_tokens,
                "avg_tokens": metrics.avg_tokens_per_request,
                "total_cost": metrics.total_cost,
                "avg_cost": metrics.avg_cost_per_request,
            }
            for provider, metrics in self._stats.items()
        }

    def reset_stats(self, provider: str | None = None):
        """Reset statistics for one or all providers"""
        if provider:
            if provider in self._stats:
                self._stats[provider] = ProviderMetrics()
        else:
            self._stats.clear()
