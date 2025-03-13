"""
Runtime context management for agents
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class RuntimeContext:
    """Context object for managing runtime state"""

    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_context: Optional["RuntimeContext"] = None
    is_processing: bool = False
    child_contexts: Dict[str, "RuntimeContext"] = field(default_factory=dict)
    execution_path: List[str] = field(default_factory=list)
    error_count: int = 0
    max_retries: int = 3
    resources: Set[str] = field(default_factory=set)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a variable from context, checking parent contexts if not found"""
        if key in self.variables:
            return self.variables[key]
        if self.parent_context:
            return self.parent_context.get(key, default)
        return default

    def set(self, key: str, value: Any) -> None:
        """Set a variable in the current context"""
        self.variables[key] = value

    def create_child(self, name: str = "") -> "RuntimeContext":
        """Create a new child context with optional name"""
        child_id = name or f"child_{len(self.child_contexts)}"
        child = RuntimeContext(parent_context=self)
        self.child_contexts[child_id] = child
        return child

    def get_child_contexts(self) -> Dict[str, "RuntimeContext"]:
        """Get all child contexts"""
        return self.child_contexts

    def start_processing(self) -> None:
        """Mark the context as processing"""
        self.is_processing = True
        # Record execution path if there's a parent
        if self.parent_context:
            path_entry = f"context_{id(self)}"
            self.parent_context.execution_path.append(path_entry)

    def end_processing(self) -> None:
        """Mark the context as not processing"""
        self.is_processing = False

    def with_metadata(self, **kwargs) -> "RuntimeContext":
        """Add metadata and return self for chaining"""
        self.metadata.update(kwargs)
        return self

    def record_error(self, error: Exception) -> int:
        """Record an error and return current error count"""
        self.error_count += 1
        self.metadata["last_error"] = str(error)
        return self.error_count

    def can_retry(self) -> bool:
        """Check if operation can be retried based on error count"""
        return self.error_count < self.max_retries

    def reset_errors(self) -> None:
        """Reset error count"""
        self.error_count = 0

    def acquire_resource(self, resource_id: str) -> bool:
        """Attempt to acquire a resource, return success status"""
        if resource_id in self.resources:
            return False
        self.resources.add(resource_id)
        return True

    def release_resource(self, resource_id: str) -> None:
        """Release a previously acquired resource"""
        if resource_id in self.resources:
            self.resources.remove(resource_id)

    def merge_from(self, other: "RuntimeContext") -> None:
        """Merge variables and metadata from another context"""
        self.variables.update(other.variables)
        self.metadata.update(other.metadata)
