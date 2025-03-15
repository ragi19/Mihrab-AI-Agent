"""
Core tracing functionality for capturing and exporting agent operations
"""

import abc
import asyncio
import contextvars
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from .logging import get_logger

logger = get_logger("utils.tracing")

# Context variables for trace context propagation across async boundaries
_current_trace = contextvars.ContextVar("current_trace", default=None)
_current_span = contextvars.ContextVar("current_span", default=None)

T = TypeVar("T")


@dataclass
class SpanError:
    """Error information for a span"""

    message: str
    data: Optional[Dict[str, Any]] = None


@dataclass
class Span(Generic[T]):
    """A single unit of work within a trace"""

    id: str
    trace_id: str
    parent_id: Optional[str]
    name: str
    data: T
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    error: Optional[SpanError] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        """Mark the span as started"""
        self.started_at = time.time()
        # Store span in context var
        _current_span.set(self)

    def end(self, error: Optional[Union[Exception, SpanError]] = None) -> None:
        """Mark the span as ended"""
        self.ended_at = time.time()
        if error:
            if isinstance(error, Exception):
                self.error = SpanError(
                    message=str(error), data={"type": error.__class__.__name__}
                )
            else:
                self.error = error
        # Clear span from context var if it matches current
        if _current_span.get() == self:
            _current_span.set(None)

    def set_error(self, error: Union[Exception, SpanError]) -> None:
        """Set error information for this span"""
        if isinstance(error, Exception):
            self.error = SpanError(
                message=str(error), data={"type": error.__class__.__name__}
            )
        else:
            self.error = error

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set metadata for this span"""
        self.metadata.update(metadata)

    @property
    def duration(self) -> Optional[float]:
        """Get the span duration in seconds"""
        if self.started_at and self.ended_at:
            return self.ended_at - self.started_at
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "id": self.id,
            "trace_id": self.trace_id,
            "name": self.name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration": self.duration,
            "metadata": self.metadata,
        }

        if self.parent_id:
            result["parent_id"] = self.parent_id

        if self.error:
            result["error"] = asdict(self.error)

        # Include data but handle non-serializable objects
        try:
            result["data"] = self.data
        except (TypeError, ValueError):
            result["data"] = str(self.data)

        return result
        
    # Context manager protocol methods
    def __enter__(self) -> 'Span[T]':
        """Enter the context manager, starting the span"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager, ending the span"""
        if exc_val:
            self.end(error=exc_val)
        else:
            self.end()


@dataclass
class Trace:
    """A complete workflow containing multiple spans"""

    id: str
    name: str
    group_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    error: Optional[SpanError] = None
    spans: List[Span[Any]] = field(default_factory=list)

    def start(self) -> None:
        """Mark the trace as started"""
        self.started_at = time.time()
        # Store trace in context var
        _current_trace.set(self)

    def end(self, error: Optional[Union[Exception, SpanError]] = None) -> None:
        """Mark the trace as ended"""
        self.ended_at = time.time()
        if error:
            if isinstance(error, Exception):
                self.error = SpanError(
                    message=str(error), data={"type": error.__class__.__name__}
                )
            else:
                self.error = error

        # Clear trace from context var if it matches current
        if _current_trace.get() == self:
            _current_trace.set(None)

    def add_span(self, span: Span[Any]) -> None:
        """Add a span to this trace"""
        self.spans.append(span)

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set metadata for this trace"""
        self.metadata.update(metadata)

    @property
    def duration(self) -> Optional[float]:
        """Get the trace duration in seconds"""
        if self.started_at and self.ended_at:
            return self.ended_at - self.started_at
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "group_id": self.group_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration": self.duration,
            "metadata": self.metadata,
            "spans": [span.to_dict() for span in self.spans],
            "error": (
                {"message": self.error.message, "data": self.error.data}
                if self.error
                else None
            ),
        }


class TracingProcessor(abc.ABC):
    """Base class for trace processors"""

    @abc.abstractmethod
    def on_trace_start(self, trace: Trace) -> None:
        """Called when a trace starts"""
        pass

    @abc.abstractmethod
    def on_trace_end(self, trace: Trace) -> None:
        """Called when a trace ends"""
        pass

    @abc.abstractmethod
    def on_span_start(self, span: Span[Any]) -> None:
        """Called when a span starts"""
        pass

    @abc.abstractmethod
    def on_span_end(self, span: Span[Any]) -> None:
        """Called when a span ends"""
        pass

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Shut down the processor"""
        pass

    @abc.abstractmethod
    def force_flush(self) -> None:
        """Force flush any pending items"""
        pass


class TracingExporter(abc.ABC):
    """Base class for trace exporters"""

    @abc.abstractmethod
    def export(self, items: List[Any]) -> None:
        """Export traces or spans"""
        pass


class BatchTraceProcessor(TracingProcessor):
    """Processor that batches traces and spans for export"""

    def __init__(
        self,
        exporter: TracingExporter,
        max_queue_size: int = 8192,
        max_batch_size: int = 128,
        schedule_delay: float = 5.0,
        export_trigger_ratio: float = 0.7,
    ):
        self.exporter = exporter
        self.max_queue_size = max_queue_size
        self.max_batch_size = max_batch_size
        self.schedule_delay = schedule_delay
        self.export_trigger_ratio = export_trigger_ratio

        self._queue = queue.Queue(maxsize=max_queue_size)
        self._shutdown = threading.Event()
        self._worker = threading.Thread(target=self._worker_thread, daemon=True)
        self._worker.start()

    def on_trace_start(self, trace: Trace) -> None:
        """Called when a trace starts"""
        pass

    def on_trace_end(self, trace: Trace) -> None:
        """Called when a trace ends"""
        self._enqueue(trace)

    def on_span_start(self, span: Span[Any]) -> None:
        """Called when a span starts"""
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        """Called when a span ends"""
        pass

    def shutdown(self) -> None:
        """Shut down the processor"""
        self._shutdown.set()
        self._worker.join(timeout=5.0)

    def force_flush(self) -> None:
        """Force flush any pending items"""
        self._export_batch()

    def _worker_thread(self) -> None:
        """Worker thread that exports batches"""
        while not self._shutdown.is_set():
            try:
                self._export_batch()
            except Exception as e:
                logger.error(f"Error in batch processor: {e}", exc_info=True)

            self._shutdown.wait(self.schedule_delay)

    def _export_batch(self) -> None:
        """Export a batch of items"""
        if self._queue.empty():
            return

        batch = []
        try:
            # Get up to max_batch_size items
            for _ in range(self.max_batch_size):
                if self._queue.empty():
                    break
                batch.append(self._queue.get_nowait())
                self._queue.task_done()

            # Export the batch
            if batch:
                self.exporter.export(batch)
        except Exception as e:
            logger.error(f"Error exporting batch: {e}", exc_info=True)

    def _enqueue(self, item: Any) -> None:
        """Add an item to the queue"""
        try:
            self._queue.put_nowait(item)

            # If queue is getting full, trigger export
            if self._queue.qsize() > self.max_queue_size * self.export_trigger_ratio:
                self._export_batch()
        except queue.Full:
            logger.warning("Trace processor queue full, dropping item")


class ConsoleExporter(TracingExporter):
    """Simple exporter that prints to console"""

    def export(self, items: List[Any]) -> None:
        """Export traces or spans to console"""
        for item in items:
            if isinstance(item, Trace):
                print(f"Trace: {item.name} ({item.duration}s)")
                for span in item.spans:
                    print(f"  Span: {span.name} ({span.duration}s)")
            elif isinstance(item, Span):
                print(f"Span: {item.name} ({item.duration}s)")


class FileExporter(TracingExporter):
    """Exporter that writes traces to a JSON file"""

    def __init__(self, file_path: str, append: bool = True):
        self.file_path = file_path
        self.append = append

        # Create directory if it doesn't exist
        file_dir = os.path.dirname(file_path)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        # Initialize file if not appending
        if not append and os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("[]")

    def export(self, items: List[Any]) -> None:
        """Export traces to file"""
        if not items:
            return

        try:
            # Convert items to dictionaries
            serialized_items = []
            for item in items:
                if hasattr(item, "to_dict"):
                    serialized_items.append(item.to_dict())
                elif isinstance(item, dict):
                    serialized_items.append(item)
                else:
                    logger.warning(f"Cannot serialize item of type {type(item)}")

            if not serialized_items:
                return

            # Read existing data if appending
            existing_data = []
            if self.append and os.path.exists(self.file_path):
                try:
                    with open(self.file_path, "r") as f:
                        content = f.read().strip()
                        if content:
                            existing_data = json.loads(content)
                        else:
                            existing_data = []
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not parse existing file {self.file_path}, overwriting"
                    )
                    existing_data = []

            # Combine and write data
            all_data = existing_data + serialized_items
            with open(self.file_path, "w") as f:
                json.dump(all_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error exporting traces to file: {e}", exc_info=True)


class TraceProvider:
    """Central component for managing traces"""

    def __init__(self):
        self._processors: List[TracingProcessor] = []

    def register_processor(self, processor: TracingProcessor) -> None:
        """Register a trace processor"""
        self._processors.append(processor)

    def create_trace(self, name: str, group_id: Optional[str] = None) -> Trace:
        """Create a new trace"""
        trace = Trace(id=str(uuid.uuid4()), name=name, group_id=group_id)
        _current_trace.set(trace)

        # Notify processors
        for processor in self._processors:
            try:
                processor.on_trace_start(trace)
            except Exception as e:
                logger.error(f"Error in processor: {e}", exc_info=True)

        return trace

    def create_span(
        self, name: str, data: T, parent_id: Optional[str] = None
    ) -> Span[T]:
        """Create a new span"""
        current_trace = _current_trace.get()
        if not current_trace:
            raise RuntimeError("No active trace")

        span = Span(
            id=str(uuid.uuid4()),
            trace_id=current_trace.id,
            parent_id=parent_id
            or (_current_span.get().id if _current_span.get() else None),
            name=name,
            data=data,
        )
        current_trace.add_span(span)
        _current_span.set(span)

        # Notify processors
        for processor in self._processors:
            try:
                processor.on_span_start(span)
            except Exception as e:
                logger.error(f"Error in processor: {e}", exc_info=True)

        return span

    def end_trace(self, trace: Trace, error: Optional[Exception] = None) -> None:
        """Mark a trace as ended"""
        trace.end(error)
        _current_trace.set(None)

        # Notify processors
        for processor in self._processors:
            try:
                processor.on_trace_end(trace)
            except Exception as e:
                logger.error(f"Error in processor: {e}", exc_info=True)

    def end_span(self, span: Span[Any], error: Optional[Exception] = None) -> None:
        """Mark a span as ended"""
        span.end(error)
        _current_span.set(None)

        # Notify processors
        for processor in self._processors:
            try:
                processor.on_span_end(span)
            except Exception as e:
                logger.error(f"Error in processor: {e}", exc_info=True)


class FileTraceProvider(TraceProvider):
    """Trace provider that writes traces to a file"""

    def __init__(
        self, file_path: str, batch_size: int = 10, flush_interval: float = 5.0
    ):
        super().__init__()

        # Create file exporter and batch processor
        exporter = FileExporter(file_path)
        processor = BatchTraceProcessor(
            exporter=exporter, max_batch_size=batch_size, schedule_delay=flush_interval
        )

        # Register processor
        self.register_processor(processor)
