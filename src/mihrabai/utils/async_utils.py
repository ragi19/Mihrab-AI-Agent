"""
Async utilities for parallel execution and task management
"""

import asyncio
from typing import Awaitable, Callable, List, Optional, TypeVar

T = TypeVar("T")


async def gather_with_concurrency(
    limit: int, tasks: List[Awaitable[T]], *, return_exceptions: bool = False
) -> List[T]:
    """Run tasks with a concurrency limit"""
    semaphore = asyncio.Semaphore(limit)

    async def wrapped_task(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task

    return await asyncio.gather(
        *[wrapped_task(task) for task in tasks], return_exceptions=return_exceptions
    )


async def with_retry(
    func: Callable[[], Awaitable[T]],
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Optional[tuple] = None,
) -> T:
    """Execute an async function with retry logic"""
    exceptions = exceptions or (Exception,)
    last_error = None

    for attempt in range(retries):
        try:
            return await func()
        except exceptions as e:
            last_error = e
            if attempt < retries - 1:
                await asyncio.sleep(delay * (backoff**attempt))

    raise last_error
