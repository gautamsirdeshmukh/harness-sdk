"""Private async execution utilities."""

import asyncio
import contextvars
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
_SYNC_BRIDGE_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "strands_sync_bridge_active",
    default=False,
)


def _is_sync_bridge_active() -> bool:
    """Return whether execution entered through the synchronous async bridge."""
    return _SYNC_BRIDGE_ACTIVE.get()


def run_async(async_func: Callable[[], Awaitable[T]]) -> T:
    """Run an async function in a separate thread to avoid event loop conflicts.

    This utility handles the common pattern of running async code from sync contexts
    by using ThreadPoolExecutor to isolate the async execution.

    Args:
        async_func: A callable that returns an awaitable

    Returns:
        The result of the async function
    """

    async def execute_async() -> T:
        token = _SYNC_BRIDGE_ACTIVE.set(True)
        try:
            return await async_func()
        finally:
            _SYNC_BRIDGE_ACTIVE.reset(token)

    def execute() -> T:
        return asyncio.run(execute_async())

    with ThreadPoolExecutor() as executor:
        context = contextvars.copy_context()
        future = executor.submit(context.run, execute)
        return future.result()
