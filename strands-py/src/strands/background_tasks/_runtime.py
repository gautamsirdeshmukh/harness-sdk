"""Process-lifetime event-loop runtime for detached background work."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import threading
from collections.abc import Awaitable, Callable
from typing import TypeVar, cast

T = TypeVar("T")


class _BackgroundTaskRuntime:
    """Own one daemon event loop shared by all background-task managers."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._start_lock = threading.Lock()

    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        existing_loop = self._current_loop()
        if existing_loop is not None:
            return existing_loop

        with self._start_lock:
            if self._loop is None:
                self._thread = threading.Thread(
                    target=self._run_loop,
                    name="strands-background-tasks",
                    daemon=True,
                )
                self._thread.start()
                self._ready.wait()

        loop = self._current_loop()
        if loop is None:
            raise RuntimeError("Background Tasks runtime failed to start")
        return loop

    def _current_loop(self) -> asyncio.AbstractEventLoop | None:
        return self._loop

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        loop.run_forever()

    async def run(self, operation: Callable[[], T | Awaitable[T]]) -> T:
        """Run an operation on the persistent loop from any caller loop."""
        loop = self._ensure_started()
        if threading.current_thread() is self._thread:
            return await self._invoke(operation)

        future = asyncio.run_coroutine_threadsafe(self._invoke(operation), loop)
        return await asyncio.wrap_future(future)

    async def run_void(self, operation: Callable[[], Awaitable[None]]) -> None:
        """Run an asynchronous operation that has no result."""
        loop = self._ensure_started()
        if threading.current_thread() is self._thread:
            await operation()
            return

        async def invoke() -> None:
            await operation()

        future = asyncio.run_coroutine_threadsafe(invoke(), loop)
        await asyncio.wrap_future(future)

    def run_sync(self, operation: Callable[[], T | Awaitable[T]]) -> T:
        """Run an operation on the persistent loop from synchronous code."""
        loop = self._ensure_started()
        if threading.current_thread() is self._thread:
            raise RuntimeError("Cannot synchronously wait on the Background Tasks runtime thread")

        future = asyncio.run_coroutine_threadsafe(self._invoke(operation), loop)
        return future.result()

    def submit(self, operation: Callable[[], T | Awaitable[T]]) -> concurrent.futures.Future[T]:
        """Schedule an operation without tying it to a caller event loop."""
        loop = self._ensure_started()
        return asyncio.run_coroutine_threadsafe(self._invoke(operation), loop)

    async def _invoke(self, operation: Callable[[], T | Awaitable[T]]) -> T:
        result = operation()
        if inspect.isawaitable(result):
            return await cast(Awaitable[T], result)
        return result


_BACKGROUND_TASK_RUNTIME = _BackgroundTaskRuntime()


def get_background_task_runtime() -> _BackgroundTaskRuntime:
    """Return the shared process-lifetime Background Tasks runtime."""
    return _BACKGROUND_TASK_RUNTIME
