"""Programmatic controls for background tool execution."""

from __future__ import annotations

import builtins

from .._async import run_async
from ._manager import _InProcessTaskManager
from .types import BackgroundTask


class BackgroundTasks:
    """Programmatic controls for one Agent's background tasks."""

    def __init__(self, manager: _InProcessTaskManager) -> None:
        """Initialize controls for an Agent-owned task manager.

        Args:
            manager: Internal manager that owns the Agent's background tasks.
        """
        self._manager = manager

    def get(self, task_id: str) -> BackgroundTask | None:
        """Return a task by ID.

        Delivered tasks are pruned after their results are added to the
        conversation, so this method returns ``None`` for unknown and delivered
        task IDs.

        Args:
            task_id: Task ID returned when background work was admitted.

        Returns:
            A read-only task snapshot, or ``None`` when the task is not visible.
        """
        return run_async(lambda: self.get_async(task_id))

    async def get_async(self, task_id: str) -> BackgroundTask | None:
        """Return a task by ID without blocking the caller event loop.

        Delivered tasks are pruned after their results are added to the
        conversation, so this method returns ``None`` for unknown and delivered
        task IDs.

        Args:
            task_id: Task ID returned when background work was admitted.

        Returns:
            A read-only task snapshot, or ``None`` when the task is not visible.
        """
        return await self._manager.get_task(task_id)

    def list(self) -> builtins.list[BackgroundTask]:
        """Return task snapshots in admission order.

        Delivered tasks are pruned after their results are added to the
        conversation and are not included.

        Returns:
            Read-only snapshots of tasks whose results have not been delivered.
        """
        return run_async(self.list_async)

    async def list_async(self) -> builtins.list[BackgroundTask]:
        """Return task snapshots without blocking the caller event loop.

        Delivered tasks are pruned after their results are added to the
        conversation and are not included.

        Returns:
            Read-only snapshots of tasks whose results have not been delivered.
        """
        return await self._manager.list_tasks()

    def cancel(self, task_id: str) -> BackgroundTask:
        """Request cooperative cancellation and return the updated task.

        Args:
            task_id: Task ID returned when background work was admitted.

        Returns:
            The task snapshot after the cancellation request.

        Raises:
            BackgroundTaskNotFoundError: If the task is not visible.
        """
        return run_async(lambda: self.cancel_async(task_id))

    async def cancel_async(self, task_id: str) -> BackgroundTask:
        """Request cancellation without blocking the caller event loop.

        Args:
            task_id: Task ID returned when background work was admitted.

        Returns:
            The task snapshot after the cancellation request.

        Raises:
            BackgroundTaskNotFoundError: If the task is not visible.
        """
        return await self._manager.cancel_task(task_id)

    def wait(self, *, timeout: float | None = None) -> None:
        """Wait until no background task is queued or physically executing.

        A timeout stops waiting without cancelling tasks.

        Args:
            timeout: Maximum wait in seconds, or ``None`` to wait indefinitely.

        Raises:
            TypeError: If ``timeout`` is not a positive finite number.
            BackgroundTasksTimeoutError: If the wait exceeds ``timeout``.
        """
        run_async(lambda: self.wait_async(timeout=timeout))

    async def wait_async(self, *, timeout: float | None = None) -> None:
        """Wait for physical idleness without blocking the caller event loop.

        A timeout stops waiting without cancelling tasks.

        Args:
            timeout: Maximum wait in seconds, or ``None`` to wait indefinitely.

        Raises:
            TypeError: If ``timeout`` is not a positive finite number.
            BackgroundTasksTimeoutError: If the wait exceeds ``timeout``.
        """
        await self._manager.wait_for_tasks(timeout)
