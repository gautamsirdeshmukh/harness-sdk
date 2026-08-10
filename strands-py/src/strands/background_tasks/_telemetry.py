"""Bounded-cardinality telemetry for background task lifecycle events."""

from __future__ import annotations

from typing import Literal

from opentelemetry import metrics as metrics_api
from opentelemetry.metrics import Counter, Histogram

from ..types.traces import AttributeValue


class _BackgroundTaskTelemetry:
    """Emit Background Tasks metrics through the configured global provider."""

    def __init__(self) -> None:
        meter = metrics_api.get_meter("strands.background_tasks")
        self._admitted: Counter = meter.create_counter(
            "gen_ai.agent.background_task.admitted.count",
            description="Number of background tasks durably admitted",
        )
        self._execution: Counter = meter.create_counter(
            "gen_ai.agent.background_task.execution.count",
            description="Number of physical background task executions started",
        )
        self._queue_duration: Histogram = meter.create_histogram(
            "gen_ai.agent.background_task.queue.duration",
            unit="ms",
            description="Time background task executions spend queued",
        )
        self._execution_duration: Histogram = meter.create_histogram(
            "gen_ai.agent.background_task.execution.duration",
            unit="ms",
            description="Duration of physical background task executions",
        )
        self._failure: Counter = meter.create_counter(
            "gen_ai.agent.background_task.failure.count",
            description="Number of failed background task attempts",
        )
        self._cancellation: Counter = meter.create_counter(
            "gen_ai.agent.background_task.cancellation.count",
            description="Number of accepted background task cancellation requests",
        )
        self._terminal: Counter = meter.create_counter(
            "gen_ai.agent.background_task.terminal.count",
            description="Number of background tasks committed to a terminal status",
        )

    def record_admission(self, tool_name: str) -> None:
        """Record durable task admission."""
        self._admitted.add(1, _tool_attributes(tool_name))

    def record_execution_started(
        self,
        *,
        tool_name: str,
        attempt: int,
        resumed: bool,
        queue_duration: float,
    ) -> None:
        """Record one physical execution start."""
        attributes: dict[str, AttributeValue] = {
            **_tool_attributes(tool_name),
            "background_task.attempt": attempt,
            "background_task.resumed": resumed,
        }
        self._execution.add(1, attributes)
        self._queue_duration.record(max(0.0, queue_duration), attributes)

    def record_execution_finished(
        self,
        *,
        tool_name: str,
        outcome: str,
        duration: float,
    ) -> None:
        """Record one physical execution finish."""
        self._execution_duration.record(
            max(0.0, duration),
            {
                **_tool_attributes(tool_name),
                "background_task.outcome": outcome,
            },
        )

    def record_failure(self, tool_name: str, failure_type: str) -> None:
        """Record a classified failure."""
        self._failure.add(
            1,
            {
                **_tool_attributes(tool_name),
                "background_task.failure.type": failure_type,
            },
        )

    def record_cancellation(self, tool_name: str) -> None:
        """Record an accepted cancellation request."""
        self._cancellation.add(1, _tool_attributes(tool_name))

    def record_terminal(
        self,
        tool_name: str,
        status: Literal["completed", "failed", "cancelled"],
    ) -> None:
        """Record a task entering a terminal state."""
        self._terminal.add(
            1,
            {
                **_tool_attributes(tool_name),
                "background_task.status": status,
            },
        )


def _tool_attributes(tool_name: str) -> dict[str, str]:
    return {"gen_ai.tool.name": tool_name}
