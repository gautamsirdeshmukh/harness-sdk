"""Public exceptions for background tool execution."""


class BackgroundTaskNotFoundError(LookupError):
    """Raised when a background task ID is not visible to callers."""

    def __init__(self, task_id: str) -> None:
        """Initialize the error for an unknown task ID.

        Args:
            task_id: Task ID that could not be found.
        """
        super().__init__(f"Background task '{task_id}' was not found")


class BackgroundTasksTimeoutError(TimeoutError):
    """Raised when waiting for Background Tasks to become idle times out."""

    def __init__(self, timeout: float) -> None:
        """Initialize the error for a wait timeout.

        Args:
            timeout: Timeout supplied to the wait operation, in seconds.
        """
        self.timeout = timeout
        super().__init__(f"Background Tasks wait timed out after {timeout}s")
