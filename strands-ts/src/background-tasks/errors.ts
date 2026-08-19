/** Raised when a background task ID is not found. @internal */
export class BackgroundTaskNotFoundError extends Error {
  /**
   * @param taskId - ID that could not be found.
   */
  constructor(taskId: string) {
    super(`Background task '${taskId}' was not found`)
    this.name = 'BackgroundTaskNotFoundError'
  }
}

/** Raised when waiting for in-process background tasks exceeds its timeout. @internal */
export class BackgroundTasksTimeoutError extends Error {
  /** Timeout supplied to the wait operation, in milliseconds. */
  readonly timeoutMs: number

  /**
   * @param timeoutMs - Timeout supplied to the wait operation, in milliseconds.
   */
  constructor(timeoutMs: number) {
    super(`Background Tasks wait timed out after ${timeoutMs}ms`)
    this.name = 'BackgroundTasksTimeoutError'
    this.timeoutMs = timeoutMs
  }
}
