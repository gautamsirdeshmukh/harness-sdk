import type { Interrupt } from '../interrupt.js'
import type { Serialized } from '../types/json.js'
import type { ToolResultContentData } from '../types/messages.js'

/** Configures in-process background task execution. @internal */
export interface InProcessTaskManagerConfig {
  /** Wait for all background tasks before an invocation returns. Defaults to `true`. */
  readonly waitForCompletion?: boolean
  /** Maximum number of physically executing background tasks. Defaults to `4`. */
  readonly maxConcurrency?: number
  /** Per-execution timeout in milliseconds. Defaults to `Infinity`. */
  readonly timeout?: number
}

/** Read-only snapshot of a background task. @internal */
export interface BackgroundTask {
  /** Stable identifier for task inspection and cancellation. */
  readonly taskId: string
  /** Tool-use identifier from the original model request. */
  readonly toolUseId: string
  /** Registered name of the executing tool. */
  readonly toolName: string
  /** Current task lifecycle status. */
  readonly status: 'queued' | 'working' | 'paused' | 'completed' | 'failed' | 'cancelled'
  /** ISO timestamp recorded when the task was admitted. */
  readonly createdAt: string
  /** ISO timestamp recorded at the latest task state change. */
  readonly updatedAt: string
  /** Serialized tool result when execution produced one. */
  readonly result?: {
    /** Tool-result content blocks. */
    readonly content: readonly Serialized<ToolResultContentData>[]
  }
  /** Task failure details. */
  readonly error?: {
    /** Failure category. */
    readonly type: 'toolError' | 'executionError' | 'timeout' | 'recoveryError'
    /** Failure message. */
    readonly message: string
  }
  /** Unanswered interrupts when the task is paused. */
  readonly interrupts?: readonly Readonly<Interrupt>[]
}
