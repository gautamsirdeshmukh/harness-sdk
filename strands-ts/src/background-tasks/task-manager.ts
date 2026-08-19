import type { SpanContext } from '@opentelemetry/api'

import type { InvocationState } from '../types/agent.js'
import type { JSONValue } from '../types/json.js'
import type { BackgroundTask } from './types.js'

/** Approved in-process tool call submitted for background execution. @internal */
export interface ToolCallSubmission {
  /** Registered name of the tool to execute. */
  readonly toolName: string
  /** Tool-use identifier from the originating model request. */
  readonly originalToolUseId: string
  /** Tool input with framework-owned routing fields removed. */
  readonly input: JSONValue
  /** Invocation-scoped state propagated to the tool. */
  readonly invocationState: InvocationState
  /** Agent-loop pass used to deduplicate task admission. */
  readonly passId: string
  /** Trace context captured when the tool call was submitted. */
  readonly originSpanContext?: SpanContext
}

/**
 * Background task lifecycle operations used by the Background Tasks plugin.
 *
 * @typeParam TSubmission - Submission shape accepted by this manager implementation.
 * @internal
 */
export interface TaskManager<TSubmission> {
  /** Initializes persisted task state and execution resources. */
  initialize(): Promise<void>
  /** Registers the manager's agent lifecycle hooks. */
  registerHooks(): void
  /** Reloads manager state after the agent restores application state. */
  appStateLoaded(): void
  /** Submits work for background execution. */
  submitTask(submission: TSubmission): Promise<BackgroundTask>
  /** Gets one task by its stable identifier. */
  getTask(taskId: string): Promise<BackgroundTask | undefined>
  /** Lists the tasks currently tracked by the manager. */
  listTasks(): Promise<readonly BackgroundTask[]>
  /** Requests cancellation of one task. */
  cancelTask(taskId: string): Promise<BackgroundTask>
  /** Waits until the manager has no queued or executing tasks. */
  waitForTasks(options?: { readonly timeout?: number }): Promise<void>
}
