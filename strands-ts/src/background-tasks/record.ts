import { isSpanContextValid, type SpanContext } from '@opentelemetry/api'
import { InterruptState, type InterruptStateData } from '../interrupt.js'
import type { JSONValue, Serialized } from '../types/json.js'
import { ToolResultBlock, type ToolResultBlockData, type ToolResultContentData } from '../types/messages.js'
import { validateStoredInProcessTask } from './in-process/record.js'
import type { StoredInProcessTask } from './in-process/types.js'
import type { BackgroundTask } from './types.js'

/** Persisted identity for one approved in-process tool execution. @internal */
export interface StoredToolTaskDescriptor {
  /** Tool-use identifier from the originating model request. */
  readonly originalToolUseId: string
  /** Registered name of the tool to execute. */
  readonly toolName: string
  /** Trace context captured when the tool call was submitted. */
  readonly originTraceContext?: SpanContext
}

/** Live descriptor retained only while the process can execute the task. @internal */
export interface ToolTaskDescriptor extends StoredToolTaskDescriptor {
  /** Tool input with framework-owned routing fields removed. */
  readonly input: JSONValue
  /** Opaque key for manager-owned invocation state. */
  readonly executionStateId: string
}

/** Persisted in-process background task record. @internal */
export type StoredBackgroundTask = StoredInProcessTask<
  StoredToolTaskDescriptor,
  ToolResultBlockData,
  InterruptStateData
>

/** In-memory background task record with transient execution data. @internal */
export type LiveBackgroundTask = StoredInProcessTask<ToolTaskDescriptor, ToolResultBlockData, InterruptStateData>

/** Validates a persisted in-process background task record. @internal */
export function validateStoredTask(value: unknown): asserts value is StoredBackgroundTask {
  validateStoredInProcessTask(value)
  const record = value as unknown as StoredBackgroundTask
  validateStoredToolTaskDescriptor(record.descriptor)
  if (record.result !== undefined) validateToolTaskResult(record.result)
  if (record.state !== undefined) validateToolTaskState(record.state)
  if (
    record.failure !== undefined &&
    !['toolError', 'executionError', 'timeout', 'recoveryError'].includes(record.failure.type)
  ) {
    invalid('task.failure.type', `unknown failure type '${record.failure.type}'`)
  }
}

function validateStoredToolTaskDescriptor(value: unknown): asserts value is StoredToolTaskDescriptor {
  const descriptor = requireObject(value, 'task.descriptor')
  for (const key of ['originalToolUseId', 'toolName'] as const) {
    requireString(descriptor[key], `task.descriptor.${key}`)
  }
  if (descriptor.originTraceContext !== undefined) validateTraceContext(descriptor.originTraceContext)
}

/** Removes transient execution data before persisting manager state. @internal */
export function storeBackgroundTask(record: LiveBackgroundTask): StoredBackgroundTask {
  const { originalToolUseId, toolName, originTraceContext } = record.descriptor
  return {
    ...record,
    descriptor: {
      originalToolUseId,
      toolName,
      ...(originTraceContext && { originTraceContext }),
    },
  }
}

/** Reconstructs the inert descriptor used while recovering a persisted task. @internal */
export function restoreBackgroundTask(record: StoredBackgroundTask): LiveBackgroundTask {
  return {
    ...record,
    descriptor: {
      ...record.descriptor,
      input: {},
      executionStateId: `restored:${record.taskId}`,
    },
  }
}

/** Projects a persisted task record into its read-only manager snapshot. @internal */
export function toBackgroundTask(
  record: StoredInProcessTask<StoredToolTaskDescriptor, ToolResultBlockData, InterruptStateData>
): BackgroundTask {
  const resultBlock = record.result ? rehydrateStoredToolResult(record.result) : undefined
  const result: BackgroundTask['result'] = resultBlock
    ? {
        content: resultBlock.toJSON().toolResult.content as Serialized<ToolResultContentData>[],
      }
    : undefined
  const error: BackgroundTask['error'] = record.failure
    ? {
        type: record.failure.type as NonNullable<BackgroundTask['error']>['type'],
        message: record.failure.message,
      }
    : undefined
  const interrupts = record.state ? InterruptState.fromJSON(record.state).getUnansweredInterrupts() : []
  return {
    taskId: record.taskId,
    toolUseId: record.descriptor.originalToolUseId,
    toolName: record.descriptor.toolName,
    status: record.status,
    createdAt: record.createdAt,
    updatedAt: record.updatedAt,
    ...(result && { result }),
    ...(error && { error }),
    ...(interrupts.length > 0 && { interrupts }),
  }
}

/** Reconstructs a tool-result block from persisted data. @internal */
export function rehydrateStoredToolResult(result: ToolResultBlockData): ToolResultBlock {
  try {
    return ToolResultBlock.fromJSON({ toolResult: result })
  } catch (error) {
    throw new Error('Stored tool result cannot be reconstructed', { cause: error })
  }
}

function validateToolTaskResult(value: unknown): asserts value is ToolResultBlockData {
  try {
    ToolResultBlock.fromJSON({ toolResult: value as ToolResultBlockData })
  } catch (error) {
    throw new Error('task.result cannot be reconstructed', { cause: error })
  }
}

function validateToolTaskState(value: unknown): asserts value is InterruptStateData {
  try {
    InterruptState.fromJSON(value as InterruptStateData)
  } catch (error) {
    throw new Error('task.state cannot be reconstructed', {
      cause: error,
    })
  }
}

function validateTraceContext(value: unknown): void {
  const traceContext = requireObject(value, 'task.descriptor.originTraceContext')
  if (!isSpanContextValid(traceContext as unknown as SpanContext)) {
    invalid('task.descriptor.originTraceContext', 'must be a valid span context')
  }
  if (traceContext.isRemote !== undefined && typeof traceContext.isRemote !== 'boolean') {
    invalid('task.descriptor.originTraceContext.isRemote', 'must be a boolean')
  }
}

function requireObject(value: unknown, path: string): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) invalid(path, 'must be an object')
  return value as Record<string, unknown>
}

function requireString(value: unknown, path: string): asserts value is string {
  if (typeof value !== 'string' || value.length === 0) invalid(path, 'must be a non-empty string')
}

function invalid(path: string, message: string): never {
  throw new Error(`${path} ${message}`)
}
