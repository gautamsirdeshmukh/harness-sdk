import type { SpanContext } from '@opentelemetry/api'

import type { InterruptStateData } from '../interrupt.js'
import type { InvocationState } from '../types/agent.js'
import type { ToolResultBlock, ToolUseBlock } from '../types/messages.js'

import type { Agent } from './agent.js'

/** Input for one tool execution detached from the foreground agent loop. @internal */
export interface DetachedToolInput {
  /** Tool call to execute. */
  readonly toolUseBlock: ToolUseBlock
  /** Invocation state propagated to hooks and the tool. */
  readonly invocationState: InvocationState
  /** Cancellation signal scoped to the detached execution. */
  readonly cancelSignal: AbortSignal
  /** Persisted interrupt state when resuming a paused execution. */
  readonly interruptState?: InterruptStateData
  /** Background execution identity recorded in telemetry. */
  readonly background: {
    readonly taskId: string
    readonly attempt: number
    readonly attemptId: string
    readonly executionId: string
  }
  /** Classifies an aborted execution after the manager commits its task state. */
  readonly classifyAbort: () => 'cancelled' | 'failed'
  /** Parent trace context captured when the task was admitted. */
  readonly originSpanContext?: SpanContext
}

/** Result from one detached tool execution. @internal */
export type DetachedToolOutcome = ToolResultBlock | { readonly interruptState: InterruptStateData }

type DetachedToolExecutor = (input: DetachedToolInput) => Promise<DetachedToolOutcome>

const executorByAgent = new WeakMap<Agent, DetachedToolExecutor>()
const initializationByAgent = new WeakMap<Agent, Promise<void>>()

/** Registers the detached-tool capability owned by an Agent instance. @internal */
export function registerDetachedToolExecutor(agent: Agent, executor: DetachedToolExecutor): void {
  executorByAgent.set(agent, executor)
}

/** Executes a tool through an Agent's internal detached-tool capability. @internal */
export async function executeDetachedTool(agent: Agent, input: DetachedToolInput): Promise<DetachedToolOutcome> {
  const executor = executorByAgent.get(agent)
  if (!executor) throw new Error(`Agent '${agent.id}' is not initialized for detached tool execution`)
  let initialization = initializationByAgent.get(agent)
  if (!initialization) {
    initialization = agent.initialize()
    initializationByAgent.set(agent, initialization)
  }
  try {
    await initialization
  } catch (error) {
    if (initializationByAgent.get(agent) === initialization) initializationByAgent.delete(agent)
    throw error
  }
  return executor(input)
}
