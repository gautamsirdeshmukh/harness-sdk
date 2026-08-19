import { isSpanContextValid } from '@opentelemetry/api'

import { normalizeError } from '../errors.js'
import { InterruptError, InterruptState, type Interrupt, type InterruptStateData } from '../interrupt.js'
import { InterruptResponseContent, type InterruptResponse } from '../types/interrupt.js'
import { deepCopyWithValidation } from '../types/json.js'
import { TextBlock, ToolUseBlock, type ToolResultBlock, type ToolResultBlockData } from '../types/messages.js'
import type { Agent } from '../agent/agent.js'
import { continuations } from '../agent/continuation.js'
import { executeDetachedTool } from '../agent/detached-tool.js'
import { getEventInterruptState, getInvocationResult, setInvocationResult } from '../hooks/event-state.js'
import { InProcessTaskEngine } from './in-process/engine.js'
import { isInProcessTaskTerminalStatus } from './in-process/record.js'
import type { InProcessTaskExecutionContext, TaskExecutionOutcome } from './in-process/types.js'
import { BackgroundTaskNotFoundError, BackgroundTasksTimeoutError } from './errors.js'
import { AfterInvocationEvent, AfterModelCallEvent, BeforeModelCallEvent } from '../hooks/events.js'
import { HookOrder } from '../hooks/types.js'
import { AgentResult, type InvocationState } from '../types/agent.js'
import type { JSONValue } from '../types/json.js'
import {
  assertDeliveryConsumed,
  removeBackgroundDeliveries,
  renderBackgroundDelivery,
  stableStringify,
  unpinBackgroundDeliveries,
} from './delivery.js'
import {
  restoreBackgroundTask,
  storeBackgroundTask,
  toBackgroundTask,
  validateStoredTask,
  type LiveBackgroundTask,
  type StoredBackgroundTask,
  type ToolTaskDescriptor,
} from './record.js'
import { BackgroundTaskTelemetry } from './telemetry.js'
import type { BackgroundTask, InProcessTaskManagerConfig } from './types.js'
import type { TaskManager, ToolCallSubmission } from './task-manager.js'

const DEFAULT_MAX_CONCURRENCY = 4
const MAX_TIMER_DELAY_MS = 2 ** 31 - 1
const STATE_RELOAD_TIMEOUT = 30_000

const BACKGROUND_TASKS_STATE_KEY = 'strands.backgroundTasks'

/** Executes and persists approved tool calls as in-process background tasks. @internal */
export class InProcessTaskManager implements TaskManager<ToolCallSubmission> {
  private readonly _agent: Agent
  private readonly _config: {
    readonly maxConcurrency: number
    readonly timeout: number
    readonly waitForCompletion: boolean
  }
  private _engine: InProcessTaskEngine<ToolTaskDescriptor, ToolResultBlockData, InterruptStateData>
  private readonly _records = new Map<string, LiveBackgroundTask>()
  private readonly _invocationStates = new Map<string, InvocationState>()
  private readonly _deliveryStates = new Map<string, 'pending' | 'ready' | 'delivered'>()
  private readonly _telemetry = new BackgroundTaskTelemetry()
  private readonly _attempts = new Map<string, number>()
  private readonly _executionStartedAt = new Map<string, number>()
  private readonly _resumedTasks = new Set<string>()
  private readonly _delivering = new Set<string>()
  private readonly _deliveryWaiters = new Set<() => void>()
  private _reload: Promise<void> | undefined

  /**
   * Creates an in-process background task manager.
   *
   * @param agent - Agent whose registered tools execute background work.
   * @param config - Execution limits and invocation waiting behavior.
   */
  constructor(agent: Agent, config: InProcessTaskManagerConfig = {}) {
    this._agent = agent
    this._config = {
      maxConcurrency: config.maxConcurrency ?? DEFAULT_MAX_CONCURRENCY,
      timeout: config.timeout ?? Infinity,
      waitForCompletion: config.waitForCompletion !== false,
    }
    this._engine = this._createEngine()
  }

  private _createEngine(): InProcessTaskEngine<ToolTaskDescriptor, ToolResultBlockData, InterruptStateData> {
    return new InProcessTaskEngine({
      maxConcurrency: this._config.maxConcurrency,
      timeout: this._config.timeout,
      execute: (context) => this._executeToolTask(context),
      onTaskUpdated: (record): void => {
        this._recordTaskUpdate(record)
        this._records.set(record.taskId, record)
        this._deliveryStates.set(record.taskId, deliveryStateFor(record, this._deliveryStates.get(record.taskId)))
        this._persistAppState()
      },
    })
  }

  /** {@inheritDoc TaskManager.initialize} */
  async initialize(): Promise<void> {
    this._initializeEngine(this._engine)
  }

  /** {@inheritDoc TaskManager.registerHooks} */
  registerHooks(): void {
    this._agent.addHook(BeforeModelCallEvent, (event) => this._onBeforeModelCall(event))
    this._agent.addHook(AfterModelCallEvent, (event) => this._onAfterModelCall(event))
    this._agent.addHook(AfterInvocationEvent, (event) => this._onAfterInvocation(event), {
      order: HookOrder.SDK_FIRST,
    })
  }

  private _initializeEngine(
    engine: InProcessTaskEngine<ToolTaskDescriptor, ToolResultBlockData, InterruptStateData>
  ): void {
    this._loadAppState()
    engine.initialize([...this._records.values()])
    const records = engine.list()
    const taskIdsToUnpin = new Set(
      records.filter((record) => this._deliveryStates.get(record.taskId) === 'delivered').map((record) => record.taskId)
    )
    this._pruneDelivered(engine, [...taskIdsToUnpin])
    unpinBackgroundDeliveries(this._agent.messages, taskIdsToUnpin)
    const readyTaskIds = new Set(
      records.filter((record) => this._deliveryStates.get(record.taskId) === 'ready').map((record) => record.taskId)
    )
    removeBackgroundDeliveries(this._agent.messages, readyTaskIds)
  }

  private _loadAppState(): void {
    this._records.clear()
    this._deliveryStates.clear()
    this._attempts.clear()
    this._executionStartedAt.clear()
    this._resumedTasks.clear()
    this._invocationStates.clear()
    const value = this._agent.appState.get(BACKGROUND_TASKS_STATE_KEY)
    if (value === undefined) return
    if (!isObject(value)) throw new Error(`${BACKGROUND_TASKS_STATE_KEY} must be an object`)

    const taskIdByIdempotencyKey = new Map<string, string>()
    for (const [taskId, storedValue] of Object.entries(value)) {
      if (!isObject(storedValue) || !('record' in storedValue) || !('deliveryState' in storedValue)) {
        throw new Error(`${BACKGROUND_TASKS_STATE_KEY}.${taskId} is invalid`)
      }
      validateStoredTask(storedValue.record)
      if (storedValue.record.taskId !== taskId) {
        throw new Error(`${BACKGROUND_TASKS_STATE_KEY}.${taskId}.record.taskId must match its map key`)
      }
      if (
        storedValue.deliveryState !== 'pending' &&
        storedValue.deliveryState !== 'ready' &&
        storedValue.deliveryState !== 'delivered'
      ) {
        throw new Error(`${BACKGROUND_TASKS_STATE_KEY}.${taskId}.deliveryState is invalid`)
      }
      const idempotencyKey = storedValue.record.idempotencyKey
      if (idempotencyKey !== undefined) {
        const existingTaskId = taskIdByIdempotencyKey.get(idempotencyKey)
        if (existingTaskId) {
          throw new Error(
            `${BACKGROUND_TASKS_STATE_KEY}.${taskId}.record.idempotencyKey duplicates task '${existingTaskId}'`
          )
        }
        taskIdByIdempotencyKey.set(idempotencyKey, taskId)
      }
      this._records.set(taskId, restoreBackgroundTask(storedValue.record))
      this._deliveryStates.set(taskId, storedValue.deliveryState)
    }
  }

  private _persistAppState(): void {
    if (this._records.size === 0) {
      this._agent.appState.delete(BACKGROUND_TASKS_STATE_KEY)
      return
    }
    const tasks: Record<
      string,
      {
        record: StoredBackgroundTask
        deliveryState: 'pending' | 'ready' | 'delivered'
      }
    > = {}
    for (const record of this._records.values()) {
      tasks[record.taskId] = {
        record: storeBackgroundTask(record),
        deliveryState: this._deliveryStates.get(record.taskId) ?? deliveryStateFor(record),
      }
    }
    this._agent.appState.set(BACKGROUND_TASKS_STATE_KEY, tasks)
  }

  private _pruneDelivered(
    engine: InProcessTaskEngine<ToolTaskDescriptor, ToolResultBlockData, InterruptStateData>,
    taskIds: readonly string[]
  ): void {
    if (taskIds.length === 0) return
    for (const taskId of taskIds) {
      const record = this._records.get(taskId)
      if (!record) throw new Error(`Background task '${taskId}' was not found`)
      const state = this._deliveryStates.get(taskId)
      if (state !== 'delivered' || !isInProcessTaskTerminalStatus(record.status)) {
        throw new Error(`Background task '${taskId}' has not been consumed`)
      }
      engine.remove(taskId)
      this._records.delete(taskId)
      this._deliveryStates.delete(taskId)
      this._clearTaskBookkeeping(taskId, record.descriptor.executionStateId)
    }
    this._persistAppState()
  }

  /** {@inheritDoc TaskManager.appStateLoaded} */
  appStateLoaded(): void {
    const restoredTasks = this._agent.appState.get(BACKGROUND_TASKS_STATE_KEY)
    const previousReload = this._reload
    const reload = previousReload
      ? previousReload.then(
          () => this._reloadFromAppState(restoredTasks),
          () => this._reloadFromAppState(restoredTasks)
        )
      : this._reloadFromAppState(restoredTasks)
    this._reload = reload
    void reload.then(
      () => {
        if (this._reload === reload) this._reload = undefined
      },
      () => undefined
    )
  }

  private async _reloadFromAppState(restoredTasks: JSONValue | undefined): Promise<void> {
    const deadline = Date.now() + STATE_RELOAD_TIMEOUT
    await this._waitForDeliveries(STATE_RELOAD_TIMEOUT)
    await this._engine.shutdown({ timeout: Math.max(1, deadline - Date.now()) })
    this._delivering.clear()

    if (restoredTasks === undefined) {
      this._agent.appState.delete(BACKGROUND_TASKS_STATE_KEY)
    } else {
      this._agent.appState.set(BACKGROUND_TASKS_STATE_KEY, restoredTasks)
    }

    const engine = this._createEngine()
    try {
      this._initializeEngine(engine)
      this._engine = engine
    } catch (error) {
      await engine.shutdown({ timeout: 1_000 }).catch(() => undefined)
      throw error
    }
  }

  private async _waitForReload(cancelSignal?: AbortSignal): Promise<void> {
    let reload = this._reload
    while (reload) {
      try {
        await waitWithSignal(reload, cancelSignal)
      } catch (error) {
        if (cancelSignal?.aborted || reload === this._reload) throw error
      }
      if (reload === this._reload || this._reload === undefined) return
      reload = this._reload
    }
  }

  /** {@inheritDoc TaskManager.submitTask} */
  async submitTask(submission: ToolCallSubmission): Promise<BackgroundTask> {
    if (this._reload) await this._waitForReload()
    const originTraceContext =
      submission.originSpanContext && isSpanContextValid(submission.originSpanContext)
        ? {
            traceId: submission.originSpanContext.traceId,
            spanId: submission.originSpanContext.spanId,
            traceFlags: submission.originSpanContext.traceFlags,
            ...(submission.originSpanContext.isRemote !== undefined && {
              isRemote: submission.originSpanContext.isRemote,
            }),
          }
        : undefined
    const input = deepCopyWithValidation(submission.input, 'background task input')
    const executionStateId = globalThis.crypto.randomUUID()
    this._invocationStates.set(executionStateId, submission.invocationState)
    const descriptor: ToolTaskDescriptor = {
      originalToolUseId: submission.originalToolUseId,
      toolName: submission.toolName,
      input,
      executionStateId,
      ...(originTraceContext && { originTraceContext }),
    }
    try {
      const stored = this._engine.submit({
        descriptor,
        idempotencyKey: JSON.stringify([submission.passId, descriptor.originalToolUseId]),
      })
      if (stored.descriptor.executionStateId !== executionStateId) {
        this._invocationStates.delete(executionStateId)
      }
      return toBackgroundTask(stored)
    } catch (error) {
      this._invocationStates.delete(executionStateId)
      throw error
    }
  }

  /** {@inheritDoc TaskManager.getTask} */
  async getTask(taskId: string): Promise<BackgroundTask | undefined> {
    if (this._reload) await this._waitForReload()
    const record = this._engine.get(taskId)
    return record ? toBackgroundTask(record) : undefined
  }

  /** {@inheritDoc TaskManager.listTasks} */
  async listTasks(): Promise<readonly BackgroundTask[]> {
    if (this._reload) await this._waitForReload()
    return this._engine.list().map(toBackgroundTask)
  }

  /** {@inheritDoc TaskManager.cancelTask} */
  async cancelTask(taskId: string): Promise<BackgroundTask> {
    if (this._reload) await this._waitForReload()
    const previousStatus = this._engine.get(taskId)?.status
    const task = this._engine.cancel(taskId, { reason: 'Cancellation requested' })
    if (task.status === 'cancelled' && previousStatus !== 'cancelled') {
      this._telemetry.recordCancellation(task.descriptor.toolName)
    }
    return toBackgroundTask(task)
  }

  /** {@inheritDoc TaskManager.waitForTasks} */
  async waitForTasks(options?: { readonly timeout?: number }): Promise<void> {
    const timeout = options?.timeout
    if (timeout !== undefined && (!Number.isSafeInteger(timeout) || timeout <= 0 || timeout > MAX_TIMER_DELAY_MS)) {
      throw new TypeError(
        `wait timeout must be a positive integer no greater than ${MAX_TIMER_DELAY_MS}, got ${timeout}`
      )
    }
    const timeoutController = timeout === undefined ? undefined : new AbortController()
    const timeoutTimer =
      timeoutController && timeout !== undefined
        ? setTimeout(() => timeoutController.abort(new BackgroundTasksTimeoutError(timeout)), timeout)
        : undefined
    try {
      if (this._reload) await this._waitForReload(timeoutController?.signal)
      await this._engine.waitForIdle(timeoutController ? { cancelSignal: timeoutController.signal } : undefined)
    } finally {
      if (timeoutTimer) clearTimeout(timeoutTimer)
    }
  }

  private _resumeTask(taskId: string, responses: readonly InterruptResponse[]): BackgroundTask {
    const current = this._engine.get(taskId)
    if (!current) throw new BackgroundTaskNotFoundError(taskId)
    if (current.status !== 'paused') {
      if (current.state && responsesAlreadyApplied(current.state, responses)) {
        return toBackgroundTask(current)
      }
      throw new Error(`Background task '${taskId}' cannot transition: status is '${current.status}', not 'paused'`)
    }
    return toBackgroundTask(
      this._engine.resume(taskId, (state) => {
        const interruptState = InterruptState.fromJSON(state)
        const knownIds = new Set(Object.keys(interruptState.interrupts))
        for (const response of responses) {
          if (!knownIds.has(response.interruptId)) {
            throw new Error(
              `Background task '${taskId}' cannot transition: unknown interrupt '${response.interruptId}'`
            )
          }
        }
        interruptState.resume(
          responses.map(
            (response) =>
              new InterruptResponseContent({
                interruptId: response.interruptId,
                response: response.response,
              })
          )
        )
        return {
          state: interruptState.toJSON(),
          ready: interruptState.getUnansweredInterrupts().length === 0,
        }
      })
    )
  }

  private async _onAfterModelCall(event: AfterModelCallEvent): Promise<void> {
    if (this._reload) await this._waitForReload()
    if (event.error || !event.stopData) return
    this._throwPausedInterrupts()
  }

  private async _onAfterInvocation(event: AfterInvocationEvent): Promise<void> {
    if (this._reload) await this._waitForReload()
    const stopReason = getInvocationResult(event)?.stopReason
    if (!this._config.waitForCompletion || !stopReason || stopReason === 'cancelled' || stopReason === 'interrupt') {
      return
    }

    const cannotContinue =
      stopReason === 'checkpoint' ||
      stopReason === 'limitTurns' ||
      stopReason === 'limitOutputTokens' ||
      stopReason === 'limitTotalTokens'
    await this._waitForTaskResult(cannotContinue)
    if (this._agent.cancelSignal.aborted) return
    if (this._surfacePausedInterrupt(event)) return
    if (cannotContinue) return
    this._deliverReady(event)
  }

  private _surfacePausedInterrupt(event: AfterInvocationEvent): boolean {
    const interrupts = this._pausedInterrupts()
    if (interrupts.length === 0) return false

    const interruptState = getEventInterruptState(event)
    const result = getInvocationResult(event)
    if (!interruptState || !result) {
      throw new Error('Background interrupt cannot be surfaced without an active invocation result')
    }
    for (const interrupt of interrupts) {
      interruptState.registerInterrupt(interrupt)
    }
    interruptState.activate()
    setInvocationResult(
      event,
      new AgentResult({
        ...result,
        stopReason: 'interrupt',
        interrupts: interruptState.getUnansweredInterrupts(),
      })
    )
    return true
  }

  private async _onBeforeModelCall(event: BeforeModelCallEvent): Promise<void> {
    if (this._reload) await this._waitForReload()
    this._routeInterruptResponses(event)
    this._throwPausedInterrupts()
    this._deliverReady(event)
  }

  private _deliverReady(event: BeforeModelCallEvent | AfterInvocationEvent): void {
    let continuationRegistered = false
    let taskIds: string[] = []
    try {
      const engine = this._engine
      const records = engine
        .list()
        .filter(
          (record) =>
            isInProcessTaskTerminalStatus(record.status) &&
            this._deliveryStates.get(record.taskId) === 'ready' &&
            !this._delivering.has(record.taskId)
        )
      if (records.length === 0) return
      taskIds = records.map((record) => record.taskId)
      for (const taskId of taskIds) this._delivering.add(taskId)

      const deliveries = records.map((record) => renderBackgroundDelivery(record))

      continuations.add(event, {
        args: deliveries.flat(),
        requireModelRequest: true,
        onConsumed: (modelRequestMessages) => {
          try {
            if (!modelRequestMessages) {
              throw new Error('Background task delivery requires a successful model request')
            }
            records.forEach((record, index) => {
              assertDeliveryConsumed(record.taskId, deliveries[index]!, modelRequestMessages)
            })
            if (this._engine === engine) {
              for (const taskId of taskIds) this._deliveryStates.set(taskId, 'delivered')
              this._persistAppState()
              this._pruneDelivered(engine, taskIds)
              unpinBackgroundDeliveries(this._agent.messages, new Set(taskIds))
            }
          } catch (error) {
            removeBackgroundDeliveries(this._agent.messages, new Set(taskIds))
            throw error
          } finally {
            this._finishDelivery(taskIds)
          }
        },
        onRejected: () => {
          removeBackgroundDeliveries(this._agent.messages, new Set(taskIds))
          this._finishDelivery(taskIds)
        },
      })
      continuationRegistered = true
    } finally {
      if (!continuationRegistered) {
        this._finishDelivery(taskIds)
      }
    }
  }

  private _finishDelivery(taskIds: readonly string[]): void {
    for (const taskId of taskIds) this._delivering.delete(taskId)
    for (const resolve of this._deliveryWaiters) resolve()
    this._deliveryWaiters.clear()
  }

  private async _executeToolTask(
    context: InProcessTaskExecutionContext<ToolTaskDescriptor, InterruptStateData>
  ): Promise<TaskExecutionOutcome<ToolResultBlockData, InterruptStateData>> {
    const descriptor = context.descriptor
    if (!this._agent.toolRegistry.get(descriptor.toolName)) {
      return {
        status: 'failed',
        failure: {
          type: 'recoveryError',
          message: `Tool '${descriptor.toolName}' is not registered on Agent '${this._agent.id}'`,
        },
      }
    }
    const originSpanContext = descriptor.originTraceContext
    const invocationState = this._invocationStates.get(descriptor.executionStateId)
    if (!invocationState) {
      return {
        status: 'failed',
        failure: {
          type: 'recoveryError',
          message: 'Background task live invocation state is unavailable',
        },
      }
    }
    const attempt = this._attempts.get(context.taskId) ?? 1
    let outcome: ToolResultBlock | { readonly interruptState: InterruptStateData }
    try {
      outcome = await executeDetachedTool(this._agent, {
        toolUseBlock: new ToolUseBlock({
          name: descriptor.toolName,
          toolUseId: descriptor.originalToolUseId,
          input: descriptor.input,
        }),
        invocationState,
        cancelSignal: context.cancelSignal,
        ...(context.state && { interruptState: context.state }),
        background: {
          taskId: context.taskId,
          attempt,
          attemptId: globalThis.crypto.randomUUID(),
          executionId: globalThis.crypto.randomUUID(),
        },
        classifyAbort: () => (this._engine.get(context.taskId)?.failure?.type === 'timeout' ? 'failed' : 'cancelled'),
        ...(originSpanContext && { originSpanContext }),
      })
    } catch (error) {
      return {
        status: 'failed',
        failure: {
          type: 'executionError',
          message: normalizeError(error).message,
        },
      }
    }
    if ('interruptState' in outcome) {
      return {
        status: 'paused',
        state: outcome.interruptState,
      }
    }
    const serialized = outcome.toJSON().toolResult
    if (outcome.status === 'error') {
      return {
        status: 'failed',
        failure: {
          type: 'toolError',
          message:
            outcome.error?.message ??
            outcome.content.find((content): content is TextBlock => content instanceof TextBlock)?.text ??
            'Tool returned an error without a message',
        },
        result: serialized,
      }
    }
    return {
      status: 'completed',
      result: serialized,
    }
  }

  private _recordTaskUpdate(record: LiveBackgroundTask): void {
    const previous = this._records.get(record.taskId)
    const toolName = record.descriptor.toolName
    if (!previous && record.status === 'queued') {
      this._telemetry.recordAdmission(toolName)
      return
    }
    if (previous?.status === 'paused' && record.status === 'queued') {
      this._resumedTasks.add(record.taskId)
      return
    }
    if (record.status === 'working' && previous?.status !== 'working') {
      const attempt = (this._attempts.get(record.taskId) ?? 0) + 1
      this._attempts.set(record.taskId, attempt)
      this._executionStartedAt.set(record.taskId, Date.now())
      this._telemetry.recordExecutionStarted({
        toolName,
        attempt,
        resumed: this._resumedTasks.delete(record.taskId),
        queueDuration: Math.max(0, Date.now() - Date.parse(previous?.updatedAt ?? record.createdAt)),
      })
      return
    }
    if (previous?.status === 'working' && record.status !== 'working') {
      const startedAt = this._executionStartedAt.get(record.taskId) ?? Date.now()
      this._executionStartedAt.delete(record.taskId)
      this._telemetry.recordExecutionFinished({
        toolName,
        outcome: executionOutcome(record),
        duration: Date.now() - startedAt,
      })
    }
    if (record.failure && previous?.failure !== record.failure) {
      this._telemetry.recordFailure(toolName, record.failure.type)
    }
    if (
      (record.status === 'completed' || record.status === 'failed' || record.status === 'cancelled') &&
      !isInProcessTaskTerminalStatus(previous?.status ?? 'queued')
    ) {
      this._telemetry.recordTerminal(toolName, record.status)
    }
    if (isInProcessTaskTerminalStatus(record.status)) {
      this._clearTaskBookkeeping(record.taskId, record.descriptor.executionStateId)
    }
  }

  private async _waitForDeliveries(timeout: number): Promise<void> {
    const deadline = Date.now() + timeout
    while (this._delivering.size > 0) {
      const remaining = deadline - Date.now()
      if (remaining <= 0) throw new Error(`Background Tasks state reload timed out after ${timeout}ms`)
      let timer: ReturnType<typeof setTimeout> | undefined
      await Promise.race([
        new Promise<void>((resolve) => {
          this._deliveryWaiters.add(resolve)
        }),
        new Promise<never>((_, reject) => {
          timer = setTimeout(
            () => reject(new Error(`Background Tasks state reload timed out after ${timeout}ms`)),
            remaining
          )
        }),
      ]).finally(() => {
        if (timer) clearTimeout(timer)
      })
    }
  }

  private async _waitForTaskResult(waitForAll: boolean): Promise<void> {
    const cancelSignal = this._agent.cancelSignal
    while (!cancelSignal.aborted) {
      const tasks = this._engine.list()
      if (tasks.some((task) => task.status === 'paused')) return
      if (
        !waitForAll &&
        tasks.some(
          (task) =>
            isInProcessTaskTerminalStatus(task.status) &&
            this._deliveryStates.get(task.taskId) === 'ready' &&
            !this._delivering.has(task.taskId)
        )
      ) {
        return
      }

      const pending = tasks.filter((task) => task.status === 'queued' || task.status === 'working')
      if (pending.length === 0) return

      const observationController = new AbortController()
      const observationSignal = AbortSignal.any([cancelSignal, observationController.signal])
      try {
        await Promise.race(pending.map((task) => this._engine.wait(task.taskId, { cancelSignal: observationSignal })))
      } catch (error) {
        if (!cancelSignal.aborted) throw error
      } finally {
        observationController.abort()
      }
    }
  }

  private _routeInterruptResponses(event: BeforeModelCallEvent): void {
    const interruptState = getEventInterruptState(event)
    const responseContents = interruptState?.resumeResponses
    if (!interruptState || !responseContents || responseContents.length === 0) return

    const paused = this._engine.list().filter((task) => task.status === 'paused')
    const taskByInterruptId = new Map<string, string>()
    for (const task of paused) {
      for (const interrupt of toBackgroundTask(task).interrupts ?? []) {
        const owner = taskByInterruptId.get(interrupt.id)
        if (owner && owner !== task.taskId) {
          throw new Error(`Background interrupt '${interrupt.id}' is ambiguous across paused tasks`)
        }
        taskByInterruptId.set(interrupt.id, task.taskId)
      }
    }

    const responsesByTask = new Map<string, InterruptResponse[]>()
    for (const content of responseContents) {
      const response = content.interruptResponse
      const taskId = taskByInterruptId.get(response.interruptId)
      if (!taskId) continue
      const responses = responsesByTask.get(taskId) ?? []
      responses.push(response)
      responsesByTask.set(taskId, responses)
    }
    if (responsesByTask.size === 0) return

    for (const [taskId, responses] of responsesByTask) {
      this._resumeTask(taskId, responses)
    }

    const foregroundInterruptIds = Object.keys(interruptState.interrupts)
    if (
      foregroundInterruptIds.length > 0 &&
      foregroundInterruptIds.every((interruptId) => taskByInterruptId.has(interruptId))
    ) {
      interruptState.deactivate()
    }
  }

  private _throwPausedInterrupts(): void {
    const interrupts = this._pausedInterrupts()
    if (interrupts.length > 0) throw new InterruptError(interrupts)
  }

  private _pausedInterrupts(): Interrupt[] {
    return this._engine
      .list()
      .filter((task) => task.status === 'paused')
      .flatMap((task) => toBackgroundTask(task).interrupts ?? [])
  }

  private _clearTaskBookkeeping(taskId: string, executionStateId?: string): void {
    this._attempts.delete(taskId)
    this._executionStartedAt.delete(taskId)
    this._resumedTasks.delete(taskId)
    if (executionStateId) this._invocationStates.delete(executionStateId)
  }
}

async function waitWithSignal<Value>(promise: Promise<Value>, cancelSignal?: AbortSignal): Promise<Value> {
  if (!cancelSignal) return promise
  if (cancelSignal.aborted) {
    throw cancelSignal.reason ?? new DOMException('Observation aborted', 'AbortError')
  }
  return new Promise<Value>((resolve, reject) => {
    const onAbort = (): void => {
      reject(cancelSignal.reason ?? new DOMException('Observation aborted', 'AbortError'))
    }
    cancelSignal.addEventListener('abort', onAbort, { once: true })
    void promise.then(
      (value) => {
        cancelSignal.removeEventListener('abort', onAbort)
        resolve(value)
      },
      (error: unknown) => {
        cancelSignal.removeEventListener('abort', onAbort)
        reject(error)
      }
    )
  })
}

function deliveryStateFor(
  record: LiveBackgroundTask,
  current?: 'pending' | 'ready' | 'delivered'
): 'pending' | 'ready' | 'delivered' {
  if (!isInProcessTaskTerminalStatus(record.status)) return 'pending'
  return current === 'delivered' ? 'delivered' : 'ready'
}

function isObject(value: unknown): value is { [key: string]: JSONValue } {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function responsesAlreadyApplied(state: InterruptStateData, responses: readonly InterruptResponse[]): boolean {
  return responses.every(
    (response) =>
      stableStringify(state.interrupts[response.interruptId]?.response) === stableStringify(response.response)
  )
}

function executionOutcome(
  record: LiveBackgroundTask
): 'completed' | 'paused' | 'failed' | 'cancelled' | 'executionError' {
  if (record.status === 'failed' && record.failure?.type === 'executionError') return 'executionError'
  if (
    record.status === 'completed' ||
    record.status === 'paused' ||
    record.status === 'failed' ||
    record.status === 'cancelled'
  ) {
    return record.status
  }
  return 'executionError'
}
