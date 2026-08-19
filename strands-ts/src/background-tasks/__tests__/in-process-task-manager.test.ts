import { afterEach, describe, expect, it, vi } from 'vitest'
import { metrics as otelMetrics, type Meter as OtelMeter } from '@opentelemetry/api'
import { z } from 'zod'

import { MockMeter } from '../../__fixtures__/mock-meter.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { Agent } from '../../agent/agent.js'
import { InterruptResponseContent } from '../../types/interrupt.js'
import { TextBlock } from '../../types/messages.js'
import { tool } from '../../tools/tool-factory.js'
import type { ToolContext } from '../../tools/tool.js'
import { BackgroundTasksTimeoutError } from '../errors.js'
import { InProcessTaskManager } from '../in-process-task-manager.js'
import type { StoredBackgroundTask } from '../record.js'
import type { BackgroundTask, InProcessTaskManagerConfig } from '../types.js'

const BACKGROUND_TASKS_STATE_KEY = 'strands.backgroundTasks'

function deferred<Value>(): { readonly promise: Promise<Value>; resolve(value: Value): void } {
  let resolve!: (value: Value) => void
  const promise = new Promise<Value>((promiseResolve) => {
    resolve = promiseResolve
  })
  return { promise, resolve }
}

const managers = new Set<InProcessTaskManager>()

afterEach(async () => {
  await Promise.allSettled([...managers].map(cancelManagerTasks))
  managers.clear()
  vi.restoreAllMocks()
})

async function cancelManagerTasks(manager: InProcessTaskManager): Promise<void> {
  const tasks = await manager.listTasks()
  await Promise.allSettled(
    tasks
      .filter((task) => task.status === 'queued' || task.status === 'working' || task.status === 'paused')
      .map((task) => manager.cancelTask(task.taskId))
  )
  await manager.waitForTasks({ timeout: 1_000 })
}

async function createManager(
  callback: (input: { value: string }, context?: ToolContext) => string | Promise<string>,
  options: InProcessTaskManagerConfig = {}
): Promise<InProcessTaskManager> {
  return (await createManagerFixture(callback, options)).manager
}

async function createManagerFixture(
  callback: (input: { value: string }, context?: ToolContext) => string | Promise<string>,
  options: InProcessTaskManagerConfig = {},
  model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'unused' })
): Promise<{ readonly agent: Agent; readonly manager: InProcessTaskManager }> {
  const agent = new Agent({
    id: 'manager-test-agent',
    model,
    tools: [
      tool({
        name: 'work',
        description: 'Perform controlled test work.',
        inputSchema: z.object({ value: z.string() }),
        callback,
      }),
    ],
    printer: false,
  })
  const manager = new InProcessTaskManager(agent, options)
  managers.add(manager)
  await manager.initialize()
  return { agent, manager }
}

function admit(manager: InProcessTaskManager, value: string) {
  return manager.submitTask({
    toolName: 'work',
    originalToolUseId: `tool-use-${value}`,
    input: { value },
    invocationState: { request: value },
    passId: globalThis.crypto.randomUUID(),
  })
}

async function waitForTask(manager: InProcessTaskManager, taskId: string): Promise<BackgroundTask> {
  let task: BackgroundTask | undefined
  await vi.waitFor(async () => {
    task = await manager.getTask(taskId)
    expect(task?.status).toMatch(/^(paused|completed|failed|cancelled)$/)
  })
  return task!
}

describe('InProcessTaskManager', () => {
  describe('execution', () => {
    it('executes admitted tool work and exposes the minimal public snapshot', async () => {
      const manager = await createManager(({ value }) => value.toUpperCase())

      const admitted = await admit(manager, 'hello')
      const completed = await waitForTask(manager, admitted.taskId)

      expect(completed).toEqual({
        taskId: admitted.taskId,
        toolUseId: 'tool-use-hello',
        toolName: 'work',
        status: 'completed',
        createdAt: expect.any(String),
        updatedAt: expect.any(String),
        result: { content: [{ text: 'HELLO' }] },
      })
    })

    it('retains live invocation state by reference without persisting execution data', async () => {
      const finished = deferred<string>()
      const invocationState: ToolContext['invocationState'] = {
        requestId: 'request-1',
        callback: (): string => 'not serializable',
      }
      const { agent, manager } = await createManagerFixture(async (_input, context) => {
        expect(context!.invocationState).toBe(invocationState)
        context!.invocationState.completed = true
        return finished.promise
      })

      const admitted = await manager.submitTask({
        toolName: 'work',
        originalToolUseId: 'tool-use-live-state',
        input: { value: 'sensitive-input' },
        invocationState,
        passId: globalThis.crypto.randomUUID(),
      })
      expect(agent.appState.get(BACKGROUND_TASKS_STATE_KEY)).toEqual({
        [admitted.taskId]: {
          deliveryState: 'pending',
          record: expect.objectContaining({
            descriptor: {
              originalToolUseId: 'tool-use-live-state',
              toolName: 'work',
            },
          }),
        },
      })
      finished.resolve('done')
      await waitForTask(manager, admitted.taskId)
      expect(invocationState).toEqual({
        requestId: 'request-1',
        callback: expect.any(Function),
        completed: true,
      })
    })

    it('records an accepted cancellation as terminal exactly once', async () => {
      const mockMeter = new MockMeter()
      vi.spyOn(otelMetrics, 'getMeter').mockReturnValue(mockMeter as unknown as OtelMeter)
      const started = deferred<void>()
      const finished = deferred<string>()
      const manager = await createManager(async () => {
        started.resolve()
        return finished.promise
      })

      const admitted = await admit(manager, 'cancel')
      await started.promise

      await manager.cancelTask(admitted.taskId)

      const cancellations = mockMeter.getCounter('gen_ai.agent.background_task.cancellation.count')
      const terminal = mockMeter.getCounter('gen_ai.agent.background_task.terminal.count')
      expect(cancellations?.sum).toBe(1)
      expect(terminal?.dataPoints.map((point) => point.attributes?.['background_task.status'])).toEqual(['cancelled'])

      finished.resolve('late')
      await manager.waitForTasks({ timeout: 1_000 })

      await manager.cancelTask(admitted.taskId)
      expect(cancellations?.sum).toBe(1)
      expect(terminal?.dataPoints).toHaveLength(1)
    })

    it('times out physical execution and aborts its tool context', async () => {
      let abortReason: unknown
      const manager = await createManager(
        async (_input, context) => {
          await new Promise<void>((resolve) => {
            const onAbort = (): void => {
              abortReason = context!.cancelSignal.reason
              resolve()
            }
            if (context!.cancelSignal.aborted) onAbort()
            else context!.cancelSignal.addEventListener('abort', onAbort, { once: true })
          })
          return 'stopped'
        },
        { timeout: 25 }
      )

      const admitted = await admit(manager, 'timeout')
      const failed = await waitForTask(manager, admitted.taskId)

      expect({ abortReason, failed }).toEqual({
        abortReason: 'Timed out after 25ms',
        failed: expect.objectContaining({
          status: 'failed',
          error: {
            type: 'timeout',
            message: 'Timed out after 25ms',
          },
        }),
      })
    })

    it('times out waits without cancelling the task', async () => {
      const finished = deferred<string>()
      const manager = await createManager(() => finished.promise)
      const admitted = await admit(manager, 'wait')

      await expect(manager.waitForTasks({ timeout: 0 })).rejects.toThrow(
        'wait timeout must be a positive integer no greater than 2147483647, got 0'
      )
      await expect(manager.waitForTasks({ timeout: 2 ** 31 })).rejects.toThrow(
        'wait timeout must be a positive integer no greater than 2147483647, got 2147483648'
      )
      await expect(manager.waitForTasks({ timeout: 25 })).rejects.toMatchObject({
        name: 'BackgroundTasksTimeoutError',
        timeoutMs: 25,
      } satisfies Partial<BackgroundTasksTimeoutError>)
      await expect(manager.getTask(admitted.taskId)).resolves.toEqual(expect.objectContaining({ status: 'working' }))

      finished.resolve('done')
      await manager.waitForTasks({ timeout: 1_000 })
    })

    it('rejects restored tasks with duplicate idempotency keys', async () => {
      const baseRecord: StoredBackgroundTask = {
        taskId: 'restored-1',
        idempotencyKey: 'duplicate-key',
        descriptor: {
          originalToolUseId: 'restored-use-1',
          toolName: 'work',
        },
        status: 'failed',
        failure: {
          type: 'recoveryError',
          message: 'Recovered terminal task',
        },
        createdAt: '2026-08-18T12:00:00.000Z',
        updatedAt: '2026-08-18T12:00:01.000Z',
      }
      const duplicateRecord: StoredBackgroundTask = {
        ...baseRecord,
        taskId: 'restored-2',
        descriptor: {
          ...baseRecord.descriptor,
          originalToolUseId: 'restored-use-2',
        },
      }
      const agent = new Agent({
        model: new MockMessageModel(),
        tools: [
          tool({
            name: 'work',
            description: 'Perform work.',
            inputSchema: z.object({ value: z.string() }),
            callback: ({ value }) => value,
          }),
        ],
        printer: false,
      })
      agent.appState.set(BACKGROUND_TASKS_STATE_KEY, {
        [baseRecord.taskId]: { record: baseRecord, deliveryState: 'ready' },
        [duplicateRecord.taskId]: { record: duplicateRecord, deliveryState: 'ready' },
      })
      const manager = new InProcessTaskManager(agent)

      await expect(manager.initialize()).rejects.toThrow(
        "strands.backgroundTasks.restored-2.record.idempotencyKey duplicates task 'restored-1'"
      )
    })

    it('surfaces and resumes a detached tool interrupt', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'textBlock', text: 'waiting' })
        .addTurn({ type: 'textBlock', text: 'resuming' })
        .addTurn({ type: 'textBlock', text: 'delivered' })
      const { agent, manager } = await createManagerFixture(
        (_input, context) => {
          const response = context!.interrupt<string>({
            name: 'approve_work',
            reason: 'Approve work?',
          })
          return `approved:${response}`
        },
        {},
        model
      )
      manager.registerHooks()
      const admitted = await admit(manager, 'interrupt')
      await waitForTask(manager, admitted.taskId)

      const interrupted = await agent.invoke('surface')
      expect(interrupted).toMatchObject({
        stopReason: 'interrupt',
        interrupts: [expect.objectContaining({ name: 'approve_work', reason: 'Approve work?' })],
      })

      const completed = await agent.invoke([
        new InterruptResponseContent({
          interruptId: interrupted.interrupts![0]!.id,
          response: 'yes',
        }),
      ])

      expect(completed.stopReason).toBe('endTurn')
      expect(agent.messages).toContainEqual(
        expect.objectContaining({
          content: [
            expect.objectContaining({
              type: 'toolResultBlock',
              content: [expect.any(TextBlock), expect.objectContaining({ text: 'approved:yes' })],
            }),
          ],
        })
      )
      await expect(manager.getTask(admitted.taskId)).resolves.toBeUndefined()
    })

    it('emits admission, execution, and terminal telemetry from manager transitions', async () => {
      const mockMeter = new MockMeter()
      vi.spyOn(otelMetrics, 'getMeter').mockReturnValue(mockMeter as unknown as OtelMeter)
      const manager = await createManager(() => 'done')
      const admitted = await admit(manager, 'telemetry')

      await waitForTask(manager, admitted.taskId)

      expect(mockMeter.getCounter('gen_ai.agent.background_task.admitted.count')?.sum).toBe(1)
      expect(mockMeter.getCounter('gen_ai.agent.background_task.execution.count')?.sum).toBe(1)
      expect(mockMeter.getCounter('gen_ai.agent.background_task.terminal.count')?.dataPoints).toEqual([
        {
          value: 1,
          attributes: {
            'gen_ai.tool.name': 'work',
            'background_task.status': 'completed',
          },
        },
      ])
    })
  })
})
