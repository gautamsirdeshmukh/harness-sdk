import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'

import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { Agent } from '../../agent/agent.js'
import { pinMessage } from '../../conversation-manager/compression/pin-message.js'
import { AfterModelCallEvent } from '../../hooks/events.js'
import { InvokeModelStage } from '../../middleware/stages.js'
import type { StreamOptions } from '../../models/model.js'
import type { ModelStreamEvent } from '../../models/streaming.js'
import { tool } from '../../tools/tool-factory.js'
import { Message, TextBlock, ToolResultBlock, ToolUseBlock } from '../../types/messages.js'
import { ContextInjector } from '../../vended-plugins/context-injector/plugin.js'
import { assertDeliveryConsumed, renderBackgroundDelivery, unpinBackgroundDeliveries } from '../delivery.js'
import { InProcessTaskManager } from '../in-process-task-manager.js'
import type { StoredBackgroundTask } from '../record.js'

const BACKGROUND_TASKS_STATE_KEY = 'strands.backgroundTasks'

class RecordingModel extends MockMessageModel {
  readonly requests: Message[][] = []

  override async *stream(messages: Message[], options?: StreamOptions): AsyncGenerator<ModelStreamEvent> {
    this.requests.push(messages.map((message) => message.clone()))
    yield* super.stream(messages, options)
  }
}

function deferred<Value>(): { readonly promise: Promise<Value>; resolve(value: Value): void } {
  let resolve!: (value: Value) => void
  const promise = new Promise<Value>((promiseResolve) => {
    resolve = promiseResolve
  })
  return { promise, resolve }
}

function terminalRecord(): StoredBackgroundTask {
  return {
    taskId: 'task-1',
    idempotencyKey: JSON.stringify(['pass-1', 'tool-use-1']),
    descriptor: {
      originalToolUseId: 'tool-use-1',
      toolName: 'work',
    },
    status: 'completed',
    result: new ToolResultBlock({
      toolUseId: 'tool-use-1',
      status: 'success',
      content: [new TextBlock('stored result')],
    }).toJSON().toolResult,
    createdAt: '2026-08-18T12:00:00.000Z',
    updatedAt: '2026-08-18T12:00:01.000Z',
  }
}

function seedTask(
  agent: Agent,
  record = terminalRecord(),
  deliveryState: 'pending' | 'ready' | 'delivered' = 'ready'
): void {
  agent.appState.set(BACKGROUND_TASKS_STATE_KEY, {
    [record.taskId]: { record, deliveryState },
  })
}

function readTaskState(agent: Agent): unknown {
  return agent.appState.get(BACKGROUND_TASKS_STATE_KEY)
}

async function createDeliveryFixture(model: RecordingModel): Promise<{
  readonly agent: Agent
  readonly manager: InProcessTaskManager
}> {
  const agent = new Agent({
    id: 'delivery-agent',
    model,
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
  const manager = new InProcessTaskManager(agent, { timeout: 5_000 })
  manager.registerHooks()
  seedTask(agent)
  await manager.initialize()
  return { agent, manager }
}

function deliveryCount(messages: readonly Message[]): number {
  return messages.filter((message) =>
    message.content.some((block) => block.type === 'toolUseBlock' && block.name === 'strands_background_task_result')
  ).length
}

describe('background task delivery', () => {
  it('ignores appended content while detecting canonical delivery changes', () => {
    const record = terminalRecord()
    const messages = renderBackgroundDelivery(record).map((message) => Message.fromJSON(message.toJSON()))
    messages[0]!.content.unshift(
      new ToolUseBlock({
        name: 'strands_background_task_result',
        toolUseId: 'unrelated-task',
        input: { unrelated: true },
      })
    )
    messages[1]!.content.push(new TextBlock('injected reminder'))

    expect(() => assertDeliveryConsumed(record.taskId, renderBackgroundDelivery(record), messages)).not.toThrow()

    messages[1]!.content[0] = new ToolResultBlock({
      toolUseId: record.taskId,
      status: 'success',
      content: [new TextBlock('altered result')],
    })
    expect(() => assertDeliveryConsumed(record.taskId, renderBackgroundDelivery(record), messages)).toThrow(
      "Background task delivery 'task-1' did not match its authoritative record"
    )

    const alteredToolUse = renderBackgroundDelivery(record).map((message) => Message.fromJSON(message.toJSON()))
    alteredToolUse[0]!.content[0] = new ToolUseBlock({
      name: 'strands_background_task_result',
      toolUseId: record.taskId,
      input: { altered: true },
    })
    expect(() => assertDeliveryConsumed(record.taskId, renderBackgroundDelivery(record), alteredToolUse)).toThrow()
  })

  it('consumes and prunes a delivery augmented by an every-turn ContextInjector', async () => {
    const model = new RecordingModel().addTurn({ type: 'textBlock', text: 'delivered' })
    const { agent } = await createDeliveryFixture(model)
    const renderContent = vi.fn(async () => 'TODOS REMINDER')
    new ContextInjector({ trigger: 'everyTurn', renderContent }).initAgent(agent)

    // Guards the provider-request delivery check used by https://github.com/strands-agents/stan/issues/16.
    await agent.invoke('deliver')

    expect(model.requests[0]!.at(-1)!.content).toEqual([
      expect.objectContaining({ type: 'toolResultBlock', toolUseId: 'task-1' }),
      expect.objectContaining({ type: 'textBlock', text: '\n\nTODOS REMINDER' }),
    ])
    expect(readTaskState(agent)).toBeUndefined()
    expect(deliveryCount(agent.messages)).toBe(1)
  })

  it('removes an unconsumed delivery folded into prior assistant history, then retries it', async () => {
    const model = new RecordingModel()
      .addTurn({ type: 'textBlock', text: 'initial' })
      .addTurn({ type: 'textBlock', text: 'recovered' })
    const completion = deferred<string>()
    const agent = new Agent({
      id: 'folded-delivery-agent',
      model,
      tools: [
        tool({
          name: 'work',
          description: 'Perform work.',
          inputSchema: z.object({ value: z.string() }),
          callback: () => completion.promise,
        }),
      ],
      printer: false,
    })
    const manager = new InProcessTaskManager(agent, { timeout: 5_000 })
    manager.registerHooks()
    await manager.initialize()
    const admitted = await manager.submitTask({
      toolName: 'work',
      originalToolUseId: 'tool-use-1',
      input: { value: 'stored' },
      invocationState: {},
      passId: 'pass-1',
    })
    agent.addHook(AfterModelCallEvent, () => {
      completion.resolve('stored result')
    })
    const cleanup = agent.addMiddleware(InvokeModelStage, async function* (context, next) {
      const hasDelivery = context.messages.some((message) =>
        message.content.some((block) => block.type === 'toolResultBlock' && block.toolUseId === admitted.taskId)
      )
      if (!hasDelivery) return yield* next(context)
      return {
        result: {
          message: new Message({ role: 'assistant', content: [new TextBlock('cached')] }),
          stopReason: 'endTurn' as const,
        },
      }
    })

    await expect(agent.invoke('deliver')).rejects.toThrow(
      'Continuation consumption requires a successful model request'
    )
    expect(readTaskState(agent)).toEqual({
      [admitted.taskId]: expect.objectContaining({ deliveryState: 'ready' }),
    })
    expect(deliveryCount(agent.messages)).toBe(0)
    expect(agent.messages.at(-1)?.content).toEqual([new TextBlock('initial')])
    cleanup()

    await agent.invoke('retry')

    expect(readTaskState(agent)).toBeUndefined()
    expect(deliveryCount(agent.messages)).toBe(1)
  })

  it('recovers a staged ready delivery instead of pruning it', async () => {
    const model = new RecordingModel().addTurn({ type: 'textBlock', text: 'recovered' })
    const record = terminalRecord()
    const agent = new Agent({
      id: 'recovery-agent',
      model,
      messages: [...renderBackgroundDelivery(record)],
      printer: false,
    })
    seedTask(agent, record)
    const manager = new InProcessTaskManager(agent, { timeout: 5_000 })
    manager.registerHooks()

    await manager.initialize()

    expect(deliveryCount(agent.messages)).toBe(0)
    await expect(manager.getTask(record.taskId)).resolves.toEqual(expect.objectContaining({ status: 'completed' }))

    await agent.invoke('retry')

    expect(readTaskState(agent)).toBeUndefined()
    expect(deliveryCount(agent.messages)).toBe(1)
  })

  it('prunes a persisted delivery only when it is marked consumed', async () => {
    const record = terminalRecord()
    const agent = new Agent({
      id: 'consumed-agent',
      model: new RecordingModel(),
      messages: [...renderBackgroundDelivery(record)],
      printer: false,
    })
    seedTask(agent, record, 'delivered')
    const manager = new InProcessTaskManager(agent, { timeout: 5_000 })

    await manager.initialize()

    expect(readTaskState(agent)).toBeUndefined()
    expect(agent.messages[0]?.metadata?.custom?.pinned).not.toBe(true)
    expect(agent.messages[1]?.metadata?.custom?.pinned).not.toBe(true)
  })

  it('preserves an existing pin when delivery is folded into unrelated assistant content', () => {
    const [deliveryAssistant, deliveryUser] = renderBackgroundDelivery(terminalRecord())
    const messages = [
      new Message({
        role: 'assistant',
        content: [new TextBlock('keep pinned'), ...deliveryAssistant.content],
      }),
      deliveryUser,
    ]
    pinMessage(messages, 0)

    unpinBackgroundDeliveries(messages, new Set(['task-1']))

    expect(messages[0]?.metadata?.custom?.pinned).toBe(true)
    expect(messages[1]?.metadata?.custom?.pinned).not.toBe(true)
  })
})
