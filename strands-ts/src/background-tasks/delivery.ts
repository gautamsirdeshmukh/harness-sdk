import type { JSONValue } from '../types/json.js'
import { Message, TextBlock, ToolResultBlock, ToolUseBlock } from '../types/messages.js'
import { unpinMessage } from '../conversation-manager/compression/pin-message.js'
import { rehydrateStoredToolResult, type StoredBackgroundTask } from './record.js'
import type { BackgroundTask } from './types.js'

/** Canonical synthetic tool name used to deliver a terminal background task result. @internal */
const BACKGROUND_RESULT_TOOL_NAME = 'strands_background_task_result'

/** Verifies that a task's canonical delivery blocks reached the provider request. @internal */
export function assertDeliveryConsumed(
  taskId: string,
  expected: readonly Message[],
  modelRequestMessages: readonly Message[]
): void {
  const candidates = findBackgroundDeliveryPairs(modelRequestMessages, taskId)
  if (candidates.length === 0) {
    throw new Error(`Background task delivery '${taskId}' was not present in the provider request`)
  }
  if (!candidates.some((actual) => deliveriesMatch(actual, expected, taskId))) {
    throw new Error(`Background task delivery '${taskId}' did not match its authoritative record`)
  }
}

/** Renders the pinned assistant and user messages that deliver a terminal task result. @internal */
export function renderBackgroundDelivery(record: StoredBackgroundTask): readonly [Message, Message] {
  if (record.status !== 'completed' && record.status !== 'failed' && record.status !== 'cancelled') {
    throw new Error(`Task '${record.taskId}' is not terminal`)
  }
  const failure = record.failure
  const input: JSONValue = {
    taskId: record.taskId,
    toolName: record.descriptor.toolName,
    status: record.status,
    ...(failure && {
      error: {
        type: failure.type,
        message: failure.message,
      },
    }),
  }
  const storedResult = record.result !== undefined ? rehydrateStoredToolResult(record.result) : undefined
  const resultContent = storedResult?.content ?? []
  const toolResult = new ToolResultBlock({
    toolUseId: record.taskId,
    status: record.status === 'completed' ? 'success' : 'error',
    content: [
      new TextBlock(
        renderTerminalHeader(
          record.taskId,
          record.descriptor.toolName,
          record.status,
          failure,
          storedResult !== undefined
        )
      ),
      ...resultContent,
    ],
  })

  return [
    new Message({
      role: 'assistant',
      content: [
        new ToolUseBlock({
          name: BACKGROUND_RESULT_TOOL_NAME,
          toolUseId: record.taskId,
          input,
        }),
      ],
      metadata: deliveryMetadata(),
    }),
    new Message({
      role: 'user',
      content: [toolResult],
      metadata: deliveryMetadata(),
    }),
  ]
}

/** Unpins successfully consumed task delivery messages. @internal */
export function unpinBackgroundDeliveries(messages: Message[], taskIds: ReadonlySet<string>): void {
  for (let index = 0; index < messages.length - 1; index++) {
    const taskId = backgroundDeliveryId(messages[index]!, messages[index + 1]!)
    if (!taskId || !taskIds.has(taskId)) continue
    if (isDedicatedDeliveryMessage(messages[index]!, taskId, 'assistant')) unpinMessage(messages, index)
    if (isDedicatedDeliveryMessage(messages[index + 1]!, taskId, 'user')) unpinMessage(messages, index + 1)
  }
}

/** Removes staged task delivery blocks that were not consumed. @internal */
export function removeBackgroundDeliveries(messages: Message[], taskIds: ReadonlySet<string>): void {
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index]!
    const content = message.content.filter((block) => {
      if (message.role === 'assistant' && block.type === 'toolUseBlock') {
        return !taskIds.has(block.toolUseId)
      }
      if (message.role === 'user' && block.type === 'toolResultBlock') {
        return !taskIds.has(block.toolUseId)
      }
      return true
    })
    if (content.length === message.content.length) continue
    if (content.length === 0) {
      messages.splice(index, 1)
      continue
    }
    messages[index] = new Message({
      role: message.role,
      content,
      trackingId: message.trackingId,
      ...(message.metadata !== undefined && { metadata: message.metadata }),
    })
  }
}

function renderTerminalHeader(
  taskId: string,
  toolName: string,
  status: Extract<BackgroundTask['status'], 'completed' | 'failed' | 'cancelled'>,
  error: { readonly type: string; readonly message: string } | undefined,
  hasResult: boolean
): string {
  if (status === 'completed') {
    return [
      'Background task completed.',
      '',
      `Task ID: ${taskId}`,
      `Tool: ${toolName}`,
      'Status: completed',
      '',
      'The final result follows.',
    ].join('\n')
  }
  if (status === 'failed') {
    if (!error) throw new Error(`Failed background task '${taskId}' has no failure detail`)
    return [
      'Background task failed.',
      '',
      `Task ID: ${taskId}`,
      `Tool: ${toolName}`,
      'Status: failed',
      `Error type: ${error.type}`,
      `Reason: ${error.message}`,
      '',
      hasResult ? 'The tool error follows.' : 'No result is available.',
    ].join('\n')
  }
  return [
    'Background task cancelled.',
    '',
    `Task ID: ${taskId}`,
    `Tool: ${toolName}`,
    'Status: cancelled',
    '',
    'The task was cancelled before producing a final result.',
  ].join('\n')
}

function findBackgroundDeliveryPairs(
  messages: readonly Message[],
  deliveryId: string
): readonly (readonly [Message, Message])[] {
  const pairs: [Message, Message][] = []
  for (let index = 0; index < messages.length - 1; index++) {
    const toolUseMessage = messages[index]!
    const toolResultMessage = messages[index + 1]!
    if (findDeliveryBlocks(toolUseMessage, toolResultMessage, deliveryId)) {
      pairs.push([toolUseMessage, toolResultMessage])
    }
  }
  return pairs
}

function backgroundDeliveryId(toolUseMessage: Message, toolResultMessage: Message): string | undefined {
  if (toolUseMessage.role !== 'assistant' || toolResultMessage.role !== 'user') return undefined
  const toolUse = toolUseMessage.content.find(
    (block) =>
      block.type === 'toolUseBlock' &&
      block.name === BACKGROUND_RESULT_TOOL_NAME &&
      toolResultMessage.content.some(
        (result) => result.type === 'toolResultBlock' && result.toolUseId === block.toolUseId
      )
  )
  return toolUse?.type === 'toolUseBlock' ? toolUse.toolUseId : undefined
}

function isDedicatedDeliveryMessage(message: Message, taskId: string, role: 'assistant' | 'user'): boolean {
  if (message.role !== role || message.content.length !== 1) return false
  const block = message.content[0]!
  return role === 'assistant'
    ? block.type === 'toolUseBlock' && block.name === BACKGROUND_RESULT_TOOL_NAME && block.toolUseId === taskId
    : block.type === 'toolResultBlock' && block.toolUseId === taskId
}

function deliveriesMatch(left: readonly Message[], right: readonly Message[], deliveryId: string): boolean {
  const project = (messages: readonly Message[]): unknown => {
    const [toolUseMessage, toolResultMessage] = messages
    const blocks =
      toolUseMessage && toolResultMessage
        ? findDeliveryBlocks(toolUseMessage, toolResultMessage, deliveryId)
        : undefined
    return blocks ? [blocks[0].toJSON(), blocks[1].toJSON()] : undefined
  }

  const leftDelivery = project(left)
  const rightDelivery = project(right)
  return (
    leftDelivery !== undefined &&
    rightDelivery !== undefined &&
    stableStringify(leftDelivery) === stableStringify(rightDelivery)
  )
}

function findDeliveryBlocks(
  toolUseMessage: Message,
  toolResultMessage: Message,
  deliveryId: string
): readonly [ToolUseBlock, ToolResultBlock] | undefined {
  if (toolUseMessage.role !== 'assistant' || toolResultMessage.role !== 'user') return undefined
  const toolUse = toolUseMessage.content.find(
    (block) =>
      block.type === 'toolUseBlock' && block.name === BACKGROUND_RESULT_TOOL_NAME && block.toolUseId === deliveryId
  )
  if (toolUse?.type !== 'toolUseBlock') return undefined
  const toolResult = toolResultMessage.content.find(
    (block) => block.type === 'toolResultBlock' && block.toolUseId === deliveryId
  )
  return toolResult?.type === 'toolResultBlock' ? [toolUse, toolResult] : undefined
}

/** Stable JSON serialization for internal background task comparisons. @internal */
export function stableStringify(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`
  if (value !== null && typeof value === 'object') {
    return `{${Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, nested]) => `${JSON.stringify(key)}:${stableStringify(nested)}`)
      .join(',')}}`
  }
  return JSON.stringify(value)
}

function deliveryMetadata(): NonNullable<Message['metadata']> {
  return {
    custom: {
      pinned: true,
    },
  }
}
