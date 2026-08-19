import { AfterInvocationEvent, BeforeModelCallEvent } from '../hooks/events.js'
import { logger } from '../logging/logger.js'

import type { InvokeArgs, LocalAgent } from '../types/agent.js'
import type { Message, StopReason } from '../types/messages.js'

/**
 * One internal input contribution to an agent or model invocation.
 *
 * A consumption failure rejects every input in the same continuation batch.
 */
interface ContinuationInput {
  /** Input that normalizes to one or more complete messages. */
  readonly args: InvokeArgs
  /** Requires the successful provider request when consuming this input. */
  readonly requireModelRequest?: boolean
  /** Runs after the input is incorporated into a successful model request. */
  readonly onConsumed?: (modelRequestMessages?: readonly Message[]) => void | Promise<void>
  /** Runs when the input cannot be incorporated into a successful model request. */
  readonly onRejected?: (reason: unknown) => void | Promise<void>
}

interface ContinuationState {
  readonly inputs: ContinuationInput[]
  readonly messages?: readonly Message[]
}

const stateByEvent = new WeakMap<AfterInvocationEvent | BeforeModelCallEvent, ContinuationState>()
const deferredInputsByAgent = new WeakMap<LocalAgent, ContinuationInput[]>()

/**
 * Internal continuation operations used by the agent loop.
 *
 * @internal
 */
export const continuations = {
  add,
  combine,
  consume,
  prepare,
  reject,
}

/**
 * Adds an input contribution for an invocation event.
 *
 * @param event - Event that owns the continuation input.
 * @param input - Input and optional settlement callbacks to register.
 */
function add(event: AfterInvocationEvent | BeforeModelCallEvent, input: ContinuationInput): void {
  const state = stateByEvent.get(event) ?? { inputs: [] }
  state.inputs.push(input)
  stateByEvent.set(event, state)
}

/**
 * Normalizes the inputs registered for an event and prepares those that form complete message sequences.
 *
 * @param event - Event whose continuation inputs should be prepared.
 * @param normalizeInput - Converts invocation input into messages.
 * @param stopReason - Stop reason that determines whether inputs can continue now or must be deferred.
 * @returns The prepared messages, or `undefined` when no continuation is ready.
 */
async function prepare(
  event: AfterInvocationEvent | BeforeModelCallEvent,
  normalizeInput: (args: InvokeArgs) => Message[],
  stopReason?: StopReason
): Promise<readonly Message[] | undefined> {
  if (stopReason === 'interrupt') {
    const inputs = consumeInputs(event)
    if (inputs.length > 0) {
      deferredInputsByAgent.set(event.agent, [...(deferredInputsByAgent.get(event.agent) ?? []), ...inputs])
    }
    return undefined
  }
  if (stopReason !== undefined && stopReason !== 'endTurn' && stopReason !== 'stopSequence') return undefined

  const deferredInputs = event instanceof AfterInvocationEvent ? deferredInputsByAgent.get(event.agent) : undefined
  if (deferredInputs) deferredInputsByAgent.delete(event.agent)
  const inputs = [...(deferredInputs ?? []), ...(stateByEvent.get(event)?.inputs ?? [])]
  const acceptedInputs: ContinuationInput[] = []
  const messages: Message[] = []

  for (const input of inputs) {
    try {
      const normalized = normalizeInput(input.args)
      if (!isCompleteMessageInput(normalized)) {
        throw new TypeError('Continuation input must contain a complete message sequence')
      }
      messages.push(...normalized)
      acceptedInputs.push(input)
    } catch (error) {
      await notifyRejected(input, error)
    }
  }

  if (acceptedInputs.length === 0) {
    stateByEvent.delete(event)
    return undefined
  }

  stateByEvent.set(event, { inputs: acceptedInputs, messages })
  return messages
}

/**
 * Prepends prepared continuation messages to invocation input.
 *
 * @param event - Event whose prepared messages should be applied.
 * @param args - Invocation input to combine with the prepared messages.
 * @param normalizeInput - Converts invocation input into messages.
 * @returns The combined input, or the original input when no prepared messages can be applied.
 */
function combine(
  event: AfterInvocationEvent | BeforeModelCallEvent | undefined,
  args: InvokeArgs,
  normalizeInput: (args: InvokeArgs) => Message[]
): InvokeArgs {
  const messages = event ? stateByEvent.get(event)?.messages : undefined
  if (!messages) return args

  const publicMessages = normalizeInput(args)
  const emptyInput = Array.isArray(args) && args.length === 0
  if (publicMessages.length === 0 && !emptyInput) {
    return args
  }

  return [...messages, ...publicMessages]
}

/**
 * Marks prepared inputs as incorporated into a successful model request.
 *
 * @param event - Event whose prepared inputs were consumed.
 * @param modelRequestMessages - Messages sent to the model provider.
 * @returns A promise that resolves after consumption callbacks finish.
 */
async function consume(
  event: AfterInvocationEvent | BeforeModelCallEvent | undefined,
  modelRequestMessages?: readonly Message[]
): Promise<void> {
  if (!event || !stateByEvent.get(event)?.messages) return
  const inputs = consumeInputs(event)
  if (modelRequestMessages === undefined && inputs.some((input) => input.requireModelRequest === true)) {
    const error = new Error('Continuation consumption requires a successful model request')
    await rejectInputs(inputs, error)
    throw error
  }

  for (let index = 0; index < inputs.length; index++) {
    const input = inputs[index]!
    try {
      await input.onConsumed?.(modelRequestMessages)
    } catch (error) {
      await rejectInputs(inputs.slice(index + 1), error)
      throw error
    }
  }
}

/**
 * Rejects prepared inputs that cannot be incorporated into a successful model request.
 *
 * @param event - Event whose prepared inputs should be rejected.
 * @param reason - Reason the inputs could not be consumed.
 * @returns A promise that resolves after rejection callbacks finish.
 */
async function reject(event: AfterInvocationEvent | BeforeModelCallEvent | undefined, reason: unknown): Promise<void> {
  if (!event) return
  await rejectInputs(consumeInputs(event), reason)
}

function consumeInputs(event: AfterInvocationEvent | BeforeModelCallEvent): readonly ContinuationInput[] {
  const state = stateByEvent.get(event)
  stateByEvent.delete(event)
  return state?.inputs ?? []
}

function isCompleteMessageInput(messages: readonly Message[]): boolean {
  if (messages.length === 0 || messages.at(-1)?.role !== 'user') return false

  let pendingToolUseIds = new Set<string>()
  for (const message of messages) {
    if (pendingToolUseIds.size > 0 && message.role !== 'user') return false

    const nextToolUseIds = new Set<string>()
    for (const block of message.content) {
      if (block.type === 'toolUseBlock') {
        if (message.role !== 'assistant' || nextToolUseIds.has(block.toolUseId)) return false
        nextToolUseIds.add(block.toolUseId)
      } else if (block.type === 'toolResultBlock') {
        if (message.role !== 'user' || !pendingToolUseIds.delete(block.toolUseId)) return false
      }
    }

    if (message.role === 'user' && pendingToolUseIds.size > 0) return false
    pendingToolUseIds = nextToolUseIds
  }
  return pendingToolUseIds.size === 0
}

async function notifyRejected(input: ContinuationInput, reason: unknown): Promise<void> {
  try {
    await input.onRejected?.(reason)
  } catch (error) {
    logger.warn(`error=<${error}> | continuation rejection callback failed`)
  }
}

async function rejectInputs(inputs: readonly ContinuationInput[], reason: unknown): Promise<void> {
  for (const input of inputs) {
    await notifyRejected(input, reason)
  }
}
