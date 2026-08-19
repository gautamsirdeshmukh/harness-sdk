import type { InterruptState } from '../interrupt.js'
import type { AgentResult } from '../types/agent.js'

import type { AfterInvocationEvent, BeforeModelCallEvent, BeforeToolCallEvent, BeforeToolsEvent } from './events.js'

type InterruptStateEvent = AfterInvocationEvent | BeforeModelCallEvent | BeforeToolCallEvent | BeforeToolsEvent

const resultByEvent = new WeakMap<AfterInvocationEvent, AgentResult>()
const interruptStateByEvent = new WeakMap<InterruptStateEvent, InterruptState>()

/** Returns the invocation result associated with an after-invocation event. @internal */
export function getInvocationResult(event: AfterInvocationEvent): AgentResult | undefined {
  return resultByEvent.get(event)
}

/** Associates an invocation result with an after-invocation event. @internal */
export function setInvocationResult(event: AfterInvocationEvent, result: AgentResult): void {
  resultByEvent.set(event, result)
}

/** Returns the interrupt state associated with an agent hook event. @internal */
export function getEventInterruptState(event: InterruptStateEvent): InterruptState | undefined {
  return interruptStateByEvent.get(event)
}

/** Associates an interrupt state with an agent hook event. @internal */
export function setEventInterruptState(event: InterruptStateEvent, interruptState: InterruptState): void {
  interruptStateByEvent.set(event, interruptState)
}
