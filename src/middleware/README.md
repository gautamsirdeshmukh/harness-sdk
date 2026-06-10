# Middleware Design Decisions

## Hooks fire outside middleware

Hooks always fire outside the middleware chain. This means:

1. **Hooks always fire, even when middleware short-circuits.** If a middleware returns a cached result without calling `next()`, the Before/After hook pair for that stage still fires.

2. **Middleware retries are not observable to hooks.** If a middleware calls `next()` multiple times (e.g., retry-on-throttle), hooks see one invocation — not three. Internal retries are an implementation detail that should be observed using middleware.

### Ordering

First registered = outermost. Within each phase, handlers execute in registration order. Phases are fixed: all Input → all Output → all Wrap → terminal.

Phases partially solve the ordering problem by giving concrete semantics to each position (transform input, transform output, wrap execution). But since middleware is contributed by plugins, plugin registration order is still significant within a phase.

Long-term, if plugin ordering becomes a pain point, two approaches are on the table:

1. An explicit priority/order field on `addMiddleware` for fine-grained control.
2. Internal SDK phases (e.g., a retry phase) that the SDK uses but doesn't expose to customers.

Both are valid; which path(s) we take depends on whether we revisit "built-in SDK functionality built on plugins" as a pattern.

### Execution model

```
BeforeHookEvent          ← always fires
  → Input phase          ← transforms context
    → Output phase       ← wraps Wrap, transforms result on the way out
      → Wrap phase       ← full wrap, may retry/short-circuit
        → terminal       ← actual operation
AfterHookEvent           ← always fires (even on short-circuit)
```

### Functional style over mutation

Middleware is designed around passing transformed values forward rather than mutating shared state in place. Input handlers return a new context, Output handlers return a new result, and Wrap handlers pass a (potentially modified) context to `next()`. This contrasts with hooks, which mutate event properties in place.

We don't prevent mutation — the `agent` reference on the context is an escape hatch for advanced use cases — but the API nudges toward a functional style where each layer's effect is explicit in what it passes downstream.

### Consequences

- A rate-limiter middleware that short-circuits still triggers Before/After hooks — consumers monitoring call volume see the attempt even though the model was never called.
- A retry middleware that calls `next()` three times produces one hook pair, not three. If consumers need per-attempt observability, they register middleware (not hooks) at the same stage.
- Resume logic (calling `next()` with different args) is invisible to hooks by design. Hooks see the outermost call/result boundary.

## Telemetry records post-middleware state

Trace spans record the data _after_ middleware has transformed it — not the original pre-middleware input. For example, the model invoke span records the messages and system prompt that the model actually received, reflecting any transformations applied by `InvokeModelStage` middleware. This is intentional: spans should reflect what actually happened, not what was requested before middleware intervened.

## Middleware replaces hooks long-term

Middleware is intended to supersede hooks. The Input/Output phases already cover the same use cases as Before/After hook pairs but within a unified model. For one-off events without a before/after pair (e.g., MessageAdded), the migration path is less clear — but the goal is to consolidate on middleware where possible.

## Custom stages are internal

`createStage` is not exported from the public API. Only the three built-in stages are public.

## Middleware cannot resume interrupts

`AgentStreamStage` middleware cannot currently resume tool-level interrupts. Interrupt resolution (`_interruptState.resume()`) runs in `stream()`'s outer loop, outside the middleware chain. When a tool-level interrupt fires, `_stream` catches the `InterruptError` internally and returns a normal `AgentResult` with `stopReason: 'interrupt'` — the middleware sees the result but cannot re-enter the stream with interrupt responses.

To programmatically resume interrupts within a single invocation, use `AfterInvocationEvent.resume` in a hook. A future enhancement could add a resume mechanism to `AgentStreamResult` or `AgentStreamContext` so middleware can signal re-entry with interrupt responses.
