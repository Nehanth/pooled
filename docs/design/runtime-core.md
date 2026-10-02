# Shared generation and pipeline (#22, first slice)

The browser and `packages/room-node` now use the same single-attempt decode loop,
prefill, worker frame processing and activation transport orchestration. The factories
use each caller's existing mutable `ai` state. They neither construct engines nor load
weights, connect peers, admit members or own a request queue.

```js
const pipeline = createPipeline({ state, transport, options, hooks, checkpointStore, getRoomCode });
const generate = createGenerator({ state, pipeline, options, hooks });
const result = await generate(ids, { stop, sample, maxNew, signal, onToken });
```

These are internal extraction seams, not a stable SDK. `state.engine` is the currently
loaded engine; the caller owns its replacement. The state retains positions, cached
token prefixes, pending control, chain membership, lap waiters and the worker queue.
`tests/headless/runtime.mjs` shows deterministic engines connected through the real
binary transport, runnable under Node or Deno without browser globals.

## Requests and stream

`ids` contains the complete prompt. `stop` is the required set of end-token IDs.
`sample` accepts logits or GPU candidates; its default follows the room's sampling
setting. `maxNew` is bounded by the engine's context capacity. `onToken(id, drafted)`
fires synchronously in order: 0/false means sampled, 1 draft-head/model, 2 prompt lookup.
End tokens are not emitted. Synchronous `onToken` errors propagate as generation failures.

The browser retains `onStatus`, `pin`/`pinTag`, `state.abort` and sampled pending tokens
for Continue. Its result is `{ tokens, reason, reused, prefilled, count, tps, acc,
copied, tPre, tDecode, preFrames, stats, capped }`; `reason` is `stop`, `max`, `ctx` or
`abort`. Failures during prompt preparation or decoding, including synchronous token
callbacks, invalidate reusable state and checkpoints before being rethrown. Readiness
checks and GPU sampler setup precede this cleanup boundary; completion observers,
the final checkpoint save and the node's `finish` hook run after it. Errors there
propagate without this cache invalidation, preserving the callers' existing behaviour.

The node selects `options.profile: "node"` on both factories. This preserves its
signal-only cancellation, numeric sampled-token flag, per-request `spec: false`,
request-local lookup history, existing errors and lack of browser Continue state.
Both callers retain dense prompt-lookup speculation and check peer capability each
step, including the node path added after #278. The browser alone retains its optional
draft model and lookup history across attempts; node lookup history is request-local.

## Checkpoint and completion policies

Browser checkpoint helpers stay in the pipeline: pinned-prefix retention, optional
`checkpointStore` disk copies, restore/prune and persistence after the save's frame
is sent. `getRoomCode()` supplies the current disk namespace. Engines that advertise
`hostCkpt: false` retain their existing checkpoint opt-out.

The node keeps its own `CkptIndex`, multiple pins, turn boundaries and capability checks.
It injects its checkpoint methods into the generator alongside the shared pipeline.
Its `hooks.preparePrompt(ids, request)` restores a reusable prefix, prefills through
the requested pin/turn boundaries and returns logits, reuse metadata and the bounded
generation limit. Each prefill uses the shared pipeline. `hooks.finish(result, ids,
aborted)` saves the end checkpoint, formats node statistics and emits its existing
`prefill` event. The node result retains `from` and `pinned` instead of browser
`preFrames`. These adapters preserve caller policy around the common decode loop.

## Transport and recovery

`transport.sendHidden(id, frame)` delivers activation frames; `sendTo(id, message)`
delivers worker errors. `chainRtt()` supplies the existing timeout estimate. The browser
retains its sender checks and passes frames to `pipeline.handleFrame(from, frame)`.
That helper queues work only for workers and accepts returned host laps only from
the current chain tail. The node retains its own dispatch, worker queue and sender/role
checks, calling the shared `workerFrame` and `lapDone` through its methods. Connection
failures call `pipeline.failWaiters(error)`. Hooks retain caller diagnostics, GPU wake behavior
and frame accounting. The node continues draining at most two queued slot drops
per frame; rollback/save/drop/reset/load order and binary framing remain unchanged.
`options.prefillWindow` defaults to 6. Finite values are floored and clamped to at
least 1; non-finite values use the default. The node supplies its wide-prefill
environment override through `hooks.prefillFrame`, keeping process globals outside
the shared module.

The generator runs one attempt. Browser `roomGenerate`/`roomRecover` and node
`generate` still compose `resumableGenerate` with their existing locking, readiness
and recovery policies. Membership, model loading, UI, Code and API adapters stay
with their callers. This extraction does not fix duplicate sessions or phone sleep.

Validation covers direct checkpoint imports, generation contracts, headless binary
transport, node checkpoint policy and the existing room/browser tests. Hardware and
physical-phone evidence must name the tested revision; older prototype results do
not validate this revision.
