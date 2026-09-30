# Long context and sessions

This is the engine side of Code mode (roadmap 30). It makes the Pooled engine (Qwen 3.8 27B and
Qwen 3.6 35B MoE across browser tabs) something a coding agent can sit on: long context, sessions
that resume without re-prefilling, a disk cache, and tool calls. The work was first done on a
branch called `tabby-kernel`; the research notes still use that name. Research behind the choices: [research/tabby-2026-09.md](research/tabby-2026-09.md);
earlier kernel work: [research/kernels-2026-09.md](research/kernels-2026-09.md).

Nothing here has been timed on a real GPU yet (the development machine only has SwiftShader).
Every change is either bit-identical to what it replaces or checked against it by a test, and
each has a switch for A/B timing on real hardware.

## Long context: f16 KV cache + split-K flash attention

| | before | now |
|---|---|---|
| KV cache | f32, 131 KB per token (whole 27B) | f16, 65.5 KB per token |
| attention | scores → softmax → out, whole row per head; the fast softmax stops at 2048 | `attn_flash` + `attn_combine`: splits of 256 positions, the 6 query heads that share a KV head read each K/V row once, online softmax, fixed-order merge |
| room context (27B) | 2048 tokens | 8192 tokens (0.54 GB of KV for the whole model, split over the devices) |

- Batched passes (prefill, verify) use `attn_flash_t2`: two columns per workgroup, each K/V row
  read once for both, same bits as the one-column kernel (`attnTile: false` to compare).
- `kv_store` writes the new K/V rows as packed f16 pairs in one dispatch (it replaces two
  buffer copies per column).
- The same kernels serve decode (1 column) and verify / prefill (many columns), and all their
  orders are fixed by absolute position, so speculative decoding stays exactly plain decoding and
  a split room stays exactly one device (engine_synth, room_synth).
- f16 K/V is not bit-identical to the old f32 path: `tests/e2e/flash_synth.mjs` compares the two
  (same greedy tokens; logit differences at 1e-6 of the logit range, also on a 2300-token prompt
  past the old 2048 limit, and at the 27B's shapes).
- Switch: `Qwen35Engine.create({ attnFlash: false })` restores the f32 path.
- int8 KV (`kvQ8: true`, room `?kv=q8`): one f32 scale per 32 values, `kv_store_q8` /
  `attn_flash_q8` from the same template as the f16 kernel. 36 KB per token for the whole 27B.
  Opt-in. The room host's `?kv=` decides for every device (sent with `ai-load`), and the layer
  deal counts int8 bytes. Timed on GB10 with the 35B MoE (bench log, 2026-09-29): the same 32
  greedy tokens as f16 at 1K, 8K and 32K and 0.35 GB of KV instead of 0.63 GB, but a 32K prompt
  prefills 3.5x slower, because tiled prefill attention is off with int8 KV. It stays off by
  default until tiled prefill attention supports it.
- Any `maxSeq` works; the split length grows past 32K so a head never has more than 128 splits.
  Memory at 32K: 2.1 GB of KV for the whole model in f16, 1.2 GB in int8 (q4 KV is not
  recommended: it hurts long documents and tool calls, research §3).
- 128K on the 35B MoE (`?ctx=131072`): each attention layer keeps K and V in one buffer each,
  bound whole, so the limit is the device's storage binding size, not memory. The MoE's K (or V)
  at 131072 positions in f16 is 2 KV heads x 256 x 2 bytes x 131072 = exactly 128 MiB, the binding
  size every WebGPU device supports (the MoE's stacked expert tensors already need more than that).
  The 27B is twice as wide: 256 MiB at 128K in f16, 128 MiB at 64K or at 128K in int8. The engine
  throws a readable error at create when a device cannot bind its KV buffer
  (`tests/unit/kv_bind_limit_test.js`), and the room host holds the context to what the smallest
  binding limit among its devices fits (`ctxForBinding` in `room/models.js`; each device reports
  `maxBindMB`, and a device that reports nothing counts as WebGPU's 128 MiB). The 27B's cap is 64K
  (128 MiB per buffer): its needle passes there; 128K on it is not checked yet. Splitting a layer's
  cache over several buffers was not needed; it would only matter for the 27B past 64K on a device
  that binds less than 256 MiB. Needle retrieval and speed at 32K..128K:
  `tests/needle_ctx.js` and docs/bench-log.md (2026-09-30).

## Sessions: save, restore, rewind, share a prefix

`engine/qwen35.js`:

- `exportState()` → `{ sig, pos, parts }`: this device's KV rows `[0, pos)`, DeltaNet states and
  conv windows, the draft block's KV, and the trunk hidden the next draft starts from. Read back
  one part at a time, so a long context never needs one huge mapping.
- `importState(state)`: refuses a state for other layers, another model or KV format
  (`stateSignature()`).
- `saveSlot(name)` / `loadSlot(name)` / `dropSlot(name)`: the same, as GPU-to-GPU copies, for
  switching sessions or rewinding an agent to an earlier turn.

A hybrid model cannot be cut back to an arbitrary position (the DeltaNet state is a running sum),
so reuse works at checkpoints: after the system prompt + tools, and after every finished turn.
`harness/prefix.js` (`PrefixIndex`) picks the longest checkpoint that is a prefix of the new
prompt; only the rest is prefilled. Coding agents resend ~96% of their input every turn
(research §3), which is what this saves.

`tests/e2e/state_synth.mjs`: restore after a reset and a different prompt, GPU slots, speculative
decoding after a restore, and an OPFS round trip all resume with bit-identical logits.

Size of a state on the 27B: ~150 MiB of DeltaNet state (fixed) + 65.5 KB per token of KV, for the
whole model; each device only holds its own layers' share.

### Across a split room (`?ckpt=N`, default 2)

After every answer the host saves the room's state on every device: a `sv` key rides on the next
frame down the chain, like reset and rollback do, so each device saves its own layers at exactly
the same point. A regenerate, an edited question or a branch then resumes from the longest saved
answer (`ld` key on the first frame) and prefills only what is new; the status line says
"(N reused)". Old checkpoints are dropped (`dp`) past N; a device rejoining with a fresh engine,
a re-deal or a failed answer clears them. Order on a device: rollback, save, drop, reset, load.
`tests/e2e/room_synth.mjs --regen --expect-reuse` checks that a regenerate over 3 devices resumes
from a checkpoint and repeats the greedy answer. `?ckpt=0` turns it off.

Each device also keeps a copy of its part on disk (OPFS, `room/ckpt-store.js`), so a reload does not
lose it. No new message: a device writes its copy when it applies `sv` and removes
it on `dp`. The host writes its own copy only once the `sv` has gone out on a frame, with the
checkpoint's token ids in the header, so it never indexes a slot the chain did not save. A copy is
named by room code, slot and a hash of the model and the engine's `stateSignature()` (layers, KV
format), and the header repeats them; a copy for other layers or another model, or in another file
format version, reads as missing. A new copy of a slot removes the old one first, so a failed write
(out of quota: the oldest copies of other rooms go first and the write is tried once more) leaves
the slot missing, never stale.

- A worker that reloads reads its copies back into GPU slots before it says it is ready. The host
  keeps the checkpoints whose save went out before the worker left and forgets the one still
  pending, so the next question resumes from the last answer the chain saved.
- A host that reloads resumes its room (the conversation is in localStorage, with the slot
  counter), reads its copies and their token ids back, and every device reads its own when the
  layers are dealt again. Slot numbers go on from the saved counter, so an old copy on a device is
  never taken for a new one.
- A device that loads its layers lists the slots it read back in its `ai-ready`, and the host
  forgets every checkpoint that device lacks (its copy never reached the disk, or the disk was
  full), so it never asks the chain to load a slot a device does not hold; the next question then
  resumes from an older checkpoint or prefills.
- When a device of the chain leaves, the host keeps its checkpoints (the other devices still hold
  them); a re-deal clears them and every device reads its copies back.
- The pinned system prompt checkpoint (below) goes to disk with `pin` in its header; a host that
  reloads indexes it pinned again (besides its newest `?ckpt=N` answer checkpoints), so answer saves
  still never evict it, and a worker reads back one more copy for it. A pinned one replaced by a new
  system prompt, or an answer checkpoint promoted to pinned, loses its disk copy like any dropped slot.
- `?ckptdisk=0` keeps checkpoints on the GPU only.

Code mode also pins the system prompt + tools (issue #73): `roomModel` passes its length as `pin`,
and when the caches do not hold it yet `roomGenerate` prefills up to there, saves a pinned
checkpoint (`ckptSave(true)`, riding the next frame like any save), then prefills the rest. The
pinned one is not counted in `?ckpt=N` and is never evicted by answer saves; a new system prompt
replaces it everywhere. When compaction rewrites old turns, the next step resumes there and
prefills only what follows. `engineModel` does the same on one engine with a GPU slot
(`pin: false` turns it off). Tests: `tests/unit/room_ckpt_test.js` ("pin: ..." scenarios, host and
workers in sync), `tests/unit/kv_reuse_test.js` (a 22-step session with compactions: every step
reuses everything the engine held, or the system prompt after a compaction).

### Several sessions on one engine

`harness/sessions.js` (`Sessions`): `switchTo(id)` parks the current conversation in a GPU slot
and brings `id` back from a GPU slot, from disk, or starts it fresh. Past `gpuSlots` parked
sessions, the least recently used go to OPFS (`exportSlot`, no switch needed); `persist()` saves
the active one; `close(id)` forgets one. Switching is exact: `tests/e2e/sessions_synth.mjs`
interleaves three sessions with one spare slot (one goes to disk and back) and compares every
logit with the same sessions decoded uninterrupted. This is time-sharing: one session computes at
a time. Batching several sessions through one pass is still on the list.

## Disk cache (OPFS)

`harness/statecache.js` (`StateCache`, `tokenKey`): states on the browser's origin-private file
system, keyed by SHA-256 of (model, this device's layer range and KV format, token ids). Temp file
+ rename, so a crash never leaves a half state under a real key; least recently used entries go
past a byte budget (8 GB default). Planned for rooms, not wired yet (only the e2e tests use it
today): every device stores its own part under the same key, and the host asks everyone to load it
and prefills instead if anyone is missing theirs.

Not done: SSD weight streaming. The research looked at Edge0 (the @SamuelZengML post: MoE experts
streamed from flash with a learned prefetcher) and SSD expert streaming on a Mac mini (the
@thefp4brain thread): both rely on MoE sparsity. A dense 27B touches every weight every token, so
streaming would pin decode to disk bandwidth; splitting the layers over peers is the answer here.

## Tool calls

`harness/tools.js`:
- `detectStyle(chatTemplate)`: "xml" (Qwen3-Coder / 3.5 / 3.8: `<function=…><parameter=…>`) or
  "json" (Qwen3 Hermes-style), read from the GGUF's own template.
- `toolsSystemPrompt(tools, { style, system })`, `toolResponses(results)` (one user turn of
  `<tool_response>` blocks), `renderCalls(calls, style)`.
- `ToolCallParser`: streaming; shows text as it comes but never half a tag, returns calls as they
  complete, accepts both formats, tolerates a missing `</parameter>` or `</tool_call>` at the end,
  converts XML parameter text to the schema's types.

`harness/constrain.js` (`ToolCallConstraint`): inside a tool call, masks the logits so the
function name and parameter names can only be declared ones; free text everywhere else. One
vocabulary scan per state, cached.

Unit tests: `tests/unit/tools_test.js`, `constrain_test.js`, `prefix_test.js`.

## The agent

- `harness/workspace.js`: the project the agent works on. `DirWorkspace` wraps a folder the user
  picks (File System Access API: the real files, nothing uploaded); `MemoryWorkspace` for tests.
  Paths cannot leave the root; `.git`, `node_modules` and build folders are skipped by search.
- `harness/codetools.js`: `list_dir`, `read_file` (numbered, paged at 400 lines), `search`
  (regex), `edit_file` (one exact match, with errors that say how to recover), `write_file`. The
  last two are marked `mutates`, so the agent asks before changing files.
- `harness/agent.js` (`Agent`): prompt with the tools, stream the answer through the tool-call
  parser, run the calls (approval hook for edits), send back `<tool_response>`s, repeat until an
  answer without calls or `maxSteps`. Knows nothing about GPUs: the model is any
  `generate({ system, turns })` stream.
- `harness/engine-model.js` (`engineModel`): that stream over a Qwen35Engine: renders the whole
  conversation each step but prefills only what the engine does not hold yet, keeps the model's
  own turns as their sampled ids, speculative decoding by default.

- `engineModel(engine, tok, { tools })` applies the tool-name constraint while sampling, on
  every position a speculative step checks too, so a call can only name declared tools and
  parameters.
- `Agent({ budget, count })`: past the token budget the oldest tool outputs are cut to a stub
  (oldest first, never the latest two, down to 60% so it does not cut every step). The stub
  (`stubResults`) and the fold line depend only on the turn, so a compacted turn renders the same
  on every later step; the `compacted` event's `at` is the first turn that changed.

Tests: `tests/unit/agent_test.js` (tools and loop with a scripted model),
`tests/e2e/agent_synth.mjs` (follow-up turns reuse the prefix and match a fresh engine; spec ==
plain; the loop on a real engine).

## Mixture of experts (Qwen3.5 / 3.6 MoE, `qwen35moe`)

The engine runs the MoE variant of the same hybrid architecture: every layer's FFN is 256 routed
experts (top-8) plus one shared expert, the rest (DeltaNet, gated attention, MTP) unchanged.
Target: Qwen3.6-35B-A3B (hidden 2048, 40 layers, expert and shared FFN 512, 16 q / 2 kv heads);
the same code reads Qwen3.5-122B-A10B. Architecture notes: `docs/research/moe-2026-09.md`.

- `engine/wgsl/moe.js`: `moe_router` (softmax over all experts, top-k, renormalised; ties to the
  lower index), `moe_gu_{q4,q8}` (the chosen expert's gate and up rows, SiLU, fused),
  `moe_dn_{q4,q8}`, `moe_combine` (fixed-order weighted sum + sigmoid-gated shared expert into
  the residual). Experts keep the ordinary Q4_0 / Q8_0 layout with nExp * dOut rows; the expert
  id is read on the GPU, so routing needs no readback.
- Every (token, slot) pair runs the same arithmetic in one-token and batched passes, so
  speculative decoding and batched prefill stay exact.
- Loader: `qwen35moe.*` keys are aliased to `qwen35.*`; 3-D expert tensors load as matrices.
- Files: the public Q8_0 GGUFs (~37 GB) or bartowski's "Q4_0" (~21 GB; a few tensors are Q4_1 /
  Q6_K and get re-quantised to Q8_0 on load). For two 24 GB Macs, the Q4_0 file.
- A stacked expert tensor is 134 MB (Q4_0) / 268 MB (Q8_0): the device must allow storage
  bindings that large (Macs do; phones are capped at 256 MB, fine for Q4_0). The engine says so
  when it does not; windowed bindings would lift it.

Tests: `tests/e2e/moe_kernels.mjs` (kernels vs a float64 reference, batched == single),
`tests/e2e/moe_synth.mjs` (each layer's MoE FFN in the engine vs a float64 reference from the
file's weights, ~4e-7; spec == plain; 7-token draft runs; batched prefill bit-identical to one
token at a time), `engine_synth.mjs --moe` (the whole suite, split == solo, on a MoE model).

Not done yet: expert-parallel splitting across devices (today a room splits MoE models by layers
like the dense ones), SSD spill of cold experts, grouping a batch's tokens by expert (prefill reads
each chosen expert once per token today), timing on real hardware.

## Also in this branch

- The kernels branch work: fused attention glue, fused DeltaNet delta + gated norm, batched
  attention for verify / prefill (all bit-identical), register-resident DeltaNet, Q8 prefill GEMM,
  replay rollback, batched draft-cache fill, the multi-device gibberish fixes.
- Prompt-lookup drafts search the whole context (16K window, was 4K): code edits copy from
  anywhere in the files the agent has read.

## Next

1. Time it on two Macs and a GB10: decode tok/s at 1K / 8K context, prefill tok/s, `?fuse=0`.
2. Room checkpoints on disk (landed, above): time a reload of one device on real hardware, and
   stream the copy part by part (today a device reads its whole part into memory to write it).
3. Stable prompt rendering for agents: done in part (#73: compaction is stable and the system
   prompt stays cached); a compaction still prefills everything after the system prompt once.
4. Several sessions at once: per-session KV / state slots batched through one pass (design:
   research/tabby-next-2026-09.md §2).
5. Pipelined speculative windows across devices (Mesh-LLM keeps several verifies in flight;
   design: research/tabby-next-2026-09.md §1).
6. Suffix / n-gram drafting chained after the MTP drafts (Mesh-LLM: +156% on code-copy loads).
7. q8 KV for 32K-64K; a tiled prefill attention kernel for long prompts.
