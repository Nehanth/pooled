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

## The 27B and the 35B MoE: a context ladder, the longest at full speed by default (2026-10-16)

Rooms no longer need `?ctx=` for a long context on the two big hybrids. Each has a ladder of lengths checked correct on it
(`room/models.js` `CTX` `ladder`), and a room opens at the longest step its memory holds (`room/ctxpick.js`
`roomCtxPlan`, called by the picker, Start, the room node behind `pooled host` / `pooled serve` / the OpenClaw plugin, and
`pooled host`'s screen, so they all agree):

| Model | Ladder | Picker's context row | Old default (`def`) | Why each length is on it |
|---|---|---|---|---|
| Qwen3.8 27B | 64K, 32K, 16K, 8K | 8K, 16K, 32K, 64K | 16K | needle found at 64K (2026-09-30) and 16K / 32K (2026-10-01); 8K is the same engine as 16K / 32K (below). 128K not checked: left out |
| Qwen3.6 35B MoE | 128K, 64K, 32K, 16K, 8K | 8K, 32K, 64K, 128K | 32K | needle found at 32K, 64K, 96K, 128K (2026-09-30); `test_moe.js` MATCHes llama.cpp in a 131072-position engine (below); 8K / 16K are the 32K engine |

The engine at 32K positions or fewer is one engine: the prefill split length is 256 for any `maxSeq` up to 32K
(`engine/qwen35.js` `faSplit`), and decode's splits follow the filled length (`attn_dec` `decSplits`), so a 8K or 16K room
computes what a 32K room does on any prompt that fits.

**The trade-off.** A longer window costs memory (f16 K+V: 64 KB per token on the 27B, 20 KB on the MoE, split over the
devices: the 27B needs ~17.0 GB at 16K and ~20 GB at 64K; the MoE ~21.0 GB at 32K and ~22.9 GB at 128K by the planner's
bytes), not speed: decode reads only the filled positions. Measured on the GB10 (Deno `tests/bench_ctx.js`, FILLS=1K,8K,
only `CTX` changed):

| Model | maxSeq | 1K fill: prefill / plain / spec tok/s | 8K fill: prefill / plain / spec tok/s | tokens |
|---|---|---|---|---|
| MoE | 32768 | 938 / 37.8 / 51.7 | 1159 / 35.6 / 50.6 | |
| MoE | 131072 | 921 / 37.6 / 50.4 | 1155 / 36.3 / 51.1 | identical to 32768 |
| 27B | 16384 | 247 / 10.58 / 19.9 | 277 / 10.23 / 21.3 | |
| 27B | 65536 | 246 / 10.59 / 19.9 | 277 / 10.21 / 21.1 | identical to 16384 |

`CTX=131072 tests/test_moe.js`: 3 of 3 MATCH llama.cpp, spec = plain (decode 60-61 tok/s; 55-58 at the usual 512 in the
run just before).

Memory can still cost speed in a room, two ways, so the default goes past `def` only when neither happens
(`roomCtxPlan` `fastAt`, judged on the deal by memory): (1) the longer context leaves no room for the experts, and a device
offloads them (each token then copies experts from RAM or disk); (2) it needs one more device holding layers than at
`def` (a phone, or one more network hop every token). Such a room opens at `def`, and the longer lengths are one click away
in the picker (Room settings changes it later; the model loads again). A room short of memory even for `def` walks the
ladder down (to 8K) and says so; one that started at `def` before still starts there or longer.

Example rooms (the picker's need per choice: `ctxOptions`, the GPU memory these devices need at that length):

| Room | Model | 8K | 16K | 32K | 64K | 128K | Opens at |
|---|---|---|---|---|---|---|---|
| Mac 16 + desktop 7 + iPhone 1 (24 GB) | MoE | 20.9 | | 21.4 | 22.1 | 23.5 | 128K, the phone holds nothing |
| Mac 17 + desktop 6 + iPhone 1 (24 GB) | MoE | | | | fits | fits with the phone | 64K (128K would put layers on the phone) |
| Mac 14 + desktop 8 + iPhone 1 (23 GB) | MoE | 21.4 | | 21.8 | 22.5 | needs 0.7 GB more | 64K |
| RTX 12 GB + 48 GB RAM (`pooled host --here`) | MoE | 6.2 | | 6.7 | 7.3 | 8.6 | 32K: experts in RAM at every length, so the default |
| Mac 16 + desktop 8 (24 GB) | 27B | 16.5 | 17.0 | 18.0 | 20.1 | | 64K |
| a 16 GB laptop alone | 27B | short | short | short | short | | short (for 8K) |

`?ctx=N` and `--ctx N` still ask for an exact length, tried alone. Tests: `tests/unit/ctxpick_test.js` (the picker and
Start agree for these rooms at every choice; the default rule), `cli/test/hostui_test.mjs` (the screen, the room node and
the page agree), `packages/room-node/test/roomnode_test.mjs`.

## Qwen3.8 Flash in rooms: long context (rooms long-context PR 5, 2026-10-13)

Rooms run Flash with QSA and f16 K / V, at up to **64K** (65,536 tokens), the longest context checked against llama.cpp
in a room. The picker offers 8K, 32K and 64K, each with the GPU memory it needs in that room and about how long a full
paste takes to read; a room opens at the longest proven length its devices hold (`room/models.js` `FLASH_CTX_PROVEN`,
`room/ctxpick.js`).

Goldens, two room nodes on one GB10 (40 / 40 GB, layers 0-23 + 24-47, f16 K / V, batched prefill, the f16 wire),
`flash_gold.mjs pair CTX=131072` (`tests/golden/q38flash_llama_greedy.json`; past 8K a case passes against llama.cpp's
golden or any of its reruns, and a needle case needs the passphrase):

| case | prompt | verdict | prefill (first answer) | decode |
|---|---|---|---|---|
| two-sum / bash / eos-mid | 23-31 | MATCH | 0.8-1.0 s | 15-18 tok/s |
| hash-map | 23 | TIE at 49 | 0.6 s | 15.8 |
| ctx2k / ctx2k_b / ctx2k1 | 1,887-2,082 | TIE at 26 / 6 / 13 | 10-11 s | 12.1-12.4 |
| ctx4k | 3,867 | TIE at 6 | 18.9 s (204 tok/s) | 12.5 |
| ctx8k | 7,663 | TIE at 6 | 37.5 s (204 tok/s) | 13.0 |
| ctx16k | 15,718 | TIE at 3 | 78.5 s (200 tok/s) | 12.3 |
| ctx32k | 31,958 | TIE at 0 (llama.cpp's rerun 1) | 160 s (200 tok/s) | 12.7 |
| needle30k | 29,737 | MATCH (rerun 1), passphrase found | 150 s (198 tok/s) | 12.5 |
| ctx64k | 64,795 | TIE at 13 (rerun 1) | 323 s (201 tok/s) | 12.6 |
| needle120k | 119,198 | **passphrase found**; MISMATCH at step 0 (1.0 nat below llama.cpp's top) | 639 s (187 tok/s) | 12.1 |
| ctx128k | 129,936 | MISMATCH at 21 (1.04 nats below llama.cpp's top; the same answer in substance) | 697 s (187 tok/s) | 12.1 |

- The same pair in f32 K / V gives the same verdicts at 16K-32K and the same two 128K differences, at the same steps:
  they are not f16's. Not resolved yet (both engines keep rope angles in f32; the next check is a solo run, no wire).
- 8K with batching off (`POOLED_FLASH_BATCH=0`, one column at a time): 15.8-18.3 tok/s, ctx8k in 485 s. Batched and
  per-token ids are identical on ctx4k and ctx8k; on ctx2k they part at step 29, after both left the golden at its near
  tie at 26.
- 128K fits by memory: f16 K / V is 128 MiB per QSA layer at 131,072 (WebGPU's default binding), and the planner holds
  the demo room (a Mac tab lending 22 GB with the disk tier, a PC lending 12 GB) at 128K with the tab's disk-tier RAM at
  4 GB (`room/flashdisk.js` `TAB_RAM_GB`, was 3). It is left out until a run passes the rule.

The demo room on real hardware (2026-10-14, `flash_tab_gold.mjs demo`): the Mac tab (Chrome, lends 20 GB, the disk
tier with 4 GB of RAM, `maxStorageBufferBindingSize` 4 GiB) alone, then the PC joins over WiFi (`pooled join --gb 10
--ram 43 --disk`, RTX 5070, its 31 layers' experts in RAM). The room opens at 65,536 (f16 K / V); the PC loads it
(layers 17-47, 40 cached experts a layer).

| case | devices | verdict | prompt read in | decode |
|---|---|---|---|---|
| two-sum | Mac alone | MATCH | 3.0 s | 5.1 tok/s |
| ctx8k | Mac alone | TIE at 6 | 45.7 s (168 tok/s) | 3.5 |
| ctx16k | Mac alone | TIE at 3 | 99.2 s (158 tok/s) | 5.5 |
| ctx2k | Mac + PC | TIE at 6 | 15.7 s | 6.3 |
| ctx8k | Mac + PC | TIE at 6 | 35.9 s (213 tok/s) | 6.6 |
| ctx16k | Mac + PC | TIE at 3 | 55-63 s | 6.1-7.2 |
| ctx32k | Mac + PC | TIE at 0 (llama.cpp's rerun 1) | 82 s (390 tok/s) | 6.1 |

- A two-turn chat past 16K (`TURNS=ctx16k`, `?ckpt=2`): turn 2 (15,735 tokens) restored turn 1's checkpoint and read
  one token, in 0.2 s; both turns' ids pass.
- Changing the context in Room settings while the room runs (64K to 8K, `CTX_SWITCH`): online again on both devices in
  50 s; the next answers MATCH, at 9.7-10.2 tok/s.
- Decode is slower at a long context: Mac alone ~5 tok/s at 64K (7.8 at 2048), Mac + PC 6-7 (9.4-10 at 2048 or 8K). The
  context's K / V and its two checkpoint slots come out of each device's GPU expert cache (the Mac alone: 108 cached
  experts a layer at 2048, 71 at 64K), so more expert reads miss.
- The picker's "a full paste" estimate for a disk room uses the Mac's solo rate (`room/ctxpick.js` `FLASH_PREFILL_DISK`).

Memory per device follows `flashLayerCost` (#380): weights, the range's state at the context, two checkpoint slots of
it, and the QSA scores once. A room node keeps at most those two slots past 2048 (one answer, one pinned prefix), so a
long chat never outgrows what the deal reserved. A tab keeps `?ckpt=2` by default.

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
