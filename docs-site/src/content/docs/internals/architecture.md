---
title: Architecture
description: How one model runs across several browser tabs, from one token end to end to where the weights and caches live.
eyebrow: How it works
sidebar:
  label: Architecture
  order: 1
---


Pooled has three parts:

- **The engine** (`engine/`) runs a model, or a slice of one, on a device's GPU through WebGPU. The kernels are our own WGSL.
- **The room** (`room.js`, `room/`) connects browsers over WebRTC and threads one generation through all of them.
- **Code mode** (`harness/`) is a coding agent that runs on the room's model.

The app has no build step and no framework. Nothing runs on a server except signaling, which only introduces devices to each other.

```txt
                 ┌──────────────────────── host browser ────────────────────────┐
 text ──tokenize──► embed ──► layers 0..k ──┐                      ┌──► final norm ──► LM head ──► sample ──► text
                                             │ hidden state (f16)   │
                                             ▼                      │
                                       peer A: layers k+1..m ──► peer B: layers m+1..N
```

## The host and the workers

One device is the **host**. It owns the conversation, the tokenizer, the embedding table, the LM head, the sampler and the draft block. It usually holds layers too.

The other devices are **workers**. Each holds one contiguous range of layers. Together they form a **chain** in layer order, and the last worker sends the result back to the host.

A device with no WebGPU, or one the room doesn't need, joins as an **ask-only guest**: it can ask questions and read answers, but holds no layers.

### Which device hosts

The host starts as the device that lends the most memory. The role then moves to a device of the same kind that copies GPU memory at least 1.5 times faster and lends at least half as much memory. A phone never takes the host role from a computer.

Each device measures its speed once at page load: it times a 64 MB buffer copy (best of 5 passes of 8 copies) and reports the result in its `hello`. The host role matters because the head, the sampler, the draft block and speculative rollback all sit on every token's critical path. In a GB10 + M5 Max room, moving the host role to the Mac made speculative decoding on the MoE 20 to 39% faster.

:::note
`?gbps=N` pins the measured speed on a device (`0` means unknown). The room harness uses it to force which device hosts.
:::

### How the layers are split

The split defaults to **For speed**. The fastest devices fill first, each up to the memory it lends, and devices that aren't needed join as guests. Until the first answer measures each device, a phone counts as 20 times slower than a computer, so a phone only gets layers the computers can't hold.

**By memory** splits in proportion to the memory each device lends. Phones still hold nothing when the computers can hold the whole model (`?phonelayers=1` deals them layers anyway).

Details and the URL options are on [How layers are split](/docs/rooms/split).

## One token, end to end

1. **Tokenize** (host CPU). Text becomes token ids, using the tokenizer stored in the GGUF file.
2. **Embed** (host). The id picks a row of the embedding table: the hidden state. It is 5,120 numbers wide on the 27B and 2,048 on the 35B MoE.
3. **Layers** (every device, in chain order). Each device runs its range of transformer blocks on its GPU. All of a device's work for one token is recorded into one command buffer and submitted once.
4. **Hop.** The hidden state is packed to f16 and sent to the next device over a WebRTC data channel. That is 10 KB per hop on the 27B and 4 KB on the MoE.
5. **Head** (host). A final RMSNorm, then the LM head over the whole vocabulary (248,320 rows), then sampling. Greedy and top-k sampling run on the GPU by default, so the host reads back a few bytes instead of the 1 MB logits vector.
6. **Draft** (host). The model's own draft layer proposes the next few tokens (see below).

The Qwen 3.8 27B has 64 layers: 48 Gated DeltaNet layers and 16 full-attention layers. A DeltaNet layer keeps a fixed-size recurrent state per head, so its memory does not grow with context. Only the attention layers keep a KV cache.

On the host, the head of the returned hidden state, the embedding of the picked token and the host's own layers run as one GPU submit with one readback (`?hostfuse=0` splits them for A/B tests).

## Prefill

The prompt is known upfront, so it goes through in batches. Each GPU pass handles up to 16 tokens as columns, and each network round carries 16 tokens. Full 16-column passes use a prefill GEMM that reads each weight block once for all columns (`engine/wgsl/gemm.js`).

Up to 6 prefill rounds are in flight around the chain at once, so the devices work as a pipeline. Prefill never runs the LM head: it only fills the KV caches and recurrent states.

Causality inside a batch holds because the recurrent kernels process columns strictly in order, and each column appends its K/V before it attends.

## Speculative decoding

Qwen 3.5 and later ship a draft block (`nextn`). Given the trunk's last hidden state and the token just sampled, it predicts the next token. Chained, it predicts several.

The host then checks `1 + K` tokens in **one** batched pass of the full model: one lap around the room instead of `1 + K` laps. Drafts come from the draft block or, when the answer is repeating text already in the context, from prompt lookup (up to 15 at once).

- A draft is accepted only while it matches the token the full model samples. The first mismatch ends acceptance and the full model's token is used.
- The DeltaNet state is rolled back by replaying the accepted columns from the state saved before the check. KV caches need no rollback: rejected positions are simply overwritten.
- The full model always decides, so the output is the same as plain decoding for any sampler.
- Draft depth is 3, 5 or 7. The room tries each and keeps the one with the best measured tokens per second.

This is what keeps a room usable on a real network: each lap pays the network latency once, and a good step accepts several tokens. See [Benchmarks](/docs/internals/benchmarks).

### Models without a draft block

The dense Qwen3 models (0.6B, 1.7B, 4B) have no draft block. In a room of two or more devices they still speculate, with prompt lookup: when the last few tokens of the answer already appear in the context, the tokens that followed them are checked in the same lap. Nothing repeats: the step is a plain lap. Code edits and quoted text gain the most. Ordinary chat rarely repeats itself and stays at plain speed.

- The check runs `1 + K` tokens (K up to 7) through every device's layers as one batched frame. Lookup starts at 3 drafts, one batched pass per device, and allows 7 only after a run was accepted in full.
- The batched matrix-vector kernels of the check use the same workgroup shape as single-token decoding, so each checked column gives **bit-identical** logits to a plain step. A verified token is exactly a decoded one, under any sampler (`tests/test_dense_spec.js`).
- Rollback is just a position: the rejected K/V rows are overwritten by the next frame.
- `?densespec=0` turns it off (plain laps). `?draft=qwen3-0.6b` (experimental) also loads Qwen3 0.6B whole on the host, and it drafts when lookup finds nothing.

## Memory and caching

| What | Where | Switch |
|---|---|---|
| Raw weight ranges | Browser Cache API. Each device range-fetches only its own tensors from Hugging Face, or takes them from another device in the room that already has them. | `?peerweights=0` stops taking ranges from other devices |
| Converted weights | OPFS. Tensors converted on the CPU (other quant types such as K-quants, Q4_1 and Q5_0 requantized to Q8, F16/BF16 to f32, the embedding repack) are kept, so a second load skips the conversion. | `?wcache=0` |
| KV cache | GPU. f16 by default; int8 on request. The host decides for every device. | `?kv=q8` |
| Checkpoints | GPU slots, plus a copy on disk (OPFS) per device, so a reload resumes. The last 2 answers by default. | `?ckpt=N`, `?ckptdisk=0` |

Q4_0 and Q8_0 matrices stream straight from the network into GPU buffers, repacked on the way into separate nibble and scale arrays so the kernels read them in contiguous stripes. RAM never holds the whole model.

Ranges taken from another device come in 64 KB parts, with at most 8 MB in flight, so a phone never buffers a whole range. Each cached range is stored with a size stamp (`x-swarm-len`) that is checked on read, which guards against a truncated entry.

More on checkpoints and the KV cache: [Long context and sessions](/docs/internals/sessions).

## Where the time goes

Measured on a GB10 (DGX Spark) with the 27B, using `benchmarks/bench_breakdown.js`, which skips one kernel family at a time and re-times:

| Part of one token | Time |
|---|---|
| All matrix-vector products (weights streamed at ~183 GB/s; the GPU's measured limit is 184) | 82 ms |
| Everything else (small kernels, encoding, submit, readback) | ~30 ms |

Decode is at the memory-bandwidth limit on this GPU, so speculative decoding is what raises tokens per second. Prefill is still the biggest gap to native llama.cpp.

:::caution[Older numbers]
This breakdown predates the prefill GEMM, flash attention and the fused kernels. It has not been re-measured since.
:::

## Code mode

The host types a request. The room's model runs a small agent loop that reads and edits files in a project, serves it on a virtual `localhost:5173` in a sandboxed frame, reads the preview's console errors and fixes them. A Code run holds the room's generation lock for all of its steps. See [Code mode](/docs/code).

## Files

### Engine (`engine/`)

| File | What it does |
|---|---|
| `engine.js` | Public entry point; re-exports the modules below |
| `qwen35.js` | `Qwen35Engine`: the hybrid DeltaNet + attention family, MoE, batched paths, speculation, sessions |
| `dense.js` | `DenseEngine`: dense models (Qwen3, SmolLM) |
| `preset.js` | The room's engine settings in one place, shared with the benchmarks |
| `wgsl/` | Kernels: `base.js` (shared), `coop.js` (cooperative GEMV), `qwen35.js` (DeltaNet and attention), `attn_tile.js` (tiled attention), `gemm.js`, `gemm_sgm.js`, `gemm_wide.js` (prefill GEMMs), `moe.js`, `moe_group.js` (experts), `dense.js` (dense models) |
| `gguf.js`, `safetensors.js` | File parsing, tokenizer extraction, requantizing and streaming upload |
| `tokenizer.js`, `sampling.js`, `topk.js`, `quant.js`, `autotune.js`, `selftest.js` | What their names say; `topk.js` is the host side of GPU sampling |

### Room (`room.js`, `room/`)

| File | What it does |
|---|---|
| `room.js` + `p2p.html` | The room: links, layer deal, downloads, the generation loop. Served at `/room` and `/r/<code>` |
| `transport.js`, `wire.js` | The hidden-state wire and frame packing; holds `PROTOCOL` |
| `signal.js`, `ice.js` | Signaling servers with fallback; STUN and the optional TURN relay |
| `liveness.js`, `resume.js`, `startstop.js` | Drop detection and lap timeouts; carrying an answer through a device that drops; stopping a failed start |
| `plan.js`, `pledge.js`, `gpuspeed.js`, `preflight.js` | The layer split and model host; how much a device may lend; GPU speed probe; can this browser hold layers |
| `gpuwake.js` | Keeps a worker's GPU clocked up while it waits for the next frame |
| `conversation.js`, `lookup.js`, `sampling.js` | The conversation and the exact tokens every device holds; prompt-lookup drafts; CPU sampling |
| `models.js` | The model list, memory needed and context per model |
| `weightcache.js`, `convertedcache.js`, `ckpt-store.js` | Cached weight ranges; converted-weights cache; checkpoints on disk |
| `api.js` | Answering API clients (`pooled serve`) on the host |
| `visibility.js`, `errors.js`, `working.js`, `markdown.js` | Who sees the chat; error wording; the "working" line; answer rendering |
| `compute.js`, `card.js`, `qr.js` | The Lend this device screen; the shareable room card; QR codes |
| `code.js`, `code-ui.js`, `code-export.js` | The Code pane, its UI and its Download button |

### Code mode (`harness/`)

| File | What it does |
|---|---|
| `agent.js` | The tool loop, with approval for edits |
| `tools.js`, `constrain.js`, `jsonschema.js`, `argfix.js` | Tool-call formats and a streaming parser; the sampling constraint; JSON Schema to a grammar; repair of loose arguments from small models |
| `code-prompt.js`, `cards.js` | The small system prompt; recovery hints added only when a call goes wrong |
| `room-model.js`, `engine-model.js`, `model-common.js` | The agent's model: the whole room or one local engine, with prefix reuse between steps |
| `codetools.js`, `workspace.js`, `projects.js`, `diff.js` | List, read, search, edit and write files in browser storage or a folder on disk |
| `preview*.js`, `preview-relay.html`, `run-js.js` | The virtual `:5173` server, the sandboxed frame, mirroring to peers, and `run_js` |
| `sessions.js`, `statecache.js`, `prefix.js` | Several conversations on one engine, parked on the GPU or on disk |
| `templates.js`, `export.js`, `app-export.js` | Starter templates; Download as `.zip` or one HTML file |
