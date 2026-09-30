---
title: Long context and sessions
description: How Pooled fits long prompts, saves and resumes a conversation's state on every device, survives reloads, and why int8 KV is still off by default.
eyebrow: How it works
sidebar:
  label: Long context and sessions
  order: 4
---


A coding agent resends almost its whole conversation on every turn: about 96% of its input is the same as last time. So Pooled needs two things: a long context, and a way to skip re-reading what the devices already hold. This page covers both.

## Context per model

| Model | Default context | Maximum |
|---|---|---|
| Qwen 3.8 27B | 16,384 tokens | 32,768 |
| Qwen 3.6 35B MoE | 32,768 tokens | 65,536 |
| Qwen3 1.7B | 8,192 tokens | 16,384 |

Ask for another size with `?ctx=` on the host's room link. It is rounded to a multiple of 256, at least 2,048, and capped at the model's maximum.

```txt
https://pooled.run/room?ctx=65536
```

The KV cache is split with the layers, so each device holds only its own attention layers' share. The practical limit is prefill time, not memory.

## The KV cache

Only the full-attention layers keep a KV cache. DeltaNet layers keep a fixed-size state instead, which is why a hybrid model's memory grows slowly with context.

| Format | Per token (whole 27B) | At 32K (whole 27B) | Switch |
|---|---|---|---|
| f16 (default) | 65.5 KB | 2.1 GB | |
| int8, one f32 scale per 32 values | 36 KB | 1.2 GB | `?kv=q8` on the host |

The host's choice goes out with every `ai-load`, so all devices build the same cache and the layer deal counts the right bytes. int8 KV is only available on the hybrid (Qwen 3.5 and later) models; other models stay f16.

Attention is split-K flash attention: positions in splits of 256, an online softmax, and a merge in fixed order. The order depends only on absolute position, so speculative decoding and split rooms still give exactly the same tokens.

### Why int8 KV is off by default

Measured on the 35B MoE on a GB10 (Deno), int8 KV gave the same 32 greedy tokens as f16 at 1K, 8K and 32K, and used 0.35 GB of KV instead of 0.63 GB. But a 32K prompt took 3.5 times as long to prefill (679.6 s against 195.8 s), because the tiled prefill attention kernel does not support int8 yet.

:::caution[Use `?kv=q8` only when memory is the limit]
It saves memory and keeps the output, but long prompts become much slower to read. It stays off by default until tiled prefill attention supports int8.
:::

The 27B at 1K, 8K and 32K has not been timed yet.

## Checkpoints

A hybrid model can't be cut back to an arbitrary position: the DeltaNet state is a running sum. So Pooled reuses work at **checkpoints**: saved copies of every device's state at a known point in the conversation.

### In a room

After every answer, the host saves the room's state on every device. The save rides as a flag on the next frame down the chain, like a reset or a rollback, so every device saves its own layers at exactly the same point.

When you regenerate, edit a question or branch, the host loads the longest saved answer that is a prefix of the new prompt and prefills only what is new. The status line says "(N reused)".

| Setting | Default | Effect |
|---|---|---|
| `?ckpt=N` | 2 | Keep the last N answer checkpoints. `?ckpt=0` turns them off. |
| `?ckptdisk=0` | on | Keep checkpoints on the GPU only, with no copy on disk. |

A device that rejoins with a fresh engine, a re-deal or a failed answer clears the checkpoints on the GPU.

### On disk, across reloads

Each device also keeps a copy of its part of every checkpoint on disk, in the browser's origin-private file system (OPFS).

- **A worker that reloads** reads its copies back into GPU slots before it says it is ready. The next question resumes from the last answer the chain saved.
- **A host that reloads** resumes its room, reads its copies and their token ids back, and every device reads its own when the layers are dealt again.
- **A device missing a copy** (the write never finished, or the disk was full) says so when it is ready. The host forgets that checkpoint, so it never asks the chain to load a slot one device lacks. The next question resumes from an older checkpoint or prefills from scratch.

A copy is named by room code, slot, and a hash of the model and the device's layers and KV format. A copy for other layers or another model reads as missing. A new copy removes the old one first, so a failed write leaves a slot missing, never stale. When the disk is full, the oldest copies of other rooms go first.

:::note[Known limit]
A save reaches the workers only with the next question's first frame. After a worker reloads, the room resumes from the checkpoint before the last answer and prefills that answer again. Reload times have not been measured on real hardware.
:::

### The system prompt stays cached

Code mode pins the system prompt and tool list as its own checkpoint. It is not counted in `?ckpt=N` and answer saves never evict it. When the agent compacts old tool output, the next step resumes from the pinned prompt and prefills only what follows. A new system prompt replaces the pin everywhere.

## Sessions on one engine

The engine can save and restore its whole state:

| Call | What it does |
|---|---|
| `exportState()` / `importState(state)` | This device's KV rows, DeltaNet states, conv windows and draft state, read back one part at a time. A state for other layers, another model or another KV format is refused. |
| `saveSlot(name)` / `loadSlot(name)` / `dropSlot(name)` | The same as GPU-to-GPU copies, for switching sessions or rewinding an agent. |

On the 27B, a state is about 150 MiB of DeltaNet state (fixed) plus 65.5 KB per token of KV, for the whole model.

`harness/sessions.js` uses this to run several conversations on one engine. Switching parks the current one in a GPU slot and brings another back from a GPU slot, from disk, or fresh. Past the number of GPU slots, the least recently used sessions go to OPFS. Switching is exact: a test interleaves three sessions and compares every logit with the same sessions run uninterrupted.

This is time-sharing. One session computes at a time; batching several through one pass is not built yet.

## What is not built yet

- A shared disk cache of states for rooms, keyed by the token ids (`harness/statecache.js` exists; only tests use it).
- Several sessions batched through one GPU pass.
- A tiled prefill attention kernel for int8 KV.
- Streaming SSD weights. A dense model touches every weight every token, so decode would run at disk speed. Splitting layers across devices is Pooled's answer instead.
