# 30 · Long context and sessions

**Phase:** code mode · **Status:** in part · design and what landed: [docs/long-context-and-sessions.md](../docs/long-context-and-sessions.md)

## Why
A coding agent reads files, writes files, and reads errors, so its context fills fast. Re-prefilling the whole conversation on every step is the slowest thing a room does. Long context and sessions that resume without re-prefilling are what make Code mode usable, and they help long chats too.

## What landed
- f16 KV cache and split-K flash attention; int8 KV cache as an option (`?kv=q8`).
- Context per model (`CTX` in `room/models.js`, from roadmap 13): the 27B defaults to 16K tokens (up to 32K with `?ctx=`), the 35B MoE to 32K (up to 64K).
- Session state export and import, GPU slots, room checkpoints after each answer (`?ckpt=N`), and several agent sessions on one engine (`harness/sessions.js`).
- Prompt-lookup drafts over the whole context, so copying code back is cheap.

## Still open
- Room checkpoints on disk (OPFS) on every device landed (`room/ckpt-store.js`); still to do: check a reload on real hardware and stream the copy to disk part by part.
- Stable prompt rendering for agents: compaction is now stable and the system prompt + tools stay cached through it (#73); a compaction still prefills the compacted turns once. Needs timing on real hardware.
- Several sessions at once through one batched pass.
- Timing on real hardware at 1K, 8K and 32K context, with rows in the bench log.

## Done when
- A Code mode session of 20+ steps never re-prefills more than what is new, including after a device reloads.
- Bench-log rows show decode and prefill tok/s at 1K, 8K and 32K on the 27B and the 35B MoE.
