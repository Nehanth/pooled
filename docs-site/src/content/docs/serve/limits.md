---
title: Limits and context
eyebrow: Serve API reference
description: How much context each model has, how long prompts and answers can be, and how requests queue and time out.
sidebar:
  label: Limits and context
  order: 8
---
<!-- Sources (origin/main): room/models.js (CTX, maxSeqFor: rounded to 256, min 2048), room.js (?ctx= read on the host page,
     QUEUE_MAX 10, 2 per member), room/api.js (API_LIMITS.reserve 32, "prompt is too long"), cli/lib/common.js (LIMITS;
     early check chars > 8 * ctx), cli/lib/http.js (KEEPALIVE_MS 10000, IDLE_MS 300000, HOST_QUEUED_MS 1800000,
     maxConnections 64, Retry-After 5, --max-queue), harness/constrain.js (MAX_CALLS 16), cli/README.md, docs/design/serve.md §7, §11.6. -->

## Context per model

The room's context is the most tokens one request can hold: prompt plus answer.

| Model | Default | Most with `?ctx=` |
|---|---|---|
| Qwen3 1.7B | 16384 (8192 when the room is short of memory for 16K) | 16384 |
| Qwen3.8 27B | 16384 | 32768 |
| Qwen3.6 35B MoE | 32768 | 65536 |

The host sets it by opening the room page with `?ctx=` before starting the model, for example `pooled.run/room?ctx=65536`. The value is rounded to a multiple of 256, at least 2048.

The bridge reports it in the banner, as `ctx` in `GET /health` and as `max_model_len` in `GET /v1/models`. Give it to coding agents so they compact in time: see [context settings](/docs/serve/recipes#context-settings-for-coding-agents).

## Prompt and answer length

- **Prompt.** It must leave at least 32 tokens of the context free. A longer one is a `400` with code `context_length_exceeded`. The bridge refuses early, without asking the room, when the text is over 8 characters per token of context.
- **Answer.** `max_tokens` (or `max_completion_tokens`, `max_output_tokens`) defaults to 16384 on Chat Completions and Responses. Messages requires it. Above 65536 it is capped, not refused. The answer always stops when the context is full.

When the context fills mid-answer: Chat Completions returns `finish_reason: "length"`, Responses `status: "incomplete"`, Messages `stop_reason: "model_context_window_exceeded"`.

## Request size

| Limit | Value | Over it |
|---|---|---|
| HTTP body | 4 MiB | `413` |
| The request as sent to the room | 3.5 MB | `413` |
| Text in the whole request | 1,500,000 characters | `400` |
| Messages | 1000 | `400` |
| `stop` / `stop_sequences` | 4 strings, 64 characters each | `400` |
| Tools | 1024 | `400` |
| Connections at once | 64 | |

Tool and schema limits are on [Tool calling](/docs/serve/tools#limits) and [Structured output](/docs/serve/structured-output#schema-size).

## Queues

The room runs one generation at a time for its chat, Code mode and API clients.

1. **In the bridge.** One request goes to the room at a time, and up to 8 more wait (`--max-queue`).
2. **In the room.** The host queues at most 10 questions, 2 per member.

When either is full, the request gets `429` (OpenAI) or `529` (Anthropic) with `Retry-After: 5`.

While a request waits or the room reads a long prompt, streams get a keep-alive after 10 s of silence: an SSE comment on Chat Completions, a `ping` on Messages, `response.in_progress` on Responses. A non-streamed request sends nothing until it is done, so use streaming if your client has a short read timeout.

## Timeouts

- If the room sends nothing for a request for 5 minutes, the bridge returns `504`. While the host holds it in its own queue, the limit is 30 minutes.
- There is no limit on total time. A long prompt on a room of phones can take minutes before the first token.
- Closing the connection stops the request in the room.

Every status code is on [HTTP endpoints](/docs/reference/endpoints#status-codes).
