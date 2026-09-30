---
title: Limits and context
eyebrow: Serve API
description: How much context each model has, how long answers can be, how requests queue and time out, and every error the bridge returns.
sidebar:
  label: Limits and context
  order: 8
---
<!-- Sources (origin/main 40bb0fa): room/models.js (CTX, maxSeqFor: rounded to 256, min 2048), room.js 2516 (?ctx=
     read on the host page), room/api.js (API_LIMITS.reserve 32, "prompt is too long"), room.js (QUEUE_MAX 10, 2 per
     member), cli/lib/common.js (LIMITS; early check chars > 8 * ctx), cli/lib/http.js (KEEPALIVE_MS 10000, IDLE_MS
     300000, HOST_QUEUED_MS 1800000, Retry-After 5, --max-queue), cli/lib/openai.js + anthropic.js (status tables),
     cli/README.md (Parameters, Errors, 404 list), docs/design/serve.md §7, §11.6, §15. -->

## Context per model

The room's context is the most tokens one request can hold: the prompt plus the answer.

| Model | Default | Most with `?ctx=` |
|---|---|---|
| Qwen3 1.7B | 8192 | 16384 |
| Qwen3.8 27B | 16384 | 32768 |
| Qwen3.6 35B MoE | 32768 | 65536 |

To change it, open the host's room page with `?ctx=` before starting the model, for example `pooled.run/room?ctx=65536`. The value is rounded to a multiple of 256, at least 2048, at most the model's maximum.

The bridge shows the room's context in three places:

- the banner: `32768 tokens of context`
- `GET /health`: `"ctx": 32768`
- `GET /v1/models`: `"max_model_len": 32768`

:::tip[Tell your agent the context]
Coding agents do not know a model called `pooled`, so they assume a large window and never compact in time. Give them the room's context. The banner prints the exact settings for Codex, Claude Code and opencode.
:::

## Prompt and answer length

- **Prompt.** A prompt must leave at least 32 tokens of the context free. A longer one is a `400` with code `context_length_exceeded`. The bridge also refuses early, without asking the room, when the text is over 8 characters per token of context.
- **Answer.** `max_tokens` (and `max_completion_tokens`, `max_output_tokens`) defaults to 16384 on Chat Completions and Responses. Messages requires it. Above 65536 it is capped, not refused, because agents send their own model's output limit. The answer always stops when the context is full.

When the context fills during an answer, Chat Completions returns `finish_reason: "length"`, Responses `status: "incomplete"`, and Messages `stop_reason: "model_context_window_exceeded"`.

## Request size

| Limit | Value | Over it |
|---|---|---|
| HTTP body | 4 MB | `413` |
| The request as sent to the room | 3.5 MB of JSON | `413` |
| Text in the whole request | 1.5 million characters | `400` |
| Messages | 1000 | `400` |
| `stop` / `stop_sequences` | 4 strings, 64 characters each | `400` |
| Tools | 1024 per request | `400` |

Tool and schema limits are on [Tool calling](/docs/serve/tools#limits) and [Structured output](/docs/serve/structured-output#schema-size).

## Queues

The room runs one generation at a time for everyone: its chat, Code mode and API clients.

1. **In the bridge.** One request goes to the room at a time. Up to 8 more wait in the bridge (`--max-queue`). Past that: `429` (OpenAI) or `529` (Anthropic), with `Retry-After: 5`.
2. **In the room.** The host queues at most 10 questions, 2 per member. When that is full, the request gets `429` or `529` too.

While a request waits or the room prefills, streams get a keep-alive after 10 s of silence: an SSE comment on Chat Completions, a `ping` on Messages, and `response.in_progress` on Responses.

:::note
A non-streamed request sends nothing until the answer is complete. If your client has a short read timeout, use streaming.
:::

## Timeouts

- If the room sends nothing for a request for 5 minutes, the bridge stops it and returns `504`.
- While the host holds the request in its own queue behind other answers, that limit is 30 minutes.
- Closing the connection stops the request in the room.

There is no limit on total time. A long prefill on a room of phones can take minutes before the first token.

## Errors

Errors come in each API's own shape: `{"error": {...}}` for OpenAI, `{"type": "error", "error": {...}}` for Anthropic.

| Situation | OpenAI | Anthropic |
|---|---|---|
| Bad request, unsupported feature, prompt over the room's context | `400` | `400` |
| Missing or wrong key (with a token set) | `401` | `401` |
| Foreign `Host`, or an `Origin` header | `403` | `403` |
| Unknown path | `404` | `404` |
| Wrong method on a known path | `405` | `405` |
| Body or request too large | `413` | `413` |
| Queue full, in the bridge or in the room | `429` | `529` |
| Model not ready, host gone, client disconnected by the host, the host pressed Stop | `503` | `529` |
| The room sent nothing for 5 minutes (30 while the host queues it) | `504` | `504` |
| Generation failed in the room | `500` | `500` |

`503` and `429` carry `Retry-After: 5` where waiting helps: the model is loading, a device left and the host is re-dealing layers, or a queue is full.

Once a stream has started, an error comes as an error chunk with no `[DONE]` (OpenAI) or an `error` event (Anthropic, and `response.failed` on Responses).

## Not available

These paths are `404`:

- `/v1/embeddings`
- `/v1/completions` (the legacy API; use `/v1/chat/completions`)
- `/v1/responses/input_tokens`
- `/v1/responses/compact`
- `/v1/messages/count_tokens`

Images, files and documents become a short note (`[image omitted: this model reads text only]`) instead of an error, so a pasted screenshot does not break a session. Audio is a `400`.
