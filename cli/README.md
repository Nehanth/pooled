# @pooled/cli

`pooled serve` turns a [Pooled](https://pooled.run) room into a local OpenAI and Anthropic compatible endpoint. Any tool that talks to Chat Completions, Responses or Messages through a base URL (coding agents such as Codex CLI, Claude Code and opencode included, with their tool calls) then runs on the room's model: the model is split across the phones and laptops in the room, and this command only relays requests.

```
$ npx @pooled/cli serve "https://pooled.run/r/4TKG9P#k=…"
pooled serve · room 4TKG9P · Qwen3.6 35B MoE · Q4 · 32768 tokens of context
  OpenAI     http://127.0.0.1:8080/v1         (OPENAI_BASE_URL, any API key: chat/completions, responses)
  Anthropic  http://127.0.0.1:8080            (ANTHROPIC_BASE_URL: messages)
  bound to 127.0.0.1 only · no token (set POOLED_TOKEN to require one)
  prompts go to the room's host and may be shown to everyone in the room

For this room's 32768-token context:
  Codex        model_context_window = 32768, model_auto_compact_token_limit = 26214  (~/.codex/config.toml)
  Claude Code  ANTHROPIC_BASE_URL=http://127.0.0.1:8080 CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768 CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 claude --model pooled
  opencode     OPENCODE_DISABLE_CLAUDE_CODE=1 opencode, with this opencode.json:
    {"provider":{"pooled":{"npm":"@ai-sdk/openai-compatible","name":"Pooled room","options":{"baseURL":"http://127.0.0.1:8080/v1","apiKey":"x"},"models":{"pooled/qwen3.6-35b-moe":{"name":"Qwen3.6 35B MoE · Q4","tool_call":true,"limit":{"context":32768,"output":8192}}}}},"model":"pooled/pooled/qwen3.6-35b-moe"}
```

The settings appear once the room's model is ready (if the host has not started it yet, the bridge prints them when it is).

Requirements: Node 22 or newer. No GPU is needed on this machine. The room's host page must be open, and the room's model started for requests to be answered (until then they get `503` with `Retry-After: 5`).

**Getting in.** Give it the room's invite link (Invite in the room, or the Serve API page), in quotes: the link carries the room's invite key after `#k=`, and the host lets a client with it in at once. With the code alone (`serve 4TK-G9P`), a host that asks before new devices join (the default) sees "pooled serve … wants to join (API client)" and the client waits until the host presses Allow. Either way the host's **Allow API clients** switch must be on. The host gives the client a pass when it lets it in, so reconnecting after the host page reloads doesn't ask again. Codes of older rooms (four characters) still work.

In the room, the black **Serve API** button in the header opens the Serve API page: this command with the room's code, the base URLs, examples, and who is connected.

## How it works

A browser tab cannot accept HTTP connections, so this small Node process joins the room as one more member: an **API client** with no layers, over the same WebRTC data channels the room's devices use ([peerjs](https://peerjs.com) on [node-datachannel](https://github.com/murat-dogan/node-datachannel)). Each HTTP request becomes one question to the room. The room shows the client as its own card, and its exchanges in the chat (under the room's "who sees the chat" setting), marked "via API".

- Every request is stateless: your tool sends the whole conversation each time, and the room answers it without touching its own chat. When your tool resends the room's previous answer unchanged, the room reuses its caches and prefills only the new turn.
- One generation runs in the room at a time. The chat, Code mode and API clients share one queue. Requests wait here first (up to `--max-queue`, 8 by default) and streams get a keep-alive every 10 s while they wait.
- If the host page reloads, in-flight requests fail and the client reconnects when the room is back. If the host disconnects the client, or turns API clients off, it stops and answers `503`.

## Options

```
pooled serve <ROOM CODE | room link> [options]

  --port <n>        HTTP port (default 8080)
  --token-file <f>  require the token in this file as "Authorization: Bearer <t>" or
                    "x-api-key: <t>" on every request (or set POOLED_TOKEN)
  --token <t>       the same, given on the command line (other local users can read it with ps)
  --name <s>        how the room shows this client (default: "pooled serve" and 4 random letters)
  --signal <h:p>    PeerJS signaling server, as the room page's ?signal= (default: PeerJS cloud)
  --max-queue <n>   requests that may wait here behind the running one before 429 / 529
                    (default 8; 0 = only when idle)
  --quiet / --json-log
```

The endpoint listens on `127.0.0.1` only. Requests whose `Host` is not `127.0.0.1:<port>` / `localhost:<port>`, and any request with an `Origin` header, are refused (403), so a web page cannot use the room through your browser. Without a token any API key is accepted (some tools insist on one). With one (`POOLED_TOKEN`, `--token-file` or `--token`), every path needs it except `/health`, which then answers only `{"ok": true}`.

## Who sees your prompts

Everything a tool sends goes to the room's host, a browser tab on someone's device, and the room's other devices compute on it. Coding tools send file contents, and sometimes secrets, so point them only at a room you trust.

- The host's screen shows every API request: the last user message (up to 2000 characters) and the answer.
- Other people in the room see the same unless the host sets the room's answers to **Only me** or **Whoever asked**.
- If the host's page closes, the bridge knocks on the room's code for a minute in case the page reloads. The room code is the only name involved, so a page that opens a room with that code within the minute would receive the next requests (it would not know the bridge's pass, so it could only take it in as a new client). Stop the bridge (Ctrl-C) when the room ends.

## APIs

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions with tool calls and JSON mode, streaming (`stream: true`, `stream_options.include_usage`) and not |
| `POST /v1/responses` | OpenAI Responses with function tools, `previous_response_id` and `text.format`, streaming and not |
| `GET` / `DELETE /v1/responses/{id}`, `GET /v1/responses/{id}/input_items` | responses kept by this process (up to 256 or 64 MB, an hour after last use) |
| `POST /v1/messages` | Anthropic Messages with tools, `tool_choice` and thinking, streaming and not |
| `GET /v1/models`, `GET /v1/models/{id}` | the room's model, `pooled/<model>`, with `max_model_len` (the room's context) as vLLM lists it (Anthropic's shape when the request has `anthropic-version` or `x-api-key`) |
| `GET /health` | `{ok, room, connected, ready, model, ctx, queue, served}` for scripts (`ctx`: the room's context in tokens); with a token set and not given, only `{ok: true}` |

Parameters: `max_tokens` / `max_completion_tokens` / `max_output_tokens` (default 16384 on Chat Completions and Responses; above 65536 capped, not refused; always capped by the room's context), `temperature` (0 is greedy; OpenAI 0 to 2, Anthropic 0 to 1; absent uses the room's setting), `top_k` (1 to 64, default 40), `stop` / `stop_sequences` (up to 4), thinking (OpenAI `reasoning_effort` other than `none` / `minimal`, returned as `reasoning_content`; Anthropic `thinking: {type: "enabled", budget_tokens}`; once the reasoning reaches `budget_tokens` the think block is closed and the rest of `max_tokens` goes to the answer). `model` can be any string. `top_p` and `seed` are accepted and ignored. `usage.prompt_tokens_details.cached_tokens` (OpenAI) and `usage.cache_read_input_tokens` (Anthropic) report the prompt tokens the room already held.

Chat Completions tools and structured output (the room's host must run a Pooled with tool calling; an older one answers these with a `400` asking to reload the host page):

- `tools` (function tools, up to 1024; `strict` is accepted and always holds: every call is constrained to its schema), `tool_choice` (`auto`, `none`, `required`, `{"type": "function", "function": {"name"}}`, `{"type": "allowed_tools", ...}`), `parallel_tool_calls`. Calls come back as `message.tool_calls` with `finish_reason: "tool_calls"` (`"stop"` for a named `tool_choice`, as OpenAI), streamed as `delta.tool_calls` (the name first, then argument fragments). Send them back as an assistant message with `tool_calls` and one `role: "tool"` message per result (`tool_call_id`); `reasoning_content` on that assistant message is fed back to the model.
- `response_format`: `json_object` and `json_schema` are constrained. Types, required keys, enums, array lengths (`minItems` / `maxItems`) and structure are guaranteed; string and number bounds and `pattern` are not. The same holds for Responses `text.format` and Messages `output_format`.
- `reasoning_effort`: `none` / `minimal` turn thinking off, `low` to `max` on (Qwen3.8 uses the level; the others think the same way at any level). `chat_template_kwargs.enable_thinking` works too when `reasoning_effort` is absent.
- A call cut by `max_tokens` is left out of a whole answer; streamed, its name and partial arguments already went out, and `finish_reason: "length"` says so.
- A string argument containing a line `</parameter>` ends there (Qwen3.5+ call format).

Images and files anywhere in the conversation become a short note (`[image omitted: this model reads text only]`, logged once), so a pasted screenshot does not break the session. Still a clear `400`: `custom` tools, the deprecated `functions` / `function_call` and `role: "function"`, audio, `n > 1`, logprobs, `prediction`, `web_search_options`, an assistant message last (prefill). Responses (`/v1/responses`), on the same machinery:

- `input` as a string or items: messages (user, assistant, system, developer), `function_call` / `function_call_output`, `custom_tool_call` / `custom_tool_call_output`, `reasoning` (a `pooled1.` `encrypted_content` restores the exact reasoning), `item_reference`. `instructions`, function `tools`, custom (free-form) `tools` (the model writes one raw string, returned as a `custom_tool_call` item with `input`; a `grammar` format is shown to the model, not enforced: Codex's `apply_patch` with a GPT-5 model name works), every `tool_choice` form, `parallel_tool_calls`, `max_tool_calls`, `text.format` (`json_object`, `json_schema`), `reasoning.effort` (absent or `none` is off), `include: ["reasoning.encrypted_content"]`.
- `previous_response_id` chains onto a response this process stored (`store`, default true); an unknown id is a `400` with code `previous_response_not_found`. Stored responses live in memory only.
- Output items come in generation order (reasoning, message, calls); a stream is the full event sequence with `sequence_number`, from `response.created` to `response.completed` / `.incomplete` / `.failed`; while the room is busy or prefilling, `response.in_progress` repeats every 10 s (clients such as Codex time out on silence, and SSE comments do not count). Without `max_output_tokens` up to 16384 tokens (capped by the room's context).
- Hosted tools such as `web_search` (Codex always sends it) and items of tools that ran elsewhere are skipped with one logged warning. `400`: hosted `tool_choice`, a malformed `allowed_tools` entry, `background`, `conversation`, `prompt`, logprobs.

Messages (`/v1/messages`):

- `tools` with `input_schema`, `tool_choice` (`auto`, `any`, `{"type": "tool", "name"}`, `none`, `disable_parallel_tool_use`). Calls come back as `tool_use` blocks with `stop_reason: "tool_use"`, streamed as `input_json_delta`; results go back as `tool_result` blocks (`is_error` too). Anthropic's own tools (web search, bash, text editor, ...) are skipped with a logged warning.
- Thinking: `enabled` with `budget_tokens`, `adaptive`, `disabled`, `display: "omitted"`. The thinking block's `signature` carries the reasoning (`pooled1.`), so a client that sends the block back gives the room its exact reasoning with nothing stored here.
- `output_config.format` / `output_format` (`json_schema`) are constrained; `output_config.effort` is accepted. `max_tokens` above 65536 is capped. `usage.input_tokens` leaves out `cache_read_input_tokens`.
- Images and documents become a short note, in user turns and in `tool_result`s. `400`: `search_result` blocks, a non-empty `mcp_servers`.

`/v1/embeddings`, `/v1/completions` (the legacy API: use `/v1/chat/completions`), `/v1/responses/input_tokens`, `/v1/responses/compact` and `/v1/messages/count_tokens` are `404`.

## Tools

### curl

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "pooled", "messages": [{"role": "user", "content": "Hi"}], "stream": true}'

curl http://127.0.0.1:8080/v1/messages -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model": "pooled", "max_tokens": 200, "messages": [{"role": "user", "content": "Hi"}]}'
```

### OpenAI SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="pooled")
r = client.chat.completions.create(model="pooled", messages=[{"role": "user", "content": "Hi"}])
print(r.choices[0].message.content)
```

```js
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://127.0.0.1:8080/v1", apiKey: "pooled" });
const stream = await client.chat.completions.create({ model: "pooled", messages: [{ role: "user", content: "Hi" }], stream: true });
for await (const chunk of stream) process.stdout.write(chunk.choices[0]?.delta?.content || "");
```

### Anthropic SDK

```python
import anthropic
client = anthropic.Anthropic(base_url="http://127.0.0.1:8080", api_key="pooled")
m = client.messages.create(model="pooled", max_tokens=500, messages=[{"role": "user", "content": "Hi"}])
print(m.content[0].text)
```

### Codex CLI

In `~/.codex/config.toml`. Codex has no size for a model it does not know: give it the room's context (the banner prints these two lines for your room, and `/health` has `ctx`), or it never compacts before the room refuses a prompt that is too long:

```toml
model = "pooled"
model_provider = "pooled"
model_context_window = 32768            # the room's context
model_auto_compact_token_limit = 26214  # 0.8 of it

[model_providers.pooled]
name = "Pooled"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
```

Codex's prompt is about 15 k tokens, so the room needs a long context and a model that calls tools reliably (the 35B MoE does; the 1.7B, capped at 16 k, often says what it would run instead). With a GPT-5 model name (`model = "gpt-5-codex"`, which also silences Codex's "model metadata not found" warning) Codex sends `apply_patch` as a custom tool; that works too. On a slow room raise `stream_idle_timeout_ms` in the provider section if a first prompt takes more than 5 minutes to prefill (keep-alive events normally reset it).

### Claude Code

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=pooled \
  CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768 CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192 \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 claude --model pooled
```

`CLAUDE_CODE_MAX_CONTEXT_TOKENS` is the room's context (the banner prints this line for your room): Claude Code does not know a model called `pooled`, and without it never compacts before the room refuses a prompt. `CLAUDE_CODE_MAX_OUTPUT_TOKENS` (a quarter of the context, at most 32000): Claude Code keeps its output limit (32000) free in the window, so on a room under about 32 k without it, it refuses its own prompt ("Prompt is too long") before sending anything. `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` keeps its side requests (titles) from interleaving with the agent's and costing a re-prefill. Its prompt is 15-20 k tokens before any history, so use a room with a long context: the 35B MoE (32 k by default, up to 64 k with `?ctx=65536`) or the 27B (up to 32 k). The 1.7B's 16 k leaves about 1 k tokens after Claude Code's prompt: a file read can already overflow it, and the 1.7B cannot write the compaction summary, so use a bigger model for Claude Code. Each step after the first reuses the room's caches and prefills only the new tool results. MCP servers are fine: up to 1024 tools per request.

### opencode

In `opencode.json` in your project (or `~/.config/opencode/opencode.json`); the banner prints this for your room, with the model id from `/v1/models` and the room's context:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "pooled": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Pooled room",
      "options": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "x" },
      "models": {
        "pooled/qwen3.6-35b-moe": { "name": "Qwen3.6 35B MoE · Q4", "tool_call": true, "limit": { "context": 32768, "output": 8192 } }
      }
    }
  },
  "model": "pooled/pooled/qwen3.6-35b-moe"
}
```

Then run `opencode` in your project. `limit.context` is the room's context, so opencode compacts before the room refuses a prompt. Two tips:

- Set `OPENCODE_DISABLE_CLAUDE_CODE=1`. Otherwise opencode loads `~/.claude/CLAUDE.md` and every skill in `~/.claude/skills` into its system prompt, which can add 10 k tokens or more to every prefill.
- Pass `--title` (with `opencode run`) or set a `small_model`, so opencode skips its extra title-generation request at the start of each session, which would otherwise queue in the room ahead of the agent's first step.

Tested with opencode 1.18.33 on the Qwen3.6 35B MoE room: it read and edited files, ran the tests and finished multi-step tasks, with prompts of about 7.2 k tokens, 45-50 tok/s output, and 97-99% of each follow-up prompt reused from the room's caches.

### Continue

In `~/.continue/config.yaml`:

```yaml
models:
  - name: Pooled room
    provider: openai
    model: pooled
    apiBase: http://127.0.0.1:8080/v1
    apiKey: pooled
    roles: [chat, edit]
```

Autocomplete wants a small, fast local model; a room's answers take a lap through every device.

### Open WebUI

Settings → Connections → OpenAI API: URL `http://127.0.0.1:8080/v1`, any key. (With Open WebUI in Docker, the bridge on the host is not at `127.0.0.1` from inside the container, and v1 does not listen on other addresses; run Open WebUI on the host, or with `--network=host`.)

### LiteLLM

```yaml
model_list:
  - model_name: pooled
    litellm_params:
      model: openai/pooled
      api_base: http://127.0.0.1:8080/v1
      api_key: pooled
```

## Errors

Errors come in each API's own shape (`{"error": {...}}` / `{"type": "error", "error": {...}}`):

| | OpenAI | Anthropic |
|---|---|---|
| bad request, unsupported feature, prompt over the room's context | 400 | 400 |
| missing or wrong key (with `--token`) | 401 | 401 |
| foreign Host, or an Origin header | 403 | 403 |
| queue full here or in the room | 429 | 529 |
| model not ready, host gone, client disconnected by the host, the host pressed Stop | 503 | 529 |
| the room sent nothing for a request for 5 minutes (30 while the host queues it) | 504 | 504 |
| generation failed in the room | 500 | 500 |

Once a stream has started, an error comes as an error chunk (OpenAI, no `[DONE]`) or an `error` event (Anthropic).

## Development

```bash
cd cli && npm install
node bin/pooled.js serve 4TKG9P --signal 127.0.0.1:9000   # against a room page opened with ?signal=127.0.0.1:9000
```

Unit tests (`tests/unit/serve_*_test.js`, Deno) cover the request mapping and the byte-exact streams; `tests/e2e/serve.mjs` runs a real room with Qwen3 1.7B in headless Chromium and checks both APIs end to end, with the official SDKs. Design: [docs/design/serve.md](../docs/design/serve.md).
