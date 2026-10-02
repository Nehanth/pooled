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
  Claude Code  ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=pooled CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768 CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 claude --model pooled
  opencode     OPENCODE_DISABLE_CLAUDE_CODE=1 opencode, with this opencode.json:
    {"provider":{"pooled":{"npm":"@ai-sdk/openai-compatible","name":"Pooled room","options":{"baseURL":"http://127.0.0.1:8080/v1","apiKey":"x"},"models":{"pooled/qwen3.6-35b-moe":{"name":"Qwen3.6 35B MoE · Q4","tool_call":true,"limit":{"context":32768,"output":8192}}}}},"model":"pooled/pooled/qwen3.6-35b-moe"}
```

## Start here

`pooled` alone, in a terminal, opens a short menu (↑/↓ and Enter, or a number):

```
pooled 0.3.1 · what would you like to do?
❯ 1  Host a room on this computer        open a room here and lend this GPU
  2  Join a room                         lend this GPU to someone's room
  3  Chat with a room                    talk to a room's model right here
  4  Use a room from your tools (serve)  a local OpenAI / Anthropic API
  5  Manage models (pull, list, rm)      download, list or delete models
↑/↓ and Enter, or 1-5 · q quits · pooled --help lists the commands
```

Each item runs that command's own flow and asks for what it needs (the room code or link, the model, how much to lend). Without a terminal, `pooled` prints a short usage; `pooled --help` lists every command with an example, and `pooled <command> --help` has the details. A mistyped command gets one line back, `pooled: unknown command "hsot". Did you mean "host"?`, and a missing argument says what to type next: `pooled serve needs a room: pooled serve 4TK-G9P, or paste the invite link in quotes.`

The interactive screens use plain ANSI (no terminfo), so they work over ssh from terminals the other computer doesn't know (Ghostty's `xterm-ghostty`), in tmux, with `NO_COLOR` (no colors), and with `TERM=dumb` (numbered questions, one line at a time).

To talk to the room's model from the terminal, see [pooled chat](#talk-to-a-room-pooled-chat-preview); to lend this computer's GPU to a room instead (it holds some of the layers), see [pooled join and pooled host](#lend-a-computer-pooled-join--pooled-host-preview).

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

## Talk to a room: pooled chat [preview]

`pooled chat` talks to a room's model from the terminal, like `ollama run`. It joins the room as an ask-only client (no layers, no GPU needed here) over the same bridge and request checks as `pooled serve`, streams each answer token by token, and keeps the conversation: every turn sends all of it, so the room's host reuses what it already computed for the earlier turns and prefills only the new one (the line under each answer says how many prompt tokens were reused). Switching `/think` changes how the conversation is rendered, so the turn after a switch prefills it again.

```
$ pooled chat "https://pooled.run/r/HJQN44#k=…"
pooled chat · room HJQ-N44 · Qwen3 1.7B · Q8 · host spark-host
  type a message · /help for commands · Ctrl-C stops an answer · /exit or Ctrl-D leaves
>>> Name the largest planet in our solar system. One sentence.
The largest planet in our solar system is Jupiter.
room HJQ-N44 · Qwen3 1.7B · Q8 · 10 tokens · 56.7 tok/s · 24 prompt tokens
>>> How many moons does it have? Answer in one sentence.
Jupiter has over 79 confirmed moons.
room HJQ-N44 · Qwen3 1.7B · Q8 · 10 tokens · 61.5 tok/s · 34 of 60 prompt tokens reused
>>> /exit
```

- **In the chat**: `/clear` starts over, `/think` (or `--think`) lets the model reason first, shown dimmed under "thinking", `/exit` or Ctrl-D leaves. Ctrl-C stops the answer being written (the room stops too; the part already written stays in the conversation) and, at the prompt, leaves. While nothing is on screen yet a spinner says what the room is doing (joining, queued, reading the conversation).
- **In scripts**: with a prompt argument, or with input that is not a terminal, it asks once, prints the answer on stdout (no reasoning, no status) and exits: `echo "What is 17 times 23?" | pooled chat "$LINK"` prints `391`. Without a started model it waits `--wait` seconds (120) and then fails.
- **Errors** say what happened: no room with that code, no model started yet (in a terminal it waits for the host to start one), the host doesn't allow API clients, the host runs an older Pooled, the host didn't let it in.
- `--system`, `--max-tokens` (8192), `--temperature`, `--name`, `--signal`; `pooled chat --help` lists them. Text from the room is printed without control characters, so a host can't send escape sequences to your terminal.

What you type goes to the room's host and, under the room's visibility setting, to the other screens in the room; every device holding layers sees its hidden states (see "Who sees your prompts" above).

## Lend a computer: pooled join / pooled host [preview]

`pooled host` opens a room on this computer and `pooled join` puts this computer in someone's room as one more device that holds layers, the way a browser tab does, without a browser: Pooled's own engine runs on [Dawn](https://dawn.googlesource.com/dawn) (npm `webgpu`) in this process. Both run in the foreground until Ctrl-C, which leaves the room and frees the GPU.

### 1. Download the model: pooled pull

Like `ollama pull`: the model's files go to `~/.pooled/models/<model>`, so the room starts from disk instead of streaming layers from Hugging Face on every start. Pooled keeps its models in that one folder, made the first time it is needed; `pooled host`, `pooled join`, `pull`, `list` and `rm` all use it (`--models <dir>` or `POOLED_MODELS` moves it, for the rare setup that needs that; `pooled pull --help`).

```
$ pooled pull qwen3.6-35b-moe
pulling qwen3.6-35b-moe (Qwen3.6 35B MoE · Q4): 19.4 GB into ~/.pooled/models/qwen3.6-35b-moe
▕███████████░░░░░░░░░░░░░░░░░░▏  37%  7.2 GB / 19.4 GB  106 MB/s  ETA 1m58s
✓ qwen3.6-35b-moe is ready (19.4 GB): pooled host qwen3.6-35b-moe
$ pooled list
MODEL              NAME                     SIZE
qwen3-1.7b         Qwen3 1.7B · Q8          1.7 GB
qwen3.6-35b-moe    Qwen3.6 35B MoE · Q4     19.4 GB

Not downloaded (pooled pull <model>):
  qwen3-0.6b       Qwen3 0.6B · Q8          610 MB
  …
```

A download is written to `<file>.part` and resumes where it stopped (Ctrl-C, a lost network: run `pooled pull` again, it asks for the rest with a Range request); the file is checked against its size and SHA-256 before it is renamed and used. `pooled download` is the same command, `pooled ls` the same as `pooled list`, and `pooled rm <model>` deletes one. A part of a name works: `pooled pull 35b`. Without a terminal the progress is a plain line every few seconds.

### 2. Open a room: pooled host

In a terminal, `pooled host` opens the room at once (its code, invite link and the `pooled join` command at the top of the screen, once, so devices can join while you choose), then asks only for what you did not give as flags: the model, from one list, smallest first, that says how much memory each takes and whether it is downloaded (or how big the download is); then how to run it. **Pool with devices** (first, and preselected) asks how much this computer lends (half the GPU's memory by default, as the room page; ←/→ or type a number) and waits for devices, spreading the layers across all of them (`s` changes it). **Run it here** lends what the model needs on this computer alone (never more than its GPU allows) and starts as soon as the model is on disk; other devices can still join to chat. It is offered only when the model fits here, and otherwise shows why ("needs 23 GB, this computer has 16 GB to lend"). A model that is not downloaded downloads right there, with its progress in the screen, while devices join. Then the room itself, redrawn in place:

```
pooled host · room 9PF-Z8T · Qwen3 1.7B
invite  https://pooled.run/r/9PFZ8T#k=GezmJYfe6-sh8uNihM_bvg
join    pooled join "https://pooled.run/r/9PFZ8T#k=GezmJYfe6-sh8uNihM_bvg"
gpu     NVIDIA GB10 · 122 GB unified memory  ·  models ~/.pooled/models
Share the link only with people you trust: devices see what is asked here.
────────────────────────────────────────────────────────────────────────
model   Qwen3 1.7B  ✓ downloaded   m: change
lending 2 GB of 122 GB   p: change

Devices (2)
  ● spark              this computer  2 GB
  ● macbook            CLI            2 GB

room    ████████████████████ 4 GB pledged, Qwen3 1.7B needs 3.8 GB
Qwen3 1.7B would run on spark alone if it lends 4 GB (now 2 GB): p raises it.
Split across 2 devices, each token waits for the network: expect roughly 14
tok/s.

Enter: start  ·  m: model  ·  p: pledge  ·  q: quit
```
Every device with its kind (CLI, browser tab, phone) and pledge, who waits to join (`a` allows, `d` denies), whether the pledges hold the model, and when they split a model this computer could hold alone, how much it would take and what the split costs (each token crosses the network; the estimate comes from the links' round trips). The fit uses the room page's own math (room/plan.js `roomFit` over room/models.js `roomBytes`, each pledge through room/pledge.js), so the terminal says "short" exactly when the room page would. Start stays off until the pledges fit (and the download is done); then `Enter: start`, each device's load progress, and "online" with the layer split:

```
Devices (2)
  ● spark-host         this computer  2 GB     layers 0–5
  ● spark-join2        CLI            6 GB     layers 6–27

● online  spark-host 0-5 · spark-join2 6-27

c: chat here  ·  Enter: re-deal  ·  q: quit
```

`c` chats with the room right there (the same REPL as `pooled chat`, over the invite link; `/exit` comes back to the room screen), so the host needs no second terminal. `m` / `p` change the model or the pledge while the room waits.

Every step has a flag, and a flag given skips its question, so `pooled host qwen3.6-35b-moe --pool --gb 64 --start --yes` runs with no questions at all:

| Flag | Skips / does |
|---|---|
| `[model]`, `--model <m>` | the model picker (a part of the name works: `35b`) |
| `--pool` | the "how do you want to run it" question: pooled with other devices (the split: spread) |
| `--here` | the same: all of it on this computer, lending what it needs, starting by itself; an error when it doesn't fit here |
| `--gb <n\|max>` | the pledge question |
| `--ram <n>` | how much system RAM a MoE model's experts may use when the GPU's pledge is short (expert offload, below); `0` never offloads |
| `-y`, `--yes` | the "not downloaded (19.4 GB). Download it now? [Y/n]" question: downloads |
| `--no-pull` | don't download: stream this computer's layers from Hugging Face, as before |
| `--start` | start as soon as the pledges hold the model |
| `--wait <n>` | start once n devices (this one included) are in and the pledges fit (`--devices <n>`, its older name) |
| `--allow-all` | no lobby: anyone with the code comes in |
| `--deny-unknown` | no lobby: a device with the code alone is turned away (the invite link still works) |
| `--chat` | chat here once the room is online |
| `--name <s>` | how the room shows this computer (default: `node-` and 3 letters from the hostname) |
| `--split <speed\|spread>` | how the layers spread: `speed` (fastest first, the default, as the room page) or `spread` (across all devices by what each lends, even when one could hold the model); `s` toggles it in the screen, which shows what each device would hold |

Without a terminal (a script, a service, `--json-log`) nothing is asked: the model defaults to `qwen3-1.7b`, the pledge to the memory rule below, a model that is not downloaded is pulled first (`--no-pull` streams instead), and the room starts once the pledges fit (`--wait N`: and N devices are in).

### 3. Lend to someone's room: pooled join

`pooled join` alone, in a terminal, asks for the code or the invite link, shows the host and its model ("downloaded here: loads from disk" when you pulled it) and asks how much to lend; `pooled join "<link>" --gb 8` asks nothing. When the host's model is not in `~/.pooled/models` yet, it downloads the whole file there first, with the same progress bar as `pooled pull` ("Qwen3 1.7B isn't downloaded yet (1.7 GB). Downloading it into ~/.pooled/models so this and later joins load from disk."), then loads from disk; a deal that comes meanwhile waits for it. `--no-pull` streams only this device's layers from Hugging Face instead, and the status line then says what it is doing: `downloading 312 of 900 MB · 18 MB/s · from Hugging Face`, then `loading onto the GPU`.

```
$ pooled join HJQ-N44
pooled join · room HJQ-N44
  GPU      NVIDIA GB10 · 121.7 GB unified memory
  lending  64 GB  (121.7 GB of unified memory, less 42.6 GB kept for the system, capped at 64 GB per device; --gb to change)
  whoever the host lets in can use what this computer lends, and it sees the room's hidden states
  (they carry the prompts and answers): lend to rooms you trust
  Ctrl-C leaves the room and frees the GPU
13:06:50 reached room HJQ-N44 as node-kqd
13:06:50 waiting for the host to let this device in (a room's invite link gets in without asking)
  host     spark-host · Qwen3 1.7B · Q8 (downloaded here: loads from disk)
  lend     ◀ 8 GB ▶  of 122 GB here, at most 64 · ←/→ or type, Enter
13:06:52 the host let this device in
13:06:55 holding layers 14-27 of qwen3-1.7b (loaded in 3.5 s)
room HJQ-N44 · online · 2 devices · layers 14-27 of qwen3-1.7b · 56.4 tok/s · 390 passes
```

- **Getting in** (the room page's gate, [docs/protocol.md](../docs/protocol.md) "Joining a room"; the room page's side is PR #268). Room codes are six characters (`4TK-G9P`; four-character codes of older rooms still work), and a code only finds a room. A device that has the room's **invite link** (its `#k=` key; quote it in the shell) is let in at once; one with the code alone waits in the host's lobby, "waiting for the host to let you in", until the host allows it. The host gives each device it lets in a pass, so a device that loses its link and knocks again is let back in without asking. `pooled join`, `pooled chat` and `pooled serve` all take the link. A host from before the gate lets everyone in, as before.
- **As a host**, `pooled host` prints the invite link and asks you about each device that comes with the code alone: `a` lets the oldest request in, `d` turns it away. Without a terminal nobody is asked, so such a device waits until you give it the link; `--allow-all` lets in anyone with the code (and room pages from before the gate, which can't wait in a lobby), `--deny-unknown` turns them away. Enter deals the layers over the devices in the room once their pledges hold the model (and again, re-deals after more join); `--wait N` deals as soon as N devices are in and fit, and deals again by itself when a device stayed away past the minute's grace (the room went on without it) and N are back.
- **How much it lends.** WebGPU does not say how much memory a GPU has, so `pooled` asks the OS: `nvidia-smi` for NVIDIA cards, sysfs for AMD on Linux, the system's memory on Apple silicon and on GPUs that share it (GB10). A discrete GPU lends its free memory less 1.5 GB; unified memory lends the total less max(8 GB, 35%) (on Linux, never more than is free right now, less 2 GB); at most 64 GB per device. On a discrete GPU it then allocates that much for a moment to make sure it is there (`--no-check` skips this; unified memory skips it, since the OS's numbers are the memory itself and touching 64 GB of it takes about 20 s). `--gb N` sets it, `--gb max` keeps only a small margin. When the memory can't be read it lends half of the GPU's largest buffer and says so.
- **Expert offload (MoE models, discrete GPUs).** A MoE layer is mostly its 256 routed experts, of which a token uses 8. When the room's pledges can't hold a MoE model (the 35B or the 122B), a `pooled` device with a discrete GPU holds layers past its pledge anyway: it keeps those layers' experts in system RAM and copies the ones each token picks into a cache on the GPU (slower than layers held whole: one GPU round trip per offloaded layer per token, plus the copies on a cache miss). This is what lets a 12 GB GPU run the 35B alone with `pooled host qwen3.6-35b-moe --here` (it lends the memory rule's GB; the cache gets what the layers don't use). How much RAM it may use is `--ram N`, like `--gb` for the GPU: by default the total less 16 GB, never more than is free at start less 4 GB, at most 64 GB; `--ram 0` turns it off. Unified memory (Apple silicon, GB10) never offloads: its pledge is that same memory. The host deals it (room/plan.js, the room page does the same): a room the pledges hold deals exactly as before; otherwise every device that doesn't offload holds all its pledge holds, and the devices that offload take the rest, each within its pledge (the layers' other weights plus the cache: one whole layer's experts for prompts and at least 16 cached experts per offloaded layer) and within its `--ram`. It offloads the fewest layers that still leave each a 64-expert cache (a quarter of its experts), else as many as its RAM takes. The header shows `RAM up to N GB ...`, the host's log which layers each device offloads. Browsers don't offload.
- **The status line** shows the room, what is happening (waiting for the host to let you in, waiting for the host to deal layers, loading, online, answering, a device left, rejoining), the devices in the room, the layers this computer holds, the room's speed on its last answer and how many passes ran here. Without a terminal (or with `--json-log`) it prints a line when that changes, and once a minute.
- **When the link drops.** A lost signaling server does not stop the room (the links are direct); `pooled` reconnects to it with backoff (2, 4, 8, 16, 30 s), as the room page does, and tries the next server in a `--signal` list when one is down. When the link to the host drops it knocks for a minute (with its pass) and the host puts it back in its slot, with its layers still loaded. If the host does not come back, it joins the room again from scratch for `--wait` minutes (10 by default), then stops. It goes back only to a host of the same name: a room code can be reused by a new room, and `pooled join` never joins a room you did not name (it leaves one with another host and says so). Without `--name`, the device's name is `node-` and 3 letters made from the hostname, the same on every run, so a `pooled join` started again after a crash takes back its slot.
- **GPU driver messages.** Starting Dawn makes the system's Vulkan drivers print harmless lines of their own (on a GB10: `MESA: error: Opening /dev/dri/card0 failed: Permission denied`, `TU: error: … VK_ERROR_INCOMPATIBLE_DRIVER`, `Warning: maxDynamicUniformBuffersPerPipelineLayout artificially reduced …`). `host` and `join` hide them (stderr points at a file while Dawn sets up the GPU, then back); `--verbose` (or `POOLED_VERBOSE=1`) shows them, and when no GPU is found the error quotes what the driver said.
- **Errors** say what to do: no WebGPU adapter (Dawn needs Metal on macOS 26+, Vulkan on Linux, D3D12 on Windows), not enough GPU memory, no room with that code, the signaling server can't be reached, the host turned this device away, a kernel this GPU's shader compiler can't build (join from Chrome or Edge instead; `--verbose` shows the compiler's error), and a host on another protocol version (update with `npx @pooled/cli@latest`, or ask the host to reload).
- **Dawn is optional.** `pooled serve` and `pooled chat` need no GPU. `webgpu` (about 95 MB with all five OS builds) is an optional dependency: `npm install -g @pooled/cli` installs it when it can, and if it can't, `join` and `host` say how to add it (`npm install -g @pooled/cli webgpu@0.6.1`).

**Who sees what.** Every device that holds layers computes on every prompt: it receives the hidden state of each token, which carries the prompts, tool results and answers, so a `pooled join` device can in principle read what is asked in the room it lends to, and every device let into a `pooled host` room can read what is asked there. Whoever is let in can also ask, and use what the devices lend. The gate decides who that is: share the invite link only with people you trust, and let in (a) only devices you recognise. The invite key is a secret in the link's fragment (browsers never send it to a server, but it shows in `ps` for the process given it and in shell history). The room's host picks the model (from Pooled's own list; a host can't make this device download anything else) and how many layers this device holds. Needs macOS 26 or newer, Linux with glibc 2.38 or newer (Ubuntu 24.04), or Windows (D3D12; tested on Windows 11 with an RTX 5070).

**Windows.** `join` and `host` run on D3D12. On a Windows computer the browser is the fallback: if `pooled join` says this GPU's shader compiler can't build a kernel, open the room's link in Chrome or Edge instead, which compile the same kernels with their own compiler. On a laptop or desktop with two GPUs (integrated and discrete), set Chrome or Edge to **High performance** in Windows Settings > System > Display > Graphics, or the browser picks the integrated GPU.

Not included yet: running as a background service (it is a foreground command), and phones (they join from the room page).

## Development

```bash
cd cli && npm install
node bin/pooled.js serve 4TKG9P --signal 127.0.0.1:9000   # against a room page opened with ?signal=127.0.0.1:9000
npm run build && (cd ../packages/room-node && npm install)   # join / host: the room node bundle (dist/) and Dawn
node bin/pooled.js join 4TKG9P --signal 127.0.0.1:9000
node bin/pooled.js chat 4TKG9P --signal 127.0.0.1:9000
```

`npm run build` bundles `packages/room-node` (with `engine/`, `room/`, `harness/` and `cli/lib/`) into `dist/room-node.js` with esbuild; `npm pack` runs it. Without the bundle, `join` and `host` use `packages/room-node` from the checkout. In a checkout, Dawn comes from `packages/room-node`'s own `webgpu` dependency: npm 11 does not install an optional peer dependency into `cli/` (`npm install --no-save webgpu` there reports "up to date" and adds nothing). `test/cli_test.mjs` covers the front door (the usage, `--help`, "did you mean", the one-line errors, run as a real process); `test/menu_test.mjs` the menu (keys, numbers, `TERM=dumb`, `NO_COLOR`, an unknown TERM) and the fd-2 swap that hides the driver's lines; `test/repl_test.mjs` that a streamed answer is never overwritten by the chat prompt; `test/lend_test.mjs` argument parsing, the memory rule, the status line and the error messages; `test/pull_test.mjs` pull / list / rm against a local server with Range requests (progress, resume after an abort, a wrong size or hash, the room node reading a pulled model); `test/hostui_test.mjs` pooled host's screen (the model list, pledge defaults, a short room, Start gating, the keys, every flag given means no question); `test/chat_test.mjs` pooled chat's arguments, the conversation it sends, the rendering and its errors; `packages/room-node/test/gate_test.mjs` the gate on a node host and device.

Unit tests (`tests/unit/serve_*_test.js`, Deno) cover the request mapping and the byte-exact streams; `tests/e2e/serve.mjs` runs a real room with Qwen3 1.7B in headless Chromium and checks both APIs end to end, with the official SDKs. Design: [docs/design/serve.md](../docs/design/serve.md).
