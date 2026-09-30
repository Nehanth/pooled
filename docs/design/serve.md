# `pooled serve`: the room as a local OpenAI and Anthropic endpoint

Status: v1 built (chat only). v2 (tools, structured output, reasoning in history; OpenAI Chat
Completions, OpenAI Responses and Anthropic Messages, no legacy `/v1/completions`) in progress: its
shared core is section 11; the endpoint mappings land with each endpoint. Roadmap item 04. Scope: a new `cli/` package, a small host-side
addition in `room.js` + a new DOM-free `room/api.js`, a panel in `p2p.html`, `docs/protocol.md`.
Nothing in `engine/`. No `PROTOCOL` bump.

```
$ npx @pooled/cli serve ABCD
pooled serve · room ABCD · Qwen3.6 35B MoE · Q4 (3 devices)
  OpenAI     http://127.0.0.1:8080/v1         (OPENAI_BASE_URL, any API key)
  Anthropic  http://127.0.0.1:8080            (ANTHROPIC_BASE_URL)
  bound to 127.0.0.1 only · no token (set POOLED_TOKEN to require one)
  prompts go to the room's host and may be shown to everyone in the room
```

Any tool that speaks the OpenAI chat API or the Anthropic Messages API to a base URL (Continue,
Open WebUI, LiteLLM, the `openai` / `anthropic` SDKs, curl) then runs on the room's model. A
browser tab cannot accept inbound HTTP, so a small Node process joins the room as one more
**ask-only guest with no layers** and turns HTTP requests into room messages.

## 1. Decisions at a glance

| # | Question | Decision |
|---|---|---|
| 1 | Node WebRTC stack | `peerjs@1.5.4` (the exact client `p2p.html` loads) on top of `node-datachannel` (libdatachannel) via its `polyfill` export. Verified below; `werift` fails the PeerJS handshake |
| 2 | Package / command | `cli/` package **`@pooled/cli`**, bin **`pooled`**, `pooled serve <ROOM CODE> [--port 8080] [--token T]`, bound to 127.0.0.1 only. `pooled` on npm is taken; the `@pooled` scope is free |
| 3 | APIs | OpenAI `POST /v1/chat/completions` (stream + non-stream), `GET /v1/models`; Anthropic `POST /v1/messages` (stream + non-stream), `GET /v1/models` (shape picked by the `anthropic-version` header) |
| 3 | Conversation mapping | An API request is **stateless and separate from the room's chat**: the bridge sends the full `messages` array, the host renders it with the same ChatML code and runs it through `roomGenerate` like Code mode does. The room's chat conversation (`ai.conv`) is never touched |
| 3 | Protocol | `ai-ask` gains `api: 1, rid, messages, params`; `ai-genstart` / `ai-token` / `ai-gendone` / `ai-busy` / `ai-queued` / `ai-stop` gain `rid`; the host advertises `meta.api: 1` in its `hello`. Old peers ignore the new fields, so `PROTOCOL` stays 4 |
| 4 | Room side | "API client *name* joined" in the log and chat; a card marked API; host can disconnect it (`bye`); an **API** panel in the room menu with the command for this room; API asks follow `ai-visibility` like any guest's, except that the asking bridge always gets the full stream |
| 5 | Limits | One generation at a time through the room's existing queue (the bridge keeps one request in flight and queues the rest locally); `max_tokens`, `stop`, `temperature`, `top_k` mapped onto the room's sampler; tools, images, `n > 1`, JSON mode: a clear 400 in v1 |
| 6 | v2 (section 11) | One pipeline, thin adapters: each API parses into one internal request; one new ask, `ai-ask {api: 2}`, negotiated (no `PROTOCOL` bump). The host does everything that needs the model (template profile, structural ids, the tool-call grammar, parsing calls out of the answer, the exact-id cache); the CLI does everything about HTTP |

## 2. The Node WebRTC stack

### What was tried (2026-09-28, on the Spark, aarch64, Node 22.23)

A local `peerjs --port 9123` server, a host page in headless Chromium registering
`pooled-room-PROBE` with the browser `peerjs@1.5.4`, and a Node guest that connects, sends `hello`
and an `ai-ask`, and receives `hello` + a 52-message fake answer stream. Probe files:
`scratchpad/serve/probe/` (not committed).

| Stack | Result |
|---|---|
| `peerjs@1.5.4` + `node-datachannel@0.33.4/polyfill` | **works.** Signaling open 33 ms, data channel open 98 ms, `serialization: "binary"` (BinaryPack, same as the browser), 50 `ai-token`s + `ai-gendone` received, whole run 102 ms on loopback |
| `peerjs@1.5.4` + `werift@0.24.4` | **fails**: `ERROR PeerJS: (Error) invalid sessionDescription` during negotiation (and exporting all of werift onto `globalThis` clobbers `Event`, breaking Node's WebSocket). Would need a PeerJS fork or our own signaling client |
| `node-datachannel` guest against the **real** `p2p.html` host (`?signal=127.0.0.1:9123`, "Create room" clicked by Playwright) | **works**: receives the host's `hello {v: 4}`, `roster`, and `ping`s. The host's negotiated `swarm-wire` channel (id 77) is created on the host side only and is harmless: a guest is never sent frames |

Shims the Node side needs before `import("peerjs")` (all three were required in the probe):

```js
import * as rtc from "node-datachannel/polyfill";
Object.assign(globalThis, rtc);                         // RTCPeerConnection & co.
globalThis.window ??= globalThis;
globalThis.navigator ??= { userAgent: "pooled-cli" };
globalThis.location ??= { protocol: "https:" };        // peerjs util.isSecure() reads it
const { Peer } = (await import("peerjs")).default;     // the CJS bundle: named exports sit on default
```

Why `node-datachannel`:

- It speaks the same SCTP/DTLS as Chrome and Safari (libdatachannel is what the room already
  interoperates with through Chrome's dcSCTP), and PeerJS runs on its W3C-shaped polyfill unmodified.
- Prebuilt binaries ship as `optionalDependencies` per platform (`@node-datachannel/linux-arm64-gnu`,
  `darwin-arm64`, `linux-x64-gnu`, `win32-x64-msvc`, …), so `npx` installs without a compiler.
  1.2 MB JS + one platform binary; `peerjs` 2.2 MB.
- Using the real PeerJS client means the bridge gets the same id rules, the same BinaryPack
  framing and chunking of large messages (a long `messages` array goes over ~16 KB chunks), and the
  same `peer.connect(PREFIX + code, {reliable: true})` path every browser guest uses.

Not chosen: `werift` (above); `@roamhq/wrtc` / `wrtc` (full libwebrtc, 30+ MB, spotty arm64
prebuilds); a Deno-compiled single binary as roadmap 04 first sketched (Deno cannot load the native
addon into a `deno compile` binary portably; revisit when roadmap 11 builds a native peer).

Node ≥ 22 (global `WebSocket`, which PeerJS's signaling socket uses). `engines: {node: ">=22"}`;
on older Node the CLI exits with "pooled needs Node 22 or newer".

## 3. The CLI

### Package

```
cli/
  package.json        name @pooled/cli, bin {"pooled": "bin/pooled.js"}, type module,
                      deps: peerjs 1.5.4 (pinned to p2p.html's), node-datachannel ^0.33
  bin/pooled.js       argv (node:util parseArgs), help, version, Node check
  lib/room.js         the bridge: PeerJS peer, hello/ping/pong, ready state, request map by rid, reconnect
  lib/http.js         node:http server, host/origin/token checks, routing, local queue
  lib/openai.js       request validation + mapping, response and SSE chunk builders
  lib/anthropic.js    same for the Messages API
  lib/sse.js          SSE writer (flush per event, keep-alive comments, client-gone detection)
  README.md           setup for curl, Continue, Open WebUI, LiteLLM, the two SDKs
  npm-shrinkwrap.json   the exact dependency tree, published with the package (npx ignores package-lock.json)
```

No other runtime dependencies (no express, no yargs). `cli/node_modules/` is git-ignored like the
root one; `cli/` is added to `.vercelignore` so the site deploy never uploads it.

### Name

`npm view` on 2026-09-28:

| Name | Status |
|---|---|
| `pooled` | **taken**: 0.0.2 by chrisdickinson, "mutate constructors into a poolable form", last published 2022 |
| `@pooled/*` | **free**: `registry.npmjs.org/-/org/pooled/package` answers "Scope not found" |
| `pooled-cli`, `pooled-run`, `pooled-serve`, `pooledrun`, `@pooled/cli` | free |

Recommendation: create the free npm org `pooled` and publish **`@pooled/cli`** with the bin
`pooled`. The scope also holds the follow-ons roadmap 22 names (`@pooled/client`, the extracted
runtime). The room shows `npx @pooled/cli serve ABCD`; after `npm i -g @pooled/cli` it is
`pooled serve ABCD`. Asking npm for the dormant `pooled` name is optional and must not block this.
Fallback if the org cannot be created: `pooled-cli` (same bin). Nothing is published by this work.

### Command

```
pooled serve <ROOM CODE | room link> [options]

  --port <n>        HTTP port (default 8080)
  --token-file <f>  require the token in this file as "Authorization: Bearer <t>" or
                    "x-api-key: <t>" on every request (or set POOLED_TOKEN)
  --token <t>       the same, given on the command line (other local users can read it with ps)
  --name <s>        how the room shows this client (default: "pooled serve" and 4 random letters;
                    never the hostname, which every guest would see)
  --signal <h:p>    PeerJS signaling server, same as the room page's ?signal= (default: PeerJS cloud)
  --max-queue <n>   HTTP requests waiting here before 429/529 (default 8)
  --quiet / --json-log
  pooled --version, pooled help serve
```

- The code is upper-cased; a room link (`https://pooled.run/r/ABCD`, `…/room?code=ABCD`) is accepted
  and the code extracted.
- **Bound to `127.0.0.1` only.** There is no `--host` flag in v1: the endpoint spends other
  people's GPUs, and exposing it to a LAN is a decision for a later version with a token required.
- Port in use: exit 1 with "port 8080 is busy (the repo's `npm run serve` uses 8080 too): pass --port".
- Without `--token`, any API key is accepted (tools like Continue insist on a non-empty key).
  With `--token`, a missing or wrong key is 401 in the right API's error shape.
- Browser-origin defence, with or without a token: requests whose `Host` is not
  `127.0.0.1:<port>` / `localhost:<port>` are 403 (DNS rebinding), and requests carrying an
  `Origin` header are 403 (a web page must not drive the endpoint through the user's browser).
  No CORS headers are sent. `--allow-origin <o>` can come later if a browser UI needs it.
- Startup: connect, wait for the host's `hello`. If the host's `meta.api` is missing: exit 1,
  "the host of room ABCD runs an older Pooled; reload the host page". If no room answers in 15 s:
  exit 1, "no room ABCD (is the host page open?)", the same words as the browser join.
- Ctrl-C: send `leaving`, destroy the peer, finish open streams with an error event, exit 0.

### Bridge behaviour (`lib/room.js`)

- `hello {name, meta: {api: 1, client: "pooled-cli/<version>", webgpu: false, ua: "API"}, v: 4}`.
  `webgpu: false` already keeps a device out of every deal (`room/plan.js` counts only WebGPU
  memory), so the bridge never gets `ai-load`.
- Answers `ping` with `pong` (the host measures RTT for every card). Ignores `roster`, `ai-map`,
  `ai-history`, `ai-typing`, chat traffic without its own `rid`, and all Code mode messages.
- Ready state: not ready until `ai-ready-all` (the host ignores `ai-ask` before its engine exists
  and sends nothing back: seen in the probe). `ai-degraded` → not ready; `ai-ready-all` /
  `ai-redeal` completion → ready. While not ready, requests get 503 + `Retry-After: 5`.
- Host link lost: in-flight requests end with an error (`503` or an SSE error event); the bridge
  knocks on `pooled-room-<CODE>` every 3 s for a minute, like browser guests, and reconnects with
  `hello {…, back: 1}` when a resumed host is back. `bye {reason}` from the host: print the reason,
  answer every later request 503 with it, and do **not** reconnect.
- One request in flight in the room at a time; the others wait in a local FIFO (`--max-queue`).
  This respects the host's "two per device" queue rule without depending on it.

## 4. The HTTP APIs

Every endpoint maps onto one internal request:

```
{ rid, api: "openai" | "anthropic", client, stream,
  system: string, messages: [{role: "user"|"assistant", text}],
  maxTokens, temperature?, topK?, stop: [string] (≤ 4, each ≤ 64 chars), thinking: bool }
```

`client` is the attribution the room shows: a label from `User-Agent` (`Continue`,
`OpenAI/Python`, `Anthropic/JS`, `curl`, …, capped at 40 chars), else `API`. The OpenAI `user`
field and Anthropic `metadata.user_id` are ignored: tools fill them with account or session ids,
and the label is shown to the room.

### Shared rules

| Input | Mapping |
|---|---|
| `max_tokens` (OpenAI: also `max_completion_tokens`) | `min(request, room context left)`. Default when absent (OpenAI only; Anthropic requires it): 1024. The host caps it at `ctxMax - promptTokens` and reports `length` / `max_tokens` when it hits that |
| `temperature` | `0` → greedy (`exact`). `> 0` → top-k sampling at that temperature. OpenAI accepts 0..2, Anthropic 0..1 (400 outside). Absent → the room's sampling preset |
| `top_k` (Anthropic; also accepted as an OpenAI extension) | clamped to 1..64 (`engine/topk.js` `TOPK_MAX`, so GPU sampling keeps working). Absent → 40, the creative preset's |
| `top_p` | accepted and ignored (the sampler has no nucleus cut; noted in the README). `seed`, `presence_penalty`, `frequency_penalty` = 0, `logit_bias` = {} likewise accepted; non-default penalties or bias → 400 |
| `stop` / `stop_sequences` | a string or up to 4 strings, applied on the host (below) |
| thinking | off unless asked: OpenAI `reasoning_effort` other than `none`/`minimal`; Anthropic `thinking: {type: "enabled"}`. Off renders the pre-closed think block (`buildIds(thinking=false)`), same as the chat |
| `model` | any string accepted; responses carry the room's model id (`pooled/<model key>`, e.g. `pooled/qwen3.6-35b-moe`, the `MODELS` key) |
| tools | `tools`, `functions`, `tool_choice` (other than `"none"` or `"auto"`, which with no tools ask for nothing), messages with role `tool`/`function`, `tool_calls`, Anthropic `tool_use`/`tool_result` blocks → **400** "tool calls are not supported by pooled serve yet (v1 is chat only)" |
| other content | image / audio / file / document parts → 400 "only text content is supported". `n` > 1 → 400. `logprobs` → 400. `response_format` other than `{type: "text"}` → 400 "JSON mode is not supported yet" |
| roles | OpenAI `system` and `developer` messages anywhere are joined (in order, blank line between) into `system`; Anthropic `system` (string or text blocks) likewise. The rest must be `user`/`assistant`; consecutive same-role messages are merged; the last must be `user` (an assistant prefill last is 400 in v1) |
| size | body ≤ 1 MB (413); total text ≤ 400 k chars. The host rejects a prompt longer than the room's context minus 32 tokens with **400 `context_length_exceeded`** (OpenAI) / `invalid_request_error` "prompt is too long: N tokens > M maximum" (Anthropic). No silent trimming: unlike the chat, the API never drops old turns |

Errors, OpenAI shape: `{"error": {"message", "type", "param", "code"}}`; Anthropic shape:
`{"type": "error", "error": {"type", "message"}}`. Status map:

| Situation | OpenAI | Anthropic |
|---|---|---|
| bad request / unsupported feature | 400 `invalid_request_error` | 400 `invalid_request_error` |
| bad or missing token (`--token` set) | 401 `invalid_api_key` | 401 `authentication_error` |
| foreign Host / Origin | 403 `permission_error` | 403 `permission_error` |
| unknown path | 404 | 404 `not_found_error` |
| body too large | 413 | 413 `request_too_large` |
| local queue full, or the room's queue full (`ai-busy`) | 429 `rate_limit_exceeded` | 529 `overloaded_error` |
| room not ready / degraded / host gone / kicked | 503 | 529 `overloaded_error` (message says why) |
| generation failed on the host | 500 `server_error` | 500 `api_error` |

### OpenAI: `POST /v1/chat/completions`

Non-stream response:

```json
{ "id": "chatcmpl-<rid>", "object": "chat.completion", "created": 1790000000,
  "model": "pooled/qwen3.6-35b-moe", "system_fingerprint": null,
  "choices": [{ "index": 0,
                "message": { "role": "assistant", "content": "…", "refusal": null },
                "logprobs": null, "finish_reason": "stop" }],
  "usage": { "prompt_tokens": 812, "completion_tokens": 143, "total_tokens": 955 } }
```

With thinking on, `message.reasoning_content` carries the think text (the field vLLM, DeepSeek and
LiteLLM use) and `content` only the answer.

Stream (`"stream": true`): `Content-Type: text/event-stream`, one `data: <json>\n\n` per chunk, no
`event:` lines, all chunks share `id`, `created`, `model`:

```
data: {"id":"chatcmpl-r7","object":"chat.completion.chunk","created":1790000000,"model":"pooled/qwen3.6-35b-moe","system_fingerprint":null,"choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}

data: {…,"choices":[{"index":0,"delta":{"content":"Hello"},"logprobs":null,"finish_reason":null}]}

data: {…,"choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}

data: {…,"choices":[],"usage":{"prompt_tokens":812,"completion_tokens":143,"total_tokens":955}}   (only with stream_options.include_usage)

data: [DONE]

```

- One content chunk per `ai-token` the host sends (it is already per token; no extra batching, so
  time-to-first-token is the room's).
- Thinking: `delta.reasoning_content` chunks before the `content` ones.
- While waiting in a queue: SSE comment lines `: queued, 2 ahead\n\n` every 10 s (clients ignore
  comments; they keep proxies from timing out).
- A failure after the headers went out: `data: {"error": {…}}\n\n`, then the connection closes
  without `[DONE]` (what OpenAI does).

`finish_reason`: `stop` (end token or a stop string), `length` (`max_tokens` or context full).
When the host presses Stop the request ends as an error (503, or an error chunk and no `[DONE]`
while streaming): a cut-off answer must not read as a finished one.
A stop pressed by the host ends with `stop` as well.

### OpenAI: `GET /v1/models`

`{"object": "list", "data": [{"id": "pooled/qwen3.6-35b-moe", "object": "model", "created": <room start>, "owned_by": "pooled"}]}`.
Empty `data` before the room's model is ready. `GET /v1/models/{id}` returns the one entry or 404.

### Anthropic: `POST /v1/messages`

Required: `model`, `max_tokens`, `messages`. `anthropic-version` is accepted and not enforced.
Non-stream response:

```json
{ "id": "msg_<rid>", "type": "message", "role": "assistant", "model": "pooled/qwen3.6-35b-moe",
  "content": [{ "type": "text", "text": "…" }],
  "stop_reason": "end_turn", "stop_sequence": null,
  "usage": { "input_tokens": 812, "output_tokens": 143 } }
```

With thinking on, a `{"type": "thinking", "thinking": "…", "signature": ""}` block comes before
the text block (the signature is empty: there is nothing to verify against; the official SDKs do
not check it client-side).

Stream: `event: <type>\ndata: <json>\n\n`, in exactly this order:

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_r7","type":"message","role":"assistant","model":"pooled/qwen3.6-35b-moe","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":812,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type":"ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":143}}

event: message_stop
data: {"type":"message_stop"}

```

- `message_start` is sent when the host's `ai-genstart` arrives (it carries the prompt token
  count, so `input_tokens` is exact). While queued: `event: ping` every 10 s.
- Thinking on: block 0 is `{"type": "thinking", "thinking": ""}` with `thinking_delta` deltas, a
  `signature_delta` with `""`, `content_block_stop`; then the text block at index 1.
- Failure mid-stream: `event: error\ndata: {"type":"error","error":{"type":"api_error","message":"…"}}`, close.

`stop_reason`: `end_turn` (end token), `stop_sequence` (with `stop_sequence` set to the one that
matched), `max_tokens`, `model_context_window_exceeded` (the context is full). When the host
presses Stop the request ends as an error (529, or an `error` event and no `message_stop`).

### Anthropic: `GET /v1/models`

When the request carries `anthropic-version` (or `x-api-key`), `/v1/models` answers in
Anthropic's shape: `{"data": [{"type": "model", "id": "pooled/qwen3.6-35b-moe", "display_name": "Qwen3.6 35B MoE · Q4 (Pooled room ABCD)", "created_at": "<ISO>"}], "has_more": false, "first_id": "pooled/qwen3.6-35b-moe", "last_id": "pooled/qwen3.6-35b-moe"}`.

`POST /v1/messages/count_tokens` → 404 `not_found_error` in v1 (it would need a host round trip;
it is listed as a follow-up because Claude Code calls it, although Claude Code also needs tools).

### Also

`GET /health` → `{"ok": true, "room": "ABCD", "connected": true, "ready": true, "model": "pooled/qwen3.6-35b-moe", "queue": 0}`
(no auth, for scripts waiting on startup). `GET /` → a one-screen plain-text banner with the base URLs.

## 5. Conversation mapping and the protocol addition

### What the room does today

- `ai-ask {text, name}` carries **one question**. The host appends it to its single conversation
  (`ai.conv.turns`) and prefills only what follows `ai.fed` (`room/conversation.js`). Chat history
  is the host's, not the asker's.
- Code mode does not use that conversation: it builds its own ids and calls `roomGenerate(ids, …)`
  under the room lock. Caches follow whichever ids were fed last; `ckptResume` reloads a saved
  checkpoint when an earlier prompt is a prefix, and otherwise the room resets and re-prefills.
  Correctness never depends on which conversation ran last.

An API client, by contrast, owns its history and resends all of it on every call. So:

**An API request never reads or writes `ai.conv`.** The host renders `{system, messages}` to ids
with the same `buildIds` the chat uses and runs `roomGenerate` on them, exactly as Code mode does.
Consequences:

- The room's chat keeps its own memory; an API exchange shown in the chat is marked "via API ·
  not part of this chat's memory". The next chat question still reuses the chat's checkpoint
  (`?ckpt` keeps the last 2 answers), or re-prefills if it was evicted.
- No new state to keep in sync, no "reset" message, no risk that an API client's history leaks
  into the room's chat or the other way round.

**Multi-turn stays cheap.** A client's turn *n+1* resends turn *n*'s answer as text. Re-tokenizing
text can split differently from the ids that were sampled, which would break the prefix match and
cost a full re-prefill every turn. The host keeps the last 8 API answers as `text → sampled ids`
(`room/api.js`, LRU, host memory only) and substitutes the exact ids when an assistant message's
text matches one byte for byte. Then `reusablePrefix(ai.fed, ids)` holds and only the new user turn
is prefilled, as in the chat. A mismatch (the client edited the answer, trimmed whitespace) just
costs a re-prefill; it is never wrong.

### Wire additions (no `PROTOCOL` bump)

| Message | Direction | Addition |
|---|---|---|
| `hello` | host → guest | `meta.api: 1` when the host understands API asks. The bridge refuses to serve without it |
| `hello` | bridge → host | `meta: {api: 1, client, webgpu: false}`; the host labels the device an API client |
| `ai-ask {api: 1, rid, messages, system, params}` | bridge → host | `rid`: bridge-chosen id (≤ 32 chars). `messages: [{role, text}]`, `params: {maxTokens, temperature?, topK?, stop?, thinking?, client}`. No `text` field, so an old host that ignores `api` drops it on its empty-text check instead of asking a chat question |
| `ai-queued {pos, rid}` | host → bridge | as today, plus `rid` |
| `ai-busy {why, rid, code?}` | host → bridge | as today, plus `rid`; `code`: `queue` (full), `ctx` (prompt too long, with `n`, `max`), `bad` (validation), `off` (API clients disabled), `degraded`, `loading` |
| `ai-genstart {…, rid, api: 1, client, promptTokens}` | host → all | the asker always gets it in full; others per `ai-visibility`. `text` is the last user message, capped at 2,000 chars for the screens |
| `ai-token {text, d, rid, th?}` | host → asker (+ screens) | `th: 1` marks think-block text. Stop-string hold-back is already applied |
| `ai-gendone {…, rid, reason, stopSeq?, usage: {in, out}}` | host → all | `reason`: `stop`, `stop_seq`, `max`, `ctx`, `abort`, `error` (+ `err` message) |
| `ai-stop {rid}` | bridge → host | stop this request: honoured when it is running for this asker (as today) **or** still in the host's queue (removed, answered with `ai-busy {rid, code: "gone"}`). Sent when the HTTP client disconnects |

The change to `sendChat`: for an API ask, the asker is added to the `full` set whatever the
visibility mode (today under `host` the asker would get only hidden stand-ins, which would leave
the API client with no tokens). Nothing else about visibility changes.

Old peers: a browser guest on an older deploy ignores `rid`/`api`/`th`/`usage` (extra fields on
known messages). An old host never advertises `meta.api`, so a new bridge never sends it an API ask.
Frames, hello version and every existing message keep their meaning, which is why this is not a
protocol change under GOVERNANCE.md's definition. `docs/protocol.md` gets a section "API clients".

### Host-side flow (`room/api.js`, DOM-free, unit-tested; wired from `room.js`)

1. `aiOnData` `ai-ask` with `api: 1` → `apiAsk(from, d)`: reject unless the sender said hello with
   `meta.api` and "Allow API clients" is on; validate shapes and sizes (≤ 200 messages, ≤ 400 k
   chars, roles, `stop` ≤ 4 × 64 chars, numbers in range); then join **the same queue** as chat
   questions (`aiAsk`'s queue, `QUEUE_MAX` 10, two per device), with an entry `{api: d, from}`.
2. `nextQueued` → `apiGenerate(entry)`, a sibling of `aiGenerate`: take `ai.busy = "gen"`, build
   ids (`buildIds(tok, {system, turns, thinking})`, exact-id substitution), check the context
   (no trimming: over → `ai-busy {code: "ctx"}`), send `ai-genstart`, and call
   `roomGenerate(ids, {onToken, stop: {imEnd, eot}, maxNew, sample, signal})`.
3. `onToken` decodes the token, appends to the answer text, and runs the stop matcher: it holds
   back the longest tail that is a prefix of any stop string, emits the rest, and on a full match
   emits the text before it, sets `stopSeq`, and aborts through the `AbortController` passed as
   `signal` (decoding ends after the lap in flight; the extra tokens are simply not emitted).
   Think-block text is tagged `th: 1` (`splitThink` logic, streaming).
4. Sampler: `makeSampler({temp, topK})` in `room/sampling.js` (the presets become calls to it);
   `temp` 0 → greedy, else top-k at `temp`, with the same `.gpu` descriptor so GPU sampling applies.
   Absent temperature → `pickSampler(ai.settings.sampling)`.
5. Done: `ai-gendone {rid, reason, usage}`; record `text → ids` for exact-id reuse; push a
   transcript item flagged `api` (so late joiners see it under `all`, marked); release the lock;
   `nextQueued`. `saveHost` does not store API exchanges.

A Code mode run holds the lock for all its steps, so API asks queue behind it like chat questions
do. The host's Stop button stops an API answer like any other.

## 6. Room UI

- **Join.** Log: "API client *name* joined". Chat note (when the chat is on screen): same line.
  The card uses an API glyph, the subtitle "API client · *client*", no GB, no memory stepper, and
  is not counted in "N devices" (the header reads "3 devices · 1 API client").
- **Disconnect (host only).** The card's menu gets "Disconnect". It sends
  `bye {reason: "the host disconnected this API client"}`, closes the link, and remembers the
  peer id for the session so an automatic reconnect with `back: 1` is refused. Restarting the CLI
  gets back in unless the host has turned API clients off.
- **Allow API clients** (host setting, room menu, default **on**, saved with the room for resume).
  Off: new API hellos get `bye {reason: "the host does not allow API clients in this room"}`,
  connected ones are disconnected. Default on because anyone with the code can already join as a
  guest and ask; the bridge adds automation, not access.
- **API panel.** A room-menu entry "Use from code" on every device (any member may run the bridge
  on their own machine), opening a panel with:
  - `npx @pooled/cli serve ABCD` with a copy button (plus `--signal host:port` when the page runs
    with `?signal=`);
  - the base URLs: OpenAI `http://127.0.0.1:8080/v1`, Anthropic `http://127.0.0.1:8080`, and one
    curl line;
  - "Connected API clients": name, client label, requests answered, with Disconnect on the host;
  - one sentence: "Runs on your computer; requests use this room's GPUs and appear in the chat
    under the room's visibility setting."
  Not a third tab next to Chat | Code: it is set up once and has no live content worth a tab. As
  built, it is the API half of the dark Serve API page (`#compute-screen`, opened by the black
  **Serve API** button in the room's header), beside this device's layers and passes; with a client
  connected the steps fold under "How to connect". `textContent` only.
- **Chat.** An API request appears as a normal exchange: the asker line reads
  "*name* · *client* (API)", the text is the last user message (2,000 chars, "…"), and the note
  "via API · not part of this chat's memory". It follows `ai-visibility`: `all` → everyone sees it;
  `host` → only the host's screen; `asker` → the host (the asker is the bridge). Reactions,
  Continue and Regenerate are hidden on API exchanges (their history is the client's).

## 7. Limits

| Limit | Value | Where |
|---|---|---|
| generations in the room | 1 at a time; the chat, Code and API share one lock | host |
| room queue | 10 total, 2 per device (unchanged) | host |
| bridge in-flight | 1; local FIFO `--max-queue` 8, then 429 / 529 | bridge |
| `max_tokens` | default 1024 (OpenAI), capped by the context left | bridge + host |
| prompt | context − 32 tokens, else 400; ≤ 200 messages, ≤ 400 k chars, body ≤ 1 MB | both |
| `stop` | ≤ 4 strings, ≤ 64 chars each | both |
| idle stream | keep-alive every 10 s while queued | bridge |
| request wall time | none by default (the room's per-lap timeouts already fail a stuck chain); the client may close, which sends `ai-stop` | bridge |

Out of scope for v1, each with a clear 400: tool calls / function calling, images, `n > 1`,
logprobs, JSON mode / structured outputs, assistant prefill, embeddings (`/v1/embeddings` 404),
legacy `/v1/completions` (404), `/v1/responses` (404), `count_tokens`.

## 8. Tests

- **Unit (Deno, `tests/unit/`)**: `serve_openai_test.js`, `serve_anthropic_test.js`: request
  validation and mapping (roles, merges, tools/image rejection, param ranges), and golden SSE
  transcripts for both APIs (a fixed `ai-genstart/ai-token×N/ai-gendone` sequence in, byte-exact
  SSE out, including thinking, stop sequence, length and error endings). `api_host_test.js` for
  `room/api.js`: stop-string hold-back across token boundaries, exact-id substitution, context
  rejection, sampler descriptors.
- **E2E (`tests/e2e/serve.mjs`, through `gpurun.sh`)**: local PeerJS server
  (`node_modules/.bin/peerjs`, as `xroom.mjs`), host = headless Chromium on `p2p.html?signal=…`
  with Qwen3 1.7B (`models/qwen17`), then `pooled serve <code> --port <free>`:
  1. curl non-stream and stream on both APIs; the official `openai` and `@anthropic-ai/sdk`
     clients (cli devDependencies) parse both;
  2. `temperature: 0` twice gives identical text; a turn-2 request resends turn 1 and the host
     reports `reused > 0` (exact-id reuse works);
  3. two concurrent requests: the second gets queue keep-alives and completes after the first;
  4. `stop: ["\n"]` ends at the first line with `finish_reason: "stop"` / `stop_sequence`;
  5. `tools` → 400 with the message; a foreign `Origin` → 403; `--token` → 401 without it;
  6. visibility `host`: the bridge still streams; a second headless guest sees hidden stand-ins;
  7. host clicks Disconnect → in-flight stream gets an error event, then 503s; the bridge does
     not reconnect;
  8. client closes mid-stream → the host logs the stop and the room is free within a lap.
- `npm run check` extended to `node --check cli/**/*.js`.

## 9. Build order

1. `room/api.js` + sampling `makeSampler` + unit tests (no UI, no wire).
2. Host wiring in `room.js`: `meta.api` in hello, `ai-ask {api}`, `apiGenerate`, `sendChat`
   asker rule, `ai-stop {rid}` for queued entries; protocol.md section.
3. `cli/` bridge + OpenAI endpoints; e2e 1-3.
4. Anthropic endpoints; stop/thinking; e2e 4-8.
5. Room UI: card, Disconnect, Allow toggle, "Use from code" panel.
6. README for the CLI (Continue, Open WebUI, LiteLLM, SDK snippets); roadmap 04 status; npm org
   and publish left to the maintainer.

## 10. Open questions

- **Port 8080** is the default asked for, and also what `npm run serve` uses for the site in this
  repo and what many dev servers pick. The clear "port busy" message covers it; 11435 (roadmap 04's
  first sketch) is the alternative if collisions show up in practice.
- **Tool calls**: resolved by v2, section 11.
- **Default thinking when a client says nothing** (Codex `reasoning: null`, opencode): off, as v1;
  a `pooled serve --reasoning` flag if agents want it on.
- **`count_tokens`** (`/v1/messages/count_tokens`, `/v1/responses/input_tokens`): needs a render-only
  ask on the host; added if Claude Code's compaction needs it.
- **Several pinned checkpoints**: two agents in one room each want their system prompt + tools pinned;
  v2 keeps one pin and does not replace a pin another client used in the last 10 minutes.
- **Several API clients** each hold one in-flight request, so N bridges can take up to N of the 10
  queue slots. Fine for v1; a per-room API share could come with roadmap 07.

## 11. v2: the shared core (tools, structured output, reasoning)

Built on `feat/serve-tools-core`. The endpoint mappings (Chat Completions, Messages, Responses) sit
on top of it, each in its own adapter file, and are documented with them.

### 11.1 The pipeline

```
 HTTP ─ adapter.parse ─► InternalReq ─ common.finishRequest ─ http.js queue ─ ai-ask {api: 2} ─►  host: validateApiAsk (v2)
                                                                                                  apiPrompt2: templateProfile + renderApi
                                                                                                  apiRun2: grammar sampler → roomGenerate
   ◄── adapter.encoder (SSE) / adapter.final (JSON) ◄── answer.js Ask ◄── ai-token / ai-call / ai-gendone ◄┘
       ThinkSplit → CallStream → StopMatcher
```

- **CLI** (`cli/lib/`): `common.js` (the internal request, `normalizeMessages`, `finishRequest`,
  `needsV2`, `askBody`, `outcome`, ids, the `pooled1.` reasoning blob, limits), `answer.js` (the
  bridge's checks and the `Encoder` / `Collector` contract), `http.js` (routing to the adapters in
  `ADAPTERS`, keep-alives on any 10 s of silence, the 413 caps, negotiation), `room.js` (`hostApi`,
  `ai-call` routing, the host's `ctx`). `openai.js` / `anthropic.js` are today's behavior on the
  adapter contract; `responses.js` is a stub (404 "not built yet").
- **Host** (`room/api.js`, `room/conversation.js`, `harness/`): below.

### 11.2 Template profiles and structural rendering (`room/conversation.js`)

`templateProfile(chatTemplate)` reads what the model's GGUF template does, once per loaded model:
`style` (`json` Hermes calls for Qwen3; `xml` `<function=…><parameter=…>` for Qwen3.5+), whether the
generation prompt opens the think block itself (`thinkInPrompt`: Qwen3.6 / 3.8 yes, Qwen3 no),
which past turns keep their think block (`thinkRule`: `all` for Qwen3.8, `afterQuery` for Qwen3.6,
`afterQueryNonEmpty` for Qwen3), the Qwen3.8 reasoning-effort sentence, and whether text is
trimmed. Nothing is keyed on a model name; a model with no template in hand (the 1.7B loads a
`tokenizer.json`) gets Qwen3's rules.

`renderApi` renders the conversation as the template does, **structurally**: every tag the template
writes (`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, `<tool_call>`, `</tool_call>`,
`<tool_response>`, `</tool_response>`) is its special id, including the tags in the template's own
tool instructions, while every client string (system text, messages, tool results, reasoning,
arguments, the tool JSON) goes through plain `tok.encode`, so no client text can become a special
token. Tools are listed with Python's `json.dumps` separators (`pyJSON`), past calls rendered as the
template renders them. Tested byte for byte against the three GGUF templates rendered by jinja2
(`tests/fixtures/api/render.json`, 66 cases).

Rendering the tags in the tool instructions as special tokens matters in practice: with them
spelled out in text, Qwen3 1.7B, Qwen3.5 2B and Qwen3.6 35B all wrote `<tool_call>` back as text
pieces, the 1.7B skipped the tool, and both Qwen3.5+ models broke parallel calls
(`</function>` followed by more parameters); with the special tokens all three called correctly
(llama.cpp recordings, `tests/e2e/serve_record.mjs`).

Deviations from the templates, each deliberate: mid-conversation system messages fold into the
user turn before them (Qwen3.6 / 3.8 raise on them; Claude Code sends them); text sent along with
tool results is its own user turn but not a new query (`aside`); `tool_choice: "none"` leaves the
tools out of the prompt (with them listed, Qwen3.6 wrote call markup in any spelling the grammar
had not banned, e.g. `<tool.call>`); and with thinking off every past answer keeps the pre-closed
empty think block it was sampled after (as v1's `buildIds`; Qwen3 and Qwen3.6 drop it before the last
query), so each prompt stays an extension of the last one and the room's caches reuse it all.

### 11.3 The grammar (`harness/constrain.js` `GrammarConstraint`, `harness/jsonschema.js`)

Every v2 answer with tools or a format is sampled under a grammar over the whole answer: free
reasoning; then, by mode, free text with calls (`auto`; after `</tool_call>` only whitespace,
another call or the end), no calls (`none`), a call first (`required`, named), or a JSON value
(`format`, or a call or the value for `auto` + format). Calls follow the model's format with
declared names (narrowed by `allowed`), each parameter once, required ones before `</function>`,
typed values (string-capable values raw, string enums unquoted, everything else a JSON value of its
schema). After the allowed number of calls only the end token is left. The JSON subset: types,
`properties`, `required`, `additionalProperties`, `items` / `prefixItems`, `enum` / `const`,
`anyOf` / `oneOf`, `allOf` (objects), `nullable`, local `$ref`; bounds and patterns are accepted,
not enforced. Whitespace: one space, or a newline and indentation. Schemas are capped (10 k nodes,
enum 1 k, anyOf 64, allOf 16, depth 32).

Tags that are special tokens are atomic symbols in the grammar: structure can only be written with
the real token, never spelled out. XML models may open a call inside the reasoning (it closes it);
for Qwen3 a call-like draft inside the reasoning stays reasoning. In the forcing modes the end token
is banned in the reasoning and it gets a default budget of min(max_tokens / 2, 4096) tokens, after
which the host closes the block and the grammar engages.

Masks are cached per grammar state in one LRU per tokenizer (64 MB). Before scanning the vocabulary
for a new state, the mask checks the model's top 64 candidates: when enough of them are allowed for
the sampler's top-k, masking only those is exact (tested against the full mask on 10,000 random
logit vectors), so most steps never scan. The garbage guard counts forced tokens only where the text
is the model's own choice (a value's contents), plus NaN logits and a vanishing allowed probability.

Recorded with llama.cpp's top-64 candidates through the host's whole pipeline, all modes work on
Qwen3 1.7B and Qwen3.6 35B: parallel calls, typed and nested arguments, `required`, named, `none`,
one call when parallel is off, JSON schema, reasoning then calls (`-g-` fixtures).

### 11.4 Parsing and streaming (`harness/tools.js` `CallStream`, `room/api.js` `apiRun2`)

The answer runs through `ThinkSplit` (reasoning → `ai-token {th}`), then `CallStream` (content →
`StopMatcher` → `ai-token`; calls → `ai-call {name}`, `{a}`, `{end}`). A call's name goes out once
complete and declared; its argument fragments concatenate exactly to its final arguments: string
values stream as JSON strings as they are written, arrays and objects as their JSON text, values
that need coercion (numbers, booleans, null, unions) at `</parameter>`. Stop strings apply to
content only. `<tool_response>` ends the answer. An answer cut inside a call reports it as `open`.
Without a grammar (never on the v2 path) a body in another shape is parsed whole at `</tool_call>`,
and one that does not parse becomes content.

Known limit: a string value containing `\n</parameter>` ends there (no lookahead).

### 11.5 The exact-id cache (`room/api.js` `TurnCache`)

Keyed by (model, hash of the history before the answer, the answer's canonical form: trimmed
content and each call's name and sorted arguments); the value is the ids after `assistant\n` as the
caches hold them (the header's think part, the reasoning, the injected budget close). 512 answers,
2 M ids, cleared when a model loads. A client that sends reasoning back hits only with the same
reasoning. With it, step k+1's prompt is step k's prompt, its answer and the new tool results, so
the room prefills only those (tested on the recordings, thinking on and off). A host-side encode
cache (32 MB) keeps agents' long histories from being re-tokenized every step.

### 11.6 Limits (v2)

| Limit | Value | Where |
|---|---|---|
| body | 4 MB | CLI 413 |
| the ask as sent to the room | 3.5 MB of JSON (PeerJS drops a message it cannot rebuild, ~4 MB) | CLI 413 |
| text | 1.5 M chars; early 400 when over 8 × the host's context | CLI and host |
| messages | 1000 | CLI and host |
| tools | 128; names `[A-Za-z0-9_.:-]{1,128}`, unique; schema ≤ 32 k chars; XML parameter names without `<`, `>`, line breaks | CLI and host |
| calls | 16 per answer (grammar), 64 accepted from the host | host, CLI |
| mask cache / encode cache / exact-id cache | 64 MB / 32 MB / 512 answers, 2 M ids | host |

### 11.7 Tests

Unit (Deno): `jsonschema_test.js`; `constrain_test.js` (modes, typed values, whitespace, caps, the
LRU, the fast path, the garbage guard, on top of Code mode's cases); `tools_test.js` (pyJSON,
coercion, `CallStream` at every split point); `api_host_test.js` (v2 validation, profiles, renderApi
against the templates, injection, every recording replayed through `apiRun2`, the grammar accepting
every token of the well-formed recordings and stopping the malformed ones, prefix reuse through the
cache); `serve_common_test.js` (normalization including Claude Code's system-message shapes, the
caps, the ask bodies, the bridge's checks). CLI (`node --test`): negotiation, v2 calls through an
adapter, the old-host 400, the 413 caps, keep-alives mid-stream.

GPU (`tests/e2e/serve_v2_smoke.mjs`, a real host room through the CLI's Bridge; 2026-09-29 on the
Spark): 17 of 17 checks on Qwen3 1.7B and on Qwen3.6 35B MoE (v1 ask unchanged, v2 plain, auto call,
tool-result follow-up reusing the caches, parallel, parallel off, required, named, none, JSON schema,
reasoning then a call, an older CLI's v1 asks, a bad named tool refused). `tests/e2e/serve.mjs` (v1
endpoints, now v2 asks underneath) passes all 57 checks.

Recordings: `node tests/e2e/serve_record.mjs --llama URL --model M --gguf F [--grammar]` against a
llama.cpp server (CPU is fine) writes `tests/fixtures/api/<model>-[g-]<case>.json`.

## 12. Chat Completions on v2 (`cli/lib/openai.js`)

Built on `feat/serve-chat` over the v2 core. It replaces the "tools" and "other content" rows of the
table in section 4 for `/v1/chat/completions`; the other rows hold as written.

**Request.** `tools` (function tools; a missing `parameters` is `{type: "object", properties: {}}`;
`strict` accepted, always true in effect; an empty list means no tools) · `tool_choice` `auto` /
`none` / `required` / `{type: "function", function: {name}}` (the flat `{type: "function", name}`
too) / `{type: "allowed_tools", allowed_tools: {mode, tools}}` → `allowed` · `parallel_tool_calls`
· assistant messages with `tool_calls` (arguments that are not a JSON object go as `{}`, logged)
and `reasoning_content` or `reasoning` (fed back as that turn's reasoning) · `role: "tool"`
messages with `tool_call_id` (text parts joined; an image part becomes a note), put back in the
calls' order · `response_format` `json_object` → `{type: "json"}`, `json_schema` → `{type:
"schema"}` · `reasoning_effort` `none` / `minimal` → off, `low` … `max` → on with that effort (other
values 400); absent, `chat_template_kwargs.enable_thinking` decides · `max_completion_tokens` over
`max_tokens` · `stream_options.include_usage` only with `stream`. System and developer messages
before the first other message are the system prompt; later ones fold into the user turn before
them (`normalizeMessages`), where v1 joined them all into the system prompt. Ignored: `seed`,
`store`, `metadata`, `user`, `safety_identifier`, `prompt_cache_key`, `service_tier`, `verbosity`,
`top_p`. 400: `custom` tools and calls, `functions` / `function_call` / `role: "function"`, `n > 1`,
logprobs, audio, `prediction`, `web_search_options`, non-text user content, an assistant message
last. `tool_choice` errors use vLLM's messages ("When using `tool_choice`, `tools` must be set.",
"The tool specified in `tool_choice` does not match any of the specified `tools`").

**Response.** `message = {role, content, refusal: null, reasoning_content?, tool_calls?}`;
`content: null` when there are calls and no text; `tool_calls[i] = {id: "call_<24>", type:
"function", function: {name, arguments}}`; `finish_reason` from `common.outcome`: `tool_calls`,
`stop` (also for a named `tool_choice`, as OpenAI and vLLM), `length` (`max_tokens` or the context);
a call cut by `max_tokens` is left out. `usage` adds `completion_tokens_details.reasoning_tokens`
when thinking was on and `prompt_tokens_details.cached_tokens` when the room reused its caches.

**Stream.** The role chunk, then one chunk per room message: `reasoning_content`, `content`, a call's
opening `{tool_calls: [{index, id, type: "function", function: {name, arguments: ""}}]}`, its
argument fragments `{tool_calls: [{index, function: {arguments}}]}` (they join to the final
arguments exactly), the finish chunk, the usage chunk with `include_usage`, `[DONE]`. A call cut by
`max_tokens` has already streamed its name and partial arguments; `finish_reason: "length"` says so.

**Tests.** `cli/test/openai_test.mjs` (the ask the room gets for a full agent request, the whole and
streamed wire format byte for byte, the OpenAI SDK's `stream().finalChatCompletion()` and a streamed
`runTools` loop, every 400, an older host) and `tests/unit/serve_openai_test.js`; on the GPU,
`tests/e2e/serve.mjs` "chat tools".
