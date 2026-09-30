# Code mode: the in-browser coding agent

Status: design, 2026-09-26. Branch base: `feat/engine-opt` @ 04531cb.

What we are building: a **Code** mode in the room page. The host types "build a Tetris game";
the room's model (spread over everyone's GPUs) runs our own light agent loop (`harness/`),
writes files into a browser-side project, then **serves** them on a virtual port (`:5173`).
The app runs in a preview pane on the host's screen and, from the same file snapshot, on every
peer's screen. The agent reads the preview's console errors and fixes its own bugs. Nothing runs
on a server: static hosting plus the PeerJS signaling server, as today.

Decisions taken before this doc (not revisited): our own harness, not opencode; the serve +
preview + console loop is the key feature; context stays frugal (16k on the 27B, 32k on the MoE,
`ai.engine.maxSeq` / `ctxMax()`).

Contents: A. room model adapter, B. preview server, C. tools, D. workspace, E. UI,
F. context budget, G. tests, H. work split and interfaces, then risks.

---

## A. Room model adapter

With `?hcore=1` the agent's model calls go through `pooled serve`'s v2 core instead of the adapters
below (the template's tool prompt, the strict grammar, `CallStream`): see [harness-core.md](harness-core.md).

### A.1 The problem

`harness/agent.js` needs `generate({ system, turns, signal }) -> async iterable of text deltas`.
`harness/engine-model.js` provides that over one local engine, and cannot drive a room: in a
chain the host engine holds only `ranges[0]` + embed/head, so its `prefillTokens` /
`forwardToken` / `specStep(next, pick, K)` (no `{runTrunk, onReject}`) would run the host's layers
only, never send frames, and bypass `ai.fed`, `ai.pos`, `pendingCtl`, checkpoints and `ai.busy`.

The room's distributed generation lives inside `aiGenerate` (`room.js:1750-1979`), mixed with chat
UI. So: split it.

### A.2 room.js refactor (implementer B)

**1. `roomGenerate(ids, opts)`** — new, next to `aiGenerate`, UI-free, host only. It is lines
~1789-1946 of `aiGenerate` with the UI calls replaced by callbacks.

```js
// The room's generation core: prefill ids (reusing whatever the caches or a checkpoint already
// hold), then decode until a stop token, maxNew, a full context or an abort. Host only; the caller
// holds the lock (ai.busy). Throws on failure after leaving the room in a clean state.
async function roomGenerate(ids, {
  onToken,              // (id, drafted: 0|1|2) => void, per emitted token, in order
  stop,                 // Set<number>: tokens that end the answer (not emitted, not piped)
  maxNew,               // answer cap, tokens
  sample,               // (logits: Float32Array) => id; a wrapper may mask (tool-name constraint)
  signal,               // AbortSignal (ai.abort still works too)
  onStatus = () => {},  // (text) => void: "prefill: N new tokens (M reused)…", "generating… N tok · X tok/s"
}) -> {
  tokens,               // number[]: emitted ids, verbatim (no stop token)
  reason,               // "stop" | "max" | "ctx" | "abort"
  reused, prefilled,    // prompt tokens taken from the caches / computed
  count, tps, acc, copied, tPre, tDecode,
  stats,                // the same one-line stats string the chat shows
}
```

What moves into it, unchanged in behaviour:
- `ckptResume(ids, reusablePrefix(ai.fed, ids))`, the Continue fast path through `ai.pending`
  (generic: it only fires when `ai.fed` equals `ids` exactly and `ai.pending.at === ai.pos`),
  `resetState()` when nothing is reusable, the `mtpRun` draft-row fill for a follow-up.
- `aiPrefill(ids.slice(reused))`.
- Both decode paths: the speculative one (`spec = ai.chain.length ? {runTrunk, onReject} : {}`,
  `pickK`, prompt lookup via `lookupDrafts` / `specStepDrafts`) and the plain one.
- `eos(t)` becomes `stop.has(t)`; `ai.abort` becomes `aborted()` = `ai.abort || signal?.aborted`;
  `emit` becomes `onToken` plus the internal count; `aiStatus` calls become `onStatus`.
- Bookkeeping: `ai.fed` pushes, `ai.pos = ai.engine.pos`, `ai.xAt`, `ai.pending`, `pushMap`,
  `noteSpeeds`, `lastSoloTps`, and `ckptSave()` on success.
- On error: `ai.fed = null; ai.pendingCtl = {}; ckptClear(true);` then rethrow.

`aiGenerate` keeps everything else (fitContext, chat bubbles, `ai-genstart` / `ai-token` /
`ai-gendone`, transcript, queue) and calls `roomGenerate` with an `onToken` that does exactly what
`emit` does today. Chat behaviour must be byte-identical: same tokens, same stats line, same
messages (checked in G.3).

**2. Abort during prefill.** `aiPrefill(ids, { aborted })` checks between batched rounds. On
abort it stops issuing rounds, awaits the rounds already in flight (`lapWait` promises, so no
worker is left mid-frame), and throws `AbortError`. `roomGenerate` catches it, sets
`ai.fed = null` (the next call resets; checkpoints stay valid because saved slots are untouched)
and returns `{ reason: "abort", tokens: [] }`. The chat Stop button gets this for free.

**3. Lock.** Two small helpers so the chat queue cannot slip a question in between agent steps
(which would reset the caches under the agent):

```js
function roomLock(kind)   // kind: "code"; false if ai.busy/!ai.engine/ai.degraded; else sets ai.busy, setBusyUI(true)
function roomUnlock()     // ai.busy=false, setBusyUI(false), setTimeout(nextQueued, 0), showRedeal if degraded
```

A code run takes the lock for the whole `Agent.run` (all steps), not per step. Peer chat questions
that arrive meanwhile queue as today (`ai-queued`); `ai-busy` gets `why: "the host's agent is
working"`.

**4. `roomApi`** — the only surface Code mode sees. Built once in room.js:

```js
const roomApi = {
  myId: () => peer.id,
  role: () => ai.role,                                 // "host" | "worker" | "guest" | undefined
  ready: () => !!ai.engine && !ai.degraded,            // host: can generate now
  tok: () => ai.tok,                                   // encode/decode/vocab, specials() works on it
  chatTemplate: () => ai.tok?.chatTemplate || "",      // for detectStyle()
  maxSeq: () => ctxMax(),
  generate: roomGenerate,                              // A.2.1
  lock: roomLock, unlock: roomUnlock,
  stop: () => { ai.abort = true; },
  // messaging
  send: (id, msg) => sendTo(id, msg),
  broadcast: (msg) => sendCode(msg),                   // visibility-filtered, see below
  hostId: () => ai.hostId,
  peers: () => [...conns.keys()],
  on: (type, fn) => codeHandlers.set(type, fn),        // ai-code-* / ai-pv* messages, fn(from, d)
  onPeerJoin: (fn) => codeJoin.push(fn),               // host: late joiner, send history + manifests
  onRole: (fn) => codeRole.push(fn),                   // role / hostId changes (re-deal, host left)
};
```

`sendCode(msg)` applies the room's visibility setting via `chatRecipients` (`room/visibility.js`):
full recipients get the message, hidden ones get nothing (unlike chat there is no stand-in).

**5. Routing.** In `aiOnData`, before the existing switch:

```js
if (d.t.startsWith("ai-code") || d.t.startsWith("ai-pv")) {
  if (CODE_FROM_HOST.has(d.t) && from !== (ai.hostId || PREFIX + roomCode)) return;
  return codeHandlers.get(d.t)?.(from, d);
}
```

`CODE_FROM_HOST = ai-code-start, ai-code-tok, ai-code-tool, ai-code-done, ai-code-files,
ai-code-history, ai-pv, ai-pv-blob, ai-pv-stop`. Only `ai-pv-want` goes peer to host. No frame
format changes, so `PROTOCOL` stays 4; peers on an old build simply ignore the new types.

**6. Loading Code mode.** `room.js` does `import("./room/code.js")` the first time the host (or
a peer, when the host announces a code session) opens the Code tab, and calls
`initCode(roomApi, { mock })`. Chat page load cost is unchanged.

**7. Mock hook (tests only).** When `?mock=code` and `location.hostname` is `127.0.0.1` or
`localhost`, room.js sets `window.__pooledMock = { model: null }` and makes `roomApi.ready()` true
without a loaded engine; `room/code.js` uses `__pooledMock.model` (an object with the Agent's
`generate` contract) instead of `roomModel`. The host role is forced (`ai.role = "host"`,
`ai.hostId = peer.id`) when the room creator enables Code mode in mock. Never active on the
deployed origin.

### A.3 `harness/room-model.js` (implementer B)

```js
export function roomModel(api, {
  thinking = false,      // code mode default off: the think block eats the answer budget
  maxNew = 4096,         // cap; the real value is min(maxNew, maxSeq - prompt - 16) per call
  tools = null,          // for the tool-name constraint
  style = "xml",
  sampling = "focused",  // pickSampler key; code wants low temperature
}) -> {
  generate({ system, turns, signal }),  // async iterable of text deltas (Agent contract)
  budget(),        // tokens the conversation may take: maxSeq - reserve(); reserve = min(maxNew, maxSeq/4) + 64
  count(text),     // exact token count (tok.encode), LRU-cached per string (256 entries)
  idsFor(text),    // the ids an assistant turn was sampled as (for Agent.toJSON)
  adopt(text, ids),// restore them (Agent.from)
  stats,           // { calls, reused, prefilled, generated, tps }
}
```

`generate`:
1. `S = specials(tok)`, `stop = new Set([S.imEnd, S.eot])`.
2. Assistant turns map to their exact sampled ids through the `own` map (fallback
   `tok.encode`); `buildIds(tok, { system, turns, thinking })` from `room/conversation.js`.
3. If `ids.length > maxSeq - 16` throw `ContextFull` (the Agent compacts before this happens,
   see F; this is the backstop).
4. Sampler: `base = pickSampler(sampling)`; with `tools`, wrap it with the
   `ToolCallConstraint` mask exactly like `engine-model`'s `pick` (`C.text = base + pending`),
   which also covers every speculative column because `specStep` samples verified columns through
   the sampler it is given.
5. Calls `api.generate(ids, { onToken, stop, maxNew: min(maxNew, maxSeq - ids.length - 16), sample, signal })`.
   `onToken` pushes decoded pieces into an async queue (UTF-8 holdback while the text ends in
   U+FFFD, as engine-model does) that the returned iterator drains. The constraint's `push` is fed
   from the same place.
6. Text stop: if the generated text contains `<tool_response>` (the model starting to invent a
   tool result), abort an internal controller chained to `signal` and cut the text there.
7. With `thinking`, `splitThink` (`room/conversation.js:82`) separates the think block: the
   visible delta stream gets only the answer, and an `onThink` delta callback (optional) gets the
   rest for the UI.
8. At the end `own.set(text, ids)`. `own` is pruned on every call to the texts present in `turns`
   (it no longer grows without bound).

Shared code with `engine-model.js` moves to **`harness/model-common.js`**:
`constrainedSampler(base, tools, tok, style)`, `deltaDecoder(tok)`, `asyncQueue()`,
`OwnIds` (map + prune). `engine-model.js` keeps its API and tests.

Prefix reuse falls out of the room: `roomGenerate` compares ids against `ai.fed` and the
checkpoints (`ckptResume`), so agent step N+1 re-prefills only the tool response and the new
assistant header. Agent and chat keep separate `turns`; switching modes changes the system prompt,
so reuse drops to the checkpoint or zero and the room resets. No ownership flag is needed because
reuse is decided by comparing exact ids.

### A.4 `harness/agent.js` upgrades (implementer B)

- **Tool interface (shared contract with A):**
  `{ name, description, parameters, mutates, run(args, ctx) -> string, preview?(args) -> {path, before, after} }`,
  `ctx = { signal, step }`.
- **Cancellation.** `run(text, { signal })` passes `signal` to `generate` and to tools. On abort
  mid-generation: keep the partial assistant text as a turn, run no tool calls, emit
  `{type:"stopped"}`. If the abort comes before any assistant text, pop the user turn so two user
  turns never follow each other.
- **Events** (add): `step {step}`, `delta {text, step}` (streamed raw visible text; `text` stays
  for compatibility), `tool-start {call, step}` (before approval), `tool {call, result, step, ms}`,
  `usage {prompt, reused, generated, tps}`, `compacted {tier, before, after}`, `stopped`.
- **Approval.** `approve(call, info)` where `info = await tool.preview?.(call.arguments)`. It returns
  `true`, `false`, or `{ ok: false, reason }`; a reason goes back to the model as
  `declined by the user: <reason>`.
- **Result cap.** Every tool result is cut to `maxResultChars = 6000`, keeping head and tail:
  `…(N chars cut)…`.
- **Compaction** replaces `_fit`, see F.3. `budget` and `count` may be functions
  (`budget: () => model.budget(), count: model.count`).
- **State.** `toJSON() -> { v:1, turns:[{role, text, ids?}] }`, `static from(json, opts)`,
  `reset()`. Assistant `ids` come from `model.idsFor` / go back through `model.adopt`.

---

## B. Preview server (virtual ports)

### B.1 Decision: sandboxed srcdoc iframe, built in the parent, no service worker

Option 1 (service worker at `/` serving `/preview/<port>/…`) is rejected. A same-origin preview
could read the room's localStorage, OPFS (the project files and the model weight cache) and
reach `window.parent`. Adding `sandbox` to fix that makes the frame's origin opaque, and a
service worker does not control an opaque-origin document, so its subresource requests would go
to the real server and 404. It would also sit in front of every weight fetch.

Option 2 is chosen, in a simpler form than "loader resolves files over postMessage": **the host
page builds one self-contained HTML document** from the snapshot and loads it with
`<iframe sandbox="allow-scripts" srcdoc=…>`. Every relative reference is rewritten to a `data:`
URL, including ES module imports (bottom-up over the import graph), so no request ever leaves the
frame and nothing needs the frame to talk back to load files.

Checked in headless Chromium (Playwright, no GPU) before writing this:

| Check | Result |
|---|---|
| `<script type=module src="data:…">` importing a `data:` module importing a `data:` module | works (value computed through 3 levels) |
| `data:` stylesheet via `<link>` | applied |
| `window.origin` in the frame | `"null"` (opaque) |
| `localStorage` in the frame | throws (so we shim it, B.4) |
| `parent.document` | blocked |
| `fetch("/x")` to the room origin under the frame's CSP meta | refused by CSP |
| `import(blobURL)` created inside the frame | works (fallback path if ever needed) |
| uncaught error | reaches the parent via `postMessage` from the capture script |

Why this is the safest option that still runs multi-file apps:
- The code runs in an opaque origin: no access to the room's storage, cookies, OPFS, Cache API,
  PeerJS objects or DOM. The parent only accepts messages whose `e.source` is that iframe's
  `contentWindow`, and treats their contents as untrusted text (rendered with `textContent`, capped).
- Sandbox flags: `allow-scripts` only. No `allow-same-origin`, `allow-popups`,
  `allow-top-navigation`, `allow-forms` or `allow-modals`. `allow=""` (no camera, mic,
  geolocation…), `referrerpolicy="no-referrer"`.
- A CSP `<meta>` is the first element of the built `<head>`, before any agent-written markup.
  Later metas can only tighten it:
  `default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' data: blob: https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; style-src 'unsafe-inline' data: https://fonts.googleapis.com; font-src data: https://fonts.gstatic.com; img-src data: blob:; media-src data: blob:; connect-src data: blob:; worker-src blob: data:`.
  Network is off except the two script CDNs the rest of the site already trusts (so the agent may
  use a library from jsdelivr).
- Peers run the same document in the same sandbox, so a peer is never exposed to more than the
  host.

No Vercel, header or rewrite changes are needed: Code is a mode of `/room`, and no new top-level
path or service worker exists. One constraint for the future: a srcdoc document inherits the
parent page's CSP. If `/room` ever gets a strict CSP, it must still allow inline and `data:`
scripts in the frame, or previews must move to a separate static origin (a
`*.preview.pooled.sh` wildcard domain with its own service worker). That would also lift the
rewriting limits in B.5. Recorded here as the upgrade path, not built now.

### B.2 Virtual ports

A port is a key in the host's `PreviewServer`: `serve({ dir, port, entry })` snapshots `dir` of
the workspace and registers it under `port` (an integer 1024-65535, default 5173). The UI shows a
tab per port labelled `:5173` with an address line `:5173/index.html`. Ports are per host session,
there is nothing to bind, and a second `serve` on the same port replaces it.

Snapshot:

```js
// Snapshot: an immutable view of the served files at one revision
{ port, dir, entry, rev, bytes,
  files: Map<path, { type, bytes: Uint8Array, hash }> }   // path relative to dir, hash = sha-256 hex (first 20 chars)
```

Limits (host enforces on `serve` and on refresh; peers enforce again on receipt): 400 files,
2 MB per file, 8 MB per snapshot. Skipped: `SKIP_DIRS`, dotfiles. Over the limit, `serve` fails
with a short message the model can act on.

### B.3 Build: `buildPreviewDoc(snapshot, { path, nonce })` (harness/preview-build.js)

Pure function, no DOM needed beyond string work, so it is unit tested in Deno.

Returns `{ html, missing: [path], warnings: [text], urlToPath: {dataURL: path} }`.

1. Entry HTML (`path`, default `snapshot.entry`) is read as text.
2. Assets become `data:<type>;base64,` URLs. CSS is rewritten first (`url(...)`, `@import`) so
   its own references resolve relative to the CSS file.
3. JS modules: parse static `import … from "x"`, `export … from "x"`, `import "x"` and
   `import("literal")` specifiers with a small tokenizer-aware regex (skips comments and
   strings), resolve relative specifiers against the importing file, rewrite depth-first so each
   module's dependencies are already `data:` URLs. Bare specifiers and `https:` URLs are left as
   they are (CDN via CSP). An import cycle is reported in `warnings` and the cycle edge is left
   unresolved (it fails with a clear console error in the frame).
4. HTML: `src`, `href` (stylesheets, icons), `srcset`, `poster`, inline `style` `url()`,
   `<style>` blocks, and inline `<script type=module>` imports are rewritten. Paths that do not
   exist go to `missing` and are also logged into the frame's console (`404 path`) so the agent
   sees them.
5. Injected at the top of `<head>`: CSP meta, then the capture script (B.4) with `nonce`, the
   `urlToPath` table (for mapping error locations back to file names), and a map of every asset
   path to its data URL for the `fetch`/XHR shim.

Mime types by extension (`.html .js .mjs .css .json .svg .png .jpg .gif .webp .wav .mp3 .ogg
.woff2 .txt`), else `application/octet-stream`.

### B.4 In-frame capture script (string constant in harness/preview-build.js)

Runs before any agent code:
- Wraps `console.log/info/warn/error/debug`; listens to `error` (capture phase: script errors
  and resource load failures) and `unhandledrejection`. Posts
  `{ pv: nonce, t: "log", level, text, src, line, col }` to `parent` with `"*"` (an opaque origin
  cannot name a target origin; the parent checks `e.source`). `src` is mapped from the data URL to
  the file path via `urlToPath`; data URLs in stack text are replaced by paths. `text` capped at
  1,000 chars; at most 200 messages per second, then one `(N messages dropped)`.
- Posts `{ t: "ready", ms }` on `load` and `{ t: "idle" }` 500 ms later (the serve tool waits for
  it, C).
- Shims: in-memory `localStorage` / `sessionStorage` (games keep high scores there, and the real
  one throws in an opaque origin); `alert`/`confirm`/`prompt` log to the console and return
  `undefined`/`false`/`null` (modals are not allowed); `fetch` and `XMLHttpRequest.open` map
  relative URLs to the asset table; clicks on relative `<a href>` post `{ t: "nav", path }` and
  the parent rebuilds the document for that page.

### B.5 Known limits (said in the system prompt in one line, and in the UI)

No server code, no network except the two CDNs, no computed import specifiers, no
`new Worker("file.js")` (a literal path is rewritten like an import), no `new URL(x, import.meta.url)`
assets, one document per load (multi-page apps work through the nav shim). These fit the target
(static games, demos and tools) and each failure shows up as a console error the agent can read.

### B.6 Live reload

`PreviewServer` wraps the workspace (`watch(ws)`, D) and listens to writes. A write under a served
`dir` marks the port dirty; 250 ms after the last write it re-snapshots (hashing only changed
paths), bumps `rev` and emits `update { port, rev, changed }`. Every mounted frame for that port
rebuilds and reassigns `srcdoc` (a full reload; game state resets, which is fine). The Reload
button calls `refresh(port)`, which re-walks the directory (catches edits made on disk outside the
agent in a DirWorkspace).

### B.7 Console to the agent

The host's mounted frame calls `server.pushLog(port, entry)` for each message. The server keeps a
ring of 500 entries per port, each `{ seq, t (ms since load of that rev), rev, level, text, src, line, col }`.
`logs(port, since)` returns the entries after `since` and the next cursor. Logs from peers' frames
stay on the peer (they run the same code; the host's copy is the one the agent reads).

Chrome throttles `requestAnimationFrame` in hidden or off-screen cross-origin frames, so a game
bug that fires on the first frame may not fire while the pane is hidden. `serve` therefore opens
the preview pane on the host (the UI listens to the server's `update`), and the pane stays
rendered while a port is served.

### B.8 Peers see the same preview (harness/preview-sync.js)

Transport-agnostic classes; room messages are wired in by the integrator through
`roomApi.on/send/broadcast`.

```js
export class PreviewPublisher {           // host
  constructor(source /* PreviewServer */, { send, broadcast, chunk = 64 << 10 })
  announce(port)            // broadcast ai-pv {port, dir, entry, rev, manifest}
  stopped(port)             // broadcast ai-pv-stop {port}
  onWant(from, d)           // handle ai-pv-want: sends ai-pv-blob chunks for listed hashes of the current rev
  helloTo(peerId)           // late joiner: ai-pv for every served port
}
export class PreviewSubscriber {          // peer; implements PreviewSource
  constructor({ send, hostId, maxBytes = 8 << 20, maxFile = 2 << 20, maxFiles = 400 })
  onManifest(from, d)       // ai-pv: keep known blobs, send ai-pv-want for missing ones
  onBlob(from, d)           // ai-pv-blob: assemble, verify sha-256, complete rev -> emit update
  onStop(from, d)
}
```

Messages (PeerJS reliable channel, binary serialization, not the activation wire):

| Type | Direction | Body |
|---|---|---|
| `ai-pv` | host → peers | `{ port, dir, entry, rev, bytes, manifest: [[path, type, hash, size]] }` |
| `ai-pv-want` | peer → host | `{ port, rev, hs: [hash] }` (host sends only hashes in that port's current manifest) |
| `ai-pv-blob` | host → peer | `{ h, i, n, b: ArrayBuffer }` chunk `i` of `n`, 64 KB each |
| `ai-pv-stop` | host → peers | `{ port }` |

The host sends chunks one at a time, waiting while the PeerJS data channel's `bufferedAmount` is
over 1 MB, so a snapshot never delays chat or agent messages much. Blobs are content addressed:
after a live reload a peer fetches only changed files. Peers verify hashes and limits and drop
anything else. A peer's preview is click-to-run the first time ("Run preview :5173"). It is
sandboxed either way, but it spends the peer's CPU and it is the peer's choice. After that it
live-reloads.

---

## C. Tools

### C.1 The tool set (8 tools, schemas minimal)

Descriptions are one line; parameter descriptions only where the name is not enough.

| Tool | Params | Mutates | Result (short) |
|---|---|---|---|
| `list_dir` | `path?` | no | names, dirs end in `/`; max 200 entries then `(+N more)` |
| `read_file` | `path, start_line?, end_line?` | no | numbered lines `41\|code`, 200 lines or 8,000 chars per call, then `(lines 1-200 of 412; read on with start_line=201)` |
| `search` | `pattern, path?, ignore_case?` | no | `path:line: text` (text cut to 160 chars), 30 hits then `(+N more)` |
| `edit_file` | `path, old, new` | yes | `edited game.js lines 40-44 (5 -> 6 lines)`; error if `old` matches 0 or >1 times, with the count |
| `write_file` | `path, content, append?` | yes | `wrote game.js (212 lines, 6.1 KB)`; with `append: true`, `appended …` |
| `serve` | `dir?="", port?=5173, entry?="index.html"` | no | see C.2 |
| `preview_logs` | `port?=5173, since?` | no | see C.2 |
| `stop_serve` | `port` | no | `stopped :5173` |

Changes to `harness/codetools.js`: `read_file` pages drop from 400 lines / 24,000 chars to
200 / 8,000; `edit_file` parameter names shorten to `old` / `new` (fewer tokens per call, the
parser coerces either); `write_file` gets `append` (a file longer than one answer's `maxNew` is
written in parts: the system prompt says so in one line); `list_dir` and `search` caps as above.
Each mutating tool gets `preview(args)` returning `{ path, before, after }` for the approval card.
After any write inside a served dir the result gets ` · preview :5173 reloaded` appended.

No delete, rename or shell tool in v1 (not needed for "build me X", and every tool costs prompt).

### C.2 serve and preview_logs formats

`serve` snapshots, registers the port, and when a host frame is mounted waits up to 2 s for the
frame's `idle` message, then reports what happened in the first 500 ms:

```
serving . on :5173 (index.html, 3 files, 9.4 KB)
loaded in 120 ms · 1 error:
[0.1s] error game.js:41:5 ReferenceError: ctx is not defined
more: preview_logs since=3
```

With no frame (headless, or the pane failed): `serving . on :5173 (…) · no preview open, logs
appear when it is`. Missing referenced files: `missing: sprites/block.png`.

`preview_logs({ port, since })`: entries after `since`, newest last, at most 40 lines /
3,000 chars (older ones summarized as `(N earlier lines, since=K)`), repeated identical lines
folded (`×12`), then `next: since=57`. With nothing new: `no new logs on :5173 (rev 4, loaded 3.2s ago)`.

Both live in **`harness/preview-tools.js`**: `previewTools(server) -> Tool[]` (serve,
preview_logs, stop_serve).

---

## D. Workspace

**`harness/projects.js`** (new):

```js
export const PROJECTS_DIR = "pooled-projects";
export async function listProjects() -> [{ id, name, kind: "opfs"|"folder", updated }]
export async function createProject(name) -> { id, ws }        // OPFS: pooled-projects/<slug>/, DirWorkspace over it
export async function openProject(id) -> { id, ws }             // folder handles re-ask permission (requestPermission "readwrite")
export async function openFolder() -> { id, ws } | null         // showDirectoryPicker; null if unsupported/cancelled
export async function deleteProject(id)                         // OPFS only; a folder is just forgotten
export const canOpenFolder = () => "showDirectoryPicker" in window
```

Project metadata (name, kind, updated, the folder `FileSystemDirectoryHandle`) lives in IndexedDB
(`pooled-projects` db), since handles are structured-cloneable and localStorage cannot hold
them. The agent session per project (`Agent.toJSON()`) is saved next to it, so reopening a project
restores the conversation (the room's caches are re-prefilled on the next step).

Defaults: "build me Tetris" from nothing → a new OPFS scratch project named from the first
message (`tetris`, `tetris-2`). "Open folder…" only where `showDirectoryPicker` exists
(Chrome/Edge desktop); elsewhere the button is hidden. Download/zip: out of scope.

**`harness/workspace.js`** additions: `readBytes(p) -> Uint8Array` and `writeBytes` on both
workspaces (images for the preview), `remove(p)` (for later), and

```js
export function watch(ws) -> ws & { onChange(fn) -> unsubscribe }   // fn({ path, kind: "write"|"remove" }) after each mutation through this wrapper
```

Approval policy: OPFS scratch projects auto-approve edits by default (the files exist only in
this browser, and the diff is still shown in the timeline). A folder on disk asks per edit, with
"Allow edits for this task" on the card. The host can switch either default in the Code settings.

---

## E. UI

All in `p2p.html` + `room/code.js` + `room/code-ui.js`, reusing the existing tokens (`--bg`,
`--panel`, `--panel-2`, `--border`, `--text`, `--muted`, `--accent`, `--ok`, `--warn`, `--err`,
`--sans`, `--mono`) and the existing button, card and chip styles. No new colours except
`--diff-add` / `--diff-del` (tints of `--ok` / `--err`). There is no dark mode in the room today,
and this doesn't add one.

**Mode switch.** A segmented control at the top of `main#chatpane`: `#mode-chat` | `#mode-code`
(`role="tablist"`). Chat mode is the current pane, untouched. Code mode hides `#ai-output`,
`#chat-tools` and `#ai-row` and shows `#code-pane`. Peers see the Code tab once the host
announces a session (`ai-code-start` or `ai-pv`), with a dot when something new happens.

**`#code-pane`** (host): two columns over 900 px, stacked below.

Phones (640 px and narrower, and short touch screens, `(max-height: 500px) and (pointer: coarse)`:
a phone in landscape, however wide) show one view at a time, picked from `#code-tabs`, a tab bar at
the bottom (`role="tablist"`, 56 px plus the safe area, 44 px on a short screen): **Agent** (the log, the prompt at the bottom;
an approval waits in `#code-dock` above the prompt), **Preview** (the app at full height with its
address bar) and **Files** (project and tree; a file opens the editor full screen with a back arrow).
A dot on a tab: Agent pulses while the agent works and holds a dot while an approval waits, Preview
gets one for a new revision served while elsewhere. The first app served in a session opens
Preview. The tab is kept per session (`sessionStorage`); guests get the same layout. While typing,
the tab bar steps aside and the prompt sits on the keyboard (`--kb` where the browser does not
resize the page). On a short screen Chat | Code moves into the header row and the devices' chip row
goes, so the pane keeps the height. The editor soft-wraps long lines on phones; the gutter numbers
the file's lines, with blank rows beside wrapped ones. Two served ports or more get a row of
port tabs of their own above the address.

- Left, the agent:
  - `#code-project`: project select, `New`, `Open folder…`, project name. Host only.
  - `#code-log`: the timeline. Items:
    - `.cm-user`: the request.
    - `.cm-text`: streamed model text (markdown via `mdChat`, like chat bubbles).
    - `.cm-tool`: one line per call, `read_file game.js 1-200`, with a status chip
      (`running` / `done` / `error` / `declined`) and a `<details>` with the result (monospace,
      capped).
    - `.cm-diff`: for `edit_file` / `write_file`, a line diff (`lineDiff`, up to 400 lines,
      else "new file, 212 lines"), with `Approve` / `Reject…` (reason input) / `Allow edits for this task`
      when approval is needed, and a compact "applied" state otherwise.
    - `.cm-note`: compaction ("older steps shortened to fit 16k"), stop, errors.
    - `.cm-stats` at the end of a run: steps, tokens, tok/s, devices (same format as chat stats).
  - `#code-row`: `#code-prompt` (textarea, Enter sends, Shift+Enter newline), `#code-send`,
    `#code-stop`. `#code-ctx`: the same meter as `setCtx`, for the agent's context.
- Right, the output: tabs `Preview` | `Files`.
  - Preview: `#pv-tabs` (one `.pv-tab` per port, `:5173`, with a close ×), `#pv-addr`
    (`:5173/index.html`, read-only), `#pv-reload`, `#pv-frame-wrap` (the sandboxed iframe, clicks
    focus it for keyboard games), and `#pv-console`: a collapsible strip with counts
    (`2 errors · 5 logs`), rows coloured by level, `Clear`, and "Send to agent", which puts
    `Fix the errors in the preview console` in the prompt.
  - Files: `#code-tree` (read-only tree from `walk()`, max 500); clicking a file opens
    `#code-view` (numbered lines, monospace, no editing in v1).

**Peers** get the same pane, and drive it like the host when the room shows answers to everyone:
one shared agent session per room, run on the model host. A member's request goes to the host as
`ai-code-ask` and queues (six at most, two per member) behind the current run; its bubble carries
the member's name. The member who asked, or the host, answers its approvals (`ai-code-approve`)
and can stop it (`ai-code-stop`); others see "waiting for <name> to approve". Any member can start
a new task, open or create a project saved in the host's browser (empty, or from a starter
template in harness/templates.js: `{cmd: "new", name, tpl}`, the host checks `tpl` against its
list), or tick auto-approve (`ai-code-cmd`); `ai-code-projects` mirrors the host's project list and `ai-code-sync` asks for the
session on opening Code. The host alone opens a folder from disk, saves in the editor, and drives
a folder project (what the agent reads there would reach the asker's screen). With "Only me" or
"Whoever asked", Code stays the host's. A line above the log says where the agent runs and where
the files live. Preview with port tabs and their own console strip, and Files from
`ai-code-files`. Joining late: `ai-code-history` (last 50 items, results already capped) plus
`ai-pv` per port.

**Agent streaming messages** (host → peers, visibility-filtered, all in `CODE_FROM_HOST`):

| Type | Body |
|---|---|
| `ai-code-start` | `{ sid, mid, name, text }` new request |
| `ai-code-tok` | `{ mid, step, text }` visible text, coalesced every 50 ms |
| `ai-code-tool` | `{ mid, step, i, name, brief, state, result?, diff? }` state `pending\|approved\|declined\|done\|error`, `brief` ≤ 200 chars, `result` ≤ 600, `diff` ≤ 4 KB |
| `ai-code-done` | `{ mid, steps, stats, reason }` |
| `ai-code-files` | `{ tree: [path] }` after each mutating step (≤ 500 paths) |
| `ai-code-history` | `{ sid, items }` to a late joiner |

Host keyboard: Esc stops the run. The Stop button and the chat `aiStop` both abort through
`roomApi.stop()` and the run's `AbortController`.

---

## F. Context frugality

Numbers for the 16k default (27B). The 32k MoE gets the same prompt and more room to work.

### F.1 Budget

| Part | Tokens |
|---|---|
| System prompt: instructions | ≤ 350 |
| Tool block (8 tools, xml style) | ≤ 850 |
| Answer reserve (`maxNew` cap + 64) | min(4096, maxSeq/4) + 64 = 4,160 at 16k |
| Conversation (history + tool results) | the rest: ~11k at 16k, ~23k at 32k |

A unit test holds the system prompt + tool block under 4,200 characters (≈1,200 tokens at the
3.5 chars/token estimate). The real-model script (G.4) logs the exact count.

### F.2 System prompt (integrator owns the text; target shape)

```
You are a coding agent in a browser. Files live in a project folder; there is no shell.
Build static web apps (HTML, CSS, JS modules). They run in a sandboxed preview: no network except
cdn.jsdelivr.net and cdnjs.cloudflare.com, no server code.
Work in small steps: write files with write_file (split files over ~150 lines with append),
fix with edit_file, then serve and check preview_logs. Fix every error before you finish.
Read files by line range. Keep answers short; when done, say what you built in one or two lines.
```

### F.3 Compaction (in `Agent`, replaces `_fit`)

Before each step, if `size > budget()`, compact down to **60% of the budget**. The margin
matters because every compaction changes the middle of the prompt, and that means a full
re-prefill across the room. So it should happen rarely and free a lot each time:

1. **Stub old tool results**: `<tool_response>` bodies over 200 chars older than the last 2
   steps become `(output of read_file game.js 1-200 dropped; run it again if needed)`. Oldest
   first. (Existing behaviour, now with the name/args in the stub.)
2. **Fold finished requests**: every step of an earlier *completed* user request becomes one
   assistant line built mechanically, without a model call:
   `[earlier: wrote index.html, game.js, style.css; served :5173; fixed 1 error]` + that
   request's final answer (first 600 chars).
3. **Drop the oldest folded exchanges** (like `fitContext`), always keeping the current
   request's user turn.
4. Still over: stop with `context full: start a new task (the files are kept)`. The UI offers
   "New task", which resets the agent turns but keeps the project and the preview.

Emits `compacted { tier, before, after }`; the UI shows one `.cm-note`.

### F.4 Other savings

- Code mode samples with `focused` (0.4 / top-20), thinking off.
- Assistant turns replay their exact sampled ids, so steps extend the cached prefix instead of
  re-prefilling it.
- Tool results are capped at the tool (C.1) and again in the Agent (6,000 chars).
- `serve` returns the first errors directly, which saves a separate `preview_logs` step in the
  common case.

---

## G. Test plan

### G.1 Unit tests (Deno, CPU: `deno test --allow-read tests/unit`)

Implementer A:
- `preview_build_test.js`: script/link/img/srcset rewrite; module graph a→b→c becomes nested
  `data:` URLs with the right import order; CSS `url()` and `@import` relative to the CSS file;
  bare and `https:` specifiers untouched; missing files listed; a cycle is a warning; the CSP meta
  is the first head element and the capture script comes before any agent script, even when the
  agent's HTML has its own `<head>` or none at all; `urlToPath` round trip; limits.
- `preview_server_test.js` (MemoryWorkspace): serve/stop/ports; snapshot limits; a write under the
  dir bumps `rev` after the debounce with the right `changed`; a write outside does not;
  `pushLog` / `logs(since)` cursor, caps, folding; the serve / preview_logs / stop_serve result
  strings (C.2) with a fake frame that answers `idle`.
- `preview_sync_test.js`: Publisher ↔ Subscriber over an in-memory `send`: first rev fetches every
  blob; the second rev fetches only changed ones; a bad hash or a size over the limit is rejected;
  `ai-pv-want` for a hash outside the manifest is ignored; stop.
- `codetools_test.js` (extend `agent_test.js` or new): `read_file` 200 lines / 8,000 chars and
  the continuation hint; `write_file append`; `edit_file` `old`/`new` with 0 and 2 matches;
  `list_dir`/`search` caps; `preview()` before/after.
- `diff_test.js`: `lineDiff` add/remove/change, the size cap.
- `prompt_size_test.js`: system prompt + tool block ≤ 4,200 chars.

Implementer B:
- `room_model_test.js`: `roomModel` over a fake `roomApi` whose `generate` emits scripted ids
  with the word tokenizer from `engine_model_test.js`: deltas stream in order; the stop set ends
  the answer; the tool-name constraint forces `read_file>` over `rm_rf>` through the `sample`
  wrapper; `<tool_response>` text stop; abort mid-stream; `ContextFull`; `budget()` formula;
  assistant ids replayed exactly on the next call (the fake asserts the new ids start with the
  previous call's ids + answer); `own` pruning.
- `agent_test.js` additions: abort mid-generation (partial text kept, no tool run, no double user
  turn); abort before any text (user turn popped); `tool-start` before approval; approval with a
  reason; result cap; compaction tiers 1-4 with a `count` that makes sizes exact; `toJSON`/`from`
  round trip with ids.
- `engine_model_test.js` still passes after the `model-common.js` extraction.

### G.2 Browser tests without WebGPU (Playwright, `launch({ args: ["--no-sandbox"] })`)

- `tests/e2e/preview_browser.mjs` (A, seconds): serves the repo with `serveRepo`, mounts
  `mountPreview` on a page with a MemoryWorkspace holding a 3-file app (module import, CSS,
  PNG). Asserts: canvas size non-zero and pixels drawn (`frame.evaluate`); `window.origin ===
  "null"`; `parent.document` throws; `localStorage` shim works; `fetch("/p2p.html")` blocked; an
  error in `game.js` arrives as `game.js:41`; live reload after `ws.write`. Second page: a
  Subscriber fed from the first page's Publisher through `page.exposeFunction` relays renders the
  same canvas. This one always runs; it needs neither PeerJS nor a room.
- `tests/e2e/code_tetris.mjs` (integrator, the acceptance test), based on `room_synth.mjs`
  (local PeerServer, `ctx.route`, `serveRepo`), without WebGPU flags:
  1. Host tab: `p2p.html?signal=127.0.0.1:9001&mock=code`, create the room; peer tab joins with
     the code.
  2. Host clicks `#mode-code`, then `New` project `tetris`. The page sets
     `__pooledMock.model = scripted([...])` (the `scripted` helper from `agent_test.js:11` moves
     to `tests/scripted-model.js` so both tests use it). Replies:
     1. `write_file index.html` (canvas `#board`, `<link style.css>`, `<script type=module src=game.js>`)
     2. `write_file style.css`
     3. `write_file game.js` with a deliberate bug (`ctx.fillRect` before `const ctx = …`, a TDZ ReferenceError)
     4. `serve {port: 5173}`, and the mock **asserts** its next `turns` contain `ReferenceError`
        and `game.js` in the serve result
     5. `edit_file game.js` moving the line
     6. `preview_logs {since}`, and the mock asserts `no new logs` or no `error` line
     7. final answer "Tetris is running on :5173."
  3. Host asserts: `.pv-tab` `:5173`; the preview frame's canvas is non-zero size with drawn
     pixels; `#pv-console` showed the error and then a new rev with no errors; `.cm-stats` present.
  4. Peer asserts: the Code tab appears; the timeline shows the 7 steps (the tool cards);
     `Run preview :5173` → its own sandboxed frame renders a canvas with drawn pixels at the same
     `rev`; peer frame `window.origin === "null"`.
  5. Collects `pageerror` and console errors on both pages (the preview's own intentional error
     excluded by source) and fails on any.

  Prerequisite, the same as `room_synth.mjs`: the PeerJS client bundle
  (`npm i --no-save peerjs@1.5.4` in `/home/nehanth/bello`, which needs the network). If it is
  missing the test prints `SKIP: peerjs client missing` and exits 0, and `preview_browser.mjs`
  still covers the host-to-peer path.

### G.3 Chat regression (B)

`roomGenerate` must not change chat output. Before/after on the refactor branch:
`node tests/e2e/room_synth.mjs --solo --greedy` and `--devices 3 --compare --expect-reuse`
(SwiftShader runs on the CPU, so it does not disturb a real-GPU benchmark, but it takes minutes:
run each once, not in a loop). The answer text
and token counts must match the pre-refactor run exactly. Plus `deno test` and `npm run check`.

### G.4 Real model, manual (later, needs the GPU)

`tests/manual/code_real.md`, a checklist the integrator writes:
1. Two devices, 27B at 16k, `/room`, host loads the model, Code tab, new project.
2. "Build a Tetris game in plain JS with a canvas. Arrow keys move, up rotates, space drops."
3. Record: system prompt tokens (from `usage`), steps, re-prefills (`compacted` notes), tok/s,
   whether serve showed errors and whether the agent fixed them unaided, final context used.
4. Play it on both screens; confirm the peer's preview updates after a host-side edit.
5. Repeat on the 35B MoE at 32k; then "add a score and a next-piece box" as a follow-up to test
   prefix reuse across requests (expect `reused` ≈ previous context).
6. Stop mid-write: the next request works and the room chat still answers.

---

## H. Work split

Three branches off `feat/engine-opt`. A and B run in parallel; the integrator starts from both.

### Implementer A: `feat/code-preview` (preview, tools, workspace)

Owns: `harness/preview-build.js`, `harness/preview.js`, `harness/preview-frame.js`,
`harness/preview-sync.js`, `harness/preview-tools.js`, `harness/diff.js`,
`harness/projects.js`, `harness/codetools.js`, `harness/workspace.js`, the G.1 A tests,
`tests/e2e/preview_browser.mjs`. Does not touch `room.js`, `agent.js`, `p2p.html`.

### Implementer B: `feat/code-model` (room adapter and agent)

Owns: `room.js` (A.2: `roomGenerate`, `aiPrefill` abort, `roomLock/Unlock`, `roomApi`,
`sendCode`, routing and `CODE_FROM_HOST`, the lazy `import("./room/code.js")` behind a stub
`room/code.js` that exports a no-op `initCode`, the mock hook), `harness/room-model.js`,
`harness/model-common.js`, `harness/engine-model.js`, `harness/agent.js`, `harness/tools.js`
if needed, the G.1 B tests, G.3. Does not touch preview or workspace files.

### Integrator: `feat/code-ui` (after A and B merge)

Owns: `room/code.js` (host controller and peer view wiring), `room/code-ui.js` (DOM), the
`p2p.html` markup and CSS, the system prompt, `tests/scripted-model.js`,
`tests/e2e/code_tetris.mjs`, `tests/manual/code_real.md`, and `docs/protocol.md` (new message
types).

### Interfaces (frozen by this doc)

```js
// ---- Tool (A implements, B's Agent calls) ----
Tool = { name, description, parameters /* JSON schema */, mutates: bool,
         run(args, ctx: { signal, step }) -> Promise<string>,
         preview?(args) -> Promise<{ path, before: string|null, after: string }> }

// ---- Workspace (A) ----
ws.read(p) -> string; ws.write(p, text); ws.readBytes(p) -> Uint8Array; ws.writeBytes(p, u8)
ws.exists(p); ws.list(dir) -> [{name, dir}]; ws.walk(limit) -> [path]; ws.remove(p)
watch(ws) -> ws & { onChange(fn({path, kind})) -> unsubscribe }
codingTools(ws) -> Tool[]                         // list_dir read_file search edit_file write_file
projects: listProjects() createProject(name) openProject(id) openFolder() deleteProject(id) canOpenFolder()

// ---- Preview (A) ----
new PreviewServer(ws /* watched */, { maxFiles: 400, maxFile: 2<<20, maxBytes: 8<<20 })
  .serve({ dir, port, entry }) -> Promise<Snapshot>     // throws Error(short message) over limits
  .stop(port); .refresh(port); .ports() -> [{ port, dir, entry, rev }]
  // PreviewSource (also implemented by PreviewSubscriber):
  .snapshot(port) -> Snapshot | null
  .onUpdate(fn({ port, rev, changed, stopped })) -> unsubscribe
  .pushLog(port, { level, text, src, line, col }); .logs(port, since) -> { lines, next }
  .whenIdle(port, ms) -> Promise<{ loadedMs } | null>   // resolved by the mounted frame's idle
previewTools(server) -> Tool[]                          // serve preview_logs stop_serve
buildPreviewDoc(snapshot, { path, nonce }) -> { html, missing, warnings, urlToPath }
mountPreview(el, source, port, { onLog, onStatus, autorun = true }) -> { reload(), destroy(), frame }
new PreviewPublisher(server, { send, broadcast }); new PreviewSubscriber({ send, hostId })
lineDiff(a, b, { max: 400 }) -> [{ op: " "|"+"|"-", text }] | null

// ---- Model (B) ----
roomApi                        // A.2.4 (room.js)
roomGenerate(ids, opts) -> result   // A.2.1
roomModel(roomApi, opts) -> { generate, budget, count, idsFor, adopt, stats }   // A.3
new Agent({ generate, tools, style, system, maxSteps, approve(call, info), onEvent, budget, count, maxResultChars })
  .run(text, { signal }) -> { text, steps, calls, reason }; .toJSON(); Agent.from(json, opts); .reset()
events: step delta text tool-start tool usage compacted trimmed done limit stopped

// ---- Room messages (B routes, integrator/A's sync produce and consume) ----
host→peers: ai-code-start ai-code-tok ai-code-tool ai-code-done ai-code-files ai-code-history
            ai-pv ai-pv-blob ai-pv-stop
peer→host:  ai-pv-want
```

Order inside each branch: A starts with `preview-build.js` + `preview_browser.mjs` (the sandbox
path is the riskiest piece, and the spike above says it works); B starts with the `roomGenerate`
extraction + the G.3 regression, before the adapter.

---

## Risks and open points

- **Rewriting is not a bundler.** Computed imports, workers from files and `import.meta.url` assets
  fail (B.5). The failures are loud (console errors reach the agent), and the separate-origin
  service-worker preview is the upgrade path if real use hits them.
- **Big files vs. `maxNew`.** A 300-line `game.js` is ~3,500 tokens: fine at the 4,096 cap on 16k,
  but a capped `write_file` becomes an unterminated call. The prompt asks for files in parts
  (`append`), and the unterminated-call error tells the model to do so.
- **Re-prefill cost.** Each compaction re-prefills ~7k tokens across the room. The 60% target
  keeps it rare. A checkpoint right after the system prompt (so a compaction re-prefills only
  history) is a follow-up: it needs `ckptSave` at an arbitrary position mid-prefill.
- **Hidden-frame throttling.** Covered by keeping the host pane visible while serving (B.7).
  A host who hides the pane may see fewer errors reach the agent.
- **Peers' CPU.** Previews are click-to-run on peers the first time.
- **Multiple hosts/re-deal.** A re-deal that moves the host role ends the code session (the
  project stays in the old host's browser). The new host can't continue it in v1; peers see
  `ai-code-done {reason: "host changed"}`.
