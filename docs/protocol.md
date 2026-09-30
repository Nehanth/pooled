# Room protocol

Browsers in a room form a WebRTC mesh (PeerJS signaling for the introduction only). The host registers with PeerJS as `pooled-room-<CODE>`; before the rename to Pooled the prefix was `swarmllm-room-`, so a pooled.run tab and an old swarmllm.ai tab never meet in one room. One browser is the **host**: it owns the conversation, the tokenizer, the embedding table, the LM head and the sampler. The others are **workers** holding contiguous layer ranges; together they form a **chain** in layer order, with the last worker sending back to the host. By memory, phones hold no layers at all while the host and the other computers can hold the model (`phonesToLeaveOut`; `?phonelayers=1` overrides): they join as ask-only guests.

## Lifecycle

| Message | Direction | Meaning |
|---|---|---|
| `ai-wait` | host → worker | join accepted; wait for assignment |
| `ai-load {v, model, range, next, host}` | host → worker | download and load layers `[range[0], range[1])`; forward to `next`. A worker on another protocol `v` refuses it with `ai-error` instead of loading |
| `ai-progress {pct}` / `ai-hostprog` | worker ↔ host | download progress for the room UI |
| `ai-ready` / `ai-ready-all` | worker → host / host → all | layers loaded; room online |
| `ai-reset` | host → all | "new chat": the host forgot the conversation; screens clear the transcript. It does **not** reset any engine: devices keep their caches between questions (multi-turn), and a reset rides on the next frame instead (see compute frames) |
| `ai-genstart` / `ai-token` / `ai-gendone` | host → all | mirror the question and streamed answer to every screen. Under `ai-visibility` `host`/`asker`, the text goes only to the allowed screens; the others get `ai-genstart`/`ai-gendone` with `hidden: true` (no `ai-token`), so every Send box still locks and unlocks |
| `ai-visibility {mode}` | host → all | who sees the chat: `all`, `host` (only the host's screen) or `asker` (the host and the peer that asked, by peer id). Sent on change and to every device that joins while it is not `all`. Every device still computes the answer; this only decides which screens get the text |
| `ai-ask` / `ai-busy {why?}` | guest → host / host → guest | anyone in the room can ask; one generation at a time |
| `ai-queued {pos}` / `ai-queue {n}` | host → asker / host → all | a question asked while the room is answering waits in the host's queue (at most 10, two per device) and runs next; the asker learns its place, every screen shows how many are waiting |
| `ai-cmd {cmd}` | guest → host | `continue` a capped answer or `regen`erate the last one; honoured from the host or whoever asked last. `ai-regen` (host → all) greys out the replaced exchange |
| `ai-react {mid, e}` / `ai-reacts {mid, counts}` | guest → host / host → all | emoji reactions on an answer; `mid` is the answer id the host puts in `ai-genstart`. The host toggles per device and broadcasts the counts |
| `ai-typing {name?}` | guest → host → others | "… is typing", relayed only while the chat is visible to everyone |
| `ai-inv-req {url}` / `ai-inv {url, have}` | host → all / all → host | before dealing, the host asks what byte ranges of the model each device has cached; the inventory goes out with every `ai-load` (`inv`) |
| `ai-wget {id, url, lo, hi}` / `ai-wpart {id, off, data \| done \| miss}` | device ↔ device | take a cached range from another device instead of the model host, in 64 KB parts; any failure falls back to the network |
| `ai-stop` | guest → host | stop the answer being generated. Honoured from the device that asked (the host can always stop); decoding ends after the lap in flight and `ai-gendone` unlocks every screen |
| `ai-degraded {why}` | host → all | a device in the chain left; every lap in flight failed at once and the room waits for a re-deal |
| `ai-redeal {by, model}` | host → all | the host is dealing the layers again over the devices now in the room (after a departure, or to include late joiners); fresh `ai-load`s follow, cached ranges reload in seconds, the conversation is kept and re-prefilled on the next question |
| `ai-ready-all {model}` to one device | host → newcomer | a device that joins an online room becomes an ask-only guest right away, followed by `ai-history {items}` (the last 20 exchanges) when the chat is visible to everyone |
| `ai-style {persona, sampling, thinking}` | host → all | the host changed the answer style (screens show a toast); takes effect on the next question |
| `ai-tele {k}` | worker → host | compute ms per frame kind (`spec` verify, `one` single token, `pre` prefill), an EMA, at most every 700 ms |
| `ai-map {nodes, st, live}` | host → all | the room map (`#swarm-map`): chain order, layers and compute per device, lap = GPUs + wire, tok/s, draft acceptance; ~1/s while answering |
| `ai-genstart {name, text, asker}` | host → all | carries the asker's peer id so that screen shows Stop |
| `ai-gendone {stats, ctx, failed}` | host → all | `ctx: {used, max}` feeds every screen's context meter |

## Compute frames

| Message | Payload | Use |
|---|---|---|
| `ai-hidden {pos}` → … → `ai-hiddenret` | one hidden state | single-token decode lap |
| `ai-hidden-b {basePos, n, spec?}` → … → `ai-hiddenret-b` | `n` hidden states (multiple of the batch width; up to 16) | batched prefill (`spec` absent) or speculative verify (`spec: 1`: the recurrent state is snapshotted after every non-final column) |

Control rides on frames. A frame's header flags byte (`room/transport.js` `packFlags`) carries `spec` (verify: snapshot the recurrent state after every column), `reset` (clear recurrent state and start from position 0 before this frame) and `rb` (restore the recurrent state to the snapshot after column `k` before this frame: the host rejected drafts after `k`). The host queues a reset or rollback and attaches it to the next frame it sends; each worker applies it, then forwards it with the frame. There is no separate `ai-rollback` message any more: sent on its own channel it could be overtaken by the next frame after a lost packet, and a worker would verify from the wrong state.

Checkpoint control rides the same way, in header bytes 24..31 (u16 each, 0 = none): `sv` saves this device's state (its layers' KV rows, DeltaNet states, conv windows) as GPU slot `sv` before the frame, `ld` loads slot `ld`, `dp` drops up to two slots (`0xffff` = all). A device applies them in the order rollback, save, drop, reset, load, then runs the frame and forwards the control with it. The host saves after every answer (`?ckpt=N` keeps the last N, default 2) and loads the longest saved answer that is a prefix of a new prompt, so a regenerate or branch prefills only what is new (docs/long-context-and-sessions.md).

Hidden states travel as binary frames: an f16-packed `Uint16Array` (10 KB for `dim = 5120`) with the wire format flag `WIRE_F16`; decoders accept f32 for older peers. Frames are correlated by position (`pos` / `basePos`), and the host keeps a timeout per outstanding lap.

## Ordering guarantees

- Data channels are ordered and reliable. Frames are sliced (≤ 4.6 KB) and striped across several associations, so consecutive frames can complete out of order at the receiver; the transport hands them over strictly in send order (a gap with no progress for 5 s is skipped, and a frame arriving after its gap was skipped is dropped rather than run out of order). A worker runs frames one at a time from a queue in that order, so recurrent states advance deterministically.
- Keep-alive: while a wire link has carried a frame in the last 1.5 s, each end sends a 1-byte message on a second negotiated channel (id 78, `swarm-ka`, unordered, never retransmitted) whenever it has sent nothing on that link for 10 ms (`?ka=ms`, `?ka=0` off). It keeps a phone's Wi-Fi out of power save between laps. Receivers ignore it; a peer without the channel drops it, so it is not a protocol change.
- Because of that, the host keeps up to 6 prefill rounds in flight: round r+1 runs on the host while round r is on a worker, and the chain works as a pipeline. Output is unchanged: every device sees the same frames in the same order.
- The prefill rounds come back as full hidden states, which the host feeds to the draft block (`mtpRun`) so the first speculative steps after a prompt draft from a warm cache.
- Inside a batched frame, columns are processed strictly in order; snapshot slots are indexed by global column (`frame.snap` packs base and total), so an 8-column verify split into two 4-column chunks on an older worker still rolls back correctly.

## Conversation state

The host owns the conversation: `{system, turns}` rendered to ChatML ids by `room/conversation.js`, with assistant turns kept as the exact sampled ids. It also tracks `fed`, the exact tokens every device's caches hold. A new question prefills only what follows `fed` when `fed` is a strict prefix of the new ids; otherwise (the persona changed, older turns were dropped to fit `MAX_SEQ`, a failure) it resets and prefills everything. Neither decode path pipes the end token through the chain, so both leave the caches holding exactly prompt + answer.

## Link lifecycle

| Message | Direction | Meaning |
|---|---|---|
| `hello {name, meta, v, died?}` | both ways on every link | `v` is the protocol version; on a mismatch each side says which one is older and who should reload (room/errors.js), and sends that as `bye {reason}` for a tab too old to word it itself. `died` is a joiner's crumb from a tab that was killed (surfaced on the host) |
| `hello {…, back: 1}` | returning guest → host | a device reconnecting to a host that resumed the room (it keeps its transcript, so no `ai-history`) |
| `leaving` | all → all | sent on `pagehide`; the receiver closes the link at once instead of waiting for ICE to notice (tens of seconds), so a departure mid-answer fails within a lap |

## Resuming a room

The host keeps `{code, name, model, turns, transcript, settings, peers}` in `localStorage` after every answer. A reloaded host page offers "resume room ABCD" for 15 minutes: it claims the same PeerJS id (retrying while the old registration expires), restores the conversation, waits up to 25 s for the devices that held layers and deals again; the next question re-prefills the history. When the host link closes, the other devices keep knocking on the host id every 3 s for a minute before calling the room over.

## Code mode

The host's coding agent and its previews (docs/design/harness-app.md, room/code.js). All of them go through the chat's visibility rules (`ai-visibility`): hidden screens get nothing. Everything but `ai-pv-want` is accepted only from the host; a peer treats the contents as untrusted text (typed and capped, shown with `textContent`), and runs previews in its own sandboxed frame.

| Message | Direction | Meaning |
|---|---|---|
| `ai-code-start {sid, mid, name, text}` | host → all | a new request to the agent; `sid` changes when the host opens another project |
| `ai-code-tok {mid, step, text}` | host → all | the model's visible text for a step, coalesced every 50 ms |
| `ai-code-tool {mid, step, i, name, brief, state, result?, diff?, ms?}` | host → all | tool call `i` of the request: `running`, `pending` (waiting for the host's approval), `approved`, `declined`, `done`, `error`; later messages for the same `i` update the card. `brief` ≤ 200 chars, `result` ≤ 600, `diff` (≤ ~4 KB) = `{path, isNew, lines, add, del, rows: [[op, text, skip?]], more}` |
| `ai-code-note {mid, text, err?}` | host → all | a line in the timeline: stopped, compaction, errors |
| `ai-code-done {mid, steps, reason, stats}` | host → all | the request ended (`done`, `stopped`, `limit`, `context`, `error`) |
| `ai-code-files {tree}` | host → all | the project's paths (≤ 500) after each file change |
| `ai-code-history {sid, items, tree}` | host → newcomer / all | the last 50 timeline items (same shapes as above) to a device that joins, and to everyone when the host switches project |
| `ai-pv {port, dir, entry, rev, bytes, manifest}` | host → all | a served port at revision `rev`: `manifest = [[path, type, hash, size]]` (sha-256, first 20 hex chars). Peers check the limits (400 files, 2 MB each, 8 MB total) and paths |
| `ai-pv-want {port, rev, hs}` | peer → host | the blobs a peer does not hold yet; the host answers only hashes in that port's current manifest |
| `ai-pv-blob {h, i, n, b}` | host → peer | chunk `i` of `n` (64 KB, `b` an ArrayBuffer) of blob `h`; the host waits while the data channel has over 1 MB buffered. The peer verifies the hash before using it |
| `ai-pv-stop {port}` | host → all | the port is no longer served |
| `ai-code-share {mid, port, rev, name, by}` | host → all | "Share with the room": a card in every timeline with Download (the app as one `.html` file, built by each device from its own hash-checked copy of that port's rev and sandboxed like a preview, harness/app-export.js) and Open full screen (the device's own preview frame). Kept in the history for late joiners |
| `ai-code-share-ask {port}` | member → host | a member who can drive asks the host to share a served port; the host sends `ai-code-share` naming them |

A code run holds the room's generation lock for all its steps, so chat questions asked meanwhile queue and run after it. None of this changes the frame format, so the protocol version stays 4: an older peer ignores these messages.

## API clients

`pooled serve` (the `cli/` package, docs/design/serve.md) joins a room as one more ask-only guest with no layers and turns HTTP requests from OpenAI / Anthropic clients into room messages. An API request is stateless and separate from the room's chat: the bridge sends the whole conversation every time, the host renders it with the model's own chat template and runs `roomGenerate` on the ids, as Code mode does. `ai.conv` is never read or written. The host keeps the exact ids of recent API answers (`room/api.js`, host memory only, cleared when another model loads) and replays them when a client sends an answer back, so a follow-up prefills only the new turn.

There are two kinds of ask. **v1** (`ai-ask {api: 1}`) is a plain conversation of user and assistant text. **v2** (`ai-ask {api: 2}`) adds tools, tool calls and their results, reasoning in the history, structured output and the tool-choice options; its answer streams tool calls as `ai-call` messages. Which one a bridge sends is negotiated, see below.

| Message | Direction | Meaning |
|---|---|---|
| `hello {…, meta: {…, api: 2, ctx}}` | host → guest | this host answers API asks, v1 and v2; `ctx` is its context size in tokens now. An older host says `api: 1` (v1 only). Guests drop `api` and `ctx` from the host's meta |
| `hello {name, v, meta: {api: 1, client, webgpu: false, ua: "API"}}` | bridge → host | an API client: never dealt layers (`webgpu: false`), shown with its own card, not counted as a device. With "Allow API clients" off, or after the host disconnected it this session, the answer is `bye {reason}` |
| `ai-ask {api: 1, rid, system, messages: [{role, text}], params: {maxTokens, temperature?, topK?, stop?, thinking?, thinkBudget?, client}}` | bridge → host | a v1 request. `rid` ≤ 32 chars (`[A-Za-z0-9_-]`), ≤ 200 messages, ≤ 400k chars, the last message the user's, `stop` ≤ 4 × 64 chars. With `thinkBudget`, once the reasoning used that many tokens the host closes the think block and continues from the same ids, so the rest of `maxTokens` goes to the answer. It joins the room's one queue (10, two per device). No `text` field: an older host drops it on its empty-question check |
| `ai-ask {api: 2, rid, system, messages, tools?, params}` | bridge → host | a v2 request; only to a host that said `api: 2`. `messages` (≤ 1000, normalized by the bridge): `{role: "user", text, aside?}` (`aside`: text sent along with tool results, its own user turn, not a new question), `{role: "assistant", text, calls?: [{name, args}], reasoning?}` (`args` an object), `{role: "tool", text}` (one result; a run of them is one turn, in call order). The last message is the user's or a tool result. `tools` ≤ 128 `{name, description, parameters}` (names `[A-Za-z0-9_.:-]{1,128}`, unique; each schema ≤ 32k chars; for XML-style models every property name must fit `<parameter=NAME>`: no `<`, `>` or line breaks). `params`: v1's plus `toolChoice` (`"auto"`, `"none"`, `"required"` or `{name}`), `allowed?` (names), `parallel` (default true), `maxCalls?`, `format?` (`{type: "json"}` or `{type: "schema", schema, name?}`), `effort?` (`low`, `medium`, `high`, `xhigh`, `max`). ≤ 1.5M chars in all. The host refuses (`ai-busy bad`) schemas past its compile caps and any tools when the model has no tool-call format |
| `ai-queued {pos, rid}` | host → bridge | waiting at position `pos` |
| `ai-busy {rid, code, why, n?, max?}` | host → bridge | refused: `queue` (full), `ctx` (the prompt is `n` tokens, over `max` = context − 32; never trimmed), `bad` (validation, v1 or v2), `off` (API clients disabled), `degraded`, `loading`, `gone` (removed from the queue by its `ai-stop`) |
| `ai-genstart {rid, api, client, promptTokens, model, style?}` | host → bridge | the answer starts; `api` echoes the ask's (1 or 2), `style` (v2) is the model's tool-call format (`json` or `xml`, for logs). The other screens get the usual `ai-genstart` (name "*name* · *client* (API)", the last real user message ≤ 2000 chars, `api: 1`) under `ai-visibility`; under `asker` only the host's screen shows it |
| `ai-token {rid, text, d, th?}` | host → bridge | answer text, stop strings already applied (a tail that may start a stop string is held back). `th: 1` marks reasoning (the tags are not sent). v2: content only; tool-call markup never goes out as `ai-token` |
| `ai-call {rid, i, name}` | host → bridge | v2: call `i` (from 0, in order) started; its name is complete and a declared (allowed) tool |
| `ai-call {rid, i, a}` | host → bridge | v2: the next fragment of call `i`'s arguments, a JSON object as text: the fragments of a call concatenate to its final arguments |
| `ai-call {rid, i, end: 1}` | host → bridge | v2: call `i` is complete |
| `ai-gendone {rid, api, reason, stopSeq?, usage: {in, out, think?}, reused, stats, failed, err?, calls?, open?}` | host → bridge | `reason`: `stop` (end token), `stop_seq`, `max`, `ctx` (context full), `abort` (the host pressed Stop), `error` (with `err`). `reused`: prompt tokens the room's caches already held. v2: `usage.think` (reasoning tokens), `calls: [{name, args}]` (the complete calls, `args` the final JSON text: the truth for a non-stream answer, a check for the streamed fragments), `open: {i, name}` when the answer ended inside call `i` |
| `ai-stop {rid}` | bridge → host | stop this request: honoured while it runs (after the lap in flight) or while it waits in the host's queue (answered `ai-busy {rid, code: "gone"}`). Sent when the HTTP client disconnects |
| `ai-ready-all {model, label, ctx?}` | host → all | as before, plus the model's display label (the bridge's `/v1/models`) and, from a v2 host, its context size |

**Negotiation.** An older host would take a v2 ask's `tools` for unknown fields and answer without them (a wrong answer, not an error), so the bridge never sends it one: it sends v2 asks only when the host's `hello` said `api: 2` (or more), and v1 asks otherwise. Against an older host, a request that needs v2 (tools, tool history, a format, `required` or a named tool) is answered 400 "the room's host runs an older Pooled without tool calling (reload the host page)"; plain requests work as before. The host dispatches on the ask's `api` (1 → the v1 validator and path, 2 → v2); an older bridge's v1 asks are answered exactly as before, and it never sees `ai-call` (only v2 asks produce them). If the host is reloaded to an older build while a v2 ask waits, its `ai-genstart` lacks `api: 2` and the bridge ends the request (500 "the room's host changed to an older Pooled; retry"). `PROTOCOL` stays 4.

**Bridge checks** (the host is another person's tab): call indexes in order and below 64, names declared, fragments only for open calls, each call's arguments ≤ 8 MB; `ai-token` and `ai-call` fragments both count toward `maxTokens` + 16; a violation ends the request.

The bridge answers every `ping` with `pong`. The host drops an API client that missed 6 pings in a row (about 15 s): a bridge that was killed never sends `leaving`, and its data channel can take over a minute to close while its answer holds the room.

API exchanges show in the chat with "via API · not part of this chat's memory", are not saved with the room (`saveHost`), and hide Continue / Regenerate (their history is the client's). A v2 answer shows its reasoning as a think block and each tool call as one line `→ name(args…)` (200 characters at most); tool results are not shown. Old peers ignore the new fields on known messages and never see `ai-ask {api}`, which is why this is not a protocol change: `PROTOCOL` stays 4.

## Versioning

`PROTOCOL` in `room/transport.js` is 4 (frame flags, ordered delivery, frame-borne reset/rollback, frame-borne checkpoints; 32-byte slice header). Protocol changes bump it; peers with another version are refused at `hello` with a message instead of failing mid-answer. See GOVERNANCE.md for what counts as a protocol change.
