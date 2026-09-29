# Room protocol

Browsers in a room form a WebRTC mesh (PeerJS signaling for the introduction only). The host registers with PeerJS as `pooled-room-<CODE>`; before the rename to Pooled the prefix was `swarmllm-room-`, so a pooled.run tab and an old swarmllm.ai tab never meet in one room. One browser is the **host**: it owns the conversation, the tokenizer, the embedding table, the LM head and the sampler. The others are **workers** holding contiguous layer ranges; together they form a **chain** in layer order, with the last worker sending back to the host. By memory, phones hold no layers at all while the host and the other computers can hold the model (`phonesToLeaveOut`; `?phonelayers=1` overrides): they join as ask-only guests.

## Lifecycle

| Message | Direction | Meaning |
|---|---|---|
| `ai-wait` | host → worker | join accepted; wait for assignment |
| `ai-load {model, range, next, host}` | host → worker | download and load layers `[range[0], range[1])`; forward to `next` |
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
- Because of that, the host keeps up to 6 prefill rounds in flight: round r+1 runs on the host while round r is on a worker, and the chain works as a pipeline. Output is unchanged: every device sees the same frames in the same order.
- The prefill rounds come back as full hidden states, which the host feeds to the draft block (`mtpRun`) so the first speculative steps after a prompt draft from a warm cache.
- Inside a batched frame, columns are processed strictly in order; snapshot slots are indexed by global column (`frame.snap` packs base and total), so an 8-column verify split into two 4-column chunks on an older worker still rolls back correctly.

## Conversation state

The host owns the conversation: `{system, turns}` rendered to ChatML ids by `room/conversation.js`, with assistant turns kept as the exact sampled ids. It also tracks `fed`, the exact tokens every device's caches hold. A new question prefills only what follows `fed` when `fed` is a strict prefix of the new ids; otherwise (the persona changed, older turns were dropped to fit `MAX_SEQ`, a failure) it resets and prefills everything. Neither decode path pipes the end token through the chain, so both leave the caches holding exactly prompt + answer.

## Link lifecycle

| Message | Direction | Meaning |
|---|---|---|
| `hello {name, meta, v, died?}` | both ways on every link | `v` is the protocol version; a mismatch gets `bye {reason}` and the newcomer is told to reload. `died` is a joiner's crumb from a tab that was killed (surfaced on the host) |
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

A code run holds the room's generation lock for all its steps, so chat questions asked meanwhile queue and run after it. None of this changes the frame format, so the protocol version stays 4: an older peer ignores these messages.

## Versioning

`PROTOCOL` in `room/transport.js` is 4 (frame flags, ordered delivery, frame-borne reset/rollback, frame-borne checkpoints; 32-byte slice header). Protocol changes bump it; peers with another version are refused at `hello` with a message instead of failing mid-answer. See GOVERNANCE.md for what counts as a protocol change.
