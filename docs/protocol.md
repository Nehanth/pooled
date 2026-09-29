# Room protocol

Browsers in a room form a WebRTC mesh (PeerJS signaling for the introduction only). The host registers with PeerJS as `pooled-room-<CODE>`; before the rename to Pooled the prefix was `swarmllm-room-`, so a pooled.run tab and an old swarmllm.ai tab never meet in one room. One browser is the **host**: it owns the conversation, the tokenizer, the embedding table, the LM head and the sampler. The others are **workers** holding contiguous layer ranges; together they form a **chain** in layer order, with the last worker sending back to the host.

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
| `ai-linklost {name}` | worker → host | the worker's link to `name` (another device in the chain) went down and is being replaced; frames on it are gone, so the host fails the laps in flight now instead of timing out. Hosts that predate it ignore it |
| `ai-degraded {why}` | host → all | a device in the chain left (or stopped responding, see Drop detection); every lap in flight failed at once and the room waits for a re-deal |
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

Hidden states travel as binary frames: an f16-packed `Uint16Array` (10 KB for `dim = 5120`) with the wire format flag `WIRE_F16`; decoders accept f32 for older peers. Frames are correlated by position (`pos` / `basePos`), and the host keeps a timeout per outstanding lap: 30 s for a token lap and 90 s for a verify or prefill round until four decode laps have been measured, then `max(25 s, 6 × the slowest recent lap + 2 × RTT + 2 s)` for decode laps (`lapTimeout` in `room/liveness.js`). A dead device is caught sooner by drop detection; the lap timeout is for a frame lost on a live chain.

## Ordering guarantees

- Data channels are ordered and reliable. Frames are sliced (≤ 4.6 KB) and striped across several associations, so consecutive frames can complete out of order at the receiver; the transport hands them over strictly in send order. A missing frame on a reliable link is late, not lost (a device's network froze, a lost packet is waiting out SCTP's retransmission timer), so the receiver waits for it: it skips the gap at once only when every open channel has already delivered a newer frame (nothing older can still be queued), after 5 s when a channel closed in the last 15 s (the frame may have gone down with it), and otherwise after a 60 s backstop. A frame arriving after its gap was skipped is dropped rather than run out of order. Frames of up to 3 slices (a decode token's hidden state) are sent twice, on two associations, so one lost packet does not stall a token behind a retransmission timeout; receivers drop the second copy (`?wiredup=0` turns this off). Slices go to the open channel with the least data queued. A worker runs frames one at a time from a queue in that order, so recurrent states advance deterministically.
- Because of that, the host keeps up to 6 prefill rounds in flight: round r+1 runs on the host while round r is on a worker, and the chain works as a pipeline. Output is unchanged: every device sees the same frames in the same order.
- The prefill rounds come back as full hidden states, which the host feeds to the draft block (`mtpRun`) so the first speculative steps after a prompt draft from a warm cache.
- Inside a batched frame, columns are processed strictly in order; snapshot slots are indexed by global column (`frame.snap` packs base and total), so an 8-column verify split into two 4-column chunks on an older worker still rolls back correctly.

## Dead links

A network that passes no packets for longer than ICE's write timeout (about 15 s: frozen Wi-Fi, a closed laptop lid, a phone changing networks) kills a link's candidate pairs for good. Chrome then reports `connectionState: "failed"` while `iceConnectionState` stays `disconnected` and SCTP and every data channel still look open, so PeerJS never closes the connection and nothing sent on it arrives again. Shorter freezes recover on their own (the transport waits for late frames, above).

Each device watches every link's `RTCPeerConnection`. When one fails:

- The side that dialed it dials a new connection to the same peer id and sends `hello` with `back: 1`; both sides swap it in with a fresh wire (frame ids restart) and close the dead one and its stripes. The device keeps its place in the chain and its layers. A failed stripe is closed and redialed the same way.
- The other side waits 45 s for that, then closes the link (the device left, as before). A device whose network died silently is therefore dropped about a minute after it went quiet (before, its link stayed open and the room waited on it indefinitely).
- The host fails every lap in flight at once ("the link to X dropped; ask again") and the next question prefills from scratch. A worker whose link to another worker dropped tells the host with `ai-linklost`.

No PROTOCOL change: a peer that predates this sees an ordinary new connection from a device it knows (it already replaces the old entry), and ignores `ai-linklost`.

## Connecting: STUN and an optional TURN relay

Links are direct WebRTC connections. Every device uses public STUN servers to find its public address; that is enough on most home and office networks. When both sides are behind symmetric NAT or carrier-grade NAT, or a firewall blocks UDP, no direct path exists and the join fails after 15 s with "found the room, but the direct connection failed" (the Network box under the join form opens). A TURN relay fixes that: it forwards the traffic between the two devices.

Pooled does not run a relay and ships no credentials; it is off by default. To use one (your own [coturn](https://github.com/coturn/coturn), or a provider's):

- **Network box** under the join form: relay URL (`turn:relay.example.org:3478`, `turns:` for TLS, comma-separate several), username and password. Saved in this browser only.
- **URL**: `?turn=turn:relay.example.org:3478&turnuser=NAME&turncred=PASSWORD`. Overrides the saved setting.
- **Self-hosted deployments**: define `window.TURN_SERVERS` (an `RTCIceServer` array) before `room.js` loads.

ICE still prefers a direct path and only falls back to the relay when it has to. `?relay=1` (or "Always go through the relay") uses only the relay, so the other devices never see this device's IP address. Every device that cannot connect directly needs the relay configured; a device with an open network can reach a relayed one without it. Join links and QR codes never include `turn`, `turnuser`, `turncred` or `relay`. `pooledDebug()` shows each link's `path` (`direct` or `relay`), and the room log notes relayed links. A relay adds a hop to every token's round trip, so decode is slower through it than over a direct path. Tested with `node tests/e2e/room_chaos.mjs --plan turn` (a local test TURN server, `tests/e2e/turn_server.mjs`, with every direct candidate dropped).

## Conversation state

The host owns the conversation: `{system, turns}` rendered to ChatML ids by `room/conversation.js`, with assistant turns kept as the exact sampled ids. It also tracks `fed`, the exact tokens every device's caches hold. A new question prefills only what follows `fed` when `fed` is a strict prefix of the new ids; otherwise (the persona changed, older turns were dropped to fit `MAX_SEQ`, a failure) it resets and prefills everything. Neither decode path pipes the end token through the chain, so both leave the caches holding exactly prompt + answer.

## Link lifecycle

| Message | Direction | Meaning |
|---|---|---|
| `hello {name, meta, v, died?}` | both ways on every link | `v` is the protocol version; a mismatch gets `bye {reason}` and the newcomer is told to reload. `died` is a joiner's crumb from a tab that was killed (surfaced on the host) |
| `hello {…, back: 1}` | returning guest → host | a device reconnecting to a host that resumed the room (it keeps its transcript, so no `ai-history`) |
| `leaving` | all → all | sent on `pagehide`; the receiver closes the link at once instead of waiting for ICE to notice (tens of seconds), so a departure mid-answer fails within a lap |
| `ping {ts}` / `pong {ts}` | all → all / reply | every 2.5 s to every link (the RTT on each card); while an answer runs the host also pings each device in the chain every 500 ms (drop detection) |

## Drop detection

A device that dies without a `leaving` (its network drops, the tab freezes or is killed) used to hold the room until ICE gave up (~30 s) or a lap timed out. While an answer runs, the host now counts anything it receives from a chain device as a sign of life (a `pong` to its 500 ms `ping`, any control message, any slice on any of its wire channels) and holds a device silent for longer than `clamp(3 s + 3 × RTT, 3.5 s, 5 s)` (`room/liveness.js`): the host's screen says "<name> stopped responding; waiting for it (Stop gives up)", the answer in flight waits (frames on a frozen link are late, not lost) and a new question waits for it too. A device that comes back (a frozen Wi-Fi, a laptop lid) finishes the answer and keeps its place and layers, with no re-deal. If ICE gives up on the link first (~15 s), the answer fails at once and the link is redialed (Dead links). It counts as back only once it answers a `ping` sent after it went quiet (a full round trip). One still silent 30 s after it went quiet (`EVICT_MS`: long enough for a link ICE gave up on at ~15 s to be redialed, see Dead links) is dropped: its link is closed, which takes the departure path, `ai-degraded {why: "<name> stopped responding (layers …)"}` goes out and the host offers a re-deal. Silence before the answer began does not count, and a host tab that itself stalled (a timer tick more than 1.2 s late) restarts the count rather than blaming every device. The limit covers SCTP head-of-line stalls on a working link: with 5% loss the longest silence measured was 2.5 s at a 300 ms round trip (limit 3.9 s) and 3.65 s at 600 ms, i.e. 300 ms one way (limit 4.8 s) (tests/e2e/room_drop.mjs). Only the host judges and it only uses `ping`/`pong`, which every protocol version answers, so this needs no protocol change. `?hb=0` turns it off.

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

## API clients

`pooled serve` (the `cli/` package, docs/design/serve.md) joins a room as one more ask-only guest with no layers and turns HTTP requests from OpenAI / Anthropic clients into room messages. An API request is stateless and separate from the room's chat: the bridge sends the whole conversation every time, the host renders it with the chat's own template (`buildIds`) and runs `roomGenerate` on the ids, as Code mode does. `ai.conv` is never read or written. The host keeps the sampled ids of its last 8 API answers (`room/api.js`, host memory only) and uses them when a client resends an answer byte for byte, so a follow-up prefills only the new turn.

| Message | Direction | Meaning |
|---|---|---|
| `hello {…, meta: {…, api: 1}}` | host → guest | this host answers API asks. The bridge refuses to serve without it (an older host) |
| `hello {name, v, meta: {api: 1, client, webgpu: false, ua: "API"}}` | bridge → host | an API client: never dealt layers (`webgpu: false`), shown with its own card, not counted as a device. With "Allow API clients" off, or after the host disconnected it this session, the answer is `bye {reason}` |
| `ai-ask {api: 1, rid, system, messages: [{role, text}], params: {maxTokens, temperature?, topK?, stop?, thinking?, thinkBudget?, client}}` | bridge → host | one request. `rid` ≤ 32 chars (`[A-Za-z0-9_-]`), ≤ 200 messages, ≤ 400k chars, the last message the user's, `stop` ≤ 4 × 64 chars. With `thinkBudget`, once the reasoning used that many tokens the host closes the think block and continues from the same ids, so the rest of `maxTokens` goes to the answer. It joins the room's one queue (10, two per device). No `text` field: an older host drops it on its empty-question check |
| `ai-queued {pos, rid}` | host → bridge | waiting at position `pos` |
| `ai-busy {rid, code, why, n?, max?}` | host → bridge | refused: `queue` (full), `ctx` (the prompt is `n` tokens, over `max` = context − 32; never trimmed), `bad` (validation), `off` (API clients disabled), `degraded`, `loading`, `gone` (removed from the queue by its `ai-stop`) |
| `ai-genstart {rid, api: 1, client, promptTokens, model}` | host → bridge | the answer starts. The other screens get the usual `ai-genstart` (name "*name* · *client* (API)", the last user message ≤ 2000 chars, `api: 1`) under `ai-visibility`; under `asker` only the host's screen shows it |
| `ai-token {rid, text, d, th?}` | host → bridge | answer text, stop strings already applied (a tail that may start a stop string is held back). `th: 1` marks think-block text (the tags are not sent). The asking bridge always gets every token, whatever the visibility |
| `ai-gendone {rid, api: 1, reason, stopSeq?, usage: {in, out}, reused, stats, failed, err?}` | host → bridge | `reason`: `stop` (end token), `stop_seq`, `max`, `ctx` (context full), `abort` (the host pressed Stop), `error` (with `err`). `reused`: prompt tokens the room's caches already held |
| `ai-stop {rid}` | bridge → host | stop this request: honoured while it runs (after the lap in flight) or while it waits in the host's queue (answered `ai-busy {rid, code: "gone"}`). Sent when the HTTP client disconnects |
| `ai-ready-all {model, label}` | host → all | as before, plus the model's display label (the bridge's `/v1/models`) |

The bridge answers every `ping` with `pong`. The host drops an API client that missed 6 pings in a row (about 15 s): a bridge that was killed never sends `leaving`, and its data channel can take over a minute to close while its answer holds the room.

API exchanges show in the chat with "via API · not part of this chat's memory", are not saved with the room (`saveHost`), and hide Continue / Regenerate (their history is the client's). Old peers ignore the new fields on known messages and never see `ai-ask {api}`, which is why this is not a protocol change: `PROTOCOL` stays 4.

## Versioning

`PROTOCOL` in `room/transport.js` is 4 (frame flags, ordered delivery, frame-borne reset/rollback, frame-borne checkpoints; 32-byte slice header). Protocol changes bump it; peers with another version are refused at `hello` with a message instead of failing mid-answer. See GOVERNANCE.md for what counts as a protocol change.
