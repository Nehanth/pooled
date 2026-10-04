---
title: Wire protocol
description: The messages devices in a room send each other, how hidden states travel, what order is guaranteed, and how the protocol is versioned.
eyebrow: Internals
sidebar:
  label: Wire protocol
  order: 2
---


The messages devices in a room send each other, and what order they are guaranteed to arrive in.

A room is a WebRTC mesh between browsers. PeerJS signaling is used only to introduce devices; once links are open, no traffic goes through a server.

The host registers with the signaling server as `pooled-room-<CODE>`. Every link starts with a `hello` that carries the device's name, its protocol version and a `meta` object (memory it lends, whether it has WebGPU, its measured GPU speed, and so on).

:::note[Full reference]
This page covers the shape of the protocol. Every message and field is listed in [`docs/protocol.md`](https://github.com/Nehanth/pooled/blob/main/docs/protocol.md) on GitHub.
:::

## Joining and loading

| Message | Direction | Meaning |
|---|---|---|
| `hello {name, meta, v}` | both ways, every link | First message on a link. On a version mismatch, each side says which one is older and who should reload. |
| `auth` / `auth-proof` / `admit {via, hp, mk}` | host ↔ joining device | The joining device proves the invite key or a pass, bound to this link, and the host proves it back; never the secret itself. See [Security model](/docs/internals/security#who-is-on-a-link). |
| `mesh {p}` | both ends, links between devices and stripes | Proof of the room's key, before the link carries anything. |
| `ai-inv-req` / `ai-inv` | host ↔ all | Before dealing, the host asks which byte ranges of the model each device has cached. |
| `ai-load {v, model, range, next, host}` | host → worker | Load layers `[range[0], range[1])` and forward results to `next`. |
| `ai-wget` / `ai-wpart` / `ai-wack` | device ↔ device | Take a cached byte range from another device in 64 KB parts, with at most 8 MB unacknowledged. Any failure falls back to the network. |
| `ai-progress`, `ai-ready`, `ai-ready-all` | worker → host, host → all | Download progress; layers loaded; the room is online. |
| `ai-share {gb \| drop}` | host → device | The device's tab was killed while loading: re-deal with half its share, or without it after a second kill. |

A device that joins a room already online becomes an ask-only guest and gets the last 20 exchanges (when the chat is visible to everyone).

## Asking and answering

Anyone can `ai-ask`; one generation runs at a time and the rest wait in the host's queue (`ai-queued`, at most 10, two per device). The host mirrors each question and streamed answer to every screen with `ai-genstart` / `ai-token` / `ai-gendone`, filtered by `ai-visibility` (`all`, `host` or `asker`). `ai-stop` ends an answer after the lap in flight, `ai-cmd` continues or regenerates, and `ai-reset` clears the screens; the engines' reset rides on the next compute frame.

## Compute frames

Hidden states travel as binary frames, never as JSON.

| Frame | Payload | Used for |
|---|---|---|
| `ai-hidden {pos}` → … → `ai-hiddenret` | one hidden state | A single-token decode lap |
| `ai-hidden-b {basePos, n, spec?}` → … → `ai-hiddenret-b` | `n` hidden states, up to 16 | Batched prefill, or a speculative verify when `spec` is set |

Each hidden state is f16 (10 KB on the 27B). Decoders still accept f32 from older peers.

### Control rides on frames

Resets, rollbacks and checkpoints are flags in the frame header, not separate messages. The host queues one and attaches it to the next frame it sends. Each worker applies it, runs the frame and forwards it with the frame.

| Header field | Effect |
|---|---|
| `spec` | Snapshot the recurrent state after every column (speculative verify) |
| `reset` | Clear the recurrent state and start from position 0 |
| `rb` | Restore the recurrent state to the snapshot after column `k`: the host rejected the drafts after it |
| `sv`, `ld`, `dp` | Save this device's state as GPU slot `sv`, load slot `ld`, drop up to two slots |

A device applies them in this order: rollback, save, drop, reset, load.

:::note[Why not a separate message]
A rollback sent on its own channel could be overtaken by the next frame after a lost packet, and a worker would then verify from the wrong state. Riding on the frame removes that race.
:::

### Lap timeouts

The host keeps a timeout for every lap in flight. Until four decode laps have been measured it is 30 s for a token lap and 90 s for a verify or prefill round. After that, a decode lap gets `max(25 s, 6 × the slowest recent lap + 2 × RTT + 2 s)`, never more than those 30 s or 90 s. Prefill rounds keep the 90 s timeout. A dead device is usually caught sooner by drop detection (below).

## Ordering

- **Frames arrive in send order.** Frames are sliced (at most 4.6 KB each) and striped over several data channels, so they can complete out of order. The transport hands them over strictly in the order they were sent.
- **A missing frame is late, not lost.** Channels are reliable, so the receiver waits. It skips a gap at once only when every open channel has already delivered a newer frame, after 5 s when a channel closed recently, and otherwise after a 60 s backstop. A frame that arrives after its gap was skipped is dropped.
- **Small frames go twice.** Frames of up to 3 slices (a decode token's hidden state) are sent on two channels, so one lost packet does not stall a token behind a retransmission timeout (`?wiredup=0` turns this off).
- **Workers run frames one at a time**, in order, so recurrent state advances the same way on every device.
- **Keep-alive.** While a link has carried a frame in the last 1.5 s, each end sends 1 byte on a separate unordered channel (id 78) whenever it has sent nothing for 10 ms. This keeps a phone's Wi-Fi out of power save between laps. `?ka=0` turns it off.

## Liveness

| Mechanism | What it does |
|---|---|
| `ping` / `pong` | Every 2.5 s on every link. While an answer runs, the host also pings each chain device every 500 ms. |
| `leaving` | Sent on `pagehide`, so a closed tab fails the answer within a lap instead of after ICE notices. |
| `ai-degraded` | A chain device silent for 15 s (a computer) or 60 s (a phone) is dropped, never while it loads its layers, and the host offers a re-deal. |
| `hello {back: 1}` | A device returning after a lock, a background tab, a reload or a dead-link redial. The host re-seats it in its old slot by name. |
| `ai-redeal` | The host deals the layers again over the devices now in the room. |

The stalled-device notice, timings and what each screen shows are on [Device drops and recovery](/docs/rooms/recovery). A reloaded host resumes its room from `localStorage` for 15 minutes, as described there. Measured timings: [Benchmarks](/docs/internals/benchmarks#bad-networks).

## Connecting: STUN and TURN

Every device uses public STUN servers to find its public address. When no direct path exists (symmetric or carrier-grade NAT, or blocked UDP), the join fails after 15 s (up to 40 s while the browser still tries paths) and the Network box opens. pooled.run hands each device short-lived TURN credentials from `POST /api/turn`; a relay the user sets wins over it, and ICE still prefers a direct path. Model weights never cross a relayed link: the serving side answers `ai-wpart {miss: 1}` and the device fetches the range from the network. See [TURN relay](/docs/self-host/turn).

## Code mode messages

Code mode sends its timeline, tool calls, file tree and preview files to the other screens with `ai-code-*` and `ai-pv*` messages. They follow the chat's visibility rule. Peers accept them only from the host, treat their content as untrusted text, and check every preview blob's SHA-256 hash and the limits (400 files, 2 MB each, 8 MB in all) before using it.

## API clients

`pooled serve` joins a room as one more ask-only guest with no layers. It turns each HTTP request into an `ai-ask {api: 1 | 2}` with the whole conversation. The host renders it with the model's chat template and runs it in the same queue as the chat.

- **v1** asks carry plain user and assistant text.
- **v2** asks add tools, tool calls and results, reasoning, structured output and tool choice. Tool calls stream back as `ai-call` messages.

The bridge sends v2 asks only to a host whose `hello` said `api: 2`. Against an older host, a request that needs tools gets a 400 asking the host to reload.

The host drops an API client that misses 6 pings in a row (about 15 s), so a killed bridge cannot hold the room. More on the bridge: [Serve API](/docs/serve).

## Versioning

`PROTOCOL` in `room/transport.js` is **4**. It covers the frame flags, ordered delivery, frame-borne reset, rollback and checkpoints, and the 32-byte slice header.

A change to the frame format bumps it. Peers on another version are refused at `hello` with a message, instead of failing in the middle of an answer. New messages that old peers can safely ignore (the keep-alive, `ai-linklost`, the Code mode and API messages) do not bump it. [GOVERNANCE.md](https://github.com/Nehanth/pooled/blob/main/GOVERNANCE.md) says what counts as a protocol change.
