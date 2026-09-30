# @pooled/room-node

A Pooled room device in a Node.js process. It runs Pooled's own WebGPU engine (unmodified, from
`engine/`) on Dawn through the npm `webgpu` package, and speaks the room protocol
([docs/protocol.md](../../docs/protocol.md)) over WebRTC through `node-datachannel` and PeerJS. So it
can sit in the same room as browser tabs and phones on pooled.run: host the room, or join one and
hold layers. There is no browser and no child process; the GPU work and the WebRTC links share one
event loop.

It is what the OpenClaw plugin ([packages/openclaw](../openclaw)) runs inside the OpenClaw gateway.
Status: proof of concept, not published to npm. It runs from a Pooled checkout (it imports
`engine/`, `room/`, `harness/` and `cli/lib/` by relative path).

## Install

```sh
cd packages/room-node && npm install --omit=dev     # webgpu 0.6.1 (Dawn), node-datachannel 0.33.4, peerjs 1.5.4
```

Needs Node 22 or newer. The `webgpu` package ships prebuilt Dawn for Linux (Vulkan), Windows (D3D12)
and macOS 26 or newer (Metal); no node-gyp.

## Use

```js
import { createRoom, joinRoom } from "./packages/room-node/index.js";

// host: this machine creates room room.code, holds the embedding, the head and its share of layers
const room = await createRoom({ model: "qwen3.6-35b-moe", pledgeGB: 16, modelDir: "/path/to/models" });
console.log(`open https://pooled.run/room/${room.code} on the other device`);
await room.start({ minDevices: 2, waitMs: 120000 });   // optional: wait for devices, then deal the layers
for await (const ev of room.ask([{ role: "user", content: "Why is the sky blue?" }], { maxTokens: 256 })) {
  if (ev.type === "token") process.stdout.write(ev.text);
}

// worker: hold layers in someone else's room (a browser tab, or another node)
const node = await joinRoom("K7QX", { pledgeGB: 16 });
```

### createRoom(options) / joinRoom(code, options) -> RoomNode

| Option | Meaning |
|---|---|
| `model` | (createRoom) a key of `room/models.js` MODELS, default `qwen3-1.7b` |
| `code` | (createRoom) the room code; default a random one of six characters (`room/joingate.js` alphabet) |
| `gate` | (createRoom) hold new links at the room page's gate (`gate.js`, docs/protocol.md "Joining a room"): a device gets in with the room's invite key (`node.inviteFragment`, `#k=…` on the room link), a pass the host gave it before, or `allowJoin()`. Default false (anyone with the code, as before); `pooled host` turns it on |
| `ask` | (createRoom, with `gate`) false lets anyone with the code in without asking (`pooled host --allow-all`); default true |
| `key`, `pass` | (joinRoom) the invite key from a room link, and a pass from an earlier admit: sent only to the host (`hello {join: 1, key, pass}`). The node waits in a gated host's lobby (`lobby`, then `admitted`; `node.pass` is what the host gave it) and walks straight into a host from before the gate |
| `pledgeGB` | GPU memory this device lends; default half its largest buffer (1 to 64) |
| `ctx` | context to ask for, clamped by `room/models.js` CTX (1.7B 16k, 27B 64k, MoE 128k); default the largest for the qwen35 models (MoE 128k, 27B 64k) and the room default (8k) for the 1.7B, then lowered to what the smallest GPU binding limit in the room holds (`maxBindMB` in each hello; none counts as 128 MiB). Every device gets it with its `ai-load` |
| `ckpt` | the host's checkpoints (`ckpt.js`): `{ answers, pins, minPin, turns }` (default 4 answer and turn checkpoints, 4 pinned prefixes, pin a fixed start of 1024+ tokens, turn checkpoints on), or `false` for none |
| `modelDir` | local model files: `<model key>/<file name>` as `pooled pull` keeps them (`qwen3-1.7b/Qwen3-1.7B-Q8_0.gguf`), or the layout of `source.js` LOCAL (`qwen17/model.gguf`, `q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf`, ...); else HTTP range reads of the model's URL, as the page makes |
| `signal` | a PeerServer `host:port` (TLS only on port 443), or `room/signal.js` specs as a comma list (`wss://host/path,cloud`): the first that answers is used, the next when one is down; default the PeerJS cloud server pooled.run uses. A signaling server that drops later is reconnected with backoff (the `signaling` event) |
| `setup` | `setupNode` options: `webgpu` (an async loader returning `{ create, globals }`, for a caller that resolves Dawn from its own install, as `pooled join` does), `dawnFlags` |
| `name` | this device's name in the room |
| `expectHost` | `joinRoom`: the host's name this device will serve (a rejoin after its room was over). Room codes are short and get reused: a host of another name is another room, and the node leaves it before it hears anything else from it (the `otherhost` event). A worker always does this when a host of another name answers under the same code later (knocking after the host link dropped) |
| `flags` | engine switches, as the room page's `?query` (`engine/preset.js`) |
| `visibility` | who sees the answers on the room's screens (`room/visibility.js`): `all` (the room page's default), `asker` (only the asker; other screens get the "answering…" stand-in) or `host`. Set once; newcomers are told the mode |
| `allowApi` | answer API asks from devices that joined as API clients (`pooled serve`, another OpenClaw in join mode); default true, `false` refuses them |
| `autoRedeal` | re-deal without a device that does not come back within 60 s (default true) |
| `gbps` | pin the GPU copy speed the room uses to pick the model host (default: measured) |
| `log` | `(line) => {}`; default: the `log` event |

### RoomNode

- `code`, `name`, `hosting()` (this device runs the model: it created the room, or a browser host
  picked it as the model host), `status()` (devices, pledges, split, online, degraded).
- `start(model?, { minDevices, waitMs })`: deal the layers over the devices in the room now (by
  pledge; phones only when the computers cannot hold the model) and load this device's share.
  `ask()` calls it when the room has not started.
- `ask(messages, opts)`: an async iterator of events. `messages` are OpenAI-style (`system`, `user`,
  `assistant` with `tool_calls`, `tool` with `tool_call_id`). `opts`: `maxTokens`, `temperature`,
  `topK`, `stop`, `thinking`, `tools` (`[{ name, description, parameters }]`), `toolChoice`,
  `parallel`, `format`, `signal`, `client`. Events: `{ type: "start", promptTokens }`,
  `{ type: "token", text, think? }`, `{ type: "call", i, id, name }`, `{ type: "call", i, a }`
  (argument fragments), `{ type: "call", i, end: 1, args }`, then
  `{ type: "done", reason, usage, reused, calls: [{ id, name, args }], stats }` or
  `{ type: "done", reason: "error", code, err }`.
- `request(body, handler, { rid })`: the raw API ask, the same `ai-ask` body `pooled serve` sends
  (`cli/lib/common.js` askBody); `handler` gets the same room messages a `Bridge.ask` handler gets.
  Returns `{ rid, stop() }`. `hostMeta` is what the host's hello says (`{ api: 2, ctx }`).
- The gate (a host created with `gate`): `inviteFragment` (`#k=<key>`), `waitingJoins()` (the
  devices in the lobby, oldest first, with `line`: "otter wants to join (Mac, 8 GB)"),
  `allowJoin(id?)` / `denyJoin(id?)` (the oldest when no id). Events `joinrequest` (`{ id, name,
  meta, line }`) and `joinrequests` (the lobby changed). A device: `admission` (`wait`, `lobby`,
  `in`), events `lobby` and `admitted`.
- `redeal(why)`, `close()` (closes every link without counting them as departures, stops the re-deal timer, frees the engine and GPU). Events: `log`, `members`, `online`, `degraded`, `loaded`, `progress`,
  `hostgone`, `back`, `roomover`, `chat`, `chatanswer`, `answer`, `loadprogress` (this device's load, %),
  `signaling` (false when the signaling server dropped, true when it is back), `otherhost` (`{ was, now }`: the code now belongs to a host of another name, and this node left it), `version` (a hello on
  another protocol: `{ theirs, theyHost, name }`), `bye`. `joinRoom` rejects with `code: "room-not-found"`
  as soon as the signaling server says there is no such room.

## From the command line

`pooled join <CODE>` and `pooled host` in [`@pooled/cli`](../../cli/README.md#lend-a-computer-pooled-join--pooled-host-preview) are the supported way (memory rule, status line, reconnects, errors). `join.mjs` is the bare version, kept for scripts. Either way, a device that holds layers receives the hidden state of every token the room computes, which carries its prompts and answers: lend only to rooms you trust.

```sh
node packages/room-node/join.mjs K7QX --gb 12 --name mac-studio [--signal host:port] [--models dir] [--state status.json]
```

## One tool-call path

`ask()` goes through exactly the code `pooled serve` and the browser host run:
`cli/lib/common.js` (normalize, check, `askBody`) -> `RoomNode.request` -> `room/api.js`
(`validateApiAsk`, `apiPrompt2`, `apiRun2`: chat template, call grammar, call parsing) ->
`cli/lib/answer.js` (`Ask`: the client's checks on what the host sent). Nothing of tool calling is
copied into this package.

## What it takes from room.js

`room.js` is one DOM module and none of it is imported. The DOM-free modules are shared as they are
(transport, wire, plan, pledge, models, conversation, sampling, liveness, lookup, resume, gpuspeed,
api). The link layer, dealing, laps, prefill, plain and speculative decode (with the host fuse:
`headAhead`, `preTrunk`), the worker's frame loop and asks are extracted with the DOM taken out
(same logic, same messages). Current as of the room protocol after #257-#260:

- the faster GPU hosts: the hello carries `gbps` (`room/gpuspeed.js`), so a browser host's
  `pickModelHost` can pick this node (`ai-start-req`);
- one submit per token on a chain host (`headAhead` / `preTrunk`), `ai-wake` for phone workers;
- devices that drop out: a chain device that leaves degrades the room, and coming back under its
  name (`hello {back: 1}`) re-seats it (`ai-load` for its slot, `ai-next {relink}` / `ai-linked`);
  an answer in flight waits and carries on (`room/resume.js`); after 60 s the room re-deals. A worker
  that still holds its layers skips the reload; a worker whose host link drops knocks for a minute;
- `ai-share` (a smaller share after a killed load), `ai-linklost` (fail the laps in flight);
- serve v2: `hello {api: 2, ctx}`, v1 and v2 asks, `ai-call`.

- checkpoints (`ckpt.js`, the browser room's design from #251 / #260): the host saves the state of
  every device (`sv` on the next frame) at a prompt's fixed start and after every answer, and resumes
  a new prompt from the longest saved prefix of it (`ld`), so only what is new is prefilled:
  - pinned prefixes: the system prompt + tools, and an agent's own cache boundary inside its system
    prompt (OpenClaw's `<!-- /openclaw:attempt:STABLE -->`: the date and model name that follow it
    change, the ~11k tokens before it do not). Answers never evict them; up to 4, by last use.
  - turn checkpoints: where the prompt's last user turn starts (`turnPoint`). OpenClaw ends every
    call with a user turn of per-call context (`<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>`) that the next
    call drops, so an answer checkpoint is never a prefix of the next call in a tool loop; the turn
    checkpoint is, and the next call reads only the assistant turn and the tool results. Kept with
    the answer checkpoints. `POOLED_CKPT_DEBUG=1` logs where each prompt leaves each checkpoint.
  - answer checkpoints: up to 4 (with the turn checkpoints), evicted by GreedyDual (the cost to rebuild past the pinned prefix,
    aged by use), so a session title or compaction request does not push out the conversation.
  - one index for the whole host: a new session with the same system prompt starts from the pin.
  - a dense model (the 1.7B) has GPU slots too (`engine/dense.js`); a chain with a tab from before
    that change (no `ckpt: 1` in its `ai-ready`) runs without checkpoints.
  - frames carry at most two drops (`room/transport.js`): evictions queue and go out two per frame.
  - a failed lap, a device leaving or coming back, or a re-deal drops every checkpoint (`dp` all).

Left out for now: disk copies of checkpoints (a gateway restart starts cold), resuming a reloaded host,
the speed split, dead-link redial (the ICE state watch), changing the visibility while the room runs (it is set when the node is created: `visibility`), Code mode, the room map,
weight caches and peer-to-peer weights, the bandwidth test.

## Tests

```sh
node --test packages/room-node/test/*_test.mjs          # no GPU, no network
# GPU (a local PeerServer, the room page from this checkout, weights from MODELS):
node packages/room-node/test/solo.mjs "Why is the sky blue? Answer in two sentences." 48 > /tmp/solo.json
REF=/tmp/solo.json MODELS=/path/to/models node packages/room-node/test/e2e.mjs nodehost   # node hosts, a tab joins
MODELS=/path/to/models node packages/room-node/test/e2e.mjs tabhost    # a tab hosts, the node joins
MODELS=/path/to/models node packages/room-node/test/e2e.mjs nodepair   # two nodes, no browser
MODELS=/path/to/models node packages/room-node/test/e2e.mjs api        # pooled serve's bridge asks a node host
```

`e2e.mjs` needs a Chromium with WebGPU (`CHROME_BIN`, default playwright's) and prints one JSON line;
with `REF` it says whether the room's answer matches the solo engine token for token (`matchSolo`).
