# @pooled/openclaw

An OpenClaw provider plugin. Pick **Pooled** in OpenClaw and this machine becomes a device in a
Pooled room: it holds its share of the model's layers on the local GPU with Pooled's own engine
(WebGPU through Dawn, inside the OpenClaw gateway process, via
[@pooled/room-node](../room-node)). No browser tab, no `pooled serve`, no API key. It can start the
room or join one. Other devices join over WebRTC: another machine with this plugin, or a browser tab
or phone on pooled.run. Together they run a model none of them can run alone (the 35B MoE needs
about 22.5 GB across the room; tested on a Spark + a 36 GB Mac, and with an iPhone added).

Status: proof of concept, not published. Tested with OpenClaw 2026.9.6 and 2026.9.7 (Linux GB10 and
macOS 27 on an M5 Max). It runs from a Pooled
checkout (it imports `packages/room-node`, `room/`, `harness/` and `cli/lib/` by relative path).

## Set up

```sh
cd packages/room-node && npm install --omit=dev           # Dawn, node-datachannel, peerjs (~140 MB: the webgpu package ships Dawn for all five OS/arch pairs)
openclaw plugins install --link --accept-capabilities --force "$PWD/packages/openclaw"
openclaw onboard        # or: openclaw models auth login --provider pooled --set-default
```

`--force`: 2026.9.7 refuses a local path outside ClawHub review without it; the path must be
absolute. With npm 11, installing OpenClaw itself needs its install scripts allowed
(`--allow-scripts=@google/genai,esbuild,koffi,protobufjs,openclaw` for a global install, or an
`allowScripts` field in a project's package.json).

Onboarding offers **Pooled (run a model across your devices)** with two choices:

- **Start a room on this device**: pick the model (Qwen3 1.7B, Qwen3.8 27B, Qwen3.6 35B MoE), the
  GPU memory this device lends (GB), and how many devices to wait for. It shows the room code and
  `https://pooled.run/room/<CODE>`.
- **Join a room**: enter the code from the device that started it and the GB to lend.

It writes:

- `models.providers.pooled`: one catalog model, `pooled/<model>` (or `pooled/room` when joining),
  with the room's context (1.7B 16k, 27B 64k, MoE 128k: `room/models.js` CTX) and no API key;
- `plugins.entries.pooled.config`: `{ mode, code, model, pledgeGB, minDevices }`;
- the default model, and for the 1.7B a file-tools-only profile (a small model cannot follow
  OpenClaw's whole tool catalog).

Non-interactive: `openclaw onboard --non-interactive --auth-choice pooled` with `POOLED_MODE`
(`host` / `join`), `POOLED_CODE`, `POOLED_MODEL`, `POOLED_PLEDGE_GB`.

### Config (`plugins.entries.pooled.config`, each overridable by a `POOLED_*` variable)

| Key | Env | Meaning |
|---|---|---|
| `mode` | `POOLED_MODE` | `host` or `join` |
| `code` | `POOLED_CODE` | the room code (4 to 6 characters; use 4 when a phone or tab joins by typing it: the room page's code box keeps 4) |
| `model` | `POOLED_MODEL` | the model a host starts (`room/models.js` key) |
| `pledgeGB` | `POOLED_PLEDGE_GB` | GPU memory this device lends |
| `minDevices` | `POOLED_MIN_DEVICES` | devices to wait for before dealing the layers (default 1) |
| `waitSeconds` | `POOLED_WAIT_S` | how long a request waits for devices or the host's model (default 120) |
| `ctx` | `POOLED_CTX` | context to ask for (default the model's largest) |
| `signal` | `POOLED_SIGNAL` | a PeerServer `host:port` (default the PeerJS cloud server pooled.run uses) |
| `modelDir` | `POOLED_MODELS` | local model files (`packages/room-node/source.js` LOCAL layout) instead of downloads |
| `name` | `POOLED_NAME` | this device's name in the room |
| `page` | `POOLED_PAGE` | a room page other than pooled.run (local tests) |
| `allowApiClients` | `POOLED_ALLOW_API` | host: answer API asks from other devices (another OpenClaw in join mode, `pooled serve`); default on, `false` keeps the GPU to this OpenClaw |

## How it works

- The gateway's `pooled-room` service opens the room at startup, so other devices can join before
  the first question. As host it deals the layers once the room has `minDevices` devices and enough
  pledged memory; as a joiner it holds layers when the host deals it some. A state file
  (`$OPENCLAW_STATE_DIR/pooled-room.json`) shows the room, the devices and the split.
- OpenClaw's model calls go through `createStreamFn`, a custom transport with no HTTP hop:
  - `src/convert.js` turns OpenClaw's context (system prompt, messages, tool calls and results,
    tools) into the request `pooled serve` builds from an OpenAI body, and runs the same
    normalization and checks (`cli/lib/common.js` finishRequest, askBody);
  - `src/pool.js` sends it to the room's host: this process (`RoomNode.request`) when this device
    runs the model, else the host over WebRTC (the `pooled serve` bridge, `cli/lib/room.js`);
  - the host renders the chat template, constrains tool calls with the grammar and parses them
    (`room/api.js` apiRun2), whether it is this node or a browser tab;
  - `src/stream.js` checks the answer as `pooled serve` does (`cli/lib/answer.js` Ask: calls in
    order, only declared tools, arguments a JSON object) and streams OpenClaw's `text_*`,
    `thinking_*` and `toolcall_*` events.

Room problems show in the chat as `Pooled: ...`: waiting for devices (with the link), not enough
memory (the pledges against what the model needs), a device left while it held layers (the room
waits for it to come back, then re-deals), the context is full, the host does not allow API
clients, the host runs an older Pooled without tool calling, no model running yet.

## Who can see and steer what

A room is open to anyone who has its code; there is no password and no approval step. Codes are 4 to 6
characters on the public PeerJS server by default, and onboarding keeps a host's code across restarts.
Use the plugin only in rooms of your own devices, and keep OpenClaw's approvals on for `exec` and writes.

- **Screens.** A hosting gateway opens its room with visibility `asker`: the other devices' room pages
  get "answering…" stand-ins, never OpenClaw's prompts, answers or tool calls.
- **Devices holding layers see the conversation anyway.** Each one gets the hidden states of every
  token, which carry the prompt (files OpenClaw read, tool results). A device holding layers can also
  send back any hidden states it likes. That lets it choose the model's output, **including the tool
  calls OpenClaw then runs on the gateway machine**. The call grammar only keeps the calls well formed.
- **Join mode trusts the room's host with everything.** It sends OpenClaw's whole context to whoever
  answers on `pooled-room-<CODE>`, and runs the tool calls it gets back. If the real host is down,
  anyone can register that id on the public signaling server.
- **Guests.** Any device in the room can ask its own chat questions, which queue with OpenClaw's
  requests. An API client's `reused` token count shows how much of its prompt matched the host's
  cached prefixes, so it can probe what the cache holds. Turn `allowApiClients` off to close that.

## Layout

- `index.js`: the provider, onboarding hooks and the room service
- `openclaw.plugin.json`: the manifest (auth choice, config schema)
- `src/setup.js`: onboarding and the catalog entry
- `src/pool.js`: the room per process, readiness, errors, the transport to the host
- `src/convert.js`: OpenClaw context -> the room's API ask
- `src/stream.js`: the StreamFn
- `test/plugin_test.mjs`: unit tests (no GPU, no OpenClaw): conversion, OpenClaw's real tool
  schemas through the host's checks, the StreamFn against a scripted room, onboarding config
- `test/offline.mjs`: recorded OpenClaw requests through the conversion and the host's prompt,
  with a real tokenizer
- `test/join_e2e.mjs`, `test/e2e_oc.mjs`, `test/onboard_pty.py`: GPU and OpenClaw end-to-end runs

## Known gaps

- The very first turn after the gateway starts prefills OpenClaw's whole prompt (14-17k tokens
  with 2026.9.7's full tool profile: 50-62 s on the MoE split over a Spark and a Mac).
  After that the host keeps it as pinned checkpoints (`packages/room-node/ckpt.js`): new sessions,
  side requests and the next day's sessions start from them, follow-ups from the last answer, and
  each step of a tool loop from where the previous call's last user turn started (turn checkpoints).
  The checkpoints live in GPU memory only, so a gateway restart starts cold again.
- macOS needs 26 or newer for the `webgpu` package's Dawn build; Windows is untested.
- Small models need the trimmed tool profile; the MoE is the model to use.
- Join mode tells OpenClaw the room has a 32K context whatever the host runs (the catalog is written at
  onboarding, before the host is reachable). A host with less context refuses longer prompts ("start a
  new session"); a 128K host is cut to 32K by OpenClaw's compaction.
- Checkpoints (up to 4 pinned + 4 answer/turn slots, each a copy of the KV rows so far) are not part of
  the memory split. At long contexts they can take GBs across the room (a 50K-token slot is ~1 GB for
  the MoE), and a phone holding an attention layer gets its share of that.
