# @pooled/openclaw

An OpenClaw provider plugin. Pick **Pooled** in OpenClaw and this machine becomes a device in a
Pooled room: it holds its share of the model's layers on the local GPU with Pooled's own engine
(WebGPU through Dawn, inside the OpenClaw gateway process, via
[@pooled/room-node](../room-node)). No browser tab, no `pooled serve`, no API key. It can start the
room or join one. Other devices join over WebRTC: another machine with this plugin, or a browser tab
or phone on pooled.run. Together they run a model none of them can run alone (the 35B MoE needs
about 22.5 GB; two 16 GB Macs can hold it).

Status: proof of concept, not published. Tested with OpenClaw 2026.9.6. It runs from a Pooled
checkout (it imports `packages/room-node`, `room/`, `harness/` and `cli/lib/` by relative path).

## Set up

```sh
cd packages/room-node && npm install                      # Dawn, node-datachannel, peerjs
openclaw plugins install --link --accept-capabilities packages/openclaw
openclaw onboard        # or: openclaw models auth login --provider pooled --set-default
```

Onboarding offers **Pooled (run a model across your devices)** with two choices:

- **Start a room on this device**: pick the model (Qwen3 1.7B, Qwen3.8 27B, Qwen3.6 35B MoE), the
  GPU memory this device lends (GB), and how many devices to wait for. It shows the room code and
  `https://pooled.run/room/<CODE>`.
- **Join a room**: enter the code from the device that started it and the GB to lend.

It writes:

- `models.providers.pooled`: one catalog model, `pooled/<model>` (or `pooled/room` when joining),
  with the room's context (1.7B 16k, 27B 32k, MoE 64k: `room/models.js` CTX) and no API key;
- `plugins.entries.pooled.config`: `{ mode, code, model, pledgeGB, minDevices }`;
- the default model, and for the 1.7B a file-tools-only profile (a small model cannot follow
  OpenClaw's whole tool catalog).

Non-interactive: `openclaw onboard --non-interactive --auth-choice pooled` with `POOLED_MODE`
(`host` / `join`), `POOLED_CODE`, `POOLED_MODEL`, `POOLED_PLEDGE_GB`.

### Config (`plugins.entries.pooled.config`, each overridable by a `POOLED_*` variable)

| Key | Env | Meaning |
|---|---|---|
| `mode` | `POOLED_MODE` | `host` or `join` |
| `code` | `POOLED_CODE` | the room code (4 to 6 characters) |
| `model` | `POOLED_MODEL` | the model a host starts (`room/models.js` key) |
| `pledgeGB` | `POOLED_PLEDGE_GB` | GPU memory this device lends |
| `minDevices` | `POOLED_MIN_DEVICES` | devices to wait for before dealing the layers (default 1) |
| `waitSeconds` | `POOLED_WAIT_S` | how long a request waits for devices or the host's model (default 120) |
| `ctx` | `POOLED_CTX` | context to ask for (default the model's largest) |
| `signal` | `POOLED_SIGNAL` | a PeerServer `host:port` (default the PeerJS cloud server pooled.run uses) |
| `modelDir` | `POOLED_MODELS` | local model files (`packages/room-node/source.js` LOCAL layout) instead of downloads |
| `name` | `POOLED_NAME` | this device's name in the room |
| `page` | `POOLED_PAGE` | a room page other than pooled.run (local tests) |

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

- The first turn prefills OpenClaw's whole prompt (8-12k tokens): about 70 s for the 1.7B split
  over two devices. The node has no pinned system-prompt cache yet (the browser room's pinned
  checkpoints, #251 / #260); follow-ups in the same session reuse the cached prefix.
- macOS needs 26 or newer for the `webgpu` package's Dawn build; Windows is untested.
- Small models need the trimmed tool profile; the MoE is the model to use.
