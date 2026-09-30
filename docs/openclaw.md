# OpenClaw on Pooled

Status: proof of concept on the `feat/openclaw-pooled` branch. Nothing is published to npm, and
nothing is listed in OpenClaw's plugin catalog yet.

## What it is

[OpenClaw](https://github.com/openclaw/openclaw) is a self-hosted agent: a gateway process on your
machine that runs the agent loop and its tools (`exec`, `read`, `edit`, ...) and calls a model
provider for every step. The `@pooled/openclaw` plugin ([packages/openclaw](../packages/openclaw))
adds **Pooled** as that provider.

When you pick Pooled, the OpenClaw gateway itself becomes a device in a Pooled room. It holds its
share of the model's layers on the local GPU and runs them with Pooled's own WebGPU engine, through
Dawn (npm `webgpu`) inside the gateway's Node process ([packages/room-node](../packages/room-node)).
Other devices join the same room and hold the rest:

- another machine with the plugin, or with `packages/room-node/join.mjs` (no OpenClaw needed);
- a browser tab or a phone on `https://pooled.run/room/<CODE>`.

So a model none of the devices can run alone serves the agent, with no API key and no cloud model.
OpenClaw's requests and tool calls take the same path as `pooled serve`: the room's host renders the
chat template, constrains tool calls with a grammar and parses them (`room/api.js`). The host also
keeps OpenClaw's system prompt and each turn as checkpoints, so only the new tokens of each step are
read.

```
 OpenClaw gateway (Spark)                Mac Studio               iPhone (optional)
 ┌───────────────────────────┐          ┌──────────────┐          ┌──────────────┐
 │ agent loop + tools        │          │ join.mjs     │          │ Safari,      │
 │ @pooled/openclaw          │  WebRTC  │ Node + Dawn  │  WebRTC  │ pooled.run   │
 │ room node: embed, layers  │ <------> │ layers 19-39 │ <------> │ 1-2 layers   │
 │ 0-18, head, checkpoints   │          │ (Metal)      │          │              │
 └───────────────────────────┘          └──────────────┘          └──────────────┘
```

## Setup

You need a Pooled checkout on every machine that runs Node (the packages import `engine/`, `room/`,
`harness/` and `cli/lib/` by relative path), Node 22 or newer (Node 24 was tested; OpenClaw ships on
24), and OpenClaw 2026.9.6 or newer (2026.9.7 was tested).

### Linux (tested: DGX Spark, GB10, Vulkan)

```sh
git clone https://github.com/Nehanth/pooled.git && cd pooled
git checkout feat/openclaw-pooled
(cd packages/room-node && npm install --omit=dev)     # Dawn, node-datachannel, peerjs (~156 MB)
npm install -g --allow-scripts=@google/genai,esbuild,koffi,protobufjs,openclaw openclaw@2026.9.7
openclaw plugins install --link --accept-capabilities --force "$PWD/packages/openclaw"
openclaw onboard          # Model/auth provider -> More... -> Pooled
```

The Vulkan driver must be installed (Dawn uses it). `--force` and an absolute path are needed by
2026.9.7 for a local plugin outside ClawHub review. npm 11 only runs the install scripts you allow,
hence `--allow-scripts` for OpenClaw's own dependencies.

### macOS (tested: Mac Studio M5 Max, 36 GB, macOS 27)

Needs macOS 26 or newer (the `webgpu` package's Dawn build). No sudo: Node goes under your home.

```sh
cd ~/.local && V=v24.21.0
curl -fsSLO https://nodejs.org/dist/$V/node-$V-darwin-arm64.tar.xz
curl -fsSL https://nodejs.org/dist/$V/SHASUMS256.txt | grep " node-$V-darwin-arm64.tar.xz$" | shasum -a 256 -c -
tar xJf node-$V-darwin-arm64.tar.xz && ln -sfn node-$V-darwin-arm64 node24 && rm node-$V-darwin-arm64.tar.xz
export PATH=~/.local/node24/bin:$PATH
```

Then the same steps as on Linux. Dawn runs on Metal. `dawn.node` is ad-hoc signed; npm does not set
the quarantine flag, so Gatekeeper does not block it. If you copied the package from a zip or a
browser download, run `xattr -d com.apple.quarantine` on `dawn.node` (the package's install script
does this, but npm 11 skips it unless allowed).

### Start or join a room

`openclaw onboard` offers **Pooled (run a model across your devices)** with two choices:

- **Start a room on this device**: pick the model (Qwen3.6 35B MoE is the one to use for agent work),
  the GPU memory to lend and how many devices to wait for. It prints the code and
  `https://pooled.run/room/<CODE>`.
- **Join a room**: the code from the device that started it and the GB to lend.

To lend a second machine's GPU without OpenClaw:

```sh
node packages/room-node/join.mjs SPKA --gb 12 --name mac-studio
```

Or open `https://pooled.run/room/SPKA` in Chrome or Safari. Use 4-character codes when a browser or
phone joins: the room page's code box keeps 4 characters.

Non-interactive setup, config keys (`allowApiClients`, `signal`, `modelDir`, ...) and how requests
flow are in [packages/openclaw/README.md](../packages/openclaw/README.md).

## Results

Qwen3.6 35B MoE (Q4_0, ~20 GB), 128K context, OpenClaw 2026.9.7 with its full tool profile. Five
tasks, each in a new session, each checked by a script the agent never sees: answer a question by
reading files, a code edit, run the tests and fix the code, add a feature with tests, and summarize
a 140 KB (~33K-token) file. No tool call in any run was malformed or came back as an error.

| Room | Layer split | Passed | First call (cold) | Decode |
|---|---|---|---|---|
| A: OpenClaw on the Spark, Mac joins | Spark 0-18 + head, Mac 19-39 | 4/5 | 61.8 s (17.4K tokens) | 17-29 tok/s |
| A2: A with turn checkpoints | same | 5/5 | 61.7 s | 16-30 tok/s |
| B: OpenClaw on the Mac, Spark joins | Mac 0-18 + head, Spark 19-39 | 4/5 | 49.2 s (14.5K tokens) | 17-32 tok/s |
| C: A plus an iPhone 14 Pro Max in Safari | Spark 0-17, iPhone 18-19, Mac 20-39 | 4/5 | 89.2 s | 9-16 tok/s |

- After the first call, each task's first model call reads 4-8 s of prompt: the system prompt comes
  from a checkpoint.
- A2 task times: 55-139 s for the four short tasks, 306 s for the long file.
- The failures are all the same model mistake on the edit task (`$-12.34` instead of `-$12.34`,
  never checked). In A2 the model tested its change and passed.
- The phone held its 2 layers through a 23-minute run and roughly halved the speed. It should only
  get layers when the computers cannot hold the model.
- Turn checkpoints (found in this POC): OpenClaw drops each call's trailing context block from the
  next call, so the host now also saves a checkpoint where the last user turn starts. Long file
  813 -> 306 s, feature 298 -> 139 s, tests 119 -> 57 s.

Per-task tables are in [openclaw-poc.md](openclaw-poc.md).

A demo recording (77.5 s, real time) shows scenario A: OpenClaw on the Spark fixes a failing test
suite in 52 s over 6 model calls, with the Mac holding layers 19-39. It is not in the repo.

## Limits

- **Cold first call.** The first call after a gateway starts reads OpenClaw's whole prompt (14-18K
  tokens): 49-64 s on two computers, 89 s with a phone. Checkpoints live in GPU memory only.
- **Decode over the network.** Two computers over Wi-Fi decode at 25-30 tok/s; the Mac alone does ~92.
  Splitting pays one network hop per token, so it is for models that do not fit on one device.
- **Trust.** A room has no password. Every device holding layers sees the hidden state of every token
  (which carries the prompt and tool results) and can steer the answer, including the tool calls
  OpenClaw runs. Use rooms of your own devices and keep OpenClaw's approvals on for `exec` and
  writes. Details: "Who can see and steer what" in the plugin README.
- **Join mode** tells OpenClaw the room has 32K context whatever the host runs.
- **Checkpoint memory** is not part of the layer split: at 50K tokens one checkpoint is ~1 GB across
  the room for the MoE, and the host keeps up to 8 (4 pinned, 4 answer/turn).
- **Install size** is ~156 MB: the `webgpu` package ships Dawn for five OS/CPU pairs.
- **Windows** is untested. Small models (the 1.7B) need a file-tools-only profile.
- **Not packaged.** The plugin runs from a Pooled checkout and is not on npm or ClawHub.
