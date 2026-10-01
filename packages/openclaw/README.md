# Pooled for OpenClaw

Run a model across your own devices: this machine joins a [Pooled](https://pooled.run) room and holds
part of the model on its GPU.

Pick **Pooled** in OpenClaw and this machine becomes a device in a Pooled room. It holds its share of
the model's layers on the local GPU with Pooled's own engine (WebGPU through Dawn, inside the OpenClaw
gateway). Other devices join the same room over WebRTC and hold the rest: another OpenClaw with this
plugin, `pooled join` in a terminal, or a browser tab or phone on pooled.run. Together they run a model
none of them could run alone. No API key, no cloud model.

Tested with OpenClaw 2026.9.6 and 2026.9.7 on Linux (DGX Spark, GB10) and macOS 27 (M5 Max).
Docs: [pooled.run/docs/openclaw](https://pooled.run/docs/openclaw/) ·
Source: [github.com/Nehanth/pooled](https://github.com/Nehanth/pooled/tree/main/packages/openclaw)

## Install

```sh
openclaw plugins install clawhub:@pooled/openclaw
openclaw onboard          # Model/auth provider -> More... -> Pooled
```

Or from npm: `openclaw plugins install @pooled/openclaw`.

OpenClaw shows the plugin's capabilities and asks you to accept them. The package bundles Pooled's
engine and room code, and installs `node-datachannel` (WebRTC), `peerjs` (signaling) and, as an
optional dependency, `webgpu` (Dawn, about 95 MB).

Needs: OpenClaw 2026.9.6 or newer, Node 22 or newer, and a GPU Dawn runs on: Linux with glibc 2.38 or
newer and a Vulkan driver, or macOS 26 or newer. Windows is untested. If `webgpu` didn't install, the
gateway says how to add it when it needs the GPU.

## Which model

**Use the Qwen3.6 35B MoE.** It needs about 23 GB of GPU memory across the room (one big machine, or
two or more pooled). Onboarding preselects it when this device's memory holds it, the Qwen3.8 27B
(about 20 GB) when only that fits, and the MoE again when neither fits on this device alone (the room
then waits for more devices).

Small models struggle with OpenClaw's long prompts and tools: expect slow turns and tool loops. Use the
35B MoE if your devices can hold it. Measured on a DGX Spark, the OpenClaw gateway hosting and a second
device (`pooled join`) holding the other half of the layers, OpenClaw 2026.9.6:

| Turn | Qwen3.6 35B MoE (128k context) | Qwen3 1.7B (16k context) |
|---|---|---|
| First question after the room came online | 143 s (OpenClaw's whole prompt, cold) | 163 s |
| Second question in the same chat | 4.6 s (17k tokens reused) | 119 s (nothing reused) |
| "Read notes.txt and tell me the secret word" | 10.4 s (2 model calls) | 25.8 s (5 model calls) |
| The same in a new session | 21.2 s | 580 s: 85 tool-search calls, then OpenClaw stopped the loop with no answer |
| OpenClaw's background memory save | not triggered | on almost every turn; holds the room for about 120 s |

The 1.7B gets OpenClaw's file tools only (`read`, `write`, `edit`, `ls`). Onboarding also offers, for a
small model only and off unless you pick it, to turn off OpenClaw's tool search
(`tools.toolSearch = false`) and its memory flush
(`agents.defaults.compaction.memoryFlush.enabled = false`). **Both are global OpenClaw settings**: they
apply to every model and agent, not only Pooled, and stay when you switch models. Undo them with
`openclaw config unset tools.toolSearch` and
`openclaw config unset agents.defaults.compaction.memoryFlush.enabled`.

## Start a room

`openclaw onboard` → **Pooled** → **Start a room on this device**:

1. How much of this GPU's memory to lend (the default is what `pooled host` picks).
2. The model: each one shows whether it's downloaded, its download size, the memory it needs across
   the room and its context.
3. If it isn't downloaded: **Download now** (with progress), **Download when the gateway starts**, or
   **Don't download** (stream this machine's layers from Hugging Face at each start).
4. How many devices to wait for, and who can join: **Devices with the invite link** (a device with
   only the code waits for your Allow) or **Anyone with the room code**.

It prints the room code (`4TK-G9P`) and the invite link (`https://pooled.run/r/4TKG9P#k=…`). The
room opens when the gateway starts and keeps its code and link across restarts. On your other devices:

- open the invite link in Chrome or Safari, or
- `npx -p @pooled/cli -p webgpu@0.6.1 pooled join "<invite link>"`, or
- pick Pooled → **Join a room** in OpenClaw there and paste the link.

## Join a room

`openclaw onboard` → **Pooled** → **Join a room**, then paste the invite link (or type the code) and
say how much memory to lend. Onboarding connects to the room right away (no GPU needed for that):

- with the invite link, it's let in at once;
- with the code alone, it shows `Waiting for <host> to let this device in…` until the host allows it.

Either way it keeps the pass the host gives it, so the gateway (and every restart) gets straight back
in. It also learns the host's model and context, so OpenClaw knows the room's real context window.

OpenClaw's own instructions and tools are about 12k tokens, so the room needs at least a 16k context.
`pooled host qwen3-1.7b` opens at 8k: its host should run `pooled host qwen3-1.7b --ctx 16384` (the
1.7B's longest). Onboarding says so when it sees a shorter one, and so does `/pooled`.

Join mode is flagged as a dangerous setting (`plugins.entries.pooled.config.mode=join`): the gateway
logs a security warning at startup, and `openclaw security audit` lists it. See
[Who can see and steer what](#who-can-see-and-steer-what).

## /pooled

In an OpenClaw chat (the TUI, the Control UI, a messaging channel), for the gateway's owner. It needs
OpenClaw's `operator.admin` scope, so `openclaw agent --message` and `gateway call chat.send` can't run it:

| Command | What it does |
|---|---|
| `/pooled` | The room: invite link, devices, pledges, whether they hold the model, who is waiting to join, the model download |
| `/pooled allow [n\|all]` | Let a waiting device in (the first one, the n-th, or all) |
| `/pooled deny [n]` | Turn one away |
| `/pooled link` | The invite link |
| `/pooled pledge <GB>` | Lend another amount of this machine's GPU memory |

The gateway log says `phone wants to join (iPhone, 0.5 GB): /pooled allow lets it in, /pooled deny
turns it away` when a device knocks with the code alone.

Room problems show in the chat as `⚠️ Pooled: …`: waiting for the host to let this device in, waiting
for devices, not enough memory, downloading, a device left while it held layers, the host doesn't allow
API clients, the host runs an older Pooled. They are not part of the conversation the model sees. When
the conversation outgrows the room's context, OpenClaw compacts it and asks again. The bundled
`pooled-room` skill tells the agent what these mean and where the room's status is.

## What it does on your machine

- **Native GPU code in the gateway.** It loads Dawn (npm `webgpu`, a native `dawn.node`) and the WebRTC
  library `node-datachannel` into the OpenClaw gateway process, and runs the model's layers there.
- **Network.** It opens the room when the gateway starts. Devices find each other through the public
  PeerJS server (`0.peerjs.com`; `signal` sets your own PeerServer) and Google's STUN servers
  (`stun.l.google.com`), then talk directly over WebRTC.
- **Model downloads from Hugging Face.** A hosting device downloads the model it runs (1.8 GB for the
  1.7B, 16 GB for the 27B, 21 GB for the MoE) into `~/.pooled/models`, the folder `pooled pull` uses,
  or with **Don't download** reads its layers from Hugging Face with range requests at each start.
  Downloads resume and are checked against their SHA-256. `modelDir` points at local copies.
- **A background service.** The gateway's `pooled-room` service keeps the room open while the gateway
  runs and, on a host, warms up OpenClaw's system prompt and tools when the room comes online.
- **Files** in OpenClaw's state folder (`~/.openclaw/pooled/`, mode 0600), never in `openclaw.json`:
  `room.json` (the hosted room's invite key and the passes it gave out; a joined room's key and pass),
  `status.json` (the room as the gateway sees it: devices, split, who is waiting, recent events) and
  `prewarm.json` (the last system prompt and tool list, for the warm-up).
- **Environment.** It reads the `POOLED_*` variables in [Config](#config) (plus `POOLED_DEBUG` for
  logging).
- **One helper program.** During onboarding it runs `nvidia-smi` (without a shell, when it is installed)
  to read the GPU's memory for the default pledge.
- **Nothing else.** No API key or account, no shell, no OpenClaw hooks, no tools of its own, no
  telemetry.

## Who can see and steer what

Use rooms of your own devices, share the invite link only with people you trust, and keep OpenClaw's
approvals on for `exec` and writes.

- **Getting in.** A hosted room asks before new devices join: the invite link's key gets a device in,
  a code alone waits for `/pooled allow`, and a device you let in keeps a pass. With **Anyone with the
  room code**, anyone who has or guesses the code gets in.
- **Screens.** A hosting gateway opens its room with visibility `asker`: other devices' room pages show
  "answering…", never OpenClaw's prompts, answers or tool calls.
- **Devices holding layers see the conversation anyway.** Each one gets the hidden states of every
  token, which carry the prompt (files OpenClaw read, tool results). A device holding layers can also
  send back any hidden states it likes, which lets it choose the model's output, **including the tool
  calls OpenClaw then runs on the gateway machine**. The call grammar only keeps the calls well formed.
- **Join mode trusts the room's host with everything**: it sends OpenClaw's whole context to the room's
  host and runs the tool calls it gets back.
- **Guests** in the room can ask their own questions, which queue with OpenClaw's. `allowApiClients:
  false` keeps API clients out.

## Models

Models live in `~/.pooled/models`, the same folder as `pooled pull`, so a model is downloaded once per
machine. Downloads resume, are checked against their SHA-256, and only one process writes a model at
a time. A hosting gateway downloads a missing model in the background while devices join; a question
meanwhile gets `Pooled is downloading Qwen3 1.7B for room 4TK-G9P: 42% (734 MB of 1.7 GB), about 18s
left. Ask again when it's done; devices can join the room meanwhile`.

## Config

Onboarding writes `plugins.entries.pooled.config`. Each key can be overridden by its variable.

| Key | Variable | Meaning |
|---|---|---|
| `mode` | `POOLED_MODE` | `host` or `join` |
| `code` | `POOLED_CODE` | The room code (6 characters; 4 for rooms opened before six) |
| — | `POOLED_LINK` | A room's invite link (join; non-interactive setup) |
| `model` | `POOLED_MODEL` | The model a host starts: `qwen3.6-35b-moe` (recommended), `qwen3.8-27b`, `qwen3-1.7b` |
| `pledgeGB` | `POOLED_PLEDGE_GB` | GPU memory this device lends |
| `minDevices` | `POOLED_MIN_DEVICES` | Devices to wait for before dealing the layers (default 1) |
| `waitSeconds` | `POOLED_WAIT_S` | How long a question waits for devices or the host's model (default 120) |
| `ask` | `POOLED_ASK` | Host: a device with only the code waits for `/pooled allow` (default on) |
| `allowApiClients` | `POOLED_ALLOW_API` | Host: answer other devices' API asks (another OpenClaw in join mode, `pooled serve`; default on) |
| `pull` | `POOLED_PULL` | Host: download a missing model (default on); off streams it at each start |
| `prewarm` | `POOLED_PREWARM` | Host: warm up OpenClaw's system prompt and tools when the room comes online (default on) |
| `modelDir` | `POOLED_MODELS` | Models folder (default `~/.pooled/models`) |
| `ctx` | `POOLED_CTX` | Context to ask for (default the model's largest: 1.7B 16k, 27B 64k, MoE 128k) |
| `name` | `POOLED_NAME` | This device's name in the room (default `<hostname> (OpenClaw)`) |
| `signal` | `POOLED_SIGNAL` | A PeerServer `host:port` (default the public PeerJS server pooled.run uses) |

Non-interactive: `openclaw onboard --non-interactive --auth-choice pooled` with `POOLED_MODE=host`
and `POOLED_MODEL=qwen3.6-35b-moe` (without `POOLED_MODEL` it starts the 1.7B), `POOLED_PLEDGE_GB`, …,
or `POOLED_LINK="<invite link>"` to join.

## How it works

- The gateway's `pooled-room` service opens the room at startup. A host deals the layers once the room
  has `minDevices` devices and enough memory, then replays the last system prompt and tools once (a
  one-token answer), so the first question after a restart starts from their checkpoint.
- OpenClaw's model calls go through a custom StreamFn, with no HTTP hop. The request is the one
  `pooled serve` builds; the room's host renders the chat template and constrains tool calls with a
  grammar; the answer is checked as `pooled serve` checks it and streamed as OpenClaw's events.
- A joined gateway asks the room's host over WebRTC (the `pooled serve` bridge). The bridge shows the
  host this device's pass, so the host sees one device, not two join requests.

Design notes, results and limits:
[docs/openclaw.md](https://github.com/Nehanth/pooled/blob/main/docs/openclaw.md).

## Known gaps

- Prefill is the slow part: OpenClaw's prompt is 14-18k tokens. The warm-up moves the cold read to
  when the room comes online; checkpoints still live in GPU memory only.
- A room hosted in a browser tab has no turn checkpoints yet, so OpenClaw re-reads more per tool call
  there than with a Node host.
- Checkpoints are not part of the memory split: at 50k tokens one is about 1 GB across the room (MoE).
- Windows is untested.

## Develop

From a checkout of [github.com/Nehanth/pooled](https://github.com/Nehanth/pooled):

```sh
cd packages/openclaw && npm install && npm run build      # dist/index.js: the bundle that ships
openclaw plugins install --link --force --accept-capabilities "$PWD"    # runs dist/index.js: rebuild after edits
npm test                                  # unit tests: no GPU, no network, no OpenClaw
sh test/pack_install.sh                   # npm pack, install the tarball into a clean OpenClaw, onboard
```

`npm run build` bundles `index.js` and what it imports from the repo's `room/`, `engine/`, `harness/`,
`cli/lib/` and `packages/room-node/` into `dist/index.js` (esbuild); `openclaw`, `webgpu`,
`node-datachannel` and `peerjs` stay external. `npm pack` builds it. Layout, tests and the GPU runs:
[packages/openclaw on GitHub](https://github.com/Nehanth/pooled/tree/main/packages/openclaw).

MIT license.
