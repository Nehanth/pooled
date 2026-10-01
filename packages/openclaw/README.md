# Pooled for OpenClaw

**Peer-to-peer inference engine for your claw.**

[Website](https://pooled.run) · [Docs](https://pooled.run/docs/kits/openclaw) · [GitHub](https://github.com/Nehanth/pooled)

![OpenClaw's Control UI: the assistant reads notes.txt and answers with a to-do list, running on the Qwen3.6 35B MoE in a Pooled room](https://raw.githubusercontent.com/Nehanth/pooled/main/site/img/openclaw/control-ui.webp)

Run a big model across your own devices, right from OpenClaw. No API key, no cloud model. This
machine joins a [Pooled](https://pooled.run) room and holds part of the model on its GPU. Your other
computers, browser tabs and phones hold the rest. Together they run a model none of them could run
alone.

## Get started

1. **Install the plugin.**

   ```sh
   openclaw plugins install clawhub:@pooled/openclaw
   ```

2. **Start a room.** Run `openclaw onboard`, pick **More… → Pooled**, then **Start a room**. Choose
   the **Qwen3.6 35B MoE** (recommended). Onboarding prints the room code and an invite link.

3. **Invite a second device.** Open the invite link in Chrome or Safari, run
   `npx @pooled/cli join "<invite link>"` on another computer, or pick **Pooled → Join a room** in
   OpenClaw there. Type `/pooled` in an OpenClaw chat to see the room.

Full guide with screenshots: [pooled.run/docs/kits/openclaw](https://pooled.run/docs/kits/openclaw)

## What to expect

The first answer is slow because the room reads OpenClaw's prompt (about 18,700 tokens of
instructions and tools) once. After that the room keeps it.

| Turn | Time |
|---|---|
| First answer | about 3 min |
| Next question, same chat | 14 s |
| Same task in a new chat | 46 s |

Qwen3.6 35B MoE, two devices on one DGX Spark, OpenClaw 2026.9.6. On other GPUs the times differ.

![/pooled in the OpenClaw TUI: room VAV-A53 online on the Qwen3.6 35B MoE, two devices in a table (GPU, what each lends, the layers each holds: 0-23 on this gateway, 24-39 on the second device) and the memory row](https://raw.githubusercontent.com/Nehanth/pooled/main/site/img/openclaw/room-online.webp)

## Which model

| Model | Use it for OpenClaw? |
|---|---|
| **Qwen3.6 35B MoE** | Recommended. About 23 GB of GPU memory across the room, 128k context. |
| Qwen3.8 27B | Works. About 20 GB across the room, 64k context. |
| Qwen3 1.7B | Not recommended. Small (about 5.5 GB), but slow on OpenClaw's long prompt and loops on its tools. |

Onboarding preselects the MoE when this device can hold it, and the 27B when only that fits. A room
can hold more than one device holds alone: it waits for more devices.

## Requirements

- OpenClaw 2026.9.6 or newer, Node 22 or newer.
- A GPU: Linux with a Vulkan driver and glibc 2.38 or newer (like Ubuntu 24.04), or macOS 26 or
  newer. Windows is untested.
- Tested on OpenClaw 2026.9.6 and 2026.9.7, on Linux (DGX Spark, GB10) and macOS 27 (M5 Max).
- Install size: the plugin bundles Pooled's engine and room code, and installs `node-datachannel`
  (WebRTC), `peerjs` (signaling) and, as an optional dependency, `webgpu` (Dawn, about 95 MB). If
  `webgpu` didn't install, the gateway says how to add it when it needs the GPU.
- Also on npm: `openclaw plugins install @pooled/openclaw`.

## What it does on your machine

OpenClaw shows the plugin's capabilities and asks you to accept them at install.

- **Runs native GPU code in the gateway.** It loads Dawn (npm `webgpu`, a native `dawn.node`) and the
  WebRTC library `node-datachannel` into the OpenClaw gateway process, and runs the model's layers
  there.
- **Opens network connections.** It opens the room when the gateway starts. Devices find each other
  through the public PeerJS server (`0.peerjs.com`; `signal` sets your own PeerServer) and Google's
  STUN servers (`stun.l.google.com`), then talk directly over WebRTC.
- **Downloads models from Hugging Face.** A hosting device downloads the model it runs (1.8 GB for
  the 1.7B, 16 GB for the 27B, 21 GB for the MoE) into `~/.pooled/models`, the folder `pooled pull`
  uses. With **Don't download** it reads its layers from Hugging Face with range requests at each
  start instead. Downloads resume, are checked against their SHA-256, and only one process writes a
  model at a time. `modelDir` points at local copies.
- **Runs a background service.** The gateway's `pooled-room` service keeps the room open while the
  gateway runs. On a host, it warms up OpenClaw's system prompt and tools when the room comes online.
- **Writes files** in OpenClaw's state folder (`~/.openclaw/pooled/`, mode 0600), never in
  `openclaw.json`:
  - `room.json`: the hosted room's invite key and the passes it gave out; a joined room's key and pass.
  - `status.json`: the room as the gateway sees it (devices, split, who is waiting, recent events).
  - `prewarm.json`: the last system prompt and tool list, for the warm-up.
- **Reads environment variables**: the `POOLED_*` variables in the config table below, plus
  `POOLED_DEBUG` for logging.
- **Runs one helper program.** During onboarding it runs `nvidia-smi` (without a shell, when it is
  installed) to read the GPU's memory for the default pledge.
- **Changes two global OpenClaw settings only if you ask.** For a small model, onboarding offers (off
  unless you pick it) to turn off OpenClaw's tool search (`tools.toolSearch = false`) and its memory
  flush (`agents.defaults.compaction.memoryFlush.enabled = false`). These apply to every model and
  agent, not only Pooled, and stay when you switch models. Undo them with
  `openclaw config unset tools.toolSearch` and
  `openclaw config unset agents.defaults.compaction.memoryFlush.enabled`. With the 1.7B, the agent
  gets OpenClaw's file tools only (`read`, `write`, `edit`, `ls`).
- **Nothing else.** No API key or account, no shell, no OpenClaw hooks, no tools of its own, no
  telemetry.

## Who can see what

Use rooms of your own devices, share the invite link only with people you trust, and keep OpenClaw's
approvals on for `exec` and writes.

- **Getting in.** A hosted room asks before new devices join. The invite link's key gets a device in;
  a code alone waits for `/pooled allow`; a device you let in keeps a pass. With **Anyone with the
  code**, anyone who has or guesses the code gets in.
- **Screens.** A hosting gateway opens its room with visibility `asker`: other devices' room pages
  show "answering…", never OpenClaw's prompts, answers or tool calls.
- **Devices holding layers see the conversation anyway.** Each one gets the hidden states of every
  token, which carry the prompt (files OpenClaw read, tool results).
- **Devices holding layers can steer the answer.** A device can send back any hidden states it likes,
  which lets it choose the model's output, **including the tool calls OpenClaw then runs on the
  gateway machine**. The call grammar only keeps the calls well formed.
- **Join mode trusts the room's host with everything.** It sends OpenClaw's whole context to the
  room's host and runs the tool calls it gets back. Join mode is flagged as a dangerous setting
  (`plugins.entries.pooled.config.mode=join`): the gateway logs a security warning at startup, and
  `openclaw security audit` lists it.
- **Guests** in the room can ask their own questions, which queue with OpenClaw's.
  `allowApiClients: false` keeps API clients out.

## More

<details>
<summary><b>Start a room: every question</b></summary>

`openclaw onboard` → **Pooled** → **Start a room**:

1. How much of this GPU's memory to lend (the default is what `pooled host` picks).
2. The model: each row shows the memory it needs across the room and whether it's downloaded (or its
   download size).
3. If it isn't downloaded: **Download now** (with a progress bar, speed and time left), **When the
   gateway starts**, or **Don't download** (stream this machine's layers from Hugging Face at each
   start).
4. How many devices to wait for, and who can join: **Invite link only** (a device with only the code
   waits for your Allow) or **Anyone with the code**.

It prints the room code (`4TK-G9P`) and the invite link (`https://pooled.run/r/4TKG9P#k=…`). The
room opens when the gateway starts and keeps its code and link across restarts. If onboarding
couldn't start the gateway, run `openclaw gateway run`. Onboarding's own AI check (OpenClaw's live
test completion) is answered at once with the room's state, so it passes even while the room waits
for its other devices; no `--skip-health` is needed.

While a model downloads, a question gets an answer like `Qwen3 1.7B is downloading for room 4TK-G9P:
42% · 734 MB of 1.7 GB · 48 MB/s · 18s left`. Devices can join meanwhile.

</details>

<details>
<summary><b>Join a room from OpenClaw</b></summary>

`openclaw onboard` → **Pooled** → **Join a room**, then paste the invite link (or type the code) and
say how much memory to lend. Onboarding connects right away (no GPU needed for that):

- with the invite link, it's let in at once;
- with the code alone, it shows `Waiting for <host> to let this device in · 0:08` until the host allows it.

Either way it keeps the pass the host gives it, so the gateway (and every restart) gets straight back
in. It also learns the host's model and context.

OpenClaw's own instructions and tools are about 12k tokens, so the room needs at least a 16k context.
`pooled host qwen3-1.7b` opens at 16k from `@pooled/cli` 0.3.2; with 0.3.0 or 0.3.1 its host should run
`pooled host qwen3-1.7b --ctx 16384` (the 1.7B's longest). Onboarding says so when it sees a shorter one, and so does `/pooled`.
A room short of memory for 16k (under about 5.5 GB across the room) opens the 1.7B at 8k, the plugin's own room too: then
`/pooled` warns that OpenClaw needs 16k, and onboarding shows `fits on this machine alone at 8k (OpenClaw needs 16k)`.

</details>

<details>
<summary><b>/pooled commands</b></summary>

In an OpenClaw chat (the TUI, the Control UI, a messaging channel), for the gateway's owner. They need
OpenClaw's `operator.admin` scope, so `openclaw agent --message` and `gateway call chat.send` can't
run them.

| Command | What it does |
|---|---|
| `/pooled` | The room: invite link, each device's GPU, what it lends and the layers it holds, the memory, who is waiting, the download |
| `/pooled allow [n\|all]` | Let a waiting device in (the first one, the n-th, or all) |
| `/pooled deny [n]` | Turn one away |
| `/pooled link` | The invite link |
| `/pooled pledge <GB>` | Lend another amount of this machine's GPU memory |

Room problems show in the chat as a short note that starts with **Pooled**, the room code and what
happened: waiting for devices, not enough memory, downloading the model, a device left, and so on.
They are not part of the conversation the model sees. The
bundled `pooled-room` skill tells the agent what they mean.

</details>

<details>
<summary><b>Config</b></summary>

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
| `ctx` | `POOLED_CTX` | Context to ask for (default the model's largest: 1.7B 16k, or 8k when the room is short of memory for 16k; 27B 64k, MoE 128k) |
| `name` | `POOLED_NAME` | This device's name in the room (default `<hostname> (OpenClaw)`) |
| `signal` | `POOLED_SIGNAL` | A PeerServer `host:port` (default the public PeerJS server pooled.run uses) |

Non-interactive: `openclaw onboard --non-interactive --auth-choice pooled` with `POOLED_MODE=host`
and `POOLED_MODEL=qwen3.6-35b-moe` (without `POOLED_MODEL` it starts the 1.7B), `POOLED_PLEDGE_GB`, …,
or `POOLED_LINK="<invite link>"` to join.

</details>

<details>
<summary><b>How it works, and known gaps</b></summary>

- The gateway's `pooled-room` service opens the room at startup. A host deals the layers once the room
  has `minDevices` devices and enough memory, then replays the last system prompt and tools once (a
  one-token answer), so the first question after a restart starts from their checkpoint.
- OpenClaw's model calls go through a custom StreamFn, with no HTTP hop. The room's host renders the
  chat template and constrains tool calls with a grammar; the answer is checked as `pooled serve`
  checks it and streamed as OpenClaw's events.
- A joined gateway asks the room's host over WebRTC (the `pooled serve` bridge), showing the host
  this device's pass, so the host sees one device, not two join requests.

Known gaps:

- Reading the prompt is the slow part: OpenClaw's prompt is 14-19k tokens. The warm-up moves the
  cold read to when the room comes online; checkpoints live in GPU memory only.
- A room hosted in a browser tab has no turn checkpoints yet, so OpenClaw re-reads more per tool call
  there than with a Node host.
- Checkpoints are not part of the memory split: at 50k tokens one is about 1 GB across the room (MoE).
- Windows is untested.

Design notes, results and limits:
[docs/openclaw.md](https://github.com/Nehanth/pooled/blob/main/docs/openclaw.md).

</details>

<details>
<summary><b>Develop</b></summary>

From a checkout of [github.com/Nehanth/pooled](https://github.com/Nehanth/pooled):

```sh
cd packages/openclaw && npm install && npm run build      # dist/index.js: the bundle that ships
openclaw plugins install --link --force --accept-capabilities "$PWD"    # rebuild after edits
npm test                                  # unit tests: no GPU, no network, no OpenClaw
sh test/pack_install.sh                   # npm pack, install the tarball into a clean OpenClaw, onboard
```

`npm run build` bundles `index.js` and what it imports from the repo's `room/`, `engine/`, `harness/`,
`cli/lib/` and `packages/room-node/` into `dist/index.js` (esbuild); `openclaw`, `webgpu`,
`node-datachannel` and `peerjs` stay external.

</details>

## Links

- Guide: [pooled.run/docs/kits/openclaw](https://pooled.run/docs/kits/openclaw)
- Reference: [pooled.run/docs/openclaw](https://pooled.run/docs/openclaw)
- Source: [github.com/Nehanth/pooled](https://github.com/Nehanth/pooled/tree/main/packages/openclaw)
- Issues: [github.com/Nehanth/pooled/issues](https://github.com/Nehanth/pooled/issues)

MIT license.
