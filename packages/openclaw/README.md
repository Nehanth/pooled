# @pooled/openclaw

An OpenClaw provider plugin. Pick **Pooled** in OpenClaw and this machine becomes a device in a
Pooled room: it runs its share of the model's layers on the local GPU with Pooled's own engine
(WebGPU through Dawn, inside the OpenClaw gateway). Other devices join the same room and hold the
rest: another OpenClaw with this plugin, `pooled join`, or a browser tab or phone on pooled.run.
Together they run a model none of them can run alone. No API key, no cloud model.

Overview, results and limits: [docs/openclaw.md](../../docs/openclaw.md).

Status: 0.2.0, preview. Tested with OpenClaw 2026.9.6 and 2026.9.7 on Linux (GB10) and macOS 27
(M5 Max). Not on npm until it is published; [install from a checkout](#develop) until then.

## Install

```sh
openclaw plugins install @pooled/openclaw
openclaw onboard          # Model/auth provider -> More... -> Pooled
```

OpenClaw asks you to confirm a package from npm (it isn't on ClawHub) and to accept its
capabilities; `--force --accept-capabilities` skips both. The package bundles Pooled's engine and
room code; it installs `node-datachannel` (WebRTC) and `peerjs`, and `webgpu` (Dawn, ~95 MB) as an
optional dependency. Dawn needs macOS 26 or newer, or Linux with glibc 2.38 or newer (Vulkan). If
`webgpu` didn't install, the gateway says how to add it when it needs the GPU.

## Start a room

`openclaw onboard` → **Pooled** → **Start a room on this device**:

1. How much of this GPU's memory to lend (the default is what `pooled host` picks).
2. The model: each one shows whether it's downloaded, its download size, the memory it needs across
   the room and its context. The one this machine holds alone is recommended.
3. If it isn't downloaded: **Download now** (with progress), **Download when the gateway starts**, or
   **Don't download** (stream this machine's layers from Hugging Face at each start).
4. How many devices to wait for, and who can join: **Devices with the invite link** (a device with
   only the code waits for your Allow) or **Anyone with the room code**.

It prints the room code (`4TK-G9P`) and the invite link (`https://pooled.run/r/4TKG9P#k=…`). The
room opens when the gateway starts. On your other devices:

- open the invite link in Chrome or Safari, or
- `npx -p @pooled/cli -p webgpu@0.6.1 pooled join "<invite link>"`, or
- pick Pooled → **Join a room** in OpenClaw there and paste the link.

The code and the link stay the same across gateway restarts.

## Join a room

`openclaw onboard` → **Pooled** → **Join a room**, then paste the invite link (or type the code) and
say how much memory to lend. Onboarding connects to the room right away (no GPU needed for that):

- with the invite link, it's let in at once;
- with the code alone, it shows `Waiting for <host> to let this device in…` until the host allows it.

Either way it keeps the pass the host gives it, so the gateway (and every restart) gets straight back
in. It also learns the host's model and context, so OpenClaw knows the room's real context window.

## /pooled

In any OpenClaw chat (the TUI, the Control UI, a messaging channel), for the gateway's owner:

| Command | What it does |
|---|---|
| `/pooled` | The room: invite link, devices, pledges, whether they hold the model, who is waiting to join, the model download |
| `/pooled allow [n\|all]` | Let a waiting device in (the first one, the n-th, or all) |
| `/pooled deny [n]` | Turn one away |
| `/pooled link` | The invite link |
| `/pooled pledge <GB>` | Lend another amount of this machine's GPU memory |

The gateway log says `phone wants to join (iPhone, 0.5 GB): /pooled allow lets it in, /pooled deny
turns it away` when a device knocks with the code alone.

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
| `model` | `POOLED_MODEL` | The model a host starts |
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
| `signal` | `POOLED_SIGNAL` | A PeerServer `host:port` (default the PeerJS cloud server pooled.run uses) |

Non-interactive: `openclaw onboard --non-interactive --auth-choice pooled` with `POOLED_MODE=host`
(and `POOLED_MODEL`, `POOLED_PLEDGE_GB`, …), or `POOLED_LINK="<invite link>"` to join.

Files the plugin keeps in OpenClaw's state folder (`~/.openclaw/pooled/`, mode 0600), never in
`openclaw.json`:

- `room.json`: the hosted room's invite key and the passes it gave out; a joined room's key and pass.
- `status.json`: the room as the gateway sees it (devices, split, who is waiting, recent events).
- `prewarm.json`: the last system prompt and tool list, for the warm-up.

## How it works

- The gateway's `pooled-room` service opens the room at startup. A host deals the layers once the room
  has `minDevices` devices and enough memory, then replays the last system prompt and tools once (a
  one-token answer), so the first question after a restart starts from their checkpoint.
- OpenClaw's model calls go through a custom StreamFn, with no HTTP hop. The request is the one
  `pooled serve` builds (`cli/lib/common.js`); the room's host renders the chat template and
  constrains tool calls with a grammar (`room/api.js`); the answer is checked as `pooled serve` checks
  it (`cli/lib/answer.js`) and streamed as OpenClaw's events.
- A joined gateway asks the room's host over WebRTC (the `pooled serve` bridge). The bridge shows the
  host this device's pass, so the host sees one device, not two join requests.

Room problems show in the chat as `⚠️ Pooled: …`: waiting for the host to let this device in, waiting
for devices, not enough memory, downloading, a device left while it held layers, the host doesn't allow
API clients, the host runs an older Pooled. They are not part of the conversation the model sees. When
the conversation outgrows the room's context, OpenClaw compacts it and asks again.

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

## Develop

From a Pooled checkout:

```sh
cd packages/openclaw && npm install && npm run build      # dist/index.js: the bundle npm ships
openclaw plugins install --link --force --accept-capabilities "$PWD"    # runs index.js from the checkout
npm test                                  # unit tests: no GPU, no network, no OpenClaw
npm pack && openclaw plugins install ./pooled-openclaw-0.2.0.tgz --force --accept-capabilities
```

- `npm run build` bundles `index.js` and what it imports from `room/`, `engine/`, `harness/`,
  `cli/lib/` and `packages/room-node/` into `dist/index.js` (esbuild, the CLI's recipe); `openclaw`,
  `webgpu`, `node-datachannel` and `peerjs` stay external. `npm pack` builds it.
- `sh test/pack_install.sh`: packs the plugin, installs the tarball into a clean OpenClaw (its own
  state folder), then checks `plugins inspect`, a non-interactive onboarding, the model list and the
  synthetic auth. No GPU.
- `node test/gate_e2e.mjs` (GPU): two gateways from the tarball, one hosting and one joining with the
  code alone; `/pooled allow`, a question from each side, a host restart (same link, the joined gateway
  back in with its pass, the warm-up).
- `test/join_e2e.mjs`, `test/e2e_oc.mjs`, `test/offline.mjs`, `test/onboard_pty.py`: earlier GPU and
  OpenClaw runs against a browser tab.

Layout: `index.js` (provider, `/pooled`, the room service), `openclaw.plugin.json` (manifest),
`src/setup.js` (onboarding), `src/pool.js` (the room, the gate, the transport), `src/stream.js`
(StreamFn), `src/convert.js` (OpenClaw context → the room's ask), `src/commands.js` (`/pooled`),
`src/models.js` + `src/download.js` + `src/pulllock.js` (models and the shared cache), `src/state.js`
(`room.json`), `src/prewarm.js` (warm-up), `src/runtime.js` (loads the room node and Dawn).

## Known gaps

- Prefill is the slow part: OpenClaw's prompt is 14-18k tokens. The warm-up moves the cold read to
  when the room comes online; checkpoints still live in GPU memory only.
- A room hosted in a browser tab has no turn checkpoints yet, so OpenClaw re-reads more per tool call
  there than with a Node host.
- Windows is untested. Small models (the 1.7B) get a file-tools-only profile; the MoE is the model for
  agent work.
- Checkpoints are not part of the memory split: at 50k tokens one is about 1 GB across the room (MoE).
