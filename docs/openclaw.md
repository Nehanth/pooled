# OpenClaw on Pooled

Status: `@pooled/openclaw` 0.2.0, preview. The package is ready for npm but not published yet, and it
isn't in OpenClaw's plugin catalog.

## What it is

[OpenClaw](https://github.com/openclaw/openclaw) is a self-hosted agent: a gateway process on your
machine that runs the agent loop and its tools (`exec`, `read`, `edit`, ...) and calls a model
provider for every step. The `@pooled/openclaw` plugin ([packages/openclaw](../packages/openclaw))
adds **Pooled** as that provider. It is a third-party plugin: nothing in OpenClaw changes.

When you pick Pooled, the OpenClaw gateway itself becomes a device in a Pooled room. It holds its
share of the model's layers on the local GPU and runs them with Pooled's own WebGPU engine, through
Dawn (npm `webgpu`) inside the gateway's Node process ([packages/room-node](../packages/room-node)).
Other devices join the same room and hold the rest:

- another machine with the plugin, or with `pooled join` ([Rooms from the terminal](../cli/README.md));
- a browser tab or a phone with the room's invite link (`https://pooled.run/r/<CODE>#k=…`).

So a model none of the devices can run alone serves the agent, with no API key and no cloud model.
OpenClaw's requests and tool calls take the same path as `pooled serve`: the room's host renders the
chat template, constrains tool calls with a grammar and parses them (`room/api.js`). The host also
keeps OpenClaw's system prompt and each turn as checkpoints, so only the new tokens of each step are
read.

```
 OpenClaw gateway (Spark)                Mac Studio               iPhone (optional)
 ┌───────────────────────────┐          ┌──────────────┐          ┌──────────────┐
 │ agent loop + tools        │          │ pooled join  │          │ Safari,      │
 │ @pooled/openclaw          │  WebRTC  │ Node + Dawn  │  WebRTC  │ pooled.run   │
 │ room node: embed, layers  │ <------> │ layers 19-39 │ <------> │ 1-2 layers   │
 │ 0-18, head, checkpoints   │          │ (Metal)      │          │              │
 └───────────────────────────┘          └──────────────┘          └──────────────┘
```

## Set up

You need Node 22 or newer (OpenClaw ships on 24), OpenClaw 2026.9.6 or newer, and a GPU Dawn runs on:
Linux with glibc 2.38 or newer and a Vulkan driver (tested: DGX Spark, GB10), or macOS 26 or newer
(tested: Mac Studio M5 Max, macOS 27). Windows is untested.

```sh
openclaw plugins install @pooled/openclaw     # --force --accept-capabilities to skip the two prompts
openclaw onboard                              # Model/auth provider -> More... -> Pooled
```

Until the package is on npm, build the tarball from a checkout and install that:

```sh
cd packages/openclaw && npm install && npm pack
openclaw plugins install ./pooled-openclaw-0.2.0.tgz --force --accept-capabilities
```

### Start a room

**Start a room on this device** asks how much GPU memory to lend, which model (downloaded or its size,
what it needs across the room, its context; the one this machine holds alone is recommended), whether
to download it now, how many devices to wait for, and who can join. It prints the code (`4TK-G9P`) and
the invite link. The room opens with the gateway and keeps its code and link across restarts.

On the other devices, open the invite link in Chrome or Safari, run
`npx -p @pooled/cli -p webgpu@0.6.1 pooled join "<invite link>"`, or pick **Join a room** in OpenClaw
there. A device with the code alone waits: `/pooled` in any OpenClaw chat shows who is waiting,
`/pooled allow` lets it in.

### Join a room

**Join a room** takes the invite link (or the code) and the GB to lend, and connects right away. With
the code alone it shows `Waiting for <host> to let this device in…` until the host allows it. It keeps
the pass the host gives it, so the gateway gets straight back in after every restart, and it learns the
host's model and context window.

Models are kept in `~/.pooled/models`, shared with `pooled pull`. Config keys, non-interactive setup
and how requests flow: [packages/openclaw/README.md](../packages/openclaw/README.md).

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
  tokens): 49-64 s on two computers, 89 s with a phone. A hosting gateway now replays the last system
  prompt and tools when its room comes online, so that read happens before the first question
  (checkpoints still live in GPU memory only).
- **Decode over the network.** Two computers over Wi-Fi decode at 25-30 tok/s; the Mac alone does ~92.
  Splitting pays one network hop per token, so it is for models that do not fit on one device.
- **Trust.** Every device holding layers sees the hidden state of every token (which carries the
  prompt and tool results) and can steer the answer, including the tool calls OpenClaw runs. A hosted
  room asks before new devices join; share the invite link only with people you trust, and keep
  OpenClaw's approvals on for `exec` and writes. Details: "Who can see and steer what" in the plugin
  README.
- **Browser hosts** have no turn checkpoints yet: OpenClaw joined to a room hosted in a tab re-reads
  more per tool call than with a Node host.
- **Checkpoint memory** is not part of the layer split: at 50K tokens one checkpoint is ~1 GB across
  the room for the MoE, and the host keeps up to 8 (4 pinned, 4 answer/turn).
- **Windows** is untested. Small models (the 1.7B) get a file-tools-only profile.
