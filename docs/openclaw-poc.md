# OpenClaw on a Pooled room: proof of concept results

Date: 2026-09-30. Branch `feat/openclaw-pooled`. OpenClaw 2026.9.7 with the `@pooled/openclaw`
plugin ([packages/openclaw](../packages/openclaw)); the model is Qwen3.6 35B MoE (Q4_0, ~20 GB) split
over real devices, 128K context. Every answer below was produced by OpenClaw's own agent loop
(`openclaw agent`), with OpenClaw's full tool profile. There is no API key and no cloud model.

## Devices

| Device | Runtime | Role |
|---|---|---|
| DGX Spark (GB10, Linux, Vulkan) | Node 24 + Dawn (`packages/room-node`) inside the OpenClaw gateway, or `join.mjs` | host in A and C, joiner in B |
| Mac Studio M5 Max, 36 GB, macOS 27 | Node 24 + Dawn on Metal, inside the OpenClaw gateway or `join.mjs` | joiner in A and C, host in B |
| iPhone 14 Pro Max, iOS 18.7, Safari 26.6.1 | the live room page on pooled.run | holds 2 layers in C |

The two computers talk over Wi-Fi on the same LAN (the Spark is on Wi-Fi too). A and B use a local
PeerJS server for signaling; C uses the public one, because the phone opens the room page on
pooled.run.

## Scenarios

- **A**: OpenClaw + plugin on the Spark hosts room SPKA. The Mac joins with `join.mjs`. Split: Spark
  layers 0-18 + embedding/head, Mac layers 19-39. The room was online 47 s after the Mac joined.
- **A2**: A again, with turn checkpoints (next section). Online after 42 s.
- **B**: OpenClaw + plugin on the Mac hosts room MACB. The Spark joins. Split: Mac 0-18 + head,
  Spark 19-39. Online after 28 s.
- **C**: A plus the iPhone. Spark 0-17 + head, iPhone 18-19, Mac 20-39. The phone loaded its
  912 MB share in 65 s and held it for the whole run (23 minutes, no errors). Online after 95 s.

The tasks run in a sandbox Node project ([packages/openclaw/test/agent](../packages/openclaw/test/agent)).
Each task starts a new OpenClaw session, and a script the agent never sees checks the result:

| Task | Prompt (short) | Check |
|---|---|---|
| read | How many attempts does the tax client make, and where is the limit defined and enforced? | answer says 4, `src/config.js`, `withRetries`, `taxclient.js` |
| edit | Make `formatMoney` print negative amounts as `-$12.34` | 7 input/output cases |
| tests | Run `npm test`, fix the source (not the tests), run it again | the fixed functions are correct, tests unchanged, suite passes |
| feature | Add `bulkPrice` in a new module, write its tests, run them until they pass | 7 cases, the test file exists, suite passes |
| long | Read the whole 140 KB journal (~33k tokens) and summarize every [MAJOR] incident in SUMMARY.md | all 5 planted incidents with the right date and root cause |

## Results

Column notes: *TTFT* is the first model call's prompt read time (the cold one on `read`). *Read /
total* adds up every model call in the task: the prompt tokens actually read, against all prompt
tokens, and the share that came from checkpoints. *Decode* is output tokens over decode time, for
all calls. Every tool call was well formed: no tool call in any run came back as an error.

### A: Spark hosts, Mac joins (before turn checkpoints)

| task | ok | wall s | model calls | tool calls | TTFT s | prompt tok first / max | read / total prompt tok (reused) | prefill s | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|
| read | PASS | 120 | 4 | 3 | 61.8 (cold) | 17439 / 20261 | 25606 / 77524 (67%) | 94.2 | 25.7 |
| edit | FAIL | 48 | 3 | 2 | 4.2 | 17451 / 18000 | 2229 / 53075 (96%) | 7.6 | 24.8 |
| tests | PASS | 119 | 8 | 8 | 4.5 | 17428 / 21950 | 17219 / 154595 (89%) | 67.0 | 28.7 |
| feature | PASS | 298 | 13 | 17 | 5.1 | 17493 / 23250 | 44280 / 268186 (83%) | 173.9 | 27.5 |
| long | PASS | 813 | 7 | 6 | 5.0 | 17472 / 53206 | 149785 / 269855 (44%) | 671.8 | 16.7 |

### A2: Spark hosts, Mac joins (with turn checkpoints)

| task | ok | wall s | model calls | tool calls | TTFT s | prompt tok first / max | read / total prompt tok (reused) | prefill s | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|
| read | PASS | 104 | 4 | 3 | 61.7 (cold) | 17448 / 20305 | 20564 / 77662 (74%) | 75.6 | 24.3 |
| edit | PASS | 55 | 4 | 3 | 5.0 | 17460 / 18278 | 2309 / 71377 (97%) | 10.1 | 25.4 |
| tests | PASS | 57 | 7 | 7 | 5.0 | 17437 / 19986 | 4183 / 132291 (97%) | 18.3 | 29.4 |
| feature | PASS | 139 | 12 | 18 | 4.2 | 17502 / 23088 | 7665 / 244926 (97%) | 31.8 | 29.7 |
| long | PASS | 306 | 7 | 6 | 5.0 | 17481 / 52997 | 37238 / 269709 (86%) | 170.3 | 15.9 |

### B: Mac hosts, Spark joins

OpenClaw on the Mac sends a smaller first prompt (14.5k tokens against 17.4k on the Spark).

| task | ok | wall s | model calls | tool calls | TTFT s | prompt tok first / max | read / total prompt tok (reused) | prefill s | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|
| read | PASS | 75 | 4 | 3 | 49.2 (cold) | 14494 / 15707 | 15966 / 61055 (74%) | 55.6 | 27.8 |
| edit | FAIL | 29 | 3 | 2 | 4.5 | 14506 / 14917 | 1739 / 44056 (96%) | 7.7 | 31.5 |
| tests | PASS | 53 | 7 | 9 | 4.6 | 14483 / 16811 | 3862 / 110675 (97%) | 16.9 | 27.7 |
| feature | PASS | 59 | 5 | 6 | 4.5 | 14548 / 16372 | 3362 / 76975 (96%) | 14.1 | 29.3 |
| long | PASS | 287 | 7 | 6 | 4.6 | 14527 / 49967 | 37062 / 248256 (85%) | 165.7 | 17.1 |

### C: Spark hosts, Mac and iPhone join

| task | ok | wall s | model calls | tool calls | TTFT s | prompt tok first / max | read / total prompt tok (reused) | prefill s | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|
| read | PASS | 154 | 4 | 3 | 89.2 (cold) | 17439 / 20261 | 20520 / 77524 (74%) | 110.3 | 14.4 |
| edit | FAIL | 59 | 3 | 2 | 7.9 | 17451 / 18000 | 1974 / 53075 (96%) | 13.2 | 16.1 |
| tests | PASS | 148 | 8 | 9 | 7.3 | 17428 / 22072 | 6338 / 155132 (96%) | 43.2 | 15.5 |
| feature | PASS | 388 | 12 | 23 | 8.2 | 17493 / 24717 | 9300 / 254832 (96%) | 67.2 | 15.5 |
| long | PASS | 576 | 7 | 6 | 7.9 | 17472 / 53049 | 37296 / 269692 (86%) | 329.0 | 8.7 |

The phone adds a third hop on every lap. Decode drops from ~27 to ~15 tok/s and prompt reading is
~2x slower. It works, but a phone should only take layers when the computers cannot hold the model.
Here the computers pledged 11 GB each on purpose, so that the phone would get layers.

## What failed

- **edit (A, B, C): a model mistake, not a room one.** The model put the sign after the currency
  (`$-12.34`), told the user it was fixed, and did not run anything to check. It made the same edit
  on both hosts. In A2 it ran a check with `exec` and got it right. Every failed and passed answer is
  in the run logs.
- **The first try at C:** the host used a 5-character room code. The plugin and the room node
  accept 4 to 6 characters, but the room page's code box keeps only 4, so the phone got "No room
  with that code". Use 4-character codes (onboarding makes 4).
- Before turn checkpoints (A), long tool loops re-read everything after the system prompt on every
  call. Fixed in this branch (next section).

## Turn checkpoints (found and fixed in this POC)

OpenClaw ends every model call with a user turn of per-call context
(`<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>> ...`) and drops it from the next call's history. So the
host's answer checkpoint, whose tokens include that block, was never a prefix of the next call. Each
call in a tool loop resumed from the pinned system prompt (17,306 tokens) and read everything after
it again. In A's long task that meant reading 7.8k, then 15.5k, 23.3k and 31k tokens (up to 140 s
per call).

The host now saves one more checkpoint where the prompt's last user turn starts
(`packages/room-node/ckpt.js` `turnPoint`). The next call resumes there and reads only the assistant
turn, the tool results and the new context. It is kept with the answer checkpoints (4 now), and
`POOLED_CKPT_DEBUG=1` logs where a prompt leaves each checkpoint. Same room, same tasks, A to A2:

| task | wall s A to A2 | prompt tokens read A to A2 |
|---|---|---|
| tests | 119 to 57 | 17.2k to 4.2k |
| feature | 298 to 139 | 44.3k to 7.7k |
| long | 813 to 306 | 149.8k to 37.2k |

Correctness: the checkpoint-free baseline (`packages/room-node/test/e2e.mjs cache`, MoE, Spark) gives
the same 6 answers, token for token, with turn checkpoints on. Unit tests cover the tool loop with an
engine whose output depends on its whole history.

## Still open

- The first call after a gateway starts is cold: 14-17k tokens take 49-62 s on two computers, and 89 s
  with the phone. Checkpoints live in GPU memory only.
- Decode is 25-30 tok/s on two computers over Wi-Fi. The Mac alone does ~92 tok/s and the Spark ~49.
  The split pays for a network hop on every token.
- The long task's decode drops to ~16 tok/s at 50k context.
- The first call of each new session reuses the pinned system prompt but reads the user's question
  and OpenClaw's per-call block again (~1k tokens). That is small.

Logs for every run (gateway, joiner, phone, per-task agent output and answers) are kept outside the
repo for the video.
