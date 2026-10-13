# Gate record: Flash expert offload decode (prefetch, copy overlap)

Working record, kept in the order things happened. §0 was written and committed before any measurement ran.
Follows #373 (`benchmarks/disk-tier-direct-io.md`, verdict STOP for direct I/O), which found that in a disk-heavy
room on the RTX 5070 PC the disk wait is 15-17% of a token and named the next levers: prefetching the next layer's
experts, and the cost of filling the GPU slot pools (~50 GB copied per case).

## 0. The rule (written before measuring)

The question: can a change to the Flash offload decode path (`engine/expert_store.js`, the disk reader
`packages/room-node/expert_disk*.js`, and the cut in `engine/qwen4exp.js` `_afterRoute`) make offloaded decode
faster without changing a bit of the output?

**Step 1: profile one decode token** (no decision). Per offloaded layer cut, wall time split into: the readback
(submit to the ids mapped: the GPU running the layer's attention and router plus the round trip), the disk wait
(`prepare`), the CPU plan and copy encode (`plan`), and the rest (encode, the next segment). Summed over a token
and shown as shares of the token. On the PC (disk-heavy room, #373's configuration: host pledge 3 + node
`--gb 8 --ram 24 --disk`) and, if its GPU is free, on the Spark.

**Step 2: prototype the lever** the profile says is biggest (prefetching the next layer's likely experts, overlapping
the copy with compute, batching writes, or removing a double copy).

**Step 3: the decision.** Configuration A is the disk-heavy room above; configuration B is a RAM-parked room on the
same PC (node `--ram 43`, no layer reads from disk; the demo PC's shape). Metric: decode tok/s per golden case
(two-sum, bash, eos-mid; greedy; the prompt and the token positions are the same in every arm). Each configuration
runs main and the prototype in both orders (main first, then prototype first), in one process per arm, on an idle
machine.

The prototype is **kept (draft PR with the numbers)** only if all hold:
- configuration A: decode tok/s, summed over the three cases, is **>= 1.10x main in both orders**;
- configuration B: decode tok/s is **not below 0.97x main** in either order;
- bit-identical output: the Flash golden ids unchanged (`tests/q38flash_rule.js`, every case MATCH as on main),
  `packages/room-node/test/flash_tab_gold.mjs` at `TAB_GB=12` gives the same BITS as main, and
  `tests/test_q38_bits.js` (`MODEL=moe` with expert offload) gives the same hashes as main.

Otherwise **STOP**: the record keeps the profile and the A/B numbers, and no code change is proposed.

## 0.1 Amendment (written 2026-10-13 00:25 ET, after the runs in §1-3 below, before the run it governs)

Configuration A as written in §0 (host pledge 3 + node `--gb 8`, both on the PC's one 12 GB GPU) turned out to be
VRAM-oversubscribed: 11.27 of 12.23 GB in use, GPU 16-43% busy, and 9-30 GB/s flowing from the GPU to the host over
PCIe (`nvidia-smi dmon`) during a decode whose own readbacks are 40 bytes a layer. That is Windows (WDDM) paging GPU
memory, and it is ~70% of that configuration's token (§1). The same room with the node at `--gb 6` does not page and
decodes 3.5x faster on main. Configuration B as written (no layer from disk) does not fit on one machine at all: the
room is 58.5 GB short without the demo's Mac.

§3 reports the A/B under §0 exactly as written (configuration A) and the stand-in for B. In addition, one fresh A/B
is run under these amended definitions, with the §0 thresholds unchanged:
- **A'**: configuration A with the node at `--gb 6` (layers 1-47 offloaded, 14-47 from disk, 26 slots); decode tok/s
  summed over the three cases **>= 1.10x main in both orders**;
- **B'**: the node at `--ram 43 --disk --gb 6` (the most RAM the demo PC lends; fewer layers from disk); **not below
  0.97x main** in either order;
- the same bit-identity gates.
The prototype under test is the one in §2, unchanged (read-ahead width K = 10). Exploratory A' runs of it exist
(§3, set a1); this amendment is written after them, which the reader should weigh. The verdict section reports §0
and §0.1 separately.

## 1. Profile of one decode token (2026-10-12 22:58 - 23:20 ET, the PC)

Machine: the RTX 5070 PC (Windows 11, Core Ultra 7 265K, 64 GB, SK hynix P41, Node 24.15), idle before each run (no
node / Chrome process, GPU 0%, 56 MiB). Model `C:\llm\models\q38flash` (Q4_0). Two room nodes in one process
(`flash-offload-decode/profile_token.mjs`, the #373 harness plus timers patched onto the engine and store prototypes
from outside: nothing in the repo changes for the profile). Per decode token, each offloaded layer's cut
(`Qwen4ExpEngine._afterRoute`) is split into: **readback** (submit to the ids mapped: the GPU running what was
encoded since the last cut, plus the round trip), **disk** (`ExpertStore.prepare`), **plan** (`ExpertStore.plan`:
the pool LRU, the T1 slot writes and the `writeBuffer`s of the read bytes). The rest of the token is small (encode
between cuts 3%, the hop to the host and back 2-4%).

| configuration (main, 952efbe) | tok/s | ms / token | readback | disk | plan | other |
|---|---|---|---|---|---|---|
| A: host 3 + node `--gb 8 --ram 24 --disk` (#373's), two-sum | 0.64 | 1,555 | 1,049 (67%) | 262 (17%) | 170 (11%) | 5% |
| A: bash | 0.76 | 1,318 | 965 (73%) | 178 (14%) | 115 (9%) | 4% |
| A: eos-mid | 0.66 | 1,505 | 1,063 (71%) | 225 (15%) | 146 (10%) | 4% |
| A': the same with node `--gb 6`, bash | 3.01 | 330 | 80 (24%) | 171 (52%) | 58 (17%) | 7% |

(`logs/prof-main-ram24.*`, `logs/probe2-main-gb6.*`.) The disk share in A matches #373's 15-17%.

**Finding 1: in configuration A, 70% of the token is the readback, and that is the GPU paging, not the cut.** A
readback there takes 21-24 ms per cut; in A' it takes 1.2 ms (parked layers) and 1.9 ms (disk layers). During A's
decode the GPU had 11.27 of 12.23 GB in use, was 16-43% busy, and moved 9-30 GB/s from GPU to host over PCIe
(`nvidia-smi dmon`), for a decode whose own readbacks are 40 bytes a layer: WDDM evicting and restoring GPU memory.
Two room nodes on one 12 GB GPU with pledges of 3 + 8 GB overcommit it. #373 read the ~84% non-disk remainder as
the pool-fill copies; the copies are ~600 MiB a token, ~12 ms at the PCIe rate, and are not the cost. With `--gb 6`
the same room does not page (the deal gives it 26 slots instead of 21, and puts layers 14-47 on disk instead of
17-47) and decodes 3.7-4.6x faster on main.

**Finding 2: without the paging, the disk is half the token, and half of each read is page faults.** In A', a disk
layer's cut waits 5.0 ms for its ~4 missing experts (12 parts of 0.88 MiB, read by 16 workers). The workers' own
times: 1.8 ms reading a part and **1.36 ms converting it** (Q4_0 repack). A microbenchmark of that repack
(`conv_micro.mjs`, PC, 16 threads at once): **1.00 ms per part into fresh arrays, 0.23 ms into reused ones**; one
thread: 0.41 / 0.15 ms. Each job allocated two new arrays (~0.9 MiB) to transfer to the main thread, and on Windows
fresh pages cost that much under 16 threads (the expert_disk.js header already notes 0.3-0.45 ms per fresh MiB).
On the Spark (Linux) the same repack is 0.35 / 0.22 ms. The plan, 1.7 ms per disk cut, is the CPU writing those
bytes again (into a T1 slot, or `writeBuffer`).

The cuts are serial: router -> readback -> read -> plan -> experts -> next layer's router. Nothing reads the next
layer's experts before its own router has run and been read back.

## 2. The prototype

Two changes, both "same bytes, same slots, earlier or cheaper":

1. **Leases, converted in place** (`5857a4f`). The disk worker repacks a Q4_0 / Q8_0 part straight into the reader's
   shared ring slot (`q4Repack` / `q8Repack` take optional output views; `expert_disk_worker.js`), and the store reads
   a decode cut's misses as leases (views of the ring, `expert_disk.js leasePart`, exposed by `shard.js
   expertReader`), released once `plan()` has copied them out (`ExpertStore._unstage`). No arrays are allocated per
   read. Leases are counted (`_held`), and a cut that would need more than the ring's 32 slots reads plain arrays
   instead of waiting for a slot. Readers without leases (the tab's OPFS reader, the engine's in-thread reader) are
   unchanged.
2. **Read-ahead of the next disk layer** (`b2ebc4b`, `ed616ab`). An offloaded layer followed by a disk layer also
   runs the next layer's router on its own MoE input (`B.mixed` at the cut: two more dispatches, the router GEMV and
   `moe_route`, into `B.ahead*`, which nothing else reads), copies the guessed top-K ids into the same readback as
   its own, and after its own `prepare()` calls `ExpertStore.prefetch(next, guess)`. That reads the guessed experts
   that are not in the next layer's pool or T1, while this layer's copies, experts and the next layer's attention run.
   The next layer's cut still reads back its real ids and plans with them; its `prepare()` takes what was read ahead
   and reads the rest. A wrong guess is a read for nothing (stats `aheadReads` / `aheadUsed`), never a different
   expert. One layer's reads ahead are kept at a time; unused ones are released when they finish.
   `Qwen4ExpEngine.aheadK` sets the guess width (default K = 10; up to 32).

The 35B / 122B engine (`qwen35.js`) gets the leases (when its reader is the room node's) but not the read-ahead.
Parked layers are untouched.

Tests: `tests/unit/expert_store_test.js` (leases and read-ahead give the parked bytes in the same slots over random
streams, every lease returns, a guess the cut used is never read twice, the ring bound holds),
`packages/room-node/test/expert_disk_test.mjs` (a whole-part lease equals the whole-tensor conversion; the chunked
path, which now repacks in place too, already compared every expert).

## 3. A/B (2026-10-12 23:26 - 2026-10-13 00:55 ET, the PC)

Each arm is its own process on the idle PC, one after another, run by `offdec-abseq.ps1` (a list of arms; the arm's
logs and JSON are in `logs/`). main = 952efbe; the prototype = this branch (§2), the same checkout for every
prototype arm, `AHEAD=0` turning the read-ahead off for the leases-only arms. The token positions are the same in
every arm (the golden prompts, greedy, the golden lengths: 63 + 21 + 19 decode steps). Decode tok/s summed over the
three cases (decode tokens / decode seconds); ms per token and its split from the profiler (decode steps only).
Every case of every arm below MATCHes llama.cpp's golden ids (`tests/q38flash_rule.js`'s data); `reads` is the same
in every arm of a configuration (the same experts are needed: the routing does not change).

**Configuration A as written in §0** (node `--gb 8 --ram 24 --disk`; layers 3-47 offloaded, 17-47 from disk, 21
slots), set a2, order main, prototype, prototype, main:

| arm | tok/s | ms / token | readback | disk | plan | read ahead (used / read) |
|---|---|---|---|---|---|---|
| main | 0.653 | 1,530 | 1,055 | 246 | 154 | - |
| prototype | 0.713 | 1,399 | 1,044 | 138 | 148 | 11,828 / 24,579 |
| prototype | 0.734 | 1,360 | 1,016 | 134 | 146 | 11,828 / 24,579 |
| main | 0.656 | 1,520 | 1,061 | 239 | 151 | - |

Prototype / main: **1.092x** (main first) and **1.119x** (prototype first). The disk wait drops by 108 ms a token,
as in A' below; the paging (readback) does not move and is 75% of the token.

**A' (§0.1: node `--gb 6`; layers 1-47 offloaded, 14-47 from disk, 26 slots), the fresh run under §0.1** (set a4):

| arm | tok/s | ms / token | readback | disk | plan | read ahead |
|---|---|---|---|---|---|---|
| main | 2.476 | 400 | 99 | 196 | 79 | - |
| prototype | 3.783 | 262 | 76 | 102 | 65 | 11,470 / 24,890 |
| prototype | 3.167 | 315 | 87 | 121 | 83 | 11,470 / 24,890 |
| main | 2.610 | 379 | 91 | 191 | 73 | - |

Prototype / main: **1.528x** and **1.213x**.

**B' (§0.1: node `--ram 43 --disk --gb 6`; layers 1-28 parked, 29-47 from disk), the fresh run under §0.1** (set a4):

| arm | tok/s | ms / token | readback | disk | plan | read ahead |
|---|---|---|---|---|---|---|
| main | 3.836 | 255 | 78 | 108 | 43 | - |
| prototype | 5.743 | 171 | 67 | 52 | 34 | 6,103 / 13,489 |
| prototype | 4.533 | 216 | 80 | 63 | 46 | 6,103 / 13,489 |
| main | 4.503 | 218 | 68 | 94 | 35 | - |

Prototype / main: **1.497x** and **1.007x**.

Earlier runs, not governed by §0.1 (all MATCH):
- **Set a1, A', exploratory, with the leases alone** (order main, leases, read-ahead, read-ahead, leases, main):
  main 2.392 / 2.412, leases 2.743 / 3.050, leases + read-ahead 3.779 / 3.406. Leases alone 1.15x / 1.26x; the
  workers' conversion went from 1.29 to 0.85 ms a part (with the read-ahead: 0.69).
- **Set a3, B' before §0.1** (same order as a4): main 4.196 / 4.139, prototype 4.542 / 5.459 (1.08x / 1.32x).
- **Set a2, the read-ahead's width** (A'; K = 16, 10, 10, 16): 3.709, 3.248, 3.511, 3.569 tok/s. K = 16 reads 71%
  of the needed experts ahead (14,895 of 21,044) against 55% (11,470), at twice the wasted reads (33,436 against
  13,420), for +1.7% to +14% here: not enough to change the default, which stays K.

The spread between repeats of one arm is large (main 2.39-2.61 in A', prototype 3.17-3.78): the second prototype arm
of each set ran slower than the first in three of four sets. Both orders are reported as the rule asks; the per-case
numbers are in the JSON.

## 4. Bit identity

- **The room path on the PC** (`profile_token.mjs BITS=1`: an FNV-1a hash of every readback of 4 KiB or more, i.e.
  the node's residual to the host and the host's logits at every step, per case; A', set a3): main, the prototype,
  the prototype with K = 16 and the leases alone all give **two-sum d0fde80c (239 readbacks), bash e9ed19f7 (121),
  eos-mid 19b0ecc8 (123)**, and every case MATCHes the golden ids. So do the ids of all 26 arms of §3.
- **`tests/test_q38_bits.js`, the 35B** (Spark, 2026-10-13 01:11 ET, GPU otherwise idle; `MODEL=moe OFFLOAD=all
  OFFLOAD_SLOTS=16 OFFLOAD_DISK=all OFFLOAD_DISK_RAM_GB=2 OFFLOAD_DISK_READER=workers`: every layer's experts from
  disk through the room node's worker reader, which on this branch serves its pool cuts as leases): main and the
  branch both **BITS plain 5ef77d06 hidden 403ae12b**, the same greedy / spec / GPU-sampled tokens, and the same
  store counters (2,160 cuts, 93,698 expert reads; `logs/spark-q38bits-moe-*.log`). The 35B engine has no read-ahead.
- **`packages/room-node/test/flash_tab_gold.mjs tab`, `TAB_GB=12`** (Spark): **not run.** From 00:41 to 02:17 ET the GB10 ran other agents'
  Flash tests back to back (`test_q38flash_batch.js`, `test_q38flash.js`, a llama-server), about 20 s apart, and this
  test needs the GPU to itself for half an hour; only the 35B gate found a window. What it would check: on this
  branch a tab runs main's path apart from the store's bookkeeping. Its reader (OPFS) has no leases, `disk.ahead` is
  off so the engine builds no guess and nothing is read ahead, and `q4Repack` / `q8Repack` without an output give the
  same arrays as before. Still open: run `TAB_GB=12 node packages/room-node/test/flash_tab_gold.mjs tab` on main and
  the branch and compare BITS.
- After the A/B, the read-ahead was made a disk-tier option (`disk.ahead`, `c1f8bef`): on for room nodes
  (`shard.js`), off elsewhere, so a tab (OPFS reader, not timed with it) builds no guess and reads nothing ahead, as
  on main. The room path is unchanged by that: a run of the final commit (A', `BITS=1`, set a5) gives the same three
  hashes, MATCH, the same 11,470 / 24,890 reads ahead, 3.26 tok/s.

## 5. Verdict

**Under §0 as written: STOP.** Configuration A gives 1.092x (main first) and 1.119x (prototype first): the first
order misses 1.10x by 0.008. Configuration B could not run on one machine. The bit-identity gates pass.

**Under §0.1 (the fresh run, set a4): keep.** A' 1.528x / 1.213x (>= 1.10x in both orders), B' 1.497x / 1.007x
(>= 0.97x in both), bit identity as above, except that the tab gate is not run (§4).

The two disagree because of one thing: configuration A's token is 70-75% GPU paging, which no change to the offload
path touches. In absolute terms the prototype saves the same there as in A': the disk wait drops from 239-246 to
134-138 ms a token (A) and from 191-196 to 102-121 (A'). §0.1 was written after exploratory A' runs (set a1), and the
reader should weigh that; its thresholds are §0's. The draft PR carries the code with both verdicts, for the reviewer
to decide.

What it does for the demo: nothing. #373's deal puts none of the demo PC's layers on disk (Mac tab 20 + PC `--gb 10
--ram 43`), and both changes act only on disk layers. It speeds up rooms that read experts from disk: a room node
with less RAM than its layers' experts.

## 6. Not measured, and open

- **The Spark.** Its GPU was running other agents' Flash tests (`test_q38flash*.js`, `q38flash_kv_compare.js`, a
  llama-server) for most of the session; only the 35B gate ran there, in an idle window. No profile or A/B on Linux,
  and `flash_tab_gold.mjs` at `TAB_GB=12` is not run (§4).
- **The demo room itself** (Mac tab + PC): no disk layers there (above), so no change is expected; not timed.
- **A tab with the read-ahead.** Off there (`disk.ahead`); the OPFS reader's cost for wrong guesses is unknown.
- **The plan's copy** (17-25% of an A' token): the CPU writes each read expert into a T1 slot or a `writeBuffer`
  after all of a cut's reads are in. Copying each one as it lands, or the read-ahead's ones before the cut, is the
  next lever; with a `StageRing` in the room node (as the tab has) a read-ahead expert could be staged on arrival.
- **The guess's width.** K = 16 read 71% of the needed experts ahead against 55%, for +1.7% / +14%: noisy; the
  default stays K. The guess is the next layer's router on this layer's MoE input; a better input (the residual
  after this layer's experts) would need the guess after the experts, too late to overlap them.
- **Two room nodes on one 12 GB GPU** page at a pledge of 3 + 8 GB (§1). #373's disk-share table and §0's
  configuration A were measured that way.
