# Gate record: direct I/O for the room node's expert disk tier

Working record, kept in the order things happened. §0 was written and committed before any measurement ran.
Method borrowed from Soup's gate-974 (`benchmarks/gate-974-disk-tier-direct-io.md` in MakazhanAlpamys/Soup):
buffered NVMe reads there ran 1.2-2.9 GB/s, direct I/O with sector-aligned ranges and K reads in flight 3.5-5.65.

## 0. The rule (written before measuring)

The question: should `packages/room-node/expert_disk_worker.js` (and `source.js`) read through direct I/O
(`O_DIRECT` / `FILE_FLAG_NO_BUFFERING`) instead of buffered `fs.readSync`?

**Step 1a: the drive.** On the RTX 5070 PC (Windows, SK hynix P41, Node 24.15), read Flash expert-sized ranges
(the real `ffn_{gate,up,down}_exps` part offsets of `C:\llm\models\q38flash`, random experts, 1 MiB pieces as the
reader issues them) four ways: buffered and unbuffered, 1 and K in flight (K = 16, the reader's default; 4 and 32
also printed). Two runs with the arm order reversed. Buffered arms run first-touch (ranges no arm read before in that
run; the files are ~100 GB against 64 GB of RAM). Reported as GB/s and p50 / p99 ms per piece. Numbers only; this
step decides nothing on its own.

**Step 1b: the decode token.** The deciding number is the disk tier's wait per decode token (the expert store's
`diskWaitMs` over decode tokens) as a share of the room's decode token time, in the demo configuration: Mac Chrome
tab (pledge 20) + PC `pooled join --gb 10 --ram 43 --disk`, Qwen3.8-Flash-Next.

- **Disk share < 15% of the token: STOP.** No prototype. Report the numbers.
- **>= 15%: proceed** to step 2.

The same share is also reported, for information only, for one configuration that does put layers on disk
(a lower `--ram`), so the next decision has a number. It does not change the verdict above.

**Step 2/3 (only if step 1b says proceed).** A direct-I/O path is kept only if all hold:
- position-matched A/B, both run orders: converted-expert GB/s (`disk_expert_bench.mjs` conv, first-touch)
  >= 1.3x buffered in both orders, and decode tok/s in the demo room not lower than buffered in either order;
- automatic fallback to buffered when the fast path is unavailable or the store fits in RAM;
- bit-identical output: `tests/q38flash_rule.js` and the Flash golden ids (`flash_tab_gold.mjs`) unchanged.

## 1a. The drive: buffered against direct, 1 to 32 reads in flight (2026-10-12 19:37-19:39 ET)

Machine: the RTX 5070 PC, Windows 11, SK hynix Platinum P41 2 TB, Core Ultra 7 265K, 64 GB RAM, Node 24.15.0.
Files: `C:\llm\models\q38flash` (bartowski Q4_0, three files, 100.6 GB). Probe: `tests/bench/direct_io_probe.mjs`
at 92196b0. Each arm reads 4,320 expert parts (4.0 GB; mean 0.88 MiB a part) that no other arm of the run reads,
from N worker threads with one fd per file each. Before each run it read 85 GB of other files (the UD-Q4_K_XL Flash
files, 25 s) to push the model's pages out of the standby list. Baseline before each run: no node / Chrome /
Python process, CPU 8-9%, GPU idle. The direct arm's first part was byte-checked against a buffered read: identical.

Windows needs no native addon for this. libuv's `UV_FS_O_DIRECT` (0x02000000) passes through `fs.openSync`
and `uv_fs_open` turns it into `FILE_FLAG_NO_BUFFERING`. A read then has to be sector-aligned in offset, length and
buffer address. Linux gets the same from `fs.constants.O_DIRECT`. macOS has neither, and `F_NOCACHE` is not
reachable from Node.

GB/s (p50 / p99 ms per part), run A in the order listed, run B reversed (slot in brackets):

| in flight | buffered, run A | direct, run A | buffered, run B | direct, run B |
|---|---|---|---|---|
| 1 | 1.50 (0.57 / 1.49) [1] | 0.91 (0.99 / 2.27) [2] | 1.63 (0.53 / 1.19) [8] | 2.04 (0.40 / 1.11) [7] |
| 4 | 2.89 (1.13 / 3.05) [3] | 3.21 (0.94 / 3.85) [4] | 3.32 (0.94 / 2.73) [6] | 2.64 (1.12 / 5.28) [5] |
| **16** (the reader's default) | **6.43** (2.08 / 6.11) [5] | **3.65** (5.54 / 8.95) [6] | **6.64** (2.10 / 4.03) [4] | **3.72** (3.06 / 9.44) [3] |
| 32 | 6.52 (4.10 / 12.84) [7] | 3.52 (1.67 / 29.19) [8] | 3.27 (7.13 / 19.94) [2] | 3.08 (1.82 / 39.33) [1] |

A pilot at 2 GB per arm (seed 1, run order A, 19:37) read the same way: buffered 1.55 / 2.93 / 5.93 / 5.82 and
direct 0.91 / 4.29 / 5.32 / 2.87 at 1 / 4 / 16 / 32. At 0.3-0.7 s per arm it was too short to quote.

**Finding 1: on this PC, direct I/O is slower than buffered where the reader runs.** At 16 in flight, buffered
reads 6.43 / 6.64 GB/s in both orders, and direct reads 3.65 / 3.72: 0.55-0.57x. Soup's result (buffered 1.2-2.9,
direct 3.5-5.65 on a Samsung PM9B1 laptop) does not carry over. Here the buffered path is not stuck at a cache-manager
ceiling. A buffered QD1 p50 of 0.53-0.57 ms per 0.9 MiB part is a drive read, not a cache hit, so the eviction held.
Direct only wins at QD1 in run B and at QD4 in run A, and loses the same pairs in the other order. At 32 in flight
both modes are erratic (buffered 6.52 vs 3.27 between orders), and direct's p99 reaches 29-39 ms.

A smoke test on the Spark (GB10, Linux `O_DIRECT`, 0.1-0.2 GB per arm, under other agents' load) read direct
3.2-4.6 GB/s against buffered 1.2-1.6 at 2-4 in flight. That is a smoke number, not a measurement: see §3.

## 1b. The decode token

**The demo configuration reads nothing from disk per token.** Flash room nodes need #368 (`feat/flash-tab-disk`,
faa8f94), because main refuses Flash offload. Its planner, given the demo devices (Mac tab pledge 20 with the OPFS
tier, PC `--gb 10 --ram 43 --disk`), gives the PC layers 13-47. Layers 17-47 park their experts in RAM (40.9 GiB),
with no `diskLo` and 0 GiB of T1 (`disk-tier-direct-io/demo_deal.mjs` and `.out`). #368's own run of that room over
the LAN reports the same shape: "PC 14-47 (17-47 parked in RAM)", 10.8-12.3 tok/s. The `--disk` flag lets the
planner use the disk, but at `--ram 43` it does not need to. The disk tier's share of the demo token is therefore
**0%**. Disk does enter the deal at `--ram 32` (layers 46-47 on disk) and `--ram 24` (43-47), per the same planner.

**Information row: a disk-heavy room, measured** (19:42-19:48 ET, PC alone, PC idle before:
`disk-tier-direct-io/flash_disk_share.mjs`, log `pc_disk_share_ram24.log`). Two room nodes in one process on the
PC's one GPU: a host (pledge 3, layer 0 + embedding/head) and a node `--gb 8 --ram 24 --disk` holding layers 1-47.
The deal: 3-16 parked (19.6 GiB), **17-47 from disk** (31 disk layers, 2.36 GiB T1), 21 pool slots per layer. That
is far more disk than any demo deal. Golden prompts, greedy, current buffered reader:

| case | ids | room tok/s | disk reads | MiB from file | disk wait | wall | **disk share of wall** |
|---|---|---|---|---|---|---|---|
| two-sum | 64 + 23 prompt | 0.69 | 10,750 | 28,345 | 22.9 s | 137.6 s | **16.7%** |
| bash | 22 + 27 | 0.70 | 5,143 | 13,561 | 11.6 s | 71.9 s | **16.2%** |
| eos-mid | 20 + 31 | 0.70 | 5,646 | 14,887 | 11.5 s | 76.1 s | **15.1%** |

All three cases MATCH llama.cpp's golden ids. Disk wait is 225-264 ms against 1,467-1,582 ms of wall per token
(prefill tokens included: with offload they run one at a time). The remaining ~84% goes to the pool cuts, about
50 GB copied to the GPU pools per case at 52-54% pool hits with 21 slots, plus compute and the host hop. Even with
2/3 of the model's layers on disk, the disk is a sixth of the token. The 28.3 GB read in 22.9 s of wait is
1.24 GB/s while waiting. That is the decode pattern's serial-per-layer latency (K = 10 experts x 3 parts per layer,
layer after layer), not the drive's 6.4 GB/s.

## 2. Verdict

**STOP, per §0.** In the demo configuration the disk tier's share of a decode token is 0%, under the 15% bar. No
prototype was built and the reader is unchanged.

The other two numbers point the same way:
- On the demo PC, direct I/O is **0.55-0.57x** buffered at the reader's 16 reads in flight (§1a). The
  Soup-style change would make the PC's disk tier slower, not faster. Windows' buffered path is not the ceiling here.
- Even a room with 31 disk layers spends only 15-17% of its token in disk wait. A doubled read rate would save at
  most ~8% of that token.

What would help a disk-heavy room is not this change. It would be fewer serial waits per token (prefetching the
next layer's experts, or the layer-major prefill of roadmap 32 item 9) and the pool-cut cost, which is ~84% of the
token in the table.

## 3. Not measured

- **Linux `O_DIRECT` at scale.** The Spark was running other agents' Flash GPU tests. The probe's Linux path
  drops the Flash files' page cache (`dd iflag=nocache`), which would have evicted pages their tests use, so only
  the 0.1-0.2 GB smoke runs exist (§1a). Direct I/O may well win there (smoke: 2-3x), but the Spark is not the
  demo machine.
- **macOS.** There is no `O_DIRECT`, and `F_NOCACHE` needs `fcntl`, which Node lacks. A native addon would be
  needed. The Mac runs the demo as a Chrome tab (OPFS), not this reader.
- **The demo room itself (Mac tab + PC).** The Mac's OPFS copy was deleted after #368, and a re-ingest is ~92 GB.
  The 0% follows from the deal (no disk layers), not from a timed run.
- **The converted-expert rate (`disk_expert_bench.mjs` conv) with direct I/O.** It was not needed once the raw
  rate lost.
- **Load time** (`source.js` reads every parked layer at join: ~41 GB at `--ram 43`). That is a separate question
  from the decode disk tier. The same raw table says buffered is the right default on this PC there too, but it
  was not timed.
