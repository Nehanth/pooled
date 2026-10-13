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
