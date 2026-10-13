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
