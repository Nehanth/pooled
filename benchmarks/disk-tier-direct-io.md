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
