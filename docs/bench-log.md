# Bench log

Every kernel/engine change gets a row. All 27B numbers are Qwen 3.8 27B Q4_0,
greedy, bit-identical output verified by the test suite (`tests/test_mtp.js`,
`tests/test_batch*.js`). GB10 = DGX Spark via Deno/wgpu (Vulkan, no subgroups,
shader-f16 available). Mac = user's MacBook, Chrome, staging site.

| Date | Change | Commit | GB10 decode plain | GB10 decode spec K=3 | GB10 spec K=7 | GB10 prefill (batched) | Mac decode | Mac prefill | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Aug 29 | Launch state | main | 3.6 | — | — | 3.0 | 2.5 | ~3 | one thread per row, f32 scales |
| Aug 31 | Coop kernels + batched prefill + fusion + f16 scales | 71b7b85..9f8b852 | 9.0 | — | — | 15.2 (4-col) | 6.7 | 14–16 | |
| Sep 1 | Multi-column batched ops (2x batched passes) | f1a87a9 | 9.1 | — | — | 39.4 | | | |
| Sep 1 | MTP self-speculation | e8d5642, 15c389f | 9.07 | 15.86 (85% acc) | — | 39.4 | 10–10.8 | 12 s/prompt | cross-network 3.5–4 tok/s |
| Sep 1 | Adaptive deep speculation (K=3/5/7 by lap RTT) | 3461d10 | 8.72–9.14 | 16.07 (85%) | 13.09 (71%, 6.0 tok/lap) | 39.4 | | | K=7 only chosen when lap >260 ms |
| Sep 1 | GPU argmax for the draft chain (8-byte readback instead of 1 MB/draft) | — | 9.06 / 8.85 | 16.13 (85%) | 12.32 (71%) | | | | neutral on GB10 (unified memory); helps discrete GPUs |
| Sep 1 | 8-wide batched columns (BCOLS=8) — prefill | — | 8.7–9.0 | | | 27.3 (4w) vs 26.1 (8w×2r) vs 27.9 (8w×4r), 86-tok prompt | | | no gain: 8-col pass ≈ 4-col pass ⇒ prefill is bound by the serial DeltaNet recurrence, not GEMV |
| Sep 1 | **Native reference: llama.cpp CUDA (build 749f688), same GGUF** | — | **7.99** (tg32) | — | — | **377** (pp86) | | | WebGPU beats native on decode (9.0 plain); prefill 14x behind ⇒ parallel recurrence is the prize |
| Sep 1 | Deep spec with single 8-col verify (BCOLS=8) | e07c0e0 | 8.68 | | K=7: 14.86 (71%); K=5: 16.65 (97%, partly contaminated) | | | | K=7 never pays solo; K=5 is the candidate |
| Sep 1 | unpack4xU8/I8 dequant in all coop kernels (probe-gated fallback) | — | 8.65 | | | 26.0 | | | bit-exact; neutral on GB10 (driver already optimized the shifts) — kept for Metal/Android where ALU is scarcer |
| Sep 1 | Bandwidth probe: achievable streaming read on GB10 via WebGPU = 184 GB/s | — | | | | | | | decode GEMVs = 15 GB / 82 ms = 183 GB/s ⇒ AT roofline; remaining decode cost is ~30 ms of small-dispatch overhead |
| Sep 2 | Accumulate matvecs (y += W·x): residual adds folded in, −192 dispatches/token | — | 8.93 | 15.64 (85%) | | 25.8 | | | bit-exact; neutral on GB10 ⇒ small dispatches are ~free on Vulkan; kept for Metal |
| Sep 2 | Gates+L2 fused (dn_pre), room at 8 cols × 2 rows, 4-col twin kernels, 2-D dispatch for tall matvecs | — | 8.8–9.2 | 15.49 (85%, room cfg) | K=5: 16.02 (97%, room cfg); K=5 @ 8×4 rows: 17.00 | 42.8 (18-tok test) | | | bug fixed: LM head at 2 rows/WG = 124k workgroups > 65535 limit, dispatch silently dropped |
| Sep 2 | GEMM prototype for prefill (benchmarks/bench_gemm.js): 16-col tiled Q4 GEMM, vec4 shared tiles, 2×4 thread tile | — | | | | 17408×5120 × 16 cols: gemm 1.83 ms vs coop_b 4×4-col 1.93 ms (parity) | | | correct (2.5e-7); only 27 GB/s / 1.5 TFLOPS ⇒ latency-bound (2 barriers × 160 k-blocks, loads exposed). Next: register prefetch of block b+1, 64-k steps, 2+ WGs/SM |
| Sep 2 | GEMM v3 (padded shared tile, 2 blocks/barrier, register prefetch, RT=2×4 cols) | — | | | | 17408×5120×16: 1.54–1.61 ms vs 1.93 (1.21–1.25×); 5120×5120: parity (occupancy: 80 WGs) | | | bank-conflict padding was the only lever that moved it; 4×4 tiles slower (occupancy). Still ~30 GB/s / 1.8 TFLOPS: next try split-K for small dOut, 256-thread WGs, check naga bounds-check cost |
| Sep 4 | **Prefill GEMM** (roadmap 02): row-stationary Q4_0 GEMM at 16 batch columns, split-K pinned per shape, `_dop` ladder GEMM→b8→b4 | — | 9.0–9.2 (unchanged) | 15.4–15.9 (unchanged, K=5 at 16 cols) | | pass-level 34.6 → **56.7** (1.64×); end-to-end `bench.js` 30.2 → **43.7** (1.45×) | | | ffn_down + ssm_out are Q8_0 ⇒ stay on the GEMV; Q8 variant is the next lever |
| Sep 4 | Width-dependent rows per workgroup for the b8/b4 twins | — | | 12.8 → 15.4 at 16 cols | | | | | fixed the decode regression the 16-column width introduced; twins byte-identical to native-width kernels (`tests/test_twins.js`) |

## Standard prompts

Room numbers are only comparable when the prompt is the same. Use these, verbatim, and name the prompt in the row. Token counts are for Qwen 3.8 27B's tokenizer, text only; the chat template adds about 9 tokens.

| Name | Tokens | Text |
|---|---|---|
| `hello` | 1 | `hello` |
| `meaning` | 6 | `what is the meaning of life` |
| `japan` | 160 | `I am planning a two week trip through Japan in late October with my partner. We land in Tokyo, want three days there, then a day trip to Nikko, then the bullet train to Kyoto for four days with a side trip to Nara, then two nights in Osaka, and we fly home from Osaka. We like food markets, old temples, hiking, and small neighborhood bars, and we want to avoid the most crowded tourist spots where we can. Our budget is moderate, around two hundred dollars a day for the two of us not counting hotels. Please give me a day by day itinerary with one main activity each morning and afternoon, a neighborhood to eat dinner in each night, and tell me which days I should buy a rail pass for and whether it is worth it at all.` |

`hello` and `meaning` measure fixed per-request overhead. `japan` is long enough that prefill runs through the 16-column GEMM path for most of its length; use it for any prefill claim. Decode numbers are for the 400-token answer cap; the first answer in a room is slower because the speculation depth starts conservative, so report the second answer or later.

## Room benchmarks by PR (real devices)

One row per merged PR that changes speed, measured in the room on real devices, before and after. "Before" is production (`main`) on the same day; "after" is the PR's preview URL. Prefill is the seconds the status line reports; decode is the tok/s in the answer footer.

| Date | PR | Devices | Prompt | Prefill before → after | Decode before → after | Notes |
|---|---|---|---|---|---|---|
| Sep 4 | #29 prefill GEMM | MacBook (Chrome, Metal) 62 layers + embed/head, iPhone 2 layers | `japan` | 13.8 s → **8.5 s** (1.62×) | 7.3 → 7.7 tok/s (unchanged, within noise) | first answer in each room; both runs hit the 512-token context overflow (#31) after ~300 generated tokens, which does not affect the prefill number |

## Per-hop telemetry on a Mac and an iPhone (PR #51, @aaryanmanchanda)

Measured 2026-09-13 by @aaryanmanchanda on his own devices, from PR #51 (closed because per-hop telemetry landed on the main work branch in a different shape; the numbers are his). MacBook Air M1 8 GB (Brave) as host, iPhone 13 Safari as worker with a 0.5 GB pledge, same Wi-Fi, local HTTPS. Qwen3 0.6B Q8, default wire (`?wire=stripe4`), `japan` prompt, second answer in the room. Generation: 400 tokens at 12.4 tok/s on 2 devices.

| | p50 | p90 |
|---|---|---|
| iPhone compute per hop | 30 ms | 35 ms |
| Derived transport (lap minus compute minus host pack and compute) | 10 ms | 64.2 ms |

His run also found two bugs that the main branch fixed too: the striped wire's fixed slice header dropped per-hop data, and the lap timer started after the host's own work, so derived transport read 0 on every lap.

## Hidden-state transport (data channel)

Measured 2026-09-04 on the GB10: two headless Chromium 131 tabs on one machine, loopback shaped with netem to a 100 ms round trip (50 ms each way), one-way delay of one message, p50 over 10 samples, 1 s apart, RTCDataChannel ordered+reliable unless noted. Harness: two RTCPeerConnections over host candidates, sender stamps `performance.timeOrigin + now` in a 16-byte header.

| message | what it is | plain, one send | sliced ≤4.6 KB sends | striped over 5 associations |
|---|---|---|---|---|
| 1 KB | token id, ping | 51 ms | 51 | 52 |
| 5 KB | | 153 | 51 | 51 |
| 10 KB | one token's hidden state (5120 × f16) | 152 | 51 | 52 |
| 20 KB | | 254 | 52 (p90 152) | 52 |
| 30 KB | K=3 verify block | 255 | 52 (p90 155) | 51 |
| 50 KB | K=5 verify block | 355 | 152 | 52 (p90 153) |
| 70 KB | K=7 verify block | 458 | 152 | 52 (p90 153) |
| 164 KB | 16-token prefill chunk | 562 (p90 664) | 153 | — |

Reading: 51 ms is the physical one-way time. On a plain channel every ~4 packets beyond the first burst cost one more round trip (dcSCTP `max_burst` 4, initial cwnd 10 MTU): a single token paid 1.5 round trips per hop, a K=5 verify block 3.5, a K=7 block 4.5, a prefill chunk 5.5. Slicing every send under four packets removes the burst penalty and fixes everything up to ~30 KB; blocks above the initial window still pay one round trip on one association, and striping over five associations removes that too.

With 1 % packet loss on the same link (12 samples): plain 10 KB p50 153 / p90 356 ms, plain 164 KB p50 1373 / p90 1979 ms; sliced 10 KB p50 152 / p90 253, sliced 70 KB p50 359 / p90 771. Loss recovery costs a round trip per event on a reliable channel, so forward error correction on an unordered channel is the next lever (issue #34, step 3).

Room change: `room/transport.js`, a negotiated data channel per peer link that PeerJS never sees, slicing at 4,600 bytes and round-robin over `?wire=stripeN` associations (default `stripe4`); `?wire=off` restores PeerJS messages. Exact by construction: bytes only. Unit test `tests/unit/transport_test.js` (byte-exact reassembly under reordering and duplicates).

End-to-end on the GB10 (two headless Chromium tabs, real PeerJS signaling and WebRTC on loopback, Qwen3 0.6B Q8 split 18+10 layers): `?wire=off` 41.4 / 42.9 tok/s, `?wire=stripe4` 53.5 / 52.0 tok/s, 4 channels open per link, 127 frames sent = 127 received each way, no console errors. The loopback gain is the PeerJS serializer and its 16 KB chunking leaving the path; the round-trip gain needs a real network and is the table above.

27B in the emulator (GB10, `npm run e2e -- --phone --model qwen3.8-27b`, host 53 layers + embed/head, worker 9, phone-shaped tab 2, `japan` prompt, 400-token answers, weights served from local disk, loopback network): `--wire off` prefill 4.3 / 4.1 s, decode 10.3 / 9.9 tok/s; `--wire stripe4` prefill 4.5 / 4.2 s, decode 10.4 / 10.7 tok/s. Equal within noise on loopback, as expected; 459 frames per link each way. Both runs trip bug #31 (prompt + 400 tokens > 512 context): 1,741 GPU validation lines in the room log, which the emulator now reports.

Topology change (host link + on-demand chain links instead of a full mesh) and `--devices N` in the emulator, GB10, 27B, `japan` prompt, local signaling, loopback: 3 devices online in 3.0 min, prefill 4.5 s, decode 10.4 tok/s; 16 devices (8 phone-shaped, 64 layers dealt 18+embed / 4-5 per worker / 2 per phone) online in 2.3 min, prefill 8.3 / 7.2 s, decode 3.8 / 4.7 tok/s, every device holding one host link and two chain links, no errors. The decode drop on a zero-latency network is per-hop processing (unpack, upload, readback, pack), about 15 ms per hop, now a measured target. 64 tabs in one Chromium fail at `vkCreateDevice` (one GPU process, driver device cap); not a room limit.

## 2026-10-01: split-K decode attention (attnDecode "v2") on by default (branch perf/kf-long-context)

`engine/wgsl/attn_dec.js` replaces attn_flash + attn_combine for decode and verify passes. Its split length comes
from the column's own position (not maxSeq), it uses coalesced vec4 K/V loads, a 32-thread softmax, and (new) a
combine that is parallel over splits (MoE 0.83 -> 0.16 ms per token at 64K). Not bit-identical to v1, so the 27B
goldens are re-baselined; `?attndecode=v1` / `ATTN_DECODE=v1` keep the old path. Details:
docs/research/long-context-2026-09.md.

Plain decode tok/s, v1 -> v2 (2 runs each; same prompts per row):

| Path | Model | 1K | 8K | 32K | 64K |
|---|---|---|---|---|---|
| GB10 Chrome `chrome_bench.mjs` (fill, rep 2) | MoE | 31.8 -> 50.4 | 32.2 -> 48.5 | 24.5 -> 41.0 | 18.2 -> 35.5 |
| GB10 Chrome `chrome_bench.mjs` (mean of 2) | 27B | 10.20 -> 11.23 | 9.65 -> 10.81 | 7.65 -> 9.76 | n/a |
| GB10 Deno `bench_attn_ctx.js` (in-process A/B) | MoE | 23.3 -> 32.3 | 23.2 -> 31.4 | 20.1 -> 28.3 | 14.7 -> 25.7 |
| GB10 Deno `bench_attn_ctx.js` (in-process A/B) | 27B | 8.61 -> 10.00 | 8.28 -> 9.69 | 6.79 -> 8.87 | 5.47 -> 7.99 |
| GB10 Deno `bench_ctx.js` (mean of 2) | 27B | 8.48 -> 10.02 | 8.10 -> 9.67 | 6.60 -> 8.81 | 5.32 -> 7.89 |
| M5 Max Chrome `chrome_bench.mjs` (mean of 2) | MoE | 69.3 -> 88.0 | 68.2 -> 82.3 | 46.5 -> 68.1 | n/a |
| M5 Max Chrome `chrome_bench.mjs` (mean of 5) | 27B | 19.7 -> 21.0 | 18.7 -> 20.3 | n/a | n/a |

Attention GPU ms per token (Deno A/B): MoE 12.6 -> 0.58 (1K), 19.4 -> 4.9 (32K), 37.6 -> 8.7 (64K); 27B 17.6 -> 1.07
(1K), 48.5 -> 13.8 (32K), 82.8 -> 26.1 (64K). GB10 Chrome MoE rep 1 ran under GPU contention (v1 1K 15.6), rep 2 shown.

Correctness: greedy tokens identical v1 vs v2 in every Chrome run (both GPUs, all fills); `test_moe` MATCH llama.cpp
on all prompts, spec == plain; `needle_ctx.js` PASS (MoE 32K/64K, 27B 16K/32K); `test_attn_dec.js` relDiff vs f64
<= 2.4e-7, verify == decode; `test_attn_dec_model.js` logits relDiff v2 vs v1 <= 1e-5, argmax and 24 greedy equal.
New 27B goldens (GB10): default `BITS plain c26dbc5 hidden 3177f9f1`, `ATTN_PREFILL_TILE=0` f0537158 / 5d287854
(v1: 4cac59d8 / a67b7bcd and 4f70a9ca / 5eb28e41), same 13 tokens, spec == plain. `test_prefill_opts.js` MoE
fails its 0.02 tolerance at 700 tokens on main too (2.44e-2 main and v1, 2.45e-2 v2; argmax, greedy, spec equal):
pre-existing, not from this change.

## 2026-09-30: 128K context on the 35B MoE (branch perf/kv-128k), GB10 Deno

`cd tests && MODEL=moe CTX=131072 LENS=32k,64k,96k,127k DEPTHS=0.5,0.25,0.75,0.5 deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights needle_ctx.js`.
Needle in a haystack: a fresh prompt of this repo's docs and source with one sentence holding a random passphrase at the given depth, then a question; greedy answer. maxSeq 131072 (2.5 GB of f16 KV for the whole MoE). Prefill is the whole prompt from an empty cache; decode is 32 tokens with that prompt in the cache.

| Model | KV | Prompt | Needle depth | Needle | Prefill tok/s (time) | Decode tok/s |
|---|---|---|---|---|---|---|
| moe | f16 | 32.7K | 0.5 | found | 294.3 (1.9 min) | 15.89 |
| moe | f16 | 65.5K | 0.25 | found | 220.6 (4.9 min) | 14.43 |
| moe | f16 | 98.3K | 0.75 | found | 177.7 (9.2 min) | 12.49 |
| moe | f16 | 130K | 0.5 | found | 149.6 (14.5 min) | 9.39 |
| moe | q8 | 130K | 0.5 | found | 34.2 (63 min) | 9.87 |
| 27b | f16 | 65.5K | 0.5 | found | 52.9 (20.6 min) | 5.07 |

bench_ctx.js at CTX=131072, fills 32K / 64K / 96K / 127K (prefill tok/s is for the 32K tokens added to reach each fill; a headless Chromium job from another run shared the GPU for part of it, so read the speeds as a floor):

| Date | Model | Hardware | KV | Context | Prefill tok/s | Plain decode tok/s | Spec decode tok/s | Spec = plain |
|---|---|---|---|---|---|---|---|---|
| Sep 30 | moe | GB10 (Deno) | f16 | 32K | 134.2 | 11.2 | 15.7 (24/30) | yes |
| Sep 30 | moe | GB10 (Deno) | f16 | 64K | 77.2 | 10.02 | 10.69 (22/30) | yes |
| Sep 30 | moe | GB10 (Deno) | f16 | 96K | 57.4 | 9.19 | 6.26 (18/39) | yes |
| Sep 30 | moe | GB10 (Deno) | f16 | 127K | 43.8 | 6.55 | 5.64 (20/33) | yes |

- No NaN and no GPU error at any length; speculative decoding identical to plain at every fill.
- 128K works but a cold 128K prompt takes ~15 min to prefill on the GB10 (f16). It is useful for sessions that reuse
  checkpoints (pinned system prompt, per-turn checkpoints), not for pasting a 128K document into a fresh room.
- int8 KV at 128K: correct, same decode speed, 4.3x slower prefill (tiled prefill attention is off with int8).
- The 27B at 128K (f16: 256 MiB per K/V buffer, 8 GB of KV) was not run to the end: its 64K run took 20.6 min and the
  128K prefill was stopped to free the GPU. Its cap moves from 32K to 64K (128 MiB per buffer, fits every device).
- tests/test_moe.js (3/3 MATCH llama.cpp, spec == plain) and tests/run.sh quick pass on this branch.

## 2026-09-29: phone memory while loading (#207, branch fix/i207-memory)

The iPhone 14 Pro Max probe (memprobe.html, PR #244) found Safari's page process gets a 1536 MB soft limit (WebGPU buffers count against it), the networking process 840 MB, and that the page died on 3-layer MoE loads because the prefetcher re-fetched tensors the loader already had (~450 MB of unread bodies per MoE layer, piling up in the networking process). This branch: phone pledges capped (iPhone/iPad 0.5 GB default, 1 GB max; Android by `navigator.deviceMemory`), no duplicate prefetches (unread ones cancelled), ranges from room devices streamed with an 8 MB flow-control window instead of whole-range JS buffers, Q4_1/Q5_0/Q5_K/Q6_K requantized to Q8 a few rows at a time on the way to the GPU (bit-identical: `tests/test_stream_requant.js` on both models, 160 MB MoE expert tensors included), and a phone killed while loading gets a smaller share or is left out by an automatic re-deal.

| Test | Before (main) | After (branch) | Notes |
|---|---|---|---|
| Spark loopback room, 27B, host + worker + iPhone-UA tab joining late (`tests/e2e/room_phone_mem.mjs --no-kill`), phone dealt 2 layers from the worker's cache | renderer peak 546 / 505 MB; 568 MB fetched from the room | renderer peak **307 MB**; 423 MB fetched from the room | same split both runs (host 58 + embed, worker 4, phone 2); peak = resident size of the phone browser's renderer, sampled every 200 ms. Answers identical to host+worker. |
| Same, phone's layers from the network | | renderer 328 MB, network service 106 MB | |
| Same, phone tab killed while loading, twice (`room_phone_mem.mjs`) | host reloads the same layers into it | 1st kill: re-dealt with 0.22 GB for it; 2nd: re-dealt without it; room online both times, same answer | |
| Real iPhone 14 Pro Max in a room with the Spark (public signaling, branch preview, 1 GB pledge), one load each | | MoE 1 layer ×1, MoE 2 layers ×3, 27B 4 layers ×2: every load online in 119–146 s, no WebKit kill of the room's page in the phone's syslog (two `long-idle-exit` kills of earlier sessions' idle tabs) | with the cap a phone can't be dealt 3 MoE layers any more (a 22 GB host pledge is the smallest that starts the MoE; the phone then gets 2); decode 22–24 tok/s MoE, 7.4 tok/s 27B |

## 2026-09-26: Qwen3.6-35B-A3B MoE on real hardware (GB10), expert kernels rebuilt

File: bartowski `Qwen_Qwen3.6-35B-A3B-Q4_0.gguf` (shared experts Q5_0 → Q8 on load, routers BF16 → f32 exact).
Reference: llama.cpp b10840 CUDA on the same file: 85.5 tok/s decode, 2520 tok/s pp512 (27B: 13.8 decode).

Correctness: greedy output matches llama.cpp on three chat prompts (tests/test_moe.js); speculative decoding
identical to plain. Plain "The capital of France is" is a near tie after " Paris" ("." 19.029 vs "," 18.968 here;
llama.cpp CUDA picks ","), so it is not used as a golden.

Per-token GPU time (timestamp queries, tests/prof_ts.js), before → after rebuilding the MoE kernels on the
cooperative-GEMV layout (4 threads per block, vec4 x reuse across rows, unpack4x dequant) and a parallel router:

| kernel      | before (µs × 40) | after |
|-------------|------------------|-------|
| moe_gu_q4   | 288              | 71    |
| moe_dn_q4   | 169              | 43    |
| moe_router  | 165              | 32    |
| GPU total   | 41.7 ms          | 22.8 ms |

Decode tok/s, plain / speculative (K=3):

| | Deno (GB10) | Chrome (GB10) |
|---|---|---|
| MoE before | 17.8 / 9–20 | n/a |
| MoE after  | 27.0 / 15–27 | 32–40 / 52–59 |
| 27B (unchanged) | 9.7 / 15.9 | 9.4–10.8 / 18.7–22.6 |

Deno adds ~11.7 ms per GPU sync (an empty submit + 4-byte readback: 11.7 ms in Deno, 0.76 ms in Chrome), so
Chrome numbers (tests/bench/chrome_bench.mjs) are the ones to quote. CPU encode is 4.5 ms per MoE token (732 dispatches).

## 2026-09-26: combined optimizations (branch opt/combined), GB10

Merged onto work/tabby-gpu in this order: opt/spec-one-submit (one-submit speculative step, draft chain on),
opt/fuse-projections (merged projection GEMVs, bit-identical), opt/moe-fuse (fused router + shared expert +
combine), opt/moe-kernel-tuning (configurable unfused expert GEMV layout; left opt-in, legacy by default),
opt/drafts (batched draft-cache refill, pre-run first draft, draftvocab auto-fallback). With a fused verify the
refill reads the trunk hiddens in place from the batch buffer (no CPU copy); acceptance counts are identical
on every path (fused / separate verify, chain on / off).

Correctness: `tests/run.sh quick` and `tests/run.sh q38` pass with the same outputs, hidden values and acceptance
counts as the baseline (27B bits unchanged); `tests/test_moe.js` matches llama.cpp on all three prompts with
spec == plain, also with DRAFTCHAIN=0, SPECFUSE=0 and MOE_FUSE=0; e2e drafts_synth and draftchain_synth pass
on the real GPU (E2E_GPU=real). moe_synth's "batched prefill == one token" check still fails with 202 logits,
as it does on work/tabby-gpu (not caused by these merges, also with gemm off; open).

Chrome decode tok/s (tests/bench/chrome_bench.mjs, 40 tokens, K=3), plain / speculative. Baseline: today's
work/tabby-gpu numbers (one re-run in this session: MoE 33.77 / 59.17 and 38.86 / 49.79, inside the range).

| | two-sum | hash-map |
|---|---|---|
| MoE baseline | 33.4–33.8 / 57.4–59.1 | 37.5–38.9 / 50.5–51.2 |
| MoE combined (bench defaults, run 1, 2) | 40.48 / 74.33, 41.63 / 74.68 | 45.84 / 66.54, 45.19 / 64.73 |
| MoE combined, draftvocab=65536 (room default) | 44.73 / 82.82, 33.61* / 84.22 | 43.25 / 65.00, 44.77 / 64.82 |
| 27B baseline | 10.1–10.2 / 22.0–22.1 | 10.6–10.7 / 18.1–18.7 |
| 27B combined (bench defaults, run 1, 2) | 10.23 / 24.05, 10.06 / 24.13 | 10.69 / 20.39, 10.69 / 20.27 |
| 27B combined, draftvocab=65536 | 10.17 / 27.06, 10.08 / 26.78 | 10.81 / 21.07, 10.72 / 20.82 |

MoE: plain +20–24 % (two-sum) and +16–22 % (hash-map); speculative +26–30 % and +26–32 %. 27B: plain unchanged,
speculative +9 % and +8–13 %. Acceptance is unchanged (MoE 28/33, 28/39; 27B 28/33, 26/39). The draftvocab=65536
rows were not measured on the baseline, so they only compare against the combined defaults.
(*) one run with a stall in the first timed decode (see below).

MoE GPU time per token (tests/prof_ts.js, timestamp queries): 22.74 ms, 732 dispatches (baseline, same session)
→ 20.46–20.63 ms, 502 dispatches. Top kernels: matvec_q8_coop 3.08 ms (15), moe_gus_q4_q8 3.05 (40, 76 µs),
matvec_q4_coop 2.97 (36), moe_dnc_q4_q8 1.87 (35, 54 µs), dn_pre 1.32, dn_delta_gn 1.28, moe_route 1.25 (40),
matvec_coop 1.11 (70), rmsnorm 1.08 (81). Deno wall is sync-bound (~35 ms/token steady on both), but with
fuseProj + moeFuse together Deno shows a one-time 1.4–1.8 s stall a dozen tokens into decode (a V8 scavenge of
~1.5 s under external-memory pressure, `--trace-gc`), so prof_ts's 20-token wall average reads 92–128 ms. Neither
branch alone shows it; the cause is open.

## 2026-09-27: Emulated latency, 1 to 3 devices (branch bench/latency), GB10

What a room feels like when the devices are not on the same machine. Harness: `tests/e2e/room_latency.mjs`.
Each device is its own headless Chromium 131 (own profile, own GPU process, weight cache on disk), real PeerJS
signaling and real WebRTC over loopback, `?wire=stripe4`, all devices on the one GB10 GPU. Latency is added in
the page, not by netem (that needs root, which this machine does not give us): every activation frame a device
sends is held for the one-way delay before it goes on the wire, the same thing `?netlag=ms` does. The harness
serves room.js with two dev-only changes (the delay is read live so one loaded room can sweep, and a switch for
plain decoding); room.js itself is unchanged. Split is even by pledge (MoE 20+20 and 14+13+13 layers, 27B
32+32 and 22+21+21; the host also holds embed and head). `japan` prompt (172 tokens with the template), exact
(greedy) sampling, new chat before every answer so every answer prefills the whole prompt, 128-token answers,
two answers per cell, mean shown (the two were within 0.4 tok/s except one MoE 2-device spec pair, 34.6 / 30.7).

Decode tok/s, plain / speculative, and time to first token:

| Model | Devices | 0 ms | 5 ms | 20 ms | 50 ms | Time to first token |
|---|---|---|---|---|---|---|
| 35B MoE | 1 | 32.5 / 44.2 | | | | 1.04 s |
| 35B MoE | 2 | 27.9 / 32.7 | 21.1 / 27.9 | 12.9 / 20.6 | 7.2 / 13.5 | 1.10 s at 0 ms, 1.13 s at 50 ms |
| 35B MoE | 3 | 24.8 / 33.3 | 17.7 / 26.9 | 9.9 / 17.5 | 5.2 / 10.8 | 1.14 s at 0 ms, 1.14 s at 50 ms |
| 27B | 1 | 9.2 / 14.6 | | | | 2.57 s |
| 27B | 2 | 8.6 / 11.9 | 7.8 / 10.9 | 6.3 / 9.3 | 4.6 / 7.6 | 2.52 s at 0 ms, 2.53 s at 50 ms |
| 27B | 3 | 8.4 / 11.0 | 7.3 / 10.2 | 5.5 / 8.6 | 3.7 / 6.6 | 2.53 s at 0 ms, 2.55 s at 50 ms |

Latency is one way, per hop. A token goes host → worker(s) → host, so a room of N devices pays N hops per lap.

Draft acceptance: MoE 59 % on one device, 38–44 % on 2, 44–51 % on 3; 27B 47–48 % on one device, 35–38 % on
2 and 3. The room picks the draft depth by lap time (K=3 alone, deeper on a chain), so a chain drafts more and
accepts a smaller share of it; the tokens per lap still go up.

Reading:
- Plain decode is exactly compute plus hops times latency. MoE on 2 devices: 36 ms per token at 0 ms, 139 ms at
  50 ms (+103 = 2 × 50); on 3 devices 40 → 194 ms (+154 = 3 × 50). 27B on 2 devices 116 → 220 ms, on 3 119 →
  270 ms. Nothing else in the pipeline grows with latency.
- Speculative decoding is what keeps a room usable on a real network: at 20 ms it is 1.6x plain for the MoE on
  2 devices and 1.8x on 3; at 50 ms 1.9x and 2.1x. On 0 ms it only buys 1.2–1.4x, because every device here
  shares one GPU and a verify block costs real GPU time.
- The MoE on 2 devices at 5 ms (a good home Wi-Fi) is 21 plain / 28 speculative, and still 13 / 21 at 20 ms
  (same city). The 27B at 20 ms is 6.3 / 9.3, close to what it does alone on this machine plain.
- Time to first token hardly moves with latency (+0.03 s at 50 ms): prefill sends the prompt in 16-token frames
  with several in flight, so the network is paid about once per prompt, not once per frame. It is dominated by
  the prefill compute itself (172 tokens in ~1.0 s on the MoE, ~2.5 s on the 27B).
- Splitting costs a little even at 0 ms (MoE 32.5 → 27.9 → 24.8 plain): per hop the hidden state is read back,
  packed, sent, unpacked and uploaded, about 3.5–4 ms per hop here.

References on the same machine:
- Chrome, one device, `tests/bench/chrome_bench.mjs` (2026-09-26, combined build, 40 tokens, K=3, coding
  prompts): MoE 40–46 plain / 65–84 speculative, 27B 10.1–10.8 / 20–27. The room numbers above are lower on one
  device mainly because the `japan` itinerary drafts worse than the coding prompts (acceptance 47–59 % against
  70–85 %) and because a room answer includes the room's own per-token work (sampling, chat, telemetry).
- llama.cpp CUDA, same GGUFs: MoE 85.5 tok/s decode and 2520 tok/s pp512 (b10840), so about 0.07 s to prefill
  this prompt; 27B 13.8 decode (b10840), 377 tok/s pp86 (749f688), about 0.5 s for this prompt. One device,
  no network.

Caveats: loopback with an added delay is not a real link. No bandwidth limit, no loss, no jitter, and none of
the congestion-window round trips that big frames pay on a fresh link (see "Hidden-state transport" above; with
`stripe4` a single token and a K≤7 verify block fit the first window, so decode should be close, but prefill
frames of 164 KB would pay more on a real link than here). Control messages are not delayed. All devices share
one GPU, so compute on different devices does not overlap the way it would on separate machines, which makes the
0 ms rows pessimistic for a real room and the high-latency rows about right. Delay is applied with setTimeout in
the sending tab, so it is at least the stated value (timer slack of about 1 ms).

Notes from getting it to run: with every device as a tab of one Playwright context (the default
`browser.newContext()`), the context is off-the-record, the Cache API weight store lives in RAM, and loading the
35B MoE grew one browser process past 13 GB until Chromium aborted (SIGTRAP) about 6 GB in. One browser per
device with an on-disk profile fixed it. With the weight store disabled instead, the MoE load stalled at about
460 MB per device in this harness (cause not found). `tests/e2e/room.mjs` uses the one-context setup and would
likely hit the same crash on the MoE (it has no MoE entry in its local-weights map today).

## 2026-09-26: load time for tests and benches (CPU side; GPU not yet measured)

Loader CPU cost, with the GPU upload stubbed (`tests/bench/load_profile.js`): the 27B took 43.8 s (6.5 s reading, 37.2 s converting, 20.1 s of that the Q4_0 repack) and the MoE 46.0 s (8.4 s reading, 37.6 s converting). The repack is now 3x faster (u16 copies). With the new converted-weights cache (`tests/weight_cache.js`), a warm load is 4 to 6 s for the 27B and about 9 s for the MoE. The Chrome bench's static server read ranges at 0.27 GB/s; 8 MB reads bring that to 2 GB/s, and pre-converted tensors (`bench.html?wcache=1`) take the MoE tab load path from 74.7 s to 17 s on the CPU side. `tests/run_q38_once.js` runs the 27B suite over one upload. Details, and the commands still to run for GPU validation, are in [testing-fast.md](testing-fast.md).

## 2026-09-27: kernel pass on decode (27B) and the dense engine (Qwen3 1.7B), branch kopt/combined, GB10

Everything here is exact: the 27B's outputs are bit-identical to kopt/base (`tests/test_q38_bits.js`: same
logits hash 85b12667 and trunk-hidden hash eba0b8d5 over a 16-column prefill, 13 plain tokens and 3 spec
steps; `tests/run.sh q38once` passes; `tests/test_moe.js` 3x MATCH llama.cpp, spec == plain), and the dense
engine's logits are bit-identical to kopt/base at 2,600 tokens of context (`tests/test_dense_exact.js`,
hashes 445aa938 / 6aef0553 on both branches; `tests/run.sh quick` passes). Changes that were not exact are
off by default (the one-kernel dense glue).

What changed (branches, each from kopt/base = feat/engine-opt + opt/load-cache):
- `kopt/dense-attn` dense attention that is not latency-bound: `attn_scores_d` (G heads per thread share
  each K row), `attn_softmax_d` (parallel max and exp, then ONE thread adds the exponentials in position
  order with 8 loads in flight), `attn_out_d` (the per-(head, dim) chains read V / p tiles staged in shared
  memory by the whole workgroup, next tile prefetched into registers). Same operations in the same order
  as attn_scores / attn_softmax (a one-thread-per-head kernel before) / attn_out. Switch: `attnFast`.
- `kopt/dense-fuse` (on dense-attn): residual adds folded into the o / down GEMVs (`_acc`), one rmsnorm
  dispatch for all batch columns, the qk-norm + rope + K/V cache writes as three reference-shaped
  multi-column kernels (`head_norm_dmc`, `rope_dmc`, `kv_store_d`): no cache copies, one compute pass per
  layer. A single fused glue kernel was measured too (`fuseGlue`, off): the same rope expression compiles
  to differently rounded code inside a bigger kernel on NVIDIA Vulkan, so it is not exact.
- `kopt/dense-qkv` (on dense-fuse): q, k, v from one GEMV over the row-concatenated weights (`mergeQKV`).
  Dawn (Chrome) rejects two writable bindings of one buffer in a dispatch where wgpu (Deno) does not, so
  the glue kernels bind the merged buffer once; the Chrome bench now logs uncaptured GPU errors.
- `kopt/encode-ahead` both engines: while the GPU runs token N, the command buffer of position N + 1 is
  recorded (a token's commands depend only on the position and the switches); logits copy in the same
  submit. Switch: `encodeAhead`.
- `kopt/rmsnorm` rmsnorm and the dn_pre q/k L2 norms with 4 / 8 loads in flight, same in-order sums.
- `kopt/head-rows` LM head GEMV with 8 rows per workgroup (rows per workgroup never enter a row's
  arithmetic): on for the dense engine, off for the hybrid (no gain on the 27B head).
- `kopt/wide-loads` opt/wide-loads (16-byte weight loads) re-based and GPU-validated: bit-identical in the
  model (COOPWIDE=4,2 gives the same 27B hashes) but no faster (GPU 90.0 -> 91.0 ms/token), not merged.

### Qwen3 1.7B Q8 (dense engine), before (kopt/base) -> after (kopt/combined)

Chrome (tests/bench/chrome_bench.mjs, `prefill=512,4096`; plain decode = 40 tokens after a short chat
prompt; the dense model has no draft head, so no speculative number):

| | base run 1 | base run 2 | combined run 1 | combined run 2 | |
|---|---|---|---|---|---|
| decode, short chat (two-sum / hash-map) | 52.6 / 52.6 | 51.3 / 51.1 | 61.1 / 63.7 | 58.7 / 61.8 | +17% |
| prefill to 512 tokens | 83.9 | 84.2 | 261.1 | 259.4 | 3.1x |
| prefill to 4096 tokens | 22.2 | 22.3 | 176.3 | 176.6 | 7.9x |
| decode at 512 | 35.4 | 35.1 | 58.4 | 58.6 | +66% |
| decode at 4096 | 10.45 | 10.43 | 32.9 | 32.8 | 3.1x |

Deno (tests/bench_dense.js, FILLS=512,4096): prefill 82.2 / 82.4 -> 219.4 / 221.2 at 512 and 19.2 / 19.3 ->
158.9 / 159.6 at 4096; decode 22.9 / 24.0 -> 36.5 / 36.7 at 512 and 7.49 / 7.42 -> 25.3 / 24.9 at 4096.
Per token at 4k context (tests/prof_dense.js): 219 ms -> 41 ms wall; the old softmax was one thread per
head walking 4k positions three times in global memory.

### Qwen 3.8 27B Q4_0, before (kopt/base) -> after (kopt/combined)

Chrome (`batchcols=16&prefill=512,4096`, 40 tokens, K=3):

| | base run 1 | base run 2 | combined run 1 | combined run 2 | combined run 3 |
|---|---|---|---|---|---|
| plain (two-sum / hash-map) | 10.62 / 10.56 | 10.56 / 10.48 | 11.11 / 11.09 | 11.03 / 11.11 | 11.11 / 11.08 |
| spec K=3 (two-sum / hash-map) | 24.01 / 20.19 | 23.69 / 20.12 | 23.99 / 20.12 | 23.96 / 20.28 | 24.08 / 20.30 |
| prefill to 512 / 4096 | 72.7 / 66.4 | 72.5 / 66.3 | 72.7 / 66.1 | 73.0 / 66.3 | 73.0 / 66.3 |
| decode at 4096 | 9.46 | 9.42 | 9.90 | 9.84 | 9.89 |

Plain decode +5%, speculative and prefill unchanged (the verify pass and prefill use the batched kernels,
which this pass did not touch; prefill is the prefill workflow's). Deno (tests/bench_ctx.js MODEL=27b,
FILLS=512,4096): plain 8.77 / 8.80 -> 9.28 / 9.29 at 512 and 8.59 / 8.60 -> 9.08 / 9.08 at 4096; spec at 512
16.98 / 16.97 -> 16.92 / 16.93; prefill 66.2 / 65.9 -> 66.0 / 66.0. (The spec number at 4096 is not
comparable: bench_ctx builds its prompt from engine/qwen35.js, whose text changed, and acceptance went
17/42 -> 14/51.) tests/prof_ts.js: wall 106.1 -> 101.3 ms/token from encode-ahead (GPU time unchanged),
then GPU 90.9 -> 88.7 ms from rmsnorm (3.18 -> 1.64 ms, 129 dispatches) and dn_pre (2.10 -> 1.35 ms).

Where the 27B token still goes (GPU 88.7 ms): the GEMVs are ~79 ms at 200-215 GB/s (gate/up 469 us x 64,
LM head 5.8 ms at 228 GB/s); exact knobs tried per shape (tests/bench_wide.js: 16-byte loads, 1-16 rows
per workgroup) are all bit-identical and within +-5% of today's kernel, so the remaining gap to llama.cpp
(13.8 tok/s) is load efficiency that an exact kernel cannot reorder its way out of.

## 2026-09-27: tiled prefill attention (attnPrefillTile, candidate E), on by default for dense models (MoE opt-in since prefill/combined)

`attn_flash_tile` (engine/wgsl/attn_tile.js) replaces attn_flash / attn_flash_t2 on full-width prefill passes: one
workgroup per (split, KV head, 64 query rows) shares each K/V tile across the whole pass. GB10, Deno,
`bench_ctx.js` CTX=16640 TOKENS=8, prefill tok/s over the segment ending at each fill, off → on:

| | 512 | 4096 | 16384 |
|---|---|---|---|
| 27B | 65.9 → 69.8 | 57.4 → 74.3 | 30.2 → 66.1 |
| MoE | 151.3 → 159.7 | 133.3 → 166.0 | 80.8 → 117.2 |

Whole 16000-token prompt: 27B 39.7 → 69.1, MoE 99.5 → 153.9. Attention kernel time at 4096 (`prof_prefill.js`):
27B 14.5 → 2.2 s, MoE 4.8 → 1.0 s. Decode never uses the kernel. Every golden passes with it on; logit relDiff vs
attn_flash and the MoE router caveat are in docs/research/prefill-profile-2026-09.md (candidate E).
`ATTN_PREFILL_TILE=0` / `?attnptile=0` restores attn_flash.

## 2026-09-27 · MoE expert-grouped prefill (candidate D, branch prefill/moe-group), GB10 Deno

- Exact grouped kernels (`moeGroupPrefill: 256`): bit-identical on GPU (logits, greedy, spec, draft acceptance) but
  slower than per-pass at every chunk size: 2048 tok 160 -> 147/151/117/111 tok/s (UC 2/4/8/16). Each pair still does
  its own 256-lane reduction; a collapsed exact tree (2 barriers) was slower still. Verdict: drop as a speed path.
- Tiled grouped kernels (`moeGroupPrefill: 256, moeGroupTiled: true`, UC 8): experts 3x faster (4096 tok: gate/up
  6733 -> 2155 ms, down 4547 -> 1385 ms, sort 337 ms). bench_ctx prefill tok/s off -> on: 512 151.8 -> 208.5,
  4096 140.3 -> 193.1, 16384 89.1 -> 109.3; decode unchanged (24.6/22.3/20.6 vs 24.1/23.5/21.2), same greedy text,
  spec identical. test_moe.js (MOEGROUP=16 MOEGROUP_TILED=1): 3/3 MATCH llama.cpp, spec == plain.
  Next-token logits relDiff vs per-pass: 5.3e-4 (300 tok), 1.7e-2 (700 tok; the per-pass path itself is 2.2e-3 off
  token-by-token there, both argmax-equal): routing flips compound, so it stays off by default.

## 2026-09-27 · prefill/combined: E + B + D + C merged, vs prefill/base (GB10)

Merged into `prefill/combined`: flash-attn (E), moe-group (D), gemm-tiles (B), f16-subgroup (C). Not merged:
deltanet-chunked (2.6x slower kernel, verdict drop) and prebaked-state (a TTFT feature, not prefill throughput; its
reviews found room-path bugs, a restored hit can drop its own staged slot after any DROP_ALL and a worker can persist
a stale slot under the prefix key, and its v2 state signature silently invalidates saved sessions on the default path).

**Defaults.** Only the tiled prefill attention is on by default, and only for dense models: on the MoE it lands
4e-3..1.6e-2 from attn_flash (over the 2e-3 prefill tolerance), so it is opt-in there. It no longer runs on
speculative verify passes at any batchCols (review must-fix: gate is now `nCols === NC && !_snapNow`). Wide GEMM
(`prefillUbatch`), expert-grouped MoE (`moeGroupPrefill`, tiled kernels by default when on) and `prefillMath` stay
off. With wide + grouped both on, each wide chunk's MoE layers run the grouped expert kernels over the whole chunk.

**Goldens on the combined branch (defaults):** `run.sh quick` 8/8, `run.sh q38` 9/9 (MATCH, test_batch_q38 relDiff
1.74e-7, twins bit-identical, spec == plain, GEMM worst 1.61e-6), `test_moe.js` 3/3 MATCH llama.cpp, spec identical,
acceptance 28/33, 28/39, 25/45.

**All options on vs all off** (`tests/test_prefill_opts.js`, SEQ_ALL=1; argmax, 24 greedy tokens and spec == plain
identical at every length):

| | 150 | 700 | 2100 |
|---|---|---|---|
| 27B relDiff on vs off (vs one-at-a-time: off / on) | 2.7e-5 (2.9e-5 / 1.9e-5) | 9.0e-5 (6.4e-5 / 1.2e-4) | 7.3e-5 (5.5e-5 / 9.0e-5) |
| MoE relDiff on vs off (vs one-at-a-time: off / on) | 2.7e-5 (1.4e-5 / 2.5e-5) | 6.7e-4 (1.2e-4 / 6.7e-4) | 1.5e-3 (6.3e-4 / 1.7e-3) |

On this prompt the MoE all-on path stays under 2e-3, but D and E alone exceeded it on other prompts/lengths (up to
1.7e-2), so the MoE options stay off until a MoE tolerance is decided.

**bench_ctx**, Deno, CTX=16640 TOKENS=32, same tokens on both branches (`CTX_SRC` = prefill/base checkout), one run
each, GPU idle (waited for other jobs). Prefill tok/s over the segment ending at each fill; decode plain / spec tok/s.
"all opts" = 27B `PREFILL_UBATCH=256`; MoE `ATTN_PREFILL_TILE=1 PREFILL_UBATCH=256 MOEGROUP=256`.

| | prefill 512 | 4096 | 16384 | plain decode 512 / 4k / 16k | spec decode 512 / 4k / 16k |
|---|---|---|---|---|---|
| 27B prefill/base | 65.9 | 59.0 | 34.9 | 8.78 / 8.55 / 7.58 | 16.92 / 11.27 / 12.70 |
| 27B combined default | 70.0 | 74.4 | 66.1 | 8.80 / 8.54 / 7.60 | 16.91 / 11.35 / 12.81 |
| 27B combined all opts | 85.9 | 94.5 | 78.4 | 8.77 / 8.50 / 7.59 | 16.78 / 10.53 / 12.75 |
| MoE prefill/base | 152.9 | 139.2 | 88.9 | 22.67 / 22.39 / 20.47 | 31.80 / 39.24 / 28.42 |
| MoE combined default | 153.2 | 139.5 | 88.9 | 24.20 / 22.39 / 21.84 | 33.65 / 39.51 / 28.86 |
| MoE combined all opts | 300.2 | 370.6 | 296.4 | 23.93 / 23.77 / 21.62 | 32.28 / 41.66 / 29.50 |

Spec output identical to plain at every fill in every run. The 27B all-opts spec dip at 4k (10.53) comes with
16/45 accepted drafts instead of 17/42 (the draft cache is filled from the wide prefill's hiddens); plain decode is
unchanged. llama.cpp CUDA (b749f688, -fa 1) for scale: 27B pp512 879 / pp4096 893 / pp16384 847, MoE 2356 / 2374 /
2271. Gap now: 27B 12.6x / 12.0x / 12.8x by default (10.2x / 9.4x / 10.8x all opts); MoE 7.8x / 6.4x / 7.7x all opts.

**Chrome** (`chrome_bench.mjs`, batchcols=16, 40 tokens, Chromium 131; decode plain / spec on two-sum, hash-map):

| | 27B plain | 27B spec | MoE plain | MoE spec | prefill 2048 tok |
|---|---|---|---|---|---|
| prefill/base | 10.61 / 10.49 | 23.99 / 20.15 | 43.48 / 44.07 | 73.31 / 64.75 | not measured (no ?prefill on base) |
| combined default | 10.63 / 10.62 | 24.08 / 20.14 | 42.20 / 43.83 | 71.98 / 64.79 | 27B 81.0, MoE 165.3 |
| combined all opts | 10.52 / 10.56 | 23.99 / 20.23 | 44.69 / 42.53 | 73.18 / 63.95 | 27B 102.0, MoE 432.9 |

Decode does not regress (all within about 3%, single runs; acceptance identical: 28/33, 26/39 and 28/33, 28/39).
Logs: scratchpad `pc/` (bc_*, cr_*, opts_*, g_*).

## 2026-09-27: test_q38_bits fingerprint drift after the merge (branch integ/kernels), GB10

After merging research/overnight and kopt/combined, `tests/test_q38_bits.js` with the tiled prefill attention off
gave `BITS plain 7ab40f4 hidden bbe1f08c` instead of kopt/combined's `85b12667 / eba0b8d5`. It was not a kernel
change. The test built its prompt from the first 300 tokens of `engine/gguf.js`, read live from disk, and the audit
commit b3f50e1 (comment-only) rewrote the header comment of that file. So the input changed, not the math. Bisect:
kopt/combined's `engine/wgsl` alone did not move the hash, its whole `engine/` did (it carries the old gguf.js),
and `WEIGHT_CACHE=0` and kopt's qwen35.js alone did not. gguf.js differs between the two only in comments.

Fix: the prompt is now a frozen fixture, `tests/golden/q38_bits_prompt.txt` (the first 150 lines of gguf.js as of
kopt/combined), and the test reads that. Same prompt everywhere, same bits:

| Branch | ATTN_PREFILL_TILE | BITS plain | hidden | spec vs plain |
|---|---|---|---|---|
| feat/engine-opt (f615846) | n/a (no tile) | 85b12667 | eba0b8d5 | == plain |
| kopt/combined | n/a | 85b12667 | eba0b8d5 | == plain |
| integ/kernels | 0 | 85b12667 | eba0b8d5 | == plain |
| integ/kernels | default (on for dense) | 8a532ef5 | 52f2ae10 | == plain, same 13 tokens |

The only deviation is the tiled prefill attention (prefill summation order), which is the accepted one.

## 2026-09-27: MoE prefill options on by default (branch integ/kernels), GB10

The Qwen 3.6 35B-A3B now gets the tiled prefill attention (`attnPrefillTile`), the wide prefill GEMM
(`prefillUbatch` 256) and the expert-grouped tiled FFN (`moeGroupPrefill` 256) by default. Wide and grouped run only
on an engine that holds the embedding (solo `prefillTokens`); a room's split prefill keeps its 16-column frames, and
workers turn both off without a warning. An option that cannot be built on a device turns itself off. Dense defaults are
unchanged: tiled attention on, wide GEMM opt-in. `false` / `0` still turns each one off (`ATTN_PREFILL_TILE=0`,
`PREFILL_UBATCH=0`, `MOEGROUP=0`; Chrome `?attnptile=0&ubatch=0&moegroup=0`).

MoE prefill tolerance in the tests is now 2e-2 (`tests/load_model.js` `prefillTol`; dense stays 2e-3). The baseline
is the MoE's own 16-column batched prefill: with the old kernels it is already 2e-3..2.3e-2 from token-by-token decode
in Deno, and 0.16..0.17 in Chrome on the bench page's HTML. The options sit inside that band.

Prefill tok/s, options off -> on:

| Where | 512 | 700 | 2048/2100 | 4k | 16k |
|---|---|---|---|---|---|
| Deno `bench_ctx.js` (fills 512 / 4096 / 16384) | 142 -> 263 | | | 140 -> 372 | 89 -> 306 |
| Deno `test_prefill_opts.js` | | 165 -> 367 | 160 -> 420 | 149 -> 411 | |
| Chrome `chrome_bench.mjs ...&prefillall=1` | | 171 -> 367 | 168 -> 402 | | |

Decode does not change (plain 26.7 / spec 33.7..37.4 at 512..4k, same acceptance and spec == plain both ways).

Checks, options on:

- `test_moe.js`: MATCH llama.cpp 3/3, spec == plain.
- `test_moe_split.js`: host and worker both use the engine defaults, as room.js does (room.js passes no prefill
  option). Solo, split plain == solo and spec == split plain on 5 prompts, including the 3674-token Code-mode tool
  prompt. 0 GPU errors. Same result with `OPTS=0`. On that tool prompt the greedy text with the options on differs
  from the text with them off at one token (`state.ines` vs `state.lines`, in a synthetic prompt). Solo and split
  agree within each setting.
- `test_prefill_opts.js` MoE, relDiff on vs off at 150 / 700 / 2100 / 4000: 3.3e-5 / 1.5e-3 / 1.6e-3 / 1.4e-2.
  Argmax is equal, greedy 24 is identical, and spec == plain. With `PROMPT_FILE=bench/bench.html` (raw HTML) at 700 /
  2048 it is 3.4e-3 / 3.0e-3, while all-off is 2.3e-2 / 2.3e-3 from token-by-token. 27B: 7.3e-5, pass.
- Phone / Mac-class device (`LIMITS=default`: WebGPU default limits, so 16 KB workgroup memory, 256 invocations and 8
  storage buffers per stage). Full MoE with adapter buffer sizes: all three on (attention TK 8, 64x64 wide tile in
  16 KB), pass, 0 GPU errors. Layers [0,10) + embedding with 256 MiB buffers and bindings (a phone in room.js): pass,
  hidden relDiff 2e-6, 0 GPU errors. The 128 MiB default binding cannot hold the MoE's 151 MB expert tensors with the
  options on or off, so the MoE cannot run at that limit either way.
- Chrome (`chrome_bench.mjs`, 16 KB workgroup memory, tile smem 16384): 0 GPU errors, golden two-sum / hash-map,
  spec == plain. relDiff on vs off at 2048 tokens is 5.7e-2 with argmax equal, but the old batched path is itself
  0.17 from token-by-token there. At 700: 2.3e-3, and both batched paths are 0.16 from token-by-token, with a
  different argmax than token-by-token (198 vs 19455). That gap already exists with the options off. It is worth
  its own look (Chrome's per-token path vs its batched path).
- Code mode end to end (`tests/eval/run.mjs --model engine`, MoE, calculator / fix-bug / logic / todo). Defaults and
  `--engine-opts '{"attnPrefillTile":false,"moeGroupPrefill":0,"prefillUbatch":0}'` give the same 2/4 with the same
  failures (calculator's "12 + 719" check and todo's missing #new). fix-bug is identical token for token, 0 GPU
  errors, 100 s vs 169 s.
- Unchanged: `test_q38_bits.js` `ATTN_PREFILL_TILE=0` -> 85b12667 / eba0b8d5, `run.sh q38once` and `quick` pass,
  `deno test tests/unit` 210 passed, `npm run check`.

## 2026-09-27: GPU sampling on by default (branch release/v1-exp, exp/gpu-sample on v1.0.0), GB10

GPU argmax / top-k (`topk_a` / `topk_b`, docs/research/exp-gpu-sample.md) in the head's submit: 16 B back per
greedy token (8k + 8 B for top-k) instead of the 1 MB logits vector, and the draft chain's argmax as the
two-stage multi-workgroup kernel. Merged onto v1.0.0's kernels (encode-ahead, head rows, MoE prefill options);
`forwardTokenIds` now rides encode-ahead like `forwardToken`. Off: `?gpusample=0` (room, bench) or `GPU_SAMPLE=0`
(Deno tests). Same tokens: sampling never touches the logits.

Chrome, one device (`tests/bench/chrome_bench.mjs`, 40 tokens, K=3, greedy, `?gpusample=0&argmaxwide=0` vs
`?gpusample=1`, same build, off/on alternated), decode tok/s plain / speculative:

| Model | Prompt | off | on | change |
|---|---|---|---|---|
| 35B MoE | two-sum (run 1 / run 2) | 47.7 / 72.7, 47.0 / 71.6 | 50.5 / 81.5, 50.4 / 79.6 | plain +6 %, spec +12 % |
| 35B MoE | hash map (run 1 / run 2) | 48.9 / 65.5, 47.8 / 64.7 | 50.4 / 71.4, 50.8 / 71.6 | plain +5 %, spec +10 % |
| 27B | two-sum | 11.07 / 24.3 | 11.20 / 25.0 | plain +1 %, spec +3 % |
| 27B | hash map | 11.06 / 20.3 | 11.13 / 21.2 | plain +1 %, spec +4 % |

Acceptance identical (MoE 28/33, 28/39; 27B 28/33, 26/39), llama.cpp golden text on the MoE, spec == plain,
gpuErrors 0 in every run.

Room (`tests/e2e/room_latency.mjs --lat 0 --maxnew 128`, `japan` prompt, exact sampling, one Chromium per device,
two answers per cell, mean), 35B MoE, `--query gpusample=0` vs `gpusample=1`:

| Devices | off plain / spec | on plain / spec |
|---|---|---|
| 1 | 33.9 / 44.5 | 36.3 / 48.7 (+7 % / +9 %) |
| 2 | 28.4 / 32.7 | 28.4 / 34.6 (0 / +6 %) |

Correctness on this branch: `test_q38_bits.js` (ATTN_PREFILL_TILE=0) BITS plain 85b12667 hidden eba0b8d5, and
GPU-sampled greedy (plain and speculative) gives the same 13 tokens as the logits path; `run.sh quick` (with
`test_selftest.js`) and `q38once`; `test_moe.js` MATCH llama.cpp on 3 prompts, spec == plain, head check 0/256
greedy and 0/32 top-40 mismatches; `test_moe_split.js`; `test_prefill_opts.js` 27B and MoE; `topk_kernel.mjs`
(251 cases) and `gpusample_synth.mjs`; `room_synth.mjs --compare` at 2 and 3 devices (solo == split).

`test_prefill_opts.js` now summarizes a frozen fixture (`tests/golden/prefill_opts_prompt.txt`) instead of the
live `engine/qwen35.js`: editing the engine changed the 700-token MoE prompt and its all-on vs all-off relDiff
(1.5e-3 -> 3.0e-2, over the 2e-2 gate) with identical kernels (same prompt file, both builds: 8.25e-3 and 2.53e-3
on each). With the fixture the numbers equal v1.0.0's (1.48e-3 / 1.27e-3).

Not in v1: `exp/moe-fused-layout` (its "tuned" preset measured +5 % speculative in Chrome on the old base, still
flag-off with no keep verdict), `exp/chain-fuse`, `exp/tail-head` and `exp/k-probe` had no GPU validation verdict
by the cut-off, and each conflicts with v1 in `engine/qwen35.js` / `room.js` / the test harnesses. `exp/one-sync-hop`
(no speedup; its one-submit readback overlaps v1's encode-ahead, conflicts in 5 files) and `exp/wire-rtt` (nothing
to gain) are left out too.

## 2026-09-27: first Apple M5 Max results (Mac Studio, Metal), main at cef5cd3

Mac Studio, Apple M5 Max (32-core GPU, Metal 4), 36 GB unified memory, macOS 27.0. Deno 2.9.7 (wgpu on Metal) for
the Deno tests; Chrome 154 headless over SSH for the browser runs. Headless Chrome gets WebGPU over SSH without a GUI
session (`--headless=new --enable-unsafe-webgpu`, adapter `apple` / `metal-3`, 4 GB buffers and bindings, 32 KB
workgroup memory, `shader-f16`, `subgroups`, `chromium-experimental-subgroup-matrix`). Models copied from the GB10.

### Correctness

| Check | Result on the M5 Max |
|---|---|
| `test_selftest.js`, `deno test tests/unit` (228) | pass |
| `run.sh quick` (Qwen3 0.6B / SmolLM goldens, stream, batch, reset, splits) | all pass; Qwen3 0.6B 45.4 tok/s, batched prefill 209.8 tok/s |
| `run.sh q38` (27B, one process per file) | **all 9 pass** (test_q38, batch_q38, mtp, b4, twins, gemm, q38_split, mtp_split, ctx) |
| `run.sh q38once` (27B, one process, shared upload) | **4 fail** (test_q38, test_b4, test_q38_split, test_ctx): every logit NaN. Test-runner problem, see below |
| `test_q38_bits.js` `ATTN_PREFILL_TILE=0` | BITS plain **b72e4d1f** hidden **ac403b4e** (GB10: 85b12667 / eba0b8d5, different GPU, different sums); spec == plain, text `\nexport const TENSOR_BYTES = {\n  [GGML` |
| `test_moe.js`, engine defaults | **FAIL**: garbage text on all 3 prompts, spec != plain. Cause: the tiled expert-grouped prefill kernel, see below |
| `test_moe.js`, `MOEGROUP=0`, `MOEGROUP_TILED=0` or `MOE_FUSE=0` | MATCH llama.cpp 3/3, spec == plain |
| `test_moe.js`, tiled kernel with the scratch fix (below) | MATCH llama.cpp 3/3, spec == plain; plain 36.4 tok/s, spec 75.0 / 67.5 / 57.1 |
| `test_moe_split.js` `SOLO=0` (scratch fix) | pass: split spec == split plain on 5 prompts incl. the tool prompt, checkpoint after rollback ok, 0 GPU errors (solo skipped: two 21 GB engines do not fit in 36 GB) |
| `test_prefill_opts.js` MoE, defaults | 150 tokens: relDiff on vs off **4.35e-1**, greedy 24 differs, spec after all-on prefill != plain; then `OperationError: validation error occurred` at `forwardToken` `mapAsync` in the 700-token case |
| `test_prefill_opts.js` MoE, scratch fix, `PREFILL_UBATCH=0` | 150 tokens: relDiff 2.2e-5, argmax equal, greedy 24 identical, spec == plain; the same `mapAsync` validation error at 700 |
| `bench_ctx.js` MoE, wide prefill and grouped prefill both on | the same `mapAsync` validation error at the first fill (either one alone: no error). Deno does not report the underlying validation message; Chrome runs the same config with 0 GPU errors |
| `bench_ctx.js` 27B at 16384 | **spec != plain** (acceptance 10/63, spec 8.4 tok/s vs plain 12.2); at 512 and 4096 spec == plain |

**MoE: tiled expert-grouped prefill is wrong on Apple GPUs, in Chrome too.** `moeGroupPrefill` with
`moeGroupTiled` (the MoE default since integ/kernels) gives garbage from the first token. In Chrome 154 on the
same Mac, `chrome_bench.mjs` with defaults also fails (golden false, spec != plain, 0 GPU errors), and
`&moegrouptiled=0` passes. So a Mac solo MoE or a Mac room host gets garbage on pooled.run. A Mac worker in a
room does not run the grouped kernels, since they only run on the engine that holds the embedding. The A/B points
at `tiledKernel` in `engine/wgsl/moe_group.js`: its reduction scratch `var<workgroup> xt: array<vec4<f32>>` is
written one component per thread (`xt[ri >> 2u][ri & 3u] = a...`, threads t..t+3 write the 4 components of one
vec4). On Metal that store appears to become a read-modify-write of the whole vec4, so neighbouring threads' writes
are lost. A scratch copy that declared `xt` as `array<f32, 4 * XT>` (vec4 loads built from 4 scalars) passed
`test_moe`, `test_moe_split` and the 150-token prefill check above. It is the only component-wise write into a
workgroup vec4 array in `engine/wgsl`. The fix is not in this PR.

**27B one-process runner (`run_q38_once.js`) loses its f32 weights on Metal.** `sharedQ38Context` →
`preuploadWeights` creates every f32 tensor with `mappedAtCreation` and writes the Q4/Q8 matrices with
`writeBuffer`, 15.14 GB in all (64 layers + head), with no GPU work in between. Read back afterwards, all 448 f32
buffers (norms, `wBeta`, `wAlpha`, `dtBias`, ...) are zero, even layer 0's. At 32 layers + head (8.4 GB) or 64
layers without the head (13.9 GB) they are intact, and so are they at 64 layers when each layer's upload is
followed by a readback. `onSubmittedWorkDone` after each layer does not help. Every per-file 27B test passes, so
this is a runner (or wgpu Metal) issue, not an engine one. Two gates did not catch it: `test_batch_q38`
("relDiff 0.00e+0", argmax 0 == 0) and `test_twins` ("identical") both pass when every logit is NaN.

### Speed

Deno (`bench_ctx.js`, fills 512 / 4096 / 16384, 32 tokens; the 512 prefill includes first-use shader compiles):

| | prefill 512 / 4k / 16k | plain decode 512 / 4k / 16k | spec decode 512 / 4k / 16k |
|---|---|---|---|
| 27B, defaults | 27.1 / 58.4 / 53.6 | 14.93 / 14.67 / 12.15 | 21.66 / 22.24 / 8.36 (16k: spec != plain) |
| MoE, scratch fix, `PREFILL_UBATCH=0` | 44.9 / 218.7 / 194.1 | 32.63 / 27.78 / 28.66 | 46.91 / 49.68 / 43.95 |
| GB10 for scale (above, combined default) | 27B 70.0 / 74.4 / 66.1, MoE 153 / 140 / 89 | 27B 8.8 / 8.5 / 7.6, MoE 24.2 / 22.4 / 21.8 | 27B 16.9 / 11.4 / 12.8, MoE 33.7 / 39.5 / 28.9 |

Decode is 1.3..1.7x the GB10's. The M5 Max has about twice the GB10's memory bandwidth. Prefill is behind on the
27B. `test_mtp` 27B: plain 15.8, spec 34.9 tok/s (85%). `test_moe` MoE: plain 35-36 tok/s.

Chrome 154 (`chrome_bench.mjs`, MoE, 40 tokens, `&moegrouptiled=0` so the output is right): plain **80.5 / 83.5**,
spec **127.6 / 115.3** tok/s (two-sum / hash-map, acceptance 28/33 and 28/39), golden, spec == plain, 0 GPU errors.
Decode in Chrome is 2.3x Deno's on the same GPU. Prefill at 2048 tokens (`prefilllen=2048&prefillall=1`): all off
178.5, all on 210.4 tok/s, relDiff 0.217 with argmax 713 vs 460. With the tiled kernel (defaults) it is 258 tok/s
but relDiff 1.02.

### First cross-machine room: GB10 + M5 Max

GB10 (headless Chromium 131, Vulkan) hosts, and the M5 Max (headless Chrome 154, Metal) joins over Tailscale, both
on the same LAN. Each machine serves its own checkout of cef5cd3 and loads its layers from its own disk
(`peerweights=0`). Signaling is a PeerJS server on the GB10, and the WebRTC link is direct (mDNS host-candidate
hiding off). The harness is a two-machine version of `room_latency.mjs` (one browser per machine, no emulated lag).
Qwen3.6 35B MoE, split by pledge: GB10 layers 1-20 + embed/head, M5 Max layers 21-40. Online 109 s after Start. The
prompt is `japan` (172 tokens with the template), exact sampling, a new chat per answer, 128 tokens.

| Mode | Answer | Prefill | Decode | TTFT | Ping RTT |
|---|---|---|---|---|---|
| plain | 1 | 172 tok in 1.2 s | 22.3 tok/s | 55 ms | 6 ms |
| plain | 2 | 172 tok in 0.5 s | 26.2 tok/s | 5 ms | 7 ms |
| spec | 1 | 172 tok in 0.5 s | 31.7 tok/s (43% accepted) | 7 ms | 10 ms |
| spec | 2 | 172 tok in 0.5 s | 27.2 tok/s (40% accepted) | 5 ms | 6 ms |

All four answers start with the same text (greedy). At 26 tok/s plain, one lap (GB10 20 layers → M5 Max 20 layers
→ back) is about 38 ms, including a ~6-7 ms WebRTC round trip. The GB10-only 2-device emulation at 0 ms (above)
gave 27.9 / 32.7.

## 2026-09-28: two-machine room, where a lap goes (branch perf/room-harness, main at c6ca8cc)

Harness: `tests/e2e/xroom.mjs` + `xroom_pair.sh` + `xroom_report.mjs` (docs/testing-fast.md, "Two-machine
rooms"), wire lab `tests/e2e/xwire_lab.mjs`. GB10 (headless Chromium 131, Vulkan) + M5 Max (headless Chrome
154, Metal). No room or engine change: every number is origin/main's code (trace marks are added at serve time).

**The link is Wi-Fi on both ends**, not wired: the GB10's Ethernet has no carrier (`wlP9s9` carries the
traffic) and the Mac's route goes through `en1` (Wi-Fi; `en0` inactive). WebRTC picks the LAN IPv6 host
candidates (`2601:…` on both ends), not Tailscale. ICMP round trip over the LAN IPv6: 3.1 / 3.4 ms min / avg
idle, 6.3 ms avg with a 4000-byte payload; over Tailscale 3.7-4.4 ms min, 5-48 ms avg depending on the hour,
with spikes to 70-120 ms and, in one session, multi-second stalls (ping p90 6 s).

### Wire lab: one frame out and back (`xwire_lab.mjs`, 200 frames per cell, 30 ms apart, 2 runs)

One-way ms p50 (run 1 / run 2), = (round trip - echo turnaround) / 2:

| send path | 4 KB | 10 KB | 16 KB | 32 KB | 40 KB | 80 KB |
|---|---|---|---|---|---|---|
| room wire (4 stripes, 4.6 KB slices, ordered) | 4.26 / 4.44 | 4.57 / 4.42 | 5.02 / 4.85 | 6.17 / 5.89 | 7.37 / 6.48 | 8.58 / 8.38 |
| 1 stripe | 4.38 / 4.42 | 4.74 / 4.69 | 5.11 / 5.04 | 6.38 / 6.20 | 6.99 / 6.43 | 9.44 / 10.44 |
| raw channel, one send per frame | 4.31 / 4.46 | 11.50 / 11.44 | 16.59 / 16.23 | 23.12 / 21.86 | 24.46 / 24.14 | 32.38 / 34.54 |
| PeerJS `send()` (`?wire=off`) | 4.25 / 4.76 | 10.98 / 12.04 | 11.71 / 12.23 | 13.42 / 13.41 | 14.07 / 14.14 | 16.33 / 16.30 |
| unordered (as `attachWire` builds it on main: maxRetransmits 0) | 4.38, 4 of 200 lost | 4.59, 4 lost | 4.94, 4 lost | 6.20 | 6.67, 2 lost | 8.71, 6 lost |
| PeerJS JSON ping (the peer card's rtt) | 2.25 / 2.55 | | | | | |

Reading: the network floor here is ~2.2 ms one-way for a tiny message and ~4.3 ms for a 4 KB hidden state
(the MoE's one column): on this Wi-Fi a 4 KB frame costs ~2 ms more than a ping, which is the medium, not
the code (ICMP shows the same with a 4000-byte payload). The room's sliced, striped wire is already the
best of these at every size, 2-4x better than one unsliced send from 10 KB up (dcSCTP's burst limit), and
2-3x better than PeerJS. Striping over 4 associations vs 1 is within noise below 40 KB and ~1-2 ms better at
80 KB. Unordered delivery on main is also unreliable (`maxRetransmits: 0`): on Wi-Fi it lost 20 of 1,200
frames, each costing the transport's 5 s gap timer. Keep it off. A raw `RTCDataChannel` layer of our own
would not beat this: the room already sends raw negotiated channels with its own binary framing, and the
floor is the link.

### MoE, GB10 host (20 layers + embed/head) + M5 Max guest (20 layers), `japan` and `twosum`, 128 tokens

tok/s per round (rounds 0, 1 untraced; round 2 traced), exact sampling:

| session | plain japan | plain twosum | spec japan (acc) | spec twosum (acc) |
|---|---|---|---|---|
| 1 (ping 3.7-88 ms, avg 13) | 9.3, 12.1, 20.9 | 30.1, 30.5, 20.0 | 18.1 (41%), 20.9 (43%), 31.7 (40%) | 47.8 (67%), 44.3 (67%), 43.9 (48%) |
| 2 (ping p90 up to 6 s) | 15.0, 4.3, 4.5 | 5.6, 4.2, 4.2 | 13.3, 6.3, 6.6 | 51.6, 4.6, 25.7 |

Same GB10, one device (`--solo`, same page): plain 36.0 / 36.8 (japan), 42.8 / 41.6 (twosum); spec 46.6 /
46.1 (59%), 65.5 / 66.7 (80%). Same GB10, two tabs on loopback (`room_prof.mjs`, twosum): plain 30.3, spec
50.7 (67%). The round-to-round spread across machines (9 to 21 tok/s for the same answer) follows the Wi-Fi,
not the code: session 2's rounds line up with the ping log's multi-second stalls.

One plain token, traced (session 1, medians over 128 laps, GB10 clock): **36.1 ms** =

| part | ms | of which |
|---|---|---|
| host: embed + its 20 layers + read the hidden back | 12.4 | GPU 9.7, `mapAsync` wait 2.6, encode 0.3 |
| host: send, wait for the Mac and the wire both ways | 16.7 | wire ~4.3 each way (lab), Mac's 20 layers + readback ~8 |
| host: head (upload, norm, LM head, top-k, map) | 4.2 | GPU 2.3, a second submit + `mapAsync` |
| host: sampling, emit, UI until the next lap | 1.7 | |
| submits / `mapAsync`s per token on the host | 3 / 2 | |

One speculative step, `japan` (52 steps, 2.46 tokens per lap): **72.8 ms** = drafting 15.0 (3 submits, 2
maps: the draft chain waits behind the previous step's rollback + refill on the GPU) + host layers for 4-8
columns 19.7 + waiting for Mac and wire 26.7 + head, sampling, rollback, refill 4.9 + between steps 3.8.
The host's GPU is busy ~30 ms of the 73, the Mac's ~16: **the host's serial share (drafts, its layers, the
head, the rollback) is ~60% of a step**, the network ~15%, the Mac ~25%.

### Draft acceptance in the room vs one device

Per draft depth, from the traces (`h.step0` / `h.step1`, draft head only):

| | K=3 | K=5 | K=7 |
|---|---|---|---|
| one GB10, twosum (spec always K=3) | 0.80 | | |
| loopback room, twosum, 3 runs (`room_prof`, incl. `mtpbatch=0`, `draftchain=0`) | 0.80 (15 steps) | 0.40 (1) | 0.36 (2) |
| GB10 + Mac, twosum (traced round) | 0.67 (3) | 0.30 (2) | 0.48 (11) |
| GB10 + Mac, japan | 0.50 (35) | 0.31 (11) | 0.14 (3) |
| 27B loopback room, twosum | 0.82 (11) | 0.30 (2) | 0.54 (4) |

The draft head is as good in the room as on one device at the same depth (0.80 = 0.80). The room's lower
headline acceptance is (1) the prompt (prose `japan` accepts ~50-59% even on one device; code ~80%) and
(2) the room's depth picker: it probes K=5 and 7 and keeps the best measured tok/s, and every extra draft
is accepted less often, so the ratio drops even when tokens per lap rise. On a noisy link the picker's
tok/s samples are noisy too: the traced twosum round ran 11 of 16 steps at K=7.

### Correctness observations (no change made)

- Within a session, every round's answer to `twosum` is identical in all modes, both sessions, the solo run
  and the 27B (sha 1df98ce0…); `twosum`'s answer is short code and insensitive.
- `japan`: session 1 plain == spec (44efa784…), but **session 2 plain (8e29cc8d…, 3 rounds) != spec
  (44efa784…, 3 rounds)**, same first 160 characters. The one-device GB10 answer (a0e7f9bd…) differs from both
  from about the 25th token ("nightlife" vs "local vibes"): expected between a Vulkan and a Metal device with
  an f16 wire (the 27B goldens already differ per machine). Forcing the GB10's GEMV autotune to (64,4),
  (128,4) or (256,4) does not change its one-device answer, so the autotune is not the cause. Session 2 is the
  one with multi-second network stalls; the harness now stores whole answers so the divergence point can be
  found on the next occurrence. Open.

Not measured in this round: the Mac as host (GB10 as guest) and the 27B across the two machines. The Mac's
GPU lock was held by other jobs for most of the window (one wait lasted 90 minutes), and then the shared SSH
connection to the Mac expired. The harness supports both (`xroom_pair.sh --here guest`, `--model
qwen3.8-27b`); the 27B two-tab loopback room on the GB10 gives plain 9.2, spec 18.3 tok/s (63%) on twosum.

## 2026-09-28: wide fused MoE expert kernels on Apple, faster moe_route (branch perf/metal-moe-fused-expert-kernels-apple)

The profile of 2026-09-27 found the fused expert kernels barrier bound on the M5 Max: moe_gus (256 threads, 4 rows,
one 32-weight block quarter per thread at dIn 2048, 8-level tree over 8 arrays) and moe_dnc (64 threads, one block per
thread at dIn 512, 6-level tree over 9 arrays) ran at ~280 / ~230 GB/s, moe_route took 16.6 µs a layer.

Changes:
- `moeFusedLayout` engine option (engine/wgsl/moe.js `gusKernelWide` / `dncKernelWide`): groups of TPR threads per R
  rows, each thread owns whole blocks (16 B vec4<u32> loads), a log2(TPR) tree per group. Default: `wide` when the
  adapter vendor is Apple (Chrome / Safari), `legacy` elsewhere (the GB10's bits are unchanged). MOEF_WIDE =
  gate/up 128 threads / 16 per row / 1 row, down 64 / 8 / 1 (picked by the sweep below). The MoE bits on Apple change
  (another summation order); batched == one-token still holds (spec == plain).
- moe_route: the top-K rank loop reads the probabilities 4 per load from a vec4 copy and stops once a thread's rank
  reaches K. Same ids and weights bit for bit (tests/e2e/moe_fused_cpu.mjs: moe_route == moe_router); on every GPU.

Kernel A/B (`tests/bench/moe_fused_sweep.js`, new: synthetic 35B-A3B shapes, 40 launches per timed pass, variants
interleaved round by round; `REF=<origin/main engine/wgsl/moe.js>` adds main's kernels as "ref"). The Mac was heavily
loaded during these runs by other processes (absolute µs are 2-4x the quiet-machine profile), so only the ratios mean
anything. Median µs per launch, M5 Max, Deno / Metal:

| kernel | main (ref) | this branch, legacy layout | this branch, wide |
|---|---|---|---|
| moe_gus | 194.9 | 194.3 | 95.7 (0.49x) |
| moe_dnc | 124.5 | 123.9 | 68.9 (0.55x) |
| moe_route | 98.8 | 52.8 (0.53x) | 52.4 |

Layout sweep (2 runs, same tool): gate/up TPR 16 or 32 at R 1 best (TPR 8 / R 2 and 256-thread R 4 slower); down TPR 8
or 16 best, TPR 1 / 2 and R 2 much slower. GB10 (Vulkan), same tool: moe_route 26.7 -> 16.5 µs; wide moe_gus 57.3 -> 51.7
but wide moe_dnc 36.8 -> 45.3 µs, hence legacy stays the default there (Chrome decode with the wide layout forced:
plain 49.2-49.6 vs 50.0-50.6, spec +3-4%).

Chrome 154 decode on the Mac (`chrome_bench.mjs <moe> 40`, base = origin/main, runs interleaved, all golden,
spec == plain, 0 GPU errors), plain / spec tok/s, two-sum then hash-map:

| run | main | this branch |
|---|---|---|
| 1 | 75.8 / 93.0, 38.6 / 13.0 | 73.3 / 96.1, 39.1 / 15.2 |
| 2 | 67.8 / 75.1, 23.2 / 10.3 | 73.7 / 97.7, 40.6 / 17.7 |
| 3 | 68.2 / 75.1, 22.9 / 9.1 | 70.7 / 88.9, 31.7 / 10.2 |

These are far below the quiet-machine numbers (85.9 / 135.5) and swing 2x between runs, so they only say "not
slower": this branch is ahead in 11 of 12 pairs. Scaling the kernel ratios onto the quiet profile (moe_gus 1.60, moe_dnc
0.98, moe_route 0.55 ms per token) gives about 1.5 ms of 11.7 ms, i.e. roughly +13% plain decode; to be re-measured on
an idle Mac.

GB10 Chrome decode (default = legacy layout, only moe_route changed), 2 runs each: main plain 50.0 / 50.5, 49.0 / 50.2,
spec 79.9 / 72.7, 80.8 / 72.8; branch plain 50.5 / 50.1, 50.4 / 50.6, spec 81.2 / 72.3, 80.9 / 72.3 (noise).

Gates: test_moe.js 3/3 llama.cpp + spec == plain on the Mac (MOE_FUSED_LAYOUT=wide, and the default) and on the GB10
(default, and MOE_FUSED_LAYOUT=wide); test_q38_bits.js ATTN_PREFILL_TILE=0 unchanged (GB10 85b12667 / eba0b8d5, Mac
b72e4d1f / ac403b4e); unit tests (tests/unit/moe_fused_wide_test.js new); tests/e2e/moe_fused_cpu.mjs (WGSL
interpreter, race detection) passes for the legacy and two wide layouts (its fused == unfused check now uses the legacy
unfused layout; it had failed since MOE_DEFAULT became the unfused default).

## 2026-09-28: Metal optimizations (overnight), branch perf/metal

This branch combines the two Metal branches that were kept:
- **perf/metal-wide-prefill-256-apple** (42136dd). `_prefillWide` submits a command buffer every `WIDE_SUBMIT_LAYERS = 8` layers instead of one per whole-model chunk. This fixes ubatch 256 on Apple Metal under Deno, and the arithmetic is unchanged.
- **perf/metal-moe-fused-expert-kernels-apple** (b46c1cd). Adds the `moeFusedLayout` wide fused expert kernels, the default on Apple adapters in Chrome and Safari, and the faster `moe_route`, which gives the same bits on every GPU.

Both cherry-picked cleanly onto main c6ca8cc, and the sections above give the details of each.

Commands:
- Deno: `D = deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights --allow-net`, run from `tests/`.
- Chrome: `node tests/bench/chrome_bench.mjs <model> 40 [extra]`.
- Base (origin/main) and branch ran alternately in the same session: base, branch, base, branch.
- bench_ctx builds its prompt from repo source that this branch edits (engine/qwen35.js, engine/wgsl/moe.js). The GB10 A/B therefore sets `CTX_SRC=<base checkout>`, so base and branch prefill the same tokens.

### GB10 (NVIDIA, Vulkan): no regression, MoE prefill +5%

| | main c6ca8cc | perf/metal |
|---|---|---|
| bench_ctx MoE defaults, prefill 512 / 4k / 16k (2 runs) | 259.0 / 360.5 / 297.7, 293.8 / 362.4 / 298.2 | 288.7 / 381.2 / 311.3, 303.9 / 382.7 / 311.1 |
| bench_ctx MoE defaults, plain decode 512 / 4k / 16k | 26.53 / 26.60 / 23.53, 26.68 / 26.39 / 23.54 | 27.32 / 26.79 / 23.87, 26.85 / 26.44 / 23.94 |
| bench_ctx MoE defaults, spec decode 512 / 4k / 16k | 33.43 / 31.37 / 34.04, 33.55 / 31.54 / 33.73 | 33.55 / 31.52 / 34.11, 33.65 / 31.88 / 33.94 |
| bench_ctx 27B `CTX=16640`, prefill / plain / spec at 4k | 75.7 / 9.05 / 13.25 | 75.6 / 9.11 / 13.27 |
| Chrome MoE decode, plain two-sum / hash-map (2 runs) | 49.97 / 50.69, 50.99 / 50.26 | 49.04 / 50.01, 50.23 / 49.59 |
| Chrome MoE decode, spec two-sum / hash-map | 79.61 / 72.39, 81.83 / 71.94 | 80.68 / 71.58, 81.66 / 71.44 |
| Chrome MoE `prefilllen=2048&prefillall=1`, all-on tok/s (relDiff) | 251.8 (2.14e-3), 249.3 (2.38e-3) | 258.2 (2.14e-3), 258.6 (2.14e-3) |

Reading the table:
- MoE prefill (bench_ctx, same tokens) is 5-6% faster at 4k and 4-5% faster at 16k, consistently across both run pairs. This is outside the noise. The likely cause is the per-8-layer submit: the GPU starts on the first layers while the rest are still being encoded.
- Chrome prefill at 2048 tokens is +3% (2 runs each).
- Decode is unchanged (<2%, noise). The GB10 keeps the legacy fused layout, so only `moe_route` changed for it.
- 27B is unchanged: wide prefill is opt-in for dense models, so bench_ctx does not use it.
- In every run, spec == plain, the Chrome output matches the golden text and there are 0 GPU errors.

### M5 Max (Metal): Chrome MoE decode +5.6% plain, +11% spec; the Deno MoE defaults run

The Chrome runs come from a quiet period of the Mac: main's numbers match the earlier quiet-machine baseline (85.9 / 135.5). They used Chrome 154 through `chrome_bench.mjs` with the macOS flag fix from PR #205, copied into both checkouts. Every run gave golden text, spec == plain and 0 GPU errors.

| Chrome 154, `chrome_bench.mjs <moe> 40` | main c6ca8cc | perf/metal |
|---|---|---|
| plain two-sum / hash-map, 5 interleaved runs | 85.32 / 86.26, 86.44 / 86.17, 86.32 / 86.25, 86.40 / 86.26, 86.11 / 85.26 | 90.80 / 91.25, 89.70 / 90.51, 91.23 / 91.31, 91.06 / 91.40, 90.97 / 91.14 |
| plain, mean | 86.1 | **90.9 (+5.6%)** |
| spec two-sum / hash-map | 137.81 / 122.61, 137.37 / 122.68, 137.76 / 122.90, 135.61 / 122.72, 136.13 / 122.24 | 152.22 / 135.72, 152.40 / 136.08, 152.46 / 136.39, 151.93 / 136.21, 150.58 / 136.03 |
| spec, mean | 136.9 / 122.6 | **151.9 / 136.1 (+11%)** |
| `prefilllen=2048&prefillall=1`, all off / all on tok/s (2 runs) | 173.9 / 242.5, 172.4 / 243.9 | 196.8 / 262.5, 195.0 / 261.1 (+13% / +8%) |
| same, relDiff (argmax) | 0.292, 0.227 (520 = 520) | 0.382, 0.173 (520 = 520) |
| 27B `prefilllen=2048&ubatch=256` (1 run): off / wide tok/s, relDiff | 63.9 / 109.8, 0.0316 | 63.7 / 112.5, 0.0172 |
| 27B decode plain / spec | 21.64 / 45.51 | 21.77 / 45.13 |

- The decode gain comes from the wide fused layout: this is the "+13% estimated" from the per-branch kernel A/B, now measured at +5.6% plain. Spec gains more, probably because its verify step runs the same fused kernels once per column.
- The Chrome prefill relDiff (0.17-0.38 on the MoE with all options on, 0.02-0.03 on the 27B) is the open Chrome-on-Metal nondeterminism noted on the wide-prefill branch. It is present on main and not changed by this branch; argmax is equal in every run.
- The 27B is unchanged, as expected: it has no MoE layers, and in Chrome, one command buffer per chunk was not failing.

Deno (wgpu/Metal), `MODEL=moe FILLS=512,4096,16384 $D bench_ctx.js` (engine defaults). This ran earlier, while the Mac was loaded: NVIDIA Sync used 240-290% CPU and plain decode on identical code swung from 6 to 32 tok/s. So only the pass/fail column means anything:

| run | main c6ca8cc | perf/metal |
|---|---|---|
| 1 | **fails**: `OperationError: validation error occurred` at `forwardToken` `mapAsync` after the first prefill | runs, spec == plain on every row, 0 GPU errors |
| 2 | **fails** (same error) | runs, spec == plain on every row, 0 GPU errors |

`PREFILL_UBATCH=0` and the 27B (`CTX=16640`) ran on both base and branch, with spec == plain on every row including the 27B at 16384 (22/36). The speed spread from the load makes those numbers meaningless, so they are not listed.

### Gates

| Gate | GB10 | M5 Max |
|---|---|---|
| `test_moe.js` (default) | MATCH llama.cpp 3/3, spec == plain on 3/3 (layout legacy) | MATCH 3/3, spec == plain on 3/3 (Deno: layout legacy, new moe_route) |
| `MOE_FUSED_LAYOUT=wide test_moe.js` | MATCH 3/3, spec == plain on 3/3 | MATCH 3/3, spec == plain on 3/3 |
| `ATTN_PREFILL_TILE=0 test_q38_bits.js` | BITS plain 85b12667 hidden eba0b8d5 (unchanged) | BITS plain b72e4d1f hidden ac403b4e (unchanged); spec == plain; GPU sampling == logits path |
| `MODEL=27b PREFILL_UBATCH=256 LENS=150,700,2100 test_prefill_wide.js` | PASS, max relDiff 7.26e-5 | **PASS**, max relDiff 6.50e-5, 0 GPU errors (main: validation error at 700) |
| `MODEL=moe [PREFILL_UBATCH=256] LENS=150,700,2100 test_prefill_wide.js` | PASS, max relDiff 1.23e-3 | 150 tokens pass (3.64e-5); **700: `validation error occurred` at `forwardToken` `mapAsync`** (see below) |
| unit tests, `node --check`, generator_smoke | 248 passed / 0 failed, pass, pass | (not run on the Mac) |
| tests/e2e/moe_fused_cpu.mjs (WGSL interpreter) | MOE FUSED CPU PASS (legacy + two wide layouts, no races) | |

**Open: MoE test_prefill_wide at 700 tokens on the Mac.** It hit the known Deno-only `mapAsync` validation error once. That is the same error bench_ctx hits on main, and on this branch bench_ctx with the same wide + grouped prefill defaults ran 2 of 2 times. The wide-prefill branch passed this case when it was measured on its own. An A/B job (this branch vs 42136dd alone, alternated 2x) was set up twice but never ran: the first time the SSH connection dropped, the second time the shared Mac GPU lock stayed taken by other jobs for over an hour. Whether this is intermittent or caused by the combination is not settled. Output never silently changes: when the error happens, the run throws.

## 2026-09-28: M5 Max profile at main c6ca8cc (Chrome 154 / Deno 2.9.7, Metal)

Same Mac Studio as above. Chrome runs: `CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"`.

**Bandwidth and floors** (`tests/bench/bw_probe.js`, new: streaming read of a 1 GiB buffer, slope between 2 and 6
passes per submit so the submit round trip cancels). Chrome: **443 GB/s** read (u32 or vec4 loads within 3%), copy
199 GB/s; per dispatch 1.2 µs, per compute pass 1.7 µs, **per compute pass + `copyBufferToBuffer` 17.1 µs** (a
blit costs ~15 µs on Metal). Deno: 486 GB/s, pass 24.8 µs, and an **empty submit + `onSubmittedWorkDone` round
trip of 12.5 ms**; a 4-byte `mapAsync` round trip is 14.5 ms (`prof_moe_decode.js` sync floor), vs 0.27 ms in
Chrome. So every Deno decode number on the Mac carries ~13 ms per sync: Deno MoE plain is 28.0 ms/token for
12.1 ms of kernels. Use Chrome for Mac speed numbers.

**Decode, Chrome** (`node tests/prof/prof_chrome.mjs <model> out.json roompath=1`):

| | plain ms/token | kernel sum | weights read per token | kernels vs 443 GB/s | spec K=3 |
|---|---|---|---|---|---|
| MoE | 11.66 (85.8 tok/s) | 10.02 ms, 420 dispatches | 2.0 GB | 45% | 26.1 ms/step, 3.67 tok/step, 140.7 tok/s |
| 27B | 47.05 (21.3 tok/s) | 40.45 ms, 563 dispatches | 15.1 GB | 84% | 75.9 ms/step, 3.5 tok/step, 46.1 tok/s |

MoE kernel families per plain token: DeltaNet projections 2.45 ms (~260 GB/s), expert gate/up 1.60, LM head 1.26
(~330 GB/s), DeltaNet core 1.11 (`dn_delta_gn` 33.7 µs x 30 for 4 MB of state each), expert down 0.98, attention
projections 0.83, attention core 0.60, `moe_route` 0.55 (16.6 µs each, one workgroup), router GEMV 0.32, norms 0.29.
The 27B's GEMVs run at ~374 GB/s (`matvec_q4_gu` 268 µs) and the LM head at ~425 GB/s: the 27B is near the roofline,
the MoE is not. Encoder structure (new counters): plain MoE token 1 submit, 87 passes, 483 dispatches, 1 copy;
one speculative step 120 passes, 627 dispatches, **142 `copyBufferToBuffer` moving 81.5 MB** (124 of them in
`_verifyFused`: the replay state `S -> S_pre` and conv / beta / decay copies per DeltaNet layer).

Room path (roadmap 24 gate 1): `embedRun` + `headFromHiddenIds` vs `forwardTokenIds`, same tab, best of 3:
MoE 12.14 vs 11.66 ms (+4.1%), 27B 48.21 vs 47.49 ms (+1.5%).

GEMV shape A/B (`chrome_bench.mjs <moe> 40 'opts={"coopWG":W,"coopRows":R}'`, plain / spec tok/s, two-sum and
hash-map, all golden, spec == plain): default 256/4 85.9 / 135.5, 86.0 / 121.2; 64/2 **93.2 / 143.9, 93.9 / 129.0**;
32/4 93.1 / 140.0; 128/4 92.2 / 140.1; 64/4 92.1 / 140.4; 64/8 90.6 / 130.9; 128/8 89.9 / 131.5. 27B: default
21.66 / 47.2, 64/4 22.48 / 47.2, 128/8 21.28 / 41.2. (At dIn 2048 a 256-thread row group gives each thread one
32-weight block, then an 8-level tree.)

**Prefill:** `chromium-experimental-subgroup-matrix` is exposed (subgroup size 32) but only with f32->f32 and
f16->f16 8x8x8 configs, so `prefillMath=sgmatrix` (needs f16->f32) is unavailable on Apple. The MoE has no 16-column
GEMM shapes (`gemmOn false`), so `prefillMath` does not apply to it at all.

**Wide prefill with 256-column chunks fails on Apple.** `MODEL=27b PREFILL_UBATCH=256 LENS=150,700
test_prefill_wide.js`: 150 tokens (one 128-column chunk) relDiff 2.2e-5, greedy and spec identical; 700 tokens
(256-column chunks): device lost, reported as `OperationError: validation error occurred` at the next `mapAsync`.
The MoE with `MOEGROUP=0` does the same. `diag_map` (MoE defaults): a 255-token prompt (a 192-column chunk) passes,
256 / 320 / 448 / 511 tokens lose the device, at maxSeq 2048, 8192 or 32768 alike (not memory). Chrome reports no
error and returns wrong numbers: 27B `prefilllen=2048&prefillall=1&ubatch=256` relDiff **0.836**, argmax 318 vs
34062. This is the Deno "wide + grouped" `mapAsync` error above; grouped prefill is not needed to trigger it.

**27B spec != plain at 16384** is the harness overrun `bench_ctx.js` now refuses: with `CTX=16640`, fill 16384 gives
spec == plain (22/36).

Deno `bench_ctx.js` (`MODEL=... FILLS=512,4096,16384`; MoE with `PREFILL_UBATCH=0`, since the default crashes as
above; 27B with `CTX=16640`): MoE prefill 36.4 / 212.6 / 190.0, plain 33.31 / 32.04 / 28.05, spec 42.26 / 42.71 /
43.92; 27B prefill 31.9 / 58.5 / 53.6, plain 14.79 / 14.35 / 12.43, spec 23.54 / 21.13 / 15.96, spec == plain on
every row. Chrome `chrome_bench.mjs` MoE defaults: plain 85.4 / 86.0, spec 135.8 / 121.5; `prefilllen=2048&
prefillall=1`: 170.3 tok/s all off, 243.2 all on (relDiff 0.22, argmax equal).

## 2026-09-28: phone placement (GB10 + iPhone 14 Pro Max, branch perf/cluster-phone-placement)

Two ideas from the device matrix below, tried on the real phone. Phone on the branch's Vercel preview
(`https://pooled-git-perf-cluster-phone-placement-nehanths-projects.vercel.app/room`, public PeerJS, HF weights,
0.5 GB pledge), GB10 host on the same commit from 127.0.0.1 with local weights, Qwen3.6 35B MoE, exact sampling,
`tests/e2e/xroom_cluster.sh --devices here,phone --phone-url <preview> -- --model qwen3.6-35b-moe --gb 40 --prompts
japan,twosum --rounds 2 --maxnew 128 [--query ...]`, 2 runs per config (each 2 plain + 2 spec rounds per prompt),
all in one evening session (19:00-20:35), the phone on power and unlocked.

**1. Phones hold no layers while the computers can hold the model (kept).** `room/plan.js phonesToLeaveOut`: by
memory, when the host and the other computers can hold every layer, phones join as ask-only guests (as the speed
split already did); `?phonelayers=1` on the host deals them layers anyway.

| config (GB10 40 GB pledge + iPhone 0.5 GB) | plain japan | plain twosum | spec japan | spec twosum | sha japan / twosum |
|---|---|---|---|---|---|
| before (cluster-matrix C_moe_1/2: GB10 39 + iPhone layer 39) | 14.7-21.7 | 19.4-25.4 | 25.0-28.7 | 24.7-38.6 | 3f42e667 / dcc61ede |
| same split on this branch, `phonelayers=1` (C_notail_moe_1/2) | 17.9-19.8 | 23.7-24.4 | 20.9-29.0 | 25.3-39.7 | 3f42e667 / dcc61ede |
| **after: phone asks, GB10 holds 40** (C_after_moe_1/2, C_final_moe_1) | **35.8-37.4** | **39.7-41.4** | **45.3-46.8** | **62.4-67.2** | a0e7f9bd / dcc61ede |
| GB10 alone (matrix reference) | 35.4-37.0 | 39.8-40.2 | 45.5-47.5 | 63.1-64.9 | a0e7f9bd / dcc61ede |

About 1.9x plain and 1.6-1.9x spec over the phone holding a layer, and the same as the GB10 alone (within noise),
with the GB10-alone answers (a0e7f9bd / dcc61ede) and plain == spec in every round. Qwen3 1.7B (13 GB host):
48.6/47.2 and 49.2/50.0 japan, 48.9/47.1 and 51.0/50.4 twosum (C_after_q17_1/2), vs 25.6-29.0 with the phone's
layer before and 49.1/51.3 alone; shas 166285f6 / e19cba7f as alone. The phone's page still joins in 4.2 s and
stays in the room as a guest.

The other combos with the phone (main's baselines from the device-matrix runs, same session evening; after = this
branch, phone on the preview, it asks and holds nothing):

| combo | plain japan | plain twosum | spec japan | spec twosum |
|---|---|---|---|---|
| B before: M5 Max 39 + iPhone 1 (30 GB host, B_moe_1) | 28.5/29.4 | 26.9/32.4 | 43.7/44.4 | 37.5/56.6 |
| **B after: M5 Max 40, phone asks** (B_after_moe_1/2) | **68.4/69.5, 68.8/70.3** | **83.1/81.8, 82.1/82.5** | **95.2/96.0, 95.1/95.3** | **130.2/133.5, 132.2/131.6** |
| D before: GB10 19 + M5 Max 20 + iPhone 1 (D_moe_here_2) | 19.5/18.7 | 18.5/20.9 | 19.1/21.3 | 27.9/27.6 |
| D after: GB10 20 + M5 Max 20, phone asks (D_after_moe_1, _3) | 26.5/18.7, 23.2/20.8 | 16.6/23.8, 25.1/22.8 | 29.0/27.2, 28.4/27.0 | 43.0/40.9, 37.5/56.4 |
| A (GB10 20 + M5 Max 20, no phone; matrix A_moe_1/2) | 29.7-29.9 | 28.4-33.3 | 26.1-38.0 | 51.6-56.8 |

B goes 2.3-2.4x plain and 2.2-3.5x spec (the Mac alone; answers 3f42e667 / dcc61ede, plain == spec, the same as
B before). D after is A's split (the phone is out of the chain), so it gains over D before (spec +35-100%) but reads
below the earlier A runs in plain (16.6-26.5 vs 28.4-33.3): the same code path as A, run 1.5-2 h later on the shared
Wi-Fi, so read the gap as run-to-run conditions, not the phone (it sends no frames). D's japan has plain 8e29cc8d
vs spec 44efa784 in both runs: the GB10 + M5 Max open item from the matrix (A run 1 did the same), not this change.
D_after_moe_2 failed at join (the public PeerJS server said no room for the code) and was rerun as D_after_moe_3.
Runs between 20:28 and 21:15 overlap a phone-lock mix-up in another job (the lock was released under running
jobs); in B and D after the phone only joins, and B_after_moe_1/2 agree to within 2%.

**2. Keep the phone off the last (full-attention) layer (tried, dropped).** The chain ends on the phone, so it held
layer 39, full attention (+3 ms over the context without subgroups, its KV cache on the phone). Tried: the host
keeps layer 39 as a tail, a second engine run on the returned hidden before the head, and every slice moves one
earlier, so the phone gets layer 38 (DeltaNet). Same answers (3f42e667 / dcc61ede, plain == spec). Numbers
(`phonelayers=1`, tail vs `phonetail=0`, 2 runs each, interleaved):

| | plain japan | plain twosum | spec japan | spec twosum | phone 1-col GPU p50 (run 1, traced) | phone verify-block GPU p50 (run 1) | host tail p50 (local stand-in) |
|---|---|---|---|---|---|---|---|
| tail (phone: layer 38, DeltaNet) | 19.3/19.9, 18.6/18.9 | 22.0/22.5, 22.8/17.8 | 22.9/21.0, 22.8/23.2 | 33.3/36.3, 36.4/32.5 | 5.7 ms | 30.0 ms | 3.2 ms |
| no tail (phone: layer 39, attention) | 18.9/17.9, 19.4/19.8 | 23.7/23.9, 24.4/23.7 | 25.0/20.9, 29.0/25.2 | 37.7/25.3, 38.5/39.7 | 7.4 ms | 25.8 ms | - |

The phone's one-column time did drop (7.4 -> 5.7 ms p50 over the first 30 s), as predicted, but its 4-8-column
verify blocks rose (DeltaNet's per-column state snapshots and serial recurrence cost more on the phone than a few
columns of attention at these contexts), and the tail's own submit + readback adds 3.2 ms p50 per lap on the host
(measured in the local stand-in, p95 77 ms: an occasional stall). Net: plain within noise, spec slightly worse. Not kept
(commit 5f4cffb has it; 4e0b2fe reverts it). With rule 1 the phone only holds layers when the computers can't hold
the model, and then the context-growth of its attention layer is the smaller problem.

Local stand-in (a GB10 tab posing as an iPhone, `xroom.mjs --ua iphone`, 22 GB host pledge): the tail gave the
same answers as the phone at layer 39 and cost 26.5-29.1 vs 28.9-32.4 plain tok/s.

## 2026-09-28: device matrix (Spark, M5 Max, iPhone 14 Pro Max)

Branch perf/cluster-matrix (main c6ca8cc + b749f99 merged, + the room harness). Every device runs main's code: the
iPhone opens https://pooled.run/room (production = main; public PeerJS signaling, HF weights, 0.5 GB pledge), the
GB10 ("Spark": headless Chromium, Vulkan) and the M5 Max (headless Chrome 154, Metal) serve their own checkout on
127.0.0.1 with local weights. Runner: `tests/e2e/xroom_cluster.sh --devices <here,there,phone> -- --model M --gb G
--prompts japan,twosum --rounds 2 --maxnew 128 [--trace-rounds 1,3,5,7]` (2 plain + 2 spec rounds per prompt,
exact sampling), per-device chain report `xroom_chain_report.mjs`, phone lap trace injected over WebDriver
(`XROOM_PHONE_TRACE=1`, GPU timestamps on the page's own passes). All on home Wi-Fi (both computers are on Wi-Fi
too, see "two-machine room" above). Numbers are per round; two runs per config unless noted.

**Coverage.** Measured: C (Spark + iPhone) MoE and 1.7B, A (Spark + Mac) MoE, the phone thermal run, per-layer
profiles on the phone and the GB10, one-device references on the GB10. **Not measured: B (Mac + iPhone) and D (all
three), the Mac as host, the 27B.** They need the Mac's GPU lock, and four batches of the Spark+Mac job were
queued on it at the same time: our D run waited the lock's full 90 minutes and gave up (`no lock`); the queued runs
(D host Spark / host Mac, B, 1.7B on B and D, Mac solo references) are still in the runner's chain and will land in
the scratch results; this section is to be extended with them.

### Matrix: tok/s (per round), acceptance, answers

Qwen3.6 35B-A3B MoE Q4, `--gb 40` (C) / `--gb 13` (A). embed + head always on the host (GB10).

| combo | split | load (host online) | plain japan | plain twosum | spec japan (acc) | spec twosum (acc) | answers |
|---|---|---|---|---|---|---|---|
| GB10 alone (3 runs) | 40+embed | 163-168 s | 35.4, 37.0 | 39.8, 40.2 | 45.5, 47.5 (59%) | 63.1, 64.9 (80%) | japan a0e7f9bd, twosum dcc61ede |
| A GB10 + M5 Max, run 1 | 20+embed / 20 | 103 s | 29.8, 29.7 | 33.3, 29.9 | 37.0, 26.1 (42, 37%) | 53.5, 51.6 (67%) | japan **plain 8e29cc8d, spec 44efa784**; twosum dcc61ede |
| A run 2 | same | 106 s | 29.3, 29.9 | 30.4, 28.4 | 38.0, 34.7 (42%) | 54.0, 56.8 (67%) | japan 44efa784 (plain == spec), twosum dcc61ede |
| C GB10 + iPhone, run 1 | 39+embed / 1 (layer 40 of 40) | 158 s; phone joined 4.2 s, slice served at 159 s | 21.7, 19.3 | 25.4, 19.4 | 28.7, 27.9 (48, 52%) | 38.6, 24.7 (67, 50%) | japan 3f42e667, twosum dcc61ede, plain == spec |
| C run 2 | same | 164 s | 14.7, 21.7 | 25.0, 23.7 | 25.0, 25.7 (48, 44%) | 34.5, 38.0 (67%) | same |
| C, twosum only, earlier main (smoke, 2 runs) | same | 157 s | | 17.7-21.2 | | 21.5-35.6 (55-67%) | 1df98ce0 = its solo |

Qwen3 1.7B Q8, `--gb 13`: GB10 alone 49.1 (japan) / 51.3 (twosum) plain, load 18 s; **C** (27+embed / 1, the
phone holds layer 28 of 28), 5 runs: plain 25.6-29.0, load 20-22 s, answers japan 166285f6 / twosum e19cba7f
= the GB10 alone (the 1.7B has no draft head: spec = plain).

Correctness: every C round gives the same answer in plain and spec, and `twosum` equals the one-device answer. `japan`
never equals the one-device GB10 answer once a second GPU is in the chain: C departs at character 359 ("the \"herd\"
mentality" vs the GB10's wording), A at character 128 ("local vibes" vs "nightlife"). A different device's layers
with an f16 wire is expected to move a long prose answer (as noted in the two-machine entry); what is not expected
is **A run 1: plain 8e29cc8d != spec 44efa784 within one session**, again reproducing the open item from the
two-machine entry (diverges at character 350: `the "herd."` vs `the "herd" where possible.`). With the phone (C)
plain == spec held in all 16 japan rounds. The japan/twosum answer shas moved (1df98ce0 -> dcc61ede for twosum)
between the smoke runs and these, on both the solo and room sides together, when main b749f99 was merged.

### Where a lap goes (traced rounds, medians, ms)

| round | lap p50 / p95 | host pre (its layers + embed) | host GPU | wait (other devices + wire) p50 / p95 | head | wire, all hops, p50 / p95 | worker residence p50 / p95 |
|---|---|---|---|---|---|---|---|
| A plain japan (GB10 20 + Mac 20) | 33.4 / 37.2 | 12.2 | 9.7 | 15.3 / 17.1 | 4.2 | 8.6 / 10.4 | Mac 6.6 / 7.8 (readback-wait 6.4) |
| C plain japan (GB10 39 + phone 1) | 47.0 / 60.3 | 22.3 | 18.3 | 17.3 / 28.9 | 4.3 | 7.9 / 17.7 | phone 9 / 13 |
| C plain twosum | 40.4 / 56.1 | 18.4 | 15.0 | 15.9 / 25.6 | 4.2 | 8.8 / 20.6 | phone 6 / 11 |
| C spec japan (step) | 89.6 / 129 | 35.7 | 31.5 | 28.1 / 51.7 | | 11.5 / 33.7 | phone 15 / 25 |
| C spec twosum (step) | 114.7 / 881 | 52.0 | 47.2 | 32.7 / 821 | | 12.4 / 807 | phone 19 / 35 |
| C 1.7B plain japan (GB10 27 + phone 1) | 33.5 / 44.9 | 14.5 | 11.8 | 14.5 / 22.6 | 3.3 | 8.3 / 16.1 | phone 5 / 11: GPU 3.9, readback 1.1 |

Per layer (one-token decode, MoE, GPU time): GB10 0.47 ms (18.3 ms / 39 layers in C, 9.7 / 20 in A; `layer_prof`
alone: 0.41 DeltaNet layer 38, 0.54 attention layer 39); M5 Max <= 0.33 ms (20 layers inside a 6.6 ms residence);
**iPhone 3 ms back-to-back** (`tests/bench/layer_prof.mjs --browser ios`, layer 38 or 39, wall 3 ms p95 4, map wait
3.7-4.1 ms) but **6.8-11.7 ms per lap inside the room** (GPU timestamps on the page, layer 39). One phone layer
costs as much GPU time as 15-25 GB10 layers, and its two Wi-Fi hops another ~8 ms; its share of the model is 1/40.
1.7B: phone 3.9 ms for one layer vs GB10 0.44 (11.8 / 27).

Phone kernels, one-token decode of layer 39 (full attention + MoE), per-dispatch timestamps (each dispatch in its own
pass, so the sum, 5.8 ms, exceeds the 3 ms wall; read the ratios): attn_flash 1.47 ms, matvec_q8_coop (q/kv proj) 2
x 0.72, moe_gus_q4_q8 0.79, moe_dnc_q4_q8 0.74, matvec_q4_coop_acc (o proj) 0.52, the rest < 0.2 each. Layer 38
(DeltaNet): the qkvz projection matvec_q4_coop 1.44 ms is the largest. The same kernels on the GB10 total 0.63 /
0.43 ms. 4-column verify (block4) on the phone: 7-11 ms for 1-2 layers vs 3-5 plain, MoE expert kernels grow the
most (0.8 -> 1.3 ms each).

**The phone's GPU cools between laps.** `layer_prof --gaps` (idle ms before each call, 60 calls each; layers 39 / 38-39):

| idle before call | 0 | 10 | 20 | 40 | 80 | 0 again |
|---|---|---|---|---|---|---|
| 1 layer, plain wall ms (p95) | 5 (9) | 7 (9) | 8 (10) | 7 (11) | 9 (11) | 5 (6) |
| 2 layers, plain wall ms (p95) | 5 (6) | | 10 (15) | 12 (14) | | 6 (7) |

In the room the phone idles ~35-40 ms of every 47 ms lap, which is why its layer takes 7-10 ms there instead of 3-5.
(A third gap run hung the phone page and timed out after 30 min; no result.)

Links: phone <-> GB10 over IPv6 host candidates, direct (no relay), STUN RTT 4-10 ms with single samples at 18 and
125 ms; peer-card rtt 4-15 ms, one 136 ms. GB10 <-> Mac: STUN 4-8 ms, ICMP 3.2 ms min / 11-56 ms avg with a 96 ms
max in A run 1. Median wire per lap is the same with the phone (7.9-8.8 ms both hops) as with the Mac (8.6), but the
p95 is twice as high (17.7-20.6 vs 10.4), and one C spec round had an 0.8 s stall (lap p95 881 ms).

Acceptance per draft depth in C (traced rounds): japan K=3 0.61 (38 steps), K=5 0.20 (4), K=7 0.14 (2, 245 ms a
step); twosum K=3 0.75 (4), K=7 0.49 (10). The picker spends steps at K=5/7, where a phone verify of 8 columns is
30+ ms; spec's gain over plain in C is 1.3-1.5x (as on one device, 1.3-1.6x), but from a lower base.

### The phone over 7.4 minutes (C, MoE, plain japan, 16 rounds x 512 tokens, 8,147 traced laps)

tok/s per round: 20.4, 18.7, 20.3, 19.4, 19.5, 19.8, 19.7, 19.9, 17.7, 19.7, 20.3, 19.9, 19.7, 19.6, 18.8, 19.3: flat,
no throttling visible in throughput, one answer (1926bd..) in all 16. The phone's GPU ms per lap (layer 39) depends
on the position (full attention: the KV it reads grows), so, binned by position and by time:

| position | 0-2 min | 2-4 min | 4-6 min | 6-7.4 min |
|---|---|---|---|---|
| 160-340 | 7.34 | 6.55 | 6.68 | 8.00 |
| 340-520 | 8.20 | 9.62 | 8.36 | 11.43 |
| 520-700 | 7.54 | 10.34 | 10.05 | 10.77 |

About +3 ms from position 250 to 600 (attention), and +10-40% in the last 1.5 minutes at the same position: the
onset of throttling, still hidden because the phone is ~20% of a lap. Longer runs or more phone layers will show it.

### What to fix, ranked (phone first; none duplicates the Spark+Mac experiments)

1. **Where the phone's layers go.** Today the phone gets the last layer, which in Qwen3.5/3.6 is a full-attention
   layer: its cost grows with context on a GPU without subgroups (+3 ms per lap from position 250 to 600) and its
   KV cache lives on the smallest device. Put phone slices on DeltaNet (linear attention) layers, whose cost and
   state are constant, and do not deal a layer to a phone at all when the model fits on the others (a phone's layer
   costs 15-25 GB10 layers of time + two Wi-Fi hops for 1/40 of the model: C runs at 50-60% of the GB10 alone).
   room/plan.js (the deal), room.js (the plan's consumers). Expected: C's phone residence flat at ~6 ms instead of
   7-12; with "no layer when not needed" the room runs at the host's own speed.
2. **Keep the phone's GPU awake for its hop.** Back to back its layer takes 3-5 ms; after the 20-40 ms idle it gets
   in a room lap it takes 7-12 ms. Let the host send a tiny "your hidden is coming" control message to each phone
   worker when a lap starts, and have the phone submit a short dummy dispatch (or its own queue writes) so the GPU
   is clocked up when the hidden arrives. room.js (host lap start, `workerFrame`), room/transport.js. Expected -3
   to -5 ms a lap in C/D (~+8-10% tok/s), to be checked against thermals (the thermal run above as the gate).
3. **No-subgroup kernels on the phone.** attn_flash (1.5 ms, 25-30% of the phone's attention layer) and the q8/q4
   coop matvecs dominate; the 4-column verify path grows the MoE expert kernels 60%. Profile with the page's
   timestamp-query on Safari and tune the decode attention split / matvec tiles for the 32 KB, no-subgroup path.
   engine/wgsl/ (attention, matvec, moe). Expected phone layer 3 -> ~2 ms hot; small in C today (phone ~20% of a
   lap) but it grows with every layer a phone holds and in B.
4. **Wi-Fi jitter to the phone.** Same median wire as GB10 <-> Mac, twice the p95, and a 0.8 s stall in one round.
   Likely iOS Wi-Fi power save between sparse frames; try a light keep-alive on the data channel while a room is
   generating and measure p95 / stalls. room/transport.js. Expected lap p95 -8 to -10 ms in C, fewer stall rounds.

Not ranked: thermal-aware pacing (no throughput loss in 7.4 min with one layer; revisit with 2+ phone layers), the
speculative depth picker with a phone in the chain (K=5/7 cost 30+ ms of phone verify; belongs to the depth-policy
work, input handed over here).

## 2026-09-28/29: keep-alive on the wire while generating (C: Spark + iPhone)

Branch perf/cluster-phone-wifi-jitter (f02efdd on perf/cluster-matrix). While a wire link carried a frame in the
last 1.5 s, each end sends a 1-byte message on a negotiated unordered, never-retransmitted channel (id 78) whenever
it sent nothing on that link for 10 ms (`?ka=ms`, `?ka=0` off; not a protocol change, see docs/protocol.md). Same
code on both sides: the phone opens the branch's Vercel preview, the GB10 its checkout; A/B by `?ka=0` vs `?ka=10`
(and `?ka=4`) on every device, interleaved in the same lock hold. `xroom_cluster.sh --devices here,phone
--phone-url <preview>/room --phone-query "dev=1&ka=K" -- --model M --gb G --prompts japan,twosum --rounds 2
--maxnew 128 --query ka=K --trace-rounds ...`, 0.5 GB phone pledge (1 layer: MoE layer 40, 1.7B layer 28).

tok/s, mean over every round of the runs (range):

| model | config | runs | plain | spec |
|---|---|---|---|---|
| 1.7B | ka=0 | 4 | 22.7 (18.5-27.1) | - |
| 1.7B | ka=10 | 3 | 26.5 (24.3-28.5) | - |
| 1.7B | ka=4 | 2 | 24.6 (14.2-28.8) | - |
| MoE | ka=0 | 3 (1 more failed to start) | 20.8 (13.6-25.5) | 30.6 (24.1-37.0) |
| MoE | ka=10 | 4 | 21.9 (19.1-26.3) | 31.3 (22.3-41.5) |
| MoE | ka=4 | 2 | 22.9 (19.1-26.2) | 32.6 (26.4-38.9) |

Phone-side hop gaps (phone send → next frame in, plain decode laps of traced rounds, per run):

| model | config | laps > 300 ms per run | max gap (ms) |
|---|---|---|---|
| 1.7B | ka=0 | 2, 3, 2, 2 | 689, 766, 663, 445 |
| 1.7B | ka=10 | 1, 0, 0 | 327, 156, 262 |
| 1.7B | ka=4 | 0, 0 | 211, 200 |
| MoE | ka=0 | 2, 2, 1 | 770, 724, 672 |
| MoE | ka=10 | 1, 0, 1, 1 | 315, 167, 308, 332 |
| MoE | ka=4 | 0, 0 | 191, 189 |

- **What it does:** it cuts the tail, not the p95. The 0.4-0.8 s stalls (2-3 per 522-lap run on the 1.7B) mostly
  go away and the worst gap halves. Lap p50/p95 and wire p50/p95 do not move (wire p95 14-24 ms in both configs;
  the expected -8 to -10 ms p95 did not happen). The phone's own residence is unchanged (8-10 ms p50).
- **Throughput:** 1.7B plain +17% (the on range sits above most of the off range; a 0.7 s stall in a ~5 s round
  is ~14% alone). MoE +5% plain / +2% spec, inside the noise of 3-4 runs; the untraced rounds show +15% plain only
  because two off rounds hit stalls (13.6, 17.7).
- **Correctness:** same answer in every round off and on (MoE japan 3f42e667, twosum dcc61ede; 1.7B 166285f6 /
  e19cba7f, = GB10 alone); unit tests 251 pass. No engine change.
- **Not measured:** A (Spark + Mac), B (Mac + iPhone), D (all three) with the keep-alive on vs off. Every attempt
  on 09-29 waited out the Mac GPU lock (90 min; the Spark+Mac integration job held it while waiting for the Spark
  GPU) and further runs were not permitted in this session. The keep-alive also runs on computer-computer links
  (~100 one-byte messages/s per link while generating); a regression there is unlikely but unverified.

## 2026-09-29: MoE decode, fewer launches per layer (branch perf/decode-moe-bandwidth), GB10 Chrome

What the MoE decode spends its time on (`tests/prof_ts.js`, Deno timestamps, one dispatch per pass, so small
kernels carry ~10-15 µs of pass overhead each): the expert kernels are close to what they can do (`moe_gus` 74 µs =
~155 GB/s in the model, 206 GB/s in `moe_fused_sweep.js`; `moe_dnc` 52 µs / 159 GB/s in the sweep, the wide and
`moeDnRows` 2 / 4 variants are no faster on the GB10); the rest of a MoE layer was four small serial launches
(post-attention `rmsnorm` 9 µs, the f32 router GEMV 15 µs, `moe_route` 20 µs, DeltaNet input `rmsnorm` 9 µs).

Changes (MoE models only; the 27B's kernels and bits are untouched):
- `moe_nrt` (`moeNormRouter`, default on): post-attention RMSNorm + router GEMV in one launch. Each workgroup
  sums x*x and W_r . (x*w) in one loop and one tree, writes inv * dot; workgroup 0 writes xn for `moe_gus`. The
  router is stored as BF16 (the file's router is BF16; the loader widened it exactly): 2.1 -> 1.05 MB per layer.
- `dn_nba` (`dnNormBA`, default on): the same kernel for a DeltaNet layer's input RMSNorm + merged F32 beta/alpha
  GEMV, before the [qkv|z] GEMV that reads its xn.
- `moe_route`: with renormalized weights (Qwen3.5/3.6) rank the logits and softmax over the 8 picked logits only
  (the full sum cancels), no 256-wide trees; the rank counts among candidates (>= the smallest of 8 group maxima)
  instead of all 256. 20 -> 17 µs in the model, 12.4 µs in the sweep.
- 502 -> 432 dispatches per token. Tried and dropped: routing inside every `moe_gus` workgroup (+34..57 µs per
  launch: the prologue runs once per wave of workgroups), routing next to the shared expert in one launch (neutral).

Correctness: `test_moe.js` MATCH llama.cpp 3/3, spec == plain 3/3, GPU sampling == logits path (also with
`GPU_SAMPLE=0`); `test_moe_split.js` PASS; `MODEL=moe LENS=150,700 test_prefill_wide.js` PASS (max relDiff 1.4e-3);
`test_q38_bits.js` BITS plain 8a532ef5 hidden 52f2ae10 (unchanged); unit tests 433/433; `tests/e2e/moe_fused_cpu.mjs`
PASS (K 3 and 8). The router weights' last bits change (llama.cpp text is the MoE golden).

Chrome decode tok/s (`chrome_bench.mjs <moe> 64 japan=1`, plain / spec K=3, mean of 3 interleaved runs vs origin/main
2f1704e in its own worktree):

| | two-sum | hash-map | japan |
|---|---|---|---|
| main | 50.16 / 82.17 | 50.01 / 69.68 | 43.38 / 52.98 |
| branch | 52.14 / 85.61 | 52.47 / 73.22 | 45.05 / 54.99 |
| gain | +3.9% / +4.2% | +4.9% / +5.1% | +3.8% / +3.8% |

Acceptance unchanged (48/54, 44/63, 39/75). Kernel sum per token 19.15 -> 18.55 ms (Deno). Not measured on the M5 Max;
the three changes apply there too (they do not depend on the fused expert layout).

## 2026-09-29: long context at 1K, 8K and 32K, f16 vs int8 KV (issue #71, branch perf/longer-context-timing), GB10 Deno

`cd tests && MODEL=moe KV=f16|q8 HW=GB10 deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights bench_ctx.js`, then `bench_ctx_compare.js f16.log q8.log`. maxSeq 33024 for both. Prefill tok/s is for the tokens added to reach that fill (1024, then 7168, then 24576). Decode is 32 tokens of whatever the text is at that point, so it is noisy; read it as "no collapse at 32K", not as a trend. Other jobs were queued on the GPU but not running.

| Date | Model | Hardware | KV | Context | Prefill tok/s | Plain decode tok/s | Spec decode tok/s | Spec = plain |
|---|---|---|---|---|---|---|---|---|
| Sep 29 | moe | GB10 (Deno) | f16 | 1K | 136.6 | 11.8 | 14.17 (21/36) | yes |
| Sep 29 | moe | GB10 (Deno) | f16 | 8K | 116 | 22.13 | 24.16 (17/42) | yes |
| Sep 29 | moe | GB10 (Deno) | f16 | 32K | 125.5 | 9.92 | 13.07 (24/27) | yes |
| Sep 29 | moe | GB10 (Deno) | q8 | 1K | 126.1 | 14.58 | 23.34 (22/33) | yes |
| Sep 29 | moe | GB10 (Deno) | q8 | 8K | 94.7 | 13.79 | 13.21 (17/42) | yes |
| Sep 29 | moe | GB10 (Deno) | q8 | 32K | 36.2 | 19.3 | 28.48 (24/27) | yes |

- int8 KV gave the same 32 greedy tokens as f16 at every fill.
- int8 KV prefill at 32K is 0.29x f16 (679.6 s vs 195.8 s): the engine turns tiled prefill attention off with kvQ8, so long prompts fall back to the slower attention path.
- KV cache: 0.63 GB f16, 0.35 GB int8 at this maxSeq.
- Not yet run: the 27B at 1K / 8K / 32K (f16 and int8), and repeated decode runs to average out the noise.
- int8 KV decision for now: keep it off by default. It saves memory and keeps the tokens, but a 32K prompt takes 3.5x as long to prefill. Revisit once tiled prefill attention supports int8 KV.

## 2026-09-28: the room host's share of a lap in one submit (branch perf/room-host-one-submit, `hostFuse`)

Change: a chain host used to run its share of each lap as separate GPU round trips (submit, map, read back).
Plain greedy decoding took 3 submits and 2 maps per token; a speculative step took 7 and 5. Now:
- **Plain** (greedy, GPU sampling, in a chain): the head of the hidden the chain returned, the GPU gather of
  its pick and the host's layers on it run as one command buffer with one readback (`engine.headAhead`).
  The hidden goes out before the token is shown. The recurrent state is saved in the same buffer and put
  back (`dropAhead`) when the pick is not piped (a stop token, the cap, an abort).
- **Speculative, draft chain on**: the draft chain, the drafts' embeddings gathered into the verify columns
  and the host's layers run in one submit (`_hostTrunkFused`).
- Same kernels, same inputs, same order, so the same bits. `?hostfuse=0` restores the old path for A/B.

The prose prompt (`japan`) does not take the speculative fused path. The reduced-vocabulary draft head
misses more than 5% there, so `draftVocabAuto` turns the draft chain off, and those steps run the
unchanged per-token drafting on both sides of the A/B. Its spec numbers below are the same code twice
(noise). Solo never reaches either path (both need a chain).

MoE, GB10 host (20 layers + embed/head) + M5 Max guest (20 layers), LAN Wi-Fi (IPv6 host candidates),
exact sampling, 128 tokens. `tests/e2e/xroom_pair.sh -- --model qwen3.6-35b-moe --gb 13 --prompts
japan,twosum --rounds 2|3 [--query hostfuse=0]`, sessions alternated off/on. Untraced rounds, tok/s median
(n = rounds):

| | plain japan | plain twosum | spec japan | spec twosum |
|---|---|---|---|---|
| off, 4 sessions (n=8) | 28.1 | 28.6 | 34.1 (41%) | 51.7 (67%) |
| **on**, 4 sessions (n=8) | **29.9 (+6.8%)** | **31.9 (+11%)** | 34.1 (same path) | **54.0 (+4.5%)** |
| off, GEMV shape 64/4 pinned on both ends (`--tune 64,4`, `--guest-tune 64,4`; n=2) | 27.0 | 30.4 | 35.7 | 50.5 |
| **on**, same pin (n=4) | **30.9** | **32.6** | 34.6 (same path) | **54.0** |
| off, `--fixk 3` (n=4) | | | 36.6 (49%) | 58.2 (80%) |
| **on**, `--fixk 3` (n=4) | | | 38.0 (same path) | 58.8 (+1%, noise) |

The spread inside a cell is 5-10 tok/s and follows ping spikes on the Wi-Fi (off: plain twosum 23.2-33.1;
on: 24.0-35.6). The traced rounds are the cleaner comparison. Medians over the laps, GB10 clock,
`xroom_report.mjs`:

| traced round | off (2 sessions) | on (2 sessions) |
|---|---|---|
| plain japan, one token | 33.3 / 34.2 ms: layers 11.8 + head 4.2, 3 submits, 2 maps | **31.5 / 31.9 ms**: head + layers 15.2, 1 submit, 1 map |
| plain twosum, one token | 31.2 / 32.4 ms | **28.5 / 28.5 ms** |
| plain, send -> hidden back | 15.1-16.0 ms | 15.0-16.2 ms (unchanged: the Mac and the wire) |
| spec twosum, drafting + host layers | 24.0 / 22.9 ms (2 submits, 2 maps) | **21.2 / 21.1 ms** (1 submit, 1 map) |

So it saves about 2-3 ms of a ~33 ms plain token and about 2 ms of a ~50-55 ms spec step. That is less than
the 4-5 ms estimated up front: the GPU work itself does not shrink (head + layers 12.3 ms of GPU), only the
extra sync and the idle gap between submits go away.

In process on the GB10 (`BENCH=2 tests/test_moe_split.js`, split at 20, same process, off / on medians):
plain +7 to +24% over five prompts, spec -7% (bash, 22 tokens) to +14%.

One device (`xroom.mjs --solo`, GB10, 4 rounds per cell, off / on): plain japan 35.7 / 35.5, plain twosum
41.9 / 41.3, spec japan 47.8 / 47.4, spec twosum 67.8 / 66.6. Same code path, so this is noise; the answers
are identical.

Correctness:
- `test_moe_split` passes with both paths: split == solo, spec == plain, and checkpoint/resume with a
  pending rollback.
- 27B `test_q38_bits` with `ATTN_PREFILL_TILE=0`: BITS 85b12667 / eba0b8d5 on the GB10 and b72e4d1f /
  ac403b4e on the Mac, unchanged.
- In the two-machine room every round of a cell gave the same answer, on and off.

The one outlier is plain `japan`: 8e29cc8d in 7 of 8 unpinned sessions, and 44efa784 (the spec answer) in
one "off" session. That session is the only one where both ends' autotune picked the 64/4 GEMV shape. With
64/4 pinned on both ends, plain == spec == 44efa784 in every round, on and off. So the plain != spec seen
in the first two-machine sessions comes from the cooperative GEMV shape the load-time autotune picks. It
predates this change and is not caused by it. The next thing to chase is which device's plain (one-column)
GEMV shape changes the bits against its batched verify.

## 2026-09-28: which device hosts, and where the split falls (branch perf/room-placement)

Same harness and machines as above, over the home LAN (Wi-Fi on both ends; Tailscale to the Mac was down).
MoE (Qwen3.6-35B-A3B Q4_0) and the 27B, exact sampling, 128 tokens, `japan` and `twosum`, 2 untraced rounds
per mode and prompt, each config run in 2 sessions interleaved with the others (so 4 samples per cell). The
link was bad all evening: ping averages 15-150 ms with peaks of 100-440 ms (p50 during a round 3.5-6 ms, p90
often 20-300 ms), so single rounds swing by 30%+ and only medians are quoted.

**The question.** A spec step spends about 60% of its time on the host (embedding, 20 layers, head, drafting,
rollback and refill), and the M5 Max moves memory about twice as fast as the GB10. Does it help to give the
Mac more layers (split by speed), or to make the Mac the host?

### MoE, split point and host (origin/perf/room-harness b25aa6a, `--split` and `--here guest`), median tok/s

| config | plain japan | plain twosum | spec japan | spec twosum |
|---|---|---|---|---|
| GB10 hosts, 20 / 20 (pledge) | 29.0 | 31.3 | 34.0 | 51.5 |
| GB10 hosts 16, Mac 24 | 29.9 | 30.6 | 36.6 | 47.3 |
| GB10 hosts 12, Mac 28 | 29.4 | 33.2 | 39.1 | 41.8 |
| **Mac hosts, 20 / 20** | **31.6** (+9%) | **34.9** (+11%) | **47.4** (+39%) | **61.8** (+20%) |
| Mac hosts 16, GB10 24 | 30.3 | 32.2 | 45.0 | 58.5 |
| Mac hosts 24, GB10 16 | 32.9 | 35.3 | 49.4 | 63.3 |

Moving layers to the Mac while the GB10 hosts is within noise (the Mac's layers overlap nothing; they are
simply 2x cheaper per layer, a few ms a lap). Moving the **host role** is what pays: the head, the sampler,
the draft block and the rollback/refill all run on the faster memory, and they sit on every token's
critical path. Mac-host 24/16 is a little ahead of 20/20 (+1-4%, within noise).

### The change: pick the model host by GPU speed

`room/gpuspeed.js` times a 64 MB buffer copy at page load (8 copies, best of 5 passes, its own device,
destroyed afterwards); the result travels in the device's `hello` meta as `gbps` (old tabs don't send it,
and a missing value leaves the pick by memory, so no protocol bump). `pickModelHost` still starts from the
device that lends the most memory, then hands the host role to a device of the same kind that copies at
least 1.5x faster and lends at least half as much memory (a phone never beats a computer). Measured in the
same headless browsers as the rooms, 6 page loads each: **GB10 185-199 GB/s** (one outlier 106), 274-454 ms
for the probe including device creation; **M5 Max 370-398 GB/s**, 11-33 ms. The ratio is about 2.0, so the Mac
hosts even when it lends less (12 GB against 13). `?gbps=N` pins the value (0 = unknown).

Checked in a real room (`xroom.mjs --speedpick`): the GB10 pressed Start and the Mac took the model host
(Mac layers 1-18 + embed/head, GB10 serving 19-40, the split by pledge), online in 63 s. The harness pins
its host page (`?gbps=0`) so `--here host|guest` still means what it says.

Same code, pinned both ways (2 sessions; the second batch had ping p90 up to 300 ms, so plain is noise):

| | plain japan | plain twosum | spec japan | spec twosum |
|---|---|---|---|---|
| MoE, GB10 hosts -> Mac hosts, all 8 samples | 25.4 -> 26.5 (+4%) | 28.4 -> 32.2 (+13%) | 30.7 -> 39.4 (+28%) | 45.3 -> 55.4 (+22%) |
| 27B (31 / 33 layers), 4 samples | 9.4 -> 9.6 (+2%, noise) | 9.3 -> 9.0 (-4%, stalls) | 10.8 -> 12.6 (+17%) | 18.2 -> 21.6 (+19%) |

The 27B across two machines is new: GB10-hosted spec twosum 18.2 matches the GB10 two-tab loopback (18.3);
Mac-hosted 21.6 beats it. One GB10 alone, `--solo`, 2 runs each, before -> after the change (the probe runs at
load), range over 4 rounds: plain japan 34.9-35.9 -> 35.9-37.2, plain twosum 40.1-41.5 -> 40.4-43.4, spec japan 45.7-47.8 -> 46.0-47.7 (59%), spec twosum 64.0-68.1 -> 65.1-67.8
(80%): unchanged within noise, same answers (a0e7f9bd / 1df98ce0), spec == plain.

### Correctness

- `twosum`: one answer (1df98ce0…) in every config, split, host direction, mode, session and the solo runs.
- 27B: `japan` 94b0f2de… and `twosum` 1df98ce0… in every round, both directions, plain == spec.
- MoE `japan` depends on where the split falls, on either host: 16/24 (either host) gives the one-device
  answer a0e7f9bd; 24/16 and 12/28 give 3f42e6… (from character 359); 20/20 gives 8e29cc… or 44efa7… (from
  character 128, "local vibes"), and those two split at character 350 (`the "herd."` vs `the "herd" where
  possible.`), a near tie that lands either way by session and by mode, on the old code as on the new (GB10
  hosting, base code: plain 8e29cc / spec 44efa7 in one session, both 44efa7 in the next; Mac hosting, new
  code: both 8e29cc in both sessions). The Vulkan + Metal split with an f16 wire moves a close argmax; the
  host change changes no split's arithmetic. Open, as before.
- No engine change (27B bit goldens not rerun). Unit tests: 245 passed, 0 failed.

## 2026-09-29: iPhones in rooms, combined branch (fix/iphone-rooms: memory + resume + GPU wake, #207)

Spark (GB10) and iPhone 14 Pro Max (iOS 26.6.2), public signaling, the phone on the branch preview, 1 GB pledge, host 22 GB with `?phonelayers=1`, Qwen 3.6 35B MoE, `twosum`, 48 tokens, 2 answers per run (`xroom.mjs` + `xroom_phone.mjs`):

| Run | Split | Phone online after | Decode tok/s (answer 1 / 2) | Answer |
|---|---|---|---|---|
| m1 | host 38+embed, iPhone 2 (layers 39-40, 917 MB from the network) | 125 s | 23.6 / 20.2 | 6e01f17d both |
| m2 | same | died at 82% of its download | | the page vanished with no jetsam entry and no crash report while the USB link (WebDriver and syslog) dropped at the same second |
| m4 | same | 129 s | 25.4 / 21.7 | 6e01f17d both |
| m5 | same | 121 s | 25.3 / 19.6 | 6e01f17d both |

Recovery with the real phone (`room_resume.mjs --url --external iphone` + `resume_phone.mjs`, Qwen3 0.6B, host + worker tabs on the Spark, phone 4 layers): Safari in the background 30 s: the answer finished in 58 s with the same text and no drop (the link survived, main's liveness keeps a silent phone 60 s); reload: back in its slot, same text, 52 s. The same with the MoE (host 22, worker 1, phone 1 GB) was tried twice; both times the Spark's host browser closed (once mid-answer, once mid-load) as other GPU jobs started on the Spark, so it is not measured.

Spark loopback: `room_resume.mjs` lock 20 s (answered in 25 s, link kept), reload (35 s, carried on), kill (72 s, re-dealt after 60 s), all the same text; `room_phone_mem.mjs` phone tab peak 311 MB (423 MB from the worker) and 364 MB (mostly from the network), killed twice while loading: 0.25 GB share, then re-dealt without it, room online with the same answer each time.

## 2026-09-29: rooms on bad networks, origin/main vs fix/network-resilience (GB10)

Branch fix/network-resilience = fix/net-signaling + fix/net-drop + fix/net-lossy, merged with origin/main
5af2f66 (which added the keep-alive and the 15 s silent-link drop in the meantime). Same harness for both
columns: `tests/e2e/room_chaos.mjs --root <checkout>` (3 headless Chromium tabs on one GB10, Qwen3 1.7B Q8
split by memory so all three hold layers, 96 new tokens, local PeerServer, every WebRTC packet through the
userland UDP shaper; `join` uses 2 tabs and no model). Base = origin/main 5af2f66. One run per cell;
"identical" = same text as that run's LAN baseline. After a failed answer the harness waits until the room
shows it is ready (re-dealing if it offers that), then asks again.

| Plan / scenario | origin/main | fix/network-resilience |
|---|---|---|
| RTT 50 / 150 / 300 ms | 7.5 / 3.5 / 1.9 tok/s, identical | 8.8 / 3.6 / 2.0 tok/s, identical |
| 1% loss, RTT 50 ms | 6.0 tok/s | 8.1 tok/s |
| 5% loss, RTT 50 ms | 4.4 tok/s | 7.3 tok/s |
| 15% loss, RTT 50 ms | 1.4 tok/s (11.5 s stall) | 4.1 tok/s (2.3 s stall) |
| 2 / 5 / 12 s freeze of one guest, 2 / 8 s of every device | complete, identical | complete, identical |
| 20 s freeze of one guest | fails at 17 s ("guest1 left"); re-deal, next answer identical on 2 devices | fails at 16 s ("the link to guest1 dropped; ask again"); re-deal, next answer identical on 2 devices |
| guest's network dies while idle | dropped after 17 s; re-deal 12 s; next answer OK | dropped after 16 s; re-deal 10.5 s; next answer OK |
| guest's network dies mid-answer | fails at 16.8 s ("guest1 left") | host shows "guest1 stopped responding; waiting for it (Stop gives up)" at 3.6-5 s; fails at 15.4 s |
| signaling down mid-answer, then the next answer | both identical | both identical |
| new device joins while signaling is down | "Can't reach the room server. Check your internet connection…" | "Can't reach the signaling server (…). … Rooms already running are not affected" + a self-host link |
| new device joins right after signaling is back | "No room with that code" (the host's registration is not back) | joins in 0.7 s |
| signaling blip of 5 s / 30 s, then a new device joins | "No room with that code" | joins in 0.25 s |
| host or guest opens the page with signaling refused | "Can't reach the room server…" | "Can't reach the signaling server…" |
| host or guest opens with signaling blackholed | "Connecting…" until the server is back | "Can't reach the signaling server…" after 20 s, buttons usable again |
| signaling down, a device leaves, re-deal | fails (new links need signaling) | fails the same way |
| no direct path, no relay | "No room with that code" (the room exists) | "Found the room, but these two devices can't reach each other… A relay (TURN) server gets around that", Network box opens |
| no direct path, TURN relay (`?turn=`) | no relay support: the join times out | all 4 links on the relay, 33.1 tok/s relay-only |

Drop detection (`tests/e2e/room_drop.mjs --devices 3`, 161 tokens, all 16 checks pass): RTT 300 ms + 5% loss
and RTT 600 ms + 5% loss finish identical with nobody flagged (longest silence 0.9 s against a 3.9 s limit and
1.6 s against 4.8 s). A dead middle / last device is flagged in 5.1 / 3.9 s, the answer fails at 15-17 s
(the link fails or the silent-link drop fires), the re-deal is offered at once and the room answers again.
Signaling fallback (`tests/e2e/signal_fallback.mjs`, no GPU): 19 of 19.

How the three branches changed when merged:
- Drop detection no longer fails the answer 3.5-5 s into a silence. As first merged it did, and a first matrix
  run showed it failing 5, 8 and 12 s freezes that origin/main finishes. It now flags the device, holds the
  answer (Stop gives up) and makes a new question wait; the device counts as back only after it answers a
  ping sent after the silence began. Dropping a silent device is left to main's ping loop (15 s for a computer,
  60 s for a phone).
- Main's 15 s silent-link drop fires before ICE gives up (~15-17 s), so fix/net-lossy's redial of a dead link
  no longer brings back a computer frozen for 20 s: it is dropped and rejoins through the re-deal, as on main.
  The redial still applies to phones (60 s) and to links that fail while packets still flow.
- The decode lap-timeout floor from fix/net-drop went from 15 s to 25 s: a 12 s freeze left a 15 s gap between
  two tokens in the first run.

Still open: a re-deal while signaling is down still fails, and an answer caught in a freeze longer than
~15 s isn't retried automatically.

## 2026-09-29: decode layer fusion, fewer dispatches per layer (branch perf/kn-layer-fusion, not kept: opt-in)

Question: how much of plain decode is launch overhead, i.e. what do fewer dispatches per layer buy when every
fusion keeps today's bits? Engine option `layerFuse` (off by default; `true`, or a list of `dn,conv,kv,comb,norm,pass`;
`engine.layerFuse.<k>` flips at runtime). One-token passes only: the batched verify/prefill kernels are untouched,
so spec == plain holds by construction and was checked.

Dispatches per decode layer today (counted by wrapping `dispatchWorkgroups` over 16 tokens, Deno):

| | DeltaNet layer | attention layer | FFN | per token (+ head) | compute passes |
|---|---|---|---|---|---|
| MoE, main | 7: norm, [qkv\|z], [beta\|alpha], conv, dn_pre, dn_delta_gn, out+res | 9: norm, q, [k\|v], glue, kv_store, flash, combine, sigmoid_mul, o+res | 5: norm, router, route, gus, dnc | 502 | 91 |
| 27B, main | 7 | 9 | 3: norm, gate/up, down+res | 674 | 145 |
| layerFuse | 5 | 6 | unchanged | MoE 412, 27B 530 | 1 |

The fusions (`engine/wgsl/layer_fuse.js`, `engine/wgsl/coop.js` CV / NRM variants):
- `dn`: `dn_pre` (q/k L2 norm, beta/decay gates) folded into `dn_delta_gn` (`dn_delta_gnp`): each value-head
  workgroup normalises its key head from workgroup memory with `dn_pre`'s in-order sum; [dt | A] packed in one buffer
  to stay at 8 storage bindings. -1 per DeltaNet layer.
- `conv`: `dn_conv` in the [qkv | z] GEMV's epilogue (`matvec_*_coop_cv`): the thread that stores a qkv row runs
  the causal conv + SiLU + state shift on it; each channel belongs to one workgroup, so no race. -1 per DeltaNet layer.
- `comb`: `sigmoid_mul` in `attn_combine` (`attn_combine_g`, gate read from q_full). -1 per attention layer.
  Written as `(O / L) * sigmoid(g)` Metal folded the two and changed bits on the M5 Max (GB10 unchanged); the quotient
  now goes through workgroup memory (store, barrier, load) and the bits match on both.
- `kv`: `kv_store` in `attn_glue` (`attn_glue_kv`: k heads packed to the f16 cache from workgroup memory, v copied);
  needs `comb` (the glue no longer copies the gate). -1 per attention layer.
- `norm`: an attention layer's input rmsnorm inside its [k | v] GEMV (run before q): every workgroup recomputes
  rmsnorm's 256 strided partials and tree (fewer threads each run several partials), normalises its x slices in
  registers, workgroup 0 writes xn for q. -1 per attention layer. The same fold for the MoE router and the DeltaNet
  [beta | alpha] GEMV was built and measured (+1.3% more on the GB10) and then dropped here: perf/decode-moe-bandwidth
  already has it (`moe_nrt`, `dn_nba`).
- `pass`: `forwardToken`'s layers and head in one compute pass instead of 2-3 per layer.

Correctness (all with `layerFuse` on): per-flag logit hashes equal to off for 24 decode tokens after prefill,
spec == plain, on the GB10 (MoE 92fafe48, 27B 4d85013d) and the M5 Max (MoE b97a1a37, 27B 2517aa46);
`test_q38_bits.js` ATTN_PREFILL_TILE=0 LAYER_FUSE=1 = reference on both (see below for the GB10; Mac
b72e4d1f / ac403b4e, GPU sampling == logits path); `test_moe.js LAYER_FUSE=1` MATCH llama.cpp 3/3, spec == plain
3/3, GPU sampling head check 0 mismatches (Mac); `tests/e2e/fusion_synth.mjs` has one row per flag; unit tests 433/433.
(Deno on the Mac: a long-running process fails `mapAsync` with "validation error" after ~6 prefill + decode + spec
cycles, with `layerFuse` never on as well: a harness/runtime limit, not these kernels.)

Speed, Chrome (`chrome_bench.mjs <model> 40`, plain tok/s, two prompts, OFF/ON interleaved in one snapshot tree):

| | OFF | ON | gain |
|---|---|---|---|
| GB10 MoE (6 samples each) | 50.57 | 52.38 | +3.6% |
| GB10 MoE, `dn,conv,kv,comb` only (6 each) | 50.67 | 52.56 | +3.7% |
| GB10 MoE, with the router / beta-alpha norm folds as well (6 each) | 50.12 | 52.96 | +5.7% |
| GB10 27B (4 / 2 samples; a later ON run hit GPU contention and is dropped) | 11.16 | 11.30 | +1.3% |
| M5 Max MoE (6 each) | 91.02 | 91.88 | +0.9% |
| M5 Max 27B (4 each) | 21.66 | 21.85 | +0.9% |

Deno GB10 plain: MoE 31.96 -> 31.0 ms/token (median of 4), 27B 101.78 -> 100.09 ms. Spec tok/s unchanged
(the verify step runs the batched kernels). About 1 ms per MoE token on the GB10 for 90 fewer dispatches and 90
fewer passes (~11 µs each), under 0.1 ms on the M5 Max: Metal launches are cheap (1.2 µs per dispatch measured
on 2026-09-28) and the Mac's MoE gap to MLX is inside the kernels, not between them. Below the 5% bar on both
machines, so `layerFuse` stays opt-in. Stacked with perf/decode-moe-bandwidth (orthogonal kernels) the GB10 MoE
gain would be expected around +7-8%; not measured together.

## 2026-09-30: layerFuse on by default for NVIDIA, Apple and Deno (branch perf/kn-layer-fusion)

The 2026-09-29 layer fusions above, merged with main (#246's `dn_nba` and `moe_nrt` now run next to them) and
turned on by default: `layerFuse` "auto" (the default) is on when the adapter vendor is NVIDIA or Apple, or when
there is no adapter info (Deno), and off on everything else until someone measures it there. `?layerfuse=0|1`
(engine/preset.js) or `layerFuse: false` turns it off or on. Every flag keeps the bits, so rooms may mix devices
with it on and off.

The Chrome numbers from separate tab loads were not usable tonight: three other worktrees' Deno jobs were on
the GB10 during the runs (nvidia-smi), and plain MoE decode swung between 15 and 52 tok/s in both arms. The
numbers below come from a new in-process A/B that flips `engine.layerFuse` at runtime in alternating blocks after
the same prompt, so both arms share the load and any contention: `tests/bench/lf_ab.js` (Deno, 8 rounds x 24
tokens) and `bench.html?lfab=R` (Chrome, R rounds x 40 tokens). Both check that the two arms give the same tokens.

| | off | on | gain |
|---|---|---|---|
| GB10 Deno MoE (8 blocks each, median ms/token) | 30.77 | 29.46 | +4.5% (every on block faster than every off block) |
| GB10 Deno 27B | 101.89 | 100.70 | +1.2% |
| GB10 Chrome 27B (2 tabs x 8 blocks, median tok/s) | 11.09 / 11.09 | 11.15 / 11.21 | +0.5% / +1.0% |
| GB10 Chrome MoE, 4 tabs (8, 8, 12, 12 blocks), all contended (quiet GB10: ~52 tok/s) | 19.0 / 43.2 / 16.0 / 12.8 | 20.1 / 47.9 / 17.0 / 13.2 | +5.7% / +10.9% / +6.0% / +3.1%, tokens identical |
| GB10 Chrome MoE, only the quiet blocks of tab 2 (3 pairs at 52-55 tok/s) | 52.3 / 52.9 / 53.4 | 54.7 / 53.0 / 54.2 | +4.6% / +0.2% / +1.5% |
| M5 Max Chrome MoE (2 tabs x 8 blocks) | 90.29 / 90.78 | 92.09 / 92.28 | +2.0% / +1.7% (every on block faster) |
| M5 Max Chrome 27B | 21.10 / 21.07 | 21.34 / 21.29 | +1.1% / +1.0% (every on block faster) |

Every GB10 Chrome MoE tab's median favoured on, but the GB10 was never quiet enough tonight for a clean Chrome MoE figure.
The 2026-09-29 quiet run (+3.6%) and the quiet tail above (~+2%) are the numbers to trust. The M5 Max, which had no
contention, gains 1-2% in Chrome and has every on block faster than every off block.

Correctness, GB10 (Deno): `test_q38_bits` with `LAYER_FUSE=0` and default (on) give the same hashes: 4cac59d8 / a67b7bcd, and
`ATTN_PREFILL_TILE=0` 4f70a9ca / 5eb28e41 both ways. These are not the old 8a532ef5 / 85b12667 references: main moved them
before this branch, and `LAYER_FUSE=0` is main's code path. Spec == plain and GPU sampling == logits in both.
`test_moe` MATCH llama.cpp 3/3, spec == plain 3/3, head check 0 mismatches (on and off). `test_moe_split` PASS (split == solo,
spec == plain). `fusion_synth.mjs` on the real GPU (`E2E_GPU=real`): every layerFuse row 0 logits differ. Its `attn_glue` row
fails on a real GPU with or without this branch, which the file's own comment already says. `test_prefill_opts` 27B PASS. MoE:
700 tokens relDiff 2.44e-2 over the 2e-2 tolerance, with identical numbers under `LAYER_FUSE=0` and `LAYER_FUSE=1`, so it
comes from main (routing near-ties, see the note in the test) and not from this branch. Argmax, greedy and spec == plain are
unchanged. M5 Max (Deno): `test_q38_bits ATTN_PREFILL_TILE=0` e3903fe5 / 8742688e for both off and on. Unit tests 942/942,
`npm run check` clean.
## 2026-10-01: moe_route top-K by a merge network (branch perf/moe-route-merge), GB10

Report item B4 (MOE-1). `moe_route` now turns each expert's value into a u32 sort key, sorts chunks of K by rank
and merges list pairs in ceil(log2(nExp / K)) rounds (5 for 256 experts), one barrier per round, all 256 threads.
The order is (value desc, id asc), the same as before, so ids and weights are the same bits.

| | origin/main 8a435fe | branch |
|---|---|---|
| `moe_fused_sweep.js` route µs (REF = main, 2 x 3 runs) | 12.40 | 10.35 |
| `prof_ts.js` moe_route in the model (2 runs) | 17.2 / 17.3 µs, 0.69 ms/token | 14.8 / 14.8 µs, 0.59 ms/token |
| `prof_ts.js` kernel sum per token | 18.39 / 18.41 ms | 18.23 / 18.28 ms |
| Chrome plain tok/s, two-sum / hash-map / japan (mean of 3, interleaved) | 50.98 / 50.91 / 44.06 | 51.08 / 51.31 / 44.65 |
| Chrome spec K=3 tok/s | 92.22 / 63.13 / 48.29 | 90.55 / 64.49 / 49.35 |

The kernel gain is clear and repeatable (-2 µs a launch, -0.1 ms a token, ~0.5%). End to end it is inside run-to-run
noise (plain +0.2..+1.3%, spec -1.8..+2.2%, with one low branch run at 86.6 two-sum spec). What I tried that did
not help: a comparator on the float values plus NaN and padding tests in each compare (14.45 µs with a linear count,
12.40 with a binary search, no better than main); bigger phase-0 chunks (F = 2, 4, 8: same 10.35 µs). Empty
selection floor: 4.2 µs. The selection part went from ~8.2 to ~6.1 µs.

Correctness: unit tests 960/960 (new cases: +0/-0 ties, nExp 1024 with K 16); a CPU differential fuzz against
origin/main's kernel (3500 cases: random, integer ties, +0/-0, nExp up to 1024, K 1..16) gives identical ids
and weight bits. Only inputs with NaN or -inf logits differ: main's group-max threshold drops them or picks nothing.
The new kernel orders NaN after every number and never writes it to a slot. `test_moe.js` MATCH llama.cpp 3/3, spec ==
plain 3/3 (same output and acceptance as main); Chrome golden true, specIdentical true, the same acceptance as main.
`test_q38_bits.js` default 4cac59d8 / a67b7bcd and ATTN_PREFILL_TILE=0 4f70a9ca / 5eb28e41, both the same as
origin/main on the same machine (the 27B does not use moe_route). Barrier lint clean; `npm run check` passes.
Not measured on the M5 Max.
## 2026-10-01: speculative verify without state copies (branch perf/decode-verify-blits)

Replay rollback used to copy every DeltaNet layer's state into `S_pre` before the verify pass and the
conv/beta/decay inputs into `L.rp` after it, then copy `S_pre` back and copy the conv state per layer in
`_restoreDN`. Now `dn_delta_mc` writes `S_pre` and each column's inputs into `L.rp` from its own registers
during the verify, and `_restoreDN` is one compute pass (replay from `S_pre` + new `dn_conv_restore`). Same
kernel, inputs and order on replay, so the bits are unchanged.

Copies per speculative step (M5 Max, `prof_chrome.mjs`, K=3): MoE 142 (81.5 MB) -> 7 (24 KB); 27B 222.8
(194 MB) -> 6.8 (57 KB). Step 19.71 -> 19.44 ms (MoE), 69.73 -> 68.98 ms (27B). `prof_chrome.mjs` and
`prof_moe_decode.js` hang in submit mode on GB10 (main too), so the counts come from the M5.

Spec K=3 tok/s, `chrome_bench.mjs <model> 64`, mean of 3 runs (plain unchanged in every pair):

| | main | branch | |
|---|---|---|---|
| GB10 Chrome MoE two-sum / hash-map | 93.4 / 65.6 | 97.9 / 68.0 | +4.9% / +3.6% |
| GB10 Chrome 27B two-sum / hash-map | 25.5 / 22.8 | 26.4 / 23.7 | +3.5% / +3.6% |
| GB10 Deno MoE (`test_moe.js`) two-sum / hash-map / bash | 59.7 / 52.7 / 44.2 | 63.2 / 55.9 / 46.2 | +5.9% / +6.0% / +4.6% (2-3 runs) |
| M5 Max Chrome MoE two-sum / hash-map | 176.0 / 125.6 | 179.1 / 128.1 | +1.8% / +2.0% |
| M5 Max Chrome 27B two-sum / hash-map | 47.35 / 42.57 | 47.91 / 43.12 | +1.2% / +1.3% |

The M5 gained less than the ~5% estimated (the copies there cost ~0.3 ms per step, not 1.2-1.5 ms); GB10
gained more. Gates: unit tests, `npm run check`, `test_q38_bits` (4cac59d8, and 4f70a9ca with
ATTN_PREFILL_TILE=0: same as main), `test_moe` MATCH llama.cpp + spec == plain, `test_mtp`, `test_mtp_split`,
`test_moe_split`, M5 bits equal to main, `specIdentical` and golden in every Chrome run.

## 2026-10-01: test_prefill_opts MoE 700-token miss is a routing near-tie in the all-off baseline (branch test/prefill-opts-tol), GB10

`MODEL=moe test_prefill_opts.js` failed on clean main at 700 tokens: relDiff all-on vs all-off 2.45e-2 over the 2e-2
gate, with argmax, greedy 24 and spec == plain all equal. Swept 18 prompt lengths with `SEQ_ALL=1` (each also run
token by token through decode), relDiff to the token-by-token logits:

| tokens | all-off (16-col batched) | all-on | | tokens | all-off | all-on |
|---|---|---|---|---|---|---|
| 150 | 1.13e-5 | 3.55e-5 | | 701 | 2.11e-4 | 3.65e-4 |
| 300 | 1.75e-5 | 5.10e-5 | | 704 | 2.06e-4 | 3.27e-4 |
| 450 | 1.82e-5 | 2.68e-4 | | 710 | 2.18e-4 | 3.40e-4 |
| 640 | 2.50e-4 | 3.21e-4 | | 730 | 2.38e-4 | 2.74e-4 |
| 670 | 1.60e-4 | **1.61e-3** | | 760 | 2.04e-4 | 3.44e-4 |
| 690 | **8.51e-3** | **8.50e-3** | | 1000 | 3.26e-4 | 2.29e-4 |
| 696 | 1.39e-4 | 4.11e-4 | | 1400 | 7.24e-4 | 4.43e-4 |
| 699 | 1.76e-4 | 3.94e-4 | | 2100 | 3.39e-4 | 3.66e-4 |
| **700** | **2.49e-2** | 3.56e-4 | | | | |

No drift: everything sits at 1e-5..7e-4 and does not grow with length, except isolated single-length spikes, and the
path that spikes changes (670: all-on; 690: token-by-token, the two prefills agree to 4e-4; 700: the all-off baseline).
That is one token's top-8 routing flipping on a near-tie in whichever path, the effect the test's prompt note already
records (1.5e-3 -> 3.0e-2 from a prompt change alone). At 700 the options are right and the baseline is the outlier.
`BATCH_COLS=8` gives the same 2.49e-2, so it is the batched kernels vs decode, not the 16-column grouping. Why 700
moved from v1.0.0's 1.48e-3: #243 (Qwen pre-tokenizer split) changed the fixture's token ids, so these are new prompts.

Fix (test only, no engine change): when on vs off is over the tolerance (or argmax differs), the test runs the prompt
token by token and passes the length if all-on is within the tolerance of that and has the same argmax, printing that
the baseline was the outlier. The 2e-2 gate itself is unchanged, and a real break (0.2..1.0 in past entries) still
fails because it would miss the token-by-token logits too. Also `OPTS=attn,wide,group` (which options the on run uses)
and `BATCH_COLS`. After: MoE default lengths PASS (700 reported as a baseline outlier), 27B PASS (max 1.44e-4), unit
tests 961/961, `npm run check` clean.

## 2026-10-01: `openclaw onboard` health check with a 2-device room (branch openclaw-onboard-health, @pooled/openclaw 0.2.3), GB10

Plain `openclaw onboard` (OpenClaw 2026.9.7, Node 24, clean throwaway profile, plugin linked) → More… → Pooled →
Start a room → lend 4 GB → Qwen3 1.7B → 2 devices → invite link only. OpenClaw then runs its setup check: one
live completion ("Reply with the single word OK. Do not use tools.", tools off, 90 s) in the onboarding process.
Before: the plugin opened a second copy of the room in that process and waited for the other device; the
spinner ("Testing your AI connection…") was still up after 7 min (the video run saw the timeout and the loop back
to the provider picker instead). After: answered at once with the room's state, "Inference verified:
pooled/qwen3-1.7b · AI check: replied in 2.7s", the default model saved, no room opened by onboarding. Then
`openclaw gateway run` opened room 2QQ-7FS, `npx @pooled/cli@0.3.5 join <link> --gb 4` joined, the room went
online (13+embed / 15 layers) and `openclaw agent` answered "2 + 2 = 4." (246 s, 15069 prompt tokens, cold).
Gates: plugin unit tests (new: the check is answered without a room, tools/multi-turn/other text is not the check).
No engine change.

## 2026-10-01: layerFuse comb/kv under attention v2 (branch perf/lf-v2), GB10

Since #294 made attnDecode v2 the default, `layerFuse.comb` and `.kv` switched themselves off on every attention
layer (they wrapped attn_flash + attn_combine), so the MoE's in-process layerFuse gain fell from +4.5% to +3.7%.
New `attn_dec_combine_g` (engine/wgsl/attn_dec.js) is attn_dec_combine with sigmoid_mul folded in: the quotient
O / L goes through workgroup memory, then each of 64 threads per slice multiplies by the gate read from q_full,
the same expressions as sigmoid_mul. With it, `attn_glue_kv` (kv_store in the glue) runs under v2 too: 2 dispatches
fewer per attention layer. Its own module, compiled only when comb is on, optional: a compile failure turns comb off
for v2 only (warning) and the separate sigmoid_mul runs; #300's fallback (any layerFuse kernel failing turns
layerFuse off) is unchanged and skips this kernel.

`tests/bench/lf_ab.js` (Deno, 8 rounds x 24 tokens, median ms/token; new `FLAGS=comb,kv` flips only those):

| | off | on | gain |
|---|---|---|---|
| MoE, branch, all flags (2 runs) | 29.95 / 29.99 | 28.63 / 28.64 | +4.6% / +4.7% |
| MoE, origin/main, all flags (2 runs) | 29.77 / 30.00 | 28.71 / 28.77 | +3.7% / +4.3% |
| MoE, branch, comb,kv only (2 runs) | 28.94 / 28.62 | 28.82 / 28.70 | +0.4% / -0.3% |
| 27B, branch, comb,kv only (2 runs) | 97.76 / 97.72 | 97.39 / 97.56 | +0.4% / +0.2% |
| 27B, branch / main, all flags (one clean run each; the others were contended) | 98.70 / 99.40 | 96.61 / 98.02 | +2.2% / +1.4% |

Small: the "on" arm is 0.1 ms/token (~0.3%) faster than main's on the MoE and the full-fusion gain is back to
main's pre-v2 +4.5%; comb,kv alone sits at the edge of the noise (20 dispatches of a few µs each per MoE token).
Kept because it is free: same bits, two dispatches fewer per attention layer.

Correctness (GB10, Deno): `test_q38_bits` default c26dbc5 / 3177f9f1 with layerFuse on (attn_dec_combine_g built)
and with `LAYER_FUSE=0`; `ATTN_PREFILL_TILE=0` f0537158 / 5d287854: all equal to main. Fallbacks, with a temporary
(not committed) hook in compile.js that throws for a named kernel: attn_dec_combine_g failing -> warning, comb off
for v2, other flags on, c26dbc5 / 3177f9f1; attn_combine_g failing (#300 path) -> layerFuse off, c26dbc5 / 3177f9f1.
`test_moe` MATCH llama.cpp 3/3, spec == plain 3/3, head check 0 mismatches. `test_dense_spec` PASS. lf_ab tokens
identical in every run (under 64 positions: the combine's one-split path; `test_q38_bits`, 300 positions, covers
the multi-split path). Barrier lint
clean, unit tests 961/961, `npm run check` passes. Not measured on the M5 Max or under FXC.

## 2026-10-01: room prompt frames through the prefill kernels (branch perf/cold-turn)

OpenClaw's cold first turn on a room was slow because a chain prefilled in 16-token frames, each through
batchCols-wide passes on every device; the MoE's wide GEMM and expert-grouped prefill ran only in a solo
`prefillTokens`. Now a node host sends frames of up to 256 tokens (`engine.prefillFrame()`) and every device
runs them through the new `prefillHidden` (wide chunks, grouped ubatches, every column read back). The MoE
prefill kernels are on by default on shards without the embedding as well. Spec frames and an older host's
16-token frames still take the old passes; an older worker runs the wide frames NC columns at a time.

TTFT of a real OpenClaw request (captured gateway prompt, paths and host name replaced: 15,846 tokens with the
12 tools; `packages/room-node/test/e2e.mjs cache`, `STEPS=turn1`, 35B MoE, split 19 + 21 layers), GB10:

| setup | main | branch |
|---|---|---|
| solo (one node) | 45.5 s | unchanged (same path) |
| 2 nodes in one process (`SETUP=pair`) | 119.8 s | 77.5 s |
| 2 nodes in 2 processes (`SETUP=proc`, new: the joiner via join.mjs) | 196.3 s | 63.7 s, 51.0 s (2 runs) |
| `SETUP=proc`, `POOLED_PREFILL_FRAME=128` / `512` | | 71.7 s / 49.7 s |

Same answer in every run. Both nodes share one GPU here, so the frames cannot overlap; on two machines the
host's next frame runs while the worker runs the last one. Gates: unit tests (962, incl. a new mock check that
`prefillHidden` keeps every column in place), `npm run check`, barrier lint clean, `test_moe_split` PASS
(now with the prompt frames; plain == solo on all 5 cases incl. the 3,449-token one, spec == plain),
`test_moe` MATCH llama.cpp + spec == plain, `test_dense_spec` PASS, `test_q38_bits` c26dbc5 / 3177f9f1
(unchanged). Not measured on Apple or Windows workers (the frames use the same kernels a solo Mac / PC host
already runs); `POOLED_PREFILL_FRAME=0` restores the 16-token frames.

Not done: a first-start warm-up from a bundled OpenClaw prompt. Two of our own OpenClaw installs differ at
character 2,978 of the system prompt (the tool list), and the prompt holds absolute skill paths, so a bundled
prompt would never be a prefix the checkpoints can resume from: it would cost a minute of GPU for nothing.

## 2026-10-01: dp4a wide prefill GEMM on the 27B (branch perf/prefill-dp4a-wide-gemm-27b, not kept as default: opt-in)

The wide prefill's projections as `dot4I8Packed` on activations quantized per 32 values (llama.cpp's MMQ numerics),
`prefillDp4a: true` / `?dp4a=1`. Fast, but it failed the gate that was set for turning it on: its next-token
log-probabilities vs llama.cpp's top 20 (`tests/test_prefill_dp4a.js`, llama-server goldens refreshed with `n_probs`)
must be no further than the f32 path's + 0.05 nats. So it ships opt-in, and the dense default is the f32 wide GEMM.

| 27B, GB10, Chrome | 16-column | f32 wide (new default) | dp4a | dp4a relDiff vs 16-col |
| --- | ---: | ---: | ---: | ---: |
| 2048 tokens (2 runs) | 82.6 / 83.2 | 102.5 / 103.1 | 186.9 / 188.6 | 2.3e-2 |
| 8192 tokens (2 runs) | 77.3 / 66.4 | 96.2 / 96.4 | 170.9 / 171.2 | 0.21-0.31 |

Max |logprob - llama.cpp| over its top 20 (narrow / wide / dp4a): 27B 150 tok 0.148 / 0.148 / 0.135, 700 tok
0.115 / 0.115 / 0.130, **2100 tok 0.260 / 0.260 / 0.951 (fail)**; MoE 150 tok 0.337 / 0.337 / 0.297, 700 tok
0.214 / 0.216 / 0.264 (fail), 2100 tok 1.99 / 1.86 / 2.17 (fail). Argmax, 32 greedy tokens vs llama.cpp and spec ==
plain matched in every mode. The metric is a max over tail tokens (logprob -13..-17 at 2100), so it is strict; a
probability-weighted gate (KL) might pass, but the rule was fixed before the run and is kept.

Kept from the branch: the f32 wide GEMM as the dense default (ubatch 256), and one multi-column SiLU on the
16-column pass. Gates: unit 967 pass, `npm run check`, barrier lint 0 findings (dp4a kernels included; they are also
optional at create, so a compiler that rejects them turns dp4a off instead of failing the load), `test_q38_bits`
PREFILL_UBATCH=0 c26dbc5 / 3177f9f1 (= main: the SiLU change is exact), ATTN_PREFILL_TILE=0 PREFILL_UBATCH=0
f0537158 / 5d287854, default re-baselined to 4612ece1 / aeb06a4d (the wide prefill; same greedy and spec tokens),
`test_moe` MATCH llama.cpp x3 + spec == plain, `test_dense_spec` PASS, `test_prefill_opts` 27B PASS (relDiff
1.44e-4). `test_prefill_opts` MoE fails at 700 tokens (relDiff 2.45e-2 vs 0.02 tolerance) on main too, same number:
pre-existing, not from this branch.

## 2026-10-02: DP4a accuracy evaluation, pre-registered (branch perf/dp4a-eval, PR #309): dp4a becomes the dense default on NVIDIA

#302 left the dp4a wide prefill opt-in because it failed a max-|Δ logprob| gate over llama.cpp's top 20 (0.951 vs 0.260
nats at 2100 tokens on the 27B), a metric set by a few tail tokens at logprob -13..-17. This run asks what a user would
see instead. The rule was written in the PR description and in `tests/eval_dp4a.js` and pushed (commit 1, PR opened as a
draft) before anything was measured, and was not changed afterwards.

Setup: `tests/eval_dp4a.js`, 15 frozen prompts (`tests/golden/dp4a_eval_prompts.json`, chat template, thinking off):
short (< 64 tokens: English chat, code, Chinese, French, Japanese; under the wide GEMM's 64-column tile, so a control),
mid (238-361 tokens: multi-turn trip plan, email rewrite, Chinese translation), 2K (code, prose), 8K (code, prose).
Reference: CPU `llama-server` (build 749f688, `-t 8`/`-t 10`, n_probs 20), greedy 256 tokens per prompt on the same ids;
ours is teacher-forced on llama.cpp's tokens so all 256 positions compare (goldens in
`tests/golden/dp4a_eval_{27b,2b}_llama.json`). "f32" = the f32 wide GEMM, the dense default until now. Models: every dense
model with the dp4a path (`qwen35` architecture): 27B (q38) and Qwen3.5-2B (q35-2b). The Qwen3 1.7B / 4B run on
`DenseEngine`, which has no wide or dp4a prefill, so the flag cannot change them.

Rule (per model, both must pass; positions = mid + 2K + 8K, 1792 per model): R1 mean KL(llama ‖ ours) over llama's top 20
(renormalized) dp4a <= 2x f32; R2 top-1 agreement (our argmax == llama.cpp's greedy token) f32 - dp4a <= 0.5 pt; R3 per
bucket KL dp4a <= max(3x f32, f32 + 0.01 nats); R4 downstream (40 items scored exactly: 20 lookups in an 80-row table at
~2.3K tokens, 10 word problems 3-shot, 10 "what does this Python print" 4-shot) correct_f32 - correct_dp4a <=
max(2, 1.96 sqrt(b + c)).

| 27B, GB10, Deno | KL f32 | KL dp4a | top-1 f32 | top-1 dp4a | top-5 overlap f32 / dp4a | greedy == llama.cpp (mean tokens of 256) f32 / dp4a | KL(f32 ‖ dp4a) | top-1 dp4a == f32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short (control, 8) | 0.00039 | 0.00039 | 99.27% | 99.27% | 98.43 / 98.40% | 95.6 / 95.6 | 0.00001 | 100.00% |
| mid (3) | 0.00040 | 0.00041 | 98.57% | 98.44% | 98.57 / 98.67% | 86.3 / 66.0 | 0.00003 | 99.87% |
| 2K (2) | 0.00050 | 0.00047 | 98.63% | 98.83% | 98.48 / 98.59% | 129.5 / 114.0 | 0.00015 | 99.41% |
| 8K (2) | 0.00049 | 0.00053 | 99.61% | 99.80% | 98.40 / 98.71% | 152.5 / 240.0 | 0.00021 | 99.80% |
| **gated (mid+2K+8K)** | **0.00045** | **0.00046** | **98.88%** | **98.94%** | 98.49 / 98.66% | 117.6 / 129.4 | 0.00012 | 99.72% |

| Qwen3.5-2B, GB10, Deno | KL f32 | KL dp4a | top-1 f32 | top-1 dp4a | top-5 overlap f32 / dp4a | greedy == llama.cpp f32 / dp4a | KL(f32 ‖ dp4a) | top-1 dp4a == f32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short (control, 8) | 0.00084 | 0.00085 | 98.34% | 98.44% | 98.13 / 98.15% | 74.9 / 76.5 | 0.00000 | 99.90% |
| mid (3) | 0.00080 | 0.00083 | 97.27% | 97.27% | 98.10 / 98.05% | 60.3 / 99.3 | 0.00012 | 99.48% |
| 2K (2) | 0.00093 | 0.00100 | 97.66% | 97.85% | 98.32 / 98.40% | 10.0 / 31.5 | 0.00028 | 98.83% |
| 8K (2) | 0.00095 | 0.00098 | 97.46% | 97.85% | 98.05 / 97.89% | 53.0 / 75.5 | 0.00022 | 99.22% |
| **gated (mid+2K+8K)** | **0.00088** | **0.00092** | **97.43%** | **97.60%** | 98.15 / 98.10% | 43.9 / 73.1 | 0.00019 | 99.22% |

Downstream (correct of 40; lookup / math / code): 27B llama.cpp 33 (20 / 10 / 3, the same items), f32 33 (20 / 10 / 3), dp4a 33 (20 / 10 / 3);
2B llama.cpp 26 (17 / 9 / 0), f32 27 (18 / 9 / 0), dp4a 27 (18 / 9 / 0). f32 and dp4a got exactly the same items right
on both models (b = c = 0).

Verdict: **PASS on both models, all four rules.** 27B: R1 0.00046 <= 2 x 0.00045; R2 -0.06 pt (dp4a higher); R3 mid
0.00041, 2K 0.00047, 8K 0.00053 (3x f32 would also pass); R4 33 vs 33. 2B: R1 0.00092 <= 2 x 0.00088; R2 -0.17 pt; R3
0.00083 / 0.00100 / 0.00098; R4 27 vs 27. So `DP4A_DEFAULT.dense = true` (NVIDIA only via `dp4aAutoDevice`, never Apple;
the compile fallback that turns dp4a off when its kernels do not build, e.g. FXC, is unchanged; MoE stays opt-in).

Notes. The KL divergence dp4a adds against f32 (0.0001-0.0003 nats) is several times smaller than the gap both have to
llama.cpp (0.0004-0.0010): dp4a moves the distribution less than the f32 path already differs from the reference.
Greedy agreement is noisy (one near-tie flips the rest of a run) and is reported, not gated: dp4a stays on llama.cpp's
path longer on average for both models, as expected from the same Q8 activation numerics llama.cpp uses on CPU and GPU.
The short bucket is identical between modes except `code-sql` (65 tokens with the template, so 64 of them go through the
wide GEMM). top-1 counts our argmax against llama.cpp's sampled token; with `ignore_eos` llama.cpp samples the runner-up
where `<|im_end|>` is its argmax (2B mid-email: 8 positions), so a second column, our argmax vs llama.cpp's argmax, is in
the script output (27B gated 99.16 / 99.22%, 2B 98.16 / 98.33%). #302's metric is still printed by
`tests/test_prefill_dp4a.js` (now reported, not gated): 2100 tokens 0.2599 / 0.2599 / 0.9510, same as #302; on this set
the first position's max |Δ logprob| on the 27B's 2K / 8K prompts is 0.174-0.337 for dp4a against 0.182-0.316 for f32, so the
2100-token number is one position's tail, not a trend.
Prefill in the eval (27B, Deno, includes the first logits): 2K 19.7-20.0 s f32 vs 10.7-10.9 s dp4a, 8K 82.2 / 82.5 s vs
46.0 / 46.7 s.

Gates: `test_q38_bits` default re-baselined to **4dc814b1 / 4f075117** (dp4a; same greedy and spec tokens),
`PREFILL_DP4A=0` 4612ece1 / aeb06a4d (= the f32 default before), `PREFILL_UBATCH=0` c26dbc5 / 3177f9f1 (unchanged);
`test_prefill_dp4a` 27B PASS (argmax, 32 greedy tokens == llama.cpp and narrow at 150 / 700 / 2100, spec == plain);
unit 968 pass; `npm run check`; `tests/run.sh q38once` PASS (9 checks incl. MTP spec, split, twins, ctx).

## 2026-10-02: OpenClaw's first answer after a restart from disk checkpoints (branch perf/ckpt-persist), GB10

Every device of a room-node room keeps its part of the pinned prefixes (OpenClaw's system prompt + tools, and its
STABLE cache boundary) on disk (`packages/room-node/ckptdisk.js`, `~/.pooled/cache/ckpt`); when a restarted room's
deal is whole, each device reads them back into GPU slots before the room goes online. Recorded OpenClaw request
(11,286 tokens with its tools), `packages/room-node/test/e2e.mjs cache`, `STEPS=turn1`, 35B MoE:

| setup | cold first turn (= every restart on main) | after a restart, with the cache |
|---|---|---|
| 2 nodes in 2 processes (`SETUP=proc`, split 19 + 21) | 52.5 s, 34.5 s (cache on); 41.6, 34.4, 42.3 s (`POOLED_CKPT_DISK=0`) | **1.26 s, 1.20 s** (231 tokens read; restore 0.4-0.8 s while the room comes online) |
| solo (`SETUP=solo`) | 31.0 s | **2.2 s** (restore 1.7 s) |

Same answer: turn1 identical in every run, and the full six-step `cache` run after a restart matched the
`POOLED_CKPT_DISK=0` run step for step (texts and token counts). Cold turns with the cache on are within the run-to-run
spread of the cache off (both nodes share one GPU here). On disk: two prefixes, ~0.6 GB for the room (host ~330 MB,
worker ~285 MB; solo 615 MB), written after the pinned save, atomically, 0600.

## 2026-10-02: MoE decode, where a token goes and what was cut (branch perf/moe-decode), GB10

Question: plain decode of the 35B MoE ran ~50-56 tok/s in Chrome at 1K context against llama.cpp CUDA's 85.2 tok/s
(`llama-bench -p 0 -n 128`, build 749f688, same GGUF; 11.7 ms/token). Where does a token go?

### Profile (main e45d9a8, Chrome GB10 unless noted)

New probes: `tests/prof/decode_gap.js` (GPU span and idle of every submit, CPU encode, back-to-back GPU time;
`bench.html ?ctx=1024&gap=1`, `tests/prof/gap_deno.js`), `tests/prof/skip_cost.js` (what each kernel family
costs inside the real one-pass token: back-to-back GPU time with the family skipped), `prof_ts.js SPEC=1`.

| | plain token, 1K | spec step K=3, 1K (2.3-2.5 tokens) |
|---|---|---|
| wall | 17.7 ms | 37.4 ms |
| GPU busy (sum of submit spans) | 15.7 ms | 33.5 ms: verify 30.1-31.2, DeltaNet replay 0.93, draft refill 0.55 |
| GPU idle between submits | 1.8 ms (the readback, JS, next submit) | 3.5-4.8 ms |
| submits / readbacks | 1 / 1 | 3.8 / 1 |
| CPU encode | 0.07 ms (Deno 0.9) | 0.5-0.6 ms |
| dispatches / compute passes | 342 / 1 | ~540 / 2 + 3 per layer |
| back-to-back GPU time (no readback between tokens) | 16.1 ms | |

In-situ cost per family, plain token (Deno, back-to-back 15.4 ms/token, `skip_cost.js`, after the DeltaNet change below):

| family | launches | ms/token | us each | bytes read | GB/s |
|---|---|---|---|---|---|
| moe_gus (8 routed experts + shared, gate/up + SiLU) | 40 | 2.76 | 69 | 11.65 MB | 169 |
| matvec_q4_coop_cv ([qkv \| z] + conv) | 30 | 2.69 | 90 | 14.2 MB | 158 |
| matvec_q8_coop (LM head 540 MB ~2.5 ms + 4 Q8 attention q) | 5 | 2.88 | | | ~215 (head) |
| moe_dnc (down + combine + residual), Q4 / Q8 | 35 / 5 | 1.59 / 0.37 | 45.5 / 73 | 5.8 MB / 10 MB | 128 / 137 |
| dn_delta_gnp (DeltaNet, 2 MB state read + written) | 30 | 1.11 | 37 | 4 MB | 108 |
| out projections Q4 / Q8 (+ residual) | 26 / 14 | 0.87 / 0.77 | 33.5 / 55 | 4.7 / 8.9 MB | 141 / 162 |
| moe_nrt (norm + router) / moe_route / dn_nba | 40 / 40 / 30 | 0.59 / 0.55 / 0.26 | 14.6 / 13.8 / 8.6 | | latency |
| attention (attn_dec, glue_kv, combine_g, Q4 q, Q8 [k \| v]) | | 1.06 | | | |
| head norm, top-k | | 0.10 | | | |

The GB10 streams ~210 GB/s through WebGPU at best (the LM head reaches it); at that rate the ~2.15 GB a token reads
would take ~10.3 ms. The rest is (a) GEMVs over 2048- and 4096-wide rows reaching 130-170 GB/s (one Q4 block per
thread, then an 8-level workgroup tree per row: short rows pay the tree on little data), (b) ~110 latency-bound
small launches a token (router, route, DeltaNet input norm: ~1.4 ms), (c) the DeltaNet update, and in Chrome (d) the
1.8 ms round trip between tokens.

llama.cpp, `nsys --cuda-graph-trace=node` over `llama-bench -n 32` (33 tokens: per token, its own profiler
overhead included): expert gate/up `mul_mat_vec_q<Q4_0, ids>` 40 x 57.8 us = 2.31 ms (ours 2.76), expert down 35 x 28.9 us
(ours 1.96 with the shared expert and combine), the Q4 projections 66 x 34.4 us = 2.27 ms (ours ~3.8 ms over the same
weights), the Q6_K head ~2.0 ms (ours 2.5 ms from Q8), `gated_delta_net` 30 x 4.5 us = 0.13 ms (ours 1.11 ms: one warp per
state column over a transposed state, rows across lanes and a warp shuffle sum), `topk_moe` 3.7 us, `rms_norm` 3.1 us, plus
`quantize_q8_1` 291 x 2.2 us = 0.63 ms that we do not pay (it multiplies Q8_1 activations with dp4a).

### Kept (every change keeps main's bits)

1. DeltaNet one-token update in two chunked sweeps (`dnTwoPass`, dn_delta / dn_delta_gn / dn_delta_gnp). The state
   column sat in 128 registers loaded and stored as one straight line of 128 accesses, which the GB10 runs nearly
   serially: 39 us for the 32 heads in isolation against 14 us for two sweeps of 16 rows (the second hits L2). In
   the model dn_delta_gnp 54.4 -> 30.6 us (prof_ts), -0.7 ms a token.
2. `verifyPass`: the fused verify's 40 layers and head in one compute pass instead of three per layer. Chrome in-tab
   A/B at 1K: spec 60.0 -> 61.8 tok/s.
3. `dn_pre_mc`'s q/k norms with 16 loads in flight: 39.8 -> 29.7 us per layer per verify (-0.3 ms a step).
4. Decode-ahead in `forwardTokenIds` (greedy, from the second call in a row): token N + 1 is queued behind token N,
   its embedding gathered on the GPU from N's top-1, after a save of the DeltaNet states, conv windows and x; the
   next call keeps it if it gets that top-1, anything else puts the state back. Chrome in-tab A/B at 1K: plain
   59.2 -> 63.8 tok/s (+7.7%); Deno 28.07 -> 18.55 ms/token. The room's solo MoE path speculates and does not use it.

### Before / after

Chrome (`chrome_bench.mjs <moe> 64 "ctx=1024,8192,32768"`: a chat turn over this repo's source with a question about it,
GPU sampling, greedy, K=3; median of 4 main / 3 branch runs; runs that overlapped another agent's GPU job, visible as
25-30 tok/s, are dropped). The decode-ahead-off column is the branch with `set={"decodeAhead":false}`, the kernel and
pass changes alone (one run).

| context | main plain | branch plain | branch, decode-ahead off | main spec | branch spec |
|---|---|---|---|---|---|
| 1K | 55.1 | **63.9** (+16%) | 58.8 (+7%) | 60.6 | 59.5 |
| 8K | 52.0 | **59.6** (+15%) | 55.3 (+6%) | 56.7 | 57.0 |
| 32K | 45.0 | **50.1** (+11%) | 47.3 (+5%) | 47.7 | 47.8 |

Acceptance 39/81, 36/81, 42/72 in every run, spec == plain in every run. Speculative decoding does not move outside the
run-to-run noise (+-2 tok/s here): the verify pass is the batched GEMVs and expert kernels, which this branch does not
change; the in-tab A/B above sees verifyPass at +3%.

Deno (`bench_ctx.js`, `CTX_SRC` = the main checkout for both, one run each). It decodes through `forwardToken` and
reads back the logits, so decode-ahead does not apply; Deno pays ~11 ms per readback.

| context | main plain | branch plain | main spec | branch spec |
|---|---|---|---|---|
| 1K | 34.23 | 34.85 | 43.92 | 43.73 |
| 8K | 32.97 | 34.21 | 43.46 | 45.45 |
| 32K | 29.15 | 30.34 | 42.39 | 43.89 |

Deno through `forwardTokenIds` (`tests/bench/decode_ab.js`, decodeAhead off / on in one process): 28.07 -> 18.55 ms/token.
Back-to-back GPU time per token (Chrome, 1K): 16.1 ms on main, 14.8 ms on the branch.

### Tried, not kept

- DeltaNet value head over 2 / 4 workgroups (GDN-5, `dnSplit`): same bits, -1.2% (per-thread latency bound, not
  occupancy bound: the 32-thread groups took as long as the 128-thread ones).
- `dn_delta_mc2`, the two-sweep DeltaNet for verify passes of <= 4 columns: 59 -> 41 us at 4 columns in isolation, but
  +4.7 ms GPU per speculative step in the model (33.5 -> 38.3 ms): the per-column re-reads miss L2 there.
- The [qkv | z] GEMV's conv weights and state loaded before the dot products: no faster (102 vs 90 us) and it moved
  the MoE bits (937cf4e4 vs 864fb862): the epilogue's contraction changed.
- dn_pre_mc with the heads staged in workgroup memory: 42 us (bank conflicts) vs 29.7 us for 16 loads in flight.
- coopRows 8 (bit-neutral): 15.11 vs 15.23 ms/token back to back, inside the noise.

### Not done: needs a pre-registered accuracy rule (bits change)

The remaining gap sits in kernels whose summation order is the bit contract: short-row GEMVs (qkvz 158 GB/s, out
projections 141-162 GB/s; the verify's 4-column twins 92-118 GB/s: matvec_q8_coop_b4 194 us, matvec_q4_coop_b4 120 us),
the expert kernels (169 / 128 GB/s), a native Q6_K LM head and attention q (417 instead of 540 MB for the head), and
a DeltaNet update that splits a column's sums across threads (llama.cpp 4.5 us vs 37 us). Together roughly 3-4 ms of
a 15 ms token on these numbers. The spec step's own overhead (replay 0.9 ms, refill 0.6 ms, ~4 ms idle around one
readback) needs acceptance on the GPU (SPEC-2).

Gates: `test_moe` MATCH llama.cpp 3/3, spec == plain 3/3, GPU sampling head check 0 mismatches; `test_moe_split` PASS;
`test_dense_spec` PASS; `test_q38_bits` 4dc814b1 / 4f075117 on main and the branch, also with `LAYER_FUSE=0` (the
dn_delta_gn path); the new MoE fingerprint (`MODEL=moe test_q38_bits.js`) 864fb862 / 46e62439 on main and the branch;
unit tests 1024/1024 (with the FXC barrier lint), `npm run check`. On the GB10 (`E2E_GPU=real`): `gpusample_synth` PASS
with a new decode-ahead check (tokens other than the top-1 and a `forwardToken` in between give exactly decodeAhead
off); `fusion_synth` 0 logits differ for every layerFuse row (its attn_glue row fails on a real GPU on main too);
`dn_delta_synth` single-token dn_delta bit-identical to the reference (its "nCols 8, replay" row fails the same way on
main since #295 records the replay inputs). 2-device room, `xroom.mjs` host + guest on this GB10 (19 + embed / 21
layers, twosum and japan, plain and spec, 2 rounds each): answer sha-256 identical to main in all 8 cells. Not measured
on the M5 Max (the two-sweep DeltaNet must keep Metal's bits too) or under FXC beyond the lint.

## 2026-10-02: MoE prefill 2x (branch perf/moe-prefill, PR #314), GB10

The 35B-A3B MoE prefilled at 350-400 tok/s in Deno against llama.cpp CUDA's ~2,400 on the same GGUF
(`llama-bench -p 512,2048,8192,16384 -n 0 -fa 1`, build 749f688, measured here: 2354 / 2411 / 2347 / 2285).
OpenClaw's prompt pays that on every cold first turn.

**Profile before** (`tests/prof_prefill.js`, timestamps per sampled dispatch, `PREFILL_UBATCH=256`, main's defaults; ms per
token of kernel time): 2048 tokens: expert gate/up (`moe_gusg`) 0.565, expert down 0.380, DeltaNet projection GEMMs 0.407,
draft-cache (MTP) fill 0.180, attention 0.149, attention projections 0.106, `dn_delta` 0.097, DeltaNet glue 0.095, `moe_nrt`
0.087, `moe_gsort` 0.082, the rest under 0.03. At 8192 tokens attention rises to 0.353 (14%). 43,374 dispatches per
2048-token prefill. A skip-a-family profile (`tests/bench/moe_prefill_ab.js SKIP=1`) overstated some families: skipping
`moe_gsort` "saved" 30% because the expert launches then run on a stale chunk list; launching a kernel twice
(`DOUBLE=`) gives its real marginal cost (`moe_gsort` 3%).

**Kept** (each measured on / off in one process at the end state, `tests/bench/moe_prefill_ab.js`, Deno, 2048 / 16384 tokens):

| change | off | on | bits |
| --- | ---: | ---: | --- |
| dp4a expert kernels (`moe_gusq` / `moe_dnq` + `moe_qx`, `moeGroupDp4a`; wide chunks only) | 588 / 492 | 860 / 675 | evaluated (below) |
| draft-cache fill with wide GEMMs, KV rows only (`mtpWide`) | 759 / 588 | 860 / 675 | same logits |
| the wide chunk's router once per chunk (`wideRouter`) | 789 / 627 | 860 / 675 | same |
| DeltaNet conv / gates / recurrence / gated norm once per chunk (`wideDn`) | 799 / 634 | 860 / 675 | same |
| dp4a projections for the MoE too (`DP4A_DEFAULT.moe`; A/B at the dp4a-experts commit) | 631 / 586 (2048 / 8192) | 731 / 680 | evaluated |
| `moe_gsort` in O(pairs / 256) steps (atomic histogram, per-column bitmask placement) | 0.53 ms | 0.04 ms a launch | same output (CPU test) |
| ubatch 256 -> 512 (`MOE_PREFILL_UBATCH`) | 803 / 732 (2048 / 8192) | 859 / 791 | same |
| tiled attention: next K / V tile prefetched to registers, f16 tile, named registers | 861 / 640 (2048 / 16384) | 869 / 674 | same (CPU test; 27B bits) |

The dp4a experts keep each thread's weight blocks as packed i8 in registers and dot4I8Packed them against activations
quantized per 32 values (llama.cpp's MMQ numerics); the f32 tiled kernels unpacked every block once per pair. Kernel
microbench (`tests/bench/moe_group_sweep.js`, synthetic, 256 tokens): gate/up 4.41 -> 2.27 ms, down 3.01 -> 1.57 ms, plus
0.63 ms quantization. They run only inside wide chunks (>= 64 tokens), like the dp4a projections: test_moe's 23-27-token
prompts (batchCols 4, so an expert-grouped ubatch without a wide chunk) flipped a near-tie against the llama.cpp golden with
them on (hash-map, token ~20), and with them off there those prompts compute exactly what main does.

**Accuracy (pre-registered, `tests/eval_dp4a.js MODEL=moe`; rule committed in ed19a35 before any MoE dp4a measurement).**
llama.cpp CPU goldens (`tests/golden/dp4a_eval_moe_llama.json`), 15 prompts, 256 teacher-forced positions each:

| 35B-A3B, GB10, Deno | KL(llama ‖ ours) wide | dp4a | dp4ax | top-1 wide | dp4a | dp4ax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short (control) | 0.00214 | 0.00223 | 0.00223 | 98.68% | 98.68% | 98.58% |
| mid | 0.00154 | 0.00194 | 0.00178 | 98.44% | 98.44% | 98.44% |
| 2K | 0.00313 | 0.00291 | 0.00325 | 98.44% | 98.63% | 97.85% |
| 8K | 0.00349 | 0.00362 | 0.00360 | 97.66% | 97.07% | 97.85% |
| **gated** | **0.00255** | **0.00270** | 0.00272 | **98.21%** | **98.10%** | 98.10% |

Downstream 40 items: llama.cpp 36, wide 36, dp4a 36 (the same items), dp4ax 35. R1 0.00270 <= 2 x 0.00255, R2 -0.11 pt,
R3 0.00194 / 0.00291 / 0.00362 against 0.0115 / 0.0131 / 0.0135, R4 36 vs 36: **PASS**, so `DP4A_DEFAULT.moe = true` with
the dp4a experts. KL(wide ‖ dp4a) 0.0019 nats, smaller than either path's distance to llama.cpp. (The run used the
experts in grouped tails too; the shipped scope is narrower.)

**Before / after**, same bench script, two processes each, alternating (Deno, GB10, tok/s; main = e45d9a8):

| prompt tokens | 512 | 2048 | 8192 | 16384 |
| --- | ---: | ---: | ---: | ---: |
| main | 353 | 402 | 379 | 337 |
| branch | 623 | 861 | 809 | 675 |
| llama.cpp CUDA (llama-bench) | 2354 | 2411 | 2347 | 2285 |

OpenClaw cold first turn (`packages/room-node/test/e2e.mjs cache STEPS=turn1`, the recorded gateway request: 11,286 prompt
tokens with its tools, 35B MoE, Node + Dawn; same answer in every run):

| setup | main | branch |
| --- | ---: | ---: |
| solo (`SETUP=solo`, one node) | 30.9 s (30.9 s) | 14.7 s |
| 2 nodes, 2 processes (`SETUP=proc`, split 18 + 22) | 33.7 s | 18.7 s (20.0 s with ubatch 512, before the attention change) |

**After** (`prof_prefill.js`, ms per token at 2048 / 8192): DeltaNet projection GEMMs 0.211 / 0.211, expert gate/up 0.175,
down 0.145, attention 0.131 / 0.318 (29% at 8K), `dn_delta` 0.074, `moe_nrt` 0.051, draft fill 0.047 / 0.012, attention
projections 0.046; 107 dispatches per 16 tokens (was 317).

**Gates** (final snapshot): `test_q38_bits` 4dc814b1 / 4f075117 (= main); `test_moe` MATCH llama.cpp 3/3, spec == plain,
acceptance 28/33, 28/39, 25/45 (= main); `test_moe_split` PASS (copy: split plain differs from solo at a 0.075 near-tie
over the f16 wire; with the dp4a experts off it matches, as on main; the other 4 cases plain == solo);
`test_prefill_opts MODEL=moe` PASS (700 tokens: the all-off path is the outlier, as on main); unit 1029 pass;
`npm run check`; barrier lint 0 findings.

**Tried, not kept**: `moe_nrt_w` (the router GEMV for 3 columns per workgroup, exact): 859 vs 860 / 675 vs 672 tok/s.
dp4a expert chunks of 16 / 32 pairs: slower than 8 (synthetic, 256 tokens: 4.54 / 5.53 vs 3.84 ms). dp4a GEMM tiles BM 128
(TM 8, or BN 32): slower than 64 x 64. ubatch 1024: +3% over 512 at 2048 / 8192 tokens for twice the wide buffers.
Attention split targets 64 / 128: no change. One wide-chunk attention launch with the per-sub-batch split boundaries (exact):
its split partials would take ~0.5 GB at 16K. Attention is the open item: at 8K one 16-column pass takes 1.5 ms (~1.5 TFLOPS);
dropping its QK FMAs saves 0.42 ms and its PV FMAs 0.27 ms, so the tile loads, four barriers per 8 positions and the softmax
are over half. TK > 8 (fewer barriers per position) needs more than 16 KB of workgroup memory or another partial-sum layout,
and changes the numerics. Then `dn_delta` (sequential over the chunk) and the DeltaNet projection GEMMs.
Windows / Metal (after merging #313 / #315): every pipeline of a worker shard (layers 20-22) and a host shard with the head
(0-2) compiles under FXC on the RTX 5070 PC (Dawn D3D12, d3dcompiler_47 only; `packages/room-node/test/compile_check.mjs`,
322 / 329 pipelines, 0 failures), the dp4a experts (`moe_gusq` / `moe_dnq` q4 and q8, `moe_qx`) and the f16-prefetch
attention tile included; the dp4a fallback was not needed. A real join of that PC into a Spark-hosted room (Spark 0-20 +
embed / head, PC 21-39 over WebRTC) answered OpenClaw's request (11,286 tokens) in 11.5 s TTFT (prefill 11.3 s). On the
M5 Max (Metal, Deno) dp4a stays off (`prefillDp4a` and `moeDp4a` false), `test_q38_bits MODEL=moe` gives e1b27cc1 /
1a71afaf on both main and this branch (the per-chunk router / DeltaNet, the KV-only draft fill and the attention tile
change no bit there), spec == plain, and prefill goes 247 / 266 / 256 -> 284 / 309 / 302 tok/s at 512 / 2048 / 8192.
After the merge on the GB10: `test_q38_bits` 4dc814b1 / 4f075117, `test_moe` MATCH 3/3 with the same acceptance,
`test_moe_split` PASS as before.

## 2026-10-02: Qwen3.5-122B-A10B loads and matches llama.cpp (branch feat/qwen35-122b, phase 1), GB10

Model: bartowski `Qwen_Qwen3.5-122B-A10B-Q4_0` (two-file split GGUF, 72.5 GB; the 35B's quant mix: Q4_0 experts, Q4_1
down experts on layers 0-5, Q8_0 / Q5_0 shared experts, Q6_K head, F32 routers). Same hybrid as the 35B, larger: 48
layers (+ MTP), dim 3072, 256 experts top-8 of width 1024, 64 DeltaNet value heads, 32 query heads on 2 KV heads.

What it needed: (1) 16 query heads per KV head, past the flash / `attn_dec` kernels' 8. Each KV head now gets
R = ceil(G / 8) workgroups of G / R heads (`dims.attnR`, `nKVa = nKV * R`); R = 1 on the 27B and the 35B, so their
kernels and bits are unchanged (the tiled prefill attention already took G up to 64). (2) Split GGUFs in the Deno
loader (`tests/load_model.js openGGUF` reads every shard; the room's loader still reads one file, so the model is not in
the picker or the ?dev=1 list). (3) A streamed load (`streamWeights`: a layer at a time, each matrix uploaded as it is
converted), since the weights do not fit twice in 121 GB. Engine GPU memory ~70 GB; 133 s from disk (no weight cache).

| | engine (Deno, GB10) | llama.cpp 749f688 CUDA |
|---|---|---|
| decode, plain | 24.9 tok/s (512 ctx), 24.2 (8K) | 27.8 (tg64) |
| decode, spec K=3 (MTP) | 30.1-35.5 tok/s, acceptance 27-31 of 33-36 | |
| prefill | 353 tok/s (512), 349 (2K), 292 (8K), 264 (12K agent prompt) | 766 (pp512) |

Correctness: `MODEL=122b tests/test_moe.js` MATCH llama.cpp (CUDA llama-server, greedy, same ids) 3/3 over 40 tokens,
token ids included, spec == plain on all three; also with `ATTN_DECODE=v1 ATTN_PREFILL_TILE=0` (attn_flash everywhere).
Gates: `test_q38_bits` 4dc814b1 / 4f075117 (= main), 35B `test_moe` MATCH 3/3 with main's acceptance (28/33, 28/39,
25/45), unit 1029 pass, `npm run check`.

Routing traces for the expert-offload design: `tests/trace_moe.js` (engine option `moeTrace`: per-layer selection
buffers, `readMoeTrace()`). Per MoE layer one expert is 5.06 MiB (gate + up + down, Q4_0; 5.25 MiB on layers 0-5);
always-resident weights are 53-74 MiB per layer, 3.0 GiB for the 48 layers, against 61 GiB of routed experts. On chat,
code, a 12K-token OpenClaw-like agent turn and a 4-turn chat, a per-layer LRU of 32 / 64 / 128 experts hits 53-58% /
69-71% / 85-86% of the decode-time picks (cold misses included).

## 2026-10-02: expert offload, phase 2a (branch feat/expert-offload, on feat/qwen35-122b), GB10 + RTX 5070 PC

Selected MoE layers keep their 256 routed experts out of the GPU's resident weights (`engine/expert_store.js`): parked
in `MAP_WRITE | COPY_SRC` buffers (system RAM on a discrete GPU), cached per layer in a VRAM slot pool (LRU, equal slots
per layer from the budget), plus one whole-layer region for prefill frames. Decode and verify cut each offloaded layer
at `moe_route`: the selection is read back, the store copies the misses into the pool and the CPU writes the slot ids
in place of the expert ids, so `moe_gus` / `moe_dnc` run unchanged on the pool (no new kernel, no extra binding: the
offloaded engine gives the resident engine's bits). Prefill frames copy the layer (or, under 32 tokens, the experts the
frame chose) into the region and run the grouped kernels there. Load: `qwen35Weights(..., { experts: store })`; Deno
tests: `OFFLOAD=lo-hi|all OFFLOAD_GB=8 | OFFLOAD_SLOTS=N` (`tests/load_model.js offloadFromEnv`).

Correctness. Bits: 35B `MODEL=moe test_q38_bits` with every layer offloaded at 16 slots (pool, region fallback in
verify, grouped prefill on the region) and with layers 0-19 at 64 slots: 5ef77d06 / 403ae12b, the resident engine's
(and phase 1's). Off: `test_q38_bits` 4dc814b1 / 4f075117, 35B `test_moe` MATCH 3/3 with phase 1's acceptance.
On: 122B `MODEL=122b test_moe` MATCH llama.cpp 3/3 (token ids) and spec == plain with layers 17-47 offloaded at 8 GB
(Deno and Dawn) and with all 48 at 8 GB (Dawn; includes layers 0-5's Q8-widened down experts); 35B MATCH 3/3 and
spec == plain with all 40 layers offloaded (Deno 16 slots; Dawn 4 GB on the GB10, on the RTX 5070 and on the PC's
Intel iGPU, both D3D12 + FXC). The store's LRU replayed on the 122B traces (layers 17-47, 50 slots) hits 78.9 / 80.8 /
75.8 / 79.8% (chat / code / agent / multiturn) vs the feasibility study's per-layer simulation 78.6 / 80.5 / 75.4 / 79.4.

The cut needs a fast readback: Deno's mapAsync costs ~14 ms (122B, 31 cuts: 2.2 tok/s), Dawn's ~0.2 ms. Perf numbers
are Node + Dawn (`tests/dawn_run.mjs`: a Deno GPU test under Node on dawn.node, as room-node runs).

GB10, Dawn (Vulkan), 122B, engine defaults (test_moe: decode over 40 tokens incl. the cold start; spec K=3):

| | decode tok/s | spec tok/s | cache | MiB copied / token |
|---|---|---|---|---|
| resident | 24.7-26.3 | 30.8-40.2 | | |
| layers 17-47 offloaded, 8 GB (43 slots + 1.27 GiB region) | 18.6-20.3 | 25.4-29.3 | 72-75% hits | 325-363 |
| all 48 offloaded, 8 GB (25 slots + 1.64 GiB region) | 16.3-17.2 | 19.4-22.0 | 51-60% hits | 867-1055 |

RTX 5070 PC (D3D12 + FXC, Node 24 + dawn.node 0.6.1, `--no-maglev`), 35B-A3B with all 40 layers offloaded (it does not
fit the 12 GB card resident; 17.5 GiB parked, process RSS 20.4 GB): test_moe at 4 GB (50 slots) decode 45.4-46.5 tok/s,
spec 54-71, 70-75% hits. One decode token (4 GB, 40 cuts): 18.5 ms, of which the cuts' resolve 16.5 ms (waiting for the
readback, which includes the GPU work of the segment, 8.8 ms; JS plan + encode + submit ~0.19 ms per cut).
`tests/bench/offload_bench.js` (room preset, 128 decode tokens after a code prompt):

| 35B on the 5070, all offloaded | prefill 512 / 2048 tok/s | decode tok/s | hits | MiB copied / token |
|---|---|---|---|---|
| 4 GB (50 slots) | 268 / 540 | 38.1 / 37.6 | 69% | 179 |
| 8 GB (109 slots) | 252 / 546 | 43.9 / 54.7 | 89-91% | 53-60 |

GB10, Dawn, 122B, `offload_bench.js` (room preset):

| 122B on the GB10 | prefill 512 / 2048 tok/s | decode tok/s (after 512 / 2048) | hits | MiB copied / token |
|---|---|---|---|---|
| resident | 234 / 373 | 25.1 / 24.8 | | |
| layers 17-47 offloaded, 8 GB | 161 / 268 | 20.3 / 19.4 | 77% / 68% | 290 / 402 |
| layers 17-47 offloaded, 4 GB (17 slots; engine defaults) | 165 / 230 | 17.7 / 17.4 | 51% / 45% | 622 / 691 |

Prefill copies each offloaded layer whole into the region per frame (a 2048-token prompt: 155 layer loads, 196 GiB on
the GB10's unified memory); on the PC that is ~0.9 s per 512-token frame for the 122B's 31 layers.

Not in this phase: the room (`room/plan.js` dealing layers to an offload device by resident VRAM + RAM for experts,
room-node's shard range and `--no-maglev`, the room loader reading split GGUFs), native Q4_1 down experts (layers 0-5 of
the 122B still widen to Q8), seeding the LRU from the prompt, and an offload path in Chrome (its readback latency is not
measured). Encode-ahead and decode-ahead are off on an offloaded engine (a token's commands depend on its routing).

## 2026-10-02: expert offload in rooms, phase 2b (branch feat/offload-rooms, on feat/expert-offload), GB10 + RTX 5070 PC

A room node with a discrete GPU offers RAM for a MoE model's experts (`--ram`, default total less 16 GB); when the
pledges can't hold the model, the host deals it layers past its pledge with their experts parked (room/plan.js
dealRoom, room-node shard.js). Answers are the CLI's `pooled chat` against the room; tok/s is the host's status line
(the room's own measure of its last answer, MTP speculation on).

| room | split | offload | answers | tok/s (two-sum / hash-map / bash / 250-word story) |
|---|---|---|---|---|
| PC alone, `pooled host qwen3.6-35b-moe --here` (lends 10 GB) | PC 0-39 | 40 layers, 17.5 GiB parked, 60 slots | two-sum, hash-map = the 35B goldens | 54.3 / 53.8 / - / 40.9 |
| Spark `--gb 6` hosts, PC `--gb 8` joins, 35B | Spark 0-7, PC 8-39 | PC: 11-39, 12.2 GiB parked, 71 slots | same | 43.4 / 36.7 / - / 34.1 (an earlier run 49.1 / 42.7 / - / 27.5) |
| Spark `--gb 26` hosts, PC `--gb 10` joins, 122B (128K context) | Spark 0-14, PC 15-47 | PC: 15-47, 41.8 GiB parked, 26 slots | the 3 llama.cpp goldens (q122_llama_greedy.json), text prefix | 19.0 / 15.2 / 15.4 / 12.0 |

PC memory with the 122B share: process working set 43.3 GB (peak 45.4), private 54.4 GB, 15 GB of the 63.5 GB free;
VRAM 10.6 of 12.2 GB. Load from disk 110 s. Three things had to change on the way (all in this branch):
- **Streamed loads kept every byte they fetched until the end** (source.js rangePrefetcher: a dropped chunk's settled
  promise still held its bytes). Streaming the 122B share from Hugging Face reached 87 GB private / 1 GB free commit
  at 85% before I stopped it. Fixed: on the GB10 a streamed 11-layer share peaks at 19.6 GB RSS instead of 32.4.
- **Parked buffers were 22% empty**: a slice that did not fit the rest of a 1 GiB buffer started a new one (402 MB
  slices: 2 per GiB). 50 GiB of upload heap for 41.8 GiB of experts; D3D12 then failed CreateCommittedResource. Now a
  slice spans buffers at an expert boundary: 42 GiB for 41.8. (A bare test allocated 60 x 1 GiB MAP_WRITE buffers on
  the PC without an error, Windows paging the working set out at the end: the limit is RAM, not a heap cap.)
- **An out-of-memory buffer in a load came online as garbage** ("!!!!" answers). shard.js runs the load in an
  out-of-memory error scope and fails it.
Streaming the share from Hugging Face (74-110 MB/s, then 16 MB/s) still peaked at 57 GB working set at 94% with the
prefetch fix (the watchdog stopped it at 1.8 GB free RAM): ~14 GB more than from disk. Not found yet; from disk is fine.

Chain host offload: an offloading **host** in a chain (node-a 0-28 offloaded + head, node-b 29-39) answered nothing:
the room's draft fill (pipeline.js fillDrafts: setHidden, mtpRun, _mtpFillBatch writing B.x) ran between the parts
of a trunk call split at a cut and overwrote its scratch. The engine now has a gate (offGate: from a cut's submit
until the call is through); the fill waits for it. Same answer as the resident host (35B, 900-token prompt: identical
48 tokens, 39.1 vs 41.8 tok/s), and the room page as host dealing offload to a node (tab 0-16 + head, node 17-39 with
29-39 offloaded): matches solo, 53.7 tok/s.

Gates: `test_q38_bits` 4dc814b1 / 4f075117; 35B `test_moe` MATCH 3/3, spec == plain (Deno, and Dawn with all 40
layers offloaded at 4 GB); `MODEL=moe OFFLOAD=all OFFLOAD_SLOTS=16 test_q38_bits` 5ef77d06 / 403ae12b (= resident).
