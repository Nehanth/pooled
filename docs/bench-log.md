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

### M5 Max (Metal): the MoE engine defaults now run under Deno; speed not measurable this session

The Mac was heavily loaded by other processes during the whole session. NVIDIA Sync used 240-290% CPU, fileproviderd 100-140%, and the GPU was 23-89% utilized when none of these jobs were running. Plain decode on identical code swung from 6 to 32 tok/s (about 33 on a quiet machine), so none of the Mac speed numbers below support a speed claim.

| `MODEL=moe FILLS=512,4096,16384 $D bench_ctx.js` (engine defaults) | main c6ca8cc | perf/metal |
|---|---|---|
| run 1 | **fails**: `OperationError: validation error occurred` at `forwardToken` `mapAsync` after the first prefill | runs: prefill 18.4 / 55.9 / 32.5, spec == plain on every row, 0 GPU errors |
| run 2 | **fails** (same error) | runs: prefill 22.6 / 112.3 / 108.7, spec == plain on every row, 0 GPU errors |

With `PREFILL_UBATCH=0`, both base and branch run. Prefill 512 / 4k / 16k was 19.4 / 24.9 / 36.1 and 36.0 / 104.3 / 85.5 on base, and 45.6 / 92.4 / 103.8 and 20.7 / 111.3 / 93.5 on the branch. That spread is load noise.

The Chrome runs on the Mac (MoE decode, prefill 2048, 27B `ubatch=256`) and the 27B Deno bench_ctx did not complete this session. The shared SSH connection to the Mac dropped partway through. The Chrome evidence for the wide fused layout on the Mac is therefore still the per-branch kernel A/B above: moe_gus 0.49x, moe_dnc 0.55x, moe_route 0.53x of main, interleaved in one run.

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

**Open: MoE test_prefill_wide at 700 tokens on the Mac.** It hit the known Deno-only `mapAsync` validation error once. That is the same error bench_ctx hits on main, and on this branch bench_ctx with the same wide + grouped prefill defaults ran 2 of 2 times. The wide-prefill branch passed this case when it was measured on its own. I set up an A/B job (this branch vs 42136dd alone, alternated 2x) but could not run it because the SSH connection dropped. Whether this is intermittent or caused by the combination is not settled. Output never silently changes: when the error happens, the run throws.
