# Fast model loading for tests and benchmarks

Every 27B/MoE test or benchmark process used to re-read and re-convert the whole GGUF. `tests/run.sh q38` did this about 8 times (once per file), and a Chrome MoE bench logged `loaded in 130s`. Three changes address this: a converted-weights disk cache, a one-process runner for the 27B suite, and a cheaper load path for the Chrome bench. None of them touches decode: the engine, the kernels and the bytes uploaded to the GPU are the same.

## Where load time went (CPU side, measured 2026-09-26, GB10)

Measured with `tests/bench/load_profile.js`. It runs the real loader (`qwen35Weights`, full model plus head and MTP block) with the GPU upload stubbed out. Both GGUFs were in the page cache (`dd` reads them at 18 GB/s).

| | 27B (`models/q38/model.gguf`, 15 GB) | MoE (`q36moe/...Q4_0.gguf`, 21 GB) |
|---|---|---|
| header parse (with / without tokenizer) | 0.17 s / 0.02 s | 0.18 s / 0.02 s |
| tensor reads, Deno `seek`+`read` | 6.5 s (2.3 GB/s) | 8.4 s (2.3 GB/s) |
| conversion, before | **37.2 s** | **37.6 s** |
| of which Q4_0 repack | 20.1 s (12.4 GB) | 28.6 s (17.0 GB) |
| of which K-quant/Q4_1/Q5_0 → Q8 requant | Q5_K 9.2 s, Q6_K 4.8 s, Q4_1 3.1 s | Q4_1 5.2 s, Q6_K 2.2 s, Q5_0 0.3 s |
| **loader total, before** | **43.8 s** | **46.0 s** |
| loader total, faster repack, no cache | 28.9 s (25.0 s with `node:fs` reads) | 25.8 s (24.3 s) |
| loader total, cache cold (convert + write) | 34.0 s | 36.0 s |
| **loader total, cache warm** | **3.8 to 6.3 s** | **8.5 to 8.7 s** |

The warm figure assumes the cache files are in the page cache. When they have been pushed out, for example by reading both GGUFs and both caches (about 73 GB), a warm load took 16.4 s (27B) and 21.8 s (MoE).

A Deno GPU test's load time is roughly this CPU total plus the GPU upload and engine build. For the Chrome bench, the tab's load path was simulated with a fake GPU device and a Node client:

| MoE, Chrome page path (CPU side only) | time |
|---|---|
| old `serve.mjs` (64 KB read chunks, 0.23 to 0.31 GB/s per range) | 74.7 s |
| new `serve.mjs` (8 MB read chunks, 1.9 to 2.2 GB/s per range), same page code | 50.8 s |
| new `serve.mjs` + `?wcache=1` (pre-converted tensors, warm) | 17.0 s (27B: 13.2 s) |

Of the 130 s the Chrome run logged, about 75 s was therefore the static server and the in-tab conversion. The rest is GPU writes plus Chrome's clamping of `setTimeout(0)` to at least 4 ms: `streamEntryToGPU` calls it after each 4 MB flush, about 4,100 flushes for the MoE, so up to about 16 s. That split is inferred by elimination and has not been measured on the GPU yet.

## The converted-weights cache

`tests/weight_cache.js`, hooked in through `engine/gguf.js` `ggufEntry` via `G.entryCache`. The browser never sets it.

- **What is stored.** For each tensor, the exact entry `convertEntry(info, bytes)` returns: repacked Q4_0/Q8_0 nibbles or int8 plus f16 scales, K-quants requantized to Q8, and f32 for the rest. These are the bytes the engine uploads. A hit skips both the GGUF read and the conversion.
- **Where it lives.** `~/.cache/swarmllm-weights/<gguf name>-<key>/<tensor>.bin`, plus a `meta.json`. `WEIGHT_CACHE=<dir>` moves it and `WEIGHT_CACHE=0` turns it off.
- **Key.** The GGUF realpath, size and mtime, plus `LOADER_VERSION`. `LOADER_VERSION` is a sha256 of `engine/gguf.js` (where all the conversion code lives) plus a manual `LOADER_EPOCH`. Any edit to `gguf.js` therefore starts a new directory. Old ones are reported on open, and `WEIGHT_CACHE_PRUNE=1` deletes them.
- **Safety.**
  - Writes go to a temp file and are then renamed, so a killed process never leaves a truncated entry.
  - On read, the header (magic, format, ggml type, element count, kind, layout) and the exact file size are checked against the GGUF tensor info.
  - `WEIGHT_CACHE_VERIFY=1` also checks a payload checksum, at about 1.5x the warm load time.
  - Anything wrong falls back to fresh conversion and rewrites the entry.
  - Without `--allow-write` the cache is read-only. Without `--allow-env` it is off, and there is never an interactive permission prompt.
- **Size.** About 16 GB for the 27B and 20 GB for the MoE (K-quants grow when requantized to Q8), in addition to the GGUFs. Check free space with `df -h ~`; there were 2.8 TB free when this was written.
- **Bit-identical by construction.** `tests/unit/weight_cache_test.js` converts real tensors of every type in both models, fresh and through the cache, and checks that the bytes are equal. The types are Q4_0, Q4_1, Q5_0, Q5_K, Q6_K, Q8_0, BF16, F32 and a stacked 3D expert tensor. The test also covers short, long, empty, garbage-header, wrong-kind and bit-flipped entries (each falls back and is rewritten), and it checks the u16 fast paths of `q4Repack`, `q8Repack` and `streamEntryToGPU` against the original per-block copy, including odd byte offsets and ragged network chunks. Run it with `deno test --allow-read --allow-write tests/unit/weight_cache_test.js` or `npm run test:cache`. Under plain `npm test` (read-only), the two model tests are skipped.

`q4Repack`, `q8Repack` and `streamEntryToGPU` now copy u16 words instead of calling `subarray` + `set` per 18- or 34-byte block. That is 3x faster, byte-identical, and also speeds up the browser's streamed load.

Every 27B test in `run.sh q38`, plus `test_moe`, `test_q38_full`, `prof_moe` and `benchmarks/bench.js MODEL=q38`, loads through `tests/load_model.js` `openGGUF(path)`. That function does `node:fs` reads (about 2x Deno's `seek`/`read`) through the cache. `run.sh` grants `--allow-write` to the cache directory.

## One-process 27B suite

`tests/run_q38_once.js` (`tests/run.sh q38once`) loads the 27B once: all 64 layers, embed, head and the MTP block. It then uploads every matrix and norm to the GPU once (`preuploadWeights`, which sets `entry.gpu`) and runs the nine checks from `run.sh q38` in order. Each check calls its test file's exported `run(ctx)`, so the assertions and goldens are the same code.

- `ctx.model.weights({lo, hi, hasEmbed, hasHead, mtp})` returns views of the shared set with the same structure as `qwen35Weights`. `Qwen35Engine.create` already uses `entry.gpu` as-is (the hook the browser's streamed upload uses), so whole-model, few-layer and split host/worker engines all share one copy. The engine never writes to or frees weight buffers, and it builds fresh per-engine layer objects.
- A CPU check with a fake GPU device compared the views against fresh `qwen35Weights` loads for every range the suite uses (0..5, 0..6 with head, 0..24 with head and MTP, 24..64, 33..64). Structure and bytes were identical, and the bytes sent to the (fake) GPU buffers matched the CPU entries.
- Each test file still runs standalone (`import.meta.main`), and `run.sh q38` is unchanged in behaviour. Two exceptions: `test_q38.js` and `test_q38_split.js` now exit 1 on MISMATCH, where before they printed MISMATCH and exited 0.
- `test_gemm` loads no model and just gets the shared device. `test_ctx` counts only the GPU errors raised during its own run.
- `ONLY=test_mtp,test_ctx` runs a subset.

## Chrome bench

`node tests/bench/chrome_bench.mjs models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf`:

- `serve.mjs` streams with 8 MB reads (7x faster ranges). This change alone helps the old page path.
- `WCACHE=1` (the default) loads `bench.html?wcache=1`. The page fetches each tensor already converted from `serve.mjs` `/__wcache`, which converts and caches it on a miss using the same cache directory as the Deno tests. The page then writes the tensor straight into its GPU buffers. The embedding and f32 tensors stay on the CPU as before. `WCACHE=0` gives the old path.
- `CHROME_PROFILE` (default `~/.cache/swarmllm-chrome-bench`) is a persistent profile, so Chrome's disk caches, such as its GPU shader cache, survive between runs. `CHROME_PROFILE=0` uses a fresh profile each run.
- The decode measurement is the same page, engine and kernels. Only loading differs.

## GPU validation commands

```sh
cd <worktree>
df -h ~                                              # cache needs ~16 GB (27B) + ~20 GB (MoE)
# 1. reference, cache off (old behaviour, faster repack only)
time WEIGHT_CACHE=0 tests/run.sh q38 2>&1 | tee /tmp/q38_nocache.log
# 2. cache: first run fills it, second run is warm
time tests/run.sh q38 2>&1 | tee /tmp/q38_cold.log
time tests/run.sh q38 2>&1 | tee /tmp/q38_warm.log
# verdicts and numerics must match; tok/s lines may differ
diff <(grep -E "PASS|FAIL|MISMATCH|argmax|identical|differing" /tmp/q38_nocache.log | sed 's/batchedPrefill=.*//') \
     <(grep -E "PASS|FAIL|MISMATCH|argmax|identical|differing" /tmp/q38_warm.log | sed 's/batchedPrefill=.*//')
# 3. one process, shared upload
time tests/run.sh q38once 2>&1 | tee /tmp/q38_once.log
# 4. MoE test, twice (cold then warm), compare "loaded in"
(cd tests && deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights test_moe.js)
# 5. Chrome bench: old path vs pre-converted path (tok/s lines must match within noise, load time should drop)
WCACHE=0 CHROME_PROFILE=0 node tests/bench/chrome_bench.mjs models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf
node tests/bench/chrome_bench.mjs models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf    # twice: second run warm
```

Estimates only, not GPU-measured: a full-27B test's load should drop from about 45 s of CPU plus the upload to about 5 to 7 s plus the upload. The q38 suite loads the full 27B 5 times across 4 files (`test_ctx` loads it twice) and 5 to 8 layers 5 times across 4 more files (`test_gemm` loads nothing). With the cache every one of those loads is a cache read, and `q38once` does one load and one upload for all of them. For the Chrome MoE load, expect about 130 s to drop to roughly 20 to 40 s (17 s of transfer plus GPU writes).

## GPU validation result (2026-09-26, branch prefill/base = feat/engine-opt + opt/load-cache, GB10)

Reference: feat/engine-opt 927696c before the merge, same session. Every check passed on both sides, and every
verdict, logit, argmax, acceptance count and output text line is identical between the reference, the merge with
`WEIGHT_CACHE=0`, the cache's cold (filling) run and its warm run. Only `test_gemm`'s relDiff digits move, as they do
between any two runs (random matrices, no model). `tests/test_moe.js`: all three prompts MATCH llama.cpp, spec
identical to plain, same acceptance (28/33, 28/39, 25/45) on all four runs. `tests/run.sh quick` passes unchanged.
`tests/unit/weight_cache_test.js` passes (4/4, every ggml type byte-identical).

| load (Deno, GPU included) | reference | merge, WEIGHT_CACHE=0 | cache cold | cache warm |
|---|---|---|---|---|
| full 27B (`test_mtp` "loaded in") | 57 s | 36 s | 39 s | 18 s |
| MoE (`test_moe` "loaded in") | 63 s | 39 s | 53 s | 29 s |
| `run.sh q38` wall, 9 files | 404 s | 316 s | 196 s | 170 s |
| `run.sh q38once` wall | | | | 58 s (8.3 s load + upload) |

`tests/bench_ctx.js` and `tests/prof_ts.js` now load through `openGGUF` too (same cache), so they need
`--allow-write=$HOME/.cache/swarmllm-weights` to fill it (read-only still works).

Fast validation for engine changes:

```sh
cd tests
../tests/run.sh quick                          # small models, ~1 min
../tests/run.sh q38once                        # 27B suite, one load, ~1 min warm
D="deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights"
$D test_moe.js                                 # MoE vs llama.cpp, ~45 s warm
MODEL=moe FILLS=1024,4096,16384 $D bench_ctx.js # prefill/decode vs context fill (MODEL=27b for the dense)
Q38=1 $D prof_ts.js                            # per-kernel GPU ms (drop Q38 for the MoE)
```

`run.sh q38` (one process per file) remains the reference; run it before landing. The Chrome bench path (`WCACHE=1`, `serve.mjs`) was not re-run in this validation.

## Rooms: a device that drops out (#207)

`tests/e2e/room_resume.mjs` (loopback, Spark: host, desktop worker and a phone-shaped tab in headless Chromium, real WebRTC) interrupts an answer mid-stream and checks it finishes with the uninterrupted text. `lock` hides the tab, closes its signaling socket and pauses its JS in the debugger for 20 s (Chromium ignores a freeze on a visible page); `reload`; `kill` (closed for good: the experimental auto re-deal after 60 s). `--scenarios code --model qwen3-1.7b` does the same to a Code run mid-step (`--code-action lock|reload`).

`tests/e2e/resume_phone.mjs` drives the real iPhone over WebDriver against `room_resume.mjs --url <preview>/room --external iphone`: `background` switches Safari to another tab for 30 s (the closest WebDriver gets to a lock: iOS stops the page's WebRTC), `reload` reloads the room's tab. A real screen lock needs a person.

Results, 2026-09-29 (Qwen3 0.6B split 3 ways; the iPhone 14 Pro Max over the branch preview, host and worker on the Spark):

| Where | Interrupt | Back in | Result |
|---|---|---|---|
| loopback | lock 20 s | 26 s | same text, the tab back in its slot with its layers |
| loopback | reload | 18 s | same text, rejoined under its name, layers reloaded |
| loopback | kill | 103 s | auto re-deal after 60 s over 2 devices; shown text kept, answer ran to the end |
| loopback, 1.7B | Code run, lock 20 s | 39 s | run waited, carried on inside its step, wrote index.html + style.css, no stop note |
| loopback, 1.7B | Code run, reload | 52 s | same |
| iPhone | Safari in the background 30 s | 67 s | host dropped the silent link after 12 s, the phone reconnected by itself, layers still loaded (no download), same text |
| iPhone | reload | 53 s | rejoined its slot from sessionStorage with the crumb shown, same text |

Before the `ai-linked` handshake the first batch prefill after a phone came back timed out once (the device before it was still opening its fresh link): the answer recovered twice and took 165 s instead of 67 s.
