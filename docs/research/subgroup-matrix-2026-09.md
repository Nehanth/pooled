# Subgroup-matrix (tensor-core) wide prefill GEMM

Branch `perf/kf-subgroup-matrix`. Opt-in engine option `prefillMma` (off by default: the default
path builds the same pipelines and produces the same bits as main). Kernel generator
`engine/wgsl/gemm_wmma.js`, engine glue in `engine/qwen35.js` (`_initMma`, `_mmaSelfTest`, `wop` /
`_dW`), kernel bench `tests/bench/wmma_gemm.{html,mjs}`, CPU tests `tests/unit/gemm_wmma_test.js`.

This is separate from `prefillMath: "sgmatrix"` (engine/wgsl/gemm_sgm.js, prefill/f16-subgroup):
that one replaces the 16-column GEMM and needs an f16 -> f32 matrix shape, which neither of our
GPUs exposes. `prefillMma` replaces the wide (ubatch) GEMM and uses the shapes the GPUs do expose.

## What is available where (measured)

| | GB10 (DGX Spark), Linux, NVIDIA Vulkan | M5 Max, macOS, Metal |
|---|---|---|
| Browser | Chrome for Testing 153.0.8010.12 headless (`--enable-unsafe-webgpu`) | Chrome 154.0.8037.58 headless (`--enable-unsafe-webgpu`) |
| Feature | `chromium-experimental-subgroup-matrix` | `chromium-experimental-subgroup-matrix` |
| Also | `subgroups`, `subgroup-size-control`; **no `shader-f16`** | `subgroups`, `shader-f16` |
| Subgroup size | 32..32 | 32..32 |
| `subgroupMatrixConfigs` | `u8->u32` and `i8->i32` at 16x16x32 and 16x8x32. **No float shapes.** | `f32->f32` 8x8x8, `f16->f16` 8x8x8. **No f16->f32, no integer.** |
| WGSL spelling | Chrome 153: template loads (`subgroupMatrixLoad<T, row_major>`) and the proposal's type order `left<T, M, K>` (the old `<T, K, M>` order is rejected for i8 16x16x32: "no matching call to subgroupMatrixMultiply"); Chromium 145 took only the "bool" form | template form; 8x8x8 is square, so type order does not matter |
| Chrome 131 (Playwright's bundled Chromium) | feature absent (only `subgroups`) | not tested |

Without `--enable-unsafe-webgpu` (chrome://flags "Unsafe WebGPU Support") the feature is not
listed at all, on either machine.

So the kernel has two families, picked from `adapterInfo.subgroupMatrixConfigs`:

- **i8** (NVIDIA / Vulkan): MMQ, the scheme llama.cpp's CUDA backend runs for Q4_0 prefill.
  Q4_0 nibbles unpack exactly to i8 (q - 8), Q8_0 is i8 already. Activations are quantized per
  (column, 32-block) to i8 with an f32 scale (`wmma_quant`, llama.cpp's `quantize_q8_1` rounding).
  One 16x16x32 MMA per block, the i32 tile goes through workgroup memory and every lane applies
  `w_scale * x_scale` into f32 accumulators it owns. **Changes numerics** (activation quantization,
  ~4e-3 rel per GEMM on random data).
- **f32** (Apple / Metal): weights dequantized with their scale into an f32 tile, activations loaded
  straight from the column-strided x, 8x8x8 `subgroupMatrixMultiplyAccumulate` into f32. Same
  numerics as the f32 wide GEMM up to summation order.

Both self-test against the f32 wide GEMM at engine init (rel L2 < 1e-4 for f32, < 2e-2 for i8);
anything missing (feature, config, compile, self-test) leaves the f32 wide GEMM and records why in
`engine.mmaWhy`.

## How users would get it

- **Today:** only with a flag. Chrome / Edge 145+ desktop with `--enable-unsafe-webgpu` (or
  chrome://flags/#enable-unsafe-webgpu). Not something to ask room users to do: that flag turns off
  WebGPU's safety checks for every site.
- **Origin trial:** none. Chrome ran origin trials for `subgroups` (128-131, shipped in 134) and
  `subgroup_id`, but there is no intent-to-experiment for subgroup-matrix on blink-dev as of
  2026-09-29.
- **Spec:** the gpuweb proposal (`proposals/subgroup-matrix.md`, issue #4195) is still "Draft".
  Chrome shipping it by default needs the proposal to settle (builtin spelling already changed
  once: the "bool" -> "template" form between Chrome 145 and 154), then an intent to ship.
  Firefox and Safari have no implementation. Realistic default-on: not before 2027, and float
  shapes on NVIDIA / integer shapes on Apple are not guaranteed even then.
- **Deno / wgpu:** not exposed (the engine's Node / Deno test paths never see it).

Therefore: ship it only as a feature-detected opt-in. Anyone who runs Chrome with the flag gets it
through `prefillMma`, everyone else keeps the f32 wide GEMM with identical bits.

## Measurements

### M5 Max, kernel alone (`node tests/bench/wmma_gemm.mjs 'reps=10&w=128'`, 2026-09-29 04:34)

f32 8x8x8 plan BM 64 BN 64 SM 32 SN 32 KB 1 PAD 4, 128 threads, 9216 B workgroup memory, "template"
syntax. Random Q4_0 / Q8_0 weights, 128 token columns. `err` = rel L2 vs float64; both kernels
~1e-6 (same numerics), the `+=` variant checked too.

| shape | f32 wide GEMM ms (TFLOPS) | subgroup-matrix ms (TFLOPS) | speedup |
|---|---|---|---|
| 17408x5120 q4 (27B gate/up) | 3.10 (7.36) | 2.53 (9.02) | 1.23x |
| 12288x5120 q4 | 2.38 (6.77) | 1.64 (9.82) | 1.45x |
| 10240x5120 q4 | 1.98 (6.78) | 1.39 (9.66) | 1.42x |
| 6144x5120 q4 | 1.99 (4.05) | 0.89 (9.05) | 2.24x |
| 5120x17408 q8 (27B down) | 3.14 (7.27) | 2.63 (8.68) | 1.19x |
| 5120x6144 q8 | 1.54 (5.23) | 0.88 (9.15) | 1.75x |
| 8192x2048 q4 (MoE) | 1.23 (3.49) | 0.49 (8.77) | 2.51x |
| 4096x2048 q4 | 0.95 (2.26) | 0.42 (5.11) | 2.26x |
| 2048x4096 q8 | 1.04 (2.06) | 0.32 (6.71) | 3.25x |
| 2048x4096 q4 | 1.00 (2.15) | 0.32 (6.71) | 3.12x |

The big 27B FFN shapes, which dominate 27B prefill, gain only 1.2-1.45x: at 8x8x8 f32 the Apple
simdgroup matrices are ~9-10 TFLOPS, not far above the tuned f32 ALU GEMM. The small (MoE
attention / DeltaNet projection) shapes gain 2.2-3.2x mostly because the f32 wide GEMM is poor on
them.

### GB10, kernel alone (Chrome 153 headless, `wmma_gemm.mjs 'reps=10&w=256'`, 2026-09-29 13:20)

f32 wide GEMM (the current default wide prefill kernel), 256 columns: 17408x5120 q4 8.88 ms (5.14
TFLOPS), 12288x5120 q4 6.93 (4.65), 10240x5120 q4 5.40 (4.97), 6144x5120 q4 3.13 (5.15), 5120x17408
q8 10.18 (4.48), 5120x6144 q8 3.63 (4.44), 8192x2048 q4 1.78 (4.83), 4096x2048 q4 1.33 (3.23),
2048x4096 q8 0.93 (4.62), 2048x4096 q4 0.91 (4.72). Compare 134 TOPS measured for the raw i8 MMA
(prefill-f16-subgroup.md): the headroom is ~25x on paper, so even a 10x-off kernel would be a
large win on NVIDIA.

**The i8 kernel did not compile in this run**: it was generated with the pre-153 type parameter
order (`left<i8, 32, 16>`), which Chrome 153 rejects. Fixed in this branch (tries the proposal
order first, then the old ones). Note: `engine/wgsl/gemm_sgm.js` on main (`prefillMath:
"sgmatrix"`) uses the same old order; it is unreachable on both our GPUs anyway (no f16->f32
config), but would need the same fix on hardware that has one.

Also found and fixed in this run: `_initMma` read `this.layers` before the layers exist (27B
TypeError at init), and bench.html did not request the 32 KB workgroup-memory limit for `?mma=`
without `?ubatch=` (MoE: "19968 B of workgroup memory (limit 16384)", falls back correctly).
MoE wide prefill at 2048 with the f32 GEMM: 88.6 tok/s (Chrome 153, GB10).

### End-to-end prefill 512 / 2048 / 8192 (27B, MoE)

Not measured yet: the GB10 GPU queue was saturated (the one engine run of this session failed on
the bugs above), and the M5 Max queue never reached this job before the cut-off. Commands:

    # GB10 (Chrome 153 headless shell): i8
    CHROME_BIN=<chrome 153> PORT=8817 node tests/bench/chrome_bench.mjs models/q38/model.gguf 4 'ubatch=256&mma=i8&prefilllen=2048&prefillmodes=default,wide,mma'
    # M5 Max (Chrome 154): f32
    node tests/bench/chrome_bench.mjs models/q38/model.gguf 4 'ubatch=128&mma=f32&prefilllen=2048&prefillmodes=default,wide,mma'

The page prints tok/s for `default` (16-column GEMM), `wide` (f32 tiled wide GEMM) and `mma`, and
the logits relDiff of `mma` vs both.

## Verdict so far

Not kept as a default (it cannot be: flag-only, i8 changes numerics). Kept as an opt-in, off by
default, zero bit change to the default path. Expected value: Apple 1.2-1.45x on the 27B's big
GEMMs (measured, kernel only), NVIDIA potentially far more (i8 tensor cores vs 5 TFLOPS f32 today),
unmeasured until the fixed i8 kernel compiles on the GB10.


