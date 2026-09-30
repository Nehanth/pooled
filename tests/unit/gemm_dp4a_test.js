// dp4a wide prefill GEMM generator (engine/wgsl/gemm_wide.js gemmDp4aWGSL), no GPU: tile config checks, the
// Q4_0 nibble re-centring identity the kernel uses, and a scalar model of the kernel's arithmetic
// (per-block exact int dot, f32 scale product) against float64 on random data. The GPU kernel itself is
// checked against float64 by tests/bench_dp4a_gemm.js.
//   deno test --no-check tests/unit/gemm_dp4a_test.js
import { dp4aTileConfig, gemmDp4aWGSL, DP4A_TILE_DEFAULT } from "../../engine/wgsl/gemm_wide.js";

const assert = (c, m) => { if (!c) throw new Error(m); };

Deno.test("dp4a tile config: default fits 16 KB, bad shapes throw", () => {
  const c = dp4aTileConfig();
  assert(c.T === 256 && c.KS === 64 && c.smem <= 16384, JSON.stringify(c));
  for (const bad of [{ BM: 62 }, { TN: 3 }, { BN: 40, TN: 4 }, { BM: 256, BN: 256, TM: 4, TN: 4 }, { KB: 0 }, { BM: 128, BN: 128, TM: 8, TN: 8, KB: 4 }])
    { let threw = false; try { dp4aTileConfig(bad); } catch { threw = true; } assert(threw, `accepted ${JSON.stringify(bad)}`); }
  assert(DP4A_TILE_DEFAULT.BN === 64, "the dp4a tile width must divide the wide tile width 64");
});

Deno.test("dp4a WGSL: one entry point per format and accumulate mode, plus the quantizer", () => {
  const w = gemmDp4aWGSL(dp4aTileConfig());
  for (const n of ["gemm_d_q4", "gemm_d_q4_acc", "gemm_d_q8", "gemm_d_q8_acc", "quant_q8_w"]) assert(w.includes(`fn ${n}(`), n);
  assert((w.match(/dot4I8Packed/g) || []).length >= 4 * 16 * 2, "the inner loop is unrolled");
});

// ((w & 0x0F0F0F0F) + 0x78787878) ^ 0x80808080 turns four nibbles q into four signed bytes q - 8
Deno.test("Q4 re-centring: every nibble value, every byte lane", () => {
  const sb = (x, i) => (((x >>> (8 * i)) & 255) << 24) >> 24;
  for (let q = 0; q < 16; q++) for (let lane = 0; lane < 4; lane++) {
    const w = (q << (8 * lane)) | (0x5 << (8 * ((lane + 1) % 4)));
    const r = ((((w & 0x0F0F0F0F) >>> 0) + 0x78787878) ^ 0x80808080) >>> 0;
    assert(sb(r, lane) === q - 8, `q ${q} lane ${lane}: ${sb(r, lane)}`);
    assert(sb(r, (lane + 1) % 4) === 5 - 8, "neighbour lane");
  }
});

// The kernel's arithmetic, scalar: x quantized per 32 (d = amax / 127, round), exact int dot per block,
// acc += f32(isum) * (dw * dx). Error vs float64 stays at the activation-quantization level (~1e-2 of max).
Deno.test("dp4a arithmetic model vs float64", () => {
  let seed = 3; const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
  const dIn = 1024, rows = 16, nb = dIn / 32, f = Math.fround;
  const x = Array.from({ length: dIn }, () => rnd() * 2 - 1);
  const xq = new Int8Array(dIn), xd = new Float32Array(nb);
  for (let b = 0; b < nb; b++) {
    let m = 0; for (let i = 0; i < 32; i++) m = Math.max(m, Math.abs(x[b * 32 + i]));
    const d = f(m / 127), id = d > 0 ? f(1 / d) : 0; xd[b] = d;
    for (let i = 0; i < 32; i++) xq[b * 32 + i] = Math.round(f(x[b * 32 + i] * id));
  }
  let worst = 0, scale = 0;
  for (let r = 0; r < rows; r++) {
    const q = Array.from({ length: dIn }, () => Math.floor(rnd() * 16) - 8), dw = Array.from({ length: nb }, () => f(0.01 + rnd() * 0.01));
    let ref = 0, acc = 0;
    for (let b = 0; b < nb; b++) {
      let is = 0; for (let i = 0; i < 32; i++) { is += q[b * 32 + i] * xq[b * 32 + i]; ref += dw[b] * q[b * 32 + i] * x[b * 32 + i]; }
      acc = f(acc + f(is * f(dw[b] * xd[b])));
    }
    worst = Math.max(worst, Math.abs(acc - ref)); scale = Math.max(scale, Math.abs(ref));
  }
  assert(worst / scale < 2e-2, `relErr ${worst / scale}`);
});
