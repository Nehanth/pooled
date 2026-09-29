// CPU checks of the subgroup-matrix wide prefill GEMM generator (engine/wgsl/gemm_wmma.js). The
// kernels themselves only run in Chrome with chromium-experimental-subgroup-matrix; the engine
// self-tests them against the f32 wide GEMM at init (Qwen35Engine._initMma) and
// tests/bench/wmma_gemm.mjs measures them. Here: config picking on the adapters we know, tile plan
// validation, the Q4_0 nibble -> i8 byte trick, the generated WGSL's entry points in both builtin
// spellings, and the i8 reference's error against float64 (activation quantization).
//   deno test --no-check tests/unit/gemm_wmma_test.js
import { pickWmmaConfig, wmmaPlan, gemmWmmaWGSL, wmmaI8Ref, wmmaXqWords, WMMA_SYNTAX } from "../../engine/wgsl/gemm_wmma.js";

const eq = (a, b, m) => { if (a !== b) throw new Error(`${m}: ${a} !== ${b}`); };
const ok = (c, m) => { if (!c) throw new Error(m); };
const cfg = (componentType, resultComponentType, M, N, K) => ({ componentType, resultComponentType, M, N, K });
// what Chrome 145..153 report (docs/research/subgroup-matrix-2026-09.md)
const GB10 = { subgroupMaxSize: 32, subgroupMatrixConfigs: [cfg("u8", "u32", 16, 16, 32), cfg("i8", "i32", 16, 16, 32), cfg("u8", "u32", 16, 8, 32), cfg("i8", "i32", 16, 8, 32)] };
const APPLE = { subgroupMaxSize: 32, subgroupMatrixConfigs: [cfg("f32", "f32", 8, 8, 8), cfg("f16", "f16", 8, 8, 8)] };

Deno.test("pickWmmaConfig: i8 16x16x32 on the GB10, f32 8x8x8 on Apple, nothing without configs", () => {
  const g = pickWmmaConfig(GB10);
  eq(g.kind, "i8", "GB10 kind"); eq(`${g.M}x${g.N}x${g.K}`, "16x16x32", "GB10 shape");
  eq(pickWmmaConfig(GB10, "f32").kind, "i8", "GB10 has no f32: falls to i8");
  const a = pickWmmaConfig(APPLE);
  eq(a.kind, "f32", "Apple kind"); eq(`${a.M}x${a.N}x${a.K}`, "8x8x8", "Apple shape");
  ok(pickWmmaConfig({ subgroupMaxSize: 32, subgroupMatrixConfigs: [] }).why, "no configs -> why");
  ok(pickWmmaConfig({ subgroupMaxSize: 32, subgroupMatrixConfigs: [cfg("f16", "f16", 8, 8, 8)] }).why, "f16->f16 only -> why");
  ok(pickWmmaConfig({ subgroupMatrixConfigs: GB10.subgroupMatrixConfigs }).why, "no subgroup size -> why");
});

// i8 stages each block's i32 tiles (BM x BN) in workgroup memory: its default needs the 32 KB the
// engine requests with wide prefill (both GPUs here have it); f32 fits the 16 KB default.
const LIM = { i8: 32768, f32: 16384 };
Deno.test("wmmaPlan: defaults fit, bad tiles throw", () => {
  for (const info of [GB10, APPLE]) {
    const mc = pickWmmaConfig(info), P = wmmaPlan(mc, {}, LIM[mc.kind]);
    ok(P.smem <= LIM[mc.kind] && P.T <= 1024 && P.T % 32 === 0, `plan ${JSON.stringify(P)}`);
    eq(P.KS, 32, "KS");
  }
  const mc = pickWmmaConfig(GB10);
  for (const bad of [{ SM: 24 }, { BN: 40 }, { PAD: 3 }, { BM: 256, BN: 256 }, { SM: 16, BM: 32 }])
    ok((() => { try { wmmaPlan(mc, bad, 32768); return false; } catch { return true; } })(), `should throw: ${JSON.stringify(bad)}`);
});

Deno.test("Q4_0 nibble -> i8 byte trick equals q - 8 for every byte", () => {
  // WGSL: (((w & 0x0F0F0F0F) | 0x80808080) - 0x08080808) ^ 0x80808080, per byte lane, no borrow across lanes
  for (let v = 0; v < 65536; v += 1) {
    const w = (v * 0x10001 ^ (v << 7)) >>> 0;
    for (const hi of [false, true]) {
      const s = hi ? w >>> 4 : w;
      const r = ((((s & 0x0F0F0F0F) | 0x80808080) >>> 0) - 0x08080808 >>> 0 ^ 0x80808080) >>> 0;
      for (let b = 0; b < 4; b++) {
        const got = (((r >>> (8 * b)) & 255) << 24) >> 24, want = ((s >>> (8 * b)) & 15) - 8;
        if (got !== want) throw new Error(`w ${w.toString(16)} hi ${hi} byte ${b}: ${got} vs ${want}`);
      }
    }
  }
});

Deno.test("generated WGSL: entry points and builtin spelling", () => {
  for (const info of [GB10, APPLE]) {
    const mc = pickWmmaConfig(info), P = wmmaPlan(mc, {}, LIM[mc.kind]);
    for (const syntax of WMMA_SYNTAX) {
      const s = gemmWmmaWGSL(P, syntax);
      ok(s.startsWith("enable chromium_experimental_subgroup_matrix;"), "enable first");
      for (const e of ["gemm_m_q4", "gemm_m_q4_acc", "gemm_m_q8", "gemm_m_q8_acc", ...(P.kind === "i8" ? ["wmma_quant"] : [])]) ok(s.includes(`fn ${e}(`), `${P.kind} ${syntax}: ${e}`);
      ok(syntax === "template" ? /, (row|col)_major>\(/.test(s) : !/_major>/.test(s), `${P.kind} ${syntax} spelling`);
      ok(!/\$\{|undefined|NaN/.test(s), `${P.kind} ${syntax}: template leftovers`);
    }
  }
  eq(wmmaXqWords(256, 5120), 256 * 5120 / 4 + 256 * 160, "xq words");
});

Deno.test("i8 reference: activation quantization error vs float64 stays ~1e-2 rel with outliers", () => {
  let seed = 99; const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
  const dOut = 24, dIn = 512, cols = 8, nb = dIn / 32;
  const wq = Float32Array.from({ length: dOut * dIn }, () => Math.floor(rnd() * 16) - 8), ws = Float32Array.from({ length: dOut * nb }, () => 0.004 + rnd() * 0.02);
  const x = Float32Array.from({ length: cols * dIn }, (_, i) => (rnd() * 2 - 1) * (i % 97 === 0 ? 20 : 1));
  const y = wmmaI8Ref({ wq, ws, x, dOut, dIn, cols });
  let n = 0, d = 0;
  for (let c = 0; c < cols; c++) for (let r = 0; r < dOut; r++) {
    let a = 0; for (let k = 0; k < dIn; k++) a += wq[r * dIn + k] * ws[r * nb + (k >> 5)] * x[c * dIn + k];
    n += (y[c * dOut + r] - a) ** 2; d += a * a;
  }
  const rel = Math.sqrt(n / d);
  ok(rel > 1e-4 && rel < 2e-2, `rel ${rel}`);
});
