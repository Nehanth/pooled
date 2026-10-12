// CPU check of Qwen3.8-Flash-Next's state accounting (engine/qwen4exp_mem.js) on the model's shape: the sizes by
// hand, f16 halving the K / V cache, the raw-key ring's fixed size, a room stage's share, and the budget inverse.
// The engine's allocations are checked against it on the GPU (tests/test_q38flash_ckpt.js).
// No GPU.   deno test --allow-read tests/unit/q38flash_mem_test.js
import { qwen4expMemory, qwen4expMaxCtx, QSA_RAW_RING } from "../../engine/qwen4exp_mem.js";

const assert = (c, m) => { if (!c) throw new Error(m); };
// the bartowski Qwen3.8-Flash-Next-Q4_0 config (qwen4expConfig), the fields the accounting reads
const cfg = { L: 48, dim: 2560, hc: 4, hDim: 10240, nHeads: 24, nKVHeads: 2, headDim: 256,
  qsa: Array.from({ length: 48 }, (_, i) => (i + 1) % 4 === 0),
  ssm: { convK: 4, stateSize: 128, nKHeads: 16, nVHeads: 48, inner: 6144 },
  ple: { layers: [1], ngram: 3, convK: 4 } };
const MiB = 2 ** 20;

Deno.test("Flash state: sizes by hand at 32K", () => {
  const m = qwen4expMemory(cfg, { ctx: 32768, kv: "f32", rawRing: 0 });
  assert(m.layers.qsa === 12 && m.layers.deltaNet === 36, "12 attention + 36 DeltaNet layers");
  assert(m.bytes.kv === 12 * 2 * 32768 * 2 * 256 * 4, "f32 K + V: 48 KB per token");
  assert(m.bytes.rawK === 12 * 32768 * 128 * 2 && m.bytes.pooled === 12 * 8192 * 128 * 2, "rawK per position, pooled per block");
  assert(m.bytes.deltaNet === 36 * (48 * 128 * 128 * 4 + 10240 * 3 * 4), "S + conv window per DeltaNet layer");
  assert(m.bytes.ple === 10 * 10240 * 4, "PLE ring: (4 - 1) * 3 + 1 slots");
  assert(m.perToken === 12 * (4096 + 256 + 64), `per token ${m.perToken}`);
  assert(m.state === m.bytes.kv + m.bytes.rawK + m.bytes.pooled + m.bytes.deltaNet + m.bytes.ple, "state is the sum");
});

Deno.test("Flash state: f16 halves the K / V cache, the ring makes rawK fixed", () => {
  for (const ctx of [8192, 32768, 131072, 262144]) {
    const a = qwen4expMemory(cfg, { ctx, kv: "f32", rawRing: 0 }), b = qwen4expMemory(cfg, { ctx });
    assert(b.kv === "f16" && b.rawRing === QSA_RAW_RING, "defaults: f16 KV, the ring");
    assert(b.bytes.kv * 2 === a.bytes.kv, `${ctx}: f16 KV is half`);
    assert(b.bytes.rawK === 12 * QSA_RAW_RING * 256, `${ctx}: rawK ring is ${QSA_RAW_RING} rows per layer`);
    assert(b.bytes.pooled === a.bytes.pooled && b.bytes.deltaNet === a.bytes.deltaNet, `${ctx}: pooled and DeltaNet unchanged`);
    assert(b.perToken === 12 * (2048 + 64), `${ctx}: 24.75 KB per token after`);
  }
  const big = qwen4expMemory(cfg, { ctx: 262144 });
  assert(Math.round(big.bytes.kv / MiB) === 6144 && Math.round(big.bytes.pooled / MiB) === 192, "256K: 6 GiB KV, 192 MiB pooled");
});

Deno.test("Flash state: a room stage holds its layers only; no QSA, no indexer state", () => {
  const s = qwen4expMemory(cfg, { ctx: 8192, layerRange: [0, 4] });
  assert(s.layers.qsa === 1 && s.layers.deltaNet === 3 && s.bytes.ple > 0, "layers 0-3: one attention layer, the PLE layer");
  const t = qwen4expMemory(cfg, { ctx: 8192, layerRange: [4, 8] });
  assert(t.bytes.ple === 0, "no PLE ring past layer 1");
  const d = qwen4expMemory(cfg, { ctx: 2048, qsa: false });
  assert(d.bytes.rawK === 0 && d.bytes.pooled === 0 && d.bytes.scores === 0, "dense: no indexer state");
});

Deno.test("Flash state: the longest context for a budget", () => {
  for (const budget of [1, 4, 8].map((g) => g * 2 ** 30)) {
    const n = qwen4expMaxCtx(cfg, budget);
    assert(qwen4expMemory(cfg, { ctx: n }).total <= budget && qwen4expMemory(cfg, { ctx: n + 1 }).total > budget, `${budget}: ${n}`);
  }
});
