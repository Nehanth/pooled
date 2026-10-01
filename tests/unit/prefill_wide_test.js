// CPU-only check of the wide prefill path's command stream (engine option prefillUbatch), on a tiny
// synthetic model (tests/e2e/synth.mjs, dense and MoE) and the recording mock device (mock_gpu.js):
// no GPU, no shader execution. What it proves:
//  * the wide path encodes valid WebGPU commands (bind-group ranges, alignment, usage, copy ranges,
//    dispatch limits) and dispatches the tiled GEMMs over the whole chunk;
//  * every non-projection kernel (DeltaNet conv / gates / recurrence / gated norm, the beta / alpha
//    GEMV, attention glue / KV store / flash / combine / gate, the MoE router and experts, the whole
//    MTP draft-cache fill) is dispatched with exactly the same bind group, grid and frame
//    (positions, column count), in the same order per bind group, as the default 16-column prefill;
//  * with prefillUbatch 0, or engine.prefillWide = false at runtime, the command stream is the 16-column path's;
//  * the defaults: wide prefill on for an engine holding the embedding (MoE and dense), wide + expert-grouped
//    prefill on for a MoE engine with or without the embedding (a room worker runs it through prefillHidden);
//  * prefillDp4a: every wide projection as quant_q8_w + an int8 GEMM (gemm_d_*), valid commands, runtime switch;
//  * prefillHidden (a room chain's prompt frames) puts every column's hidden back in its place: the mock runs
//    no shaders, so what comes back is what went in, through the wide, grouped and batchCols-wide stages.
// The MoE comparison runs with moeGroupPrefill 0 on both sides (the per-sub-batch expert kernels are what
// it checks; the grouped kernels are tests/unit/moe_group_engine_test.js's).
//   deno test --no-check --allow-read tests/unit/prefill_wide_test.js
import { mockDevice } from "./mock_gpu.js";
import { buildSynthGGUF, SYNTH_MOE } from "../e2e/synth.mjs";
import { parseGGUFHeader, qwen35Weights, GGML_EMBED } from "../../engine/gguf.js";
import { Qwen35Engine, DP4A_DEFAULT, dp4aAutoDevice } from "../../engine/qwen35.js";

async function engineFor(buf, opts = {}) {
  const G = parseGGUFHeader(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const bytesOf = async (i) => buf.slice(i.byteOffset, i.byteOffset + i.byteLength);
  const L = G.meta["qwen35.block_count"] - 1;
  const device = mockDevice({ wgMem: opts.wgMem ?? 16384 });
  const origWarn = console.warn; const warns = []; console.warn = (...a) => warns.push(a.join(" "));
  try {
    const eng = await Qwen35Engine.create({ device, meta: G.meta, layerRange: [0, L], hasEmbed: true, hasHead: true, vocab: G.tensors[GGML_EMBED].shape[0],
      maxSeq: 512, batchCols: 16, coopRowsB: 1, weights: await qwen35Weights(G, bytesOf, { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true }), ...opts.engine });
    return { eng, device, warns };
  } finally { console.warn = origWarn; }
}
const models = {
  dense: buildSynthGGUF({ layers: 8 }).bytes,
  q8out: buildSynthGGUF({ layers: 8, q8out: true, seed: 3 }).bytes,   // ffn_down, ssm_out, attn_output in Q8_0 like the real 27B
  moe: buildSynthGGUF({ layers: 8, moe: SYNTH_MOE }).bytes,
};
const ids = (n) => Array.from({ length: n }, (_, i) => 33 + ((i * 7919) % 90));

async function trace(model, n, engine = {}, wgMem) {
  const { eng, device, warns } = await engineFor(models[model], { engine, wgMem });
  device.log.length = 0;
  await eng.prefillTokens(ids(n));
  return { eng, log: device.log.slice(), warns };
}

// per bind group: the ordered list of (pipeline, grid, frame) it was dispatched with
const byBg = (log) => {
  const m = new Map();
  for (const e of log) { if (!m.has(e.bg1)) m.set(e.bg1, []); m.get(e.bg1).push(`${e.pipe} ${e.grid} ${e.frame}`); }
  return m;
};
// kernels the wide path replaces: the projection GEMVs / 16-column GEMM (+ transposes, reduces), the
// trunk's batched norms and the per-column SiLU. Everything else must be dispatched identically.
const PROJ = /^(matvec_.*coop_b|matvec_.*gu_b|gemm_(q4|q8|red|xpose)|rmsnorm_mc$|silu_mul(_w)?$|add_res_mc$)/;
const eq = (a, b, m) => { if (a !== b) throw new Error(`${m}: ${a} != ${b}`); };

async function compare(model, n, U, wgMem) {
  const base = await trace(model, n, { prefillUbatch: 0, moeGroupPrefill: 0 });
  const wide = await trace(model, n, { prefillUbatch: U, moeGroupPrefill: 0, prefillDp4a: false }, wgMem);
  if (wide.warns.length || !wide.eng.ubatch) throw new Error(`wide prefill did not turn on: ${wide.warns}`);
  const A = byBg(base.log), B = byBg(wide.log);
  let same = 0;
  for (const [bg, seq] of A) {
    if (!B.has(bg)) {
      const pipes = new Set(seq.map((x) => x.split(" ")[0]));
      for (const p of pipes) if (!PROJ.test(p)) throw new Error(`${model}: ${p} (bind group ${bg}) never ran on the wide path`);
      continue;
    }
    const w = B.get(bg);
    if (seq.every((x) => PROJ.test(x.split(" ")[0]))) continue;   // replaced on full chunks, still used by the tail
    if (seq.join("|") !== w.join("|")) {
      const k = seq.findIndex((x, i) => x !== w[i]);
      throw new Error(`${model}: bind group ${bg} differs at dispatch ${k}: default "${seq[k]}" vs wide "${w[k]}" (${seq.length} vs ${w.length} dispatches)`);
    }
    same++;
  }
  // the wide GEMMs: every trunk projection over the whole chunk, grid y = width / BN
  const cfg = wide.eng.wideCfg, L = wide.eng.layers.length, chunks = [];
  for (let i = 0; n - i >= cfg.BN; ) { const w = Math.min(U, Math.floor((n - i) / cfg.BN) * cfg.BN); chunks.push(w); i += w; }
  const gw = wide.log.filter((e) => e.pipe.startsWith("gemm_w_"));
  const perLayer = wide.eng.layers.reduce((s, Ly) => s + (Ly.isFull ? (Ly.fKV ? 3 : 4) : (Ly.fQZ ? 2 : 3)) + (Ly.moe ? 0 : 3), 0);
  eq(gw.length, perLayer * chunks.length, `${model}: wide GEMM dispatches`);
  let k = 0;
  for (const w of chunks) for (let j = 0; j < perLayer; j++, k++) eq(gw[k].grid[1], w / cfg.BN, `${model}: GEMM grid y`);
  eq(wide.log.filter((e) => e.pipe === "silu_mul_w").length, wide.eng.layers.filter((Ly) => !Ly.moe).length * chunks.length, `${model}: wide SiLU`);
  // same hidden handed to decode: the last prompt column lands in engine.x on both paths (by position)
  eq(wide.eng.pos, base.eng.pos, `${model}: position after prefill`);
  return { same, chunks, kinds: [...new Set(gw.map((e) => e.pipe))].sort().join(",") };
}

Deno.test("wide prefill: non-projection kernels dispatch exactly as the 16-column path (dense, Q4 + Q8 projections)", async () => {
  const r1 = await compare("dense", 150, 64);
  if (r1.same < 60) throw new Error(`only ${r1.same} shared bind groups compared`);
  const r2 = await compare("q8out", 200, 128, 32768);
  if (!r2.kinds.includes("gemm_w_q8_acc") || !r2.kinds.includes("gemm_w_q4")) throw new Error("Q8 / Q4 GEMMs not both used: " + r2.kinds);
  eq(r2.chunks.join(","), "128,64", "chunking");
});

Deno.test("wide prefill: MoE (experts, router and draft fill per 16-column sub-batch)", async () => {
  const r = await compare("moe", 150, 64);
  if (r.same < 60) throw new Error(`only ${r.same} shared bind groups compared`);
});

Deno.test("wide prefill: off with prefillUbatch 0 and at runtime (engine.prefillWide = false), short prompts", async () => {
  const base = await trace("dense", 100, { prefillUbatch: 0 });
  const def = await engineFor(models.dense);   // the default engine (wide on) switched off at runtime: the 16-column path
  def.eng.prefillWide = false; def.device.log.length = 0;
  await def.eng.prefillTokens(ids(100));
  eq(def.device.log.map((e) => `${e.pipe} ${e.grid} ${e.frame}`).join("|"), base.log.map((e) => `${e.pipe} ${e.grid} ${e.frame}`).join("|"), "prefillWide = false");
  const { eng, device } = await engineFor(models.dense, { engine: { prefillUbatch: 64 } });
  eng.prefillWide = false; device.log.length = 0;
  await eng.prefillTokens(ids(100));
  if (device.log.some((e) => e.pipe.startsWith("gemm_w_"))) throw new Error("runtime switch ignored");
  if (JSON.stringify(eng.stateSignature()).includes('"ub"')) throw new Error("signature with wide prefill off");
  eng.prefillWide = true;
  if (!JSON.stringify(eng.stateSignature()).includes('"ub":64')) throw new Error("signature without the ubatch");
  const short = await trace("dense", 40, { prefillUbatch: 64 });   // shorter than one tile: default path only
  if (short.log.some((e) => e.pipe.startsWith("gemm_w_"))) throw new Error("wide path on a 40-token prompt");
  const bad = await trace("dense", 40, { prefillUbatch: 48 });
  if (bad.eng.ubatch || !bad.warns.some((w) => w.includes("wide prefill off"))) throw new Error("a ubatch that is not a multiple of BN was accepted");
});

Deno.test("prefill defaults: wide on for an engine with the embedding (MoE + expert-grouped, dense), MoE workers too (prefillHidden)", async () => {
  const host = await engineFor(models.moe);
  eq(host.eng.ubatch, 256, "MoE host prefillUbatch default"); eq(host.eng.moeGrpU, 256, "MoE host moeGroupPrefill default");
  eq(!!host.eng.attnPrefillTile, true, "MoE attnPrefillTile default");
  eq(host.warns.length, 0, "MoE host warnings");
  const G = parseGGUFHeader(models.moe.buffer.slice(models.moe.byteOffset, models.moe.byteOffset + models.moe.byteLength));
  const L = G.meta["qwen35.block_count"] - 1, bytesOf = async (i) => models.moe.slice(i.byteOffset, i.byteOffset + i.byteLength);
  const warns = [], origWarn = console.warn; console.warn = (...a) => warns.push(a.join(" "));
  let worker;
  try {
    worker = await Qwen35Engine.create({ device: mockDevice({ wgMem: 16384 }), meta: G.meta, layerRange: [4, L], hasEmbed: false, hasHead: false,
      maxSeq: 512, batchCols: 16, coopRowsB: 1, weights: await qwen35Weights(G, bytesOf, { lo: 4, hi: L, hasEmbed: false, hasHead: false }) });
  } finally { console.warn = origWarn; }
  eq(worker.ubatch, 256, "MoE worker prefillUbatch"); eq(worker.moeGrpU, 256, "MoE worker moeGroupPrefill");
  eq(worker.prefillFrame(), 256, "MoE worker prefill frame");
  eq(!!worker.attnPrefillTile, true, "MoE worker attnPrefillTile"); eq(warns.length, 0, "MoE worker warnings: " + warns.join("; "));
  const dense = await engineFor(models.dense);
  eq(dense.eng.ubatch, 256, "dense prefillUbatch default"); eq(dense.eng.moeGrpU, 0, "dense moeGroupPrefill");
  // dp4a "auto": per model kind, on the devices dp4aAutoDevice accepts (here: Deno, no vendor, not macOS)
  const autoDev = dp4aAutoDevice(null);
  eq(!!dense.eng.dp4aCfg, DP4A_DEFAULT.dense && autoDev, "dense prefillDp4a default"); eq(!!host.eng.dp4aCfg, DP4A_DEFAULT.moe && autoDev, "MoE prefillDp4a default");
  eq(dp4aAutoDevice({ vendor: "nvidia" }), true, "auto on NVIDIA"); eq(dp4aAutoDevice({ vendor: "apple" }), false, "auto on Apple");
  eq(dp4aAutoDevice({ vendor: "amd" }), false, "auto on unmeasured vendors");
  const nv = await engineFor(models.dense, { engine: { adapterInfo: { vendor: "apple" } } });
  eq(!!nv.eng.dp4aCfg, false, "dense prefillDp4a default on Apple"); eq(nv.warns.length, 0, "Apple auto: no warning");
  eq(dense.warns.length, 0, "dense warnings");
});

Deno.test("prefill defaults on a MoE: wide chunks with the expert-grouped kernels inside, valid commands", async () => {
  const r = await trace("moe", 300);   // 256-token wide chunk, then a 32-token grouped ubatch, then 16-column passes
  const n = (p) => r.log.filter((e) => e.pipe.startsWith(p)).length;
  if (!n("gemm_w_")) throw new Error("no wide GEMMs");
  if (!n("moe_gsort") || !n("moe_gusg_") || !n("moe_dng_")) throw new Error("no expert-grouped kernels");
  eq(r.eng.pos, 300, "position after prefill");
});

Deno.test("dp4a wide prefill: each projection input quantized once, int8 GEMMs over the chunk, valid commands, runtime switch", async () => {
  for (const [model, n, U, wgMem] of [["dense", 150, 64], ["q8out", 200, 128, 32768], ["moe", 150, 64]]) {
    const r = await trace(model, n, { prefillUbatch: U, moeGroupPrefill: 0, prefillDp4a: true }, wgMem);
    if (r.warns.length || !r.eng.dp4aCfg) throw new Error(`${model}: dp4a did not turn on: ${r.warns}`);
    const dc = r.eng.dp4aCfg, BN = r.eng.wideCfg.BN, chunks = [];
    for (let i = 0; n - i >= BN; ) { const w = Math.min(U, Math.floor((n - i) / BN) * BN); chunks.push(w); i += w; }
    eq(r.log.filter((e) => e.pipe.startsWith("gemm_w_")).length, 0, `${model}: f32 wide GEMMs with dp4a on`);
    const gd = r.log.filter((e) => e.pipe.startsWith("gemm_d_")), qz = r.log.filter((e) => e.pipe === "quant_q8_w");
    const Ls = r.eng.layers;
    const perLayer = Ls.reduce((s, Ly) => s + (Ly.isFull ? (Ly.fKV ? 3 : 4) : (Ly.fQZ ? 2 : 3)) + (Ly.moe ? 0 : 3), 0);
    const qPerLayer = Ls.reduce((s, Ly) => s + 2 + (Ly.moe ? 0 : 2), 0);   // xn + attention / DeltaNet output (+ xn, SiLU output)
    eq(gd.length, perLayer * chunks.length, `${model}: dp4a GEMM dispatches`);
    eq(qz.length, qPerLayer * chunks.length, `${model}: quantize dispatches`);
    let k = 0;
    for (const w of chunks) for (let j = 0; j < perLayer; j++, k++) eq(gd[k].grid[1], w / dc.BN, `${model}: dp4a GEMM grid y`);
    if (model === "q8out" && !(gd.some((e) => e.pipe === "gemm_d_q8_acc") && gd.some((e) => e.pipe === "gemm_d_q4"))) throw new Error("Q8 / Q4 dp4a GEMMs not both used");
    if (!JSON.stringify(r.eng.stateSignature()).includes('"dp4a":1')) throw new Error(`${model}: signature without dp4a`);
    // runtime switch: the f32 wide GEMMs again, no quantization, signature without dp4a
    r.eng.prefillDp4a = false; r.eng.reset();
    const { log } = r.eng.device;
    log.length = 0;
    await r.eng.prefillTokens(ids(n));
    if (log.some((e) => e.pipe.startsWith("gemm_d_") || e.pipe === "quant_q8_w") || !log.some((e) => e.pipe.startsWith("gemm_w_"))) throw new Error(`${model}: prefillDp4a = false ignored`);
    if (JSON.stringify(r.eng.stateSignature()).includes('"dp4a"')) throw new Error(`${model}: signature with dp4a off`);
  }
  // Apple GPUs: an explicit request warns and stays off
  const apple = await engineFor(models.dense, { engine: { prefillDp4a: true, adapterInfo: { vendor: "apple" } } });
  if (apple.eng.dp4aCfg || !apple.warns.some((w) => w.includes("prefillDp4a"))) throw new Error("dp4a on an Apple GPU");
});

// (the real 27B's full 16-column passes run gate / up as GEMMs, then this SiLU; the synthetic models have no GEMM
// shapes, so plain GEMVs (no fused gate/up) stand in for that here)
Deno.test("16-column pass: one multi-column SiLU (silu_mul_w) per dense FFN, none per column", async () => {
  for (const model of ["dense", "q8out"]) {
    const r = await trace(model, 100, { prefillUbatch: 0, matvecVariant: "plain" });
    // per pass: every dense trunk layer, plus the MTP draft layer when the prefill fills the draft cache
    const passes = Math.ceil(100 / 16), dense = r.eng.layers.filter((Ly) => !Ly.moe).length + (r.eng.mtp ? 1 : 0);
    eq(r.log.filter((e) => e.pipe === "silu_mul").length, 0, `${model}: per-column silu_mul`);
    const sw = r.log.filter((e) => e.pipe === "silu_mul_w");
    eq(sw.length, dense * passes, `${model}: silu_mul_w dispatches`);
    // grid y = the pass's column count (16, then the 4-column tail)
    eq(sw.filter((e) => e.grid[1] === 16).length, dense * Math.floor(100 / 16), `${model}: full passes`);
    // the tail pass: its 4 columns (the draft layer's fill may cover one column fewer: it pairs each token with the next)
    eq(sw.filter((e) => e.grid[1] > 0 && e.grid[1] < 16 && e.grid[1] >= 100 % 16 - 1).length, dense, `${model}: tail pass`);
  }
});

Deno.test("prefillHidden: a room worker's prompt frame through wide, grouped and batchCols passes, every column in place", async () => {
  const G = parseGGUFHeader(models.moe.buffer.slice(models.moe.byteOffset, models.moe.byteOffset + models.moe.byteLength));
  const L = G.meta["qwen35.block_count"] - 1, bytesOf = async (i) => models.moe.slice(i.byteOffset, i.byteOffset + i.byteLength);
  const device = mockDevice({ wgMem: 16384 });
  const worker = await Qwen35Engine.create({ device, meta: G.meta, layerRange: [4, L], hasEmbed: false, hasHead: false,
    maxSeq: 512, batchCols: 16, coopRowsB: 1, weights: await qwen35Weights(G, bytesOf, { lo: 4, hi: L, hasEmbed: false, hasHead: false }) });
  const dim = worker.dims.dim, n = 300, base = 7;   // 256 wide (+ grouped inside), 32 grouped, 12 in one pass
  const xs = Float32Array.from({ length: n * dim }, (_, k) => (k % dim) + 1000 * Math.floor(k / dim));
  device.log.length = 0;
  const out = await worker.prefillHidden(xs, base);
  eq(out.length, n * dim, "output length");
  for (let k = 0; k < out.length; k++) if (out[k] !== xs[k]) throw new Error(`column ${Math.floor(k / dim)} row ${k % dim}: ${out[k]} != ${xs[k]}`);
  eq(worker.pos, base + n, "position after the frame");
  const nk = (p) => device.log.filter((e) => e.pipe.startsWith(p)).length;
  if (!nk("gemm_w_")) throw new Error("no wide GEMMs on the worker");
  if (!nk("moe_gsort")) throw new Error("no expert-grouped kernels on the worker");
  // the shard with the embedding: ids in, the embedding rows out (the mock runs no layers), no draft-cache fill
  const host = await engineFor(models.moe);
  const ids2 = ids(40);   // under one wide tile: grouped (32) + one 8-column pass
  const h = await host.eng.prefillHidden(ids2, 0);
  for (let c = 0; c < ids2.length; c++) {
    const e = host.eng._embedRowF32(ids2[c]);
    for (let r = 0; r < dim; r++) if (h[c * dim + r] !== e[r]) throw new Error(`host column ${c}: not its embedding row`);
  }
  eq(host.eng.pos, 40, "host position");
});
