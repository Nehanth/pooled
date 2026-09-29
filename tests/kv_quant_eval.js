// Quality of the q8_0 KV cache (kvQ8) against the f16 cache at long context, on the real model.
//
// Three engines, run one after another (each loads its own copy of the weights: the engine consumes
// some uploaded tensors, so engines cannot share them; the previous engine's buffers are destroyed):
//   A  f16 KV, engine defaults (the reference; its logits and greedy tokens are kept in memory)
//   B  q8_0 KV (kvQ8: true), otherwise the same
//   C  f16 KV with the tiled prefill attention off (attn_flash for prefill): a noise floor. A and C
//      hold the same KV format and differ only in the prefill summation order, so A vs C is what
//      "numerically equivalent" looks like on this model at this length.
// For each fill F: every engine prefills the same F tokens of a long document, then
//   teacher-forced: the next T document tokens are fed; per position the KL divergence
//     KL(A || X), top-1 agreement, max |logit difference| and the mean NLL of the true token
//   greedy: from the same saved state, G greedy tokens per engine; first divergence and position-wise match
//   CORPUS=code (this repo's source, like bench_ctx.js) | docs (docs/**/*.md prose)
//   MODEL=moe|27b  CTX=<maxSeq>  FILLS=1024,8192,32768  T=128  G=256  NOISE=1 (engine C; 0 skips it)
//   cd tests && MODEL=moe deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights kv_quant_eval.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { argmax } from "../engine/engine.js";
import { openGGUF, gpuDevice, watchGpuErrors, wideOpts, MOE_PATH, Q38_PATH } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "moe");
const FILLS = env("FILLS", "1024,8192,32768").split(",").map(Number);
const T = +env("T", 128), GN = +env("G", 256), NOISE = env("NOISE", "1") !== "0";
const MAXSEQ = +env("CTX", Math.max(...FILLS) + T + GN + 64);
const CORPUS = env("CORPUS", "code");

const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(MODEL === "27b" ? Q38_PATH : MOE_PATH);
const L = model.trunkLayers, tok = model.tokenizer();
console.log(`${MODEL}: ${L} layers; maxSeq ${MAXSEQ}, fills ${FILLS.join(" / ")}, T ${T}, G ${GN}, corpus ${CORPUS}`);
const CFG = { A: {}, B: { kvQ8: true }, ...(NOISE ? { C: { attnPrefillTile: false } } : {}) };
async function mk(name) {
  const t0 = performance.now();
  const w = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: false });
  // not preuploaded: the engine uploads (and fuses) the tensors itself, as in the room
  const { attnPrefillTile, ...o } = CFG[name];
  const e = await Qwen35Engine.create({ device, meta: model.meta, weights: w, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: MAXSEQ, batchCols: 16, coopRowsB: 1, ...wideOpts(), ...o });
  if (attnPrefillTile === false) e.attnPrefillTile = false;
  console.log(`engine ${name}: loaded in ${((performance.now() - t0) / 1000).toFixed(0)} s, kvQ8 ${!!e.kvQ8}, attnPrefillTile ${e.attnPrefillTile}`);
  return { e, w };
}
// destroy every GPUBuffer reachable from the engine and its weights (no engine.destroy() exists)
function destroyAll(root) {
  const seen = new Set(), st = [root];
  while (st.length) {
    const o = st.pop();
    if (!o || typeof o !== "object" || seen.has(o)) continue;
    seen.add(o);
    if (o instanceof GPUBuffer) { try { o.destroy(); } catch {} continue; }
    if (ArrayBuffer.isView(o) || o instanceof ArrayBuffer || o instanceof GPUDevice) continue;
    for (const v of (o instanceof Map ? o.values() : o instanceof Set ? o.values() : Object.values(o))) st.push(v);
  }
}

// document tokens
const need = Math.max(...FILLS) + T + 16;
let ids = [];
const files = CORPUS === "docs"
  ? [...Deno.readDirSync(new URL("../docs/", import.meta.url))].filter((f) => f.isFile && f.name.endsWith(".md")).map((f) => "../docs/" + f.name).sort()
  : ["../engine/qwen35.js", "../room.js", "../engine/gguf.js", "../engine/wgsl/base.js", "../engine/wgsl/moe.js", "../harness/agent.js", "../room/plan.js"];
for (const f of files) {
  try { ids.push(...tok.encode(`\n// file: ${f}\n` + await Deno.readTextFile(new URL(f, import.meta.url)))); } catch {}
  if (ids.length >= need) break;
}
while (ids.length < need) ids = ids.concat(ids);
ids = ids.slice(0, need);

// log-softmax in f64
function logSoftmax(lg) {
  let m = -Infinity; for (let i = 0; i < lg.length; i++) if (lg[i] > m) m = lg[i];
  let s = 0; for (let i = 0; i < lg.length; i++) s += Math.exp(lg[i] - m);
  const lz = m + Math.log(s), out = new Float64Array(lg.length);
  for (let i = 0; i < lg.length; i++) out[i] = lg[i] - lz;
  return out;
}
function compare(la, lx, rawA, rawX) {
  let kl = 0, md = 0;
  for (let i = 0; i < la.length; i++) { const p = Math.exp(la[i]); if (p > 0) kl += p * (la[i] - lx[i]); const d = Math.abs(rawA[i] - rawX[i]); if (d > md) md = d; }
  return { kl: Math.max(kl, 0), md };
}
const q = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };

// run[name][fillIndex] = { prefillS, decodeMs, logits: [Float32Array] (A only), stats, nll, gen }
const run = {};
let mem = null;
const perTok = (e) => { const D = e.dims, n = e.layers.filter((l) => l.isFull).length; return n * 2 * (e.kvQ8 ? D.kvDim + D.kvDim / 32 * 2 : D.kvDim * 2); };
for (const name of Object.keys(CFG)) {
  const { e, w } = await mk(name);
  if (name === "B" && !e.kvQ8) throw new Error("engine B did not enable kvQ8 (needs the flash path and headDim % 32 == 0)");
  if (name === "A") mem = { f16PerTok: perTok(e) };
  if (name === "B") { mem.q8PerTok = perTok(e); mem.ratio = +(mem.q8PerTok / mem.f16PerTok).toFixed(4);
    console.log(`KV per token: f16 ${(mem.f16PerTok / 1024).toFixed(1)} KB, q8_0 ${(mem.q8PerTok / 1024).toFixed(1)} KB (${(mem.ratio * 100).toFixed(1)}%); at ${MAXSEQ}: ${(mem.f16PerTok * MAXSEQ / 2 ** 30).toFixed(2)} GB vs ${(mem.q8PerTok * MAXSEQ / 2 ** 30).toFixed(2)} GB`); }
  run[name] = [];
  let pos = 0;
  for (let fi = 0; fi < FILLS.length; fi++) {
    const fill = FILLS[fi], chunk = ids.slice(pos, fill);
    let t0 = performance.now();
    if (chunk.length > 1) await e.prefillTokens(chunk.slice(0, -1));
    await e.forwardToken(chunk[chunk.length - 1]);
    const prefillS = (performance.now() - t0) / 1000;
    e.saveSlot("fill");
    pos = fill;
    const r = { prefillS, nll: 0 };
    const ref = name === "A" ? null : run.A[fi];
    if (!ref) r.logits = [];
    else r.st = { kl: [], md: [], top1: 0 };
    t0 = performance.now();
    for (let t = 0; t < T; t++) {
      const lg = await e.forwardToken(ids[fill + t]);
      const ls = logSoftmax(lg);
      r.nll -= ls[ids[fill + t + 1]];
      if (!ref) r.logits.push(Float32Array.from(lg));
      else { const la = logSoftmax(ref.logits[t]); const c = compare(la, ls, ref.logits[t], lg); r.st.kl.push(c.kl); r.st.md.push(c.md); if (argmax(lg) === argmax(ref.logits[t])) r.st.top1++; }
    }
    r.tfS = (performance.now() - t0) / 1000;
    e.loadSlot("fill");
    let next = ids[fill]; r.gen = [];
    t0 = performance.now();
    for (let i = 0; i < GN; i++) { const l = await e.forwardToken(next); next = argmax(l); r.gen.push(next); }
    r.decodeTokS = GN / ((performance.now() - t0) / 1000);
    e.loadSlot("fill"); e.dropSlot("fill");
    console.log(`engine ${name} fill ${fill}: prefill ${prefillS.toFixed(1)} s, greedy decode ${r.decodeTokS.toFixed(1)} tok/s (with readback of full logits), ppl ${Math.exp(r.nll / T).toFixed(4)}`);
    run[name].push(r);
  }
  destroyAll(e); destroyAll(w);
  await device.queue.onSubmittedWorkDone();
}

const rows = [];
for (let fi = 0; fi < FILLS.length; fi++) {
  const fill = FILLS[fi], A = run.A[fi];
  const r = { fill, prefillS: {}, decodeTokS: {}, ppl: {} };
  for (const n of Object.keys(run)) { const x = run[n][fi]; r.prefillS[n] = +x.prefillS.toFixed(1); r.decodeTokS[n] = +x.decodeTokS.toFixed(1); r.ppl[n] = +Math.exp(x.nll / T).toFixed(4); }
  for (const n of Object.keys(run).filter((n) => n !== "A")) {
    const x = run[n][fi], s = x.st, mean = s.kl.reduce((a, b) => a + b, 0) / T;
    let first = A.gen.findIndex((t, i) => x.gen[i] !== t); if (first < 0) first = GN;
    r[n] = { klMean: +mean.toExponential(3), klP99: +q(s.kl, 0.99).toExponential(3), klMax: +Math.max(...s.kl).toExponential(3), top1: +(s.top1 / T).toFixed(4),
      maxLogitDiffMean: +(s.md.reduce((a, b) => a + b, 0) / T).toFixed(4), greedyFirstDiverge: first, greedySame: +(A.gen.filter((t, i) => x.gen[i] === t).length / GN).toFixed(4) };
    console.log(`fill ${String(fill).padStart(6)} ${n === "B" ? "q8_0 " : "noise"} vs f16: KL mean ${r[n].klMean} p99 ${r[n].klP99} max ${r[n].klMax} · top-1 ${(r[n].top1 * 100).toFixed(1)}% · max|dlogit| ${r[n].maxLogitDiffMean} · greedy same until ${first}/${GN} (${(r[n].greedySame * 100).toFixed(1)}% positions)`);
  }
  console.log(`fill ${String(fill).padStart(6)} ppl ${JSON.stringify(r.ppl)} prefill s ${JSON.stringify(r.prefillS)} decode tok/s ${JSON.stringify(r.decodeTokS)}`);
  if (A.gen.join() !== run.B[fi].gen.join()) console.log("  f16 : " + JSON.stringify(tok.decode(A.gen).slice(0, 200)) + "\n  q8_0: " + JSON.stringify(tok.decode(run.B[fi].gen).slice(0, 200)));
  rows.push(r);
}
console.log("RESULT " + JSON.stringify({ model: MODEL, maxSeq: MAXSEQ, corpus: CORPUS, T, G: GN, mem, rows, gpuErrors: errors.count }));
if (errors.count) Deno.exit(1);
