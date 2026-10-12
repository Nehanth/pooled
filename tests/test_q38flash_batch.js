// Qwen3.8-Flash-Next batched passes (engine/qwen4exp_batch.js) and QSA (engine/qwen4exp_qsa.js) on the engine
// (engine/qwen4exp.js). Roadmap 33 PR plan items 5 and 9. The model: bartowski Qwen3.8-Flash-Next-Q4_0 in
// models/q38flash (Q38F_DIR=<dir> elsewhere), loaded with the indexer weights (qsa: true).
//   deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights tests/test_q38flash_batch.js
//
// Default: the first LAYERS (4) layers, no head (DeltaNet 0, the PLE layer 1, DeltaNet 2, QSA layer 3), on the
// first NTOK (4300) tokens of the ctx8k golden's prompt, so the context passes QSA's 2051 cells. The reference is
// the decode engine, one token at a time; every check compares the output residual (4 x 2560 floats) bit for bit:
//   QSA vs reference    past 2051 cells, the decode engine's QSA state on the real activations of the last QSA layer
//                       (layer 3) against the CPU reference (tests/reference/qsa_ref.mjs): raw keys, every pooled
//                       key (re-pooled on the CPU from the GPU's raw keys), and at sampled positions the indexer
//                       query, the block scores and the selected row list. The only check of qsa_score / qsa_select
//                       against something independent on real data; prefill == decode below carries it to the passes
//   prefill == decode   passes of mixed sizes (1 .. 64 columns, some across the 2051 / 2052 boundary)
//   interleave          a prefill, then decode steps, then a prefill again
//   QSA == dense        below 2052 cells: every pass forced onto the gathered (QSA) attention, decode and prefill
//   verify / accept     verify passes of 6 columns with random accepts (state restored and re-run), across 2052
//   timing              decode vs prefill tok/s for these layers
// OFFLOAD=lo-hi [OFFLOAD_SLOTS=N]: those layers' routed experts parked in the expert store (engine/expert_store.js),
// the engine built with expertRegion: true, so the batched passes route them through the store's region and decode
// through its pools. Every check above still compares against the decode engine; the "BITS" line (a hash of the
// 64-column prefill's residuals) must equal the resident run's.
// FULL=1: all 48 layers and the head (~75 GB of GPU memory): greedy ids vs llama.cpp
// (tests/golden/q38flash_llama_greedy.json, every case including ctx4k / ctx8k; ctx2k and ctx2k_b stay dense at
// long positions (1887 and 1943 prompt tokens, + 64 <= 2051 cells), ctx2k1 (2082) is just past 2051 cells, where
// QSA drops a few blocks) with the prompt run by the batched prefill, judged by tests/q38flash_rule.js (a difference
// passes only at a backend near tie: our token within 0.5 nats of llama.cpp's top there, our own top-2 gap under
// 0.5); the prefill's greedy id == the decode engine's on the prompts of <= 256 tokens; and a verify pass of 8
// columns over the golden continuation giving the decode steps' ids, accept(8) (the full-length accept only; partial
// accepts are covered by the layered mode). Cases that share a prompt prefix and fail at the same step on the same
// token count as one failure seen again: a near tie in the later one does not excuse it.
// LONG=1: long context on the full model (all 48 layers + head, QSA on): the long goldens (ctx16k, ctx32k and
// needle30k, ~15.7K / ~32K / ~29.7K prompt tokens; CASES=a,b picks others) run by the batched prefill, then decode,
// judged by tests/q38flash_rule.js; per case the prefill time and tok/s, and decode tok/s. needle30k is a needle in
// a haystack: a passphrase planted ~2.5K tokens in, asked for at the end; it fails when llama.cpp's answer has the
// passphrase and ours does not (and its ids are judged as any golden's). MAXSEQ defaults to the longest case's prompt
// + ids, rounded up to 256 (f32 K / V for 12 layers: ~1.6 GB at 32K). llama.cpp's 3 runs of these prompts part early
// (ctx32k at step 0), so it also judges us against each run alone (that run's ids and top-5), and a case also passes
// when the rule passes against one of llama.cpp's reruns. No verify pass or decode-only prefill here (one
// 32K prefill is ~15 minutes on the GB10 at ~35 tok/s).
import fs from "node:fs";
import path from "node:path";
import { Qwen4ExpEngine } from "../engine/qwen4exp.js";
import { Qwen4ExpBatch } from "../engine/qwen4exp_batch.js";
import { qwen4expWeights, qwen4expConfig, gpuUploadEntry } from "../engine/gguf.js";
import { ngramConfig, NgramTable } from "../engine/ngram.js";
import { openGGUF, gpuDevice, watchGpuErrors, offloadFromEnv } from "./load_model.js";
import { judgeGolden } from "./q38flash_rule.js";
import { QSA, QsaRefCache, poolBlock, indexerQuery, scoreBlocks, selectBlocks, selectedCells, f16bits, h2f } from "./reference/qsa_ref.mjs";

const env = (k, d) => Deno.env.get(k) ?? d;
const DIR = env("Q38F_DIR", new URL("../models/q38flash", import.meta.url).pathname);
const first = fs.readdirSync(DIR).filter((f) => /^Qwen3\.8-Flash-Next-.*-00001-of-\d{5}\.gguf$/.test(f))[0];
if (!first) throw new Error("no Qwen3.8-Flash-Next shards in " + DIR);
const LONG = env("LONG") === "1", FULL = LONG || env("FULL") === "1";
const gold = JSON.parse(fs.readFileSync(new URL("./golden/q38flash_llama_greedy.json", import.meta.url), "utf8"));
const LONG_CASES = env("CASES") ? env("CASES").split(",") : ["ctx16k", "ctx32k", "needle30k"];
if (LONG) for (const c of LONG_CASES) if (!gold[c]) throw new Error(`LONG=1: no golden "${c}" (tests/golden/make_q38flash_llama.js)`);
const longSeq = LONG ? Math.ceil(Math.max(...LONG_CASES.map((c) => gold[c].prompt.length + gold[c].ids.length)) / 256) * 256 : 0;

const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(path.join(DIR, first));
const G = model.G, cfg = qwen4expConfig(G.meta);
const nL = FULL ? cfg.L : +env("LAYERS", 4), hasHead = nL === cfg.L;
const MAXSEQ = +env("MAXSEQ", LONG ? longSeq : FULL ? 8448 : 4608);

const experts = await offloadFromEnv(device, nL, model);   // (async since the disk tier, #349; a disk layer is refused by Qwen4ExpBatch.create)
let t0 = performance.now();
const up = (e, name) => { if (name !== "token_embd.weight") gpuUploadEntry(device, e, false); };
const flush = async () => { device.queue.submit([device.createCommandEncoder().finish()]); await device.queue.onSubmittedWorkDone(); };
const layers = [];
for (let i = 0; i < nL; i++) {
  layers.push((await qwen4expWeights(G, model.bytesOf, { lo: i, hi: i + 1, hasEmbed: false, hasHead: false, qsa: true, experts }, undefined, up)).layers[0]);
  await flush();
  if (i % 8 === 7 || i === nL - 1) console.log(`  layer ${i + 1}/${nL} on the GPU, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
const rest = await qwen4expWeights(G, model.bytesOf, { lo: nL, hi: nL, hasEmbed: true, hasHead, qsa: true }, undefined, up);
await flush();
const weights = { ...rest, layers };
const fds = model.paths.map((p) => fs.openSync(p, "r"));
const readAt = async (file, off, len) => { const b = new Uint8Array(len); fs.readSync(fds[file], b, 0, len, off); return b; };
const ngram = new NgramTable(ngramConfig(G), { readAt, layout: "tensor" });
const eng = await Qwen4ExpEngine.create({ device, meta: G.meta, weights, layerRange: [0, nL], hasEmbed: true, hasHead, ngram, maxSeq: MAXSEQ, expertRegion: !!experts });
const pb = await Qwen4ExpBatch.create(eng, weights, { maxCols: +env("MAXCOLS", 64), verifyCols: 8 });
if (experts) console.log(`offload: layers ${experts.layers[0]}-${experts.layers.at(-1)}, ${experts.P} slots per layer, region ${(experts.regionBytes / 2 ** 30).toFixed(2)} GiB`);
console.log(`qwen4exp: ${nL} layers${hasHead ? " + head" : ""}, QSA ${eng.qsa ? "on" : "off"}, maxSeq ${MAXSEQ}, loaded in ${((performance.now() - t0) / 1000).toFixed(0)} s`);

let fail = 0;
const check = (ok, msg) => { console.log(`${ok ? "ok  " : "FAIL"} ${msg}`); if (!ok) fail++; };
if (LONG) await long(); else if (FULL) await full(); else await layered();
if (experts) console.log(experts.summary());
if (errors.count) { console.log(`${errors.count} GPU errors`); fail++; }
console.log(fail ? "Q38FLASH BATCH FAIL" : "Q38FLASH BATCH PASS ✓");
Deno.exit(fail ? 1 : 0);

// ---- bit comparison of residuals: -> { bad (columns that differ), worst (max |a - b| / max |b|) }
function cmpCols(got, ref, t0, n) {
  const HD = cfg.hDim;
  let bad = 0, worst = 0, firstBad = -1;
  for (let c = 0; c < n; c++) {
    const a = new Uint32Array(got.buffer, got.byteOffset + c * HD * 4, HD), r = ref[t0 + c], rb = new Uint32Array(r.buffer, r.byteOffset, HD);
    let diff = false;
    for (let i = 0; i < HD; i++) if (a[i] !== rb[i]) { diff = true; break; }
    if (diff) {
      bad++; if (firstBad < 0) firstBad = t0 + c;
      const af = new Float32Array(a.buffer, a.byteOffset, HD);
      let md = 0, mr = 0; for (let i = 0; i < HD; i++) { md = Math.max(md, Math.abs(af[i] - r[i])); mr = Math.max(mr, Math.abs(r[i])); }
      worst = Math.max(worst, md / mr);
    }
  }
  return { bad, worst, firstBad };
}

async function layered() {
  const NTOK = Math.min(+env("NTOK", 4300), MAXSEQ, gold.ctx8k.prompt.length);
  const ids = gold.ctx8k.prompt.slice(0, NTOK);
  // the reference: decode steps
  eng.reset();
  const ref = [];
  // positions where the QSA check reads the decode step's indexer scratch (past 2051 cells: scored and selected)
  const qsaAt = eng.qsa ? [2051, 2052, 2053, 2600, 3333, NTOK - 1].filter((p, i, a) => p < NTOK && a.indexOf(p) === i) : [];
  const qsaSnap = new Map();
  let t = performance.now(), tRead = 0;
  for (let p = 0; p < NTOK; p++) {
    ref.push((await eng.forwardToken(ids[p])).hidden);
    if (qsaAt.includes(p)) { const tr = performance.now(); qsaSnap.set(p, await readQsaScratch(p)); tRead += performance.now() - tr; }
  }
  const decTs = NTOK / ((performance.now() - t - tRead) / 1000);
  console.log(`decode reference: ${NTOK} tokens, ${decTs.toFixed(1)} tok/s (${nL} layers)`);
  if (eng.qsa) await qsaVsRef(NTOK, qsaSnap);

  // prefill == decode, mixed pass sizes
  const run = async (label, chunks, n = NTOK, from = 0) => {
    let bad = 0, worst = 0, firstBad = -1, passes = 0, h = 0x811c9dc5;
    const tt = performance.now();
    await pb.prefill(ids.slice(from, from + n), { chunks, head: "none", wantHidden: true, onPass: (r, o, sz) => {
      const c = cmpCols(r.hiddens, ref, from + o, sz); passes++;
      for (const w of new Uint32Array(r.hiddens.buffer, r.hiddens.byteOffset, r.hiddens.length)) h = Math.imul(h ^ w, 0x01000193) >>> 0;
      bad += c.bad; worst = Math.max(worst, c.worst); if (firstBad < 0 && c.firstBad >= 0) firstBad = c.firstBad;
    } });
    const ts = n / ((performance.now() - tt) / 1000);
    check(bad === 0, `${label}: ${n} tokens in ${passes} passes, ${bad} columns differ from decode${bad ? ` (first at ${firstBad}, worst rel ${worst.toExponential(2)})` : " (bit-equal)"}; ${ts.toFixed(1)} tok/s incl. readback`);
    return { ts, hash: h.toString(16).padStart(8, "0") };
  };
  eng.reset();
  await run("prefill == decode, passes 1,5,64,37,64,3,64,64,2,63,17,64", [1, 5, 64, 37, 64, 3, 64, 64, 2, 63, 17, 64]);
  eng.reset();
  const b64 = await run("prefill == decode, 64-column passes", [64]);
  console.log(`BITS ${nL} layers, ${NTOK} tokens, 64-column prefill residual hash ${b64.hash}${experts ? " (offloaded)" : " (resident)"}`);
  // around the boundary: passes that end at 2050, 2051, 2052 and straddle it
  eng.reset();
  await run("prefill == decode, passes across 2051 / 2052", [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 2, 1, 1, 1, 1, 7, 64], Math.min(NTOK, 2200));
  // the plain prefill speed (no hidden readback per pass)
  eng.reset();
  t = performance.now();
  await pb.prefill(ids, { head: "none" });
  await device.queue.onSubmittedWorkDone();
  const pfTs = NTOK / ((performance.now() - t) / 1000);
  console.log(`prefill: ${NTOK} tokens, ${pfTs.toFixed(1)} tok/s (${nL} layers, ${pb.maxCols}-column passes) vs decode ${decTs.toFixed(1)}: ${(pfTs / decTs).toFixed(1)}x`);

  // interleave: prefill, decode steps, prefill again
  {
    eng.reset();
    let bad = 0;
    const a = Math.min(1500, NTOK - 140), b = a + 20;
    const r1 = await pb.prefill(ids.slice(0, a), { head: "none", wantHidden: true });
    bad += cmpCols(r1.hiddens, ref, a - r1.hiddens.length / cfg.hDim, r1.hiddens.length / cfg.hDim).bad;
    for (let p = a; p < b; p++) bad += cmpCols((await eng.forwardToken(ids[p])).hidden, ref, p, 1).bad;
    const r2 = await pb.prefill(ids.slice(b, b + 100), { head: "none", wantHidden: true, chunks: [100] });
    bad += cmpCols(r2.hiddens, ref, b + 100 - r2.hiddens.length / cfg.hDim, r2.hiddens.length / cfg.hDim).bad;
    check(bad === 0, `interleave: prefill 0..${a}, decode ${a}..${b}, prefill ${b}..${b + 100}: ${bad} columns differ`);
  }

  // QSA == dense below 2052 cells: every pass on the gathered attention
  if (eng.qsa) {
    eng.reset(); eng.qsaAlways = true;
    let bad = 0;
    const nd = Math.min(600, NTOK);
    for (let p = 0; p < nd; p++) bad += cmpCols((await eng.forwardToken(ids[p])).hidden, ref, p, 1).bad;
    check(bad === 0, `QSA == dense, decode: positions 0..${nd - 1} on the gathered attention, ${bad} differ from dense`);
    eng.reset();
    await run("QSA == dense, prefill: positions 0..2050 on the gathered attention", [64, 31, 64, 1], Math.min(2051, NTOK));
    eng.qsaAlways = false;
  }

  // verify / accept across 2052
  {
    eng.reset();
    const start = Math.min(2030, NTOK - 80);
    await pb.prefill(ids.slice(0, start), { head: "none" });
    let p = start, bad = 0, rounds = 0, seed = 7;
    const rnd = () => { seed = (seed * 1103515245 + 12345) >>> 0; return seed / 2 ** 32; };
    while (p + 6 <= Math.min(NTOK, start + 70)) {
      const r = await pb.verify(ids.slice(p, p + 6), { wantHidden: true });
      bad += cmpCols(r.hiddens, ref, p, 6).bad;
      const k = Math.floor(rnd() * 7);
      await pb.accept(k);
      p += k; rounds++;
    }
    // the state after the accepts: the next pass must still match the decode steps
    const r = await pb.prefill(ids.slice(p, p + 8), { head: "none", wantHidden: true, chunks: [8] });
    bad += cmpCols(r.hiddens, ref, p, 8).bad;
    check(bad === 0, `verify / accept: ${rounds} verify passes of 6 from ${start} (random accepts, to ${p}), then a pass: ${bad} columns differ from decode`);
  }
}

// ---- QSA on the real activations vs the CPU reference (decode engine, the last QSA layer in range)
async function readBuf(b, bytes, off = 0) {
  const st = device.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = device.createCommandEncoder(); e.copyBufferToBuffer(b, off, st, 0, Math.ceil(bytes / 4) * 4); device.queue.submit([e.finish()]);
  await st.mapAsync(GPUMapMode.READ); const out = st.getMappedRange().slice(0); st.unmap(); st.destroy(); return out;
}
// the decode scratch after the step at p: the projection [q 512 ; k 128], the indexer query, the scores and the row list
async function readQsaScratch(p) {
  const S = eng.qsaS, nv = Math.floor((p + 1) / QSA.KPOOL);
  const sel = new Uint32Array(await readBuf(S.sel, (QSA.MAX_SEL + 1) * 4));
  return { proj: new Float32Array(await readBuf(S.proj, 640 * 4)), qi: new Float32Array(await readBuf(S.qi, 512 * 4)),
    scores: nv > QSA.N_TOP ? new Float32Array(await readBuf(S.scores, nv * 4)) : null, rows: sel.slice(1, 1 + sel[0]) };
}
function f16s(ab) { return Float32Array.from(new Uint16Array(ab), h2f); }
function rel(g, r) { let md = 0, mr = 0; for (let i = 0; i < r.length; i++) { md = Math.max(md, Math.abs(g[i] - r[i])); mr = Math.max(mr, Math.abs(r[i])); } return md / mr; }

async function qsaVsRef(n, snaps) {
  const li = eng.layers.findLastIndex((l) => l.isQ), l = eng.layers[li], ix = weights.layers[li].indexer;
  if (cfg.ropeTheta !== QSA.THETA || cfg.ropeDims !== QSA.NROT) console.log(`  note: the reference ropes with ${QSA.NROT} dims, base ${QSA.THETA}; the model has ${cfg.ropeDims}, ${cfg.ropeTheta}`);
  const tag = `QSA vs reference, layer ${l.i}`;
  // raw keys: the GPU's cache, and at the sampled positions == the step's own k projection rounded to f16
  const cache = new QsaRefCache({ maxSeq: n });
  cache.rawK.set(f16s(await readBuf(l.st.qsa.rawK, n * QSA.ID * 2)));
  cache.n = n;
  let rawBad = 0;
  for (const [p, s] of snaps) for (let d = 0; d < QSA.ID; d++) if (f16bits(s.proj[512 + d]) !== f16bits(cache.rawK[p * QSA.ID + d])) rawBad++;
  // pooled keys: every complete block re-pooled on the CPU from those raw keys, vs the GPU's
  const nb = Math.floor(n / QSA.KPOOL), gp = new Uint16Array(await readBuf(l.st.qsa.pooled, nb * QSA.ID * 2));
  for (let b = 0; b < nb; b++) poolBlock(cache, b, ix.kNorm.data);
  let pd = 0, pmu = 0;
  for (let i = 0; i < nb * QSA.ID; i++) if (gp[i] !== f16bits(cache.pooled[i])) { pd++; pmu = Math.max(pmu, Math.abs(h2f(gp[i]) - cache.pooled[i])); }
  check(rawBad <= 2 && pmu <= 2e-3, `${tag}: raw keys at ${snaps.size} positions == k_proj (${rawBad} f16 off); ${nb} pooled keys re-pooled on the CPU: ${pd}/${nb * QSA.ID} f16 values differ (max |d| ${pmu.toExponential(1)}, rounding flips only)`);
  // with the GPU's pooled keys from here on, so a pooling flip does not count twice
  cache.pooled.set(Float32Array.from(gp, h2f));
  for (const [p, s] of snaps) {
    const nv = Math.floor((p + 1) / QSA.KPOOL);
    const qi = indexerQuery(s.proj, 0, p, ix.qNorm.data), qiRel = rel(s.qi, qi);
    const sR = nv > QSA.N_TOP ? scoreBlocks(cache, qi, nv) : null;
    const want = selectedCells(selectBlocks(sR, nv), p), got = s.rows;
    const same = got.length === want.length && got.every((v, i) => v === want[i]);
    let note = "", ok = qiRel < 1e-5 && (!sR || rel(s.scores, sR) < 1e-4);
    // the GPU list must be the reference rule applied to the GPU's own scores, exactly
    const own = s.scores ? selectedCells(selectBlocks(s.scores, nv), p) : want;
    const ownSame = own.length === got.length && own.every((v, i) => v === got[i]);
    if (!same && sR) {   // vs the reference's scores: allowed only as swaps within the measured score error of the cut
      let err = 0; for (let b = 0; b < nv; b++) err = Math.max(err, Math.abs(s.scores[b] - sR[b]));
      const gs = new Set(), ws = new Set();
      for (let i = 0; i + 3 < got.length && i < QSA.N_TOP * 4; i += 4) gs.add(got[i] / 4);
      for (let i = 0; i + 3 < want.length && i < QSA.N_TOP * 4; i += 4) ws.add(want[i] / 4);
      const onlyG = [...gs].filter((b) => !ws.has(b)), onlyW = [...ws].filter((b) => !gs.has(b));
      const T = Float32Array.from(sR).sort().reverse()[QSA.N_TOP - 1];
      const off = Math.max(...onlyG.concat(onlyW).map((b) => Math.abs(sR[b] - T)));
      const tie = onlyG.length === onlyW.length && off <= 2 * err;
      note = `, ${onlyG.length} block swap(s) at ${off.toExponential(1)} from the cut (score error ${err.toExponential(1)})${tie ? ": near tie" : ""}`;
      ok = ok && tie;
    } else if (!same) ok = false;
    check(ok && ownSame, `${tag}, decode at ${p} (${nv} blocks): ${got.length} rows, list ${same ? "== reference" : "differs"}${note}; == rule on GPU scores: ${ownSame}; qi rel ${qiRel.toExponential(1)}${sR ? `, scores rel ${rel(s.scores, sR).toExponential(1)}` : ""}`);
  }
}

function gap(lg) {
  let a = -Infinity, b = -Infinity;
  for (const v of lg) { if (v > a) { b = a; a = v; } else if (v > b) b = v; }
  return a - b;
}

async function full() {
  const tok = model.tokenizer();
  const pick = env("CASES") ? env("CASES").split(",") : null;
  const results = [];
  const sharedPrefix = (a, b) => { let i = 0; while (i < a.length && i < b.length && a[i] === b[i]) i++; return i; };
  const cases = Object.entries(gold).filter(([k, v]) => !k.startsWith("_") && (!pick || pick.includes(k)) && v.prompt.length + v.ids.length <= eng.maxSeq);
  for (const [name, g] of cases) {
    const N = Math.min(+env("TOKENS", g.ids.length), g.ids.length);
    // decode-only greedy id after the prompt, for the short prompts: the prefill's must equal it
    let decodeFirst = null;
    if (g.prompt.length <= 256) { eng.reset(); decodeFirst = (await eng.prefillTokens(g.prompt)).id; }
    eng.reset();
    let t = performance.now();
    let r = await pb.prefill(g.prompt, { wantLogits: true });
    const pf = (performance.now() - t) / 1000;
    if (decodeFirst !== null) check(r.id === decodeFirst, `${name}: prefill greedy id ${r.id} == decode's ${decodeFirst}`);
    const gen = [r.id], gaps = [gap(r.logits)];
    t = performance.now();
    while (gen.length < N) { r = await eng.forwardToken(gen[gen.length - 1], { wantLogits: true }); gen.push(r.id); gaps.push(gap(r.logits)); }
    const ts = (N - 1) / ((performance.now() - t) / 1000);
    const J = judgeGolden(g, gen, gaps, N), { diff, ok } = J;
    let tieOk = J.excused;
    // a near tie excuses a divergence only when it is a failure of its own: a case whose prompt shares a long prefix
    // with an earlier case that already failed at the same step on the same token is that failure seen again
    // (ctx8k repeats ctx4k's first 3835 prompt tokens and its first 6 ids), so it counts as failed too
    const twin = !ok && results.find((o) => !o.ok && !o.tieOk && o.diff === diff && o.gen[diff] === gen[diff]
      && g.ids.slice(0, diff + 1).every((x, i) => x === o.ids[i]) && sharedPrefix(o.prompt, g.prompt) >= 1024);
    const withdrawn = twin && tieOk;
    if (twin) tieOk = false;
    results.push({ name, ok, tieOk, diff, gen, ids: g.ids, prompt: g.prompt });
    console.log(`${ok || tieOk ? "ok  " : "FAIL"} ${name}: prompt ${g.prompt.length} tok, prefill ${pf.toFixed(1)} s (${(g.prompt.length / pf).toFixed(0)} tok/s) · decode ${ts.toFixed(2)} tok/s · ${J.msg}${twin ? ` (the same divergence as ${twin.name}: shared prompt prefix, same step and token${withdrawn ? "; the near tie does not excuse it" : ""})` : ""}`);
    if (!ok) console.log(`  engine: ${JSON.stringify(tok.decode(gen))}\n  golden: ${JSON.stringify(g.text)}\n  top-2 gap (nats) ours / llama.cpp, steps 0..${diff}: ${gaps.slice(0, diff + 1).map((x, i) => `${x.toFixed(3)}/${g.gap?.[i]?.toFixed(3)}`).join(" ")}`);
    if (!ok && !tieOk) fail++;
    // verify: the prompt again, then one verify pass over [last prompt token, gen[0..6]] gives gen[0..7]
    if (gen.length >= 8) {
      eng.reset();
      await pb.prefill(g.prompt.slice(0, -1), { head: "none" });
      const v = await pb.verify([g.prompt[g.prompt.length - 1], ...gen.slice(0, 7)]);
      await pb.accept(8);
      const same = v.ids.every((x, i) => x === gen[i]);
      check(same, `${name}: verify pass ids ${v.ids.join(",")} == decode ${gen.slice(0, 8).join(",")}`);
    }
  }
  // verifyCols: 1: the logits hold one column, so the prefill's head (on its last column) must run on exactly one
  // column (the frame's nCols set to 1 before it; gemv_mc would otherwise fill GEMV_MC_COLS columns, past B.hl).
  // Same engine, a second batch: its 64-column prefill's logits == the verifyCols 8 batch's bit for bit, its id ==
  // the decode engine's, and a 1-column verify gives decode's next id.
  {
    const g = gold["two-sum"], pb1 = await Qwen4ExpBatch.create(eng, weights, { maxCols: 64, verifyCols: 1 });
    eng.reset(); const dec = await eng.prefillTokens(g.prompt, { wantLogits: true }), next = (await eng.forwardToken(dec.id)).id;
    eng.reset(); const r8 = await pb.prefill(g.prompt, { wantLogits: true });
    eng.reset(); const r1 = await pb1.prefill(g.prompt, { wantLogits: true });
    const same = new Uint32Array(r1.logits.buffer).every((x, i) => x === new Uint32Array(r8.logits.buffer)[i]);
    const v = await pb1.verify([r1.id]);
    await pb1.accept(1);
    check(same && r1.id === dec.id && v.ids[0] === next && errors.count === 0,
      `verifyCols 1: prefill logits ${same ? "==" : "differ from"} the verifyCols 8 batch's, id ${r1.id} vs decode ${dec.id}; 1-column verify id ${v.ids[0]} vs decode ${next}`);
  }
  // by attention phase: dense (every step <= 2051 cells) and sparse (QSA drops blocks); exact matches only
  const phase = (sparse) => results.filter((r) => (r.prompt.length + r.ids.length > 2051) === sparse);
  for (const [label, rs] of [["dense (<= 2051 cells)", phase(false)], ["long context (QSA past 2051 cells)", phase(true)]]) {
    if (rs.length) console.log(`${label}: ${rs.filter((r) => r.ok).length}/${rs.length} goldens match llama.cpp exactly (${rs.map((r) => `${r.name} ${r.ok ? "match" : `${r.tieOk ? "near tie" : "MISMATCH"} at ${r.diff}`}`).join(", ")})`);
  }
}

// ---- LONG=1: the long-context goldens, batched prefill (QSA) then decode
async function long() {
  const tok = model.tokenizer(), rows = [];
  for (const name of LONG_CASES) {
    const g = gold[name], N = Math.min(+env("TOKENS", g.ids.length), g.ids.length);
    if (g.prompt.length + N > eng.maxSeq) { check(false, `${name}: ${g.prompt.length} + ${N} > maxSeq ${eng.maxSeq}`); continue; }
    eng.reset();
    let t = performance.now(), next = 4096;
    let r = await pb.prefill(g.prompt, { wantLogits: true, onPass: (_, o, sz) => {
      if (o + sz >= next) { next += 4096; console.log(`  ${name}: prefill ${o + sz}/${g.prompt.length}, ${((performance.now() - t) / 1000).toFixed(0)} s, ${((o + sz) / ((performance.now() - t) / 1000)).toFixed(1)} tok/s`); }
    } });
    const pf = (performance.now() - t) / 1000;
    const gen = [r.id], gaps = [gap(r.logits)];
    t = performance.now();
    while (gen.length < N) { r = await eng.forwardToken(gen[gen.length - 1], { wantLogits: true }); gen.push(r.id); gaps.push(gap(r.logits)); }
    const ts = (N - 1) / ((performance.now() - t) / 1000);
    const J = judgeGolden(g, gen, gaps, N), text = tok.decode(gen);
    // llama.cpp is not reproducible here (its 3 runs part at steps 0-40): the rule against each run on its own, with
    // that run's ids and top-5. The case passes by the rule on the golden (as FULL=1), or by the rule on one of
    // llama.cpp's own reruns: matching a run llama.cpp itself produced, through the same rule, is as good as matching
    // the first (needle30k's run 1: llama.cpp's golden and rerun take different tokens at step 40, our gap there 0.53)
    const runs = [{ ids: g.ids, top: g.top }, ...(g.runs || [])];
    const JR = runs.map((rn) => judgeGolden({ ids: rn.ids, top: rn.top }, gen, gaps, Math.min(N, rn.ids.length)));
    const byRun = JR.findIndex((Jk, k) => k > 0 && (Jk.ok || Jk.excused));
    let ok = J.ok || J.excused || byRun > 0, needle = "";
    if (g.needle) {
      const ours = text.includes(g.needle), theirs = g.found ?? g.text.includes(g.needle);
      needle = ` · needle "${g.needle}": ours ${ours ? "found" : "MISSING"}, llama.cpp ${theirs ? "found" : "missing"}`;
      if (theirs && !ours) ok = false;
    }
    console.log(`${ok ? "ok  " : "FAIL"} ${name}: prompt ${g.prompt.length} tok, prefill ${pf.toFixed(0)} s (${(g.prompt.length / pf).toFixed(1)} tok/s) · decode ${ts.toFixed(2)} tok/s at ${g.prompt.length} ctx · ${J.msg}${needle}`);
    console.log(`  engine: ${JSON.stringify(text)}`);
    if (!J.ok) console.log(`  golden: ${JSON.stringify(g.text)}\n  top-2 gap (nats) ours / llama.cpp, steps 0..${J.diff}: ${gaps.slice(0, J.diff + 1).map((x, i) => `${x.toFixed(3)}/${g.gap?.[i]?.toFixed(3)}`).join(" ")}`);
    console.log(`  vs each llama.cpp run (0: the golden): ${JR.map((Jk, k) => `run ${k}: ${Jk.ok ? `all ${Math.min(N, runs[k].ids.length)} ids` : `${Jk.diff} ids, then ${Jk.excused ? "near tie" : "MISMATCH"}`}`).join(", ")}${!(J.ok || J.excused) && byRun > 0 ? ` (passes by run ${byRun})` : ""}`);
    if (!ok) fail++;
    rows.push(`${name} | ${g.prompt.length} | ${pf.toFixed(0)} s (${(g.prompt.length / pf).toFixed(1)} tok/s) | ${ts.toFixed(2)} | ${J.ok ? "match" : J.excused ? `near tie at ${J.diff}` : `MISMATCH at ${J.diff}`}${!(J.ok || J.excused) && byRun > 0 ? `, matches llama.cpp's run ${byRun} (${JR[byRun].ok ? "all ids" : `near tie at ${JR[byRun].diff}`})` : ""}`);
  }
  console.log(`long context (QSA, maxSeq ${eng.maxSeq}, ${pb.maxCols}-column passes):\n  case | prompt | prefill | decode tok/s | vs llama.cpp\n  ${rows.join("\n  ")}`);
}
