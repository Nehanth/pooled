// Qwen3.8-Flash-Next batched passes (engine/qwen4exp_batch.js) and QSA (engine/qwen4exp_qsa.js) on the engine
// (engine/qwen4exp.js). Roadmap 33 PR plan items 5 and 9. The model: bartowski Qwen3.8-Flash-Next-Q4_0 in
// models/q38flash (Q38F_DIR=<dir> elsewhere), loaded with the indexer weights (qsa: true).
//   deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights tests/test_q38flash_batch.js
//
// Default: the first LAYERS (4) layers, no head (DeltaNet 0, the PLE layer 1, DeltaNet 2, QSA layer 3), on the
// first NTOK (4300) tokens of the ctx8k golden's prompt, so the context passes QSA's 2051 cells. The reference is
// the decode engine, one token at a time; every check compares the output residual (4 x 2560 floats) bit for bit:
//   prefill == decode   passes of mixed sizes (1 .. 64 columns, some across the 2051 / 2052 boundary)
//   interleave          a prefill, then decode steps, then a prefill again
//   QSA == dense        below 2052 cells: every pass forced onto the gathered (QSA) attention, decode and prefill
//   verify / accept     verify passes of 6 columns with random accepts (state restored and re-run), across 2052
//   timing              decode vs prefill tok/s for these layers
// FULL=1: all 48 layers and the head (~75 GB of GPU memory): greedy ids vs llama.cpp
// (tests/golden/q38flash_llama_greedy.json, every case including ctx4k / ctx8k) with the prompt run by the
// batched prefill, the prefill's greedy id == the decode engine's on the short prompts, and a verify pass over the
// golden continuation giving the decode steps' ids.
import fs from "node:fs";
import path from "node:path";
import { Qwen4ExpEngine } from "../engine/qwen4exp.js";
import { Qwen4ExpBatch } from "../engine/qwen4exp_batch.js";
import { qwen4expWeights, qwen4expConfig, gpuUploadEntry } from "../engine/gguf.js";
import { ngramConfig, NgramTable } from "../engine/ngram.js";
import { openGGUF, gpuDevice, watchGpuErrors } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const DIR = env("Q38F_DIR", new URL("../models/q38flash", import.meta.url).pathname);
const first = fs.readdirSync(DIR).filter((f) => /^Qwen3\.8-Flash-Next-.*-00001-of-\d{5}\.gguf$/.test(f))[0];
if (!first) throw new Error("no Qwen3.8-Flash-Next shards in " + DIR);
const FULL = env("FULL") === "1";
const gold = JSON.parse(fs.readFileSync(new URL("./golden/q38flash_llama_greedy.json", import.meta.url), "utf8"));

const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(path.join(DIR, first));
const G = model.G, cfg = qwen4expConfig(G.meta);
const nL = FULL ? cfg.L : +env("LAYERS", 4), hasHead = nL === cfg.L;
const MAXSEQ = +env("MAXSEQ", FULL ? 8448 : 4608);

let t0 = performance.now();
const up = (e, name) => { if (name !== "token_embd.weight") gpuUploadEntry(device, e, false); };
const flush = async () => { device.queue.submit([device.createCommandEncoder().finish()]); await device.queue.onSubmittedWorkDone(); };
const layers = [];
for (let i = 0; i < nL; i++) {
  layers.push((await qwen4expWeights(G, model.bytesOf, { lo: i, hi: i + 1, hasEmbed: false, hasHead: false, qsa: true }, undefined, up)).layers[0]);
  await flush();
  if (i % 8 === 7 || i === nL - 1) console.log(`  layer ${i + 1}/${nL} on the GPU, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
const rest = await qwen4expWeights(G, model.bytesOf, { lo: nL, hi: nL, hasEmbed: true, hasHead, qsa: true }, undefined, up);
await flush();
const weights = { ...rest, layers };
const fds = model.paths.map((p) => fs.openSync(p, "r"));
const readAt = async (file, off, len) => { const b = new Uint8Array(len); fs.readSync(fds[file], b, 0, len, off); return b; };
const ngram = new NgramTable(ngramConfig(G), { readAt, layout: "tensor" });
const eng = await Qwen4ExpEngine.create({ device, meta: G.meta, weights, layerRange: [0, nL], hasEmbed: true, hasHead, ngram, maxSeq: MAXSEQ });
const pb = await Qwen4ExpBatch.create(eng, weights, { maxCols: +env("MAXCOLS", 64), verifyCols: 8 });
console.log(`qwen4exp: ${nL} layers${hasHead ? " + head" : ""}, QSA ${eng.qsa ? "on" : "off"}, maxSeq ${MAXSEQ}, loaded in ${((performance.now() - t0) / 1000).toFixed(0)} s`);

let fail = 0;
const check = (ok, msg) => { console.log(`${ok ? "ok  " : "FAIL"} ${msg}`); if (!ok) fail++; };
if (FULL) await full(); else await layered();
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
  let t = performance.now();
  for (const id of ids) ref.push((await eng.forwardToken(id)).hidden);
  const decTs = NTOK / ((performance.now() - t) / 1000);
  console.log(`decode reference: ${NTOK} tokens, ${decTs.toFixed(1)} tok/s (${nL} layers)`);

  // prefill == decode, mixed pass sizes
  const run = async (label, chunks, n = NTOK, from = 0) => {
    let bad = 0, worst = 0, firstBad = -1, passes = 0;
    const tt = performance.now();
    await pb.prefill(ids.slice(from, from + n), { chunks, head: "none", wantHidden: true, onPass: (r, o, sz) => {
      const c = cmpCols(r.hiddens, ref, from + o, sz); passes++;
      bad += c.bad; worst = Math.max(worst, c.worst); if (firstBad < 0 && c.firstBad >= 0) firstBad = c.firstBad;
    } });
    const ts = n / ((performance.now() - tt) / 1000);
    check(bad === 0, `${label}: ${n} tokens in ${passes} passes, ${bad} columns differ from decode${bad ? ` (first at ${firstBad}, worst rel ${worst.toExponential(2)})` : " (bit-equal)"}; ${ts.toFixed(1)} tok/s incl. readback`);
    return ts;
  };
  eng.reset();
  await run("prefill == decode, passes 1,5,64,37,64,3,64,64,2,63,17,64", [1, 5, 64, 37, 64, 3, 64, 64, 2, 63, 17, 64]);
  eng.reset();
  await run("prefill == decode, 64-column passes", [64]);
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

function gap(lg) {
  let a = -Infinity, b = -Infinity;
  for (const v of lg) { if (v > a) { b = a; a = v; } else if (v > b) b = v; }
  return a - b;
}

async function full() {
  const tok = model.tokenizer();
  const pick = env("CASES") ? env("CASES").split(",") : null;
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
    const diff = gen.findIndex((x, i) => x !== g.ids[i]);
    const tie = (g.nearTies || []).map(([i]) => i).filter((i) => i < N);
    const ok = diff < 0, tieOk = !ok && tie.some((i) => i <= diff);
    console.log(`${ok || tieOk ? "ok  " : "FAIL"} ${name}: prompt ${g.prompt.length} tok, prefill ${pf.toFixed(1)} s (${(g.prompt.length / pf).toFixed(0)} tok/s) · decode ${ts.toFixed(2)} tok/s · ${ok ? `MATCH llama.cpp (${N} ids)` : tieOk ? `differs at ${diff} after llama.cpp's near tie at ${tie.filter((i) => i <= diff).join(",")}` : `MISMATCH at ${diff}`}`);
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
}
