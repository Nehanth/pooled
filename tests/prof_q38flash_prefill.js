// Qwen3.8-Flash-Next batched prefill (engine/qwen4exp_batch.js): tok/s, and per-kernel GPU time from timestamp
// queries (every dispatch in its own compute pass, labelled by its pipeline's label), summed by kernel.
//   TIMESTAMPS=1 Q38F_DIR=~/llm/models/q38flash deno run --unstable-webgpu --allow-read --allow-env \
//     --allow-write=$HOME/.cache/swarmllm-weights tests/prof_q38flash_prefill.js
// LAYERS (4; 48 = the whole model + head), NTOK (2048), MAXCOLS (the batch's pass width), PROF=0 (tok/s only),
// LENS=512,2048,8192 (tok/s per prompt length; QSA on past 2051 cells),
// OFFLOAD=lo-hi (as tests/test_q38flash_batch.js), MODES=f32,dp4a,moe,dense,exact (the batch configurations timed),
// PROF_AT / PROF_NTOK: the profiled tokens and the context they start at.
import fs from "node:fs";
import path from "node:path";
import { Qwen4ExpEngine } from "../engine/qwen4exp.js";
import { Qwen4ExpBatch } from "../engine/qwen4exp_batch.js";
import { qwen4expWeights, qwen4expConfig, gpuUploadEntry } from "../engine/gguf.js";
import { ngramConfig, NgramTable } from "../engine/ngram.js";
import { openGGUF, gpuDevice, watchGpuErrors, offloadFromEnv } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const DIR = env("Q38F_DIR", new URL("../models/q38flash", import.meta.url).pathname);
const first = fs.readdirSync(DIR).filter((f) => /^Qwen3\.8-Flash-Next-.*-00001-of-\d{5}\.gguf$/.test(f))[0];
const gold = JSON.parse(fs.readFileSync(new URL("./golden/q38flash_llama_greedy.json", import.meta.url), "utf8"));
const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(path.join(DIR, first));
const G = model.G, cfg = qwen4expConfig(G.meta);
const nL = +env("LAYERS", 4), hasHead = nL === cfg.L;
const LENS = env("LENS", env("NTOK", "2048")).split(",").map(Number);
const MAXSEQ = +env("MAXSEQ", Math.max(...LENS) + 64);
const experts = await offloadFromEnv(device, nL, model);
const t0 = performance.now();
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
// MODES: comma list of batch configurations, each timed on every length: exact | f32 | moe | dense | dp4a (default dp4a)
const mk = (m) => Qwen4ExpBatch.create(eng, weights, { ...(m === "exact" ? { exact: true } : { dp4a: m === "f32" ? false : m === "dp4a" ? true : m }),
  ...(env("MAXCOLS") ? { maxCols: +env("MAXCOLS") } : {}) });
const MODES = env("MODES", "f32").split(",");
console.log(`qwen4exp: ${nL} layers${hasHead ? " + head" : ""}${experts ? `, offload ${experts.layers[0]}-${experts.layers.at(-1)}` : ""}, loaded in ${((performance.now() - t0) / 1000).toFixed(0)} s`);

// a prompt of n tokens: ctx8k's prompt (7663 tokens), repeated (fixed, so the timings stay comparable as goldens are added)
const base = gold.ctx8k.prompt;
const prompt = (n) => Array.from({ length: n }, (_, i) => base[i % base.length]);
let pb = null;
for (const mode of MODES) {
  pb = await mk(mode);
  const rows = [];
  // warm-up (every pipeline's first use, the sparse ones included): passes up to 3 x maxCols tokens, untimed
  eng.reset();
  await pb.prefill(prompt(Math.min(Math.max(...LENS), 3 * pb.maxCols, MAXSEQ)), { head: hasHead ? "last" : "none" });
  for (const n of LENS) {
    const ids = prompt(n);
    eng.reset();
    const t = performance.now();
    await pb.prefill(ids, { head: hasHead ? "last" : "none" });
    await device.queue.onSubmittedWorkDone();
    const s = (performance.now() - t) / 1000;
    rows.push({ n, s, ts: n / s });
    console.log(`${mode}: prefill ${n} tokens: ${s.toFixed(2)} s, ${(n / s).toFixed(1)} tok/s (${nL} layers, ${pb.maxCols}-column passes${n > 2051 ? ", QSA past 2051" : ""})`);
  }
  console.log(`TABLE ${nL} layers ${experts ? "offload " + env("OFFLOAD") : "resident"} ${mode}: ` + rows.map((r) => `${r.n}=${r.ts.toFixed(1)}`).join(" "));
}

if (env("PROF", "1") !== "0" && device.features.has("timestamp-query")) {
  // PROF_AT: tokens prefilled first, unprofiled (the profiled passes then run at that context)
  const at = +env("PROF_AT", 0), n = Math.min(+env("PROF_NTOK", 2 * pb.maxCols), MAXSEQ - at);
  const ids = prompt(at + n);
  eng.reset();
  if (at) await pb.prefill(ids.slice(0, at), { head: "none" });
  const MAXQ = 4096, qs = device.createQuerySet({ type: "timestamp", count: MAXQ });
  const res = device.createBuffer({ size: MAXQ * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  let names = [], nq = 0;
  const encs = [];
  const orig = device.createCommandEncoder.bind(device);
  device.createCommandEncoder = (d) => {
    const enc = orig(d); const ob = enc.beginComputePass.bind(enc); encs.push(enc);
    enc.beginComputePass = () => {
      let pipe = null; const bgs = {};
      const run = (f) => {
        if (nq + 2 > MAXQ) { const p = ob(); p.setPipeline(pipe); for (const i in bgs) p.setBindGroup(+i, bgs[i]); f(p); p.end(); return; }
        const p = ob({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: nq, endOfPassWriteIndex: nq + 1 } }); nq += 2; names.push(pipe.label || "?");
        p.setPipeline(pipe); for (const i in bgs) p.setBindGroup(+i, bgs[i]); f(p); p.end();
      };
      return { setPipeline(p) { pipe = p; }, setBindGroup(i, b) { bgs[i] = b; },
        dispatchWorkgroups(x, y = 1, z = 1) { run((p) => p.dispatchWorkgroups(x, y, z)); },
        dispatchWorkgroupsIndirect(b, o) { run((p) => p.dispatchWorkgroupsIndirect(b, o)); }, end() {} };
    };
    return enc;
  };
  const t = performance.now();
  await pb.prefill(ids.slice(at), { head: hasHead ? "last" : "none" });
  await device.queue.onSubmittedWorkDone();
  const wall = performance.now() - t;
  device.createCommandEncoder = orig;
  const e = orig(); e.resolveQuerySet(qs, 0, nq, res, 0);
  const rd = device.createBuffer({ size: nq * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  e.copyBufferToBuffer(res, 0, rd, 0, nq * 8); device.queue.submit([e.finish()]);
  await rd.mapAsync(GPUMapMode.READ); const tv = new BigUint64Array(rd.getMappedRange().slice(0)); rd.unmap();
  const by = new Map(); let tot = 0;
  for (let i = 0; i < names.length; i++) { const us = Number(tv[2 * i + 1] - tv[2 * i]) / 1000; tot += us; const o = by.get(names[i]) || { us: 0, n: 0 }; o.us += us; o.n++; by.set(names[i], o); }
  console.log(`per-kernel GPU time, ${n} tokens from position ${at} (${names.length} dispatches, one pass each; wall ${wall.toFixed(0)} ms incl. the timestamps): sum ${(tot / 1000).toFixed(1)} ms = ${(n / (tot / 1e6)).toFixed(1)} tok/s`);
  for (const [k, o] of [...by].sort((a, b) => b[1].us - a[1].us)) console.log(`  ${k.padEnd(28)} ${(o.us / 1000).toFixed(2).padStart(9)} ms ${(100 * o.us / tot).toFixed(1).padStart(5)}%  ${String(o.n).padStart(6)} launches`);
}
if (errors.count) console.log(`${errors.count} GPU errors`);
Deno.exit(errors.count ? 1 : 0);
