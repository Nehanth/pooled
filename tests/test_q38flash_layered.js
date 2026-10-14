// Qwen3.8-Flash-Next's layered prompt passes (engine/qwen4exp_batch.js passesLayered: disk-tier layers, each read from
// its file once for up to layeredCols tokens) against pass after pass through every layer (_pass, as room/flash.js
// _passes ran them), on the first NTOK (3000) tokens of the ctx8k golden's prompt:
//   deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights tests/test_q38flash_layered.js
// with OFFLOAD=lo-hi OFFLOAD_DISK=lo-hi (tests/load_model.js offloadFromEnv; default 0-3 and 1-3). For the wide batch
// (WIDE_COLS, default 256 columns: a room's frame) and the exact one (64): the output residuals of every column, bit
// for bit, pass after pass vs layered (one group) vs layered in groups of 2 x maxCols (the group edge); the disk
// tier's whole-layer reads and bytes and the time of each.
import fs from "node:fs";
import path from "node:path";
import { Qwen4ExpEngine } from "../engine/qwen4exp.js";
import { Qwen4ExpBatch } from "../engine/qwen4exp_batch.js";
import { qwen4expWeights, qwen4expConfig, gpuUploadEntry } from "../engine/gguf.js";
import { ngramConfig, NgramTable } from "../engine/ngram.js";
import { openGGUF, gpuDevice, watchGpuErrors, offloadFromEnv } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
if (!env("OFFLOAD")) Deno.env.set("OFFLOAD", "0-3");
if (!env("OFFLOAD_DISK")) Deno.env.set("OFFLOAD_DISK", "1-3");
const DIR = env("Q38F_DIR", new URL("../models/q38flash", import.meta.url).pathname);
const first = fs.readdirSync(DIR).filter((f) => /^Qwen3\.8-Flash-Next.*-00001-of-\d+\.gguf$/.test(f)).sort()[0];
const gold = JSON.parse(fs.readFileSync(new URL("./golden/q38flash_llama_greedy.json", import.meta.url), "utf8"));
const NTOK = +env("NTOK", 3000), nL = +env("LAYERS", 4);
const prompt = (gold.ctx8k?.prompt || []).slice(0, NTOK);
if (prompt.length < NTOK) { console.log(`FAIL the ctx8k golden has ${prompt.length} prompt tokens (< ${NTOK})`); Deno.exit(1); }

const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(path.join(DIR, first));
const G = model.G;
qwen4expConfig(G.meta);
const experts = await offloadFromEnv(device, nL, model);
const up = (e, name) => { if (name !== "token_embd.weight") gpuUploadEntry(device, e, false); };
const flush = async () => { device.queue.submit([device.createCommandEncoder().finish()]); await device.queue.onSubmittedWorkDone(); };
const layers = [];
for (let i = 0; i < nL; i++) { layers.push((await qwen4expWeights(G, model.bytesOf, { lo: i, hi: i + 1, hasEmbed: false, hasHead: false, qsa: true, experts }, undefined, up)).layers[0]); await flush(); }
const rest = await qwen4expWeights(G, model.bytesOf, { lo: nL, hi: nL, hasEmbed: true, hasHead: false, qsa: true }, undefined, up);
await flush();
const weights = { ...rest, layers };
const fds = model.paths.map((p) => fs.openSync(p, "r"));
const readAt = async (file, off, len) => { const b = new Uint8Array(len); fs.readSync(fds[file], b, 0, len, off); return b; };
const ngram = new NgramTable(ngramConfig(G), { readAt, layout: "tensor" });
const eng = await Qwen4ExpEngine.create({ device, meta: G.meta, weights, layerRange: [0, nL], hasEmbed: true, hasHead: false, ngram,
  maxSeq: Math.ceil((NTOK + 16) / 256) * 256, expertRegion: !!experts });
if (experts) console.log(`offload: layers ${experts.layers.join(",")}, disk ${[...experts.diskSet].join(",") || "none"}, ${experts.P} slots per layer`);

const hash = (a) => { const u = new Uint32Array(a.buffer, a.byteOffset, a.length); let h = 2166136261; for (const x of u) h = Math.imul(h ^ x, 16777619) >>> 0; return h.toString(16); };
let fails = 0;
const run = async (b, how) => {
  eng.reset();
  const s0 = { ...experts?.stats }, t = performance.now();
  let out;
  if (how === "passes") {
    out = new Float32Array(NTOK * eng.dims.hDim);
    for (let o = 0; o < NTOK; o += b.maxCols) {
      const m = Math.min(b.maxCols, NTOK - o);
      const r = await b._pass(prompt.slice(o, o + m), { head: "none", wantHidden: true });
      out.set(r.hiddens, o * eng.dims.hDim);
    }
  } else {
    const keep = b.layeredCols;
    if (how === "layered2") b.layeredCols = 2 * b.maxCols;
    try { out = (await b.passesLayered(prompt, { wantHidden: true })).hiddens; } finally { b.layeredCols = keep; }
  }
  const s = (performance.now() - t) / 1000, st = experts?.stats || {};
  return { out, s, loads: (st.layerLoads || 0) - (s0.layerLoads || 0), gb: ((st.diskBytes || 0) - (s0.diskBytes || 0)) / 2 ** 30 };
};
for (const [name, b] of [["wide", await Qwen4ExpBatch.create(eng, weights, { maxCols: +env("WIDE_COLS", 256), verifyCols: 1 })],
  ["exact", await Qwen4ExpBatch.create(eng, weights, { maxCols: 64, verifyCols: 1, exact: true })]]) {
  if (!b.layered(NTOK)) { console.log(`FAIL ${name}: layered(${NTOK}) is false (disk layers: ${experts?.diskSet?.size || 0})`); fails++; continue; }
  const ref = await run(b, "passes");
  console.log(`${name} (${b.maxCols} columns): pass after pass ${ref.s.toFixed(1)} s, ${ref.loads} whole-layer disk reads, ${ref.gb.toFixed(1)} GiB from disk · bits ${hash(ref.out)}`);
  for (const how of ["layered", "layered2"]) {
    const r = await run(b, how);
    let diff = 0;
    for (let i = 0; i < ref.out.length; i++) if (Object.is(ref.out[i], r.out[i]) === false) diff++;
    const ok = diff === 0;
    if (!ok) fails++;
    console.log(`${ok ? "ok  " : "FAIL"} ${name} ${how}${how === "layered2" ? ` (groups of ${2 * b.maxCols})` : ""}: ${r.s.toFixed(1)} s, ${r.loads} whole-layer disk reads, ${r.gb.toFixed(1)} GiB from disk · ${diff} values differ · bits ${hash(r.out)}`);
  }
}
if (errors.count) { console.log(`FAIL ${errors.count} GPU errors`); fails++; }
console.log(fails ? `LAYERED FAIL (${fails})` : "LAYERED PASS");
Deno.exit(fails ? 1 : 0);
