// Qwen3.8-Flash-Next: the f16 K / V cache against the f32 one on a long prompt, teacher-forced, with no golden in
// the loop. One model load, two engines on the same weights (kv "f32", then kv "f16", QSA on, the raw-key ring):
// each prefills the prompt but its last N tokens (batched), then steps those N prompt tokens one by one
// (forwardToken, wantLogits), so both formats predict the same N positions from the same context. Per position:
// KL(f32 || f16) of the next-token distributions, max |d logprob| over f32's top 20, top-1 agreement, and each
// format's top-2 gap. The golden rule (tests/q38flash_rule.js) cannot tell the formats apart at 32K: llama.cpp's
// own reruns spread ~1 nat there and only step 0 is compared, and the goldens ran llama.cpp with its default f16
// K / V cache (no -ctk / -ctv). This is the deterministic check of what f16 costs.
// The prompt: a case of the golden file (GOLD, default tests/golden/q38flash_llama_greedy.json; ctx32k is in the
// long-context goldens, PR #374). Full model, ~70 GB: check `free -g` first.
//   CASE=ctx32k N=32 deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights tests/q38flash_kv_compare.js
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
const gold = JSON.parse(fs.readFileSync(env("GOLD", new URL("./golden/q38flash_llama_greedy.json", import.meta.url).pathname), "utf8"));
const CASE = env("CASE", "ctx32k"), N = +env("N", 32);
if (!gold[CASE]?.prompt) throw new Error(`no case ${CASE} in the golden file (ctx32k: the long-context goldens, PR #374; or GOLD=<file>)`);
const prompt = gold[CASE].prompt;
if (prompt.length <= N + 64) throw new Error(`${CASE}: ${prompt.length} tokens, too short for N ${N}`);

const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(path.join(DIR, first));
const G = model.G, cfg = qwen4expConfig(G.meta), L = cfg.L;
let t0 = performance.now();
const up = (e, name) => { if (name !== "token_embd.weight") gpuUploadEntry(device, e, false); };
const flush = async () => { device.queue.submit([device.createCommandEncoder().finish()]); await device.queue.onSubmittedWorkDone(); };
const layers = [];
for (let i = 0; i < L; i++) {
  layers.push((await qwen4expWeights(G, model.bytesOf, { lo: i, hi: i + 1, hasEmbed: false, hasHead: false, qsa: true }, undefined, up)).layers[0]);
  await flush();
  if (i % 8 === 7 || i === L - 1) console.log(`  layer ${i + 1}/${L} on the GPU, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
const rest = await qwen4expWeights(G, model.bytesOf, { lo: L, hi: L, hasEmbed: true, hasHead: true, qsa: true }, undefined, up);
await flush();
const weights = { ...rest, layers };
const fds = model.paths.map((p) => fs.openSync(p, "r"));
const readAt = async (file, off, len) => { const b = new Uint8Array(len); fs.readSync(fds[file], b, 0, len, off); return b; };
const maxSeq = Math.ceil((prompt.length + 8) / 256) * 256;

const logSoftmax = (lg) => {
  let m = -Infinity;
  for (const v of lg) if (v > m) m = v;
  let s = 0;
  for (const v of lg) s += Math.exp(v - m);
  const z = m + Math.log(s), out = new Float64Array(lg.length);
  for (let i = 0; i < lg.length; i++) out[i] = lg[i] - z;
  return out;
};
const topk = (lp, k) => {
  const idx = [];
  for (let i = 0; i < lp.length; i++) {
    if (idx.length < k) { idx.push(i); idx.sort((a, b) => lp[b] - lp[a]); } else if (lp[i] > lp[idx[k - 1]]) { idx[k - 1] = i; idx.sort((a, b) => lp[b] - lp[a]); }
  }
  return idx;
};

// one format: prefill the prompt but its last N tokens, then step those N -> N log-prob vectors
async function runFormat(kv) {
  const ngram = new NgramTable(ngramConfig(G), { readAt, layout: "tensor" });
  const eng = await Qwen4ExpEngine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, ngram, maxSeq, kv });
  const pb = await Qwen4ExpBatch.create(eng, weights, { maxCols: 64 });
  eng.reset();
  const t = performance.now();
  await pb.prefill(prompt.slice(0, prompt.length - N), { head: "none" });
  const pf = (performance.now() - t) / 1000;
  const out = [];
  for (const id of prompt.slice(prompt.length - N)) out.push(logSoftmax((await eng.forwardToken(id, { wantLogits: true })).logits));
  console.log(`KV ${kv}: QSA ${eng.qsa ? "on" : "off"}, ring ${eng.qsaRing}, maxSeq ${maxSeq}, state ${(eng.stateBytes().state / 2 ** 20).toFixed(0)} MiB; `
    + `prefill ${prompt.length - N} tokens in ${pf.toFixed(0)} s (${((prompt.length - N) / pf).toFixed(1)} tok/s), then ${N} teacher-forced steps`);
  eng.destroy?.();
  return out;
}

console.log(`${CASE}: ${prompt.length} prompt tokens; the last ${N} positions, f32 vs f16 K / V (weights loaded in ${((performance.now() - t0) / 1000).toFixed(0)} s)`);
const A = await runFormat("f32"), B = await runFormat("f16");
let agree = 0, klSum = 0, klMax = 0, dMax = 0, dSum = 0;
const rows = [];
for (let p = 0; p < N; p++) {
  const a = A[p], b = B[p];
  let kl = 0;
  for (let i = 0; i < a.length; i++) if (a[i] > -60) kl += Math.exp(a[i]) * (a[i] - b[i]);
  const ta = topk(a, 20), tb = topk(b, 2);
  let d = 0;
  for (const i of ta) d = Math.max(d, Math.abs(a[i] - b[i]));
  const same = ta[0] === tb[0];
  agree += same;
  klSum += kl; klMax = Math.max(klMax, kl); dMax = Math.max(dMax, d); dSum += d;
  rows.push(`  pos ${prompt.length - N + p}: top-1 ${ta[0]}${same ? "" : ` vs f16 ${tb[0]}`}, gap f32 ${(a[ta[0]] - a[ta[1]]).toFixed(3)} / f16 ${(b[tb[0]] - b[tb[1]]).toFixed(3)}, `
    + `KL ${kl.toExponential(2)}, max |d logprob| (f32 top 20) ${d.toFixed(4)}`);
}
for (const r of rows) console.log(r);
console.log(`summary ${CASE}, last ${N} positions: top-1 agreement ${agree}/${N}; KL(f32 || f16) mean ${(klSum / N).toExponential(2)}, max ${klMax.toExponential(2)} nats; `
  + `max |d logprob| over f32's top 20: mean ${(dSum / N).toFixed(4)}, max ${dMax.toFixed(4)} nats`);
if (errors.count) console.log(`${errors.count} GPU errors`);
Deno.exit(errors.count ? 1 : 0);
