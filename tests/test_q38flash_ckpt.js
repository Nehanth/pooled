// Qwen3.8-Flash-Next (engine/qwen4exp.js): a GPU checkpoint round trip on the first LAYERS layers (default 4: the
// DeltaNet layers 0-2, the PLE layer 1 and the attention layer 3, ~9 GB). The engine's token-dependent state is the
// KV rows [0, pos) (position-major), each DeltaNet layer's S and conv window, the PLE ring and the n-gram
// predecessors (CPU); a checkpoint copies the GPU part on the GPU, as Qwen35Engine.saveSlot does, which needs
// COPY_SRC on every one of those buffers. The check: run a prefix, copy the state out, run a tail (hidden states A),
// run other tokens over it, copy the state back, run the tail again (B): A and B bit for bit. Model and
// environment as tests/test_q38flash.js. The parts are the engine's own list (Qwen4ExpEngine._stateParts: K / V rows
// in the engine's KV format, and with QSA the raw-key ring and the complete blocks' pooled keys).
//   KV=f16       the f16 K / V cache (default f32)
//   QSA=1        load the indexer weights: the prefix is the first 2302 tokens of the ctx8k golden prompt (batched
//                prefill), past QSA's 2051 cells and past the 1024-row raw-key ring's first wrap, so the restored
//                state includes a ring whose rows the other tokens overwrote (the open block 575: rows 2300, 2301)
// Also: the engine's allocated state (stateBytes) == qwen4expMemory's sizes for its layers and formats.
//   LAYERS=4 deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights tests/test_q38flash_ckpt.js
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

const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(path.join(DIR, first));
const G = model.G, cfg = qwen4expConfig(G.meta), nL = +env("LAYERS", 4), KV = env("KV", "f32"), QSA = env("QSA") === "1";
if (!cfg.qsa.slice(0, nL).some(Boolean)) throw new Error(`LAYERS=${nL} holds no attention layer (no KV cache to copy)`);

const t0 = performance.now();
const up = (e, name) => { if (name !== "token_embd.weight") gpuUploadEntry(device, e, false); };
const weights = await qwen4expWeights(G, model.bytesOf, { lo: 0, hi: nL, hasEmbed: true, hasHead: false, qsa: QSA }, undefined, up);
const fds = model.paths.map((p) => fs.openSync(p, "r"));
const readAt = async (file, off, len) => { const b = new Uint8Array(len); fs.readSync(fds[file], b, 0, len, off); return b; };
const ngram = new NgramTable(ngramConfig(G), { readAt, layout: "tensor" });
const eng = await Qwen4ExpEngine.create({ device, meta: G.meta, weights, layerRange: [0, nL], hasEmbed: true, hasHead: false, ngram, kv: KV, ...(QSA ? { maxSeq: 4608 } : {}) });
const pb = QSA ? await Qwen4ExpBatch.create(eng, weights, { maxCols: 64 }) : null;
console.log(`qwen4exp: layers [0, ${nL}) loaded in ${((performance.now() - t0) / 1000).toFixed(0)} s; KV ${eng.kv}, QSA ${eng.qsa ? `on (raw-key ring ${eng.qsaRing} rows)` : "off"}`);

const U = GPUBufferUsage, kvRow = cfg.nKVHeads * cfg.headDim * (KV === "f16" ? 2 : 4);
let fail = 0;
// the engine's allocated state == the accounting function's
{
  const got = eng.stateBytes(), want = eng.memory();
  const same = Object.keys(got.bytes).every((k) => got.bytes[k] === want.bytes[k]) && got.state === want.state;
  console.log(`${same ? "ok  " : "FAIL"} state bytes at maxSeq ${eng.maxSeq}: allocated ${JSON.stringify(got.bytes)} ${same ? "==" : "!="} qwen4expMemory ${JSON.stringify(want.bytes)}`);
  if (!same) fail++;
}
// the GPU state of every layer: [buffer, bytes that matter at pos]
const parts = (pos) => eng._stateParts(pos).map(({ buf, bytes }) => [buf, bytes]);
const save = () => {
  const enc = device.createCommandEncoder();
  const bufs = parts(eng.pos).map(([b, n]) => {
    const c = device.createBuffer({ size: Math.max(4, n), usage: U.COPY_SRC | U.COPY_DST });
    if (n) enc.copyBufferToBuffer(b, 0, c, 0, n);
    return c;
  });
  device.queue.submit([enc.finish()]);
  return { pos: eng.pos, ng: eng.ngs?.export(), bufs };
};
const load = (s) => {
  const enc = device.createCommandEncoder();
  parts(s.pos).forEach(([b, n], i) => { if (n) enc.copyBufferToBuffer(s.bufs[i], 0, b, 0, n); });
  device.queue.submit([enc.finish()]);
  eng.pos = s.pos;
  eng.ngs?.import(s.ng);
};
const run = async (ids) => { const out = []; for (const t of ids) out.push((await eng.forwardToken(t)).hidden); return out; };

const gold = QSA ? JSON.parse(fs.readFileSync(new URL("./golden/q38flash_llama_greedy.json", import.meta.url), "utf8")) : null;
const prefix = QSA ? gold.ctx8k.prompt.slice(0, 2302) : [9707, 11, 1879, 374, 264], tail = [1273, 13, 576, 3974], other = [785, 6722, 315, 9625, 374, 12095];
eng.reset();
if (pb) await pb.prefill(prefix, { head: "none" }); else await run(prefix);
const s = save();
// the copied K rows hold the prefix: non-zero and finite
const kSlot = eng.layers.findIndex((l) => l.st.kc), kBuf = s.bufs[parts(s.pos).findIndex(([b]) => b === eng.layers[kSlot].st.kc)];
const kStage = device.createBuffer({ size: kBuf.size, usage: U.MAP_READ | U.COPY_DST }), kEnc = device.createCommandEncoder();
kEnc.copyBufferToBuffer(kBuf, 0, kStage, 0, kBuf.size);
device.queue.submit([kEnc.finish()]);
await kStage.mapAsync(GPUMapMode.READ);
const kw = new Uint32Array(kStage.getMappedRange().slice(0));
kStage.unmap();
const h2f = (h) => { const e = (h >> 10) & 31, m = h & 1023, v = e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1 + m / 1024) * 2 ** (e - 15); return h & 0x8000 ? -v : v; };
const k = KV === "f16" ? Float32Array.from({ length: kw.length * 2 }, (_, i) => h2f((kw[i >> 1] >>> ((i & 1) * 16)) & 0xffff)) : new Float32Array(kw.buffer);
const kOk = k.length === prefix.length * kvRow / (KV === "f16" ? 2 : 4) && k.every(Number.isFinite) && k.some((v) => v !== 0);
console.log(`K rows copied from layer ${eng.layers[kSlot].i}: ${k.length} ${KV} values, ${kOk ? "non-zero and finite" : "BAD"}`);
if (!kOk) fail++;
const A = await run(tail);
await run(other);
load(s);
const B = await run(tail);
let diff = 0;
for (let p = 0; p < tail.length; p++) for (let i = 0; i < A[p].length; i++) if (!Object.is(A[p][i], B[p][i])) diff++;
console.log(`restored at pos ${s.pos}: tail hidden states ${diff ? diff + " values differ" : "bit for bit"}`);
if (diff) fail++;
if (errors.count) { console.log(`${errors.count} GPU errors`); fail++; }
console.log(fail ? "Q38FLASH CKPT FAIL" : "Q38FLASH CKPT PASS ✓");
Deno.exit(fail ? 1 : 0);
