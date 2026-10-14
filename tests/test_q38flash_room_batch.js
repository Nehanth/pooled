// Qwen3.8-Flash-Next in a room (room/flash.js FlashRoomEngine) with batched passes (engine/qwen4exp_batch.js), on the
// first LAYERS layers (default 4: DeltaNet 0-2, the PLE layer 1, the attention layer 3, ~9 GB) split like a room: a
// "host" engine [0, 1) with the embedding and a "worker" engine [1, LAYERS) without it, which needs the frame's token
// ids for its n-gram (PLE) layer. Each stage is wrapped twice over the same engine: per token (no batch: one decode
// step a column, the room before batched prefill) and with a Qwen4ExpBatch. Checks:
//   exact  the batch on the bit-exact kernels: the host's embedRunBatch / prefillHidden and the worker's
//          runHiddenBatch / prefillHidden give the per-token residuals bit for bit (several frame widths, a frame
//          wider than the batch's maxCols, so it takes several passes), and a decode step after them too
//   wide   the default (wide kernels, #377): within a tolerance of per token (max |a - b| / max |b| per column)
//   solo   prefillTokens (a solo room's prompt) through the batch, then one step: the same next residual as per token
//          (exact) / within the tolerance (wide)
//   bytes  the batch's GPU scratch (room/flash.js qwen4expBatchBytes) at maxCols 64, 256 and 1024: the numbers behind
//          FLASH_BATCH_BYTES_PER_COL / FLASH_BATCH_BYTES_FIXED
//   LAYERS=4 deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights tests/test_q38flash_room_batch.js
import fs from "node:fs";
import path from "node:path";
import { Qwen4ExpEngine } from "../engine/qwen4exp.js";
import { Qwen4ExpBatch } from "../engine/qwen4exp_batch.js";
import { qwen4expWeights, qwen4expConfig, gpuUploadEntry } from "../engine/gguf.js";
import { ngramConfig, NgramTable } from "../engine/ngram.js";
import { FlashRoomEngine, qwen4expBatchBytes, flashBatchScratch } from "../room/flash.js";
import { openGGUF, gpuDevice, watchGpuErrors } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const DIR = env("Q38F_DIR", new URL("../models/q38flash", import.meta.url).pathname);
const first = fs.readdirSync(DIR).filter((f) => /^Qwen3\.8-Flash-Next-.*-00001-of-\d{5}\.gguf$/.test(f))[0];
if (!first) throw new Error("no Qwen3.8-Flash-Next shards in " + DIR);
const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(path.join(DIR, first));
const G = model.G, cfg = qwen4expConfig(G.meta), nL = +env("LAYERS", 4), TOL = +env("TOL", 2e-2);
const up = (e, name) => { if (name !== "token_embd.weight") gpuUploadEntry(device, e, false); };
const fds = model.paths.map((p) => fs.openSync(p, "r"));
const readAt = async (file, off, len) => { const b = new Uint8Array(len); fs.readSync(fds[file], b, 0, len, off); return b; };
const ngram = new NgramTable(ngramConfig(G), { readAt, layout: "tensor" });
const t0 = performance.now();
const stage = async (lo, hi, hasEmbed) => {
  const weights = await qwen4expWeights(G, model.bytesOf, { lo, hi, hasEmbed, hasHead: false, qsa: false }, undefined, up);
  const eng = await Qwen4ExpEngine.create({ device, meta: G.meta, weights, layerRange: [lo, hi], hasEmbed, hasHead: false, maxSeq: 2048,
    ngram: cfg.ple.layers.some((l) => l >= lo && l < hi) ? ngram : null });
  const exact = await Qwen4ExpBatch.create(eng, weights, { maxCols: 64, verifyCols: 1, exact: true });
  const wide = await Qwen4ExpBatch.create(eng, weights, { maxCols: 256, verifyCols: 1 });
  const bytes = { exact64: qwen4expBatchBytes(exact), wide256: qwen4expBatchBytes(wide) };
  const w1024 = await Qwen4ExpBatch.create(eng, weights, { maxCols: 1024, verifyCols: 1 });
  bytes.wide1024 = qwen4expBatchBytes(w1024);
  return { eng, bytes, per: FlashRoomEngine.wrap(eng), ex: FlashRoomEngine.wrap(eng, { batch: exact }), wd: FlashRoomEngine.wrap(eng, { batch: wide }) };
};
const H = await stage(0, 1, true), W = await stage(1, nL, false);
console.log(`qwen4exp room batch: host [0, 1) + worker [1, ${nL}) loaded in ${((performance.now() - t0) / 1000).toFixed(0)} s`);
const HD = cfg.hDim;

// a prompt: tokens of the golden's first prompt, repeated to N
const gold = JSON.parse(fs.readFileSync(new URL("./golden/q38flash_llama_greedy.json", import.meta.url), "utf8"));
const base = Object.values(gold).find((g) => g?.prompt?.length > 100).prompt;
const N = +env("N", 600), ids = Array.from({ length: N }, (_, i) => base[i % base.length]);
let fail = 0;
const ok = (c, what) => { console.log(`${c ? "ok  " : "FAIL"} ${what}`); if (!c) fail++; };
const same = (a, b) => a.length === b.length && a.every((x, i) => Object.is(x, b[i]));
// worst column's max |a - b| / max |b|
const relCol = (a, b) => {
  let worst = 0;
  for (let c = 0; c < a.length / HD; c++) {
    let d = 0, m = 0;
    for (let k = c * HD; k < (c + 1) * HD; k++) { d = Math.max(d, Math.abs(a[k] - b[k])); m = Math.max(m, Math.abs(b[k])); }
    worst = Math.max(worst, d / (m || 1));
  }
  return worst;
};
// the room per token: the host's columns, then the worker's (frames of `widths`, as room/pipeline.js sends them)
const room = async (h, w, frames, via) => {
  h.reset(); w.reset();
  const hs = new Float32Array(N * HD), ws = new Float32Array(N * HD);
  let o = 0, k = 0;
  while (o < N) {
    const n = Math.min(frames[k++ % frames.length], N - o), tk = new Uint32Array(ids.slice(o, o + n));
    const a = via === "frame" ? await h.prefillHidden(ids.slice(o, o + n), o) : await h.embedRunBatch(ids.slice(o, o + n), o);
    hs.set(a, o * HD);
    const b = via === "frame" ? await w.prefillHidden(a, o, { ids: tk }) : await w.runHiddenBatch(a, o, false, { ids: tk });
    ws.set(b, o * HD);
    o += n;
  }
  // one decode step after the prompt
  const t = ids[N % ids.length], a = await h.embedRun(t, N), b = await w.runHidden(a, N, { ids: new Uint32Array([t]) });
  return { hs, ws, next: b };
};
const ref = await room(H.per, W.per, [16], "batch");
for (const [frames, via] of [[[64], "batch"], [[16, 64, 8, 4], "batch"], [[256], "frame"], [[200, 37], "frame"]]) {
  const r = await room(H.ex, W.ex, frames, via);
  ok(same(r.hs, ref.hs) && same(r.ws, ref.ws) && same(r.next, ref.next), `exact batch, frames ${frames.join("/")} (${via}): host, worker and the next step == per token, bit for bit`);
}
for (const [frames, via] of [[[64], "batch"], [[256], "frame"]]) {
  const r = await room(H.wd, W.wd, frames, via);
  const e = [relCol(r.hs, ref.hs), relCol(r.ws, ref.ws), relCol(r.next, ref.next)];
  ok(e.every((x) => x < TOL), `wide batch, frames ${frames.join("/")} (${via}): worst column ${e.map((x) => x.toExponential(1)).join(" / ")} (host / worker / next step) < ${TOL}`);
}
// solo: one engine with the embedding through every test layer would be the solo room; here the host stage stands in
// (prefillTokens, then a step: the next residual)
{
  const solo = async (E) => { E.reset(); E.pos = 0; await E.prefillTokens(ids); return E.embedRun(ids[0], N); };
  const a = await solo(H.per), b = await solo(H.ex), c = await solo(H.wd);
  ok(same(a, b), "solo prefillTokens, exact batch: the next residual == per token, bit for bit");
  ok(relCol(c, a) < TOL, `solo prefillTokens, wide batch: ${relCol(c, a).toExponential(1)} < ${TOL}`);
}
// timing: the worker's frames, per token vs batched (wide), on this subset of layers
{
  const t = async (E, fn) => { E.reset(); const s = performance.now(); await fn(E); await device.queue.onSubmittedWorkDone(); return (performance.now() - s) / 1000; };
  const xs = ref.hs, tk = new Uint32Array(ids);
  const run = (w) => async (E) => { for (let o = 0; o < N; o += w) { const n = Math.min(w, N - o); await E.runHiddenBatch(xs.subarray(o * HD, (o + n) * HD), o, false, { ids: tk.subarray(o, o + n) }); } };
  const sp = await t(W.per, run(16)), sw = await t(W.wd, run(256)), se = await t(W.ex, run(64));
  console.log(`worker layers [1, ${nL}), ${N} columns: per token ${sp.toFixed(2)} s (${(N / sp).toFixed(0)} tok/s), exact batch ${se.toFixed(2)} s (${(N / se).toFixed(0)} tok/s), wide batch ${sw.toFixed(2)} s (${(N / sw).toFixed(0)} tok/s)`);
}
for (const [k, v] of Object.entries(W.bytes)) console.log(`batch scratch (worker [1, ${nL})), ${k}: ${(v / 2 ** 20).toFixed(1)} MiB`);
for (const [k, v] of Object.entries(H.bytes)) console.log(`batch scratch (host [0, 1) + embedding), ${k}: ${(v / 2 ** 20).toFixed(1)} MiB`);
console.log(`flashBatchScratch (the constant): 256 cols ${(flashBatchScratch(256) / 2 ** 20).toFixed(1)} MiB, 1024 cols ${(flashBatchScratch(1024) / 2 ** 20).toFixed(1)} MiB`);
ok(!errors.count, `no GPU errors${errors.count ? ` (${errors.count})` : ""}`);
console.log(fail ? `FLASH ROOM BATCH FAIL (${fail})` : "FLASH ROOM BATCH PASS");
Deno.exit(fail ? 1 : 0);
