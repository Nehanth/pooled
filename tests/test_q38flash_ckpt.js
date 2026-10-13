// Qwen3.8-Flash-Next (engine/qwen4exp.js): a GPU checkpoint round trip on the first LAYERS layers (default 4: the
// DeltaNet layers 0-2, the PLE layer 1 and the attention layer 3, ~9 GB). The engine's token-dependent state is the
// KV rows [0, pos) (position-major), each DeltaNet layer's S and conv window, the PLE ring and the n-gram
// predecessors (CPU); a checkpoint copies the GPU part on the GPU, as Qwen35Engine.saveSlot does, which needs
// COPY_SRC on every one of those buffers. The check: run a prefix, copy the state out, run a tail (hidden states A),
// overwrite the state with other tokens, copy the state back, run the tail again (B): A and B bit for bit. Model and
// environment as tests/test_q38flash.js. The parts are the engine's own list (Qwen4ExpEngine._stateParts: K / V rows
// in the engine's KV format, and with QSA the raw-key ring and the complete blocks' pooled keys).
//   KV=f16       the f16 K / V cache (default f32)
//   QSA=1        load the indexer weights: the prefix is the first 2302 tokens of the ctx8k golden prompt (batched
//                prefill), past QSA's 2051 cells and past the 1024-row raw-key ring's first wrap. The "other" tokens
//                are then a reset and a batched prefill of a different 2302-token chunk of that prompt, which rewrites
//                every ring slot, every pooled block below 575, the K / V rows and the DeltaNet state, so the
//                restore has to bring all of them back (the open block 575's rows 2300, 2301 come from the ring).
//                Negative controls: restoring without the ring, and without the pooled keys, must each be seen
//                (otherwise the test could not see a missing part and fails). At 2302 QSA's selection is the same
//                with the other chunk's pooled keys, so the tail's hidden states alone do not see them: the check
//                also compares the whole state after the tail (every _stateParts buffer, bit for bit), where a
//                missing ring shows in block 575's pooled key and missing pooled keys in blocks 0-574.
//   ROOM=1       the same checks through the room's adapter (room/flash.js FlashRoomEngine over qwen4expCore): the
//                save is saveSlot, the restore loadSlot, and again importState(exportSlot) (the disk copy's path, which
//                reads every part back to the CPU); the tail runs through the room's step (embedRun). The negative
//                controls import the slot's export with the left-out parts taken from the overwritten state. Also
//                checks the signature is the engine's with v 2 and that a v1 one is refused, and that the room's step
//                gives the engine's forwardToken hidden states on the tail (so a room step without QSA is seen).
// Also: the engine's allocated state (stateBytes) == qwen4expMemory's sizes for its layers and formats.
//   LAYERS=4 deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights tests/test_q38flash_ckpt.js
import fs from "node:fs";
import path from "node:path";
import { Qwen4ExpEngine } from "../engine/qwen4exp.js";
import { Qwen4ExpBatch } from "../engine/qwen4exp_batch.js";
import { qwen4expWeights, qwen4expConfig, gpuUploadEntry } from "../engine/gguf.js";
import { ngramConfig, NgramTable } from "../engine/ngram.js";
import { openGGUF, gpuDevice, watchGpuErrors } from "./load_model.js";
import { FlashRoomEngine } from "../room/flash.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const DIR = env("Q38F_DIR", new URL("../models/q38flash", import.meta.url).pathname);
const first = fs.readdirSync(DIR).filter((f) => /^Qwen3\.8-Flash-Next-.*-00001-of-\d{5}\.gguf$/.test(f))[0];
if (!first) throw new Error("no Qwen3.8-Flash-Next shards in " + DIR);

const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(path.join(DIR, first));
const G = model.G, cfg = qwen4expConfig(G.meta), nL = +env("LAYERS", 4), KV = env("KV", "f32"), QSA = env("QSA") === "1", ROOM = env("ROOM") === "1";
if (!cfg.qsa.slice(0, nL).some(Boolean)) throw new Error(`LAYERS=${nL} holds no attention layer (no KV cache to copy)`);

const t0 = performance.now();
const up = (e, name) => { if (name !== "token_embd.weight") gpuUploadEntry(device, e, false); };
const weights = await qwen4expWeights(G, model.bytesOf, { lo: 0, hi: nL, hasEmbed: true, hasHead: false, qsa: QSA }, undefined, up);
const fds = model.paths.map((p) => fs.openSync(p, "r"));
const readAt = async (file, off, len) => { const b = new Uint8Array(len); fs.readSync(fds[file], b, 0, len, off); return b; };
const ngram = new NgramTable(ngramConfig(G), { readAt, layout: "tensor" });
const eng = await Qwen4ExpEngine.create({ device, meta: G.meta, weights, layerRange: [0, nL], hasEmbed: true, hasHead: false, ngram, kv: KV, ...(QSA ? { maxSeq: 4608 } : {}) });
const pb = QSA ? await Qwen4ExpBatch.create(eng, weights, { maxCols: 64 }) : null;
const room = ROOM ? FlashRoomEngine.wrap(eng) : null;
console.log(`qwen4exp: layers [0, ${nL}) loaded in ${((performance.now() - t0) / 1000).toFixed(0)} s; KV ${eng.kv}, QSA ${eng.qsa ? `on (raw-key ring ${eng.qsaRing} rows)` : "off"}${ROOM ? "; through the room's adapter" : ""}`);

const U = GPUBufferUsage, kvRow = cfg.nKVHeads * cfg.headDim * (KV === "f16" ? 2 : 4);
let fail = 0;
// the engine's allocated state == the accounting function's
{
  const got = eng.stateBytes(), want = eng.memory();
  const same = Object.keys(got.bytes).every((k) => got.bytes[k] === want.bytes[k]) && got.state === want.state;
  console.log(`${same ? "ok  " : "FAIL"} state bytes at maxSeq ${eng.maxSeq}: allocated ${JSON.stringify(got.bytes)} ${same ? "==" : "!="} qwen4expMemory ${JSON.stringify(want.bytes)}`);
  if (!same) fail++;
}
// the room adapter: checkpoints on, the signature the engine's with the adapter's v 2, a v1 state refused
if (room) {
  const sig = room.stateSignature(), es = eng.stateSignature();
  const sigOk = room.ckpt && sig.v === 2 && Object.keys(es).every((k) => k === "v" || sig[k] === es[k]) && sig.hDim === cfg.hDim && sig.qsa === !!eng.qsa && sig.kv === KV;
  console.log(`${sigOk ? "ok  " : "FAIL"} room signature ${JSON.stringify(sig)}`);
  if (!sigOk) fail++;
}
// the GPU state of every layer: [buffer, bytes that matter at pos]
const parts = (pos) => eng._stateParts(pos).map(({ buf, bytes }) => [buf, bytes]);
const roomSave = () => {
  room.saveSlot("s");
  return { pos: eng.pos, room: true, bufs: room.slots.get("s").bufs };
};
// the room's restores: loadSlot, or importState of the slot's export (the disk path). skip: the parts the import takes
// from the state as it is now (the overwritten one) instead of the slot, for the negative controls
const roomLoad = async (s, { skip = null, viaImport = false } = {}) => {
  if (!viaImport && !skip) { room.loadSlot("s"); return; }
  const st = await room.exportSlot("s");
  if (skip) {
    const now = await room.exportState(), cur = parts(eng.pos);
    // exportState is at the overwritten pos; the skipped parts (ring: whole; pooled: blocks below pos) are taken from it
    parts(s.pos).forEach(([b, n], i) => {
      if (!skip.has(b)) return;
      const j = cur.findIndex(([c]) => c === b), have = new Uint8Array(n);
      have.set(new Uint8Array(now.parts[j]).subarray(0, Math.min(n, now.parts[j].byteLength)));
      st.parts[i] = have.buffer;
    });
  }
  let refused = false;
  try { room.importState({ ...st, sig: { ...st.sig, v: 1 } }); } catch { refused = true; }
  if (!refused) { console.log("FAIL importState took a v1 signature"); fail++; }
  room.importState(st);
};
const save = () => {
  if (room) return roomSave();
  const enc = device.createCommandEncoder();
  const bufs = parts(eng.pos).map(([b, n]) => {
    const c = device.createBuffer({ size: Math.max(4, n), usage: U.COPY_SRC | U.COPY_DST });
    if (n) enc.copyBufferToBuffer(b, 0, c, 0, n);
    return c;
  });
  device.queue.submit([enc.finish()]);
  return { pos: eng.pos, ng: eng.ngs?.export(), bufs };
};
// skip: buffers to leave out of the restore (the negative controls)
const load = async (s, skip = null) => {
  if (room) return roomLoad(s, { skip });
  const enc = device.createCommandEncoder();
  parts(s.pos).forEach(([b, n], i) => { if (n && !skip?.has(b)) enc.copyBufferToBuffer(s.bufs[i], 0, b, 0, n); });
  device.queue.submit([enc.finish()]);
  eng.pos = s.pos;
  eng.ngs?.import(s.ng);
};
// the state's bytes (every part at pos), to compare two states bit for bit
const snap = async (pos) => {
  const ps = parts(pos), enc = device.createCommandEncoder();
  const st = ps.map(([b, n]) => { const c = device.createBuffer({ size: Math.max(4, n), usage: U.MAP_READ | U.COPY_DST }); if (n) enc.copyBufferToBuffer(b, 0, c, 0, n); return c; });
  device.queue.submit([enc.finish()]);
  const out = [];
  for (const c of st) { await c.mapAsync(GPUMapMode.READ); out.push(new Uint8Array(c.getMappedRange().slice(0))); c.unmap(); c.destroy(); }
  return out;
};
const stateDiff = (X, Y) => X.reduce((n, x, i) => n + (x.length !== Y[i].length || x.some((v, j) => v !== Y[i][j]) ? 1 : 0), 0);
const run = async (ids) => { const out = []; for (const t of ids) out.push(room ? await room.embedRun(t, eng.pos) : (await eng.forwardToken(t)).hidden); return out; };

const gold = QSA ? JSON.parse(fs.readFileSync(new URL("./golden/q38flash_llama_greedy.json", import.meta.url), "utf8")) : null;
const prefix = QSA ? gold.ctx8k.prompt.slice(0, 2302) : [9707, 11, 1879, 374, 264], tail = [1273, 13, 576, 3974], other = [785, 6722, 315, 9625, 374, 12095];
eng.reset();
if (pb) await pb.prefill(prefix, { head: "none" }); else await run(prefix);
const s = save();
// the copied K and V rows hold the prefix: non-zero and finite (f16 V is not normalized: |v| >= 65520 would be inf)
const h2f = (h) => { const e = (h >> 10) & 31, m = h & 1023, v = e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1 + m / 1024) * 2 ** (e - 15); return h & 0x8000 ? -v : v; };
const kSlot = eng.layers.findIndex((l) => l.st.kc);
for (const [name, gpuBuf] of [["K", eng.layers[kSlot].st.kc], ["V", eng.layers[kSlot].st.vc]]) {
  const cBuf = s.bufs[parts(s.pos).findIndex(([b]) => b === gpuBuf)];   // (the room's slot keeps the same list's copies)
  const stage = device.createBuffer({ size: cBuf.size, usage: U.MAP_READ | U.COPY_DST }), enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(cBuf, 0, stage, 0, cBuf.size);
  device.queue.submit([enc.finish()]);
  await stage.mapAsync(GPUMapMode.READ);
  const w = new Uint32Array(stage.getMappedRange().slice(0));
  stage.unmap();
  const x = KV === "f16" ? Float32Array.from({ length: w.length * 2 }, (_, i) => h2f((w[i >> 1] >>> ((i & 1) * 16)) & 0xffff)) : new Float32Array(w.buffer);
  let mx = 0;
  for (const v of x) if (Math.abs(v) > mx) mx = Math.abs(v);
  const ok = x.length === prefix.length * kvRow / (KV === "f16" ? 2 : 4) && x.every(Number.isFinite) && x.some((v) => v !== 0);
  console.log(`${name} rows copied from layer ${eng.layers[kSlot].i}: ${x.length} ${KV} values, max |${name}| ${mx.toPrecision(4)}, ${ok ? "non-zero and finite" : "BAD"}`);
  if (!ok) fail++;
}
const A = await run(tail), sA = await snap(eng.pos);
// overwrite the state: other tokens (QSA: a reset and a different 2302-token chunk, which wraps the ring twice)
const overwrite = async () => { if (pb) { eng.reset(); await pb.prefill(gold.ctx8k.prompt.slice(2400, 2400 + prefix.length), { head: "none" }); } else await run(other); };
const cmp = (X, Y) => { let d = 0; for (let p = 0; p < tail.length; p++) for (let i = 0; i < X[p].length; i++) if (!Object.is(X[p][i], Y[p][i])) d++; return d; };
// negative controls (QSA): a restore that leaves out the ring, or the pooled keys, must change the tail or the state
// (each over a freshly overwritten state)
if (pb) {
  const qs = eng.layers.filter((l) => l.st.qsa).map((l) => l.st.qsa);
  for (const [name, skip] of [["the raw-key ring", new Set(qs.map((q) => q.rawK))], ["the pooled keys", new Set(qs.map((q) => q.pooled))]]) {
    await overwrite();
    await load(s, skip);
    const d = cmp(A, await run(tail)), ds = stateDiff(sA, await snap(eng.pos));
    console.log(`${d || ds ? "ok  " : "FAIL"} control: restored without ${name}: tail ${d} hidden values and ${ds} state buffers differ ${d || ds ? "(the test sees it)" : "(the test cannot see this part)"}`);
    if (!d && !ds) fail++;
  }
}
for (const how of room ? ["loadSlot", "importState(exportSlot)"] : ["copy"]) {
  await overwrite();
  if (room) await roomLoad(s, { viaImport: how !== "loadSlot" }); else await load(s);
  const B = await run(tail);
  const diff = cmp(A, B), sdiff = stateDiff(sA, await snap(eng.pos));
  console.log(`restored at pos ${s.pos}${room ? ` by ${how}` : ""}: tail hidden states ${diff ? diff + " values differ" : "bit for bit"}, state after the tail ${sdiff ? sdiff + " buffers differ" : `bit for bit (${sA.length} buffers)`}`);
  if (diff || sdiff) fail++;
}
// ROOM: A and B both ran through the room's step, so the checks above cannot see a room step that computes something
// else than the engine (a missing QSA op or uniform in stepOps). Restore once more and run the tail through the
// engine's own forwardToken: it has to give A.
if (room) {
  await overwrite();
  await roomLoad(s, {});
  const E = []; for (const t of tail) E.push((await eng.forwardToken(t)).hidden);
  let d = 0, mx = 0, ref = 0;
  for (let p = 0; p < tail.length; p++) for (let i = 0; i < A[p].length; i++) { const e = Math.abs(A[p][i] - E[p][i]); if (!Object.is(A[p][i], E[p][i])) d++; if (!(e <= mx)) mx = e; ref = Math.max(ref, Math.abs(E[p][i])); }
  const ok = Number.isFinite(mx) && mx <= 1e-3 * ref;
  console.log(`${ok ? "ok  " : "FAIL"} room step vs engine forwardToken on the tail: ${d ? `${d} values differ, max |diff| ${mx.toPrecision(3)} (max |h| ${ref.toPrecision(3)})` : "bit for bit"}`);
  if (!ok) fail++;
}
if (errors.count) { console.log(`${errors.count} GPU errors`); fail++; }
console.log(fail ? "Q38FLASH CKPT FAIL" : "Q38FLASH CKPT PASS ✓");
Deno.exit(fail ? 1 : 0);
