// Qwen3.8-Flash-Next (llama.cpp arch qwen4exp) in a room: the room's engine interface (room/pipeline.js,
// engine/generate.js) over the Flash engine (engine/qwen4exp.js, PR #346). Roadmap 33 PR plan item 8.
//
// What is different from a Qwen 3.5 room:
//   - The hidden state between devices is the 4-stream hyper-connection residual, hDim = 4 x 2560 = 10240 floats
//     (20 KB per token in f16), not the 2560 of the block input. dims.hDim says so; the pipeline sizes every frame,
//     prefill round and returned hidden by it (room/wire.js wireDim).
//   - Layer 1 adds the n-gram (PLE) injection: its rows come from the token and its two predecessors, so the device
//     holding layer 1 needs the token ids, not only the hidden. frameIds: true makes the host append them to every
//     frame (wire.js packWire; 4 bytes per token), and runHidden / runHiddenBatch take them as { ids }. The table
//     itself (28.8 GB) stays on that device's disk (room/plan.js placeNgram).
//   - A checkpoint holds, past the DeltaNet / attention state, the PLE conv history (a GPU ring of 10 positions x
//     hDim on the PLE layer) and the n-gram predecessors (NgramState p1 / p2 / pos, on the CPU). Both are per
//     sequence: a reset clears them, a slot restores them, exportSlot / importState carry them to and from disk.
//   - Phase 1 (dense attention, context <= 2048, no MTP): no speculative verify (spec frames are refused), no GPU
//     sampling (the plain decode path: headFromHidden, then the room's sampler).
//   - Prompt frames (a node host's wide frames, a browser host's 64-column ones, a worker's frames) and a solo room's
//     prompt run in batched passes (engine/qwen4exp_batch.js, the wide kernels: within a tolerance of per-token
//     decode, not bit-equal) when the loader built one for this device; else one decode step a column, as before.
//
// The core: what this adapter needs from the Flash engine, one token at a time. PR #346's Qwen4ExpEngine runs its
// layers and its head in one forwardToken and has no slots; qwen4expCore below builds this interface from its
// fields (B.R, layers[].st, headOps, _begin, _encode) until the engine exposes it itself:
//   device                      the GPUDevice (slots copy GPU buffers on it)
//   dims { dim, hDim, vocab }   lo, hi (the layers [lo, hi) it runs), hasEmbed, hasHead, maxSeq, pos (read / write)
//   ngs                         the n-gram predecessors (engine/ngram.js NgramState), or null without a PLE layer
//   pleLayers                   the PLE layers in [lo, hi)
//   step(id, hidden) -> Promise<Float32Array(hDim)>   one token at pos through the layers, from the embedding of id
//                               (hidden null: hasEmbed) or from `hidden` (the previous device's residual); id feeds
//                               the n-gram rows on a PLE layer. Advances pos (and ngs).
//   head(hidden) -> Promise<Float32Array(vocab)>      the head's HC mixer and output.weight on a final residual
//   reset()                     pos 0, recurrent state, PLE ring and ngs cleared
//   stateParts(pos) -> [{ buf, bytes }]               the GPU buffers of everything per sequence at pos: DeltaNet S
//                               and conv window, K / V rows [0, pos) in the engine's KV format, with QSA the raw-key
//                               ring and the complete blocks' pooled keys, the PLE ring (the engine's _stateParts)
//   signature() -> {}           what a saved state must match (the engine's stateSignature: arch, lo, hi, kv, qsa,
//                               qsaRing, dim); optional, a core without it is the dense f32 layout
import { NgramState } from "../engine/ngram.js";
import { QSA_KPOOL } from "../engine/wgsl/qsa.js";
import { FLASH_PREFILL_FRAME, flashBatchScratch, FLASH_BATCH_BYTES_PER_COL, FLASH_BATCH_BYTES_FIXED } from "./models.js";

// WebGPU's flag values (the spec's), for a page or test without the globals until a device exists
const U = globalThis.GPUBufferUsage || { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8 };
const MAP_READ = globalThis.GPUMapMode?.READ ?? 1;

// A device with disk-tier layers runs a prompt of up to DISK_STEP_COLS tokens one column at a time (core steps, through
// its expert cache, as before #400), not in batched passes. A batched pass of 26+ columns loads each offloaded layer's
// experts whole into the region, and on a disk layer that is a read of the whole layer from the file: on the Mac tab
// alone (all 48 layers on disk) a 27-token question read ~77 GB and waited 28 s for its first token, where core steps
// take ~4 s (2026-10-14, main, flash_tab_gold.mjs demo: bash 4.2 s on #397, 28.4 s after #400). Steps cost a decode
// token's time each (~0.15 s on the Mac, ~0.5 s on the GB10 tab), a batched group of up to 4096 tokens ~20-36 s, so
// longer prompts keep the passes.
export const DISK_STEP_COLS = 128;

export class FlashRoomEngine {
  // opts.batch: a Qwen4ExpBatch on the same engine (engine/qwen4exp_batch.js), which runs every multi-column call below
  // (prompt frames, a solo room's prompt) in passes of up to its maxCols columns. The loaders create it when this device
  // can batch: no disk-tier layers, offloaded ones only with the store's whole-layer region. Without it every column
  // is one core step, as before.
  // opts.frame: the prompt frame width a node host sends (prefillFrame), FLASH_PREFILL_FRAME by default
  constructor(core, { batch = null, frame = FLASH_PREFILL_FRAME } = {}) {
    this.core = core;
    this.device = core.device;
    this.frameIds = true;   // frames carry the token ids (the n-gram layer may be on the first worker)
    this.mtp = null;        // no draft block in rooms yet: plain decoding
    this.batch = batch || null;
    // columns per pipeline batch: a browser host's prompt frames (room/pipeline.js widths) and a worker's runHiddenBatch
    // chunks. With the batch, FLASH_BATCH_NC columns per frame (one pass each); without it 4 (one core step a column)
    this.NC = this.batch ? FLASH_BATCH_NC : 4;
    this.frame = this.batch ? Math.max(this.NC, Math.floor(frame) || FLASH_PREFILL_FRAME) : 0;
    this.slots = new Map();
    // checkpoints only when every state buffer can be copied out. PR #346 creates its K/V caches without COPY_SRC, so a
    // device holding an attention layer can't save one. Then this engine has no slot methods at all: every reader of the
    // capability (room/pipeline.js ckptEngine and workerFrame, the ai-ready `ckpt` flag on the room page and the room
    // node, the room node's disk copies) tests for saveSlot / exportSlot / importState / stateSignature, so the host
    // leaves checkpoints off for the room instead of a worker throwing on an `sv` frame
    this.ckpt = core.stateParts(0).every((p) => copyable(p.buf));
    if (!this.ckpt) for (const k of SLOT_API) this[k] = undefined;
  }
  static wrap(engine, opts = {}) { return new FlashRoomEngine(qwen4expCore(engine), opts); }

  get dims() { return this.core.dims; }
  get maxSeq() { return this.core.maxSeq; }
  get pos() { return this.core.pos; }
  set pos(v) { this.core.pos = v; }
  get lo() { return this.core.lo; }
  get hi() { return this.core.hi; }

  // the token ids of a frame: needed on a PLE layer (its n-gram rows), else unused
  _ids(ids, n, what) {
    if (ids && ids.length >= n) return ids;
    if (this.core.ngs) throw new Error(`qwen4exp: layer ${this.core.pleLayers.join(", ")} needs the token ids with the hidden state (${what}): the host must send them (engine.frameIds)`);
    return new Uint32Array(n);
  }
  _noSpec(spec) { if (spec) throw new Error("qwen4exp: no speculative verify in rooms yet (phase 1 has no MTP)"); }

  // n columns from basePos through the batch, in passes of at most its maxCols: ids (the tokens; the n-gram rows, and
  // the embedding when xs is null) and xs (the input residuals, n x hDim, on a device without the embedding).
  // want: return the output residuals (n x hDim); else nothing is read back (a solo prompt: the caches are what count)
  async _passes(ids, xs, basePos, want = true) {
    const b = this.batch, W = this.dims.hDim, n = ids.length;
    this.core.pos = basePos;
    if (n <= DISK_STEP_COLS && this.core.experts?.diskSet?.size) {
      const out = want ? new Float32Array(n * W) : null;
      for (let c = 0; c < n; c++) {
        const r = await this.core.step(ids[c], xs ? xs.subarray(c * W, (c + 1) * W) : null);
        if (want) out.set(r, c * W);
      }
      return out;
    }
    // disk-tier layers: layer by layer, each one read from disk once for up to b.layeredCols tokens (same bits)
    if (b.layered?.(n)) return (await b.passesLayered(Array.from(ids), { hidden: xs, wantHidden: want })).hiddens;
    const out = want ? new Float32Array(n * W) : null;
    for (let o = 0; o < n; o += b.maxCols) {
      const m = Math.min(b.maxCols, n - o);
      // (_pass: a pass that throws part way marks the engine failed, as a core step does)
      const r = await b._pass(Array.from(ids.slice(o, o + m)), { head: "none", wantHidden: want, ...(xs ? { hidden: xs.subarray(o * W, (o + m) * W) } : {}) });
      if (want) out.set(r.hiddens, o * W);
    }
    return out;
  }

  // ---- the host: embedding + its layers; the head on what comes back
  async embedRun(id, pos) { this.core.pos = pos; return this.core.step(id, null); }
  async embedRunBatch(ids, basePos, spec = false) {
    this._noSpec(spec);
    if (this.batch && ids.length > 1) return this._passes(ids, null, basePos);
    const W = this.dims.hDim, out = new Float32Array(ids.length * W);
    this.core.pos = basePos;
    for (let c = 0; c < ids.length; c++) out.set(await this.core.step(ids[c], null), c * W);
    return out;
  }
  async headFromHidden(h) { return this.core.head(h); }
  // solo (every layer here): layers only for the prompt, the last token through aiPipeToken's head
  async prefillToken(id) { await this.core.step(id, null); }
  async prefillTokens(ids) {
    if (this.batch && ids.length > 1) { await this._passes(ids, null, this.core.pos, false); return; }
    for (const t of ids) await this.core.step(t, null);
  }
  // a node host's wide prompt frames (room/pipeline.js aiPrefill) and a worker's (workerFrame): frame columns per frame
  // when this device batches, else 0 (the frames then come in NC-column chunks, one core step a column)
  // A device with disk-tier layers runs a frame layer by layer (each disk layer read once for it: batch.passesLayered),
  // so it asks for frames of the batch's layeredCols: a host with them sends them that wide (room/pipeline.js aiPrefill,
  // browser hosts too: diskFrames)
  get diskFrames() { return !!this.batch?.layered?.(this.batch.layeredCols); }
  prefillFrame() { return !this.batch ? 0 : this.diskFrames ? Math.max(this.frame, this.batch.layeredCols) : this.frame; }
  // src: the host's token ids (it has the embedding), or a worker's input residuals (n x hDim) with opts.ids
  async prefillHidden(src, basePos, { ids = null } = {}) {
    if (!this.batch) throw new Error("qwen4exp: prefillHidden without batched passes on this device (prefillFrame() is 0)");
    if (this.core.hasEmbed) return this._passes(src, null, basePos);
    const n = src.length / this.dims.hDim;
    return this._passes(this._ids(ids, n, `${n} tokens`), src, basePos);
  }

  // ---- a worker: a residual in, its layers, a residual out. opts.ids: the frame's token ids (wire.js splitWire)
  async runHidden(h, pos, { ids = null } = {}) {
    const tk = this._ids(ids, 1, "one token");
    this.core.pos = pos;
    return this.core.step(tk[0], h);
  }
  async runHiddenBatch(xs, basePos, spec = false, { ids = null } = {}) {
    this._noSpec(spec);
    const W = this.dims.hDim, n = xs.length / W, tk = this._ids(ids, n, `${n} tokens`);
    if (this.batch && n > 1) return this._passes(tk, xs, basePos);
    const out = new Float32Array(xs.length);
    this.core.pos = basePos;
    for (let c = 0; c < n; c++) out.set(await this.core.step(tk[c], xs.subarray(c * W, (c + 1) * W)), c * W);
    return out;
  }

  reset() { this.core.reset(); }
  // what the GPU device does not free: the n-gram table's open files or OPFS workers (room-node shard.js, room/model-loader.js).
  // The holder calls it when it drops the engine, before or after destroying the device
  close() { const p = this.core.close?.(); this.core.close = null; return p; }
  // the expert store of offloaded layers (engine/expert_store.js), or null: generate.js turns speculation off with one,
  // the room's logs read its stats
  get experts() { return this.core.experts || null; }

  // ---- checkpoints (Qwen35Engine's contract: room/pipeline.js ckpt*, room/ckpt-store.js)
  // A state: { sig, pos, parts: [ArrayBuffer] }: the core's GPU parts, then (with a PLE layer) the n-gram
  // predecessors as one more part, Int32Array [p1, p2, pos].
  // The signature is the engine's own (its KV format, QSA, the raw-key ring, the layer range) with this adapter's
  // fields after it: v 2 (the engine's parts, QSA state included) overwrites the engine's v 1, so a v1 copy (the dense
  // f32 list without QSA state, before the room used the engine's parts) never matches: it reads as missing on disk
  // (room/ckpt-store.js, packages/room-node/ckptdisk.js key on the sig) and importState refuses it.
  stateSignature() {
    const c = this.core;
    const base = c.signature ? c.signature() : { v: 1, arch: "qwen4exp", lo: c.lo, hi: c.hi, kv: "f32" };
    return { ...base, v: 2, hDim: c.dims.hDim, ple: c.pleLayers.slice() };
  }
  _ngsPart(st = this.core.ngs?.export()) { return st ? new Int32Array([st.p1, st.p2, st.pos]).buffer : null; }
  saveSlot(name) {
    this.dropSlot(name);
    const parts = this.core.stateParts(this.pos), enc = this.device.createCommandEncoder();
    const bufs = parts.map((p) => {
      const b = this.device.createBuffer({ size: Math.max(4, p.bytes), usage: U.COPY_SRC | U.COPY_DST });
      if (p.bytes) enc.copyBufferToBuffer(p.buf, 0, b, 0, p.bytes);
      return b;
    });
    this.device.queue.submit([enc.finish()]);
    this.slots.set(name, { pos: this.pos, bufs, ngs: this.core.ngs?.export() || null });
  }
  loadSlot(name) {
    const sl = this.slots.get(name);
    if (!sl) throw new Error("no saved slot " + name);
    const parts = this.core.stateParts(sl.pos), enc = this.device.createCommandEncoder();
    parts.forEach((p, i) => { if (p.bytes) enc.copyBufferToBuffer(sl.bufs[i], 0, p.buf, 0, p.bytes); });
    this.device.queue.submit([enc.finish()]);
    this.core.pos = sl.pos;
    if (this.core.ngs) sl.ngs ? this.core.ngs.import(sl.ngs) : this.core.ngs.reset();
    this.core.clearFailed?.();   // the whole sequence state is replaced: a step that failed part way left nothing behind
  }
  dropSlot(name) {
    const sl = this.slots.get(name);
    if (!sl) return;
    this.slots.delete(name);
    if (sl.reading) sl.dropped = true; else for (const b of sl.bufs) b.destroy();
  }
  dropAllSlots() { for (const k of [...this.slots.keys()]) this.dropSlot(k); }
  async exportState() { return this._read(this.core.stateParts(this.pos), this.pos, this.core.ngs?.export()); }
  async exportSlot(name) {
    const sl = this.slots.get(name);
    if (!sl) throw new Error("no saved slot " + name);
    sl.reading = (sl.reading || 0) + 1;
    try { return { ...(await this._read(this.core.stateParts(sl.pos).map((p, i) => ({ buf: sl.bufs[i], bytes: p.bytes })), sl.pos, sl.ngs)), dropped: !!sl.dropped }; }
    finally { if (!--sl.reading && sl.dropped) for (const b of sl.bufs) b.destroy(); }
  }
  async _read(parts, pos, ngs) {
    const out = [], most = Math.max(4, ...parts.map((p) => p.bytes));
    const stage = this.device.createBuffer({ size: most, usage: U.MAP_READ | U.COPY_DST });
    try {
      for (const p of parts) {
        if (!p.bytes) { out.push(new ArrayBuffer(0)); continue; }
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(p.buf, 0, stage, 0, p.bytes);
        this.device.queue.submit([enc.finish()]);
        await stage.mapAsync(MAP_READ, 0, p.bytes);
        out.push(stage.getMappedRange(0, p.bytes).slice(0));
        stage.unmap();
      }
    } finally { stage.destroy(); }
    if (this.core.ngs) out.push(this._ngsPart(ngs || new NgramState(this.core.ngs.cfg).export()));
    return { sig: this.stateSignature(), pos, parts: out };
  }
  importState(st) {
    if (JSON.stringify(st.sig) !== JSON.stringify(this.stateSignature())) throw new Error("saved state is for a different model, layer range or KV format");
    const parts = this.core.stateParts(st.pos), extra = this.core.ngs ? 1 : 0;
    if (st.parts.length !== parts.length + extra || parts.some((p, i) => p.bytes !== st.parts[i].byteLength)) throw new Error("saved state has the wrong shape");
    let ngs = null;
    if (extra) {
      const a = st.parts[parts.length];
      if (a.byteLength !== 12) throw new Error("saved state has the wrong shape");
      const [p1, p2, pos] = new Int32Array(a);
      ngs = { p1, p2, pos };
    }
    parts.forEach((p, i) => { if (p.bytes) this.device.queue.writeBuffer(p.buf, 0, st.parts[i]); });
    this.core.pos = st.pos;
    if (ngs) this.core.ngs.import(ngs);
    this.core.clearFailed?.();   // (as loadSlot)
  }
}

// layers per submit in a step (engine/qwen4exp.js forwardToken's step)
const SUBMIT_LAYERS = 12;
// With batched passes: the columns of a browser host's prompt frames and of a worker's runHiddenBatch chunks (NC). A node
// host's prompt frames are FLASH_PREFILL_FRAME wide (room/models.js; POOLED_PREFILL_FRAME overrides it there)
export const FLASH_BATCH_NC = 64;
export { FLASH_PREFILL_FRAME, flashBatchScratch, FLASH_BATCH_BYTES_PER_COL, FLASH_BATCH_BYTES_FIXED };
// Batched passes (the loaders: room-node shard.js loadFlash, room/model-loader.js): columns per pass. A solo device
// (every layer, the embedding and the head: a solo room's prompt goes through it in one prefillTokens) runs the batch's
// widest default; a room's devices run the host's prompt frames (FLASH_PREFILL_FRAME columns), so wider passes would
// only cost scratch (flashBatchScratch).
// ring (a QSA engine's raw-key ring rows, ai-load.flash): a pass completes blocks of up to cols + 3 rows of it
// (engine/qwen4exp_batch.js refuses wider passes), so the width stays under ring - 3, in whole 64-column steps
export const flashBatchCols = (solo, ring = 0) => {
  const cols = solo ? FLASH_SOLO_COLS : FLASH_PREFILL_FRAME;
  return ring > 0 ? Math.min(cols, Math.max(64, Math.floor((ring - (QSA_KPOOL - 1)) / 64) * 64)) : cols;
};
const FLASH_SOLO_COLS = 1024;
// what a Qwen4ExpBatch holds on the GPU: every buffer it created (scratch, QSA scratch, stages) -> bytes
export function qwen4expBatchBytes(pb) {
  const seen = new Set();
  const add = (v) => { if (!v || typeof v !== "object") return; if (Array.isArray(v)) { v.forEach(add); return; } if (typeof v.size === "number" && typeof v.destroy === "function") seen.add(v); };
  for (const v of Object.values(pb.B || {})) add(v);
  for (const v of Object.values(pb.qsaS || {})) add(v);
  for (const k of ["amStage", "logitStage", "hStage", "idStage"]) add(pb[k]);
  for (const v of pb._stages?.values?.() || []) add(v);
  let b = 0;
  for (const x of seen) b += x.size;
  return b;
}

// what a device without checkpoints lacks (dropSlot / dropAllSlots stay: with no slots they do nothing)
const SLOT_API = ["saveSlot", "loadSlot", "exportSlot", "exportState", "importState", "stateSignature"];
// a GPU buffer copies can read (a test's fake buffer says nothing: yes)
const copyable = (b) => b?.usage == null || !!(b.usage & U.COPY_SRC);

// The core over PR #346's Qwen4ExpEngine (engine/qwen4exp.js), from its fields: the residual buffer B.R, each
// layer's ops and state (layers[i].attnOps / ffnOps / st), the head's ops (headOps), _begin (the frame, the n-gram
// rows, the embedding into R) and _encode. Once the engine has step / head / stateParts of its own, this goes.
export function qwen4expCore(e) {
  const dev = e.device, cfg = e.cfg, HD = cfg.hDim;
  const kvRow = cfg.nKVHeads * cfg.headDim * 4;   // f32 K (or V) per position: only the fallback list below (no _stateParts)
  // Checkpoints use the engine's own state list and signature (engine/qwen4exp.js _stateParts / stateSignature, #375:
  // K / V rows in its KV format, QSA's raw-key ring and pooled keys). An engine without _stateParts (before #375) gets
  // the dense f32 list below, which only fits an f32 engine without QSA: anything else is refused rather than
  // checkpointed wrong.
  const own = typeof e._stateParts === "function";
  if (!own && ((e.kv ?? "f32") !== "f32" || e.qsa)) throw new Error(`room/flash.js: an engine without _stateParts checkpoints as f32 K / V without QSA state, not kv ${e.kv}${e.qsa ? " with QSA" : ""}`);
  // One step's ops: the layers' halves fused as forwardToken runs them (each HC combine folded into the next site),
  // laid out without the head, so the last combine stays in and B.R is the finished residual (forwardToken's own
  // lists fold the last layer's combine into the head's site when the engine has the head). Built on first use.
  // Without _halfOps (an engine before the fused sites): the per-half lists, which compute the same R.
  // Per layer, in submits of SUBMIT_LAYERS layers as forwardToken does: one submit of all 48 layers' dispatches
  // lost the device on the GB10 (an AbortError on the readback), 12 per submit is what the engine runs.
  // Expert offload (engine/expert_store.js; the layers with l.off): such a layer's list splits after its router (the op
  // marked cut), and between the two parts the engine's own cut runs (_afterRoute: submit, read the K ids back, let the
  // store copy the misses into the layer's pool, write the slot ids), as forwardToken does. Without the cut its experts
  // would run on whatever the pool held.
  let ops = null;
  const stepOps = () => {
    if (ops) return ops;
    const per = e._halfOps ? e._halfOps(e.layers.flatMap((l) => l.halves), true) : e.layers.flatMap((l) => [l.attnOps, l.ffnOps]);
    ops = [];
    for (let k = 0; k < e.layers.length; k += SUBMIT_LAYERS)
      ops.push(e.layers.slice(k, k + SUBMIT_LAYERS).map((l, j) => {
        const list = [...per[2 * (k + j)], ...per[2 * (k + j) + 1]];
        if (!l.off) return { l, list };
        const at = list.findIndex((o) => o.cut) + 1;
        if (!at) throw new Error(`qwen4exp: offloaded layer ${l.i} has no router cut in its ops`);
        return { l, list: list.slice(0, at), exp: list.slice(at) };
      }));
    return ops;
  };
  // one chunk's layers into enc (a cut layer submits what came before it) -> { enc, pass }
  const encodeChunk = async (c) => {
    let enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
    for (const { l, list, exp } of c) {
      e._encode(pass, list);
      if (!exp) continue;
      pass.end();
      enc = await e._afterRoute(enc, l);
      pass = enc.beginComputePass();
      e._encode(pass, exp);
    }
    return { enc, pass };
  };
  const run = async (id, hidden) => {
    await e._begin(id);
    if (hidden) dev.queue.writeBuffer(e.B.R, 0, hidden);
    const chunks = stepOps();
    for (const c of chunks.slice(0, -1)) {
      const { enc, pass } = await encodeChunk(c);
      pass.end();
      dev.queue.submit([enc.finish()]);
    }
    const { enc, pass } = await encodeChunk(chunks.at(-1) || []);
    pass.end();
    enc.copyBufferToBuffer(e.B.R, 0, e.hStage, 0, HD * 4);
    dev.queue.submit([enc.finish()]);
    e.pos++;
    await e.hStage.mapAsync(MAP_READ);
    const out = new Float32Array(e.hStage.getMappedRange()).slice();
    e.hStage.unmap();
    return out;
  };
  return {
    device: dev, dims: e.dims, lo: e.lo, hi: e.hi, hasEmbed: e.hasEmbed, hasHead: e.hasHead, maxSeq: e.maxSeq,
    get pos() { return e.pos; }, set pos(v) { e.pos = v; },
    get ngs() { return e.ngs || null; },
    pleLayers: (cfg.ple?.layers || []).filter((l) => l >= e.lo && l < e.hi),
    // A step that throws part way (a lost device, a failed readback) may have advanced state without pos: the engine is
    // marked failed and refuses later steps until reset(), as its forwardToken does (engine/qwen4exp.js _check).
    // The checks that change nothing (_check, a worker without the residual) throw before that.
    async step(id, hidden) {
      e._check(id);
      if (!hidden && !e.hasEmbed) throw new Error("qwen4exp: no embedding here: a worker's step needs the residual");
      try { return await run(id, hidden); } catch (err) { e._failed = err; throw err; }
    },
    async head(hidden) {
      if (!e.hasHead) throw new Error("qwen4exp: no head on this device");
      dev.queue.writeBuffer(e.B.R, 0, hidden);
      const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
      e._encode(pass, e.headOps);
      pass.end();
      enc.copyBufferToBuffer(e.B.hl, 0, e.logitStage, 0, e.dims.vocab * 4);
      dev.queue.submit([enc.finish()]);
      await e.logitStage.mapAsync(MAP_READ);
      const out = new Float32Array(e.logitStage.getMappedRange(), 0, e.dims.vocab).slice();
      e.logitStage.unmap();
      return out;
    },
    reset() { e.reset(); },
    // a checkpoint restored (FlashRoomEngine loadSlot / importState) replaces every state a failed step may have left
    // half advanced, as Qwen35Engine's do: the engine takes steps again. The expert store's region (a batched pass's)
    // may hold a layer a failed pass never finished loading: forgotten, as reset() does
    clearFailed() { e._failed = null; if (e.experts?.region) e.experts.region.layer = -1; },
    // the n-gram table's reader (open files, OPFS workers), when the loader gave it a close, and the expert store's
    // (expert offload: its disk reader's workers or threads, its RAM cache)
    close() { return Promise.all([e.ngram?.close?.(), e.experts?.close?.()]); },
    // the expert store, when layers are offloaded (stats: hits, disk reads)
    get experts() { return e.experts || null; },
    stateParts(pos) {
      if (own) return e._stateParts(pos);
      const parts = [];
      for (const l of e.layers) {
        const st = l.st;
        if (st.kc) parts.push({ buf: st.kc, bytes: pos * kvRow }, { buf: st.vc, bytes: pos * kvRow });
        for (const b of [st.S, st.conv, st.ring]) if (b) parts.push({ buf: b, bytes: b.size });
      }
      return parts;
    },
    ...(typeof e.stateSignature === "function" ? { signature: () => e.stateSignature() } : {}),
  };
}
