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
//                               and conv window, K / V rows [0, pos), the PLE ring
import { NgramState } from "../engine/ngram.js";

// WebGPU's flag values (the spec's), for a page or test without the globals until a device exists
const U = globalThis.GPUBufferUsage || { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8 };
const MAP_READ = globalThis.GPUMapMode?.READ ?? 1;

export class FlashRoomEngine {
  constructor(core) {
    this.core = core;
    this.device = core.device;
    this.frameIds = true;   // frames carry the token ids (the n-gram layer may be on the first worker)
    this.mtp = null;        // no draft block in rooms yet: plain decoding
    this.NC = 4;            // columns per pipeline batch (the core runs them one at a time)
    this.slots = new Map();
  }
  static wrap(engine) { return new FlashRoomEngine(qwen4expCore(engine)); }

  // checkpoints (room/pipeline.js ckptEngine): on when every state buffer can be copied out. PR #346 creates its K/V
  // caches without COPY_SRC, so a core with attention layers has them off until the engine adds it
  get hostCkpt() { return this.core.stateParts(0).every((p) => copyable(p.buf)); }
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

  // ---- the host: embedding + its layers; the head on what comes back
  async embedRun(id, pos) { this.core.pos = pos; return this.core.step(id, null); }
  async embedRunBatch(ids, basePos, spec = false) {
    this._noSpec(spec);
    const W = this.dims.hDim, out = new Float32Array(ids.length * W);
    this.core.pos = basePos;
    for (let c = 0; c < ids.length; c++) out.set(await this.core.step(ids[c], null), c * W);
    return out;
  }
  async headFromHidden(h) { return this.core.head(h); }
  // solo (every layer here): layers only for the prompt, the last token through aiPipeToken's head
  async prefillToken(id) { await this.core.step(id, null); }
  async prefillTokens(ids) { for (const t of ids) await this.core.step(t, null); }

  // ---- a worker: a residual in, its layers, a residual out. opts.ids: the frame's token ids (wire.js splitWire)
  async runHidden(h, pos, { ids = null } = {}) {
    const tk = this._ids(ids, 1, "one token");
    this.core.pos = pos;
    return this.core.step(tk[0], h);
  }
  async runHiddenBatch(xs, basePos, spec = false, { ids = null } = {}) {
    this._noSpec(spec);
    const W = this.dims.hDim, n = xs.length / W, tk = this._ids(ids, n, `${n} tokens`), out = new Float32Array(xs.length);
    this.core.pos = basePos;
    for (let c = 0; c < n; c++) out.set(await this.core.step(tk[c], xs.subarray(c * W, (c + 1) * W)), c * W);
    return out;
  }

  reset() { this.core.reset(); }

  // ---- checkpoints (Qwen35Engine's contract: room/pipeline.js ckpt*, room/ckpt-store.js)
  // A state: { sig, pos, parts: [ArrayBuffer] }: the core's GPU parts, then (with a PLE layer) the n-gram
  // predecessors as one more part, Int32Array [p1, p2, pos].
  stateSignature() {
    const c = this.core;
    return { v: 1, arch: "qwen4exp", lo: c.lo, hi: c.hi, hDim: c.dims.hDim, kv: "f32", ple: c.pleLayers.slice() };
  }
  _ngsPart(st = this.core.ngs?.export()) { return st ? new Int32Array([st.p1, st.p2, st.pos]).buffer : null; }
  saveSlot(name) {
    this.dropSlot(name);
    const parts = this.core.stateParts(this.pos), enc = this.device.createCommandEncoder();
    if (!parts.every((p) => copyable(p.buf))) throw new Error("qwen4exp: a state buffer has no COPY_SRC usage, so it cannot be checkpointed (engine/qwen4exp.js K/V caches)");
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
  }
}

// a GPU buffer copies can read (a test's fake buffer says nothing: yes)
const copyable = (b) => b?.usage == null || !!(b.usage & U.COPY_SRC);

// The core over PR #346's Qwen4ExpEngine (engine/qwen4exp.js), from its fields: the residual buffer B.R, each
// layer's ops and state (layers[i].attnOps / ffnOps / st), the head's ops (headOps), _begin (the frame, the n-gram
// rows, the embedding into R) and _encode. Once the engine has step / head / stateParts of its own, this goes.
export function qwen4expCore(e) {
  const dev = e.device, cfg = e.cfg, HD = cfg.hDim;
  const kvRow = cfg.nKVHeads * cfg.headDim * 4;   // f32 K (or V) per position
  return {
    device: dev, dims: e.dims, lo: e.lo, hi: e.hi, hasEmbed: e.hasEmbed, hasHead: e.hasHead, maxSeq: e.maxSeq,
    get pos() { return e.pos; }, set pos(v) { e.pos = v; },
    get ngs() { return e.ngs || null; },
    pleLayers: (cfg.ple?.layers || []).filter((l) => l >= e.lo && l < e.hi),
    async step(id, hidden) {
      await e._begin(id);
      if (hidden) dev.queue.writeBuffer(e.B.R, 0, hidden);
      else if (!e.hasEmbed) throw new Error("qwen4exp: no embedding here: a worker's step needs the residual");
      const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
      for (const l of e.layers) { e._encode(pass, l.attnOps); e._encode(pass, l.ffnOps); }
      pass.end();
      enc.copyBufferToBuffer(e.B.R, 0, e.hStage, 0, HD * 4);
      dev.queue.submit([enc.finish()]);
      e.pos++;
      await e.hStage.mapAsync(MAP_READ);
      const out = new Float32Array(e.hStage.getMappedRange()).slice();
      e.hStage.unmap();
      return out;
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
    stateParts(pos) {
      const parts = [];
      for (const l of e.layers) {
        const st = l.st;
        if (st.kc) parts.push({ buf: st.kc, bytes: pos * kvRow }, { buf: st.vc, bytes: pos * kvRow });
        for (const b of [st.S, st.conv, st.ring]) if (b) parts.push({ buf: b, bytes: b.size });
      }
      return parts;
    },
  };
}
