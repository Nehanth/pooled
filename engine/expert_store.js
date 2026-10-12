// Expert offload (engine option via the loader: qwen35Weights(..., { experts: ExpertStore })).
//
// A MoE layer's 256 routed experts are most of its bytes (Qwen3.5-122B: 1.31 GiB of 1.39 per layer). With offload,
// selected layers keep their experts out of the GPU's resident weights:
//
//   park    every expert of those layers lives in large MAP_WRITE | COPY_SRC buffers ("parked"). On a discrete GPU
//           (Dawn D3D12, the RTX 5070 PC) these sit in the upload heap, i.e. system RAM, and the copy engine pulls
//           an expert into VRAM at PCIe speed (51 GB/s measured) with no CPU work per byte. They are filled once at
//           load, straight from the converted tensor into the mapped range, in the engine's layout (the repacked
//           quant and f16 scale arrays the kernels read), then unmapped for good.
//   pool    per offloaded layer, a VRAM slot pool of P experts (six buffers: gate qs / scales, up qs / scales, down
//           qs / scales, slot s at s * slice bytes), filled on demand and evicted least recently used.
//   region  one transient whole-layer region (the six arrays of the largest offloaded layer, all 256 experts) shared
//           by every offloaded layer: prefill frames copy a layer's whole expert set into it and run the existing
//           expert-grouped kernels on it, and a batched pass whose experts do not fit the pool copies them there.
//
// Decode splits each offloaded layer at moe_route (engine/qwen35.js _cutAt / _cutResolve): the 8 selections are
// read back, plan() maps them to pool slots (copying the misses from the parked buffers) and the expert kernels
// (moe_gus / moe_dnc, unchanged) run on the pool with the slot ids in place of the expert ids: the CPU applies the
// expert -> slot table and writes the remapped selection buffer, so the kernels read expert slot[sel[k]] with no
// extra binding (moe_gus already uses 8 storage bindings, the WebGPU default limit). The same kernels and the same
// weight bytes at another base address: an offloaded engine gives the resident engine's bits.
//
// Per-layer LRU partitions are within 0.6 pt of a global LRU on the traces (feasibility study, q122/traces), so
// each layer gets the same slot count: P = (vramBytes - region) / (sum over offloaded layers of one expert's bytes).
//
// Disk layers (option disk: { layers, reader }; roadmap/32-ssd-expert-offload.md): offloaded layers whose experts are
// never parked. The loader attaches each part's tensor entry (where its experts are in the GGUF) instead, and an expert
// is read from the file and converted when a cut or a frame needs it (engine/expert_disk.js reader.readPart). The
// pool, the region and the kernels are the same; only the source of a copy changes: queue.writeBuffer of the read
// bytes instead of copyBufferToBuffer from a parked buffer. A read is async, so the engine awaits the store before it
// plans (prepare(), before plan() at a cut) and fills the region with loadLayerAsync / loadExpertsAsync, each only
// after it has submitted every encoder that uses the region: a writeBuffer lands at once in queue order, so issued in
// the middle of an unsubmitted encoder it would land before that encoder's earlier kernels ran. Same bytes in the same
// slots: a disk layer gives the parked layer's bits.
//
// T1 (disk.ramBytes > 0; engine/expert_t1.js): a RAM cache of disk layers' experts in per-expert MAP_WRITE slot
// buffers. prepare() reads only the pool misses T1 does not hold; plan() copies a T1 hit from its slot
// (copyBufferToBuffer, as from a parked buffer), and writes a miss T1 admits into a free mapped slot, unmaps it and
// copies from there. A miss T1 does not admit (its first miss: admit-on-second-miss), or one with no free slot, goes
// by writeBuffer as above. Only decode cuts that fit the pool fill T1: a batch through the region and prefill frames
// read past it (they use what it holds, but a prompt's burst of experts does not evict the decode set).
import { expertGeometry } from "./expert_disk.js";
import { ExpertT1 } from "./expert_t1.js";

const MAP_PARK = () => GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC;
const POOL_USAGE = () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
export const OFFLOAD_PARTS = ["gate", "up", "down"];

export class ExpertStore {
  // layers: absolute layer indices whose routed experts are offloaded. vramBytes: the GPU budget for the slot pools
  // plus the prefill region. slots: a fixed slot count per layer instead (tests). parkBytes: size of each parked buffer.
  // ramBytes: the most system RAM the parked buffers may take (a room node's ramGB: a promise, as a pledge is). Parking
  // past it throws, so the load fails instead of the device coming online over what it lends. expectBytes: what the
  // whole store will park (room/plan.js offloadPlan ramBytes), so the last buffer is sized to what is left instead of
  // a whole parkBytes.
  // disk: { layers, reader, ramBytes, reserve } (reader: engine/expert_disk.js expertDisk(), or the room node's
  // packages/room-node/expert_disk.js): of the offloaded layers, those whose experts stay in the file; ramBytes: RAM for
  // T1, the cache of their experts (0: none); reserve: free T1 slots kept mapped (default 2 K, at least 16)
  constructor(device, { layers = [], vramBytes = 8 * 2 ** 30, slots = 0, parkBytes = 2 ** 30, ramBytes = Infinity, expectBytes = 0, disk = null } = {}) {
    this.device = device;
    this.layerSet = new Set(layers);
    this.diskSet = new Set((disk?.layers || []).filter((l) => this.layerSet.has(l)));
    this.reader = disk?.reader || null;
    this.t1Bytes = this.diskSet.size && disk?.ramBytes > 0 ? disk.ramBytes : 0;
    this.t1Reserve = disk?.reserve;
    this.t1 = null;
    if (this.diskSet.size && !this.reader) throw new Error("ExpertStore: disk layers need a reader (engine/expert_disk.js)");
    const notOff = (disk?.layers || []).filter((l) => !this.layerSet.has(l));
    if (notOff.length) console.warn(`ExpertStore: disk layers ${notOff.join(",")} are not offloaded layers: their experts stay resident on the GPU`);
    if (disk && !this.diskSet.size && (disk.layers || []).length) console.warn("ExpertStore: none of the disk layers is offloaded: no disk tier");
    this._staged = null;
    this.vramBytes = vramBytes; this.slotsOpt = slots;
    this.parkBytes = Math.min(parkBytes, device.limits?.maxBufferSize ?? parkBytes);
    this.L = new Map();          // layer -> { parts: { gate|up|down: { kind, nExp, rows, cols, sl: [qs, sc] } } }
    this.park_ = [];             // { buf, view (until seal), used }
    this.sealed = false;
    this.parkedBytes = 0;        // the experts' bytes (what room/plan.js plans: offloadPlan ramBytes)
    this.bufBytes = 0;           // the parked buffers' sizes (the RAM they take: parkedBytes plus alignment and tails)
    this.placed = 0;             // bytes placed so far (for expectBytes)
    this.ramBytes = ramBytes > 0 ? ramBytes : Infinity; this.expectBytes = expectBytes > 0 ? expectBytes : 0;
    this.stats = this._zeroStats();
  }
  _zeroStats() { return { cuts: 0, lookups: 0, hits: 0, misses: 0, bytes: 0, regionCuts: 0, regionBytes: 0, layerLoads: 0, layerBytes: 0, diskReads: 0, diskBytes: 0, diskWaitMs: 0, t1Copies: 0, t1Writes: 0 }; }
  resetStats() { this.stats = this._zeroStats(); if (this.t1) this.t1.stats = this.t1._zero(); }
  has(i) { return this.layerSet.has(i); }
  isDisk(i) { return this.diskSet.has(i); }

  // ---- load: park one converted tensor ({ kind: "q4" | "q8", qs, scales, shape: [nExp * rows, cols] }) ----
  // Room in the parked buffers for up to `want` experts of `per` bytes each: as many as the last buffer still holds
  // (at least one: else a new buffer), so a tensor's slice may span buffers at an expert boundary. Before this a slice
  // that did not fit whole started a new buffer: the 122B's 402 MB slices left 220 MB of every 1 GiB buffer unused
  // (50 GiB of buffers for 41.8 GiB of experts on the RTX 5070 PC, which then ran out of RAM).
  // -> { buf, off, view, n }
  _alloc(per, want) {
    if (this.sealed) throw new Error("ExpertStore: parked buffers are sealed");
    let p = this.park_[this.park_.length - 1];
    if (!p || p.used + per > p.buf.size) {
      const one = Math.ceil(per / 256) * 256, left = this.expectBytes - this.placed;
      let size = Math.max(one, this.expectBytes > 0 ? Math.min(this.parkBytes, Math.ceil(Math.max(0, left) / 256) * 256 + 4096) : this.parkBytes);
      if (one > (this.device.limits?.maxBufferSize ?? Infinity)) throw new Error(`ExpertStore: a ${per}-byte expert exceeds maxBufferSize`);
      size = Math.min(size, this.device.limits?.maxBufferSize ?? Infinity);
      // never past the RAM this device lends: a smaller last buffer if that still holds an expert, else fail the load
      const room = Math.floor((this.ramBytes - this.bufBytes) / 256) * 256;
      if (room < one) throw new Error(`ExpertStore: parking more experts would take over the ${(this.ramBytes / 2 ** 30).toFixed(2)} GiB of RAM this device lends (${(this.bufBytes / 2 ** 30).toFixed(2)} GiB parked so far)`);
      size = Math.min(size, room);
      this.bufBytes += size;
      const buf = this.device.createBuffer({ size, usage: MAP_PARK(), mappedAtCreation: true });
      p = { buf, view: new Uint8Array(buf.getMappedRange()), used: 0 };
      this.park_.push(p);
    }
    const n = Math.min(want, Math.floor((p.buf.size - p.used) / per));
    const at = { buf: p.buf, off: p.used, view: p.view, n };
    p.used += Math.ceil(n * per / 256) * 256;
    this.placed += n * per;
    return at;
  }
  // -> the placeholder entry the engine sees in place of the stacked expert tensor
  park(layer, part, e, nExp) {
    // a disk layer's experts stay in the file (attach): a loader without that branch must not park them
    if (this.isDisk(layer)) throw new Error(`ExpertStore: layer ${layer} is a disk layer: its experts are attached (attach()), not parked`);
    if (e.kind !== "q4" && e.kind !== "q8") throw new Error(`ExpertStore: layer ${layer} ${part} experts must be Q4_0 / Q8_0 after conversion (got ${e.kind})`);
    const [allRows, cols] = e.shape, rows = allRows / nExp;
    if (rows % 1 || cols % 32) throw new Error(`ExpertStore: layer ${layer} ${part}: shape ${e.shape} is not ${nExp} experts of whole blocks`);
    const qsB = rows * cols / (e.kind === "q4" ? 2 : 1), scB = rows * cols / 32 * 2;   // per expert
    if (qsB % 16 || scB % 4) throw new Error(`ExpertStore: layer ${layer} ${part}: per-expert slices ${qsB} / ${scB} B are not copyable`);
    if (this.parkedBytes + (qsB + scB) * nExp > this.ramBytes)
      throw new Error(`ExpertStore: layer ${layer} ${part}: parking its experts would take ${((this.parkedBytes + (qsB + scB) * nExp) / 2 ** 30).toFixed(2)} GiB of RAM, over the ${(this.ramBytes / 2 ** 30).toFixed(2)} GiB this device lends`);
    // each slice: segs [{ buf, off, e0, n }]: experts e0 .. e0 + n - 1 at off + (e - e0) * per in buf
    const sl = [[e.qs, qsB], [e.scales, scB]].map(([src, per]) => {
      const bytes = per * nExp;
      const s = src instanceof Uint8Array ? src : new Uint8Array(src.buffer, src.byteOffset, src.byteLength);
      if (s.byteLength < bytes) throw new Error(`ExpertStore: layer ${layer} ${part}: ${s.byteLength} B for ${bytes}`);
      const segs = [];
      for (let e0 = 0; e0 < nExp;) {
        const at = this._alloc(per, nExp - e0);
        at.view.set(s.subarray(e0 * per, (e0 + at.n) * per), at.off);
        segs.push({ buf: at.buf, off: at.off, e0, n: at.n });
        e0 += at.n;
      }
      return { segs, per };
    });
    this.parkedBytes += (qsB + scB) * nExp;
    e.qs = e.scales = null;   // the CPU copy is no longer needed
    let S = this.L.get(layer);
    if (!S) this.L.set(layer, S = { parts: {} });
    S.parts[part] = { kind: e.kind, nExp, rows, cols, sl };
    return { kind: e.kind, shape: e.shape, offload: true, store: this, layer, part };
  }
  // ---- load: a disk layer's part. info: the stacked tensor's header entry ([nExp][rows][cols], ggmlType, byteOffset,
  // shard); nothing is read. -> the same placeholder park() returns
  attach(layer, part, info, nExp) {
    if (!this.isDisk(layer)) throw new Error(`ExpertStore: layer ${layer} is not a disk layer`);
    const g = expertGeometry(info);
    if (g.nExp !== nExp) throw new Error(`ExpertStore: layer ${layer} ${part}: ${g.nExp} experts in the file, ${nExp} expected`);
    const { rows, cols, kind } = g;
    if (cols % 32) throw new Error(`ExpertStore: layer ${layer} ${part}: shape ${info.shape} is not ${nExp} experts of whole blocks`);
    const qsB = rows * cols / (kind === "q4" ? 2 : 1), scB = rows * cols / 32 * 2;
    if (qsB % 16 || scB % 4) throw new Error(`ExpertStore: layer ${layer} ${part}: per-expert slices ${qsB} / ${scB} B are not copyable`);
    let S = this.L.get(layer);
    if (!S) this.L.set(layer, S = { parts: {} });
    S.parts[part] = { kind, nExp, rows, cols, info, disk: true, sl: [{ per: qsB }, { per: scB }] };
    return { kind, shape: [nExp * rows, cols], offload: true, store: this, layer, part };
  }
  // unmap every parked buffer (the copy engine reads them from here on)
  seal() {
    if (this.sealed) return;
    for (const p of this.park_) { p.buf.unmap(); p.view = null; }
    this.sealed = true;
  }

  // ---- engine side ----
  // the six [part, slice] pairs of a layer, in pool / region buffer order
  static SLICES = [["gate", 0], ["gate", 1], ["up", 0], ["up", 1], ["down", 0], ["down", 1]];
  _per(S) { return ExpertStore.SLICES.map(([p, s]) => S.parts[p].sl[s].per); }
  // Allocate the pools and the region. K: experts per token (the pool must hold one token's).
  build({ K }) {
    if (this.built) return this;
    this.seal();
    const dev = this.device, layers = [...this.L.keys()].sort((a, b) => a - b);
    for (const l of layers) for (const p of OFFLOAD_PARTS) if (!this.L.get(l).parts[p]) throw new Error(`ExpertStore: layer ${l} has no parked ${p} experts`);
    const nExp = this.L.get(layers[0]).parts.gate.nExp;
    // region: the largest layer's whole expert arrays, slice by slice
    const regionSz = ExpertStore.SLICES.map((_, j) => Math.max(...layers.map((l) => this._per(this.L.get(l))[j])) * nExp);
    const regionBytes = regionSz.reduce((a, b) => a + b, 0);
    const perExp = layers.reduce((a, l) => a + this._per(this.L.get(l)).reduce((x, y) => x + y, 0), 0);
    let P = this.slotsOpt > 0 ? this.slotsOpt : Math.floor((this.vramBytes - regionBytes) / perExp);
    P = Math.min(nExp, P);
    if (!(P >= K)) throw new Error(`ExpertStore: ${(this.vramBytes / 2 ** 30).toFixed(2)} GiB holds ${P} experts per offloaded layer after the ${(regionBytes / 2 ** 30).toFixed(2)} GiB prefill region; one token needs ${K}`);
    dev.pushErrorScope("out-of-memory");
    this.region = { bufs: regionSz.map((n) => dev.createBuffer({ size: n, usage: POOL_USAGE() })), layer: -1 };
    for (const l of layers) {
      const S = this.L.get(l), per = this._per(S);
      S.pool = per.map((n) => dev.createBuffer({ size: n * P, usage: POOL_USAGE() }));
      S.slotOf = new Int32Array(nExp).fill(-1);   // expert -> slot (the table)
      S.expOf = new Int32Array(P).fill(-1);       // slot -> expert
      S.used = new Float64Array(P);               // LRU stamps
      S.perExp = per.reduce((a, b) => a + b, 0);
    }
    this._oom = dev.popErrorScope();
    this.P = P; this.K = K; this.nExp = nExp; this.layers = layers;
    this.regionBytes = regionBytes; this.poolBytes = P * perExp;
    this.stamp = 0;
    if (this.t1Bytes > 0) {
      // one slot holds one expert of any disk layer: each slice at the largest disk layer's size, 256-aligned
      const dl = layers.filter((l) => this.isDisk(l)), off = [];
      let at = 0;
      for (let j = 0; j < 6; j++) { off.push(at); at += Math.ceil(Math.max(...dl.map((l) => this._per(this.L.get(l))[j])) / 256) * 256; }
      this.t1Off = off;
      this.t1 = new ExpertT1(dev, { bytes: this.t1Bytes, slotBytes: at, reserve: this.t1Reserve ?? Math.max(16, 2 * K) });
    }
    this.built = true;
    return this;
  }
  async checkAlloc() { const e = await this._oom; if (e) throw new Error("ExpertStore: GPU out of memory for the expert pools: " + e.message); }
  // engine entries ({ kind, qs, sc }) over the region / a layer's pool
  regionEntry(layer, part) { const j = OFFLOAD_PARTS.indexOf(part) * 2; return { kind: this.L.get(layer).parts[part].kind, qs: this.region.bufs[j], sc: this.region.bufs[j + 1] }; }
  poolEntry(layer, part) { const S = this.L.get(layer), j = OFFLOAD_PARTS.indexOf(part) * 2; return { kind: S.parts[part].kind, qs: S.pool[j], sc: S.pool[j + 1] }; }
  layerBytes(layer) { return this.L.get(layer).perExp * this.nExp; }
  _copyExpert(enc, S, e, dst, at) {   // dst: pool or region buffers; at: slot (pool) or expert id (region)
    let n = 0;
    ExpertStore.SLICES.forEach(([p, s], j) => {
      const src = S.parts[p].sl[s], g = src.segs.find((x) => e >= x.e0 && e < x.e0 + x.n);
      enc.copyBufferToBuffer(g.buf, g.off + (e - g.e0) * src.per, dst[j], at * src.per, src.per);
      n += src.per;
    });
    return n;
  }
  // The whole layer into the region (prefill frames: a frame of 512+ tokens touches nearly every expert). Skipped when
  // the region already holds it (encode order is execution order: one queue).
  loadLayer(enc, layer) {
    const R = this.region;
    if (R.layer === layer) return 0;
    if (this.isDisk(layer)) throw new Error(`expert offload: disk layer ${layer} must be loaded into the region with loadLayerAsync, after a submit`);
    const S = this.L.get(layer);
    let n = 0;
    ExpertStore.SLICES.forEach(([p, s], j) => {
      const src = S.parts[p].sl[s];
      for (const g of src.segs) { enc.copyBufferToBuffer(g.buf, g.off, R.bufs[j], g.e0 * src.per, g.n * src.per); n += g.n * src.per; }
    });
    R.layer = layer;
    this.stats.layerLoads++; this.stats.layerBytes += n;
    return n;
  }
  // Plan one cut: sel = the route's output (nCols x KS u32, slot K of each column is the shared expert). Encodes the
  // copies into enc. -> { pool: true, remap: Uint32Array(nCols * KS) } (the slot ids, for the pool bind groups), or
  // { pool: false } when the columns' distinct experts outnumber the pool: they are copied into the region at their
  // own ids instead and the region bind groups (the original selection) run.
  plan(enc, layer, sel, nCols, KS) {
    const S = this.L.get(layer), K = KS - 1, st = this.stats;
    st.cuts++;
    const uniq = [];
    for (let c = 0; c < nCols; c++) for (let k = 0; k < K; k++) {
      const e = sel[c * KS + k];
      if (e >= this.nExp) throw new Error(`ExpertStore: layer ${layer} routed to expert ${e}`);
      if (!uniq.includes(e)) uniq.push(e);
    }
    st.lookups += uniq.length;
    if (uniq.length > this.P) {   // a wide verify / prompt-tail pass: through the region, no caching
      this._toRegion(enc, layer, uniq);
      this._staged = null;
      return { pool: false };
    }
    // recency is one stamp per use (hits in selection order, then the misses), so the order is total; the
    // step's experts are stamped past t0 and never evicted by its own misses
    const t0 = this.stamp, miss = [];
    for (const e of uniq) {
      const s = S.slotOf[e];
      if (s >= 0) { S.used[s] = ++this.stamp; st.hits++; } else miss.push(e);
    }
    for (const e of miss) {   // victim: a free slot, else the least recently used one not in this step
      let v = -1, best = Infinity;
      for (let s = 0; s < this.P; s++) {
        if (S.expOf[s] < 0) { v = s; break; }
        if (S.used[s] <= t0 && S.used[s] < best) { best = S.used[s]; v = s; }
      }
      if (S.expOf[v] >= 0) S.slotOf[S.expOf[v]] = -1;
      S.expOf[v] = e; S.slotOf[e] = v; S.used[v] = ++this.stamp;
      st.bytes += this._putExpert(enc, layer, S, e, S.pool, v, true);
      st.misses++;
    }
    this._staged = null;
    const remap = new Uint32Array(nCols * KS);
    for (let c = 0; c < nCols; c++) for (let k = 0; k < K; k++) remap[c * KS + k] = S.slotOf[sel[c * KS + k]];
    return { pool: true, remap };
  }
  // these experts of a layer into the region at their own ids (the rest of it is stale from here on)
  _toRegion(enc, layer, ids) {
    const S = this.L.get(layer), done = this._staged?.layer === layer ? this._staged.region : null;
    // (a disk layer: prepare() already wrote what it read; only T1's experts are copied here)
    for (const e of ids) this.stats.regionBytes += done?.has(e) ? S.perExp : this._putExpert(enc, layer, S, e, this.region.bufs, e);
    this.region.layer = -1;
    this.stats.regionCuts++;
  }
  // a short prefill frame (engine _regionLoad): the distinct experts of its nCols x KS selection into the region
  loadExperts(enc, layer, sel, nCols, KS) {
    if (this.isDisk(layer)) throw new Error(`expert offload: disk layer ${layer} takes loadExpertsAsync, after a submit`);
    const ids = this._frameIds(sel, nCols, KS);
    this._toRegion(enc, layer, ids);
    return ids.length;
  }
  _frameIds(sel, nCols, KS) {
    const ids = [], seen = new Uint8Array(this.nExp);
    for (let c = 0; c < nCols; c++) for (let k = 0; k < KS - 1; k++) { const e = sel[c * KS + k]; if (e < this.nExp && !seen[e]) { seen[e] = 1; ids.push(e); } }
    return ids;
  }

  // ---- disk layers ----
  // one expert's six slices (SLICES order), read and converted
  async _readExpert(S, e) {
    const out = [];
    const parts = await Promise.all(OFFLOAD_PARTS.map((p) => this.reader.readPart(S.parts[p].info, e)));
    for (const r of parts) out.push(r.qs, r.sc);
    ExpertStore.SLICES.forEach(([p, s], j) => {
      // (a reader may pad the scales to whole u32 words, as convertEntry's Uint32Array does: only `want` bytes count)
      const want = S.parts[p].sl[s].per;
      if (out[j].byteLength < want || out[j].byteLength > want + 4) throw new Error(`expert offload: expert ${e} ${p} slice ${s} converted to ${out[j].byteLength} B, expected ${want}`);
      if (out[j].byteLength !== want) out[j] = out[j].subarray(0, want);
    });
    return out;
  }
  // these experts of a disk layer, read in parallel -> Map(e -> slices); stats count the wait. Callers keep ids to a
  // run (REGION_RUN): each expert is held whole in memory until written (~10 MB on the 122B)
  async _readMany(layer, ids) {
    const S = this.L.get(layer), t0 = performance.now(), m = new Map();
    const got = await Promise.all(ids.map((e) => this._readExpert(S, e)));
    ids.forEach((e, i) => m.set(e, got[i]));
    this.stats.diskWaitMs += performance.now() - t0;
    this.stats.diskReads += ids.length;
    this.stats.diskBytes += ids.length * OFFLOAD_PARTS.reduce((a, p) => a + expertGeometry(S.parts[p].info).expBytes, 0);
    return m;
  }
  // Before plan() on a disk layer (engine _cutResolve, after the selection is read back and the cut's encoder is
  // submitted): read the experts plan() will copy, the same set it computes (no LRU state changes in between).
  // Experts T1 holds are not read (plan() copies them from their slots). Everything encoded so far is submitted by
  // now (the engine's contract), so this is where T1 evicts to refill its reserve of free slots (flush).
  // flush false: something encoded but not yet submitted may still copy from T1 slots (engine _specCut after its first
  // prepare: several plan()s go into one encoder), so nothing is evicted this time.
  async prepare(layer, sel, nCols, KS, { flush = true } = {}) {
    if (!this.isDisk(layer)) return 0;
    if (flush) this.t1?.flush();
    const S = this.L.get(layer), K = KS - 1, uniq = [];
    for (let c = 0; c < nCols; c++) for (let k = 0; k < K; k++) { const e = sel[c * KS + k]; if (e < this.nExp && !uniq.includes(e)) uniq.push(e); }
    const ids = (uniq.length > this.P ? uniq : uniq.filter((e) => S.slotOf[e] < 0)).filter((e) => !this.t1 || this.t1.find(this._key(layer, e), false) < 0);
    this._staged = null;
    if (uniq.length > this.P) {
      // plan() takes the region path: up to every expert of the layer, so they are written into the region here, a run
      // at a time, instead of all held until plan() (everything that uses the region is submitted: the contract above)
      await this._writeRegion(layer, ids);
      this._staged = { layer, region: new Set(ids) };
    } else this._staged = { layer, bytes: await this._readMany(layer, ids) };
    return ids.length;
  }
  // these experts of a disk layer into the region at their own ids, read in runs of `run` (each run written, then
  // dropped: a wide pass or a short frame can name ~all 256 experts, ~2.5 GB on the 122B if read at once). Only once
  // everything that uses the region is submitted (a writeBuffer lands in queue order). -> bytes written
  async _writeRegion(layer, ids, run = ExpertStore.REGION_RUN) {
    const S = this.L.get(layer), q = this.device.queue, R = this.region;
    R.layer = -1;
    let n = 0;
    for (let i = 0; i < ids.length; i += run) {
      const part = ids.slice(i, i + run), got = await this._readMany(layer, part);
      for (const e of part) {
        const b = got.get(e);
        ExpertStore.SLICES.forEach(([p, s], j) => { const per = S.parts[p].sl[s].per; q.writeBuffer(R.bufs[j], e * per, b[j]); n += per; });
      }
    }
    return n;
  }
  static REGION_RUN = 32;
  _key(layer, e) { return layer * this.nExp + e; }
  // an expert into dst slot / region id `at`: a copy from its parked buffer, or (disk layer) from its T1 slot, or the
  // bytes prepare() read: into a free T1 slot and copied from there when T1 admits it (admit: a pool fill, enc given),
  // else a writeBuffer, which lands in queue order, after what was submitted and before enc's commands.
  _putExpert(enc, layer, S, e, dst, at, admit = false) {
    if (!this.isDisk(layer)) return this._copyExpert(enc, S, e, dst, at);
    const b = this._staged?.layer === layer ? this._staged.bytes.get(e) : null, T = this.t1, key = this._key(layer, e);
    let n = 0;
    if (!b) {   // not read: T1 holds it (prepare left it out)
      const s = T && enc ? T.find(key) : -1;
      if (s < 0) throw new Error(`expert offload: disk layer ${layer} expert ${e} was not prepared`);
      ExpertStore.SLICES.forEach(([p, sl], j) => { const per = S.parts[p].sl[sl].per; enc.copyBufferToBuffer(T.buf(s), this.t1Off[j], dst[j], at * per, per); n += per; });
      this.stats.t1Copies++;
      return n;
    }
    const slot = admit && enc && T && T.wants(key) ? T.take(key) : null;
    if (slot) {   // into the mapped slot, unmapped, then copied like a hit
      ExpertStore.SLICES.forEach((_, j) => slot.view.set(b[j], this.t1Off[j]));
      const buf = T.commit(slot.s, key);
      ExpertStore.SLICES.forEach(([p, sl], j) => { const per = S.parts[p].sl[sl].per; enc.copyBufferToBuffer(buf, this.t1Off[j], dst[j], at * per, per); n += per; });
      return n;
    }
    ExpertStore.SLICES.forEach(([p, s], j) => { const per = S.parts[p].sl[s].per; this.device.queue.writeBuffer(dst[j], at * per, b[j]); n += per; });
    this.stats.t1Writes++;
    return n;
  }
  // The whole disk layer into the region (a prefill frame). Only after the caller has submitted everything that uses
  // the region. Reads in runs of `run` experts, so a layer is never held whole in memory.
  async loadLayerAsync(layer, run = ExpertStore.REGION_RUN) {
    const R = this.region;
    if (R.layer === layer) return 0;
    if (!this.isDisk(layer)) throw new Error(`ExpertStore: layer ${layer} is not a disk layer`);
    this.t1?.flush();
    const S = this.L.get(layer), q = this.device.queue;
    R.layer = -1;
    let n = 0;
    for (let e0 = 0; e0 < this.nExp; e0 += run) {
      const ids = Array.from({ length: Math.min(run, this.nExp - e0) }, (_, i) => e0 + i);
      const got = await this._readMany(layer, ids);
      ExpertStore.SLICES.forEach(([p, s], j) => {
        const per = S.parts[p].sl[s].per, buf = new Uint8Array(per * ids.length);
        ids.forEach((e, i) => buf.set(got.get(e)[j], i * per));
        q.writeBuffer(R.bufs[j], e0 * per, buf);
        n += buf.byteLength;
      });
    }
    R.layer = layer;
    this.stats.layerLoads++; this.stats.layerBytes += n;
    return n;
  }
  // A short frame's experts of a disk layer into the region (engine _regionLoad), after the caller's submit.
  async loadExpertsAsync(layer, sel, nCols, KS) {
    if (!this.isDisk(layer)) throw new Error(`ExpertStore: layer ${layer} is not a disk layer`);
    this.t1?.flush();
    const ids = this._frameIds(sel, nCols, KS);
    this._staged = null;
    this.stats.regionBytes += await this._writeRegion(layer, ids);
    this.region.layer = -1;
    this.stats.regionCuts++;
    return ids.length;
  }
  // drop every cached expert (tests: a cold cache)
  clear() { for (const l of this.layers || []) { const S = this.L.get(l); S.slotOf.fill(-1); S.expOf.fill(-1); S.used.fill(0); } }
  summary() {
    const s = this.stats, h = s.lookups ? (100 * s.hits / s.lookups).toFixed(1) : "-";
    return `expert offload: ${this.layers.length} layers, ${this.P} slots each (${(this.poolBytes / 2 ** 30).toFixed(2)} GiB) + region ${(this.regionBytes / 2 ** 30).toFixed(2)} GiB, parked ${(this.parkedBytes / 2 ** 30).toFixed(2)} GiB in ${(this.bufBytes / 2 ** 30).toFixed(2)} GiB of buffers; `
      + `${s.cuts} cuts, hits ${h}% (${s.hits}/${s.lookups}), ${(s.bytes / 2 ** 20).toFixed(0)} MiB copied to the pools, ${s.regionCuts} region cuts (${(s.regionBytes / 2 ** 20).toFixed(0)} MiB), ${s.layerLoads} whole-layer loads (${(s.layerBytes / 2 ** 30).toFixed(1)} GiB)`
      + (this.diskSet.size ? `; ${this.diskSet.size} disk layers: ${s.diskReads} expert reads, ${(s.diskBytes / 2 ** 20).toFixed(0)} MiB from the file, ${(s.diskWaitMs / 1000).toFixed(2)} s waiting` : "")
      + (this.t1 ? `; RAM cache ${this.t1.full}/${this.t1.cap} experts (${(this.t1.bytes / 2 ** 30).toFixed(2)} of ${(this.t1Bytes / 2 ** 30).toFixed(2)} GiB): ${this.t1.stats.hits} hits, ${this.t1.stats.admits} admitted, ${this.t1.stats.skips} first misses, ${this.t1.stats.noSlot} with no free slot, ${this.t1.stats.evicts} evicted` : "");
  }
  // the RAM the disk tier's cache takes now (its slot buffers)
  get t1RamBytes() { return this.t1 ? this.t1.bytes : 0; }
  // frees the RAM cache and stops the reader (room node: when the layers are freed)
  async close() {
    this.t1?.destroy(); this.t1 = null;
    const r = this.reader; this.reader = null;
    await r?.close?.();
  }
}
