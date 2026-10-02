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
const MAP_PARK = () => GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC;
const POOL_USAGE = () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
export const OFFLOAD_PARTS = ["gate", "up", "down"];

export class ExpertStore {
  // layers: absolute layer indices whose routed experts are offloaded. vramBytes: the GPU budget for the slot pools
  // plus the prefill region. slots: a fixed slot count per layer instead (tests). parkBytes: size of each parked buffer.
  constructor(device, { layers = [], vramBytes = 8 * 2 ** 30, slots = 0, parkBytes = 2 ** 30 } = {}) {
    this.device = device;
    this.layerSet = new Set(layers);
    this.vramBytes = vramBytes; this.slotsOpt = slots;
    this.parkBytes = Math.min(parkBytes, device.limits?.maxBufferSize ?? parkBytes);
    this.L = new Map();          // layer -> { parts: { gate|up|down: { kind, nExp, rows, cols, sl: [qs, sc] } } }
    this.park_ = [];             // { buf, view (until seal), used }
    this.sealed = false;
    this.parkedBytes = 0;
    this.stats = this._zeroStats();
  }
  _zeroStats() { return { cuts: 0, lookups: 0, hits: 0, misses: 0, bytes: 0, regionCuts: 0, regionBytes: 0, layerLoads: 0, layerBytes: 0 }; }
  resetStats() { this.stats = this._zeroStats(); }
  has(i) { return this.layerSet.has(i); }

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
      const size = Math.max(this.parkBytes, Math.ceil(per / 256) * 256);
      if (size > (this.device.limits?.maxBufferSize ?? Infinity)) throw new Error(`ExpertStore: a ${per}-byte expert exceeds maxBufferSize`);
      const buf = this.device.createBuffer({ size, usage: MAP_PARK(), mappedAtCreation: true });
      p = { buf, view: new Uint8Array(buf.getMappedRange()), used: 0 };
      this.park_.push(p);
    }
    const n = Math.min(want, Math.floor((p.buf.size - p.used) / per));
    const at = { buf: p.buf, off: p.used, view: p.view, n };
    p.used += Math.ceil(n * per / 256) * 256;
    return at;
  }
  // -> the placeholder entry the engine sees in place of the stacked expert tensor
  park(layer, part, e, nExp) {
    if (e.kind !== "q4" && e.kind !== "q8") throw new Error(`ExpertStore: layer ${layer} ${part} experts must be Q4_0 / Q8_0 after conversion (got ${e.kind})`);
    const [allRows, cols] = e.shape, rows = allRows / nExp;
    if (rows % 1 || cols % 32) throw new Error(`ExpertStore: layer ${layer} ${part}: shape ${e.shape} is not ${nExp} experts of whole blocks`);
    const qsB = rows * cols / (e.kind === "q4" ? 2 : 1), scB = rows * cols / 32 * 2;   // per expert
    if (qsB % 16 || scB % 4) throw new Error(`ExpertStore: layer ${layer} ${part}: per-expert slices ${qsB} / ${scB} B are not copyable`);
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
      st.bytes += this._copyExpert(enc, S, e, S.pool, v);
      st.misses++;
    }
    const remap = new Uint32Array(nCols * KS);
    for (let c = 0; c < nCols; c++) for (let k = 0; k < K; k++) remap[c * KS + k] = S.slotOf[sel[c * KS + k]];
    return { pool: true, remap };
  }
  // these experts of a layer into the region at their own ids (the rest of it is stale from here on)
  _toRegion(enc, layer, ids) {
    const S = this.L.get(layer);
    for (const e of ids) this.stats.regionBytes += this._copyExpert(enc, S, e, this.region.bufs, e);
    this.region.layer = -1;
    this.stats.regionCuts++;
  }
  // a short prefill frame (engine _regionLoad): the distinct experts of its nCols x KS selection into the region
  loadExperts(enc, layer, sel, nCols, KS) {
    const ids = [], seen = new Uint8Array(this.nExp);
    for (let c = 0; c < nCols; c++) for (let k = 0; k < KS - 1; k++) { const e = sel[c * KS + k]; if (e < this.nExp && !seen[e]) { seen[e] = 1; ids.push(e); } }
    this._toRegion(enc, layer, ids);
    return ids.length;
  }
  // drop every cached expert (tests: a cold cache)
  clear() { for (const l of this.layers || []) { const S = this.L.get(l); S.slotOf.fill(-1); S.expOf.fill(-1); S.used.fill(0); } }
  summary() {
    const s = this.stats, h = s.lookups ? (100 * s.hits / s.lookups).toFixed(1) : "-";
    return `expert offload: ${this.layers.length} layers, ${this.P} slots each (${(this.poolBytes / 2 ** 30).toFixed(2)} GiB) + region ${(this.regionBytes / 2 ** 30).toFixed(2)} GiB, parked ${(this.parkedBytes / 2 ** 30).toFixed(2)} GiB; `
      + `${s.cuts} cuts, hits ${h}% (${s.hits}/${s.lookups}), ${(s.bytes / 2 ** 20).toFixed(0)} MiB copied to the pools, ${s.regionCuts} region cuts (${(s.regionBytes / 2 ** 20).toFixed(0)} MiB), ${s.layerLoads} whole-layer loads (${(s.layerBytes / 2 ** 30).toFixed(1)} GiB)`;
  }
}
