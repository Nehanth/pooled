// Batched passes for Qwen3.8-Flash-Next (engine/qwen4exp.js): prompt prefill in chunks of up to maxCols tokens and
// speculative verify passes. Roadmap 33 PR plan items 5 (batched / wide prefill, verify) and 9 (QSA in them).
//
// A pass runs nCols consecutive tokens through every layer at once, on the decode engine's own state (DeltaNet conv
// and S, the PLE ring, the K / V cache, QSA's rawK / pooled keys, the n-gram history and pos): a pass leaves the
// engine exactly where nCols decode steps would, bit for bit, so prefill and decode interleave freely. Kernels:
// engine/wgsl/qwen4exp_batch.js (the decode kernels with a column index; gemv_mc shares each weight read across
// GEMV_MC_COLS columns). QSA: engine/qwen4exp_qsa.js, the decode path's rule per pass (dense while every column has
// <= 2051 cells, the selected cells past that).
//
//   const pb = await Qwen4ExpBatch.create(engine, weights, { maxCols: 64 });
//   const r = await pb.prefill(promptIds, { wantLogits: true });    // { id, logits } (head) or { hidden }
//   const v = await pb.verify([last, ...draft]);                     // per column: greedy ids (head) / hiddens
//   await pb.accept(n);                                              // keep the first n columns of that pass
//
// Verify keeps a copy of the recurrent state (DeltaNet conv + S, the PLE ring) from before the pass; accept(n < N)
// restores it, rolls the n-gram history back and re-runs the first n columns (no head). The K / V cache and QSA's
// rawK need no copy (the re-run rewrites them) and a pooled key is re-pooled when its block completes again.
// Cost: the re-run is the whole trunk for n columns (MoE expert reads included), about one more pass, and every
// verify copies ~115 MB of state. Replace it with the Qwen3.5 engine's replay rollback (dn_delta_mc: replay only the
// DeltaNet step from S before the pass) before MTP drafting is built on verify (docs/kernels.md).
import * as MC from "./wgsl/qwen4exp_batch.js";
import { qsaScratch, qsaSetFrame, qsaBind, qsaStateOps, qsaSelectOps, QSA_SEL_STRIDE } from "./qwen4exp_qsa.js";

const U = GPUBufferUsage;
const fr = Math.fround;

export class Qwen4ExpBatch {
  static async create(engine, weights, opts = {}) { const b = new Qwen4ExpBatch(); await b._init(engine, weights, opts); return b; }

  // maxCols: columns per pass (longer prompts run in several); verifyCols: the most columns a pass with the head on
  // every column (verify) takes, which sizes its logits (vocab x 4 B per column).
  async _init(eng, weights, { maxCols = 64, verifyCols = 8 } = {}) {
    const { device, cfg } = eng, wq = eng._wq;
    Object.assign(this, { eng, device, cfg, maxCols, verifyCols: Math.min(verifyCols, maxCols) });
    const d = cfg.dim, hc = cfg.hc, HD = cfg.hDim, ss = cfg.ssm, K = cfg.K, nH = cfg.nHeads, nKV = cfg.nKVHeads, hd = cfg.headDim;
    const { convDim, keyDim } = eng.dn, N = maxCols, eps = cfg.eps;
    const C = MC.GEMV_MC_COLS;

    // ---- pipelines
    const pipe = async (code, label) => device.createComputePipelineAsync({ layout: "auto", label,
      compute: { module: device.createShaderModule({ code, label }), entryPoint: "main" } });
    const P = this.P = {};
    await Promise.all([
      ...["q4", "q8", "f32"].map(async (k) => { P["gemv_" + k] = await pipe(MC.gemvMcWGSL(k), "gemv_mc_" + k); }),
      ...["q4", "q8"].map(async (k) => { P["gu_" + k] = await pipe(MC.moeGuMcWGSL(k), "moe_gu_mc_" + k); P["dn_" + k] = await pipe(MC.moeDnMcWGSL(k), "moe_dn_mc_" + k); }),
      ...Object.entries({ route: MC.MOE_ROUTE_MC_WGSL, hcNorm: MC.HC_NORM_MC_WGSL, hcMix: MC.HC_MIX_MC_WGSL, hcComb: MC.HC_COMBINE_MC_WGSL,
        dnConv: MC.DN_CONV_MC_WGSL, dnL2: MC.DN_L2_MC_WGSL, dnDelta: MC.DN_DELTA_MC_WGSL, dnGnorm: MC.DN_GNORM_MC_WGSL,
        attnPrep: MC.ATTN_PREP_MC_WGSL, attnDense: MC.attnDecMcWGSL(), attnGather: MC.attnDecMcWGSL({ gather: QSA_SEL_STRIDE }),
        pleGate: MC.PLE_GATE_MC_WGSL, pleConv: MC.PLE_CONV_MC_WGSL, pleRing: MC.PLE_RING_MC_WGSL, am1: MC.ARGMAX1_MC_WGSL, am2: MC.ARGMAX2_MC_WGSL })
        .map(async ([k, code]) => { P[k] = await pipe(code, k); }),
    ]);

    // ---- scratch, N columns of each of the decode engine's buffers
    const buf = (n, extra = 0) => device.createBuffer({ size: Math.max(16, Math.ceil(n * 4 / 16) * 16), usage: U.STORAGE | U.COPY_SRC | U.COPY_DST | extra });
    const B = this.B = {
      R: buf(N * HD), xn: buf(N * HD), lo: buf(N * cfg.hcRank), gate: buf(N * HD), mixed: buf(N * d), inj: buf(N * hc), out: buf(N * d),
      qkv: buf(N * convDim), z: buf(N * ss.inner), alpha: buf(N * ss.nVHeads), beta: buf(N * ss.nVHeads), co: buf(N * convDim),
      o: buf(N * ss.inner), go: buf(N * ss.inner),
      qf: buf(N * 2 * nH * hd), kb: buf(N * nKV * hd), vb: buf(N * nKV * hd), qn: buf(N * nH * hd), ao: buf(N * nH * hd),
      logits: buf(N * cfg.nExp), ids: buf(N * K), wts: buf(N * K), h: buf(N * K * cfg.expFF), moe: buf(N * d),
      sg: buf(N), shIds: buf(N), hs: buf(N * cfg.shFF), sh: buf(N * d),
      F: device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST }), cs: buf(N * MC.MC_FRAME_COL),
    };
    if (eng.B.emb) Object.assign(B, { emb: buf(N * d), pk: buf(N * HD), pv: buf(N * d), nb: buf(N * HD), gb: buf(N * HD) });
    if (eng.qsa) this.qsaS = qsaScratch(device, { maxCols: N, maxBlocks: eng.qsaSh.maxBlocks });

    const uni = (fields) => {
      const n = Math.max(4, Math.ceil(fields.length / 4) * 4), ab = new ArrayBuffer(n * 4), dv = new DataView(ab);
      fields.forEach((v, i) => (typeof v === "number" ? dv.setUint32(i * 4, v, true) : dv.setFloat32(i * 4, v.f, true)));
      const b = device.createBuffer({ size: ab.byteLength, usage: U.UNIFORM | U.COPY_DST, mappedAtCreation: true });
      new Uint8Array(b.getMappedRange()).set(new Uint8Array(ab));
      b.unmap();
      return b;
    };
    const bg = (p, list) => device.createBindGroup({ layout: p.getBindGroupLayout(0),
      entries: list.map((b, i) => (b ? { binding: i, resource: { buffer: b } } : null)).filter(Boolean) });
    const f = (v) => ({ f: v });
    const cdiv = (a, b) => Math.ceil(a / b);
    // y (column stride ys) = mat x (column stride xs); shapes as the engine's gemv
    const gemv = (Mw, x, y, { dIn = Mw.shape?.[1], dOut = Mw.shape?.[0] ?? Mw.n / dIn, xs = dIn, ys = dOut, mode = 0, scale = 0 } = {}) => {
      if (!(dIn > 0 && dOut > 0 && Number.isInteger(dOut))) throw new Error(`qwen4exp batch: gemv shape ${dOut} x ${dIn}`);
      if (xs % 4) throw new Error("qwen4exp batch: gemv column stride must be a multiple of 4");
      const p = P["gemv_" + Mw.kind], u = uni([dOut, dIn, 0, 0, 0, mode, f(scale), xs / 4, ys, 0, 0, 0]);
      const g = Mw.kind === "f32" ? bg(p, [Mw.buf, null, x, y, u, B.F]) : bg(p, [Mw.qs, Mw.sc, x, y, u, B.F]);
      return { pipe: p, bg: g, dyn: (pos, n) => [cdiv(dOut, 8), cdiv(n, C), 1] };
    };
    const hcMix = (hw, inject = true) => {
      const norm = wq(hw.norm, "hc norm"), ops = [];
      ops.push({ pipe: P.hcNorm, bg: bg(P.hcNorm, [B.R, norm.buf, B.xn, uni([d, f(eps), hc, 0])]), dyn: (pos, n) => [hc, n, 1] });
      ops.push(gemv(wq(hw.down, "hc down"), B.xn, B.lo, { dIn: HD, mode: 1, scale: 1 / hc }));
      ops.push(gemv(wq(hw.up, "hc up"), B.lo, B.gate, { dIn: cfg.hcRank }));
      ops.push({ pipe: P.hcMix, bg: bg(P.hcMix, [B.xn, B.gate, B.mixed, uni([d, hc, 0, 0])]), dyn: (pos, n) => [cdiv(d, 256), n, 1] });
      if (inject) ops.push(gemv(wq(hw.inject, "hc inject"), B.xn, B.inj, { dIn: HD }));
      return ops;
    };
    const hcComb = (a, b) => ({ pipe: P.hcComb, bg: bg(P.hcComb, [B.R, a, b || a, B.inj, uni([d, hc, b ? 1 : 0, 0])]), dyn: (pos, n) => [cdiv(HD, 256), n, 1] });

    // ---- per layer ops, on the engine's per-layer state
    this.layers = [];
    for (const el of eng.layers) {
      const i = el.i, L = weights.layers[i - eng.lo], st = el.st, ops = [];
      if (st.pw) {   // PLE: gate + normed value per column, the conv, then the ring
        const pc = L.ple, { K1, slots } = st.pleU, pu = uni([d, K1, cfg.ple.ngram, slots, f(eps), hc, 0, 0]);
        ops.push(gemv(wq(pc.key, "ple key"), B.emb, B.pk));
        ops.push(gemv(wq(pc.value, "ple value"), B.emb, B.pv));
        ops.push({ pipe: P.pleGate, bg: bg(P.pleGate, [B.R, B.pk, B.pv, st.pw, B.nb, B.gb, pu]), dyn: (pos, n) => [hc, n, 1] });
        ops.push({ pipe: P.pleConv, bg: bg(P.pleConv, [B.R, B.gb, st.pw, st.ring, B.nb, pu, B.F]), dyn: (pos, n) => [cdiv(HD, 256), n, 1] });
        ops.push({ pipe: P.pleRing, bg: bg(P.pleRing, [st.ring, B.nb, pu, B.F]), dyn: (pos, n) => [cdiv(HD, 256), n, 1] });
      }
      ops.push(...hcMix(L.hcAttn));
      if (el.isQ) {
        ops.push(gemv(wq(L.wq, "attn_q"), B.mixed, B.qf), gemv(wq(L.wk, "attn_k"), B.mixed, B.kb), gemv(wq(L.wv, "attn_v"), B.mixed, B.vb));
        let sel = [];
        if (eng.qsa) {
          const G = qsaBind(device, eng.qsaK, eng.qsaSh, this.qsaS, st.qsa, B.mixed);
          ops.push(...qsaStateOps(eng.qsaK, G));
          sel = qsaSelectOps(eng.qsaK, G, eng.sparse);
        }
        const qkw = new Float32Array(2 * hd);
        qkw.set(L.qNorm.data, 0); qkw.set(L.kNorm.data, hd);
        ops.push({ pipe: P.attnPrep, bg: bg(P.attnPrep, [B.qf, B.kb, B.vb, B.qn, st.kc, st.vc, wq({ kind: "f32", data: qkw }).buf, B.cs, st.ap, B.F]),
          dyn: (pos, n) => [nH + nKV, n, 1] });
        ops.push(...sel);
        ops.push({ pipe: P.attnDense, bg: bg(P.attnDense, [B.qn, st.kc, st.vc, B.qf, B.ao, st.ap, B.F]),
          dyn: (pos, n) => (eng.qsa && eng.sparse(pos, n) ? null : [nH, n, 1]) });
        if (eng.qsa) ops.push({ pipe: P.attnGather, bg: bg(P.attnGather, [B.qn, st.kc, st.vc, B.qf, B.ao, st.ap, null, this.qsaS.sel]),
          dyn: (pos, n) => (eng.sparse(pos, n) ? [nH, n, 1] : null) });
        ops.push(gemv(wq(L.wo, "attn_output"), B.ao, B.out));
      } else {
        ops.push(gemv(wq(L.wqkv, "qkv"), B.mixed, B.qkv), gemv(wq(L.wz, "z"), B.mixed, B.z),
          gemv(wq(L.wBeta, "beta"), B.mixed, B.beta, { dIn: d }), gemv(wq(L.wAlpha, "alpha"), B.mixed, B.alpha, { dIn: d }));
        ops.push({ pipe: P.dnConv, bg: bg(P.dnConv, [B.qkv, st.conv, wq(L.conv, "conv").buf, B.co, uni([convDim, ss.convK, 0, 0]), B.F]), dyn: () => [cdiv(convDim, 256), 1, 1] });
        ops.push({ pipe: P.dnL2, bg: bg(P.dnL2, [B.co, uni([ss.stateSize, f(eps), convDim, 0])]), dyn: (pos, n) => [2 * ss.nKHeads, n, 1] });
        ops.push({ pipe: P.dnDelta, bg: bg(P.dnDelta, [B.co, st.S, B.alpha, B.beta, wq(L.dtBias).buf, wq(L.ssmA).buf, B.o,
          uni([ss.nKHeads, ss.stateSize, keyDim, convDim, ss.nVHeads, ss.inner, 0, 0]), B.F]), dyn: () => [ss.nVHeads, 1, 1] });
        ops.push({ pipe: P.dnGnorm, bg: bg(P.dnGnorm, [B.o, B.z, wq(L.ssmNorm).buf, B.go, uni([ss.stateSize, f(eps), ss.inner, 0])]), dyn: (pos, n) => [ss.nVHeads, n, 1] });
        ops.push(gemv(wq(L.wOut, "ssm_out"), B.go, B.out));
      }
      ops.push(hcComb(B.out, null));
      // MoE
      ops.push(...hcMix(L.hcFfn));
      ops.push(gemv(wq(L.router, "router"), B.mixed, B.logits, { dIn: d }));
      ops.push({ pipe: P.route, bg: bg(P.route, [B.logits, B.ids, B.wts, uni([cfg.nExp, K, 0, 0])]), dyn: (pos, n) => [n, 1, 1] });
      const eg = wq(L.expGate, "experts gate"), eu = wq(L.expUp, "experts up"), ed = wq(L.expDown, "experts down"), ff = cfg.expFF;
      ops.push({ pipe: P["gu_" + eg.kind], bg: bg(P["gu_" + eg.kind], [eg.qs, eg.sc, eu.qs, eu.sc, B.mixed, B.h, B.ids, uni([ff, d, K, d / 4])]), dyn: (pos, n) => [cdiv(ff, 8), K, n] });
      ops.push({ pipe: P["dn_" + ed.kind], bg: bg(P["dn_" + ed.kind], [ed.qs, ed.sc, B.h, B.moe, B.ids, B.wts, uni([d, ff, K, 0])]), dyn: (pos, n) => [cdiv(d, 8), 1, n] });
      const sgw = wq(L.shGate, "shared gate"), suw = wq(L.shUp, "shared up"), sdw = wq(L.shDown, "shared down"), sff = cfg.shFF;
      ops.push(gemv(wq(L.shRouter, "shared router"), B.mixed, B.sg, { dOut: 1, dIn: d }));
      ops.push({ pipe: P["gu_" + sgw.kind], bg: bg(P["gu_" + sgw.kind], [sgw.qs, sgw.sc, suw.qs, suw.sc, B.mixed, B.hs, B.shIds, uni([sff, d, 1, d / 4])]), dyn: (pos, n) => [cdiv(sff, 8), 1, n] });
      ops.push({ pipe: P["dn_" + sdw.kind], bg: bg(P["dn_" + sdw.kind], [sdw.qs, sdw.sc, B.hs, B.sh, B.shIds, B.sg, uni([d, sff, 1, 1])]), dyn: (pos, n) => [cdiv(d, 8), 1, n] });
      ops.push(hcComb(B.moe, B.sh));
      this.layers.push({ i, st, ops });
    }

    // ---- head on every column (verify): the head mixer, the LM head, a greedy id per column
    if (eng.hasHead) {
      const V = eng.dims.vocab, per = 1024, nPart = cdiv(V, per), NV = this.verifyCols;
      Object.assign(B, { hl: buf(NV * V), part: buf(NV * nPart * 2), am: buf(NV * 2) });
      const head = wq(weights.head, "output");
      const hg = gemv(head, B.mixed, B.hl, { dIn: d });
      this.headOps = [...hcMix(weights.headHC, false), hg,
        { pipe: P.am1, bg: bg(P.am1, [B.hl, B.part, uni([V, per, nPart, 0])]), dyn: (pos, n) => [nPart, n, 1] },
        { pipe: P.am2, bg: bg(P.am2, [B.part, B.am, uni([V, per, nPart, 0])]), dyn: (pos, n) => [n, 1, 1] }];
      this.amStage = device.createBuffer({ size: NV * 8, usage: U.MAP_READ | U.COPY_DST });
    }
    this._stages = new Map();
    this._snap = null;
  }

  _stage(bytes) {
    let s = this._stages.get(bytes);
    if (!s) { s = this.device.createBuffer({ size: bytes, usage: U.MAP_READ | U.COPY_DST }); this._stages.set(bytes, s); }
    return s;
  }
  async _read(s, bytes, Type = Float32Array) {
    await s.mapAsync(GPUMapMode.READ, 0, bytes);
    const a = new Type(s.getMappedRange(0, bytes).slice(0));
    s.unmap();
    return a;
  }

  // One pass of n <= maxCols tokens at the engine's position. head: "last" (the engine's head on the last column, or
  // its hidden when the engine has no head), "all" (a greedy id per column; n <= verifyCols), "none".
  // hidden: the input residual (n x hDim) when the engine has no embedding (a later room stage).
  // -> { id, logits } | { hidden } | { ids } , plus hiddens (n x hDim, the output residual) when wantHidden.
  async _pass(ids, { head = "last", wantLogits = false, wantHidden = false, hidden = null } = {}) {
    const { eng, device, cfg, B } = this, q = device.queue, n = ids.length, pos = eng.pos, HD = cfg.hDim, d = cfg.dim;
    if (!(n >= 1 && n <= this.maxCols)) throw new Error(`qwen4exp batch: ${n} columns (1 .. ${this.maxCols})`);
    if (pos + n > eng.maxSeq) throw new Error(`qwen4exp: context full (${eng.maxSeq})`);
    if (head === "all" && !eng.hasHead) head = "none";
    if (head === "all" && n > this.verifyCols) throw new Error(`qwen4exp batch: a verify pass takes at most ${this.verifyCols} columns`);
    // the residual
    if (eng.hasEmbed) {
      const R = new Float32Array(n * HD);
      for (let c = 0; c < n; c++) { const e = eng._embedRow(ids[c]); for (let s = 0; s < cfg.hc; s++) R.set(e, c * HD + s * d); }
      q.writeBuffer(B.R, 0, R);
    } else {
      if (!hidden || hidden.length !== n * HD) throw new Error("qwen4exp batch: this stage needs the input residual (hidden, n x hDim)");
      q.writeBuffer(B.R, 0, hidden);
    }
    // the n-gram rows (the PLE layer's input), the history either way
    if (B.emb) {
      q.writeBuffer(B.emb, 0, await eng.ngram.rows(eng.ngs.push(ids)));
    } else eng.ngs?.push(ids);
    // the frame: pos and n, and every column's rope table (the decode frame's recurrence)
    q.writeBuffer(B.F, 0, new Uint32Array([pos, n, 0, 0]));
    const cs = new Float32Array(n * MC.MC_FRAME_COL);
    for (let c = 0; c < n; c++) {
      let th = fr(pos + c);
      for (let i = 0; i < cfg.ropeDims / 2; i++) { cs[c * MC.MC_FRAME_COL + i * 4] = fr(Math.cos(th)); cs[c * MC.MC_FRAME_COL + i * 4 + 1] = fr(Math.sin(th)); th = fr(th * eng._ropeTs); }
    }
    q.writeBuffer(B.cs, 0, cs);
    if (eng.qsa) qsaSetFrame(device, this.qsaS, pos, n);

    const enc = device.createCommandEncoder();
    let pass = enc.beginComputePass();
    for (const l of this.layers) eng._encode(pass, l.ops, pos, n);
    if (head === "all") eng._encode(pass, this.headOps, pos, n);
    pass.end();
    const out = {};
    const last = (n - 1) * HD * 4;
    if (head === "last") {
      if (eng.hasHead) {
        enc.copyBufferToBuffer(B.R, last, eng.B.R, 0, HD * 4);
        pass = enc.beginComputePass();
        eng._encode(pass, eng.headOps, pos + n - 1, 1);
        pass.end();
        enc.copyBufferToBuffer(eng.B.am, 0, eng.amStage, 0, 16);
        if (wantLogits) enc.copyBufferToBuffer(eng.B.hl, 0, eng.logitStage, 0, eng.dims.vocab * 4);
      } else enc.copyBufferToBuffer(B.R, last, eng.hStage, 0, HD * 4);
    }
    if (head === "all") enc.copyBufferToBuffer(B.am, 0, this.amStage, 0, n * 8);
    let hs = null;
    if (wantHidden) { hs = this._stage(this.maxCols * HD * 4); enc.copyBufferToBuffer(B.R, 0, hs, 0, n * HD * 4); }
    q.submit([enc.finish()]);
    eng.pos += n;
    if (head === "last") {
      if (eng.hasHead) {
        await eng.amStage.mapAsync(GPUMapMode.READ);
        out.id = new Uint32Array(eng.amStage.getMappedRange())[0];
        eng.amStage.unmap();
        if (wantLogits) out.logits = await this._read(eng.logitStage, eng.dims.vocab * 4);
      } else out.hidden = await this._read(eng.hStage, HD * 4);
    }
    if (head === "all") { const a = await this._read(this.amStage, n * 8, Uint32Array); out.ids = Array.from({ length: n }, (_, c) => a[2 * c]); }
    if (wantHidden) out.hiddens = await this._read(hs, n * HD * 4);
    if (head === "none" && !wantHidden) await q.onSubmittedWorkDone();
    return out;
  }

  // Prompt tokens in passes of maxCols (the last pass gets the head). Returns the last pass's result (as _pass).
  // opts.chunks: explicit pass sizes (tests), repeated until the ids run out.
  async prefill(ids, { chunks = null, ...opts } = {}) {
    let r = null, o = 0, k = 0;
    while (o < ids.length) {
      const sz = Math.min(chunks ? chunks[k++ % chunks.length] : this.maxCols, this.maxCols, ids.length - o);
      const end = o + sz >= ids.length;
      const res = await this._pass(ids.slice(o, o + sz), end ? opts : { ...opts, head: "none" });
      if (opts.onPass) opts.onPass(res, o, sz);
      r = res; o += sz;
    }
    return r;
  }

  // A verify pass: ids = [the last accepted token, ...draft]; -> { ids: greedy id per column } (head), and hiddens
  // when wantHidden. The state before the pass is kept for accept().
  async verify(ids, { wantHidden = false } = {}) {
    this._save();
    const pos0 = this.eng.pos;
    const r = await this._pass(ids, { head: "all", wantHidden });
    this._pending = { ids: ids.slice(), pos0 };
    return r;
  }

  // Keep the first n columns of the last verify pass (n = its length: nothing to do).
  async accept(n) {
    const v = this._pending;
    if (!v) throw new Error("qwen4exp batch: accept without a verify pass");
    this._pending = null;
    if (!(Number.isInteger(n) && n >= 0 && n <= v.ids.length)) throw new Error(`qwen4exp batch: accept(${n}) of ${v.ids.length}`);
    if (n === v.ids.length) return;
    this._restore();
    this.eng.pos = v.pos0;
    this.eng.ngs?.rollback(v.ids.length);
    if (n) await this._pass(v.ids.slice(0, n), { head: "none" });
  }

  // the recurrent state (DeltaNet conv + S, the PLE ring), copied on the GPU
  _bufs() { return this.eng.layers.flatMap((l) => [l.st.conv, l.st.S, l.st.ring].filter(Boolean)); }
  _save() {
    const src = this._bufs();
    if (!this._snap) this._snap = src.map((b) => this.device.createBuffer({ size: b.size, usage: U.COPY_DST | U.COPY_SRC }));
    const enc = this.device.createCommandEncoder();
    src.forEach((b, k) => enc.copyBufferToBuffer(b, 0, this._snap[k], 0, b.size));
    this.device.queue.submit([enc.finish()]);
  }
  _restore() {
    const enc = this.device.createCommandEncoder();
    this._bufs().forEach((b, k) => enc.copyBufferToBuffer(this._snap[k], 0, b, 0, b.size));
    this.device.queue.submit([enc.finish()]);
  }
}
