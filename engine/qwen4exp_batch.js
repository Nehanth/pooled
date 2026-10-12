// Batched passes for Qwen3.8-Flash-Next (engine/qwen4exp.js): prompt prefill in chunks of up to maxCols tokens and
// speculative verify passes. Roadmap 33 PR plan items 5 (batched / wide prefill, verify) and 9 (QSA in them).
//
// A pass runs nCols consecutive tokens through every layer at once, on the decode engine's own state (DeltaNet conv
// and S, the PLE ring, the K / V cache, QSA's rawK / pooled keys, the n-gram history and pos) and its weight buffers
// (the joined / fused ones from each layer's st.w): a pass leaves the engine exactly where nCols decode steps would,
// bit for bit, so prefill and decode interleave freely. The op lists mirror the decode engine's: per layer two halves
// { pre, site, rest, comb }, laid out fused as forwardToken runs them (each HC combine folded into the next site,
// _halfOps), with the multi-column kernels of engine/wgsl/qwen4exp_batch.js (the decode kernels with a column index;
// gemv_mc shares each weight read across GEMV_MC_COLS columns). Ops carry dyn(pos, nCols) and run through the
// engine's _encode. QSA: engine/qwen4exp_qsa.js, the decode path's rule per pass (dense while every column has
// <= 2051 cells, the selected cells past that), on this module's own nCols-wide scratch.
//
//   const pb = await Qwen4ExpBatch.create(engine, weights, { maxCols: 64 });
//   const r = await pb.prefill(promptIds, { wantLogits: true });    // { id, logits } (head) or { hidden }
//   const v = await pb.verify([last, ...draft]);                     // { ids }: a greedy id per column
//   await pb.accept(n);                                              // keep the first n columns of that pass
//
// Verify keeps a copy of the recurrent state (DeltaNet conv + S, the PLE ring) from before the pass; accept(n < N)
// restores it, rolls the n-gram history back and re-runs the first n columns (no head). The K / V cache and QSA's
// rawK need no copy (the re-run rewrites them) and a pooled key is re-pooled when its block completes again.
// Cost: the re-run is the whole trunk for n columns (MoE expert reads included), about one more pass, and every
// verify copies ~115 MB of state. Deferred (#348 review finding 3): replace it with the Qwen3.5 engine's replay
// rollback (save each column's DeltaNet inputs, replay only the delta step from S before the pass) before MTP
// drafting is built on verify (docs/kernels.md).
//
// Expert offload (engine/expert_store.js): the router of every layer is marked (op.cut) where an offloaded layer
// would split the pass: read the nCols x K ids back, copy the experts into the store's whole-layer region
// (ExpertStore.loadLayer / loadExperts) and run the expert kernels on regionEntry() with the original ids. Until that
// is wired, create() refuses an engine with offloaded layers.
import * as MC from "./wgsl/qwen4exp_batch.js";
import { qsaScratch, qsaSetFrame, qsaBind, qsaStateOps, qsaSelectOps, QSA_SEL_STRIDE } from "./qwen4exp_qsa.js";

const U = GPUBufferUsage;
const fr = Math.fround;
const SUBMIT_LAYERS = 12;   // layers per submit, as forwardToken

export class Qwen4ExpBatch {
  static async create(engine, weights, opts = {}) { const b = new Qwen4ExpBatch(); await b._init(engine, weights, opts); return b; }

  // weights: the object the engine was built from (its unfused entries; the joined / fused ones come from st.w).
  // maxCols: columns per pass (longer prompts run in several); verifyCols: the most columns a pass with the head on
  // every column (verify) takes, which sizes its logits (vocab x 4 B per column).
  async _init(eng, W, { maxCols = 64, verifyCols = 8 } = {}) {
    const off = eng.offLayers?.length ? eng.offLayers : eng.layers.filter((l) => l.off).map((l) => l.i);
    if (off.length || eng.experts) throw new Error(`qwen4exp batch: layers ${off.join(",")} have offloaded experts; batched passes do not route them `
      + "through ExpertStore.loadLayer / loadExperts yet: run them with forwardToken / prefillTokens");
    const { device, cfg } = eng, wq = eng._wq;
    if (!wq) throw new Error("qwen4exp batch: the engine does not expose its weights (_wq)");
    Object.assign(this, { eng, device, cfg, maxCols, verifyCols: Math.min(verifyCols, maxCols) });
    const d = cfg.dim, hc = cfg.hc, HD = cfg.hDim, ss = cfg.ssm, K = cfg.K, nH = cfg.nHeads, nKV = cfg.nKVHeads, hd = cfg.headDim;
    const { convDim, keyDim } = eng.dn, N = maxCols, eps = cfg.eps, C = MC.GEMV_MC_COLS, rank = cfg.hcRank;
    const nQ = 2 * nH * hd, nK = nKV * hd, qs = nQ + 2 * nK, qkvzS = convDim + ss.inner, ls = cfg.nExp + 1;
    if (rank % 4) throw new Error("qwen4exp batch: the HC rank must be a multiple of 4");

    // ---- pipelines (the GEMVs per lanes-per-row variant, compiled on first use)
    const pipe = (code, label) => device.createComputePipelineAsync({ layout: "auto", label,
      compute: { module: device.createShaderModule({ code, label }), entryPoint: "main" } });
    const P = this.P = {};
    await Promise.all([
      ...["q4", "q8"].map(async (k) => { P["gu_" + k] = await pipe(MC.moeGuMcWGSL(k), "moe_gu_mc_" + k); P["dn_" + k] = await pipe(MC.moeDnMcWGSL(k), "moe_dn_mc_" + k); }),
      ...Object.entries({ route: MC.MOE_ROUTE_MC_WGSL, hcSite: MC.HC_SITE_MC_WGSL, hcMix: MC.HC_MIX_MC_WGSL, hcComb: MC.HC_COMBINE_MC_WGSL,
        dnConv: MC.DN_CONV_MC_WGSL, dnDelta: MC.DN_DELTA_MC_WGSL, attnPrep: MC.ATTN_PREP_MC_WGSL,
        attnDense: MC.attnDecMcWGSL(), attnGather: MC.attnDecMcWGSL({ gather: QSA_SEL_STRIDE }),
        ple: MC.PLE_MC_WGSL, am1: MC.ARGMAX1_MC_WGSL, am2: MC.ARGMAX2_MC_WGSL })
        .map(async ([k, code]) => { P[k] = await pipe(code, k); }),
    ]);
    const lazy = new Map();
    const gemvPipe = (key, code) => { if (!lazy.has(key)) lazy.set(key, pipe(code, key)); return lazy.get(key); };

    // ---- scratch: N columns of each of the decode engine's buffers
    const buf = (n) => device.createBuffer({ size: Math.max(16, Math.ceil(n * 4 / 16) * 16), usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    const B = this.B = {
      R: buf(N * HD), xn: buf(N * HD), lo: buf(N * rank), gate: buf(N * HD), mixed: buf(N * d), out: buf(N * d),
      inj: [buf(N * hc * hc), buf(N * hc * hc)], xnTmp: buf(HD),
      qkvz: buf(N * qkvzS), ba: buf(N * 2 * ss.nVHeads), co: buf(N * convDim), o: buf(N * ss.inner), go: buf(N * ss.inner),
      qkvA: buf(N * qs), qn: buf(N * nH * hd), ao: buf(N * nH * hd),
      logits: buf(N * ls), ids: buf(N * K), wts: buf(N * K), h: buf(N * K * cfg.expFF), moe: buf(N * d),
      shIds: buf(4), hs: buf(N * cfg.shFF), sh: buf(N * d),
      F: device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST }), cs: buf(N * MC.MC_FRAME_COL),
    };
    device.queue.writeBuffer(B.shIds, 0, new Uint32Array(4));
    if (eng.B.emb) Object.assign(B, { emb: buf(N * d), pk: buf(N * HD), pv: buf(N * d), dbg: buf(2 * HD + hc) });
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
    const pending = [];   // ops whose pipeline is still compiling: { op, p (promise), list }
    const later = (pp, list, dyn, extra = {}) => { const op = { pipe: null, bg: null, dyn, ...extra }; pending.push({ op, pp, list }); return op; };

    // y (column stride ys) = mat x (column stride xs): the decode GEMV's lanes per row (gemvLanes of its shape)
    const gemv = (M, x, y, { dIn = M.shape?.[1], dOut = M.shape?.[0] ?? M.n / dIn, xs = dIn, ys = dOut, xOff = 0, yOff = 0, mode = 0, scale = 0, cut = false } = {}) => {
      if (!(dIn > 0 && dOut > 0 && Number.isInteger(dOut))) throw new Error(`qwen4exp batch: gemv shape ${dOut} x ${dIn}`);
      if (xs % 4 || xOff % 4) throw new Error("qwen4exp batch: gemv column stride / offset must be a multiple of 4");
      const lanes = MC.gemvLanes(M.kind, dOut, dIn), nwg = cdiv(dOut, 256 / lanes);
      const u = uni([dOut, dIn, 0, xOff / 4, yOff, mode, f(scale), xs / 4, ys, 0, 0, 0]);
      const list = M.kind === "f32" ? [M.buf, null, x, y, u, B.F] : [M.qs, M.sc, x, y, u, B.F];
      return later(gemvPipe(`gemv_mc_${M.kind}_${lanes}`, MC.gemvMcWGSL(M.kind, lanes)), list,
        (pos, n) => [Math.min(nwg, 32768), cdiv(nwg, 32768), cdiv(n, C)], cut ? { cut } : {});
    };
    const ewg = (n) => (pos, c) => [cdiv(n, 256), c, 1];

    // HC sites, as the engine's: site h's inject partials go to inj[h % 2]
    let nSite = 0;
    const site = (hw, inject = true) => ({ h: nSite++, w: wq(hw.norm, "hc norm").buf, injW: inject ? wq(hw.inject, "hc inject").buf : null });
    const siteOp = (S, comb) => {
      const out = B.inj[S.h % 2], prev = comb ? B.inj[comb.h % 2] : B.inj[(S.h + 1) % 2];
      const u = uni([d, hc, f(eps), comb ? (comb.b ? 2 : 1) : 0, S.injW ? 1 : 0, 0, 0, 0]);
      return { pipe: P.hcSite, bg: bg(P.hcSite, [B.R, comb ? comb.a : B.mixed, comb?.b || B.mixed, prev, S.w, B.xn, S.injW || S.w, out, u]), dyn: (pos, n) => [hc, n, 1] };
    };
    const combOp = (Cb) => ({ pipe: P.hcComb, bg: bg(P.hcComb, [B.R, Cb.a, Cb.b || Cb.a, B.inj[Cb.h % 2], uni([d, hc, Cb.b ? 1 : 0, 0])]), dyn: ewg(HD) });
    const hcMixRest = (hw) => {
      const ops = [gemv(wq(hw.down, "hc down"), B.xn, B.lo, { dIn: HD, ys: rank, mode: 1, scale: 1 / hc })];
      const up = wq(hw.up, "hc up");
      if (up.kind !== "f32") {
        const lanes = MC.gemvLanes(up.kind, HD, rank), u = uni([HD, rank, 0, 0, 0, 0, f(0), d, rank / 4, HD, 0, 0]);
        ops.push(later(gemvPipe(`gemvmix_mc_${up.kind}_${lanes}`, MC.gemvMixMcWGSL(up.kind, lanes)), [up.qs, up.sc, B.lo, B.gate, u, B.xn, B.mixed],
          (pos, n) => [cdiv(d, 256 / lanes), 1, n]));
      } else {
        ops.push(gemv(up, B.lo, B.gate, { dIn: rank, ys: HD }));
        ops.push({ pipe: P.hcMix, bg: bg(P.hcMix, [B.xn, B.gate, B.mixed, uni([d, hc, 0, 0])]), dyn: ewg(d) });
      }
      return ops;
    };
    const halfOps = (halves) => halves.map((H, k) => {
      const prev = k > 0 ? halves[k - 1] : null, next = halves[k + 1];
      const fold = prev && prev.comb && !H.pre.length;
      const list = [...H.pre, siteOp(H.site, fold ? prev.comb : null), ...H.rest];
      if (H.comb && !(next && !next.pre.length)) list.push(combOp(H.comb));
      return list;
    });

    // ---- per layer ops, on the engine's per-layer state and weights
    const halves = [], lmeta = [];
    for (const el of eng.layers) {
      const i = el.i, L = W.layers[i - eng.lo], st = el.st, w = st.w, pre = [], rest = [];
      if (w.pw) {   // PLE: key / value projections of the n-gram rows, then the injection over the columns in order
        const pc = L.ple, { K1, slots } = st.pleU;
        pre.push(gemv(wq(pc.key, "ple key"), B.emb, B.pk));
        pre.push(gemv(wq(pc.value, "ple value"), B.emb, B.pv));
        pre.push({ pipe: P.ple, bg: bg(P.ple, [B.R, B.pk, B.pv, w.pw, st.ring, B.dbg, uni([d, K1, cfg.ple.ngram, slots, f(eps), hc, 0, 0]), B.F]), dyn: () => [hc, 1, 1] });
      }
      const sA = site(L.hcAttn);
      rest.push(...hcMixRest(L.hcAttn));
      if (el.isQ) {
        if (w.qkvA) rest.push(gemv(w.qkvA, B.mixed, B.qkvA, { ys: qs }));
        else rest.push(gemv(wq(L.wq, "attn_q"), B.mixed, B.qkvA, { ys: qs }), gemv(wq(L.wk, "attn_k"), B.mixed, B.qkvA, { ys: qs, yOff: nQ }),
          gemv(wq(L.wv, "attn_v"), B.mixed, B.qkvA, { ys: qs, yOff: nQ + nK }));
        let sel = [];
        if (eng.qsa) {
          const G = qsaBind(device, eng.qsaK, eng.qsaSh, this.qsaS, st.qsa, B.mixed);
          rest.push(...qsaStateOps(eng.qsaK, G));
          sel = qsaSelectOps(eng.qsaK, G, eng.sparse);
        }
        const ms = uni([qs, nQ, nQ + nK, 0]);
        rest.push({ pipe: P.attnPrep, bg: bg(P.attnPrep, [B.qkvA, B.qn, st.kc, st.vc, wq(L.qNorm).buf, wq(L.kNorm).buf, B.cs, st.ap, ms, B.F]),
          dyn: (pos, n) => [nH + nKV, n, 1] });
        rest.push(...sel);
        rest.push({ pipe: P.attnDense, bg: bg(P.attnDense, [B.qn, st.kc, st.vc, B.qkvA, B.ao, st.ap, B.F, ms]),
          dyn: (pos, n) => (eng.qsa && eng.sparse(pos, n) ? null : [nH, n, 1]) });
        if (eng.qsa) rest.push({ pipe: P.attnGather, bg: bg(P.attnGather, [B.qn, st.kc, st.vc, B.qkvA, B.ao, st.ap, null, ms, this.qsaS.sel]),
          dyn: (pos, n) => (eng.sparse(pos, n) ? [nH, n, 1] : null) });
        rest.push(gemv(wq(L.wo, "attn_output"), B.ao, B.out));
      } else {
        if (w.qkvz) rest.push(gemv(w.qkvz, B.mixed, B.qkvz, { ys: qkvzS }));
        else rest.push(gemv(wq(L.wqkv, "qkv"), B.mixed, B.qkvz, { ys: qkvzS }), gemv(wq(L.wz, "z"), B.mixed, B.qkvz, { ys: qkvzS, yOff: convDim }));
        rest.push(gemv(w.ba, B.mixed, B.ba, { dIn: d }));
        rest.push({ pipe: P.dnConv, bg: bg(P.dnConv, [B.qkvz, st.conv, wq(L.conv, "conv").buf, B.co, uni([ss.convK, 2 * ss.nKHeads, ss.stateSize, f(eps), qkvzS, convDim, 0, 0]), B.F]),
          dyn: () => [convDim / ss.stateSize, 1, 1] });
        rest.push({ pipe: P.dnDelta, bg: bg(P.dnDelta, [B.co, st.S, B.ba, w.dta, B.qkvz, wq(L.ssmNorm).buf, B.o, B.go,
          uni([ss.nKHeads, ss.stateSize, keyDim, ss.nVHeads, f(eps), convDim, 2 * ss.nVHeads, qkvzS, convDim, ss.inner, 0, 0]), B.F]), dyn: () => [ss.nVHeads, 1, 1] });
        rest.push(gemv(wq(L.wOut, "ssm_out"), B.go, B.out));
      }
      const combA = { a: B.out, b: null, h: sA.h };
      // MoE: router + shared gate (one GEMV), top-K per column, routed experts, shared expert
      const sF = site(L.hcFfn), ffn = [...hcMixRest(L.hcFfn)];
      ffn.push(gemv(w.router, B.mixed, B.logits, { dIn: d }));
      ffn.push({ pipe: P.route, bg: bg(P.route, [B.logits, B.ids, B.wts, uni([cfg.nExp, K, ls, 0])]), dyn: (pos, n) => [n, 1, 1], cut: true });
      const eg = wq(L.expGate, "experts gate"), eu = wq(L.expUp, "experts up"), ed = wq(L.expDown, "experts down"), ff = cfg.expFF;
      ffn.push({ pipe: P["gu_" + eg.kind], bg: bg(P["gu_" + eg.kind], [eg.qs, eg.sc, eu.qs, eu.sc, B.mixed, B.h, B.ids, uni([ff, d, K, d / 4, K * ff, K, 0, 0])]),
        dyn: (pos, n) => [cdiv(ff, 8), K, n] });
      ffn.push({ pipe: P["dn_" + ed.kind], bg: bg(P["dn_" + ed.kind], [ed.qs, ed.sc, B.h, B.moe, B.ids, B.wts, uni([d, ff, K, 0, 0, K, K, K * ff / 4])]),
        dyn: (pos, n) => [cdiv(d, 8), 1, n] });
      const sgw = wq(L.shGate, "shared gate"), suw = wq(L.shUp, "shared up"), sdw = wq(L.shDown, "shared down"), sff = cfg.shFF;
      ffn.push({ pipe: P["gu_" + sgw.kind], bg: bg(P["gu_" + sgw.kind], [sgw.qs, sgw.sc, suw.qs, suw.sc, B.mixed, B.hs, B.shIds, uni([sff, d, 1, d / 4, sff, 0, 0, 0])]),
        dyn: (pos, n) => [cdiv(sff, 8), 1, n] });
      ffn.push({ pipe: P["dn_" + sdw.kind], bg: bg(P["dn_" + sdw.kind], [sdw.qs, sdw.sc, B.hs, B.sh, B.shIds, B.logits, uni([d, sff, 1, 1, cfg.nExp, ls, 0, sff / 4])]),
        dyn: (pos, n) => [cdiv(d, 8), 1, n] });
      const combF = { a: B.moe, b: B.sh, h: sF.h };
      halves.push({ pre, site: sA, rest, comb: combA }, { pre: [], site: sF, rest: ffn, comb: combF });
      lmeta.push({ i, st });
    }

    // ---- head: the head mixer's site (which folds the last layer's combine) on every column, then the mixer, the LM
    // head and a greedy id per column ("all": verify), or on the last column only ("last": prefill)
    if (eng.hasHead) {
      const V = eng.dims.vocab, per = 1024, nPart = cdiv(V, per), NV = this.verifyCols;
      Object.assign(B, { hl: buf(NV * V), part: buf(NV * nPart * 2), am: buf(NV * 2) });
      halves.push({ pre: [], site: site(W.headHC, false), comb: null, rest: [...hcMixRest(W.headHC),
        gemv(wq(W.head, "output"), B.mixed, B.hl, { dIn: d }),
        { pipe: P.am1, bg: bg(P.am1, [B.hl, B.part, uni([V, per, nPart, 0])]), dyn: (pos, n) => [nPart, n, 1] },
        { pipe: P.am2, bg: bg(P.am2, [B.part, B.am, uni([V, per, nPart, 0])]), dyn: (pos, n) => [n, 1, 1] }] });
      this.amStage = device.createBuffer({ size: NV * 8, usage: U.MAP_READ | U.COPY_DST });
      this.logitStage = device.createBuffer({ size: Math.ceil(V * 4 / 16) * 16, usage: U.MAP_READ | U.COPY_DST });
    }
    await Promise.all(pending.map(async ({ op, pp, list }) => { op.pipe = await pp; op.bg = bg(op.pipe, list); }));
    const lists = halfOps(halves);
    this.layers = lmeta.map((m, k) => ({ ...m, ops: [...lists[2 * k], ...lists[2 * k + 1]] }));
    if (eng.hasHead) { const h = lists[lists.length - 1]; this.headSite = h.slice(0, 1); this.headRest = h.slice(1); }
    this.hStage = device.createBuffer({ size: HD * 4, usage: U.MAP_READ | U.COPY_DST });
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

  // One pass of n <= maxCols tokens at the engine's position. head: "last" (the head on the last column, or its
  // hidden when the engine has no head), "all" (a greedy id per column; n <= verifyCols), "none".
  // hidden: the input residual (n x hDim) when the engine has no embedding (a later room stage).
  // -> { id, logits } | { hidden } | { ids }, plus hiddens (n x hDim, the output residual) when wantHidden.
  async _pass(ids, opts) {
    const eng = this.eng;
    eng._check(ids[0]);
    try { return await this._run(ids, opts); } catch (e) { eng._failed = e; throw e; }
  }

  async _run(ids, { head = "last", wantLogits = false, wantHidden = false, hidden = null } = {}) {
    const { eng, device, cfg, B } = this, q = device.queue, n = ids.length, pos = eng.pos, HD = cfg.hDim, d = cfg.dim;
    if (!(n >= 1 && n <= this.maxCols)) throw new Error(`qwen4exp batch: ${n} columns (1 .. ${this.maxCols})`);
    if (pos + n > eng.maxSeq) throw new Error(`qwen4exp: context full (${eng.maxSeq})`);
    if (eng.hasEmbed && !ids.every((id) => Number.isInteger(id) && id >= 0 && id < eng.nVocab)) throw new Error("qwen4exp batch: token id outside the vocab");
    if (head === "all" && !eng.hasHead) head = "none";
    if (head === "all" && n > this.verifyCols) throw new Error(`qwen4exp batch: a verify pass takes at most ${this.verifyCols} columns`);
    if (!eng.hasEmbed && !(hidden && hidden.length === n * HD)) throw new Error("qwen4exp batch: this stage needs the input residual (hidden, n x hDim)");
    // the residual
    if (eng.hasEmbed) {
      const R = new Float32Array(n * HD);
      for (let c = 0; c < n; c++) { const e = eng._embedRow(ids[c]); for (let s = 0; s < cfg.hc; s++) R.set(e, c * HD + s * d); }
      q.writeBuffer(B.R, 0, R);
    } else q.writeBuffer(B.R, 0, hidden);
    // the n-gram rows (the PLE layer's input); the history either way
    if (B.emb) q.writeBuffer(B.emb, 0, await eng.ngram.rows(eng.ngs.push(ids)));
    else eng.ngs?.push(ids);
    // the frame: pos and n, and every column's rope table (the decode frame's recurrence)
    q.writeBuffer(B.F, 0, new Uint32Array([pos, n, 0, 0]));
    const cs = new Float32Array(n * MC.MC_FRAME_COL);
    for (let c = 0; c < n; c++) {
      let th = fr(pos + c);
      for (let i = 0; i < cfg.ropeDims / 2; i++) { cs[c * MC.MC_FRAME_COL + i * 4] = fr(Math.cos(th)); cs[c * MC.MC_FRAME_COL + i * 4 + 1] = fr(Math.sin(th)); th = fr(th * eng._ropeTs); }
    }
    q.writeBuffer(B.cs, 0, cs);
    if (eng.qsa) qsaSetFrame(device, this.qsaS, pos, n);

    let at = 0;
    for (; at + SUBMIT_LAYERS < this.layers.length; at += SUBMIT_LAYERS) {
      const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
      for (const l of this.layers.slice(at, at + SUBMIT_LAYERS)) eng._encode(pass, l.ops, pos, n);
      pass.end();
      q.submit([enc.finish()]);
    }
    const enc = device.createCommandEncoder();
    let pass = enc.beginComputePass();
    for (const l of this.layers.slice(at)) eng._encode(pass, l.ops, pos, n);
    if (eng.hasHead) eng._encode(pass, this.headSite, pos, n);   // the last layer's combine, on every column
    if (head === "all") eng._encode(pass, this.headRest, pos, n);
    pass.end();
    const out = {}, HB = HD * 4;
    if (head === "last") {
      if (eng.hasHead) {   // the head on the last column: its normed residual into column 0, then one column
        if (n > 1) { enc.copyBufferToBuffer(B.xn, (n - 1) * HB, B.xnTmp, 0, HB); enc.copyBufferToBuffer(B.xnTmp, 0, B.xn, 0, HB); }
        pass = enc.beginComputePass();
        eng._encode(pass, this.headRest, pos + n - 1, 1);
        pass.end();
        enc.copyBufferToBuffer(B.am, 0, this.amStage, 0, 8);
        if (wantLogits) enc.copyBufferToBuffer(B.hl, 0, this.logitStage, 0, eng.dims.vocab * 4);
      } else enc.copyBufferToBuffer(B.R, (n - 1) * HB, this.hStage, 0, HB);
    }
    if (head === "all") enc.copyBufferToBuffer(B.am, 0, this.amStage, 0, n * 8);
    let hs = null;
    if (wantHidden) { hs = this._stage(this.maxCols * HB); enc.copyBufferToBuffer(B.R, 0, hs, 0, n * HB); }
    q.submit([enc.finish()]);
    eng.pos += n;
    if (head === "last") {
      if (eng.hasHead) {
        const a = await this._read(this.amStage, 8, Uint32Array);
        out.id = a[0];
        if (out.id >= eng.dims.vocab) throw new Error(`qwen4exp: no greedy id at position ${eng.pos - 1} (non-finite logits)`);
        if (wantLogits) out.logits = await this._read(this.logitStage, eng.dims.vocab * 4);
      } else out.hidden = await this._read(this.hStage, HB);
    }
    if (head === "all") { const a = await this._read(this.amStage, n * 8, Uint32Array); out.ids = Array.from({ length: n }, (_, c) => a[2 * c]); }
    if (wantHidden) out.hiddens = await this._read(hs, n * HB);
    if (head === "none" && !wantHidden) await q.onSubmittedWorkDone();
    return out;
  }

  // Prompt tokens in passes of maxCols (the last pass gets the head). Returns the last pass's result (as _pass).
  // opts.chunks: explicit pass sizes (tests), repeated until the ids run out; opts.onPass(result, offset, size).
  async prefill(ids, { chunks = null, onPass = null, ...opts } = {}) {
    let r = null, o = 0, k = 0;
    while (o < ids.length) {
      const sz = Math.min(chunks ? chunks[k++ % chunks.length] : this.maxCols, this.maxCols, ids.length - o);
      const end = o + sz >= ids.length;
      const res = await this._pass(ids.slice(o, o + sz), end ? opts : { ...opts, head: "none", hidden: opts.hidden?.subarray(o * this.cfg.hDim, (o + sz) * this.cfg.hDim) });
      if (onPass) onPass(res, o, sz);
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
