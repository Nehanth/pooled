// Qwen3.8-Flash-Next (llama.cpp arch "qwen4exp") on WebGPU: the decode path, weights resident, context <= 2048.
// Roadmap 33 (roadmap/33-qwen38-flash.md) PR plan item 4.
//
// A separate engine module, not a mode of Qwen35Engine: the residual is 4 streams wide (hyper-connections), every
// norm around the blocks is an HC mixer, layer 1 adds the n-gram (PLE) injection, the MoE has 512 experts / top-10
// and there is no output norm, so every one of Qwen35Engine's fused decode paths would need an HC branch. This file
// keeps the math in one readable place, one token at a time, with its own kernels (engine/wgsl/qwen4exp.js); the
// batched / prefill paths, MTP, offload and room plumbing (PR plan items 5-8) build on it.
//
// One token (forwardToken):
//   R = 4 copies of the token embedding                               (hc_init)
//   per layer: [PLE on its layer] -> HC mix -> DeltaNet or gated attention -> HC combine
//                                 -> HC mix -> MoE (top-10 of 512 + shared expert) -> HC combine
//   head: HC mix (output_hc_*, no inject) -> output.weight -> greedy id on the GPU (and the logits on request)
//
// The math is tests/reference/ref_q38flash.mjs's with f32 activations (its --act f32 mode). Attention is dense causal:
// QSA (PR 9) selects every cell below 2051 cached tokens, so dense is exact up to the 2048 cap here. The hook for
// QSA is _attnOps: a selection pass between attn_prep and attn_dec that gives attn_dec a cell list.
import { qwen4expConfig } from "./gguf.js";
import { NgramState } from "./ngram.js";
import * as W from "./wgsl/qwen4exp.js";

const U = GPUBufferUsage;
const fr = Math.fround;

export class Qwen4ExpEngine {
  static _ids = 0;
  static async create(opts) { const e = new Qwen4ExpEngine(); await e._init(opts); return e; }

  // weights: qwen4expWeights' structure (engine/gguf.js), each matrix with entry.gpu (streamed) or its CPU bytes.
  // ngram: an NgramTable (engine/ngram.js) for the PLE layer's rows; required when layerRange holds a PLE layer.
  // layerRange [lo, hi): the trunk layers this engine runs; hasEmbed / hasHead as Qwen35Engine.
  async _init({ device, meta, weights, layerRange = null, hasEmbed = true, hasHead = true, maxSeq = 2048, ngram = null }) {
    const cfg = this.cfg = qwen4expConfig(meta);
    this.device = device;
    const [lo, hi] = layerRange || [0, cfg.L];
    Object.assign(this, { lo, hi, hasEmbed, hasHead, pos: 0 });
    if (maxSeq > 2048) throw new Error("qwen4exp: the dense decode path holds at most 2048 positions (QSA is PR 9)");
    this.maxSeq = maxSeq;
    const d = cfg.dim, hc = cfg.hc, HD = cfg.hDim, ss = cfg.ssm;
    this.dims = { dim: d, hDim: HD, vocab: hasHead ? weights.head.shape[0] : 0 };
    const pleLayers = (cfg.ple?.layers || []).filter((l) => l >= lo && l < hi);
    if (pleLayers.length && !ngram) throw new Error("qwen4exp: layers " + pleLayers + " need the n-gram table (ngram: an NgramTable)");
    this.ngram = ngram;
    this.ngs = ngram ? new NgramState(ngram.cfg) : null;
    this.embed = weights.embed || null;

    // ---- pipelines
    const pipe = async (code, label) => device.createComputePipelineAsync({ layout: "auto", label,
      compute: { module: device.createShaderModule({ code, label }), entryPoint: "main" } });
    const kinds = ["q4", "q8", "f32"];
    const P = this.P = {};
    await Promise.all([
      ...kinds.map(async (k) => { P["gemv_" + k] = await pipe(W.gemvWGSL(k), "gemv_" + k); }),
      ...["q4", "q8"].map(async (k) => { P["gu_" + k] = await pipe(W.moeGuWGSL(k), "moe_gu_" + k); P["dn_" + k] = await pipe(W.moeDnWGSL(k), "moe_dn_" + k); }),
      ...Object.entries({ route: W.MOE_ROUTE_WGSL, hcNorm: W.HC_NORM_WGSL, hcMix: W.HC_MIX_WGSL, hcComb: W.HC_COMBINE_WGSL,
        dnConv: W.DN_CONV_WGSL, dnDelta: W.DN_DELTA_WGSL,
        attnPrep: W.ATTN_PREP_WGSL, attnDec: W.ATTN_DEC_WGSL, ple: W.PLE_WGSL, am1: W.ARGMAX1_WGSL, am2: W.ARGMAX2_WGSL })
        .map(async ([k, code]) => { P[k] = await pipe(code, k); }),
    ]);

    // ---- scratch (shared by every layer: one token's ops run in order)
    const buf = (n, extra = 0) => device.createBuffer({ size: Math.max(16, Math.ceil(n * 4 / 16) * 16), usage: U.STORAGE | U.COPY_SRC | U.COPY_DST | extra });
    const convDim = 2 * ss.nKHeads * ss.stateSize + ss.inner, keyDim = ss.nKHeads * ss.stateSize;
    const K = cfg.K, nH = cfg.nHeads, nKV = cfg.nKVHeads, hd = cfg.headDim;
    this.dn = { convDim, keyDim };
    const B = this.B = {
      R: buf(HD), xn: buf(HD), lo: buf(cfg.hcRank), gate: buf(HD), mixed: buf(d), inj: buf(hc), out: buf(d),
      qkvz: buf(convDim + ss.inner), ba: buf(2 * ss.nVHeads), co: buf(convDim), o: buf(ss.inner), go: buf(ss.inner),
      qkvA: buf(2 * nH * hd + 2 * nKV * hd), qn: buf(nH * hd), ao: buf(nH * hd),
      logits: buf(cfg.nExp + 1), ids: buf(K), wts: buf(K), h: buf(K * cfg.expFF), moe: buf(d),
      shIds: buf(1), hs: buf(cfg.shFF), sh: buf(d),
      frame: device.createBuffer({ size: W.FRAME_FLOATS * 4, usage: U.UNIFORM | U.COPY_DST }),
    };
    if (pleLayers.length) Object.assign(B, { emb: buf(d), pk: buf(HD), pv: buf(d), dbg: buf(2 * HD + hc) });
    device.queue.writeBuffer(B.shIds, 0, new Uint32Array(4));

    // ---- weights to the GPU (entry.gpu when the loader streamed it; f32 tensors are uploaded here)
    const wq = (e, what) => {
      if (!e) throw new Error("qwen4exp: missing weight " + what);
      if (e.kind === "f32") {
        if (!e.gpuF32) {
          const src = e.data;
          const b = device.createBuffer({ size: Math.ceil(src.byteLength / 16) * 16, usage: U.STORAGE | U.COPY_DST, mappedAtCreation: true });
          new Float32Array(b.getMappedRange()).set(src);
          b.unmap();
          e.gpuF32 = b;
        }
        return { kind: "f32", buf: e.gpuF32, n: e.data.length };
      }
      if (!e.gpu) {
        const mk = (data) => {
          const u8 = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
          const b = device.createBuffer({ size: Math.ceil(u8.byteLength / 4) * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC, mappedAtCreation: true });
          new Uint8Array(b.getMappedRange()).set(u8);
          b.unmap();
          return b;
        };
        e.gpu = { kind: e.kind, qs: mk(e.qs), sc: mk(e.scales) };
      }
      return { kind: e.kind, qs: e.gpu.qs, sc: e.gpu.sc, shape: e.shape };
    };

    // f32 tensors end to end, as one entry (kept per pair: the cache key is the entries themselves)
    const cats = new Map();
    const cat = (...es) => {
      const k = es.map((e) => (e.__id ||= ++Qwen4ExpEngine._ids)).join(",");
      if (!cats.has(k)) {
        const data = new Float32Array(es.reduce((n, e) => n + e.data.length, 0));
        let o = 0;
        for (const e of es) { data.set(e.data, o); o += e.data.length; }
        cats.set(k, { kind: "f32", data });
      }
      return cats.get(k);
    };

    // ---- op builders: an op is { pipe, bg, wg: [x, y, z], tap?: [name, buffer, count, "f32" | "u32"] }
    const uni = (fields) => {
      const n = Math.max(4, Math.ceil(fields.length / 4) * 4), ab = new ArrayBuffer(n * 4), dv = new DataView(ab);
      fields.forEach((v, i) => (typeof v === "number" ? dv.setUint32(i * 4, v, true) : dv.setFloat32(i * 4, v.f, true)));
      const b = device.createBuffer({ size: ab.byteLength, usage: U.UNIFORM | U.COPY_DST, mappedAtCreation: true });
      new Uint8Array(b.getMappedRange()).set(new Uint8Array(ab));
      b.unmap();
      return b;
    };
    // list: buffers, or { buffer, offset, size } views (offsets multiples of 256 bytes)
    const bg = (p, list) => device.createBindGroup({ layout: p.getBindGroupLayout(0),
      entries: list.map((b, i) => (b ? { binding: i, resource: b.buffer ? b : { buffer: b } } : null)).filter(Boolean) });
    const view = (buffer, off, n) => ({ buffer, offset: off * 4, size: n * 4 });
    // Quantized matrices of one kind and width stacked into one (rows end to end), so projections that read the
    // same input run as one GEMV. The engine owns its weights: the parts' GPU buffers are freed. null when the
    // kinds or widths differ.
    const fuseRows = (...es) => {
      const ms = es.map((e) => wq(e));
      if (ms.some((m) => m.kind === "f32" || m.kind !== ms[0].kind || m.shape[1] !== ms[0].shape[1])) return null;
      const cols = ms[0].shape[1], nb = cols / 32, qb = ms[0].kind === "q4" ? 16 : 32;
      const rows = ms.reduce((n, m) => n + m.shape[0], 0);
      if (ms.some((m) => (m.shape[0] * nb) % 2)) return null;   // scales are packed two per word
      const qs = device.createBuffer({ size: rows * nb * qb, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
      const sc = device.createBuffer({ size: rows * nb * 2, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
      const enc = device.createCommandEncoder();
      let r = 0;
      for (const m of ms) {
        enc.copyBufferToBuffer(m.qs, 0, qs, r * nb * qb, m.shape[0] * nb * qb);
        enc.copyBufferToBuffer(m.sc, 0, sc, r * nb * 2, m.shape[0] * nb * 2);
        r += m.shape[0];
      }
      device.queue.submit([enc.finish()]);
      for (const e of es) { e.gpu.qs.destroy(); e.gpu.sc.destroy(); e.gpu = { kind: e.kind, qs, sc, fused: true }; }
      return { kind: ms[0].kind, qs, sc, shape: [rows, cols] };
    };
    const f = (v) => ({ f: v });
    const eps = cfg.eps;
    // y[yOff..] = mat x (mode 1: silu(acc * scale)); dOut / dIn from the weight's shape unless given
    // f32 entries carry no shape: pass dIn (dOut is then their length / dIn)
    const gemv = (M, x, y, { dIn = M.shape?.[1], dOut = M.shape?.[0] ?? M.n / dIn, rowOff = 0, xOff = 0, yOff = 0, mode = 0, scale = 0, tap } = {}) => {
      if (!(dIn > 0 && dOut > 0 && Number.isInteger(dOut))) throw new Error(`qwen4exp: gemv shape ${dOut} x ${dIn}`);
      const p = P["gemv_" + M.kind];
      const u = uni([dOut, dIn, rowOff, xOff / 4, yOff, mode, f(scale), 0]);
      const g = M.kind === "f32" ? bg(p, [M.buf, null, x, y, u]) : bg(p, [M.qs, M.sc, x, y, u]);
      const nwg = Math.ceil(dOut / 8);
      return { pipe: p, bg: g, wg: [Math.min(nwg, 32768), Math.ceil(nwg / 32768), 1], tap };
    };
    const ewg = (n) => [Math.ceil(n / 256), 1, 1];
    // HC mixer: xn = per-stream rmsnorm(R) * norm; lo = silu(down xn / hc); gate = up lo; mixed = mean_c xn * sigmoid(gate);
    // inj = inject xn. tag: "-i" for the dump names; inject null for the head
    const hcMix = (hw, tag, nm, inject = true) => {
      const norm = wq(hw.norm, "hc norm"), ops = [];
      ops.push({ pipe: P.hcNorm, bg: bg(P.hcNorm, [B.R, norm.buf, B.xn, uni([d, f(eps), 0, 0])]), wg: [hc, 1, 1] });
      ops.push(gemv(wq(hw.down, "hc down"), B.xn, B.lo, { dIn: HD, mode: 1, scale: 1 / hc }));
      ops.push(gemv(wq(hw.up, "hc up"), B.lo, B.gate, { dIn: cfg.hcRank, tap: ["hc_gate" + tag, B.gate, HD] }));
      ops.push({ pipe: P.hcMix, bg: bg(P.hcMix, [B.xn, B.gate, B.mixed, uni([d, hc, 0, 0])]), wg: ewg(d), tap: [nm, B.mixed, d] });
      if (inject) ops.push(gemv(wq(hw.inject, "hc inject"), B.xn, B.inj, { dIn: HD, tap: ["hc_inject" + tag, B.inj, hc] }));
      return ops;
    };
    const hcComb = (a, b, name) => ({ pipe: P.hcComb, bg: bg(P.hcComb, [B.R, a, b || a, B.inj, uni([d, hc, b ? 1 : 0, 0])]), wg: ewg(HD), tap: [name, B.R, HD] });

    // ---- per layer state and ops
    const frameU = B.frame;
    this.layers = [];
    for (let i = lo; i < hi; i++) {
      const L = weights.layers[i - lo], isQ = cfg.qsa[i], st = {}, attnOps = [], ffnOps = [];
      if (!!L.isFull !== !!isQ) throw new Error(`qwen4exp: layer ${i} type does not match the config`);
      // PLE (n-gram) injection, before the layer's first mixer
      if (pleLayers.includes(i)) {
        const pc = L.ple, K1 = pc.conv.data.length / HD;
        const slots = (K1 - 1) * cfg.ple.ngram + 1;
        st.ring = buf(slots * HD);
        attnOps.push(gemv(wq(pc.key, "ple key"), B.emb, B.pk, { tap: ["ple_embd", B.emb, d] }));
        attnOps.push(gemv(wq(pc.value, "ple value"), B.emb, B.pv));
        // the PLE norms and the conv kernel in one buffer (8 storage buffers per stage is WebGPU's default limit)
        const pw = new Float32Array(3 * HD + HD * K1);
        [pc.normKey, pc.normQuery, pc.normConv, pc.conv].forEach((e, j) => pw.set(e.data, j * HD));
        attnOps.push({ pipe: P.ple, bg: bg(P.ple, [B.R, B.pk, B.pv, wq({ kind: "f32", data: pw, shape: [pw.length] }).buf,
          st.ring, B.dbg, uni([d, K1, cfg.ple.ngram, slots, f(eps), hc, 0, 0]), frameU]), wg: [hc, 1, 1],
          taps: [[`ple_gate-${i}`, B.dbg, hc, "f32", 2 * HD], [`ple_gated_value-${i}`, B.dbg, HD], [`ple_conv_out-${i}`, B.dbg, HD, "f32", HD]] });
      }
      attnOps.push(...hcMix(L.hcAttn, `-${i}`, `hc_mixed-${i}`));
      if (isQ) attnOps.push(...this._attnOps(L, i, st, { gemv, wq, bg, uni, f, view, fuseRows }));
      else {
        st.conv = buf(convDim * (ss.convK - 1)); st.S = buf(ss.nVHeads * ss.stateSize * ss.stateSize);
        // [qkv | z]: one GEMV when both have one type (Q4_0 in every layer of the bartowski file)
        const zTap = { tap: [`z-${i}`, B.qkvz, ss.inner, "f32", convDim] }, fz = fuseRows(L.wqkv, L.wz);
        if (fz) attnOps.push(gemv(fz, B.mixed, B.qkvz, zTap));
        else attnOps.push(gemv(wq(L.wqkv, "qkv"), B.mixed, B.qkvz), gemv(wq(L.wz, "z"), B.mixed, B.qkvz, { yOff: convDim, ...zTap }));
        // beta and alpha as one GEMV, dt_bias and A as one buffer (cat: CPU copies of the f32 tensors)
        attnOps.push(gemv(wq(cat(L.wBeta, L.wAlpha)), B.mixed, B.ba, { dIn: d }));
        attnOps.push({ pipe: P.dnConv, bg: bg(P.dnConv, [view(B.qkvz, 0, convDim), st.conv, wq(L.conv, "conv").buf, B.co, uni([ss.convK, 2 * ss.nKHeads, ss.stateSize, f(eps)])]),
          wg: [convDim / ss.stateSize, 1, 1] });
        attnOps.push({ pipe: P.dnDelta, bg: bg(P.dnDelta, [B.co, st.S, B.ba, wq(cat(L.dtBias, L.ssmA)).buf, view(B.qkvz, convDim, ss.inner), wq(L.ssmNorm).buf, B.o, B.go,
          uni([ss.nKHeads, ss.stateSize, keyDim, ss.nVHeads, f(eps), 0, 0, 0])]), wg: [ss.nVHeads, 1, 1],
          taps: [[`attn_output-${i}`, B.o, ss.inner], [`final_output-${i}`, B.go, ss.inner]] });
        attnOps.push(gemv(wq(L.wOut, "ssm_out"), B.go, B.out, { tap: [`linear_attn_out-${i}`, B.out, d] }));
      }
      attnOps.push(hcComb(B.out, null, `hc_combine-${i}`));
      // MoE: router, top-K, routed experts, shared expert (gated by sigmoid(shRouter . x))
      ffnOps.push(...hcMix(L.hcFfn, `-${i}`, `hc_mixed-${i}`));
      // the router and the shared expert's gate as one GEMV: logits[nExp] is the shared gate's
      ffnOps.push(gemv(wq(cat(L.router, L.shRouter)), B.mixed, B.logits, { dIn: d }));
      ffnOps.push({ pipe: P.route, bg: bg(P.route, [B.logits, B.ids, B.wts, uni([cfg.nExp, K, 0, 0])]), wg: [1, 1, 1],
        taps: [[`ffn_moe_topk-${i}`, B.ids, K, "u32"], [`ffn_moe_weights_norm-${i}`, B.wts, K]] });
      const eg = wq(L.expGate, "experts gate"), eu = wq(L.expUp, "experts up"), ed = wq(L.expDown, "experts down");
      if (eg.kind !== eu.kind) throw new Error(`qwen4exp: layer ${i} gate/up experts differ in type`);
      const ff = cfg.expFF;
      ffnOps.push({ pipe: P["gu_" + eg.kind], bg: bg(P["gu_" + eg.kind], [eg.qs, eg.sc, eu.qs, eu.sc, B.mixed, B.h, B.ids, uni([ff, d, K, 0, 0, 0, 0, 0])]), wg: [Math.ceil(ff / 8), K, 1] });
      ffnOps.push({ pipe: P["dn_" + ed.kind], bg: bg(P["dn_" + ed.kind], [ed.qs, ed.sc, B.h, B.moe, B.ids, B.wts, uni([d, ff, K, 0, 0, 0, 0, 0])]), wg: [Math.ceil(d / 8), 1, 1],
        tap: [`ffn_moe_out-${i}`, B.moe, d] });
      const sgw = wq(L.shGate, "shared gate"), suw = wq(L.shUp, "shared up"), sdw = wq(L.shDown, "shared down"), sff = cfg.shFF;
      if (sgw.kind !== suw.kind) throw new Error(`qwen4exp: layer ${i} shared gate/up differ in type`);
      ffnOps.push({ pipe: P["gu_" + sgw.kind], bg: bg(P["gu_" + sgw.kind], [sgw.qs, sgw.sc, suw.qs, suw.sc, B.mixed, B.hs, B.shIds, uni([sff, d, 1, 0, 0, 0, 0, 0])]), wg: [Math.ceil(sff / 8), 1, 1] });
      ffnOps.push({ pipe: P["dn_" + sdw.kind], bg: bg(P["dn_" + sdw.kind], [sdw.qs, sdw.sc, B.hs, B.sh, B.shIds, B.logits, uni([d, sff, 1, 1, cfg.nExp, 0, 0, 0])]), wg: [Math.ceil(d / 8), 1, 1],
        tap: [`ffn_shexp_gated-${i}`, B.sh, d] });
      ffnOps.push(hcComb(B.moe, B.sh, `l_last-${i}`));
      this.layers.push({ i, isQ, st, attnOps, ffnOps });
    }

    // ---- head: the last HC mixer (no inject) stands in for the output norm, then the LM head and a GPU argmax
    if (hasHead) {
      const V = this.dims.vocab, per = 1024, nPart = Math.ceil(V / per);
      Object.assign(B, { hl: buf(V), part: buf(nPart * 2), am: buf(4) });
      this.headOps = [...hcMix(weights.headHC, "", "result_norm", false),
        gemv(wq(weights.head, "output"), B.mixed, B.hl, { tap: ["result_output", B.hl, V] }),
        { pipe: P.am1, bg: bg(P.am1, [B.hl, B.part, uni([V, per, 0, 0])]), wg: [nPart, 1, 1] },
        { pipe: P.am2, bg: bg(P.am2, [B.part, B.am, uni([nPart, 0, 0, 0])]), wg: [1, 1, 1] }];
      this.amStage = device.createBuffer({ size: 16, usage: U.MAP_READ | U.COPY_DST });
      this.logitStage = device.createBuffer({ size: Math.ceil(V * 4 / 16) * 16, usage: U.MAP_READ | U.COPY_DST });
    }
    this.hStage = device.createBuffer({ size: HD * 4, usage: U.MAP_READ | U.COPY_DST });
    this._ropeTs = fr(Math.pow(cfg.ropeTheta, -2 / cfg.ropeDims));
    this.reset();
  }

  // Gated full attention (dense causal). The QSA hook (PR 9): an indexer pass would go between attn_prep and
  // attn_dec and hand attn_dec a selected-cell list; below 2051 cached tokens it selects every cell.
  _attnOps(L, i, st, { gemv, wq, bg, uni, f, view, fuseRows }) {
    const { cfg, B, P, device } = this, nH = cfg.nHeads, nKV = cfg.nKVHeads, hd = cfg.headDim;
    if (hd !== 256) throw new Error("qwen4exp: attn kernels assume head dim 256");
    const sz = this.maxSeq * nKV * hd * 4;
    st.kc = device.createBuffer({ size: sz, usage: U.STORAGE | U.COPY_DST });
    st.vc = device.createBuffer({ size: sz, usage: U.STORAGE | U.COPY_DST });
    const ap = uni([nH, nKV, hd, cfg.ropeDims, f(cfg.eps), 0, 0, 0]);
    const nQ = 2 * nH * hd, nK = nKV * hd, qf = view(B.qkvA, 0, nQ), fq = fuseRows(L.wq, L.wk, L.wv);
    return [
      // [q|gate, k, v] into one buffer: one GEMV when the three share a type (Q6_K and Q8_0 both load as q8)
      ...(fq ? [gemv(fq, B.mixed, B.qkvA)] : [gemv(wq(L.wq, "attn_q"), B.mixed, B.qkvA), gemv(wq(L.wk, "attn_k"), B.mixed, B.qkvA, { yOff: nQ }),
        gemv(wq(L.wv, "attn_v"), B.mixed, B.qkvA, { yOff: nQ + nK })]),
      { pipe: P.attnPrep, bg: bg(P.attnPrep, [qf, view(B.qkvA, nQ, nK), view(B.qkvA, nQ + nK, nK), B.qn, st.kc, st.vc, wq(L.qNorm).buf, wq(L.kNorm).buf, ap, B.frame]), wg: [nH + nKV, 1, 1] },
      { pipe: P.attnDec, bg: bg(P.attnDec, [B.qn, st.kc, st.vc, qf, B.ao, ap, B.frame]), wg: [nH, 1, 1], tap: [`attn_gated-${i}`, B.ao, nH * hd] },
      gemv(wq(L.wo, "attn_output"), B.ao, B.out, { tap: [`attn_output-${i}`, B.out, cfg.dim] }),
    ];
  }

  reset() {
    this.pos = 0;
    this.ngs?.reset();
    const enc = this.device.createCommandEncoder();
    for (const l of this.layers) for (const b of [l.st.conv, l.st.S, l.st.ring]) if (b) enc.clearBuffer(b);
    this.device.queue.submit([enc.finish()]);
  }

  // The token's embedding row (Q4_0 on the CPU, dequantized as dequantF32 does)
  _embedRow(id) {
    const e = this.embed, d = this.cfg.dim, nb = d / 32, out = new Float32Array(d);
    if (e.kind === "f32") { out.set(e.data.subarray(id * d, (id + 1) * d)); return out; }
    const sc16 = new Uint16Array(e.scales.buffer, e.scales.byteOffset, e.scales.byteLength / 2);
    for (let b = 0; b < nb; b++) {
      const blk = id * nb + b, s = f16(sc16[blk]);
      if (e.kind === "q4") {
        const o = blk * 16;
        for (let j = 0; j < 16; j++) { const q = e.qs[o + j]; out[b * 32 + j] = s * ((q & 15) - 8); out[b * 32 + j + 16] = s * ((q >> 4) - 8); }
      } else {
        const o = blk * 32;
        for (let j = 0; j < 32; j++) { const q = e.qs[o + j]; out[b * 32 + j] = s * (q > 127 ? q - 256 : q); }
      }
    }
    return out;
  }

  // Everything one token needs before its layers run: the residual (from the embedding), the n-gram rows, the frame.
  async _begin(id) {
    if (this.pos >= this.maxSeq) throw new Error(`qwen4exp: context full (${this.maxSeq})`);
    const q = this.device.queue, { cfg, B } = this;
    if (this.hasEmbed) {
      const e = this._embedRow(id), R = new Float32Array(cfg.hDim);
      for (let c = 0; c < cfg.hc; c++) R.set(e, c * cfg.dim);
      q.writeBuffer(B.R, 0, R);
    }
    if (B.emb) {
      const rows = this.lastRows = this.ngs.push([id]);
      const emb = await this.ngram.rows(rows);
      q.writeBuffer(B.emb, 0, emb);
    } else this.ngs?.push([id]);
    const F = new Float32Array(W.FRAME_FLOATS), Fu = new Uint32Array(F.buffer);
    Fu[0] = this.pos; Fu[1] = this.pos + 1;
    let th = fr(this.pos);
    for (let i = 0; i < cfg.ropeDims / 2; i++) { F[4 + i * 4] = fr(Math.cos(th)); F[5 + i * 4] = fr(Math.sin(th)); th = fr(th * this._ropeTs); }
    q.writeBuffer(B.frame, 0, F);
  }

  _encode(pass, ops) { for (const o of ops) { pass.setPipeline(o.pipe); pass.setBindGroup(0, o.bg); pass.dispatchWorkgroups(...o.wg); } }

  // One decode step. -> { id (greedy, when the engine has the head), logits (wantLogits) }
  async forwardToken(id, { wantLogits = false } = {}) {
    await this._begin(id);
    const enc = this.device.createCommandEncoder(), pass = enc.beginComputePass();
    for (const l of this.layers) { this._encode(pass, l.attnOps); this._encode(pass, l.ffnOps); }
    if (this.hasHead) this._encode(pass, this.headOps);
    pass.end();
    if (this.hasHead) {
      enc.copyBufferToBuffer(this.B.am, 0, this.amStage, 0, 16);
      if (wantLogits) enc.copyBufferToBuffer(this.B.hl, 0, this.logitStage, 0, this.dims.vocab * 4);
    } else enc.copyBufferToBuffer(this.B.R, 0, this.hStage, 0, this.cfg.hDim * 4);
    this.device.queue.submit([enc.finish()]);
    this.pos++;
    const out = {};
    if (this.hasHead) {
      await this.amStage.mapAsync(GPUMapMode.READ);
      out.id = new Uint32Array(this.amStage.getMappedRange())[0];
      this.amStage.unmap();
      if (wantLogits) {
        await this.logitStage.mapAsync(GPUMapMode.READ);
        out.logits = new Float32Array(this.logitStage.getMappedRange(), 0, this.dims.vocab).slice();
        this.logitStage.unmap();
      }
    } else {
      await this.hStage.mapAsync(GPUMapMode.READ);
      out.hidden = new Float32Array(this.hStage.getMappedRange()).slice();
      this.hStage.unmap();
    }
    return out;
  }

  // Prompt tokens one at a time (the batched prefill is PR plan item 5). Returns the last step's result.
  async prefillTokens(ids, opts) { let r = null; for (const t of ids) r = await this.forwardToken(t, opts); return r; }

  // Per-layer check (tests/test_q38flash.js): one token, each half layer submitted on its own with every tapped
  // tensor read back. force(name) -> Float32Array | null: the residual to start a half layer from ("in-<i>" before
  // layer i's first mixer, "mid-<i>" before its MoE mixer; the dumps' hc_init / l_last-(i-1) / hc_combine-i), for a
  // teacher-forced run. -> Map of "<name>.<k>" (k counts repeats within the token, as the dumps do) -> typed array.
  async debugToken(id, force = () => null) {
    await this._begin(id);
    const out = new Map(), seen = new Map(), dev = this.device;
    const run = async (ops, Rin) => {
      if (Rin) dev.queue.writeBuffer(this.B.R, 0, Rin);
      const enc = dev.createCommandEncoder(), reads = [];
      for (const o of ops) {
        const pass = enc.beginComputePass();
        this._encode(pass, [o]);
        pass.end();
        for (const [name, b, n, type = "f32", off = 0] of o.taps || (o.tap ? [o.tap] : [])) {
          const s = dev.createBuffer({ size: n * 4, usage: U.MAP_READ | U.COPY_DST });
          enc.copyBufferToBuffer(b, off * 4, s, 0, n * 4);
          reads.push([name, s, n, type]);
        }
      }
      dev.queue.submit([enc.finish()]);
      for (const [name, s, n, type] of reads) {
        await s.mapAsync(GPUMapMode.READ);
        const a = type === "u32" ? new Int32Array(s.getMappedRange().slice(0, n * 4)) : new Float32Array(s.getMappedRange().slice(0, n * 4));
        s.unmap(); s.destroy();
        const k = seen.get(name) || 0; seen.set(name, k + 1);
        out.set(`${name}.${k}`, a);
      }
    };
    for (const l of this.layers) {
      await run(l.attnOps, force(`in-${l.i}`));
      await run(l.ffnOps, force(`mid-${l.i}`));
    }
    if (this.hasHead) await run(this.headOps, force("head"));
    this.pos++;
    return out;
  }
}

function f16(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
}
