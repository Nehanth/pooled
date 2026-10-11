// Qwen3.8-Flash-Next MTP speculative decoding on the Flash decode engine (engine/qwen4exp.js). Roadmap 33
// (roadmap/33-qwen38-flash.md) PR plan item 6.
//
// The draft block ships in its own GGUF (mtp-Qwen3.8-Flash-Next-*.gguf, blk.48 + blk.48.nextn.*), loaded by
// qwen4expWeights' mtp option (engine/gguf.js). One draft step, as llama.cpp's graph_mtp (src/models/qwen4exp.cpp):
//   x_s   = [enorm(embed(t)) ; hnorm(h_s)]  per HC stream s       (h: a 4-stream residual, see below)
//   R_s   = eh_proj x_s                                            (the block's own 4-stream residual)
//   R     = one full-attention layer (HC mix -> gated attention -> HC combine -> HC mix -> MoE 512/top-10 -> HC combine)
//   draft = argmax(head(HC mix with nextn.hc_head_*, no inject))
// h is the trunk's residual after its last layer and before its head mixer (llama.cpp's h_nextn) for the first draft
// of a step, and the block's own output residual R for each draft after it (chained drafting). The block's KV cache
// row p pairs the token at p with the hidden of p - 1 (zeros at p = 0), as llama.cpp's draft-mtp process() does.
//
// Which LM head (llama.cpp #29761 loads the MTP file as a model of its own, so it uses that file's token_embd and
// output, both Q4_0): the MTP file's output.weight is the trunk's (same tensor, Q4_0 there and Q6_K in the trunk), and
// its token_embd is byte-for-byte the trunk's (tests/test_q38flash_mtp.js checks both). Here the default is the
// trunk's head (ownHead: false): it is already resident (no extra 358 MB), it reads the more precise Q6_K copy, and
// drafts only decide the speed, never the output. ownHead: true uses the file's own Q4_0 head, as llama.cpp does.
//
// Verification: specStep hands [t, ...drafts] to a verifier with Qwen4ExpBatch's verify / accept contract
// (engine/qwen4exp_batch.js, PR #348: verify(ids, { wantHidden }) -> { ids, hiddens }, accept(n)); BatchVerifier wraps
// one. Until that lands on this branch, DecodeVerifier runs the columns as decode steps: the same ids, but no speedup.
// Either way the output is exactly plain greedy decoding's: every emitted token is the trunk's argmax.
// On a rejection, accept(n) restores the recurrent state (DeltaNet conv + S, the PLE conv ring), rolls the n-gram
// predecessors back and re-runs the n kept columns; the draft block's own KV rows past the accepted position are
// simply rewritten by later steps (attention reads rows 0 .. pos only).
import { MTP_IN_WGSL } from "./wgsl/qwen4exp_mtp.js";

const U = GPUBufferUsage;

export class Qwen4ExpMtp {
  // eng: a Qwen4ExpEngine with the embedding and the head. mtp: qwen4expWeights' out.mtp. head: the LM head entry to
  // use (the trunk's weights.head by default; out.mtp.head for the file's own). maxDrafts: the most drafts per step.
  static async create(eng, mtp, { head, maxDrafts = 7 } = {}) { const m = new Qwen4ExpMtp(); await m._init(eng, mtp, { head, maxDrafts }); return m; }

  async _init(eng, mtp, { head, maxDrafts }) {
    if (!eng.hasEmbed || !eng.hasHead) throw new Error("qwen4exp mtp: the engine needs the embedding and the head");
    if (!eng._build) throw new Error("qwen4exp mtp: the engine has no op builders (engine/qwen4exp.js _build)");
    if (!mtp?.layer || !mtp.ehProj || !head) throw new Error("qwen4exp mtp: missing the MTP block or the head");
    const { cfg, device, B } = eng, d = cfg.dim, HD = cfg.hDim, hc = cfg.hc;
    const { gemv, wq, bg, uni, f, buf, hcMix, layerOps } = eng._build;
    Object.assign(this, { eng, device, cfg, maxDrafts });
    this.stats = { steps: 0, drafts: 0, accepted: 0 };
    const pIn = await device.createComputePipelineAsync({ layout: "auto", label: "mtp_in",
      compute: { module: device.createShaderModule({ code: MTP_IN_WGSL, label: "mtp_in" }), entryPoint: "main" } });
    // e: the token's embedding row; hT: the trunk hidden the next first draft reads; hC: the block's last output
    // (what a chained draft reads); x: the eh_proj input, 2d per stream
    this.e = buf(d); this.hT = buf(HD); this.hC = buf(HD); this.x = buf(2 * d * hc);
    const ew = wq(mtp.enorm, "nextn enorm").buf, hw = wq(mtp.hnorm, "nextn hnorm").buf, eh = wq(mtp.ehProj, "nextn eh_proj");
    if (eh.shape[0] !== d || eh.shape[1] !== 2 * d) throw new Error(`qwen4exp mtp: eh_proj is ${eh.shape}, expected ${d} x ${2 * d}`);
    const inOps = (h) => [{ pipe: pIn, bg: bg(pIn, [this.e, h, ew, hw, this.x, uni([d, f(cfg.eps), 0, 0])]), wg: [hc, 1, 1] },
      ...Array.from({ length: hc }, (_, s) => gemv(eh, this.x, B.R, { xOff: s * 2 * d, yOff: s * d, tap: s === hc - 1 ? ["mtp_eh_proj", B.R, HD] : undefined }))];
    this.inTrunk = inOps(this.hT);
    this.inChain = inOps(this.hC);
    // the block: a full-attention layer (its own KV cache, eng.maxSeq rows), built exactly as a trunk layer
    this.layer = layerOps(mtp.layer, cfg.L, true);
    // the draft head: the block's head mixer, the LM head, the engine's two-stage argmax into B.am
    const hw2 = wq(head, "mtp head"), V = hw2.shape[0], per = 1024, nPart = Math.ceil(V / per);
    if (V !== eng.dims.vocab) throw new Error(`qwen4exp mtp: head of ${V} rows, the engine's has ${eng.dims.vocab}`);
    this.headOps = [...hcMix(mtp.hcHead, "-mtp", "mtp_result_norm", false),
      gemv(hw2, B.mixed, B.hl, { tap: ["mtp_result_output", B.hl, V] }),
      { pipe: eng.P.am1, bg: bg(eng.P.am1, [B.hl, B.part, uni([V, per, 0, 0])]), wg: [nPart, 1, 1] },
      { pipe: eng.P.am2, bg: bg(eng.P.am2, [B.part, B.am, uni([nPart, 0, 0, 0])]), wg: [1, 1, 1] }];
    this.stage = device.createBuffer({ size: 16, usage: U.MAP_READ | U.COPY_DST });
    this.reset();
  }

  // a new sequence: the hidden before position 0 is zeros (llama.cpp's pending_h starts zeroed)
  reset() {
    const enc = this.device.createCommandEncoder();
    enc.clearBuffer(this.hT); enc.clearBuffer(this.hC);
    this.device.queue.submit([enc.finish()]);
  }

  // hT <- the trunk's residual (B.R right after a trunk step, before anything else runs on the engine's scratch)
  adoptTrunk(enc = null) {
    const own = !enc;
    enc ||= this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.eng.B.R, 0, this.hT, 0, this.cfg.hDim * 4);
    if (own) this.device.queue.submit([enc.finish()]);
  }

  // One run of the block for token `tok` at position `pos` (writes its KV row pos). from: "trunk" reads hT, "chain"
  // reads hC (the block's previous output). hFrom: { buffer, offset } copied into hT first (a verify column's
  // hidden). head: also run the draft head -> the drafted id (else null). The block's output residual lands in hC.
  // Uses the engine's scratch, B.R included: run it between trunk steps, never inside one.
  async run(tok, pos, { from = "trunk", hFrom = null, head = false } = {}) {
    const { eng, device, cfg } = this;
    if (pos >= eng.maxSeq) throw new Error(`qwen4exp mtp: position ${pos} past the context (${eng.maxSeq})`);
    device.queue.writeBuffer(this.e, 0, eng._embedRow(tok));
    eng._writeFrame(pos);
    const enc = device.createCommandEncoder();
    if (hFrom) enc.copyBufferToBuffer(hFrom.buffer, hFrom.offset || 0, this.hT, 0, cfg.hDim * 4);
    let pass = enc.beginComputePass();
    eng._encode(pass, from === "chain" ? this.inChain : this.inTrunk);
    eng._encode(pass, this.layer.attnOps);
    eng._encode(pass, this.layer.ffnOps);
    pass.end();
    enc.copyBufferToBuffer(eng.B.R, 0, this.hC, 0, cfg.hDim * 4);
    if (head) {
      pass = enc.beginComputePass();
      eng._encode(pass, this.headOps);
      pass.end();
      enc.copyBufferToBuffer(eng.B.am, 0, this.stage, 0, 16);
    }
    device.queue.submit([enc.finish()]);
    if (!head) return null;
    await this.stage.mapAsync(GPUMapMode.READ);
    const id = new Uint32Array(this.stage.getMappedRange())[0];
    this.stage.unmap();
    return id;
  }

  // Prompt tokens through the trunk one at a time, with the block's KV row for each (row p: token p, hidden p - 1).
  // -> the trunk's last result ({ id }: the first generated token). Leaves hT = the last prompt token's hidden.
  async prefill(ids, opts) {
    const { eng } = this;
    let r = null;
    for (let i = 0; i < ids.length; i++) {
      await this.run(ids[i], eng.pos, { from: "trunk" });   // hT: zeros at 0, else the hidden adopted below
      r = await eng.forwardToken(ids[i], opts);
      this.adoptTrunk();
    }
    return r;
  }

  // One speculative step. Precondition: the trunk has run positions 0 .. eng.pos - 1, hT holds the hidden of
  // eng.pos - 1, and tNext is the trunk's greedy id for eng.pos. Drafts K tokens (chained), verifies [tNext, ...drafts]
  // with `verifier`, keeps the accepted prefix. -> the new tokens: tNext's successors, 1 .. K + 1 of them (the last one
  // is the next step's tNext). refill: rewrite the block's KV rows of the accepted drafts from the exact trunk
  // hiddens (llama.cpp's draft-mtp catch-up does); false keeps the rows the drafts wrote.
  async specStep(tNext, verifier, K = 3, { refill = true } = {}) {
    const { eng, cfg } = this, pos = eng.pos, HD4 = cfg.hDim * 4;
    K = Math.max(1, Math.min(this.maxDrafts, K, eng.maxSeq - pos - 1));
    const drafts = [];
    for (let k = 0; k < K; k++) drafts.push(await this.run(k ? drafts[k - 1] : tNext, pos + k, { from: k ? "chain" : "trunk", head: true }));
    const v = await verifier.verify([tNext, ...drafts]);
    let a = 0;
    while (a < K && a < v.ids.length - 1 && v.ids[a] === drafts[a]) a++;
    await verifier.accept(a + 1);
    const out = v.ids.slice(0, a + 1);
    this.stats.steps++; this.stats.drafts += K; this.stats.accepted += a;
    // rows pos + 1 .. pos + a: token out[k - 1] with the trunk's hidden of column k - 1; then hT <- column a's
    if (refill) for (let k = 1; k <= a; k++) await this.run(out[k - 1], pos + k, { hFrom: { buffer: v.hcols, offset: (k - 1) * HD4 } });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(v.hcols, a * HD4, this.hT, 0, HD4);
    this.device.queue.submit([enc.finish()]);
    return out;
  }
}

// Verify by decode steps (no batched pass on this branch yet): Qwen4ExpBatch's verify / accept contract on
// forwardToken. earlyStop (default): a column whose greedy id is not the next draft ends the pass, so the state is
// already the accepted one and accept() has nothing to undo; the ids are the same as a full pass's up to there.
// earlyStop: false runs every column, as a batched pass does, and accept(n < N) rolls back: the recurrent state copy
// (conv, S, PLE ring) restored, the n-gram predecessors rolled back, the n kept columns re-run.
export class DecodeVerifier {
  constructor(eng, { maxCols = 8, earlyStop = true } = {}) {
    Object.assign(this, { eng, maxCols, earlyStop });
    this.hcols = eng.device.createBuffer({ size: maxCols * eng.cfg.hDim * 4, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    this._snap = null; this._pending = null;
  }

  // ids = [the last accepted token, ...draft] -> { ids: the trunk's greedy id after each column run, hcols: a GPU
  // buffer with each run column's output residual (hDim floats per column) }
  async verify(ids) {
    const { eng } = this, n = ids.length, HD4 = eng.cfg.hDim * 4;
    if (!(n >= 1 && n <= this.maxCols)) throw new Error(`qwen4exp verify: ${n} columns (1 .. ${this.maxCols})`);
    if (!this.earlyStop) this._save();
    const pos0 = eng.pos, out = [];
    for (let c = 0; c < n; c++) {
      const r = await eng.forwardToken(ids[c]);
      out.push(r.id);
      this._keepHidden(c);
      if (this.earlyStop && c + 1 < n && r.id !== ids[c + 1]) break;
    }
    this._pending = { ids: ids.slice(0, out.length), pos0, HD4 };
    return { ids: out, hcols: this.hcols };
  }

  // keep the first n columns of the last verify
  async accept(n) {
    const v = this._pending, { eng } = this;
    if (!v) throw new Error("qwen4exp verify: accept without a verify pass");
    this._pending = null;
    if (!(Number.isInteger(n) && n >= 0 && n <= v.ids.length)) throw new Error(`qwen4exp verify: accept(${n}) of ${v.ids.length}`);
    if (n === v.ids.length) return;
    if (this.earlyStop) throw new Error("qwen4exp verify: an early-stopped pass keeps every column it ran");
    this._restore();
    eng.pos = v.pos0;
    eng.ngs?.rollback(v.ids.length);
    for (let c = 0; c < n; c++) { await eng.forwardToken(v.ids[c]); this._keepHidden(c); }
  }

  _keepHidden(c) {
    const { eng } = this, HD4 = eng.cfg.hDim * 4, enc = eng.device.createCommandEncoder();
    enc.copyBufferToBuffer(eng.B.R, 0, this.hcols, c * HD4, HD4);
    eng.device.queue.submit([enc.finish()]);
  }
  // the recurrent state (DeltaNet conv + S, the PLE ring), copied on the GPU (as Qwen4ExpBatch keeps it)
  _bufs() { return this.eng.layers.flatMap((l) => [l.st.conv, l.st.S, l.st.ring].filter(Boolean)); }
  _save() {
    const src = this._bufs(), dev = this.eng.device;
    if (!this._snap) this._snap = src.map((b) => dev.createBuffer({ size: b.size, usage: U.COPY_DST | U.COPY_SRC }));
    const enc = dev.createCommandEncoder();
    src.forEach((b, k) => enc.copyBufferToBuffer(b, 0, this._snap[k], 0, b.size));
    dev.queue.submit([enc.finish()]);
  }
  _restore() {
    const dev = this.eng.device, enc = dev.createCommandEncoder();
    this._bufs().forEach((b, k) => enc.copyBufferToBuffer(this._snap[k], 0, b, 0, b.size));
    dev.queue.submit([enc.finish()]);
  }
}

// A Qwen4ExpBatch (engine/qwen4exp_batch.js, PR #348) as a verifier: its verify pass with the hiddens, copied to the
// GPU buffer specStep reads the next first draft's hidden and the refill rows from.
export class BatchVerifier {
  constructor(eng, batch, { maxCols = 8 } = {}) {
    Object.assign(this, { eng, batch });
    this.hcols = eng.device.createBuffer({ size: maxCols * eng.cfg.hDim * 4, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
  }
  async verify(ids) {
    const r = await this.batch.verify(ids, { wantHidden: true });
    this.eng.device.queue.writeBuffer(this.hcols, 0, r.hiddens);
    return { ids: r.ids, hcols: this.hcols };
  }
  accept(n) { return this.batch.accept(n); }
}
