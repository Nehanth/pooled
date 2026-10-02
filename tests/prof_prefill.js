// Prefill profile: end-to-end prefill tok/s at several prompt lengths, plus per-kernel GPU time
// from timestamp queries, grouped by what the kernel does (DeltaNet recurrence / conv / glue,
// DeltaNet / attention / FFN projection GEMMs, attention core, MoE router / experts / shared
// expert, norms and glue, the draft-block (MTP) cache fill), and the achieved GB/s and TFLOPS of
// every prefill GEMM / batched GEMV shape.
//
// One prefill of max(LENS) tokens, fed through prefillTokens in segments ending at each length,
// so the wall time to each length is the cumulative time of the segments before it (all lengths
// are multiples of the batch width: no tail passes). A second, instrumented prefill of the same
// tokens runs every dispatch of the SAMPLED passes in its own compute pass with begin/end
// timestamps; each sampled pass stands for `stride` passes (uniform stride per segment, so the
// position-dependent attention cost integrates correctly). Kernel time excludes CPU encode,
// submits and syncs: wall - kernel sum is the overhead.
//
//   MODEL=moe|27b  LENS=512,4096,16384  SAMPLES=32 (instrumented passes per segment)  MTP_FILL=1
//   MOEGROUP=U (MoE: expert-grouped prefill in U-token ubatches; LENS must be multiples of U; the instrumented
//   run then samples whole ubatches instead of passes)  MOEGROUP_UC=8
//   PREFILL_UBATCH=256: profile the wide prefill (prefillUbatch); LENS must then be multiples of it and
//   whole wide chunks are sampled (SAMPLES = chunks per segment); a wide GEMM covers the chunk's columns.
//   cd tests && MODEL=27b deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights prof_prefill.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { makeTokenizer } from "../engine/engine.js";
import { qwen35Weights, tokenizerFromGGUF } from "../engine/gguf.js";
import { openGGUF, wideOpts } from "./load_model.js";
import { GEMM_S } from "../engine/wgsl/gemm.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "moe");
const PATH = MODEL === "27b" ? "../models/q38/model.gguf" : "../models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf";
const LENS = env("LENS", "512,4096,16384").split(",").map(Number);
const SAMPLES = +env("SAMPLES", 32);
const NC = +env("NC", 16);
const MAXLEN = Math.max(...LENS);
const MAXSEQ = +env("CTX", Math.ceil((MAXLEN + 256) / 256) * 256);
if (LENS.some((l) => l % NC)) throw new Error(`LENS must be multiples of ${NC}`);

// GEMM_EXTRA='{"8192x2048":2,...}': pin extra split-K shapes (experiment: puts those shapes on the prefill GEMM)
if (Deno.env.get("GEMM_EXTRA")) Object.assign(GEMM_S, JSON.parse(Deno.env.get("GEMM_EXTRA")));
const model = openGGUF(PATH);
const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredFeatures: ["timestamp-query"], requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
let gpuErrors = 0;
device.addEventListener?.("uncapturederror", (e) => { if (gpuErrors++ < 4) console.error("GPU ERROR:", e.error?.message?.slice(0, 200)); });
const G = model.G, m = G.meta;
const nBlk = m["qwen35.block_count"], L = nBlk - (m["qwen35.nextn_predict_layers"] || 0);
const hasMtp = Object.keys(G.tensors).some((k) => k.startsWith(`blk.${nBlk - 1}.`)) && L < nBlk;
const tok = makeTokenizer(tokenizerFromGGUF(m));
let t0 = performance.now();
const weights = await qwen35Weights(G, (i) => model.readAt(i.byteOffset, i.byteLength), { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: hasMtp });
const eng = await Qwen35Engine.create({ device, meta: m, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: MAXSEQ, batchCols: NC, coopRowsB: 1,
  moeGroupPrefill: env("MOEGROUP") === undefined ? undefined : +env("MOEGROUP"), moeGroupUC: +env("MOEGROUP_UC", 8), moeGroupTiled: env("MOEGROUP_TILED") === undefined ? undefined : env("MOEGROUP_TILED") === "1", ...wideOpts() });
console.log(`moeGroupPrefill ${eng.moeGrpU || "off"}${eng.moeGrpU ? ` UC ${eng.moeGrpUC}` : ""}`);
const UB = eng.ubatch;
if (UB && LENS.some((l) => l % UB)) throw new Error(`LENS must be multiples of the ubatch ${UB}`);
eng.mtpFill = env("MTP_FILL", "1") !== "0";
const D = eng.dims;
console.log(`${MODEL}: loaded in ${((performance.now() - t0) / 1000).toFixed(0)}s · layers ${L} (${eng.layers.filter((x) => x.isFull).length} full attn) · dims ${JSON.stringify(D)}`);
console.log(`  prefillMath ${eng.prefillMath} · moe ${JSON.stringify(eng.moe)} · gemmOn ${eng.gemmOn} shapes ${JSON.stringify([...eng._gemmShapes])} q8pairs ${JSON.stringify(eng._gemm8Pairs)} · flash ${eng.flash} faSplit ${eng.faSplit} · mtp ${!!eng.mtp} fill ${eng.mtpFill} · ubatch ${UB} tile ${JSON.stringify(eng.wideCfg)}`);

// prompt: this repo's source (same recipe as bench_ctx.js)
let ids = [];
for (const f of ["../engine/qwen35.js", "../room.js", "../engine/gguf.js", "../engine/wgsl/base.js", "../engine/wgsl/moe.js", "../harness/agent.js", "../room/plan.js"]) {
  try { ids.push(...tok.encode(`\n// file: ${f}\n` + await Deno.readTextFile(new URL(f, import.meta.url)))); } catch {}
  if (ids.length >= MAXLEN) break;
}
while (ids.length < MAXLEN) ids = ids.concat(ids);
ids = ids.slice(0, MAXLEN);

// ---- 1. wall time (no instrumentation) ----
eng.reset(); if (eng.mtp) eng.mtpFill = env("MTP_FILL", "1") !== "0";
await eng.prefillTokens(ids.slice(0, NC));   // warm-up pass (pipelines, bind groups)
eng.reset();
const wall = {};
{
  let pos = 0, acc = 0;
  for (const len of LENS) {
    const t = performance.now();
    await eng.prefillTokens(ids.slice(pos, len));
    acc += performance.now() - t; pos = len;
    wall[len] = acc;
    console.log(`wall: ${len} tokens in ${(acc / 1000).toFixed(2)} s = ${(len / acc * 1000).toFixed(1)} tok/s`);
  }
}

// ---- 2. instrumented run ----
// op labels: which projection each batched op belongs to, with its shape and weight bytes
const kindOf = (w) => w?.kind || "?";
const wBytes = (w, dOut, dIn) => { const k = kindOf(w); return k === "q4" ? dOut * dIn * 18 / 32 : k === "q8" ? dOut * dIn * 34 / 32 : k === "f32" ? dOut * dIn * 4 : dOut * dIn * 2; };
const opInfo = new Map();
const note = (op, cat, name, w, dOut, dIn) => { if (op && !opInfo.has(op)) opInfo.set(op, { cat, name, kind: kindOf(w), dOut, dIn, bytes: wBytes(w, dOut, dIn) }); };
eng.layerB.forEach((LB, i) => {
  const Ly = i < eng.layers.length ? eng.layers[i] : eng.mtpLayer, W = weights.layers[i] || weights.mtp?.layer || Ly;
  const src = (k) => W?.[k] ?? Ly?.[k];
  if (Ly.isFull) {
    note(LB.qkvOps?.[0], "attn_proj", "attn_q+gate", src("wq"), D.nH * D.hd * 2, D.dim);
    note(LB.qkvOps?.[1], "attn_proj", "attn_k", src("wk"), D.kvDim, D.dim);
    note(LB.qkvOps?.[2], "attn_proj", "attn_v", src("wv"), D.kvDim, D.dim);
    note(LB.kv, "attn_proj", "attn_kv(merged)", src("wk"), 2 * D.kvDim, D.dim);
    note(LB.o, "attn_proj", "attn_out", src("wo"), D.dim, D.qDim);
  } else if (LB.dnOps) {
    note(LB.dnOps[0], "dn_proj", "dn_qkv", src("wqkv"), D.convDim, D.dim);
    note(LB.dnOps[1], "dn_proj", "dn_z", src("wz"), D.dInner, D.dim);
    note(LB.dnOps[2], "dn_proj", "dn_beta", src("wBeta"), D.nVH, D.dim);
    note(LB.dnOps[3], "dn_proj", "dn_alpha", src("wAlpha"), D.nVH, D.dim);
    note(LB.qz, "dn_proj", "dn_qkv+z(merged)", src("wqkv"), D.convDim + D.dInner, D.dim);
    note(LB.ba, "dn_proj", "dn_beta+alpha(merged)", src("wBeta"), 2 * D.nVH, D.dim);
    note(LB.out, "dn_proj", "dn_out", src("wOut"), D.dim, D.dInner);
  }
  const moe = !!eng.moe;
  (LB.gateUp || []).forEach((op, j) => note(op, moe ? "moe_shared" : "ffn", j ? "ffn_up" : "ffn_gate", src(j ? "ffnUp" : "ffnGate"), D.inter, D.dim));
  note(LB.gu, moe ? "moe_shared" : "ffn", "ffn_gate+up(fused)", src("ffnGate"), 2 * D.inter, D.dim);
  note(LB.down, "ffn", "ffn_down", src("ffnDown"), D.dim, D.inter);
  note(LB.shDown, "moe_shared", "sh_down", src("ffnDown"), D.dim, D.inter);
  note(LB.router, "moe_router", "router", src("router"), (eng.moe?.nExp || 0) + (Ly.fused ? 1 : 0), D.dim);
  note(LB.shRouter, "moe_router", "sh_router", src("shRouter"), 1, D.dim);
  note(LB.rs, "moe_router", "router+sh(merged)", src("router"), (eng.moe?.nExp || 0) + 1, D.dim);
});
if (eng.mtp?.projB) note(eng.mtp.projB, "mtp", "mtp_eh_proj", eng.mtp.ehProj, D.dim, 2 * D.dim);
for (const [k, n] of [["xposeXn", "xpose"], ["xposeG", "xpose"], ["xposeGated", "xpose"], ["xposeAttnOut", "xpose"]]) if (eng[k]) opInfo.set(eng[k], { cat: "glue", name: "gemm_xpose" });

// wide prefill: label the wide GEMM ops (built lazily by _initWide) and remember the chunk width
let wideW = 0;
const labelWide = () => {
  if (!eng.layerW || opInfo.has(eng.layerW[0].out)) return;
  eng.layerW.forEach((R, i) => {
    const Ly = eng.layers[i], W = weights.layers[i] || Ly, src = (k) => W?.[k] ?? Ly?.[k];
    if (Ly.isFull) {
      note(R.proj[0], "attn_proj", "attn_q+gate", src("wq"), D.nH * D.hd * 2, D.dim);
      if (R.proj.length === 2) note(R.proj[1], "attn_proj", "attn_kv(merged)", src("wk"), 2 * D.kvDim, D.dim);
      else { note(R.proj[1], "attn_proj", "attn_k", src("wk"), D.kvDim, D.dim); note(R.proj[2], "attn_proj", "attn_v", src("wv"), D.kvDim, D.dim); }
      note(R.out, "attn_proj", "attn_out", src("wo"), D.dim, D.qDim);
    } else {
      if (R.proj.length === 1) note(R.proj[0], "dn_proj", "dn_qkv+z(merged)", src("wqkv"), D.convDim + D.dInner, D.dim);
      else { note(R.proj[0], "dn_proj", "dn_qkv", src("wqkv"), D.convDim, D.dim); note(R.proj[1], "dn_proj", "dn_z", src("wz"), D.dInner, D.dim); }
      note(R.out, "dn_proj", "dn_out", src("wOut"), D.dim, D.dInner);
    }
    if (R.gate) { note(R.gate, "ffn", "ffn_gate", src("ffnGate"), D.inter, D.dim); note(R.up, "ffn", "ffn_up", src("ffnUp"), D.inter, D.dim); note(R.down, "ffn", "ffn_down", src("ffnDown"), D.dim, D.inter); }
  });
};
if (UB) {
  const origDW = eng._dW.bind(eng);
  eng._dW = (p, op, w) => { labelWide(); const prev = curOp; curOp = op; wideW = w; try { origDW(p, op, w); } finally { curOp = prev; wideW = 0; } };
}

const pipeCat = (p) => {
  if (p === "dn_delta_mc") return "dn_recurrence";
  if (p === "dn_conv_mc") return "dn_conv";
  if (/^dn_/.test(p)) return "dn_glue";
  if (/^(kv_store|attn_flash|attn_combine|attn_scores|attn_softmax|attn_out)/.test(p)) return "attn_core";
  if (/^(attn_glue|sigmoid_mul|qsplit|head_norm|rope_part)/.test(p)) return "attn_glue";
  if (/^moe_route|^moe_router|^moe_nrt/.test(p)) return "moe_router";
  if (p === "moe_gsort") return "moe_group_sort";
  if (/^moe_gus|^moe_gu_/.test(p)) return "moe_experts_gu";   // (moe_gusg: grouped)
  if (/^moe_dnc|^moe_dn_|^moe_dng|^moe_dnq|^moe_combine|^moe_combw/.test(p)) return "moe_experts_down";
  if (p === "moe_qx") return "moe_quant";   // moeGroupDp4a: the experts' activation quantization
  if (/^(rmsnorm|add_res|silu_mul)/.test(p)) return "norms_glue";
  if (p === "gemm_xpose") return "glue";
  return "other";
};

const pname = new Map(Object.entries(eng.pipes).map(([k, v]) => [v, k]));
const QN = 4096;
const qsPool = [];
let curOp = null, inMtp = false, sampleOn = false, wideWant = null, wideChunk = 0;
// wrap the engine's dispatch helpers to know which op / phase a dispatch belongs to
const origDop = eng._dop.bind(eng);
eng._dop = (pass, op, nCols) => { const prev = curOp; curOp = op; try { origDop(pass, op, nCols); } finally { curOp = prev; } };
const origLB = eng._encodeLayerBatch.bind(eng);
eng._encodeLayerBatch = (enc, i, ...rest) => { const prev = inMtp; inMtp = inMtp || i >= eng.layers.length; try { return origLB(enc, i, ...rest); } finally { inMtp = prev; } };
if (UB) {
  const origPW = eng._prefillWide.bind(eng);
  eng._prefillWide = async (...a) => { sampleOn = !!wideWant?.has(wideChunk++); try { return await origPW(...a); } finally { sampleOn = false; } };
}
const origMF = eng._mtpFillBatch.bind(eng);
eng._mtpFillBatch = (...a) => { inMtp = true; try { return origMF(...a); } finally { inMtp = false; } };

let pending = [];          // encoders finished with timestamps, waiting for their submit
const reads = [];          // promises of { recs, t: BigUint64Array }
const agg = {};            // per segment end: key -> ms (weighted)
let weight = 1, segKey = 0;
const origCreate = device.createCommandEncoder.bind(device);
device.createCommandEncoder = (d) => {
  const enc = origCreate(d);
  if (!sampleOn) return enc;
  const recs = [];   // { qs, idx, label }
  let qs = null, qn = QN;
  const nextQ = () => { if (qn + 2 > QN) { qs = device.createQuerySet({ type: "timestamp", count: QN }); qsPool.push(qs); recs.push({ newSet: qs, list: [] }); qn = 0; } const i = qn; qn += 2; return i; };
  const ob = enc.beginComputePass.bind(enc);
  enc.beginComputePass = () => {
    let pipe = null; const bgs = {};
    return {
      setPipeline(p) { pipe = p; }, setBindGroup(i, b) { bgs[i] = b; },
      dispatchWorkgroups(x, y = 1, z = 1) {
        const i = nextQ();
        const p = ob({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: i, endOfPassWriteIndex: i + 1 } });
        p.setPipeline(pipe); for (const k in bgs) p.setBindGroup(+k, bgs[k]); p.dispatchWorkgroups(x, y, z); p.end();
        const pn = pname.get(pipe) || "?";
        const info = curOp && opInfo.get(curOp);
        let cat, name;
        if (inMtp) { cat = "mtp"; name = "mtp:" + (info ? info.name : pn); }
        else if (info && info.cat === "glue") { cat = "glue"; name = "gemm_xpose"; }
        else if (info) { const red = /^gemm_red/.test(pn); cat = info.cat; name = `${info.name}${red ? " (split-K reduce)" : ""} [${pn.replace(/_s\d+$/, "")}${red ? "" : ` ${info.dOut}x${info.dIn} ${info.kind}`}]`; }
        else { cat = pipeCat(pn); name = pn; }
        if (wideW) name = name.replace(/ \[gemm_w_/, ` [wide ${wideW} cols gemm_w_`);
        recs[recs.length - 1].list.push({ i, cat, name, pn, cols: wideW || NC, info: !inMtp && info && !/^gemm_red/.test(pn) ? info : null });
      },
      dispatchWorkgroupsIndirect(buf, off) {   // grouped MoE prefill: sizes written on the GPU
        const i = nextQ();
        const p = ob({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: i, endOfPassWriteIndex: i + 1 } });
        p.setPipeline(pipe); for (const k in bgs) p.setBindGroup(+k, bgs[k]); p.dispatchWorkgroupsIndirect(buf, off); p.end();
        const pn = pname.get(pipe) || "?";
        recs[recs.length - 1].list.push({ i, cat: inMtp ? "mtp" : pipeCat(pn), name: inMtp ? "mtp:" + pn : pn, pn, info: null });
      },
      end() {},
    };
  };
  const ofin = enc.finish.bind(enc);
  enc.finish = () => {
    const bufs = [];
    for (const r of recs) {
      const n = r.list.length * 2; if (!n) continue;
      const res = device.createBuffer({ size: n * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      const rd = device.createBuffer({ size: n * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      enc.resolveQuerySet(r.newSet, 0, n, res, 0); enc.copyBufferToBuffer(res, 0, rd, 0, n * 8);
      bufs.push({ r, res, rd });
    }
    pending.push({ bufs, w: weight, seg: segKey });
    return ofin();
  };
  return enc;
};
const origSubmit = device.queue.submit.bind(device.queue);
device.queue.submit = (cbs) => {
  origSubmit(cbs);
  for (const pe of pending) for (const b of pe.bufs) reads.push(b.rd.mapAsync(GPUMapMode.READ).then(() => {
    const t = new BigUint64Array(b.rd.getMappedRange().slice(0)); b.rd.unmap(); b.rd.destroy(); b.res.destroy();
    const A = (agg[pe.seg] ||= { cat: {}, name: {}, gemm: {}, disp: 0 });
    for (const d of b.r.list) {
      const ms = Number(t[d.i + 1] - t[d.i]) / 1e6 * pe.w;
      A.cat[d.cat] = (A.cat[d.cat] || 0) + ms;
      const nk = (A.name[d.name] ||= { ms: 0, n: 0, cat: d.cat }); nk.ms += ms; nk.n += pe.w;
      if (d.info) { const g = (A.gemm[d.name] ||= { ms: 0, n: 0, bytes: d.info.bytes, fl: 0 }); g.ms += ms; g.n += pe.w; g.fl += 2 * d.info.dOut * d.info.dIn * d.cols * pe.w; }
      A.disp += pe.w;
    }
  }));
  pending = [];
};

eng.reset();
await eng.prefillTokens(ids.slice(0, NC)); eng.reset();
{
  let pos = 0;
  for (const len of LENS) {
    segKey = len;
    const GU = eng.moeGrpU && eng.moeGroup !== false ? eng.moeGrpU : NC;   // grouped: sample whole ubatches
    if ((len - pos) % GU) throw new Error(`segment ${pos}..${len} is not a multiple of ${GU}`);
    const nPass = (len - pos) / GU, stride = Math.max(1, Math.floor(nPass / SAMPLES));
    // sample passes at uniform stride, the middle pass of every block of `stride`
    const want = new Set(); for (let p = Math.floor(stride / 2); p < nPass; p += stride) want.add(pos + p * GU);
    const nSampled = want.size;
    weight = nPass / nSampled;
    // prefillTokens encodes each full pass at eng.pos == basePos (trunk, then MTP fill): toggle there
    const hookFrame = device.queue.writeBuffer.bind(device.queue);
    if (UB) {   // whole wide chunks at a uniform stride
      const nCh = (len - pos) / UB, cs = Math.max(1, Math.floor(nCh / SAMPLES));
      wideWant = new Set(); for (let c = Math.floor(cs / 2); c < nCh; c += cs) wideWant.add(c);
      weight = nCh / wideWant.size; wideChunk = 0;
      const t = performance.now();
      await eng.prefillTokens(ids.slice(pos, len));
      await Promise.all(reads.splice(0));
      console.log(`instrumented segment ..${len}: ${wideWant.size}/${nCh} wide chunks sampled (x${weight.toFixed(2)}), ${((performance.now() - t) / 1000).toFixed(1)} s`);
      pos = len; continue;
    }
    device.queue.writeBuffer = (buf, off, data, ...r) => {
      if (buf === eng.frameBufsB?.[0] && data instanceof Uint32Array && data[2] === NC) {
        const bp = data[0];
        // trunk frame: basePos; MTP fill frame: basePos + 1 (same pass, keep the flag)
        if (want.has(bp)) sampleOn = true; else if (!want.has(bp - 1)) sampleOn = false;
      }
      return hookFrame(buf, off, data, ...r);
    };
    const origPG = eng._prefillGrouped;
    if (GU !== NC) {   // a grouped ubatch starts at eng.pos (trunk and its MTP fills inside); the frame hook stays off
      device.queue.writeBuffer = hookFrame;
      eng._prefillGrouped = async (...a) => { sampleOn = want.has(eng.pos); try { return await origPG.apply(eng, a); } finally { sampleOn = false; } };
    }
    const t = performance.now();
    await eng.prefillTokens(ids.slice(pos, len));
    device.queue.writeBuffer = hookFrame; sampleOn = false; eng._prefillGrouped = origPG;
    await Promise.all(reads.splice(0));
    console.log(`instrumented segment ..${len}: ${nSampled}/${nPass} passes sampled (x${weight.toFixed(2)}), ${((performance.now() - t) / 1000).toFixed(1)} s`);
    pos = len;
  }
}
for (const q of qsPool) q.destroy();

// ---- 3. report ----
const cum = (len) => {   // aggregate segments up to len
  const out = { cat: {}, name: {}, gemm: {}, disp: 0 };
  for (const l of LENS.filter((x) => x <= len)) {
    const A = agg[l]; if (!A) continue;
    for (const [k, v] of Object.entries(A.cat)) out.cat[k] = (out.cat[k] || 0) + v;
    for (const [k, v] of Object.entries(A.name)) { const o = (out.name[k] ||= { ms: 0, n: 0, cat: v.cat }); o.ms += v.ms; o.n += v.n; }
    for (const [k, v] of Object.entries(A.gemm)) { const o = (out.gemm[k] ||= { ms: 0, n: 0, bytes: v.bytes, fl: 0 }); o.ms += v.ms; o.n += v.n; o.fl += v.fl; }
    out.disp += A.disp;
  }
  return out;
};
const result = { model: MODEL, NC, lens: {}, gpuErrors };
for (const len of LENS) {
  const A = cum(len), gpu = Object.values(A.cat).reduce((a, b) => a + b, 0);
  const w = wall[len];
  console.log(`\n=== ${MODEL} prefill ${len} tokens: wall ${(w / 1000).toFixed(2)} s (${(len / w * 1000).toFixed(1)} tok/s) · kernel sum ${(gpu / 1000).toFixed(2)} s (${(len / gpu * 1000).toFixed(1)} tok/s if no overhead) · overhead ${((w - gpu) / w * 100).toFixed(1)}% · ${Math.round(A.disp / (len / NC))} dispatches/pass`);
  console.log("  by category (ms, % of kernel time, ms per token):");
  for (const [k, ms] of Object.entries(A.cat).sort((a, b) => b[1] - a[1])) console.log(`    ${k.padEnd(18)} ${ms.toFixed(0).padStart(8)} ms ${(ms / gpu * 100).toFixed(1).padStart(5)}%  ${(ms / len).toFixed(3)} ms/tok`);
  console.log("  top kernels:");
  for (const [k, v] of Object.entries(A.name).sort((a, b) => b[1].ms - a[1].ms).slice(0, 28)) console.log(`    ${k.slice(0, 70).padEnd(70)} ${v.ms.toFixed(0).padStart(7)} ms ${(v.ms / gpu * 100).toFixed(1).padStart(5)}%  ${(v.ms / v.n * 1000).toFixed(1).padStart(8)} µs each`);
  console.log(`  projections (each dispatch = one pass of ${UB ? "the wide chunk's" : "NC"} columns): µs each, weight GB/s, TFLOPS`);
  const gem = [];
  for (const [k, v] of Object.entries(A.gemm).sort((a, b) => b[1].ms - a[1].ms)) {
    const us = v.ms / v.n * 1000, gbs = v.bytes / (us * 1e-6) / 1e9, tf = v.fl / (v.ms * 1e-3) / 1e12;
    gem.push({ k, ms: +v.ms.toFixed(1), us: +us.toFixed(1), gbs: +gbs.toFixed(1), tflops: +tf.toFixed(3) });
    console.log(`    ${k.slice(0, 70).padEnd(70)} ${us.toFixed(1).padStart(8)} µs ${gbs.toFixed(0).padStart(5)} GB/s ${tf.toFixed(2).padStart(6)} TFLOPS`);
  }
  result.lens[len] = { wallS: +(w / 1000).toFixed(3), tokPerS: +(len / w * 1000).toFixed(1), kernelS: +(gpu / 1000).toFixed(3),
    cat: Object.fromEntries(Object.entries(A.cat).map(([k, v]) => [k, +v.toFixed(1)])), gemm: gem,
    top: Object.entries(A.name).sort((a, b) => b[1].ms - a[1].ms).slice(0, 40).map(([k, v]) => ({ k, ms: +v.ms.toFixed(1), us: +(v.ms / v.n * 1000).toFixed(1), cat: v.cat })) };
}
const out = env("OUT", "");
if (out) Deno.writeTextFileSync(out, JSON.stringify(result, null, 1));
console.log("RESULT " + JSON.stringify({ model: MODEL, gpuErrors, lens: Object.fromEntries(Object.entries(result.lens).map(([k, v]) => [k, { tokPerS: v.tokPerS, kernelS: v.kernelS, cat: v.cat }])) }));
