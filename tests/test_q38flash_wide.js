// The wide (fast) Qwen3.8-Flash-Next prefill kernels (engine/wgsl/qwen4exp_wide.js) against a float64 CPU reference
// on random data, no model: the tiled GEMM (q4 / q8 / f32 / bf16 weights, odd row counts, column guards, offsets and
// strides, each epilogue mode), and the expert-grouped MoE (sort, gate/up, down, combine) for every pair of a pass,
// including experts with more pairs than one tile holds and experts no column picked. Bound: max |gpu - ref| /
// max |ref| < 5e-6 per output tensor (f32 sums in another order than the reference). Also: two runs give the same bits.
//   deno run --unstable-webgpu --allow-read tests/test_q38flash_wide.js
import * as X from "../engine/wgsl/qwen4exp_wide.js";
import * as MC from "../engine/wgsl/qwen4exp_batch.js";

const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize, maxBufferSize: ad.limits.maxBufferSize } });
let gpuErr = 0;
device.addEventListener?.("uncapturederror", (e) => { if (gpuErr++ < 4) console.error("GPU ERROR:", e.error?.message?.slice(0, 400)); });
const SU = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const mk = (bytes) => device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: SU });
const put = (a) => { const b = mk(a.byteLength); device.queue.writeBuffer(b, 0, a); return b; };
const uni = (fields, usage = GPUBufferUsage.UNIFORM) => {
  const n = Math.max(4, Math.ceil(fields.length / 4) * 4), ab = new ArrayBuffer(n * 4), dv = new DataView(ab);
  fields.forEach((v, i) => (typeof v === "number" ? dv.setUint32(i * 4, v, true) : dv.setFloat32(i * 4, v.f, true)));
  const b = device.createBuffer({ size: ab.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(b, 0, ab);
  return b;
};
const pipes = new Map();
const pipeOf = (code) => {
  if (!pipes.has(code)) pipes.set(code, device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } }));
  return pipes.get(code);
};
function run(code, list, wg) {
  const p = pipeOf(code);
  const bg = device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: list.map((b, i) => (b ? { binding: i, resource: { buffer: b } } : null)).filter(Boolean) });
  const e = device.createCommandEncoder(), pass = e.beginComputePass();
  pass.setPipeline(p); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(...wg); pass.end();
  device.queue.submit([e.finish()]);
}
async function read(b, n, Type = Float32Array) {
  const s = device.createBuffer({ size: Math.ceil(n * 4 / 4) * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = device.createCommandEncoder(); e.copyBufferToBuffer(b, 0, s, 0, n * 4); device.queue.submit([e.finish()]);
  await s.mapAsync(GPUMapMode.READ); const a = new Type(s.getMappedRange().slice(0)); s.unmap(); s.destroy(); return a;
}

let seed = 20261012, fails = 0;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 2 ** 32 - 0.5; };
const randF = (n, s = 1) => Float32Array.from({ length: n }, () => rnd() * s);
const check = (ok, msg) => { console.log(`${ok ? "ok  " : "FAIL"} ${msg}`); if (!ok) fails++; };
const silu = (v) => v / (1 + Math.exp(-v)), sigm = (v) => 1 / (1 + Math.exp(-v));
// random weights in the engine's layouts, and their dequantized values (row-major rows x dIn)
const f16 = (m, e) => ({ bits: ((e + 15) << 10) | m, v: (1 + m / 1024) * 2 ** e });
function quant(kind, rows, dIn) {
  const nBlk = rows * dIn / 32, deq = new Float64Array(rows * dIn);
  if (kind === "f32") { const w = randF(rows * dIn, 0.2); deq.set(w); return { kind, buf: put(w), deq }; }
  if (kind === "bf16") {   // u32 pairs, low half = the even value (QSA's indexer weights, engine/qwen4exp_qsa.js)
    const w = randF(rows * dIn, 0.2), wb = new Uint32Array(w.buffer), pk = new Uint32Array(rows * dIn / 2);
    for (let i = 0; i < pk.length; i++) pk[i] = (wb[2 * i] >>> 16) | (wb[2 * i + 1] & 0xffff0000);
    for (let i = 0; i < w.length; i++) deq[i] = new Float32Array(new Uint32Array([wb[i] & 0xffff0000]).buffer)[0];
    return { kind, buf: put(pk), deq };
  }
  const sc = new Uint32Array(Math.ceil(nBlk / 2)), sv = new Float64Array(nBlk);
  for (let b = 0; b < nBlk; b++) { const h = f16(Math.floor((rnd() + 0.5) * 1023), -7 - Math.floor((rnd() + 0.5) * 3)); sc[b >> 1] |= h.bits << ((b & 1) * 16); sv[b] = h.v; }
  const per = kind === "q4" ? 16 : 32, qs = new Uint8Array(nBlk * per);
  for (let i = 0; i < qs.length; i++) qs[i] = Math.floor((rnd() + 0.5) * 256) & 255;
  for (let b = 0; b < nBlk; b++) for (let j = 0; j < 32; j++) {
    let q;
    if (kind === "q4") { const byte = qs[b * 16 + (j & 15)]; q = (j < 16 ? byte & 15 : byte >> 4) - 8; }
    else { const byte = qs[b * 32 + j]; q = byte > 127 ? byte - 256 : byte; }
    deq[b * 32 + j] = sv[b] * q;
  }
  return { kind, qs: put(qs), sc: put(sc), deq, sv };
}
const relErr = (g, r) => { let md = 0, mr = 0; for (let i = 0; i < r.length; i++) { md = Math.max(md, Math.abs(g[i] - r[i])); mr = Math.max(mr, Math.abs(r[i])); } return md / mr; };
const BOUND = 5e-6, DP4A_BOUND = 2e-3;

// activation quantization as quant_x does it (f32 math, ties to even), per 32 values: { q: Int8Array, d: Float32Array }
const fr = Math.fround;
const rne = (v) => { const f = Math.floor(v), r = v - f; return r > 0.5 ? f + 1 : r < 0.5 ? f : (f % 2 === 0 ? f : f + 1); };
function quantRef(x, off, n) {
  const nb = n / 32, q = new Int8Array(n), d = new Float32Array(nb);
  for (let b = 0; b < nb; b++) {
    let m = 0; for (let i = 0; i < 32; i++) m = Math.max(m, Math.abs(x[off + b * 32 + i]));
    const dd = fr(m / 127), id = dd > 0 ? fr(1 / dd) : 0;
    d[b] = dd; for (let i = 0; i < 32; i++) q[b * 32 + i] = rne(fr(x[off + b * 32 + i] * id));
  }
  return { q, d };
}
// the weights' integer values and block scales (dequantized = scale * int), for the dp4a reference
function intW(W, rows, dIn) { const nb = rows * dIn / 32, s = new Float64Array(nb), q = new Float64Array(rows * dIn); for (let b = 0; b < nb; b++) { let m = 0; for (let i = 0; i < 32; i++) m = Math.max(m, Math.abs(W.deq[b * 32 + i])); s[b] = W.sv[b]; for (let i = 0; i < 32; i++) q[b * 32 + i] = Math.round(W.deq[b * 32 + i] / W.sv[b]); } return { q, s }; }
const dp4aOk = await (async () => { const m = device.createShaderModule({ code: "@compute @workgroup_size(1) fn main() { let v = dot4I8Packed(pack4xI8(vec4<i32>(1, -2, 3, -4)), 0x01010101u); }" }); return !(await m.getCompilationInfo()).messages.some((x) => x.type === "error"); })();
console.log(`dp4a (packed_4x8_integer_dot_product): ${dp4aOk ? "yes" : "no: the dp4a kernels are skipped"}`);

// ---- gemm_t (f32 tiles) and gemm_d (dp4a), with and without split-K
const dummy = mk(16);
for (const kind of ["q4", "q8", "f32", "bf16"]) for (const [dOut, dIn, nCols, maxCols] of [[200, 320, 37, 64], [64, 2560, 128, 128], [513, 640, 5, 64]]) for (const mode of [0, 1, 2]) for (const dp4a of kind === "f32" || kind === "bf16" || !dp4aOk ? [false] : [false, true]) for (const split of [1, 3]) {
  const W = quant(kind, dOut, dIn), xs4 = dIn / 4 + 3, xOff4 = 2, ys = dOut + 7, yOff = 3, scale = 0.25, nb = dIn / 32;
  const kPer = split > 1 ? Math.ceil(nb / split / 2) * 2 : 1 << 30;
  const x = randF(xOff4 * 4 + maxCols * xs4 * 4), aux = randF(maxCols * ys + yOff + 64, 4);
  const xb = put(x), yb = put(new Float32Array(maxCols * ys + 64).fill(-1234)), ab = put(aux);
  const F = uni([0, nCols, 0, 0]), P = uni([dOut, dIn, xOff4, xs4, yOff, ys, mode, { f: scale }, kPer, split > 1 ? 1 : 0, 0, 0]);
  const part = split > 1 ? put(new Float32Array(split * maxCols * dOut)) : null, out = part || yb;
  const grid = (BM, BN) => [Math.ceil(dOut / BM), Math.ceil(nCols / BN), split];
  let xq = null;
  if (dp4a) {
    const qb = mk(maxCols * dIn), db = mk(maxCols * nb * 4);
    run(X.QUANT_X_WGSL, [xb, qb, db, uni([dIn, xOff4, xs4, 1]), F], [Math.ceil(nb / 32), nCols, 1]);
    run(X.gemmDWGSL(kind), [W.qs, W.sc, qb, db, out, P, F, ab], grid(X.DP4A_T.BM, X.DP4A_T.BN));
    xq = Array.from({ length: nCols }, (_, c) => quantRef(x, (xOff4 + c * xs4) * 4, dIn));
  } else run(X.gemmTWGSL(kind), kind === "f32" || kind === "bf16" ? [W.buf, null, xb, out, P, F, ab] : [W.qs, W.sc, xb, out, P, F, ab], grid(X.WIDE_T.BM, X.WIDE_T.BN));
  if (split > 1) run(X.SPLIT_SUM_WGSL, [part, yb, uni([dOut, dIn, xOff4, xs4, yOff, ys, mode, { f: scale }, split, 0, 0, 0]), F, ab], [Math.ceil(dOut / 256), nCols, 1]);
  const got = await read(yb, maxCols * ys + 64);
  const ref = new Float64Array(nCols * dOut), g = new Float64Array(nCols * dOut);
  const I = dp4a ? intW(W, dOut, dIn) : null;
  for (let c = 0; c < nCols; c++) for (let r = 0; r < dOut; r++) {
    let a = 0;
    if (dp4a) for (let b = 0; b < nb; b++) { let is = 0; for (let i = 0; i < 32; i++) is += I.q[r * dIn + b * 32 + i] * xq[c].q[b * 32 + i]; a += is * I.s[r * nb + b] * xq[c].d[b]; }
    else for (let k = 0; k < dIn; k++) a += W.deq[r * dIn + k] * x[(xOff4 + c * xs4) * 4 + k];
    const o = c * ys + yOff + r;
    ref[c * dOut + r] = mode === 1 ? silu(a * scale) : mode === 2 ? silu(aux[o]) * a : a;
    g[c * dOut + r] = got[o];
  }
  // nothing written outside the columns' row ranges
  const inside = new Set(); for (let c = 0; c < nCols; c++) for (let r = 0; r < dOut; r++) inside.add(c * ys + yOff + r);
  let outside = 0;
  for (let i = 0; i < got.length; i++) if (!inside.has(i) && got[i] !== -1234) outside++;
  const e = relErr(g, ref);
  // dp4a: the reference quantizes as quant_x does, but WGSL's 1 / d is not correctly rounded, so a value at a rounding
  // midpoint can land on the other integer (~1e-4 relative); a layout or index error would be O(1)
  check(e < (dp4a ? DP4A_BOUND : BOUND) && outside === 0, `${dp4a ? "gemm_d" : "gemm_t"} ${kind} ${dOut} x ${dIn}, ${nCols}/${maxCols} columns, mode ${mode}${split > 1 ? `, split-K ${split}` : ""}: rel ${e.toExponential(2)}, ${outside} writes outside`);
}

// ---- expert-grouped MoE
// variants: "rows" (f32, a row per thread), "dp4a" (coalesced tiles, quantized activations)
for (const kind of ["q4", "q8"]) for (const [nExp, K, nCols, d, ff, hot] of [[16, 4, 50, 256, 192, false], [64, 10, 64, 128, 64, true], [512, 10, 300, 64, 32, false]]) for (const vr of dp4aOk ? ["rows", "dp4a"] : ["rows"]) {
  const dp4a = vr === "dp4a";
  const maxCols = nCols + 3, S = X.moeSortWSizes({ maxCols, K, nExp });
  const G = quant(kind, nExp * ff, d), Uq = quant(kind, nExp * ff, d), D = quant(kind === "q4" ? "q8" : "q4", nExp * d, ff);
  // ids: K distinct experts per column; "hot" sends most columns to experts 0..K-1 (more pairs than a tile)
  const ids = new Uint32Array(maxCols * K), wts = randF(maxCols * K, 1), ls = nExp + 1, lg = randF(maxCols * ls, 4);
  for (let c = 0; c < nCols; c++) {
    const pick = new Set();
    while (pick.size < K) pick.add(hot && rnd() < 0.3 ? pick.size : Math.floor((rnd() + 0.5) * nExp) % Math.max(1, nExp - 3));
    [...pick].forEach((e, k) => { ids[c * K + k] = e; });
  }
  const x = randF(maxCols * d), sh = randF(maxCols * d);
  const idb = put(ids), wb = put(wts), lgb = put(lg), xb = put(x), shb = put(sh);
  const grp = mk(S.words * 4), h = put(new Float32Array(maxCols * K * ff)), y = put(new Float32Array(maxCols * K * d)), out = put(new Float32Array(maxCols * d));
  const F = uni([0, nCols, 0, 0]);
  const xq = mk(maxCols * d), xd = mk(maxCols * d / 8), hq = mk(maxCols * K * ff), hd = mk(maxCols * K * ff / 8);
  const runAll = async () => {
    run(X.moeSortWWGSL(), [idb, grp, uni([nExp, K, S.PO, 0]), F], [1, 1, 1]);
    const tiles = Math.ceil(nCols * K / X.MOE_BP) + Math.min(nExp, nCols * K);
    if (dp4a) {
      run(X.QUANT_X_WGSL, [xb, xq, xd, uni([d, 0, d / 4, 1]), F], [Math.ceil(d / 1024), nCols, 1]);
      run(X.moeGuCWGSL(kind), [G.qs, G.sc, Uq.qs, Uq.sc, xq, xd, h, grp, uni([ff, d, K, S.PO, ff, 0, 0, 0])], [Math.ceil(ff / X.MOE_RC), tiles, 1]);
      run(X.QUANT_X_WGSL, [h, hq, hd, uni([ff, 0, ff / 4, K]), F], [Math.ceil(ff / 1024), nCols * K, 1]);
      run(X.moeDnCWGSL(D.kind), [D.qs, D.sc, hq, hd, y, grp, uni([d, ff, K, S.PO, 0, 0, 0, 0])], [Math.ceil(d / X.MOE_RC), tiles, 1]);
    } else {
      run(X.moeGuWWGSL(kind), [G.qs, G.sc, Uq.qs, Uq.sc, xb, h, grp, uni([ff, d, K, S.PO, d / 4, ff, 0, 0])], [Math.ceil(ff / X.MOE_RW), tiles, 1]);
      run(X.moeDnWWGSL(D.kind), [D.qs, D.sc, h, y, grp, uni([d, ff, K, S.PO, 0, 0, 0, 0])], [Math.ceil(d / X.MOE_RW), tiles, 1]);
    }
    run(X.MOE_COMB_W_WGSL, [y, wb, shb, lgb, out, uni([d, K, ls, nExp])], [Math.ceil(d / 256), nCols, 1]);
    return { g: await read(grp, S.words, Uint32Array), h: await read(h, nCols * K * ff), out: await read(out, nCols * d) };
  };
  const r1 = await runAll(), r2 = await runAll();
  // the sort: every pair exactly once, tiles of one expert each, at most BP pairs, the expert's pairs only
  const nT = r1.g[0], seen = new Uint8Array(nCols * K);
  let sortOk = nT <= S.maxTiles;
  for (let t = 0; t < nT && sortOk; t++) {
    const [p0, np, e] = r1.g.subarray(4 + 4 * t, 7 + 4 * t);
    if (!(np >= 1 && np <= X.MOE_BP)) sortOk = false;
    for (let i = 0; i < np; i++) { const p = r1.g[S.PO + p0 + i]; if (p >= nCols * K || seen[p]++ || ids[p] !== e) sortOk = false; }
  }
  sortOk = sortOk && seen.every((v) => v === 1);
  // reference (dp4a: on the activations quantized as quant_x does, h from the GPU's own)
  const GI = dp4a ? intW(G, nExp * ff, d) : null, UI = dp4a ? intW(Uq, nExp * ff, d) : null, DI = dp4a ? intW(D, nExp * d, ff) : null;
  const xQ = dp4a ? Array.from({ length: nCols }, (_, c) => quantRef(x, c * d, d)) : null, hQ = dp4a ? Array.from({ length: nCols * K }, (_, p) => quantRef(r1.h, p * ff, ff)) : null;
  const hRef = new Float64Array(nCols * K * ff), oRef = new Float64Array(nCols * d);
  for (let c = 0; c < nCols; c++) for (let k = 0; k < K; k++) {
    const e = ids[c * K + k], p = c * K + k;
    for (let r = 0; r < ff; r++) {
      let a = 0, b = 0; const ro = (e * ff + r) * d;
      if (dp4a) for (let bk = 0; bk < d / 32; bk++) {
        let ia = 0, ib = 0;
        for (let i = 0; i < 32; i++) { const xi = xQ[c].q[bk * 32 + i]; ia += GI.q[ro + bk * 32 + i] * xi; ib += UI.q[ro + bk * 32 + i] * xi; }
        a += ia * GI.s[ro / 32 + bk] * xQ[c].d[bk]; b += ib * UI.s[ro / 32 + bk] * xQ[c].d[bk];
      } else for (let i = 0; i < d; i++) { a += G.deq[ro + i] * x[c * d + i]; b += Uq.deq[ro + i] * x[c * d + i]; }
      hRef[p * ff + r] = silu(a) * b;
    }
  }
  for (let c = 0; c < nCols; c++) for (let r = 0; r < d; r++) {
    let acc = 0;
    for (let k = 0; k < K; k++) {
      const e = ids[c * K + k], p = c * K + k; let a = 0; const ro = (e * d + r) * ff;
      if (dp4a) for (let bk = 0; bk < ff / 32; bk++) { let ia = 0; for (let i = 0; i < 32; i++) ia += DI.q[ro + bk * 32 + i] * hQ[p].q[bk * 32 + i]; a += ia * DI.s[ro / 32 + bk] * hQ[p].d[bk]; }
      else for (let i = 0; i < ff; i++) a += D.deq[ro + i] * r1.h[p * ff + i];
      acc += wts[p] * a;
    }
    oRef[c * d + r] = acc + sigm(lg[c * ls + nExp]) * sh[c * d + r];
  }
  const eh = relErr(r1.h, hRef), eo = relErr(r1.out, oRef);
  const same = r1.out.every((v, i) => Object.is(v, r2.out[i])) && r1.h.every((v, i) => Object.is(v, r2.h[i]));
  const bnd = dp4a ? DP4A_BOUND : BOUND;
  check(sortOk && eh < bnd && eo < bnd && same, `moe grouped ${vr} ${kind} (down ${D.kind}), ${nExp} experts top-${K}, ${nCols} columns, ${nT} tiles: sort ${sortOk ? "ok" : "BAD"}, h rel ${eh.toExponential(2)}, out rel ${eo.toExponential(2)}, rerun ${same ? "same bits" : "DIFFERS"}`);
}
// ---- attn_w: dense (causal, column groups) and gathered (row lists), vs float64
{
  const nH = 24, nKV = 2, hd = 256, G = nH / nKV, qs = 2 * nH * hd + 2 * nKV * hd, stride = 2049;
  for (const [pos, nCols, gather] of [[0, 37, false], [1500, 64, false], [2200, 23, true], [100, 7, true]]) {
    const maxSeq = pos + nCols + 8, kc = randF(maxSeq * nKV * hd), vc = randF(maxSeq * nKV * hd), qn = randF(nCols * nH * hd, 2), qkv = randF(nCols * qs, 4);
    // gathered: per column a sorted random subset of 0 .. pos + col (up to 2048 rows), always holding the last
    const lists = [], sel = new Uint32Array(nCols * stride);
    for (let c = 0; c < nCols; c++) {
      let L = [];
      for (let p = 0; p <= pos + c; p++) if (!gather || p === pos + c || rnd() < 0.45) L.push(p);
      if (L.length > 2048) L = L.slice(L.length - 2048);
      lists.push(L); sel[c * stride] = L.length; sel.set(L, c * stride + 1);
    }
    const out = put(new Float32Array(nCols * nH * hd).fill(7));
    const F = uni([pos, nCols, 0, 0]), AP = uni([nH, nKV, hd, 64, { f: 1e-6 }, 0, 0, 0]), MS = uni([qs, 2 * nH * hd, 2 * nH * hd + nKV * hd, 0]);
    const cw = gather ? 1 : X.ATTN_W.dense.CW;
    run(X.attnWWGSL({ gather: gather ? stride : 0, G }), [put(qn), put(kc), put(vc), put(qkv), out, AP, F, MS, ...(gather ? [put(sel)] : [])], [nKV, Math.ceil(nCols / cw), 1]);
    const got = await read(out, nCols * nH * hd), ref = new Float64Array(nCols * nH * hd);
    for (let c = 0; c < nCols; c++) for (let h = 0; h < nH; h++) {
      const kvh = Math.floor(h / G), L = lists[c], sc = L.map((p) => { let a = 0; for (let i = 0; i < hd; i++) a += qn[(c * nH + h) * hd + i] * kc[(p * nKV + kvh) * hd + i]; return a / Math.sqrt(hd); });
      const mx = Math.max(...sc), e = sc.map((v) => Math.exp(v - mx)), tot = e.reduce((a, b) => a + b, 0);
      for (let i = 0; i < hd; i++) {
        let a = 0; L.forEach((p, k) => { a += e[k] * vc[(p * nKV + kvh) * hd + i]; });
        ref[(c * nH + h) * hd + i] = (a / tot) * sigm(qkv[c * qs + h * 2 * hd + hd + i]);
      }
    }
    const e = relErr(got, ref);
    check(e < BOUND, `attn_w ${gather ? "gathered" : "dense"}, pos ${pos}, ${nCols} columns: rel ${e.toExponential(2)}`);
  }
}
// ---- dn_delta_w + dn_gnorm_w == the exact DN_DELTA_MC (bit for bit: the same expressions, the norm moved out)
{
  const nKH = 16, dS = 128, nVH = 48, keyDim = nKH * dS, convDim = 2 * keyDim + nVH * dS, inner = nVH * dS, nCols = 37, qkvzS = convDim + inner;
  const co = put(randF(nCols * convDim)), ba = put(randF(nCols * 2 * nVH, 4)), dta = put(randF(2 * nVH, 2)), z = put(randF(nCols * qkvzS, 4)), nw = put(randF(dS, 2));
  const S0 = randF(nVH * dS * dS, 0.1), SA = put(S0), SB = put(S0);
  const oA = put(new Float32Array(nCols * inner)), yA = put(new Float32Array(nCols * inner)), oB = put(new Float32Array(nCols * inner)), yB = put(new Float32Array(nCols * inner));
  const ep = uni([nKH, dS, keyDim, nVH, { f: 1e-6 }, convDim, 2 * nVH, qkvzS, convDim, inner, 0, 0]), F = uni([0, nCols, 0, 0]);
  run(MC.DN_DELTA_MC_WGSL, [co, SA, ba, dta, z, nw, oA, yA, ep, F], [nVH, 1, 1]);
  run(X.DN_DELTA_W_WGSL, [co, SB, ba, dta, oB, ep, F], [nVH, 1, 1]);
  run(X.DN_GNORM_W_WGSL, [oB, z, nw, yB, ep], [nVH, nCols, 1]);
  const [a, b, sa, sb] = [await read(yA, nCols * inner, Uint32Array), await read(yB, nCols * inner, Uint32Array), await read(SA, nVH * dS * dS, Uint32Array), await read(SB, nVH * dS * dS, Uint32Array)];
  const dy = a.filter((v, i) => v !== b[i]).length, ds = sa.filter((v, i) => v !== sb[i]).length;
  check(dy === 0 && ds === 0, `dn_delta_w + dn_gnorm_w vs the exact kernel, ${nCols} columns: ${dy} outputs and ${ds} state values differ`);
}
// ---- dn_conv_w + dn_conv_st_w == the exact DN_CONV_MC (bit for bit), including the state, for passes longer and
// shorter than the conv window
for (const nCols of [37, 2, 1]) {
  const K = 4, dS = 128, nQK = 32, nH = 80, convDim = nH * dS, xs = convDim + 64, cs = convDim + 32;
  const xin = put(randF(nCols * xs, 2)), w = put(randF(convDim * K)), st0 = randF(convDim * (K - 1), 2), stA = put(st0), stB = put(st0);
  const coA = put(new Float32Array(nCols * cs)), coB = put(new Float32Array(nCols * cs)), F = uni([0, nCols, 0, 0]);
  run(MC.DN_CONV_MC_WGSL, [xin, stA, w, coA, uni([K, nQK, dS, { f: 1e-6 }, xs, cs, 0, 0]), F], [nH, 1, 1]);
  run(X.DN_CONV_W_WGSL, [xin, stB, w, coB, uni([K, nQK, dS, { f: 1e-6 }, xs, cs, 0, 0])], [nH, nCols, 1]);
  run(X.DN_CONV_ST_W_WGSL, [xin, stB, null, null, uni([K, nQK, dS, { f: 1e-6 }, xs, cs, nH, 0]), F], [Math.ceil(convDim / 256), 1, 1]);
  const [a, b, sa, sb] = [await read(coA, nCols * cs, Uint32Array), await read(coB, nCols * cs, Uint32Array), await read(stA, convDim * (K - 1), Uint32Array), await read(stB, convDim * (K - 1), Uint32Array)];
  const dy = a.filter((v, i) => v !== b[i]).length, ds = sa.filter((v, i) => v !== sb[i]).length;
  check(dy === 0 && ds === 0, `dn_conv_w + dn_conv_st_w vs the exact kernel, ${nCols} columns: ${dy} outputs and ${ds} state values differ`);
}
if (gpuErr) { console.log(`${gpuErr} GPU errors`); fails++; }
console.log(fails ? `Q38FLASH WIDE FAIL (${fails})` : "Q38FLASH WIDE PASS ✓");
Deno.exit(fails ? 1 : 0);
