// The multi-column Qwen3.8-Flash-Next kernels (engine/wgsl/qwen4exp_batch.js) against the decode kernels
// (engine/wgsl/qwen4exp.js) on random data, no model: every column of a multi-column dispatch must give the bits
// the decode kernel gives at that column (for the stateful ones, conv / DeltaNet / PLE, the decode kernel run one
// position after another). Also the gathered attention over the full row list == dense.
//   deno run --unstable-webgpu --allow-read tests/test_q38flash_mc.js
import * as W from "../engine/wgsl/qwen4exp.js";
import * as M from "../engine/wgsl/qwen4exp_batch.js";
import { f16bits } from "./reference/qsa_ref.mjs";

const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize, maxBufferSize: ad.limits.maxBufferSize,
  maxStorageBuffersPerShaderStage: Math.min(10, ad.limits.maxStorageBuffersPerShaderStage) } });
let gpuErr = 0;
device.addEventListener?.("uncapturederror", (e) => { if (gpuErr++ < 4) console.error("GPU ERROR:", e.error?.message?.slice(0, 300)); });
const SU = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const mk = (bytes) => device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: SU });
const put = (a) => { const b = mk(a.byteLength); device.queue.writeBuffer(b, 0, a); return b; };
const uni = (fields) => {
  const n = Math.max(4, Math.ceil(fields.length / 4) * 4), ab = new ArrayBuffer(n * 4), dv = new DataView(ab);
  fields.forEach((v, i) => (typeof v === "number" ? dv.setUint32(i * 4, v, true) : dv.setFloat32(i * 4, v.f, true)));
  const b = device.createBuffer({ size: ab.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(b, 0, ab);
  return b;
};
const f = (v) => ({ f: v });
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
async function read(b, n, off = 0) {
  const s = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = device.createCommandEncoder(); e.copyBufferToBuffer(b, off * 4, s, 0, n * 4); device.queue.submit([e.finish()]);
  await s.mapAsync(GPUMapMode.READ); const a = new Uint32Array(s.getMappedRange().slice(0)); s.unmap(); s.destroy(); return a;
}
const copy = (src, so, dst, d0, n) => { const e = device.createCommandEncoder(); e.copyBufferToBuffer(src, so * 4, dst, d0 * 4, n * 4); device.queue.submit([e.finish()]); };

let seed = 20261011, fails = 0;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 2 ** 32 - 0.5; };
const randF = (n, s = 1) => Float32Array.from({ length: n }, () => rnd() * s);
const check = (ok, msg) => { console.log(`${ok ? "ok  " : "FAIL"} ${msg}`); if (!ok) fails++; };
const same = (a, b) => { let d = 0; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d++; return d; };
// f32 -> f16 bits (round to nearest), for random quant scales
const h16 = (v) => { const fb = new Float32Array([v]), u = new Uint32Array(fb.buffer)[0]; const s = (u >>> 16) & 0x8000, e = ((u >>> 23) & 255) - 112, m = u & 0x7fffff; return e <= 0 ? s : s | (e << 10) | ((m + 0x1000) >>> 13); };
const scales = (nBlk) => { const w = new Uint32Array(Math.ceil(nBlk / 2)); for (let i = 0; i < nBlk; i++) w[i >> 1] |= h16(0.01 + Math.abs(rnd()) * 0.05) << ((i & 1) * 16); return w; };
const quant = (kind, rows, dIn) => {
  const nBlk = rows * dIn / 32;
  if (kind === "f32") return { kind, buf: put(randF(rows * dIn, 0.2)) };
  const qs = new Uint32Array(nBlk * (kind === "q4" ? 4 : 8)); for (let i = 0; i < qs.length; i++) qs[i] = (rnd() + 0.5) * 2 ** 32 >>> 0;
  return { kind, qs: put(qs), sc: put(scales(nBlk)) };
};

// ---- GEMV: the decode kernel (its lanes per row and rows per lane group) per column vs gemv_mc over the columns
for (const [dOut, dIn] of [[203, 2560], [4096, 2560], [16, 10240], [2600, 512]]) for (const kind of ["q4", "q8", "f32"]) for (const mode of [0, 1]) {
  if (mode && dOut > 300) continue;
  const N = 7, Wt = quant(kind, dOut, dIn), x = put(randF(N * dIn));
  const lanes = W.gemvLanes(kind, dOut, dIn), R = W.gemvRows(kind, dOut), nwg = Math.ceil(dOut / (256 / lanes * R));
  const yd = mk(N * dOut * 4), ym = mk(N * dOut * 4);
  for (let c = 0; c < N; c++) {
    const u = uni([dOut, dIn, 0, c * dIn / 4, c * dOut, mode, f(0.25), 0]);
    run(W.gemvWGSL(kind, lanes, R), kind === "f32" ? [Wt.buf, null, x, yd, u] : [Wt.qs, Wt.sc, x, yd, u], [Math.min(nwg, 32768), Math.ceil(nwg / 32768), 1]);
  }
  const F = uni([0, N, 0, 0]), u = uni([dOut, dIn, 0, 0, 0, mode, f(0.25), dIn / 4, dOut, 0, 0, 0]), nm = Math.ceil(dOut / (256 / lanes));
  run(M.gemvMcWGSL(kind, lanes), kind === "f32" ? [Wt.buf, null, x, ym, u, F] : [Wt.qs, Wt.sc, x, ym, u, F], [Math.min(nm, 32768), Math.ceil(nm / 32768), Math.ceil(N / M.GEMV_MC_COLS)]);
  check(same(await read(yd, N * dOut), await read(ym, N * dOut)) === 0, `gemv_mc ${kind} ${dOut}x${dIn} (${lanes} lanes, decode rows ${R}) mode ${mode}: ${N} columns == gemv per column`);
}

// ---- the HC up projection + mix
for (const kind of ["q4", "q8"]) {
  const d = 512, hc = 4, HD = d * hc, rank = 32, N = 5;
  const Wt = quant(kind, HD, rank), lo = put(randF(N * rank, 2)), xn = put(randF(N * HD, 2));
  const lanes = W.gemvLanes(kind, HD, rank);
  const gD = mk(N * HD * 4), mD = mk(N * d * 4), g1 = mk(HD * 4), m1 = mk(d * 4), lo1 = mk(rank * 4), xn1 = mk(HD * 4);
  for (let c = 0; c < N; c++) {
    copy(lo, c * rank, lo1, 0, rank); copy(xn, c * HD, xn1, 0, HD);
    run(W.gemvWGSL(kind, lanes, 4, true), [Wt.qs, Wt.sc, lo1, g1, uni([HD, rank, 0, 0, 0, 0, f(0), d]), xn1, m1], [Math.ceil(d / (256 / lanes)), 1, 1]);
    copy(g1, 0, gD, c * HD, HD); copy(m1, 0, mD, c * d, d);
  }
  const gM = mk(N * HD * 4), mM = mk(N * d * 4);
  run(M.gemvMixMcWGSL(kind, lanes), [Wt.qs, Wt.sc, lo, gM, uni([HD, rank, 0, 0, 0, 0, f(0), d, rank / 4, HD, 0, 0]), xn, mM], [Math.ceil(d / (256 / lanes)), 1, N]);
  check(same(await read(gD, N * HD), await read(gM, N * HD)) === 0 && same(await read(mD, N * d), await read(mM, N * d)) === 0, `gemvmix_mc ${kind}: ${N} columns' gates and mix == per column`);
}

// ---- MoE: route, gate-up, down, and the shared expert (sigmoid of the router row nExp)
{
  const nExp = 64, K = 10, d = 256, ff = 128, N = 5, ls = nExp + 1;
  const lg = randF(N * ls, 4); for (let i = 0; i < N * ls; i += 7) lg[i] = lg[(i + 3) % lg.length];   // some ties
  const L = put(lg), idsD = mk(N * K * 4), wtsD = mk(N * K * 4), idsM = mk(N * K * 4), wtsM = mk(N * K * 4);
  const one = mk(ls * 4), oi = mk(K * 4), ow = mk(K * 4);
  for (let c = 0; c < N; c++) { copy(L, c * ls, one, 0, ls); run(W.MOE_ROUTE_WGSL, [one, oi, ow, uni([nExp, K, 0, 0])], [1, 1, 1]); copy(oi, 0, idsD, c * K, K); copy(ow, 0, wtsD, c * K, K); }
  run(M.MOE_ROUTE_MC_WGSL, [L, idsM, wtsM, uni([nExp, K, ls, 0])], [N, 1, 1]);
  check(same(await read(idsD, N * K), await read(idsM, N * K)) === 0 && same(await read(wtsD, N * K), await read(wtsM, N * K)) === 0, `moe_route_mc: ${N} columns == moe_route per column`);
  for (const kind of ["q4", "q8"]) {
    const g = quant(kind, nExp * ff, d), u = quant(kind, nExp * ff, d), dn = quant(kind, nExp * d, ff), x = put(randF(N * d));
    const hM = mk(N * K * ff * 4), yM = mk(N * d * 4), hD = mk(N * K * ff * 4), yD = mk(N * d * 4);
    const x1 = mk(d * 4), h1 = mk(K * ff * 4), y1 = mk(d * 4), l1 = mk(ls * 4);
    // shared expert: ids [0], K = 1, weight sigmoid(logits[nExp])
    const sg = quant(kind, ff, d), su = quant(kind, ff, d), sdn = quant(kind, d, ff), sh0 = put(new Uint32Array(4));
    const hsD = mk(N * ff * 4), shD = mk(N * d * 4), hsM = mk(N * ff * 4), shM = mk(N * d * 4), hs1 = mk(ff * 4), sh1 = mk(d * 4);
    for (let c = 0; c < N; c++) {
      copy(x, c * d, x1, 0, d); copy(idsM, c * K, oi, 0, K); copy(wtsM, c * K, ow, 0, K); copy(L, c * ls, l1, 0, ls);
      run(W.moeGuWGSL(kind), [g.qs, g.sc, u.qs, u.sc, x1, h1, oi, uni([ff, d, K, 0, 0, 0, 0, 0])], [ff / 8, K, 1]);
      run(W.moeDnWGSL(kind), [dn.qs, dn.sc, h1, y1, oi, ow, uni([d, ff, K, 0, 0, 0, 0, 0])], [d / 8, 1, 1]);
      run(W.moeGuWGSL(kind), [sg.qs, sg.sc, su.qs, su.sc, x1, hs1, sh0, uni([ff, d, 1, 0, 0, 0, 0, 0])], [ff / 8, 1, 1]);
      run(W.moeDnWGSL(kind), [sdn.qs, sdn.sc, hs1, sh1, sh0, l1, uni([d, ff, 1, 1, nExp, 0, 0, 0])], [d / 8, 1, 1]);
      copy(h1, 0, hD, c * K * ff, K * ff); copy(y1, 0, yD, c * d, d); copy(hs1, 0, hsD, c * ff, ff); copy(sh1, 0, shD, c * d, d);
    }
    run(M.moeGuMcWGSL(kind), [g.qs, g.sc, u.qs, u.sc, x, hM, idsM, uni([ff, d, K, d / 4, K * ff, K, 0, 0])], [ff / 8, K, N]);
    run(M.moeDnMcWGSL(kind), [dn.qs, dn.sc, hM, yM, idsM, wtsM, uni([d, ff, K, 0, 0, K, K, K * ff / 4])], [d / 8, 1, N]);
    run(M.moeGuMcWGSL(kind), [sg.qs, sg.sc, su.qs, su.sc, x, hsM, sh0, uni([ff, d, 1, d / 4, ff, 0, 0, 0])], [ff / 8, 1, N]);
    run(M.moeDnMcWGSL(kind), [sdn.qs, sdn.sc, hsM, shM, sh0, L, uni([d, ff, 1, 1, nExp, ls, 0, ff / 4])], [d / 8, 1, N]);
    check(same(await read(hD, N * K * ff), await read(hM, N * K * ff)) === 0 && same(await read(yD, N * d), await read(yM, N * d)) === 0,
      `moe_gu_mc / moe_dn_mc ${kind}: ${N} columns x top-${K} == per column`);
    check(same(await read(hsD, N * ff), await read(hsM, N * ff)) === 0 && same(await read(shD, N * d), await read(shM, N * d)) === 0,
      `moe_gu_mc / moe_dn_mc ${kind}, shared expert (sigmoid gate from the router row): ${N} columns == per column`);
  }
}

// ---- HC site (every combine / inject mode), mix, combine
{
  const d = 512, hc = 4, HD = d * hc, N = 6, eps = 1e-6;
  const R0 = randF(N * HD, 2), w = put(randF(HD, 1)), injW = put(randF(hc * HD, 0.05)), gate = put(randF(N * HD, 3)), a = put(randF(N * d)), b = put(randF(N * d));
  const injP = put(randF(N * hc * hc, 2));
  for (const [comb, inj] of [[0, 1], [1, 1], [2, 1], [2, 0]]) {
    const RD = put(R0), RM = put(R0), xnD = mk(N * HD * 4), xnM = mk(N * HD * 4), ioD = mk(N * hc * hc * 4), ioM = mk(N * hc * hc * 4);
    const R1 = mk(HD * 4), xn1 = mk(HD * 4), a1 = mk(d * 4), b1 = mk(d * 4), ip1 = mk(hc * hc * 4), io1 = mk(hc * hc * 4);
    const u = uni([d, hc, f(eps), comb, inj, 0, 0, 0]);
    for (let c = 0; c < N; c++) {
      copy(RD, c * HD, R1, 0, HD); copy(a, c * d, a1, 0, d); copy(b, c * d, b1, 0, d); copy(injP, c * hc * hc, ip1, 0, hc * hc);
      run(W.HC_SITE_WGSL, [R1, a1, b1, ip1, w, xn1, injW, io1, u], [hc, 1, 1]);
      copy(R1, 0, RD, c * HD, HD); copy(xn1, 0, xnD, c * HD, HD); copy(io1, 0, ioD, c * hc * hc, hc * hc);
    }
    run(M.HC_SITE_MC_WGSL, [RM, a, b, injP, w, xnM, injW, ioM, u], [hc, N, 1]);
    const okInj = !inj || same(await read(ioD, N * hc * hc), await read(ioM, N * hc * hc)) === 0;
    check(same(await read(RD, N * HD), await read(RM, N * HD)) === 0 && same(await read(xnD, N * HD), await read(xnM, N * HD)) === 0 && okInj,
      `hc_site_mc comb ${comb} inject ${inj}: ${N} columns' R, xn${inj ? ", inject partials" : ""} == per column`);
  }
  const xn = put(randF(N * HD, 2)), R = put(R0), mixM = mk(N * d * 4);
  const R1 = mk(HD * 4), xn1 = mk(HD * 4), g1 = mk(HD * 4), m1 = mk(d * 4), a1 = mk(d * 4), b1 = mk(d * 4), i1 = mk(hc * hc * 4);
  const mixD = mk(N * d * 4), RD = mk(N * HD * 4);
  for (let c = 0; c < N; c++) {
    copy(R, c * HD, R1, 0, HD); copy(xn, c * HD, xn1, 0, HD); copy(gate, c * HD, g1, 0, HD); copy(a, c * d, a1, 0, d); copy(b, c * d, b1, 0, d); copy(injP, c * hc * hc, i1, 0, hc * hc);
    run(W.HC_MIX_WGSL, [xn1, g1, m1, uni([d, hc, 0, 0])], [d / 256, 1, 1]);
    run(W.HC_COMBINE_WGSL, [R1, a1, b1, i1, uni([d, hc, 1, 0])], [HD / 256, 1, 1]);
    copy(m1, 0, mixD, c * d, d); copy(R1, 0, RD, c * HD, HD);
  }
  run(M.HC_MIX_MC_WGSL, [xn, gate, mixM, uni([d, hc, 0, 0])], [d / 256, N, 1]);
  run(M.HC_COMBINE_MC_WGSL, [R, a, b, injP, uni([d, hc, 1, 0])], [HD / 256, N, 1]);
  check(same(await read(mixD, N * d), await read(mixM, N * d)) === 0, `hc_mix_mc: ${N} columns == per column`);
  check(same(await read(RD, N * HD), await read(R, N * HD)) === 0, `hc_combine_mc: ${N} columns == per column`);
}

// ---- DeltaNet: conv + l2 (state), delta rule + gated norm (state), 9 columns after 3 decode steps
{
  const nKH = 2, nV = 6, dS = 128, keyDim = nKH * dS, inner = nV * dS, n = 2 * keyDim + inner, K = 4, N = 9, pre = 3, eps = 1e-6;
  const zs = n + inner;   // the [qkv | z] buffer's column
  const qz = put(randF((pre + N) * zs, 2)), cw = put(randF(n * K, 0.5)), ba = put(randF((pre + N) * 2 * nV, 2));
  const dta = put(Float32Array.from([...randF(nV), ...Array.from({ length: nV }, () => -Math.abs(rnd()) * 2)])), nw = put(randF(dS, 2));
  const mkState = () => ({ st: mk(n * (K - 1) * 4), S: mk(nV * dS * dS * 4) });
  const D = mkState(), Mc = mkState();
  const x1 = mk(n * 4), co1 = mk(n * 4), b1 = mk(2 * nV * 4), o1 = mk(inner * 4), z1 = mk(inner * 4), y1 = mk(inner * 4);
  const coD = mk((pre + N) * n * 4), oD = mk((pre + N) * inner * 4), yD = mk((pre + N) * inner * 4);
  const step = (S, p) => {   // one decode position on state S
    copy(qz, p * zs, x1, 0, n); copy(qz, p * zs + n, z1, 0, inner); copy(ba, p * 2 * nV, b1, 0, 2 * nV);
    run(W.DN_CONV_WGSL, [x1, S.st, cw, co1, uni([K, 2 * nKH, dS, f(eps)])], [n / dS, 1, 1]);
    run(W.DN_DELTA_WGSL, [co1, S.S, b1, dta, z1, nw, o1, y1, uni([nKH, dS, keyDim, nV, f(eps), 0, 0, 0])], [nV, 1, 1]);
    copy(co1, 0, coD, p * n, n); copy(o1, 0, oD, p * inner, inner); copy(y1, 0, yD, p * inner, inner);
  };
  for (let p = 0; p < pre + N; p++) step(D, p);
  for (let p = 0; p < pre; p++) step(Mc, p);
  const qzs = mk(N * zs * 4), bas = mk(N * 2 * nV * 4);
  copy(qz, pre * zs, qzs, 0, N * zs); copy(ba, pre * 2 * nV, bas, 0, N * 2 * nV);
  const co = mk(N * n * 4), o = mk(N * inner * 4), y = mk(N * inner * 4), F = uni([pre, N, 0, 0]);
  run(M.DN_CONV_MC_WGSL, [qzs, Mc.st, cw, co, uni([K, 2 * nKH, dS, f(eps), zs, n, 0, 0]), F], [n / dS, 1, 1]);
  run(M.DN_DELTA_MC_WGSL, [co, Mc.S, bas, dta, qzs, nw, o, y, uni([nKH, dS, keyDim, nV, f(eps), n, 2 * nV, zs, n, inner, 0, 0]), F], [nV, 1, 1]);
  check(same(await read(coD, N * n, pre * n), await read(co, N * n)) === 0 && same(await read(D.st, n * (K - 1)), await read(Mc.st, n * (K - 1))) === 0,
    `dn_conv_mc (conv + l2): ${N} columns after ${pre} decode steps == decode, state too`);
  check(same(await read(oD, N * inner, pre * inner), await read(o, N * inner)) === 0 && same(await read(yD, N * inner, pre * inner), await read(y, N * inner)) === 0
    && same(await read(D.S, nV * dS * dS), await read(Mc.S, nV * dS * dS)) === 0, `dn_delta_mc (delta rule + gated norm): ${N} columns == decode, S too`);
}

// ---- attention: prep (rope per column) + dense, and gathered over the full list, after 20 decode steps; per KV
// format. f16: the cache words are the f32 run's values rounded to nearest even (f16bits), and the outputs stay
// within f16 rounding of the f32 run's.
const attnRuns = {};
for (const kv of ["f32", "f16"]) {
  const nH = 4, nKV = 2, hd = 256, nRot = 64, maxSeq = 64, pre = 20, N = 7, eps = 1e-6, theta = 1e7, kb = kv === "f16" ? 2 : 4;
  const nQ = 2 * nH * hd, nK = nKV * hd, qs = nQ + 2 * nK, nw = (n) => n * kb / 4;   // cache words of n values
  const ts = Math.fround(Math.pow(theta, -2 / nRot));
  const csOf = (p) => { const F = new Float32Array(64 * 4); let th = Math.fround(p); for (let i = 0; i < nRot / 2; i++) { F[i * 4] = Math.fround(Math.cos(th)); F[i * 4 + 1] = Math.fround(Math.sin(th)); th = Math.fround(th * ts); } return F; };
  seed = 20261012;   // the same data for both formats
  const qkv = put(randF((pre + N) * qs, 2)), qw = put(randF(hd, 2)), kw = put(randF(hd, 2));
  const ap = uni([nH, nKV, hd, nRot, f(eps), 0, 0, 0]);
  const kcD = mk(maxSeq * nK * kb), vcD = mk(maxSeq * nK * kb), kcM = mk(maxSeq * nK * kb), vcM = mk(maxSeq * nK * kb);
  const q1 = mk(nQ * 4), k1 = mk(nK * 4), v1 = mk(nK * 4), qn1 = mk(nH * hd * 4), ao1 = mk(nH * hd * 4);
  const frame = device.createBuffer({ size: W.FRAME_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const aoD = mk(N * nH * hd * 4);
  const step = (kc, vc, p, keep) => {
    const Fr = new Float32Array(W.FRAME_FLOATS);
    new Uint32Array(Fr.buffer, 0, 2).set([p, p + 1]); Fr.set(csOf(p), 4);
    device.queue.writeBuffer(frame, 0, Fr);
    copy(qkv, p * qs, q1, 0, nQ); copy(qkv, p * qs + nQ, k1, 0, nK); copy(qkv, p * qs + nQ + nK, v1, 0, nK);
    run(W.attnPrepWGSL(kv), [q1, k1, v1, qn1, kc, vc, qw, kw, ap, frame], [nH + nKV, 1, 1]);
    run(W.attnDecWGSL(kv), [qn1, kc, vc, q1, ao1, ap, frame], [nH, 1, 1]);
    if (keep >= 0) copy(ao1, 0, aoD, keep * nH * hd, nH * hd);
  };
  for (let p = 0; p < pre + N; p++) step(kcD, vcD, p, p - pre);
  for (let p = 0; p < pre; p++) step(kcM, vcM, p, -1);
  const qkvs = mk(N * qs * 4); copy(qkv, pre * qs, qkvs, 0, N * qs);
  const cs = new Float32Array(N * M.MC_FRAME_COL); for (let c = 0; c < N; c++) cs.set(csOf(pre + c), c * M.MC_FRAME_COL);
  const csb = put(cs), F = uni([pre, N, 0, 0]), ms = uni([qs, nQ, nQ + nK, 0]), qn = mk(N * nH * hd * 4), ao = mk(N * nH * hd * 4), aoG = mk(N * nH * hd * 4);
  run(M.attnPrepMcWGSL(kv), [qkvs, qn, kcM, vcM, qw, kw, csb, ap, ms, F], [nH + nKV, N, 1]);
  run(M.attnDecMcWGSL({ kv }), [qn, kcM, vcM, qkvs, ao, ap, F, ms], [nH, N, 1]);
  const STR = 2052, sel = new Uint32Array(N * STR); for (let c = 0; c < N; c++) { const p = pre + c; sel[c * STR] = p + 1; for (let r = 0; r <= p; r++) sel[c * STR + 1 + r] = r; }
  run(M.attnDecMcWGSL({ gather: STR, kv }), [qn, kcM, vcM, qkvs, aoG, ap, null, ms, put(sel)], [nH, N, 1]);
  const kD = await read(kcD, nw((pre + N) * nK)), vD = await read(vcD, nw((pre + N) * nK));
  check(same(kD, await read(kcM, nw((pre + N) * nK))) === 0 && same(vD, await read(vcM, nw((pre + N) * nK))) === 0,
    `attn_prep_mc (${kv} KV): ${N} columns' K / V cache rows == decode`);
  const aD = await read(aoD, N * nH * hd);
  check(same(aD, await read(ao, N * nH * hd)) === 0, `attn_dec_mc (dense, ${kv} KV): ${N} columns == decode`);
  check(same(aD, await read(aoG, N * nH * hd)) === 0, `attn_dec_mc (gathered, the full row list, ${kv} KV) == dense`);
  attnRuns[kv] = { kD, vD, aD };
}
{
  const a = attnRuns.f32, b = attnRuns.f16;
  const half = (w, i) => (w[i >> 1] >>> ((i & 1) * 16)) & 0xffff, f32 = (u) => new Float32Array(u.buffer, u.byteOffset, u.length);
  let off = 0;
  for (const [x, y] of [[a.kD, b.kD], [a.vD, b.vD]]) { const xf = f32(x); for (let i = 0; i < xf.length; i++) if (half(y, i) !== f16bits(xf[i])) off++; }
  check(off === 0, `f16 KV store: every K / V value == the f32 run's rounded to nearest even (${off} differ)`);
  const xf = f32(a.aD), yf = f32(b.aD);
  let md = 0, mx = 0; for (let i = 0; i < xf.length; i++) { md = Math.max(md, Math.abs(xf[i] - yf[i])); mx = Math.max(mx, Math.abs(xf[i])); }
  check(md / mx < 2e-3, `f16 KV attention output vs f32: max rel ${(md / mx).toExponential(2)} (< 2e-3)`);
}

// ---- PLE: 13 columns from position 5 (ring of 10 slots: a pass longer than the ring) == 18 decode steps
{
  const d = 512, hc = 4, HD = d * hc, K = 4, dil = 3, slots = (K - 1) * dil + 1, pre = 5, N = 13, eps = 1e-6, T = pre + N;
  const R0 = randF(T * HD, 2), pk = put(randF(T * HD, 2)), pv = put(randF(T * d, 2)), pw = put(randF(3 * HD + HD * K, 1));
  const pu = uni([d, K, dil, slots, f(eps), hc, 0, 0]);
  const ringD = mk(slots * HD * 4), ringM = mk(slots * HD * 4), RD = put(R0), RM = put(R0);
  const R1 = mk(HD * 4), pk1 = mk(HD * 4), pv1 = mk(d * 4), dbg = mk((2 * HD + hc) * 4);
  const frame = device.createBuffer({ size: W.FRAME_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const step = (ring, R, p) => {
    device.queue.writeBuffer(frame, 0, new Uint32Array([p, p + 1, 0, 0]));
    copy(R, p * HD, R1, 0, HD); copy(pk, p * HD, pk1, 0, HD); copy(pv, p * d, pv1, 0, d);
    run(W.PLE_WGSL, [R1, pk1, pv1, pw, ring, dbg, pu, frame], [hc, 1, 1]);
    copy(R1, 0, R, p * HD, HD);
  };
  for (let p = 0; p < T; p++) step(ringD, RD, p);
  for (let p = 0; p < pre; p++) step(ringM, RM, p);
  const Rs = mk(N * HD * 4), pks = mk(N * HD * 4), pvs = mk(N * d * 4), F = uni([pre, N, 0, 0]);
  copy(RM, pre * HD, Rs, 0, N * HD); copy(pk, pre * HD, pks, 0, N * HD); copy(pv, pre * d, pvs, 0, N * d);
  run(M.PLE_MC_WGSL, [Rs, pks, pvs, pw, ringM, dbg, pu, F], [hc, 1, 1]);
  check(same(await read(RD, N * HD, pre * HD), await read(Rs, N * HD)) === 0, `ple_mc: ${N} columns from ${pre} == decode`);
  check(same(await read(ringD, slots * HD), await read(ringM, slots * HD)) === 0, `ple_mc: the ring after the pass == decode's`);
}

// ---- greedy head per column
{
  const V = 5000, N = 3, per = 1024, nPart = Math.ceil(V / per);
  const lg = randF(N * V, 10); lg[V + 17] = lg[V + 4000] = 99;   // a tie: the lower index
  const L = put(lg), part = mk(N * nPart * 8), am = mk(N * 8);
  run(M.ARGMAX1_MC_WGSL, [L, part, uni([V, per, nPart, 0])], [nPart, N, 1]);
  run(M.ARGMAX2_MC_WGSL, [part, am, uni([V, per, nPart, 0])], [N, 1, 1]);
  const got = await read(am, 2 * N), want = [0, 1, 2].map((c) => { let b = 0; for (let i = 1; i < V; i++) if (lg[c * V + i] > lg[c * V + b]) b = i; return b; });
  check([0, 1, 2].every((c) => got[2 * c] === want[c]), `argmax_mc: ${N} columns (${[0, 1, 2].map((c) => got[2 * c]).join(", ")}), ties to the lower index`);
}

if (gpuErr) { console.log(`${gpuErr} GPU errors`); fails++; }
console.log(fails ? `Q38FLASH MC FAIL (${fails})` : "Q38FLASH MC PASS ✓");
Deno.exit(fails ? 1 : 0);
