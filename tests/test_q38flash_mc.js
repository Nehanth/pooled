// The multi-column Qwen3.8-Flash-Next kernels (engine/wgsl/qwen4exp_batch.js) against the decode kernels
// (engine/wgsl/qwen4exp.js) on random data, no model: every column of a multi-column dispatch must give the bits
// the decode kernel gives at that column (for the stateful ones, conv / DeltaNet / PLE, the decode kernel run one
// position after another). Also the gathered attention over the full row list == dense.
//   deno run --unstable-webgpu --allow-read tests/test_q38flash_mc.js
import * as W from "../engine/wgsl/qwen4exp.js";
import * as M from "../engine/wgsl/qwen4exp_batch.js";

const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize, maxBufferSize: ad.limits.maxBufferSize } });
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

// ---- GEMV
for (const kind of ["q4", "q8", "f32"]) for (const mode of [0, 1]) {
  const dOut = 203, dIn = 2560, N = 7, Wt = quant(kind, dOut, dIn), x = put(randF(N * dIn));
  const yd = mk(N * dOut * 4), ym = mk(N * dOut * 4);
  for (let c = 0; c < N; c++) run(W.gemvWGSL(kind), Wt.kind === "f32" ? [Wt.buf, null, x, yd, uni([dOut, dIn, 0, c * dIn / 4, c * dOut, mode, f(0.25), 0])] : [Wt.qs, Wt.sc, x, yd, uni([dOut, dIn, 0, c * dIn / 4, c * dOut, mode, f(0.25), 0])], [Math.ceil(dOut / 8), 1, 1]);
  const F = uni([0, N, 0, 0]);
  run(M.gemvMcWGSL(kind), Wt.kind === "f32" ? [Wt.buf, null, x, ym, uni([dOut, dIn, 0, 0, 0, mode, f(0.25), dIn / 4, dOut, 0, 0, 0]), F] : [Wt.qs, Wt.sc, x, ym, uni([dOut, dIn, 0, 0, 0, mode, f(0.25), dIn / 4, dOut, 0, 0, 0]), F], [Math.ceil(dOut / 8), Math.ceil(N / M.GEMV_MC_COLS), 1]);
  check(same(await read(yd, N * dOut), await read(ym, N * dOut)) === 0, `gemv_mc ${kind} mode ${mode}: ${N} columns == gemv per column`);
}

// ---- MoE: route, gate-up, down (and the shared expert's sigmoid weight)
{
  const nExp = 64, K = 10, d = 256, ff = 128, N = 5;
  const lg = randF(N * nExp, 4); for (let i = 0; i < N * nExp; i += 7) lg[i] = lg[(i + 3) % lg.length];   // some ties
  const L = put(lg), idsD = mk(N * K * 4), wtsD = mk(N * K * 4), idsM = mk(N * K * 4), wtsM = mk(N * K * 4);
  const one = mk(nExp * 4), oi = mk(K * 4), ow = mk(K * 4);
  for (let c = 0; c < N; c++) { copy(L, c * nExp, one, 0, nExp); run(W.MOE_ROUTE_WGSL, [one, oi, ow, uni([nExp, K, 0, 0])], [1, 1, 1]); copy(oi, 0, idsD, c * K, K); copy(ow, 0, wtsD, c * K, K); }
  run(M.MOE_ROUTE_MC_WGSL, [L, idsM, wtsM, uni([nExp, K, 0, 0])], [N, 1, 1]);
  check(same(await read(idsD, N * K), await read(idsM, N * K)) === 0 && same(await read(wtsD, N * K), await read(wtsM, N * K)) === 0, `moe_route_mc: ${N} columns == moe_route per column`);
  for (const kind of ["q4", "q8"]) {
    const g = quant(kind, nExp * ff, d), u = quant(kind, nExp * ff, d), dn = quant(kind, nExp * d, ff), x = put(randF(N * d));
    const hM = mk(N * K * ff * 4), yM = mk(N * d * 4), hD = mk(N * K * ff * 4), yD = mk(N * d * 4);
    const x1 = mk(d * 4), h1 = mk(K * ff * 4), y1 = mk(d * 4);
    for (let c = 0; c < N; c++) {
      copy(x, c * d, x1, 0, d); copy(idsM, c * K, oi, 0, K); copy(wtsM, c * K, ow, 0, K);
      run(W.moeGuWGSL(kind), [g.qs, g.sc, u.qs, u.sc, x1, h1, oi, uni([ff, d, K, 0])], [ff / 8, K, 1]);
      run(W.moeDnWGSL(kind), [dn.qs, dn.sc, h1, y1, oi, ow, uni([d, ff, K, c & 1])], [d / 8, 1, 1]);
      copy(h1, 0, hD, c * K * ff, K * ff); copy(y1, 0, yD, c * d, d);
    }
    run(M.moeGuMcWGSL(kind), [g.qs, g.sc, u.qs, u.sc, x, hM, idsM, uni([ff, d, K, d / 4])], [ff / 8, K, N]);
    // sig per column is a uniform: run the down kernel twice (sig 0 / 1) and keep the matching columns
    const yM1 = mk(N * d * 4);
    run(M.moeDnMcWGSL(kind), [dn.qs, dn.sc, hM, yM, idsM, wtsM, uni([d, ff, K, 0])], [d / 8, 1, N]);
    run(M.moeDnMcWGSL(kind), [dn.qs, dn.sc, hM, yM1, idsM, wtsM, uni([d, ff, K, 1])], [d / 8, 1, N]);
    const a = await read(yD, N * d), b0 = await read(yM, N * d), b1 = await read(yM1, N * d);
    let bad = 0; for (let c = 0; c < N; c++) for (let i = 0; i < d; i++) if (a[c * d + i] !== (c & 1 ? b1 : b0)[c * d + i]) bad++;
    check(same(await read(hD, N * K * ff), await read(hM, N * K * ff)) === 0 && bad === 0, `moe_gu_mc / moe_dn_mc ${kind}: ${N} columns x top-${K} == per column (sig 0 and 1)`);
  }
}

// ---- HC norm / mix / combine
{
  const d = 512, hc = 4, HD = d * hc, N = 6, eps = 1e-6;
  const R0 = randF(N * HD, 2), w = put(randF(HD, 1)), gate = put(randF(N * HD, 3)), a = put(randF(N * d)), b = put(randF(N * d)), inj = put(randF(N * hc, 8));
  const R = put(R0), xnM = mk(N * HD * 4), mixM = mk(N * d * 4);
  const R1 = mk(HD * 4), xn1 = mk(HD * 4), g1 = mk(HD * 4), m1 = mk(d * 4), a1 = mk(d * 4), b1 = mk(d * 4), i1 = mk(16);
  const xnD = mk(N * HD * 4), mixD = mk(N * d * 4), RD = mk(N * HD * 4);
  for (let c = 0; c < N; c++) {
    copy(R, c * HD, R1, 0, HD); copy(gate, c * HD, g1, 0, HD); copy(a, c * d, a1, 0, d); copy(b, c * d, b1, 0, d); copy(inj, c * hc, i1, 0, hc);
    run(W.HC_NORM_WGSL, [R1, w, xn1, uni([d, f(eps), 0, 0])], [hc, 1, 1]);
    run(W.HC_MIX_WGSL, [xn1, g1, m1, uni([d, hc, 0, 0])], [d / 256, 1, 1]);
    run(W.HC_COMBINE_WGSL, [R1, a1, b1, i1, uni([d, hc, 1, 0])], [HD / 256, 1, 1]);
    copy(xn1, 0, xnD, c * HD, HD); copy(m1, 0, mixD, c * d, d); copy(R1, 0, RD, c * HD, HD);
  }
  run(M.HC_NORM_MC_WGSL, [R, w, xnM, uni([d, f(eps), hc, 0])], [hc, N, 1]);
  run(M.HC_MIX_MC_WGSL, [xnM, gate, mixM, uni([d, hc, 0, 0])], [d / 256, N, 1]);
  run(M.HC_COMBINE_MC_WGSL, [R, a, b, inj, uni([d, hc, 1, 0])], [HD / 256, N, 1]);
  check(same(await read(xnD, N * HD), await read(xnM, N * HD)) === 0, `hc_norm_mc: ${N} columns == per column`);
  check(same(await read(mixD, N * d), await read(mixM, N * d)) === 0, `hc_mix_mc: ${N} columns == per column`);
  check(same(await read(RD, N * HD), await read(R, N * HD)) === 0, `hc_combine_mc: ${N} columns == per column`);
}

// ---- DeltaNet: conv (state), l2, delta rule (state), gated norm, over 9 columns after 3 decode steps
{
  const nKH = 2, nV = 6, dS = 128, keyDim = nKH * dS, inner = nV * dS, n = 2 * keyDim + inner, K = 4, N = 9, pre = 3, eps = 1e-6;
  const xin = put(randF((pre + N) * n, 2)), cw = put(randF(n * K, 0.5)), al = put(randF((pre + N) * nV, 2)), be = put(randF((pre + N) * nV, 2));
  const dtb = put(randF(nV)), A = put(Float32Array.from({ length: nV }, () => -Math.abs(rnd()) * 2)), z = put(randF((pre + N) * inner, 2)), nw = put(randF(dS, 2));
  const mkState = () => ({ st: mk(n * (K - 1) * 4), S: mk(nV * dS * dS * 4) });
  const D = mkState(), Mc = mkState();
  const x1 = mk(n * 4), co1 = mk(n * 4), a1 = mk(nV * 4), b1 = mk(nV * 4), o1 = mk(inner * 4), z1 = mk(inner * 4), y1 = mk(inner * 4);
  const coD = mk((pre + N) * n * 4), yD = mk((pre + N) * inner * 4);
  const step = (S, p) => {   // one decode position on state S
    copy(xin, p * n, x1, 0, n); copy(al, p * nV, a1, 0, nV); copy(be, p * nV, b1, 0, nV); copy(z, p * inner, z1, 0, inner);
    run(W.DN_CONV_WGSL, [x1, S.st, cw, co1, uni([n, K, 0, 0])], [Math.ceil(n / 256), 1, 1]);
    run(W.DN_L2_WGSL, [co1, uni([dS, f(eps), 0, 0])], [2 * nKH, 1, 1]);
    run(W.DN_DELTA_WGSL, [co1, S.S, a1, b1, dtb, A, o1, uni([nKH, dS, keyDim, 0])], [nV, 1, 1]);
    run(W.DN_GNORM_WGSL, [o1, z1, nw, y1, uni([dS, f(eps), 0, 0])], [nV, 1, 1]);
    copy(co1, 0, coD, p * n, n); copy(y1, 0, yD, p * inner, inner);
  };
  for (let p = 0; p < pre + N; p++) step(D, p);
  for (let p = 0; p < pre; p++) step(Mc, p);
  const xs = mk(N * n * 4), as = mk(N * nV * 4), bs = mk(N * nV * 4), zs = mk(N * inner * 4);
  copy(xin, pre * n, xs, 0, N * n); copy(al, pre * nV, as, 0, N * nV); copy(be, pre * nV, bs, 0, N * nV); copy(z, pre * inner, zs, 0, N * inner);
  const co = mk(N * n * 4), o = mk(N * inner * 4), y = mk(N * inner * 4), F = uni([pre, N, 0, 0]);
  run(M.DN_CONV_MC_WGSL, [xs, Mc.st, cw, co, uni([n, K, 0, 0]), F], [Math.ceil(n / 256), 1, 1]);
  run(M.DN_L2_MC_WGSL, [co, uni([dS, f(eps), n, 0])], [2 * nKH, N, 1]);
  run(M.DN_DELTA_MC_WGSL, [co, Mc.S, as, bs, dtb, A, o, uni([nKH, dS, keyDim, n, nV, inner, 0, 0]), F], [nV, 1, 1]);
  run(M.DN_GNORM_MC_WGSL, [o, zs, nw, y, uni([dS, f(eps), inner, 0])], [nV, N, 1]);
  check(same(await read(coD, N * n, pre * n), await read(co, N * n)) === 0 && same(await read(D.st, n * (K - 1)), await read(Mc.st, n * (K - 1))) === 0,
    `dn_conv_mc + dn_l2_mc: ${N} columns after ${pre} decode steps == decode, state too`);
  check(same(await read(yD, N * inner, pre * inner), await read(y, N * inner)) === 0 && same(await read(D.S, nV * dS * dS), await read(Mc.S, nV * dS * dS)) === 0,
    `dn_delta_mc + dn_gnorm_mc: ${N} columns == decode, S too`);
}

// ---- attention: prep (rope per column) + dense, and gathered over the full list, after 20 decode steps
{
  const nH = 4, nKV = 2, hd = 256, nRot = 64, maxSeq = 64, pre = 20, N = 7, eps = 1e-6, theta = 1e7;
  const ts = Math.fround(Math.pow(theta, -2 / nRot));
  const csOf = (p) => { const F = new Float32Array(64 * 4); let th = Math.fround(p); for (let i = 0; i < nRot / 2; i++) { F[i * 4] = Math.fround(Math.cos(th)); F[i * 4 + 1] = Math.fround(Math.sin(th)); th = Math.fround(th * ts); } return F; };
  const qf = put(randF((pre + N) * 2 * nH * hd, 2)), kin = put(randF((pre + N) * nKV * hd, 2)), vin = put(randF((pre + N) * nKV * hd, 2));
  const qwArr = randF(hd, 2), kwArr = randF(hd, 2), qw = put(qwArr), kw = put(kwArr), qkw = put(Float32Array.from([...qwArr, ...kwArr]));
  const ap = uni([nH, nKV, hd, nRot, f(eps), 0, 0, 0]);
  const kcD = mk(maxSeq * nKV * hd * 4), vcD = mk(maxSeq * nKV * hd * 4), kcM = mk(maxSeq * nKV * hd * 4), vcM = mk(maxSeq * nKV * hd * 4);
  const q1 = mk(2 * nH * hd * 4), k1 = mk(nKV * hd * 4), v1 = mk(nKV * hd * 4), qn1 = mk(nH * hd * 4), ao1 = mk(nH * hd * 4);
  const frame = device.createBuffer({ size: W.FRAME_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const aoD = mk(N * nH * hd * 4);
  const step = (kc, vc, p, keep) => {
    const Fr = new Float32Array(W.FRAME_FLOATS);
    new Uint32Array(Fr.buffer, 0, 2).set([p, p + 1]); Fr.set(csOf(p), 4);
    device.queue.writeBuffer(frame, 0, Fr);
    copy(qf, p * 2 * nH * hd, q1, 0, 2 * nH * hd); copy(kin, p * nKV * hd, k1, 0, nKV * hd); copy(vin, p * nKV * hd, v1, 0, nKV * hd);
    run(W.ATTN_PREP_WGSL, [q1, k1, v1, qn1, kc, vc, qw, kw, ap, frame], [nH + nKV, 1, 1]);
    run(W.ATTN_DEC_WGSL, [qn1, kc, vc, q1, ao1, ap, frame], [nH, 1, 1]);
    if (keep >= 0) copy(ao1, 0, aoD, keep * nH * hd, nH * hd);
  };
  for (let p = 0; p < pre + N; p++) step(kcD, vcD, p, p - pre);
  for (let p = 0; p < pre; p++) step(kcM, vcM, p, -1);
  const qs = mk(N * 2 * nH * hd * 4), ks = mk(N * nKV * hd * 4), vs = mk(N * nKV * hd * 4);
  copy(qf, pre * 2 * nH * hd, qs, 0, N * 2 * nH * hd); copy(kin, pre * nKV * hd, ks, 0, N * nKV * hd); copy(vin, pre * nKV * hd, vs, 0, N * nKV * hd);
  const cs = new Float32Array(N * M.MC_FRAME_COL); for (let c = 0; c < N; c++) cs.set(csOf(pre + c), c * M.MC_FRAME_COL);
  const csb = put(cs), F = uni([pre, N, 0, 0]), qn = mk(N * nH * hd * 4), ao = mk(N * nH * hd * 4), aoG = mk(N * nH * hd * 4);
  run(M.ATTN_PREP_MC_WGSL, [qs, ks, vs, qn, kcM, vcM, qkw, csb, ap, F], [nH + nKV, N, 1]);
  run(M.attnDecMcWGSL(), [qn, kcM, vcM, qs, ao, ap, F], [nH, N, 1]);
  const STR = 2052, sel = new Uint32Array(N * STR); for (let c = 0; c < N; c++) { const p = pre + c; sel[c * STR] = p + 1; for (let r = 0; r <= p; r++) sel[c * STR + 1 + r] = r; }
  run(M.attnDecMcWGSL({ gather: STR }), [qn, kcM, vcM, qs, aoG, ap, null, put(sel)], [nH, N, 1]);
  check(same(await read(kcD, (pre + N) * nKV * hd), await read(kcM, (pre + N) * nKV * hd)) === 0, `attn_prep_mc: ${N} columns' K / V cache rows == decode`);
  const aD = await read(aoD, N * nH * hd);
  check(same(aD, await read(ao, N * nH * hd)) === 0, `attn_dec_mc (dense): ${N} columns == decode`);
  check(same(aD, await read(aoG, N * nH * hd)) === 0, `attn_dec_mc (gathered, the full row list) == dense`);
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
  const Rs = mk(N * HD * 4), pks = mk(N * HD * 4), pvs = mk(N * d * 4), nb = mk(N * HD * 4), gb = mk(N * HD * 4), F = uni([pre, N, 0, 0]);
  copy(RM, pre * HD, Rs, 0, N * HD); copy(pk, pre * HD, pks, 0, N * HD); copy(pv, pre * d, pvs, 0, N * d);
  run(M.PLE_GATE_MC_WGSL, [Rs, pks, pvs, pw, nb, gb, pu], [hc, N, 1]);
  run(M.PLE_CONV_MC_WGSL, [Rs, gb, pw, ringM, nb, pu, F], [HD / 256, N, 1]);
  run(M.PLE_RING_MC_WGSL, [ringM, nb, pu, F], [HD / 256, N, 1]);
  check(same(await read(RD, N * HD, pre * HD), await read(Rs, N * HD)) === 0, `ple_gate_mc + ple_conv_mc: ${N} columns from ${pre} == decode`);
  check(same(await read(ringD, slots * HD), await read(ringM, slots * HD)) === 0, `ple_ring_mc: the ring after the pass == decode's`);
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
