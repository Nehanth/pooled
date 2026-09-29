// The no-subgroup decode kernels (phones, Safari) on a room worker's slice of a real model (no embedding or
// head, like a phone's layers), fed random hiddens across the first flash split boundary (faSplit = 256):
//   1. attnHeads (engine/wgsl/qwen35.js attn_flash_h: 1, 2 or 4 query heads per flash workgroup) must give
//      attn_flash's bits: the same schedule of one-token steps (runHidden) and 4-column verifies
//      (runHiddenBatch with snapshots), every output float compared with an attn_flash engine.
//   2. coopWide (engine/wgsl/coop.js wideGEMV) must keep decode == verify where the coop layout has it: 4 tokens decoded one at a time and
//      the same 4 tokens as one 4-column verify pass, from the same state, give the same bits (what makes
//      spec == plain under exact sampling). Its sums are in another order than coop's, so against coop it
//      is a tolerance (relDiff, the tests/test_batch.js measure).
//   MODEL=moe|27b LO=36 HI=40 STEPS=300 HEADS=1,2,4 WIDE='[{"WG":64,"TPR":8,"R":1},{"WG":128,"TPR":16,"R":1}]' CTX=4096
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights test_nosubgroup.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { openGGUF, gpuDevice, watchGpuErrors, Q38_PATH, MOE_PATH } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "moe"), LO = +env("LO", 36), HI = +env("HI", 40), STEPS = +env("STEPS", 300), CTX = +env("CTX", 4096);
const HEADS = env("HEADS", "1,2,4").split(",").filter(Boolean).map(Number);
const WIDE = JSON.parse(env("WIDE", '[{"WG":64,"TPR":8,"R":1},{"WG":128,"TPR":16,"R":1},{"WG":64,"TPR":4,"R":2}]'));
const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(MODEL === "27b" ? Q38_PATH : MOE_PATH);
const G = model.G;
// fresh weights per engine (creation repacks and releases some buffers); the phone's GEMV shape (WG 64, 4 rows)
const mk = async (o) => Qwen35Engine.create({ device, meta: G.meta, weights: await model.weights({ lo: LO, hi: HI, hasEmbed: false, hasHead: false, mtp: false }),
  layerRange: [LO, HI], hasEmbed: false, hasHead: false, maxSeq: CTX, batchCols: 16, coopRowsB: 1, coopWG: 64, coopRows: 4, ...o });
const ref = await mk({});
const D = ref.dims, grp = D.nH / D.nKV;
console.log(`${MODEL} layers ${LO}-${HI - 1}: G = ${grp}, hd ${D.hd}, faSplit ${ref.faSplit}`);
let seed = 12345;
const rnd = (n) => { const x = new Float32Array(n); for (let i = 0; i < n; i++) { seed = (seed * 1103515245 + 12345) >>> 0; x[i] = (seed / 2 ** 32 - 0.5) * 0.5; } return x; };
const sameBits = (a, b) => { const u = new Uint32Array(a.buffer, a.byteOffset, a.length), v = new Uint32Array(b.buffer, b.byteOffset, b.length); for (let i = 0; i < u.length; i++) if (u[i] !== v[i]) return false; return true; };
const relOf = (r, g) => { let md = 0, sc = 1e-6; for (let i = 0; i < r.length; i++) { md = Math.max(md, Math.abs(g[i] - r[i])); sc = Math.max(sc, Math.abs(r[i])); } return md / sc; };
let fail = 0;

// 1. attnHeads: bitwise against attn_flash on a mixed schedule (plain tokens, a 4-column verify every 16 positions)
const sched = [];
for (let p = 0; p < STEPS;) {
  if (p % 16 === 12) { sched.push({ pos: p, xs: rnd(4 * D.dim) }); p += 4; } else { sched.push({ pos: p, x: rnd(D.dim) }); p++; }
}
const trace = async (eng) => {
  eng.reset();
  const out = [];
  for (const s of sched) out.push(s.x ? (await eng.runHidden(s.x, s.pos)).slice() : (await eng.runHiddenBatch(s.xs, s.pos, true)).slice());
  return out;
};
const want = await trace(ref);
for (const hg of HEADS) {
  const eng = await mk({ attnHeads: hg });
  if (eng.faHG !== hg) { console.log(`SKIP attnHeads ${hg}: not available for G = ${grp}`); continue; }
  const got = await trace(eng);
  let bad = 0, first = -1;
  for (let i = 0; i < want.length; i++) if (!sameBits(want[i], got[i])) { bad++; if (first < 0) first = sched[i].pos; }
  console.log(`${bad ? "FAIL" : "PASS"} attnHeads ${hg}: ${sched.length} steps (${sched.filter((s) => s.xs).length} 4-column verifies) to position ${STEPS}, ${bad} differ${bad ? ` (first at position ${first})` : ""}`);
  if (bad) fail++;
  eng.destroy?.();
}

// 2. coopWide: decode == verify bitwise at a few positions, and within tolerance of coop
const prefix = Array.from({ length: STEPS }, () => rnd(D.dim));
const checkAt = [8, 60, STEPS - 4].filter((p) => p >= 0 && p + 4 <= STEPS);
const block = (p) => { const xs = new Float32Array(4 * D.dim); for (let c = 0; c < 4; c++) xs.set(prefix[p + c], c * D.dim); return xs; };
const decodeVsVerify = async (eng) => {
  const res = [];
  for (const p of checkAt) {
    eng.reset(); for (let i = 0; i < p; i++) await eng.runHidden(prefix[i], i);
    const seq = []; for (let c = 0; c < 4; c++) seq.push((await eng.runHidden(prefix[p + c], p + c)).slice());
    eng.reset(); for (let i = 0; i < p; i++) await eng.runHidden(prefix[i], i);
    const bat = (await eng.runHiddenBatch(block(p), p, true)).slice();
    res.push({ p, seq, bat });
  }
  return res;
};
const base = await decodeVsVerify(ref);
// the reference itself (coop, WG 64): decode == verify? (on the GB10 it holds for a full-attention + MoE layer and
// not for a DeltaNet layer, whose batched recurrence already rounds differently; coopWide must not add a mismatch)
let refEq = true;
for (const k of base) for (let c = 0; c < 4; c++) if (!sameBits(k.seq[c], k.bat.subarray(c * D.dim, (c + 1) * D.dim))) refEq = false;
console.log(`INFO coop WG 64 (reference): decode ${refEq ? "==" : "!="} verify bitwise`);
for (const w of WIDE) {
  const eng = await mk({ coopWide: w });
  const r = await decodeVsVerify(eng);
  let ok = true, worst = 0;
  for (let k = 0; k < r.length; k++) for (let c = 0; c < 4; c++) {
    if (!sameBits(r[k].seq[c], r[k].bat.subarray(c * D.dim, (c + 1) * D.dim))) ok = false;
    worst = Math.max(worst, relOf(base[k].seq[c], r[k].seq[c]));
  }
  const tolOk = worst < 2e-3;
  console.log(`${(ok || !refEq) && tolOk ? "PASS" : "FAIL"} coopWide ${JSON.stringify(w)}: decode ${ok ? "==" : "!="} verify bitwise at positions ${checkAt.join(", ")}; relDiff vs coop ${worst.toExponential(2)}`);
  if ((!ok && refEq) || !tolOk) fail++;
}
if (errors.count) fail++;
console.log(fail ? "NOSUBGROUP FAIL" : "NOSUBGROUP PASS");
Deno.exit(fail ? 1 : 0);
