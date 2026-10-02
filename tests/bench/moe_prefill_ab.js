// MoE prefill tok/s with the room's engine settings, A/B over runtime switches and a skip-family profile.
// Each run prefills a fresh prompt (a slice of this repo's source) of N tokens after reset() and takes the
// first logits (prefillTokens(n - 1) + forwardToken), like tests/bench_dp4a_prefill.js.
//   LENS=512,2048,8192 RUNS=2
//   MODES='{"base":{},"dp4a":{"prefillDp4a":true}}'  engine properties set before each run (runtime switches)
//   OPTS='{"moeGroupUC":8}'                           extra Qwen35Engine.create options
//   SKIP=1: also time each kernel family skipped (outputs are wrong while skipping; only time matters),
//           at each length, in the first mode. The family's share = (base - skipped) / base.
//   MODEL=moe|27b
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights bench/moe_prefill_ab.js
import { Qwen35Engine } from "../../engine/qwen35.js";
import { argmax } from "../../engine/engine.js";
import { roomQwen35Options } from "../../engine/preset.js";
import { openGGUF, gpuDevice, watchGpuErrors, trunkLayers, MOE_PATH, Q38_PATH, roomFlags } from "../load_model.js";
const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "moe"), LENS = env("LENS", "512,2048,8192").split(",").map(Number), RUNS = +env("RUNS", 2);
const MODES = JSON.parse(env("MODES", '{"base":{}}')), OPTS = JSON.parse(env("OPTS", "{}"));
const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(MODEL === "moe" ? MOE_PATH : Q38_PATH);
const G = model.G, L = trunkLayers(G), nBlk = G.meta["qwen35.block_count"];
const tok = model.tokenizer();
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: L < nBlk });
const maxSeq = Math.ceil((Math.max(...LENS) + 64) / 256) * 256;
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq,
  ...roomQwen35Options(roomFlags()), ...OPTS });
console.log(`${MODEL}: ${L} layers, ubatch ${eng.ubatch}, dp4a ${eng.prefillDp4a}, moeGroup ${eng.moeGrpU} UC ${eng.moeGrpUC}, opts ${JSON.stringify(OPTS)}`);
let src = [];
for (const f of ["../engine/qwen35.js", "../engine/gguf.js", "../harness/agent.js", "../engine/wgsl/qwen35.js", "../engine/wgsl/moe.js", "../room.js"])
  try { src.push(...tok.encode(await Deno.readTextFile(new URL("../" + f, import.meta.url)))); } catch { /* optional */ }
let off = 0;
const fresh = (n) => { if (off + n > src.length) off = 0; const s = src.slice(off, off + n); off += 997; return s; };
const rel = (a, b) => { let md = 0, sc = 1e-6; for (let i = 0; i < a.length; i++) { md = Math.max(md, Math.abs(a[i] - b[i])); sc = Math.max(sc, Math.abs(b[i])); } return md / sc; };
const saved = {};
const set = (m) => { for (const [k, v] of Object.entries(MODES[m])) { if (!(k in saved)) saved[k] = eng[k]; } for (const k in saved) eng[k] = k in MODES[m] ? MODES[m][k] : saved[k]; };
// dispatch count of one prefill
let nDisp = 0;
const oc = device.createCommandEncoder.bind(device);
device.createCommandEncoder = (d) => {
  const e = oc(d), ob = e.beginComputePass.bind(e);
  e.beginComputePass = (pd) => { const p = ob(pd), od = p.dispatchWorkgroups.bind(p), oi = p.dispatchWorkgroupsIndirect.bind(p);
    p.dispatchWorkgroups = (...a) => { nDisp++; od(...a); }; p.dispatchWorkgroupsIndirect = (...a) => { nDisp++; oi(...a); }; return p; };
  return e;
};
const run = async (ids) => {
  eng.reset(); if (eng.mtp) eng.mtpFill = true;
  await device.queue.onSubmittedWorkDone();
  const t0 = performance.now();
  await eng.prefillTokens(ids.slice(0, -1)); const l = await eng.forwardToken(ids.at(-1));
  return { ms: performance.now() - t0, l: Float32Array.from(l) };
};
// DOUBLE=pipe1,pipe2: dispatch those pipelines twice (the second launch's cost = that kernel's real cost in the stream)
if (env("DOUBLE")) {
  const dbl = new Set(env("DOUBLE").split(",")), ox = eng._dxyz.bind(eng), oi = eng._dInd.bind(eng), om = eng._dMC.bind(eng);
  eng._dxyz = (p, n, ...a) => { ox(p, n, ...a); if (dbl.has(n) && eng.dbl !== false) ox(p, n, ...a); };
  eng._dInd = (p, n, ...a) => { oi(p, n, ...a); if (dbl.has(n) && eng.dbl !== false) oi(p, n, ...a); };
  eng._dMC = (p, n, ...a) => { om(p, n, ...a); if (dbl.has(n) && eng.dbl !== false) om(p, n, ...a); };
}
const names = Object.keys(MODES);
set(names[0]); await run(fresh(600));
for (const m of names.slice(1)) { set(m); await run(fresh(300)); }
const out = {};
for (const n of LENS) {
  const ids = fresh(n), lg = {};
  for (const m of names) { set(m); nDisp = 0; lg[m] = (await run(ids)).l; out[`${m}_${n}_disp`] = nDisp; }
  console.log(`${n}: ` + names.map((m) => `${m} disp ${out[`${m}_${n}_disp`]}` + (m === names[0] ? "" : ` relDiff ${rel(lg[m], lg[names[0]]).toExponential(2)} argmax ${argmax(lg[m])}/${argmax(lg[names[0]])}`)).join(" · "));
  for (const m of names) {
    const r = out[`${m}_${n}`] = [];
    for (let k = 0; k < RUNS; k++) { set(m); r.push(+(n / ((await run(fresh(n))).ms / 1000)).toFixed(1)); }
    console.log(`  ${n} ${m.padEnd(10)} prefill ${r.join(" & ")} tok/s`);
  }
  if (env("SKIP")) {
    set(names[0]);
    const pipes = Object.keys(eng.pipes);
    const fam = {
      "experts gate/up (moe_gusg)": /^moe_gusg/, "experts down + combine (moe_dng, moe_combw)": /^moe_(dng|combw)/, "expert sort (moe_gsort)": /^moe_gsort$/,
      "router (moe_nrt, moe_route)": /^moe_(nrt|route|router)$/, "wide GEMMs (proj, dn_out, attn_out)": /^(gemm_w_|gemm_d_|quant_q8_w)/,
      "attention core (attn_flash*, combine)": /^(attn_flash|attn_combine|kv_store)/, "attention glue": /^(attn_glue|qsplit|head_norm|rope_part|sigmoid_mul)/,
      "DeltaNet recurrence (dn_delta)": /^dn_delta/, "DeltaNet other (dn_pre, conv, gatenorm)": /^dn_(?!delta)/,
      "batched GEMV (beta/alpha, MTP)": /^matvec/, "fused per-pair MoE (MTP: moe_gus/moe_dnc)": /^moe_(gus|dnc)_/, "norms": /^(rmsnorm|add_res|silu)/, "ALL kernels (floor: encode, submit, sync)": /./,
    };
    const base = [];
    for (let k = 0; k < RUNS; k++) base.push((await run(fresh(n))).ms);
    const b = Math.min(...base);
    console.log(`  skip profile ${n}: base ${b.toFixed(0)} ms`);
    for (const [f, re] of Object.entries(fam)) {
      const sk = pipes.filter((p) => re.test(p)); if (!sk.length) continue;
      eng.skip = new Set(sk);
      const t = []; for (let k = 0; k < RUNS; k++) t.push((await run(fresh(n))).ms);
      eng.skip = null;
      const s = Math.min(...t);
      console.log(`    ${f.padEnd(48)} ${(b - s).toFixed(0).padStart(6)} ms ${((b - s) / b * 100).toFixed(1).padStart(5)}%`);
    }
    { eng.mtpFill = false; const t = []; for (let k = 0; k < RUNS; k++) { eng.reset(); eng.mtpFill = false; await device.queue.onSubmittedWorkDone(); const ids2 = fresh(n), t0 = performance.now(); await eng.prefillTokens(ids2.slice(0, -1)); await eng.forwardToken(ids2.at(-1)); t.push(performance.now() - t0); }
      const s = Math.min(...t); console.log(`    ${"MTP draft-cache fill (mtpFill = false)".padEnd(48)} ${(b - s).toFixed(0).padStart(6)} ms ${((b - s) / b * 100).toFixed(1).padStart(5)}%`); }
  }
}
console.log("RESULT " + JSON.stringify({ model: MODEL, opts: OPTS, out, gpuErrors: errors.count }));
Deno.exit(errors.count ? 1 : 0);
