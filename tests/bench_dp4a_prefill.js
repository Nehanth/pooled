// Prefill A/B for the dp4a wide GEMM on a real model, one process, room engine settings (engine/preset.js):
// modes "narrow" (16-column passes), "wide" (f32 wide GEMM) and "dp4a" (int8 wide GEMM), switched at runtime
// on one engine (engine.prefillWide / engine.prefillDp4a). Each run prefills a fresh prompt (a different
// slice of this repo's source) of N tokens after reset(); prints tok/s per run, the next-token logits
// relDiff of wide and dp4a vs narrow, and plain decode tok/s after each prefill.
//   MODEL=27b|moe LENS=512,2048,8192 RUNS=2 MODES=narrow,wide,dp4a DECODE=32
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights bench_dp4a_prefill.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { argmax } from "../engine/engine.js";
import { roomQwen35Options } from "../engine/preset.js";
import { openGGUF, gpuDevice, watchGpuErrors, trunkLayers, MOE_PATH, Q38_PATH, roomFlags } from "./load_model.js";
const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "27b"), LENS = env("LENS", "512,2048,8192").split(",").map(Number), RUNS = +env("RUNS", 2);
const MODES = env("MODES", "narrow,wide,dp4a").split(","), DEC = +env("DECODE", 32);
const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(MODEL === "moe" ? MOE_PATH : Q38_PATH);
const G = model.G, L = trunkLayers(G), nBlk = G.meta["qwen35.block_count"];
const tok = model.tokenizer();
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: L < nBlk });
const maxSeq = Math.ceil((Math.max(...LENS) + DEC + 64) / 256) * 256;
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq,
  ...roomQwen35Options(roomFlags()), prefillUbatch: 256, prefillDp4a: true });
console.log(`${MODEL}: ${L} layers, ubatch ${eng.ubatch}, dp4a ${JSON.stringify(eng.dp4aCfg)}, moeGroup ${eng.moeGrpU}`);
let src = [];
for (const f of ["../engine/qwen35.js", "../engine/gguf.js", "../harness/agent.js", "../engine/wgsl/qwen35.js", "../engine/wgsl/moe.js", "../app/room.js"])
  try { src.push(...tok.encode(await Deno.readTextFile(new URL(f, import.meta.url)))); } catch { /* optional */ }
let off = 0;
const fresh = (n) => { if (off + n > src.length) off = 0; const s = src.slice(off, off + n); off += 997; return s; };
const rel = (a, b) => { let md = 0, sc = 1e-6; for (let i = 0; i < a.length; i++) { md = Math.max(md, Math.abs(a[i] - b[i])); sc = Math.max(sc, Math.abs(b[i])); } return md / sc; };
const set = (m) => { eng.prefillWide = m !== "narrow"; eng.prefillDp4a = m === "dp4a"; };
const out = {};
set("dp4a"); eng.reset(); await eng.prefillTokens(fresh(600)); await eng.forwardToken(1);   // warm-up (compiles)
set("narrow"); eng.reset(); await eng.prefillTokens(fresh(64)); await eng.forwardToken(1);
for (const n of LENS) {
  // accuracy: one prompt, all modes
  const ids = fresh(n), lg = {};
  for (const m of MODES) { set(m); eng.reset(); if (eng.mtp) eng.mtpFill = true; await eng.prefillTokens(ids.slice(0, -1)); lg[m] = Float32Array.from(await eng.forwardToken(ids.at(-1))); }
  const acc = MODES.filter((m) => m !== "narrow" && lg.narrow).map((m) => `${m} vs narrow relDiff ${rel(lg[m], lg.narrow).toExponential(2)} argmax ${argmax(lg[m])}/${argmax(lg.narrow)}`).join(" · ");
  console.log(`${n}: ${acc}`);
  for (const m of MODES) {
    const r = out[`${m}_${n}`] = { prefill: [], decode: [] };
    for (let k = 0; k < RUNS; k++) {
      set(m); eng.reset(); if (eng.mtp) eng.mtpFill = true;
      const p = fresh(n);
      await device.queue.onSubmittedWorkDone();
      const t0 = performance.now();
      await eng.prefillTokens(p.slice(0, -1)); let l = await eng.forwardToken(p.at(-1));
      const t1 = performance.now();
      let nx = argmax(l);
      for (let i = 0; i < DEC; i++) nx = argmax(await eng.forwardToken(nx));
      const t2 = performance.now();
      r.prefill.push(+(n / ((t1 - t0) / 1000)).toFixed(1)); r.decode.push(+(DEC / ((t2 - t1) / 1000)).toFixed(2));
    }
    console.log(`  ${n} ${m.padEnd(6)} prefill ${r.prefill.join(" & ")} tok/s · decode ${r.decode.join(" & ")} tok/s`);
  }
}
console.log("RESULT " + JSON.stringify({ model: MODEL, out, gpuErrors: errors.count }));
Deno.exit(errors.count ? 1 : 0);
