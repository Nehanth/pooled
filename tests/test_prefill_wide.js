// Wide prefill (engine option prefillUbatch, engine/wgsl/gemm_wide.js) vs the default 16-column prefill
// on a real model, same engine, same tokens (engine.prefillWide toggles it):
//   * logits of the token after the prompt: argmax equal and relDiff under prefillTol (load_model.js: 2e-3,
//     the tolerance of tests/test_batch_q38.js, dense; 2e-2 MoE); also both against one-token-at-a-time (sequential) on the shortest prompt;
//   * greedy continuation of GEN tokens: identical to the default path (reported; a difference means the
//     option must stay off by default under the correctness policy);
//   * speculative decoding after a wide prefill: identical to plain decoding after the same prefill;
//   * prefill tok/s both ways.
//   MODEL=27b|moe  PREFILL_UBATCH=256  LENS=150,700,2100  GEN=24  [PREFILL_TILE=json] [WGMEM=0]
//   cd tests && MODEL=27b PREFILL_UBATCH=256 deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights test_prefill_wide.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { argmax } from "../engine/engine.js";
import { openGGUF, gpuDevice, watchGpuErrors, trunkLayers, MOE_PATH, Q38_PATH, wideOpts, prefillTol } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "27b"), GEN = +env("GEN", 24), K = +env("K", 3);
if (!Deno.env.get("PREFILL_UBATCH")) Deno.env.set("PREFILL_UBATCH", "256");
const LENS = env("LENS", "150,700,2100").split(",").map(Number);
const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(MODEL === "moe" ? MOE_PATH : Q38_PATH);
const G = model.G, L = trunkLayers(G), nBlk = G.meta["qwen35.block_count"];
const hasMtp = L < nBlk;
const tok = model.tokenizer();
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: hasMtp });
const maxSeq = Math.ceil((Math.max(...LENS) + GEN + 64) / 256) * 256;
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq, batchCols: 16, coopRowsB: 1, ...wideOpts(),
  // the f32 wide GEMM (its 2e-3 / 2e-2 tolerance); the dp4a GEMM is tests/test_prefill_dp4a.js (PREFILL_DP4A=1 forces it here)
  ...(Deno.env.get("PREFILL_DP4A") ? {} : { prefillDp4a: false }) });
if (!eng.ubatch) { console.log("WIDE FAIL: prefillUbatch did not turn on (see the warning above)"); Deno.exit(1); }
console.log(`${MODEL}: ${L} layers, mtp ${!!eng.mtp}, ubatch ${eng.ubatch}, tile ${JSON.stringify(eng.wideCfg)}, gemm16 ${eng.gemmOn}`);

// a realistic coding prompt: this repo's source
let src = [];
for (const f of ["../engine/qwen35.js", "../engine/gguf.js", "../harness/agent.js"]) src.push(...tok.encode(await Deno.readTextFile(new URL(f, import.meta.url))));
const V = tok.vocab;
const prompt = (n) => {
  const tail = [V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n")];
  const head = [V["<|im_start|>"], ...tok.encode("user\nSummarize what this code does:\n")];
  return [...head, ...src.slice(0, n - head.length - tail.length), ...tail];
};
const rel = (a, b) => { let md = 0, sc = 1e-6; for (let i = 0; i < a.length; i++) { md = Math.max(md, Math.abs(a[i] - b[i])); sc = Math.max(sc, Math.abs(b[i])); } return md / sc; };

async function run(ids, wide, spec = false) {
  eng.reset(); eng.prefillWide = wide; if (eng.mtp) eng.mtpFill = spec;
  const t0 = performance.now();
  await eng.prefillTokens(ids.slice(0, -1));
  let lg = Float32Array.from(await eng.forwardToken(ids.at(-1)));
  const s = (performance.now() - t0) / 1000;
  const first = lg;
  let next = argmax(lg); const gen = [next];
  if (spec && eng.mtp) { while (gen.length < GEN) { for (const t of await eng.specStep(next, argmax, K)) gen.push(t); next = gen.at(-1); } gen.length = GEN; }
  else for (let i = 1; i < GEN; i++) { lg = await eng.forwardToken(next); next = argmax(lg); gen.push(next); }
  return { lg: first, gen, tokps: ids.length / s };
}

let fail = 0, maxRel = 0;
for (const n of LENS) {
  const ids = prompt(n);
  const d = await run(ids, false), w = await run(ids, true);
  const r = rel(w.lg, d.lg); maxRel = Math.max(maxRel, r);
  const sameGen = d.gen.every((t, i) => t === w.gen[i]);
  let line = `${n} tokens: prefill default ${d.tokps.toFixed(1)} tok/s, wide ${w.tokps.toFixed(1)} tok/s (${(w.tokps / d.tokps).toFixed(2)}x) · logits relDiff wide vs default ${r.toExponential(2)} · argmax ${argmax(d.lg)} / ${argmax(w.lg)} · greedy ${GEN} ${sameGen ? "identical" : "DIFFERS"}`;
  if (argmax(d.lg) !== argmax(w.lg) || !(r < prefillTol(!!eng.moe))) fail++;
  if (!sameGen) { line += `\n  default: ${JSON.stringify(tok.decode(d.gen))}\n  wide:    ${JSON.stringify(tok.decode(w.gen))}`; fail++; }
  if (eng.mtp) {
    const s = await run(ids, true, true), same = s.gen.every((t, i) => t === w.gen[i]);
    line += ` · spec after wide prefill ${same ? "identical to plain" : "DIFFERS from plain"}`;
    if (!same) fail++;
  }
  console.log(line);
  if (n === Math.min(...LENS) || env("SEQ_ALL", "") === "1") {   // both prefill paths against one token at a time (SEQ_ALL=1: every length)
    eng.reset(); eng.prefillWide = false; if (eng.mtp) eng.mtpFill = false;
    let lg = null; for (const id of ids) lg = await eng.forwardToken(id);
    const seq = Float32Array.from(lg);
    console.log(`  vs sequential: default prefill ${rel(d.lg, seq).toExponential(2)}, wide prefill ${rel(w.lg, seq).toExponential(2)}`);
  }
}
eng.prefillWide = true;
console.log(`max logits relDiff wide vs default ${maxRel.toExponential(2)}, GPU errors ${errors.count}`);
console.log(fail || errors.count ? "WIDE PREFILL FAIL" : "WIDE PREFILL PASS ✓");
Deno.exit(fail || errors.count ? 1 : 0);
