// Every prefill option together vs the pre-2026-09 prefill path (all off), same engine, same tokens:
// tiled prefill attention (attnPrefillTile), wide prefill GEMM (prefillUbatch) and, on the MoE, the
// expert-grouped tiled kernels (moeGroupPrefill; inside the wide chunks when both are on). Runtime
// switches: engine.attnPrefillTile, engine.prefillWide, engine.moeGroup.
//   * logits of the token after the prompt: relDiff vs the all-off path (must be under prefillTol:
//     2e-3 dense, 2e-2 MoE, see load_model.js) and vs one-token-at-a-time (SEQ_ALL=1: at every length,
//     else the shortest and any length over the tolerance), argmax. A length over the tolerance passes when
//     all-on matches one-token-at-a-time instead (the all-off path took a routing near-tie; see below);
//   * greedy continuation of GEN tokens vs the all-off path (reported);
//   * speculative decoding after an all-on prefill: identical to plain decoding after the same prefill;
//   * prefill tok/s both ways (single runs, second of two when REPS=2).
//   MODEL=27b|moe  LENS=150,700,2100  GEN=24  PREFILL_UBATCH (27B: 256; MoE: the engine default)  MOEGROUP (the
//   engine default; tiled attention: the engine default, on)  OPTS=attn,wide,group: the options the "on" run turns on
//   (default all)  BATCH_COLS (16): columns of the all-off batched prefill
//   LIMITS=default: a device with the WebGPU default limits (16 KB workgroup memory, 256 invocations, ...) except
//   BIND_MB (default 256: the MoE's 151 MB expert tensors do not fit the 128 MiB default) and BUF_MB (default 256),
//   i.e. what a phone-class device gives room.js; each option must work there or switch itself off cleanly.
//   BUF_MB=max / BIND_MB=max: the adapter's. The whole MoE needs max (its 508 MB LM head), so a 256 MiB device
//   is tested with LAYERS=N: layers [0, N) + embedding, no head; the last prompt token's hidden is compared.
//   cd tests && MODEL=moe deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights test_prefill_opts.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { argmax } from "../engine/engine.js";
import { openGGUF, gpuDevice, watchGpuErrors, trunkLayers, MOE_PATH, Q38_PATH, wideOpts, prefillTol } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "27b"), GEN = +env("GEN", 24), K = +env("K", 3), REPS = +env("REPS", 1);
if (MODEL !== "moe" && !Deno.env.get("PREFILL_UBATCH")) Deno.env.set("PREFILL_UBATCH", "256");   // dense: opt-in
const LENS = env("LENS", "150,700,2100").split(",").map(Number);
const LIMITS = env("LIMITS", "");
const { device } = LIMITS === "default" ? await (async () => {
  const adapter = await navigator.gpu.requestAdapter(), MB = 2 ** 20;
  const lim = (k, e) => env(e, "256") === "max" ? adapter.limits[k] : Math.min(adapter.limits[k], +env(e, "256") * MB);
  const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: lim("maxBufferSize", "BUF_MB"),
    maxStorageBufferBindingSize: lim("maxStorageBufferBindingSize", "BIND_MB") } });
  const l = device.limits;
  console.log(`device limits: workgroup memory ${l.maxComputeWorkgroupStorageSize} B, invocations ${l.maxComputeInvocationsPerWorkgroup}, storage buffers/stage ${l.maxStorageBuffersPerShaderStage}, binding ${l.maxStorageBufferBindingSize / MB} MiB, buffer ${l.maxBufferSize / MB} MiB, workgroups/dim ${l.maxComputeWorkgroupsPerDimension}`);
  return { device };
})() : await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(MODEL === "moe" ? MOE_PATH : Q38_PATH);
const PART = +env("LAYERS", 0);   // > 0: layers [0, PART) + embedding only (hidden compared, no decode)
const G = model.G, L = PART || trunkLayers(G), nBlk = G.meta["qwen35.block_count"];
const hasMtp = !PART && L < nBlk, hasHead = !PART;
const tok = model.tokenizer();
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead, mtp: hasMtp });
const maxSeq = Math.ceil((Math.max(...LENS) + GEN + 64) / 256) * 256;
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead, maxSeq, batchCols: +env("BATCH_COLS", 16), coopRowsB: 1,
  layerFuse: { "0": false, "1": true }[env("LAYER_FUSE", "")],   // LAYER_FUSE=0 / 1: decode layer fusion off / on (unset: engine default)
  ...wideOpts(), ...(env("MOEGROUP") ? { moeGroupPrefill: +env("MOEGROUP") } : {}), ...(env("MOEGROUP_UC") ? { moeGroupUC: +env("MOEGROUP_UC") } : {}) });
const TOL = prefillTol(!!eng.moe);
// OPTS=attn,wide,group (default all): which options the "on" run turns on, to find which one a difference comes from
const OPTS = new Set(env("OPTS", "attn,wide,group").split(","));
const set = (on) => { eng.attnPrefillTile = on && OPTS.has("attn") && !!eng.attnPTCfg; eng.prefillWide = on && OPTS.has("wide") && eng.ubatch > 0; eng.moeGroup = on && OPTS.has("group") && eng.moeGrpU > 0; };
console.log(`${MODEL}: ${L} layers, mtp ${!!eng.mtp}, attnPrefillTile ${!!eng.attnPTCfg}${eng.attnPTCfg ? ` (TK ${eng.attnPTCfg.TK})` : ""}, ubatch ${eng.ubatch}, moeGroupPrefill ${eng.moeGrpU || "off"}${eng.moeGrpU ? ` UC ${eng.moeGrpUC} tiled ${eng.moeGrpTiled}` : ""}`);

// The code the chat prompt summarizes is a frozen fixture (golden/prefill_opts_prompt.txt: the first 400 lines
// of engine/qwen35.js as of v1.0.0, ~8k tokens, repeated past that). It used to be read live from
// engine/qwen35.js, so any edit there changed the prompts and the MoE's on-vs-off relDiff with them (700 tokens:
// 1.5e-3 at v1.0.0, 3.0e-2 after the GPU sampling merge, with identical kernels: routing near-ties).
let src = tok.encode(await Deno.readTextFile(new URL("./golden/prefill_opts_prompt.txt", import.meta.url)));
while (src.length < Math.max(...LENS)) src = src.concat(src);
const V = tok.vocab;
// PROMPT_FILE=path (relative to tests/): that file's raw tokens repeated to each length, no chat template (the
// Chrome bench's ?prefilllen prompt is its page's HTML repeated this way)
const PF = env("PROMPT_FILE", "");
let raw = PF ? tok.encode(await Deno.readTextFile(new URL(PF, import.meta.url))) : null;
const prompt = (n) => {
  if (raw) { while (raw.length < n) raw = raw.concat(raw); return raw.slice(0, n); }
  const tail = [V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n")];
  const head = [V["<|im_start|>"], ...tok.encode("user\nSummarize what this code does:\n")];
  return [...head, ...src.slice(0, n - head.length - tail.length), ...tail];
};
const rel = (a, b) => { let md = 0, sc = 1e-6; for (let i = 0; i < a.length; i++) { md = Math.max(md, Math.abs(a[i] - b[i])); sc = Math.max(sc, Math.abs(b[i])); } return md / sc; };

async function run(ids, on, spec = false) {
  let out;
  for (let r = 0; r < (spec ? 1 : REPS); r++) {
    eng.reset(); set(on); if (eng.mtp) eng.mtpFill = spec;
    const t0 = performance.now();
    await eng.prefillTokens(ids.slice(0, -1));
    if (PART) {   // no head: the last prompt token's hidden after layer PART - 1
      const h = Float32Array.from(await eng.embedRun(ids.at(-1), eng.pos));
      out = { lg: h, gen: [], tokps: ids.length / ((performance.now() - t0) / 1000) };
      continue;
    }
    let lg = Float32Array.from(await eng.forwardToken(ids.at(-1)));
    const s = (performance.now() - t0) / 1000;
    const first = lg;
    let next = argmax(lg); const gen = [next];
    if (spec && eng.mtp) { while (gen.length < GEN) { for (const t of await eng.specStep(next, argmax, K)) gen.push(t); next = gen.at(-1); } gen.length = GEN; }
    else for (let i = 1; i < GEN; i++) { lg = await eng.forwardToken(next); next = argmax(lg); gen.push(next); }
    out = { lg: first, gen, tokps: ids.length / s };
  }
  return out;
}

// The one-token-at-a-time logits break ties: on the MoE a 1-ulp difference can flip one token's top-8 routing
// (a near-tie) in any one of the three paths, which moves the last logits by up to ~3e-2 at isolated prompt
// lengths and not at the lengths next to them. If all-on vs all-off is over the tolerance but all-on matches
// one-token-at-a-time (relDiff and argmax), the all-off path is the outlier and the options are not at fault.
const seqLogits = async (ids) => {
  eng.reset(); set(false); if (eng.mtp) eng.mtpFill = false;
  let lg = null; for (const id of ids) lg = await eng.forwardToken(id);
  return Float32Array.from(lg);
};
let fail = 0, maxRel = 0, tieBreaks = 0;
for (const n of LENS) {
  const ids = prompt(n);
  const d = await run(ids, false), w = await run(ids, true);
  const r = rel(w.lg, d.lg); maxRel = Math.max(maxRel, r);
  const sameGen = d.gen.every((t, i) => t === w.gen[i]);
  let line = `${n} tokens: prefill all-off ${d.tokps.toFixed(1)} tok/s, all-on ${w.tokps.toFixed(1)} tok/s (${(w.tokps / d.tokps).toFixed(2)}x) · logits relDiff on vs off ${r.toExponential(2)} · argmax ${argmax(d.lg)} / ${argmax(w.lg)} · greedy ${GEN} ${sameGen ? "identical" : "DIFFERS"}`;
  if (PART) {
    if (!(r < TOL)) fail++;
    console.log(`${n} tokens (layers [0, ${L}) + embed): prefill all-off ${d.tokps.toFixed(1)} tok/s, all-on ${w.tokps.toFixed(1)} tok/s · hidden relDiff on vs off ${r.toExponential(2)}`); continue;
  }
  if (!sameGen) line += `\n  off: ${JSON.stringify(tok.decode(d.gen))}\n  on:  ${JSON.stringify(tok.decode(w.gen))}`;
  if (eng.mtp) {
    const s = await run(ids, true, true), same = s.gen.every((t, i) => t === w.gen[i]);
    line += ` · spec after all-on prefill ${same ? "identical to plain" : "DIFFERS from plain"}`;
    if (!same) fail++;
  }
  console.log(line);
  const over = !(r < TOL) || argmax(d.lg) !== argmax(w.lg);
  if (over || n === Math.min(...LENS) || env("SEQ_ALL", "") === "1") {
    const seq = await seqLogits(ids), rOff = rel(d.lg, seq), rOn = rel(w.lg, seq);
    console.log(`  vs one-token-at-a-time: all-off prefill ${rOff.toExponential(2)}, all-on prefill ${rOn.toExponential(2)}`);
    if (over) {
      if (rOn < TOL && argmax(w.lg) === argmax(seq)) { tieBreaks++; console.log(`  all-on matches one-token-at-a-time: the all-off path is the outlier at ${n} tokens (routing near-tie), not counted`); }
      else fail++;
    }
  }
}
set(true);
console.log(`max logits relDiff all-on vs all-off ${maxRel.toExponential(2)} (tolerance ${TOL}${tieBreaks ? `; ${tieBreaks} length(s) over it with all-off the outlier` : ""}), GPU errors ${errors.count}`);
console.log(fail || errors.count ? "PREFILL OPTS FAIL (argmax / relDiff / spec / GPU errors)" : "PREFILL OPTS PASS (argmax, relDiff under tolerance, spec == plain)");
Deno.exit(fail || errors.count ? 1 : 0);
