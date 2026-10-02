// Qwen3.6-35B-A3B (MoE, 256 experts, top-8) on the engine: full model, greedy output vs llama.cpp, tok/s.
// Goldens: llama.cpp b10840 CUDA, same Q4_0 file (bartowski), --temp 0.
// MODEL=122b: Qwen3.5-122B-A10B (bartowski Q4_0, two-file split GGUF, 48 layers, 256 experts top-8, 16 query heads per kv
// head) on the same prompts, goldens from llama.cpp 749f688 CUDA (llama-server, greedy) in tests/golden/q122_llama_greedy.json;
// its weights stream to the GPU a layer at a time (tests/load_model.js streamWeights).
import { Qwen35Engine } from "../engine/qwen35.js";
import { argmax } from "../engine/engine.js";
import { openGGUF, gpuDevice, watchGpuErrors, trunkLayers, MOE_PATH, Q122_PATH, streamWeights, wideOpts } from "./load_model.js";
import { GPU_SAMPLE, ARGMAX_WIDE, gpuGreedy, checkHeadIds } from "./gpusample_check.js";
const N = +(Deno.env.get("TOKENS") || 40), K = +(Deno.env.get("K") || 3);
const MOEFL = Deno.env.get("MOE_FUSED_LAYOUT") ? (Deno.env.get("MOE_FUSED_LAYOUT").startsWith("{") ? JSON.parse(Deno.env.get("MOE_FUSED_LAYOUT")) : Deno.env.get("MOE_FUSED_LAYOUT")) : undefined;   // moeFusedLayout: legacy | wide | JSON (unset: auto)
const MOEK = Deno.env.get("MOE_KERNEL") ? (Deno.env.get("MOE_KERNEL").startsWith("{") ? JSON.parse(Deno.env.get("MOE_KERNEL")) : Deno.env.get("MOE_KERNEL")) : undefined;   // moeKernel: legacy | default | JSON
const BIG = Deno.env.get("MODEL") === "122b";
const PATH = Deno.env.get("MOE") || (BIG ? Q122_PATH : MOE_PATH);
const { device } = await gpuDevice();
watchGpuErrors(device);
const model = openGGUF(PATH);   // converted-weights cache: tests/weight_cache.js (WEIGHT_CACHE=0 disables)
const G = model.G;
const arch = G.meta["general.architecture"], nBlk = G.meta[arch + ".block_count"];
const hasMtp = G.tensors ? Object.keys(G.tensors).some((k) => k.startsWith(`blk.${nBlk - 1}.`)) : false;
const L = trunkLayers(G);
const tok = model.tokenizer();
let t0 = performance.now();
const weights = BIG ? await streamWeights(model, device, { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: hasMtp }, (i) => { if (i % 8 === 7) console.log(`  layer ${i + 1}/${L} on the GPU, ${((performance.now() - t0) / 1000).toFixed(0)} s`); })
  : await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: hasMtp });
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 512,
  // DRAFTCHAIN=0 / SPECFUSE=0: per-submit drafts / separate verify submits (A/B; same output)
  draftChain: Deno.env.get("DRAFTCHAIN") !== "0", specFuse: Deno.env.get("SPECFUSE") !== "0",
  moeNormRouter: Deno.env.get("MOE_NORM_ROUTER") !== "0",   // MOE_NORM_ROUTER=0: rmsnorm + router GEMV launches instead of moe_nrt (A/B)
  layerFuse: { "0": false, "1": true }[Deno.env.get("LAYER_FUSE")],   // LAYER_FUSE=0 / 1: the decode layer fusions off / on (unset: engine default)
  moeFuse: Deno.env.get("MOE_FUSE") !== "0", moeDnRows: +(Deno.env.get("MOE_DN_ROWS") || 1), moeKernel: MOEK, moeFusedLayout: MOEFL,   // MOE_FUSE=0: unfused MoE kernels (A/B)
  // MOEGROUP=U (unset: the engine default, 256 tiled; 0: off): expert-grouped prefill in ubatches of up to U tokens (U a multiple of BCOLS; these prompts are ~25 tokens:
  // with the default BCOLS=4, MOEGROUP=16 puts all but the last 0..7 prompt tokens through it), MOEGROUP_UC: pairs per chunk
  ...(Deno.env.get("BCOLS") ? { batchCols: +Deno.env.get("BCOLS"), coopRowsB: +Deno.env.get("BCOLS") >= 16 ? 1 : 4 } : {}),
  moeGroupPrefill: Deno.env.get("MOEGROUP") === undefined ? undefined : +Deno.env.get("MOEGROUP"), moeGroupUC: +(Deno.env.get("MOEGROUP_UC") || 8), moeGroupTiled: Deno.env.get("MOEGROUP_TILED") === undefined ? undefined : Deno.env.get("MOEGROUP_TILED") === "1", ...wideOpts(),
  // GPU_SAMPLE=0: the logits path (GPU sampling and the two-stage draft argmax are on by default; ARGMAX_WIDE=0|1 alone: only the latter)
  gpuSample: GPU_SAMPLE, argmaxWide: ARGMAX_WIDE });   // PREFILL_UBATCH: wide prefill (these prompts are < 64 tokens: test_prefill_wide.js exercises it)
console.log(`moeGroupPrefill ${eng.moeGrpU || "off"}${eng.moeGrpU ? ` UC ${eng.moeGrpUC}, batchCols ${eng.NC}` : ""}`);
console.log(`draftChain ${!!eng.draftChain}, specFuse ${eng.specFuse}, gpuSample ${eng.gpuSample}, argmaxWide ${eng.argmaxWide}`);
console.log(`${arch}: ${L} layers, mtp tensors ${hasMtp}, engine mtp ${!!eng.mtp}, moeFuse ${eng.moeFuse}; loaded in ${((performance.now() - t0) / 1000).toFixed(0)}s`);
if (eng.moeK) console.log("moeKernel", JSON.stringify(eng.moeK));
if (eng.moeFuse) console.log("moeFusedLayout", JSON.stringify(eng.moe.layout || "legacy"));
const V = tok.vocab;
const chat = (q) => [V["<|im_start|>"], ...tok.encode("user\n" + q), V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n"), V["<think>"], ...tok.encode("\n\n"), V["</think>"], ...tok.encode("\n\n")];
// (plain "The capital of France is" is a near tie after " Paris": "." 19.029 vs "," 18.968 here, llama.cpp CUDA picks ",". Not used as a golden.)
const CASES = BIG ? Object.entries(JSON.parse(Deno.readTextFileSync(new URL("./golden/q122_llama_greedy.json", import.meta.url)))).filter(([k]) => !k.startsWith("_")).map(([name, { q, text, ids }]) => [name, chat(q), text, ids]) : [
  ["two-sum", chat("Write the Python code for two sum. Code only."), "```python\ndef two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:"],
  ["hash-map", chat("Explain what a hash map is in two sentences."), "A hash map is a data structure that stores key-value pairs, allowing for efficient retrieval, insertion, and deletion operations. It uses a hash function to compute an index into an array of buckets or slots"],
  ["bash", chat("Write a bash one-liner that counts lines in all .js files."), "```bash\nfind . -name '*.js' -exec cat {} + | wc -l\n```"],
];
let fail = 0;
for (const [name, prompt, golden, goldenIds] of CASES) {
  eng.reset(); if (eng.mtp) eng.mtpFill = false;
  t0 = performance.now(); await eng.prefillTokens(prompt.slice(0, -1)); let logits = GPU_SAMPLE ? await eng.forwardTokenIds(prompt[prompt.length - 1]) : await eng.forwardToken(prompt[prompt.length - 1]); const pf = (performance.now() - t0) / 1000;
  let next = gpuGreedy(logits); const gen = [next]; const tp0 = performance.now();
  for (let i = 1; i < N; i++) { logits = GPU_SAMPLE ? await eng.forwardTokenIds(next) : await eng.forwardToken(next); next = gpuGreedy(logits); gen.push(next); }
  const ts = (N - 1) / ((performance.now() - tp0) / 1000), text = tok.decode(gen), n = Math.min(text.length, golden.length), ok = n > 20 && text.slice(0, n) === golden.slice(0, n)
    && (!goldenIds || goldenIds.slice(0, N).every((t, i) => gen[i] === t));   // 122B: llama.cpp's token ids too
  console.log(`${name}: prefill ${prompt.length} tok ${pf.toFixed(2)}s · decode ${ts.toFixed(2)} tok/s · ${ok ? "MATCH llama.cpp" : "MISMATCH"}\n  engine: ${JSON.stringify(text)}${ok ? "" : "\n  golden: " + JSON.stringify(golden)}`);
  if (!ok) fail++;
  if (eng.mtp) {
    eng.reset(); eng.mtpFill = true; eng.mtp.stats = { drafts: 0, accepted: 0 };
    await eng.prefillTokens(prompt.slice(0, -1)); logits = await eng.forwardToken(prompt[prompt.length - 1]);
    next = argmax(logits); const spec = [next]; const ts0 = performance.now();
    while (spec.length < N) { for (const t of await eng.specStep(next, GPU_SAMPLE ? gpuGreedy : argmax, K)) spec.push(t); next = spec[spec.length - 1]; }
    const sts = (spec.length - 1) / ((performance.now() - ts0) / 1000), same = gen.every((t, i) => spec[i] === t), st = eng.mtp.stats;
    console.log(`  spec K=${K}: ${sts.toFixed(2)} tok/s, acceptance ${st.accepted}/${st.drafts}, ${same ? "identical to plain" : "DIFFERS from plain"}`);
    if (!same) fail++;
  }
}
if (GPU_SAMPLE) fail += await checkHeadIds(eng) ? 1 : 0;
console.log(fail ? "MOE FAIL" : "MOE PASS ✓"); if (fail) Deno.exit(1);
