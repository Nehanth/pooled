// GPU sampling (exp/gpu-sample) through the whole engine on the synthetic Qwen 3.5 model (dense and
// MoE, tests/e2e/synth.mjs): with gpuSample + argmaxWide on,
//   * headFromHiddenIds greedy == greedy(headFromHidden) on 256 random hiddens, and its top-40 ==
//     the top 40 of the same logits (sorted value desc, index asc), with bad = 0
//   * forwardTokenIds greedy decoding == forwardToken + argmax (plain)
//   * specStep with a .gpu greedy sampler == plain, for the fused one-submit path, the separate
//     path (specFuse off, draftChain off), the runTrunk path (verifyN -> headBatchIds) and prompt
//     lookup drafts (specStepDrafts)
//   * argmaxWide on vs off: the same speculative tokens and acceptance (drafts identical)
// Runs on any WebGPU adapter; on a CPU Vulkan driver it needs no GPU:
//   CPU_ONLY=1 WGPU_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
//     deno run --allow-read --allow-env --unstable-webgpu tests/e2e/gpusample_synth.mjs
import { Qwen35Engine } from "../../engine/qwen35.js";
import { parseGGUFHeader, qwen35Weights, GGML_EMBED } from "../../engine/gguf.js";
import { argmax } from "../../engine/engine.js";
import { buildSynthGGUF, SYNTH_MOE } from "./synth.mjs";
import { topkNaive, readCands } from "../../engine/topk.js";

const adapter = await navigator.gpu.requestAdapter();
const ai = adapter.info || {};
console.log("adapter:", ai.vendor, ai.architecture, ai.device, ai.description);
if (Deno.env.get("CPU_ONLY") === "1" && !/llvmpipe|lavapipe|swiftshader|cpu/i.test([ai.vendor, ai.architecture, ai.device, ai.description].join(" "))) { console.log("not a CPU adapter; exiting"); Deno.exit(2); }
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
const errs = []; device.addEventListener?.("uncapturederror", (e) => errs.push(e.error?.message));

let fail = 0;
const check = (name, ok, d = "") => { console.log(`${ok ? "PASS" : "FAIL"} ${name}${d ? "  " + d : ""}`); if (!ok) fail++; };
const G1 = Object.assign((x) => (x && x.ids instanceof Uint32Array ? x.ids[0] : argmax(x)), { gpu: { kind: "greedy" } });
let seed = 99;
const rnd = () => { let t = (seed = (seed + 0x6d2b79f5) | 0); t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32; };

for (const [label, opts] of [["dense", {}], ["moe", { moe: SYNTH_MOE }]]) {
  const { bytes } = buildSynthGGUF(opts);
  const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const G = parseGGUFHeader(buf), M = G.meta, L = M["qwen35.block_count"] - 1;
  const bytesOf = async (i) => new Uint8Array(buf, i.byteOffset, i.byteLength).slice();
  const mk = async (o = {}) => Qwen35Engine.create({ device, meta: M, layerRange: [0, L], hasEmbed: true, hasHead: true, vocab: G.tensors[GGML_EMBED].shape[0],
    maxSeq: 512, batchCols: 16, coopRowsB: 1, coopWG: 64, gpuSample: true, argmaxWide: true, ...o,
    weights: await qwen35Weights(G, bytesOf, { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true }) });
  const eng = await mk();
  const { dim, vocab } = eng.dims;
  console.log(`--- ${label}: vocab ${vocab}, dim ${dim}, ${L} layers, mtp ${!!eng.mtp}`);

  // 1. head on random hiddens
  let bad = 0, badTop = 0;
  for (let i = 0; i < 256; i++) {
    const h = Float32Array.from({ length: dim }, () => (rnd() * 2 - 1) * 3);
    const lg = await eng.headFromHidden(h);
    const g = await eng.headFromHiddenIds(h, { kind: "greedy" });
    if (g.ids[0] !== argmax(lg) || g.bad !== 0 || g.vals[0] !== lg[g.ids[0]]) bad++;
    if (i % 8 === 0) {
      const t = await eng.headFromHiddenIds(h, { kind: "topk", k: 40, temp: 0.8 });
      const ref = readCands(topkNaive(lg, { n: vocab, k: 40 }), 0, 40);
      if (t.ids.join() !== ref.ids.join() || t.vals.join() !== ref.vals.join()) badTop++;
    }
  }
  check(`${label}: headFromHiddenIds greedy == argmax(headFromHidden), 256 hiddens`, bad === 0, `${bad} differ`);
  check(`${label}: headFromHiddenIds top-40 == sorted logits, 32 hiddens`, badTop === 0, `${badTop} differ`);

  // 2. plain decoding
  const prompt = Array.from({ length: 40 }, (_, i) => 33 + ((i * 7919) % 90));
  const N = 24;
  const start = async (e, ids) => { e.reset(); if (e.mtp) e.mtpFill = true; await e.prefillTokens(prompt.slice(0, -1)); return ids ? (await e.forwardTokenIds(prompt.at(-1))).ids[0] : argmax(await e.forwardToken(prompt.at(-1))); };
  let t = await start(eng, false); const plain = [t];
  for (let i = 1; i < N; i++) { t = argmax(await eng.forwardToken(t)); plain.push(t); }
  t = await start(eng, true); const plainIds = [t];
  for (let i = 1; i < N; i++) { t = (await eng.forwardTokenIds(t)).ids[0]; plainIds.push(t); }
  check(`${label}: forwardTokenIds greedy == forwardToken + argmax (${N} tokens)`, plainIds.join() === plain.join());
  // decode-ahead (forwardTokenIds queues the next token on its top-1): a token that is not the top-1, and another
  // engine call while a step is queued, must give exactly what the engine gives with decodeAhead off
  const off = async (feed) => {   // feed(i, top1) -> the token to feed at step i
    const lg = [], ids = [];
    let c = await start(eng, true); ids.push(c.ids ? c.ids[0] : c);
    for (let i = 1; i < N; i++) { const f = feed(i, ids.at(-1)); const r = await eng.forwardTokenIds(f); ids.push(r.ids[0]); lg.push(r.vals[0]); if (i === 12) { const x = await eng.forwardToken(ids.at(-1)); lg.push(argmax(x)); eng.pos--; } }
    return ids.join() + "|" + lg.join();
  };
  const feed = (i, top) => (i % 5 === 3 ? (top + 7) % 90 + 33 : top);
  eng.decodeAhead = false; const refA = await off(feed);
  eng.decodeAhead = true; const gotA = await off(feed);
  check(`${label}: decode-ahead == off with off-top-1 tokens and a forwardToken in between`, gotA === refA);

  // 3. speculative, several paths
  const spec = async (e, sample, o = {}) => {
    let next = await start(e, false); const out = [next];
    e.mtp.stats = { drafts: 0, accepted: 0 };
    while (out.length < N) { const s = await e.specStep(next, sample, 3, o); out.push(...s); next = s.at(-1); }
    return { toks: out.slice(0, N), acc: `${e.mtp.stats.accepted}/${e.mtp.stats.drafts}` };
  };
  const a0 = await spec(eng, argmax), a1 = await spec(eng, G1);
  check(`${label}: specStep (fused one-submit) GPU greedy == plain`, a1.toks.join() === plain.join(), `acceptance ${a1.acc}`);
  check(`${label}: specStep logits path == plain`, a0.toks.join() === plain.join(), `acceptance ${a0.acc}`);
  check(`${label}: GPU-sampled spec has the same acceptance as the logits path`, a0.acc === a1.acc, `${a0.acc} vs ${a1.acc}`);
  const runTrunk = (toks, pos) => eng.embedRunBatch(toks, pos, true);
  const a2 = await spec(eng, G1, { runTrunk });
  check(`${label}: specStep with runTrunk (verifyN -> headBatchIds) == plain`, a2.toks.join() === plain.join(), `acceptance ${a2.acc}`);
  eng.specFuse = false; eng.chainOn = false;
  const a3 = await spec(eng, G1);
  check(`${label}: specStep separate submits (specFuse off, chain off) == plain`, a3.toks.join() === plain.join(), `acceptance ${a3.acc}`);
  eng.specFuse = true; eng.chainOn = true;
  // prompt-lookup style drafts (plain's own next tokens: long accepted runs through headBatchIds / fused)
  let nx = await start(eng, false); const vd = [nx];
  while (vd.length < N) { const i = vd.length - 1; const o = await eng.specStepDrafts(nx, G1, plain.slice(i + 1, i + 8)); vd.push(...o); nx = o.at(-1); }
  check(`${label}: specStepDrafts GPU greedy == plain`, vd.slice(0, N).join() === plain.join());
  // argmaxWide off: drafts from the old argmax, same result and acceptance
  const eOld = await mk({ argmaxWide: false, gpuSample: false });
  const b0 = await spec(eOld, argmax);
  check(`${label}: argmaxWide on vs off: same tokens and acceptance`, b0.toks.join() === a0.toks.join() && b0.acc === a0.acc, `${b0.acc} (old) vs ${a0.acc} (wide)`);
  // the drafts themselves (the synthetic draft head rarely guesses right, so acceptance alone says
  // little): the same state in both engines -> the same draft ids, per-submit and chained, full
  // head and a 128-row draft head
  for (const dv of [0, 128]) {
    const w = dv ? await mk({ draftVocab: dv, draftVocabAuto: false }) : eng, o = dv ? await mk({ draftVocab: dv, draftVocabAuto: false, argmaxWide: false }) : eOld;
    const got = [[], []];
    for (const [j, e] of [[0, w], [1, o]]) {
      let nx2 = await start(e, false);
      for (let s2 = 0; s2 < 6; s2++) {
        const pos = e.pos;
        got[j].push(...await e._draftChain(nx2, pos, 3));
        const out = await e.specStep(nx2, argmax, 3);   // advance; both engines go through the same (perturbed) states
        nx2 = out.at(-1);
        got[j].push(await e.mtpRun(null, nx2, e.pos, "argmax"));
        e._pre = null;
      }
    }
    check(`${label}: draft ids, wide vs old argmax${dv ? `, draftVocab ${dv}` : ""}`, got[0].join() === got[1].join(), `${got[0].length} drafts: ${got[0].slice(0, 8).join(",")}…`);
  }
}
check("no GPU validation errors", !errs.length, errs.slice(0, 2).join(" | "));
console.log(fail ? `GPUSAMPLE FAIL (${fail})` : "GPUSAMPLE PASS ✓");
if (fail) Deno.exit(1);
