// Dispatch fusions vs the kernels they replace: attn_glue (qsplit + q/k head_norm + rope),
// dn_delta_gn (dn_delta + dn_gatenorm), the batched attention of prefill passes
// (attn_*_mc: all columns per dispatch; attn_flash_t2: two columns per workgroup) and the decode layer
// fusions (engine.layerFuse: dn_pre in dn_delta_gn, conv in the qkv GEMV, kv_store / sigmoid_mul in the
// attention glue / combine, rmsnorm in the small GEMVs, one compute pass per token). Every logit of the decode after a batched prefill must be
// bit-identical with each fusion on and off; also times decode.
//   NODE_PATH=... node tests/e2e/fusion_synth.mjs [--model f.gguf] [--prompt-len 200] [--tokens 24]
import fs from "fs"; import os from "os"; import path from "path";
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";
import { writeSynth } from "./synth.mjs";
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const PORT = +arg("port", 18993);

async function pageMain({ plen, ntok }) {
  const { Qwen35Engine } = await import("/engine/qwen35.js");
  const { parseGGUFHeader, qwen35Weights, GGML_EMBED } = await import("/engine/gguf.js");
  const { argmax } = await import("/engine/engine.js");
  const out = [], say = (s) => out.push(s);
  const buf = await (await fetch("/__m.gguf")).arrayBuffer();
  const G = parseGGUFHeader(buf);
  const bytesOf = async (i) => new Uint8Array(buf, i.byteOffset, i.byteLength).slice();
  const L = G.meta["qwen35.block_count"] - 1;
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  const errs = []; device.addEventListener("uncapturederror", (e) => errs.push(e.error?.message));
  const eng = await Qwen35Engine.create({ device, meta: G.meta, layerRange: [0, L], hasEmbed: true, hasHead: true, vocab: G.tensors[GGML_EMBED].shape[0],
    maxSeq: 2048, batchCols: 16, coopRowsB: 1, coopWG: 64, layerFuse: true, weights: await qwen35Weights(G, bytesOf, { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true }) });
  const prompt = Array.from({ length: plen }, (_, i) => 33 + ((i * 7919) % 90));
  const LFK = Object.keys(eng.layerFuse);
  const run = async (glue, dn, mc, tile = false, lf = "") => {
    eng.attnGlue = glue; eng.dnFuse = dn; eng.attnMC = mc; eng.attnTile = tile; eng.reset(); eng.mtpFill = false;
    for (const k of LFK) eng.layerFuse[k] = lf === "all" || lf.split(",").includes(k);
    await eng.prefillTokens(prompt.slice(0, -1));
    const logs = [];
    let lg = await eng.forwardToken(prompt[plen - 1]); logs.push(lg);
    await device.queue.onSubmittedWorkDone();
    const t0 = performance.now();
    for (let i = 1; i < ntok; i++) { lg = await eng.forwardToken(argmax(lg)); logs.push(lg); }
    return { logs, msPerTok: (performance.now() - t0) / (ntok - 1) };
  };
  const a = await run(false, false, false);
  let ok = !errs.length;
  for (const [glue, dn, mc, tile, name, lf] of [[true, false, false, false, "attn_glue"], [false, true, false, false, "dn_delta_gn"], [false, false, true, false, "attn_*_mc"], [false, false, false, true, "attn_flash_t2"],
    [false, true, false, false, "layerFuse.dn", "dn"], [false, false, false, false, "layerFuse.conv", "conv"], [false, false, false, false, "layerFuse.comb", "comb"], [true, false, false, false, "layerFuse.kv", "kv,comb"],
    [false, false, false, false, "layerFuse.norm", "norm"], [false, false, false, false, "layerFuse.pass", "pass"], [true, true, true, true, "all", "all"]]) {
    const b = await run(glue, dn, mc, tile, lf || "");
    // a layerFuse flag is compared with the same kernel switches and layerFuse off (so on a real GPU, where
    // attn_glue's bits are not the five kernels', layerFuse.kv is still checked alone)
    const ref = lf ? await run(glue, dn, mc, tile, "") : a;
    let diff = 0;
    for (let i = 0; i < ntok; i++) for (let j = 0; j < ref.logs[i].length; j++) if (!Object.is(ref.logs[i][j], b.logs[i][j])) diff++;
    ok = ok && diff === 0;
    say(`${diff === 0 ? "PASS" : "FAIL"} ${name}: ${ntok} decode steps after a ${plen}-token prompt, ${diff} logits differ from ${lf ? "layerFuse off" : "the unfused kernels"}; ${b.msPerTok.toFixed(0)} vs ${a.msPerTok.toFixed(0)} ms/token (SwiftShader, not a GPU number)`);
  }
  ok = ok && !errs.length;
  if (errs.length) say("GPU errors: " + errs.slice(0, 2).join(" | "));
  return { out, ok };
}
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-ag-"));
const model = arg("model") || writeSynth(path.join(tmp, "m.gguf"), {}).file;
const srv = serveRepo(PORT, { "/__m.gguf": path.resolve(model) });
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: GPU_ARGS });
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${PORT}/favicon.svg`);
const res = await page.evaluate(pageMain, { plen: +arg("prompt-len", 200), ntok: +arg("tokens", 24) });
for (const l of res.out) console.log(l);
console.log(res.ok ? "FUSION PASS" : "FUSION FAIL");
await browser.close(); srv.close(); fs.rmSync(tmp, { recursive: true, force: true });
process.exit(res.ok ? 0 : 1);
