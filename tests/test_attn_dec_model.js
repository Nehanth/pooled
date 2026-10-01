// attnDecode "v2" (split-K decode attention, engine/wgsl/attn_dec.js) vs the default attn_flash decode on a
// real model at long context, one engine (engine.attnDecode toggled at runtime). The prompt is prefilled
// the default way; then, from the same KV state, the last prompt token and GEN greedy tokens are decoded
// with each kernel:
//   * logits of the token after the prompt: argmax equal, relDiff under prefillTol (load_model.js);
//   * greedy continuation: reported (identical expected; a difference fails);
//   * speculative decoding (MTP verify passes through attn_dec) == plain decoding, both with v2.
//   MODEL=moe|27b  LENS=1000,8000,30000  GEN=24  K=3
//   cd tests && MODEL=moe deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights test_attn_dec_model.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { argmax } from "../engine/engine.js";
import { openGGUF, gpuDevice, watchGpuErrors, trunkLayers, MOE_PATH, Q38_PATH, prefillTol } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "moe"), GEN = +env("GEN", 24), K = +env("K", 3);
const LENS = env("LENS", "1000,8000,30000").split(",").map(Number);
const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(MODEL === "27b" ? Q38_PATH : MOE_PATH);
const G = model.G, L = trunkLayers(G), nBlk = G.meta["qwen35.block_count"];
const hasMtp = L < nBlk;
const tok = model.tokenizer();
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: hasMtp });
const maxSeq = Math.ceil((Math.max(...LENS) + GEN + 64) / 256) * 256;
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq, batchCols: 16, coopRowsB: 1, attnDecode: "v2" });
if (eng.attnDecode !== "v2") { console.log("ATTN DEC MODEL FAIL: attnDecode v2 did not turn on (see the warning above)"); Deno.exit(1); }
console.log(`${MODEL}: ${L} layers, mtp ${!!eng.mtp}, maxSeq ${maxSeq}, attn_flash ${eng.faSplit} x ${eng.faSplits}, attn_dec S ${eng.adCfg.S}`);

// long realistic prompt: this repo's source, repeated as needed
let src = [];
for (const f of ["../engine/qwen35.js", "../engine/gguf.js", "../harness/agent.js", "../room.js", "../engine/wgsl/qwen35.js"]) src.push(...tok.encode(`\n// file: ${f}\n` + await Deno.readTextFile(new URL(f, import.meta.url))));
while (src.length < Math.max(...LENS)) src = src.concat(src);
const V = tok.vocab;
const prompt = (n) => {
  const tail = [V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n")];
  const head = [V["<|im_start|>"], ...tok.encode("user\nSummarize what this code does:\n")];
  return [...head, ...src.slice(0, n - head.length - tail.length), ...tail];
};
const rel = (a, b) => { let md = 0, sc = 1e-6; for (let i = 0; i < a.length; i++) { md = Math.max(md, Math.abs(a[i] - b[i])); sc = Math.max(sc, Math.abs(b[i])); } return md / sc; };

async function run(ids, mode, spec = false) {
  eng.reset(); eng.attnDecode = "v1"; if (eng.mtp) eng.mtpFill = spec;
  await eng.prefillTokens(ids.slice(0, -1));
  eng.attnDecode = mode;
  let lg = Float32Array.from(await eng.forwardToken(ids.at(-1)));
  const first = lg;
  let next = argmax(lg); const gen = [next];
  const t0 = performance.now();
  if (spec && eng.mtp) { while (gen.length < GEN) { for (const t of await eng.specStep(next, argmax, K)) gen.push(t); next = gen.at(-1); } gen.length = GEN; }
  else for (let i = 1; i < GEN; i++) { lg = await eng.forwardToken(next); next = argmax(lg); gen.push(next); }
  return { lg: first, gen, tokps: (GEN - 1) / ((performance.now() - t0) / 1000) };
}

let fail = 0;
for (const n of LENS) {
  const ids = prompt(n);
  const a = await run(ids, "v1"), b = await run(ids, "v2");
  const r = rel(b.lg, a.lg), sameGen = a.gen.every((t, i) => t === b.gen[i]);
  let ok = argmax(a.lg) === argmax(b.lg) && r < prefillTol(!!eng.moe) && sameGen;
  let line = `${n} tokens: logits relDiff v2 vs v1 ${r.toExponential(2)} · argmax ${argmax(a.lg)} / ${argmax(b.lg)} · greedy ${GEN} ${sameGen ? "identical" : "DIFFERS"} · decode ${a.tokps.toFixed(2)} -> ${b.tokps.toFixed(2)} tok/s`;
  if (!sameGen) line += `\n  v1: ${JSON.stringify(tok.decode(a.gen))}\n  v2: ${JSON.stringify(tok.decode(b.gen))}`;
  if (eng.mtp) {
    const s = await run(ids, "v2", true), same = s.gen.every((t, i) => t === b.gen[i]);
    line += ` · v2 spec ${same ? "identical to plain" : "DIFFERS from plain"}`;
    if (!same) ok = false;
  }
  if (!ok) fail++;
  console.log((ok ? "PASS " : "FAIL ") + line);
}
console.log(`GPU errors ${errors.count}`);
console.log(fail || errors.count ? "ATTN DEC MODEL FAIL" : "ATTN DEC MODEL PASS");
Deno.exit(fail || errors.count ? 1 : 0);
