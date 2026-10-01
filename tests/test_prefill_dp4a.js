// dp4a wide prefill (engine option prefillDp4a, engine/wgsl/gemm_wide.js gemmDp4aWGSL) on a real model, against
// the f32 16-column prefill ("narrow"), the f32 wide GEMM ("wide") and llama.cpp. Same engine, same tokens
// (engine.prefillWide / engine.prefillDp4a switch at runtime). Prompts: plain text slices of the frozen fixture
// golden/prefill_opts_prompt.txt (no special tokens, so llama.cpp gets exactly the same ids).
//   * next-token argmax equal to narrow's at every length; logits relDiff vs narrow reported (dp4a quantizes
//     activations like llama.cpp's MMQ, so it is above the f32 prefill tolerance by design);
//   * greedy continuation of GEN tokens per mode vs llama.cpp's greedy continuation of the same ids
//     (golden/q38_llama_greedy.json, from llama-server: LLAMA_URL=http://127.0.0.1:8080 refreshes it, GOLD_ONLY=1
//     then exits before touching the GPU); dp4a
//     must match llama.cpp for at least as many tokens as narrow does;
//   * the next token's log-probabilities vs llama.cpp's (its top 20, n_probs, stored in the golden as "<key>#lp"):
//     max |logprob - llama.cpp's| per mode. dp4a must be no further from llama.cpp than narrow (+ LP_SLACK, 0.05 nats),
//     the condition for turning dp4a on by default, since its relDiff vs narrow is above the f32 prefill tolerance;
//   * speculative decoding after a dp4a prefill: identical to plain decoding after the same prefill.
//   MODEL=27b|moe  LENS=150,700,2100  GEN=32  LLAMA_URL=
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env --allow-net --allow-write=$HOME/.cache/swarmllm-weights,golden test_prefill_dp4a.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { argmax } from "../engine/engine.js";
import { openGGUF, gpuDevice, watchGpuErrors, trunkLayers, MOE_PATH, Q38_PATH } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "27b"), GEN = +env("GEN", 32), K = +env("K", 3), LLAMA = env("LLAMA_URL", "");
const LENS = env("LENS", "150,700,2100").split(",").map(Number), LP_SLACK = +env("LP_SLACK", 0.05);
async function llama(ids) {
  const r = await fetch(`${LLAMA}/completion`, { method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt: ids, n_predict: GEN, temperature: 0, top_k: 1, samplers: ["top_k"], cache_prompt: false, return_tokens: true, ignore_eos: true, n_probs: 20 }) });
  const j = await r.json();
  // the first generated token's top 20 (pre-sampling log-probabilities of the full distribution after the prompt)
  const lp = (j.completion_probabilities?.[0]?.top_logprobs || []).map((e) => [e.id, e.logprob]);
  return { tokens: j.tokens, lp };
}
const GOLD = new URL(`./golden/${MODEL === "moe" ? "q36moe" : "q38"}_llama_greedy.json`, import.meta.url);
const model = openGGUF(MODEL === "moe" ? MOE_PATH : Q38_PATH);
const G = model.G, L = trunkLayers(G), nBlk = G.meta["qwen35.block_count"];
const tok = model.tokenizer();
const src = tok.encode(await Deno.readTextFile(new URL("./golden/prefill_opts_prompt.txt", import.meta.url)));
const prompt = (n, i) => src.slice(i * 211, i * 211 + n);
let gold = {}; try { gold = JSON.parse(await Deno.readTextFile(GOLD)); } catch { /* none yet */ }
if (LLAMA) {   // refresh the llama.cpp goldens first (no GPU needed for this part)
  for (const [pi, n] of LENS.entries()) { const k = `${n}@${pi * 211}`, r = await llama(prompt(n, pi)); gold[k] = r.tokens; if (r.lp.length) gold[`${k}#lp`] = r.lp; }
  await Deno.writeTextFile(GOLD, JSON.stringify(gold) + "\n");
  console.log(`llama.cpp goldens written to ${GOLD.pathname}`);
  if (env("GOLD_ONLY", "") === "1") Deno.exit(0);
}
const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: L < nBlk });
const maxSeq = Math.ceil((Math.max(...LENS) + GEN + 64) / 256) * 256;
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq, batchCols: 16, coopRowsB: 1, prefillUbatch: 256, prefillDp4a: true });
if (!eng.dp4aCfg) { console.log("SKIP: prefillDp4a unavailable on this device (see the warning above)"); Deno.exit(0); }
console.log(`${MODEL}: ${L} layers, ubatch ${eng.ubatch}, dp4a tile ${JSON.stringify(eng.dp4aCfg)}`);
const rel = (a, b) => { let md = 0, sc = 1e-6; for (let i = 0; i < a.length; i++) { md = Math.max(md, Math.abs(a[i] - b[i])); sc = Math.max(sc, Math.abs(b[i])); } return md / sc; };
// max |log-softmax(lg)[id] - llama.cpp's logprob| over llama.cpp's top tokens
const lpDiff = (lg, ref) => { let m = -Infinity; for (const v of lg) m = Math.max(m, v); let z = 0; for (const v of lg) z += Math.exp(v - m);
  const lz = m + Math.log(z); return Math.max(...ref.map(([id, lp]) => Math.abs(lg[id] - lz - lp))); };
const same = (a, b) => { let k = 0; while (k < a.length && k < b.length && a[k] === b[k]) k++; return k; };

async function run(ids, mode, spec = false) {
  eng.reset(); eng.prefillWide = mode !== "narrow"; eng.prefillDp4a = mode === "dp4a"; if (eng.mtp) eng.mtpFill = true;
  await eng.prefillTokens(ids.slice(0, -1));
  let lg = Float32Array.from(await eng.forwardToken(ids.at(-1)));
  const first = lg; let next = argmax(lg); const gen = [next];
  if (spec) { while (gen.length < GEN) { for (const t of await eng.specStep(next, argmax, K)) gen.push(t); next = gen.at(-1); } gen.length = GEN; }
  else for (let i = 1; i < GEN; i++) { lg = await eng.forwardToken(next); next = argmax(lg); gen.push(next); }
  return { lg: first, gen };
}
let fail = 0;
for (const [pi, n] of LENS.entries()) {
  const ids = prompt(n, pi), key = `${n}@${pi * 211}`;
  const ref = gold[key], refLp = gold[`${key}#lp`];
  const R = {}; for (const m of ["narrow", "wide", "dp4a"]) R[m] = await run(ids, m);
  const s = await run(ids, "dp4a", true), specSame = s.gen.every((t, i) => t === R.dp4a.gen[i]);
  const a0 = argmax(R.narrow.lg);
  let line = `${n} tokens: relDiff vs narrow: wide ${rel(R.wide.lg, R.narrow.lg).toExponential(2)}, dp4a ${rel(R.dp4a.lg, R.narrow.lg).toExponential(2)} · argmax ${a0} / ${argmax(R.wide.lg)} / ${argmax(R.dp4a.lg)}`
    + ` · greedy ${GEN} vs narrow: wide ${same(R.wide.gen, R.narrow.gen)}, dp4a ${same(R.dp4a.gen, R.narrow.gen)}`
    + (ref ? ` · vs llama.cpp: narrow ${same(R.narrow.gen, ref)}, wide ${same(R.wide.gen, ref)}, dp4a ${same(R.dp4a.gen, ref)}` : " · no llama.cpp golden")
    + (refLp ? ` · logprob vs llama.cpp (top ${refLp.length}): narrow ${lpDiff(R.narrow.lg, refLp).toFixed(4)}, wide ${lpDiff(R.wide.lg, refLp).toFixed(4)}, dp4a ${lpDiff(R.dp4a.lg, refLp).toFixed(4)}` : "")
    + ` · spec after dp4a prefill ${specSame ? "identical to plain" : "DIFFERS from plain"}`;
  if (refLp && lpDiff(R.dp4a.lg, refLp) > lpDiff(R.narrow.lg, refLp) + LP_SLACK) { fail++; line += " · dp4a further from llama.cpp's logprobs than narrow"; }
  if (argmax(R.dp4a.lg) !== a0 || !specSame) fail++;
  if (ref && same(R.dp4a.gen, ref) < Math.min(GEN, same(R.narrow.gen, ref))) { fail++; line += " · dp4a matches llama.cpp for fewer tokens than narrow"; }
  console.log(line);
  if (same(R.dp4a.gen, R.narrow.gen) < GEN) console.log(`  narrow: ${JSON.stringify(tok.decode(R.narrow.gen))}\n  dp4a:   ${JSON.stringify(tok.decode(R.dp4a.gen))}${ref ? `\n  llama:  ${JSON.stringify(tok.decode(ref))}` : ""}`);
}
console.log(fail || errors.count ? `DP4A PREFILL FAIL (GPU errors ${errors.count})` : "DP4A PREFILL PASS");
Deno.exit(fail || errors.count ? 1 : 0);
