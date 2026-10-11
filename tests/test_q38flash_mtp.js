// Qwen3.8-Flash-Next MTP speculative decoding (engine/qwen4exp_mtp.js) on the decode engine, the whole model resident
// (~77 GB of GPU memory on the GB10: the trunk as tests/test_q38flash.js loads it, plus the MTP block). Roadmap 33
// PR plan item 6. The model: models/q38flash (Q38F_DIR=<dir>) with the three trunk shards and
// mtp-Qwen3.8-Flash-Next-Q4_0.gguf.
//   deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights tests/test_q38flash_mtp.js
//
// Gates, per golden prompt (tests/golden/q38flash_llama_greedy.json, those under 2048 tokens):
//   1. spec ids == plain greedy ids, with the decode verifier stopping at the first rejection, and with it running
//      every column then rolling back (restore conv / S / PLE ring, n-gram rollback, re-run): the rollback path
//   2. the draft acceptance rate and the tokens per verify, per K (K=1,3 by default: KS=1,2,3)
//   and the tok/s of each. The decode verifier runs the columns one at a time, so spec is no faster than plain here;
//   a step's cost model (draft ms, verify ms per column) is printed, and with a batched verify (PR #348) it would be
//   one pass. HEAD=own drafts with the MTP file's own Q4_0 head instead of the trunk's (the default; see
//   engine/qwen4exp_mtp.js). TOKENS=n caps the generated ids (default: the golden's). CASES=a,b picks prompts.
//   REFILL=0: no draft-row refill after a verify. ROLLBACK=0 skips the full-pass rollback check.
import fs from "node:fs";
import path from "node:path";
import { Qwen4ExpEngine } from "../engine/qwen4exp.js";
import { Qwen4ExpMtp, DecodeVerifier } from "../engine/qwen4exp_mtp.js";
import { qwen4expWeights, qwen4expConfig, gpuUploadEntry } from "../engine/gguf.js";
import { ngramConfig, NgramTable } from "../engine/ngram.js";
import { openGGUF, gpuDevice, watchGpuErrors } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const DIR = env("Q38F_DIR", new URL("../models/q38flash", import.meta.url).pathname);
const first = fs.readdirSync(DIR).filter((f) => /^Qwen3\.8-Flash-Next-.*-00001-of-\d{5}\.gguf$/.test(f))[0];
const mtpFile = fs.readdirSync(DIR).filter((f) => /^mtp-Qwen3\.8-Flash-Next-.*\.gguf$/.test(f))[0];
if (!first || !mtpFile) throw new Error("need the Qwen3.8-Flash-Next shards and its mtp-*.gguf in " + DIR);
const OWN = env("HEAD") === "own", KS = env("KS", "1,3").split(",").map(Number), REFILL = env("REFILL") !== "0";

const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(path.join(DIR, first)), mtpModel = openGGUF(path.join(DIR, mtpFile), { skipTokenizer: true });
const G = model.G, cfg = qwen4expConfig(G.meta);

let t0 = performance.now();
const up = (e, name) => { if (name !== "token_embd.weight") gpuUploadEntry(device, e, false); };
const flush = async () => { device.queue.submit([device.createCommandEncoder().finish()]); await device.queue.onSubmittedWorkDone(); };
const layers = [];
for (let i = 0; i < cfg.L; i++) {
  layers.push((await qwen4expWeights(G, model.bytesOf, { lo: i, hi: i + 1, hasEmbed: false, hasHead: false, qsa: false }, undefined, up)).layers[0]);
  await flush();
  if (i % 8 === 7) console.log(`  layer ${i + 1}/${cfg.L} on the GPU, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
// the head with the MTP block, and the MTP file's own token_embd / output for the checks below: both heads keep
// their CPU copy until then; the file's own head is uploaded only when it drafts (HEAD=own: the engine's wq does it)
let nOut = 0;
const upRest = (e, name) => {
  if (name === "token_embd.weight") return;
  if (name === "output.weight") { if (++nOut === 1) gpuUploadEntry(device, e, true); return; }
  gpuUploadEntry(device, e, false);
};
const rest = await qwen4expWeights(G, model.bytesOf, { lo: cfg.L, hi: cfg.L, hasEmbed: true, hasHead: true, qsa: false,
  mtp: mtpModel.G, mtpBytesOf: mtpModel.bytesOf, mtpOwnHead: true }, undefined, upRest);
await flush();
const weights = { ...rest, layers };
const fds = model.paths.map((p) => fs.openSync(p, "r"));
const readAt = async (file, off, len) => { const b = new Uint8Array(len); fs.readSync(fds[file], b, 0, len, off); return b; };
const ngram = new NgramTable(ngramConfig(G), { readAt, layout: "tensor" });
const eng = await Qwen4ExpEngine.create({ device, meta: G.meta, weights, hasEmbed: true, hasHead: true, ngram });
let fail = 0;

// which head (engine/qwen4exp_mtp.js): the MTP file's token_embd and output against the trunk's
{
  const m = weights.mtp, rowOf = (e, r) => eng._embedRow.call({ embed: e, cfg }, r);
  const same = (a, b) => a.kind === b.kind && a.qs.length === b.qs.length && a.qs.every((v, i) => v === b.qs[i]) && a.scales.every((v, i) => v === b.scales[i]);
  console.log(`MTP file: token_embd ${same(m.embed, weights.embed) ? "byte-equal to" : "DIFFERS from"} the trunk's`);
  let num = 0, den = 0, worst = 0;
  for (let r = 0; r < eng.dims.vocab; r += 997) {
    const a = rowOf(m.head, r), b = rowOf(weights.head, r);
    let n = 0, dd = 0;
    for (let j = 0; j < a.length; j++) { n += (a[j] - b[j]) ** 2; dd += b[j] ** 2; }
    num += n; den += dd; worst = Math.max(worst, Math.sqrt(n / (dd || 1)));
  }
  console.log(`MTP file: output (${m.head.kind}) vs the trunk's (${weights.head.kind}): rel L2 ${Math.sqrt(num / den).toExponential(2)}, worst row ${worst.toExponential(2)} (Q4_0 vs Q6_K of one tensor: ~1e-1)`);
}
weights.head.qs = weights.head.scales = null;   // on the GPU already
const mtp = await Qwen4ExpMtp.create(eng, weights.mtp, { head: OWN ? weights.mtp.head : weights.head });
weights.mtp.head.qs = weights.mtp.head.scales = weights.mtp.embed.qs = weights.mtp.embed.scales = null;
console.log(`qwen4exp + MTP (${OWN ? "its own Q4_0 head" : "the trunk's head"}) loaded in ${((performance.now() - t0) / 1000).toFixed(0)} s`);

const gold = JSON.parse(fs.readFileSync(new URL("./golden/q38flash_llama_greedy.json", import.meta.url), "utf8"));
const pick = env("CASES") ? env("CASES").split(",") : null;
const cases = Object.entries(gold).filter(([k, v]) => !k.startsWith("_") && (!pick || pick.includes(k)) && v.prompt.length + v.ids.length + 8 <= eng.maxSeq);
const sum = { plainMs: 0, plainN: 0 }, perK = new Map();
for (const [name, g] of cases) {
  const N = Math.min(+env("TOKENS", g.ids.length), g.ids.length);
  // plain greedy, the reference for the gates
  eng.reset(); mtp.reset();
  let r = await eng.prefillTokens(g.prompt);
  const plain = [r.id];
  let t = performance.now();
  while (plain.length < N) plain.push((await eng.forwardToken(plain.at(-1))).id);
  const plainMs = performance.now() - t;
  sum.plainMs += plainMs; sum.plainN += N - 1;
  const diffG = plain.findIndex((x, i) => x !== g.ids[i]);
  console.log(`${name}: prompt ${g.prompt.length}, ${N} ids · plain ${((N - 1) / plainMs * 1000).toFixed(2)} tok/s · ${diffG < 0 ? "== llama.cpp" : `differs from llama.cpp at ${diffG}`}`);
  for (const K of KS) for (const earlyStop of env("ROLLBACK") === "0" ? [true] : [true, false]) {
    eng.reset(); mtp.reset();
    mtp.stats = { steps: 0, drafts: 0, accepted: 0 };
    const ver = new DecodeVerifier(eng, { maxCols: K + 1, earlyStop });
    r = await mtp.prefill(g.prompt);
    const spec = [r.id];
    let draftMs = 0;
    t = performance.now();
    // per step: draft K, verify, refill (timed apart: the draft block's share is what a batched verify keeps)
    const run0 = mtp.run.bind(mtp);
    mtp.run = async (...a) => { const s = performance.now(); const x = await run0(...a); if (a[2]?.head) draftMs += performance.now() - s; return x; };
    while (spec.length < N) spec.push(...await mtp.specStep(spec.at(-1), ver, K, { refill: REFILL }));
    const ms = performance.now() - t;
    mtp.run = run0;
    spec.length = N;
    const diff = spec.findIndex((x, i) => x !== plain[i]);
    const st = mtp.stats, acc = st.accepted / st.drafts, perStep = (st.accepted + st.steps) / st.steps;
    console.log(`  K=${K} ${earlyStop ? "stop at rejection" : "full pass + rollback"}: ${diff < 0 ? "spec == plain" : `MISMATCH at ${diff}`} · accepted ${st.accepted}/${st.drafts} (${(acc * 100).toFixed(1)}%), ${perStep.toFixed(2)} tokens per verify · ${((N - 1) / ms * 1000).toFixed(2)} tok/s · drafting ${(draftMs / st.steps).toFixed(1)} ms per step (${(draftMs / st.drafts).toFixed(1)} ms per draft)`);
    if (diff >= 0) { fail++; console.log(`    plain ${plain.slice(Math.max(0, diff - 2), diff + 3)} spec ${spec.slice(Math.max(0, diff - 2), diff + 3)}`); }
    if (earlyStop) {
      const k = perK.get(K) || { steps: 0, drafts: 0, accepted: 0, draftMs: 0 };
      k.steps += st.steps; k.drafts += st.drafts; k.accepted += st.accepted; k.draftMs += draftMs;
      perK.set(K, k);
    }
  }
}
// what a batched verify would give: one pass of K + 1 columns (cost vK, from PR #348's measurements: pass it as
// VERIFY_MS=k:ms,...; default: the decode step's time, i.e. a pass as cheap as one token) plus the drafts
const step = sum.plainMs / sum.plainN;
const vMs = Object.fromEntries((env("VERIFY_MS") || "").split(",").filter(Boolean).map((s) => s.split(":").map(Number)));
console.log(`plain decode: ${(1000 / step).toFixed(2)} tok/s (${step.toFixed(1)} ms per token)`);
for (const [K, k] of perK) {
  const tokens = (k.accepted + k.steps) / k.steps, draft = k.draftMs / k.steps, v = vMs[K + 1] ?? step;
  console.log(`K=${K}: acceptance ${(k.accepted / k.drafts * 100).toFixed(1)}%, ${tokens.toFixed(2)} tokens per verify, drafts ${draft.toFixed(1)} ms per step; with a ${v.toFixed(1)} ms verify pass of ${K + 1} columns: ${(tokens / (v + draft) * 1000).toFixed(2)} tok/s (plain ${(1000 / step).toFixed(2)})`);
}
if (errors.count) { console.log(`${errors.count} GPU errors`); fail++; }
console.log(fail ? "Q38FLASH MTP FAIL" : "Q38FLASH MTP PASS ✓");
Deno.exit(fail ? 1 : 0);
