// Dense speculation (engine/dense.js specStepDrafts, prompt-lookup drafts from room/lookup.js):
// the output must be EXACTLY plain decoding's. For each prompt, on one engine (solo) and on a
// two-engine split in this process (host: embed + first half + head, worker: second half; the
// verify goes "round the chain" as one batched frame, as in a room):
//   * greedy: the same tokens as plain greedy decoding, and every verify column's logits
//     bit-identical to the plain step's logits at that position (so no near-tie can ever flip);
//   * sampled (t 0.8, top-k 40, seeded Math.random): the same tokens as plain sampling with the
//     same seed (each emitted token is one draw in both paths);
// and prints the acceptance, tokens per verify lap and the solo tok/s of both paths.
// DRAFT=qwen (a model dir; same tokenizer): also a run where that model drafts DRAFTK tokens when
// lookup finds nothing (room/draftmodel.js), greedy, with the same checks.
//   cd tests && [DENSE=qwen17] [N=96] [DRAFT=qwen] deno run --unstable-webgpu --allow-read --allow-env test_dense_spec.js
import { DenseEngine, makeTokenizer } from "../engine/engine.js";
import { parseGGUFHeader, ggufWeights } from "../engine/gguf.js";
import { lookupDrafts } from "../room/lookup.js";
import { greedy, aiSample } from "../room/sampling.js";
import { DraftModel } from "../room/draftmodel.js";
import { argmax } from "../engine/sampling.js";
import fs from "node:fs";

const env = (k, d) => Deno.env.get(k) ?? d;
const M = env("DENSE", "qwen17"), N = +env("N", 96);
const dir = new URL(`../models/${M}/`, import.meta.url).pathname;
const fd = fs.openSync(dir + "model.gguf", "r");
const readAt = (off, len) => { const o = new Uint8Array(len); let g = 0; while (g < len) { const n = fs.readSync(fd, o, g, len - g, off + g); if (n <= 0) break; g += n; } return o; };
const G = parseGGUFHeader(readAt(0, 32 << 20).buffer, { skipTokenizer: true });
const cfg = JSON.parse(await Deno.readTextFile(dir + "config.json"));
const tok = makeTokenizer(JSON.parse(await Deno.readTextFile(dir + "tokenizer.json")));
const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
let gpuErrors = 0;
device.addEventListener?.("uncapturederror", (e) => { if (gpuErrors++ < 4) console.error("GPU ERROR:", e.error?.message?.slice(0, 300)); });
const L = cfg.num_hidden_layers, H = Math.floor(L / 2);
const W = (lo, hi, hasEmbed, hasHead) => ggufWeights(G, (i) => readAt(i.byteOffset, i.byteLength), { lo, hi, hasEmbed, hasHead });
const OPTS = JSON.parse(env("OPTS", "{}"));
const solo = await DenseEngine.create({ device, cfg, weights: await W(0, L, true, true), layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 2048, ...OPTS });
const host = await DenseEngine.create({ device, cfg, weights: await W(0, H, true, true), layerRange: [0, H], hasEmbed: true, hasHead: true, maxSeq: 2048, ...OPTS });
const work = await DenseEngine.create({ device, cfg, weights: await W(H, L, false, false), layerRange: [H, L], hasEmbed: false, hasHead: false, maxSeq: 2048, ...OPTS });
const dim = solo.dims.dim;
let drafter = null;
const DRAFT = env("DRAFT", ""), DRAFTK = +env("DRAFTK", 4);
if (DRAFT) {
  const dd = new URL(`../models/${DRAFT}/`, import.meta.url).pathname, dfd = fs.openSync(dd + "model.gguf", "r");
  const rd = (off, len) => { const o = new Uint8Array(len); let g = 0; while (g < len) { const n = fs.readSync(dfd, o, g, len - g, off + g); if (n <= 0) break; g += n; } return o; };
  const DG = parseGGUFHeader(rd(0, 32 << 20).buffer, { skipTokenizer: true });
  const dcfg = JSON.parse(await Deno.readTextFile(dd + "config.json")), DL = dcfg.num_hidden_layers;
  const de = await DenseEngine.create({ device, cfg: dcfg, weights: await ggufWeights(DG, (i) => rd(i.byteOffset, i.byteLength), { lo: 0, hi: DL, hasEmbed: true, hasHead: true }), layerRange: [0, DL], hasEmbed: true, hasHead: true, maxSeq: 2048 });
  drafter = () => new DraftModel(de, { argmax });
}

const V = tok.vocab;
const chat = (u) => [V["<|im_start|>"], ...tok.encode("user\n" + u), V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n"), V["<think>"], ...tok.encode("\n\n"), V["</think>"], ...tok.encode("\n\n")];
const CODE = `def load_users(path):
    users = []
    with open(path) as f:
        for line in f:
            name, age, city = line.strip().split(",")
            users.append({"name": name, "age": int(age), "city": city})
    return users


def adults(users):
    return [u for u in users if u["age"] >= 18]


def by_city(users):
    out = {}
    for u in users:
        out.setdefault(u["city"], []).append(u)
    return out
`;
const PROMPTS = {
  chat: "What are three good habits for staying focused while working from home? Answer in a short paragraph.",
  code: "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring. Code only.",
  edit: "Rename the variable `users` to `people` everywhere in this code and return the whole file. Code only.\n\n```python\n" + CODE + "```",
};
const EOS = new Set([V["<|im_end|>"], V["<|endoftext|>"]]);

// seeded Math.random (mulberry32) for the sampled runs
const seeded = (seed) => () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const R0 = Math.random;
const samplers = { greedy: (lg) => greedy(lg), sampled: (lg) => aiSample(lg, 0.8, 40) };

// the two topologies: prefill, plain step (token at pos -> logits), verify (tokens at pos -> final hiddens)
const topo = {
  solo: {
    prefill: async (ids) => { solo.reset(); await solo.prefillTokens(ids); },
    step: async (t) => solo.forwardToken(t),
    eng: solo, trunk: null,
  },
  split: {
    prefill: async (ids) => {   // the room's batched prefill: 4 columns per frame, the tail one token at a time
      host.reset(); work.reset();
      let i = 0;
      for (; ids.length - i >= 4; i += 4) await work.runHiddenBatch(await host.embedRunBatch(ids.slice(i, i + 4), i), i);
      for (; i < ids.length; i++) await work.runHidden(await host.embedRun(ids[i], i), i);
      host.pos = ids.length;
    },
    step: async (t) => { const p = host.pos; const h = await work.runHidden(await host.embedRun(t, p), p); host.pos = p + 1; return host.headFromHidden(h); },
    eng: host,
    trunk: async (tokens, pos) => {   // one frame round the chain: host layers (batched), then the worker's
      const n = tokens.length, hb = new Float32Array(n * dim), out = new Float32Array(n * dim);
      for (let c = 0; c < n; c += 4) { const m = Math.min(4, n - c); hb.set(await host.embedRunBatch(tokens.slice(c, c + m), pos + c, { base: c, total: n }), c * dim); }
      for (let c = 0; c < n; c += 4) { const m = Math.min(4, n - c); out.set(await work.runHiddenBatch(hb.subarray(c * dim, (c + m) * dim), pos + c, { base: c, total: n }), c * dim); }
      return out;
    },
  },
};

// plain: N tokens; returns tokens and the logits bits that produced each (step i's logits pick token i+1)
async function plain(T, ids, sample) {
  await T.prefill(ids.slice(0, -1));
  const toks = [], bits = [];
  let lg = await T.step(ids[ids.length - 1]);
  const t0 = performance.now();
  for (let i = 0; i < N; i++) {
    bits.push(new Uint32Array(lg.buffer.slice(lg.byteOffset, lg.byteOffset + lg.byteLength)));
    const t = sample(lg); toks.push(t);
    if (EOS.has(t) || i === N - 1) break;
    lg = await T.step(t);
  }
  return { toks, bits, ms: performance.now() - t0 };
}
// speculative: lookup drafts, specStepDrafts; a step with no drafts is a plain step (as the room does)
async function spec(T, ids, sample, ref, dm = null) {
  await T.prefill(ids.slice(0, -1));
  const E = T.eng;
  E.specStats = { steps: 0, drafts: 0, accepted: 0 };
  let lg = await T.step(ids[ids.length - 1]);
  const ctx = [...ids];
  const toks = [];
  let bitDiff = 0, cols = 0, laps = 0;
  const cmp = (lgk, at) => { if (!ref || !ref.bits[at]) return; cols++; const a = new Uint32Array(lgk.buffer, lgk.byteOffset, lgk.length), b = ref.bits[at]; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { bitDiff++; break; } };
  cmp(lg, 0);
  const t0 = performance.now();
  let next = sample(lg); toks.push(next);
  while (!EOS.has(next) && toks.length < N) {
    let lk = lookupDrafts([...ctx, next], Math.min(E.maxDrafts, N - toks.length));
    if (!lk.length && dm) lk = await dm.propose([...ctx, next], Math.min(DRAFTK, N - toks.length));
    laps++;
    let out;
    if (lk.length) {
      const at = toks.length;   // verify column k gives the logits for output token at + k
      const sampleSpy = (x) => { cmp(x, at + sampleSpy.k++); return sample(x); };
      sampleSpy.k = 0;
      out = await E.specStepDrafts(next, sampleSpy, lk, { runTrunk: T.trunk });
    } else {
      lg = await T.step(next);
      cmp(lg, toks.length);
      out = [sample(lg)];
    }
    ctx.push(next, ...out.slice(0, -1));
    for (const t of out) { if (toks.length >= N) break; toks.push(t); if (EOS.has(t)) break; }
    next = toks[toks.length - 1];
  }
  return { toks, ms: performance.now() - t0, bitDiff, cols, laps, st: { ...E.specStats } };
}

let fail = 0;
for (const [pname, text] of Object.entries(PROMPTS)) {
  const ids = chat(text);
  for (const tname of ["solo", "split"]) {
    const T = topo[tname];
    for (const [sname, sample] of Object.entries(samplers)) {
      if (sname === "sampled") Math.random = seeded(1234);
      const p = await plain(T, ids, sample);
      if (sname === "sampled") Math.random = seeded(1234);
      const s = await spec(T, ids, sample, sname === "greedy" ? p : null);
      Math.random = R0;
      const same = p.toks.length === s.toks.length && p.toks.every((t, i) => t === s.toks[i]);
      let firstDiff = -1; for (let i = 0; i < Math.min(p.toks.length, s.toks.length); i++) if (p.toks[i] !== s.toks[i]) { firstDiff = i; break; }
      const acc = s.st.drafts ? s.st.accepted / s.st.drafts : 0;
      const line = `${pname.padEnd(4)} ${tname.padEnd(5)} ${sname.padEnd(7)}: ${p.toks.length} tok, spec ${same ? "IDENTICAL to plain" : `DIFFERS from plain (first at ${firstDiff})`}`
        + (sname === "greedy" ? ` · verify logits ${s.bitDiff ? `${s.bitDiff}/${s.cols} columns DIFFER` : `bit-identical (${s.cols} columns)`}` : "")
        + ` · ${s.st.steps} verifies, ${s.st.accepted}/${s.st.drafts} drafts accepted (${(acc * 100).toFixed(0)}%) · ${(s.toks.length / s.laps).toFixed(2)} tok/lap`
        + ` · ${(p.toks.length / (p.ms / 1000)).toFixed(1)} → ${(s.toks.length / (s.ms / 1000)).toFixed(1)} tok/s`;
      console.log(line);
      if (!same || s.bitDiff) fail++;
      if (drafter && sname === "greedy") {   // lookup, else the draft model
        const d = await spec(T, ids, sample, p, drafter());
        const same2 = p.toks.length === d.toks.length && p.toks.every((t, i) => t === d.toks[i]);
        const acc2 = d.st.drafts ? d.st.accepted / d.st.drafts : 0;
        console.log(`${pname.padEnd(4)} ${tname.padEnd(5)} +draft ${DRAFT}: spec ${same2 ? "IDENTICAL to plain" : "DIFFERS from plain"} · verify logits ${d.bitDiff ? `${d.bitDiff}/${d.cols} columns DIFFER` : `bit-identical (${d.cols} columns)`}`
          + ` · ${d.st.accepted}/${d.st.drafts} drafts accepted (${(acc2 * 100).toFixed(0)}%) · ${(d.toks.length / d.laps).toFixed(2)} tok/lap · ${(d.toks.length / (d.ms / 1000)).toFixed(1)} tok/s`);
        if (!same2 || d.bitDiff) fail++;
      }
    }
  }
}
if (gpuErrors) fail++;
console.log(fail ? `DENSE SPEC FAIL (${fail})` : "DENSE SPEC PASS (spec == plain, greedy bit-exact, sampled same draws)");
Deno.exit(fail ? 1 : 0);
