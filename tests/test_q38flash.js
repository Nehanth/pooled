// Qwen3.8-Flash-Next (qwen4exp) on the engine (engine/qwen4exp.js): the decode path, every weight resident on the
// GPU (~75 GB on the GB10), the n-gram table read from the model file (engine/ngram.js), context <= 2048.
// Roadmap 33 PR plan item 4. The model: bartowski Qwen3.8-Flash-Next-Q4_0 (3 shards) in models/q38flash
// (Q38F_DIR=<dir> elsewhere). Run it through tests/run.sh's environment, e.g.
//   deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights tests/test_q38flash.js
//
// Modes:
//   (default)  greedy ids vs llama.cpp (tests/golden/q38flash_llama_greedy.json, the prompts under 2048 tokens):
//              MATCH per prompt, with the golden's near ties reported; decode tok/s. TOKENS=n caps the ids compared
//              (default: all of the golden's). CASES=a,b picks prompts.
//   LAYERS=N   per-layer check against llama.cpp's own tensors (the q38dump dumps, Q38F_DUMPS, default
//              <Q38F_DIR>/dumps; README there): layers 0..N-1 (48: and the head), every tapped tensor's relative
//              L2 error per token, and the expert sets. Teacher-forced by default (each half layer starts from
//              llama.cpp's residual, as tests/reference/ref_q38flash.mjs checks); FREE=1 chains our own.
//              FREE=both runs both; GREEDY=1 (with LAYERS=48) runs the goldens after, on the same load.
//              SEQS=s0,s1 picks sequences (default all four). Q38F_DUMPS=<dir> from tests/reference/q38flash_refdump.mjs
//              compares against the JS reference (f32 activations, as the engine) instead: there TOL (default 2e-2)
//              gates every tensor and the expert sets must equal. Against llama.cpp's Q8-rounded activations the
//              errors compound within a half layer, so there only the argmax, n-gram rows and sizes gate.
import fs from "node:fs";
import path from "node:path";
import { Qwen4ExpEngine } from "../engine/qwen4exp.js";
import { qwen4expWeights, qwen4expConfig, gpuUploadEntry } from "../engine/gguf.js";
import { ngramConfig, NgramTable } from "../engine/ngram.js";
import { openGGUF, gpuDevice, watchGpuErrors } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const DIR = env("Q38F_DIR", new URL("../models/q38flash", import.meta.url).pathname);
const first = fs.readdirSync(DIR).filter((f) => /^Qwen3\.8-Flash-Next-.*-00001-of-\d{5}\.gguf$/.test(f))[0];
if (!first) throw new Error("no Qwen3.8-Flash-Next shards in " + DIR);
const LAYERS = env("LAYERS") ? +env("LAYERS") : 0;

const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(path.join(DIR, first));
const G = model.G, cfg = qwen4expConfig(G.meta);
const nL = LAYERS || cfg.L, hasHead = nL === cfg.L;

// the weights, a layer at a time, each matrix uploaded as it is converted (tests/load_model.js streamWeights, with
// the qwen4exp loader); f32 tensors keep their CPU copy and the engine uploads them
let t0 = performance.now();
const up = (e, name) => { if (name !== "token_embd.weight") gpuUploadEntry(device, e, false); };
const flush = async () => { device.queue.submit([device.createCommandEncoder().finish()]); await device.queue.onSubmittedWorkDone(); };
const layers = [];
for (let i = 0; i < nL; i++) {
  layers.push((await qwen4expWeights(G, model.bytesOf, { lo: i, hi: i + 1, hasEmbed: false, hasHead: false, qsa: false }, undefined, up)).layers[0]);
  await flush();
  if (i % 8 === 7 || i === nL - 1) console.log(`  layer ${i + 1}/${nL} on the GPU, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
const rest = await qwen4expWeights(G, model.bytesOf, { lo: nL, hi: nL, hasEmbed: true, hasHead, qsa: false }, undefined, up);
await flush();
const weights = { ...rest, layers };

// the n-gram table: rows read straight from the shard that holds it
const fds = model.paths.map((p) => fs.openSync(p, "r"));
const readAt = async (file, off, len) => { const b = new Uint8Array(len); fs.readSync(fds[file], b, 0, len, off); return b; };
const ngram = new NgramTable(ngramConfig(G), { readAt, layout: "tensor" });
const eng = await Qwen4ExpEngine.create({ device, meta: G.meta, weights, layerRange: [0, nL], hasEmbed: true, hasHead, ngram });
console.log(`qwen4exp: ${nL} layers${hasHead ? " + head" : ""} loaded in ${((performance.now() - t0) / 1000).toFixed(0)} s${model.cache ? "; " + model.cache.summary() : ""}`);

let fail = 0;
if (LAYERS) {
  for (const free of env("FREE") === "both" ? [false, true] : [env("FREE") === "1"]) await perLayer(free);
  if (hasHead && env("GREEDY") === "1") await greedy();   // the same load: per-layer, then the goldens
} else await greedy();
if (hasHead && env("BENCH") === "1") await bench();
if (errors.count) { console.log(`${errors.count} GPU errors`); fail++; }
console.log(fail ? "Q38FLASH FAIL" : "Q38FLASH PASS ✓");
Deno.exit(fail ? 1 : 0);

// ---- per-layer check against the q38dump tensors
async function perLayer(FREE) {
  // free-running, small differences compound over 48 layers (the reference's own --act f32 run ends ~9e-2 off at
  // layer 47): the gate there is the argmax, the tolerance only flags a blow-up
  const DUMPS = env("Q38F_DUMPS", path.join(DIR, "dumps")), TOL = +env("TOL", FREE ? 2 : 2e-2);
  const seqs = (env("SEQS") || fs.readdirSync(DUMPS).filter((s) => /^s\d+$/.test(s)).sort().join(",")).split(",");
  // a q38flash_refdump.mjs directory: the reference's tensors ("REF"), with llama.cpp's forced residuals ("LLAMA")
  let refMode = false;
  const forced = new Set();
  const loadDump = (dir) => {
    const out = new Map();
    for (const line of fs.readFileSync(path.join(dir, "shapes.tsv"), "utf8").trim().split("\n")) {
      const [key, op, , , , , type] = line.split("\t");
      if (op === "REF") refMode = true;
      if (op === "LLAMA") forced.add(key);
      const b = fs.readFileSync(path.join(dir, key + (type === "i32" ? ".i32" : ".f32")));
      out.set(key, type === "i32" ? new Int32Array(b.buffer, b.byteOffset, b.length / 4) : new Float32Array(b.buffer, b.byteOffset, b.length / 4));
    }
    return out;
  };
  const rel = (a, b) => { let n = 0, dd = 0; for (let i = 0; i < b.length; i++) { const e = a[i] - b[i]; n += e * e; dd += b[i] * b[i]; } return Math.sqrt(n / (dd || 1)); };
  const worst = new Map();   // tensor kind -> [rel, key]
  let expertMiss = 0, rowMiss = 0, argmaxSame = 0, argmaxN = 0, compared = 0;
  const kind = (k) => k.replace(/-\d+(\.\d+)?$/, "").replace(/\.\d+$/, "");
  const note = (key, r) => { const k = kind(key); if (!worst.has(k) || worst.get(k)[0] < r) worst.set(k, [r, key]); compared++; };
  for (const s of seqs) {
    eng.reset();
    const chunks = fs.readdirSync(path.join(DUMPS, s)).filter((c) => /^c\d+$/.test(c)).sort((a, b) => +a.slice(1) - +b.slice(1));
    for (const c of chunks) {
      const cd = path.join(DUMPS, s, c), D = loadDump(cd);
      const ids = fs.readFileSync(path.join(cd, "tokens.txt"), "utf8").trim().split(/[\s,]+/).map(Number), T = ids.length;
      const slice = (key, t) => { const a = D.get(key); if (!a) return null; const n = a.length / T; return a.subarray(t * n, (t + 1) * n); };
      for (let t = 0; t < T; t++) {
        // the embedding and the n-gram rows (CPU side)
        note(`model.input_embed.0`, rel(eng._embedRow(ids[t]), slice("model.input_embed.0", t)));
        const force = FREE ? () => null : (name) => {
          const i = +name.split("-")[1];
          if (name.startsWith("in-")) return i === 0 ? slice("hc_init.0", t) : slice(`l_last-${i - 1}.0`, t);
          if (name.startsWith("mid-")) return slice(`hc_combine-${i}.0`, t);
          return slice(`l_last-${cfg.L - 1}.0`, t);
        };
        const got = await eng.debugToken(ids[t], force);
        const rows = slice("ple_rows.0", t);
        if (rows && eng.lastRows) for (let j = 0; j < rows.length; j++) if (rows[j] !== eng.lastRows[j]) rowMiss++;
        let moe = null;
        for (const [key, a] of got) {
          const ref = slice(key, t);
          if (!ref) continue;
          if (key.startsWith("ffn_moe_topk")) { const set = new Set(a); for (const e of ref) if (!set.has(e)) expertMiss++; continue; }
          if (ref.length !== a.length) { console.log(`  ${key}: size ${a.length} vs ${ref.length}`); fail++; continue; }
          if (forced.has(key)) continue;   // llama.cpp's residual, the reference's input: not the reference's output
          note(key, rel(a, ref));
          if (key.startsWith("ffn_moe_out")) moe = a;
          if (key.startsWith("ffn_shexp_gated") && moe) {
            const i = key.split("-")[1].split(".")[0], fo = slice(`ffn_out-${i}.0`, t);
            if (fo) note(`ffn_out-${i}.0`, rel(moe.map((v, j) => v + a[j]), fo));
          }
          if (key === "result_output.0") {
            const am = (x) => { let b = 0; for (let j = 1; j < x.length; j++) if (x[j] > x[b]) b = j; return b; };
            argmaxN++; if (am(a) === am(ref)) argmaxSame++;
          }
        }
      }
      console.log(`  ${s}/${c}: ${T} tokens`);
    }
  }
  // against llama.cpp's own tensors the engine's f32 activations (vs its Q8-rounded ones) compound within a half layer
  // and flip near-tied experts, so the tolerance is informational there (the argmax, the n-gram rows and the sizes
  // gate); against the JS reference (q38flash_refdump.mjs) it gates: every op equal to ~1e-6 where both read the
  // same weights, ~1e-2 where the engine reads a requantized copy, and the same expert sets
  console.log(`per-layer vs ${refMode ? "the JS reference (f32 activations)" : "llama.cpp's tensors"}, ${FREE ? "free-running" : "teacher-forced"}, layers 0..${nL - 1}${hasHead ? " + head" : ""}: ${compared} tensors compared`);
  let bad = 0;
  for (const [k, [r, key]] of [...worst].sort((a, b) => b[1][0] - a[1][0])) {
    const over = r > TOL && (refMode || FREE);
    if (over) bad++;
    console.log(`  ${k.padEnd(24)} worst rel ${r.toExponential(2)}  (${key})${over ? "  OVER " + TOL : ""}`);
  }
  console.log(`  expert ids not in llama.cpp's set: ${expertMiss}; n-gram rows differing: ${rowMiss}${argmaxN ? `; argmax equal ${argmaxSame}/${argmaxN}` : ""}`);
  if (bad || rowMiss || (refMode && !FREE && expertMiss) || (argmaxN && argmaxSame !== argmaxN && !FREE)) fail++;
}

// ---- greedy ids vs llama.cpp
async function greedy() {
  const gold = JSON.parse(fs.readFileSync(new URL("./golden/q38flash_llama_greedy.json", import.meta.url), "utf8"));
  const pick = env("CASES") ? env("CASES").split(",") : null;
  const cases = Object.entries(gold).filter(([k, v]) => !k.startsWith("_") && (!pick || pick.includes(k)) && v.prompt.length + v.ids.length <= eng.maxSeq);
  const tok = model.tokenizer();
  let tsAll = [];
  for (const [name, g] of cases) {
    eng.reset();
    const N = Math.min(+env("TOKENS", g.ids.length), g.ids.length);
    let t = performance.now();
    let r = await eng.prefillTokens(g.prompt, { wantLogits: true });
    const pf = (performance.now() - t) / 1000;
    const gen = [r.id], gaps = [gap(r.logits)];
    t = performance.now();
    while (gen.length < N) { r = await eng.forwardToken(gen[gen.length - 1], { wantLogits: true }); gen.push(r.id); gaps.push(gap(r.logits)); }
    const ts = (N - 1) / ((performance.now() - t) / 1000);
    tsAll.push(ts);
    // the first differing id; a difference at or after one of llama.cpp's near ties (top-2 gap < 0.25 there)
    // is reported but not a failure, as for the other goldens
    const diff = gen.findIndex((x, i) => x !== g.ids[i]);
    const tie = (g.nearTies || []).map(([i]) => i).filter((i) => i < N);
    const ok = diff < 0, tieOk = !ok && tie.some((i) => i <= diff);
    console.log(`${name}: prompt ${g.prompt.length} tok ${pf.toFixed(1)} s · decode ${ts.toFixed(2)} tok/s · ${ok ? `MATCH llama.cpp (${N} ids)` : tieOk ? `differs at ${diff} after llama.cpp's near tie at ${tie.filter((i) => i <= diff).join(",")} (not a failure)` : `MISMATCH at ${diff}`}`);
    console.log(`  engine: ${JSON.stringify(tok.decode(gen))}`);
    if (!ok) console.log(`  golden: ${JSON.stringify(g.text)}\n  ours ${gen.slice(Math.max(0, diff - 2), diff + 3)} gap ${gaps[diff]?.toFixed(3)} vs llama.cpp ${g.ids.slice(Math.max(0, diff - 2), diff + 3)} gap ${g.gap?.[diff]}`);
    const ourTies = gaps.map((x, i) => [i, x]).filter(([, x]) => x < 0.25).map(([i, x]) => `${i}:${x.toFixed(3)}`);
    console.log(`  near ties (top-2 gap < 0.25): ours [${ourTies.join(" ")}] llama.cpp [${(g.nearTies || []).map(([i, x]) => `${i}:${x}`).join(" ")}]`);
    if (!ok && !tieOk) fail++;
  }
  if (tsAll.length) console.log(`decode: ${(tsAll.reduce((a, b) => a + b) / tsAll.length).toFixed(2)} tok/s mean`);
}

// BENCH=1: decode tok/s without the logits readback (greedy ids only), and where one token's GPU time goes
// (each part submitted alone, 20 times, timed to queue completion; frame state as after the run)
async function bench() {
  const tok = model.tokenizer(), N = +env("BENCH_TOKENS", 64);
  eng.reset();
  let r = await eng.prefillTokens(tok.encode("Write a long story about a lighthouse keeper."));
  const t = performance.now();
  for (let i = 1; i < N; i++) r = await eng.forwardToken(r.id);
  console.log(`bench: decode ${((N - 1) / ((performance.now() - t) / 1000)).toFixed(2)} tok/s over ${N - 1} tokens (ids only)`);
  const time = async (name, ops) => {
    const reps = 20, t0 = performance.now();
    for (let k = 0; k < reps; k++) {
      const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
      eng._encode(pass, ops); pass.end(); device.queue.submit([enc.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    console.log(`  ${name.padEnd(28)} ${((performance.now() - t0) / reps).toFixed(2)} ms  (${ops.length} dispatches)`);
  };
  const all = (f) => eng.layers.flatMap(f);
  await time("token mixers + HC (attn half)", all((l) => l.attnOps));
  await time("  DeltaNet halves", all((l) => (l.isQ ? [] : l.attnOps)));
  await time("  attention halves", all((l) => (l.isQ ? l.attnOps : [])));
  await time("MoE halves", all((l) => l.ffnOps));
  await time("head", eng.headOps);
  await time("whole token", [...all((l) => [...l.attnOps, ...l.ffnOps]), ...eng.headOps]);
  // per kernel and launch shape, summed over the layers
  const groups = new Map();
  for (const o of [...all((l) => [...l.attnOps, ...l.ffnOps]), ...eng.headOps]) {
    const k = `${o.pipe.label} [${o.wg.join("x")}]`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(o);
  }
  for (const [k, ops] of groups) await time(k, ops);
}

function gap(lg) {
  let a = -Infinity, b = -Infinity;
  for (const v of lg) { if (v > a) { b = a; a = v; } else if (v > b) b = v; }
  return a - b;
}
