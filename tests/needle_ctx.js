// Needle in a haystack at long context: for each length, a fresh prompt of about that many tokens
// (this repo's docs and source as the haystack) hides one sentence with a random passphrase at a
// given depth, then asks for it. Greedy answer must contain the passphrase, logits stay finite and
// the GPU reports no error. Also prints the whole-prompt prefill speed and the decode speed with
// that much context in the cache.
//   MODEL=moe|27b  LENS=32k,64k,96k,127k  DEPTHS=0.5 (one per length, or one for all)  KV=f16|q8
//   CTX=<maxSeq> (default: the smallest multiple of 256 that holds the longest prompt + decode)
//   TOKENS=32 (decode steps timed after the prompt)
//   cd tests && MODEL=moe LENS=32k,128k deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights needle_ctx.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { argmax } from "../engine/engine.js";
import { openGGUF, wideOpts, wideLimits } from "./load_model.js";
import { parseFills, parseKV, kvBytesPerPos } from "./ctx_plan.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "moe");
const PATH = MODEL === "27b" ? "../models/q38/model.gguf" : "../models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf";
const LENS = parseFills(env("LENS", "32k,64k,96k,127k"));
const DEPTHS = env("DEPTHS", "0.5").split(",").map(Number);
const KV = parseKV(env("KV", "f16"));
const N = +env("TOKENS", 32);
const MAXSEQ = +env("CTX", 0) || Math.ceil((Math.max(...LENS) + N + 16) / 256) * 256;
if (Math.max(...LENS) + N > MAXSEQ) { console.error(`LENS ${Math.max(...LENS)} + ${N} does not fit CTX ${MAXSEQ}`); Deno.exit(2); }

const model = openGGUF(PATH);
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, ...wideLimits(adapter) } });
let gpuErrors = 0;
device.addEventListener?.("uncapturederror", (e) => { if (gpuErrors++ < 4) console.error("GPU ERROR:", e.error?.message?.slice(0, 200)); });
console.log(`adapter limits: maxBufferSize ${(adapter.limits.maxBufferSize / 2 ** 20).toFixed(0)} MiB, maxStorageBufferBindingSize ${(adapter.limits.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MiB`);

const G = model.G, m = G.meta, nBlk = m["qwen35.block_count"], L = nBlk - (m["qwen35.nextn_predict_layers"] || 0);
console.log(`${MODEL}: maxSeq ${MAXSEQ}, KV cache ${(kvBytesPerPos(m, L, KV) * MAXSEQ / 2 ** 30).toFixed(2)} GB (${KV}), lengths ${LENS.join(" / ")}`);
const tok = model.tokenizer();
let t0 = performance.now();
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true });
const eng = await Qwen35Engine.create({ device, meta: m, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: MAXSEQ, batchCols: 16, coopRowsB: 1, kvQ8: KV === "q8", ...wideOpts() });
if ((KV === "q8") !== !!eng.kvQ8) { console.error(`KV=${KV} asked, engine kvQ8 ${!!eng.kvQ8}`); Deno.exit(2); }
console.log(`loaded in ${((performance.now() - t0) / 1000).toFixed(0)}s; KV buffers: ${eng.kvLayout ? JSON.stringify(eng.kvLayout) : "one per layer"}`);

// haystack: paragraphs of this repo's docs and source, tokenized once, reused in order (and repeated)
const V = tok.vocab, nl = tok.encode("\n");
const files = [];
const walk = (dir, re) => { for (const e of Deno.readDirSync(new URL(dir, import.meta.url))) { if (e.isDirectory) walk(dir + e.name + "/", re); else if (re.test(e.name)) files.push(dir + e.name); } };
walk("../docs/", /\.md$/); walk("../engine/", /\.js$/); walk("../room/", /\.js$/);
files.sort();
const paras = [];
let total = 0;
const need = Math.max(...LENS);
for (const f of files) {
  for (const p of Deno.readTextFileSync(new URL(f, import.meta.url)).split(/\n\s*\n/)) {
    if (!p.trim() || /passphrase/i.test(p)) continue;
    const ids = tok.encode(p.slice(0, 4000) + "\n\n");
    paras.push(ids); total += ids.length;
  }
  if (total > need * 1.1) break;
}
console.log(`haystack pool: ${paras.length} paragraphs, ${total} tokens from ${files.length} files`);

const words = ["amber", "falcon", "quartz", "willow", "harbor", "cobalt", "saffron", "glacier", "meadow", "ember", "orchid", "lantern"];
const rnd = (n) => Math.floor(Math.random() * n);
const Q = "What is the secret passphrase for the blue vault that was mentioned in the documents above? Reply with the passphrase only.";
const rows = [];
let fail = 0;
for (let li = 0; li < LENS.length; li++) {
  const len = LENS[li], depth = DEPTHS[Math.min(li, DEPTHS.length - 1)];
  const pass = `${words[rnd(12)]}-${words[rnd(12)]}-${1000 + rnd(9000)}`;
  const needle = tok.encode(`Important note: the secret passphrase for the blue vault is ${pass}. Remember it.\n\n`);
  const head = [V["<|im_start|>"], ...tok.encode("user\nRead the following documents carefully.\n\n")];
  const tail = [...tok.encode("\n" + Q), V["<|im_end|>"], ...nl, V["<|im_start|>"], ...tok.encode("assistant\n"), V["<think>"], ...tok.encode("\n\n"), V["</think>"], ...tok.encode("\n\n")];
  const budget = len - head.length - tail.length - needle.length;
  const hay = []; let pi = 0, placed = false;
  const at = Math.floor(budget * depth);
  while (hay.length < budget) {
    if (!placed && hay.length >= at) { for (const x of needle) hay.push(x); placed = true; continue; }
    const p = paras[pi++ % paras.length];
    for (let j = 0; j < p.length && hay.length < budget + (placed ? 0 : needle.length); j++) hay.push(p[j]);
  }
  if (!placed) for (const x of needle) hay.push(x);
  const ids = [...head, ...hay.slice(0, budget + needle.length), ...tail];
  eng.reset();
  t0 = performance.now();
  await eng.prefillTokens(ids.slice(0, -1));
  let logits = await eng.forwardToken(ids[ids.length - 1]);
  const pfS = (performance.now() - t0) / 1000;
  let finite = true;
  const chk = (lg) => { for (let i = 0; i < lg.length; i += 97) if (!Number.isFinite(lg[i])) finite = false; };
  chk(logits);
  const gen = []; let next = argmax(logits); gen.push(next);
  t0 = performance.now();
  for (let i = 1; i < N; i++) { logits = await eng.forwardToken(next); chk(logits); next = argmax(logits); gen.push(next); }
  const decTs = (N - 1) / ((performance.now() - t0) / 1000);
  const end = gen.indexOf(V["<|im_end|>"]);
  const answer = tok.decode(end >= 0 ? gen.slice(0, end) : gen).trim();
  const found = answer.includes(pass);
  const ok = found && finite && gpuErrors === 0;
  if (!ok) fail++;
  const r = { len: ids.length, depth, prefillTokPerS: +(ids.length / pfS).toFixed(1), prefillS: +pfS.toFixed(1), decodeTokPerS: +decTs.toFixed(2), passphrase: pass, answer: answer.slice(0, 80), found, finite };
  rows.push(r);
  console.log(`len ${String(r.len).padStart(6)} depth ${depth}: prefill ${r.prefillTokPerS} tok/s (${r.prefillS}s) · decode ${r.decodeTokPerS} tok/s · needle ${found ? "FOUND" : "MISSED"} ${JSON.stringify(r.answer)} (want ${pass})${finite ? "" : " NaN!"}`);
}
console.log("RESULT " + JSON.stringify({ model: MODEL, kv: KV, maxSeq: MAXSEQ, rows, gpuErrors }));
console.log(fail || gpuErrors ? "NEEDLE FAIL" : "NEEDLE PASS");
if (fail || gpuErrors) Deno.exit(1);
