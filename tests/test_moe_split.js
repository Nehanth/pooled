// Qwen 3.6 35B-A3B MoE split across two engines on one GPU, driven the way the room drives a
// laptop + phone chain (room.js aiPrefill / roomGenerate / workerFrame), compared token by token
// with one engine holding every layer.
//   host   = layers [0, S) + embedding + LM head + the MTP draft block (blk.40)
//   worker = layers [S, L)
// Phases per prompt:
//   solo plain   prefillTokens + forwardToken, greedy                      (the reference)
//   solo spec    the room's solo speculative loop (prompt lookup + MTP)     must == solo plain
//   split plain  room-style split prefill (16/8/4-column frames + tail frame, draft-cache fill)
//                then one token per lap                                     compared with solo
//   split spec   same prefill, then the room's speculative loop over the chain (runTrunk, verify
//                chunks of NC, rollback riding on the next frame)          must == split plain
// Hidden states cross the "wire" as f16 like the room (WIRE=f32 to send f32).
//   deno run --unstable-webgpu --allow-read --allow-env tests/test_moe_split.js
// env: MOE=<gguf>  SPLIT=20[,5,36]  TOKENS=64  CTX=4096  NC=16  CASES=two-sum,hash-map,bash,copy
//      HOST_TUNE / WORKER_TUNE=WG,ROWS  (per-device cooperative GEMV tuning, as autotune picks it)
//      CKPT=0 (skip the checkpoint-after-rollback check)  CKPT_OLD=1 (also run it with the pre-fix protocol)
//      WIRE=f16|f32  MOE_FUSE=0  SPECFUSE=0  DRAFTCHAIN=0  SOLO=0 (skip the solo engine)
//      HOSTFUSE=0: the host's share of a lap as separate submits (default: the room's one-submit
//      paths, headAhead for split plain with one undone pick per answer, _hostTrunkFused for split spec)
//      BENCH=R: after the checks, R alternating rounds of split plain / split spec decode per case with
//      the host's one-submit paths off and on (same process, same GPU queue: no network), tokens/s
//      OPTS=0: tiled prefill attention, wide prefill GEMM and expert-grouped MoE prefill off (default: engine defaults)
//      SYNTH=1: a synthetic file (tests/e2e/synth.mjs --moe), prompts are token ids
// CPU-only check with a synthetic model (lavapipe):
//   node tests/e2e/synth.mjs /tmp/m.gguf --moe --mtp random
//   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json SYNTH=1 MOE=/tmp/m.gguf SPLIT=2,5 \
//     deno run --unstable-webgpu --allow-read --allow-env tests/test_moe_split.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { makeTokenizer, argmax } from "../engine/engine.js";
import { parseGGUFHeader, qwen35Weights, tokenizerFromGGUF, f32ToF16, f16ToF32 } from "../engine/gguf.js";
import { lookupDrafts } from "../room/lookup.js";
import { roomQwen35Options, applyRoomFlags } from "../engine/preset.js";
import { roomFlags } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const N = +env("TOKENS", 64), NC = +env("NC", 16), SYNTH = env("SYNTH", "0") === "1";
const PATH = env("MOE", new URL("../models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf", import.meta.url).pathname);
const CTX = +env("CTX", SYNTH ? 512 : 4096), WIRE = env("WIRE", "f16");
const openFile = async (path) => { const fh = await Deno.open(path);
  return async (off, len) => { await fh.seek(off, Deno.SeekMode.Start); const out = new Uint8Array(len); let got = 0;
    while (got < len) { const n = await fh.read(out.subarray(got)); if (n === null) break; got += n; } return out; }; };
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
const gpuErrs = [];
device.addEventListener?.("uncapturederror", (e) => { gpuErrs.push(e.error?.message); if (gpuErrs.length < 4) console.error("GPU ERROR:", e.error?.message); });
const readAt = await openFile(PATH);
const G = parseGGUFHeader((await readAt(0, 64 << 20)).buffer);
const nBlk = G.meta["qwen35.block_count"], nextn = G.meta["qwen35.nextn_predict_layers"] || 0, L = nBlk - nextn;
const bytesOf = (i) => readAt(i.byteOffset, i.byteLength);
const vocabRows = G.tensors["token_embd.weight"].shape[0];
// the room's engine options (engine/preset.js, what room.js aiLoadShard passes); the env knobs below are
// the room's ?flags (DRAFTCHAIN=0 is ?draftchain=0 ...), and ROOM_FLAGS takes any other
const flags = roomFlags({ ...(SYNTH ? { draftvocab: "0" } : {}), draftchain: env("DRAFTCHAIN", "1"), specfuse: env("SPECFUSE", "1"), moefuse: env("MOE_FUSE", "1"), hostfuse: env("HOSTFUSE", "1") });
const common = { device, meta: G.meta, maxSeq: CTX, vocab: vocabRows, ...roomQwen35Options(flags), batchCols: NC, coopRowsB: 1,
  // the prefill options come from the engine defaults, as in room.js (tiled prefill attention on every
  // device; wide GEMM + expert-grouped MoE on the device that holds the embedding, used by solo
  // prefillTokens only: the split prefill runs 16-column frames). OPTS=0: all three off everywhere (A/B).
  ...(env("OPTS", "") === "0" ? { attnPrefillTile: false, moeGroupPrefill: 0, prefillUbatch: 0 } : {}) };
// per-device kernel tuning (room.js autotuneCoop picks these per GPU): WG,ROWS for the worker,
// e.g. WORKER_TUNE=64,8 to stand in for a phone whose autotune differs from the host's
const tune = (v) => v ? (([wg, rows]) => ({ coopWG: wg, coopRows: rows }))(v.split(",").map(Number)) : {};
const mk = async (lo, hi, hasEmbed, hasHead) => {
  const t0 = performance.now();
  const weights = await qwen35Weights(G, bytesOf, { lo, hi, hasEmbed, hasHead, mtp: hasHead });
  const t = lo > 0 ? tune(env("WORKER_TUNE", "")) : hasEmbed && hi < L ? tune(env("HOST_TUNE", "")) : {};
  const e = applyRoomFlags(await Qwen35Engine.create({ ...common, ...t, weights, layerRange: [lo, hi], hasEmbed, hasHead }), flags);
  console.log(`  engine [${lo},${hi})${hasEmbed ? " +embed" : ""}${hasHead ? " +head" : ""}${e.mtp ? " +mtp" : ""}: moeFuse ${e.moeFuse}, attnPrefillTile ${e.attnPrefillTile}, moeGroupPrefill ${e.moeGrpU || "off"}, prefillUbatch ${e.ubatch || "off"}, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
  return e;
};

// ---- prompts ----
let CASES;
if (SYNTH) {
  // the synthetic model's answer length grows with the <|im_start|> markers in the context
  // (tests/e2e/synth.mjs counter): four turns' worth keeps it talking for ~100+ tokens
  const tv = G.meta["tokenizer.ggml.tokens"] || [], S0 = tv.indexOf("<|im_start|>"), E0 = tv.indexOf("<|im_end|>");
  const seq = (n, f) => Array.from({ length: n }, (_, i) => f(i) % Math.min(300, vocabRows));
  const turn = (body) => S0 >= 0 ? [S0, ...body, E0] : body;
  const conv = (...bodies) => [...bodies.flatMap(turn), ...(S0 >= 0 ? [S0] : [])];
  CASES = [["synth-a", conv(seq(9, (i) => 33 + i), seq(11, (i) => 50 + ((i * 7919) % 90)), seq(7, (i) => 3 + i * 5), seq(9, (i) => 90 + i))],
    ["synth-b", conv(seq(20, (i) => 40 + i * 3), seq(20, (i) => 40 + i * 3), seq(9, (i) => 7 + i), seq(5, (i) => 70 + i))],
    ["synth-c", conv(seq(30, (i) => 5 + ((i * i * 31) % 200)), seq(3, (i) => 9 + i), seq(12, (i) => 100 + i * 2), seq(4, (i) => 1 + i))]];
} else {
  const tok = makeTokenizer(tokenizerFromGGUF(G.meta)), V = tok.vocab;
  globalThis.__tok = tok;
  const chat = (q) => [V["<|im_start|>"], ...tok.encode("user\n" + q), V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n"), V["<think>"], ...tok.encode("\n\n"), V["</think>"], ...tok.encode("\n\n")];
  const file = "def total(items):\n    s = 0\n    for it in items:\n        s += it.price * it.qty\n    return s\n\n\ndef count(items):\n    return len(items)\n";
  CASES = [
    ["two-sum", chat("Write the Python code for two sum. Code only.")],
    ["hash-map", chat("Explain what a hash map is in two sentences.")],
    ["bash", chat("Write a bash one-liner that counts lines in all .js files.")],
    // a copy-heavy answer: prompt lookup drafts long runs (verify frames up to 15 columns)
    // Code-mode-like: a long system prompt with tool definitions (~2k tokens), answer is a tool call
    ["tools", [V["<|im_start|>"], ...tok.encode("system\nYou are a coding agent. You can call tools. To call a tool write <tool_call>\n<function=NAME>\n<parameter=ARG>\nVALUE\n</parameter>\n</function>\n</tool_call>.\nTools:\n"
      + Array.from({ length: 24 }, (_, i) => `- ${["read_file", "write_file", "list_dir", "run_shell", "search", "edit_file"][i % 6]}_${i}: ${"Takes a path and returns the contents of the file, or an error when it does not exist. ".repeat(3)}Parameters: path (string), start (int), end (int).\n`).join("")
      + "Here is the project file game.js:\n```js\n" + Array.from({ length: 40 }, (_, i) => `function step${i}(state) { state.t += ${i}; if (state.t > ${i * 7}) state.lines.push(${i}); return state; }\n`).join("") + "```"),
      V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("user\nAdd a score counter to game.js: write the complete new file with the write_file_1 tool."), V["<|im_end|>"], ...tok.encode("\n"),
      V["<|im_start|>"], ...tok.encode("assistant\n"), V["<think>"], ...tok.encode("\n\n"), V["</think>"], ...tok.encode("\n\n")]],
    ["copy", chat("Here is a file shop.py:\n```python\n" + file + "```\nRewrite the whole file with the function total renamed to compute_total. Output only the full file in a python code block.")],
  ];
}
const pick = env("CASES", "").split(",").filter(Boolean);
if (pick.length) CASES = CASES.filter(([n]) => pick.includes(n));
const dec = (ids) => globalThis.__tok ? JSON.stringify(globalThis.__tok.decode(ids)) : ids.join(",");

// ---- the wire ----
let wireMax = 0;
const wire = (f) => {
  if (WIRE !== "f16") return Float32Array.from(f);
  const out = new Float32Array(f.length);
  for (let i = 0; i < f.length; i++) { const a = Math.abs(f[i]); if (a > wireMax) wireMax = a; out[i] = f16ToF32(f32ToF16(f[i])); }
  return out;
};

// ---- solo (one engine, all layers) ----
async function soloPlain(eng, prompt) {
  eng.reset(); eng.mtpFill = true;
  await eng.prefillTokens(prompt.slice(0, -1));
  let lg = await eng.forwardToken(prompt[prompt.length - 1]);
  const gen = [argmax(lg)], margins = [margin(lg)];
  while (gen.length < N) { lg = await eng.forwardToken(gen[gen.length - 1]); gen.push(argmax(lg)); margins.push(margin(lg)); }
  return { gen, margins };
}
function margin(lg) { let a = -Infinity, b = -Infinity; for (const v of lg) { if (v > a) { b = a; a = v; } else if (v > b) b = v; } return a - b; }
// the room's speculative loop (roomGenerate): prompt lookup first, else the MTP head with K picked
// from 3, 5, 7 (here a fixed deterministic schedule instead of measured speed)
async function specLoop(host, first, fed0, spec) {
  const gen = [first], fed = [...fed0];
  let next = first, step = 0, lkFull = false, lookups = 0, rejects = 0;
  const Ks = [3, 3, 3, 5, 7, 3, 5, 7, 7, 7];
  while (gen.length < N) {
    const roomLeft = host.maxSeq - host.pos - 2;
    const K = Math.min(Ks[step++ % Ks.length], roomLeft, N - gen.length);
    const lkMax = lkFull ? (host.maxDrafts || 7) : 7;
    const lk = lookupDrafts([...fed, next], Math.min(lkMax, roomLeft, N - gen.length));
    const via = lk.length >= 2;
    const r0 = spec.rejects?.n || 0;
    const toks = via ? await host.specStepDrafts(next, argmax, lk, spec) : await host.specStep(next, argmax, K, spec);
    if (via) lookups++;
    lkFull = via && toks.length === lk.length + 1;
    fed.push(next, ...toks.slice(0, -1));
    gen.push(...toks); next = toks[toks.length - 1];
    rejects += (spec.rejects?.n || 0) - r0;
  }
  return { gen: gen.slice(0, N), lookups, rejects };
}
async function soloSpec(eng, prompt) {
  eng.reset(); eng.mtpFill = true; eng.mtp.stats = { drafts: 0, accepted: 0 };
  await eng.prefillTokens(prompt.slice(0, -1));
  const h = await eng.embedRun(prompt[prompt.length - 1], eng.pos);   // room solo: aiPipeToken (embedRun + head)
  eng.pos = prompt.length;
  const lg = await eng.headFromHidden(h);
  eng.setHidden(h);
  return specLoop(eng, argmax(lg), prompt, {});
}

// ---- split (room-style chain over host + worker) ----
function chain(host, worker) {
  const C = { pendingRb: null, frames: 0 };
  // workerFrame: a pending rollback applies before the frame's own work
  C.workerBatch = async (xs, basePos, n, specFlag) => {
    if (C.pendingRb != null) { worker.restoreDN(C.pendingRb); C.pendingRb = null; }
    const dim = worker.dims.dim, out = new Float32Array(n * dim);
    for (let c = 0; c < n; c += NC) {
      const m = Math.min(NC, n - c);
      out.set(await worker.runHiddenBatch(xs.subarray(c * dim, (c + m) * dim), basePos + c, specFlag ? { base: c, total: n } : false), c * dim);
    }
    C.frames++;
    return wire(out);
  };
  C.workerOne = async (x, pos) => {
    if (C.pendingRb != null) { worker.restoreDN(C.pendingRb); C.pendingRb = null; }
    C.frames++;
    return wire(await worker.runHidden(x, pos));
  };
  return C;
}
function fillDrafts(host, h, ids, i0, basePos, n) {   // room.js fillDrafts
  const dim = host.dims.dim;
  if (host._mtpFillBatch && host.B && n > 1 && n <= host.NC) {
    for (let c = 0; c < n; c++) host.device.queue.writeBuffer(host.B.x.buf, c * host.B.x.stride, h.subarray(c * dim, (c + 1) * dim));
    host._mtpFillBatch(ids, i0, basePos, n);
    return;
  }
  for (let c = 0; c < n; c++) {
    const next = ids[i0 + c + 1];
    if (next === undefined) break;
    host.setHidden(h.subarray(c * dim, (c + 1) * dim));
    host.mtpRun(null, next, basePos + c + 1, false);
  }
}
// room.js aiPrefill, split branch (TAIL_FRAME on): returns { logits, lastHidden }
async function splitPrefill(host, C, ids) {
  const dim = host.dims.dim;
  let i = 0, pos = 0;
  for (const W of [NC, ...[8, 4].filter((w) => w < NC)]) while (ids.length - 1 - i >= W) {
    const nChunks = Math.max(1, Math.min(Math.floor(16 / W), Math.floor((ids.length - 1 - i) / W)));
    const n = nChunks * W, basePos = pos, i0 = i;
    const hb = new Float32Array(n * dim);
    for (let c = 0; c < nChunks; c++) hb.set(await host.embedRunBatch(ids.slice(i + c * W, i + (c + 1) * W), basePos + c * W), c * W * dim);
    const h = await C.workerBatch(wire(hb), basePos, n, false);
    fillDrafts(host, h, ids, i0, basePos, n);
    pos = basePos + n; i += n;
  }
  const n = ids.length - i, basePos = pos, i0 = i;
  const hb = await host.embedRunBatch(ids.slice(i), basePos);
  const h = await C.workerBatch(wire(hb), basePos, n, false);
  fillDrafts(host, h, ids, i0, basePos, n);
  const lastHidden = h.slice((n - 1) * dim, n * dim);
  return { logits: await host.headFromHidden(lastHidden), lastHidden, pos: basePos + n };
}
async function splitPlain(host, worker, prompt) {
  host.reset(); worker.reset(); host.mtpFill = true;
  const C = chain(host, worker);
  let { logits, pos } = await splitPrefill(host, C, prompt);
  const gen = [argmax(logits)], margins = [margin(logits)];
  const t0 = performance.now();
  const ahead = host.hostFuse && host.canHeadAhead();
  let h1 = null;
  while (gen.length < N) {   // aiPipeToken: one token per lap
    h1 ||= await host.embedRun(gen[gen.length - 1], pos);
    const h = await C.workerOne(wire(h1), pos);
    pos++;
    h1 = null;
    if (ahead && gen.length + 1 < N) {
      // room.js plain loop (hostFuse): this lap's head and the next token's host layers in one submit
      const r = await host.headAhead(h, pos, { kind: "greedy" });
      gen.push(r.cands.ids[0]); margins.push(0);
      // once per answer, undo the step as a stop token would and take the separate path instead
      if (gen.length === 8 || !r.h) { host.dropAhead(); continue; }
      host.keepAhead(); h1 = r.h;
      continue;
    }
    logits = await host.headFromHidden(h);
    gen.push(argmax(logits)); margins.push(margin(logits));
  }
  return { gen, margins, ms: performance.now() - t0 };
}
async function splitSpec(host, worker, prompt) {
  host.reset(); worker.reset(); host.mtpFill = true; host.mtp.stats = { drafts: 0, accepted: 0 };
  const C = chain(host, worker);
  const { logits, lastHidden, pos } = await splitPrefill(host, C, prompt);
  host.setHidden(lastHidden); host.pos = pos;
  const rejects = { n: 0 };
  const spec = {
    rejects,
    preTrunk: true,   // room.js: the host's layers may already have run with the drafts (hostFuse)
    runTrunk: async (tokens, p, pre) => {
      const n = tokens.length, dim = host.dims.dim, hb = pre?.hs || new Float32Array(n * dim);
      if (!pre) for (let c = 0; c < n; c += NC) {
        const m = Math.min(NC, n - c);
        hb.set(await host.embedRunBatch(tokens.slice(c, c + m), p + c, { base: c, total: n }), c * dim);
      }
      return C.workerBatch(wire(hb), p, n, true);
    },
    onReject: async (k) => { C.pendingRb = k; rejects.n++; },
  };
  const t0 = performance.now();
  const r = await specLoop(host, argmax(logits), prompt, spec);
  return { ...r, stats: host.mtp.stats, ms: performance.now() - t0 };
}

// greedy runs keep going past <|im_end|>; what comes after the first end token is not an answer
const EOS = SYNTH ? new Set() : new Set(["<|im_end|>", "<|endoftext|>"].map((t) => globalThis.__tok.vocab[t]));
const answerLen = (g) => { const i = g.findIndex((t) => EOS.has(t)); return i < 0 ? g.length : i + 1; };
// Checkpoint after an answer whose last speculative step rejected drafts, the room's Code-mode path
// (room.js ckptSave / resetState / ckptResume, workerFrame): the rollback k is still pending for the
// worker when the save rides the next frame. The worker must roll back before saving (the host
// already did); then a reset, a load of the checkpoint and plain decoding from it must give exactly
// the tokens that continuing directly gives. oldOrder = true replays the pre-fix protocol (the
// rollback was dropped when that frame also carried a reset).
async function ckptCheck(host, worker, prompt, M, oldOrder) {
  host.reset(); worker.reset(); host.mtpFill = true;
  const C = chain(host, worker);
  const { logits, lastHidden, pos } = await splitPrefill(host, C, prompt);
  host.setHidden(lastHidden); host.pos = pos;
  const spec = {
    runTrunk: async (tokens, p) => {
      const n = tokens.length, dim = host.dims.dim, hb = new Float32Array(n * dim);
      for (let c = 0; c < n; c += NC) hb.set(await host.embedRunBatch(tokens.slice(c, c + Math.min(NC, n - c)), p + c, { base: c, total: n }), c * dim);
      return C.workerBatch(wire(hb), p, n, true);
    },
    onReject: async (k) => { C.pendingRb = k; },
  };
  let next = argmax(logits), steps = 0;
  const gen = [next];
  // speculate until a step ends with a rejection (its rollback still pending for the worker)
  while (steps < 8 || C.pendingRb == null) {
    const toks = await host.specStep(next, argmax, 3, spec);
    gen.push(...toks); next = toks[toks.length - 1];
    if (++steps > 200) throw new Error("no rejected step");
  }
  const rb = C.pendingRb; C.pendingRb = null;
  const p0 = host.pos;
  // the save frame: host saved its (rolled-back) state at the end of the answer
  host.saveSlot(1);
  if (!oldOrder) worker.restoreDN(rb);
  worker.saveSlot(1);
  // reference: continue straight on (the rollback applied), then the checkpoint path
  const cont = async () => {
    const out = [];
    let t = next, p = p0;
    for (let i = 0; i < M; i++) {
      const h = await C.workerOne(wire(await host.embedRun(t, p)), p);
      p++; t = argmax(await host.headFromHidden(h)); out.push(t);
    }
    return out;
  };
  if (oldOrder) worker.restoreDN(rb);   // the reference state (what a correct protocol leaves)
  const direct = await cont();
  host.reset(); worker.reset();         // another request in between started from scratch
  host.loadSlot(1); worker.loadSlot(1); // a later request resumes the checkpoint
  const resumed = await cont();
  host.dropAllSlots(); worker.dropAllSlots();
  return { d: firstDiff(direct, resumed), direct, resumed, rb, steps };
}

const firstDiff = (a, b) => { const n = Math.min(a.length, b.length); for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i; return a.length === b.length ? -1 : n; };
let fail = 0;
const splits = env("SPLIT", SYNTH ? "2,5" : "20").split(",").map(Number);
console.log(`${PATH.split("/").pop()}: ${L} trunk layers + ${nextn} MTP, ctx ${CTX}, NC ${NC}, wire ${WIRE}, ${N} tokens, splits ${splits.join(",")}`);
const ref = {};
if (env("SOLO", "1") !== "0") {
  const solo = await mk(0, L, true, true);
  for (const [name, prompt] of CASES) {
    const p = await soloPlain(solo, prompt);
    const s = await soloSpec(solo, prompt);
    const d = firstDiff(p.gen, s.gen);
    ref[name] = p;
    const inAns = d >= 0 && d < answerLen(p.gen);
    console.log(`[solo] ${name}: prompt ${prompt.length} tok, answer ${answerLen(p.gen)} tok, spec ${d < 0 ? "== plain" : `DIFFERS at ${d}${inAns ? "" : " (after the end token)"}, plain top-2 margin there ${p.margins[d].toFixed(3)}`} (${s.lookups} lookup steps)\n  ${dec(p.gen.slice(0, answerLen(p.gen)))}`);
    if (inAns) { fail++; console.log(`  spec: ${dec(s.gen)}`); }
  }
  solo.__drop = true;
}
for (const S of splits) {
  console.log(`--- split at ${S}: host [0,${S}) + embed/head/mtp, worker [${S},${L})`);
  const host = await mk(0, S, true, true);
  const worker = await mk(S, L, false, false);
  if (env("CKPT", "1") !== "0") {
    const [name, prompt] = CASES[0], M = +env("CKPT_TOKENS", 24);
    for (const oldOrder of env("CKPT_OLD", "0") === "1" ? [true, false] : [false]) {
      const r = await ckptCheck(host, worker, prompt, M, oldOrder);
      console.log(`[split ${S}] ${name}: checkpoint saved with a pending rollback (k=${r.rb}) then resumed, ${oldOrder ? "OLD protocol (rollback dropped)" : "fixed protocol"}: ${r.d < 0 ? `== continuing directly (${M} tokens)` : `DIFFERS at ${r.d}`}`);
      if (r.d >= 0) console.log(`  direct : ${dec(r.direct)}\n  resumed: ${dec(r.resumed)}`);
      if (r.d >= 0 && !oldOrder) fail++;
    }
  }
  for (const [name, prompt] of CASES) {
    const p = await splitPlain(host, worker, prompt);
    const s = await splitSpec(host, worker, prompt);
    const dSpec = firstDiff(p.gen, s.gen);
    const line = [`[split ${S}] ${name}: spec ${dSpec < 0 ? "== split plain" : "DIFFERS from split plain at " + dSpec}`,
      `(acc ${s.stats.accepted}/${s.stats.drafts}, ${s.lookups} lookup steps, ${s.rejects} rollbacks)`];
    if (ref[name]) {
      const d = firstDiff(ref[name].gen, p.gen);
      line.push(d < 0 ? "· plain == solo" : `· plain differs from solo at ${d} (solo top-2 margin there ${ref[name].margins[d].toFixed(3)})`);
    }
    console.log(line.join(" ") + `\n  plain: ${dec(p.gen.slice(0, answerLen(p.gen)))}`);
    if (dSpec >= 0) { fail++; console.log(`  spec : ${dec(s.gen)}`); }
    if (ref[name]) {
      const d = firstDiff(ref[name].gen, p.gen);
      if (d >= answerLen(ref[name].gen)) continue;   // both answers ended the same way
      // a split is not bit-exact with solo (f16 wire, batched prefill tail), but it must not go
      // off the rails: a divergence on a clear top-1 (margin > 1 logit) is a failure
      if (d >= 0 && ref[name].margins[d] > 1) { fail++; console.log(`  FAIL: split diverges from solo on a clear token (margin ${ref[name].margins[d].toFixed(3)})`); }
    }
  }
  const R = +env("BENCH", "0");
  if (R) for (const [name, prompt] of CASES) {
    const t = { plain: { off: [], on: [] }, spec: { off: [], on: [] } };
    const fuse0 = host.hostFuse;
    for (let r = 0; r < R; r++) for (const on of r % 2 ? [true, false] : [false, true]) {
      host.hostFuse = on;
      const p = await splitPlain(host, worker, prompt), s2 = await splitSpec(host, worker, prompt);
      if (firstDiff(p.gen, s2.gen) >= 0) { fail++; console.log(`  FAIL: bench ${name} hostFuse ${on}: spec != plain`); }
      t.plain[on ? "on" : "off"].push((N - 1) / (p.ms / 1000)); t.spec[on ? "on" : "off"].push((N - 1) / (s2.ms / 1000));
    }
    host.hostFuse = fuse0;
    const med = (a) => [...a].sort((x, y) => x - y)[a.length >> 1], f = (a) => a.map((x) => x.toFixed(1)).join(" ");
    for (const m of ["plain", "spec"]) console.log(`[bench split ${S}] ${name} ${m}: hostFuse off ${f(t[m].off)} (median ${med(t[m].off).toFixed(1)}) · on ${f(t[m].on)} (median ${med(t[m].on).toFixed(1)}) tok/s: ${((med(t[m].on) / med(t[m].off) - 1) * 100).toFixed(1)}%`);
  }
}
console.log(`largest |activation| on the wire: ${wireMax.toFixed(1)}; GPU errors: ${gpuErrs.length}`);
if (gpuErrs.length) fail++;
console.log(fail ? `MOE SPLIT FAIL (${fail})` : "MOE SPLIT PASS ✓");
if (fail) Deno.exit(1);
