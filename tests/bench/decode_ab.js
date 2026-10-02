// Decode A/B in one process (Deno): two arms that differ only in engine fields set at runtime, run in
// alternating blocks so both share the weights, the load and any GPU contention. Per block:
//   wall: plain greedy decode through forwardTokenIds (GPU sampling, the room's path), ms/token
//   gpu:  the same N positions pre-encoded and submitted back to back (no readback in between), ms/token:
//         the GPU's own time per token, without Deno's ~11 ms per-sync cost that dominates the wall
// and checks that both arms produce the same tokens and the same logits bits (hash of a forwardToken).
//   cd tests && A='{"dnSplit":0}' B='{"dnSplit":4}' OPTS='{"dnSplit":4}' deno run --unstable-webgpu --allow-read --allow-env \
//     --allow-write=$HOME/.cache/swarmllm-weights bench/decode_ab.js
//   MODEL=moe|27b  ROUNDS=6  BLOCK=24  CTX=0 (tokens of context before decoding; 0 = a short chat prompt)
//   OPTS: extra Qwen35Engine.create options (the B arm's kernels must be built at create)
//   SPEC=1: speculative steps (K=3, GPU sampling) instead: wall ms per token only (no back-to-back), tokens compared
import { Qwen35Engine } from "../../engine/qwen35.js";
import { openGGUF, gpuDevice, trunkLayers, MOE_PATH, Q38_PATH } from "../load_model.js";
import { roomQwen35Options } from "../../engine/preset.js";
import { gpuGreedy } from "../gpusample_check.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const SPEC = env("SPEC", "0") === "1";
const MODEL = env("MODEL", "moe"), ROUNDS = +env("ROUNDS", 6), BLOCK = +env("BLOCK", 24), CTX = +env("CTX", 0);
const A = JSON.parse(env("A", "{}")), B = JSON.parse(env("B", "{}")), OPTS = JSON.parse(env("OPTS", "{}"));
const model = openGGUF(MODEL === "moe" ? MOE_PATH : Q38_PATH);
const { device } = await gpuDevice();
const G = model.G, L = trunkLayers(G), tok = model.tokenizer();
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true });
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: Math.max(2048, CTX + 4 * BLOCK + 64),
  ...roomQwen35Options(""), ...OPTS });
const setArm = (o) => { for (const [k, v] of Object.entries(o)) { if (v && typeof v === "object" && eng[k] && typeof eng[k] === "object") Object.assign(eng[k], v); else eng[k] = v; } eng._fwdPre = null; };
let ids = tok.encode("<|im_start|>user\nWrite a long, detailed essay about the history of computing.<|im_end|>\n<|im_start|>assistant\n");
if (CTX > ids.length) {
  let src = [];
  for (const f of ["../../room.js", "../../engine/gguf.js", "../../engine/wgsl/base.js", "../../harness/agent.js"]) { src.push(...tok.encode(await Deno.readTextFile(new URL(f, import.meta.url)))); if (src.length > CTX) break; }
  while (src.length < CTX) src = src.concat(src);
  ids = [...src.slice(0, CTX - ids.length), ...ids];
}
console.log(`${MODEL}: A ${JSON.stringify(A)} vs B ${JSON.stringify(B)}, opts ${JSON.stringify(OPTS)}, context ${ids.length} tokens`);
const fnv = (u32) => { let h = 0x811c9dc5; for (let i = 0; i < u32.length; i++) { h ^= u32[i]; h = Math.imul(h, 0x01000193) >>> 0; } return h.toString(16); };

async function specBlock(arm) {
  setArm(arm);
  eng.reset(); eng.mtpFill = true; eng.mtp.stats = { drafts: 0, accepted: 0 };
  await eng.prefillTokens(ids.slice(0, -1));
  const pick = Object.assign((c) => c.ids[0], { gpu: { kind: "greedy" } });
  let t = gpuGreedy(await eng.forwardTokenIds(ids.at(-1)));
  const out = [t];
  for (let i = 0; i < 2; i++) { const r = await eng.specStep(t, pick, 3); out.push(...r); t = r.at(-1); }
  const n0 = out.length, t0 = performance.now();
  while (out.length - n0 < BLOCK) { const r = await eng.specStep(t, pick, 3); out.push(...r); t = r.at(-1); }
  const wall = (performance.now() - t0) / (out.length - n0);
  return { wall, gpu: 0, out: out.slice(0, n0 + BLOCK), hash: `${eng.mtp.stats.accepted}/${eng.mtp.stats.drafts}` };
}
async function block(arm) {
  if (SPEC) return specBlock(arm);
  setArm(arm);
  eng.reset();
  await eng.prefillTokens(ids.slice(0, -1));
  let t = gpuGreedy(await eng.forwardTokenIds(ids.at(-1)));
  const out = [t];
  for (let i = 0; i < 4; i++) { t = gpuGreedy(await eng.forwardTokenIds(t)); out.push(t); }
  let t0 = performance.now();
  for (let i = 0; i < BLOCK; i++) { t = gpuGreedy(await eng.forwardTokenIds(t)); out.push(t); }
  const wall = (performance.now() - t0) / BLOCK;
  eng._fwdPre = null;
  const lg = await eng.forwardToken(t);   // the logits bits of the next position
  const hash = fnv(new Uint32Array(lg.buffer));
  // back to back: the next BLOCK positions, pre-encoded (state is garbage afterwards; the next block resets)
  const pos0 = eng.pos, emb = eng._embedRowF32(t), desc = { kind: "greedy" }, jobs = [];
  for (let i = 0; i < BLOCK; i++) jobs.push(eng._encodeForward(pos0 + i, desc));
  await device.queue.onSubmittedWorkDone();
  t0 = performance.now();
  for (let i = 0; i < BLOCK; i++) { eng._setFrame(pos0 + i, pos0 + i + 1); device.queue.writeBuffer(eng.x, 0, emb); device.queue.submit([jobs[i].cb]); }
  await device.queue.onSubmittedWorkDone();
  const gpu = (performance.now() - t0) / BLOCK;
  eng._fwdPre = null;
  return { wall, gpu, out, hash };
}
const res = { A: { wall: [], gpu: [] }, B: { wall: [], gpu: [] } };
let ref = null, same = true; const hashes = { A: new Set(), B: new Set() };
for (let r = 0; r < ROUNDS; r++) {
  for (const k of r % 2 ? ["B", "A"] : ["A", "B"]) {
    const b = await block(k === "A" ? A : B);
    res[k].wall.push(b.wall); res[k].gpu.push(b.gpu); hashes[k].add(b.hash);
    if (!ref) ref = b.out; else same &&= b.out.every((x, i) => i >= ref.length || x === ref[i]);
  }
}
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };
for (const k of ["A", "B"]) console.log(`${k}: wall ${res[k].wall.map((x) => x.toFixed(2)).join(" ")} | gpu ${res[k].gpu.map((x) => x.toFixed(2)).join(" ")} ms/token | logits ${[...hashes[k]].join(",")}`);
const wa = med(res.A.wall), wb = med(res.B.wall), ga = med(res.A.gpu), gb = med(res.B.gpu);
const bitsSame = hashes.A.size === 1 && hashes.B.size === 1 && [...hashes.A][0] === [...hashes.B][0];
console.log(`median wall A ${wa.toFixed(2)} B ${wb.toFixed(2)} ms (${((wa / wb - 1) * 100).toFixed(1)}% tok/s); gpu A ${ga.toFixed(3)} B ${gb.toFixed(3)} ms (${((ga / gb - 1) * 100).toFixed(1)}%); tokens ${same ? "identical" : "DIFFER"}; logits ${bitsSame ? "same bits" : "DIFFER"}`);
Deno.exit(same ? 0 : 1);
