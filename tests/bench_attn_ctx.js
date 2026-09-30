// Decode speed and attention GPU time vs context position, without prefilling the context.
// The KV rows past the short prompt are left as they are (zeros): attention reads every row anyway,
// so its cost at position P is the same as with a real P-token context. Decode tok/s here is plain
// decode (forwardToken), wall time, at each fill; attention GPU time comes from timestamp queries
// around the attention dispatches of one token (kv_store + attn_flash* + attn_combine*).
// Output text is meaningless past the prompt; for correctness at long context use bench_ctx.js.
//   MODEL=moe|27b  FILLS=4096,16384,32768,65536  TOKENS=24  CTX=<maxSeq, default max fill + 256>
//   ATTN_DECODE=v1|v2|both (engine option attnDecode; both = A/B in one process, alternating)
//   cd tests && MODEL=moe deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights bench_attn_ctx.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { openGGUF, trunkLayers, roomFlags } from "./load_model.js";
import { roomQwen35Options, applyRoomFlags } from "../engine/preset.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "moe");
const PATH = MODEL === "27b" ? "../models/q38/model.gguf" : "../models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf";
const FILLS = env("FILLS", "1024,4096,16384,32768,65536").split(",").map(Number);
const N = +env("TOKENS", 24);
const MAXSEQ = +env("CTX", Math.max(...FILLS) + 256);
const MODES = env("ATTN_DECODE", "v1") === "both" ? ["v1", "v2"] : [env("ATTN_DECODE", "v1")];
const REPS = +env("REPS", 2);

const model = openGGUF(PATH);
const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredFeatures: ["timestamp-query", ...(ad.features.has("subgroups") ? ["subgroups"] : [])],
  requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
let gpuErrors = 0;
device.addEventListener?.("uncapturederror", (e) => { if (gpuErrors++ < 4) console.error("GPU ERROR:", e.error?.message?.slice(0, 300)); });
const G = model.G, L = trunkLayers(G);
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true });
// the room's settings (engine/preset.js; ROOM_FLAGS for others), with the decode attention under test
const flags = roomFlags();
const eng = applyRoomFlags(await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: MAXSEQ,
  ...roomQwen35Options(flags), attnDecode: MODES.includes("v2") ? "v2" : "v1" }), flags);
console.log(`${MODEL}: maxSeq ${MAXSEQ}, faSplit ${eng.faSplit} (${eng.faSplits} slots), attnDecode compiled: ${!!eng.pipes.attn_dec}, subgroups ${device.features.has("subgroups")}`);
const tok = model.tokenizer();
const prompt = tok.encode("Write a short story about a lighthouse keeper.");

// timestamp instrumentation (prof_ts.js style): each dispatch in its own timed pass
const MAXQ = 4096, qs = device.createQuerySet({ type: "timestamp", count: MAXQ });
const res = device.createBuffer({ size: MAXQ * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
const rd = device.createBuffer({ size: MAXQ * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const pname = new Map(Object.entries(eng.pipes).map(([k, v]) => [v, k]));
let names = [], nq = 0, hook = false;
const origCreate = device.createCommandEncoder.bind(device);
device.createCommandEncoder = (d) => {
  const enc = origCreate(d);
  if (!hook) return enc;
  const ob = enc.beginComputePass.bind(enc);
  enc.beginComputePass = () => { let pipe = null, name = "?"; const bgs = {};
    const run = (f) => { const p = ob({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: nq, endOfPassWriteIndex: nq + 1 } }); nq += 2; names.push(name);
      p.setPipeline(pipe); for (const i in bgs) p.setBindGroup(+i, bgs[i]); f(p); p.end(); };
    return { setPipeline(p) { pipe = p; name = pname.get(p) || "?"; }, setBindGroup(i, b) { bgs[i] = b; },
      dispatchWorkgroups(x, y = 1, z = 1) { run((p) => p.dispatchWorkgroups(x, y, z)); },
      dispatchWorkgroupsIndirect(b, o) { run((p) => p.dispatchWorkgroupsIndirect(b, o)); }, end() {} }; };
  const ofin = enc.finish.bind(enc);
  enc.finish = () => { if (nq) { enc.resolveQuerySet(qs, 0, nq, res, 0); enc.copyBufferToBuffer(res, 0, rd, 0, nq * 8); } return ofin(); };
  return enc;
};
const isAttn = (n) => /^(kv_store|attn_flash|attn_combine|attn_dec)/.test(n);
async function attnTime(id) {
  eng._fwdPre = null; const ea = eng.encodeAhead; eng.encodeAhead = false;
  hook = true; names = []; nq = 0;
  await eng.forwardToken(id);
  hook = false; eng.encodeAhead = ea;
  await rd.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(rd.getMappedRange().slice(0, nq * 8)); rd.unmap();
  let attn = 0, all = 0; const per = {};
  names.forEach((n, i) => { const ms = Number(t[2 * i + 1] - t[2 * i]) / 1e6; all += ms; if (isAttn(n)) { attn += ms; per[n] = (per[n] || 0) + ms; } });
  return { attn, all, per };
}

eng.reset();
let logits = null;
for (const id of prompt) logits = await eng.forwardToken(id);
const base = eng.pos;
const rows = [];
for (const fill of FILLS) {
  for (let r = 0; r < REPS; r++) for (const mode of MODES) {
    eng.attnDecode = mode; eng._fwdPre = null;
    eng.pos = fill;
    for (let i = 0; i < 3; i++) await eng.forwardToken(1);   // warm-up (+ first-use compiles)
    const t0 = performance.now();
    for (let i = 0; i < N; i++) await eng.forwardToken(1);
    const ms = (performance.now() - t0) / N;
    eng.pos = fill + 3 + N;
    const at = [await attnTime(1), await attnTime(1)].sort((a, b) => a.attn - b.attn)[0];
    const row = { fill, mode, msPerTok: +ms.toFixed(2), tokPerS: +(1000 / ms).toFixed(2), attnMs: +at.attn.toFixed(3), gpuMs: +at.all.toFixed(2),
      per: Object.fromEntries(Object.entries(at.per).map(([k, v]) => [k, +v.toFixed(3)])) };
    rows.push(row);
    console.log(`fill ${String(fill).padStart(6)} ${mode}: ${row.tokPerS} tok/s (${row.msPerTok} ms) · attention ${row.attnMs} ms of ${row.gpuMs} ms GPU · ${JSON.stringify(row.per)}`);
  }
}
eng.pos = base;
console.log("RESULT " + JSON.stringify({ model: MODEL, maxSeq: MAXSEQ, rows, gpuErrors }));
if (gpuErrors) Deno.exit(1);
