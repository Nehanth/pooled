// Per-kernel GPU time for one decode token, from timestamp queries: every dispatch runs in its own compute
// pass with begin/end timestamps (so per-pass overhead is included, but kernels are measured on the GPU).
// Usage: MOE=<gguf> or Q38=1 for the dense 27B.
import { Qwen35Engine } from "../engine/qwen35.js";
import { makeTokenizer } from "../engine/engine.js";
import { qwen35Weights, tokenizerFromGGUF } from "../engine/gguf.js";
import { openGGUF, roomFlags } from "./load_model.js";
import { roomQwen35Options, applyRoomFlags } from "../engine/preset.js";
const MOEK = Deno.env.get("MOE_KERNEL") ? (Deno.env.get("MOE_KERNEL").startsWith("{") ? JSON.parse(Deno.env.get("MOE_KERNEL")) : Deno.env.get("MOE_KERNEL")) : undefined;   // moeKernel: legacy | default | JSON
const PATH = Deno.env.get("Q38") ? "../models/q38/model.gguf" : (Deno.env.get("MOE") || "../models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf");
const model = openGGUF(PATH);   // node:fs reads through the converted-weights cache (tests/weight_cache.js; WEIGHT_CACHE=0 disables)
const readAt = model.readAt;
const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredFeatures: ["timestamp-query"], requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
const G = model.G; const tok = makeTokenizer(tokenizerFromGGUF(G.meta));
const arch = G.meta["general.architecture"], L = G.meta[arch + ".block_count"] - (G.meta[arch + ".nextn_predict_layers"] || 0);
const weights = await qwen35Weights(G, (i) => readAt(i.byteOffset, i.byteLength), { lo: 0, hi: L, hasEmbed: true, hasHead: true });
// the room's settings (engine/preset.js); MOE_FUSE=0 / MOE_DN_ROWS=N are ?moefuse / ?moednrows, ROOM_FLAGS takes any other
const flags = roomFlags({ ...(Deno.env.get("MOE_FUSE") ? { moefuse: Deno.env.get("MOE_FUSE") } : {}), ...(Deno.env.get("MOE_DN_ROWS") ? { moednrows: Deno.env.get("MOE_DN_ROWS") } : {}) });
const eng = applyRoomFlags(await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 512,
  vocab: G.tensors["token_embd.weight"]?.shape?.[0], ...roomQwen35Options(flags), moeKernel: MOEK,
  moeNormRouter: Deno.env.get("MOE_NORM_ROUTER") !== "0" }), flags);   // MOE_NORM_ROUTER=0: rmsnorm + router GEMV launches instead of moe_nrt (A/B)   // MOE_FUSE=0: unfused MoE kernels (A/B)
if (eng.moeK) console.log("moeKernel", JSON.stringify(eng.moeK), "moeFuse", eng.moeFuse);
const ids = tok.encode("The capital of France is"); for (const id of ids) await eng.forwardToken(id);
// wall time, normal path
let t0 = performance.now(); for (let i = 0; i < 20; i++) await eng.forwardToken(1); const wall = (performance.now() - t0) / 20;
// instrumented path (encode-ahead off: a command buffer recorded before the hook would carry no timestamps)
eng.encodeAhead = false; eng._fwdPre = null;
const MAXQ = 4096, qs = device.createQuerySet({ type: "timestamp", count: MAXQ });
const res = device.createBuffer({ size: MAXQ * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
const rd = device.createBuffer({ size: MAXQ * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
let names = [], nq = 0;
const origCreate = device.createCommandEncoder.bind(device);
device.createCommandEncoder = (d) => { const enc = origCreate(d); const ob = enc.beginComputePass.bind(enc);
  enc.beginComputePass = () => { let pipe = null, name = "?"; const bgs = {};
    return { setPipeline(p) { pipe = p; name = eng._pname.get(p) || "?"; }, setBindGroup(i, b) { bgs[i] = b; },
      dispatchWorkgroups(x, y = 1, z = 1) { const p = ob({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: nq, endOfPassWriteIndex: nq + 1 } }); nq += 2; names.push(name);
        p.setPipeline(pipe); for (const i in bgs) p.setBindGroup(+i, bgs[i]); p.dispatchWorkgroups(x, y, z); p.end(); },
      dispatchWorkgroupsIndirect(buf, off) { const p = ob({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: nq, endOfPassWriteIndex: nq + 1 } }); nq += 2; names.push(name);
        p.setPipeline(pipe); for (const i in bgs) p.setBindGroup(+i, bgs[i]); p.dispatchWorkgroupsIndirect(buf, off); p.end(); }, end() {} }; };
  const ofin = enc.finish.bind(enc); enc.finish = () => { if (nq) enc.resolveQuerySet(qs, 0, nq, res, 0); if (nq) enc.copyBufferToBuffer(res, 0, rd, 0, nq * 8); return ofin(); };
  return enc; };
eng._pname = new Map(Object.entries(eng.pipes).map(([k, v]) => [v, k]));
const agg = {}; const RUNS = 5;
for (let r = 0; r < RUNS; r++) { names = []; nq = 0; await eng.forwardToken(1); await rd.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(rd.getMappedRange().slice(0, nq * 8)); rd.unmap();
  names.forEach((n, i) => { const ns = Number(t[2 * i + 1] - t[2 * i]); (agg[n] ||= [0, 0])[0] += ns / 1e6 / RUNS; agg[n][1] += 1 / RUNS; }); }
const gpu = Object.values(agg).reduce((a, b) => a + b[0], 0);
console.log(`${arch}: wall ${wall.toFixed(2)} ms/token (${(1000 / wall).toFixed(1)} tok/s) · sum of kernel GPU time ${gpu.toFixed(2)} ms · dispatches ${Math.round(Object.values(agg).reduce((a, b) => a + b[1], 0))}`);
for (const [k, [ms, n]] of Object.entries(agg).sort((a, b) => b[1][0] - a[1][0])) console.log(`  ${k.padEnd(22)} ${ms.toFixed(2).padStart(7)} ms  ${String(Math.round(n)).padStart(4)}×  ${(ms / n * 1000).toFixed(1).padStart(7)} µs each`);
