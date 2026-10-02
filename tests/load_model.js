// Shared model loading for the Deno GPU tests and benchmarks.
//
// openGGUF(path): header + tokenizer + a weights() loader, reading with node:fs (about 2x the
// throughput of Deno.FsFile seek/read) through the converted-weights cache (tests/weight_cache.js;
// WEIGHT_CACHE=0 disables it). Fresh weights per call, exactly as each test loaded them before.
//
// sharedQ38(device): the one-process runner's model. Loads the whole 27B once (64 layers, embed,
// head, MTP block), uploads every matrix and norm to the GPU once, and hands out views of that
// set: weights({lo, hi, hasEmbed, hasHead, mtp}) returns the same structure qwen35Weights would,
// with each entry already carrying its GPU buffer (entry.gpu). Qwen35Engine.create uses entry.gpu
// as-is (the same hook the browser's streamed upload uses), so any number of engines, whole or
// partial, share one copy of the weights. The engine never writes to weight buffers.
import fs from "node:fs";
import { parseGGUFHeader, qwen35Weights, tokenizerFromGGUF, gpuUploadEntry } from "../engine/gguf.js";
import { makeTokenizer } from "../engine/engine.js";
import { attachWeightCache } from "./weight_cache.js";
import { Qwen35Engine, prefillMathFeatures } from "../engine/qwen35.js";

// A/B switches for every test and bench that loads through this file (engine defaults, not per test):
//   ATTN_PREFILL_TILE=0|1 tiled causal flash attention for full-width prefill passes (engine/wgsl/attn_tile.js;
//                         default on for every model; 0 forces attn_flash)
//   ATTN_PREFILL_TK=4|8|16  its positions per tile (default: the largest that fits the workgroup memory;
//                         16 needs 32 KB, which gpuDevice() then requests from the adapter)
//   ATTN_PREFILL_SPLITS=N its target number of context splits per pass (default 32)
//   PREFILL_DP4A=0|1      the wide prefill GEMM as int8 dot products on Q8_1-quantized activations
//                         (engine option prefillDp4a; needs dot4I8Packed and the wide path)
//   ATTN_DECODE=v1|v2     decode/verify attention: v2 = split-K attn_dec (engine/wgsl/attn_dec.js, the default), v1 = attn_flash
//   ATTN_DECODE_SPLITS=N  its most splits per column (default 256 / kv heads)
const envGet = (k) => globalThis.Deno?.env.get(k);
if (envGet("ATTN_PREFILL_TILE")) Qwen35Engine.defaults.attnPrefillTile = envGet("ATTN_PREFILL_TILE") !== "0";
if (envGet("ATTN_PREFILL_TK")) Qwen35Engine.defaults.attnPrefillTK = +envGet("ATTN_PREFILL_TK");
if (envGet("ATTN_TILE_KVH")) Qwen35Engine.defaults.attnTileKvh = envGet("ATTN_TILE_KVH") !== "0";
if (envGet("ATTN_TILE_PF")) Qwen35Engine.defaults.attnTilePf = envGet("ATTN_TILE_PF") !== "0";
if (envGet("ATTN_TILE_NR")) Qwen35Engine.defaults.attnTileNr = envGet("ATTN_TILE_NR") !== "0";
if (envGet("ATTN_PREFILL_SPLITS")) Qwen35Engine.defaults.attnPrefillSplits = +envGet("ATTN_PREFILL_SPLITS");
if (envGet("PREFILL_DP4A")) Qwen35Engine.defaults.prefillDp4a = envGet("PREFILL_DP4A") !== "0";
// MOE_DP4A=0|1: the MoE's dp4a expert kernels (engine option moeGroupDp4a) off / on for every test
if (envGet("MOE_DP4A")) Qwen35Engine.defaults.moeGroupDp4a = envGet("MOE_DP4A") !== "0";
// PREFILL_UBATCH=N (0: off) for every test, including those that do not spread wideOpts() (e.g. test_q38_bits.js)
if (envGet("PREFILL_UBATCH")) Qwen35Engine.defaults.prefillUbatch = +envGet("PREFILL_UBATCH");
if (envGet("ATTN_DECODE")) Qwen35Engine.defaults.attnDecode = envGet("ATTN_DECODE");
if (envGet("ATTN_DECODE_SPLITS")) Qwen35Engine.defaults.attnDecodeSplits = +envGet("ATTN_DECODE_SPLITS");

// The room's engine settings for a Deno test or bench (engine/preset.js): ROOM_FLAGS takes the room's own
// query-string switches, e.g. ROOM_FLAGS="draftvocab=0&kv=q8". Unset: exactly what the room runs.
// extra: { flag: value } defaults a script sets on top (ROOM_FLAGS still wins).
export function roomFlags(extra = {}) {
  const q = new URLSearchParams(extra);
  for (const [k, v] of new URLSearchParams(envGet("ROOM_FLAGS") || "")) q.set(k, v);
  return q;
}

export const Q38_PATH = new URL("../models/q38/model.gguf", import.meta.url).pathname;
export const MOE_PATH = new URL("../models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf", import.meta.url).pathname;

// layers before the MTP ("nextn") block, when the file has one
export function trunkLayers(G) {
  const n = G.meta["qwen35.block_count"];
  return n - (G.meta["qwen35.nextn_predict_layers"] || (G.tensors[`blk.${n - 1}.nextn.eh_proj.weight`] ? 1 : 0));
}

export function openGGUF(path, { skipTokenizer = false, cache = true, headerBytes = 64 << 20 } = {}) {
  const fd = fs.openSync(path, "r");
  const readAt = (off, len) => {
    const out = new Uint8Array(len);
    let o = 0;
    while (o < len) { const n = fs.readSync(fd, out, o, Math.min(len - o, 1 << 30), off + o); if (n <= 0) break; o += n; }
    return out;
  };
  const G = parseGGUFHeader(readAt(0, headerBytes).buffer, { skipTokenizer });
  const wcache = cache ? attachWeightCache(G, path) : null;
  const bytesOf = (info) => readAt(info.byteOffset, info.byteLength);
  let tok = null;
  return {
    path, G, meta: G.meta, readAt, bytesOf, cache: wcache,
    trunkLayers: trunkLayers(G),
    tokenizer: () => (tok ||= makeTokenizer(tokenizerFromGGUF(G.meta))),
    weights: (range, onProgress, onEntry) => qwen35Weights(G, bytesOf, range, onProgress, onEntry),
    close: () => fs.closeSync(fd),
  };
}

export async function gpuDevice() {
  const adapter = await navigator.gpu.requestAdapter();
  // PREFILL_MATH=sgmatrix asks for the tensor-core features where the adapter has them (Chrome only; none in Deno)
  const device = await adapter.requestDevice({ requiredFeatures: prefillMathFeatures(adapter), requiredLimits: {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, ...wideLimits(adapter),
    ...(Qwen35Engine.defaults.attnPrefillTK >= 16 ? { maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize } : {}) } });
  return { adapter, device };
}

// Wide prefill A/B for any Deno test or bench (engine option prefillUbatch; default 256 on the MoE, off on dense):
//   PREFILL_UBATCH=256 [PREFILL_TILE='{"BM":64,"BN":64,"TM":4,"TN":4}'] [WGMEM=0: keep the 16 KB default]
//   PREFILL_UBATCH=0 forces it off (unset: the engine default)
// wideOpts() -> engine options; wideLimits(adapter) -> device limits (the adapter's workgroup memory,
// so the tile can take two quant blocks per K stage).
export function wideOpts() {
  const e = Deno.env.get("PREFILL_UBATCH"), T = Deno.env.get("PREFILL_TILE");
  return e ? { prefillUbatch: +e, ...(T ? { prefillTile: JSON.parse(T) } : {}) } : T ? { prefillTile: JSON.parse(T) } : {};
}
export function wideLimits(adapter) {
  return +(Deno.env.get("PREFILL_UBATCH") || 0) && Deno.env.get("WGMEM") !== "0"
    ? { maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize } : {};
}

// Prefill logits tolerance: relDiff (max |diff| / max |logit| of the next-token logits) allowed between two
// prefill paths of the same prompt. Dense: 2e-3 (tests/test_batch_q38.js). MoE: 2e-2. Top-8 routing over 256
// experts turns a last-bit summation-order change into a different expert for tokens whose router scores
// nearly tie, and that compounds over 40 layers. The baseline is the MoE's own 16-column batched prefill with
// the old kernels: it already sits 2e-3..2.3e-2 from token-by-token decode at 700+ tokens in Deno (0.16..0.17 in
// Chrome on the bench page's HTML), with argmax, greedy text and spec == plain identical. The default MoE prefill
// options (tiled attention, wide GEMM, expert-grouped FFN) land inside that band (up to 1.7e-2 in Deno, 5.7e-2 in
// Chrome, docs/bench-log.md 2026-09-27), so 2e-2 is the baseline's width, not a loosening for them. Since the
// baseline can itself land past 2e-2 (2.49e-2 from token-by-token at 700 tokens of test_prefill_opts' prompt,
// 2026-10-01, with every neighbouring length under 1e-3), test_prefill_opts breaks a miss with the token-by-token
// logits: the options pass a length when they match those even though the baseline does not.
export const prefillTol = (moe) => (moe ? 2e-2 : 2e-3);

// Count (and print the first few) uncaptured GPU errors; tests read errors.count.
export function watchGpuErrors(device, print = 3) {
  const errors = { count: 0 };
  device.addEventListener?.("uncapturederror", (e) => { errors.count++; if (errors.count <= print) console.error("GPU ERROR:", e.error?.message?.slice(0, 200)); });
  return errors;
}

// Everything a q38 test needs. Standalone: fresh weights per weights() call (cached conversion).
export async function q38Context({ skipTokenizer = false } = {}) {
  const { adapter, device } = await gpuDevice();
  const model = openGGUF(Q38_PATH, { skipTokenizer });
  return { adapter, device, model, errors: watchGpuErrors(device), shared: false };
}

// Upload every quantized matrix and f32 tensor of a weight set once; engines then reuse entry.gpu.
// The embedding keeps its CPU copy (per-token row lookups) and is never uploaded here.
// Flushes every ~2 GB (a submit, then waiting for the queue): on Metal (Deno/wgpu), ~15 GB of staged writes with no
// submit in between came out with every f32 tensor reading back as zeros (the 27B's norms, so NaN logits in
// run_q38_once.js); 13.9 GB was still intact. docs/bench-log.md, 2026-09-27 M5 Max.
export async function preuploadWeights(device, w, flushBytes = 2 * 2 ** 30) {
  let bytes = 0, since = 0;
  const flush = async () => { device.queue.submit([device.createCommandEncoder().finish()]); await device.queue.onSubmittedWorkDone(); since = 0; };
  const f32 = (e) => {
    const src = new Uint8Array(e.data.buffer, e.data.byteOffset, e.data.byteLength);
    const buf = device.createBuffer({ size: Math.ceil(src.byteLength / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
    new Uint8Array(buf.getMappedRange()).set(src);
    buf.unmap();
    return { kind: "f32", buf };
  };
  const up = async (e) => {
    if (!e || e.gpu) return;
    const b0 = bytes;
    if (e.kind === "q4" || e.kind === "q8") { bytes += e.qs.byteLength + e.scales.byteLength; gpuUploadEntry(device, e, false); }
    else if (e.kind === "f32") { bytes += e.data.byteLength; e.gpu = f32(e); }   // keep e.data: some layers read it directly
    if ((since += bytes - b0) >= flushBytes) await flush();
  };
  const layer = async (L) => { for (const v of Object.values(L)) if (v && typeof v === "object" && "kind" in v) await up(v); };
  for (const L of w.layers) await layer(L);
  await up(w.finalNorm); await up(w.head);
  if (w.mtp) { await layer(w.mtp.layer); await up(w.mtp.ehProj); await up(w.mtp.enorm); await up(w.mtp.hnorm); await up(w.mtp.sharedHeadNorm); }
  await flush();
  return bytes;
}

// The one-process runner's context: the 27B loaded and uploaded once, shared by every engine.
export async function sharedQ38Context() {
  const { adapter, device } = await gpuDevice();
  const errors = watchGpuErrors(device);
  const model = openGGUF(Q38_PATH);
  const L = model.trunkLayers;
  let t0 = performance.now();
  const full = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true });
  const tRead = performance.now() - t0;
  t0 = performance.now();
  const up = await preuploadWeights(device, full);
  const tUp = performance.now() - t0;
  console.log(`shared 27B: ${L} layers + head + mtp read/convert ${(tRead / 1000).toFixed(1)} s, GPU upload ${(up / 2 ** 30).toFixed(2)} GB in ${(tUp / 1000).toFixed(1)} s${model.cache ? "; " + model.cache.summary() : ""}`);
  // same structure as qwen35Weights(G, ..., range), minus the load
  model.weights = async ({ lo, hi, hasEmbed = false, hasHead = false, mtp = false }) => {
    if (lo < 0 || hi > L || lo >= hi) throw new Error(`shared weights hold layers 0..${L - 1}, asked ${lo}..${hi - 1}`);
    const out = { layers: full.layers.slice(lo, hi) };
    if (hasEmbed || hasHead) out.embed = full.embed;
    if (hasHead) { out.finalNorm = full.finalNorm; out.head = full.head; }
    if (mtp && hasHead && full.mtp) out.mtp = full.mtp;
    return out;
  };
  return { adapter, device, model, errors, shared: true };
}
