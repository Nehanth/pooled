// Load this device's layers: room.js aiLoadShard without the DOM. Same GPU bring-up (a throwaway
// device for the self-test and kernel micro tests, then autotuneCoop), the same weight loaders
// (ggufWeights / qwen35Weights, each matrix uploaded as it is converted so RAM stays flat) and the
// same engine options (engine/preset.js roomQwen35Options + applyRoomFlags), so a node's layers
// compute what a browser's would. Qwen3.8-Flash-Next (kind qwen4exp) loads with qwen4expWeights into its own engine
// (engine/qwen4exp.js, imported when needed: it reads the WebGPU globals as it loads) behind the room adapter
// (room/flash.js), and the device holding the n-gram layer reads that table's rows from the model file.
import fs from "node:fs";
import { autotuneCoop, makeTokenizer, DenseEngine, gpuSelfTest, kernelMicroTests } from "../../engine/engine.js";
import { Qwen35Engine } from "../../engine/qwen35.js";
import { ggufWeights, ggufShardBytes, qwen35Weights, qwen35ShardBytes, tokenizerFromGGUF, gpuUploadEntry, GGML_EMBED, GGML_FINAL_NORM, GGML_OUTPUT,
  ggmlLayerNames, qwen35NamesFor, qwen4expConfig, qwen4expNamesFor, qwen4expWeights, qwen4expShardBytes, QWEN4EXP_HEAD } from "../../engine/gguf.js";
import { ngramConfig, NgramTable, fileReadAt, httpReadAt } from "../../engine/ngram.js";
import { FlashRoomEngine } from "../../room/flash.js";
import { hostMemory } from "./env.js";
import { roomQwen35Options, applyRoomFlags } from "../../engine/preset.js";
import { maxSeqFor, kvModeFor } from "../../room/models.js";
import { convertPool } from "./convert.js";
import { ExpertStore } from "../../engine/expert_store.js";

// The tensors a shard's loader reads (engine/gguf.js ggufWeights / qwen35Weights), in about its order,
// as { byteOffset, byteLength, shard? }: what source.js fetches ahead of it when it streams (shard: the file of a split GGUF)
export function shardTensors(G, kind, { lo, hi, hasEmbed, hasHead, mtp = false }) {
  const names = [];
  const strings = (o) => Object.values(o).filter((v) => typeof v === "string");
  // qwen4exp: the dense phase skips QSA's indexer (qwen4expWeights qsa: false), and the n-gram table is read by rows
  const layer = (i) => (kind === "qwen35" ? strings(qwen35NamesFor(G, i)) : kind === "qwen4exp" ? qwen4expTensorNames(G, i) : strings(ggmlLayerNames(i)));
  for (let i = lo; i < hi; i++) names.push(...layer(i));
  if (hasEmbed || hasHead) names.push(GGML_EMBED);
  if (hasHead) names.push(...(kind === "qwen4exp" ? Object.values(QWEN4EXP_HEAD) : [GGML_FINAL_NORM]), GGML_OUTPUT);
  if (kind === "qwen35" && mtp && hasHead) {
    const N = G.meta["qwen35.block_count"] - 1, p = `blk.${N}.nextn.`;
    if (G.tensors[p + "eh_proj.weight"]) names.push(...strings(qwen35NamesFor(G, N, true)), ...["eh_proj", "enorm", "hnorm", "shared_head_norm"].map((n) => p + n + ".weight"));
  }
  const seen = new Set(), out = [];
  for (const n of names) { const t = G.tensors[n]; if (t && !seen.has(n)) { seen.add(n); out.push({ byteOffset: t.byteOffset, byteLength: t.byteLength, ...(t.shard ? { shard: t.shard } : {}) }); } }
  return out;
}

// The tensors of one qwen4exp layer qwen4expWeights loads in the dense phase (no QSA indexer), in its order
const QWEN4EXP_INDEXER = new Set(["idxQ", "idxK", "idxQNorm", "idxKNorm"]);
export function qwen4expTensorNames(G, i) {
  return Object.entries(qwen4expNamesFor(G, i)).filter(([k, v]) => typeof v === "string" && !QWEN4EXP_INDEXER.has(k)).map(([, v]) => v);
}

// offload (expert offload, room/plan.js dealRoom): { lo, hi, vramBytes, ramBytes, ramCap }: the routed experts of
// layers [lo, hi) are parked in RAM (engine/expert_store.js ExpertStore) with a GPU cache of vramBytes (its slot pools +
// prefill region) instead of uploaded. ramBytes: what the deal says they park; ramCap: the RAM this device lends
// less the RAM the deal keeps free there (ramSpare; roomnode.js offloadOk): parking past it fails the load. A Qwen3.5 / 3.6 MoE shard on a room node only.
// the layers an ai-load's offload covers within this shard's range, or null for none
export function offloadLayers(o, range) {
  const lo = Math.max(range[0], +o?.lo || 0), hi = Math.min(range[1], +o?.hi || 0);
  return hi > lo && o.vramBytes > 0 ? Array.from({ length: hi - lo }, (_, i) => lo + i) : null;
}

// A model's transformer layers: config.json's num_hidden_layers for a dense GGUF (qwen3-*), the GGUF
// index for qwen35 (block_count less the MTP layers). Keyed on the model's kind, not its name: reading
// qwen35.* keys from a dense model's index gives NaN, and a [0, NaN] range loads no layers at all.
export async function modelLayers(src) {
  if (src.M.kind === "qwen4exp") return qwen4expConfig((await src.header(false)).meta).L;
  if (src.M.kind === "qwen35") {
    const m = (await src.header(false)).meta;
    return m["qwen35.block_count"] - (m["qwen35.nextn_predict_layers"] || 0);
  }
  return (await src.cfg()).num_hidden_layers;
}

// [lo, hi) must be whole layer numbers, 0 <= lo <= hi. A NaN bound used to load an engine with no
// layers that still "worked": embed -> final norm -> tied head, the same token every step.
export function checkRange(range) {
  const [lo, hi] = range || [];
  if (!Number.isInteger(lo) || !Number.isInteger(hi) || lo < 0 || hi < lo) throw new Error(`bad layer range [${lo}, ${hi}]`);
}

// -> { device, engine, tok, cfg, G, tune, gpuErrors, experts }
export async function loadShard({ modelKey, range, hasEmbed, hasHead, ctx = maxSeqFor(modelKey), kv = kvModeFor(modelKey, null),
  src, flags = "", onProgress = () => {}, log = () => {}, selfTest = true, onGpuError = () => {}, offload = null }) {
  checkRange(range);
  const M = src.M;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter on this machine");
  const device = await adapter.requestDevice({ requiredLimits: {
    maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  const out = { device, gpuErrors: 0 };
  let pool = null;
  device.addEventListener?.("uncapturederror", (ev) => { if (out.gpuErrors++ < 3) onGpuError(ev.error?.message || "GPU error"); });
  // a lost device fails every later readback with a bare AbortError: say why it was lost
  device.lost?.then((i) => { out.lost = i; if (i?.reason !== "destroyed") log(`GPU device lost (${i?.reason || "unknown"}): ${String(i?.message || "").split("\n")[0].slice(0, 300)}`); }).catch(() => {});
  // a buffer the GPU had no memory for is an error buffer: the layers would load "fine" and every answer come out as
  // garbage (seen on the RTX 5070 with the 122B's experts filling the RAM WDDM spills VRAM into). The whole load runs
  // in one out-of-memory scope and fails when anything in it ran out
  device.pushErrorScope?.("out-of-memory");
  let scoped = !!device.pushErrorScope;
  try {
    // the index first, so a streamed shard's weights download while the GPU tests and tunes below run
    const kind = M.kind === "qwen35" || M.kind === "qwen4exp" ? M.kind : "gguf";
    const needTok = kind !== "gguf" && (hasEmbed || hasHead);
    const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead, ...(kind === "qwen35" ? { mtp: hasHead } : {}) };
    const G0 = await src.header(needTok);
    // requantizing the tensors the GPU has no kernel for (Q4_1 / K-quants -> Q8) on worker threads:
    // seconds per tensor on the event loop otherwise (convert.js)
    pool = convertPool({ log });
    G0.convert = pool.convert;
    src.plan?.(shardTensors(G0, kind, opts));
    if (selfTest) {
      // an adapter gives out one device only; the same GPU as the shard's (env.js highPerformance)
      const tdev = await (await navigator.gpu.requestAdapter({ powerPreference: "high-performance" })).requestDevice();
      const st = await gpuSelfTest(tdev);
      if (!st.ok) throw new Error("GPU self-test FAILED on this device: " + st.detail);
      const mt = await kernelMicroTests(tdev);
      if (!mt.ok) throw new Error("GPU kernel FAILED on this device: " + mt.firstFail);
      try { tdev.destroy(); } catch {}
    }
    out.tune = await autotuneCoop(device).catch(() => ({ wg: 256, rows: 4 }));
    log(`autotune WG=${out.tune.wg} ROWS=${out.tune.rows}`);
    const upload = (e, name) => gpuUploadEntry(device, e, name === GGML_EMBED);   // straight to the GPU
    if (kind === "qwen4exp") {
      const r = await loadFlash({ device, G0, src, range, hasEmbed, hasHead, ctx, offload, onProgress, log, out });
      out.engine = r;
    } else if (M.kind === "qwen35") {
      const G = out.G = G0;
      out.cfg = { num_hidden_layers: G.meta["qwen35.block_count"] - (G.meta["qwen35.nextn_predict_layers"] || 0) };
      if (needTok) { out.tok = makeTokenizer(tokenizerFromGGUF(G.meta)); out.tok.chatTemplate = G.meta["tokenizer.chat_template"] || ""; }
      const total = qwen35ShardBytes(G, opts);
      const off = offload && offloadLayers(offload, range);
      if (off?.length) {
        out.experts = new ExpertStore(device, { layers: off, vramBytes: offload.vramBytes, ramBytes: +offload.ramCap || Infinity, expectBytes: +offload.ramBytes || 0 });
        log(`expert offload: layers ${off[0]}-${off[off.length - 1]} park their experts in RAM, ${(offload.vramBytes / 2 ** 30).toFixed(2)} GB GPU cache`);
      }
      const weights = await qwen35Weights(G, src.bytesOf, { ...opts, experts: out.experts || null }, (done) => onProgress(done, total), upload);
      out.engine = await Qwen35Engine.create({
        device, meta: G.meta, weights, vocab: G.tensors[GGML_EMBED]?.shape?.[0],
        layerRange: range, hasEmbed, hasHead, maxSeq: ctx, coopWG: out.tune.wg, coopRows: out.tune.rows,
        ...roomQwen35Options(flags), kvQ8: kv === "q8",
      });
    } else {
      out.cfg = await src.cfg();
      if (hasEmbed || hasHead) out.tok = makeTokenizer(await src.tokJson());
      const G = out.G = G0;   // vocab comes from tokenizer.json
      const total = ggufShardBytes(G, opts);
      const weights = await ggufWeights(G, src.bytesOf, opts, (done) => onProgress(done, total), upload);
      out.engine = await DenseEngine.create({ coopWG: out.tune.wg, coopRows: out.tune.rows, device, cfg: out.cfg, weights, layerRange: range, hasEmbed, hasHead, maxSeq: ctx });
    }
    applyRoomFlags(out.engine, flags);
    if (scoped) {
      scoped = false;
      const oom = await device.popErrorScope();
      if (oom) throw Object.assign(new Error(`the GPU ran out of memory while loading layers ${range[0]}-${range[1] - 1}: ${String(oom.message || "").split("\n")[0].slice(0, 200)}`), { oom: true });
    }
    // a device lost during the load (on the GB10 an allocation past the memory there is VK_ERROR_OUT_OF_DEVICE_MEMORY
    // and a lost device, not an out-of-memory error in the scope): say so now, not with the first answer's AbortError
    await new Promise((r) => setTimeout(r, 0));
    if (out.lost) throw Object.assign(new Error(`the GPU was lost while loading layers ${range[0]}-${range[1] - 1}: ${String(out.lost.message || out.lost.reason || "").split("\n")[0].slice(0, 200)}`), { oom: /MEMORY/i.test(String(out.lost.message)) });
    if (out.experts) {
      log(out.experts.summary().replace(/; \d+ cuts.*/, ""));
      if (offload.ramBytes > 0 && out.experts.parkedBytes > offload.ramBytes * 1.0001)
        log(`expert offload: parked ${(out.experts.parkedBytes / 2 ** 30).toFixed(2)} GiB, more than the ${(offload.ramBytes / 2 ** 30).toFixed(2)} GiB the deal planned (within what this device lends)`);
    }
    return out;
  } catch (e) { try { await out.engine?.close?.(); } catch {} try { await out.ngram?.close?.(); } catch {} try { device.destroy(); } catch {} throw e; }
  finally { if (out.G) delete out.G.convert; await pool?.close(); }
}

// ---------------- Qwen3.8-Flash-Next (qwen4exp) ----------------
// What a qwen4exp shard holds on the GPU besides its weights: K / V in f32 for its attention layers at ctx positions,
// the DeltaNet state per recurrent layer, the PLE ring on the n-gram layer, scratch. -> bytes
export function flashShardBytes(G, { lo, hi, hasHead, ctx }) {
  const c = qwen4expConfig(G.meta), ss = c.ssm;
  let b = qwen4expShardBytes(G, { lo, hi, hasEmbed: false, hasHead, qsa: false });
  for (let i = lo; i < hi; i++) {
    if (c.qsa[i]) b += 2 * ctx * c.nKVHeads * c.headDim * 4;
    else b += ss.nVHeads * ss.stateSize * ss.stateSize * 4 + (2 * ss.nKHeads * ss.stateSize + ss.inner) * (ss.convK - 1) * 4;
  }
  if (c.ple && c.ple.layers.some((l) => l >= lo && l < hi)) b += 10 * c.hDim * 4;
  return b + 64 * 2 ** 20;   // scratch, uniforms, staging
}

// Before a qwen4exp load: does this device have the memory for its share? On unified memory (a GB10, Apple silicon)
// an allocation past what is free does not fail in WebGPU: the OS kills the process or swaps, so say it up front.
// mem: env.js hostMemory() (detectMemory). -> null when it fits (or nothing is known), else the reason
export function flashMemoryShort(bytes, mem, { slackGB = 2 } = {}) {
  const gb = (n) => (n / 2 ** 30).toFixed(1);
  if (mem?.kind === "unified") {
    const free = mem.availGB ?? mem.freeGB;
    if (free > 0 && bytes / 2 ** 30 > free - slackGB) return `needs ${gb(bytes)} GB and this computer has ${free.toFixed(1)} GB of unified memory free (${slackGB} GB kept for the system)`;
  } else if (mem?.kind === "discrete" && mem.freeGB > 0 && bytes / 2 ** 30 > mem.freeGB - 0.5)
    return `needs ${gb(bytes)} GB of GPU memory and ${mem.name || "the GPU"} has ${mem.freeGB.toFixed(1)} GB free`;
  return null;
}

// The n-gram table's row reader for the device holding the n-gram layer: positional reads in the local model file
// (its own handle: the source's are closed when the load ends), or HTTP range requests to the shard's URL when the
// model is streamed (16 round trips a token: it works, slowly). close() releases it.
export async function openNgramTable(src, G, log = () => {}) {
  const cfg = ngramConfig(G), gb = (cfg.nRows * cfg.rowBytes / 2 ** 30).toFixed(1);
  const local = src.localPaths?.[cfg.shard];
  if (local) {
    const fh = await fs.promises.open(local, "r");
    const handles = [];
    handles[cfg.shard] = fh;
    const t = new NgramTable(cfg, { readAt: fileReadAt(handles), layout: "tensor" });
    t.close = async () => { t.close = null; await fh.close().catch(() => {}); };
    t.where = local;
    log(`n-gram table: ${gb} GB, rows read from ${local}`);
    return t;
  }
  const urls = src.urls || [src.url];
  if (!urls[cfg.shard]) throw new Error("the n-gram table's file has no URL");
  const t = new NgramTable(cfg, { readAt: httpReadAt(urls), layout: "tensor", inflight: 16 });
  t.close = async () => { t.close = null; };
  t.where = urls[cfg.shard];
  log(`n-gram table: ${gb} GB, rows read by HTTP range from ${urls[cfg.shard]} (no local copy: pooled pull keeps one)`);
  return t;
}

// One qwen4exp shard: the weights (each matrix uploaded as it is converted; the embedding stays on the CPU, where the
// engine reads its rows), the n-gram table when the range holds its layer, the engine, the room adapter.
// -> FlashRoomEngine
async function loadFlash({ device, G0: G, src, range, hasEmbed, hasHead, ctx, offload, onProgress, log, out, mem = null }) {
  const c = qwen4expConfig(G.meta);
  out.G = G;   // (loadShard's finally takes the conversion pool off it)
  out.cfg = { num_hidden_layers: c.L };
  if (range[1] > c.L) throw new Error(`bad layer range [${range[0]}, ${range[1]}]: the model has ${c.L} layers`);
  if (offload && offloadLayers(offload, range)) throw new Error("Qwen3.8-Flash-Next has no expert offload yet: this device can hold only layers that fit in what it lends");
  if (hasEmbed || hasHead) { out.tok = makeTokenizer(tokenizerFromGGUF(G.meta)); out.tok.chatTemplate = G.meta["tokenizer.chat_template"] || ""; }
  const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead, qsa: false };
  const need = flashShardBytes(G, { ...opts, ctx });
  const short = flashMemoryShort(need, mem || hostMemory());
  if (short) throw Object.assign(new Error(`can't hold layers ${range[0]}-${range[1] - 1} of Qwen3.8-Flash-Next here: it ${short}`), { oom: true, short: true });
  const total = qwen4expShardBytes(G, opts);
  // straight to the GPU, but not the embedding (Qwen4ExpEngine reads its rows on the CPU) nor f32 tensors (the engine
  // uploads those itself, some joined into one buffer). A layer at a time, the queue flushed after each: Dawn keeps
  // a writeBuffer's staging copy until the queue moves on, and 67 GB of uploads in one go ran the GB10 out of memory
  // (VK_ERROR_OUT_OF_DEVICE_MEMORY, the device lost) with the weights only half on the GPU
  const up = (e, name) => { if (name !== GGML_EMBED) gpuUploadEntry(device, e, false); };
  const flush = async () => { device.queue.submit([device.createCommandEncoder().finish()]); await device.queue.onSubmittedWorkDone(); };
  let base = 0;
  const part = async (o) => {
    const w = await qwen4expWeights(G, src.bytesOf, { ...o, qsa: false }, (done) => onProgress(base + done, total), up);
    base += qwen4expShardBytes(G, { ...o, qsa: false });
    await flush();
    return w;
  };
  const layers = [];
  let pleTable = null;
  for (let i = range[0]; i < range[1]; i++) {
    const w = await part({ lo: i, hi: i + 1, hasEmbed: false, hasHead: false });
    layers.push(w.layers[0]);
    pleTable ||= w.pleTable || null;
  }
  const weights = { ...(await part({ lo: range[1], hi: range[1], hasEmbed, hasHead })), layers, ...(pleTable ? { pleTable } : {}) };
  if (weights.pleTable) out.ngram = await openNgramTable(src, G, log);
  const { Qwen4ExpEngine } = await import("../../engine/qwen4exp.js");
  const eng = await Qwen4ExpEngine.create({ device, meta: G.meta, weights, layerRange: range, hasEmbed, hasHead, maxSeq: ctx, ngram: out.ngram || null });
  const fe = FlashRoomEngine.wrap(eng);
  out.flash = eng;   // the engine under the adapter (tests and benches: forwardToken, debugToken)
  log(`Qwen3.8-Flash-Next layers ${range[0]}-${range[1] - 1}${hasHead ? " + embedding/head" : ""}: ${(need / 2 ** 30).toFixed(1)} GB on the GPU${out.ngram ? ", the n-gram layer here" : ""}${fe.ckpt ? "" : "; no checkpoints (its K/V can't be copied out)"}`);
  return fe;
}
