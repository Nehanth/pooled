// Load this device's layers: room.js aiLoadShard without the DOM. Same GPU bring-up (a throwaway
// device for the self-test and kernel micro tests, then autotuneCoop), the same weight loaders
// (ggufWeights / qwen35Weights, each matrix uploaded as it is converted so RAM stays flat) and the
// same engine options (engine/preset.js roomQwen35Options + applyRoomFlags), so a node's layers
// compute what a browser's would. Qwen3.8-Flash-Next (kind qwen4exp) loads with qwen4expWeights into its own engine
// (engine/qwen4exp.js, imported when needed: it reads the WebGPU globals as it loads) behind the room adapter
// (room/flash.js), and the device holding the n-gram layer reads that table's rows from the model file.
import fs from "node:fs";
import os from "node:os";
import { autotuneCoop, makeTokenizer, DenseEngine, gpuSelfTest, kernelMicroTests } from "../../engine/engine.js";
import { Qwen35Engine } from "../../engine/qwen35.js";
import { ggufWeights, ggufShardBytes, qwen35Weights, qwen35ShardBytes, tokenizerFromGGUF, gpuUploadEntry, GGML_EMBED, GGML_FINAL_NORM, GGML_OUTPUT,
  ggmlLayerNames, qwen35NamesFor, qwen4expConfig, qwen4expNamesFor, qwen4expWeights, qwen4expShardBytes, QWEN4EXP_HEAD } from "../../engine/gguf.js";
import { ngramConfig, NgramTable, fileReadAt, httpReadAt } from "../../engine/ngram.js";
import { FlashRoomEngine, flashBatchCols, flashBatchScratch, qwen4expBatchBytes } from "../../room/flash.js";
import { hostMemory, qsaAlways, flashBatchMode, flashRegionForced, flashLayered, flashSharedRegion } from "./env.js";
import { roomQwen35Options, applyRoomFlags } from "../../engine/preset.js";
import { maxSeqFor, kvModeFor, flashRoomMem, ckptSlotsAt, flashLoadOpts } from "../../room/models.js";
import { QSA_RAW_RING } from "../../engine/qwen4exp_mem.js";
import { convertPool } from "./convert.js";
import { ExpertStore } from "../../engine/expert_store.js";
import { OFFLOAD_GOOD_SLOTS, OFFLOAD_MIN_SLOTS } from "../../room/plan.js";
import { expertDiskReader } from "./expert_disk.js";
import { expertDisk } from "../../engine/expert_disk.js";
import { localFileReader } from "./source.js";
import { fastLoad, fastLoadOn } from "./fastload.js";

// The tensors a shard's loader reads (engine/gguf.js ggufWeights / qwen35Weights), in about its order,
// as { byteOffset, byteLength, shard? }: what source.js fetches ahead of it when it streams (shard: the file of a split GGUF)
export function shardTensors(G, kind, { lo, hi, hasEmbed, hasHead, mtp = false, qsa = false }) {
  const names = [];
  const strings = (o) => Object.values(o).filter((v) => typeof v === "string");
  // qwen4exp: the dense phase skips QSA's indexer (qwen4expWeights qsa: false; qsa: a QSA room's load reads it), and the
  // n-gram table is read by rows
  const layer = (i) => (kind === "qwen35" ? strings(qwen35NamesFor(G, i)) : kind === "qwen4exp" ? qwen4expTensorNames(G, i, qsa) : strings(ggmlLayerNames(i)));
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

// The tensors of one qwen4exp layer qwen4expWeights loads in the dense phase (no QSA indexer; qsa: with it), in its order
const QWEN4EXP_INDEXER = new Set(["idxQ", "idxK", "idxQNorm", "idxKNorm"]);
export function qwen4expTensorNames(G, i, qsa = false) {
  return Object.entries(qwen4expNamesFor(G, i)).filter(([k, v]) => typeof v === "string" && (qsa || !QWEN4EXP_INDEXER.has(k))).map(([, v]) => v);
}

// offload (expert offload, room/plan.js dealRoom): { lo, hi, vramBytes, ramBytes, ramCap }: the routed experts of
// layers [lo, hi) are parked in RAM (engine/expert_store.js ExpertStore) with a GPU cache of vramBytes (its slot pools +
// prefill region) instead of uploaded. ramBytes: what the deal says they park; ramCap: the RAM this device lends
// less the RAM the deal keeps free there (ramSpare; roomnode.js offloadOk): parking past it fails the load. A Qwen3.5 / 3.6 MoE shard on a room node only.
// With the disk tier (roadmap/32): diskLo, diskRamBytes: the experts of layers [diskLo, hi) stay in the model's files
// and are read when a token picks them (expert_disk.js's worker threads, from src.localPaths), through a RAM cache of
// diskRamBytes (engine/expert_t1.js). diskReader: makes the reader ({ paths } -> expertDiskReader; tests stub it).
// the layers an ai-load's offload covers within this shard's range, or null for none
export function offloadLayers(o, range) {
  const lo = Math.max(range[0], +o?.lo || 0), hi = Math.min(range[1], +o?.hi || 0);
  return hi > lo && o.vramBytes > 0 ? Array.from({ length: hi - lo }, (_, i) => lo + i) : null;
}
// of those, the ones whose experts stay on disk ([diskLo, hi)), or [] for none
export const diskLayers = (o, off) => (off && Number.isInteger(o?.diskLo) ? off.filter((l) => l >= o.diskLo) : []);

// The disk tier's reader: expert_disk.js's worker threads, and when they can't run at all (no worker left: a bundle
// without expert_disk_worker.js, threads that keep dying) the engine's own reader in this thread (engine/expert_disk.js
// over positional reads of the same files): slower, the same bytes. make: the worker reader (tests stub it).
export function expertReader({ paths, make = expertDiskReader, log = () => {} }) {
  const w = make({ paths });
  let fb = null, files = null;
  const inThread = () => {
    if (fb) return fb;
    log("expert disk reader: no worker threads, reading experts on the main thread (slower)");
    files = paths.map((p) => localFileReader(p));   // source.js: a handle per lane, as the model source reads
    return fb = expertDisk({ readAt: (shard, off, len) => files[shard].readAt(off, len) });
  };
  return {
    worker: w,
    async readPart(info, e) {
      if (fb) return fb.readPart(info, e);
      try { return await w.readPart(info, e); }
      catch (err) { if (!/no worker left/.test(String(err?.message))) throw err; return inThread().readPart(info, e); }
    },
    async close() {
      await w.close?.();
      for (const f of files || []) try { await f.close(); } catch {}
      files = null;
    },
  };
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
// flash: Qwen3.8-Flash-Next's format from the host's ai-load (room/models.js flashRoomFormat / flashLoadOpts): null for the
// dense engine at <= 2048 positions, { qsa, kv, ring } for a QSA one
// pledgeBytes: what this device lends the room (a Qwen3.8-Flash-Next shard with offloaded experts takes the batched
// passes' whole-layer region only when the pledge has room for it past the share; 0: unknown, no region)
export async function loadShard({ modelKey, range, hasEmbed, hasHead, ctx = maxSeqFor(modelKey), kv = kvModeFor(modelKey, null),
  src, flags = "", onProgress = () => {}, log = () => {}, selfTest = true, onGpuError = () => {}, offload = null, diskReader = expertDiskReader, flash = null,
  pledgeBytes = 0, batch = flashBatchMode() }) {
  checkRange(range);
  const M = src.M;
  // the host's Flash format, checked before the GPU is asked for or anything streams (a format this build can't load throws)
  const fq = M.kind === "qwen4exp" ? flashLoadOpts(flash) : null;
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
    const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead, ...(kind === "qwen35" ? { mtp: hasHead } : {}), ...(fq ? { qsa: true } : {}) };
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
      const r = await loadFlash({ device, G0, src, range, hasEmbed, hasHead, ctx, offload, onProgress, log, out, diskReader, pool, flash, pledgeBytes, batch });
      out.engine = r;
    } else if (M.kind === "qwen35") {
      const G = out.G = G0;
      out.cfg = { num_hidden_layers: G.meta["qwen35.block_count"] - (G.meta["qwen35.nextn_predict_layers"] || 0) };
      if (needTok) { out.tok = makeTokenizer(tokenizerFromGGUF(G.meta)); out.tok.chatTemplate = G.meta["tokenizer.chat_template"] || ""; }
      const off = offload && offloadLayers(offload, range), dl = diskLayers(offload, off);
      const total = qwen35ShardBytes(G, { ...opts, diskLayers: dl });
      if (off?.length) {
        let disk = null;
        if (dl.length) {
          if (!src.localPaths) throw new Error("the host asked this device to read experts from disk, but the model's files are not on this device");
          disk = { layers: dl, reader: expertReader({ paths: src.localPaths, make: diskReader, log }), ramBytes: +offload.diskRamBytes || 0 };
        }
        out.experts = new ExpertStore(device, { layers: off, vramBytes: offload.vramBytes, ramBytes: +offload.ramCap || Infinity, expectBytes: +offload.ramBytes || 0, disk });
        const parked = off.length - dl.length;
        log(`expert offload: ${parked ? `layers ${off[0]}-${off[parked - 1]} park their experts in RAM` : ""}${parked && dl.length ? "; " : ""}`
          + `${dl.length ? `layers ${dl[0]}-${dl[dl.length - 1]} read theirs from disk (${(disk.ramBytes / 2 ** 30).toFixed(1)} GB RAM cache)` : ""}, ${(offload.vramBytes / 2 ** 30).toFixed(2)} GB GPU cache`);
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
  } catch (e) { out.fastLoad?.close(); await out.experts?.close().catch(() => {}); try { await out.engine?.close?.(); } catch {} try { await out.ngram?.close?.(); } catch {} try { device.destroy(); } catch {} throw e; }
  finally { if (out.G) delete out.G.convert; await pool?.close(); }
}

// ---------------- Qwen3.8-Flash-Next (qwen4exp) ----------------
// What a qwen4exp shard needs in memory besides its GPU weights: its layers' state at ctx positions and the checkpoint
// reserve the deal set aside for them, from the room's one memory model (room/models.js flashRoomMem: K / V on its
// attention layers, the QSA indexer's keys when qsa, the DeltaNet state per recurrent layer, the PLE ring on the
// n-gram layer; ckptSlotsAt(ctx) copies of it), so the planner and this refusal agree; scratch; the embedding the
// engine keeps on the CPU (flashCpuBytes: RAM, which on unified memory is the same memory); and Dawn's upload
// staging: a writeBuffer's copy is kept until the queue moves on, and loadFlash flushes after each layer, so the
// biggest one part (a layer, or the head) is staged on top of what is already up. -> bytes
// offLayers (expert offload): layers whose routed experts are not on the GPU (parked or on disk): their bytes and their
// staging leave those out; the store's GPU cache is the caller's to add (offload.vramBytes)
// kv / qsa / ring: the load's engine options (the K / V format, the QSA indexer, its rawK ring rows); slots: the
// checkpoint slots reserved (ckptSlotsAt(ctx): none at the dense phase's 2048)
export const flashExpertBytes = (G, i) => ["gate", "up", "down"].reduce((a, p) => a + (G.tensors[`blk.${i}.ffn_${p}_exps.weight`]?.byteLength || 0), 0);
export function flashShardBytes(G, { lo, hi, hasEmbed = false, hasHead, ctx, offLayers = null, kv = "f32", qsa = false, ring = QSA_RAW_RING, slots = ckptSlotsAt(ctx) }) {
  const off = (i) => (offLayers?.includes?.(i) ? flashExpertBytes(G, i) : 0);
  let b = qwen4expShardBytes(G, { lo, hi, hasEmbed: false, hasHead, qsa });
  let staged = hasHead ? qwen4expShardBytes(G, { lo: hi, hi, hasEmbed: false, hasHead: true, qsa }) - (G.tensors[GGML_EMBED]?.byteLength || 0) : 0;
  for (let i = lo; i < hi; i++) {
    b -= off(i);
    staged = Math.max(staged, qwen4expShardBytes(G, { lo: i, hi: i + 1, hasEmbed: false, hasHead: false, qsa }) - off(i));
  }
  if (hi > lo) b += flashRoomMem(G, ctx, { kv, qsa, ring, range: [lo, hi], ckptSlots: slots }).need;
  return b + staged + flashCpuBytes(G, { hasEmbed, hasHead }) + 64 * 2 ** 20;   // + scratch, uniforms
}
// the part of flashShardBytes held on the CPU: the embedding (~0.36 GB), whose rows the engine reads there
export const flashCpuBytes = (G, { hasEmbed = false, hasHead = false } = {}) => (hasEmbed || hasHead ? G.tensors[GGML_EMBED]?.byteLength || 0 : 0);

// Before a qwen4exp load: does this device have the memory for its share? On unified memory (a GB10, Apple silicon)
// an allocation past what is free does not fail in WebGPU: the OS kills the process or swaps, so say it up front.
// bytes: everything the share takes (flashShardBytes, plus with expert offload the store's GPU cache and what it parks
// in RAM); cpuBytes: the part of it in RAM (flashCpuBytes, the parked experts, the disk tier's RAM cache), not counted
// against a discrete GPU but against the system's RAM there (sysFreeGB: os.freemem(), when known).
// mem: env.js hostMemory() (detectMemory). -> null when it fits (or nothing is known), else the reason
export function flashMemoryShort(bytes, mem, { slackGB = 2, cpuBytes = 0, sysFreeGB = null } = {}) {
  const gb = (n) => (n / 2 ** 30).toFixed(1);
  if (mem?.kind === "unified") {
    const free = mem.availGB ?? mem.freeGB;
    if (free > 0 && bytes / 2 ** 30 > free - slackGB) return `needs ${gb(bytes)} GB and this computer has ${free.toFixed(1)} GB of unified memory free (${slackGB} GB kept for the system)`;
  } else if (mem?.kind === "discrete") {
    if (mem.freeGB > 0 && (bytes - cpuBytes) / 2 ** 30 > mem.freeGB - 0.5)
      return `needs ${gb(bytes - cpuBytes)} GB of GPU memory and ${mem.name || "the GPU"} has ${mem.freeGB.toFixed(1)} GB free`;
    if (sysFreeGB > 0 && cpuBytes / 2 ** 30 > sysFreeGB - slackGB)
      return `needs ${gb(cpuBytes)} GB of system RAM (the embedding, the parked experts, the disk tier's cache) and this computer has ${sysFreeGB.toFixed(1)} GB free (${slackGB} GB kept for the system)`;
  }
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
  // not on this disk (a node reads the model from disk only when every one of its files is there): Hugging Face
  const urls = src.urls || [src.url];
  if (!urls[cfg.shard]) throw new Error("the n-gram table's file has no URL");
  const t = new NgramTable(cfg, { readAt: httpReadAt(urls), layout: "tensor", inflight: 16 });
  t.close = async () => { t.close = null; };
  t.where = urls[cfg.shard];
  log(`n-gram table: ${gb} GB, NOT on this disk (the model's files are not all here: pooled pull keeps a local copy); its rows are read by HTTP range from ${urls[cfg.shard]}, 16 round trips a token: slow`);
  return t;
}

// Qwen3.8-Flash-Next's context without QSA (a host that sends no ai-load.flash; room/model-loader.js clamps a tab's too)
const FLASH_MAX_CTX = 2048;

// One qwen4exp shard: the weights (each matrix uploaded as it is converted; the embedding stays on the CPU, where the
// engine reads its rows), the n-gram table when the range holds its layer, the engine, the room adapter.
// offload (expert offload, as a Qwen3.5 shard's: offloadLayers / diskLayers): the routed experts of [lo, hi) are parked
// in RAM, those of [diskLo, hi) read from the model's files on this disk (--disk), through the store's GPU slot pools;
// engine/qwen4exp.js cuts each such layer after its router.
// flash (ai-load.flash, room/models.js flashLoadOpts): with qsa, the indexer weights are loaded and the engine runs QSA
// on its attention layers (a range without one builds none: its signature says qsa false), with the host's K / V format
// and ring; without, the dense engine as before, its context capped at 2048. POOLED_QSA_ALWAYS=1 (env.js qsaAlways, a
// test switch): QSA's gathered path on every step, also below 2052 cells (the same bits as dense there).
// batch (env.js flashBatchMode: POOLED_FLASH_BATCH): "wide" / "exact" build the batched passes (engine/qwen4exp_batch.js)
// at load time, so prompt frames (and a solo room's prompt) run several columns per pass (room/flash.js); "off": one
// decode step per column, as before. An offloaded layer needs the store's whole-layer region for them (~1.4 GB of GPU;
// below: from pledgeBytes' slack past the share, or out of a cache that stays good, else this device runs its frames
// per token; or, with every offloaded layer in RAM, borrowed from the cache for each prompt frame: the store's shared
// region, engine/expert_store.js). Disk-tier layers never batch (Qwen4ExpBatch refuses them). The batch is built from the weights object
// (its unfused entries) but its ops bind GPU buffers only: nothing of the object stays alive past the load.
// -> FlashRoomEngine
async function loadFlash({ device, G0: G, src, range, hasEmbed, hasHead, ctx, offload, onProgress, log, out, mem = null, diskReader = expertDiskReader, pool = null, flash = null,
  pledgeBytes = 0, batch = "wide" }) {
  const c = qwen4expConfig(G.meta), q = flashLoadOpts(flash);
  if (!q) ctx = Math.min(ctx, FLASH_MAX_CTX);   // the dense phase's cap, before anything is sized or uploaded (as a tab's load)
  out.G = G;   // (loadShard's finally takes the conversion pool off it)
  out.cfg = { num_hidden_layers: c.L };
  if (range[1] > c.L) throw new Error(`bad layer range [${range[0]}, ${range[1]}]: the model has ${c.L} layers`);
  if (hasEmbed || hasHead) { out.tok = makeTokenizer(tokenizerFromGGUF(G.meta)); out.tok.chatTemplate = G.meta["tokenizer.chat_template"] || ""; }
  const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead, qsa: !!q };
  const off = offload && offloadLayers(offload, range), dl = diskLayers(offload, off);
  const m = mem || hostMemory();
  // the GPU holds the layers less their offloaded experts, plus the store's cache; the parked experts and the disk tier's
  // RAM cache are in RAM: on unified memory the same memory, on a discrete GPU the system's (checked against it too)
  const ramOff = off ? (+offload.ramBytes || 0) + (dl.length ? +offload.diskRamBytes || 0 : 0) : 0;
  // (the batched passes' scratch is not in need: it is decided once the weights are up, and a device without room for it
  // runs its prompt frames one column at a time instead of refusing the load)
  const solo = hasEmbed && hasHead && range[0] === 0 && range[1] === c.L;
  const need = flashShardBytes(G, { ...opts, ctx, offLayers: off || null, ring: q?.ring }) + (off ? +offload.vramBytes || 0 : 0) + ramOff;
  const cpu = flashCpuBytes(G, opts) + ramOff;
  const sysFreeGB = m?.kind === "discrete" ? (m.sysFreeGB ?? os.freemem() / 2 ** 30) : null;
  const short = flashMemoryShort(need, m, { cpuBytes: cpu, sysFreeGB });
  if (short) throw Object.assign(new Error(`can't hold layers ${range[0]}-${range[1] - 1} of Qwen3.8-Flash-Next here: it ${short}`), { oom: true, short: true });
  if (off?.length) {
    let disk = null;
    if (dl.length) {
      if (!src.localPaths) throw new Error("the host asked this device to read experts from disk, but the model's files are not on this device");
      disk = { layers: dl, reader: expertReader({ paths: src.localPaths, make: diskReader, log }), ramBytes: +offload.diskRamBytes || 0 };
    }
    out.experts = new ExpertStore(device, { layers: off, vramBytes: offload.vramBytes, ramBytes: +offload.ramCap || Infinity, expectBytes: +offload.ramBytes || 0, disk });
    const parked = off.length - dl.length;
    log(`expert offload: ${parked ? `layers ${off[0]}-${off[parked - 1]} park their experts in RAM` : ""}${parked && dl.length ? "; " : ""}`
      + `${dl.length ? `layers ${dl[0]}-${dl[dl.length - 1]} read theirs from disk (${(disk.ramBytes / 2 ** 30).toFixed(1)} GB RAM cache)` : ""}, ${(offload.vramBytes / 2 ** 30).toFixed(2)} GB GPU cache`);
  }
  const total = qwen4expShardBytes(G, opts) - dl.reduce((a, l) => a + flashExpertBytes(G, l), 0);
  // straight to the GPU, but not the embedding (Qwen4ExpEngine reads its rows on the CPU) nor f32 tensors (the engine
  // uploads those itself, some joined into one buffer). A layer at a time, the queue flushed after each: Dawn keeps
  // a writeBuffer's staging copy until the queue moves on, and 67 GB of uploads in one go ran the GB10 out of memory
  // (VK_ERROR_OUT_OF_DEVICE_MEMORY, the device lost) with the weights only half on the GPU
  // where the load's time goes (out.loadTimes, seconds; with fastload.js reads run ahead and repacks on threads, so
  // read is the loader's wait for its bytes and the stages no longer add up to the load)
  const T = out.loadTimes = { read: 0, convert: 0, park: 0, upload: 0, flush: 0 };
  const timed = (k, f) => async (...a) => { const t = performance.now(); try { return await f(...a); } finally { T[k] += (performance.now() - t) / 1000; } };
  // fastload.js (POOLED_FASTLOAD=0: off): the tensors this load reads, in the loader's order (the disk layers' experts stay in the file)
  const dlExp = new Set(dl.flatMap((l) => ["gate", "up", "down"].map((p) => `blk.${l}.ffn_${p}_exps.weight`)));
  const byKey = new Map(Object.values(G.tensors).map((t) => [`${t.shard || 0}:${t.byteOffset}`, t]));
  const plan = shardTensors(G, "qwen4exp", opts).map((t) => byKey.get(`${t.shard || 0}:${t.byteOffset}`)).filter((t) => t && !dlExp.has(t.name));
  const fl = fastLoadOn() ? fastLoad({ src, G, plan, pool, log, ...(+process.env.POOLED_FASTLOAD_AHEAD > 0 ? { aheadBytes: +process.env.POOLED_FASTLOAD_AHEAD * 2 ** 30 } : {}) }) : null;
  // the parked buffers up front, while the RAM is still free (ExpertStore.prealloc); the first reads run meanwhile
  if (fl && out.experts) { const t = performance.now(); out.experts.prealloc(); T.prealloc = (performance.now() - t) / 1000; }
  out.fastLoad = fl;
  const bytesOf = timed("read", fl ? fl.bytesOf : src.bytesOf);
  const conv0 = fl ? fl.convert : G.convert;
  if (conv0) G.convert = timed("convert", conv0);
  if (out.experts) {
    const p0 = out.experts.park.bind(out.experts);
    out.experts.park = (layer, part, e, nExp) => { const t = performance.now(); try { return p0(layer, part, e, nExp); } finally { fl?.release(e); T.park += (performance.now() - t) / 1000; } };
  }
  const up = (e, name) => { if (name !== GGML_EMBED) { const t = performance.now(); gpuUploadEntry(device, e, false); fl?.release(e); const d = performance.now() - t; T.upload += d / 1000; fl?.upTrace(G.tensors[name], d); } };
  const flush = timed("flush", async () => { device.queue.submit([device.createCommandEncoder().finish()]); await device.queue.onSubmittedWorkDone(); });
  let base = 0;
  const part = async (o) => {
    const w = await qwen4expWeights(G, bytesOf, { ...o, qsa: !!q, experts: out.experts || null }, (done) => onProgress(base + done, total), up);
    base += qwen4expShardBytes(G, { ...o, qsa: !!q }) - (o.hi > o.lo && dl.includes(o.lo) ? flashExpertBytes(G, o.lo) : 0);
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
  // (disk layers batch too: a pass reads each one's experts from its file into the region, engine/qwen4exp_batch.js)
  let noBatch = batch === "off" ? "POOLED_FLASH_BATCH=0" : null;
  let expertRegion = false, batchCols = flashBatchCols(solo, q?.ring), scratch = 0;
  const gb = (n) => (n / 2 ** 30).toFixed(2);
  // real memory, as the check above, with extra bytes on the GPU
  const fits = (extra) => !flashMemoryShort(need + extra, m, { cpuBytes: cpu, sysFreeGB });
  // the pledge's slack past this device's share: the pledge is GPU memory (on a discrete GPU the CPU embedding, the
  // parked experts and the disk tier's RAM cache are not in it: flashMemoryShort's discrete branch). A dealer keeps
  // room/models.js flashDealReserve back from every Flash pledge, so a device dealt all of it still has that much.
  // null: the pledge is unknown (real memory only; no region from it)
  const slack = pledgeBytes > 0 ? pledgeBytes - (m?.kind === "discrete" ? need - cpu : need) : null;
  // a solo device's widest passes only when its pledge (when known) and its memory have room for their scratch, else
  // the room's frame width
  if (!noBatch && solo && !((slack == null || slack >= flashBatchScratch(batchCols)) && fits(flashBatchScratch(batchCols)))) batchCols = flashBatchCols(false, q?.ring);
  if (!noBatch) scratch = flashBatchScratch(batchCols);
  // The deal gives the expert cache the rest of the pledge (room/plan.js offloadPlan), so the region (and the scratch)
  // come from the pledge's slack past the share when there is that much (the cache keeps every slot it was dealt),
  // else out of the cache (what the slack does not cover) when its pools keep OFFLOAD_GOOD_SLOTS experts per layer
  // after it (the deal's own bar for a good cache: decode misses a little more often, prompt frames run batched), else
  // not at all (prompt frames one column at a time)
  if (!noBatch && off?.length) {
    // (the scratch may also come from the dealer's reserve for it, offload.batchReserve, which the load's own count of
    // its share can hide: it counts upload staging the deal does not)
    const st = out.experts, rb = st.regionSize(), sl = Math.max(0, slack ?? 0);
    const fromSlack = Math.min(scratch, Math.max(sl, +offload.batchReserve || 0)), kept = st.poolSlots(rb + scratch - fromSlack);
    const enough = flashRegionForced() ? OFFLOAD_MIN_SLOTS : OFFLOAD_GOOD_SLOTS;
    if (sl >= rb + scratch && fits(rb + scratch)) {
      expertRegion = true; st.vramBytes += rb;
      log(`expert offload: a ${gb(rb)} GB whole-layer region for batched prompt passes, from the pledge's ${gb(sl)} GB past the share`);
    } else if (kept >= enough && fits(fromSlack)) {
      const was = st.poolSlots();
      expertRegion = true; st.vramBytes -= scratch - fromSlack;
      log(`expert offload: a ${gb(rb)} GB whole-layer region for batched prompt passes, out of the expert cache (${was} -> ${kept} experts per layer)`);
    } else {
      // A cache about a region's size (a node lending little GPU memory past its layers): the region is borrowed from it
      // for each prompt frame (ExpertStore build "shared": the frame overwrites the cache, which decode refills from the
      // parked experts in RAM, at copy speed; the frame's passes go layer by layer, so each layer is copied in once).
      // Not with disk layers, whose refills would read the disk again. The cache must hold a whole layer's experts in
      // all, so the passes narrow (their scratch shrinks: 169 MiB at 256 columns, 48 at 64) until it does, the
      // dealer's reserve past the scratch going to the cache.
      const vb = st.vramBytes, res = +offload.batchReserve || 0;
      for (const w of [batchCols, 128, 64].filter((x, i, a) => x <= batchCols && a.indexOf(x) === i)) {
        if (!flashSharedRegion() || st.diskSet.size) break;
        const sc = flashBatchScratch(w), fs = Math.min(sc, Math.max(sl, res)), spare = Math.max(0, res - sc);
        st.vramBytes = vb - (sc - fs) + spare;
        if (st.sharedFits() && st.poolSlots() >= OFFLOAD_MIN_SLOTS && fits(fs + spare)) {
          expertRegion = "shared"; batchCols = w; scratch = sc;
          log(`expert offload: batched prompt passes of ${w} columns borrow the whole-layer region from the expert cache (${gb(rb)} GB; ${st.poolSlots()} experts per layer, refilled from RAM after each prompt frame)`);
          break;
        }
      }
      if (!expertRegion) {
        st.vramBytes = vb;
        noBatch = `the offloaded layers need a ${gb(rb)} GB whole-layer region: the pledge has ${gb(sl)} GB past the share and the cache would keep ${Math.max(0, kept)} experts per layer (< ${enough})`;
      }
    }
  } else if (!noBatch && !fits(scratch)) noBatch = `no GPU memory left for the batched passes' ${gb(scratch)} GB of scratch`;
  if (noBatch) scratch = 0;
  // an offloader without batched passes gives the dealer's scratch reserve (offload.batchReserve: none from an older
  // host, which kept nothing back) to its cache, so its cache is what the deal would have made without the reserve
  if (noBatch && off?.length && +offload.batchReserve > 0) {
    const give = +offload.batchReserve;
    if (fits(give)) { out.experts.vramBytes += give; log(`expert offload: the ${gb(give)} GB kept for batched passes goes to the expert cache`); }
  }
  const { Qwen4ExpEngine } = await import("../../engine/qwen4exp.js");
  const eng = await Qwen4ExpEngine.create({ device, meta: G.meta, weights, layerRange: range, hasEmbed, hasHead, maxSeq: ctx, ngram: out.ngram || null, expertRegion,
    ...(q ? { qsa: null, kv: q.kv, qsaRing: q.ring } : { qsa: false }) });
  if (eng.qsa && qsaAlways()) { eng.qsaAlways = true; log("POOLED_QSA_ALWAYS=1: QSA's gathered attention on every step (a test switch)"); }
  if (fl && process.env.POOLED_FASTLOAD_TRACE) fs.writeFileSync(process.env.POOLED_FASTLOAD_TRACE, JSON.stringify(fl.trace()));
  fl?.close(); out.fastLoad = null;
  log(`load: ${Object.entries(T).map(([k, v]) => `${k} ${v.toFixed(1)} s`).join(", ")}${fl ? ` (read ahead, ${fl.stats.repacked} tensors repacked on ${pool.threads} threads, ${(fl.stats.newBytes / 2 ** 30).toFixed(1)} GiB of shared buffers)` : ""}`);
  let pb = null;
  if (!noBatch) {
    const { Qwen4ExpBatch } = await import("../../engine/qwen4exp_batch.js");
    // created now: its pipelines compile during the load, not on the first prompt
    pb = await Qwen4ExpBatch.create(eng, weights, { maxCols: batchCols, verifyCols: 1, exact: batch === "exact", ...(flashLayered() ? {} : { layeredCols: 0 }) });
    out.batchBytes = qwen4expBatchBytes(pb);
  }
  const fe = FlashRoomEngine.wrap(eng, { batch: pb });
  out.flash = eng;   // the engine under the adapter (tests and benches: forwardToken, debugToken)
  out.flashBatch = pb;
  log(`Qwen3.8-Flash-Next layers ${range[0]}-${range[1] - 1}${hasHead ? " + embedding/head" : ""}: ${((need + scratch) / 2 ** 30).toFixed(1)} GB (weights${off ? " less the offloaded experts, the expert cache" : ""}, K/V, the CPU embedding, upload staging${scratch ? ", batch scratch" : ""})${out.ngram ? ", the n-gram layer here" : ""}; ${ctx} positions, ${eng.qsa ? `QSA (${eng.kv} K/V, ring ${eng.qsaRing})` : q ? "QSA loaded, no attention layer here" : "dense"}${fe.ckpt ? "" : "; no checkpoints (its K/V can't be copied out)"}`);
  log(pb ? `batched prompt passes: up to ${pb.maxCols} columns a pass (${pb.exact ? "exact" : "wide"} kernels), ${(out.batchBytes / 2 ** 20).toFixed(0)} MiB of scratch`
    : `prompt frames one column at a time: ${noBatch}`);
  return fe;
}
