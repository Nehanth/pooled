// Browser model headers, weight loading and engine construction. The room owns GPU
// acquisition, progress/pacing, membership and recovery. Engine constructors arrive
// after room.js's lazy import; importing this module does not load the GPU engine.
// Main-model dense/safetensors configuration and tokenizer preparation remain in room.js.
import { parseGGUFHeader, ggufWeights, ggufShardBytes, GGML_EMBED, GGML_OUTPUT, GGML_FINAL_NORM,
  ggmlLayerNames, qwen35Weights, qwen35ShardBytes, qwen35NamesFor, tokenizerFromGGUF, gpuUploadEntry, streamEntryToGPU,
  qwen4expConfig, qwen4expNamesFor, qwen4expWeights, qwen4expShardBytes, QWEN4EXP_HEAD }
  from "../engine/gguf.js";
import { ngramConfig, NgramTable, httpReadAt } from "../engine/ngram.js";
import { NGRAM_OPFS_DIR, ngramDir, ngramContentSig, tabOffloadLayers, tabExpertTier, opfsResidentExperts, ingestNgram,
  tabDiskGate } from "./flashdisk.js";
import { FlashRoomEngine } from "./flash.js";
import { MODELS, MAX_SEQ, mergeSplitHeaders, roomBytes, ngramOf } from "./models.js";
import { cacheKey } from "./weightcache.js";
import { attachBrowserWeightCache } from "./convertedcache.js";

// Where a tab keeps Qwen3.8-Flash-Next's n-gram table split per head (engine/ngram.js ngramFileName), from the OPFS root
export { NGRAM_OPFS_DIR };
// one qwen4exp layer's tensors as qwen4expWeights loads them in the dense phase (no QSA indexer), for the prefetcher
const QWEN4EXP_INDEXER = new Set(["idxQ", "idxK", "idxQNorm", "idxKNorm"]);
// one layer's routed expert tensors in the file (what a disk layer leaves out of the download)
const flashExpertBytes = (G, i) => ["gate", "up", "down"].reduce((a, p) => a + (G.tensors[`blk.${i}.ffn_${p}_exps.weight`]?.byteLength || 0), 0);
// What a tab's question says it would store (before the model's header is read): the layers' experts as OPFS records
// (room/models.js's expert profile: one record per expert, 4 KiB aligned) and the n-gram table when the range holds
// its layer. -> { bytes, ngramBytes }
function tabDiskAskBytes(model, layers, range) {
  const ex = roomBytes(model, 2048)?.experts, ng = ngramOf(model);
  const rec = (l) => Math.ceil((ex?.slices?.[l] || []).reduce((a, b) => a + b, 0) / 4096) * 4096 * (ex?.nExp || 0);
  const ngramBytes = ng && ng.layer >= range[0] && ng.layer < range[1] ? ng.bytes : 0;
  return { bytes: layers.reduce((a, l) => a + rec(l), 0) + ngramBytes, ngramBytes };
}
const flashLayerNames = (G, i) => Object.entries(qwen4expNamesFor(G, i)).filter(([k, v]) => typeof v === "string" && !QWEN4EXP_INDEXER.has(k)).map(([, v]) => v);

export function createModelLoader({ state: ai, rangeFetch, getWeightCache, getMeta, getEnginePreset, options = {}, hooks = {} }) {
  if (typeof getEnginePreset !== "function") throw new TypeError("getEnginePreset must be a function");
  const { onStatus: aiStatus = () => {}, crumb = () => {}, onCacheHit = () => {}, onModelLoaded: apiModelLoaded = () => {} } = hooks;
  // ngramIngest: the device holding Flash's n-gram layer copies the table into OPFS when it is not there and this tab
  // may write to its disk (else it reads its rows by HTTP range, 16 round trips a token); expertWorkers / expertRing:
  // the OPFS disk tier's reader workers and staging ring slots (room/flashdisk.js)
  const { wcache: WCACHE = true, wcacheVerify: WCACHE_VERIFY = false, prefetch = null, ngramIngest = true, expertWorkers = 4, expertRing = 64 } = options;
  // Consent for this tab's disk (room/flashdisk.js): hooks.tabDisk() -> { tabDisk: what this tab advertises (meta.tabDisk),
  // choice: its user's answer ("allow" | "deny" | null) }; hooks.askDisk({ model, bytes, layers, ngramBytes }) ->
  // Promise<boolean>: ask the user (a worker whose deal needs its disk and who was not asked yet). Without them a tab
  // refuses every offload and never copies the n-gram table: nothing is written to a disk nobody agreed to.
  const tabDiskNow = () => { try { return hooks.tabDisk?.() || {}; } catch { return {}; } };
  const askDisk = hooks.askDisk || (async () => false);
  const diskRefused = (why) => Object.assign(new Error(`this device refused to store the model on its disk: ${why}`), { noDisk: true });

  // ---- converted weights on disk (room/convertedcache.js, OPFS): a second load of the same layers skips
  // the CPU conversion (K-quant -> Q8, BF16/Q5_0 -> f32, the embedding's repack). Keyed by model URL,
  // its pinned revision, the GGUF header and engine/gguf.js itself, so any of those changing starts
  // fresh. The caller can disable it or also verify each entry's payload hash.
  async function useConvertedCache(G, url) {
    if (!WCACHE) { G.entryCache = null; return null; }
    return attachBrowserWeightCache(G, url, { srcUrl: new URL("../engine/gguf.js", import.meta.url).href, verify: WCACHE_VERIFY });
  }
  async function convertedSummary(c, t0) {
    if (!c) return;
    await c.flush();   // background writes done before the engine is built
    if (!(c.stats.hit || c.stats.write || c.stats.full)) return;
    const msg = `${c.summary()}; this device's layers loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s`;
    console.info(msg);
    if (c.stats.hit) onCacheHit(msg);
  }

  // a model's index: a split GGUF (MODELS[key].shards) reads every file's and merges them (room/models.js
  // mergeSplitHeaders: each tensor of a later file carries that file's url, which rangeOf fetches it from)
  async function fetchModelHeader(M, needTokenizer = true) {
    if (!M.shards?.length) return fetchGGUFHeader(M.gguf, needTokenizer);
    const hs = await Promise.all(M.shards.map((u, i) => fetchGGUFHeader(u, needTokenizer && i === 0)));
    return mergeSplitHeaders(hs, M.shards);
  }
  async function fetchGGUFHeader(url, needTokenizer = true) {
    let size = 12 * 2 ** 20;
    for (;;) {
      const r = await rangeFetch(url, 0, size - 1);   // 206 from the network, 200 from the cache
      const buf = await r.arrayBuffer();
      try { return parseGGUFHeader(buf, { skipTokenizer: !needTokenizer }); }
      catch (e) { if (size > 256 * 2 ** 20) throw e; size *= 2; }
    }
  }
  const streamWithRetry = (url, streamOpts, pace) => async (info) => {
    try { return await streamEntryToGPU(ai.device, info, openRangeOf(url, pace), streamOpts); }
    catch (e) {
      if (!/short tensor/.test(String(e)) && !e?.retryNet) throw e;   // (retryNet: a room device stopped sending it)
      const c = await getWeightCache(), u = info.url || url;
      if (c) c.delete(cacheKey(u, info.byteOffset, info.byteOffset + info.byteLength - 1)).catch(() => {});
      return streamEntryToGPU(ai.device, info, (i) => rangeFetch(i.url || url, i.byteOffset, i.byteOffset + i.byteLength - 1, true), streamOpts);
    }
  };
  // Prefetch: a shard is hundreds of tensors (a 27B worker with 30 layers fetches ~450), and fetching
  // them one after another pays the model host's time-to-first-byte every time. When the loader
  // asks for a tensor, the next few tensors of the shard (file order) are requested too, so
  // several are in flight at once. Phones keep one: every buffered body is RAM they do not have.
  // An explicit prefetch count overrides the device policy (0 = off).
  // The loader asks for tensors in model order, not file order, so "the next one in the file" is often
  // one it already has: those are never fetched again (taken). Fetching them anyway was ~450 MB of
  // unread downloads per MoE layer, and on an iPhone they piled up in Safari's networking process
  // until iOS killed it and the page with it (#207, measured with memprobe.html ?pfdedupe).
  const prefetcher = { url: null, list: [], at: new Map(), pending: new Map(), taken: new Set() };
  // (a split GGUF's tensors carry their own file's url: the list goes file by file, and every key names the file:
  // "<url>@<offset>")
  function planPrefetch(url, infos) {
    clearPrefetch();
    prefetcher.url = url;
    prefetcher.list = infos.filter(Boolean).sort((a, b) => (a.shard || 0) - (b.shard || 0) || a.byteOffset - b.byteOffset);
    prefetcher.at = new Map(prefetcher.list.map((x, i) => [(x.url || url) + "@" + x.byteOffset, i]));
  }
  // drop what nobody will read: cancel the bodies so the browser lets go of them now
  function clearPrefetch() {
    for (const p of prefetcher.pending.values()) p.then((r) => r.body?.cancel?.()).catch(() => {});
    prefetcher.pending = new Map(); prefetcher.taken = new Set(); prefetcher.url = null;
  }
  function rangeOf(url, info) {
    const lo = info.byteOffset, hi = info.byteOffset + info.byteLength - 1, u = info.url || url;
    if (url !== prefetcher.url) return rangeFetch(u, lo, hi);
    const ahead = prefetch ?? (getMeta()?.phone ? 1 : 4);
    const key = u + "@" + lo, i = prefetcher.at.get(key);
    prefetcher.taken.add(key);
    if (i !== undefined) for (let k = i + 1; k <= i + ahead && k < prefetcher.list.length; k++) {
      const n = prefetcher.list[k], nk = (n.url || url) + "@" + n.byteOffset;
      if (!prefetcher.pending.has(nk) && !prefetcher.taken.has(nk)) {
        const p = rangeFetch(n.url || url, n.byteOffset, n.byteOffset + n.byteLength - 1);
        p.catch(() => {});
        prefetcher.pending.set(nk, p);
      }
    }
    const p = prefetcher.pending.get(key);
    if (p) { prefetcher.pending.delete(key); return p.catch(() => rangeFetch(u, lo, hi)); }   // a failed prefetch retries in line
    return rangeFetch(u, lo, hi);
  }
  // the tensors a shard loads, for the prefetcher (a superset is harmless: the list only orders fetches)
  function shardInfos(G, names) { return [...new Set(names)].map((n) => G.tensors[n]).filter(Boolean); }
  const openRangeOf = (url, pace) => async (info) => {
    if (pace) await pace();
    crumb("streaming " + info.name + " (" + (info.byteLength / 2 ** 20).toFixed(0) + " MB)");
    return rangeOf(url, info);
  };
  const rangeBytesOf = (url, pace) => async (info) => {
    if (pace) await pace();
    crumb("fetching " + info.name + " (" + (info.byteLength / 2 ** 20).toFixed(0) + " MB)");
    let r = await rangeOf(url, info);
    let bytes = await r.arrayBuffer().then((b) => new Uint8Array(b), (e) => { if (e?.retryNet) return new Uint8Array(0); throw e; });
    if (bytes.length !== info.byteLength) {
      r = await rangeFetch(info.url || url, info.byteOffset, info.byteOffset + info.byteLength - 1, true);
      bytes = new Uint8Array(await r.arrayBuffer());
      if (bytes.length !== info.byteLength) throw new Error(`short download for ${info.name}: ${bytes.length}/${info.byteLength} bytes`);
    }
    return bytes;
  };

  // offload (Qwen3.8-Flash-Next only, a tab with the OPFS disk tier): the deal's { lo, hi, diskLo, vramBytes,
  // diskRamBytes } for this device (room/plan.js dealOffload, diskOnly): those layers' experts are read from OPFS
  async function loadShard({ modelKey, range, hasEmbed, hasHead, ctx, kv, streamOpts, onProgress: onProg, pace, offload = null }, engines) {
    const M = MODELS[modelKey];
    if (offload && M.kind !== "qwen4exp") throw new Error(`the host asked this tab to offload experts of ${M.label || modelKey}; a tab reads offloaded experts from its disk for Qwen3.8-Flash-Next only`);
    // the receiving side: an offload this tab did not offer, or its user turned down, is refused before anything is
    // read or written (the host deals again without this device's disk: room.js ai-error noDisk)
    const offL = offload ? tabOffloadLayers(offload, range) : null;
    const gate = offL ? tabDiskGate(offload, { model: modelKey, ...tabDiskNow() }) : null;
    if (gate && gate !== "ask") throw diskRefused(gate);
    if (gate === "ask") {
      // a worker whose user was not asked yet: ask now, before anything is read, with what it would store (its layers'
      // experts as OPFS records, from the model's profile, and the n-gram table when it holds that layer)
      const yes = await askDisk({ model: modelKey, layers: offL.disk, ...tabDiskAskBytes(modelKey, offL.disk, range) });
      if (!yes) throw diskRefused("its user said no");
      const g2 = tabDiskGate(offload, { model: modelKey, ...tabDiskNow() });
      if (g2) throw diskRefused(g2 === "ask" ? "its user did not answer" : g2);
    }
    const { makeTokenizer, DenseEngine, Qwen35Engine, fetchModelShard, shardTensorNames } = engines;
    if (M.kind === "qwen35") {
      aiStatus("reading model index\u2026");
      const needTok = hasEmbed || hasHead;
      const cachedOk = ai.G && ai.GModel === modelKey && (!needTok || ai.G.meta["tokenizer.ggml.tokens"]);
      const G = cachedOk ? ai.G : await fetchModelHeader(M, needTok);
      ai.G = G; ai.GModel = modelKey;
      ai.cfg = { num_hidden_layers: G.meta["qwen35.block_count"] - (G.meta["qwen35.nextn_predict_layers"] || 0) };
      if (hasEmbed || hasHead) {
        ai.tok = makeTokenizer(tokenizerFromGGUF(G.meta));
        // the model's own chat template: Code mode picks the tool-call format from it (Qwen 3.5+ use
        // XML <function=...> calls, with the full call grammar); without it every model got JSON
        ai.tok.chatTemplate = G.meta["tokenizer.chat_template"] || "";
        apiModelLoaded();
      }
      // the host also loads the model's multi-token-prediction block: it drafts
      // tokens that the trunk then verifies in one batched pass (same output, faster)
      const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead, mtp: hasHead };
      const total = qwen35ShardBytes(G, opts);
      const names = [];
      for (let l = range[0]; l < range[1]; l++) names.push(...Object.values(qwen35NamesFor(G, l)).filter((v) => typeof v === "string"));
      if (hasEmbed || hasHead) names.push(GGML_EMBED);
      if (hasHead) {
        names.push(GGML_FINAL_NORM, GGML_OUTPUT);
        const N = G.meta["qwen35.block_count"] - 1;
        names.push(...Object.values(qwen35NamesFor(G, N, true)).filter((v) => typeof v === "string"), ...["eh_proj", "enorm", "hnorm", "shared_head_norm"].map((x) => `blk.${N}.nextn.${x}.weight`));
      }
      planPrefetch(M.gguf, shardInfos(G, names));
      G.streamEntry = streamWithRetry(M.gguf, streamOpts, pace);
      const wc = await useConvertedCache(G, M.gguf), tw = performance.now();
      const weights = await qwen35Weights(G, rangeBytesOf(M.gguf, pace), opts, (done) => onProg(done, total),
        (e, name) => gpuUploadEntry(ai.device, e, name === GGML_EMBED));   // straight to the GPU, RAM stays flat
      await convertedSummary(wc, tw);
      aiStatus("building GPU pipelines (compiling shaders)\u2026");
      ai.engine = await Qwen35Engine.create({
        device: ai.device, meta: G.meta, weights, vocab: G.tensors[GGML_EMBED]?.shape?.[0],
        layerRange: range, hasEmbed, hasHead, maxSeq: ctx,
        coopWG: ai.tune?.wg, coopRows: ai.tune?.rows,
        // Read the caller's preset at construction time; the host's KV choice wins for every device.
        ...getEnginePreset(),
        kvQ8: kv === "q8",
      });
    } else if (M.kind === "qwen4exp") {
      // Qwen3.8-Flash-Next: its own engine (engine/qwen4exp.js) behind the room adapter (room/flash.js); the dense
      // phase (no QSA indexer, context <= 2048). The embedding stays in JS memory (the engine reads its rows), and the
      // device holding the n-gram layer reads that table's rows from OPFS (split per head) or by HTTP range
      aiStatus("reading model index\u2026");
      const needTok = hasEmbed || hasHead;
      const cachedOk = ai.G && ai.GModel === modelKey && (!needTok || ai.G.meta["tokenizer.ggml.tokens"]);
      const G = cachedOk ? ai.G : await fetchModelHeader(M, needTok);
      ai.G = G; ai.GModel = modelKey;
      const c = qwen4expConfig(G.meta);
      if (range[1] > c.L) throw new Error(`bad layer range [${range[0]}, ${range[1]}]: the model has ${c.L} layers`);
      ai.cfg = { num_hidden_layers: c.L };
      if (needTok) {
        ai.tok = makeTokenizer(tokenizerFromGGUF(G.meta));
        ai.tok.chatTemplate = G.meta["tokenizer.chat_template"] || "";
        apiModelLoaded();
      }
      const urls = M.shards?.length ? M.shards : [M.gguf];
      // expert offload's disk tier (room/flashdisk.js): the offloaded layers' experts into OPFS (once), read from there
      const off = tabOffloadLayers(offload, range), dev = ai.device;
      // straight from the network, never through the weight cache (rangeFetch stores what it fetches: 65 GB of experts
      // there would be a second copy on this disk) nor the HTTP cache
      const readAt = httpReadAt(urls, { fetch: (u, init) => fetch(u, { ...init, cache: "no-store" }) });
      let experts = null, resident = null;
      const streamPlain = streamWithRetry(M.gguf, streamOpts, pace);
      try {
        if (off) {
          if (!globalThis.navigator?.storage?.getDirectory) throw new Error("the host asked this tab to read experts from disk, but this browser has no OPFS");
          let line = null;
          const t = await tabExpertTier(dev, G, off.disk, { readAt, offload, workers: expertWorkers, ring: expertRing,
            onStatus: (s) => { line = s; aiStatus(s); }, onProgress: (d, tot, s) => onProg(d, tot, s || line), log: (s) => { console.info(s); crumb(s); } });
          experts = t.store;
          console.info(`expert offload: layers ${off.disk[0]}-${off.disk[off.disk.length - 1]} read their experts from OPFS, ${(offload.vramBytes / 2 ** 30).toFixed(2)} GiB GPU cache, ${((+offload.diskRamBytes || 0) / 2 ** 30).toFixed(2)} GiB RAM cache`);
        }
        // resident layers whose experts this tab already copied to OPFS (a re-deal that moved them off disk): read from
        // there, not downloaded again into the weight cache (room/flashdisk.js opfsResidentExperts; reading only)
        const resLayers = [];
        for (let l = range[0]; l < range[1]; l++) if (!off?.disk.includes(l)) resLayers.push(l);
        resident = await opfsResidentExperts(dev, G, resLayers, { readAt, workers: expertWorkers }).catch((e) => {
          console.warn(`expert OPFS copy: could not read resident layers from it (${e?.message || e}); downloading them`);
          return null;
        });
        if (resident) {
          const s = `layers ${resident.layers[0]}-${resident.layers[resident.layers.length - 1]}: experts from this device's disk copy, not the network`;
          console.info(s); crumb(s);
        }
        const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead, qsa: false };
        const skip = new Set(off?.disk || []);
        const total = qwen4expShardBytes(G, opts) - [...skip].reduce((a, l) => a + flashExpertBytes(G, l), 0);
        const names = [];
        for (let l = range[0]; l < range[1]; l++) names.push(...flashLayerNames(G, l).filter((n) => !skip.has(l) || !/_exps\.weight$/.test(n)));
        if (hasEmbed || hasHead) names.push(GGML_EMBED);
        if (hasHead) names.push(...Object.values(QWEN4EXP_HEAD), GGML_OUTPUT);
        const fromDisk = new Set(resident?.layers || []);
        planPrefetch(M.gguf, shardInfos(G, names.filter((n) => { const m = /^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$/.exec(n); return !m || !fromDisk.has(+m[1]); })));
        // (a read of the copy that fails, a short read or a bad record, is said and that tensor comes from the network)
        const fromCopy = async (info) => {
          try { return await resident.entry(info); } catch (e) {
            const s = `expert OPFS copy: ${info.name} could not be read from it (${e?.message || e}); downloading it`;
            console.warn(s); crumb(s);
            return null;
          }
        };
        G.streamEntry = resident ? async (info) => (await fromCopy(info)) ?? streamPlain(info) : streamPlain;
        const wc = await useConvertedCache(G, M.gguf), tw = performance.now();
        // straight to the GPU, but not the embedding (the engine reads its rows on the CPU). A layer at a time, the
        // queue flushed after each, as the room node loads it (packages/room-node/shard.js loadFlash): Dawn keeps a
        // writeBuffer's staging copy until the queue moves on, and one batch of a big share's uploads ran the GB10 out
        // of memory with the weights only half on the GPU
        const bytesOf = rangeBytesOf(M.gguf, pace);
        const up = (e, name) => { if (name !== GGML_EMBED) gpuUploadEntry(dev, e, false); };
        let base = 0;
        const part = async (o) => {
          const w = await qwen4expWeights(G, bytesOf, { ...o, qsa: false, experts }, (done) => onProg(base + done, total), up);
          base += qwen4expShardBytes(G, { ...o, qsa: false }) - (o.hi > o.lo && skip.has(o.lo) ? flashExpertBytes(G, o.lo) : 0);
          dev.queue.submit([dev.createCommandEncoder().finish()]);
          await dev.queue.onSubmittedWorkDone();
          return w;
        };
        const layers = [];
        let pleTable = null;
        for (let l = range[0]; l < range[1]; l++) {
          const w = await part({ lo: l, hi: l + 1, hasEmbed: false, hasHead: false });
          layers.push(w.layers[0]);
          pleTable ||= w.pleTable || null;
        }
        const weights = { ...(await part({ lo: range[1], hi: range[1], hasEmbed, hasHead })), layers, ...(pleTable ? { pleTable } : {}) };
        await convertedSummary(wc, tw);
        if (resident) {
          const s = `${(resident.bytes / 2 ** 30).toFixed(1)} GB of resident experts read from this device's disk copy`;
          console.info(s); crumb(s);
        }
        let ngram = null;
        try {
          ngram = weights.pleTable ? await openNgramTable(G, M) : null;
          aiStatus("building GPU pipelines (compiling shaders)\u2026");
          const { Qwen4ExpEngine } = await import("../engine/qwen4exp.js");
          ai.engine = FlashRoomEngine.wrap(await Qwen4ExpEngine.create({ device: ai.device, meta: G.meta, weights, layerRange: range,
            hasEmbed, hasHead, maxSeq: Math.min(ctx, 2048), ngram }));
          if (experts) console.info(experts.summary().replace(/; \d+ cuts.*/, ""));
        } catch (e) { await ngram?.close?.(); throw e; }
      } catch (e) {
        // a failed load (a weight download, the engine) frees what the disk tier holds: its reader's workers and file
        // handles, the RAM cache, the staging ring
        await experts?.close?.().catch(() => {});
        throw e;
      } finally {
        await resident?.close().catch(() => {});
        G.streamEntry = streamPlain;   // (ai.G outlives this load: no reader left behind on it)
      }
    } else if (M.kind === "gguf") {
      aiStatus("reading model index\u2026");
      const G = ai.G && ai.GModel === modelKey ? ai.G : await fetchGGUFHeader(M.gguf, false);   // vocab comes from tokenizer.json
      ai.G = G; ai.GModel = modelKey;
      const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead };
      const total = ggufShardBytes(G, opts);
      const names = [];
      for (let l = range[0]; l < range[1]; l++) names.push(...Object.values(ggmlLayerNames(l)));
      if (hasEmbed || hasHead) names.push(GGML_EMBED);
      if (hasHead) names.push(GGML_FINAL_NORM, GGML_OUTPUT);
      planPrefetch(M.gguf, shardInfos(G, names));
      G.streamEntry = streamWithRetry(M.gguf, streamOpts, pace);
      const wc = await useConvertedCache(G, M.gguf), tw = performance.now();
      const weights = await ggufWeights(G, rangeBytesOf(M.gguf, pace), opts, (done) => onProg(done, total),
        (e, name) => gpuUploadEntry(ai.device, e, name === GGML_EMBED));
      await convertedSummary(wc, tw);
      aiStatus("building GPU pipelines\u2026");
      ai.engine = await DenseEngine.create({
        coopWG: ai.tune?.wg, coopRows: ai.tune?.rows,
        device: ai.device, cfg: ai.cfg, weights,
        layerRange: range, hasEmbed, hasHead, maxSeq: ctx,
      });
    } else {
      const names = shardTensorNames(ai.cfg, range, hasEmbed, hasHead);
      const tensors = await fetchModelShard(M.st, names, (_part, done, total) => onProg(done, total));
      aiStatus("building GPU pipelines\u2026");
      ai.engine = await DenseEngine.create({
        coopWG: ai.tune?.wg, coopRows: ai.tune?.rows,
        device: ai.device, cfg: ai.cfg, tensors,
        layerRange: range, hasEmbed, hasHead, maxSeq: MAX_SEQ,
      });
    }
    clearPrefetch();
  }

  // The n-gram table for the device holding Qwen3.8-Flash-Next's n-gram layer: its 16 per-head files in this origin's
  // OPFS (NGRAM_OPFS_DIR, engine/ngram.js splitHeads; read by NgramWorkers, several heads at once) when they are
  // there, else HTTP range requests into the GGUF shard that holds it (16 round trips a token: a cold start).
  // close() releases the workers and their file locks (room/flash.js close, when the engine is dropped).
  // The OPFS copy is keyed by the table's content (room/flashdisk.js ngramDir / ngramContentSig), and it is copied only
  // when this tab may write to its disk (its user said yes: hooks.tabDisk); a copy already there is read either way.
  async function openNgramTable(G, M) {
    const cfg = ngramConfig(G), urls = M.shards?.length ? M.shards : [M.gguf];
    const readAt = httpReadAt(urls, { fetch: (u, init) => fetch(u, { ...init, cache: "no-store" }) });
    let dir = null;
    if (globalThis.navigator?.storage?.getDirectory) {
      try { dir = ngramDir(await ngramContentSig(cfg, readAt, { url: urls[cfg.shard] })); }
      catch (e) { console.warn(`n-gram table: could not read its signature (${e?.message || e}); reading its rows by HTTP range`); }
    }
    const copy = await copyNgram(cfg, readAt, dir);
    if (copy.skipped) { console.info(`n-gram table: not copied to this device's disk (${copy.skipped})`); crumb(`n-gram table: no OPFS copy (${copy.skipped})`); }
    if (dir) try {
      const { NgramWorkers } = await import("../engine/ngram_worker.js");
      const t = await NgramWorkers.open(cfg, { dir, workers: 4 });
      t.where = "opfs";
      crumb("n-gram table: OPFS");
      return t;
    } catch (e) {
      // NotFoundError: no split copy here (the usual case until a tab ingests one). Anything else (a locked file,
      // no OPFS or module workers in this browser, a short file) is said plainly before the HTTP fallback
      if (e?.name !== "NotFoundError") {
        const why = `${e?.name || "Error"}: ${String(e?.message || e).split("\n")[0].slice(0, 200)}`;
        console.warn(`n-gram table: the OPFS copy (${dir.join("/")}) could not be opened (${why}); reading its rows by HTTP range instead (16 round trips a token)`);
        crumb(`n-gram table: OPFS open failed (${why})`);
      }
    }
    const t = new NgramTable(cfg, { readAt: httpReadAt(urls), layout: "tensor", inflight: 16 });
    t.where = urls[cfg.shard];
    crumb(`n-gram table: HTTP range from ${t.where}`);
    return t;
  }
  // Copy the n-gram table into dir (room/flashdisk.js ingestNgram) when this tab's user allowed its disk; a failure falls
  // back to HTTP range reads. -> { copied: bytes } | { skipped: why } (nothing touched on disk when skipped)
  async function copyNgram(cfg, readAt, dir, { storage = globalThis.navigator?.storage } = {}) {
    if (!ngramIngest) return { skipped: "?ngramcopy=0" };
    if (!dir || !storage?.getDirectory) return { skipped: "no OPFS here" };
    const { tabDisk, choice } = tabDiskNow();
    if (choice !== "allow" || !tabDisk?.models?.length) return { skipped: choice === "deny" ? "its user did not allow models on its disk" : "its user was not asked to store models on its disk" };
    try {
      return { copied: await ingestNgram(cfg, readAt, { dir, storage, onStatus: aiStatus, log: (s) => { console.info(s); crumb(s); } }) };
    } catch (e) {
      const why = `${e?.name || "Error"}: ${String(e?.message || e).split("\n")[0].slice(0, 200)}`;
      console.warn(`n-gram table: copying it into OPFS failed (${why}); reading its rows by HTTP range instead`);
      crumb(`n-gram table: OPFS copy failed (${why})`);
      return { skipped: "the copy failed" };
    }
  }

  async function loadDraft({ modelKey, pace }, { DenseEngine }) {
    const M = MODELS[modelKey];
    const cfg = await (await fetch(M.cfg)).json();
    const G = await fetchGGUFHeader(M.gguf, false);
    const L = cfg.num_hidden_layers;
    const weights = await ggufWeights(G, rangeBytesOf(M.gguf, pace), { lo: 0, hi: L, hasEmbed: true, hasHead: true });
    return DenseEngine.create({ device: ai.device, cfg, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: ai.engine.maxSeq,
      coopWG: ai.tune?.wg, coopRows: ai.tune?.rows });
  }

  // (copyNgram: for tests of the consent gate)
  return { fetchModelHeader, fetchGGUFHeader, loadShard, loadDraft, copyNgram };
}
