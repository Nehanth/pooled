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
  tabDiskGate, tabExpertBytes, expertTierCheck, expertTierCopy, expertTierOpen, expertStoreDeferred, ngramCheck, savePoolFor, protocolOf, onHttp1 } from "./flashdisk.js";
import { partsTracker, runFlashParts } from "./flashload.js";
import { FlashRoomEngine, flashBatchCols } from "./flash.js";
import { planSteps, loadTracker } from "./loadsteps.js";
import { MODELS, MAX_SEQ, mergeSplitHeaders, roomBytes, ngramOf, flashLoadOpts } from "./models.js";
import { cacheKey } from "./weightcache.js";
import { OFFLOAD_GOOD_SLOTS } from "./plan.js";
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
// Whether layers [range[0], range[1]) of a qwen4exp model (c: qwen4expConfig) hold its n-gram (PLE) layer: the test
// engine/gguf.js qwen4expWeights uses to return the table's reader (weights.pleTable), asked before the load
// tensors a Flash tab's GPU step keeps in flight (room/model-loader.js planPrefetch; ?prefetch=N overrides)
const FLASH_PREFETCH = 12;
export const ngramLayerIn = (c, range) => !!c?.ple?.layers?.some((l) => l >= range[0] && l < range[1]);
// (qsa: with QSA's indexer tensors, when the host's ai-load.flash asks for a QSA engine)
const flashLayerNames = (G, i, qsa = false) => Object.entries(qwen4expNamesFor(G, i)).filter(([k, v]) => typeof v === "string" && (qsa || !QWEN4EXP_INDEXER.has(k))).map(([, v]) => v);

export function createModelLoader({ state: ai, rangeFetch, getWeightCache, getMeta, getEnginePreset, options = {}, hooks = {} }) {
  if (typeof getEnginePreset !== "function") throw new TypeError("getEnginePreset must be a function");
  const { onStatus: aiStatus = () => {}, crumb = () => {}, onCacheHit = () => {}, onModelLoaded: apiModelLoaded = () => {} } = hooks;
  // ngramIngest: the device holding Flash's n-gram layer copies the table into OPFS when it is not there and this tab
  // may write to its disk (else it reads its rows by HTTP range, 16 round trips a token); expertWorkers / expertRing:
  // the OPFS disk tier's reader workers and staging ring slots (room/flashdisk.js)
  // parload: a Flash tab runs its three parts at the same time (room/flashload.js; false: one after the other, as
  // before); ngramWorker: the n-gram table's copy runs in a worker (false: on the main thread, after the experts');
  // save: { workers, chunk, ahead, early } the experts' first save in workers (room/models.js modelLoadOptions;
  // workers 0: on the main thread); cdnResolve: range reads follow the model host's redirect once (engine/ngram.js);
  // diskBatch: a tab with disk-tier layers runs batched prompt passes over them (a whole-layer region out of its cache)
  const { wcache: WCACHE = true, wcacheVerify: WCACHE_VERIFY = false, prefetch = null, ngramIngest = true, expertWorkers = 4, expertRing = 64,
    parload = true, ngramWorker = true, save: saveOpts = { workers: 0, chunk: 8, ahead: 3, early: true }, cdnResolve = true, diskBatch = true } = options;
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
  // ahead: tensors in flight for this load (the explicit prefetch flag and phones still win)
  function planPrefetch(url, infos, ahead = 4) {
    clearPrefetch();
    prefetcher.url = url; prefetcher.ahead = ahead;
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
    const ahead = prefetch ?? (getMeta()?.phone ? 1 : prefetcher.ahead || 4);
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
  // onParts(snapshot): the load's parts (room/flashload.js partsTracker: { parts: [{ id, done, total, state }], note? }),
  // at most ~4 a second and on every state change; every model sends it (one "gpu" part but a Flash tab's)
  // flash (Qwen3.8-Flash-Next): the host's ai-load.flash (room/models.js flashLoadOpts): QSA's indexer weights loaded and
  // the engine built with QSA, the host's K / V format and ring, the context as dealt; null: the dense engine at <= 2048
  async function loadShard(o, engines) {
    if (!o.onParts) return loadShardIn(o, engines);
    // the other models' one part, from their onProgress (a Flash tab's parallel load makes its own tracker; its
    // sequential load, ?parload=0, sends only onProgress, as before)
    let gpuParts = null, sentParts = false;
    const onParts = (snap) => { sentParts = true; o.onParts(snap); };
    // (the parts first, then the old callback: a page that reads the last snapshot in onProgress sees this one)
    const onProgress = (done, total, info) => {
      if (!(info?.parts || info?.step)) {
        gpuParts ||= partsTracker({ sizes: { gpu: Math.max(1, total) }, onParts });
        gpuParts.at("gpu", done);
      }
      o.onProgress(done, total, info);
    };
    try {
      const r = await loadShardIn({ ...o, onProgress, onParts }, engines);
      gpuParts?.state("gpu", "done");
      // a load that said no progress at all (all from the cache): one last snapshot, so the page still gets one
      if (!sentParts) onParts({ parts: [{ id: "gpu", done: 1, total: 1, state: "done" }, { id: "save", done: 0, total: 0, state: "skip" }, { id: "words", done: 0, total: 0, state: "skip" }] });
      return r;
    } catch (e) { gpuParts?.failRunning(); throw e; }
    finally { gpuParts?.flush(); }
  }
  async function loadShardIn({ modelKey, range, hasEmbed, hasHead, ctx, kv, streamOpts, onProgress: onProg, onParts = null, pace, offload = null, flash = null }, engines) {
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
      aiStatus("Looking up the model\u2026");
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
      aiStatus("Almost ready\u2026");
      ai.engine = await Qwen35Engine.create({
        device: ai.device, meta: G.meta, weights, vocab: G.tensors[GGML_EMBED]?.shape?.[0],
        layerRange: range, hasEmbed, hasHead, maxSeq: ctx,
        coopWG: ai.tune?.wg, coopRows: ai.tune?.rows,
        // Read the caller's preset at construction time; the host's KV choice wins for every device.
        ...getEnginePreset(),
        kvQ8: kv === "q8",
      });
    } else if (M.kind === "qwen4exp") {
      // Qwen3.8-Flash-Next: its own engine (engine/qwen4exp.js) behind the room adapter (room/flash.js); with QSA when
      // the host asks (flash), else the dense phase (no QSA indexer, context <= 2048). The embedding stays in JS memory (the engine reads its rows), and the
      // device holding the n-gram layer reads that table's rows from OPFS (split per head) or by HTTP range
      aiStatus("Looking up the model\u2026");
      const needTok = hasEmbed || hasHead;
      const cachedOk = ai.G && ai.GModel === modelKey && (!needTok || ai.G.meta["tokenizer.ggml.tokens"]);
      const G = cachedOk ? ai.G : await fetchModelHeader(M, needTok);
      ai.G = G; ai.GModel = modelKey;
      const c = qwen4expConfig(G.meta), q = flashLoadOpts(flash);
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
      const readAt = httpReadAt(urls, { fetch: (u, init) => fetch(u, { ...init, cache: "no-store" }), resolve: cdnResolve });
      let experts = null, resident = null;
      // The GPU step's pacing (room.js aiLoadPace: hold it to the room's slowest GPU step), except while the parallel
      // load says not to (room/flashload.js runFlashParts paced(): this tab's disk copy still runs after its GPU step,
      // and held to a slow device the whole copy would start after that device is done). Unpaced, a Stop still ends
      // it: the load's abort (gpuHold.signal) is checked instead
      let gpuHold = null;   // { paced(), signal } during the parallel load's GPU step
      const gpuPace = pace && (async () => {
        if (gpuHold && !gpuHold.paced()) { if (gpuHold.signal.aborted) throw gpuHold.signal.reason; return; }
        return pace();
      });
      const streamPlain = streamWithRetry(M.gguf, streamOpts, gpuPace);
      // what goes onto the GPU: the range's weights but the disk layers' experts
      const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead, qsa: !!q };
      const skip = new Set(off?.disk || []);
      const total = qwen4expShardBytes(G, opts) - [...skip].reduce((a, l) => a + flashExpertBytes(G, l), 0);
      // the same test qwen4expWeights uses to return the table's reader (engine/gguf.js: pleTable with the n-gram layer)
      const ngCfg = ngramLayerIn(c, range) ? (() => { try { return ngramConfig(G); } catch { return null; } })() : null;
      const wordsBytes = ngCfg ? ngCfg.vocab.reduce((a, v) => a + v * ngCfg.rowBytes, 0) : 0;
      // The GPU step, either order's: resident layers whose experts this tab already copied to OPFS (a re-deal that
      // moved them off disk) are read from there, not downloaded again into the weight cache (room/flashdisk.js
      // opfsResidentExperts; reading only), then a layer at a time onto the GPU. at(bytes): its progress -> weights
      const gpuStep = async (at) => {
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
        const names = [];
        for (let l = range[0]; l < range[1]; l++) names.push(...flashLayerNames(G, l, !!q).filter((n) => !skip.has(l) || !/_exps\.weight$/.test(n)));
        if (hasEmbed || hasHead) names.push(GGML_EMBED);
        if (hasHead) names.push(...Object.values(QWEN4EXP_HEAD), GGML_OUTPUT);
        const fromDisk = new Set(resident?.layers || []);
        // 12 tensors in flight, not 4: a Flash layer is ~40 small tensors, and over Hugging Face each one pays a
        // redirect and the CDN's first byte (HF-like latency, 40 MB/s a request: 14 -> 27 MB/s with 16)
        planPrefetch(M.gguf, shardInfos(G, names.filter((n) => { const m = /^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$/.exec(n); return !m || !fromDisk.has(+m[1]); })), FLASH_PREFETCH);
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
        const bytesOf = rangeBytesOf(M.gguf, gpuPace);
        const up = (e, name) => { if (name !== GGML_EMBED) gpuUploadEntry(dev, e, false); };
        let base = 0;
        const part = async (o) => {
          const w = await qwen4expWeights(G, bytesOf, { ...o, qsa: !!q, experts }, (done) => at(base + done), up);
          base += qwen4expShardBytes(G, { ...o, qsa: !!q }) - (o.hi > o.lo && skip.has(o.lo) ? flashExpertBytes(G, o.lo) : 0);
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
        return weights;
      };
      const makeEngine = async (weights, ngram) => {
        const { Qwen4ExpEngine } = await import("../engine/qwen4exp.js");
        // the disk tier's layers batch through the expert store's whole-layer region (1.4-1.8 GB, out of the deal's GPU
        // cache), read from OPFS once a pass instead of once a column (engine/qwen4exp_batch.js), when the cache keeps
        // OFFLOAD_GOOD_SLOTS experts per layer after it (room-node shard.js's bar); else one column at a time
        let region = false;
        if (experts && diskBatch) {
          const rb = experts.regionSize(), kept = experts.poolSlots(rb);
          region = kept >= OFFLOAD_GOOD_SLOTS;
          console.info(region ? `expert offload: a ${(rb / 2 ** 30).toFixed(2)} GB whole-layer region for batched prompt passes over the disk layers, out of the GPU cache (${experts.poolSlots()} -> ${kept} experts per layer)`
            : `expert offload: no batched prompt passes over the disk layers: the GPU cache would keep ${kept} experts per layer after a ${(rb / 2 ** 30).toFixed(2)} GB region (< ${OFFLOAD_GOOD_SLOTS})`);
        }
        const eng = await Qwen4ExpEngine.create({ device: ai.device, meta: G.meta, weights, layerRange: range,
          hasEmbed, hasHead, maxSeq: q ? ctx : Math.min(ctx, 2048), ngram, expertRegion: region, ...(q ? { qsa: null, kv: q.kv, qsaRing: q.ring } : { qsa: false }) });
        // batched prompt passes (room/flash.js): built now so their pipelines compile during the load. Over the disk
        // tier only with the region above; a batch that fails to build leaves this tab on one decode step per column,
        // as before, and the host narrows its frames for it.
        // Always the room's frame width, solo too: a tab has no memory check, and the dealer keeps back only that
        // width's scratch from its pledge (room/models.js flashDealReserve), not a solo device's 1024 columns
        let pb = null;
        if (!experts || region) {
          try {
            const { Qwen4ExpBatch } = await import("../engine/qwen4exp_batch.js");
            pb = await Qwen4ExpBatch.create(eng, weights, { maxCols: flashBatchCols(false, q?.ring), verifyCols: 1 });
          } catch (e) { console.warn(`Flash: no batched prompt passes on this tab (${e?.message || e}); one column at a time`); }
        }
        ai.engine = FlashRoomEngine.wrap(eng, { batch: pb });
        if (experts) console.info(experts.summary().replace(/; \d+ cuts.*/, ""));
      };
      const offInfo = () => console.info(`expert offload: layers ${off.disk[0]}-${off.disk[off.disk.length - 1]} read their experts from OPFS, ${(offload.vramBytes / 2 ** 30).toFixed(2)} GiB GPU cache, ${((+offload.diskRamBytes || 0) / 2 ** 30).toFixed(2)} GiB RAM cache`);
      const log = (s) => { console.info(s); crumb(s); };
      try {
        if (off && !globalThis.navigator?.storage?.getDirectory) throw new Error("the host asked this tab to read experts from disk, but this browser has no OPFS");
        if (!parload) await loadFlashInOrder();
        else await loadFlashParallel();
      } catch (e) {
        // a failed load (a weight download, the engine) frees what the disk tier holds: its reader's workers and file
        // handles, the RAM cache, the staging ring
        await experts?.close?.().catch(() => {});
        throw e;
      } finally {
        await resident?.close().catch(() => {});
        G.streamEntry = streamPlain;   // (ai.G outlives this load: no reader left behind on it)
      }

      // The three parts at the same time (room/flashload.js): the GPU step from the start, the experts' and the n-gram
      // table's checks next to it, the table's copy in a worker next to it, the experts' copy once the GPU step is done
      async function loadFlashParallel() {
        const ac = new AbortController();
        // Stop (or another device's failed load): the copies stop too, not only the GPU step's paced fetches
        const wordsOn = !!ngCfg && !ngramCopyBlocked();
        const parts = partsTracker({ sizes: { gpu: total, save: off ? tabExpertBytes(G, off.disk) : 0, words: wordsOn ? wordsBytes : 0 }, onParts, onProg });
        // (made after the tracker: sizing it can throw, and nothing would clear this then)
        const stopTick = setInterval(() => { if (ai.startFailed && !ac.signal.aborted) ac.abort(new Error(ai.startFailed)); }, 250);
        let checks = (off ? 1 : 0) + (wordsOn ? 1 : 0);
        const checked = () => { if (--checks === 0) parts.note(null); };
        if (checks) parts.note("Checking what's already on this computer\u2026");
        let ngram = null;
        try {
          if (off) experts = expertStoreDeferred(dev, off.disk, offload);
          const inWorker = ngramWorker && typeof globalThis.Worker === "function";
          // the experts' save in workers (engine/expert_ingest_pool.js), or null: on this thread after the GPU step
          // (fewer requests on an HTTP/1.1 host, whose 6 connections the GPU step and the n-gram copy share: savePoolFor)
          const savePool = typeof globalThis.Worker === "function" ? savePoolFor(urls, { ...saveOpts, resolve: cdnResolve }, { protocolOf }) : null;
          await runFlashParts({ parts, ac,
            gpu: async ({ at, paced }) => {
              gpuHold = { paced, signal: ac.signal };
              try { return await gpuStep(at); } finally { gpuHold = null; }
            },
            save: off ? {
              check: () => expertTierCheck(G, off.disk, { readAt }).finally(checked),
              copy: (chk, { at, signal }) => expertTierCopy(chk, { readAt, signal, onStatus: crumb, onProgress: (d) => at(d), log, pool: savePool }),
              // in workers the copy leaves the main thread to the GPU step, so it starts next to it
              early: !!savePool && !savePool.h1 && saveOpts.early,
              open: async (chk) => { await expertTierOpen(dev, chk, experts, { workers: expertWorkers, ring: expertRing }); offInfo(); },
            } : null,
            words: wordsOn ? {
              check: () => ngramCheck(ngCfg, readAt, { url: urls[ngCfg.shard] }).finally(checked),
              copy: async (c, { at, signal, parallel, after, onState }) => {
                // without a worker the copy runs on this thread, after the experts' (two there at once starve both)
                if (!inWorker) { onState("wait"); await after; onState("run"); }
                return ingestNgram(ngCfg, readAt, { dir: c.dir, parallel, signal, onProgress: (d) => at(d), onStatus: crumb, log,
                  // (one read in flight a head on an HTTP/1.1 host: its 6 connections are shared with the save and the GPU step)
                  worker: inWorker ? { urls, resolve: cdnResolve, ...(onHttp1(urls, protocolOf) ? { readAhead: 1 } : {}) } : null, mainAfter: after, onState });
              },
            } : null,
            onWordsFail: (e) => {
              const why = `${e?.name || "Error"}: ${String(e?.message || e).split("\n")[0].slice(0, 200)}`;
              console.warn(`n-gram table: copying it into OPFS failed (${why}); reading its rows by HTTP range instead`);
              crumb(`n-gram table: OPFS copy failed (${why})`);
            },
            finish: async ({ weights, words: w }) => {
              parts.note("Almost ready\u2026");
              aiStatus("Almost ready\u2026");
              // the table's copy: opened where words left it (no second signature read, no second copy); a copy that
              // failed reads its rows by HTTP range (#388: never a second, unseen copy after the bar is done)
              ngram = weights.pleTable ? await openNgramTable(G, M, !w ? { copy: false } : w.ok ? { dir: w.dir, copy: false } : { copy: false, opfs: false }) : null;
              await makeEngine(weights, ngram);
            },
          });
        } catch (e) { await ngram?.close?.(); throw e; }
        finally { clearInterval(stopTick); }
      }

      // One after the other, as before (?parload=0, for one release): the experts saved, the n-gram table saved, then
      // the GPU, on one bar of named steps (room/loadsteps.js)
      async function loadFlashInOrder() {
        // The load's steps on one bar (room/loadsteps.js), sized before any of them starts: the offloaded layers' experts
        // saved to this computer, the n-gram table saved there (when this device holds its layer and may write to its
        // disk), then the layers onto the GPU. Each step's bytes go to onProgress as the whole load's done / total, with
        // info { step, stepNo, steps, stepDone, stepTotal, line } (line: the technical detail, for the console and a
        // tooltip): the bar never starts over between steps, and it reaches 100% only with the last GPU byte
        const steps = loadTracker(planSteps({ saveBytes: off ? off.disk.reduce((a, l) => a + flashExpertBytes(G, l), 0) : 0,
          wordsBytes: ngCfg && !ngramCopyBlocked() ? wordsBytes : 0, gpuBytes: total }));
        const prog = (step, d, tot, line = null) => {
          const o = steps.at(step, tot > 0 ? d / tot : 0);
          onProg(o.done, o.total, { step, stepNo: o.stepNo, steps: o.steps, stepDone: d, stepTotal: tot, line });
        };
        if (off) {
          let line = null;
          aiStatus("Checking what's already saved on this computer\u2026");
          // (the copy's own lines are detail: the breadcrumbs and the bar's tooltip, not the status)
          const t = await tabExpertTier(dev, G, off.disk, { readAt, offload, workers: expertWorkers, ring: expertRing,
            onStatus: (s) => { line = s; crumb(s); }, onProgress: (d, tot, s) => prog("save", d, tot, s || line), log });
          experts = t.store;
          if (!t.ingest) steps.drop("save");   // on this computer already: the bar skips the step, never goes back
          offInfo();
        }
        // The n-gram table's copy before the GPU step: openNgramTable below then finds the copy there and only opens
        // it; a copy that fails is said in the console, and openNgramTable then reads the rows by HTTP range
        let ngramEarlyFailed = false;
        if (ngCfg && steps.plan.some((s) => s.id === "words")) {
          aiStatus("Checking what's already saved on this computer\u2026");
          const copied = await saveNgramEarly(G, M, ngCfg, (d, tot) => prog("words", d, tot), { onFail: () => { ngramEarlyFailed = true; } });
          if (!copied) steps.drop("words");
        }
        prog("gpu", 0, total);
        const weights = await gpuStep((d) => prog("gpu", d, total));
        let ngram = null;
        try {
          // (a copy that failed just now is not tried again here, after the bar said the load was done: rows by HTTP range)
          ngram = weights.pleTable ? await openNgramTable(G, M, { copy: !ngramEarlyFailed }) : null;
          aiStatus("Almost ready\u2026");
          await makeEngine(weights, ngram);
        } catch (e) { await ngram?.close?.(); throw e; }
      }
    } else if (M.kind === "gguf") {
      aiStatus("Looking up the model\u2026");
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
      aiStatus("Almost ready\u2026");
      ai.engine = await DenseEngine.create({
        coopWG: ai.tune?.wg, coopRows: ai.tune?.rows,
        device: ai.device, cfg: ai.cfg, weights,
        layerRange: range, hasEmbed, hasHead, maxSeq: ctx,
      });
    } else {
      const names = shardTensorNames(ai.cfg, range, hasEmbed, hasHead);
      const tensors = await fetchModelShard(M.st, names, (_part, done, total) => onProg(done, total));
      aiStatus("Almost ready\u2026");
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
  // opts: dir (the copy's directory, known already: no signature read), copy (false: never copy here), opfs (false:
  // straight to HTTP range, the copy failed in this load)
  async function openNgramTable(G, M, { copy: tryCopy = true, dir: known = null, opfs = true } = {}) {
    const cfg = ngramConfig(G), urls = M.shards?.length ? M.shards : [M.gguf];
    const readAt = httpReadAt(urls, { fetch: (u, init) => fetch(u, { ...init, cache: "no-store" }) });
    let dir = opfs ? known : null;
    if (opfs && !dir && globalThis.navigator?.storage?.getDirectory) {
      try { dir = ngramDir(await ngramContentSig(cfg, readAt, { url: urls[cfg.shard] })); }
      catch (e) { console.warn(`n-gram table: could not read its signature (${e?.message || e}); reading its rows by HTTP range`); }
    }
    const copy = tryCopy ? await copyNgram(cfg, readAt, dir) : { skipped: opfs ? (known ? "copied earlier in this load" : "not copied in this load") : "the copy failed earlier in this load" };
    if (copy.skipped && !known) { console.info(`n-gram table: not copied to this device's disk (${copy.skipped})`); crumb(`n-gram table: no OPFS copy (${copy.skipped})`); }
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
  // onProgress(done, total): the copy's bytes (room/loadsteps.js "words" step); its status lines are detail (breadcrumbs)
  async function copyNgram(cfg, readAt, dir, { storage = globalThis.navigator?.storage, onProgress = () => {} } = {}) {
    if (!ngramIngest) return { skipped: "?ngramcopy=0" };
    if (!dir) return { skipped: "no OPFS here" };
    const no = ngramCopyBlocked(storage);
    if (no) return { skipped: no };
    try {
      return { copied: await ingestNgram(cfg, readAt, { dir, storage, onProgress, onStatus: crumb, log: (s) => { console.info(s); crumb(s); } }) };
    } catch (e) {
      const why = `${e?.name || "Error"}: ${String(e?.message || e).split("\n")[0].slice(0, 200)}`;
      console.warn(`n-gram table: copying it into OPFS failed (${why}); reading its rows by HTTP range instead`);
      crumb(`n-gram table: OPFS copy failed (${why})`);
      return { skipped: "the copy failed" };
    }
  }

  // Why this tab won't copy the n-gram table to its disk (null: it may): the copy's gate, also asked before a load
  // sizes its steps (a copy that won't happen is no step on the bar)
  function ngramCopyBlocked(storage = globalThis.navigator?.storage) {
    if (!ngramIngest) return "?ngramcopy=0";
    if (!storage?.getDirectory) return "no OPFS here";
    const { tabDisk, choice } = tabDiskNow();
    if (choice !== "allow" || !tabDisk?.models?.length) return choice === "deny" ? "its user did not allow models on its disk" : "its user was not asked to store models on its disk";
    return null;
  }
  // The n-gram table's copy ahead of the GPU step (loadShard qwen4exp): openNgramTable's signature read and copy, but
  // no workers opened (no file handle or lock lives across the GPU load). -> bytes copied; 0 when it was all there,
  // not allowed, or it failed (said in the console; a copy that failed calls onFail and openNgramTable then reads by HTTP range)
  // onFail(): the copy started and failed (not called when it was not allowed or already there)
  async function saveNgramEarly(G, M, cfg, onProgress = () => {}, { storage = globalThis.navigator?.storage, readAt = null, onFail = () => {} } = {}) {
    const urls = M.shards?.length ? M.shards : [M.gguf];
    readAt ||= httpReadAt(urls, { fetch: (u, init) => fetch(u, { ...init, cache: "no-store" }) });
    try {
      if (ngramCopyBlocked(storage)) return 0;
      const dir = ngramDir(await ngramContentSig(cfg, readAt, { url: urls[cfg.shard] }));
      const r = await copyNgram(cfg, readAt, dir, { storage, onProgress });
      if (r.skipped === "the copy failed") onFail();
      return r.copied || 0;
    } catch (e) {
      console.warn(`n-gram table: the early copy did not run (${e?.message || e}); it is tried again when the table opens`);
      return 0;
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

  // (copyNgram, saveNgramEarly: for tests of the consent gate and the early copy)
  return { fetchModelHeader, fetchGGUFHeader, loadShard, loadDraft, copyNgram, saveNgramEarly };
}
