// Browser model headers, weight loading and engine construction. The room owns GPU
// acquisition, progress/pacing, membership and recovery. Engine constructors arrive
// after room.js's lazy import; importing this module does not load the GPU engine.
import { parseGGUFHeader, ggufWeights, ggufShardBytes, GGML_EMBED, GGML_OUTPUT, GGML_FINAL_NORM,
  ggmlLayerNames, qwen35Weights, qwen35ShardBytes, qwen35NamesFor, tokenizerFromGGUF, gpuUploadEntry, streamEntryToGPU }
  from "../engine/gguf.js";
import { roomQwen35Options } from "../engine/preset.js";
import { MODELS, MAX_SEQ, mergeSplitHeaders } from "./models.js";
import { cacheKey } from "./weightcache.js";
import { attachBrowserWeightCache } from "./convertedcache.js";

export function createModelLoader({ state: ai, rangeFetch, getWeightCache, getMeta, hooks }) {
  const { onStatus: aiStatus, crumb, onCacheHit, onModelLoaded: apiModelLoaded } = hooks;

  // ---- converted weights on disk (room/convertedcache.js, OPFS): a second load of the same layers skips
  // the CPU conversion (K-quant -> Q8, BF16/Q5_0 -> f32, the embedding's repack). Keyed by model URL,
  // its pinned revision, the GGUF header and engine/gguf.js itself, so any of those changing starts
  // fresh. ?wcache=0 turns it off (A/B); ?wcacheverify=1 also checks each entry's payload hash.
  const WCACHE = new URLSearchParams(location.search).get("wcache") !== "0";
  const WCACHE_VERIFY = new URLSearchParams(location.search).get("wcacheverify") === "1";
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
  let pacerHook = null;
  const streamWithRetry = (url, streamOpts) => async (info) => {
    try { return await streamEntryToGPU(ai.device, info, openRangeOf(url), streamOpts); }
    catch (e) {
      if (!/short tensor/.test(String(e)) && !e?.retryNet) throw e;   // (retryNet: a room device stopped sending it)
      const c = await getWeightCache(), u = info.url || url;
      if (c) c.delete(cacheKey(u, info.byteOffset, info.byteOffset + info.byteLength - 1)).catch(() => {});
      return streamEntryToGPU(ai.device, info, (i) => rangeFetch(i.url || url, i.byteOffset, i.byteOffset + i.byteLength - 1, true), streamOpts);
    }
  };
  // Prefetch: a shard is hundreds of tensors (a 27B worker with 30 layers fetches ~450), and fetching
  // them one after another pays the model host's time-to-first-byte every time. When the loader
  // asks for a tensor, the next PREFETCH tensors of the shard (file order) are requested too, so
  // several are in flight at once. Phones keep one: every buffered body is RAM they do not have.
  // ?prefetch=N overrides (0 = off, for A/B).
  const PREFETCH_Q = new URLSearchParams(location.search).get("prefetch");
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
    const ahead = PREFETCH_Q != null ? Math.max(0, parseInt(PREFETCH_Q, 10) || 0) : getMeta()?.phone ? 1 : 4;
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
  const openRangeOf = (url) => async (info) => {
    if (pacerHook) await pacerHook();
    crumb("streaming " + info.name + " (" + (info.byteLength / 2 ** 20).toFixed(0) + " MB)");
    return rangeOf(url, info);
  };
  const rangeBytesOf = (url) => async (info) => {
    if (pacerHook) await pacerHook();
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

  async function loadShard({ modelKey, range, hasEmbed, hasHead, ctx, kv, streamOpts, onProgress: onProg, pace }, engines) {
    const M = MODELS[modelKey];
    const { makeTokenizer, DenseEngine, Qwen35Engine, fetchModelShard, shardTensorNames } = engines;
    pacerHook = pace;
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
      G.streamEntry = streamWithRetry(M.gguf, streamOpts);
      const wc = await useConvertedCache(G, M.gguf), tw = performance.now();
      const weights = await qwen35Weights(G, rangeBytesOf(M.gguf), opts, (done) => onProg(done, total),
        (e, name) => gpuUploadEntry(ai.device, e, name === GGML_EMBED));   // straight to the GPU, RAM stays flat
      await convertedSummary(wc, tw);
      aiStatus("building GPU pipelines (compiling shaders)\u2026");
      ai.engine = await Qwen35Engine.create({
        device: ai.device, meta: G.meta, weights, vocab: G.tensors[GGML_EMBED]?.shape?.[0],
        layerRange: range, hasEmbed, hasHead, maxSeq: ctx,
        coopWG: ai.tune?.wg, coopRows: ai.tune?.rows,
        // the room's settings (engine/preset.js: 16 batch columns, the small draft head, fused kernels, GPU
        // sampling, and the ?flags that change them). The benchmarks and profilers build their engines from
        // the same preset, so their numbers come from these settings. Prefill options are not set there:
        // every device takes the engine's defaults, so host and workers agree.
        ...roomQwen35Options(location.search),
        // ?kv=q8 on the host: int8 KV cache. The host decides for every device and sends its choice with
        // ai-load (room/models.js kvModeFor, kvForLoad), so this overrides the preset's own ?kv reading.
        kvQ8: kv === "q8",
      });
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
      G.streamEntry = streamWithRetry(M.gguf, streamOpts);
      const wc = await useConvertedCache(G, M.gguf), tw = performance.now();
      const weights = await ggufWeights(G, rangeBytesOf(M.gguf), opts, (done) => onProg(done, total),
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

  async function loadDraft(M, DenseEngine) {
    const cfg = await (await fetch(M.cfg)).json();
    const G = await fetchGGUFHeader(M.gguf, false);
    const L = cfg.num_hidden_layers;
    const weights = await ggufWeights(G, rangeBytesOf(M.gguf), { lo: 0, hi: L, hasEmbed: true, hasHead: true });
    return DenseEngine.create({ device: ai.device, cfg, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: ai.engine.maxSeq,
      coopWG: ai.tune?.wg, coopRows: ai.tune?.rows });
  }

  return { fetchModelHeader, fetchGGUFHeader, loadShard, loadDraft };
}
