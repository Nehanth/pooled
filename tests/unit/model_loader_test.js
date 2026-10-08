// Real header/weight loading with in-memory ranges and recording engine constructors.
// Browser integration tests cover GPU construction and room lifecycle around this boundary.
import { createModelLoader } from "../../room/model-loader.js";
import { GGML_EMBED, GGML_OUTPUT, GGML_FINAL_NORM, GGML_Q4_0, GGML_BF16, ggmlLayerNames, qwen35LayerNames } from "../../engine/gguf.js";
import { MODELS, MAX_SEQ } from "../../room/models.js";
import { HDR, entryFile, MIN_ENTRY_BYTES, attachBrowserWeightCache } from "../../room/convertedcache.js";
import { FakeDir } from "../helpers/fake-opfs.js";
import { roomQwen35Options } from "../../engine/preset.js";

const eq = (a, b) => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${JSON.stringify(a)} != ${JSON.stringify(b)}`); };
const ok = (v, message = "assertion failed") => { if (!v) throw new Error(message); };
async function rejects(fn, message) {
  try { await fn(); } catch (e) { ok(String(e).includes(message), String(e)); return; }
  throw new Error("expected rejection: " + message);
}

function deferred() {
  let resolve;
  const promise = new Promise((r) => { resolve = r; });
  return { promise, resolve };
}

async function withGPU(fn) {
  const usage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { STORAGE: 128, COPY_DST: 8, COPY_SRC: 4 } });
  const device = { createBuffer: ({ size }) => ({ size, destroy() {} }), queue: { writeBuffer() {} },
    pushErrorScope() {}, popErrorScope: async () => null };
  try { await fn(device); }
  finally {
    if (usage) Object.defineProperty(globalThis, "GPUBufferUsage", usage);
    else delete globalThis.GPUBufferUsage;
  }
}

async function withStorage(root, fn) {
  const fetch = globalThis.fetch, storage = Object.getOwnPropertyDescriptor(navigator, "storage");
  globalThis.fetch = async () => new Response("converter source");
  Object.defineProperty(navigator, "storage", { configurable: true, value: {
    getDirectory: async () => root, estimate: async () => ({ quota: 100 * 2 ** 30, usage: 0 }),
  } });
  try { await fn(); }
  finally {
    globalThis.fetch = fetch;
    if (storage) Object.defineProperty(navigator, "storage", storage);
    else delete navigator.storage;
  }
}

const test = Deno.test;

function loader(state, rangeFetch, { getMeta = () => ({}), getWeightCache = async () => null, hooks,
  options = { wcache: false, prefetch: 0 }, getEnginePreset = () => ({}) } = {}) {
  return createModelLoader({ state, rangeFetch, getWeightCache, getMeta, getEnginePreset, options, hooks });
}

// A small f32 GGUF index, optionally with metadata, shapes and distinct tensor payloads.
function header(name, { meta = { "tokenizer.ggml.model": "test" }, shapes = {}, payload = false } = {}) {
  const names = Array.isArray(name) ? name : [name];
  const bytes = [], enc = new TextEncoder();
  const u32 = (n) => { for (let i = 0; i < 4; i++) bytes.push(n >>> (8 * i) & 255); };
  const u64 = (n) => { u32(n); u32(0); };
  const str = (s) => { const b = enc.encode(s); u64(b.length); bytes.push(...b); };
  u32(0x46554747); u32(3); u64(names.length); u64(Object.keys(meta).length);
  for (const [key, value] of Object.entries(meta)) {
    str(key);
    if (Array.isArray(value)) { u32(9); u32(8); u64(value.length); for (const v of value) str(v); }
    else if (typeof value === "string") { u32(8); str(value); }
    else { u32(4); u32(value); }
  }
  let offset = 0;
  for (const n of names) {
    const shape = shapes[n] || [1];
    str(n); u32(shape.length); for (const dim of [...shape].reverse()) u64(dim);
    u32(0); u64(offset); offset += shape.reduce((a, b) => a * b, 1) * 4;
    if (payload) offset = Math.ceil(offset / 32) * 32;
  }
  if (payload) {
    while (bytes.length % 32) bytes.push(0);
    for (const [i, n] of names.entries()) {
      const data = new Float32Array((shapes[n] || [1]).reduce((a, b) => a * b, 1)).fill(i + 1);
      bytes.push(...new Uint8Array(data.buffer));
      while (bytes.length % 32) bytes.push(0);
    }
  }
  return new Uint8Array(bytes);
}

function qwen35() {
  const names = [GGML_EMBED, GGML_OUTPUT, GGML_FINAL_NORM,
    ...[0, 1].flatMap((i) => Object.values(qwen35LayerNames(i, true)).filter((n) => typeof n === "string")),
    ...["eh_proj", "enorm", "hnorm", "shared_head_norm"].map((n) => `blk.1.nextn.${n}.weight`)];
  const meta = { "qwen35.block_count": 2, "qwen35.nextn_predict_layers": 1, "qwen35.full_attention_interval": 1,
    "tokenizer.ggml.tokens": ["a", "b", "c"], "tokenizer.ggml.merges": ["a b"],
    "tokenizer.ggml.pre": "qwen35", "tokenizer.chat_template": "host template {{ messages }}" };
  const shapes = Object.fromEntries(names.map((n) => [n, n === GGML_EMBED || n === GGML_OUTPUT ? [3, 2] : [2]]));
  const file = header(names, { meta, shapes, payload: true }), requests = [], tokenizers = [];
  const state = { device: {}, tune: { wg: 64, rows: 4 } };
  const engines = { makeTokenizer(tj) { tokenizers.push(tj); return {}; }, Qwen35Engine: { async create(opts) { return opts; } } };
  const opts = { modelKey: "qwen3.8-27b", range: [0, 1], hasEmbed: true, hasHead: true, ctx: 8192, kv: "q8",
    streamOpts: {}, onProgress() {} };
  const rangeFetch = async (url, lo, hi) => { requests.push([url, lo, hi]); return new Response(file.slice(lo, hi + 1)); };
  return { names, meta, file, requests, tokenizers, state, engines, opts, rangeFetch };
}

function dense(names = [GGML_EMBED]) {
  const tensors = Object.fromEntries(names.map((name, i) => [name,
    { name, shape: [1, 1], nElems: 1, ggmlType: 0, byteOffset: i * 4, byteLength: 4 }]));
  const state = { G: { meta: {}, tensors }, GModel: "qwen3-0.6b", cfg: {}, device: {}, tune: { wg: 64, rows: 4 } };
  let built;
  const engines = { DenseEngine: { async create(opts) { built = opts; return opts; } } };
  const opts = { modelKey: state.GModel, range: [0, 0], hasEmbed: true, hasHead: false, ctx: 4096,
    streamOpts: {}, onProgress() {}, pace: async () => {} };
  return { state, engines, opts, built: () => built };
}

// A real BF16 conversion large enough for the cache's ordinary admission policy.
function cacheModel() {
  const f = dense(), n = MIN_ENTRY_BYTES / 2;
  Object.assign(f.state.G.tensors[GGML_EMBED], { ggmlType: GGML_BF16, shape: [n / 2, 2], nElems: n, byteLength: n * 2 });
  const bytes = new Uint16Array(n).fill(0x3f80); // BF16 1.0
  let fetched = 0;
  const rangeFetch = async () => { fetched++; return new Response(bytes); };
  return { ...f, rangeFetch, fetched: () => fetched };
}

test("model loader: grow a short header and retain tokenizer only when requested", async () => {
  const requests = [];
  const m = loader({}, async (url, lo, hi) => {
    requests.push([url, lo, hi]);
    return new Response(requests.length === 1 ? new Uint8Array(4) : header("x"));
  });
  const g = await m.fetchGGUFHeader("model", false);
  eq(requests.map((r) => r[2] + 1), [12 * 2 ** 20, 24 * 2 ** 20]);
  eq(g.meta, {});
  const withTok = await m.fetchGGUFHeader("model");
  eq(withTok.meta["tokenizer.ggml.model"], "test");
});

test("model loader: split headers preserve tensor source URLs", async () => {
  const m = loader({}, async (url) => new Response(header(url)));
  const g = await m.fetchModelHeader({ shards: ["first", "second"] });
  eq(g.tensors.first.url, undefined); // first-file tensors use the model's base URL
  eq(g.tensors.second.url, "second");
  eq(g.meta["tokenizer.ggml.model"], "test");
});

test("model loader: cached dense header, short range retry, progress and constructor options", async () => {
  const { state, engines, opts, built } = dense(), requests = [], progress = [];
  const m = loader(state, async (url, lo, hi, noCache = false) => {
    requests.push([url, lo, hi, noCache]);
    return new Response(noCache ? new Float32Array([7]) : new Uint8Array(0));
  });
  await m.loadShard({ ...opts, onProgress: (done, total) => progress.push([done, total]) }, engines);
  eq(requests, [[MODELS[opts.modelKey].gguf, 0, 3, false], [MODELS[opts.modelKey].gguf, 0, 3, true]]);
  eq(progress, [[4, 4]]);
  ok(state.engine === built());
  eq(built().maxSeq, 4096);
  eq(built().coopWG, 64);
  eq(built().layerRange, [0, 0]);
});

test("model loader: peer stream failure retries over the network", async () => {
  const { state, engines, opts } = dense(), retries = [];
  const m = loader(state, async (_url, _lo, _hi, noCache = false) => {
    retries.push(noCache);
    if (noCache) return new Response(new Float32Array([8]));
    return new Response(new ReadableStream({ start(c) { c.error(Object.assign(new Error("peer left"), { retryNet: true })); } }));
  });
  await m.loadShard(opts, engines);
  eq(retries, [false, true]);
});

test("model loader: short network retry rejects without publishing an engine", async () => {
  const { state, engines, opts, built } = dense();
  const m = loader(state, async () => new Response(new Uint8Array(0)));
  await rejects(() => m.loadShard(opts, engines), "short download");
  eq(built(), undefined);
  eq(state.engine, undefined);
});

test("model loader: a room pacing failure propagates before fetching weights", async () => {
  const { state, engines, opts } = dense();
  let fetched = false;
  const m = loader(state, async () => { fetched = true; return new Response(new Uint8Array(4)); });
  await rejects(() => m.loadShard({ ...opts, pace: async () => { throw new Error("start stopped"); } }, engines), "start stopped");
  ok(!fetched);
});

test("model loader: prefetch reads live phone metadata and never refetches taken tensors", async () => {
  const names = Object.values(ggmlLayerNames(0));
  const { state, engines, opts } = dense(names), offsets = [];
  let meta = { phone: false };
  const m = loader(state, async (_url, lo) => { offsets.push(lo); return new Response(new Float32Array([1])); }, { getMeta: () => meta, options: { wcache: false } });
  meta = { phone: true };
  await m.loadShard({ ...opts, range: [0, 1], hasEmbed: false }, engines);
  eq(offsets.slice(0, 2), [4, 0]); // one range ahead, then the requested range
  eq(offsets.length, names.length);
  eq(new Set(offsets).size, names.length);
});

test("model loader: explicit prefetch overrides phone policy and is captured at creation", async () => {
  for (const [phone, prefetch, first] of [[true, 2, [4, 8, 0]], [false, 0, [0, 28, 4]]]) {
    const { state, engines, opts } = dense(Object.values(ggmlLayerNames(0))), offsets = [];
    const options = { wcache: false, prefetch };
    const m = loader(state, async (_url, lo) => { offsets.push(lo); return new Response(new Float32Array([1])); },
      { options, getMeta: () => ({ phone }) });
    options.prefetch = 5;
    await m.loadShard({ ...opts, range: [0, 1], hasEmbed: false }, engines);
    eq(offsets.slice(0, 3), first);
  }
});

test("model loader: prefetch fetches each tensor once when file and model order differ", async () => {
  for (const phone of [true, false]) {
    const names = Object.values(ggmlLayerNames(0)), { state, engines, opts } = dense(names), offsets = [];
    // Model order visits the second half of the file before its first half.
    for (const [i, name] of names.entries()) state.G.tensors[name].byteOffset = ((i + 6) % names.length) * 4;
    const m = loader(state, async (_url, lo) => { offsets.push(lo); return new Response(new Float32Array([1])); }, { getMeta: () => ({ phone }), options: { wcache: false } });
    await m.loadShard({ ...opts, range: [0, 1], hasEmbed: false }, engines);
    eq(offsets.length, names.length);
    eq(new Set(offsets).size, names.length);
  }
});

test("model loader: split-shard prefetch distinguishes identical offsets in different files", async () => {
  const layerNames = ggmlLayerNames(0), names = Object.values(layerNames), { state, engines, opts } = dense(names), requests = [];
  const first = MODELS[opts.modelKey].gguf, second = first + ".part2", perFile = Math.ceil(names.length / 2);
  for (const [i, name] of names.entries()) Object.assign(state.G.tensors[name], {
    url: i < perFile ? first : second, shard: Math.floor(i / perFile), byteOffset: (i % perFile) * 4,
  });
  const m = loader(state, async (url, lo, hi) => {
    requests.push([url, lo, hi]);
    return new Response(new Float32Array([lo / 4 + (url === first ? 1 : perFile + 1)]));
  }, { options: { wcache: false, prefetch: 4 } });
  await m.loadShard({ ...opts, range: [0, 1], hasEmbed: false }, engines);
  for (const [key, name] of Object.entries(layerNames)) eq([...state.engine.weights.layers[0][key].data], [names.indexOf(name) + 1]);
  eq(requests[0], [first, 4, 7]); // the URL-keyed index actually schedules work ahead of the first tensor
  eq(requests.length, names.length);
  eq(new Set(requests.map((r) => JSON.stringify(r))).size, names.length);
});

test("model loader: successful load consumes or cancels every prefetched response body", async () => {
  const f = qwen35(), bodies = [];
  const m = loader(f.state, async (_url, lo, hi) => {
    const body = { state: "unread" }; bodies.push(body);
    return new Response(new ReadableStream({
      start(c) { c.enqueue(f.file.slice(lo, hi + 1)); },
      pull(c) { body.state = "consumed"; c.close(); },
      cancel() { body.state = "cancelled"; },
    }, { highWaterMark: 0 }));
  }, { options: { wcache: false, prefetch: 4 } });
  f.state.G = await m.fetchModelHeader(MODELS[f.opts.modelKey]);
  f.state.GModel = f.opts.modelKey;
  // Without eh_proj the optional nextn block is skipped; its other tensors are still in the prefetch plan.
  delete f.state.G.tensors["blk.1.nextn.eh_proj.weight"];
  await m.loadShard(f.opts, f.engines);
  ok(f.state.engine); eq(f.state.engine.weights.mtp, undefined);
  ok(bodies.some((b) => b.state === "consumed"));
  ok(bodies.some((b) => b.state === "cancelled"), "unused nextn prefetches must be cancelled");
  eq(bodies.filter((b) => b.state === "unread").length, 0);
});

test("model loader: the next load cancels prefetched bodies left unread by a failed load", async () => {
  const { state, engines, opts } = dense(Object.values(ggmlLayerNames(0))), cancelled = [];
  const m = loader(state, async (_url, lo) => new Response(new ReadableStream({
    start(c) { c.enqueue(new Uint8Array(new Float32Array([1]).buffer)); },
    pull(c) { c.close(); },
    cancel() { cancelled.push(lo); },
  })), { options: { wcache: false, prefetch: 2 } });
  let n = 0;
  await rejects(() => m.loadShard({ ...opts, range: [0, 1], hasEmbed: false,
    pace: async () => { if (++n > 1) throw new Error("stopped"); } }, engines), "stopped");
  await m.loadShard({ ...opts, hasEmbed: false }, engines);
  eq(cancelled.sort((a, b) => a - b), [4, 8]);
});

test("model loader: safetensors retains MAX_SEQ and passes shard selection and progress", async () => {
  const state = { cfg: { marker: 1 }, device: {}, tune: { wg: 64, rows: 4 } }, calls = [];
  const tensors = {}, m = loader(state, () => { throw new Error("unexpected range fetch"); });
  const engines = {
    shardTensorNames(...args) { calls.push(args); return ["selected"]; },
    async fetchModelShard(url, names, progress) { calls.push([url, names]); progress("part", 2, 3); return tensors; },
    DenseEngine: { async create(opts) { return opts; } },
  };
  await m.loadShard({ modelKey: "smollm-135m", range: [1, 2], hasEmbed: false, hasHead: false, ctx: 8192,
    onProgress: (...args) => calls.push(args) }, engines);
  eq(calls, [[state.cfg, [1, 2], false, false], [MODELS["smollm-135m"].st, ["selected"]], [2, 3]]);
  ok(state.engine.tensors === tensors);
  eq(state.engine.maxSeq, MAX_SEQ);
});

test("model loader: Qwen35 worker keeps context, KV choice and layer range", async () => {
  const state = { GModel: "qwen3.8-27b", G: { meta: { "qwen35.block_count": 5, "qwen35.nextn_predict_layers": 1 }, tensors: {} },
    device: {}, tune: { wg: 64, rows: 4 } };
  const m = loader(state, () => { throw new Error("unexpected fetch of cached header"); });
  await m.loadShard({ modelKey: state.GModel, range: [0, 0], hasEmbed: false, hasHead: false, ctx: 8192, kv: "q8",
    streamOpts: {}, onProgress() {} }, { Qwen35Engine: { async create(opts) { return opts; } } });
  eq(state.cfg.num_hidden_layers, 4);
  eq(state.engine.maxSeq, 8192);
  eq(state.engine.kvQ8, true);
  eq(state.engine.layerRange, [0, 0]);
  ok(state.engine.device === state.device);
});

test("model loader: Qwen35 host loads trunk, nextn, tokenizer, template, vocab and progress", async () => {
  const f = qwen35(), progress = [], notifications = [];
  const m = loader(f.state, f.rangeFetch, { hooks: { onModelLoaded() { notifications.push(f.state.tok.chatTemplate); } } });
  await m.loadShard({ ...f.opts, onProgress: (done, total) => progress.push([done, total]) }, f.engines);
  const { weights, vocab, layerRange, hasEmbed, hasHead } = f.state.engine;
  eq(f.state.cfg.num_hidden_layers, 1); // nextn is not a trunk layer
  eq(vocab, 3); // vocabulary and hidden dimension (2) deliberately differ
  eq(layerRange, [0, 1]); ok(hasEmbed && hasHead);
  eq(f.tokenizers, [{ model: { vocab: { a: 0, b: 1, c: 2 }, merges: ["a b"] }, pre: "qwen35" }]);
  eq(f.state.tok.chatTemplate, f.meta["tokenizer.chat_template"]);
  eq(notifications, [f.meta["tokenizer.chat_template"]]);
  const value = (entry, name) => eq([...entry.data], Array(f.state.G.tensors[name].nElems).fill(f.names.indexOf(name) + 1));
  value(weights.embed, GGML_EMBED); value(weights.head, GGML_OUTPUT); value(weights.finalNorm, GGML_FINAL_NORM);
  eq(weights.layers.length, 1); ok(weights.mtp, "host must load nextn");
  const layerKeys = { attnNorm: "attn_norm", postNorm: "post_attention_norm", ffnGate: "ffn_gate", ffnUp: "ffn_up", ffnDown: "ffn_down",
    wq: "attn_q", wk: "attn_k", wv: "attn_v", wo: "attn_output", qNorm: "attn_q_norm", kNorm: "attn_k_norm" };
  for (const [i, layer] of [weights.layers[0], weights.mtp.layer].entries()) {
    ok(layer.isFull);
    for (const [key, name] of Object.entries(layerKeys)) value(layer[key], `blk.${i}.${name}.weight`);
  }
  for (const [key, name] of Object.entries({ ehProj: "eh_proj", enorm: "enorm", hnorm: "hnorm", sharedHeadNorm: "shared_head_norm" }))
    value(weights.mtp[key], `blk.1.nextn.${name}.weight`);
  const sizes = Object.values(f.state.G.tensors).map((t) => t.byteLength), total = sizes.reduce((a, b) => a + b, 0);
  eq(progress.length, f.names.length);
  ok(progress.every(([done, t], i) => done > (progress[i - 1]?.[0] || 0) && t === total));
  eq(progress.at(-1), [total, total]);
  eq(f.requests.length, f.names.length + 1); // header plus every tensor
});

test("model loader: Qwen35 worker promotion refetches tokens; a complete host header is reused", async () => {
  const f = qwen35(), m = loader(f.state, f.rangeFetch);
  const headers = () => f.requests.filter(([, lo]) => lo === 0).length;
  await m.loadShard({ ...f.opts, hasEmbed: false, hasHead: false }, f.engines);
  eq(headers(), 1); eq(f.tokenizers.length, 0);
  eq(f.state.G.meta["tokenizer.ggml.tokens"], undefined);
  await m.loadShard(f.opts, f.engines);
  eq(headers(), 2); eq(f.tokenizers.length, 1);
  eq(f.state.G.meta["tokenizer.ggml.tokens"], ["a", "b", "c"]);
  eq(f.state.tok.chatTemplate, f.meta["tokenizer.chat_template"]);
  await m.loadShard(f.opts, f.engines);
  eq(headers(), 2); eq(f.tokenizers.length, 2);
  eq(f.tokenizers[1], f.tokenizers[0]);
  eq(f.state.tok.chatTemplate, f.meta["tokenizer.chat_template"]);
  eq(f.state.engine.vocab, 3); ok(f.state.engine.weights.mtp);
});

test("model loader: preset is read at engine construction and host KV wins without mutation", async () => {
  const f = qwen35();
  let preset = Object.freeze(roomQwen35Options("")), reads = 0;
  const m = loader(f.state, f.rangeFetch, { getEnginePreset() { reads++; return preset; } });
  eq(reads, 0);
  for (const [i, kv] of ["f16", "q8"].entries()) {
    const query = i ? "fuse=0&draftchain=0&kv=f16&gpusample=0" : "draftvocab=64&specfuse=0&kv=q8";
    const next = Object.freeze(roomQwen35Options(query));
    await m.loadShard({ ...f.opts, kv, onProgress() { preset = next; } }, f.engines);
    eq(reads, i + 1);
    for (const [k, v] of Object.entries(next)) eq(f.state.engine[k], k === "kvQ8" ? kv === "q8" : v);
    eq(next, roomQwen35Options(query));
  }
});

test("model loader: default options and omitted hooks work without browser location", async () => {
  eq(globalThis.location, undefined);
  await withStorage(new FakeDir(), async () => {
    const f = cacheModel();
    const m = createModelLoader({ state: f.state, rangeFetch: f.rangeFetch, getMeta: () => ({ phone: true }),
      getWeightCache: async () => null, getEnginePreset: () => ({}) });
    await m.loadShard(f.opts, f.engines);
    ok(f.state.G.entryCache, "cache must be enabled by default");
    eq(f.state.G.entryCache.stats.write, 1);
    const file = await f.state.G.entryCache.dir.getFileHandle(entryFile(GGML_EMBED));
    new DataView(file.data.buffer).setFloat32(HDR, 9, true);
    await m.loadShard(f.opts, f.engines);
    eq(f.fetched(), 1); eq(f.state.engine.weights.embed.data[0], 9); // verification defaults off
  });
});

test("model loader: explicit cache options are captured at creation", async () => {
  for (const enabled of [false, true]) await withStorage(new FakeDir(), async () => {
    const f = cacheModel(), options = { wcache: enabled, wcacheVerify: true, prefetch: 0 };
    const m = loader(f.state, f.rangeFetch, { options });
    options.wcache = !enabled; options.wcacheVerify = false;
    await m.loadShard(f.opts, f.engines);
    if (enabled) {
      const file = await f.state.G.entryCache.dir.getFileHandle(entryFile(GGML_EMBED));
      new DataView(file.data.buffer).setFloat32(HDR, 9, true);
    } else ok(f.state.G.entryCache === null, "explicit wcache: false must disable the cache");
    await m.loadShard(f.opts, f.engines);
    eq(f.fetched(), 2); eq(f.state.engine.weights.embed.data[0], 1);
    if (enabled) eq(f.state.G.entryCache.stats.bad, 1);
  });
});

test("model loader: converted cache hashes the original converter URL and tolerates unavailable storage", async () => {
  // Prime another converter so this assertion does not depend on which cache test ran first.
  await attachBrowserWeightCache({}, "model", { srcUrl: "previous-converter", fetchFn: async () => new Response("old source"),
    storage: { getDirectory: async () => new FakeDir() } });
  const { state, engines, opts } = dense(), urls = [];
  const fetch = globalThis.fetch, storage = Object.getOwnPropertyDescriptor(navigator, "storage");
  Object.defineProperty(navigator, "storage", { configurable: true,
    value: { getDirectory() { throw new Error("storage unavailable"); } } });
  globalThis.fetch = async (url) => { urls.push(url); return new Response("converter source"); };
  try {
    await loader(state, async () => new Response(new Float32Array([1])), { options: { prefetch: 0 } }).loadShard(opts, engines);
    eq(urls, [new URL("../../engine/gguf.js", import.meta.url).href]);
    eq(state.G.entryCache, null);
    ok(state.engine);
  } finally {
    globalThis.fetch = fetch;
    if (storage) Object.defineProperty(navigator, "storage", storage);
    else delete navigator.storage;
  }
});

test("model loader: converted cache reports a warm hit without fetching weights again", async () => {
  await withStorage(new FakeDir(), async () => {
    const f = cacheModel(), hits = [], m = loader(f.state, f.rangeFetch, { options: { prefetch: 0 }, hooks: { onCacheHit: (s) => hits.push(s) } });
    await m.loadShard(f.opts, f.engines);
    eq(f.state.G.entryCache.stats.write, 1); eq(hits, []);
    await m.loadShard(f.opts, f.engines);
    eq(f.fetched(), 1); eq(f.state.G.entryCache.stats.hit, 1);
    eq(hits.length, 1); ok(hits[0].includes("1 from this device"));
    ok(f.state.engine.weights.embed.data.every((v) => v === 1));
  });
});

test("model loader: wcacheverify detects a corrupt payload and reloads it", async () => {
  for (const verify of [false, true]) await withStorage(new FakeDir(), async () => {
    const f = cacheModel();
    await loader(f.state, f.rangeFetch, { options: { prefetch: 0 } }).loadShard(f.opts, f.engines);
    const file = await f.state.G.entryCache.dir.getFileHandle(entryFile(GGML_EMBED));
    new DataView(file.data.buffer).setFloat32(HDR, 9, true); // valid file layout, wrong payload hash
    await loader(f.state, f.rangeFetch, { options: { prefetch: 0, wcacheVerify: verify } }).loadShard(f.opts, f.engines);
    eq(f.state.engine.weights.embed.data[0], verify ? 1 : 9);
    eq(f.fetched(), verify ? 2 : 1);
    eq(f.state.G.entryCache.stats.bad, verify ? 1 : 0);
  });
});

test("model loader: converted cache flush finishes before engine construction", async () => {
  const root = new FakeDir(), release = deferred(), entered = deferred();
  let closed = false, built = false, wrapped = false;
  root.beforeClose = async (name) => { if (name.includes(".bin")) { await release.promise; closed = true; } };
  await withStorage(root, async () => {
    const f = cacheModel(), m = loader(f.state, f.rangeFetch, { options: { prefetch: 0 } });
    const loading = m.loadShard({ ...f.opts, onProgress() {
      if (wrapped) return;
      wrapped = true;
      const cache = f.state.G.entryCache, flush = cache.flush.bind(cache);
      cache.flush = () => { entered.resolve("flush"); return flush(); };
    } }, { DenseEngine: { async create(opts) {
      built = true; entered.resolve("engine");
      ok(closed, "engine constructed before the pending cache write finished");
      return opts;
    } } }).then(() => null, (error) => error);
    try {
      eq(await Promise.race([entered.promise, loading]), "flush");
      ok(!built, "engine must wait for cache flush");
      ok(f.state.G.entryCache.pending.size > 0, "a real cache write must still be pending");
    } finally { release.resolve(); await loading; }
    const error = await loading;
    if (error) throw error;
    ok(closed && built); eq(f.state.G.entryCache.stats.write, 1);
  });
});

test("model loader: GPU stream awaits room pacing before opening a range", async () => {
  await withGPU(async (device) => {
    const { state, engines, opts } = dense(), paced = deferred(), release = deferred(), requests = [];
    state.device = device;
    const m = loader(state, async (...args) => { requests.push(args); return new Response(new Uint8Array(18)); });
    await m.loadShard({ ...opts, hasEmbed: false, pace() { paced.resolve(); return release.promise; } }, engines);
    const pending = state.G.streamEntry({ name: "matrix", ggmlType: GGML_Q4_0, shape: [1, 32], nElems: 32, byteOffset: 64, byteLength: 18 });
    try { eq(await Promise.race([paced.promise.then(() => "paced"), pending.then(() => "streamed")]), "paced"); eq(requests, []); }
    finally { release.resolve(); await pending; }
    eq(requests, [[MODELS[opts.modelKey].gguf, 64, 81]]);
    ok((await pending).gpu);
  });
});

test("model loader: GPU stream propagates a room pacing rejection without fetching", async () => {
  await withGPU(async (device) => {
    const { state, engines, opts } = dense(), requests = [];
    state.device = device;
    const m = loader(state, async (...args) => { requests.push(args); return new Response(new Uint8Array(18)); });
    await m.loadShard({ ...opts, hasEmbed: false, pace: async () => { throw new Error("stream stopped"); } }, engines);
    await rejects(() => state.G.streamEntry({ name: "matrix", ggmlType: GGML_Q4_0, shape: [1, 32], nElems: 32,
      byteOffset: 64, byteLength: 18 }), "stream stopped");
    eq(requests, []);
  });
});

test("model loader: a short GPU stream evicts the actual shard's range and retries without cache", async () => {
  await withGPU(async (device) => {
    const { state, engines, opts } = dense(), requests = [], deleted = [];
    state.device = device;
    const m = loader(state, async (url, lo, hi, noCache = false) => {
      requests.push([url, lo, hi, noCache]);
      return new Response(new Uint8Array(noCache ? 18 : 0));
    }, { getWeightCache: async () => ({ delete: async (key) => { deleted.push(key); } }) });
    await m.loadShard({ ...opts, hasEmbed: false }, engines);
    await state.G.streamEntry({ name: "matrix", url: "second-shard", ggmlType: GGML_Q4_0, shape: [1, 32], nElems: 32, byteOffset: 64, byteLength: 18 });
    eq(requests, [["second-shard", 64, 81, false], ["second-shard", 64, 81, true]]);
    eq(deleted.length, 1);
    ok(deleted[0].includes("second-shard") && deleted[0].endsWith("/64-81"));
  });
});

async function withDraft(fn) {
  const f = dense(), requests = [], file = header([GGML_EMBED, GGML_FINAL_NORM], { payload: true });
  f.state.engine = { maxSeq: 8192 };
  const fetch = globalThis.fetch;
  globalThis.fetch = async (url) => { eq(url, MODELS["qwen3-0.6b"].cfg); return new Response(JSON.stringify({ num_hidden_layers: 0 })); };
  const rangeFetch = async (url, lo, hi) => {
    eq(url, MODELS["qwen3-0.6b"].gguf); requests.push([lo, hi]); return new Response(file.slice(lo, hi + 1));
  };
  try { await fn({ ...f, requests, m: loader(f.state, rangeFetch) }); }
  finally { globalThis.fetch = fetch; }
}

test("model loader: draft construction uses the loaded context without replacing the main engine", async () => {
  await withDraft(async ({ state, m }) => {
    const engine = state.engine;
    let paced = 0;
    const draft = await m.loadDraft({ modelKey: "qwen3-0.6b", pace: () => { paced++; } }, { DenseEngine: { async create(opts) { return opts; } } });
    eq(paced, 2);
    eq(draft.maxSeq, 8192);
    eq(draft.coopWG, 64);
    eq(draft.coopRows, 4);
    eq(draft.layerRange, [0, 0]);
    eq([...draft.weights.embed.data], [1]); eq([...draft.weights.finalNorm.data], [2]);
    ok(draft.hasEmbed && draft.hasHead);
    ok(draft.device === state.device && state.engine === engine);
  });
});

test("model loader: an independent draft load waits for its explicit pacer", async () => {
  await withDraft(async ({ state, m, requests }) => {
    const engine = state.engine, entered = deferred(), release = deferred();
    let built = false;
    const pending = m.loadDraft({ modelKey: "qwen3-0.6b", pace() { entered.resolve("paced"); return release.promise; } },
      { DenseEngine: { async create(opts) { built = true; return opts; } } });
    try {
      eq(await Promise.race([entered.promise, pending.then(() => "built")]), "paced");
      eq(requests.length, 1); // header only; no tensor range may open yet
      ok(!built);
    } finally { release.resolve(); await pending; }
    eq(requests.length, 3); ok(built && state.engine === engine);
  });
});

test("model loader: a draft pacing rejection stops tensor fetches and engine construction", async () => {
  await withDraft(async ({ state, m, requests }) => {
    const engine = state.engine;
    let built = false;
    await rejects(() => m.loadDraft({ modelKey: "qwen3-0.6b", pace: async () => { throw new Error("draft stopped"); } },
      { DenseEngine: { async create() { built = true; } } }), "draft stopped");
    eq(requests.length, 1); ok(!built && state.engine === engine);
  });
});

test("model loader: draft pacing never inherits the preceding shard's pacer", async () => {
  await withDraft(async ({ state, m, opts, engines }) => {
    await m.loadShard({ ...opts, hasEmbed: false, ctx: 8192, pace() { throw new Error("stale shard pacer"); } }, engines);
    const engine = state.engine;
    let paced = 0;
    await m.loadDraft({ modelKey: "qwen3-0.6b", pace: () => { paced++; } }, engines);
    eq(paced, 2); ok(state.engine === engine);
    await m.loadDraft({ modelKey: "qwen3-0.6b" }, engines);
    eq(paced, 2); ok(state.engine === engine);
  });
});
