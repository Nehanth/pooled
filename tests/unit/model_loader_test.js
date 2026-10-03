// Real header/weight loading with in-memory ranges and recording engine constructors.
// Browser integration tests cover GPU construction and room lifecycle around this boundary.
import { createModelLoader } from "../../room/model-loader.js";
import { GGML_EMBED, GGML_FINAL_NORM, GGML_Q4_0, ggmlLayerNames } from "../../engine/gguf.js";
import { MODELS, MAX_SEQ } from "../../room/models.js";

const eq = (a, b) => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${JSON.stringify(a)} != ${JSON.stringify(b)}`); };
const ok = (v, message = "assertion failed") => { if (!v) throw new Error(message); };
async function rejects(fn, message) {
  try { await fn(); } catch (e) { ok(String(e).includes(message), String(e)); return; }
  throw new Error("expected rejection: " + message);
}

function test(name, fn) {
  Deno.test(name, async () => {
    const location = Object.getOwnPropertyDescriptor(globalThis, "location");
    Object.defineProperty(globalThis, "location", { configurable: true, value: { search: "?wcache=0&prefetch=0" } });
    try { await fn(); }
    finally {
      if (location) Object.defineProperty(globalThis, "location", location);
      else delete globalThis.location;
    }
  });
}

function loader(state, rangeFetch, { getMeta = () => ({}), getWeightCache = async () => null } = {}) {
  return createModelLoader({ state, rangeFetch, getWeightCache, getMeta,
    hooks: { onStatus() {}, crumb() {}, onCacheHit() {}, onModelLoaded() {} } });
}

// Minimal valid GGUF index: f32 tensors and a tokenizer marker, no tensor payload.
function header(name) {
  const names = Array.isArray(name) ? name : [name];
  const bytes = [], enc = new TextEncoder();
  const u32 = (n) => { for (let i = 0; i < 4; i++) bytes.push(n >>> (8 * i) & 255); };
  const u64 = (n) => { u32(n); u32(0); };
  const str = (s) => { const b = enc.encode(s); u64(b.length); bytes.push(...b); };
  u32(0x46554747); u32(3); u64(names.length); u64(1);
  str("tokenizer.ggml.model"); u32(8); str("test");
  for (const [i, n] of names.entries()) { str(n); u32(1); u64(1); u32(0); u64(i * 4); }
  return new Uint8Array(bytes);
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
  location.search = "?wcache=0";
  const names = Object.values(ggmlLayerNames(0));
  const { state, engines, opts } = dense(names), offsets = [];
  let meta = { phone: false };
  const m = loader(state, async (_url, lo) => { offsets.push(lo); return new Response(new Float32Array([1])); }, { getMeta: () => meta });
  meta = { phone: true };
  await m.loadShard({ ...opts, range: [0, 1], hasEmbed: false }, engines);
  eq(offsets.slice(0, 2), [4, 0]); // one range ahead, then the requested range
  eq(offsets.length, names.length);
  eq(new Set(offsets).size, names.length);
});

test("model loader: prefetch fetches each tensor once when file and model order differ", async () => {
  location.search = "?wcache=0";
  for (const phone of [true, false]) {
    const names = Object.values(ggmlLayerNames(0)), { state, engines, opts } = dense(names), offsets = [];
    // Model order visits the second half of the file before its first half.
    for (const [i, name] of names.entries()) state.G.tensors[name].byteOffset = ((i + 6) % names.length) * 4;
    const m = loader(state, async (_url, lo) => { offsets.push(lo); return new Response(new Float32Array([1])); }, { getMeta: () => ({ phone }) });
    await m.loadShard({ ...opts, range: [0, 1], hasEmbed: false }, engines);
    eq(offsets.length, names.length);
    eq(new Set(offsets).size, names.length);
  }
});

test("model loader: the next load cancels prefetched bodies left unread by a failed load", async () => {
  location.search = "?wcache=0&prefetch=2";
  const { state, engines, opts } = dense(Object.values(ggmlLayerNames(0))), cancelled = [];
  const m = loader(state, async (_url, lo) => new Response(new ReadableStream({
    start(c) { c.enqueue(new Uint8Array(new Float32Array([1]).buffer)); },
    pull(c) { c.close(); },
    cancel() { cancelled.push(lo); },
  })));
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

test("model loader: converted cache hashes the original converter URL and tolerates unavailable storage", async () => {
  location.search = "?prefetch=0";
  const { state, engines, opts } = dense(), urls = [];
  const fetch = globalThis.fetch, storage = Object.getOwnPropertyDescriptor(navigator, "storage");
  Object.defineProperty(navigator, "storage", { configurable: true,
    value: { getDirectory() { throw new Error("storage unavailable"); } } });
  globalThis.fetch = async (url) => { urls.push(url); return new Response("converter source"); };
  try {
    await loader(state, async () => new Response(new Float32Array([1]))).loadShard(opts, engines);
    eq(urls, [new URL("../../engine/gguf.js", import.meta.url).href]);
    eq(state.G.entryCache, null);
    ok(state.engine);
  } finally {
    globalThis.fetch = fetch;
    if (storage) Object.defineProperty(navigator, "storage", storage);
    else delete navigator.storage;
  }
});

test("model loader: a short GPU stream evicts the actual shard's range and retries without cache", async () => {
  const { state, engines, opts } = dense(), requests = [], deleted = [];
  const usage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { STORAGE: 128, COPY_DST: 8, COPY_SRC: 4 } });
  state.device = { createBuffer: ({ size }) => ({ size, destroy() {} }), queue: { writeBuffer() {} },
    pushErrorScope() {}, popErrorScope: async () => null };
  try {
    const m = loader(state, async (url, lo, hi, noCache = false) => {
      requests.push([url, lo, hi, noCache]);
      return new Response(new Uint8Array(noCache ? 18 : 0));
    }, { getWeightCache: async () => ({ delete: async (key) => { deleted.push(key); } }) });
    await m.loadShard({ ...opts, hasEmbed: false }, engines);
    await state.G.streamEntry({ name: "matrix", url: "second-shard", ggmlType: GGML_Q4_0, shape: [1, 32], nElems: 32, byteOffset: 64, byteLength: 18 });
    eq(requests, [["second-shard", 64, 81, false], ["second-shard", 64, 81, true]]);
    eq(deleted.length, 1);
    ok(deleted[0].includes("second-shard") && deleted[0].endsWith("/64-81"));
  } finally {
    if (usage) Object.defineProperty(globalThis, "GPUBufferUsage", usage);
    else delete globalThis.GPUBufferUsage;
  }
});

test("model loader: draft construction uses the loaded context without replacing the main engine", async () => {
  const engine = { maxSeq: 8192 }, state = { engine, device: {}, tune: { wg: 64, rows: 4 } };
  const fetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({ num_hidden_layers: 0 }));
  try {
    const m = loader(state, async (_url, lo) => new Response(lo === 0 ? header([GGML_EMBED, GGML_FINAL_NORM]) : new Float32Array([1])));
    const draft = await m.loadDraft({ cfg: "config", gguf: "draft" }, { async create(opts) { return opts; } });
    eq(draft.maxSeq, 8192);
    eq(draft.layerRange, [0, 0]);
    ok(draft.hasEmbed && draft.hasHead);
    ok(draft.device === state.device && state.engine === engine);
  } finally { globalThis.fetch = fetch; }
});
