// Direct single-device inference: no PeerJS, rooms, inference service or prompt network requests.
// Reuses the room's loader and engine adapter; this module is loaded only after a user submits.
import { MODELS } from '../room/models.js';
import { createModelLoader } from '../room/model-loader.js';
import { engineModel } from '../harness/engine-model.js';
import { CACHE_NAME, cacheKey } from '../room/weightcache.js';
export const LOCAL_MODEL = 'qwen3-0.6b';
export async function loadLocal({ signal, onStatus = () => {}, onProgress = () => {} } = {}) {
  if (!navigator.gpu) throw new Error('Local inference needs WebGPU. Use Chrome or Edge on a supported desktop, or Safari 26. You can also join a trusted room.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('This browser could not access a WebGPU device. Try another supported browser or join a trusted room.');
  signal?.throwIfAborted();
  const device = await adapter.requestDevice({ requiredLimits: {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  } });
  const state = { device };
  let gpuError = '';
  device.addEventListener('uncapturederror', e => { gpuError ||= e.error?.message || 'GPU execution failed'; });
  const checkGPU = () => { if (gpuError) throw new Error('This device could not run the local model. Try another supported device or a trusted room.'); };
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); } catch { /* storage is optional */ }
  const checkedFetch = async (url, opts = {}) => {
    signal?.throwIfAborted();
    const r = await fetch(url, { ...opts, signal, credentials: 'omit' });
    if (!r.ok) throw new Error('Model download failed. Check your connection and try again.');
    return r;
  };
  const rangeFetch = localRangeFetcher({ cache, signal, fetchImpl: checkedFetch });
  try {
    onStatus('Preparing local inference…');
    const engines = await import('../engine/engine.js');
    const M = MODELS[LOCAL_MODEL];
    const [cfg, tok] = await Promise.all([checkedFetch(M.cfg).then(r => r.json()), checkedFetch(M.tok).then(r => r.json())]);
    state.cfg = cfg; state.tok = engines.makeTokenizer(tok);
    const loader = createModelLoader({ state, rangeFetch, getWeightCache: async () => cache,
      getMeta: () => ({}), getEnginePreset: () => ({}), options: { wcache: false, prefetch: 0 }, hooks: { onStatus } });
    await loader.loadShard({ modelKey: LOCAL_MODEL, range: [0, cfg.num_hidden_layers], hasEmbed: true, hasHead: true,
      ctx: 2048, onProgress, pace: async () => { signal?.throwIfAborted(); } }, engines);
    signal?.throwIfAborted();
    checkGPU();
    const model = engineModel(state.engine, state.tok, { maxNew: 384, spec: false, pin: false });
    device.lost.then(() => { model.lost = true; });
    return { async *generate(args) {
      checkGPU();
      if (model.lost) throw new Error('The graphics device was disconnected. Reload to prepare Local again.');
      for await (const delta of model.generate(args)) { checkGPU(); yield delta; }
      checkGPU();
    }, dispose: () => device.destroy() };
  } catch (e) { device.destroy(); throw e; }
}

// Kept separate to test truncated responses, cancellation and cache failure without a GPU.
export function localRangeFetcher({ cache = null, signal, fetchImpl = fetch } = {}) {
  return async (url, lo, hi, noCache = false) => {
    signal?.throwIfAborted();
    if (!Number.isSafeInteger(lo) || !Number.isSafeInteger(hi) || lo < 0 || hi < lo) throw new Error('Invalid model range.');
    const key = cacheKey(url, lo, hi);
    let hit = null;
    try { if (!noCache) hit = await cache?.match(key); } catch { /* optional storage */ }
    if (hit) {
      const data = await hit.arrayBuffer();
      if (data.byteLength === hi - lo + 1) return new Response(data);
    }
    const r = await fetchImpl(url, { headers: { Range: `bytes=${lo}-${hi}` }, signal, credentials: 'omit' });
    if (r.status !== 206) throw new Error('The model host did not honor a partial download. Try again later.');
    const data = await r.arrayBuffer();
    if (data.byteLength !== hi - lo + 1) throw new Error('Model download was incomplete. Try again.');
    signal?.throwIfAborted();
    const saved = new Response(data, { headers: { 'Content-Type': 'application/octet-stream' } });
    try { await cache?.put(key, saved.clone()); } catch { /* quota: inference can continue */ }
    return saved;
  };
}
