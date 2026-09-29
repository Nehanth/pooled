// Bookkeeping for the on-disk weight cache (Cache API), per model: what each model's cached ranges
// add up to, and deleting one model's ranges while leaving the others. DOM-free so it can be unit
// tested against a stub cache.
//
// Every entry's key is PREFIX + encodeURIComponent(weights url) + "/" + lo + "-" + hi, and its
// "x-swarm-len" header is the byte count of a complete entry (see rangeFetch in room.js). The
// "swarmllm" names are from before the rename to Pooled and stay so existing caches keep working.
export const CACHE_NAME = "swarmllm-weights-v1";
export const PREFIX = "https://weights.swarmllm.ai/";

export const cacheKey = (url, lo, hi) => PREFIX + encodeURIComponent(url) + "/" + lo + "-" + hi;

// A key (string or Request url) -> { url, lo, hi }, or null for anything that is not a weight range.
export function parseKey(key) {
  if (typeof key !== "string" || !key.startsWith(PREFIX)) return null;
  const rest = key.slice(PREFIX.length), cut = rest.lastIndexOf("/");
  if (cut < 1) return null;
  const m = /^(\d+)-(\d+)$/.exec(rest.slice(cut + 1));
  if (!m) return null;
  let url;
  try { url = decodeURIComponent(rest.slice(0, cut)); } catch { return null; }
  return { url, lo: +m[1], hi: +m[2] };
}

// The weights urls of a model (a GGUF file, or a safetensors file for the small test model).
const weightUrls = (m) => [m?.gguf, m?.st].filter(Boolean);

// A name for weights no catalogue model points at any more (an old URL, a ?dev model): the file name.
const fileName = (url) => { try { return decodeURIComponent(new URL(url).pathname.split("/").pop()) || url; } catch { return url; } };

// entries: [{ key, bytes }] with bytes from each entry's x-swarm-len (0 if missing).
// -> [{ url, model, label, bytes, ranges }], biggest first. model is the catalogue key or null.
export function groupByModel(entries, models = {}) {
  const byUrl = new Map();
  for (const { key, bytes } of entries) {
    const k = parseKey(key); if (!k) continue;
    let g = byUrl.get(k.url);
    if (!g) {
      const model = Object.keys(models).find((id) => weightUrls(models[id]).includes(k.url)) || null;
      g = { url: k.url, model, label: model ? models[model].label : fileName(k.url), bytes: 0, ranges: 0 };
      byUrl.set(k.url, g);
    }
    g.bytes += Math.max(0, +bytes || 0); g.ranges++;
  }
  return [...byUrl.values()].sort((a, b) => b.bytes - a.bytes || a.label.localeCompare(b.label));
}

// Read the cache's keys and sizes. Only headers are read (Response bodies are never touched).
export async function cachedEntries(cache) {
  const out = [];
  for (const req of await cache.keys()) {
    const key = typeof req === "string" ? req : req.url;
    const hit = await cache.match(req).catch(() => null);
    out.push({ key, bytes: +(hit?.headers.get("x-swarm-len") || 0) });
  }
  return out;
}

export const cachedModels = async (cache, models) => groupByModel(await cachedEntries(cache), models);

// Delete every cached range of one weights url; the other models' ranges stay.
// -> { ranges, bytes } that were removed.
export async function deleteModel(cache, url) {
  let ranges = 0, bytes = 0;
  for (const req of await cache.keys()) {
    const key = typeof req === "string" ? req : req.url;
    if (parseKey(key)?.url !== url) continue;
    const hit = await cache.match(req).catch(() => null);
    if (await cache.delete(req)) { ranges++; bytes += +(hit?.headers.get("x-swarm-len") || 0); }
  }
  return { ranges, bytes };
}
