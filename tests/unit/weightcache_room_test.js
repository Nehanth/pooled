// room/weightcache.js: the per-model view of the browser weight cache behind the room menu's
// "delete one model" rows. Runs against a stub of the Cache API (keys/match/delete) holding
// Request-like { url } keys and Response-like entries with an x-swarm-len header.
import { CACHE_NAME, PREFIX, cacheKey, parseKey, groupByModel, cachedEntries, cachedModels, deleteModel } from "../../room/weightcache.js";
import { MODELS } from "../../room/models.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

const G27 = MODELS["qwen3.8-27b"].gguf, MOE = MODELS["qwen3.6-35b-moe"].gguf, ST = MODELS["smollm-135m"].st;

function stubCache(entries) {
  const m = new Map(entries.map(([key, len]) => [key, len]));
  return {
    map: m,
    async keys() { return [...m.keys()].map((url) => ({ url })); },
    async match(req) {
      const k = typeof req === "string" ? req : req.url;
      if (!m.has(k)) return undefined;
      const len = m.get(k);
      return { headers: { get: (h) => h === "x-swarm-len" && len != null ? String(len) : null } };
    },
    async delete(req) { return m.delete(typeof req === "string" ? req : req.url); },
  };
}

Deno.test("cache names stay the pre-rename ones (existing downloads keep working)", () => {
  eq(CACHE_NAME, "swarmllm-weights-v1");
  eq(PREFIX, "https://weights.swarmllm.ai/");
  eq(cacheKey("https://h/a b.gguf", 0, 99), "https://weights.swarmllm.ai/https%3A%2F%2Fh%2Fa%20b.gguf/0-99");
});

Deno.test("parseKey: round trip and rejects", () => {
  eq(parseKey(cacheKey(G27, 12, 3456)), { url: G27, lo: 12, hi: 3456 });
  eq(parseKey(cacheKey("https://h/x/y/z.gguf?rev=1", 0, 0)), { url: "https://h/x/y/z.gguf?rev=1", lo: 0, hi: 0 });
  for (const bad of [null, 5, "", "https://other/" + encodeURIComponent(G27) + "/0-1", PREFIX, PREFIX + "0-1",
    PREFIX + encodeURIComponent(G27) + "/0-", PREFIX + encodeURIComponent(G27) + "/a-b", PREFIX + "%E0%A4%A/0-1"])
    eq(parseKey(bad), null, "bad key " + bad);
});

Deno.test("groupByModel: sums per model, names catalogue models, biggest first", () => {
  const entries = [
    { key: cacheKey(G27, 0, 99), bytes: 100 },
    { key: cacheKey(MOE, 0, 999), bytes: 1000 },
    { key: cacheKey(G27, 100, 199), bytes: 100 },
    { key: cacheKey(ST, 0, 9), bytes: 10 },
    { key: cacheKey("https://h/old/Gone-Q4.gguf", 0, 49), bytes: 50 },
    { key: cacheKey(MOE, 1000, 1499), bytes: null },   // no length header: counted as 0 bytes, still listed
    { key: "https://unrelated/thing", bytes: 7 },       // not a weight range: ignored
  ];
  const g = groupByModel(entries, MODELS);
  eq(g.map((x) => [x.model, x.bytes, x.ranges]), [
    ["qwen3.6-35b-moe", 1000, 2], ["qwen3.8-27b", 200, 2], [null, 50, 1], ["smollm-135m", 10, 1]]);
  eq(g[0].label, MODELS["qwen3.6-35b-moe"].label);
  eq(g[0].url, MOE);
  eq(g[2].label, "Gone-Q4.gguf", "unknown weights are named by file");
  eq(groupByModel([], MODELS), []);
});

Deno.test("cachedModels reads sizes from headers", async () => {
  const c = stubCache([[cacheKey(G27, 0, 99), 100], [cacheKey(G27, 100, 149), 50], [cacheKey(MOE, 0, 9), 10]]);
  eq((await cachedEntries(c)).length, 3);
  const g = await cachedModels(c, MODELS);
  eq(g.map((x) => [x.model, x.bytes]), [["qwen3.8-27b", 150], ["qwen3.6-35b-moe", 10]]);
});

Deno.test("deleteModel removes one model and keeps the others", async () => {
  const c = stubCache([
    [cacheKey(G27, 0, 99), 100], [cacheKey(MOE, 0, 999), 1000], [cacheKey(G27, 100, 199), 100],
    [cacheKey(MOE, 1000, 1999), 1000], ["https://unrelated/thing", 7]]);
  const r = await deleteModel(c, G27);
  eq(r, { ranges: 2, bytes: 200 });
  const left = await cachedModels(c, MODELS);
  eq(left.map((x) => [x.model, x.bytes, x.ranges]), [["qwen3.6-35b-moe", 2000, 2]]);
  ok(c.map.has("https://unrelated/thing"), "entries that are not weight ranges are left alone");
  eq(await deleteModel(c, G27), { ranges: 0, bytes: 0 }, "a second delete finds nothing");
  eq(await deleteModel(c, "https://h/not-cached.gguf"), { ranges: 0, bytes: 0 });
});

Deno.test("deleteModel does not touch a url that only shares a prefix", async () => {
  const c = stubCache([[cacheKey("https://h/m.gguf", 0, 9), 10], [cacheKey("https://h/m.gguf.part2", 0, 9), 10]]);
  eq(await deleteModel(c, "https://h/m.gguf"), { ranges: 1, bytes: 10 });
  eq([...c.map.keys()], [cacheKey("https://h/m.gguf.part2", 0, 9)]);
});
