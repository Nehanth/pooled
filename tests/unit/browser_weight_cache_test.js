// Browser converted-weights cache (room/weightcache.js, issue #77): cache keys, invalidation on a
// revision / header / converter change, the entry format, quota handling, and a second load through
// engine/gguf.js ggufEntry that skips the conversion. OPFS is an in-memory fake here.
//   deno test --allow-read --no-check tests/unit/browser_weight_cache_test.js
import { assert, assertEquals, assertNotEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { ggufEntry, convertEntry, GGML_Q8_0, GGML_BF16, GGML_F32 } from "../../engine/gguf.js";
import { LOADER_VERSION } from "../weight_cache.js";
import {
  WCACHE_DIR, HDR, revisionOf, modelOf, headerFingerprint, cacheDirName, converterVersion, worthCaching, quotaAllows,
  encodeEntry, decodeEntry, entryFile, BrowserWeightCache, convertedBytes, clearConverted, attachBrowserWeightCache,
} from "../../room/weightcache.js";

// ---- a small in-memory OPFS: directories, files, writables, move ----
const notFound = (n) => Object.assign(new Error(`${n} not found`), { name: "NotFoundError" });
class FakeFile {
  constructor(parent, name) { this.kind = "file"; this.parent = parent; this.name = name; this.data = new Uint8Array(0); }
  getFile() { const d = this.data; return Promise.resolve({ size: d.byteLength, arrayBuffer: async () => d.slice().buffer, text: async () => new TextDecoder().decode(d) }); }
  createWritable() {
    const chunks = [], root = this.parent.root;
    return Promise.resolve({
      write: async (p) => {
        if (root.failWrites) throw Object.assign(new Error("quota"), { name: "QuotaExceededError" });
        chunks.push(new Uint8Array(p.buffer ? p.buffer.slice(p.byteOffset, p.byteOffset + p.byteLength) : p));
      },
      close: async () => {   // like Chrome: the file changes only on close
        const n = chunks.reduce((a, c) => a + c.byteLength, 0), out = new Uint8Array(n);
        let o = 0; for (const c of chunks) { out.set(c, o); o += c.byteLength; }
        this.data = out;
      },
      abort: async () => {},
    });
  }
  async move(name) { this.parent.children.delete(this.name); this.name = name; this.parent.children.set(name, this); }
}
class FakeDir {
  constructor(name = "", root = null) { this.kind = "directory"; this.name = name; this.children = new Map(); this.root = root || this; }
  async getDirectoryHandle(n, { create = false } = {}) {
    let h = this.children.get(n);
    if (!h) { if (!create) throw notFound(n); h = new FakeDir(n, this.root); this.children.set(n, h); }
    return h;
  }
  async getFileHandle(n, { create = false } = {}) {
    let h = this.children.get(n);
    if (!h) { if (!create) throw notFound(n); h = new FakeFile(this, n); this.children.set(n, h); }
    return h;
  }
  async removeEntry(n) { if (!this.children.delete(n)) throw notFound(n); }
  async *entries() { for (const e of [...this.children.entries()]) yield e; }
}
const big = () => ({ quota: 100 * 2 ** 30, usage: 0 });

// ---- synthetic tensors: one Q8_0 matrix (repack) and one BF16 matrix (-> f32) ----
function makeModel({ offsetShift = 0 } = {}) {
  const rows = 256, cols = 1024, n = rows * cols;
  const q8 = { name: "blk.0.ffn_down.weight", ggmlType: GGML_Q8_0, shape: [rows, cols], nElems: n, byteLength: (n / 32) * 34, offset: 0 };
  const bf = { name: "token_embd.weight", ggmlType: GGML_BF16, shape: [rows, cols], nElems: n, byteLength: n * 2, offset: q8.byteLength + offsetShift };
  const norm = { name: "blk.0.attn_norm.weight", ggmlType: GGML_F32, shape: [cols], nElems: cols, byteLength: cols * 4, offset: bf.offset + bf.byteLength };
  const dataStart = 4096, tensors = {};
  for (const t of [q8, bf, norm]) tensors[t.name] = { ...t, byteOffset: dataStart + t.offset };
  const file = new Uint8Array(dataStart + norm.offset + norm.byteLength);
  let x = 12345;
  for (let i = 0; i < file.length; i++) { x = (Math.imul(x, 1103515245) + 12345) >>> 0; file[i] = x >>> 24; }
  // keep the Q8_0 scales and BF16 values finite: f16 scale / bf16 high byte below the exponent max
  for (let b = 0; b < n / 32; b++) file[tensors[q8.name].byteOffset + b * 34 + 1] &= 0x3b;
  for (let i = 0; i < n; i++) file[tensors[bf.name].byteOffset + i * 2 + 1] &= 0x3f;
  const G = { meta: { "general.architecture": "qwen3", "qwen3.block_count": 1, "tokenizer.ggml.tokens": ["a", "b"] }, tensors, dataStart };
  let reads = 0;
  const bytesOf = async (info) => { reads++; return file.slice(info.byteOffset, info.byteOffset + info.byteLength); };
  return { G, bytesOf, reads: () => reads, file };
}
const same = (a, b) => {
  const u = (v) => new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
  if (a.kind !== b.kind) return false;
  const pairs = a.kind === "f32" ? [[a.data, b.data]] : [[a.qs, b.qs], [a.scales, b.scales]];
  return pairs.every(([x, y]) => { const p = u(x), q = u(y); return p.length === q.length && p.every((v, i) => v === q[i]); });
};
const URL_MAIN = "https://huggingface.co/org/repo/resolve/main/Model-Q4_0.gguf";
const URL_PIN = "https://huggingface.co/org/repo/resolve/0123abcd/Model-Q4_0.gguf";
const open = (root, G, extra = {}) => BrowserWeightCache.open(root, { url: URL_PIN, G, converter: "e1-test", estimate: async () => big(), ...extra });

Deno.test("revisionOf reads the pinned revision out of a resolve URL", () => {
  assertEquals(revisionOf(URL_PIN), "0123abcd");
  assertEquals(revisionOf(URL_MAIN), "main");
  assertEquals(revisionOf("https://huggingface.co/org/repo/resolve/refs%2Fpr%2F3/m.gguf"), "refs/pr/3");
  assertEquals(revisionOf("https://example.com/models/m.gguf"), "");
  assertEquals(revisionOf(undefined), "");
  assertEquals(modelOf(URL_PIN), modelOf(URL_MAIN));
  assertEquals(modelOf(URL_MAIN), "https://huggingface.co/org/repo/resolve/*/Model-Q4_0.gguf");
  assertNotEquals(modelOf(URL_MAIN), modelOf(URL_MAIN.replace("Q4_0", "Q8_0")));
});

Deno.test("cache key: stable for the same inputs, new for any change", async () => {
  const { G } = makeModel();
  const base = { url: URL_PIN, revision: "0123abcd", fingerprint: headerFingerprint(G), converter: "e1-aaaa" };
  const k = await cacheDirName(base);
  assertEquals(k, await cacheDirName({ ...base }));
  assert(/^Model-Q4_0-[0-9a-f]{16}$/.test(k), k);
  for (const change of [{ url: URL_MAIN }, { revision: "fedcba98" }, { converter: "e1-bbbb" }, { variant: "x" },
    { fingerprint: headerFingerprint(makeModel({ offsetShift: 32 }).G) }])
    assertNotEquals(await cacheDirName({ ...base, ...change }), k, JSON.stringify(Object.keys(change)));
});

Deno.test("header fingerprint ignores the tokenizer, sees tensor and metadata changes", () => {
  const { G } = makeModel();
  const f = headerFingerprint(G);
  const noTok = { ...G, meta: { ...G.meta } }; delete noTok.meta["tokenizer.ggml.tokens"];
  assertEquals(headerFingerprint(noTok), f);
  assertNotEquals(headerFingerprint({ ...G, meta: { ...G.meta, "qwen3.block_count": 2 } }), f);
  const retyped = { ...G, tensors: { ...G.tensors, "token_embd.weight": { ...G.tensors["token_embd.weight"], ggmlType: GGML_F32 } } };
  assertNotEquals(headerFingerprint(retyped), f);
  // table order does not matter
  assertEquals(headerFingerprint({ ...G, tensors: Object.fromEntries(Object.entries(G.tensors).reverse()) }), f);
});

Deno.test("converter version matches the CLI cache's LOADER_VERSION for the same engine/gguf.js", async () => {
  const src = await Deno.readFile(new URL("../../engine/gguf.js", import.meta.url));
  assertEquals(await converterVersion(src), LOADER_VERSION);
  const edited = new Uint8Array([...src, 10]);
  assertNotEquals(await converterVersion(edited), LOADER_VERSION);
});

Deno.test("what gets cached, and the quota reserve", () => {
  assert(worthCaching({ ggmlType: GGML_BF16, byteLength: 2 ** 20 }));
  assert(!worthCaching({ ggmlType: GGML_BF16, byteLength: 1024 }));     // small: converting is cheaper
  assert(!worthCaching({ ggmlType: GGML_F32, byteLength: 2 ** 30 }));   // plain copy, nothing saved
  const GB = 2 ** 30;
  assert(quotaAllows({ quota: 100 * GB, usage: 10 * GB }, GB));
  assert(!quotaAllows({ quota: 100 * GB, usage: 80 * GB }, 10 * GB));   // would eat into the 15 GB reserve
  assert(!quotaAllows({ quota: 4 * GB, usage: 2.5 * GB }, 0.6 * GB));   // small quota: 1 GB reserve
  assert(!quotaAllows(null, 1));
  assert(!quotaAllows({ quota: 0, usage: 0 }, 1));
});

Deno.test("entry format: round trip for q8 / f32, rejects the wrong tensor, a short file and a bad payload", async () => {
  const { G, bytesOf } = makeModel();
  for (const name of ["blk.0.ffn_down.weight", "token_embd.weight"]) {
    const info = G.tensors[name], e = convertEntry(info, await bytesOf(info));
    const enc = encodeEntry(info, e);
    assert(enc, name);
    const buf = new Uint8Array(enc.bytes); let o = 0;
    for (const p of enc.parts) { buf.set(new Uint8Array(p.buffer, p.byteOffset, p.byteLength), o); o += p.byteLength; }
    assertEquals(o, enc.bytes);
    assert(same(decodeEntry(info, buf.buffer, true), e), name);
    const other = { ...info, ggmlType: info.ggmlType === GGML_Q8_0 ? GGML_BF16 : GGML_Q8_0 };
    let threw = false; try { decodeEntry(other, buf.buffer); } catch { threw = true; } assert(threw, "wrong type");
    threw = false; try { decodeEntry(info, buf.slice(0, enc.bytes - 4).buffer); } catch { threw = true; } assert(threw, "short");
    const bad = buf.slice(); bad[HDR + 7] ^= 1;
    decodeEntry(info, bad.buffer, false);   // header-only check passes
    threw = false; try { decodeEntry(info, bad.buffer, true); } catch { threw = true; } assert(threw, "checksum");
  }
  assertEquals(encodeEntry(G.tensors["token_embd.weight"], { kind: "f32", data: new Float32Array(3) }), null);
  assertEquals(encodeEntry(G.tensors["token_embd.weight"], { gpu: {} }), null);
});

Deno.test("second load through ggufEntry takes the cached entries, no read and the same bytes", async () => {
  const root = new FakeDir();
  const m = makeModel();
  const G1 = { ...m.G, entryCache: await open(root, m.G) };
  const names = Object.keys(m.G.tensors);
  const fresh = {};
  for (const n of names) fresh[n] = await ggufEntry(G1, m.bytesOf, n, false);
  assertEquals(G1.entryCache.stats.write, 2);           // the f32 norm is not worth a file
  assertEquals(m.reads(), 3);

  const G2 = { ...m.G, entryCache: await open(root, m.G) };
  for (const n of names) assert(same(await ggufEntry(G2, m.bytesOf, n, false), fresh[n]), n);
  assertEquals(G2.entryCache.stats.hit, 2);
  assertEquals(m.reads(), 4);                           // only the norm was read again
  assert(G2.entryCache.summary().includes("2 from this device"));
});

Deno.test("invalidation: a new revision, header or converter drops the old entries", async () => {
  const root = new FakeDir();
  const m = makeModel();
  const c1 = await open(root, m.G);
  await ggufEntry({ ...m.G, entryCache: c1 }, m.bytesOf, "token_embd.weight", false);
  const top = await root.getDirectoryHandle(WCACHE_DIR);
  assertEquals(top.children.size, 1);

  // same URL, new header (a re-upload under the same revision)
  const m2 = makeModel({ offsetShift: 32 });
  const c2 = await open(root, m2.G);
  assertNotEquals(c2.meta.dir, c1.meta.dir);
  assertEquals([...top.children.keys()], [c2.meta.dir]);
  assertEquals(await c2.get(m2.G.tensors["token_embd.weight"]), null);

  // new converter version
  const c3 = await open(root, m2.G, { converter: "e1-other" });
  assertEquals([...top.children.keys()], [c3.meta.dir]);

  // a new pinned revision of the same file replaces it
  const c4 = await open(root, m2.G, { url: URL_MAIN, converter: "e1-other" });
  assertEquals([...top.children.keys()], [c4.meta.dir]);
  // another model keeps its own directory
  const other = URL_PIN.replace("Q4_0", "Q8_0");
  const c5 = await open(root, m2.G, { url: other, converter: "e1-other" });
  assertEquals(new Set(top.children.keys()), new Set([c4.meta.dir, c5.meta.dir]));
});

Deno.test("a damaged entry is dropped and converted fresh; leftover temp files are cleaned up", async () => {
  const root = new FakeDir();
  const m = makeModel();
  const c1 = await open(root, m.G);
  const info = m.G.tensors["token_embd.weight"];
  const e = await ggufEntry({ ...m.G, entryCache: c1 }, m.bytesOf, info.name, false);
  const dir = c1.dir, f = dir.children.get(entryFile(info.name));
  f.data = f.data.slice(0, f.data.byteLength - 8);          // torn write
  await dir.getFileHandle(entryFile(info.name) + ".tmp-dead", { create: true });

  const c2 = await open(root, m.G);
  assert(![...dir.children.keys()].some((n) => n.includes(".tmp-")));
  assertEquals(await c2.get(info), null);
  assertEquals(c2.stats.bad, 1);
  assert(!dir.children.has(entryFile(info.name)));
  const again = await ggufEntry({ ...m.G, entryCache: c2 }, m.bytesOf, info.name, false);
  assert(same(again, e));
  assert(dir.children.has(entryFile(info.name)));           // rewritten
});

Deno.test("quota: no write near the limit, a quota error makes the cache read-only", async () => {
  const m = makeModel();
  const info = m.G.tensors["token_embd.weight"];
  const e = convertEntry(info, await m.bytesOf(info));

  const root = new FakeDir();
  const full = await open(root, m.G, { estimate: async () => ({ quota: 4 * 2 ** 30, usage: 3.5 * 2 ** 30 }) });
  await full.put(info, e);
  assertEquals(full.stats.full, 1);
  assertEquals(full.stats.write, 0);
  assert(!full.readOnly);   // may still fit later tensors after something else is freed

  const root2 = new FakeDir();
  const c = await open(root2, m.G);
  root2.failWrites = true;
  await c.put(info, e);
  assert(c.readOnly);
  assertEquals(c.stats.full, 1);
  assertEquals([...c.dir.children.keys()].filter((n) => n.endsWith(".bin") || n.includes(".tmp-")), []);
  root2.failWrites = false;
  await c.put(info, e);                                     // read-only for the rest of this load
  assertEquals(c.stats.write, 0);

  const noEst = await open(new FakeDir(), m.G, { estimate: null });
  assert(noEst.readOnly);
  const throws = await open(new FakeDir(), m.G, { estimate: async () => { throw new Error("no"); } });
  await throws.put(info, e);
  assertEquals(throws.stats.write, 0);
});

Deno.test("attach, size and clear", async () => {
  const m = makeModel();
  const root = new FakeDir();
  const storage = { getDirectory: async () => root, estimate: async () => big() };
  let fetches = 0;
  const fetchFn = async () => { fetches++; return new Response(await Deno.readFile(new URL("../../engine/gguf.js", import.meta.url))); };
  const G = { ...m.G, entryCache: "stale" };
  const c = await attachBrowserWeightCache(G, URL_PIN, { storage, srcUrl: "https://x/engine/gguf.js", fetchFn });
  assert(c && G.entryCache === c);
  assertEquals(c.meta.converter, LOADER_VERSION);
  assertEquals(c.meta.revision, "0123abcd");
  await ggufEntry(G, m.bytesOf, "token_embd.weight", false);
  await attachBrowserWeightCache({ ...m.G }, URL_PIN, { storage, srcUrl: "https://x/engine/gguf.js", fetchFn });
  assertEquals(fetches, 1);                                 // hashed once per page
  assertEquals(await convertedBytes(root), HDR + m.G.tensors["token_embd.weight"].nElems * 4);
  await clearConverted(root);
  assertEquals(await convertedBytes(root), 0);

  // no OPFS (older browser, some private windows): no cache, the load converts as before
  const G2 = { ...m.G, entryCache: "stale" };
  assertEquals(await attachBrowserWeightCache(G2, URL_PIN, { storage: {}, srcUrl: "x", fetchFn }), null);
  assertEquals(G2.entryCache, null);
  const G3 = { ...m.G };
  assertEquals(await attachBrowserWeightCache(G3, URL_PIN, { storage: { getDirectory: async () => { throw new Error("SecurityError"); } }, srcUrl: "y", fetchFn }), null);
  assertEquals(G3.entryCache, null);
});
