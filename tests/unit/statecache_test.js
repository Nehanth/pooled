// harness/statecache.js (tokenKey + StateCache over an in-memory stand-in for OPFS) and
// harness/sessions.js (Sessions over a fake engine). Before this only the GPU e2e
// sessions_synth / state_synth ran them.
import { tokenKey, StateCache } from "../../harness/statecache.js";
import { Sessions } from "../../harness/sessions.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

// ---------- tokenKey ----------

Deno.test("tokenKey: 32 hex chars, stable for the same input", async () => {
  const a = await tokenKey({ lo: 0, hi: 16, kv: "f16" }, [1, 2, 3], "qwen3.8-27b");
  ok(/^[0-9a-f]{32}$/.test(a), a);
  eq(await tokenKey({ lo: 0, hi: 16, kv: "f16" }, [1, 2, 3], "qwen3.8-27b"), a);
  eq(await tokenKey({ lo: 0, hi: 16, kv: "f16" }, Uint32Array.of(1, 2, 3), "qwen3.8-27b"), a, "typed array ids hash the same");
  eq(await tokenKey({ lo: 0, hi: 16, kv: "f16" }, Int32Array.of(1, 2, 3), "qwen3.8-27b"), a);
});

Deno.test("tokenKey: any change in sig, ids or model changes the key", async () => {
  const base = [{ lo: 0, hi: 16, kv: "f16" }, [1, 2, 3], "m"];
  const variants = [
    ["sig lo", [{ lo: 1, hi: 16, kv: "f16" }, [1, 2, 3], "m"]],
    ["sig kv", [{ lo: 0, hi: 16, kv: "f32" }, [1, 2, 3], "m"]],
    ["sig as string", ["0-16", [1, 2, 3], "m"]],
    ["one more id", [base[0], [1, 2, 3, 4], "m"]],
    ["one id less", [base[0], [1, 2], "m"]],
    ["ids reordered", [base[0], [3, 2, 1], "m"]],
    ["id changed", [base[0], [1, 2, 4], "m"]],
    ["no ids", [base[0], [], "m"]],
    ["model", [base[0], [1, 2, 3], "n"]],
    ["model default", [base[0], [1, 2, 3]]],
    ["big id", [base[0], [1, 2, 151643], "m"]],
  ];
  const k0 = await tokenKey(...base);
  const seen = new Map([[k0, "base"]]);
  for (const [why, args] of variants) {
    const k = await tokenKey(...args);
    ok(!seen.has(k), `${why} collides with ${seen.get(k)}`);
    seen.set(k, why);
  }
});

Deno.test("tokenKey: sig key order matters (JSON), so callers must build sig the same way", async () => {
  // documents the behaviour: { lo, hi } and { hi, lo } are different keys. The engine's
  // stateSignature() builds its object in one fixed order, so this is fine in practice.
  ok(await tokenKey({ lo: 0, hi: 1 }, [1]) !== await tokenKey({ hi: 1, lo: 0 }, [1]));
});

Deno.test("tokenKey: model/sig text cannot bleed into the token bytes", async () => {
  // the head is a JSON object, so a model name that ends in bytes that look like ids is still different
  ok(await tokenKey("s", [0x7d], "m") !== await tokenKey("s", [], "m}"));
  ok(await tokenKey("s", [], "") !== await tokenKey("", [], "s"));
});

// ---------- an in-memory OPFS directory ----------

class MemFile {
  constructor(bytes, lastModified) { this.bytes = bytes; this.size = bytes.length; this.lastModified = lastModified; }
  async arrayBuffer() { return this.bytes.slice().buffer; }
}
class MemDir {
  constructor({ move = true } = {}) { this.files = new Map(); this.clock = 1000; this.moveOk = move; this.removed = []; }
  async getFileHandle(name, { create = false } = {}) {
    if (!this.files.has(name)) {
      if (!create) { const e = new Error("not found: " + name); e.name = "NotFoundError"; throw e; }
      this.files.set(name, { bytes: new Uint8Array(0), t: ++this.clock });
    }
    const dir = this;
    const h = {
      kind: "file", name,
      async getFile() { const f = dir.files.get(h.name); return new MemFile(f.bytes, f.t); },
      async createWritable() {
        const chunks = [];
        return {
          async write(c) { chunks.push(new Uint8Array(c.buffer ? c.buffer.slice(c.byteOffset, c.byteOffset + c.byteLength) : c)); },
          async close() {
            const n = chunks.reduce((s, c) => s + c.length, 0), all = new Uint8Array(n);
            let o = 0; for (const c of chunks) { all.set(c, o); o += c.length; }
            dir.files.set(h.name, { bytes: all, t: ++dir.clock });
          },
        };
      },
    };
    if (this.moveOk) h.move = async (to) => { const f = dir.files.get(h.name); dir.files.delete(h.name); dir.files.set(to, f); h.name = to; };
    return h;
  }
  async removeEntry(name) { if (!this.files.delete(name)) throw new Error("not found"); this.removed.push(name); }
  async *entries() { for (const name of [...this.files.keys()]) yield [name, await this.getFileHandle(name)]; }
}
// localStorage for the LRU memory: an in-memory one, so tests do not touch Deno's on-disk store
const LS = new Map();
Object.defineProperty(globalThis, "localStorage", { configurable: true, value: {
  getItem: (k) => (LS.has(k) ? LS.get(k) : null), setItem: (k, v) => LS.set(k, String(v)), removeItem: (k) => LS.delete(k),
} });
let dirN = 0;
function cache(opts = {}, dirOpts = {}) {
  const c = new StateCache({ dirName: "t" + ++dirN, ...opts });
  c.dir = new MemDir(dirOpts);
  return c;
}
const state = (sig, pos, ...sizes) => ({ sig, pos, parts: sizes.map((n, i) => Uint8Array.from({ length: n }, (_, j) => (i * 31 + j) & 255).buffer) });
const bytesOf = (b) => [...new Uint8Array(b)];

// ---------- StateCache ----------

for (const move of [true, false]) {
  Deno.test(`StateCache: put/get round trip (${move ? "rename" : "copy fallback"})`, async () => {
    const c = cache({}, { move });
    const s = state({ lo: 0, hi: 4 }, 123, 16, 0, 7, 1024);
    ok(!(await c.has("k1")));
    eq(await c.get("k1"), null, "missing key -> null");
    await c.put("k1", s, { n: 123 });
    ok(await c.has("k1"));
    const g = await c.get("k1");
    eq([g.sig, g.pos, g.meta], [s.sig, 123, { n: 123 }]);
    eq(g.parts.length, 4);
    for (let i = 0; i < 4; i++) eq(bytesOf(g.parts[i]), bytesOf(s.parts[i]), `part ${i}`);
    eq([...c.dir.files.keys()], ["k1.bin"], "no .tmp left behind");
  });
}

Deno.test("StateCache: a state with no parts, and overwriting a key", async () => {
  const c = cache();
  await c.put("k", state("a", 0));
  eq((await c.get("k")).parts, []);
  await c.put("k", state("b", 5, 3));
  const g = await c.get("k");
  eq([g.sig, g.pos, g.parts.length], ["b", 5, 1]);
  eq((await c.entries()).length, 1);
});

Deno.test("StateCache: entries ignore non-.bin files; clear removes only .bin", async () => {
  const c = cache();
  await c.put("a", state("s", 1, 10)); await c.put("b", state("s", 2, 10));
  c.dir.files.set("junk.tmp", { bytes: new Uint8Array(4), t: 1 });
  eq((await c.entries()).map((e) => e.key).sort(), ["a", "b"]);
  await c.clear();
  eq([...c.dir.files.keys()], ["junk.tmp"]);
});

Deno.test("StateCache: eviction past the budget drops the least recently used", async () => {
  // each file = 4 + header (~70-90 bytes) + 1000; a budget of 2500 holds two
  const c = cache({ budgetBytes: 2500 });
  await c.put("a", state("s", 1, 1000));
  await c.put("b", state("s", 2, 1000));
  eq((await c.entries()).map((e) => e.key).sort(), ["a", "b"]);
  await c.put("c", state("s", 3, 1000));
  eq((await c.entries()).map((e) => e.key).sort(), ["b", "c"], "a is the oldest");
  // a get marks b as used now: the next put evicts c instead
  await c.get("b");
  await new Promise((r) => setTimeout(r, 2));   // _touch runs after get resolves
  const used = JSON.parse(LS.get("statecache:" + c.dirName));
  ok(used.b > 1e12, "the LRU memory holds a wall-clock time for b");
  await c.put("d", state("s", 4, 1000));
  const left = (await c.entries()).map((e) => e.key).sort();
  ok(left.includes("b"), `b was just used, it must stay (left: ${left})`);
});

Deno.test("StateCache: a single state bigger than the budget is evicted at once (put then get -> null)", async () => {
  const c = cache({ budgetBytes: 100 });
  await c.put("huge", state("s", 1, 1000));
  eq(await c.get("huge"), null);
});

Deno.test("StateCache: budget 0 keeps nothing; a huge budget keeps everything", async () => {
  const z = cache({ budgetBytes: 0 });
  await z.put("a", state("s", 1, 1));
  eq(await z.entries(), []);
  const big = cache({ budgetBytes: 2 ** 40 });
  for (let i = 0; i < 20; i++) await big.put("k" + i, state("s", i, 100));
  eq((await big.entries()).length, 20);
});

Deno.test("StateCache: localStorage that throws (private mode) is not fatal", async () => {
  const c = cache({ budgetBytes: 2500 });
  const saved = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  Object.defineProperty(globalThis, "localStorage", { configurable: true, get() { throw new Error("SecurityError"); } });
  try {
    await c.put("a", state("s", 1, 10));
    eq((await c.get("a")).pos, 1);
    await new Promise((r) => setTimeout(r, 2));
    eq((await c.entries()).length, 1);
  } finally { Object.defineProperty(globalThis, "localStorage", saved); }
});

// ---------- Sessions over a fake engine ----------

// The engine's "state" is the token list it has seen. Slots copy it; export/import round trip it.
class FakeEngine {
  constructor() { this.toks = []; this.slots = new Map(); this.log = []; }
  get pos() { return this.toks.length; }
  feed(...t) { this.toks.push(...t); }
  saveSlot(n) { this.log.push("save " + n); this.slots.set(n, this.toks.slice()); }
  loadSlot(n) { this.log.push("load " + n); if (!this.slots.has(n)) throw new Error("no slot " + n); this.toks = this.slots.get(n).slice(); }
  dropSlot(n) { this.log.push("drop " + n); this.slots.delete(n); }
  reset() { this.log.push("reset"); this.toks = []; }
  async exportSlot(n) { return { sig: "fake", pos: this.slots.get(n).length, parts: [Uint32Array.from(this.slots.get(n)).buffer] }; }
  async exportState() { return { sig: "fake", pos: this.toks.length, parts: [Uint32Array.from(this.toks).buffer] }; }
  importState(st) { this.log.push("import"); this.toks = [...new Uint32Array(st.parts[0])]; }
}
function sessions(gpuSlots = 1) {
  const e = new FakeEngine(), c = cache();
  return { e, c, S: new Sessions(e, { gpuSlots, cache: c, prefix: "t" }) };
}

Deno.test("Sessions: a table of switches with 1 GPU slot (one goes to disk and back)", async () => {
  const { e, c, S } = sessions(1);
  const steps = [
    // [switchTo, expected from, tokens fed after, expected parked where]
    ["A", "new", [1, 2], {}],
    ["B", "new", [10], { A: "gpu" }],
    ["C", "new", [20, 21, 22], { A: "disk", B: "gpu" }],   // A is the least recently used: spilled
    ["A", "disk", [3], { B: "disk", C: "gpu" }],           // B spilled when C parked
    ["A", "active", [4], { B: "disk", C: "gpu" }],
    ["C", "gpu", [], { A: "gpu", B: "disk" }],
    ["B", "disk", [11], { A: "disk", C: "gpu" }],
  ];
  const want = { A: [1, 2, 3, 4], B: [10, 11], C: [20, 21, 22] };
  for (const [id, from, feed, where] of steps) {
    eq(await S.switchTo(id), from, `switchTo ${id}`);
    e.feed(...feed);
    const parked = Object.fromEntries(S.list().filter((x) => x.where !== "active").map((x) => [x.id, x.where]).sort());
    eq(parked, where, `after switchTo ${id}`);
    eq(S.list().find((x) => x.where === "active").id, id);
  }
  eq(e.toks, want.B, "B resumed exactly");
  await S.switchTo("A"); eq(e.toks, want.A, "A resumed exactly");
  await S.switchTo("C"); eq(e.toks, want.C, "C resumed exactly");
  // table: 3 fresh, A/B from disk, C from GPU; then B->A and A->C each come back from disk (each park spills the LRU)
  eq(S.stats, { gpuHits: 1, diskHits: 4, fresh: 3, spills: 5 });
  ok(e.slots.size <= 1, "never more than gpuSlots parked on the GPU");
  ok((await c.entries()).length >= 1);
});

Deno.test("Sessions: 0 GPU slots sends every parked session to disk; 5 slots never spill", async () => {
  for (const [slots, spills] of [[0, 4], [5, 0]]) {
    const { e, S } = sessions(slots);
    for (const id of ["a", "b", "c", "d", "e"]) { await S.switchTo(id); e.feed(id.charCodeAt(0)); }
    eq(S.stats.spills, spills, `gpuSlots=${slots}`);
    for (const id of ["a", "b", "c", "d"]) { await S.switchTo(id); eq(e.toks, [id.charCodeAt(0)], `${id} with gpuSlots=${slots}`); }
  }
});

Deno.test("Sessions: a disk state that vanished (evicted) starts fresh", async () => {
  const { e, c, S } = sessions(0);
  await S.switchTo("a"); e.feed(1, 2);
  await S.switchTo("b");
  eq(S.list().find((x) => x.id === "a").where, "disk");
  await c.clear();
  eq(await S.switchTo("a"), "new");
  eq(e.toks, []);
  eq(S.stats.fresh, 3);
});

Deno.test("Sessions: close forgets a session wherever it is", async () => {
  const { e, c, S } = sessions(1);
  await S.switchTo("a"); e.feed(1);
  await S.switchTo("b"); e.feed(2);
  await S.switchTo("c"); e.feed(3);   // a on disk, b on GPU
  await S.close("a");
  eq((await c.entries()).length, 0, "a's file removed");
  await S.close("b");
  eq(e.slots.size, 0, "b's slot dropped");
  await S.close("c");
  eq(S.active, null); eq(e.toks, [], "closing the active one resets the engine");
  eq(S.list(), []);
  await S.close("never-seen");   // no-op
  eq(await S.switchTo("a"), "new", "a closed session comes back empty");
});

Deno.test("Sessions: ids with odd characters get safe, distinct-enough cache keys", async () => {
  const { e, c, S } = sessions(0);
  await S.switchTo("proj/x y"); e.feed(7);
  await S.switchTo("other");
  const keys = (await c.entries()).map((x) => x.key);
  eq(keys, ["t-proj_x_y"]);
  eq(await S.switchTo("proj/x y"), "disk");
  eq(e.toks, [7]);
});

Deno.test("Sessions: persist writes the active state without parking it", async () => {
  const { e, c, S } = sessions(1);
  await S.persist();   // nothing active: no-op
  eq((await c.entries()).length, 0);
  await S.switchTo("a"); e.feed(5, 6);
  await S.persist();
  eq(S.list(), [{ id: "a", where: "active" }]);
  const g = await c.get("t-a");
  eq([...new Uint32Array(g.parts[0])], [5, 6]);
  eq(g.meta, { id: "a" });
});
