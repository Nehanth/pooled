// CPU check of the 3-tier cache simulator (tests/sim/tiersim.js): the .npy reader, the VRAM pool against the real
// ExpertStore plan() on random traces (same hits, so the simulator's pool is the engine's), T1 bookkeeping on a hand
// worked case, the 512-expert mapping, an oracle prefetch, where a prefetch lands (T1 on a room node, the pool on a
// tab), and the checks that keep a trace from being replayed as the wrong model.
// No GPU.   deno test --allow-read tests/unit/tiersim_test.js
import { parseNpy, simulate, splitFlash, hash01, roomRun, MODELS, PROFILES, range, traceOf, modelFor, concatTraces } from "../sim/tiersim.js";
import { ExpertStore } from "../../engine/expert_store.js";

globalThis.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
const assert = (c, m) => { if (!c) throw new Error(m); };

// the writer of tests/trace_moe.js
function npy(arr, descr, shape) {
  let h = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${shape.join(", ")}${shape.length === 1 ? "," : ""}), }`;
  const pad = 64 - ((10 + h.length + 1) % 64);
  h += " ".repeat(pad % 64) + "\n";
  const head = new Uint8Array(10 + h.length);
  head.set([0x93, ...new TextEncoder().encode("NUMPY"), 1, 0, h.length & 255, h.length >> 8]);
  head.set(new TextEncoder().encode(h), 10);
  const body = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  const out = new Uint8Array(head.length + body.length); out.set(head); out.set(body, head.length);
  return out;
}
// a routing trace with some locality: each layer reuses an expert of its previous step with probability `stay`
function randTrace({ T = 200, L = 3, K = 4, nExp = 32, stay = 0.5, seed = 3 }) {
  const sel = new Uint16Array(T * L * K);
  for (let t = 0; t < T; t++) for (let l = 0; l < L; l++) {
    const ids = [];
    for (let k = 0, i = 0; ids.length < K; i++) {
      const prev = t ? sel[((t - 1) * L + l) * K + (k % K)] : -1;
      const e = prev >= 0 && hash01(seed, 1, t, l, i) < stay ? prev : Math.floor(hash01(seed, 2, t, l, i) * nExp);
      if (!ids.includes(e)) ids.push(e);
      k++;
    }
    sel.set(ids, (t * L + l) * K);
  }
  return { name: "rand", T, L, K, nExp, sel, w: null };
}
const unit = () => ({ file: 1, engine: 1 });

Deno.test("parseNpy: the trace writer's uint8, uint16 and float32 arrays come back", () => {
  const a = new Uint8Array([1, 2, 3, 250, 0, 7]), b = new Uint16Array([511, 0, 300, 2]), c = new Float32Array([0.5, -1, 3.25]);
  const ra = parseNpy(npy(a, "|u1", [2, 1, 3])), rb = parseNpy(npy(b, "<u2", [1, 2, 2])), rc = parseNpy(npy(c, "<f4", [3]));
  assert(ra.data instanceof Uint8Array && ra.shape.join() === "2,1,3" && ra.data.join() === a.join(), "u1");
  assert(rb.data instanceof Uint16Array && rb.shape.join() === "1,2,2" && rb.data.join() === b.join(), "u2");
  assert(rc.data instanceof Float32Array && rc.shape.join() === "3" && rc.data.join() === c.join(), "f4");
});

// a store with the same fake device as expert_store_test.js
function store({ layers, nExp, slots, K }) {
  const dev = { limits: { maxBufferSize: 2 ** 31 }, pushErrorScope() {}, popErrorScope: async () => null,
    createBuffer({ size, mappedAtCreation }) { return { size, bytes: new Uint8Array(size), mapped: !!mappedAtCreation, getMappedRange() { return this.bytes.buffer; }, unmap() { this.mapped = false; } }; } };
  const st = new ExpertStore(dev, { layers, slots, parkBytes: 1 << 20 });
  const entry = (rows, cols, kind) => ({ kind, qs: new Uint8Array(rows * cols / (kind === "q4" ? 2 : 1) * nExp), scales: new Uint32Array(rows * cols / 64 * nExp), shape: [nExp * rows, cols] });
  for (const l of layers) ["gate", "up", "down"].forEach((p, i) => st.park(l, p, i < 2 ? entry(32, 64, "q4") : entry(64, 32, "q8"), nExp));
  st.build({ K });
  return st;
}

Deno.test("pool: same hits as ExpertStore.plan() on random traces", () => {
  for (const [P, stay] of [[4, 0.3], [6, 0.5], [12, 0.7], [20, 0.2]]) {
    const tr = randTrace({ stay, seed: P });
    const st = store({ layers: range(0, tr.L - 1), nExp: tr.nExp, slots: P, K: tr.K });
    const enc = { copyBufferToBuffer() {} };
    for (let t = 0; t < tr.T; t++) for (let l = 0; l < tr.L; l++) {
      const s = new Uint32Array(tr.K + 1); s.set(tr.sel.subarray((t * tr.L + l) * tr.K, (t * tr.L + l + 1) * tr.K));
      st.plan(enc, l, s, 1, tr.K + 1);
    }
    const r = simulate(tr, { layers: range(0, tr.L - 1), P, parked: new Set(range(0, tr.L - 1)), expert: unit, warm: 0 });
    const hits = r.hitV.reduce((a, b) => a + b, 0), misses = r.hitT1.reduce((a, b) => a + b, 0);
    assert(hits === st.stats.hits && misses === st.stats.misses, `P=${P}: sim ${hits}/${misses} vs store ${st.stats.hits}/${st.stats.misses}`);
  }
});

Deno.test("T1: a global LRU below the pool, by hand", () => {
  // one layer, K = 2, 2 slots; steps [0,1] [2,3] [0,1] [4,5] [2,3]
  const tr = { T: 5, L: 1, K: 2, nExp: 8, sel: new Uint8Array([0, 1, 2, 3, 0, 1, 4, 5, 2, 3]) };
  const run = (t1Bytes, t1Policy) => { const r = simulate(tr, { layers: [0], P: 2, t1Bytes, t1Policy, expert: unit, warm: 0 }); return [r.tot.hitV, r.tot.hitT1 * 10, r.tot.disk * 10]; };
  // T1 of 2 experts holds only what the pool holds: nothing to gain
  assert(run(2, "use").join() === "0,0,10", "T1 = pool size: all disk " + run(2, "use"));
  // T1 of 4: step 3 finds 0 and 1 in T1, everything else is read from disk
  assert(run(4, "use").join() === "0,2,8", "T1 of 4 " + run(4, "use"));
  assert(run(0, "use").join() === "0,0,10", "no T1");
  // parked: every pool miss is a RAM hit
  const p = simulate(tr, { layers: [0], P: 2, parked: new Set([0]), expert: unit, warm: 0 });
  assert(p.tot.hitT1 === 1 && p.tot.disk === 0, "parked");
});

Deno.test("splitFlash: 512 experts, top-10, distinct ids, deterministic; q = 0 keeps each expert's child", () => {
  const tr = randTrace({ T: 60, L: 4, K: 8, nExp: 256, seed: 9 });
  tr.w = new Float32Array(tr.sel.length).map((_, i) => hash01(5, i));
  for (const q of [0, 0.25, 0.5]) {
    const f = splitFlash(tr, { q }), g = splitFlash(tr, { q });
    assert(f.K === 10 && f.nExp === 512 && f.sel.length === tr.T * tr.L * 10, "shape");
    assert(f.sel.every((e, i) => e === g.sel[i] && e < 512), "deterministic, in range");
    for (let x = 0; x < tr.T * tr.L; x++) {
      const ids = f.sel.subarray(x * 10, x * 10 + 10);
      assert(new Set(ids).size === 10, "distinct per step");
      for (let k = 0; k < 8; k++) assert(ids[k] >> 1 === tr.sel[x * 8 + k], "children of the traced experts");
    }
  }
  const f0 = splitFlash(tr, { q: 0 }), seen = new Map();
  for (let t = 0; t < tr.T; t++) for (let l = 0; l < tr.L; l++) for (let k = 0; k < 8; k++) {
    const id = f0.sel[(t * tr.L + l) * 10 + k], key = l * 256 + (id >> 1);
    assert(!seen.has(key) || seen.get(key) === id, "q = 0: one child per expert");
    seen.set(key, id);
  }
});

Deno.test("prefetch: an oracle one layer ahead serves every disk layer but the first from T1; rooms add up", () => {
  const tr = randTrace({ T: 80, L: 6, K: 4, nExp: 64, stay: 0.2, seed: 4 });
  const layers = range(0, 5), big = { file: 1e6, engine: 1e6 };
  const r = simulate(tr, { layers, P: 4, t1Bytes: 1e12, prefetch: { ahead: 1, top: 4, recall: 1 }, expert: () => big, warm: 0 });
  for (let x = 0; x < r.N * r.nl; x++) if (x % r.nl) assert(r.disk[x] === 0, "layers past the first: no demand reads");
  assert(r.pfUsed.reduce((a, b) => a + b, 0) > 0 && r.pfWasted === 0, "prefetches used, none wasted");
  // a room's ms are its devices' plus the net; no offload, no cuts
  const m = MODELS.q122, one = roomRun(tr, m, [{ prof: PROFILES.pcCli, resident: 6 }]), two = roomRun(tr, m, [{ prof: PROFILES.pcCli, resident: 3 }, { prof: PROFILES.pcCli, resident: 3 }]);
  assert(Math.abs(two.ms.mid - one.ms.mid - 6) < 1e-9, "two devices: + the LAN's 6 ms");
});

Deno.test("traces: no .json and 192k uint16 ids (a Flash-sized trace) give nExp from the ids; bad geometry throws", () => {
  const T = 400, L = 48, K = 10, sel = new Uint16Array(T * L * K).map((_, i) => i % 509);
  const tr = traceOf("flash", { data: sel, shape: [T, L, K] });
  assert(tr.nExp === 509 && tr.L === 48 && tr.K === 10, "nExp from the largest id " + tr.nExp);
  let threw = false;
  try { traceOf("bad", { data: sel, shape: [T, L, K] }, null, { layers: 40 }); } catch { threw = true; }
  assert(threw, ".json layers != .npy layers throws");
  // a 35B trace (40 layers, 256 experts, top-8) is not the 122B: picked by geometry it has no model, forced it throws
  const t35 = { name: "35b", T: 4, L: 40, K: 8, nExp: 256, sel: new Uint8Array(4 * 40 * 8), meta: { model: "Qwen3.5 35B A3B" } };
  for (const key of [null, "q122"]) { threw = false; try { modelFor([t35], key); } catch { threw = true; } assert(threw, `35B trace as ${key}`); }
  const t122 = { ...t35, name: "122b", L: 48, sel: new Uint8Array(4 * 48 * 8), meta: { model: "Qwen3.5 122B A10B" } };
  assert(modelFor([t122]) === "q122", "122B by name");
  threw = false; try { modelFor([t122, t35], "q122"); } catch { threw = true; } assert(threw, "every trace is checked");
  // simulate() refuses layers past the trace's
  threw = false; try { simulate(t35, { layers: range(0, 47), P: 8, expert: unit }); } catch { threw = true; } assert(threw, "layers past L");
});

Deno.test("prefetch lands in T1 on a room node and in the pool on a tab, page cache or not", () => {
  const tr = randTrace({ T: 80, L: 6, K: 4, nExp: 64, stay: 0.2, seed: 5 }), layers = range(0, 5), big = { file: 1e6, engine: 1e6 };
  const pf = { ahead: 1, top: 4, recall: 1 };
  const node = simulate(tr, { layers, P: 4, t1Bytes: 1e12, prefetch: pf, prefetchTo: "t1", expert: () => big, warm: 0 });
  const tab = simulate(tr, { layers, P: 4, t1Bytes: 1e12, prefetch: pf, prefetchTo: "pool", expert: () => big, warm: 0 });
  // oracle: on the tab every layer past the first finds its experts in the pool; on the node, in T1
  const sum = (a, from) => a.reduce((s, v, x) => s + (x % 6 >= from ? v : 0), 0);
  assert(sum(tab.hitT1, 1) === 0 && sum(tab.disk, 1) === 0 && sum(tab.hitV, 1) === (tr.T) * 5 * 4, "tab: pool hits " + sum(tab.hitV, 1));
  assert(sum(node.disk, 1) === 0 && sum(node.hitT1, 1) > 0, "node: T1 hits");
  // a tab with a page cache uploads a prediction once, from the page cache when it is there
  assert(tab.pfT1Eng.reduce((a, b) => a + b, 0) > 0, "page-cache sourced prefetches");
  // and roomRun picks the destination from the profile
  const m = { ...MODELS.q122, L: 6, nExp: 64, K: 4, expert: () => big };
  const r = roomRun(tr, m, [{ prof: PROFILES.pcTab, offload: { layers, P: 4, t1Bytes: 1e12, prefetch: pf } }]);
  const want = simulate(tr, { layers, P: 4, t1Bytes: 1e12, prefetch: pf, prefetchTo: "pool", expert: () => big }).tot.hitV;
  const rn = roomRun(tr, m, [{ prof: PROFILES.pcCli, offload: { layers, P: 4, t1Bytes: 1e12, prefetch: pf } }]);
  assert(r.sim.hitV === want && rn.sim.hitV < want, `tab room: into the pool (${r.sim.hitV} = ${want}); room node: T1 (${rn.sim.hitV})`);
});

Deno.test("concatTraces: back to back, shapes checked", () => {
  const a = randTrace({ T: 10, seed: 1 }), b = randTrace({ T: 7, seed: 2 });
  const c = concatTraces([a, b]);
  assert(c.T === 17 && c.sel.length === 17 * a.L * a.K && c.sel[10 * a.L * a.K] === b.sel[0], "concat");
  let threw = false; try { concatTraces([a, randTrace({ T: 5, K: 3 })]); } catch { threw = true; } assert(threw, "shape mismatch");
});
