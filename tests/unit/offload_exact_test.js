// Expert offload's RAM promise, exactly (review of #320): the deal (room/plan.js offloadPlan) sizes what a device parks
// from the experts' own formats, layer by layer (room/models.js expertsOf), the way ExpertStore (engine/expert_store.js)
// parks them: Q4_0 stays Q4 (18/32 B per weight), Q4_1 and the other quants become Q8 (34/32). The 35B's layers 0-4
// and the 122B's 0-5 have Q4_1 down experts, so a layer-0-3 average under- or overstates a range. Here a fake device
// parks what the deal planned in a real ExpertStore: the bytes it parks are the deal's ramBytes, its buffers stay
// within the device's RAM, and its slot count is the deal's.
// No GPU.   deno test --allow-read tests/unit/offload_exact_test.js
import { dealRoom, offloadPlan, OFFLOAD_MIN_SLOTS } from "../../room/plan.js";
import { roomBytes, expertsOf, expertSlicesOf, SHAPE } from "../../room/models.js";
import { ExpertStore } from "../../engine/expert_store.js";
import { ggmlTypeBytes, GGML_Q4_0, GGML_Q4_1, GGML_Q5_0, GGML_Q8_0 } from "../../engine/gguf.js";

globalThis.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const eq = (a, b, m) => { if (a !== b) throw new Error((m || "mismatch") + `: ${a} != ${b}`); };
const GB = 2 ** 30;
function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const int = (r, lo, hi) => lo + Math.floor(r() * (hi - lo + 1));

function fakeDevice() {
  const bufs = [];
  return {
    bufs, limits: { maxBufferSize: 2 ** 31 },
    createBuffer({ size, usage, mappedAtCreation }) {
      const b = { size, usage, bytes: new Uint8Array(size), mapped: !!mappedAtCreation, destroy() {},
        getMappedRange() { return this.bytes.buffer; }, unmap() { this.mapped = false; } };
      bufs.push(b);
      return b;
    },
    pushErrorScope() {}, popErrorScope: async () => null,
  };
}
// a toy MoE header: per layer the GGML types of its gate / up / down experts (nExp experts of ffn x hidden)
function header(types, nExp, ffn, hidden) {
  const tensors = {};
  types.forEach((ts, i) => ["gate", "up", "down"].forEach((p, j) => {
    const shape = j < 2 ? [nExp, ffn, hidden] : [nExp, hidden, ffn];
    tensors[`blk.${i}.ffn_${p}_exps.weight`] = { shape, ggmlType: ts[j], byteLength: ggmlTypeBytes(ts[j], nExp * ffn * hidden) };
  }));
  return { meta: {}, tensors };
}
// what the loader hands ExpertStore for one tensor (engine/gguf.js convertEntry's shape and sizes)
function converted(t) {
  const [nExp, rows, cols] = t.shape, n = nExp * rows * cols, q4 = t.ggmlType === GGML_Q4_0;
  return { kind: q4 ? "q4" : "q8", qs: new Uint8Array(q4 ? n / 2 : n), scales: new Uint8Array(n / 16), shape: [nExp * rows, cols] };
}
// park layers [lo, hi) of G in an ExpertStore as shard.js does (ramBytes: the device's RAM; expectBytes: the deal's)
function parkAll(G, lo, hi, { vramBytes, ramBytes = Infinity, expectBytes = 0, K = 8 }) {
  const dev = fakeDevice(), layers = Array.from({ length: hi - lo }, (_, i) => lo + i);
  const st = new ExpertStore(dev, { layers, vramBytes, ramBytes, expectBytes });
  for (const l of layers) for (const p of ["gate", "up", "down"]) {
    const t = G.tensors[`blk.${l}.ffn_${p}_exps.weight`];
    st.park(l, p, converted(t), t.shape[0]);
  }
  st.build({ K });
  return st;
}
const parkedOf = (prof, lo, hi) => { let s = 0; for (let l = lo; l < hi; l++) s += prof.slices[l].reduce((a, b) => a + b, 0) * prof.nExp; return s; };

Deno.test("expertsOf: Q4_0 parks 18/32 B per weight, Q4_1 / Q5_0 / Q8_0 34/32; SHAPE's profiles are the real headers'", () => {
  const G = header([[GGML_Q4_0, GGML_Q4_0, GGML_Q4_1], [GGML_Q4_0, GGML_Q8_0, GGML_Q5_0]], 16, 64, 128);
  const p = expertsOf(G, 2), n = 64 * 128;
  eq(JSON.stringify(p.slices[0]), JSON.stringify([n / 2, n / 16, n / 2, n / 16, n, n / 16]));
  eq(JSON.stringify(p.slices[1]), JSON.stringify([n / 2, n / 16, n, n / 16, n, n / 16]));
  eq(parkedOf(p, 0, 1), 16 * n * (18 + 18 + 34) / 32);
  eq(p.E, (G.tensors["blk.0.ffn_gate_exps.weight"].byteLength * 2 + G.tensors["blk.0.ffn_down_exps.weight"].byteLength
    + G.tensors["blk.1.ffn_gate_exps.weight"].byteLength + G.tensors["blk.1.ffn_up_exps.weight"].byteLength + G.tensors["blk.1.ffn_down_exps.weight"].byteLength) / 2);
  ok(expertsOf({ tensors: {} }, 2) === null && expertSlicesOf({ tensors: {} }, 0) === null, "a dense model has none");
  // the 35B: layers 0-4 park 0.547 GiB (Q4_1 down -> Q8), 5-39 0.422 GiB; the 122B: 0-5 1.641, 6-47 1.266
  const p35 = roomBytes("qwen3.6-35b-moe", 4096).experts, p122 = roomBytes("qwen3.5-122b-moe", 4096).experts;
  eq(p35.slices.length, 40); eq(p122.slices.length, 48);
  eq(parkedOf(p35, 0, 1), 587202560); eq(parkedOf(p35, 4, 5), 587202560); eq(parkedOf(p35, 5, 6), 452984832);
  eq(parkedOf(p122, 5, 6), 1761607680); eq(parkedOf(p122, 6, 7), 1358954496);
  eq(p35.E, SHAPE["qwen3.6-35b-moe"].exp);
});

Deno.test("offloadPlan: a range over Q4_1 and Q4_0 layers parks exactly what ExpertStore parks, with ExpertStore's slots", () => {
  // layers 0-3: Q4_1 down experts (as the 35B's 0-4), 4-11: Q4_0
  const types = Array.from({ length: 12 }, (_, i) => [GGML_Q4_0, GGML_Q4_0, i < 4 ? GGML_Q4_1 : GGML_Q4_0]);
  const nExp = 64, G = header(types, nExp, 64, 128), prof = expertsOf(G, 12);
  const LB = prof.E * 1.06, host = 0;
  // one device holds layers 0-11 within a pledge of 7 layers: it offloads most of them, the Q4_1 ones among them
  const bytes = 7 * LB, ram = 64 * GB;
  const o = offloadPlan(12, bytes, ram, LB, prof, OFFLOAD_MIN_SLOTS, 256, 12 + host);
  ok(o && o.layers >= 9, JSON.stringify(o));
  const lo = 12 - o.layers;
  ok(lo < 4, "the offloaded range has Q4_1 layers");
  eq(o.ramBytes, parkedOf(prof, lo, 12), "the plan's RAM is the exact parked bytes");
  const st = parkAll(G, lo, 12, { vramBytes: o.vramBytes, ramBytes: ram, expectBytes: o.ramBytes });
  eq(st.parkedBytes, o.ramBytes, "ExpertStore parks the plan's ramBytes");
  eq(st.P, o.slots, "ExpertStore's slots are the plan's");
  ok(st.regionBytes + st.poolBytes <= o.vramBytes, "region + pools within the cache");
  // the old estimate (every layer the layer-0-3 average of the file) said less than this range really parks
  const old = offloadPlan(12, bytes, ram, LB, prof.E, OFFLOAD_MIN_SLOTS, nExp);
  ok(old.layers * prof.E < st.parkedBytes, `the file-bytes estimate ${old.layers * prof.E} < parked ${st.parkedBytes}`);
});

Deno.test("offload, the 35B: a range with layers 0-4 plans its Q8-widened experts; the 7-39 range its Q4 ones", () => {
  const rb = roomBytes("qwen3.6-35b-moe", 32768), prof = rb.experts;
  const pc = (gb, ram) => ({ gb, ram });
  for (const [devs, lo0] of [[[pc(10, 48)], 0], [[pc(6, 0), pc(8, 48)], 7]]) {
    const off = { expertBytes: rb.expertBytes, experts: prof, ram: devs.map((d) => d.ram * GB) };
    const d = dealRoom({ L: rb.L, layerBytes: rb.layerBytes, hostBytes: rb.hostBytes, pledges: devs.map((x) => x.gb * GB), mode: "speed", off });
    ok(d.fit.fits, JSON.stringify(devs));
    const k = d.offload.findIndex(Boolean), o = d.offload[k];
    ok(o.lo >= lo0, `${o.lo}`);
    eq(o.ramBytes, parkedOf(prof, o.lo, o.hi));
    // ExpertStore's slots: (cache - region) / one expert of every offloaded layer
    let region = 0, per = 0;
    for (let j = 0; j < 6; j++) region += Math.max(...prof.slices.slice(o.lo, o.hi).map((s) => s[j])) * 256;
    for (let l = o.lo; l < o.hi; l++) per += prof.slices[l].reduce((a, b) => a + b, 0);
    eq(o.slots, Math.min(256, Math.floor((o.vramBytes - region) / per)));
  }
});

Deno.test("fuzz: what ExpertStore parks never exceeds the deal's ramBytes or the device's RAM; its slots are the deal's", () => {
  const r = rng(320), TYPES = [GGML_Q4_0, GGML_Q4_0, GGML_Q4_1, GGML_Q8_0, GGML_Q5_0];
  let offloaded = 0, tight = 0;
  for (let t = 0; t < 400; t++) {
    const L = int(r, 4, 20), nExp = [64, 128, 256][int(r, 0, 2)], ffn = 32, hidden = [64, 128][int(r, 0, 1)];
    // runs of layers that park alike (as real files: a few Q4_1 layers first, then Q4_0), or any mix
    const mixed = r() < 0.5, head = int(r, 0, L);
    const types = Array.from({ length: L }, (_, i) => (mixed ? [0, 1, 2].map(() => TYPES[int(r, 0, TYPES.length - 1)]) : [GGML_Q4_0, GGML_Q4_0, i < head ? GGML_Q4_1 : GGML_Q4_0]));
    const G = header(types, nExp, ffn, hidden), prof = expertsOf(G, L);
    const maxPark = Math.max(...prof.slices.map((s, l) => parkedOf(prof, l, l + 1)));
    const LB = prof.E * (1.02 + r() * 0.3) + int(r, 1, 30) * 1024, host = int(r, 0, 3) * LB / 2;
    const n = int(r, 1, 3);
    const pledges = Array.from({ length: n }, (_, i) => (0.2 + r() * 0.8) * (L * LB) / n + (i === 0 ? host : 0));
    const ram = Array.from({ length: n }, () => (r() < 0.6 ? r() * L * maxPark * 1.1 : 0));
    const off = ram.some((x) => x > 0) ? { expertBytes: prof.E, experts: prof, ram } : null;
    const d = dealRoom({ L, layerBytes: LB, hostBytes: host, pledges, mode: r() < 0.5 ? "speed" : "memory", off });
    const ctx = `case ${t}: L=${L} nExp=${nExp} pledges=${pledges.map(Math.round)} ram=${ram.map(Math.round)}`;
    if (!d.fit.fits) continue;
    d.used.forEach((i, k) => {
      const o = d.offload?.[k];
      if (!o) { ok(d.assigned[k] * LB + (i === 0 ? host : 0) <= pledges[i] + 1, ctx + " resident within the pledge"); return; }
      offloaded++;
      ok(o.lo >= d.ranges[k][0] && o.hi === d.ranges[k][1], ctx + " the last layers of its range");
      eq(o.ramBytes, parkedOf(prof, o.lo, o.hi), ctx + " planned = exact");
      ok(o.ramBytes <= ram[i], ctx + " planned within its RAM");
      const vram = d.assigned[k] * LB + (i === 0 ? host : 0) - o.layers * prof.E + o.vramBytes;
      ok(vram <= pledges[i] + 1, ctx + " within its pledge");
      const st = parkAll(G, o.lo, o.hi, { vramBytes: o.vramBytes, ramBytes: ram[i], expectBytes: o.ramBytes });
      eq(st.parkedBytes, o.ramBytes, ctx + " parked = planned");
      ok(st.bufBytes <= ram[i], ctx + ` buffers ${st.bufBytes} within the RAM ${ram[i]}`);
      eq(st.P, o.slots, ctx + " slots");
      ok(st.P >= OFFLOAD_MIN_SLOTS, ctx + " the minimum cache");
      if (ram[i] < 1.2 * st.bufBytes) tight++;
    });
  }
  ok(offloaded > 50 && tight > 5, `the fuzz offloads (${offloaded}, ${tight} with little RAM to spare)`);
});

Deno.test("ExpertStore: parking past the RAM it may use fails the load instead of going over", () => {
  const types = Array.from({ length: 4 }, (_, i) => [GGML_Q4_0, GGML_Q4_0, i < 2 ? GGML_Q4_1 : GGML_Q4_0]);
  const G = header(types, 16, 64, 128), prof = expertsOf(G, 4), all = parkedOf(prof, 0, 4);
  // exactly what it needs (and the planner's buffer slack): fine
  const st = parkAll(G, 0, 4, { vramBytes: 1e9, ramBytes: all + 8192, expectBytes: all });
  eq(st.parkedBytes, all); ok(st.bufBytes <= all + 8192);
  // the file-bytes estimate the old deal used (under what layers 0-1 really park): the load fails, naming the RAM
  const est = 4 * prof.E;
  ok(est < all);
  let err = null;
  try { parkAll(G, 0, 4, { vramBytes: 1e9, ramBytes: est, expectBytes: est }); } catch (e) { err = e; }
  ok(err && /RAM this device lends|over the/.test(err.message), String(err?.message));
});
