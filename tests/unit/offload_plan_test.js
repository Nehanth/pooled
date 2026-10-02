// Expert offload in the room's deal (room/plan.js): a room node with a discrete GPU offers RAM for a MoE model's
// routed experts (meta.offload, meta.ramGB); when the pledges alone can't hold the model, it holds layers past its
// pledge with their experts parked in RAM and a VRAM cache. Its pledge and its RAM are never exceeded, devices
// that don't offload keep the pledge path, and a room the pledges hold deals exactly as before.
import { roomFit, dealRoom, offloadNeed, offloadCap, offloadPlan, layerCaps, OFFLOAD_MIN_SLOTS, OFFLOAD_GOOD_SLOTS } from "../../room/plan.js";
import { roomBytes, mergeSplitHeaders, expertBytesOf } from "../../room/models.js";
import { ramGB, offloadFor, pledgeGB } from "../../room/pledge.js";
import { sameShard } from "../../room/resume.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const sum = (a) => a.reduce((s, x) => s + x, 0);
const GB = 2 ** 30;
function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const int = (r, lo, hi) => lo + Math.floor(r() * (hi - lo + 1));

const pc = (gb, ram) => ({ webgpu: true, contribGB: gb, offload: true, ramGB: ram });
const tab = (gb) => ({ webgpu: true, contribGB: gb });
// a room as the host deals it (room-node dealPlan, room.js aiStart): pledges, RAM, the model's bytes at ctx
function deal(model, metas, { ctx = 32768, mode = "speed", phone = [] } = {}) {
  const rb = roomBytes(model, ctx);
  const pledges = metas.map((m) => pledgeGB(m) * GB);
  return { rb, pledges, d: dealRoom({ L: rb.L, layerBytes: rb.layerBytes, hostBytes: rb.hostBytes, pledges, mode, phone, off: offloadFor(metas, rb.experts) }) };
}
// what a dealt device keeps in VRAM: its layers less the parked experts, plus its cache; and in RAM the parked experts
const vramOf = (rb, d, k) => {
  const host = d.used[k] === 0 ? rb.hostBytes : 0, o = d.offload?.[k];
  return host + d.assigned[k] * rb.layerBytes - (o ? o.layers * rb.expertBytes - o.vramBytes : 0);
};

Deno.test("offload: the pledges hold the model -> the deal is the one without offload (no offload field set)", () => {
  for (const mode of ["speed", "memory"]) {
    const rb = roomBytes("qwen3.6-35b-moe", 32768);
    const pledges = [12 * GB, 12 * GB];
    const plain = dealRoom({ L: rb.L, layerBytes: rb.layerBytes, hostBytes: rb.hostBytes, pledges, mode });
    const off = dealRoom({ L: rb.L, layerBytes: rb.layerBytes, hostBytes: rb.hostBytes, pledges, mode, off: offloadFor([tab(12), pc(12, 48)], rb.expertBytes) });
    eq(off, plain, mode);
    ok(!off.offload && !off.fit.offload);
  }
});

Deno.test("offload: a 12 GB GPU runs the 35B alone with its experts in RAM, within its pledge and its RAM", () => {
  const { rb, d } = deal("qwen3.6-35b-moe", [pc(10, 48)]);
  ok(!roomFit(rb.L, [10 * GB], rb.layerBytes, rb.hostBytes).fits, "it can't hold the 35B without offload");
  ok(d.fit.fits && d.fit.offload);
  eq(d.assigned, [40]);
  const o = d.offload[0];
  ok(o && o.hi === 40 && o.lo === 40 - o.layers, "the last layers of its range are offloaded");
  ok(vramOf(rb, d, 0) <= 10 * GB + 1, "within the pledge");
  ok(o.ramBytes <= 48 * GB, "within its RAM");
  ok(o.slots >= OFFLOAD_GOOD_SLOTS, `a useful cache: ${o.slots} slots`);
  // the fewest layers that leave it that cache
  const less = offloadPlan(40, 10 * GB - rb.hostBytes, (o.layers - 1) * rb.expertBytes, rb.layerBytes, rb.expertBytes);
  ok(!less || less.slots < OFFLOAD_GOOD_SLOTS || less.layers < o.layers);
  eq(d.held, [10 * GB]);
  // no RAM offered (--ram 0, or a browser): it stays short, as before
  const { d: none } = deal("qwen3.6-35b-moe", [tab(10)]);
  ok(!none.fit.fits);
});

Deno.test("offload: two devices: the one that doesn't offload holds its whole pledge, the offloading one the rest", () => {
  // a Spark lending 6 GB hosts, the PC lends 8 GB of its 12 GB GPU and 48 GB of RAM
  const { rb, d } = deal("qwen3.6-35b-moe", [tab(6), pc(8, 48)]);
  const rc = layerCaps([6 * GB, 8 * GB], rb.layerBytes, rb.hostBytes);
  ok(sum(rc) < rb.L, "short without offload");
  ok(d.fit.fits && d.fit.offload);
  eq(d.used, [0, 1]);
  eq(d.assigned[0], rc[0], "the host: all its pledge holds");
  eq(sum(d.assigned), rb.L);
  ok(d.offload[0] === null && d.offload[1]?.layers > 0);
  eq(d.offload[1].hi, rb.L);
  ok(vramOf(rb, d, 0) <= 6 * GB && vramOf(rb, d, 1) <= 8 * GB + 1);
  // the host can offload too: then both keep within pledge and RAM
  const both = deal("qwen3.6-35b-moe", [pc(6, 8), pc(4, 8)]).d;
  ok(both.fit.fits);
  both.used.forEach((i, k) => { ok(vramOf(roomBytes("qwen3.6-35b-moe", 32768), both, k) <= [6, 4][i] * GB + 1); ok(!both.offload[k] || both.offload[k].ramBytes <= 8 * GB); });
});

Deno.test("offload: the 122B on a Spark lending 30 GB and the 5070 PC (10 GB, 48 GB of RAM)", () => {
  const { rb, d } = deal("qwen3.5-122b-moe", [tab(30), pc(10, 48)]);
  ok(d.fit.fits && d.fit.offload);
  const o = d.offload[1];
  ok(o && o.ramBytes <= 48 * GB && o.slots >= OFFLOAD_MIN_SLOTS, JSON.stringify(o));
  ok(vramOf(rb, d, 1) <= 10 * GB + 1);
  // not enough RAM for what is missing: short, nothing dealt
  const { d: shortD } = deal("qwen3.5-122b-moe", [tab(30), pc(10, 16)]);
  ok(!shortD.fit.fits && !shortD.used.length);
});

Deno.test("offloadNeed / offloadCap: never past the pledge or the RAM, monotonic", () => {
  const LB = 515e6, E = 470e6;
  eq(offloadNeed(10, 10 * LB, 0, LB, E), 0, "fits: nothing to offload");
  eq(offloadNeed(11, 10 * LB, 0, LB, E), -1, "no RAM: can't");
  for (let n = 1; n <= 60; n++) {
    const m = offloadNeed(n, 8e9, 30e9, LB, E);
    if (m > 0) {
      ok(n * LB - m * E + E + m * E * OFFLOAD_MIN_SLOTS / 256 <= 8e9 + 1, `n=${n} m=${m} within VRAM`);
      ok(m * E <= 30e9 && m <= n);
      if (m > 1) ok(n * LB - (m - 1) * E + E + (m - 1) * E * OFFLOAD_MIN_SLOTS / 256 > 8e9, "the fewest");
    }
  }
  const cap = offloadCap(200, 8e9, 30e9, LB, E);
  ok(offloadNeed(cap, 8e9, 30e9, LB, E) >= 0 && offloadNeed(cap + 1, 8e9, 30e9, LB, E) < 0);
  ok(offloadCap(200, 8e9, 60e9, LB, E) >= cap, "more RAM never holds fewer");
  ok(offloadCap(200, 9e9, 30e9, LB, E) >= cap, "a bigger pledge never holds fewer");
});

Deno.test("offload: random rooms keep every pledge and every RAM promise, and deal all layers or none", () => {
  const r = rng(7);
  for (let t = 0; t < 3000; t++) {
    const n = int(r, 1, 4), L = int(r, 4, 64), E = int(r, 50, 1400) * 1e6, LB = E + int(r, 10, 200) * 1e6, host = int(r, 0, 4000) * 1e6;
    const pledges = Array.from({ length: n }, () => int(r, 0, 40) * GB / 2);
    const ram = Array.from({ length: n }, () => (r() < 0.5 ? int(r, 0, 64) * GB : 0));
    const phone = Array.from({ length: n }, (_, i) => i > 0 && r() < 0.2);
    const off = ram.some((x) => x > 0) ? { expertBytes: E, ram } : null;
    const d = dealRoom({ L, layerBytes: LB, hostBytes: host, pledges, mode: r() < 0.5 ? "speed" : "memory", phone, off });
    const ctx = `case ${t}: L=${L} LB=${LB} E=${E} host=${host} pledges=${pledges.map((p) => p / GB)} ram=${ram.map((x) => x / GB)}`;
    if (!d.fit.fits) { eq(d.used.length, 0, ctx); continue; }
    eq(sum(d.assigned), L, ctx);
    eq(d.used[0], 0, ctx);
    d.used.forEach((i, k) => {
      const o = d.offload?.[k];
      const vram = d.assigned[k] * LB + (i === 0 ? host : 0) - (o ? o.layers * E - o.vramBytes : 0);
      ok(vram <= pledges[i] + 1, ctx + ` device ${i} VRAM ${vram} > ${pledges[i]}`);
      if (o) {
        ok(ram[i] > 0 && o.ramBytes <= ram[i] + 1, ctx + ` device ${i} RAM`);
        ok(o.layers >= 1 && o.lo >= d.ranges[k][0] && o.hi === d.ranges[k][1] && o.hi - o.lo === o.layers, ctx + " range");
        ok(o.vramBytes - E >= o.layers * E * OFFLOAD_MIN_SLOTS / 256 - 1, ctx + " the cache holds the region and the minimum slots");
      } else ok(d.assigned[k] * LB + (i === 0 ? host : 0) <= pledges[i] + 1, ctx);
    });
    if (!d.fit.offload) ok(!d.offload, ctx + " no offload when the pledges hold it");
  }
});

Deno.test("ramGB / offloadFor: only devices that offer it, held to 64 GB; browsers never", () => {
  eq(ramGB({ webgpu: true, offload: true, ramGB: 48 }), 48);
  eq(ramGB({ webgpu: true, offload: true, ramGB: 200 }), 64);
  eq(ramGB({ webgpu: true, ramGB: 48 }), 0, "no offload flag");
  eq(ramGB({ webgpu: false, offload: true, ramGB: 48 }), 0, "no GPU");
  eq(offloadFor([tab(8), tab(8)], 1e9), null);
  eq(offloadFor([tab(8), pc(8, 0)], 1e9), null);
  eq(offloadFor([tab(8), pc(8, 10)], 0), null, "a dense model");
  eq(offloadFor([tab(8), pc(8, 10)], 1e9), { expertBytes: 1e9, ram: [0, 10 * GB] });
});

Deno.test("sameShard: a different offload is a different shard", () => {
  const held = { model: "m", ctx: 8, range: [0, 4], offload: { lo: 2, hi: 4, vramBytes: 9 } };
  ok(sameShard(held, { model: "m", ctx: 8, range: [0, 4], offload: { lo: 2, hi: 4, vramBytes: 9 } }));
  ok(!sameShard(held, { model: "m", ctx: 8, range: [0, 4] }));
  ok(!sameShard(held, { model: "m", ctx: 8, range: [0, 4], offload: { lo: 1, hi: 4, vramBytes: 9 } }));
  ok(sameShard({ model: "m", ctx: 8, range: [0, 4] }, { model: "m", ctx: 8, range: [0, 4] }));
});

Deno.test("mergeSplitHeaders / expertBytesOf: a split GGUF's tensors keep their file", () => {
  const h0 = { meta: { "split.count": 2 }, tensors: { "blk.0.ffn_gate_exps.weight": { byteOffset: 10, byteLength: 5 } } };
  const h1 = { meta: {}, tensors: { "blk.0.ffn_up_exps.weight": { byteOffset: 10, byteLength: 6 }, "blk.0.ffn_down_exps.weight": { byteOffset: 20, byteLength: 7 } } };
  const G = mergeSplitHeaders([h0, h1], ["a", "b"]);
  eq(G.tensors["blk.0.ffn_up_exps.weight"], { byteOffset: 10, byteLength: 6, shard: 1, url: "b" });
  eq(G.tensors["blk.0.ffn_gate_exps.weight"].shard, undefined);
  eq(expertBytesOf(G, 0, 1), 18);
  let threw = false;
  try { mergeSplitHeaders([{ meta: { "split.count": 3 }, tensors: {} }, { meta: {}, tensors: {} }]); } catch { threw = true; }
  ok(threw, "a missing file is an error");
});
