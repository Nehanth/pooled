// A pledge is a promise: no device is dealt more layer bytes than it pledged (the model host's
// pledge also pays for the embedding, the head and the draft block), a room whose pledges can't
// hold the model is told how short it is instead of being overfilled, and a device left out says why.
// The reported case (a laptop hosting Qwen3 1.7B with 3 GB, an iPhone with 1 GB): the laptop used
// ~4 GB and the phone got nothing, because a dense layer's f32 KV cache (64 MB at 8k, more than its
// 51 MB of weights) and the host's second copy of the embedding were not counted, so the laptop
// "fit" all 28 layers; and when pledges fell short both splits spread the rest over everyone.
import { planSplit, planForSpeed, roomFit, dealRoom, layerCaps, shortNote, shortBy, gbUp } from "../../room/plan.js";
import { roomBytes, hostHeldBytes, SHAPE, NEED_GB, PICKER, maxSeqFor } from "../../room/models.js";
import { pledgeGB } from "../../room/pledge.js";
import { lendStatus } from "../../room/compute.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const sum = (a) => a.reduce((s, x) => s + x, 0);
const GB = 2 ** 30;
function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const int = (r, lo, hi) => lo + Math.floor(r() * (hi - lo + 1));
const pick = (r, xs) => xs[Math.floor(r() * xs.length)];

// the reported room, as the host sees it: pledges held to each kind's cap (room/pledge.js)
const laptop = { ua: "Device", contribGB: 3, webgpu: true };
const iphone = { ua: "iPhone", contribGB: 1, webgpu: true, phone: true };
const reported = () => {
  const rb = roomBytes("qwen3-1.7b", maxSeqFor("qwen3-1.7b"));
  return { ...rb, pledges: [pledgeGB(laptop) * GB, pledgeGB(iphone) * GB], phone: [false, true] };
};

Deno.test("reported case: a dense layer counts its f32 KV cache, the host its embedding twice", () => {
  const { L, layerBytes, hostBytes } = reported();
  eq(L, 28);
  // 51 MB of Q8 weights + 8192 positions x (K + V) x 1024 x 4 bytes = 64 MB of KV cache
  eq(layerBytes, 53494784 + 8192 * 2 * 1024 * 4);
  // the tied embedding: once in JS memory for row lookups, once on the GPU as the output head
  eq(hostBytes, 2 * 330612736);
  eq(hostHeldBytes("gguf", { embed: 10, out: 0 }), 20);
  eq(hostHeldBytes("gguf", { embed: 10, out: 7 }), 17);
  // qwen35: CPU rows, head, draft block, and the draft chain's GPU copy of the embedding
  eq(hostHeldBytes("qwen35", { embed: 10, out: 7, mtp: 3 }), 30);
  eq(hostHeldBytes("qwen35", { embed: 10, out: 7, mtp: 0 }), 17);
});

for (const mode of ["speed", "memory"]) {
  Deno.test(`reported case (${mode}): laptop 3 GB + iPhone 1 GB for Qwen3 1.7B stay within their pledges, and the phone helps`, () => {
    const r = reported();
    const d = dealRoom({ L: r.L, layerBytes: r.layerBytes, hostBytes: r.hostBytes, pledges: r.pledges, mode, phone: r.phone });
    ok(d.fit.fits, "the room fits");
    eq(d.used, [0, 1], "the phone is needed and holds layers");
    eq(sum(d.assigned), 28);
    d.used.forEach((i, k) => ok(d.held[k] <= r.pledges[i], `device ${i} holds ${(d.held[k] / GB).toFixed(2)} GB of ${(r.pledges[i] / GB).toFixed(2)}`));
    ok(d.held[0] <= 3 * GB && d.held[1] <= 1 * GB);
    eq(d.out, {});
  });
}

Deno.test("reported case, the old accounting: the laptop was dealt all 28 layers, ~3.7 GB for a 3 GB pledge", () => {
  // what room.js did before: a layer's weights only, the embedding once, and a laptop cap that "held" 54 layers
  const r = reported();
  const oldLayer = 53494784, oldHost = 330612736;
  const oldCaps = [Math.floor((r.pledges[0] - oldHost) / oldLayer), Math.floor(r.pledges[1] / oldLayer)];
  eq(oldCaps, [54, 20]);
  eq(planForSpeed(28, oldCaps, [], [false, true]).assigned, [28, 0], "the old deal: the phone left out");
  ok(28 * r.layerBytes + r.hostBytes > 3.7 * GB, "what the laptop really held");
  // the same room dealt with the real costs: the laptop within 3 GB, the phone within 1 GB
  eq(layerCaps(r.pledges, r.layerBytes, r.hostBytes), [21, 8]);
  eq(planForSpeed(28, [21, 8], [], [false, true]).assigned, [21, 7]);
});

Deno.test("short rooms: nothing is dealt past a pledge, in either split", () => {
  // for speed: everyone full and 34 layers left over used to be spread over everyone
  const s = planForSpeed(64, [10, 10, 10]);
  eq(s.short, 34); eq(s.assigned, [0, 0, 0]);
  // by memory with maxes: the same
  const m = planSplit(64, [10, 10, 10], [10, 10, 10]);
  eq(m.short, 34); eq(m.assigned, [0, 0, 0]);
  // by memory within maxes: in proportion, a full device's excess goes to the others
  eq(planSplit(10, [8, 1, 1], [4, 5, 5]).assigned, [4, 3, 3]);
  eq(planSplit(64, [62, 2], [62, 2]).assigned, [62, 2], "the MacBook + iPhone demo");
  eq(planSplit(28, [21, 8], [21, 8]).assigned, [20, 8]);
  // a device with room for no layer holds none, the host needs one
  eq(planSplit(4, [5, 0.1, 5], [3, 0, 3]).assigned, [2, 0, 2]);
  eq(planSplit(4, [0, 5], [0, 9]).short, 1);
});

Deno.test("a room too small for the model: how short, and who could close the gap", () => {
  // the reported laptop and phone, for the 27B: many GB short; the laptop could give that much, the phone is at its cap
  const rb = roomBytes("qwen3.8-27b", maxSeqFor("qwen3.8-27b"));
  const pledges = [3 * GB, 1 * GB];
  const f = roomFit(rb.L, pledges, rb.layerBytes, rb.hostBytes);
  ok(!f.fits);
  ok(f.raise[1] > 10 * GB, "the phone alone would need more than 10 GB more");
  const note = shortNote("Qwen3.8 27B", f, ["Laptop", "iPhone"], [60, 0]);
  ok(note.startsWith(`This room is ${gbUp(shortBy(f, [60, 0]))} GB short for Qwen3.8 27B`), note);
  // the laptop lends 2 GB for the 1.7B: the phone (at its cap) would need less, but can't give it, so
  // the headline and the laptop's ask are the same number
  const r = reported();
  const f2 = roomFit(r.L, [2 * GB, 1 * GB], r.layerBytes, r.hostBytes);
  ok(!f2.fits && f2.raise[1] < f2.raise[0]);
  eq(shortNote("Qwen3 1.7B", f2, ["Laptop", "iPhone"], [6, 0]), `This room is ${gbUp(f2.raise[0])} GB short for Qwen3 1.7B. Add a device or raise a pledge: Laptop could give ${gbUp(f2.raise[0])} GB more.`);
  eq(shortBy(f2, [0, 0]), f2.short, "nobody can give: the smallest raise");
  ok(/\. Add a device or raise a pledge: Laptop could give [\d.]+ GB more\.$/.test(note), note);
  // a host whose pledge can't hold the embedding, the head and a layer: only the host can fix it
  const h = roomFit(10, [4 * GB, 100 * GB], GB, 4 * GB);
  ok(!h.fits); ok(h.raise[1] === Infinity); eq(gbUp(h.short), 1);
  ok(/the model host \(Laptop\) needs room for the embedding, the output head and a layer\. Add a device or raise a pledge: Laptop could give 1 GB more\.$/.test(shortNote("M", h, ["Laptop", "Desk"], [8, 8])), shortNote("M", h, ["Laptop", "Desk"], [8, 8]));
  // raising the laptop by what the note says makes it fit
  ok(roomFit(rb.L, [3 * GB + gbUp(f.raise[0]) * GB, 1 * GB], rb.layerBytes, rb.hostBytes).fits);
  // nobody can give more: the note still says what to do, and names nobody
  eq(shortNote("X", roomFit(10, [2, 2], 1, 1), ["a", "b"], [0, 0]), "This room is 0.1 GB short for X. Add a device or raise a pledge.");
  // a room that fits has nothing to say
  eq(shortNote("X", roomFit(2, [2, 2], 1, 1), ["a", "b"], [0, 0]), "");
});

Deno.test("picker shapes: SHAPE matches NEED_GB within reason for every picker model", () => {
  for (const k of PICKER) {
    ok(SHAPE[k], k + " has a SHAPE");
    const rb = roomBytes(k, maxSeqFor(k));
    const need = (rb.L * rb.layerBytes + rb.hostBytes) / GB;
    // NEED_GB (the ladder's label) is at least what the deal needs, and not far above it
    ok(need <= NEED_GB[k] && need >= NEED_GB[k] * 0.85, `${k}: deal needs ${need.toFixed(2)} GB, NEED_GB ${NEED_GB[k]}`);
  }
  // the reported pair fits the 1.7B (4 GB pooled) by the deal, as the picker says
  const r = reported();
  ok(roomFit(r.L, r.pledges, r.layerBytes, r.hostBytes).fits);
});

Deno.test("properties: over 4000 random rooms no device is dealt more than it pledged", () => {
  const r = rng(0x91ed6e);
  for (let t = 0; t < 4000; t++) {
    const n = int(r, 1, 7);
    const L = pick(r, [int(r, 1, 8), 28, 36, 40, 64, int(r, 1, 100)]);
    const layerBytes = pick(r, [120603648, 240747680, 515e6, int(r, 1, 50) * 1e6]);
    const hostBytes = pick(r, [0, 661225472, 2738e6, int(r, 0, 3000) * 1e6]);
    const pledges = Array.from({ length: n }, () => pick(r, [0.5, 1, 2, 3, 4, 8, 12, 16, 24, 64, r() * 20, 0, NaN]) * GB);
    const phone = pledges.map((_, i) => i > 0 && r() < 0.3);
    const mode = pick(r, ["speed", "memory"]);
    const ms = pick(r, [[], pledges.map(() => (r() < 0.5 ? null : 0.5 + r() * 20))]);
    const phoneLayers = r() < 0.2;
    const ctx = `case ${t}: dealRoom(L=${L}, layer=${layerBytes}, host=${hostBytes}, pledges=${JSON.stringify(pledges.map((p) => p / GB))}, ${mode}, phone=${JSON.stringify(phone)})`;
    const d = dealRoom({ L, layerBytes, hostBytes, pledges, mode, ms, phone, phoneLayers });
    const p = (i) => (Number.isFinite(pledges[i]) && pledges[i] > 0 ? pledges[i] : 0);
    if (!d.fit.fits) {
      eq(d.used, [], ctx + " a short room deals nothing");
      ok(d.fit.short > 0 && d.fit.short < Infinity, ctx + " says how short");
      // raising the device with the smallest raise by that much makes it fit
      const i = d.fit.raise.indexOf(d.fit.short);
      const more = pledges.map((x, j) => p(j) + (j === i ? d.fit.short : 0));
      ok(roomFit(L, more, layerBytes, hostBytes).fits, ctx + ` raising device ${i} by ${d.fit.short} fits`);
      continue;
    }
    eq(d.used[0], 0, ctx + " the host holds layers");
    eq(sum(d.assigned), L, ctx + " every layer placed once");
    let acc = 0;
    d.ranges.forEach(([lo, hi], k) => { eq(lo, acc, ctx + " contiguous"); eq(hi - lo, d.assigned[k], ctx + " range = count"); acc = hi; });
    d.used.forEach((i, k) => {
      ok(d.assigned[k] >= 1, ctx + ` device ${i} is in the chain with no layer`);
      const bytes = d.assigned[k] * layerBytes + (i === 0 ? hostBytes : 0);
      eq(d.held[k], bytes, ctx + " held");
      ok(bytes <= p(i), ctx + ` device ${i} holds ${bytes} > pledge ${p(i)}`);
    });
    // every device is either used or says why not
    for (let i = 1; i < n; i++) ok(d.used.includes(i) !== (i in d.out), ctx + ` device ${i} used or out`);
    // phones left out by memory only while the computers hold the model
    if (mode === "memory" && !phoneLayers) for (const [i, why] of Object.entries(d.out)) if (why === "unneeded" && phone[i]) {
      const comp = sum(d.fit.caps.filter((_, j) => j === 0 || !phone[j]));
      ok(comp >= L, ctx + " a phone left out while the computers can't hold the model");
    }
  }
});

Deno.test("properties: planSplit with maxes never exceeds a max, over 3000 random rooms", () => {
  const r = rng(0xca95);
  for (let t = 0; t < 3000; t++) {
    const n = int(r, 1, 6), L = pick(r, [int(r, 1, n + 1), int(r, 1, 130), 28, 64]);
    const max = Array.from({ length: n }, () => pick(r, [0, 1, 2, int(r, 0, 10), int(r, 0, 80)]));
    const caps = max.map((m) => pick(r, [m * 1e6, r() * 1e9, 0, NaN]));
    const ctx = `case ${t}: planSplit(${L}, ${JSON.stringify(caps)}, ${JSON.stringify(max)})`;
    const p = planSplit(L, caps, max);
    if (max[0] < 1 || sum(max) < L) { ok(p.short > 0, ctx + " short"); eq(sum(p.assigned), 0, ctx + " nothing dealt"); continue; }
    eq(p.short, 0, ctx);
    eq(sum(p.assigned), L, ctx + " every layer once");
    p.assigned.forEach((a, i) => ok(a <= max[i], ctx + ` device ${i}: ${a} > ${max[i]}`));
    ok(p.assigned[0] >= 1, ctx + " host keeps a layer");
  }
});

Deno.test("lend screen: a device holding no layers says why", () => {
  const base = { phase: "serving", lo: null, hi: null, model: "Qwen3 1.7B" };
  eq(lendStatus({ ...base, out: "unneeded", phone: true }), { title: "Not needed", sub: "The computers hold Qwen3 1.7B, so this phone stays free: it can still ask." });
  eq(lendStatus({ ...base, out: "unneeded" }).sub, "The other devices hold Qwen3 1.7B, so this one stays free: it can still ask.");
  eq(lendStatus({ ...base, out: "late" }).title, "Waiting for a re-deal");
  eq(lendStatus({ ...base, out: "small" }).title, "Not holding layers");
  eq(lendStatus({ ...base, phase: "loading", out: "unneeded", phone: true }).title, "Not needed");
  // holding layers: what it holds of what it pledged
  eq(lendStatus({ phase: "serving", lo: 21, hi: 28, model: "Qwen3 1.7B", held: 0.786, pledge: 1 }).sub, "Layers 22–28 · 0.8 of 1 GB · Qwen3 1.7B");
});
