// Expert offload's disk tier in the room's deal (room/plan.js DISK_RAM_MIN, offloadNeedDisk, offloadPlan disk;
// room/pledge.js diskFor): a room node started with --disk, holding the model's files, may leave the experts of the
// last layers of its offloaded range in the GGUF. Disk is the last resort: a room that fits without it deals exactly
// as before, field for field. With it, every pledge holds (the VRAM pays the same pool and region for a disk layer)
// and every RAM promise holds: parked experts + the spare + T1 (at least DISK_RAM_MIN) within what the device lends.
import { roomFit, dealRoom, offloadNeed, offloadNeedDisk, offloadCap, offloadPlan, DISK_RAM_MIN, OFFLOAD_MIN_SLOTS } from "../../room/plan.js";
import { roomBytes } from "../../room/models.js";
import { ramGB, offloadFor, pledgeGB, diskFor } from "../../room/pledge.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const sum = (a) => a.reduce((s, x) => s + x, 0);
const GB = 2 ** 30;
function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const int = (r, lo, hi) => lo + Math.floor(r() * (hi - lo + 1));

const M35 = "qwen3.6-35b-moe", M122 = "qwen3.5-122b-moe";
const pc = (gb, ram, disk = false, local = [M35, M122]) => ({ webgpu: true, contribGB: gb, offload: true, ramGB: ram, ...(disk ? { disk: true, local } : {}) });
const tab = (gb) => ({ webgpu: true, contribGB: gb });
function deal(model, metas, { ctx = 32768, mode = "speed" } = {}) {
  const rb = roomBytes(model, ctx);
  const pledges = metas.map((m) => pledgeGB(m) * GB);
  const args = { L: rb.L, layerBytes: rb.layerBytes, hostBytes: rb.hostBytes, pledges, mode };
  return { rb, pledges, d: dealRoom({ ...args, off: offloadFor(metas, rb.experts, model) }), plain: dealRoom({ ...args, off: offloadFor(metas, rb.experts) }) };
}
// a dealt device's RAM: what it parks, keeps free, and gives T1
const ramOf = (o) => (o ? o.ramBytes + (o.ramSpare || 0) + (o.diskRamBytes || 0) : 0);

Deno.test("disk: diskFor / offloadFor: only a device that offloads, says --disk and holds the model's files", () => {
  ok(diskFor(pc(10, 24, true), M35));
  ok(!diskFor(pc(10, 24, true), "qwen3-1.7b"), "the model is not local");
  ok(!diskFor(pc(10, 24), M35), "no --disk");
  ok(!diskFor(pc(10, 0, true), M35), "no RAM offered");
  ok(!diskFor({ webgpu: true, contribGB: 10, disk: true, local: [M35] }, M35), "a device that doesn't offload");
  ok(!diskFor(pc(10, 24, true), null), "no model named");
  eq(offloadFor([tab(8), pc(8, 10, true)], 1e9), { expertBytes: 1e9, ram: [0, 10 * GB] }, "no model: as before");
  eq(offloadFor([tab(8), pc(8, 10, true)], 1e9, M35), { expertBytes: 1e9, ram: [0, 10 * GB], disk: [false, true] });
  eq(offloadFor([tab(8), pc(8, 10)], 1e9, M35), { expertBytes: 1e9, ram: [0, 10 * GB] }, "nobody has a disk tier: no field");
  eq(ramGB(pc(8, 10, true)), 10);
});

Deno.test("disk: a room that fits without disk deals exactly as before (real models, --disk on every PC)", () => {
  for (const [model, metas] of [[M35, [pc(10, 48, true)]], [M35, [tab(6), pc(8, 48, true)]], [M122, [tab(30), pc(10, 48, true)]], [M35, [tab(12), pc(12, 48, true)]]]) {
    const { d, plain } = deal(model, metas);
    ok(d.fit.fits && !d.fit.disk, `${model}: fits without disk`);
    eq(d, plain, `${model} ${JSON.stringify(metas)}`);
  }
});

Deno.test("disk: the 35B on a 10 GB GPU with 8 GB of RAM: short with parking, fits with --disk", () => {
  const { rb, d, plain } = deal(M35, [pc(10, 8, true)]);
  ok(!plain.fit.fits, "parking alone can't: 8 GB of RAM");
  ok(d.fit.fits && d.fit.disk && d.fit.offload, "the disk tier holds it");
  eq(d.assigned, [rb.L]);
  const o = d.offload[0];
  ok(o.diskLo > o.lo && o.diskLo < o.hi && o.hi === rb.L, `parked layers, then disk layers: ${JSON.stringify(o)}`);
  ok(ramOf(o) <= 8 * GB + 1, `RAM ${ramOf(o) / GB}`);
  ok(o.diskRamBytes >= DISK_RAM_MIN, "T1 at least its minimum");
  ok(o.slots >= OFFLOAD_MIN_SLOTS);
  // the fewest on disk: one layer fewer would not fit its RAM
  const fewer = offloadNeedDisk(rb.L, 10 * GB - rb.hostBytes, 8 * GB, rb.layerBytes, rb.experts, OFFLOAD_MIN_SLOTS, 256, rb.L);
  eq(fewer, { m: o.layers, d: o.hi - o.diskLo });
  // no --disk (or the files not here): short, as before
  ok(!deal(M35, [pc(10, 8, true, ["qwen3-1.7b"])]).d.fit.fits);
  ok(!deal(M35, [pc(10, 8)]).d.fit.fits);
  // too little RAM even for T1 and the spare: short
  ok(!deal(M35, [pc(10, 2, true)]).d.fit.fits);
});

Deno.test("disk: the 122B on a Mac (26 GB tab) + the 5070 PC (10 GB, --ram 24 --disk): the PC's last layers on disk", () => {
  const { d, plain } = deal(M122, [tab(26), pc(10, 24, true)]);
  ok(!plain.fit.fits, "--ram 24 alone is short");
  ok(d.fit.fits && d.fit.disk);
  eq(d.used, [0, 1]);
  ok(d.offload[0] === null, "the tab never offloads");
  const o = d.offload[1];
  ok(o && o.diskLo > o.lo && o.hi === 48 && ramOf(o) <= 24 * GB + 1, JSON.stringify(o));
  ok(d.assigned[0] === d.fit.offloadCaps[0], "the Mac holds all its pledge does first");
});

Deno.test("disk: random rooms: identical deals when parking holds the model, every promise kept otherwise", () => {
  const r = rng(11);
  let withDisk = 0, same = 0;
  for (let t = 0; t < 4000; t++) {
    const n = int(r, 1, 4), L = int(r, 4, 64), E = int(r, 50, 1400) * 1e6, LB = E + int(r, 10, 200) * 1e6, host = int(r, 0, 4000) * 1e6;
    const pledges = Array.from({ length: n }, () => int(r, 0, 40) * GB / 2);
    const ram = Array.from({ length: n }, () => (r() < 0.6 ? int(r, 0, 64) * GB : 0));
    const disk = ram.map((x) => x > 0 && r() < 0.6);
    if (!ram.some((x) => x > 0)) continue;
    const mode = r() < 0.5 ? "speed" : "memory", args = { L, layerBytes: LB, hostBytes: host, pledges, mode };
    const plain = dealRoom({ ...args, off: { expertBytes: E, ram } });
    const d = dealRoom({ ...args, off: { expertBytes: E, ram, ...(disk.some(Boolean) ? { disk } : {}) } });
    const ctx = `case ${t}: L=${L} LB=${LB} E=${E} host=${host} pledges=${pledges.map((p) => p / GB)} ram=${ram.map((x) => x / GB)} disk=${disk}`;
    if (plain.fit.fits) { eq(d, plain, ctx + " (parking holds it: the same deal)"); same++; continue; }
    if (!d.fit.fits) { eq(d.used.length, 0, ctx); continue; }
    withDisk++;
    ok(d.fit.disk, ctx);
    eq(sum(d.assigned), L, ctx);
    d.used.forEach((i, k) => {
      const o = d.offload?.[k];
      const vram = d.assigned[k] * LB + (i === 0 ? host : 0) - (o ? o.layers * E - o.vramBytes : 0);
      ok(vram <= pledges[i] + 1, ctx + ` device ${i} VRAM ${vram} > ${pledges[i]}`);
      if (!o) { ok(d.assigned[k] * LB + (i === 0 ? host : 0) <= pledges[i] + 1, ctx); return; }
      ok(ramOf(o) <= ram[i] + 1, ctx + ` device ${i} RAM ${ramOf(o)} > ${ram[i]}`);
      ok(o.hi === d.ranges[k][1] && o.lo === o.hi - o.layers && o.lo >= d.ranges[k][0], ctx + " range");
      ok(o.vramBytes - E >= o.layers * E * OFFLOAD_MIN_SLOTS / 256 - 1, ctx + " the cache holds the region and the minimum slots");
      if (o.diskLo != null) {
        ok(disk[i], ctx + ` device ${i} has no disk tier`);
        ok(o.diskLo >= o.lo && o.diskLo < o.hi && o.diskLayers === o.hi - o.diskLo, ctx + " disk range");
        ok(o.diskRamBytes >= DISK_RAM_MIN, ctx + " T1");
        // the fewest on disk: one more parked layer would not fit
        ok(o.ramBytes + o.ramSpare + DISK_RAM_MIN <= ram[i] + 1, ctx);
      } else ok(o.diskRamBytes == null, ctx);
    });
  }
  ok(withDisk > 50 && same > 500, `exercised: ${withDisk} disk deals, ${same} identical`);
});

Deno.test("disk: caps never shrink with disk, more RAM or a bigger pledge", () => {
  const r = rng(3);
  for (let t = 0; t < 400; t++) {
    const L = int(r, 4, 64), E = int(r, 50, 1400) * 1e6, LB = E + int(r, 10, 200) * 1e6;
    const bytes = int(r, 1, 40) * GB / 2, ram = int(r, 0, 64) * GB, ctx = `case ${t}: L=${L} E=${E} LB=${LB} bytes=${bytes / GB} ram=${ram / GB}`;
    const c0 = offloadCap(L, bytes, ram, LB, E), c1 = offloadCap(L, bytes, ram, LB, E, OFFLOAD_MIN_SLOTS, null, true);
    ok(c1 >= c0, ctx + ` disk ${c1} < ${c0}`);
    ok(offloadCap(L, bytes, ram + 8 * GB, LB, E, OFFLOAD_MIN_SLOTS, null, true) >= c1, ctx + " more RAM");
    ok(offloadCap(L, bytes + GB, ram, LB, E, OFFLOAD_MIN_SLOTS, null, true) >= c1, ctx + " bigger pledge");
    // with disk the RAM no longer bounds the layers once the spare and T1 fit: only the VRAM does
    if (ram >= DISK_RAM_MIN) ok(c1 === offloadCap(L, bytes, 1e15, LB, E), ctx + ` disk caps are the VRAM's: ${c1}`);
    // a plan for what the cap holds always exists, and has no disk layers when parking alone holds it
    if (c1 > Math.floor(bytes / LB)) {
      const p = offloadPlan(c1, bytes, ram, LB, E, OFFLOAD_MIN_SLOTS, 256, null, true);
      ok(p && p.layers >= 1, ctx + " plan");
      if (offloadNeed(c1, bytes, ram, LB, E) > 0) eq(p, offloadPlan(c1, bytes, ram, LB, E), ctx + " parking holds it: the plan without disk");
    }
  }
});
