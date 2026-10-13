// Qwen3.8-Flash-Next with expert offload in the room's deal (room/plan.js; room/models.js flashOffload; room/pledge.js
// offloadFor diskOnly; room/flashdisk.js tabOffloadLayers): a browser tab with the OPFS disk tier (meta.tabDisk) never
// parks experts and reads every layer it offloads from disk, with the layers it offloads chosen by tabDiskCost; Flash's
// store has no whole-layer region; every pledge and every RAM promise holds.
import { dealRoom, roomFit, offloadPlan, poolHitRate, tabDiskCost, OFFLOAD_MIN_SLOTS } from "../../room/plan.js";
import { roomBytes, ngramOf, SHAPE, flashOffload, FLASH_CKPT_TOKENS } from "../../room/models.js";
import { offloadFor, ngramFor, pledgeGB, ramGB, tabDiskGB } from "../../room/pledge.js";
import { tabOffloadLayers, tabDiskChoice, saveTabDiskChoice, tabDiskMeta, tabDiskGate, tabRamCap, TAB_DISK_KEY } from "../../room/flashdisk.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const GB = 2 ** 30, F = "qwen3.8-flash", M35 = "qwen3.6-35b-moe", M122 = "qwen3.5-122b-moe";
const tab = (gb, ram = 3, ask = false) => ({ webgpu: true, contribGB: gb, tabDisk: { ramGB: ram, models: [F], ...(ask ? { ask: true } : {}) } });
const plainTab = (gb) => ({ webgpu: true, contribGB: gb });
const pc = (gb, ram, disk = false, local = [F, M35, M122]) => ({ webgpu: true, contribGB: gb, offload: true, ramGB: ram, offloadFlash: true, ...(disk ? { disk: true, local } : {}) });
// a released @pooled/cli 0.4.1 node: it offloads the Qwen 3.5 MoEs but its shard.js refuses Flash's experts
const pc041 = (gb, ram, disk = false, local = [F, M35, M122]) => { const m = pc(gb, ram, disk, local); delete m.offloadFlash; return m; };
function deal(model, metas, ctx = 2048) {
  const rb = roomBytes(model, ctx);
  const d = dealRoom({ L: rb.L, layerBytes: rb.layerBytes, hostBytes: rb.hostBytes, pledges: metas.map((m) => pledgeGB(m) * GB), mode: "memory",
    off: offloadFor(metas, rb.experts, model), ngram: ngramFor(metas, ngramOf(model)) });
  return { rb, d };
}
const perExp = (rb) => rb.experts.slices[47].reduce((a, b) => a + b, 0);

Deno.test("flash profile: no region, Flash's own K / V, state and checkpoint rows, K = 10", () => {
  const p = roomBytes(F, 2048).experts;
  eq([p.nExp, p.K, p.region, p.ckptTokens], [512, 10, false, FLASH_CKPT_TOKENS]);
  eq(p.kvPos, 1024, "2 KV heads x 256 x f32 x K+V on 12 of 48 layers");
  eq(p.state, (48 * 128 * 128 * 4 + (2 * 16 * 128 + 6144) * 3 * 4) * 36 / 48, "the DeltaNet layers' S and conv window, averaged");
  eq(flashOffload(SHAPE[F].flash), { kvPos: p.kvPos, state: p.state, work: p.work, region: false, ckptTokens: FLASH_CKPT_TOKENS, K: 10 });
  eq(p.slices.length, 48);
});

Deno.test("flash: the Mac tab alone (22 GB pledge, OPFS disk tier) holds all 48 layers, its experts on disk", () => {
  const { rb, d } = deal(F, [tab(22)]);
  ok(d.fit.fits && d.fit.disk && d.fit.offload, JSON.stringify(d.fit));
  eq(d.ranges, [[0, 48]]);
  const o = d.offload[0];
  ok(o.diskOnly && o.ramBytes === 0 && o.diskLo === o.lo && o.hi === 48, `every offloaded layer reads from disk, none parked: ${JSON.stringify(o)}`);
  ok(o.layers >= 40, `it offloads (nearly) every layer, not the fewest: ${o.layers}`);
  ok(o.slots >= 100, `a pool of ${o.slots} experts per layer`);
  // the pledge holds: resident layers + the pools + the spare (no region: Flash's store has none)
  const res = (48 - o.layers) * rb.layerBytes + o.layers * (rb.layerBytes - rb.experts.E) + rb.hostBytes;
  ok(res + o.vramBytes + o.vramSpare <= 22 * GB + 1, `${(res + o.vramBytes + o.vramSpare) / GB} GB of 22`);
  const pes = rb.experts.slices.slice(o.lo, o.hi).reduce((a, sl) => a + sl.reduce((x, y) => x + y, 0), 0);
  ok(o.vramBytes - o.slots * pes < pes + 1, "the whole cache is pools: no region set aside");
  // its RAM: the spare and T1 within ramGB
  ok(o.ramSpare + o.diskRamBytes <= 3 * GB + 1 && o.diskRamBytes >= 0);
  eq(d.ngram?.at, 0, "the host holds the n-gram layer");
});

Deno.test("flash: offloading more layers beats the fewest on a tab (tabDiskCost, the recorded hit curve)", () => {
  ok(poolHitRate(0) === 0 && poolHitRate(1) === 1 && Math.abs(poolHitRate(0.25) - 0.854) < 1e-9 && poolHitRate(0.2) > poolHitRate(0.17));
  const rb = roomBytes(F, 2048), bytes = 22 * GB - rb.hostBytes, pe = perExp(rb);
  const fewest = offloadPlan(48, bytes, 8 * GB, rb.layerBytes, rb.experts, OFFLOAD_MIN_SLOTS, 512, 48, true, false);   // (a node's: T1's 2 GiB minimum)
  const tabp = offloadPlan(48, bytes, 3 * GB, rb.layerBytes, rb.experts, OFFLOAD_MIN_SLOTS, 512, 48, true, true);
  ok(tabp.layers > fewest.layers, `${tabp.layers} > ${fewest.layers}`);
  ok(tabDiskCost(tabp.layers, tabp.slots, 512, 10, pe) < tabDiskCost(fewest.layers, fewest.slots, 512, 10, pe) / 2, "at least twice cheaper");
});

Deno.test("flash: a tab with the disk tier never parks, and counts only for Flash", () => {
  // parking pass: the tab is pledge-only; Mac + PC with --ram 43: the PC parks, the tab holds what its pledge holds
  const { rb, d } = deal(F, [tab(22), pc(10, 43)]);
  ok(d.fit.fits && !d.fit.disk, "parking holds it");
  ok(d.offload[0] === null, "the tab offloads nothing");
  eq(d.assigned[0], Math.floor((22 * GB - rb.hostBytes) / rb.layerBytes), "the tab: its pledge's layers");
  ok(d.offload[1] && d.offload[1].ramBytes > 0 && !Number.isInteger(d.offload[1].diskLo));
  // the 35B and the 122B: a tab's disk tier is Flash only, so the deal is the one without it, field for field
  for (const [model, metas, plain] of [[M35, [tab(12), pc(8, 48)], [plainTab(12), pc(8, 48)]], [M122, [tab(26), pc(10, 24, true)], [plainTab(26), pc(10, 24, true)]], [M35, [tab(30)], [plainTab(30)]]]) {
    const a = deal(model, metas, 32768).d, b = deal(model, plain, 32768).d;
    eq(a, b, model);
  }
  // a short Flash room without any disk tier stays short
  ok(!deal(F, [plainTab(22)]).d.fit.fits);
  eq(tabDiskGB(tab(22, 3), F), 3);
  eq(ramGB(tab(22, 3)), 0, "a tab never offers RAM to park experts in (meta.offload / ramGB stay unset)");
  eq(tabDiskGB(tab(22, 3), M35), 0, "nor a disk tier for another model");
  eq(offloadFor([tab(22), pc(10, 24)], 1e9, F).diskOnly, [true, false]);
  eq(offloadFor([plainTab(22), pc(10, 24)], 1e9, F).diskOnly, undefined, "no disk-only device: no field");
});

Deno.test("flash: Mac tab + PC (--ram 24 --disk): both offload, the tab from OPFS, the PC parks then reads its last layers from disk", () => {
  const { d } = deal(F, [tab(22), pc(10, 24, true)]);
  ok(d.fit.fits && d.fit.disk);
  const [t, p] = d.offload;
  ok(t?.diskOnly && t.ramBytes === 0 && t.diskLo === t.lo, JSON.stringify(t));
  ok(p && p.ramBytes > 0 && p.diskLo > p.lo && p.hi === 48, JSON.stringify(p));
  ok(p.ramBytes + p.ramSpare + p.diskRamBytes <= 24 * GB + 1, "the PC's RAM promise");
  eq(d.ngram?.at, 0);
});

Deno.test("flashdisk: tabOffloadLayers takes disk layers only", () => {
  eq(tabOffloadLayers(null, [0, 48]), null);
  eq(tabOffloadLayers({ lo: 46, hi: 48, diskLo: 46, vramBytes: 1 }, [0, 48]), { layers: [46, 47], disk: [46, 47] });
  eq(tabOffloadLayers({ lo: 46, hi: 48, diskLo: 46, vramBytes: 0 }, [0, 48]), null, "no cache: nothing");
  let threw = false;
  try { tabOffloadLayers({ lo: 40, hi: 48, diskLo: 44, vramBytes: 1 }, [0, 48]); } catch { threw = true; }
  ok(threw, "a plan that parks layers 40-43 is refused");
  threw = false;
  try { tabOffloadLayers({ lo: 40, hi: 48, vramBytes: 1 }, [0, 48]); } catch { threw = true; }
  ok(threw, "no disk layers at all: refused");
});

Deno.test("flash: a small tab pledge (12 GB, the Spark gate) still holds every layer from disk", () => {
  const { d } = deal(F, [tab(12)]);
  ok(d.fit.fits && d.fit.disk);
  const o = d.offload[0];
  ok(o.slots >= OFFLOAD_MIN_SLOTS && o.diskOnly, JSON.stringify(o));
  // and roomFit agrees with the deal
  const rb = roomBytes(F, 2048);
  ok(roomFit(rb.L, [12 * GB], rb.layerBytes, rb.hostBytes, offloadFor([tab(12)], rb.experts, F)).fits);
});

Deno.test("flash: the tab's disk tier is its own meta field, which a host from before it ignores", () => {
  // a released host (@pooled/cli 0.4.1) reads meta.offload / meta.ramGB: the tab sets neither, so it never deals a tab
  // an offload (as with a plain tab)
  const t = tab(12);
  ok(!t.offload && t.ramGB == null && !t.diskOnly && !t.disk && !t.local, JSON.stringify(t));
  // a tab whose user was not asked yet counts as one with the tier (it asks when the deal arrives)
  const a = deal(F, [tab(22, 3, true)]).d, b = deal(F, [tab(22, 3)]).d;
  eq(a, b, "ask: true deals as a yes would");
  ok(a.offload[0]?.diskOnly);
  // non-Flash models: the deal is the plain tab's, field for field (the reviewer's fuzz, scratchpad rv368_fuzz.mjs)
  for (const model of [M35, M122, "qwen3-1.7b", "qwen3.8-27b"]) for (const ctx of [2048, 32768])
    eq(deal(model, [tab(12), pc(8, 24, true)], ctx).d, deal(model, [plainTab(12), pc(8, 24, true)], ctx).d, `${model} @ ${ctx}`);
});

// ---------------------------------------------------------------- consent

const store = (init = {}) => { const m = new Map(Object.entries(init)); return { getItem: (k) => m.get(k) ?? null, setItem: (k, v) => m.set(k, String(v)), removeItem: (k) => m.delete(k), m }; };

Deno.test("tab disk consent: the answer is remembered per origin, ?tabdisk=0 says no for one page, the URL never says yes", () => {
  const st = store();
  eq(tabDiskChoice({ storage: st }), null, "not asked yet");
  saveTabDiskChoice("allow", st);
  eq(st.m.get(TAB_DISK_KEY), "allow");
  eq(tabDiskChoice({ storage: st }), "allow");
  eq(tabDiskChoice({ search: "?tabdisk=0", storage: st }), "deny", "the page's answer wins");
  saveTabDiskChoice("deny", st);
  eq(tabDiskChoice({ storage: st }), "deny");
  saveTabDiskChoice(null, st);
  // the URL can only say no: a host's link with ?tabdisk=1 (room links keep their query) does not skip the question
  eq(tabDiskChoice({ search: "?tabdisk=1", storage: st }), null, "?tabdisk=1 is no answer");
  saveTabDiskChoice("deny", st);
  eq(tabDiskChoice({ search: "?tabdisk=1", storage: st }), "deny", "nor does it beat a stored no");
  saveTabDiskChoice(null, st);
  eq(tabDiskChoice({ storage: st }), null);
  eq(tabDiskChoice({ storage: { getItem() { throw new Error("private mode"); } } }), null, "no storage: not asked");
});

Deno.test("tab disk consent: a tab advertises the tier only as its user answered (ask until then, nothing after a no)", () => {
  const mac = { webgpu: true, phone: false };
  eq(tabDiskMeta(mac, { opfs: true }), { ramGB: 3, models: [F], ask: true });
  eq(tabDiskMeta(mac, { opfs: true, choice: "allow" }), { ramGB: 3, models: [F] });
  eq(tabDiskMeta(mac, { opfs: true, choice: "deny" }), null);
  eq(tabDiskMeta(mac, { opfs: false, choice: "allow" }), null, "no OPFS");
  eq(tabDiskMeta({ webgpu: true, phone: true }, { opfs: true, choice: "allow" }), null, "phones never");
  eq(tabDiskMeta(mac, { opfs: true, choice: "allow", search: "?tabram=0" }), null, "?tabram=0 turns it off");
  // ?tabram is held to what the device can spare
  eq(tabDiskMeta(mac, { opfs: true, choice: "allow", search: "?tabram=40" }).ramGB, 16);
  eq(tabDiskMeta(mac, { opfs: true, choice: "allow", search: "?tabram=6", deviceMemory: 4 }).ramGB, 2, "half of a 4 GB device");
  eq(tabRamCap(8), 16, "deviceMemory 8 means 8 or more");
});

Deno.test("tab disk consent: the receiving side refuses an offload it did not offer or its user turned down", () => {
  const off = { lo: 0, hi: 48, diskLo: 0, vramBytes: 6 * GB, ramBytes: 0, diskRamBytes: GB, ramSpare: 1.6 * GB };
  const offered = tabDiskMeta({ webgpu: true }, { opfs: true, choice: "allow" });
  eq(tabDiskGate(null, { model: F }), null, "no offload: nothing to refuse");
  eq(tabDiskGate(off, { model: F, tabDisk: offered, choice: "allow" }), null, "offered and allowed: go");
  eq(tabDiskGate(off, { model: F, tabDisk: { ...offered, ask: true }, choice: null }), "ask", "not asked yet: ask");
  ok(/did not offer its disk/.test(tabDiskGate(off, { model: F, tabDisk: null, choice: null })), "?tabdisk=0 / a plain tab");
  ok(/did not offer its disk/.test(tabDiskGate(off, { model: F, tabDisk: null, choice: "allow" })), "allowed but not offered (?tabram=0)");
  ok(/did not allow/.test(tabDiskGate(off, { model: F, tabDisk: offered, choice: "deny" })));
  ok(/did not offer its disk for qwen3.6-35b-moe/.test(tabDiskGate(off, { model: M35, tabDisk: offered, choice: "allow" })), "another model");
  ok(/RAM for the disk tier/.test(tabDiskGate({ ...off, diskRamBytes: 8 * GB }, { model: F, tabDisk: offered, choice: "allow" })), "more RAM than it offered");
});

Deno.test("flash: a 0.4.1 node (no meta.offloadFlash) is never dealt Flash offload, and still serves layers in its pledge", () => {
  const { rb, d } = deal(F, [tab(22), pc041(10, 43, true)]);
  ok(!d.offload?.[1], `no offload for the old node: ${JSON.stringify(d.offload)}`);
  if (d.fit.fits) ok(d.assigned[1] <= Math.floor(10 * GB / rb.layerBytes), "within its pledge");
  // the tab offloads from disk instead, so the room still fits
  ok(d.fit.fits && d.offload[0]?.diskOnly, JSON.stringify(d.fit));
  eq(offloadFor([pc041(10, 43, true)], 1e9, F), null, "alone: no device offloads Flash");
  ok(offloadFor([pc(10, 43, true)], 1e9, F)?.ram[0] > 0, "a new node does");
  // the Qwen 3.5 MoEs: the old node offloads as it always did, field for field
  for (const model of [M35, M122]) eq(deal(model, [plainTab(12), pc041(8, 24, true)], 32768).d, deal(model, [plainTab(12), pc(8, 24, true)], 32768).d, model);
});
