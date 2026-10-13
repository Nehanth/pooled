// The picker's words for a model kept on disk (room/diskpick.js): every amount from room/models.js DISK_COPY, so a
// future big model gets the same row, cost figures, consent and Start label from its own entry. The consent state is
// #368's stored answer (room/flashdisk.js TAB_DISK_KEY), read through room/diskstore.js diskState.
import { DISK_COPY, MODELS, PICKER, SHAPE, diskCopyBytes } from "../../room/models.js";
import { diskCopyOf, rowChip, rowChipHTML, paramsText, rowTitle, groupTitle, GPU_HINT, diskRow, diskRowHTML, settingsHTML, sizeText, pickNote, shortNote, allowLabel, inviteHint,
  poolGpuLine, poolStoreLine, poolFix, devStoreTag, devStoreTitle } from "../../room/diskpick.js";
import { storageOf, offloadFor } from "../../room/pledge.js";
import { diskState } from "../../room/diskstore.js";
import { minFitPledge, roomFit } from "../../room/plan.js";
import { TAB_DISK_KEY, TAB_DISK_MODELS } from "../../room/flashdisk.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const GB = 2 ** 30;
const mem = (v) => { const o = v == null ? {} : { [TAB_DISK_KEY]: v }; return { getItem: (k) => o[k] ?? null, setItem: (k, x) => { o[k] = x; }, removeItem: (k) => { delete o[k]; } }; };

Deno.test("DISK_COPY: Qwen3.8 Flash's entry is a real model with a disk tier, a row of its own beside PICKER's", () => {
  const m = DISK_COPY["qwen3.8-flash"];
  ok(m && MODELS["qwen3.8-flash"], "an entry for a model in MODELS");
  ok(!PICKER.includes("qwen3.8-flash"), "PICKER keeps the three models the room picks by itself");
  ok(Object.keys(DISK_COPY).every((k) => TAB_DISK_MODELS.includes(k)), "every disk row is a model room/flashdisk.js serves");
  eq(diskCopyOf("qwen3-1.7b"), null, "a normal model has none");
  eq(Object.keys(m).sort(), ["active", "arch", "diskBytes", "name", "params"], "no figure kept here that the code does not compute");
});

Deno.test("DISK_COPY: the stored size is every layer's experts and the n-gram table (the host's disk question's 92.4 GB)", () => {
  const s = SHAPE["qwen3.8-flash"], m = DISK_COPY["qwen3.8-flash"];
  eq(m.diskBytes, s.exp * s.L + s.ngram.bytes);
  eq(diskCopyBytes("qwen3.8-flash"), m.diskBytes);
  eq(sizeText(m.diskBytes), "92.4 GB", "what room.js fmtBytes shows in the question");
});

Deno.test("diskpick: the row, and the line under the rows", () => {
  const m = DISK_COPY["qwen3.8-flash"];
  eq(rowChip(m), "125B · runs from disk", "the model card's size (roadmap/33), not the file's 177B");
  eq(paramsText(m), "125B (6B active)");
  eq(rowChipHTML(m).replace(/<[^>]+>/g, ""), rowChip(m), "the rendered chip says the same");
  ok(rowChipHTML(m).includes('<span class="dkz">125B · </span>runs from disk'), "a phone keeps only \"runs from disk\"");
  ok(!/<svg/.test(rowChipHTML(m)), "no star");
  eq(rowTitle(m), "Qwen3.8 Flash: a 125B (6B active) model that keeps most of itself on disk, so it needs far less GPU memory");
  const END = "Alone it reads from disk, so answers start slowly. Each device that joins makes it faster.";
  eq(pickNote(m, 5.7), `Qwen3.8 Flash is a 125B model. It keeps most of itself on disk, so it needs only about 6 GB of GPU memory. ${END}`);
  eq(pickNote(m, 8.1), pickNote(m, 8.6), "a device joining doesn't grow the figure beside \"makes it faster\"");
  ok(pickNote(m, 8.1).includes("only about 9 GB of GPU memory"));
  ok(pickNote(m, null).includes("far less GPU memory than its size"), "before the planner has a figure");
  // what the first Start downloads to this device, by its disk answer
  const say = (disk) => pickNote(m, 8.1, true, true, disk).replace(/^.*?GPU memory\. /, "").replace(END, "").trim();
  eq(say({ state: "allow", shareBytes: null, storedBytes: 0 }), "The first Start downloads 92.4 GB to this device's disk, once; that can take an hour or more on home internet.");
  eq(say({ state: "allow", shareBytes: 46.2 * GB, storedBytes: 0 }), "The first Start downloads 46.2 GB to this device's disk, once; that can take an hour or more on home internet.", "its share");
  eq(say({ state: "allow", shareBytes: 46.2 * GB, storedBytes: 46.2 * GB }), "It's already saved on this device.");
  eq(say({ state: "allow", shareBytes: null, storedBytes: m.diskBytes }), "It's already saved on this device.");
  eq(say({ state: "allow", shareBytes: 46.2 * GB, storedBytes: 10 * GB }), "Start finishes saving 46.2 GB to this device's disk.");
  eq(say({ state: "ask", shareBytes: null, storedBytes: 0 }), "Start asks first, then downloads 92.4 GB to this device's disk, once.");
  for (const st of ["deny", "url", "small", "none"]) eq(say({ state: st, shareBytes: null, storedBytes: 0 }), "", st);
  eq(say({ state: "allow", shareBytes: 0, storedBytes: 0 }), "", "the room doesn't need this device's disk");
  eq(say({ state: "ask", shareBytes: 0, storedBytes: 0 }), "");
  eq(pickNote(m, 67, false), "Qwen3.8 Flash is a 125B model that keeps most of itself on disk.", "no device stores it: never a whole-model GPU figure, and the fix is said under Start and in the pool");
  eq(pickNote(m, null, false), pickNote(m, 67, false), "no planner figure (a small room): the same, never \"far less GPU memory\"");
  eq(pickNote(m, 67, false, true), "Qwen3.8 Flash is a 125B model. This room's GPU memory holds all of it.");
  ok(!/67| on GPUs/.test(pickNote(m, 67, false) + pickNote(m, 67, false, true)), "no 67");
  ok(!/MoE|expert|OPFS|pledge/i.test(["allow", "ask", "deny"].map((st) => pickNote(m, 5.7, true, true, { state: st })).join(" ") + rowChip(m) + rowTitle(m)), "plain words");
});

Deno.test("diskpick: short, under Start: by how much, and the fix this device's disk answer leaves", () => {
  eq(shortNote(55, "ask"), "55 GB of GPU memory short. Allow disk storage, or invite more devices.");
  eq(shortNote(55, "deny"), shortNote(55, "ask"));
  eq(shortNote(2.5, "allow"), "2.5 GB of GPU memory short. Invite more devices, or give more GPU memory.");
  for (const st of ["url", "small", "none"]) eq(shortNote(55, st), "55 GB of GPU memory short. Invite more devices with GPUs.", st);
  eq(shortNote(55, "deny", false), "Disk storage is off on this device. Allow it, or invite a computer that can store it.", "no device can store it: no GPU figure");
  eq(shortNote(55, "ask", false), shortNote(55, "deny", false), "not asked yet: the same");
  eq(shortNote(55, "url", false), "Needs a device that can store it. Invite a computer that can.", "the link turns disk off: not a GPU fix");
  eq(shortNote(55, "allow", false), "Needs a device that can store it. Invite a computer with a GPU and free disk space.");
  for (const st of ["url", "small", "none", "allow"]) ok(shortNote(55, st, false).endsWith(poolFix(st, { nStore: 0 })), `the same fix under Start and in the pool, ${st}`);
  for (const st of ["ask", "deny"]) eq(poolFix(st, { nStore: 0 }), "", `disk storage off: the fix once, under Start beside its Allow, ${st}`);
  const m = DISK_COPY["qwen3.8-flash"];
  eq(allowLabel(m), "Allow disk storage (92.4 GB)");
  eq(allowLabel(m, 46.2 * GB), "Allow disk storage (46.2 GB)", "this device's share");
  eq(allowLabel(m, 0), allowLabel(m));
  for (const st of ["ask", "deny", "url", "small", "none", "allow"]) {
    ok(!/GPU memory short|GPUs\./.test(shortNote(55, st, false)), st);
  }
});

Deno.test("diskpick: the right column's group for the picked disk model", () => {
  eq(groupTitle(DISK_COPY["qwen3.8-flash"]), "What this device gives");
  ok(!/OPFS|tabram|tabdisk|pledge/i.test(GPU_HINT), "the GPU memory hint in plain words");
  eq(inviteHint(DISK_COPY["qwen3.8-flash"]), "Scan it, or type the code on another device. Each device that joins makes Qwen3.8 Flash faster; a computer with lots of RAM helps most.");
  ok(!/memory it gives/.test(inviteHint(DISK_COPY["qwen3.8-flash"])), "speed, not memory");
});

Deno.test("diskpick: Disk storage shows #368's answer, the stored size, the tier's RAM, Allow and Delete", () => {
  const m = DISK_COPY["qwen3.8-flash"], acts = (d) => d.acts.map((a) => a.act), size = sizeText(m.diskBytes);
  eq(size, "92.4 GB", "DISK_COPY's diskBytes, as the code computes it");
  let d = diskRow(m, { state: "allow", storedBytes: 0, ramGB: 3 });
  eq([d.k, d.chip, d.h, d.title, acts(d)], ["Disk storage", "On", "Saves 92.4 GB to this device's disk the first time it starts", "Uses 3 GB of RAM to read it", []]);
  eq(diskRow(m, { state: "allow", storedBytes: 0, ramGB: 1.5 }).title, "Uses 1.5 GB of RAM to read it", "tabRamCap on a small device");
  d = diskRow(m, { state: "allow", storedBytes: 40 * GB, ramGB: 3 });
  eq([d.h, acts(d)], ["40.0 GB of 92.4 GB saved on this device", ["free"]]);
  eq(diskRow(m, { state: "allow", storedBytes: 40 * GB, armed: true }).acts[0].label, "Delete 40.0 GB?", "the second press deletes");
  d = diskRow(m, { state: "ask", storedBytes: 0, ramGB: 3 });
  eq([d.chip, acts(d)], ["", ["allow"]], "not asked: no chip, Allow is the answer");
  eq([d.h, d.title], ["Lets Qwen3.8 Flash save 92.4 GB to this device's disk, once", "Uses 3 GB of RAM to read it"], "what Allow does, no second dialog");
  ok(!diskRowHTML(d).includes('class="more"'), "no empty chip");
  d = diskRow(m, { state: "deny", storedBytes: 0, ramGB: 3 });
  eq([d.chip, acts(d)], ["", ["allow"]], "a no: Allow again, no chip");
  ok(d.h.startsWith("Lets Qwen3.8 Flash save 92.4 GB"));
  eq(diskRow(m, { state: "deny", storedBytes: GB, ramGB: 3 }).h.split(" · ")[1], "1.0 GB saved");
  for (const st of ["url", "small", "none"]) { d = diskRow(m, { state: st, storedBytes: 0 }); eq([d.chip, acts(d)], ["Unavailable", []], st); }
  eq(diskRow(m, { state: "none" }).h, "This browser can't store it on disk", "plain words");
  eq(diskRow(m, { state: "url" }).h, "This link turns disk storage off on this device", "?tabram=0 / ?tabdisk=0 on a desktop");
  eq(diskRow(m, { state: "small" }).h, "This device doesn't have enough spare memory to read from disk");
  const all = ["url", "small", "none"].map((st) => diskRow(m, { state: st }).h).join(" ");
  ok(!/OPFS|tabram|tabdisk|phone/.test(all), "the jargon only in the tooltip");
  ok(!/GPU memory/.test(all), "never \"the whole model has to fit in GPU memory\"");
  ok(diskRow(m, { state: "url" }).title.includes("?tabdisk=0") && diskRow(m, { state: "none" }).title.includes("OPFS"));
  ok(diskRowHTML(diskRow(m, { state: "none" })).includes(' title="No disk tier'));
  eq(acts(diskRow(m, { state: "none", storedBytes: GB })), ["free"], "a copy left behind can still be deleted");
  ok(diskRowHTML(diskRow(m, { state: "allow" })).includes("<b>On</b>"));
  // a screen reader hears what each button does, not just "Allow"
  ok(diskRowHTML(diskRow(m, { state: "ask" })).includes('data-act="allow" aria-label="Allow disk storage for Qwen3.8 Flash">Allow</button>'));
  ok(diskRowHTML(diskRow(m, { state: "allow", storedBytes: 40 * GB })).includes('aria-label="Delete Qwen3.8 Flash&#39;s saved copy (40.0 GB)">Delete</button>'));
  eq(diskRow(m, { state: "allow", storedBytes: 40 * GB, armed: true }).acts[0].aria, "Confirm: delete Qwen3.8 Flash's saved copy (40.0 GB)");
  ok(!diskRowHTML(diskRow(m, { state: "allow", storedBytes: 0 })).includes("data-act"), "nothing to do");
  const h = settingsHTML(m, { state: "ask", storedBytes: 0, ramGB: 3 });
  ok(!/\b20 GB\b|tok\/s|dk-cost|dk-agree/.test(h), "no tiles, no made-up figure, no repeated sentence");
  eq([...h.matchAll(/<span class="k">([^<]*)/g)].map((x) => x[1]), ["Disk storage"], "one row: Disk storage, with its RAM in its words");
  ok(!/fits|short/.test(h), "no second fit answer: that is the row's chip and the pool's");
});

Deno.test("diskState: #368's stored answer; the URL can only say no", () => {
  const k = "qwen3.8-flash";
  eq(diskState(k, { can: true, storage: mem(null) }), "ask");
  eq(diskState(k, { can: true, storage: mem("allow") }), "allow");
  eq(diskState(k, { can: true, storage: mem("deny") }), "deny");
  eq(diskState(k, { can: true, search: "?tabdisk=0", storage: mem("allow") }), "url", "?tabdisk=0 wins over a stored yes");
  eq(diskState(k, { can: true, search: "?tabdisk=1", storage: mem(null) }), "ask", "and nothing in the URL says yes");
  eq(diskState(k, { can: false, storage: mem("allow") }), "none", "a tab without a disk tier");
  eq(diskState(k, { able: true, can: false, search: "?tabram=0", storage: mem("allow") }), "url", "?tabram=0 turns the tier off: the address, not the browser");
  eq(diskState(k, { able: true, can: false, storage: mem("allow") }), "small", "no RAM to spare (tabRamCap)");
  eq(diskState(k, { able: false, can: false, search: "?tabram=0", storage: mem("allow") }), "none", "a phone stays a phone");
  eq(diskState("qwen3-1.7b", { can: true, storage: mem("allow") }), "none", "a model without one");
  eq(diskState(k, { can: true, storage: { getItem() { throw new Error("blocked"); } } }), "ask", "storage that throws: not asked");
});

Deno.test("minFitPledge: the least total pledge at which the planner's own check holds", () => {
  const fitsAbove = (gb) => (pl) => pl.reduce((a, b) => a + b, 0) >= gb * GB;
  const near = (a, b, m) => ok(Math.abs(a - b) < 0.01 * GB, `${m}: ${a / GB} vs ${b / GB}`);
  near(minFitPledge([12 * GB], fitsAbove(9.4)).bytes, 9.4 * GB, "fits: below what is pledged");
  near(minFitPledge([4 * GB, 2 * GB], fitsAbove(20)).bytes, 20 * GB, "short: above it");
  eq(minFitPledge([4 * GB], fitsAbove(1000)), null, "never within 16x");
  eq(minFitPledge([], fitsAbove(1)), null, "nobody pledges");
  const m = minFitPledge([12 * GB, 6 * GB], fitsAbove(9));
  near(m.scaled[0], 6 * GB, "scaled together"); near(m.scaled[1], 3 * GB, "scaled together");
  // with roomFit itself: 10 layers of 1 GB and a 0.5 GB host part fit at 10.5 GB on one device
  near(minFitPledge([40 * GB], (pl) => roomFit(10, pl, GB, GB / 2).fits).bytes, 10.5 * GB, "roomFit's own number");
});

const FL = "qwen3.8-flash";
Deno.test("storageOf: what each device brings beyond GPU memory, by offloadFor's own rules", () => {
  const tab = { webgpu: true, contribGB: 12, tabDisk: { ramGB: 3, models: [FL] } };
  const asks = { webgpu: true, contribGB: 12, tabDisk: { ramGB: 3, models: [FL], ask: true } };
  const said = { webgpu: true, contribGB: 12 };
  const node = { webgpu: true, contribGB: 10, offload: true, offloadFlash: true, ramGB: 43, disk: true, local: [FL] };
  const nodeNoLocal = { ...node, local: [] };
  const nodeOld = { webgpu: true, contribGB: 10, offload: true, ramGB: 43, disk: true, local: [FL] };
  const noGpu = { webgpu: false, contribGB: 0, tabDisk: { ramGB: 3, models: [FL] } };
  const cases = [[tab, "disk", false], [asks, "disk", true], [said, null, false], [node, "disk", false], [nodeNoLocal, "ram", false], [nodeOld, null, false], [noGpu, null, false]];
  const metas = cases.map((c) => c[0]), off = offloadFor(metas, 1e9, FL);
  cases.forEach(([meta, kind, ask], i) => {
    const s = storageOf(meta, FL);
    eq([s.kind, s.ask], [kind, ask], `case ${i}`);
    eq(s.kind === "disk", !!off.disk?.[i], `disk agrees with offloadFor, case ${i}`);
    eq(s.kind != null, off.ram[i] > 0, `stores agrees with offloadFor's ram, case ${i}`);
  });
  eq(storageOf(nodeNoLocal, FL).ramGB, 43);
  eq(storageOf(node, FL).ramGB, 43, "a --disk node keeps its RAM figure (the pool's tag shows it)");
  eq(storageOf(tab, "qwen3-1.7b").kind, null, "another model: a plain tab");
});

Deno.test("pool: the GPU line per state (G1-G5), never a whole-model GPU figure", () => {
  const m = DISK_COPY[FL], line = (w) => { const g = poolGpuLine(m, w); return [g.text, g.num, g.chip, g.ok]; };
  eq(line({ needGB: 5.7, needDisk: true, fits: true, shortGB: 0, nStore: 2 }), ["Qwen3.8 Flash needs", "5.7", "fits", true], "G1");
  eq(line({ needGB: 5.7, needDisk: true, fits: false, shortGB: 1.7, nStore: 1 }), ["Qwen3.8 Flash needs", "5.7", "1.7 GB short", false], "G2");
  eq(line({ needGB: 67, needDisk: false, fits: true, nStore: 0 }), ["Qwen3.8 Flash fits in this room's GPU memory", null, "fits", true], "G3");
  eq(line({ needGB: 67, needDisk: false, fits: false, shortGB: 55, nStore: 0 }), ["Qwen3.8 Flash runs once a device stores it", null, null, false], "G4");
  eq(line({ needGB: null, needDisk: false, fits: false, shortGB: 55, nStore: 0 }), ["Qwen3.8 Flash runs once a device stores it", null, null, false], "G4, no figure");
  eq(line({ needGB: null, needDisk: false, fits: false, shortGB: 55, nStore: 1 }), ["Not enough GPU memory in this room yet", null, "short", false], "G5");
});

Deno.test("pool: the Storage line (S1-S5), the fix line and each device's tag", () => {
  const m = DISK_COPY[FL], line = (w) => { const s = poolStoreLine(m, w); return [s.text, s.chip, s.ok]; };
  eq(line({ nDisk: 2 }), ["Split across 2 devices' disks, 92.4 GB in all", "ready", true], "S1");
  eq(line({ nDisk: 1 }), ["Kept on disk, 92.4 GB in all", "ready", true], "S1, one device");
  eq(line({ nDisk: 1, nRam: 1 }), ["Kept on disk, 92.4 GB in all", "ready", true], "S2: the device that asks says so in its own tag");
  eq(line({ nRam: 1 }), ["Held in spare memory on 1 device", "ready", true], "S3");
  eq(line({ nRam: 2 })[0], "Held in spare memory on 2 devices", "S3, plural");
  eq(line({ needDisk: false, fits: true }), ["Not needed here, it all fits in GPU memory", null, false], "S4");
  eq(line({ needDisk: false, fits: false }), ["No device here can store it", "needs disk", false], "S5");
  for (const st of ["ask", "deny"]) eq(line({ needDisk: false, fits: false, state: st }), ["Disk storage is off on this device", "off", false], `S5, ${st}`);
  for (const st of ["url", "small", "none", "allow"]) eq(line({ needDisk: false, fits: false, state: st })[0], "No device here can store it", st);
  eq(poolStoreLine(m, { state: "deny" }).text, "Disk storage is off on this device");
  const s5 = { nStore: 0, fits: false, needDisk: false, needGB: 67 };
  for (const st of ["ask", "deny"]) eq(poolFix(st, s5), "", `${st}: said once, under Start`);
  eq(poolFix("url", s5), "Invite a computer that can.", "the Disk storage row says the link turns it off");
  for (const st of ["small", "none", "allow"]) eq(poolFix(st, s5), "Invite a computer with a GPU and free disk space.", st);
  eq(poolFix("allow", { nStore: 1, fits: false, needDisk: false, needGB: null }), "Give more GPU memory below, or invite a device.", "G5");
  eq(poolFix("ask", { nStore: 0, fits: true, needDisk: false, needGB: 67 }), "", "S4: nothing to fix");
  eq(poolFix("allow", { nStore: 2, fits: false, needDisk: true, needGB: 5.7 }), "", "G2: the GPU line's chip says it");
  eq(poolFix("allow", { nStore: 2, fits: true, needDisk: true, needGB: 5.7 }), "");
  eq(devStoreTag({ kind: "disk", ask: false }), "disk");
  eq(devStoreTag({ kind: "disk", ask: true }), "disk · will ask");
  eq(devStoreTag({ kind: "disk", ask: false, ramGB: 0 }, null, false, m), "up to 92.4 GB disk", "a tab whose share is not known yet");
  eq(devStoreTag({ kind: "disk", ask: true, ramGB: 0 }, null, false, m), "up to 92.4 GB disk · will ask");
  eq(devStoreTag({ kind: "disk", ask: false, ramGB: 43 }, null, false, m), "43 GB RAM · disk", "a room node with --disk: its RAM shows");
  eq(devStoreTag({ kind: "ram", ramGB: 43 }), "43 GB RAM");
  eq(devStoreTag({ kind: null }), "no disk");
  eq(devStoreTag({ kind: null }, null, true), "disk off", "this device, its disk storage off");
  // a tab's share in this room (roomFitFor share)
  eq(devStoreTag({ kind: "disk", ask: false }, 46.2 * GB), "46.2 GB disk");
  eq(devStoreTag({ kind: "disk", ask: true }, 46.2 * GB), "46.2 GB disk · will ask");
  eq(devStoreTag({ kind: "disk", ask: false }, 0), "disk not needed");
  eq(devStoreTitle("mac", 12, { kind: "disk", ask: false }, 46.2 * GB), "mac: 12 GB GPU memory, disk, saves 46.2 GB there");
  eq(devStoreTitle("mac", 12, { kind: "disk", ask: true }), "mac: 12 GB GPU memory, asks before saving to disk");
  eq(devStoreTitle("pc", 10, { kind: "ram", ramGB: 43 }), "pc: 10 GB GPU memory, 43 GB RAM");
  eq(devStoreTitle("pc", 10, { kind: "disk", ask: false, ramGB: 43 }), "pc: 10 GB GPU memory, 43 GB RAM, disk");
  eq(devStoreTitle("pc", 10, { kind: null }), "pc: 10 GB GPU memory, no disk");
  eq(devStoreTitle("mac", 12, { kind: null }, null, true), "mac: 12 GB GPU memory, disk storage off");
});

Deno.test("pool: with needDisk false nothing says the whole model's GPU figure; plain words throughout", () => {
  const m = DISK_COPY[FL], out = [];
  for (const fits of [true, false]) for (const nStore of [0, 1]) for (const needGB of [67, null]) {
    const w = { needGB, needDisk: false, fits, shortGB: 55, nStore, nDisk: 0, nRam: nStore };
    const g = poolGpuLine(m, w), s = poolStoreLine(m, w);
    out.push(g.text, g.num ?? "", g.chip ?? "", s.text, s.chip ?? "", ...["ask", "deny", "url", "small", "none", "allow"].map((st) => poolFix(st, w)));
    out.push(pickNote(m, needGB, false, fits), shortNote(55, "ask", false));
  }
  const all = out.join(" | ");
  ok(!/\b67\b/.test(all) && !all.includes(" on GPUs"), all);
  const words = all + [true, false].map((a) => poolStoreLine(m, { nDisk: a ? 1 : 0, nRam: 1 }).text).join(" ")
    + ["disk", "ram", null].map((k) => devStoreTag({ kind: k, ramGB: 43 }) + devStoreTitle("x", 1, { kind: k, ramGB: 43 })).join(" ");
  ok(!/MoE|expert|OPFS|pledge|offload/i.test(words), "banned words");
});

// A small room no device can store Flash in (one 4 GB tab, no disk tier): the planner finds no GPU figure within 16x
// (minFitPledge null, so roomFitFor sets no needGB / needDisk), and every place reads "no device can store it" from
// what the devices bring (storageOf, roomFitFor stores), never the whole model's GPU gap
Deno.test("pool: a single 4 GB tab with no disk: no row, Start or picker text says a whole-model GPU figure", () => {
  const m = DISK_COPY[FL], tab = { webgpu: true, contribGB: 4 };
  eq(storageOf(tab, FL).kind, null, "it brings no storage");
  eq(offloadFor([tab], 1e9, FL), null, "and the planner sees none either");
  const stores = [tab].some((d) => !!storageOf(d, FL).kind);
  eq(stores, false);
  const s = SHAPE[FL], whole = s.L * s.layer + s.embed + s.out;
  eq(minFitPledge([4 * GB], (pl) => roomFit(s.L, pl, s.layer, s.embed + s.out).fits), null, "no planner figure within 16x");
  const gb = Math.round(whole / GB);
  const texts = ["ask", "deny", "url", "small", "none", "allow"].flatMap((st) => [shortNote(gb - 4, st, stores), poolFix(st, { nStore: 0 })]);
  const w = { needGB: null, needDisk: false, fits: false, shortGB: gb - 4, nStore: 0 };
  const g = poolGpuLine(m, w), sl = poolStoreLine(m, w);
  texts.push(pickNote(m, null, stores, false), g.text, g.chip ?? "", sl.text, sl.chip ?? "");
  const all = texts.join(" | ");
  ok(!new RegExp(`\\b(${gb - 4}|${gb - 5}|${gb - 3}|${gb})\\b`).test(all) && !/GB short|far less GPU memory|on GPUs/.test(all), all);
  eq(sl.chip, "needs disk");
});

Deno.test("diskpick: Disk storage says this device's share of the copy in this room", () => {
  const m = DISK_COPY["qwen3.8-flash"];
  eq(diskRow(m, { state: "allow", storedBytes: 0, ramGB: 3, shareBytes: 46.2 * GB }).h,
    "Saves 46.2 GB of 92.4 GB to this device's disk the first time it starts");
  eq(diskRow(m, { state: "ask", storedBytes: 0, ramGB: 3, shareBytes: 46.2 * GB }).h,
    "Lets Qwen3.8 Flash save 46.2 GB of 92.4 GB to this device's disk, once");
  eq(diskRow(m, { state: "allow", storedBytes: 0, ramGB: 3, shareBytes: 0 }).title, "", "nothing to read here");
  eq(diskRow(m, { state: "allow", storedBytes: 0, ramGB: 3, shareBytes: m.diskBytes }).h.split(" · ")[0], "Saves 92.4 GB to this device's disk the first time it starts", "alone: the whole copy");
  eq(diskRow(m, { state: "allow", storedBytes: 20 * GB, ramGB: 3, shareBytes: 46.2 * GB }).h.split(" · ")[0], "20.0 GB of 46.2 GB saved on this device");
  eq(diskRow(m, { state: "allow", storedBytes: 92.4 * GB, ramGB: 3, shareBytes: 46.2 * GB }).h.split(" · ")[0], "92.4 GB saved on this device", "more than it needs: no \"of\"");
  eq(diskRow(m, { state: "allow", storedBytes: 0, ramGB: 3, shareBytes: 0 }).h, "This room doesn't need this device's disk right now");
  eq(diskRow(m, { state: "ask", storedBytes: 0, ramGB: 3, shareBytes: 0 }).h, "Lets Qwen3.8 Flash save up to 92.4 GB to this device's disk when the room needs it");
});
