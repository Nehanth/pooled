// The picker's words for a model kept on disk (room/diskpick.js): every amount from room/models.js DISK_COPY, so a
// future big model gets the same row, cost figures, consent and Start label from its own entry. The consent state is
// #368's stored answer (room/flashdisk.js TAB_DISK_KEY), read through room/diskstore.js diskState.
import { DISK_COPY, MODELS, PICKER, SHAPE, diskCopyBytes } from "../../room/models.js";
import { diskCopyOf, rowChip, rowChipHTML, paramsText, rowTitle, groupTitle, GPU_HINT, diskRow, diskRowHTML, settingsHTML, sizeText, pickNote, shortNote } from "../../room/diskpick.js";
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
  eq(pickNote(m, 5.7), "Qwen3.8 Flash is a 125B model. It keeps most of itself on disk, so it needs only 5.7 GB of GPU memory. Each device that joins makes it faster.");
  ok(pickNote(m, null).includes("far less GPU memory than its size"), "before the planner has a figure");
  eq(pickNote(m, 67, false), "Qwen3.8 Flash is a 125B model that normally keeps most of itself on disk. With no disk storage in this room, it needs 67 GB of GPU memory.", "never \"only 67 GB\"");
  ok(!/MoE|expert|OPFS|pledge/i.test(pickNote(m, 5.7) + rowChip(m) + rowTitle(m)), "plain words");
});

Deno.test("diskpick: short, under Start: by how much, and the fix this device's disk answer leaves", () => {
  eq(shortNote(55, "ask"), "55 GB of GPU memory short. Allow disk storage, or invite more devices.");
  eq(shortNote(55, "deny"), shortNote(55, "ask"));
  eq(shortNote(2.5, "allow"), "2.5 GB of GPU memory short. Invite more devices, or give more GPU memory.");
  for (const st of ["url", "small", "none"]) eq(shortNote(55, st), "55 GB of GPU memory short. Invite more devices with GPUs.", st);
});

Deno.test("diskpick: the right column's group for the picked disk model", () => {
  eq(groupTitle(DISK_COPY["qwen3.8-flash"]), "What this device gives");
  ok(!/OPFS|tabram|tabdisk|pledge/i.test(GPU_HINT), "the GPU memory hint in plain words");
});

Deno.test("diskpick: Disk storage shows #368's answer, the stored size, the tier's RAM, Allow and Delete", () => {
  const m = DISK_COPY["qwen3.8-flash"], acts = (d) => d.acts.map((a) => a.act), size = sizeText(m.diskBytes);
  eq(size, "92.4 GB", "DISK_COPY's diskBytes, as the code computes it");
  let d = diskRow(m, { state: "allow", storedBytes: 0, ramGB: 3 });
  eq([d.k, d.chip, d.h, acts(d)], ["Disk storage", "On", "Saves 92.4 GB to this device's disk the first time it starts · Uses 3 GB of RAM to read it", []]);
  eq(diskRow(m, { state: "allow", storedBytes: 0, ramGB: 1.5 }).h.split(" · ")[1], "Uses 1.5 GB of RAM to read it", "tabRamCap on a small device");
  d = diskRow(m, { state: "allow", storedBytes: 40 * GB, ramGB: 3 });
  eq([d.h, acts(d)], ["40.0 GB of 92.4 GB saved on this device · Uses 3 GB of RAM to read it", ["free"]]);
  eq(diskRow(m, { state: "allow", storedBytes: 40 * GB, armed: true }).acts[0].label, "Delete 40.0 GB?", "the second press deletes");
  d = diskRow(m, { state: "ask", storedBytes: 0, ramGB: 3 });
  eq([d.chip, acts(d)], ["", ["allow"]], "not asked: no chip, Allow is the answer");
  eq(d.h, "Lets Qwen3.8 Flash save 92.4 GB to this device's disk, once, and read it with 3 GB of RAM", "what Allow does, no second dialog");
  ok(!diskRowHTML(d).includes('class="more"'), "no empty chip");
  d = diskRow(m, { state: "deny", storedBytes: 0, ramGB: 3 });
  eq([d.chip, acts(d)], ["", ["allow"]], "a no: Allow again, no chip");
  ok(d.h.startsWith("Lets Qwen3.8 Flash save 92.4 GB"));
  eq(diskRow(m, { state: "deny", storedBytes: GB, ramGB: 3 }).h.split(" · ")[1], "1.0 GB saved");
  for (const st of ["url", "small", "none"]) { d = diskRow(m, { state: st, storedBytes: 0 }); eq([d.chip, acts(d)], ["Unavailable", []], st); }
  eq(diskRow(m, { state: "none" }).h, "This browser can't store it on disk", "plain words");
  eq(diskRow(m, { state: "none", needDisk: false }).h, "This browser can't store it on disk, so the whole model has to fit in GPU memory", "and the cause of the fit");
  eq(diskRow(m, { state: "url", needDisk: false }).h, "This link turns disk storage off, so the whole model has to fit in GPU memory", "?tabram=0 / ?tabdisk=0 on a desktop");
  eq(diskRow(m, { state: "small" }).h, "This device doesn't have enough spare memory to read from disk");
  const all = ["url", "small", "none"].map((st) => diskRow(m, { state: st, needDisk: false }).h).join(" ");
  ok(!/OPFS|tabram|tabdisk|phone/.test(all), "the jargon only in the tooltip");
  ok(diskRow(m, { state: "url" }).title.includes("?tabdisk=0") && diskRow(m, { state: "none" }).title.includes("OPFS"));
  ok(diskRowHTML(diskRow(m, { state: "none" })).includes(' title="No disk tier'));
  eq(acts(diskRow(m, { state: "none", storedBytes: GB })), ["free"], "a copy left behind can still be deleted");
  ok(diskRowHTML(diskRow(m, { state: "allow" })).includes("<b>On</b>"));
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
