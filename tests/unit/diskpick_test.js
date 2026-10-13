// The picker's words for a model kept on disk (room/diskpick.js): every amount from room/models.js DISK_COPY, so a
// future big model gets the same row, cost figures, consent and Start label from its own entry. The consent state is
// #368's stored answer (room/flashdisk.js TAB_DISK_KEY), read through room/diskstore.js diskState.
import { DISK_COPY, MODELS, PICKER, SHAPE, diskCopyBytes } from "../../room/models.js";
import { diskCopyOf, rowChip, paramsText, rowTitle, gpuRow, diskRow, gpuRowHTML, diskRowHTML, settingsHTML, sizeText } from "../../room/diskpick.js";
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

Deno.test("diskpick: the row", () => {
  const m = DISK_COPY["qwen3.8-flash"];
  eq(rowChip(m), "125B · MoE · from disk", "the model card's size (roadmap/33), not the file's 177B");
  eq(paramsText(m), "125B (6B active)");
  eq(rowTitle(m), "Qwen3.8 Flash: 125B (6B active), MoE, its experts streamed from this device's disk");
});

Deno.test("diskpick: GPU memory shows the room's own need and verdict", () => {
  const fit = gpuRow({ needGB: 5.7, hasGB: 12, fits: true, needDisk: true, mineGB: 12 });
  eq([fit.k, fit.v, fit.chip], ["GPU memory", "needs 5.7 GB · room 12 GB", "fits"]);
  eq(fit.h, "The least the room's GPUs need; the rest streams from disk. This device gives 12 GB.");
  const short = gpuRow({ needGB: 67, hasGB: 12, fits: false, shortGB: 55, needDisk: false, mineGB: 12 });
  eq([short.v, short.chip], ["needs 67 GB · room 12 GB", "55 GB short"]);
  ok(short.h.startsWith("No disk tier here"));
  ok(gpuRowHTML(fit).includes("<b>fits</b>"), "the rows' own fits chip");
  ok(gpuRowHTML(short).includes('<span class="more">55 GB short</span>'), "and their short chip");
});

Deno.test("diskpick: Disk shows #368's answer, the stored size, the tier's RAM, Allow and Free", () => {
  const m = DISK_COPY["qwen3.8-flash"], acts = (d) => d.acts.map((a) => a.act);
  let d = diskRow(m, { state: "allow", storedBytes: 0, ramGB: 3 });
  eq([d.chip, d.h, acts(d)], ["Allowed", "0 GB of 92.4 GB stored · 3 GB RAM for its cache", []]);
  d = diskRow(m, { state: "allow", storedBytes: 40 * GB, ramGB: 3 });
  eq([d.h, acts(d)], ["40.0 GB of 92.4 GB stored · 3 GB RAM for its cache", ["free"]]);
  eq(diskRow(m, { state: "allow", storedBytes: 40 * GB, armed: true }).acts[0].label, "Free 40.0 GB?", "the second press frees");
  d = diskRow(m, { state: "ask", storedBytes: 0, ramGB: 3 });
  eq([d.chip, acts(d)], ["Not asked yet", ["allow"]]); ok(d.h.endsWith("Start asks first"));
  d = diskRow(m, { state: "deny", storedBytes: 0 });
  eq([d.chip, acts(d)], ["Not allowed", ["allow"]]);
  for (const st of ["url", "none"]) { d = diskRow(m, { state: st, storedBytes: 0, ramGB: null }); eq([d.chip, acts(d)], ["Not available", []], st); }
  eq(diskRow(m, { state: "none" }).h, "This browser can't store it (phones and some browsers can't)", "plain words");
  ok(!/OPFS|tabram|tabdisk/.test(diskRow(m, { state: "none" }).h + diskRow(m, { state: "url" }).h), "the jargon only in the tooltip");
  ok(diskRow(m, { state: "url" }).title.includes("?tabdisk=0") && diskRow(m, { state: "none" }).title.includes("OPFS"));
  ok(diskRowHTML(diskRow(m, { state: "none" })).includes(' title="No disk tier'));
  eq(acts(diskRow(m, { state: "none", storedBytes: GB })), ["free"], "a copy left behind can still be freed");
  ok(diskRowHTML(diskRow(m, { state: "allow" })).includes("<b>Allowed</b>"));
  ok(!diskRowHTML(diskRow(m, { state: "allow", storedBytes: 0 })).includes("data-act"), "nothing to do");
  const h = settingsHTML(m, { needGB: 5.7, hasGB: 12, fits: true, needDisk: true }, { state: "ask", storedBytes: 0, ramGB: 3 });
  ok(!/\b20 GB\b|tok\/s|dk-cost|dk-agree/.test(h), "no tiles, no made-up figure, no repeated sentence");
});

Deno.test("diskState: #368's stored answer; the URL can only say no", () => {
  const k = "qwen3.8-flash";
  eq(diskState(k, { can: true, storage: mem(null) }), "ask");
  eq(diskState(k, { can: true, storage: mem("allow") }), "allow");
  eq(diskState(k, { can: true, storage: mem("deny") }), "deny");
  eq(diskState(k, { can: true, search: "?tabdisk=0", storage: mem("allow") }), "url", "?tabdisk=0 wins over a stored yes");
  eq(diskState(k, { can: true, search: "?tabdisk=1", storage: mem(null) }), "ask", "and nothing in the URL says yes");
  eq(diskState(k, { can: false, storage: mem("allow") }), "none", "a tab without a disk tier");
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
