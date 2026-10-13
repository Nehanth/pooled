// The picker's words for a model kept on disk (room/diskpick.js): every amount from room/models.js DISK_COPY, so a
// future big model gets the same row, cost figures, consent and Start label from its own entry. The consent state is
// #368's stored answer (room/flashdisk.js TAB_DISK_KEY), read through room/diskstore.js diskState.
import { DISK_COPY, MODELS, PICKER, SHAPE, diskCopyBytes } from "../../room/models.js";
import { diskCopyOf, rowChip, paramsText, rowTitle, speedRange, costFigures, consentText, startLabel, advSummary, storedText,
  costHTML, consentHTML, advRowsHTML, sizeText } from "../../room/diskpick.js";
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
  ok(m.speeds.every((s, i) => s.lo <= s.hi && (i === 0 || s.devices > m.speeds[i - 1].devices)), "speeds: fewest devices first, lo <= hi");
});

Deno.test("DISK_COPY: the stored size is every layer's experts and the n-gram table (the host's disk question's 92.4 GB)", () => {
  const s = SHAPE["qwen3.8-flash"], m = DISK_COPY["qwen3.8-flash"];
  eq(m.diskBytes, s.exp * s.L + s.ngram.bytes);
  eq(diskCopyBytes("qwen3.8-flash"), m.diskBytes);
  eq(sizeText(m.diskBytes), "92.4 GB", "what room.js fmtBytes shows in the question");
});

Deno.test("diskpick: Qwen3.8 Flash's words, from its entry", () => {
  const m = DISK_COPY["qwen3.8-flash"];
  eq(rowChip(m), "125B · MoE · from disk", "the model card's size (roadmap/33), not the file's 177B");
  eq(paramsText(m), "125B (6B active)");
  eq(rowTitle(m), "Qwen3.8 Flash: 125B (6B active), MoE, its experts streamed from this device's disk");
  eq(speedRange(m), "7–12");
  // the memory tile is the room's figure (roomFitFor needGB) and this tab's disk-tier RAM, never a number kept here
  eq(costFigures(m).map((f) => f.k), ["Stored once", "Speed"], "no need computed: no memory tile");
  eq(costFigures(m, { needGB: 9.4, needDisk: true, ramGB: 3 }).map((f) => [f.k, f.v, f.u, f.note]), [
    ["Stored once", "92.4", "GB", "Copied once, then starts fast"],
    ["Memory", "9.4", "GB", "On GPUs; the rest streams from disk, with 3 GB of RAM here for it"],
    ["Speed", "7–12", "tok/s", "Gets faster as devices join"]]);
  eq(costFigures(m, { needGB: 74.2, needDisk: false })[1].note, "On GPUs: the whole model, no disk here");
  eq(advSummary(m), "92.4 GB disk");
  eq(advSummary(m, { ramGB: 3 }), "92.4 GB disk · 3 GB RAM");
  ok(advRowsHTML(m, { ramGB: 3 }).includes("RAM for the disk tier</span><span class=\"v\">3 GB"));
  ok(advRowsHTML(m, {}).includes(">None<"), "no disk tier: no RAM figure");
  ok(!/\b20 GB\b/.test(costHTML(m, { needGB: 9.4, needDisk: true, ramGB: 3 }) + advRowsHTML(m, { ramGB: 3 })), "no made-up memory figure");
});

Deno.test("diskpick: the consent line says what the disk question says, and shows the answer already given", () => {
  const m = DISK_COPY["qwen3.8-flash"];
  // room.js askTabDisk: "<name> streams from your disk" / "It will store up to N in this browser, so it can run more
  // than this device's memory holds. It stays for next time; Room settings can free it."
  eq(consentText(m, "ask"), "Qwen3.8 Flash streams from your disk: it will store up to 92.4 GB in this browser, so it can run more than this device's memory holds. It stays for next time; Room settings can free it.");
  eq(startLabel(m, "ask"), "Allow 92.4 GB and start", "the press is the answer, for the same bytes");
  for (const st of ["allow", "deny", "url", "none"]) eq(startLabel(m, st), "Start Qwen3.8 Flash", st + ": nothing to ask");
  ok(consentText(m, "allow").startsWith("This device lets big models use its disk"));
  ok(consentText(m, "allow", 40 * GB).includes("40.0 GB stored now"));
  ok(consentText(m, "deny").includes("Room settings changes that"), "a no is changed where #368 changes it");
  ok(consentText(m, "url").includes("?tabdisk=0"));
  ok(!/Allow|store up to/.test(consentText(m, "none")), "no disk tier: nothing to agree to");
  ok(consentHTML(m).includes("<b>up to 92.4 GB in this browser,</b>"));
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

Deno.test("diskpick: another big model gets its own numbers, nothing of Flash's", () => {
  const m = { name: "Big One", params: "400B", arch: "MoE", diskBytes: 210.5 * GB, speeds: [{ devices: 2, lo: 3, hi: 4 }] };
  eq(rowChip(m), "400B · MoE · from disk");
  eq(paramsText(m), "400B", "no active count: just the size");
  eq(costFigures(m, { needGB: 36 }).map((f) => f.v), ["210.5", "36", "3–4"]);
  eq(costFigures(m).at(-1).note, "Measured", "one measured size: no claim that it speeds up");
  ok(consentText(m).startsWith("Big One streams from your disk: it will store up to 210.5 GB"));
  eq(startLabel(m, "ask"), "Allow 210.5 GB and start");
  for (const h of [costHTML(m, { needGB: 36 }), consentHTML(m), advRowsHTML(m)]) ok(!/Flash|92|125B|6B active/.test(h), h);
});

Deno.test("diskpick: the stored copy and its two-press delete", () => {
  eq(sizeText(92 * GB), "92.0 GB"); eq(sizeText(300 * 2 ** 20), "300 MB");
  eq(storedText(0), "Nothing stored on this device yet."); eq(storedText(null), "Not known in this browser.");
  ok(storedText(91.5 * GB).startsWith("91.5 GB on this disk."));
  const m = DISK_COPY["qwen3.8-flash"];
  ok(/dk-del" disabled/.test(advRowsHTML(m, { storedBytes: 0 })), "nothing stored: delete is off");
  ok(!/dk-del" disabled/.test(advRowsHTML(m, { storedBytes: GB })), "a copy: delete is on");
  ok(advRowsHTML(m, { storedBytes: GB, armed: true }).includes("Delete 1.0 GB?"), "armed: the second press deletes");
  ok(!advRowsHTML(m, { storedBytes: 0, armed: true }).includes("?</button>"), "nothing to delete: never armed");
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
