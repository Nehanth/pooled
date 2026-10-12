// The picker's words for a model kept on disk (room/diskpick.js): every amount from room/models.js DISK_COPY, so a
// future big model gets the same row, cost figures, consent, Start label and chart from its own entry.
import { DISK_COPY, MODELS, PICKER } from "../../room/models.js";
import { diskCopyOf, rowChip, speedRange, costFigures, consentText, keptText, freeText, startLabel, advSummary, storedText,
  speedRows, costHTML, consentHTML, advRowsHTML, chartHTML, sizeText } from "../../room/diskpick.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const GB = 2 ** 30;

Deno.test("DISK_COPY: Qwen3.8 Flash's entry is a real model, not in the default picker (dev-gated)", () => {
  const m = DISK_COPY["qwen3.8-flash"];
  ok(m && MODELS["qwen3.8-flash"], "an entry for a model in MODELS");
  ok(!PICKER.includes("qwen3.8-flash"), "Flash stays out of the default picker");
  eq(diskCopyOf("qwen3-1.7b"), null, "a normal model has none");
  ok(m.speeds.every((s, i) => s.lo <= s.hi && (i === 0 || s.devices > m.speeds[i - 1].devices)), "speeds: fewest devices first, lo <= hi");
});

Deno.test("diskpick: Qwen3.8 Flash's words, from its entry", () => {
  const m = DISK_COPY["qwen3.8-flash"];
  eq(rowChip(m), "177B · MoE · from disk");
  eq(speedRange(m), "7–12");
  eq(costFigures(m).map((f) => [f.k, f.v, f.u, !!f.approx, f.note]), [
    ["Stored once", "92", "GB", false, "Copied once, then starts fast"],
    ["Memory", "20", "GB", true, "For the experts it uses most"],
    ["Speed", "7–12", "tok/s", false, "Gets faster as devices join"]]);
  eq(consentText(m), "This device keeps a copy of the model: up to 92 GB in this browser, one time. Delete it any time.");
  eq(startLabel(m, false), "Allow 92 GB and start");
  eq(startLabel(m, true), "Start Qwen3.8 Flash");
  eq(advSummary(m), "92 GB disk · 20 GB memory");
  const r2 = speedRows(m, 2), r1 = (n) => Math.round(n * 10) / 10;
  eq(r2.rows.map((r) => [r.label, r.range, r.on, r1(r.w), r1(r.lo)]), [
    ["1 device", "7–8.5", false, 54.5, 82.4],
    ["2 devices", "11–12", true, 76.9, 91.7]], "the bars: the top speed at 1/1.3 of the track, solid to the low number");
  eq(r2.next, "3 devices");
  eq(speedRows(m, 1).next, "3 devices", "a room of one still invites the next size it has no measure for");
  eq(speedRows(m, 4).next, "5 devices");
});

Deno.test("diskpick: another big model gets its own numbers, nothing of Flash's", () => {
  const m = { name: "Big One", params: "400B", arch: "MoE", diskGB: 210.5, memGB: 36, speeds: [{ devices: 2, lo: 3, hi: 4 }] };
  eq(rowChip(m), "400B · MoE · from disk");
  eq(costFigures(m).map((f) => f.v), ["210.5", "36", "3–4"]);
  eq(costFigures(m)[2].note, "Measured", "one measured size: no claim that it speeds up");
  eq(consentText(m), "This device keeps a copy of the model: up to 210.5 GB in this browser, one time. Delete it any time.");
  eq(startLabel(m, false), "Allow 210.5 GB and start");
  eq(speedRows(m, 1).next, "3 devices");
  for (const h of [costHTML(m), consentHTML(m), advRowsHTML(m), chartHTML(m)]) ok(!/Flash|92|177B/.test(h), h);
});

Deno.test("diskpick: stored copy, free disk and the kept line", () => {
  const m = DISK_COPY["qwen3.8-flash"];
  eq(sizeText(92 * GB), "92.0 GB"); eq(sizeText(300 * 2 ** 20), "300 MB");
  eq(freeText(412.6 * GB), "412 GB free on this disk"); eq(freeText(null), "");
  eq(storedText(0), "Nothing stored on this device yet."); eq(storedText(null), "Not known in this browser.");
  ok(storedText(91.5 * GB).startsWith("91.5 GB on this disk."));
  ok(keptText(m, 91.5 * GB).includes("91.5 GB in this browser"));
  ok(consentHTML(m, { freeBytes: 412 * GB }).includes("412 GB free on this disk"), "free disk shown when known");
  ok(!consentHTML(m, {}).includes("free"), "and not when it is unknown");
  ok(/dk-del" disabled/.test(advRowsHTML(m, { storedBytes: 0 })), "nothing stored: delete is off");
  ok(!/dk-del" disabled/.test(advRowsHTML(m, { storedBytes: GB })), "a copy: delete is on");
  ok(consentHTML(m).includes("<b>up to 92 GB in this browser, one time.</b>"));
});
