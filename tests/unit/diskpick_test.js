// The picker's words for a model kept on disk (room/diskpick.js): every amount from room/models.js DISK_COPY, so a
// future big model gets the same row, cost figures, consent and Start label from its own entry.
import { DISK_COPY, MODELS, PICKER } from "../../room/models.js";
import { diskCopyOf, rowChip, paramsText, rowTitle, speedRange, costFigures, consentText, keptText, freeText, startLabel, advSummary, storedText,
  costHTML, consentHTML, advRowsHTML, sizeText } from "../../room/diskpick.js";

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
  eq(rowChip(m), "125B · MoE · from disk", "the model card's size (roadmap/33), not the file's 177B");
  eq(paramsText(m), "125B (6B active)");
  eq(rowTitle(m), "Qwen3.8 Flash: 125B (6B active), MoE, its experts streamed from this device's disk");
  eq(speedRange(m), "7–12");
  eq(costFigures(m).map((f) => [f.k, f.v, f.u, !!f.approx, f.note]), [
    ["Stored once", "92", "GB", false, "Copied once, then starts fast"],
    ["Memory", "20", "GB", true, "For the experts it uses most"],
    ["Speed", "7–12", "tok/s", false, "Gets faster as devices join"]]);
  eq(consentText(m), "This device keeps a copy of the model: up to 92 GB in this browser, one time. Delete it any time.");
  eq(startLabel(m, false), "Allow 92 GB and start");
  eq(startLabel(m, true), "Start Qwen3.8 Flash");
  eq(advSummary(m), "92 GB disk · 20 GB memory");
});

Deno.test("diskpick: another big model gets its own numbers, nothing of Flash's", () => {
  const m = { name: "Big One", params: "400B", arch: "MoE", diskGB: 210.5, memGB: 36, speeds: [{ devices: 2, lo: 3, hi: 4 }] };
  eq(rowChip(m), "400B · MoE · from disk");
  eq(paramsText(m), "400B", "no active count: just the size");
  eq(costFigures(m).map((f) => f.v), ["210.5", "36", "3–4"]);
  eq(costFigures(m)[2].note, "Measured", "one measured size: no claim that it speeds up");
  eq(consentText(m), "This device keeps a copy of the model: up to 210.5 GB in this browser, one time. Delete it any time.");
  eq(startLabel(m, false), "Allow 210.5 GB and start");
  for (const h of [costHTML(m), consentHTML(m), advRowsHTML(m)]) ok(!/Flash|92|125B|6B active/.test(h), h);
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
