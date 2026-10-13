// The picker for a model that keeps a copy on this device's disk (room/models.js DISK_COPY): its row's chip, the
// three cost figures, the consent line, the Start label and the Advanced rows.
// Pure: every word and number comes from the model's DISK_COPY entry, so another big model reuses all of it.
// The consent is room/flashdisk.js's (#368): one stored answer per site (TAB_DISK_KEY) and one question (room.js
// askTabDisk, "<name> streams from your disk"). The picker's line says what that question says, for the same bytes,
// and shows the answer already given (room/diskstore.js diskState); it never asks a second way.
// room.js puts the HTML in place; room/diskstore.js answers the state, the stored size and the delete.
import { DISK_COPY } from "./models.js";
import { esc } from "./markdown.js";

export const diskCopyOf = (key) => DISK_COPY[key] || null;

// the ✦ (site/js/sparkle.js's star)
export const STAR = "M8 0C8.6 4.6 11.4 7.4 16 8 11.4 8.6 8.6 11.4 8 16 7.4 11.4 4.6 8.6 0 8 4.6 7.4 7.4 4.6 8 0Z";
export const starSVG = (cls = "") => `<svg class="${cls}" viewBox="0 0 16 16" aria-hidden="true"><path d="${STAR}" fill="currentColor"/></svg>`;

// "20", "8.5", "1.5": a number as the picker writes it
const num = (n) => String(+(+n).toFixed(1));
// bytes as room.js fmtBytes writes them (the disk question's number): "92.4 GB", "640 MB"
export const sizeText = (bytes) => bytes >= 2 ** 30 ? `${(bytes / 2 ** 30).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 2 ** 20))} MB`;
const sizeParts = (bytes) => { const [v, u] = sizeText(bytes).split(" "); return { v, u }; };

// the row's chip: "125B · MoE · from disk"
export const rowChip = (m) => [m.params, m.arch, "from disk"].filter(Boolean).join(" · ");
// its size in words: "125B (6B active)"
export const paramsText = (m) => m.active ? `${m.params} (${m.active} active)` : m.params;
// the row's tooltip: "Qwen3.8 Flash: 125B (6B active), MoE, its experts streamed from this device's disk"
export const rowTitle = (m) => `${m.name}: ${[paramsText(m), m.arch].filter(Boolean).join(", ")}, its experts streamed from this device's disk`;

// the measured speed over every room size: "7–12"
export const speedRange = (m) => {
  const s = m.speeds || [];
  return s.length ? `${num(Math.min(...s.map((x) => x.lo)))}–${num(Math.max(...s.map((x) => x.hi)))}` : "";
};

// The cost row's figures: [{ k, v, u, approx, note }]. The memory is what the room computes, never a figure kept
// here: need: { needGB, needDisk, ramGB } (room.js roomFitFor's needGB, the GPU memory the planner needs from the
// room; ramGB: this tab's disk-tier RAM, room/flashdisk.js tabDiskMeta: TAB_RAM_GB held to tabRamCap). Without a
// needGB the memory tile is left out.
export function costFigures(m, { needGB = null, needDisk = false, ramGB = null } = {}) {
  const more = (m.speeds || []).length > 1, d = sizeParts(m.diskBytes);
  const mem = needGB == null ? null : { k: "Memory", v: num(needGB), u: "GB", note: needDisk
    ? `On GPUs; the rest streams from disk${ramGB > 0 ? `, with ${num(ramGB)} GB of RAM here for it` : ""}`
    : "On GPUs: the whole model, no disk here" };
  return [
    { k: "Stored once", v: d.v, u: d.u, note: "Copied once, then starts fast" },
    ...(mem ? [mem] : []),
    { k: "Speed", v: speedRange(m), u: "tok/s", note: more ? "Gets faster as devices join" : "Measured" },
  ];
}

// This device's answer (room/diskstore.js diskState): "ask" (not asked yet), "allow", "deny" (said no; Room settings
// changes it), "url" (?tabdisk=0: no for this page) or "none" (this tab has no disk tier: a phone, no OPFS)
// The line under the cost row, split so the amount can be bold: what room.js askTabDisk asks, for the same bytes
export const consentParts = (m, state = "ask", storedBytes = null) => {
  const amt = `up to ${sizeText(m.diskBytes)} in this browser`;
  if (state === "allow") return [storedBytes > 0 ? `This device lets big models use its disk: ${sizeText(storedBytes)} stored now, ` : "This device lets big models use its disk: ", amt + ".", " Free it under Advanced or in Room settings."];
  if (state === "deny") return ["This device keeps models off its disk. ", "Room settings changes that.", " The room can still deal it to devices that allow theirs."];
  if (state === "url") return ["This page keeps models off this device's disk ", "(?tabdisk=0).", " The room can still deal it to devices that allow theirs."];
  if (state === "none") return ["This browser can't keep models on its disk. ", "The room deals it to devices that can.", ""];
  return [`${m.name} streams from your disk: it will store `, amt + ",", " so it can run more than this device's memory holds. It stays for next time; Room settings can free it."];
};
export const consentText = (m, state, storedBytes) => consentParts(m, state, storedBytes).join("");
// Start: asks for the space while this device was not asked yet (the press is its answer), else just starts
export const startLabel = (m, state = "ask") => state === "ask" ? `Allow ${sizeText(m.diskBytes)} and start` : `Start ${m.name}`;
// Advanced's summary line: "92.4 GB disk · 3 GB RAM" (the RAM only when this tab has a disk tier)
export const advSummary = (m, { ramGB = null } = {}) => [`${sizeText(m.diskBytes)} disk`, ramGB > 0 ? `${num(ramGB)} GB RAM` : ""].filter(Boolean).join(" · ");
// the stored copy's line: what is on this disk now
export const storedText = (storedBytes) => storedBytes > 0 ? `${sizeText(storedBytes)} on this disk. Deleting frees the space; the next start that needs it copies it again.`
  : storedBytes === 0 ? "Nothing stored on this device yet." : "Not known in this browser.";

// ---- HTML ----
export function costHTML(m, need = {}) {
  return `<dl class="dk-cost" aria-label="What it takes">${costFigures(m, need).map((f) =>
    `<div><dt>${esc(f.k)}</dt><dd>${f.approx ? '<span class="ap">~</span>' : ""}${esc(f.v)}<small>${esc(f.u)}</small></dd><p>${esc(f.note)}</p></div>`).join("")}</dl>`;
}
// the consent line for this device's answer
export function consentHTML(m, { state = "ask", storedBytes = null } = {}) {
  const [a, b, c] = consentParts(m, state, storedBytes);
  return `<p class="dk-agree">${esc(a)}<b>${esc(b)}</b>${esc(c)}</p>`;
}
// Advanced's rows: disk, this tab's disk-tier RAM (ramGB, null: no disk tier here), the stored copy with its delete
// (armed: the second press deletes, like Room settings' "Free disk space")
export function advRowsHTML(m, { storedBytes = null, armed = false, ramGB = null } = {}) {
  const d = sizeText(m.diskBytes);
  return `<div class="dk-row"><span class="k">Disk to use</span><span class="v">${d}</span><span class="h">The most this browser stores. ${d} holds the whole model.</span></div>`
    + `<div class="dk-row"><span class="k">RAM for the disk tier</span><span class="v">${ramGB > 0 ? `${num(ramGB)} GB` : "None"}</span><span class="h">${ramGB > 0 ? "The experts it reads most stay in this device's memory." : "This device has no disk tier, so it holds no layers from disk."}</span></div>`
    + `<div class="dk-row"><span class="k">Stored copy</span><button type="button" class="dk-del"${storedBytes > 0 ? "" : " disabled"}>${armed && storedBytes > 0 ? `Delete ${sizeText(storedBytes)}?` : "Delete stored copy"}</button><span class="h">${esc(armed && storedBytes > 0 ? "Press again to delete it." : storedText(storedBytes))}</span></div>`;
}
