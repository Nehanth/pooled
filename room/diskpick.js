// The picker for a model that keeps a copy on this device's disk (room/models.js DISK_COPY): its row's chip, the
// three cost figures, the consent line, the Start label and the Advanced rows.
// Pure: every word and number comes from the model's DISK_COPY entry, so another big model reuses all of it.
// room.js puts the HTML in place; room/diskstore.js answers consent, stored size, delete and free disk.
import { DISK_COPY } from "./models.js";
import { esc } from "./markdown.js";

export const diskCopyOf = (key) => DISK_COPY[key] || null;

// the ✦ (site/js/sparkle.js's star)
export const STAR = "M8 0C8.6 4.6 11.4 7.4 16 8 11.4 8.6 8.6 11.4 8 16 7.4 11.4 4.6 8.6 0 8 4.6 7.4 7.4 4.6 8 0Z";
export const starSVG = (cls = "") => `<svg class="${cls}" viewBox="0 0 16 16" aria-hidden="true"><path d="${STAR}" fill="currentColor"/></svg>`;

// "92", "8.5", "1.5": a number as the picker writes it
const num = (n) => String(+(+n).toFixed(1));
// a GB amount from bytes: "92.0 GB", "640 MB"
export const sizeText = (bytes) => bytes >= 2 ** 30 ? `${(bytes / 2 ** 30).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 2 ** 20))} MB`;

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

// the three figures: [{ k, v, u, approx, note }]
export function costFigures(m) {
  const more = (m.speeds || []).length > 1;
  return [
    { k: "Stored once", v: num(m.diskGB), u: "GB", note: "Copied once, then starts fast" },
    { k: "Memory", v: num(m.memGB), u: "GB", approx: true, note: "For the experts it uses most" },
    { k: "Speed", v: speedRange(m), u: "tok/s", note: more ? "Gets faster as devices join" : "Measured" },
  ];
}

// what this device agrees to, as one sentence (split so the amount can be bold)
export const consentParts = (m) => ["This device keeps a copy of the model: ", `up to ${num(m.diskGB)} GB in this browser, one time.`, " Delete it any time."];
export const consentText = (m) => consentParts(m).join("");
// once agreed: what is there now (storedBytes null: unknown)
export const keptText = (m, storedBytes) => storedBytes > 0
  ? `This device keeps its copy of the model: ${sizeText(storedBytes)} in this browser. Delete it under Advanced.`
  : `This device keeps a copy of the model, up to ${num(m.diskGB)} GB in this browser. Delete it under Advanced.`;
// the free disk line: "412 GB free on this disk", or "" when it is unknown
export const freeText = (freeBytes) => freeBytes > 0 ? `${Math.floor(freeBytes / 2 ** 30)} GB free on this disk` : "";
// Start: asks for the space the first time, then just starts
export const startLabel = (m, agreed) => agreed ? `Start ${m.name}` : `Allow ${num(m.diskGB)} GB and start`;
// Advanced's summary line: "92 GB disk · 20 GB memory"
export const advSummary = (m) => `${num(m.diskGB)} GB disk · ${num(m.memGB)} GB memory`;
// the stored copy's line: what is on this disk now
export const storedText = (storedBytes) => storedBytes > 0 ? `${sizeText(storedBytes)} on this disk. Deleting frees the space; the next start copies it again.`
  : storedBytes === 0 ? "Nothing stored on this device yet." : "Not known in this browser.";

// ---- HTML ----
export function costHTML(m) {
  return `<dl class="dk-cost" aria-label="What it takes">${costFigures(m).map((f) =>
    `<div><dt>${esc(f.k)}</dt><dd>${f.approx ? '<span class="ap">~</span>' : ""}${esc(f.v)}<small>${esc(f.u)}</small></dd><p>${esc(f.note)}</p></div>`).join("")}</dl>`;
}
// the consent (or what is kept, once agreed) and the free disk when known
export function consentHTML(m, { agreed = false, storedBytes = null, freeBytes = null } = {}) {
  const [a, b, c] = consentParts(m), free = freeText(freeBytes);
  const share = freeBytes > 0 ? Math.min(100, m.diskGB * 2 ** 30 / freeBytes * 100) : 0;
  return `<p class="dk-agree">${agreed ? esc(keptText(m, storedBytes)) : `${esc(a)}<b>${esc(b)}</b>${esc(c).replace(" time.", "&nbsp;time.")}`}</p>`
    + (free && !agreed ? `<span class="dk-disk"><i style="--f:${share.toFixed(1)}%" aria-hidden="true"></i><span>${esc(free)}</span></span>` : "");
}
// Advanced's rows: disk, memory, the stored copy with its delete
export function advRowsHTML(m, { storedBytes = null } = {}) {
  return `<div class="dk-row"><span class="k">Disk to use</span><span class="v">${num(m.diskGB)} GB</span><span class="h">The most this browser stores. ${num(m.diskGB)} GB holds the whole model.</span></div>`
    + `<div class="dk-row"><span class="k">Memory for the expert cache</span><span class="v">~${num(m.memGB)} GB</span><span class="h">The experts it uses most stay in memory.</span></div>`
    + `<div class="dk-row"><span class="k">Stored copy</span><button type="button" class="dk-del"${storedBytes > 0 ? "" : " disabled"}>Delete stored copy</button><span class="h">${esc(storedText(storedBytes))}</span></div>`;
}
