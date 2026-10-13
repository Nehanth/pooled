// The picker for a model that keeps a copy on this device's disk (room/models.js DISK_COPY): its row's chip and
// tooltip, and its two settings under Start: GPU memory (what the room puts on GPUs, against the planner's own
// minimum) and Disk (this device's answer, what is stored, the RAM its disk tier uses, Allow / Free). Pure: the words come from the DISK_COPY entry and the figures room.js computes, so another big
// model reuses all of it.
// The consent is room/flashdisk.js's (#368): one stored answer per site (TAB_DISK_KEY) and one question (room.js
// askTabDisk); the Disk row shows that answer (room/diskstore.js diskState) and its Allow saves the same one.
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

// the row's chip: "125B · MoE · from disk"
export const rowChip = (m) => [m.params, m.arch, "from disk"].filter(Boolean).join(" · ");
// its size in words: "125B (6B active)"
export const paramsText = (m) => m.active ? `${m.params} (${m.active} active)` : m.params;
// the row's tooltip: "Qwen3.8 Flash: 125B (6B active), MoE, its experts streamed from this device's disk"
export const rowTitle = (m) => `${m.name}: ${[paramsText(m), m.arch].filter(Boolean).join(", ")}, its experts streamed from this device's disk`;

// ---- GPU memory ----
// needGB / hasGB / fits / shortGB: the figures room.js updateNeed shows (roomFitFor's needGB for a disk model: the
// planner's least GPU memory for this room, with its disk tiers; roomFit's verdict); needDisk: the rest streams from
// disk; mineGB: what this device gives (its pledge)
export function gpuRow({ needGB = null, hasGB = 0, fits = false, shortGB = 0, needDisk = false, mineGB = null } = {}) {
  const v = needGB == null ? `room ${num(hasGB)} GB` : `needs ${num(needGB)} GB · room ${num(hasGB)} GB`;
  const chip = fits ? "fits" : `${num(shortGB)} GB short`;
  const mine = mineGB > 0 ? ` This device gives ${num(mineGB)} GB.` : "";
  const h = needGB == null ? `What the room's devices give.${mine}`
    : needDisk ? `The least the room's GPUs need; the rest streams from disk.${mine}`
    : `No disk tier here, so the whole model must fit on GPUs.${mine}`;
  return { k: "GPU memory", v, chip, fits, h };
}

// ---- Disk ----
// state: room/diskstore.js diskState: "ask" | "allow" | "deny" | "url" (?tabdisk=0) | "none" (a phone, no OPFS)
export const DISK_STATE = { allow: "Allowed", ask: "Not asked yet", deny: "Not allowed", url: "Not available", none: "Not available" };
// storedBytes: room/flashdisk.js flashDiskBytes (null: unknown); ramGB: this tab's disk-tier RAM (tabDiskMeta's ramGB:
// TAB_RAM_GB held to tabRamCap; null without a tier); armed: Free was pressed once (the second press frees)
export function diskRow(m, { state = "ask", storedBytes = null, ramGB = null, armed = false } = {}) {
  const stored = storedBytes > 0 ? sizeText(storedBytes) : storedBytes === 0 ? "0 GB" : "?";
  const can = state === "allow" || state === "ask" || state === "deny";
  const parts = can ? [`${stored} of ${sizeText(m.diskBytes)} stored`, ...(ramGB > 0 ? [`${num(ramGB)} GB RAM for its cache`] : [])]
    : [state === "url" ? "This page keeps the model off this device's disk" : "This browser can't store it (phones and some browsers can't)"];
  // the developer's detail, only as the tooltip
  const title = state === "url" ? "?tabdisk=0 in the address" : state === "none" ? "No disk tier: a phone, no OPFS (origin-private file system), or ?tabram=0" : "";
  if (state === "ask") parts.push("Start asks first");
  if (!can && storedBytes > 0) parts.push(`${stored} stored`);
  const acts = [];
  if (state === "ask" || state === "deny") acts.push({ act: "allow", label: "Allow" });
  if (storedBytes > 0) acts.push({ act: "free", label: armed ? `Free ${stored}?` : "Free" });
  return { k: "Disk", chip: DISK_STATE[state] || DISK_STATE.none, ok: state === "allow", h: parts.join(" · "), title, acts };
}

// ---- HTML (p2p.html .dk-row, with the rows' chips: .fig, b for "fits" / "Allowed", .more otherwise) ----
const chipHTML = (text, ok) => ok ? `<b>${esc(text)}</b>` : `<span class="more">${esc(text)}</span>`;
export function gpuRowHTML(g) {
  return `<div class="dk-row"><span class="k">${esc(g.k)}</span><span class="fig"><span class="nd">${esc(g.v)}</span>${chipHTML(g.chip, g.fits)}</span><span class="h">${esc(g.h)}</span></div>`;
}
export function diskRowHTML(d) {
  const btn = (a) => `<button type="button" class="${a.act === "free" ? "dk-del" : "dk-act"}" data-act="${a.act}">${esc(a.label)}</button>`;
  return `<div class="dk-row"><span class="k">${esc(d.k)}</span><span class="fig">${chipHTML(d.chip, d.ok)}${d.acts.map(btn).join("")}</span><span class="h"${d.title ? ` title="${esc(d.title)}"` : ""}>${esc(d.h)}</span></div>`;
}
export function settingsHTML(m, gpu, disk) {
  return `<div class="dk-rows">${gpuRowHTML(gpuRow(gpu))}${diskRowHTML(diskRow(m, disk))}</div>`;
}
