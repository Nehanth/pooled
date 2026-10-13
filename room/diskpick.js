// The picker for a model that keeps a copy on this device's disk (room/models.js DISK_COPY): its row's chip and
// tooltip, and its per-device settings in the right column's "This device, for <model>" group, under the pledge
// stepper (room.js renderPool, relabelled "GPU memory"): Disk (this device's answer, what is stored, the RAM its disk
// tier reads it with: room/flashdisk.js tabDiskMeta, Allow / Free). Whether it fits is one answer, roomFitFor's: the
// row's own chip, the pool's line (and the dev need line), all from the same rows (room.js updateNeed, renderPool).
// Pure: the words come from the DISK_COPY entry and the figures room.js computes, so another big model reuses all of it.
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
// the chip as the row renders it: the size and architecture drop away on a phone (p2p.html .dkz), "from disk" stays
export const rowChipHTML = (m) => {
  const head = [m.params, m.arch].filter(Boolean).join(" · ");
  return `<span class="dkc">${head ? `<span class="dkz">${esc(head)} · </span>` : ""}from disk</span>`;
};
// its size in words: "125B (6B active)"
export const paramsText = (m) => m.active ? `${m.params} (${m.active} active)` : m.params;
// the row's tooltip: "Qwen3.8 Flash: 125B (6B active), MoE, its experts streamed from this device's disk"
export const rowTitle = (m) => `${m.name}: ${[paramsText(m), m.arch].filter(Boolean).join(", ")}, its experts streamed from this device's disk`;

// ---- Disk ----
// state: room/diskstore.js diskState: "ask" | "allow" | "deny" | "url" (the address keeps it off: ?tabdisk=0, ?tabram=0)
// | "small" (no RAM to spare for its tier: tabRamCap) | "none" (no disk tier at all: a phone, no OPFS, no WebGPU)
export const DISK_STATE = { allow: "Allowed", ask: "Not asked", deny: "Not allowed", url: "Not available", small: "Not available", none: "Not available" };
// the group's heading: "This device, for Qwen3.8 Flash"
export const groupTitle = (m) => `This device, for ${m.name}`;
// the GPU memory stepper's one-line hint while a disk model is picked (its pledge paragraph steps aside)
export const GPU_HINT = "The most this device's GPU holds for the room";
// why this device can't keep the copy, in plain words (the developer's detail goes in the tooltip)
const WHY = { url: "This page keeps the model off this device's disk", small: "This device has too little memory to spare for it",
  none: "This browser can't store it" };
const TIP = { url: "?tabdisk=0 or ?tabram=0 in the address", small: "tabRamCap leaves no RAM for the disk tier",
  none: "No disk tier: a phone, no WebGPU, or no OPFS (origin-private file system)" };
// storedBytes: room/flashdisk.js flashDiskBytes (null: unknown); ramGB: this tab's disk-tier RAM (tabDiskMeta's ramGB:
// TAB_RAM_GB held to tabRamCap; null without a tier); needDisk: the room's fit streams from some device's disk
// (roomFitFor), false when the whole model must fit on GPUs; armed: Free was pressed once (the second press frees)
export function diskRow(m, { state = "ask", storedBytes = null, ramGB = null, needDisk = true, armed = false } = {}) {
  const stored = storedBytes > 0 ? sizeText(storedBytes) : storedBytes === 0 ? "0 GB" : "?";
  const ram = ramGB > 0 ? `${num(ramGB)} GB of RAM to read it` : "";
  const parts = [];
  if (state === "allow") parts.push([`${stored} of ${sizeText(m.diskBytes)} stored`, ram].filter(Boolean).join(" · "));
  else if (state === "ask" || state === "deny") {
    parts.push(`Allow lets ${m.name} store up to ${sizeText(m.diskBytes)} on this device's disk${ram ? ` and use ${ram}` : ""}`
      + (needDisk ? "" : "; without it, the whole model must fit on GPUs"));
    if (storedBytes > 0) parts.push(`${stored} stored`);
  } else {
    parts.push((WHY[state] || WHY.none) + (needDisk ? "" : ", so the whole model must fit on GPUs"));
    if (storedBytes > 0) parts.push(`${stored} stored`);
  }
  const acts = [];
  if (state === "ask" || state === "deny") acts.push({ act: "allow", label: "Allow" });
  if (storedBytes > 0) acts.push({ act: "free", label: armed ? `Free ${stored}?` : "Free" });
  const can = state === "allow" || state === "ask" || state === "deny";
  return { k: "Disk", chip: DISK_STATE[state] || DISK_STATE.none, ok: state === "allow", h: parts.join(" · "), title: can ? "" : TIP[state] || TIP.none, acts };
}

// ---- HTML (p2p.html #ap-flash .dk-row, with the rows' chips: .fig, b for "Allowed", .more otherwise) ----
const chipHTML = (text, ok) => ok ? `<b>${esc(text)}</b>` : `<span class="more">${esc(text)}</span>`;
export function diskRowHTML(d) {
  const btn = (a) => `<button type="button" class="${a.act === "free" ? "dk-del" : "dk-act"}" data-act="${a.act}">${esc(a.label)}</button>`;
  return `<div class="dk-row"><span class="k">${esc(d.k)}</span><span class="fig">${chipHTML(d.chip, d.ok)}${d.acts.map(btn).join("")}</span><span class="h"${d.title ? ` title="${esc(d.title)}"` : ""}>${esc(d.h)}</span></div>`;
}
// the group's row under the pledge stepper (relabelled "GPU memory"): Disk, with the RAM its tier reads it with
export function settingsHTML(m, disk) {
  return diskRowHTML(diskRow(m, disk));
}
