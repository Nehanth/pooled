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

// "20", "8.5", "1.5": a number as the picker writes it
const num = (n) => String(+(+n).toFixed(1));
// bytes as room.js fmtBytes writes them (the disk question's number): "92.4 GB", "640 MB"
export const sizeText = (bytes) => bytes >= 2 ** 30 ? `${(bytes / 2 ** 30).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 2 ** 20))} MB`;

// the row's chip: "125B · runs from disk"
export const rowChip = (m) => [m.params, "runs from disk"].filter(Boolean).join(" · ");
// the chip as the row renders it: the size drops away on a phone or beside "N GB short" (p2p.html .dkz)
export const rowChipHTML = (m) => `<span class="dkc">${m.params ? `<span class="dkz">${esc(m.params)} · </span>` : ""}runs from disk</span>`;
// its size in words: "125B (6B active)"
export const paramsText = (m) => m.active ? `${m.params} (${m.active} active)` : m.params;
// the row's tooltip: "Qwen3.8 Flash: a 125B model (6B active) that keeps most of itself on disk"
export const rowTitle = (m) => `${m.name}: a ${paramsText(m)} model that keeps most of itself on disk, so it needs far less GPU memory`;
// the line under the rows while it is picked (in place of the download note), in plain words: what it is, why it
// fits where smaller models don't, and what more devices do. needGB / needDisk: roomFitFor's (null: not known yet;
// needDisk false: no device here streams from disk, so needGB is the whole model on GPUs)
export const pickNote = (m, needGB = null, needDisk = true) => !needDisk && needGB > 0
  ? `${m.name} is a ${m.params} model that normally keeps most of itself on disk. With no disk storage in this room, it needs ${num(needGB)} GB of GPU memory.`
  : `${m.name} is a ${m.params} model. It keeps most of itself on disk, so it needs `
    + (needGB > 0 ? `only ${num(needGB)} GB of GPU memory` : "far less GPU memory than its size") + ". Each device that joins makes it faster.";
// under Start, when it doesn't fit: by how much, and the fix this device's disk answer (diskState) leaves
export const shortNote = (shortGB, state) => `${num(shortGB)} GB of GPU memory short. `
  + (state === "ask" || state === "deny" ? "Allow disk storage, or invite more devices."
    : state === "allow" ? "Invite more devices, or give more GPU memory." : "Invite more devices with GPUs.");

// ---- Disk ----
// state: room/diskstore.js diskState: "ask" | "allow" | "deny" | "url" (the address keeps it off: ?tabdisk=0, ?tabram=0)
// | "small" (no RAM to spare for its tier: tabRamCap) | "none" (no disk tier at all: a phone, no OPFS, no WebGPU)
// (not asked or a no: no chip, the Allow button is the answer)
export const DISK_STATE = { allow: "On", ask: "", deny: "", url: "Unavailable", small: "Unavailable", none: "Unavailable" };
// the group's heading, like the pool's "The room pools"
export const groupTitle = () => "What this device gives";
// the GPU memory stepper's one-line hint while a disk model is picked (its pledge paragraph steps aside)
export const GPU_HINT = "How much of this device's graphics memory the model may use";
// why this device can't keep the copy, in plain words (the developer's detail goes in the tooltip)
const WHY = { url: "This link turns disk storage off", small: "This device doesn't have enough spare memory to read from disk",
  none: "This browser can't store it on disk" };
const TIP = { url: "?tabdisk=0 or ?tabram=0 in the address", small: "tabRamCap leaves no RAM for the disk tier",
  none: "No disk tier: a phone, no WebGPU, or no OPFS (origin-private file system)" };
// storedBytes: room/flashdisk.js flashDiskBytes (null: unknown); ramGB: this tab's disk-tier RAM (tabDiskMeta's ramGB:
// TAB_RAM_GB held to tabRamCap; null without a tier); needDisk: the room's fit streams from some device's disk
// (roomFitFor), false when the whole model must fit on GPUs; armed: Free was pressed once (the second press frees)
export function diskRow(m, { state = "ask", storedBytes = null, ramGB = null, needDisk = true, armed = false } = {}) {
  const stored = storedBytes > 0 ? sizeText(storedBytes) : storedBytes === 0 ? "0 GB" : "?";
  const total = sizeText(m.diskBytes), ram = ramGB > 0 ? `${num(ramGB)} GB of RAM` : "";
  const parts = [];
  if (state === "allow") {
    parts.push(storedBytes > 0 ? `${stored} of ${total} saved on this device` : `Saves ${total} to this device's disk the first time it starts`);
    if (ram) parts.push(`Uses ${ram} to read it`);
  } else if (state === "ask" || state === "deny") {
    parts.push(`Lets ${m.name} save ${total} to this device's disk, once, and read it with ${ram || "a little RAM"}`);
    if (storedBytes > 0) parts.push(`${stored} saved`);
  } else {
    parts.push((WHY[state] || WHY.none) + (needDisk ? "" : ", so the whole model has to fit in GPU memory"));
    if (storedBytes > 0) parts.push(`${stored} saved`);
  }
  const acts = [];
  if (state === "ask" || state === "deny") acts.push({ act: "allow", label: "Allow" });
  if (storedBytes > 0) acts.push({ act: "free", label: armed ? `Delete ${stored}?` : "Delete" });
  const can = state === "allow" || state === "ask" || state === "deny";
  return { k: "Disk storage", chip: DISK_STATE[state] ?? DISK_STATE.none, ok: state === "allow", h: parts.join(" · "), title: can ? "" : TIP[state] || TIP.none, acts };
}

// ---- HTML (p2p.html #ap-flash .dk-row, with the rows' chips: .fig, b for "Allowed", .more otherwise) ----
const chipHTML = (text, ok) => !text ? "" : ok ? `<b>${esc(text)}</b>` : `<span class="more">${esc(text)}</span>`;
export function diskRowHTML(d) {
  const btn = (a) => `<button type="button" class="${a.act === "free" ? "dk-del" : "dk-act"}" data-act="${a.act}">${esc(a.label)}</button>`;
  return `<div class="dk-row"><span class="k">${esc(d.k)}</span><span class="fig">${chipHTML(d.chip, d.ok)}${d.acts.map(btn).join("")}</span><span class="h"${d.title ? ` title="${esc(d.title)}"` : ""}>${esc(d.h)}</span></div>`;
}
// the group's row under the pledge stepper (relabelled "GPU memory"): Disk storage, with the RAM its tier reads it with
export function settingsHTML(m, disk) {
  return diskRowHTML(diskRow(m, disk));
}
