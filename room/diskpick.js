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
// fits where smaller models don't, and what more devices do. needGB: roomFitFor's GPU figure (null: not known yet).
// stores: some device in the room can store it (roomFitFor stores, room/pledge.js storageOf); false: no GPU figure is
// said at all (the planner's would be the whole model on GPUs), and the fix is said once, under Start and in the pool.
// fits: roomFitFor's verdict (with stores false, the whole model fits in this room's GPU memory). disk: this device's
// part, { state (room/diskstore.js diskState), shareBytes (roomFitFor share, null: not known), storedBytes (what is
// saved: room/flashdisk.js flashDiskBytes, null: not known) }: what the first Start downloads, and that it happens once.
// The GPU figure is rounded up to a whole GB here (8.1 and 8.6 both read "about 9 GB", so a device joining doesn't
// grow it beside "makes it faster"); the pool's line keeps the exact one.
export const pickNote = (m, needGB = null, stores = true, fits = false, disk = {}) => !stores
  ? (fits ? `${m.name} is a ${m.params} model. This room's GPU memory holds all of it.`
    : `${m.name} is a ${m.params} model that keeps most of itself on disk.`)
  : [`${m.name} is a ${m.params} model. It keeps most of itself on disk, so it needs `
    + (needGB > 0 ? `only about ${Math.ceil(needGB)} GB of GPU memory.` : "far less GPU memory than its size."),
  downloadNote(m, disk), "Alone it reads from disk, so answers start slowly. Each device that joins makes it faster."].filter(Boolean).join(" ");
// pickNote's sentence on what Start downloads to this device, by its disk answer: "" when it downloads nothing here
function downloadNote(m, { state = null, shareBytes = null, storedBytes = null } = {}) {
  const need = sizeText(shareBytes > 0 ? shareBytes : m.diskBytes);
  if (state === "ask") return shareBytes === 0 ? "" : `Start asks first, then downloads ${need} to this device's disk, once.`;
  if (state !== "allow" || shareBytes === 0) return "";
  const share = shareBytes > 0 ? shareBytes : m.diskBytes;
  if (storedBytes > 0 && storedBytes >= share * 0.99) return "It's already saved on this device.";
  return storedBytes > 0 ? `Start finishes saving ${need} to this device's disk.` : `The first Start downloads ${need} to this device's disk, once; that can take an hour or more on home internet.`;
}
// What to do when no device in the room can store it, by this device's disk answer (room/diskstore.js diskState): under
// Start (shortNote) and, when it is not this device's answer, under the pool's Storage line too (poolFix). Asked or a
// no: this device could store it, so say that its disk storage is off (Start's Allow button, allowLabel, is the fix)
export const storeFix = (state) => state === "ask" || state === "deny" ? "Disk storage is off on this device. Allow it, or invite a computer that can store it."
  : state === "url" ? "Invite a computer that can." : "Invite a computer with a GPU and free disk space.";
// under Start, when it doesn't fit: by how much, and the fix this device's disk answer (diskState) leaves. stores
// false: no device here can store it, which is the gap (not a GPU figure for the whole model)
const FIX = (state) => state === "ask" || state === "deny" ? "Allow disk storage, or invite more devices."
  : state === "allow" ? "Invite more devices, or give more GPU memory." : "Invite more devices with GPUs.";
export const shortNote = (shortGB, state, stores = true) => stores ? `${num(shortGB)} GB of GPU memory short. ${FIX(state)}`
  : state === "ask" || state === "deny" ? storeFix(state) : `Needs a device that can store it. ${storeFix(state)}`;
// the button under Start that turns this device's disk storage on (p2p.html #ai-allow-disk): with what it would save
// (shareBytes: roomFitFor share; null or 0: the whole copy)
export const allowLabel = (m, shareBytes = null) => `Allow disk storage (${sizeText(shareBytes > 0 ? shareBytes : m.diskBytes)})`;

// ---- The pool (room.js renderPool, while a disk model is picked): a GPU memory bar and a Storage bar ----
// What each device brings beyond GPU memory is room/pledge.js storageOf: { kind: "disk" | "ram" | null, ask, ramGB }.
// The figures are roomFitFor's (needGB, needDisk, fits, shortGB), never a whole-model GPU figure.
// -> { text, num (GB, or null), chip (or null), ok }
export function poolGpuLine(m, { needGB = null, needDisk = false, fits = false, shortGB = 0, nStore = 0 } = {}) {
  if (needDisk && needGB != null) return { text: `${m.name} needs`, num: num(needGB), chip: fits ? "fits" : `${num(shortGB)} GB short`, ok: !!fits };
  if (fits) return { text: `${m.name} fits in this room's GPU memory`, num: null, chip: "fits", ok: true };
  if (nStore === 0) return { text: `${m.name} runs once a device stores it`, num: null, chip: null, ok: false };
  return { text: "Not enough GPU memory in this room yet", num: null, chip: "short", ok: false };
}
// state: this device's disk answer (room/diskstore.js diskState): asked or a no, nothing here stores it because this
// device's disk storage is off, not because no device could
// -> { text, chip (or null), ok }
export function poolStoreLine(m, { nDisk = 0, nRam = 0, needDisk = false, fits = false, state = null } = {}) {
  const size = sizeText(m.diskBytes);
  // "ready", not "fits": no free-space check has run (a device that asks first says so in its own tag, devStoreTag)
  if (nDisk > 0) return { text: `${nDisk > 1 ? `Split across ${nDisk} devices' disks` : "Kept on disk"}, ${size} in all`, chip: "ready", ok: true };
  if (nRam > 0) return { text: `Held in spare memory on ${nRam} device${nRam === 1 ? "" : "s"}`, chip: "ready", ok: true };
  if (!needDisk && fits) return { text: "Not needed here, it all fits in GPU memory", chip: null, ok: false };
  if (state === "ask" || state === "deny") return { text: "Disk storage is off on this device", chip: "off", ok: false };
  return { text: "No device here can store it", chip: "needs disk", ok: false };
}
// the fix under the Storage line: only when no device can store it, or the GPU memory is short with storage present
// (no planner figure). state: this device's disk answer (room/diskstore.js diskState); asked or a no, the fix is said
// once, under Start, beside its Allow button
export function poolFix(state, { nStore = 0, fits = false, needDisk = false, needGB = null } = {}) {
  if (nStore === 0) return (!needDisk && fits) || state === "ask" || state === "deny" ? "" : storeFix(state);
  return !fits && (needGB == null || !needDisk) ? "Give more GPU memory below, or invite a device." : "";
}
// a device row's tag in the pool's list, and the words its tooltip uses
// share: what this tab would copy to its disk in this room (roomFitFor share: bytes, 0 when the room doesn't need its
// disk), null for a room node (it reads its own files) or when not known. off: this device, whose disk storage is off
// (a no), so not "no disk". m: the model (its whole copy, "up to 92.4 GB disk", while a tab's share is not known). A
// room node on disk says the RAM it reads into ("43 GB RAM · disk")
export const devStoreTag = (st, share = null, off = false, m = null) => st?.kind === "disk"
  ? (st.ramGB > 0 ? `${num(st.ramGB)} GB RAM · disk`
    : (share > 0 ? `${sizeText(share)} disk` : share === 0 ? "disk not needed" : m ? `up to ${sizeText(m.diskBytes)} disk` : "disk")
      + (st.ask && share !== 0 ? " · will ask" : ""))
  : st?.kind === "ram" ? `${num(st.ramGB)} GB RAM` : st?.kind == null && off ? "disk off" : "no disk";
export const devStoreTitle = (name, gb, st, share = null) => `${name}: ${gb} GB GPU memory, `
  + (st?.kind === "disk" ? (st.ramGB > 0 ? `${num(st.ramGB)} GB RAM, ` : "") + (st.ask ? "asks before saving to disk" : "disk") + (share > 0 ? `, saves ${sizeText(share)} there` : "")
    : st?.kind === "ram" ? `${num(st.ramGB)} GB RAM` : "no disk");

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
const WHY = { url: "This link turns disk storage off on this device", small: "This device doesn't have enough spare memory to read from disk",
  none: "This browser can't store it on disk" };
const TIP = { url: "?tabdisk=0 or ?tabram=0 in the address", small: "tabRamCap leaves no RAM for the disk tier",
  none: "No disk tier: a phone, no WebGPU, or no OPFS (origin-private file system)" };
// storedBytes: room/flashdisk.js flashDiskBytes (null: unknown); ramGB: this tab's disk-tier RAM (tabDiskMeta's ramGB:
// TAB_RAM_GB held to tabRamCap; null without a tier); armed: Free was pressed once (the second press frees);
// shareBytes: what this device would copy in this room (roomFitFor share: its layers' experts and, on the device that
// holds its layer, the n-gram table; 0 when the room doesn't need its disk), null when not known (the whole copy then)
export function diskRow(m, { state = "ask", storedBytes = null, ramGB = null, armed = false, shareBytes = null } = {}) {
  const stored = storedBytes > 0 ? sizeText(storedBytes) : storedBytes === 0 ? "0 GB" : "?";
  const total = sizeText(m.diskBytes), ram = ramGB > 0 ? `${num(ramGB)} GB of RAM` : "";
  const part = shareBytes > 0 && shareBytes < m.diskBytes * 0.99, need = part ? sizeText(shareBytes) : total;
  const whose = part ? ` of ${total}` : "";
  const parts = [];
  if (state === "allow") {
    if (shareBytes === 0 && !(storedBytes > 0)) parts.push("This room doesn't need this device's disk right now");
    else {
      parts.push(storedBytes > 0 ? (shareBytes != null && storedBytes >= shareBytes * 0.99 ? `${stored} saved on this device` : `${stored} of ${need} saved on this device`)
        : `Saves ${need}${whose} to this device's disk the first time it starts`);
    }
  } else if (state === "ask" || state === "deny") {
    parts.push(shareBytes === 0 ? `Lets ${m.name} save up to ${total} to this device's disk when the room needs it`
      : `Lets ${m.name} save ${need}${whose} to this device's disk, once`);
    if (storedBytes > 0) parts.push(`${stored} saved`);
  } else {
    parts.push(WHY[state] || WHY.none);
    if (storedBytes > 0) parts.push(`${stored} saved`);
  }
  const acts = [];
  // aria: what a screen reader says for the button (not just "Allow")
  if (state === "ask" || state === "deny") acts.push({ act: "allow", label: "Allow", aria: `Allow disk storage for ${m.name}` });
  if (storedBytes > 0) acts.push({ act: "free", label: armed ? `Delete ${stored}?` : "Delete", aria: `${armed ? "Confirm: delete" : "Delete"} ${m.name}'s saved copy (${stored})` });
  const can = state === "allow" || state === "ask" || state === "deny";
  // the RAM it reads the copy with goes in the tooltip: one line less, so the column doesn't jump as devices join
  const title = !can ? TIP[state] || TIP.none : ram && shareBytes !== 0 ? `Uses ${ram} to read it` : "";
  return { k: "Disk storage", chip: DISK_STATE[state] ?? DISK_STATE.none, ok: state === "allow", h: parts.join(" · "), title, acts };
}

// ---- HTML (p2p.html #ap-flash .dk-row, with the rows' chips: .fig, b for "Allowed", .more otherwise) ----
const chipHTML = (text, ok) => !text ? "" : ok ? `<b>${esc(text)}</b>` : `<span class="more">${esc(text)}</span>`;
export function diskRowHTML(d) {
  const btn = (a) => `<button type="button" class="${a.act === "free" ? "dk-del" : "dk-act"}" data-act="${a.act}"${a.aria ? ` aria-label="${esc(a.aria)}"` : ""}>${esc(a.label)}</button>`;
  return `<div class="dk-row"><span class="k">${esc(d.k)}</span><span class="fig">${chipHTML(d.chip, d.ok)}${d.acts.map(btn).join("")}</span><span class="h"${d.title ? ` title="${esc(d.title)}"` : ""}>${esc(d.h)}</span></div>`;
}
// the group's row under the pledge stepper (relabelled "GPU memory"): Disk storage, with the RAM its tier reads it with
export function settingsHTML(m, disk) {
  return diskRowHTML(diskRow(m, disk));
}
