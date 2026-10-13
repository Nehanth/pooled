// The picker's part for a model kept on disk (room/models.js DISK_COPY), in the page: a plain line under the rows
// (what it is, what it needs), why it is short under Start, and the right column's "What this device gives" group: the pledge stepper (room.js's, relabelled "GPU memory", its pledge
// paragraph swapped for a one-line hint), then Disk (#ap-flash). Shown only while that model is picked. (Its row,
// black and twinkling when picked, with its fit chip, is room.js renderLadder's; the pool's line is renderPool's.)
// The words and HTML are room/diskpick.js's; the answer, the stored size, Allow and Free are #368's, through
// room/diskstore.js (one stored answer, one question: room.js askTabDisk; one delete: room.js freeTabDisk).
import { diskCopyOf, groupTitle, settingsHTML, GPU_HINT, pickNote, shortNote } from "./diskpick.js";
import { diskStore } from "./diskstore.js";

const $ = (id) => document.getElementById(id);
const START = "Start the model", GIVES = "This device gives";
let shown = null, stored = null, asked = 0, armed = 0, last = {}, covers = null;

// Show (or hide) the disk model's settings for the picked model `key`. info: { ramGB, stores, needGB, fits, shareBytes } (this tab's
// disk-tier RAM, room/flashdisk.js tabDiskMeta's, or null without a tier; whether some device in the room can store it,
// the GPU memory it needs and whether it fits, and what this tab would copy to its disk in this room: roomFitFor's share,
// null when not known). Cheap when nothing changed: it runs on every room update.
export function showDiskPick(key, info = last) {
  const m = diskCopyOf(key), on = !!m;
  last = info;
  for (const id of ["ap-flash", "ap-for"]) $(id).hidden = !on;
  $("ap-me").classList.toggle("flash", on);
  if (covers == null) covers = $("ap-covers").textContent;
  if (!on) { if (shown) { $("ai-start").textContent = START; $("ap-me-k").textContent = GIVES; $("ap-covers").textContent = covers; } shown = null; asked = 0; disarm(); return; }
  if (asked !== key) stored = null;
  const state = diskStore.state(key);
  $("ai-start").textContent = `Start ${m.name}`;
  $("ap-for").textContent = groupTitle(m);
  $("ap-me-k").textContent = "GPU memory";
  $("ap-covers").textContent = GPU_HINT;
  $("ap-note").textContent = pickNote(m, info.needGB ?? null, info.stores !== false, !!info.fits, { state, shareBytes: info.shareBytes ?? null, storedBytes: stored });
  const sig = JSON.stringify([key, state, stored, !!armed, info]);
  if (sig === shown) return;
  shown = sig;
  $("ap-flash").innerHTML = settingsHTML(m, { state, storedBytes: stored, ramGB: info.ramGB ?? null, armed: !!armed, shareBytes: info.shareBytes ?? null });
  // what is stored: asked once per pick (and after Free, or Room settings freeing it: refreshDiskPick)
  if (asked !== key) { asked = key; refresh(key); }
}
async function refresh(key) {
  const s = await diskStore.storedBytes(key);
  if (asked !== key) return;
  stored = s; shown = null; showDiskPick(key);
}
function disarm() { if (armed) { clearTimeout(armed); armed = 0; } }
// the stored copy changed elsewhere (Room settings' "Free disk space", a load that copied it): read it again
export function refreshDiskPick() { const key = $("ai-model").value; if (diskCopyOf(key)) { asked = 0; shown = null; showDiskPick(key); } }

// why the picked disk model doesn't fit, under Start: by how much and the fix this device's disk answer leaves
export const diskShortNote = (key, shortGB, stores = true) => shortNote(shortGB, diskStore.state(key), stores);
// this device's disk answer for `key` (room/diskstore.js diskState), for the pool's fix line (room.js renderPool)
export const diskStateOf = (key) => diskStore.state(key);

// Wire the buttons: Allow (the same stored answer the disk question saves); Delete (two presses, like Room settings'
// "Free disk space"). toast(text): the room's, for a delete that could not run.
export function initDiskPick({ toast = () => {} }) {
  // the same Allow, under Start (p2p.html #ai-allow-disk: room.js updateNeed shows it), then on to Start
  $("ai-allow-disk").addEventListener("click", () => {
    const key = $("ai-model").value;
    if (!diskCopyOf(key)) return;
    diskStore.allow(key); shown = null; showDiskPick(key);
    $("ai-start").focus();
  });
  $("ap-flash").addEventListener("click", async (e) => {
    const b = e.target.closest("button[data-act]"), key = $("ai-model").value;
    if (!b || !diskCopyOf(key)) return;
    if (b.dataset.act === "allow") { diskStore.allow(key); shown = null; showDiskPick(key); return; }
    if (!armed) {
      const t = armed = setTimeout(() => { if (armed === t) { armed = 0; shown = null; showDiskPick($("ai-model").value); } }, 6000);
      shown = null; showDiskPick(key); return;
    }
    disarm();
    b.disabled = true;
    const r = await diskStore.deleteCopy(key).catch((err) => ({ refused: String(err?.message || err) }));
    if (r?.refused) toast(r.refused);
    asked = 0; shown = null; showDiskPick($("ai-model").value);
  });
}
