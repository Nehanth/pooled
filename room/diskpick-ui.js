// The picker's part for a model kept on disk (room/models.js DISK_COPY), in the page: "Not now" beside Start, and the
// right column's "This device, for <model>" group: the pledge stepper (room.js's, relabelled "GPU memory", its pledge
// paragraph swapped for a one-line hint), then Disk (#ap-flash). Shown only while that model is picked. (Its row,
// black and twinkling when picked, with its fit chip, is room.js renderLadder's; the pool's line is renderPool's.)
// The words and HTML are room/diskpick.js's; the answer, the stored size, Allow and Free are #368's, through
// room/diskstore.js (one stored answer, one question: room.js askTabDisk; one delete: room.js freeTabDisk).
import { diskCopyOf, groupTitle, settingsHTML, GPU_HINT } from "./diskpick.js";
import { diskStore } from "./diskstore.js";

const $ = (id) => document.getElementById(id);
const START = "Start the model", GIVES = "This device gives";
let shown = null, stored = null, asked = 0, armed = 0, last = {}, covers = null;

// Show (or hide) the disk model's settings for the picked model `key`. info: { ramGB, needDisk } (this tab's disk-tier
// RAM, room/flashdisk.js tabDiskMeta's, or null without a tier; whether the room's fit streams from a disk,
// roomFitFor's). Cheap when nothing changed: it runs on every room update.
export function showDiskPick(key, info = last) {
  const m = diskCopyOf(key), on = !!m;
  last = info;
  for (const id of ["ap-flash", "ap-for", "ai-notnow"]) $(id).hidden = !on;
  $("ap-note").hidden = on;
  $("ap-me").classList.toggle("flash", on);
  if (covers == null) covers = $("ap-covers").textContent;
  if (!on) { if (shown) { $("ai-start").textContent = START; $("ap-me-k").textContent = GIVES; $("ap-covers").textContent = covers; } shown = null; asked = 0; disarm(); return; }
  if (asked !== key) stored = null;
  const state = diskStore.state(key);
  $("ai-start").textContent = `Start ${m.name}`;
  $("ap-for").textContent = groupTitle(m);
  $("ap-me-k").textContent = "GPU memory";
  $("ap-covers").textContent = GPU_HINT;
  const sig = JSON.stringify([key, state, stored, !!armed, info]);
  if (sig === shown) return;
  shown = sig;
  $("ap-flash").innerHTML = settingsHTML(m, { state, storedBytes: stored, ramGB: info.ramGB ?? null, needDisk: info.needDisk !== false, armed: !!armed });
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

// Wire the buttons: Not now (let go of the pick); Allow (the same stored answer the disk question saves);
// Free (two presses, like Room settings' "Free disk space"). toast(text): the room's, for a Free that could not run.
export function initDiskPick({ notNow, toast = () => {} }) {
  $("ai-notnow").addEventListener("click", notNow);
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
