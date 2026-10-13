// The picker's part for a model kept on disk (room/models.js DISK_COPY), in the page: the cost row and consent under
// the rows, Start's label, "Not now" and Advanced. (Its row, black and twinkling when picked, is room.js renderLadder's.)
// The words and HTML are room/diskpick.js's; the answer, the stored size and the delete are #368's, through
// room/diskstore.js (one stored answer, one question: room.js askTabDisk).
import { diskCopyOf, costHTML, consentHTML, advRowsHTML, advSummary, startLabel } from "./diskpick.js";
import { diskStore } from "./diskstore.js";

const $ = (id) => document.getElementById(id);
const START = "Start the model";
let shown = null, stored = null, asked = 0, armed = 0;

// Show (or hide) the disk model's parts for the picked model `key`.
// Cheap when nothing changed: it runs on every room update (room.js updateNeed).
export function showDiskPick(key) {
  const m = diskCopyOf(key), on = !!m;
  for (const id of ["ai-disk", "ai-adv", "ai-notnow"]) $(id).hidden = !on;
  $("ap-note").hidden = on;
  if (!on) { if (shown) $("ai-start").textContent = START; shown = null; asked = 0; disarm(); return; }
  if (asked !== key) stored = null;
  const state = diskStore.state(key);
  $("ai-start").textContent = startLabel(m, state);
  const sig = `${key}|${state}|${stored}|${!!armed}`;
  if (sig === shown) return;
  if (shown?.split("|")[0] !== key) $("ai-adv").open = false;
  shown = sig;
  $("ai-disk").innerHTML = costHTML(m) + consentHTML(m, { state, storedBytes: stored });
  $("ai-adv").querySelector(".def").textContent = advSummary(m);
  $("ai-adv").querySelector(".dk-rows").innerHTML = advRowsHTML(m, { storedBytes: stored, armed: !!armed });
  // what is stored: asked once per pick (and after a delete, or Room settings freeing it: refreshDiskPick)
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

// Wire the buttons: Not now (let go of the pick), Delete stored copy (two presses, like Room settings' Free disk space).
// toast(text): the room's toast, for a delete that could not run (a load or the running model reads the copy)
export function initDiskPick({ notNow, toast = () => {} }) {
  $("ai-notnow").addEventListener("click", notNow);
  $("ai-adv").addEventListener("click", async (e) => {
    const b = e.target.closest(".dk-del"), key = $("ai-model").value;
    if (!b || !diskCopyOf(key)) return;
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
// Start pressed on a disk model while this device was not asked yet: the press is its answer (the button said
// "Allow N GB and start"), saved where the disk question saves it, so that question is not asked again
export function agreeOnStart(key) { if (diskCopyOf(key)) diskStore.allow(key); }
