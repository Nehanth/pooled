// The picker's part for a model kept on disk (room/models.js DISK_COPY), in the page: the cost row and consent under
// the rows, Start's label, "Not now" and Advanced. (Its row, black and twinkling when picked, is room.js renderLadder's.)
// The words and HTML are room/diskpick.js's; consent and storage go through room/diskstore.js.
import { diskCopyOf, costHTML, consentHTML, advRowsHTML, advSummary, startLabel } from "./diskpick.js";
import { diskStore } from "./diskstore.js";

const $ = (id) => document.getElementById(id);
const START = "Start the model";
let shown = null, stored = null, free = null, asked = 0;

// Show (or hide) the disk model's parts for the picked model `key`.
// Cheap when nothing changed: it runs on every room update (room.js updateNeed).
export function showDiskPick(key) {
  const m = diskCopyOf(key), on = !!m;
  for (const id of ["ai-disk", "ai-adv", "ai-notnow"]) $(id).hidden = !on;
  $("ap-note").hidden = on;
  if (!on) { if (shown) $("ai-start").textContent = START; shown = null; asked = 0; return; }
  if (asked !== key) stored = null;
  const agreed = diskStore.consent(key);
  $("ai-start").textContent = startLabel(m, agreed);
  const sig = `${key}|${agreed}|${stored}|${free}`;
  if (sig === shown) return;
  if (shown?.split("|")[0] !== key) $("ai-adv").open = false;
  shown = sig;
  $("ai-disk").innerHTML = costHTML(m) + consentHTML(m, { agreed, storedBytes: stored, freeBytes: free });
  $("ai-adv").querySelector(".def").textContent = advSummary(m);
  $("ai-adv").querySelector(".dk-rows").innerHTML = advRowsHTML(m, { storedBytes: stored });
  // what is stored and how much disk is free: asked once per pick (and after a delete), then shown
  if (asked !== key) { asked = key; refresh(key); }
}
async function refresh(key) {
  const [s, f] = await Promise.all([diskStore.storedBytes(key).catch(() => null), diskStore.freeBytes().catch(() => null)]);
  if (asked !== key) return;
  stored = s; free = f; shown = null; showDiskPick(key);
}

// Wire the buttons: Not now (let go of the pick), Delete stored copy
export function initDiskPick({ notNow }) {
  $("ai-notnow").addEventListener("click", notNow);
  $("ai-adv").addEventListener("click", async (e) => {
    const b = e.target.closest(".dk-del"), key = $("ai-model").value;
    if (!b || !diskCopyOf(key)) return;
    b.disabled = true;
    await diskStore.deleteCopy(key).catch(() => false);
    asked = 0; shown = null; showDiskPick(key);
  });
}
// Start pressed on a disk model: the press is the consent (the button says how much it allows)
export function agreeOnStart(key) { if (diskCopyOf(key) && !diskStore.consent(key)) diskStore.setConsent(key, true); }
