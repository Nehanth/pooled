// The picker's part for a model kept on disk (room/models.js DISK_COPY), in the page: the cost row and consent under
// the rows, Start's label, "Not now", Advanced, the speed chart on the side, and the ✦ burst when its row is picked.
// The words and HTML are room/diskpick.js's; consent and storage go through room/diskstore.js.
import { diskCopyOf, costHTML, consentHTML, advRowsHTML, advSummary, chartHTML, startLabel, STAR } from "./diskpick.js";
import { diskStore } from "./diskstore.js";

const $ = (id) => document.getElementById(id);
const START = "Start the model";
let shown = null, stored = null, free = null, asked = 0, lastDevices = 1;

// Show (or hide) the disk model's parts for the picked model `key`. devices: WebGPU devices in the room now.
// Cheap when nothing changed: it runs on every room update (room.js updateNeed).
export function showDiskPick(key, devices = lastDevices) {
  lastDevices = devices;
  const m = diskCopyOf(key), on = !!m;
  for (const id of ["ai-disk", "ai-adv", "ai-notnow", "ap-speed"]) { const el = $(id); if (el) el.hidden = !on; }
  $("ap-note").hidden = on;
  $("ai-panel").classList.toggle("disk", on);
  if (!on) { if (shown) $("ai-start").textContent = START; shown = null; asked = 0; return; }
  if (asked !== key) stored = null;
  const agreed = diskStore.consent(key);
  $("ai-start").textContent = startLabel(m, agreed);
  const sig = `${key}|${agreed}|${stored}|${free}|${devices}`;
  if (sig === shown) return;
  if (shown?.split("|")[0] !== key) $("ai-adv").open = false;
  shown = sig;
  $("ai-disk").innerHTML = costHTML(m) + consentHTML(m, { agreed, storedBytes: stored, freeBytes: free });
  $("ai-adv").querySelector(".def").textContent = advSummary(m);
  $("ai-adv").querySelector(".dk-rows").innerHTML = advRowsHTML(m, { storedBytes: stored });
  $("ap-speed").innerHTML = chartHTML(m, devices);
  // what is stored and how much disk is free: asked once per pick (and after a delete), then shown
  if (asked !== key) { asked = key; refresh(key, devices); }
}
async function refresh(key, devices) {
  const [s, f] = await Promise.all([diskStore.storedBytes(key).catch(() => null), diskStore.freeBytes().catch(() => null)]);
  if (asked !== key) return;
  stored = s; free = f; shown = null; showDiskPick(key, devices);
}

// Wire the buttons: Not now (deselect), Delete stored copy, invite one
export function initDiskPick({ notNow, invite }) {
  $("ai-notnow").addEventListener("click", notNow);
  $("ap-speed").addEventListener("click", (e) => { if (e.target.closest(".dk-invite")) invite(); });
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

// The ✦ burst: 7 tiny stars twinkle out of the row's ✦ once, in the site's blue (transform and opacity only, on
// a fixed layer over the page, so the ladder's re-renders and the card's overflow never touch it). None under
// reduced motion.
export function sparkleFrom(el) {
  if (!el || matchMedia("(prefers-reduced-motion: reduce)").matches || typeof el.animate !== "function") return;
  const r = el.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
  const layer = document.createElement("div");
  layer.className = "dk-burst";
  layer.setAttribute("aria-hidden", "true");
  layer.style.cssText = `left:${cx}px;top:${cy}px`;
  const N = 7, done = [];
  for (let i = 0; i < N; i++) {
    const a = -Math.PI / 2 + (i / N) * Math.PI * 2 + (i % 2 ? 0.25 : -0.1), d = 16 + (i % 3) * 6, s = 5 + (i % 3) * 1.5;
    const g = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    g.setAttribute("viewBox", "0 0 16 16");
    g.style.cssText = `width:${s}px;height:${s}px;margin:${-s / 2}px 0 0 ${-s / 2}px`;
    g.innerHTML = `<path d="${STAR}" fill="currentColor"/>`;
    layer.appendChild(g);
    const x = Math.cos(a) * d, y = Math.sin(a) * d;
    done.push(g.animate([
      { transform: "translate(0,0) scale(.2) rotate(0deg)", opacity: 0 },
      { transform: `translate(${x * 0.5}px,${y * 0.5}px) scale(1.15) rotate(40deg)`, opacity: 1, offset: 0.3 },
      { transform: `translate(${x * 0.8}px,${y * 0.8}px) scale(.8) rotate(70deg)`, opacity: .8, offset: 0.6 },
      { transform: `translate(${x}px,${y}px) scale(.3) rotate(90deg)`, opacity: 0 },
    ], { duration: 900, delay: i * 18, easing: "cubic-bezier(.2,.7,.2,1)", fill: "both" }).finished);
  }
  document.body.appendChild(layer);
  Promise.all(done).catch(() => {}).finally(() => layer.remove());
}
