// Pooled room: signaling, WebRTC mesh, layer assignment, weight streaming and the
// generation loop (prefill, decode, speculative verify). Served with p2p.html at /room.
import { autotuneCoop, makeTokenizer, DenseEngine, argmax, fetchModelShard, shardTensorNames, gpuSelfTest, kernelMicroTests }
  from "./engine/engine.js";
import { f32ToF16, f16ToF32, parseGGUFHeader, ggufWeights, ggufShardBytes, GGML_EMBED, GGML_OUTPUT, GGML_FINAL_NORM,
  ggmlLayerNames, qwen35Weights, qwen35ShardBytes, qwen35MtpBytes, qwen35LayerNames, qwen35NamesFor, tokenizerFromGGUF, gpuUploadEntry, streamEntryToGPU }
  from "./engine/gguf.js";
import { Qwen35Engine } from "./engine/qwen35.js";
import { WIRE_F16, badF32, f32ToB64, packF16, unpackF16, asU16, packWire, unpackWire, asF32, b64ToF32, wireStats } from "./room/wire.js";
import { esc, md, mdChat } from "./room/markdown.js";
import { pickSampler, SAMPLING } from "./room/sampling.js";
import { chatRecipients } from "./room/visibility.js";
import { PrefixIndex } from "./harness/prefix.js";
import { MODELS, NEED_GB, PICKER, MAX_SEQ, MAX_NEW, MAX_NEW_THINKING, MIN_ROOM, maxSeqFor, kvBytesPerLayerPos } from "./room/models.js";
// the context window of the loaded engine (per model: room/models.js CTX; 2048 for the small ones)
const ctxMax = () => ai.engine?.maxSeq || MAX_SEQ;
// ?ckpt=N: keep the room's state after the last N answers on every device (GPU copies), so a
// regenerate, an edited question or a branch resumes from the longest saved turn instead of
// prefilling the whole conversation again. 0 turns it off.
const CKPT_MAX = Math.max(0, parseInt(new URLSearchParams(location.search).get("ckpt") ?? "2", 10) || 0);
import { makeLink, attachWire, wireReady, sendFrame, PROTOCOL, DROP_ALL } from "./room/transport.js";
import { PERSONAS, specials, fitContext, reusablePrefix } from "./room/conversation.js";
import { planSplit, planForSpeed, ladder, bestFit, codeFromLocation, pickModelHost } from "./room/plan.js";
import { qrSVG } from "./room/qr.js";
import { lookupDrafts } from "./room/lookup.js";
import { drawCard } from "./room/card.js";
import { probe as preflight, deviceKind } from "./room/preflight.js";
import { computeScreen } from "./room/compute.js";
import { working, liveWords } from "./room/working.js";

// Hidden-state transport (room/transport.js). ?wire=off falls back to PeerJS messages;
// ?wire=slice uses one sliced channel; ?wire=stripeN spreads slices over N peer connections.
const WIRE = (new URLSearchParams(location.search).get("wire") || "stripe4").toLowerCase();
const WIRE_STRIPES = WIRE === "off" ? 0 : WIRE.startsWith("stripe") ? Math.max(1, Math.min(8, parseInt(WIRE.slice(6), 10) || 1)) : 1;
// Signaling: ?signal=host:port points PeerJS at our own PeerServer (the emulator and big
// rooms use one); default is the public PeerJS cloud.
const SIGNAL = new URLSearchParams(location.search).get("signal");
const SIGNAL_OPTS = SIGNAL ? (() => { const [host, port] = SIGNAL.split(":"); return { host, port: +port || 443, path: "/", secure: location.protocol === "https:" }; })() : {};

// Topology: every device keeps ONE link to the host (control, roster, tokens). Data links
// between chain neighbours open when the layers are dealt (ensureLink), so a room of N
// devices has N-1 host links plus N-1 chain links, not N*(N-1)/2. Workers learn about the
// other devices from the host's roster message and draw cards from it.
const members = new Map();   // id -> { name, meta } for everyone in the room except me
const cards = new Map();     // id -> card element

const $ = (id) => document.getElementById(id);
// a short note top right that goes by itself; at most three at a time (the oldest goes first).
// sw: a device's colour for the dot (joined / left)
function toast(text, { sw = null, kind = "" } = {}) {
  const box = $("toasts"), t = document.createElement("div");
  t.className = "toast" + (kind ? " " + kind : "");
  if (sw) t.style.setProperty("--sw", sw);
  t.textContent = text;
  box.appendChild(t);
  while (box.children.length > 3) box.firstElementChild.remove();
  setTimeout(() => t.remove(), 4200);
}
// someone joined or left: a toast, but not for the devices already here when this tab came in
let roomSince = Infinity;
const presence = (name, joined) => { if (performance.now() - roomSince > 2500) toast(`${name} ${joined ? "joined" : "left"}`, { sw: joined ? devColor(name) : "var(--faint)", kind: "presence" }); };
function mascot() {}
const PREFIX = "pooled-room-";   // PeerJS id prefix (was "swarmllm-room-" before the rename; PROTOCOL did not change)
const HOST_KEY = "pooled-host", OLD_HOST_KEY = "swarm-host";   // localStorage: what a host needs to resume its room after a reload (the old key is still read)
const rand = (n) => Array.from(crypto.getRandomValues(new Uint8Array(n)))
  .map(b => "ABCDEFGHJKMNPQRSTVWXYZ23456789"[b % 30]).join("");

let peer = null;          // my PeerJS peer
let isHost = false;
let roomCode = null;
let myName = null;
let myMeta = {};
// conns: peerId -> { conn, name, meta, rtt, mbps, card }
const conns = new Map();
// host only: roster of member peer ids -> {name, meta}
const roster = new Map();

// --- GPU capability probe (runs at page load so the join screen can offer
// contribution presets) ---
async function probeGPU() {
  const meta = { ua: deviceKind({ ua: navigator.userAgent, touchPoints: navigator.maxTouchPoints || 0, mobile: !!navigator.userAgentData?.mobile }),
                 webgpu: false, gpu: "no WebGPU", maxBufGB: 0 };
  if (navigator.gpu) {
    try {
      const a = await navigator.gpu.requestAdapter();
      if (a) {
        meta.webgpu = true;
        const info = a.info || {};
        meta.gpu = [...new Set([info.vendor, info.architecture || info.device].filter(Boolean))].join(" ") || "GPU";
        meta.maxBufGB = +(a.limits.maxBufferSize / 2 ** 30).toFixed(1);
        // browsers hide real GPU memory (fingerprinting). Default to the
        // conservative per-buffer limit; the user can opt in to a real
        // measurement (see measureBudgetGB) which replaces this estimate.
        meta.budgetGB = meta.maxBufGB;
        meta.canMeasure = meta.ua !== "iPhone" && meta.ua !== "Android";
      }
    } catch {}
  }
  return meta;
}

async function measureBudgetGB(adapter, capGB) {
  try {
    const dev = await adapter.requestDevice();
    let lost = false;
    dev.lost.then(() => { lost = true; });
    const chunk = 512 * 2 ** 20;
    const bufs = [];
    let total = 0;
    while (total < capGB * 2 ** 30 && !lost) {
      dev.pushErrorScope("out-of-memory");
      const b = dev.createBuffer({ size: chunk, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      try { // commit the pages for real, or lazy allocation lies to us
        const enc = dev.createCommandEncoder();
        enc.clearBuffer(b);
        dev.queue.submit([enc.finish()]);
        await dev.queue.onSubmittedWorkDone();
      } catch { lost = true; }
      const err = await dev.popErrorScope().catch(() => true);
      if (err || lost) { try { b.destroy(); } catch {} break; }
      bufs.push(b);
      total += chunk;
    }
    for (const b of bufs) { try { b.destroy(); } catch {} }
    try { dev.destroy(); } catch {}
    return +(total / 2 ** 30).toFixed(1);
  } catch { return 0; }
}

// the join screen says up front whether this browser can hold layers, and what to do if not
preflight().then((v) => {
  if (v.ok) return;
  $("join-gpu-t").textContent = v.line;
  $("join-gpu-d").textContent = v.detail || "";
  $("join-gpu").querySelector("details").hidden = !v.detail;
  $("join-gpu").hidden = false;
  $("join-pledge").classList.add("no-gpu");
  $("ap-no").textContent = v.line;
});
// probe once at load; fill the contribution selector
const metaPromise = (async () => {
  const m = await probeGPU();
  if (m.webgpu && m.budgetGB) m.contribGB = Math.max(0.2, Math.round(m.budgetGB * 0.5 * 10) / 10);
  m.phone = m.ua === "iPhone" || m.ua === "Android";
  if (m.phone) { m.contribGB = 0.5; $("join-gb").min = "0.5"; $("join-gb").step = "0.5"; }
  else if (m.contribGB) m.contribGB = Math.max(1, Math.round(m.contribGB));
  if (m.contribGB) $("join-gb").value = m.contribGB;
  return m;
})();

// --- UI helpers ---
function log(from, text) {
  const div = document.createElement("div");
  div.innerHTML = `<b></b> `;
  div.querySelector("b").textContent = from;
  div.appendChild(document.createTextNode(text));
  $("chat-log").appendChild(div);
  $("chat-log").scrollTop = $("chat-log").scrollHeight;
}

// A device card: name, what kind of device, the memory it gives the room, its status. This device's
// card has a quiet -/+ on its GB. The link numbers (rtt, bandwidth, GPU) and the bandwidth test show with ?dev=1.
const lends = (gb) => gb + " GB";
const ICONS = {
  laptop: '<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path d="M3 3.5h10v7H3zM1.2 12.5h13.6" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/></svg>',
  desk: '<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path d="M1.8 2.8h12.4v8.4H1.8zM8 11.2v2.6M5.2 13.8h5.6" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/></svg>',
  phone: '<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><rect x="4.5" y="1.5" width="7" height="13" rx="1.6" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M7 12.2h2" stroke="currentColor" stroke-width="1.3"/></svg>',
};
const iconFor = (meta) => meta.phone || /iPhone|Android$/.test(meta.ua || "") ? ICONS.phone : /Mac|iPad/.test(meta.ua || "") ? ICONS.laptop : ICONS.desk;
// "0–19" (what the deal sends) -> "1–20", the way people count layers
const humanRange = (r) => { const m = /^(\d+)\D+(\d+)$/.exec(String(r || "")); return m ? `${+m[1] + 1}\u2013${+m[2] + 1}` : String(r || ""); };
// One colour per device, everywhere (chips, pool bar, loading rows, band, Lend screen): given once,
// in join order, to each device that can hold layers. A device that only asks is grey everywhere.
const SWATCH = ["#2A45E0", "#2B2F3C", "#7C8FFF", "#5E616B", "#B9C6FF", "#1C33B8",
  // devices 7 to 16: more of the same family (blues, indigo, slate), each distinct from its neighbours
  "#4F6BFF", "#3E4454", "#9AABFF", "#7B7F8A", "#2F3FA8", "#D3DBFF", "#454D8F", "#9DA1AB", "#6E86FF", "#1A1D26"];
// past 16 devices: shades generated in the same blue-to-slate range (hue 222-232), so no two neighbours match
function swatch(i) {
  if (i < SWATCH.length) return SWATCH[i];
  const k = i - SWATCH.length, hue = 222 + (k * 7) % 11, sat = k % 3 === 2 ? 12 : 55 + (k * 13) % 30, light = 28 + (k * 17) % 50;
  return `hsl(${hue} ${sat}% ${light}%)`;
}
const devSlots = new Map();   // name -> slot: the host's roster order on every device (see the roster message)
function metaOf(name) {
  if (name === myName) return myMeta;
  for (const c of conns.values()) if (c.name === name) return c.meta || {};
  for (const m of members.values()) if (m.name === name) return m.meta || {};
  return null;
}
// the room's one order (the host first, then join order: the same slots the colours use), so every screen lists
// the devices the same way instead of "me first"
const bySlot = (a, b) => (devSlots.has(a) ? devSlots.get(a) : 1e6) - (devSlots.has(b) ? devSlots.get(b) : 1e6);
function orderCards() {
  const box = $("peers"); if (!box) return;
  const cards = [...box.querySelectorAll(":scope > .peer-card")];
  const sorted = [...cards].sort((x, y) => bySlot(x.dataset.name, y.dataset.name));
  if (sorted.some((c, i) => c !== cards[i])) for (const c of sorted) box.appendChild(c);
}
function devColor(name) {
  if (name == null) return "var(--faint)";
  const meta = metaOf(name);
  if (meta && meta.webgpu === false) return "var(--ink-4)";
  if (!devSlots.has(name)) devSlots.set(name, devSlots.size);
  return swatch(devSlots.get(name));
}
// text on a device's colour: ink on the light blues and greys, white on the rest
// text on a light colour is ink, on a dark one white (by perceived lightness, for the generated shades too)
const onSwatch = (c) => {
  if (/faint/.test(c)) return "var(--ink)";
  let l = 0;
  const hex = /^#([0-9a-f]{6})$/i.exec(c), hsl = /^hsl\(\S+ \S+% (\d+)%\)$/.exec(c);
  if (hex) { const n = parseInt(hex[1], 16); l = (0.299 * (n >> 16) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255)) / 255 * 100; }
  else if (hsl) l = +hsl[1];
  return l > 58 ? "var(--ink)" : "#fff";
};
function peerCard(id, name, meta, self) {
  // a chip in the room bar (the landing's: dot, icon, name, GB); a click opens the device's card
  const card = document.createElement("div");
  card.className = "peer-card" + (self ? " self" : "");
  card.setAttribute("role", "listitem");
  card.dataset.name = name;
  card.innerHTML = `
    <button class="pchip" type="button" aria-expanded="false"><span class="dot ${self || meta.webgpu ? "ok" : "warn"}"></span><span class="pic">${iconFor(meta)}</span><span class="pname"></span><span class="cg"></span><span class="cst"></span></button>
    <div class="pop" hidden>
      <div class="pop-h"><span class="pic2">${iconFor(meta)}</span><span class="pn"></span><span class="pst"></span></div>
      <div class="peer-sub"><span class="pkind"></span><span aria-hidden="true">\u00b7</span><span class="buf">-</span><span class="play"></span></div>
      <div class="peer-gpu dev-only"></div>
      <div class="peer-stats dev-only">
        <span>rtt <b class="rtt">-</b></span>
        <span>bw <b class="bw">-</b></span>
      </div>
      ${self ? "" : '<button class="bw-btn dev-only" type="button">test bandwidth</button>'}
    </div>`;
  paintCard(card, name, meta, self);
  $("peers").appendChild(card);
  if (devSlots.size) orderCards();
  card.querySelector(".pchip").addEventListener("click", (e) => { e.stopPropagation(); chipPop(card); });
  if (!self) card.querySelector(".bw-btn").addEventListener("click", () => bwTest(id));
  return card;
}
// one device card open at a time, placed under its chip (fixed, so the scrolling chip row never clips it)
function chipPop(card) {
  for (const c of document.querySelectorAll("#peers .peer-card")) {
    const open = c === card && c.querySelector(".pop").hidden;
    c.querySelector(".pop").hidden = !open;
    c.querySelector(".pchip").setAttribute("aria-expanded", String(open));
    if (open) {
      const r = c.querySelector(".pchip").getBoundingClientRect(), pop = c.querySelector(".pop");
      const w = Math.min(272, innerWidth - 24);
      pop.style.width = w + "px";
      pop.style.left = Math.max(12, Math.min(r.left, innerWidth - w - 12)) + "px";
      pop.style.top = r.bottom + 8 + "px";
    }
  }
}
document.addEventListener("click", (e) => { if (!e.target.closest?.(".pop")) chipPop(null); if (!e.target.closest?.("#room-menu") || e.target === $("room-menu")) $("room-menu").open = false; });   // (a click on the phone sheet's backdrop lands on the details itself)
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  chipPop(null); $("room-menu").open = false;
  if (!$("share").hidden) closeShare();
  if (!$("card").hidden) $("card").hidden = true;
});
document.addEventListener("pointerdown", (e) => {
  const t = e.target.closest?.("[data-tip]"); if (!t) return;
  t.classList.add("tip-off");
  t.addEventListener("pointerleave", () => t.classList.remove("tip-off"), { once: true });
});
addEventListener("resize", () => chipPop(null));
$("peers").addEventListener("scroll", () => chipPop(null), { passive: true });
// what a card says about its device (the sim hook repaints with made-up devices)
function paintCard(card, name, meta, self) {
  card.querySelector(".pname").textContent = name;
  card.querySelector(".pn").textContent = name;
  if (self) card.querySelector(".pn").insertAdjacentHTML("beforeend", " <small>(you)</small>");
  card.querySelector(".pic").innerHTML = card.querySelector(".pic2").innerHTML = iconFor(meta);
  card.querySelector(".dot").className = "dot " + (self || meta.webgpu ? "ok" : "warn");
  card.querySelector(".pkind").textContent = !meta.ua || meta.ua === "Device" ? "Computer" : meta.ua;
  // the GPU name is missing when the browser hides adapter info (and on sim devices): show only what we know
  const kind = !meta.ua || meta.ua === "Device" ? "Computer" : meta.ua;
  card.querySelector(".peer-gpu").textContent = meta.webgpu === false ? `${kind} · no WebGPU` : meta.gpu ? `${kind} · ${meta.gpu}` : kind;
  const budget = meta.budgetGB || meta.maxBufGB;
  setBuf(card, meta.webgpu === false ? "no WebGPU" : meta.contribGB ? lends(meta.contribGB) : (budget ? budget + " GB" : "-"));
  card.querySelector(".cg").textContent = meta.webgpu === false ? "chat only" : meta.contribGB ? meta.contribGB + " GB" : "";
  card.querySelector(".pchip").title = meta.webgpu === false ? `${name}: this device can ask but can't hold model layers` : `${name}: ${card.querySelector(".buf").textContent.replace(/[\u2212+]/g, "").trim()}`;
  card.style.setProperty("--sw", devColor(name));
  peerStatus(card, meta.webgpu === false ? "chat only" : self ? "this device" : "connected");
}
// the status word on a device card: connected, loading N%, ready. While it loads, its chip shows the %
function peerStatus(card, text, ok = false) {
  const el = card?.querySelector(".pst"); if (!el) return;
  el.textContent = text; el.classList.toggle("ok", ok);
  const pct = /(\d+)%$/.exec(text);
  card.classList.toggle("loading", !!pct && +pct[1] < 100);
  card.querySelector(".cst").textContent = pct ? pct[1] + "%" : "";
}
function setLends(card, gb) { setBuf(card, lends(gb)); card.querySelector(".cg").textContent = gb + " GB"; }
// the GB on a card (this device's card keeps its -/+ around the number)
function setBuf(card, text) { const b = card.querySelector(".buf"); (b.querySelector(".bv") || b).textContent = text; }

let wasReady = false;
// The model ladder: every model with what this room still needs for it, smallest first. Until
// someone picks a model by hand, the select follows the largest model the room can run.
let modelTouched = false;
const shortName = (key) => (MODELS[key]?.label || key).split("\u00b7")[0].trim();
// the picker offers three models; ?dev=1 (and the local test rooms) offer every one in room/models.js
const DEV = document.documentElement.classList.contains("dev");
const PICK_NEED = DEV ? NEED_GB : Object.fromEntries(PICKER.map((k) => [k, NEED_GB[k]]));
function addModelOption(key) {
  const sel = $("ai-model");
  if (!MODELS[key] || [...sel.options].some((o) => o.value === key)) return;
  sel.add(new Option(shortName(key), key));
}
if (DEV) Object.keys(MODELS).forEach(addModelOption);
// set the picker to a model, adding it when another device started one the picker does not list
function setModelValue(key) { if (!MODELS[key]) return; addModelOption(key); $("ai-model").value = key; }
function renderLadder(pledged) {
  const el = $("ai-ladder"); if (!el) return;
  const none = !(pledged > 0);
  el.innerHTML = (none ? '<p class="ai-nogpu">Needs a device with WebGPU</p>' : "") + ladder(PICK_NEED, pledged).map((x) => {
    const gb = `<span class="nd">${NEED_GB[x.key] ?? ""} GB</span>`;
    const fig = x.ok ? `${gb}<b>fits</b>` : none ? gb : `<span class="more">needs ${x.short} GB more</span>`;
    return `<button type="button" class="rung${x.ok ? " ok" : " short"}${x.key === $("ai-model").value ? " sel" : ""}" data-k="${x.key}" aria-pressed="${x.key === $("ai-model").value}"${x.ok ? "" : ' title="Invite a device to fit this"'}><span class="rn">${esc(shortName(x.key))}</span><span class="fig">${fig}</span></button>`;
  }).join("");
}
$("ai-ladder").addEventListener("click", (e) => {
  const b = e.target.closest(".rung"); if (!b || $("ai-model").disabled) return;
  setModelValue(b.dataset.k); modelTouched = true; updateCluster();
});
function updateNeed(pledged) {
  if (!modelTouched && !ai.engine && !ai.busy && !$("ai-model").disabled) $("ai-model").value = bestFit(PICK_NEED, pledged);
  renderLadder(pledged);
  const need = NEED_GB[$("ai-model").value] || 1;
  const ok = pledged >= need;
  $("need-fill").style.width = Math.min(100, pledged / need * 100).toFixed(1) + "%";
  const has = +pledged.toFixed(1);
  $("need-text").textContent = ok
    ? `Needs ${need} GB. The room has ${has} GB.`
    : `Needs ${need} GB. The room has ${has} GB, ${(need - pledged).toFixed(1)} GB short.`;
  $("ai-need").classList.toggle("ok", ok);
  if (!ai.busy && !ai.engine) $("ai-start").disabled = !ok;
  if (ok && !wasReady) { $("ai-start").classList.remove("unlocked"); void $("ai-start").offsetWidth; $("ai-start").classList.add("unlocked"); }
  wasReady = ok;
}
$("ai-model").addEventListener("change", () => { modelTouched = true; updateCluster(); });
function updateCluster() {
  const all = [myMeta, ...[...members.values()].map(m => m.meta)];
  const gpus = all.filter(m => m && m.webgpu).length;
  // only devices with WebGPU hold layers; the others join as ask-only guests
  const pledged = all.reduce((s, m) => s + (m?.webgpu ? m?.contribGB || 0 : 0), 0);
  updateNeed(pledged);
  const mem = all.reduce((s, m) => s + (m?.budgetGB || m?.maxBufGB || 0), 0);
  $("cluster-summary").textContent = DEV
    ? `${all.length} device${all.length > 1 ? "s" : ""} \u00b7 ${gpus} WebGPU \u00b7 ${pledged.toFixed(1)} GB pledged`
    : `${all.length} device${all.length > 1 ? "s" : ""} \u00b7 ${+pledged.toFixed(1)} GB pooled`;
  $("hdr-sum").innerHTML = `<b>${+pledged.toFixed(1)} GB</b> pooled`;
  $("peers-n").textContent = String(all.length);
  renderPool(pledged);
}
// The model card's side: what the room pools (one segment per device, in its colour, with a tick at
// each model's need), what this device lends (+/-), and the invite (QR, code, copy link).
function renderPool(pledged) {
  const devs = [{ name: myName, meta: myMeta }, ...[...members.values()].map((m) => ({ name: m.name || "device", meta: m.meta || {} }))].sort((x, y) => bySlot(x.name, y.name))
    .filter((d) => d.meta?.webgpu && d.meta?.contribGB);
  const needs = Object.entries(PICK_NEED).sort((a, b) => a[1] - b[1]);
  const top = Math.max(pledged, ...needs.map((x) => x[1])) * 1.06 || 1;
  $("ap-total").textContent = `${+pledged.toFixed(1)} GB`;
  const meter = $("ap-meter");
  meter.querySelector(".apm-fill").innerHTML = devs.map((d) => `<i style="--sw:${devColor(d.name)};width:${(d.meta.contribGB / top * 100).toFixed(2)}%" title="${esc(String(d.name))}: ${d.meta.contribGB} GB"></i>`).join("");
  // a tick where each model starts to fit; the picked one says what it needs
  const sel = $("ai-model").value;
  meter.querySelector(".apm-ticks").innerHTML = needs.map(([k, gb]) => `<span class="${pledged >= gb ? "ok" : ""}${k === sel ? " sel" : ""}${gb / top > 0.6 ? " r" : gb / top < 0.3 ? " l" : ""}" style="left:${(gb / top * 100).toFixed(2)}%" title="${esc(shortName(k))} needs ${gb} GB"></span>`).join("");
  // what the selected model needs, as its own line under the meter (not a label hanging off its tick)
  const selNeed = needs.find(([k]) => k === sel);
  $("ap-need").hidden = !selNeed;
  if (selNeed) { const [k, gb] = selNeed, short = gb - pledged; $("ap-need").innerHTML = `${esc(shortName(k))} needs <b>${gb} GB</b><span>${short > 0 ? `${+short.toFixed(1)} GB short` : "fits"}</span>`; $("ap-need").classList.toggle("ok", short <= 0); }
  $("ap-devs").innerHTML = devs.map((d) => `<li style="--sw:${devColor(d.name)}"><i></i><span>${esc(String(d.name))}${d.name === myName ? " <small>(this device)</small>" : ""}</span><b>${d.meta.contribGB} GB</b></li>`).join("")
    || '<li class="none">No device with WebGPU yet</li>';
  const can = !!myMeta.webgpu;
  $("ap-step").hidden = !can; $("ap-no").hidden = can;
  if (can) { if (document.activeElement !== $("ap-gb")) $("ap-gb").value = myMeta.contribGB; $("ap-minus").disabled = myMeta.contribGB <= lendMin(); $("ap-plus").disabled = myMeta.contribGB >= 64; }
}
const lendMin = () => (myMeta.phone ? 0.5 : 1);
function selfSteps(card) { const s = card.querySelectorAll(".gbstep .step"); if (s.length) { s[0].disabled = myMeta.contribGB <= lendMin(); s[1].disabled = myMeta.contribGB >= 64; } }
// lend a different amount: this device's card, the room's total, and every other device hear it
function lendGB(v) {
  v = Math.round(Math.min(64, Math.max(lendMin(), v)) * 10) / 10;
  if (!myMeta.webgpu || v === myMeta.contribGB) return;
  myMeta.contribGB = v;
  const selfCard = document.querySelector(".peer-card.self");
  if (selfCard) { setLends(selfCard, v); selfSteps(selfCard); }
  updateCluster(); broadcastAll({ t: "pledge", gb: v });
}
$("ap-minus").addEventListener("click", () => lendGB(myMeta.contribGB - (myMeta.phone ? 0.5 : 1)));
$("ap-plus").addEventListener("click", () => lendGB(myMeta.contribGB + (myMeta.phone ? 0.5 : 1)));
// or type the amount: applied on Enter or on leaving the box, clamped like the steps (and shown back as applied)
function typedGB() {
  const el = $("ap-gb"), v = parseFloat(String(el.value).replace(",", "."));
  if (Number.isFinite(v)) lendGB(v);
  el.value = myMeta.contribGB;
}
$("ap-gb").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); typedGB(); e.target.blur(); } else if (e.key === "Escape") { e.target.value = myMeta.contribGB; e.target.blur(); } });
$("ap-gb").addEventListener("blur", typedGB);
$("ap-gb").addEventListener("focus", (e) => e.target.select());
$("ap-copy").addEventListener("click", copyRoomLink);

function enterRoom() {
  $("join-screen").style.display = "none";
  $("room-screen").style.display = "flex";
  $("room-badge").style.display = "";
  document.body.classList.add("in-room");
  roomSince = performance.now();
  $("compute-open").hidden = false;
  $("room-badge").textContent = roomCode;
  $("side-code").textContent = roomCode;
  $("side-code").addEventListener("click", openShare);
  $("ap-qr").innerHTML = qrSVG(roomLink(), { size: 112 });
  // Chat | Code shows once a model is ready (?mock=code: at once, there is no model); peers see Code once the host starts a session
  if (isHost) { $("host-controls").hidden = false; if (MOCK) $("mode-bar").hidden = false; }
  peerCard("self", myName, myMeta, true);
  updateCluster();
  log("room", `${roomCode}: type this code on your other devices`);
  $("ai-panel").style.display = "flex";
  aiStatus("");
  emptyText("Pick a model and press Start. Anyone in the room can.");
  selfStepper();
}
// a quiet -/+ around this device's GB on its card, the same steps as the model card's
function selfStepper() {
  const selfCard = document.querySelector(".peer-card.self");
  if (!selfCard || !myMeta.webgpu) return;
  const buf = selfCard.querySelector(".buf");
  if (!buf.classList.contains("gbstep")) {
    const text = buf.textContent;
    buf.classList.add("gbstep");
    buf.innerHTML = '<button class="step" type="button" data-d="-1" aria-label="Less memory">\u2212</button><span class="bv"></span><button class="step" type="button" data-d="1" aria-label="More memory">+</button>';
    buf.querySelector(".bv").textContent = text;
    buf.addEventListener("click", (e) => { const b = e.target.closest(".step"); if (b) lendGB(myMeta.contribGB + +b.dataset.d * (myMeta.phone ? 0.5 : 1)); });
  }
  selfSteps(selfCard);
}

// --- connection wiring ---
function wire(conn, name, meta, initiator = false) {
  const entry = { conn, name: name || conn.peer, meta: meta || {}, rtt: null, card: null, link: makeLink(), stripes: [] };
  conns.set(conn.peer, entry);
  if (WIRE_STRIPES > 0) {
    attachWire(entry.link, conn, (m) => onData(conn.peer, m));
    // extra associations for striping: the side that dialed opens them, the other side accepts
    // them in peer.on("connection") by label and attaches its end of the wire channel
    if (initiator) for (let i = 1; i < WIRE_STRIPES; i++) {
      const sc = peer.connect(conn.peer, { reliable: true, label: "stripe" });
      sc.on("open", () => { attachWire(entry.link, sc, (m) => onData(conn.peer, m)); });
      sc.on("error", () => {});
      entry.stripes.push(sc);
    }
  }

  conn.on("data", (d) => onData(conn.peer, d));
  conn.on("close", () => {
    const e = conns.get(conn.peer);
    if (e && e.conn !== conn) return;   // an older link to the same device
    conns.delete(conn.peer);
    if (isHost) {   // on the host a closed link means the device left; workers wait for the roster
      dropCard(conn.peer); members.delete(conn.peer); roster.delete(conn.peer); broadcastRoster();
      log("room", `${e?.name || conn.peer} left`);
      aiPeerLeft(conn.peer, e?.name);
    } else if (conn.peer === PREFIX + roomCode) { log("room", "lost the link to the host"); hostGone(); }
    updateCluster();
  });
  conn.on("error", () => {});
  return entry;
}

function ensureCard(id, name, meta) {
  let card = cards.get(id);
  if (!card) {
    card = peerCard(id, name || id, meta || {}, false);
    cards.set(id, card);
    updateCluster();
    log("room", `${name || id} joined`);
    presence(name || id, true);
    if ($("ai-output").style.display === "block") sysNote(`${name || id} joined${meta?.contribGB && meta?.webgpu ? ` with ${meta.contribGB} GB` : ""}`, "join");
    mascot(`${name || id} joined! ${members.size + 1} devices in the room.`);
  }
  const e = conns.get(id);
  if (e) e.card = card;
  return card;
}
function dropCard(id) { const c = cards.get(id); if (c) { c.remove(); cards.delete(id); presence(c.dataset.name || id, false); } }
// open a data link to a chain neighbour if we do not have one yet; resolves when it is up
function ensureLink(id, timeoutMs = 60000) {
  if (!id || id === "host" || conns.has(id)) return Promise.resolve(true);
  if (!ensureLink.pending.has(id)) { ensureLink.pending.add(id); meshConnect(id); }
  return new Promise((res) => {
    const t0 = performance.now();
    const t = setInterval(() => {
      if (conns.has(id)) { clearInterval(t); ensureLink.pending.delete(id); res(true); }
      else if (performance.now() - t0 > timeoutMs) { clearInterval(t); ensureLink.pending.delete(id); res(false); }
    }, 100);
  });
}
ensureLink.pending = new Set();

function sendTo(id, obj) { conns.get(id)?.conn.send(obj); }
// debug: per-peer wire state (channels open, frames sent/received) — `pooledDebug()` in the console (`swarmDebug()` still works)
window.pooledDebug = window.swarmDebug = () => [...conns].map(([id, e]) => ({ id, name: e.name, chans: e.link?.chans.filter((c) => c.readyState === "open").length ?? 0, sent: e.link?.sent ?? 0, recv: e.link?.recv ?? 0 }));
// activations go over the sliced wire channel when it is up, else as a normal message
// ?netlag=ms delays every activation frame this device sends, to emulate a slow link in tests
// (equal delays keep send order)
const NETLAG = Math.max(0, parseInt(new URLSearchParams(location.search).get("netlag"), 10) || 0);
function sendHidden(id, msg) {
  if (NETLAG) { setTimeout(() => sendHiddenNow(id, msg), NETLAG); return; }
  sendHiddenNow(id, msg);
}
function sendHiddenNow(id, msg) {
  const e = conns.get(id);
  if (e?.link && wireReady(e.link) && sendFrame(e.link, msg)) return;
  sendTo(id, msg);
}
function broadcastAll(obj) { for (const [id] of conns) sendTo(id, obj); }

// bandwidth test state
const bwRecv = new Map(); // fromId -> {bytes, t0}

function onData(from, d) {
  // binary chunk = bandwidth test payload
  if (d instanceof ArrayBuffer || ArrayBuffer.isView(d)) {
    const st = bwRecv.get(from);
    if (st) st.bytes += d.byteLength || d.length;
    return;
  }
  const e = conns.get(from);
  if (!d || typeof d.t !== "string") return;
  if (d.t.startsWith("ai-")) { aiOnData(from, d); return; }
  switch (d.t) {
    case "hello":
      // one protocol per room: a tab from an older or newer deploy is told to reload
      if (d.v !== PROTOCOL) {
        sendTo(from, { t: "bye", reason: `this room runs Pooled protocol ${PROTOCOL} and your tab runs ${d.v ?? 1}: reload both pages so they match` });
        log("room", `${d.name || from} runs a different Pooled version (protocol ${d.v ?? 1}); asked it to reload`);
        break;
      }
      // a peer picks its own name: keep it a short plain string (it is also escaped wherever it is shown)
      d.name = String(d.name ?? from).replace(/[\u0000-\u001f\u007f<>"'`&]/g, "").trim().slice(0, 40) || String(from).slice(0, 8);
      e.name = d.name; e.meta = d.meta;
      members.set(from, { name: d.name, meta: d.meta });
      ensureCard(from, d.name, d.meta);
      if (isHost) {
        roster.set(from, { name: d.name, meta: d.meta }); broadcastRoster();
        aiRejoin(from, d.name);
        if (d.died?.during && d.died.ago > 2) log("room", `${d.name} came back: its tab was killed ${d.died.ago} s ago while ${d.died.during}. Phones kill background tabs; keep the screen on.`);
        if (ai.visibility !== "all") sendTo(from, { t: "ai-visibility", mode: ai.visibility });
        if (!d.back) aiWelcome(from); else offerRedealForNewcomers();
        codeWelcome(from);
      }
      break;
    case "leaving":   // the tab is closing: treat the link as gone now instead of waiting for ICE to time out
      conns.get(from)?.conn.close();
      break;
    case "bye":
      toast(d.reason);
      log("room", d.reason);
      if (from === PREFIX + roomCode) { $("room-over").hidden = false; $("room-over-why").textContent = d.reason; }
      break;
    case "roster": {
      // the host's view of the room: draw a card per device, no mesh connections
      // colours follow the host's order (the host first, then join order), so a device has the
      // same colour on every screen (they used to go by first-seen order, which put "me" first)
      const order = d.members.map((m) => m.name);
      if (order.join("\n") !== [...devSlots.keys()].slice(0, order.length).join("\n")) {
        devSlots.clear(); order.forEach((n) => devSlots.set(n, devSlots.size));
        for (const c of document.querySelectorAll(".peer-card")) c.style.setProperty("--sw", devColor(c.dataset.name));
      }
      queueMicrotask(orderCards);   // after this message's cards exist
      const seen = new Set();
      for (const m of d.members) {
        if (m.id === peer.id) continue;
        seen.add(m.id);
        members.set(m.id, { name: m.name, meta: m.meta });
        const c = ensureCard(m.id, m.name, m.meta);
        if (m.meta?.contribGB) setLends(c, m.meta.contribGB);
        const ce = conns.get(m.id); if (ce) ce.meta = m.meta;
      }
      for (const id of [...members.keys()]) if (!seen.has(id)) { members.delete(id); dropCard(id); }
      updateCluster();
      break;
    }
    case "ping": sendTo(from, { t: "pong", ts: d.ts }); break;
    case "pong": {
      e.rtt = Math.round(performance.now() - d.ts);
      if (e.card) e.card.querySelector(".rtt").textContent = e.rtt + " ms";
      break;
    }
    case "pledge":
      if (e) { e.meta = { ...e.meta, contribGB: d.gb }; if (e.card) setLends(e.card, d.gb); }
      if (members.has(from)) members.get(from).meta = { ...members.get(from).meta, contribGB: d.gb };
      if (isHost && roster.has(from)) { roster.get(from).meta = { ...roster.get(from).meta, contribGB: d.gb }; broadcastRoster(); }
      updateCluster();
      break;
    case "bw-start": bwRecv.set(from, { bytes: 0, t0: performance.now() }); break;
    case "bw-end": {
      const st = bwRecv.get(from);
      if (st) {
        const secs = (performance.now() - st.t0) / 1000;
        const mbps = (st.bytes * 8 / 1e6 / secs).toFixed(0);
        sendTo(from, { t: "bw-result", mbps });
        bwRecv.delete(from);
      }
      break;
    }
    case "bw-result":
      if (e.card) e.card.querySelector(".bw").textContent = d.mbps + " Mbps";
      log("room", `bandwidth to ${e.name}: ${d.mbps} Mbps`);
      break;
  }
}

function broadcastRoster() {
  const members = [{ id: peer.id, name: myName, meta: myMeta },
    ...[...roster.entries()].map(([id, m]) => ({ id, ...m }))];
  broadcastAll({ t: "roster", members });
}

function meshConnect(targetId) {
  const conn = peer.connect(targetId, { reliable: true });
  conn.on("open", () => {
    wire(conn, undefined, undefined, true);
    conn.send({ t: "hello", name: myName, meta: myMeta, v: PROTOCOL });
  });
}

async function bwTest(id) {
  const e = conns.get(id);
  if (!e) return;
  log("room", `testing bandwidth to ${e.name}…`);
  sendTo(id, { t: "bw-start" });
  const chunk = new Uint8Array(64 * 1024);
  const total = 4 * 1024 * 1024;
  for (let sent = 0; sent < total; sent += chunk.length) {
    e.conn.send(chunk);
    // yield so the datachannel buffer can drain
    if (e.conn.dataChannel && e.conn.dataChannel.bufferedAmount > 1 << 20)
      await new Promise(r => setTimeout(r, 20));
  }
  sendTo(id, { t: "bw-end" });
}

// a closing tab says so, so the others fail fast (the data channel close can take ~30 s to surface)
window.addEventListener("pagehide", () => { try { broadcastAll({ t: "leaving" }); } catch {} });

// --- ping loop ---
setInterval(() => broadcastAll({ t: "ping", ts: performance.now() }), 2500);

const stepGB = (d) => { const i = $("join-gb"); const lo = parseFloat(i.min) || 1; const st = parseFloat(i.step) || 1; i.value = Math.min(64, Math.max(lo, (parseFloat(i.value) || lo) + d * st)); };
$("gb-minus").addEventListener("click", () => stepGB(-1));
$("gb-plus").addEventListener("click", () => stepGB(1));
// a typed amount is clamped like the steps once the box is left (100 becomes 64, -5 the minimum)
$("join-gb").addEventListener("change", () => stepGB(0));
// a friendly name for this device ("otter"): one lowercase word, filled in on the join screen; any edit wins
const NAMES = ["otter", "falcon", "panda", "fox", "heron", "koala", "lynx", "robin", "badger", "dolphin", "owl", "tiger", "wombat", "sparrow",
  "moose", "gecko", "puffin", "beaver", "marten", "crane", "finch", "orca", "bison", "lemur", "raven", "tapir", "walrus", "yak", "zebra",
  "ibis", "kestrel", "magpie", "narwhal", "ocelot", "pelican", "quokka", "seal", "stoat", "toucan", "vole", "wren", "hare", "egret",
  "jackal", "kiwi", "llama", "mole", "newt", "okapi", "plover", "swift", "tern", "urchin", "viper", "weasel", "ferret", "gibbon", "hyena",
  "iguana", "jay", "koi", "loris", "mink", "numbat", "osprey", "pika", "quail", "rook", "shrew", "trout", "alpaca", "bobcat",
  "cougar", "dingo", "eland", "gazelle", "hornbill", "impala", "kudu", "lark", "manatee", "nightjar", "oriole", "panther", "sloth", "tamarin"];
const pick = (a) => a[crypto.getRandomValues(new Uint32Array(1))[0] % a.length];
function friendlyName() {
  const now = $("name-input").value;
  let n = now;
  for (let i = 0; i < 8 && n === now; i++) n = pick(NAMES);
  return n;
}
$("name-input").value = friendlyName();
$("name-shuffle").addEventListener("click", (e) => { e.preventDefault(); $("name-input").value = friendlyName(); });
// joining or opening a room: the logo's wave where the panel was, until the room shows (or it fails)
function joinWait(on, text = "") {
  $("join-screen").classList.toggle("waiting", !!on);
  $("join-wait").hidden = !on;
  if (text) {
    const m = /^(.*room )([A-Z0-9]{4,6})(.*)$/.exec(text), el = $("jw-t");
    if (m) { const b = document.createElement("b"); b.textContent = m[2]; el.replaceChildren(m[1], b, m[3]); } else el.textContent = text;
  }
}
function joinFailed(text) {
  joinWait(false);
  $("join-status").textContent = text;
  $("create-btn").disabled = $("join-btn").disabled = false;
}
// --- join / create ---
async function start(create, resume = null) {
  myName = resume?.name || $("name-input").value.trim() || (create ? "host" : "peer") + "-" + rand(2);
  const code = resume?.code || (create ? rand(4) : $("code-input").value.trim().toUpperCase());
  if (!code) { $("join-status").textContent = "Enter a room code"; return; }
  $("create-btn").disabled = $("join-btn").disabled = true;
  joinWait(true, create ? (resume ? `Opening room ${code} again` : "Opening your room") : `Joining room ${code}`);
  $("join-status").textContent = "Connecting…";
  myMeta = await metaPromise;
  const gbIn = parseFloat($("join-gb").value);
  myMeta.contribGB = Math.min(64, Math.max(myMeta.phone ? 0.5 : 1, gbIn > 0 ? gbIn : (myMeta.contribGB || 1)));

  // STUN for hole-punching; TURN as fallback for symmetric NAT / CGNAT peers.
  // ICE prefers direct candidates, so TURN only carries traffic when a direct
  // path is impossible.
  const ICE = {
    iceServers: [
      { urls: ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"] },
      // TURN fallback for symmetric-NAT peers goes here (needs credentials —
      // see TURN_CREDS below); without it, strict-NAT peers can't join.
      ...(window.TURN_SERVERS || []),
    ],
  };
  // host claims the well-known id for the code; joiners get random ids
  peer = new Peer(create ? PREFIX + code : undefined, { debug: 1, config: ICE, ...SIGNAL_OPTS });

  peer.on("open", () => {
    isHost = create;
    roomCode = code;
    if (create) { enterRoom(); if (resume) resumeHost(resume); return; }
    // joiner: connect to host
    $("join-status").textContent = "Reaching the other devices…";
    const conn = peer.connect(PREFIX + code, { reliable: true });
    const timeout = setTimeout(() => {
      const ice = conn.peerConnection?.iceConnectionState;
      joinFailed(ice === "checking" || ice === "failed" || ice === "disconnected"
        ? "found the room, but the direct connection failed (strict NAT/firewall on one side). Trying a relay: give it ~20 s, or try another network"
        : "no room with that code (is the host page open?)");
    }, 15000);
    conn.on("open", () => {
      clearTimeout(timeout);
      wire(conn, "host", undefined, true);
      let died = null;
      if (!VQ.get("embed")) try { const c = JSON.parse(localStorage.getItem("pooled-crumb") || "null"); if (c && Date.now() - c.t < 10 * 60 * 1000) died = { during: c.s, ago: Math.round((Date.now() - c.t) / 1000) }; } catch {}
      conn.send({ t: "hello", name: myName, meta: myMeta, died, v: PROTOCOL });
      enterRoom();
    });
  });

  peer.on("connection", (conn) => {
    conn.on("open", () => {
      if (conn.label === "stripe") {   // extra association for the hidden-state wire, not a new peer
        const e = conns.get(conn.peer);
        if (e) { attachWire(e.link, conn, (m) => onData(conn.peer, m)); e.stripes.push(conn); }
        return;
      }
      wire(conn);
      conn.send({ t: "hello", name: myName, meta: myMeta, v: PROTOCOL });
    });
  });

  peer.on("error", (err) => {
    // resuming: the old tab's id is still registered until the signaling server notices it left
    if (resume && err.type === "unavailable-id" && (resume.tries = (resume.tries || 0) + 1) < 30) {
      $("join-status").textContent = `waiting for room ${code} to be free again (the old tab is still registered)…`;
      try { peer.destroy(); } catch {}
      peer = null;
      setTimeout(() => start(true, resume), 3000);
      return;
    }
    if ($("room-screen").style.display === "flex") { $("join-status").textContent = "error: " + err.type; return; }   // in the room already: not a join failure
    joinFailed(err.type === "unavailable-id" ? "that code is already hosting a room: press Join instead"
      : err.type === "peer-unavailable" ? "no room with that code"
      : "error: " + err.type);
  });
}

let wakeLock = null, awakeVideo = null;
function awakeStatus(s) { const el = $("awake"); if (el && myMeta?.phone) el.textContent = s; }
async function keepAwake() {
  // 1. the real API (iOS 16.4+, must be called from a tap)
  try {
    if (!wakeLock && navigator.wakeLock) {
      wakeLock = await navigator.wakeLock.request("screen");
      wakeLock.addEventListener("release", () => { wakeLock = null; awakeStatus("screen lock: released"); });
      awakeStatus("screen stays awake \u2713");
    }
  } catch (e) { awakeStatus("wake lock failed: " + (e?.message || e)); }
  // 2. belt and braces: a silent looping video keeps iOS from locking the screen
  try {
    if (!awakeVideo) {
      awakeVideo = document.createElement("video");
      awakeVideo.setAttribute("playsinline", ""); awakeVideo.muted = true; awakeVideo.loop = true;
      awakeVideo.style.cssText = "position:fixed;width:1px;height:1px;opacity:0.01;pointer-events:none;bottom:0;left:0";
      awakeVideo.src = "data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAbBbW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD6AAAB9AAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAy50cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAB9AAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAEAAAABAAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAAfQAAAAAAABAAAAAAKmbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAAAoAAAAUABVxAAAAAAALWhkbHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAACUW1pbmYAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAhFzdGJsAAAAuXN0c2QAAAAAAAAAAQAAAKlhdmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAEAAQABIAAAASAAAAAAAAAABFUxhdmM2MC4zMS4xMDIgbGlieDI2NAAAAAAAAAAAAAAAGP//AAAAL2F2Y0MBQsAe/+EAF2dCwB7ZBCbARAAAAwAEAAADAFA8WLkgAQAFaMuDyyAAAAAQcGFzcAAAAAEAAAABAAAAFGJ0cnQAAAAAAAANNAAADTQAAAAYc3R0cwAAAAAAAAABAAAAFAAABAAAAAAUc3RzcwAAAAAAAAABAAAAAQAAAHBzdHNjAAAAAAAAAAgAAAABAAAAAQAAAAEAAAAFAAAAAgAAAAEAAAAGAAAAAQAAAAEAAAAJAAAAAgAAAAEAAAAKAAAAAQAAAAEAAAAMAAAAAgAAAAEAAAANAAAAAQAAAAEAAAAQAAAAAgAAAAEAAABkc3RzegAAAAAAAAAAAAAAFAAAAo8AAAAKAAAACgAAAAoAAAAKAAAACgAAAAoAAAAKAAAACgAAAAoAAAAKAAAACgAAAAoAAAAKAAAACgAAAAoAAAAKAAAACgAAAAoAAAAKAAAAUHN0Y28AAAAAAAAAEAAABwYAAAmZAAAJpwAACbUAAAnDAAAJ2wAACekAAAn3AAAKBQAACh0AAAorAAAKOQAAClEAAApfAAAKbQAACnsAAAK9dHJhawAAAFx0a2hkAAAAAwAAAAAAAAAAAAAAAgAAAAAAAAfQAAAAAAAAAAAAAAABAQAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAJGVkdHMAAAAcZWxzdAAAAAAAAAABAAAH0AAABAAAAQAAAAACNW1kaWEAAAAgbWRoZAAAAAAAAAAAAAAAAAAAH0AAAEKAVcQAAAAAAC1oZGxyAAAAAAAAAABzb3VuAAAAAAAAAAAAAAAAU291bmRIYW5kbGVyAAAAAeBtaW5mAAAAEHNtaGQAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAaRzdGJsAAAAfnN0c2QAAAAAAAAAAQAAAG5tcDRhAAAAAAAAAAEAAAAAAAAAAAABABAAAAAAH0AAAAAAADZlc2RzAAAAAAOAgIAlAAIABICAgBdAFQAAAAAAH0AAAAE/BYCAgAUViFblAAaAgIABAgAAABRidHJ0AAAAAAAAH0AAAAE/AAAAIHN0dHMAAAAAAAAAAgAAABAAAAQAAAAAAQAAAoAAAAAcc3RzYwAAAAAAAAABAAAAAQAAAAEAAAABAAAAWHN0c3oAAAAAAAAAAAAAABEAAAAVAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAFRzdGNvAAAAAAAAABEAAAbxAAAJlQAACaMAAAmxAAAJvwAACdcAAAnlAAAJ8wAACgEAAAoZAAAKJwAACjUAAApNAAAKWwAACmkAAAp3AAAKjwAAABpzZ3BkAQAAAHJvbGwAAAACAAAAAf//AAAAHHNiZ3AAAAAAcm9sbAAAAAEAAAARAAAAAQAAAGJ1ZHRhAAAAWm1ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALWlsc3QAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABMYXZmNjAuMTYuMTAwAAAACGZyZWUAAAOqbWRhdN4CAExhdmM2MC4zMS4xMDIAAjBADgAAAnEGBf//bdxF6b3m2Ui3lizYINkj7u94MjY0IC0gY29yZSAxNjQgcjMxMDggMzFlMTlmOSAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjMgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0wIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDE6MHgxMTEgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTAgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz0yIGxvb2thaGVhZF90aHJlYWRzPTEgc2xpY2VkX3RocmVhZHM9MCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MCB3ZWlnaHRwPTAga2V5aW50PTI1MCBrZXlpbnRfbWluPTEwIHNjZW5lY3V0PTQwIGludHJhX3JlZnJlc2g9MCByY19sb29rYWhlYWQ9NDAgcmM9Y3JmIG1idHJlZT0xIGNyZj0yMy4wIHFjb21wPTAuNjAgcXBtaW49MCBxcG1heD02OSBxcHN0ZXA9NCBpcF9yYXRpbz0xLjQwIGFxPTE6MS4wMACAAAAAFmWIhA/yYoAAw+ycnJ1111111111114BGCAHAAAABkGaOB/hGAEYIAcAAAAGQZpUB/hGARggBwAAAAZBmmA/wjABGCAHAAAABkGagD/CMAAAAAZBmqA/wjABGCAHAAAABkGawD/CMAEYIAcAAAAGQZrgP8IwARggBwAAAAZBmwA/wjABGCAHAAAABkGbID/CMAAAAAZBm0A/wjABGCAHAAAABkGbYD/CMAEYIAcAAAAGQZuAP8IwARggBwAAAAZBm6A/wjAAAAAGQZvAP8IwARggBwAAAAZBm+A/wjABGCAHAAAABkGaAD/CMAEYIAcAAAAGQZogP8IwARggBwAAAAZBmkA7wjAAAAAGQZpgN8IwARggBw==";
      document.body.appendChild(awakeVideo);
    }
    await awakeVideo.play();
    if (!wakeLock) awakeStatus("screen stays awake (video) \u2713");
  } catch (e) { if (!wakeLock) awakeStatus("This screen can\u2019t stay awake on its own: set Auto-Lock to Never"); }
}
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") { keepAwake(); document.title = "pooled \u00b7 room"; } });
document.addEventListener("touchstart", keepAwake, { passive: true });
// Lend this device: this device as a full screen that shows its layers and the passes going through it
function computeState() {
  const by = ai.layersByName || {};
  const spanMax = Object.values(by).reduce((t, r) => { const m = /(\d+)\D*$/.exec(String(r)); return m ? Math.max(t, +m[1] + 1) : t; }, 0);
  const online = $("ai-panel").classList.contains("online");
  const loading = $("ai-panel").classList.contains("loading");
  const mineDeal = /^(\d+)\D+(\d+)$/.exec(String(by[myName] || ""));   // "0-19": layers 1-20
  return {
    code: roomCode, devices: 1 + members.size, role: ai.role,
    model: shortName(ai.model || $("ai-model").value),
    // this device's layers: its engine's range once loaded, before that the deal the download card shows
    lo: ai.range ? ai.range[0] : mineDeal ? +mineDeal[1] : null, hi: ai.range ? ai.range[1] : mineDeal ? +mineDeal[2] + 1 : null,
    total: ai.cfg?.num_hidden_layers || spanMax || 0,
    phase: online ? "serving" : loading ? "loading" : "idle",
    pct: ai.myPct ?? (ai.prog || {})[myName] ?? null,
    color: devColor(myName),
  };
}
const compute = computeScreen({ state: computeState, keepAwake });
$("compute-open").addEventListener("click", () => compute.open());
// the header's dots button says whether this device is working: "on" while it holds layers
function deviceMark() {
  const s = computeState(), b = $("compute-open");
  const on = s.lo != null && s.hi != null && s.phase !== "idle";
  const tip = on ? (s.phase === "loading" ? `This device \u00b7 loading layers ${s.lo + 1}\u2013${s.hi}` : `This device \u00b7 holds layers ${s.lo + 1}\u2013${s.hi}`) : "This device";
  if (b.dataset.tip === tip && b.classList.contains("on") === on) return;
  b.classList.toggle("on", on);
  b.dataset.tip = tip;
  b.setAttribute("aria-label", tip.replace(" \u00b7 ", ": "));
}
setInterval(deviceMark, 1000);
$("create-btn").addEventListener("click", () => { keepAwake(); start(true); });
// (auto-rejoin removed: the user prefers to see what happened)
$("join-btn").addEventListener("click", () => { keepAwake(); start(false); });
$("code-input").addEventListener("keydown", (e) => { if (e.key === "Enter") start(false); });
const codeReady = () => {
  $("join-btn").classList.toggle("ready", /^[A-Z0-9]{4,6}$/i.test($("code-input").value.trim()));
  $("code-input").parentElement.classList.toggle("full", $("code-input").value.length >= 4);
};
// four boxes, four characters: letters and digits only; the fourth one hands off to Join (on a
// phone that also closes the keyboard), so no box waits for a fifth
$("code-input").addEventListener("input", (e) => {
  const el = e.target, v = el.value.toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 4);
  if (el.value !== v) el.value = v;
  codeReady();
  if (v.length === 4 && e.isTrusted && document.activeElement === el) $("join-btn").focus();
});
// Virtual devices: the host can add devices that are iframes of this page on this same computer.
// Each joins the room like any other device (its own WebGPU device, its own WebRTC link, its own
// layers), which shows what a room does before friends arrive; the GPU is shared, so it is a
// demo, not a speed-up. Removing one closes it like a tab (fail fast, re-deal).
let virtualN = 0;
function addVirtual() {
  if (!roomCode) return;
  const q = new URLSearchParams(location.search);
  q.set("code", roomCode); q.set("vname", `virtual-${++virtualN}`); q.set("vgb", "2"); q.set("embed", "1");
  const path = location.pathname.startsWith("/r/") ? "/room" : location.pathname;
  const box = document.createElement("div");
  box.className = "vdev";
  box.innerHTML = `<iframe title="virtual device ${virtualN}" src="${esc(path + "?" + q)}" allow="clipboard-write"></iframe><button type="button" title="close this virtual device">\u00d7</button>`;
  box.querySelector("button").addEventListener("click", () => box.remove());
  $("virtual").appendChild(box);
  $("virtual").hidden = false;
  toast(`virtual-${virtualN} is joining from this computer`);
}
$("add-virtual").addEventListener("click", addVirtual);

// Join links: pooled.run/r/ABCD opens this page and joins the room with no typing. Served
// elsewhere (a local static server, the emulator), the link keeps this page's path and query
// (signal=, wire=) and adds ?code=.
function roomLink() {
  if (location.pathname === "/room" || location.pathname.startsWith("/r/")) return `${location.origin}/r/${roomCode}`;
  const q = new URLSearchParams(location.search); q.set("code", roomCode);
  return `${location.origin}${location.pathname}?${q}`;
}
function copyRoomLink() {
  const url = roomLink();
  if (!navigator.clipboard) { toast("room code: " + roomCode); return; }
  navigator.clipboard.writeText(url).then(() => toast("join link copied")).catch(() => { navigator.clipboard.writeText(roomCode); toast("room code copied"); });
}
function openShare() {
  const url = roomLink();
  $("share-qr").innerHTML = qrSVG(url, { size: 220 });
  $("share-url").textContent = url;
  $("share-code").textContent = roomCode;
  $("share-native").hidden = !navigator.share;
  $("share").hidden = false;
  $("share-close").focus({ preventScroll: true });
}
$("room-badge").addEventListener("click", openShare);
$("share-btn").addEventListener("click", openShare);
for (const b of document.querySelectorAll("[data-invite]")) b.addEventListener("click", (e) => { e.preventDefault(); openShare(); });
function closeShare() { $("share").hidden = true; if (document.body.classList.contains("in-room")) $("share-btn").focus({ preventScroll: true }); }
$("share-close").addEventListener("click", closeShare);
$("share").addEventListener("click", (e) => { if (e.target === $("share")) closeShare(); });
$("room-over-close").addEventListener("click", () => { $("room-over").hidden = true; });
$("share-copy").addEventListener("click", copyRoomLink);
$("share-native").addEventListener("click", () => navigator.share?.({ title: "Join my Pooled room", text: `Room ${roomCode}: add this device to the AI model we run together`, url: roomLink() }).catch(() => {}));
$("room-over-new").addEventListener("click", () => { location.href = location.pathname.startsWith("/r/") ? "/room" : location.pathname.replace(/\?.*$/, ""); });
// A host that reloads its tab goes straight back into its room (no note on the join screen): only on a
// real reload of this tab, and only while the guests are still waiting for it (HOST_WAIT_MS).
const reloaded = (() => { try { return performance.getEntriesByType("navigation")[0]?.type === "reload"; } catch { return false; } })();
const backAsHost = reloaded ? savedHost() : null;
// a link with a room code fills it in and joins once the GPU probe is done
const linkCode = codeFromLocation(location.pathname, location.search, location.hash);
// a virtual device (an iframe the host added, see addVirtual): its name, pledge and a compact page
const VQ = new URLSearchParams(location.search);
if (VQ.get("embed") === "1") document.documentElement.classList.add("embed");
if (VQ.get("vname")) $("name-input").value = VQ.get("vname").slice(0, 20);
if (+VQ.get("vgb") > 0) $("join-gb").value = +VQ.get("vgb");
if (backAsHost && Date.now() - backAsHost.t < 60000 && !(linkCode && linkCode !== backAsHost.code)) {
  metaPromise.then(() => { if (!peer) start(true, backAsHost); });
} else if (linkCode) {
  $("code-input").value = linkCode; codeReady();
  joinWait(true, `Joining room ${linkCode}`);
  $("join-status").textContent = "Checking this device\u2026";
  metaPromise.then(() => { if (!peer) start(false); });
}

// ================= distributed inference =================

// ---- on-disk cache of weight ranges (Cache API): a second start skips the download ----
let weightCache = null, cacheHits = 0;
// The names "swarmllm-weights-v1", "https://weights.swarmllm.ai/" (a cache key namespace, never
// fetched) and the "x-swarm-len" header are from before the rename to Pooled. They stay so weights
// people already downloaded keep working; the Cache API is per site, so pooled.run starts empty anyway.
async function getWeightCache() {
  if (weightCache !== null) return weightCache;
  try { weightCache = await caches.open("swarmllm-weights-v1"); } catch { weightCache = false; }
  return weightCache;
}
function cacheKey(url, lo, hi) { return "https://weights.swarmllm.ai/" + encodeURIComponent(url) + "/" + lo + "-" + hi; }
async function rangeFetch(url, lo, hi, noCache = false) {
  const c = await getWeightCache();
  const key = cacheKey(url, lo, hi);
  if (c && !noCache) {
    try {
      const hit = await c.match(key);
      if (hit) {
        // only trust a complete entry: a tab that died mid-write leaves a short one behind
        if (hit.headers.get("x-swarm-len") === String(hi - lo + 1)) { cacheHits += hi - lo + 1; return hit; }
        c.delete(key).catch(() => {});
      }
    } catch {}
  }
  // another device in the room has this range cached: take it over WebRTC (same Wi-Fi is
  // usually far faster than the model host), falling back to the network on any failure
  const src = !noCache && ai.wsrc?.url === url ? ai.wsrc.map.get(lo + "-" + hi) : null;
  if (src) {
    try {
      const buf = await peerGet(src, url, lo, hi);
      ai.peerBytes = (ai.peerBytes || 0) + buf.byteLength;
      const resp = new Response(buf, { status: 200, headers: { "content-type": "application/octet-stream", "x-swarm-len": String(buf.byteLength) } });
      if (c && !myMeta?.phone) c.put(key, resp.clone()).catch(() => {});
      return resp;
    } catch (err) { crumb(`peer weights from ${conns.get(src)?.name || src} failed (${err.message}); using the network`); ai.wsrc.map.delete(lo + "-" + hi); }
  }
  ai.netBytes = (ai.netBytes || 0) + (hi - lo + 1);
  const r = await fetch(url, { headers: { Range: `bytes=${lo}-${hi}` } });
  if (r.status !== 206) throw new Error("model host refused range requests");
  if (c && !myMeta?.phone) {   // phones skip the store (no spare RAM for the copy); Cache API refuses 206s, so store as a plain 200
    try {
      // buffer the copy fully first, so a complete body is the only thing that ever gets stored
      r.clone().arrayBuffer().then((buf) => {
        if (buf.byteLength !== hi - lo + 1) return;
        return c.put(key, new Response(buf, { status: 200, headers: { "content-type": "application/octet-stream", "x-swarm-len": String(buf.byteLength) } }));
      }).then(() => { ai.cachedBytes = (ai.cachedBytes || 0) + (hi - lo + 1); }, () => {});
    } catch {}
  }
  return r;
}
// ---- weights from the room: devices share the ranges they have cached ----
// Inventory: the "lo-hi" byte ranges of `url` this device has cached (phones cache nothing).
const PEER_WEIGHTS = new URLSearchParams(location.search).get("peerweights") !== "0";
async function cachedRanges(url) {
  const c = await getWeightCache(); if (!c) return [];
  const prefix = "https://weights.swarmllm.ai/" + encodeURIComponent(url) + "/";
  try { return (await c.keys()).map((r) => r.url).filter((u) => u.startsWith(prefix)).map((u) => u.slice(prefix.length)).filter((x) => /^\d+-\d+$/.test(x)); }
  catch { return []; }
}
// host: ask every device what it has, wait briefly; -> { peerId: ["lo-hi", ...] }
async function gatherInventory(url, ms = 1500) {
  if (!PEER_WEIGHTS) return {};
  const inv = {};
  ai.invWait = { url, inv };
  broadcastAll({ t: "ai-inv-req", url });
  await new Promise((r) => setTimeout(r, conns.size ? ms : 0));
  ai.invWait = null;
  return inv;
}
// who to ask for each range: the first device (other than me) that has it
function weightSources(url, inv) {
  const map = new Map();
  for (const [id, have] of Object.entries(inv || {})) if (id !== peer.id) for (const k of have || []) if (!map.has(k)) map.set(k, id);
  return { url, map };
}
const wGets = new Map();   // request id -> { buf, got, res, rej, timer }
let wSeq = 0;
async function peerGet(src, url, lo, hi) {
  if (!(await ensureLink(src, 10000))) throw new Error("no link");
  const len = hi - lo + 1, id = `${peer.id}:${++wSeq}`;
  return new Promise((res, rej) => {
    const w = { buf: new Uint8Array(len), got: 0, res, rej, timer: null };
    const idle = () => { clearTimeout(w.timer); w.timer = setTimeout(() => { wGets.delete(id); rej(new Error("stalled")); }, 15000); };
    w.idle = idle; idle();
    wGets.set(id, w);
    sendTo(src, { t: "ai-wget", id, url, lo, hi });
  });
}
function onWeightPart(d) {
  const w = wGets.get(d.id); if (!w) return;
  if (d.miss) { clearTimeout(w.timer); wGets.delete(d.id); w.rej(new Error("not cached there")); return; }
  if (d.data) {
    const part = d.data instanceof Uint8Array ? d.data : new Uint8Array(d.data);
    if (d.off >= 0 && d.off + part.length <= w.buf.length) { w.buf.set(part, d.off); w.got += part.length; }
    w.idle();
  }
  if (d.done) {
    clearTimeout(w.timer); wGets.delete(d.id);
    if (w.got === w.buf.length) w.res(w.buf.buffer); else w.rej(new Error(`short: ${w.got}/${w.buf.length}`));
  }
}
// serve a cached range to a device in the room, 64 KB at a time, minding the channel's buffer
async function serveWeight(from, d) {
  const e = conns.get(from); if (!e) return;
  const c = await getWeightCache();
  const hit = c && Number.isInteger(d.lo) && Number.isInteger(d.hi) ? await c.match(cacheKey(d.url, d.lo, d.hi)).catch(() => null) : null;
  if (!hit || hit.headers.get("x-swarm-len") !== String(d.hi - d.lo + 1)) { sendTo(from, { t: "ai-wpart", id: d.id, miss: 1 }); return; }
  const buf = new Uint8Array(await hit.arrayBuffer());
  const CH = 64 * 1024;
  for (let off = 0; off < buf.length; off += CH) {
    if (!conns.has(from)) return;
    e.conn.send({ t: "ai-wpart", id: d.id, off, data: buf.subarray(off, Math.min(buf.length, off + CH)) });
    while (e.conn.dataChannel && e.conn.dataChannel.bufferedAmount > 4 * 2 ** 20) await new Promise((r) => setTimeout(r, 10));
  }
  sendTo(from, { t: "ai-wpart", id: d.id, done: 1 });
  ai.servedBytes = (ai.servedBytes || 0) + buf.length;
}

async function fetchGGUFHeader(url, needTokenizer = true) {
  let size = 12 * 2 ** 20;
  for (;;) {
    const r = await rangeFetch(url, 0, size - 1);   // 206 from the network, 200 from the cache
    const buf = await r.arrayBuffer();
    try { return parseGGUFHeader(buf, { skipTokenizer: !needTokenizer }); }
    catch (e) { if (size > 256 * 2 ** 20) throw e; size *= 2; }
  }
}
let pacerHook = null;
const streamWithRetry = (url, streamOpts) => async (info) => {
  try { return await streamEntryToGPU(ai.device, info, openRangeOf(url), streamOpts); }
  catch (e) {
    if (!/short tensor/.test(String(e))) throw e;
    const c = await getWeightCache();
    if (c) c.delete(cacheKey(url, info.byteOffset, info.byteOffset + info.byteLength - 1)).catch(() => {});
    return streamEntryToGPU(ai.device, info, (i) => rangeFetch(url, i.byteOffset, i.byteOffset + i.byteLength - 1, true), streamOpts);
  }
};
// Prefetch: a shard is hundreds of tensors (a 27B worker with 30 layers fetches ~450), and fetching
// them one after another pays the model host's time-to-first-byte every time. When the loader
// asks for a tensor, the next PREFETCH tensors of the shard (file order) are requested too, so
// several are in flight at once. Phones keep one: every buffered body is RAM they do not have.
// ?prefetch=N overrides (0 = off, for A/B).
const PREFETCH_Q = new URLSearchParams(location.search).get("prefetch");
const prefetcher = { url: null, list: [], at: new Map(), pending: new Map() };
function planPrefetch(url, infos) {
  prefetcher.url = url;
  prefetcher.list = infos.filter(Boolean).sort((a, b) => a.byteOffset - b.byteOffset);
  prefetcher.at = new Map(prefetcher.list.map((x, i) => [x.byteOffset, i]));
  prefetcher.pending = new Map();
}
function rangeOf(url, info) {
  const lo = info.byteOffset, hi = info.byteOffset + info.byteLength - 1;
  if (url !== prefetcher.url) return rangeFetch(url, lo, hi);
  const ahead = PREFETCH_Q != null ? Math.max(0, parseInt(PREFETCH_Q, 10) || 0) : myMeta?.phone ? 1 : 4;
  const i = prefetcher.at.get(lo);
  if (i !== undefined) for (let k = i + 1; k <= i + ahead && k < prefetcher.list.length; k++) {
    const n = prefetcher.list[k];
    if (!prefetcher.pending.has(n.byteOffset)) {
      const p = rangeFetch(url, n.byteOffset, n.byteOffset + n.byteLength - 1);
      p.catch(() => {});
      prefetcher.pending.set(n.byteOffset, p);
    }
  }
  const p = prefetcher.pending.get(lo);
  if (p) { prefetcher.pending.delete(lo); return p.catch(() => rangeFetch(url, lo, hi)); }   // a failed prefetch retries in line
  return rangeFetch(url, lo, hi);
}
// the tensors a shard loads, for the prefetcher (a superset is harmless: the list only orders fetches)
function shardInfos(G, names) { return [...new Set(names)].map((n) => G.tensors[n]).filter(Boolean); }
const openRangeOf = (url) => async (info) => {
  if (pacerHook) await pacerHook();
  crumb("streaming " + info.name + " (" + (info.byteLength / 2 ** 20).toFixed(0) + " MB)");
  return rangeOf(url, info);
};
const rangeBytesOf = (url) => async (info) => {
  if (pacerHook) await pacerHook();
  crumb("fetching " + info.name + " (" + (info.byteLength / 2 ** 20).toFixed(0) + " MB)");
  let r = await rangeOf(url, info);
  let bytes = new Uint8Array(await r.arrayBuffer());
  if (bytes.length !== info.byteLength) {
    r = await rangeFetch(url, info.byteOffset, info.byteOffset + info.byteLength - 1, true);
    bytes = new Uint8Array(await r.arrayBuffer());
    if (bytes.length !== info.byteLength) throw new Error(`short download for ${info.name}: ${bytes.length}/${info.byteLength} bytes`);
  }
  return bytes;
};

let ai = {
  visibility: "all",   // who sees the chat: all | host | asker (room/visibility.js)
  engine: null, tok: null, cfg: null, device: null,
  role: null,            // "host" | "worker" | "guest"
  chain: [],             // host: worker peer ids in pipeline order
  next: null,            // worker: peer id to forward hidden to, or "host"
  readyPeers: new Set(),
  pos: 0,
  waiters: new Map(),    // host: lap key (pos, or "b" + basePos) -> { res, rej } for a frame on its way round the chain
  busy: false,
  abort: false,          // host: Stop was pressed; the decode loop ends after the lap in flight
  degraded: false,       // host: a device in the chain left; generation needs a re-deal first
  askerId: null,         // host: who asked the question being answered
  conv: { turns: [] },   // host: the conversation (room/conversation.js)
  fed: [],               // host: the exact tokens every device's caches hold, in order; null = unknown, reset first
  pendingCtl: {},        // host: control for the chain that rides on the next frame ({ reset } or { rb })
  settings: { persona: "default", sampling: "creative", thinking: false, length: "normal" },
  transcript: [],        // host: [{ name, text, reply, stats }] for devices that join later
  teleBy: new Map(),     // host: worker id -> compute ms per frame kind, from ai-tele
  msPerLayer: new Map(), // host: device name -> measured verify compute per layer (the speed split uses it)
  q: Promise.resolve(),  // worker: frames run strictly one after another, in arrival order
};

function aiStatus(s) { $("ai-status").textContent = s; crumb(s); if ($("load-card").classList.contains("on") && !lcBytes) lcStatus(null, s); }
// breadcrumb: if iOS kills the tab, the reloaded page can say where it died
function crumb(s) { try { localStorage.setItem("pooled-crumb", JSON.stringify({ s, t: Date.now(), mem: performance.memory?.usedJSHeapSize })); } catch {} }
// (crumb is kept in localStorage for debugging, not shown on the join screen)
function aiLoading(show, title) {
  $("ai-loading").style.display = show ? "block" : "none";
  if (title) $("ldg-title").textContent = title;
  $("ai-panel").classList.toggle("loading", !!show);
  $("load-card").classList.toggle("on", !!show);
  $("ai-empty").style.display = show ? "none" : "";
  if (show) { $("lc-model").textContent = MODELS[$("ai-model").value]?.label.split("·")[0].trim() || ""; lcBytes = false; eta.t0 = 0; lcStatus(null, "Getting this device ready"); }
  lcStarting(false);
  if (show) loadCardRender();
}
// Every device has its layers: the card turns to the Pooled mark in its wave, "Starting <model>", and
// the step it is on, until the room opens. After a short wait, so a small model that starts at once
// goes straight to the chat without the card flashing.
let lcStartT = 0;
function lcStarting(on, n = 0) {
  const card = $("load-card");
  if (!on) { clearTimeout(lcStartT); lcStartT = 0; if (card.classList.contains("starting")) { card.classList.remove("starting"); $("lc-verb").textContent = "Loading"; $("load-card").querySelector(".lc-note").textContent = "Each device downloads only its own layers."; } return; }
  if (lcStartT || card.classList.contains("starting")) return;
  lcStartT = setTimeout(() => {
    lcStartT = 0;
    if (!card.classList.contains("on")) return;
    card.classList.add("starting");
    $("lc-verb").textContent = "Starting";
    card.querySelector(".lc-note").textContent = n > 2 ? `All ${n} devices have their layers` : n === 2 ? "Both devices have their layers" : "The layers are all here";
  }, 700);
}
function loadCardRender() {
  const rows = $("lc-rows"); if (!rows) return;
  const names = [myName, ...[...conns.values()].map((c) => c.name)].sort(bySlot);
  const by = ai.layersByName || {};
  const order = Object.keys(by);
  rows.innerHTML = names.map((nm) => {
    const pct = Math.max(0, Math.min(100, (ai.prog || {})[nm] ?? 0));
    const l = by[nm];
    return `<div class="lc-row${pct >= 100 ? " done" : ""}${l || !order.length ? "" : " out"}" style="--sw:${devColor(nm)}"><i class="sw"></i><div class="n"><span class="nm">${esc(String(nm))}${nm === myName ? " <small>(you)</small>" : ""}</span>${l ? `<span class="lr">${pct >= 100 ? "" : '<span class="lw">downloading </span>'}layers ${esc(humanRange(l))}</span>` : ""}</div><div class="bar"><div class="fill" style="width:${pct}%"></div></div><div class="pct">${pct >= 100 ? "ready" : pct + "%"}</div></div>`;
  }).join("");
  // the model as a strip of layers: each device's share fills in as its download goes
  const spans = order.map((nm) => { const m = /^(\d+)\D+(\d+)$/.exec(by[nm]); return m ? { nm, lo: +m[1], hi: +m[2] + 1 } : null; }).filter(Boolean);
  const total = spans.reduce((t, x) => Math.max(t, x.hi), 0);
  const strip = $("lc-strip");
  if (!total) { strip.innerHTML = ""; $("lc-sum").textContent = ""; return; }
  const n = Math.min(total, 64), per = total / n;
  let html = "";
  for (let c = 0; c < n; c++) {
    const L = c * per, sp = spans.find((x) => L >= x.lo && L < x.hi);
    const pct = sp ? (ai.prog || {})[sp.nm] ?? 0 : 0;
    const got = sp && (L - sp.lo) / Math.max(1, sp.hi - sp.lo) * 100 < pct;
    html += `<i${got ? ` style="background:${devColor(sp.nm)}"` : ""}></i>`;
  }
  strip.innerHTML = html;
  $("lc-sum").textContent = `${total} layers · ${spans.length} device${spans.length > 1 ? "s" : ""}`;
  const allIn = spans.length > 0 && spans.every((x) => ((ai.prog || {})[x.nm] ?? 0) >= 100);
  if (allIn) lcStarting(true, spans.length); else lcStarting(false);
}
// This device's line under the card: where its bytes come from (the network, devices in the room,
// or the browser's cache), how far along, and a time left once the rate has settled (30 s of data,
// or 10 s with a steady rate), so the first guess is not a wild one.
let lcBytes = false;
const eta = { t0: 0, t: 0, done: 0, rate: 0, hist: [] };
const fmtBytes = (b) => b >= 2 ** 30 ? (b / 2 ** 30).toFixed(1) + " GB" : Math.max(1, Math.round(b / 2 ** 20)) + " MB";
function etaText(s) {
  if (s < 45) return "less than a minute left";
  if (s < 90) return "about a minute left";
  return `about ${Math.round(s / 60)} min left`;
}
function lcStatus(p, text) {
  const el = $("lc-status"); if (!el) return;
  if (!p) { el.innerHTML = `<span class="src gpu">This device</span><span>${esc(String(text || "").replace(/^./, (c) => c.toUpperCase()))}</span>`; return; }
  const src = p.src === "cache" ? ["cache", "Loading from cache"] : p.src === "peer" ? ["", "Copying from the room"] : ["", "Downloading"];
  const et = p.left == null ? '<span class="eta wait">estimating time left</span>' : p.left > 1 ? `<span class="eta">${etaText(p.left)}</span>` : "";
  el.innerHTML = `<span class="src ${src[0]}">${src[1]}</span><span class="b">${fmtBytes(p.done)} of ${fmtBytes(p.total)}</span>${p.done < p.total ? et : ""}`;
}
function aiProgress(done, total, note) {
  const pct = total ? Math.min(100, Math.round(done / total * 100)) : 0;
  const now = performance.now();
  if (done < eta.done || !eta.t0) { eta.t0 = eta.t = now; eta.done = done; eta.rate = 0; eta.hist = []; }
  else if (now - eta.t > 500) {
    const r = (done - eta.done) / ((now - eta.t) / 1000);
    eta.rate = eta.rate ? 0.8 * eta.rate + 0.2 * r : r; eta.t = now; eta.done = done;
    eta.hist.push({ t: now, rate: eta.rate });
    while (eta.hist.length && now - eta.hist[0].t > 10000) eta.hist.shift();
  }
  const left = eta.rate > 0 && total > done ? (total - done) / eta.rate : 0;
  const recent = eta.hist.filter((h) => now - h.t < 8000).map((h) => h.rate);
  const steady = now - eta.t0 > 10000 && recent.length >= 6 && Math.max(...recent) / Math.max(1, Math.min(...recent)) < 1.18;
  const known = left > 0 && (now - eta.t0 > 30000 || steady);
  const leftTxt = known && left > 1 ? ` · ${etaText(left)}` : "";
  $("ldg-fill").style.width = pct + "%";
  $("ldg-sub").textContent = `${(done / 2 ** 20).toFixed(0)} MB of ${(total / 2 ** 20).toFixed(0)} MB · ${pct}%${leftTxt}` + (note ? " · " + note : "");
  lcBytes = done < total;
  if (lcBytes) lcStatus({ done, total, left: known ? left : null, src: ai.netBytes ? "net" : ai.peerBytes ? "peer" : cacheHits ? "cache" : "net" });
}
function emptyText(s) { $("ai-empty-t").textContent = s; }
function aiOut() { const o = $("ai-output"); o.style.display = "block"; $("ai-empty").style.display = "none"; return o; }

// ---- chat transcript ----
// A bot message keeps its answer as pieces ({ t: text, d: 1 when the token was an accepted
// speculative draft }) so "show drafts" can re-render it with the drafted tokens marked.
let botEl = null;
let draftView = false;
function scrollChat() { const o = $("ai-output"); o.scrollTop = o.scrollHeight; }
function chatUser(name, text) {
  const o = aiOut();
  const m = document.createElement("div");
  m.className = "m user";
  m.innerHTML = `<div class="who">${esc(name)}</div><div class="bubble">${esc(text).replace(/\n/g, "<br>")}</div>`;
  m.dataset.name = name; m.dataset.text = text;
  o.appendChild(m); scrollChat();
}
function chatBotStart(mid) {
  const o = aiOut();
  const m = document.createElement("div");
  m.className = "m bot";
  if (mid != null) m.dataset.mid = mid;
  m.innerHTML = `<div class="who"><span class="wn"></span><span class="wd" aria-hidden="true"><i></i><i></i><i></i></span></div><div class="bubble"></div>`;
  m.querySelector(".wn").textContent = shortName(ai.model || $("ai-model").value) || "room";
  // until the first token: the working line (the first piece replaces it)
  m.querySelector(".bubble").append(working());
  m.classList.add("live");
  m.pieces = [];
  o.appendChild(m); scrollChat();
  botEl = m;
}
function renderBot(m, live) {
  const b = m.querySelector(".bubble");
  if (draftView && m.pieces.length) {
    b.classList.add("drafts");
    b.innerHTML = m.pieces.map((p) => p.d ? `<span class="dr${p.d === 2 ? " lk" : ""}">${esc(p.t)}</span>` : esc(p.t)).join("") + (live ? '<span class="cursor"></span>' : "");
  } else {
    b.classList.remove("drafts");
    b.innerHTML = mdChat(m.pieces.map((p) => p.t).join("")) + (live ? '<span class="cursor"></span>' : "");
    if (live) { const cur = b.lastElementChild, last = cur?.previousElementSibling; if (last && /^(P|LI|UL|OL|H3|H4)$/.test(last.tagName)) ((last.tagName === "UL" || last.tagName === "OL") ? last.lastElementChild || last : last).appendChild(cur); }
    if (!live) for (const pre of b.querySelectorAll("pre")) {   // finished code blocks get a copy button
      const w = document.createElement("div"); w.className = "code-wrap";
      pre.replaceWith(w); w.appendChild(pre);
      w.insertAdjacentHTML("beforeend", `<button type="button" class="copy-code icon-act" aria-label="Copy the code" title="Copy the code">${COPY_SVG}</button>`);
    }
  }
}
function chatBotPiece(text, d) {
  if (!botEl) chatBotStart();
  botEl.pieces.push({ t: text, d: d === 2 ? 2 : d ? 1 : 0 });
  renderBot(botEl, true);
  scrollChat();
}
function chatBotEnd(note, stats) {
  if (!botEl) chatBotStart();
  if (note) botEl.pieces = [{ t: note, d: 0 }];
  renderBot(botEl, false);
  botEl.classList.remove("live");
  // a finished answer in a background tab: say so in the tab title until the tab is looked at
  if (!note && document.hidden) { document.title = "\u2713 answer ready \u00b7 pooled"; }
  // under the answer: a copy icon (answers only, not notes), then the numbers
  const acts = document.createElement("div");
  acts.className = "m-acts";
  if (!note && botEl.pieces.length) acts.innerHTML = `<button type="button" class="copy-ans icon-act" aria-label="Copy the answer" title="Copy">${COPY_SVG}</button>`;
  if (stats) { const s = document.createElement("div"); s.className = "stats"; s.textContent = stats; acts.appendChild(s); }
  if (acts.childElementCount) botEl.appendChild(acts);
  if (!note && botEl.dataset.mid && readAloud && botEl.pieces.length) speak(botEl.pieces.map((p) => p.t).join(""));
  botEl = null;
}
// the clipboard icon, and the tick it turns into for a moment once copied
const COPY_SVG = '<svg class="cp" width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" aria-hidden="true"><rect x="5.5" y="5.5" width="8" height="8.5" rx="1.8"/><path d="M10.5 5.5V3.8c0-1-.8-1.8-1.8-1.8H4.3c-1 0-1.8.8-1.8 1.8v4.4c0 1 .8 1.8 1.8 1.8h1.2"/></svg><svg class="ok" width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3.5 8.5l3 3 6-7"/></svg>';

// ---- the room's social bits: who is typing, answers read aloud (emoji reactions were removed) ----
function copyText(text, what, btn) {
  if (!navigator.clipboard) { toast("this browser can't copy here"); return; }
  navigator.clipboard.writeText(text).then(() => {
    if (!btn) { toast(`${what} copied`); return; }
    btn.classList.add("done"); btn.setAttribute("aria-label", "Copied");
    clearTimeout(btn._t); btn._t = setTimeout(() => { btn.classList.remove("done"); btn.setAttribute("aria-label", `Copy the ${what}`); }, 1600);
  }, () => toast("couldn't copy"));
}
// copy an answer (its raw text, markdown and all) or one code block
$("ai-output").addEventListener("click", (ev) => {
  const ca = ev.target.closest(".copy-ans");
  if (ca) { const m = ca.closest(".m.bot"); if (m?.pieces) copyText(m.pieces.map((p) => p.t).join("").replace(/<think>[\s\S]*?<\/think>\s*/g, ""), "answer", ca); return; }
  const cc = ev.target.closest(".copy-code");
  if (cc) { copyText(cc.parentElement.querySelector("pre")?.textContent || "", "code", cc); return; }
});
let typingAt = 0;
function noteTyping() {
  const now = Date.now();
  if (now - typingAt < 2000 || !$("ai-prompt").value.trim()) return;
  typingAt = now;
  if (ai.role === "host") { if (ai.visibility === "all") broadcastAll({ t: "ai-typing", name: myName }); }
  else if (ai.hostId && conns.has(ai.hostId)) sendTo(ai.hostId, { t: "ai-typing" });
}
let typingTimer = null;
function showTyping(name) {
  $("typing-note").textContent = `${name} is typing\u2026`;
  clearTimeout(typingTimer);
  typingTimer = setTimeout(() => { $("typing-note").textContent = ""; }, 3500);
}
let readAloud = false;
function speak(raw) {
  try {
    const text = raw.replace(/<think>[\s\S]*?(<\/think>|$)/g, "").replace(/[*_`#>]+/g, "").trim();
    if (!text) return;
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(text.slice(0, 4000)));
  } catch {}
}
function setDraftView(on) {
  draftView = on;
  $("draft-view").classList.toggle("on", on);
  $("draft-view").textContent = on ? "Hide drafts" : "Show drafts";
  for (const m of document.querySelectorAll("#ai-output .m.bot")) if (m.pieces) renderBot(m, m === botEl);
  if (on) toast("blue: guessed by the draft head · green: copied from earlier in the chat · both confirmed by the whole room in one lap");
}
// the room card: this room's best finished answer speed, its devices and layers, as a PNG
function openCard() {
  const nodes = lastMap?.nodes?.length ? lastMap.nodes : [{ name: myName, layers: "", host: 1 }];
  const tps = bestTps || lastMap?.st?.tps || lastSoloTps || 0;
  drawCard($("card-canvas"), { model: (MODELS[ai.model || $("ai-model").value]?.label || "").split("\u00b7")[0].trim(),
    code: roomCode, nodes, tps, acc: lastMap?.st?.acc, lap: lastMap?.st?.lap, date: new Date().toISOString().slice(0, 10) });
  $("card").hidden = false;
}
async function cardBlob() { return new Promise((res) => $("card-canvas").toBlob(res, "image/png")); }
$("card-btn").addEventListener("click", () => { $("room-menu").open = false; openCard(); });
$("card-close").addEventListener("click", () => { $("card").hidden = true; });
$("card").addEventListener("click", (e) => { if (e.target === $("card")) $("card").hidden = true; });
$("card-save").addEventListener("click", async () => {
  const a = document.createElement("a"); a.href = URL.createObjectURL(await cardBlob()); a.download = `pooled-${roomCode || "room"}.png`; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
});
$("card-share").addEventListener("click", async () => {
  const file = new File([await cardBlob()], `pooled-${roomCode || "room"}.png`, { type: "image/png" });
  if (navigator.canShare?.({ files: [file] })) navigator.share({ files: [file], title: "Our Pooled room" }).catch(() => {});
  else toast("this browser can't share images: use save");
});
let lastSoloTps = 0;
function exportChat() {
  const lines = [`# Pooled room ${roomCode || ""}`, "", `_${new Date().toISOString().slice(0, 16).replace("T", " ")} · ${MODELS[ai.model || $("ai-model").value]?.label || ""}_`, ""];
  for (const m of document.querySelectorAll("#ai-output .m")) {
    if (m.classList.contains("user")) lines.push(`**${m.dataset.name || "?"}:** ${m.dataset.text || ""}`, "");
    else if (m.pieces) {
      lines.push(m.pieces.map((p) => p.t).join(""), "");
      const st = m.querySelector(".stats")?.textContent;
      if (st) lines.push(`<sub>${st}</sub>`, "");
    }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/markdown" }));
  a.download = `pooled-chat-${roomCode || "room"}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

// Send turns into Stop while an answer streams: always on the host, on the asker's screen for
// guests (the host honours ai-stop from the asker only).
function setBusyUI(busy, canStop) {
  const b = $("ai-send");
  b.dataset.canStop = busy && canStop ? "1" : "";
  b.disabled = false;
  $("ai-row").classList.toggle("busy", !!busy);
  sendLabel();
}
// while busy, the button stops the answer when the box is empty and queues the text otherwise
function sendLabel() {
  const b = $("ai-send"), busy = $("ai-row").classList.contains("busy"), typed = !!$("ai-prompt").value.trim();
  const stop = busy && b.dataset.canStop === "1" && !typed;
  b.classList.toggle("stop", stop);
  b.textContent = stop ? "Stop" : busy ? "Queue" : "Send";
}
const kfmt = (n) => n >= 10000 ? (n / 1000).toFixed(0) + "k" : n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n);
function setCtx(used, max) {
  const sm = $("sm-ctx");
  if (sm) { sm.textContent = used ? `${kfmt(used)} / ${max % 1024 === 0 && max >= 1024 ? max / 1024 + "k" : kfmt(max)}` : "-"; sm.classList.toggle("warn", !!used && used > max * 0.8); }
  // by the composer: the same ring as Code's (room/code-ui.js ctx), how much of the context the chat uses
  const el = $("ctx-meter"); if (!el) return;
  if (!used || !max) { el.replaceChildren(); el.removeAttribute("title"); el.classList.remove("warn"); return; }
  const pct = Math.min(100, Math.max(1, Math.round(used / max * 100)));
  el.innerHTML = `<i style="--p:${pct}" aria-hidden="true"></i>${pct}% of context`;
  el.title = `${used.toLocaleString("en-US")} of ${max.toLocaleString("en-US")} tokens`;
  el.classList.toggle("warn", used > max * 0.8);
}

async function aiLoadShard(modelKey, range, hasEmbed, hasHead, ctx = maxSeqFor(modelKey)) {
  const M = MODELS[modelKey];
  aiLoading(true, `loading layers ${range[0]}\u2013${range[1] - 1} of ${M.label.split("\u00b7")[0].trim()}`);
  aiStatus("requesting GPU\u2026");
  mascot("Grabbing my slice of the model… hang tight.");
  // a previous attempt in this tab still owns its weights: release them first, or the
  // second load doubles GPU memory and every buffer after the limit comes back invalid
  if (ai.device) { try { ai.device.destroy(); } catch {} ai.device = null; ai.engine = null; }
  ai.firstGpuError = null;
  ai.peerBytes = 0; ai.netBytes = 0; cacheHits = 0;   // per load: a count left from an earlier load in this tab mislabels the status
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("no WebGPU on this device");
  ai.device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: myMeta?.phone ? Math.min(adapter.limits.maxBufferSize, 256 * 2 ** 20) : adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: myMeta?.phone ? Math.min(adapter.limits.maxStorageBufferBindingSize, 256 * 2 ** 20) : adapter.limits.maxStorageBufferBindingSize,
    },
  });
  ai.device.addEventListener?.("uncapturederror", (ev) => {
    const gmsg = ev.error?.message || "";
    if (!ai.firstGpuError) { ai.firstGpuError = gmsg; aiStatus("GPU error: " + gmsg.slice(0, 300)); log("room", "\u26a0 FIRST GPU error on " + myName + ": " + gmsg.slice(0, 600)); }
    crumb("GPU validation error: " + gmsg.slice(0, 400));
    if (ai.hostId && ai.role !== "host") sendTo(ai.hostId, { t: "ai-error", message: "GPU error: " + (ev.error?.message || "").slice(0, 300) });
    log("room", "\u26a0 GPU error on " + myName + ": " + (ev.error?.message || "").slice(0, 140));
  });
  if (location.hash === "#debug") log("room", `${myName}: maxBuf ${(adapter.limits.maxBufferSize / 2 ** 30).toFixed(1)} GB \u00b7 maxBind ${(adapter.limits.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MB`);
  aiStatus("testing GPU kernels on this device\u2026");
  const tAdapter = await navigator.gpu.requestAdapter();   // an adapter gives out one device only
  const tdev = await tAdapter.requestDevice();               // throwaway: its test buffers die with it
  const st = await gpuSelfTest(tdev);
  if (!st.ok) log("room", `${myName} GPU self-test: ${st.detail}`);
  if (!st.ok) throw new Error("GPU self-test FAILED on this device: " + st.detail + " \u2014 please screenshot this");
  const mt = await kernelMicroTests(tdev);
  if (!mt.ok) log("room", `${myName} kernels: ${mt.detail}`);
  if (!mt.ok) throw new Error("GPU kernel FAILED on this device \u2192 " + mt.firstFail + " \u2014 please send me this line");
  try { tdev.destroy(); } catch {}
  ai.device.lost.then((l) => crumb("GPU device lost: " + l.reason + " " + l.message));
  aiStatus("tuning kernels for this GPU\u2026");
  ai.tune = await autotuneCoop(ai.device).catch(() => ({ wg: 256, rows: 4 }));
  crumb(`autotune: WG=${ai.tune.wg} ROWS=${ai.tune.rows}`);
  const isPhone = myMeta?.phone;
  ai.myPct = 0;
  ai.prog = { [myName]: 0 }; ai.progAt = { [myName]: Date.now() };
  const streamOpts = { pace: isPhone ? 300 : 0, staging: isPhone ? 2 * 2 ** 20 : 8 * 2 ** 20 };
  if (M.cfg) {
    ai.cfg = await (await fetch(M.cfg)).json();
    if (hasEmbed || hasHead) ai.tok = makeTokenizer(await (await fetch(M.tok)).json());
  }

  const onProg = (done, total) => {
    aiProgress(done, total);
    // name where the bytes are coming from right now: the network, devices in the room, or this device's cache
    const gb = (b) => (b / 2 ** 30).toFixed(1) + " GB";
    aiStatus(ai.netBytes ? `downloading weights\u2026${cacheHits ? ` (${gb(cacheHits)} was already on this device)` : ""}`
      : ai.peerBytes ? `getting weights from devices in the room\u2026`
      : cacheHits ? `loading weights from this device's cache\u2026` : `downloading weights\u2026`);
    ai.myPct = total ? done / total * 100 : 0;
    ai.prog = ai.prog || {}; ai.progAt = ai.progAt || {};
    ai.prog[myName] = Math.round(ai.myPct); ai.progAt[myName] = Date.now();
    if (ai.role === "worker") sendTo(ai.hostId, { t: "ai-progress", pct: Math.round(ai.myPct) });
    loadCardRender();
  };
  if (ai.role === "host") {
    clearInterval(ai.progTimer);
    ai.progTimer = setInterval(() => { if (ai.role === "host") broadcastAll({ t: "ai-hostprog", all: ai.prog || {}, at: Date.now() }); }, 600);
  }
  // every device (host included, even when its weights come from cache) keeps within a few
  // percent of the slowest device, so the bars climb together and the room finishes as one
  const slowest = () => {
    const now = Date.now();
    let m = Infinity;
    for (const [nm, pct] of Object.entries(ai.prog || {})) {
      if (nm === myName || pct >= 100) continue;
      if (now - ((ai.progAt || {})[nm] || 0) > 30000) continue;     // silent for 30 s: don't wait on it
      m = Math.min(m, pct);
    }
    return m;
  };
  const pacer = async () => {
    while (ai.myPct < 100 && ai.myPct > slowest() + 4) {
      aiStatus(`downloading weights\u2026 in step with the room (${Math.round(ai.myPct)}%)`);
      await new Promise((r) => setTimeout(r, 250));
    }
  };
  pacerHook = pacer;

  if (M.kind === "qwen35") {
    aiStatus("reading model index\u2026");
    const needTok = hasEmbed || hasHead;
    const cachedOk = ai.G && ai.GModel === modelKey && (!needTok || ai.G.meta["tokenizer.ggml.tokens"]);
    const G = cachedOk ? ai.G : await fetchGGUFHeader(M.gguf, needTok);
    ai.G = G; ai.GModel = modelKey;
    ai.cfg = { num_hidden_layers: G.meta["qwen35.block_count"] - (G.meta["qwen35.nextn_predict_layers"] || 0) };
    if (hasEmbed || hasHead) {
      ai.tok = makeTokenizer(tokenizerFromGGUF(G.meta));
      // the model's own chat template: Code mode picks the tool-call format from it (Qwen 3.5+ use
      // XML <function=...> calls, with the full call grammar); without it every model got JSON
      ai.tok.chatTemplate = G.meta["tokenizer.chat_template"] || "";
    }
    // the host also loads the model's multi-token-prediction block: it drafts
    // tokens that the trunk then verifies in one batched pass (same output, faster)
    const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead, mtp: hasHead };
    const total = qwen35ShardBytes(G, opts);
    const names = [];
    for (let l = range[0]; l < range[1]; l++) names.push(...Object.values(qwen35NamesFor(G, l)).filter((v) => typeof v === "string"));
    if (hasEmbed || hasHead) names.push(GGML_EMBED);
    if (hasHead) {
      names.push(GGML_FINAL_NORM, GGML_OUTPUT);
      const N = G.meta["qwen35.block_count"] - 1;
      names.push(...Object.values(qwen35NamesFor(G, N, true)).filter((v) => typeof v === "string"), ...["eh_proj", "enorm", "hnorm", "shared_head_norm"].map((x) => `blk.${N}.nextn.${x}.weight`));
    }
    planPrefetch(M.gguf, shardInfos(G, names));
    G.streamEntry = streamWithRetry(M.gguf, streamOpts);
    const weights = await qwen35Weights(G, rangeBytesOf(M.gguf), opts, (done) => onProg(done, total),
      (e, name) => gpuUploadEntry(ai.device, e, name === GGML_EMBED));   // straight to the GPU, RAM stays flat
    aiStatus("building GPU pipelines (compiling shaders)\u2026");
    ai.engine = await Qwen35Engine.create({
      device: ai.device, meta: G.meta, weights, vocab: G.tensors[GGML_EMBED]?.shape?.[0],
      layerRange: range, hasEmbed, hasHead, maxSeq: ctx,
      coopWG: ai.tune?.wg, coopRows: ai.tune?.rows,
      // 16 batch columns: prefill passes go through the row-stationary GEMM
      // (docs/research/prefill-gemm-v2.md). Speculative verifies are <= 8
      // columns and drop to the 8- or 4-column GEMV twins automatically, so
      // the generated stream is unchanged.
      batchCols: 16, coopRowsB: 1,
      // ?draftvocab=N: draft over the first N vocabulary rows only (engine/qwen35.js). Default 65536:
      // the head is the biggest matrix a draft reads (1.35 GB of Q8 on the 27B, 0.54 GB on the MoE),
      // and on English prose and code only 1-2.5% of tokens lie above 65536
      // (benchmarks/draftvocab_coverage.js). For other scripts (Chinese ~84% above it) the engine
      // falls back to the full head by itself (draftVocabAuto). ?draftvocab=0: full head always.
      // ?dvauto=0: the small head always. Drafts only, the output never changes.
      draftVocab: DRAFT_VOCAB,
      draftVocabAuto: new URLSearchParams(location.search).get("dvauto") !== "0",
      // the K drafts of a speculative step, its verify and its LM head in one submit (keeps the
      // embedding table, or its first draftvocab rows, on the GPU); ?draftchain=0 turns it off
      draftChain: new URLSearchParams(location.search).get("draftchain") !== "0",
      // ?specfuse=0: speculative verify as separate trunk / head submits (A/B; same output bits)
      specFuse: new URLSearchParams(location.search).get("specfuse") !== "0",
      // ?fuse=0: the unfused kernels (attention glue, DeltaNet delta + gated norm, batched
      // attention) for A/B timing; both give the same bits, so devices may differ
      ...(new URLSearchParams(location.search).get("fuse") === "0" ? { attnGlue: false, dnFuse: false, attnMC: false } : {}),
      // ?kv=q8: int8 KV cache (~56% of f16's memory) for long contexts; changes the numerics a little
      kvQ8: new URLSearchParams(location.search).get("kv") === "q8",
      // ?moefuse=0: the unfused MoE FFN kernels (A/B). The fused path (the default) gives different
      // MoE bits, so every device of a room should run the same setting; ?moednrows=1|2|4 tunes it
      moeFuse: new URLSearchParams(location.search).get("moefuse") !== "0",
      moeDnRows: parseInt(new URLSearchParams(location.search).get("moednrows"), 10) || 1,
      // Prefill options (attnPrefillTile, prefillUbatch, moeGroupPrefill) are deliberately not passed: every
      // device takes the engine's defaults, so host and workers agree. Tiled prefill attention runs on the
      // 16-column prefill frames of every device; wide GEMM + expert-grouped MoE only in solo prefillTokens
      // (the device holding the embedding; a split prefill sends 16-column frames). ?kv=q8 turns the tiled
      // attention off and ?moefuse=0 the grouped MoE on that device only.
      // GPU sampling, on by default (?gpusample=0: off): argmax / top-k of the head in the same submit,
      // 16-520 bytes back instead of the 1 MB logits vector; a masked sampler (tool-name constraint)
      // still gets the logits. ?argmaxwide=0|1 (default: same as gpusample): the draft argmax as the
      // two-stage multi-workgroup kernel.
      gpuSample: GPU_SAMPLE,
      argmaxWide: ARGMAX_WIDE,
    });
  } else if (M.kind === "gguf") {
    aiStatus("reading model index\u2026");
    const G = ai.G && ai.GModel === modelKey ? ai.G : await fetchGGUFHeader(M.gguf, false);   // vocab comes from tokenizer.json
    ai.G = G; ai.GModel = modelKey;
    const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead };
    const total = ggufShardBytes(G, opts);
    const names = [];
    for (let l = range[0]; l < range[1]; l++) names.push(...Object.values(ggmlLayerNames(l)));
    if (hasEmbed || hasHead) names.push(GGML_EMBED);
    if (hasHead) names.push(GGML_FINAL_NORM, GGML_OUTPUT);
    planPrefetch(M.gguf, shardInfos(G, names));
    G.streamEntry = streamWithRetry(M.gguf, streamOpts);
    const weights = await ggufWeights(G, rangeBytesOf(M.gguf), opts, (done) => onProg(done, total),
      (e, name) => gpuUploadEntry(ai.device, e, name === GGML_EMBED));
    aiStatus("building GPU pipelines\u2026");
    ai.engine = await DenseEngine.create({
      coopWG: ai.tune?.wg, coopRows: ai.tune?.rows,
      device: ai.device, cfg: ai.cfg, weights,
      layerRange: range, hasEmbed, hasHead, maxSeq: ctx,
    });
  } else {
    const names = shardTensorNames(ai.cfg, range, hasEmbed, hasHead);
    const tensors = await fetchModelShard(M.st, names, (p, done, total) => onProg(done, total));
    aiStatus("building GPU pipelines\u2026");
    ai.engine = await DenseEngine.create({
      coopWG: ai.tune?.wg, coopRows: ai.tune?.rows,
      device: ai.device, cfg: ai.cfg, tensors,
      layerRange: range, hasEmbed, hasHead, maxSeq: MAX_SEQ,
    });
  }
  prefetcher.pending.clear(); prefetcher.url = null;
  if (ai.peerBytes) log("room", `${myName}: ${(ai.peerBytes / 2 ** 20).toFixed(1)} MB of weights came from devices in the room, ${((ai.netBytes || 0) / 2 ** 20).toFixed(1)} MB from the network`);
  if (ai.engine) ai.engine.mtpBatchFill = MTP_BATCH;
  // after a verify: the draft-cache refill as one batched pass (?mtprefill=0: one submit per row)
  // and the next step's first draft run in that same pass (?predraft=0: off). Drafts only.
  if (ai.engine) { ai.engine.mtpBatchRefill = MTP_REFILL; ai.engine.mtpPreDraft = PRE_DRAFT; }
  ai.range = range;
  ai.model = modelKey;
  aiLoading(false);
}

// ---- host ----
// the model host (room/plan.js pickModelHost): the strongest device, a computer before a phone,
// whoever pressed Start. The room's creator stays the PeerJS hub either way.
function biggestPeerId() {
  return pickModelHost([{ id: peer.id, meta: myMeta }, ...[...conns].map(([id, e]) => ({ id, meta: e.meta }))]) || peer.id;
}
function aiStartAnywhere() {
  const model = $("ai-model").value;
  const boss = biggestPeerId();
  if (boss === peer.id) { aiStart(model); return; }
  $("ai-start").disabled = true; $("ai-model").disabled = true;
  aiLoading(true, `starting ${MODELS[model].label.split("·")[0].trim()}`);
  $("ldg-sub").textContent = `${conns.get(boss)?.name || "the biggest device"} is dealing the layers`;
  $("ldg-fill").style.width = "0%";
  aiStatus(`asked ${conns.get(boss)?.name || "the biggest device"} to start ${MODELS[model].label.split("·")[0].trim()}…`);
  broadcastAll({ t: "ai-start-req", model, boss, by: myName });
}
async function aiStart(modelArg) {
  if (ai.engine || ai.busy) return;
  ai.busy = true;
  if (typeof modelArg === "string") setModelValue(modelArg);
  $("ai-start").disabled = true;
  $("ai-model").disabled = true;
  try {
    ai.role = "host";
    ai.degraded = false;
    ai.readyPeers = new Set();
    ai.teleBy = new Map();
    const modelKey = $("ai-model").value;
    const M = MODELS[modelKey];
    // context for this room: the model's default, or ?ctx=N up to its cap (room/models.js CTX); every device builds its engine with it
    const ROOM_CTX = maxSeqFor(modelKey, +new URLSearchParams(location.search).get("ctx") || 0);
    // devices without WebGPU join as ask-only guests: they get the chat, not layers
    ai.chain = [...conns.keys()].filter((id) => conns.get(id)?.meta?.webgpu).sort();
    ai.leftOut = new Set();
    ai.plan = new Map();                      // name -> load message, so a reloaded device can be re-seated
    ai.chainNames = ai.chain.map((id) => conns.get(id)?.name || id);
    const n = ai.chain.length + 1;
    let L, layerBytes, embedBytes, cfg = null;
    if (M.kind === "qwen35") {
      aiStatus("reading model index… (11 MB)");
      ai.G = await fetchGGUFHeader(M.gguf);
      ai.GModel = modelKey;
      L = ai.G.meta["qwen35.block_count"] - (ai.G.meta["qwen35.nextn_predict_layers"] || 0);
      layerBytes = qwen35ShardBytes(ai.G, { lo: 0, hi: 4, hasEmbed: false, hasHead: false }) / 4
        + ROOM_CTX * kvBytesPerLayerPos(ai.G.meta);   // the attention layers' KV cache at this room's context
      embedBytes = (ai.G.tensors[GGML_EMBED]?.byteLength || 0) + (ai.G.tensors[GGML_OUTPUT]?.byteLength || 0) + qwen35MtpBytes(ai.G);
    } else {
      cfg = await (await fetch(M.cfg)).json();
      L = cfg.num_hidden_layers;
    }

    // real per-shard byte costs (gguf: from the file's own index)
    if (M.kind === "gguf") {
      aiStatus("reading model index…");
      ai.G = await fetchGGUFHeader(M.gguf, false);
      ai.GModel = modelKey;
      layerBytes = Object.values(ggmlLayerNames(0))
        .reduce((s, nm) => s + (ai.G.tensors[nm]?.byteLength || 0), 0);
      embedBytes = (ai.G.tensors[GGML_EMBED]?.byteLength || 0) + (ai.G.tensors[GGML_OUTPUT]?.byteLength || 0);
    } else if (M.kind === "safetensors") {
      const d = cfg.hidden_size;
      const kvDim = cfg.num_key_value_heads * ((cfg.head_dim || d / cfg.num_attention_heads));
      layerBytes = (2 * d * d + 2 * kvDim * d + 3 * cfg.intermediate_size * d) * 4;
      embedBytes = cfg.vocab_size * d * 4;
    }
    const pledgeOf = (m) => ((m?.contribGB ?? (m?.maxBufGB ? m.maxBufGB * 0.5 : 0.5))) * 2 ** 30;
    let caps = [Math.max(pledgeOf(myMeta) - embedBytes, layerBytes / 2),
      ...ai.chain.map((id) => Math.max(pledgeOf(conns.get(id)?.meta), layerBytes / 2))];
    let assigned, ranges;
    if ($("ai-split").value === "speed") {
      // fastest devices first (measured ms per layer from earlier answers), fewest hops; devices
      // that are not needed stay in the room as ask-only guests
      const nameOf = (id) => conns.get(id)?.name || id;
      const sp = planForSpeed(L, caps.map((c) => Math.floor(c / layerBytes)), [ai.msPerLayer.get(myName), ...ai.chain.map((id) => ai.msPerLayer.get(nameOf(id)))]);
      const keep = sp.used.filter((i) => i > 0).map((i) => i - 1);
      ai.leftOut = new Set(ai.chain.filter((_, i) => !keep.includes(i)));
      ai.chain = keep.map((i) => ai.chain[i]);
      ai.chainNames = ai.chain.map(nameOf);
      assigned = sp.used.map((i) => sp.assigned[i]);
      ranges = sp.used.map((i) => sp.ranges[i]);
      caps = sp.used.map((i) => caps[i]);
    } else ({ assigned, ranges } = planSplit(L, caps));
    ai.layersN = Object.fromEntries([[myName, assigned[0]], ...ai.chain.map((id, i) => [conns.get(id)?.name || id, assigned[i + 1]])]);

    const needGB = (L * layerBytes + embedBytes) / 2 ** 30;
    const haveGB = caps.reduce((s, c) => s + c, embedBytes) / 2 ** 30;
    if (needGB > haveGB * 1.15)
      log("room", `⚠ this model needs ~${needGB.toFixed(1)} GB but the room pledged ~${haveGB.toFixed(1)} GB — it may not fit`);

    ai.deferred = [];
    // what every device already has cached, so each one can take its missing ranges from the room.
    // The host itself is never a source: it is loading its own layers and serving the whole room,
    // so a device missing a range goes to the model host instead of queueing behind it.
    const inv = M.gguf && conns.size ? await gatherInventory(M.gguf) : {};
    ai.wsrc = M.gguf ? weightSources(M.gguf, inv) : null;
    ai.chain.forEach((id, i) => {
      const msg = {
        t: "ai-load", model: modelKey, range: ranges[i + 1], ctx: ROOM_CTX,
        next: i + 1 < ai.chain.length ? ai.chain[i + 1] : "host",
        host: peer.id,
        inv,
      };
      ai.plan.set(conns.get(id)?.name || id, { msg, small: false });
      sendTo(id, msg);
    });
    ai.layersByName = Object.fromEntries([[myName, `${ranges[0][0]}–${ranges[0][1] - 1}`], ...ai.chain.map((id, i) => [conns.get(id)?.name || id, `${ranges[i + 1][0]}–${ranges[i + 1][1] - 1}`])]);
    broadcastAll({ t: "ai-layers", by: ai.layersByName });
    const splitDesc = [`you ${assigned[0]}+embed`, ...ai.chain.map((id, i) =>
      `${conns.get(id)?.name || id} ${assigned[i + 1]}`)].join(" · ");
    log("room", `${M.label} — layer split ${$("ai-split").value === "speed" ? "for speed" : "by pledge"}: ${splitDesc}`);
    ai.loadingShard = true;
    try { await aiLoadShard(modelKey, ranges[0], true, true, ROOM_CTX); } finally { ai.loadingShard = false; }
    aiStatus(n === 1
      ? `solo: all ${L} layers local — ready`
      : `layers ${ranges[0][0]}–${ranges[0][1] - 1} ready · syncing with ${ai.chain.length} device${ai.chain.length > 1 ? "s" : ""}…`);
    ai.fed = [];                              // fresh engines everywhere: nothing cached yet
    ai.pendingCtl = {};
    ckptClear();
    aiMaybeReady();
  } catch (err) {
    clearInterval(ai.progTimer);
    aiLoading(false);
    ai.engine = null;
    $("ai-panel").classList.remove("online");
    aiStatus("failed: " + err.message);
    ai.busy = false;
    $("ai-start").disabled = false;
    $("ai-model").disabled = false;
  }
}

// Deal the layers again over whoever is in the room now: after a device left (the room is
// degraded) or to bring in devices that joined after the start. Cached ranges reload in seconds;
// the conversation is kept and re-prefilled on the next question.
async function aiRedeal() {
  if (ai.role !== "host" || ai.busy === "gen" || ai.busy === "code") return;
  if (ai.loadingShard) { toast("wait for this device's layers to finish loading, then re-deal"); return; }
  $("chat-tools").hidden = true;
  const model = ai.model || $("ai-model").value;
  failWaiters(new Error("re-dealing the layers"));
  ckptClear();
  ai.engine = null; ai.busy = false; ai.fed = null;
  $("ai-panel").classList.remove("online");
  $("ai-row").style.display = "none";
  showRedeal(false);
  broadcastAll({ t: "ai-redeal", by: myName, model });
  codeRoleChanged();
  aiLoading(true, "re-dealing the layers");
  await aiStart(model);
}
function showRedeal(on, why) {
  const b = $("ai-redeal");
  b.hidden = !on || ai.role !== "host";
  if (why) $("redeal-why").textContent = why;
  $("redeal-why").hidden = b.hidden;
}
// devices with a GPU that are in the room but hold no layers (joined after the start)
function sparePeers() { return [...conns.keys()].filter((id) => conns.get(id)?.meta?.webgpu && !ai.chain.includes(id) && !ai.leftOut?.has(id)); }
function offerRedealForNewcomers() {
  if (ai.role !== "host" || !ai.engine || ai.degraded) return;
  const spare = sparePeers();
  if (spare.length) showRedeal(true, `${spare.map((id) => conns.get(id)?.name || id).join(", ")} joined after the start; re-deal to give ${spare.length > 1 ? "them" : "it"} layers`);
}

// a device in the chain left: every lap in flight fails now instead of timing out, and the room
// waits for a re-deal
function aiPeerLeft(id, name) {
  if (ai.role !== "host") return;
  if (!ai.chain.includes(id)) { offerRedealForNewcomers(); if (!sparePeers().length && !ai.degraded) showRedeal(false); return; }
  const layers = ai.layersByName?.[name];
  const why = `${name || "a device"} left${layers ? ` (layers ${layers})` : ""}`;
  ai.degraded = true;
  ai.readyPeers.delete(id);
  ai.fed = null; ckptClear();
  failWaiters(new Error(why));
  $("ai-row").style.display = ai.engine ? "flex" : "none";
  aiStatus(`${why}: re-deal the layers to keep going`);
  showRedeal(true, `${why}. Re-deal to split the model over the devices still here; cached layers reload in seconds.`);
  broadcastAll({ t: "ai-degraded", why });
  codeRoleChanged();
}

// a newcomer while the room is online gets the chat as a guest, and the conversation so far
function aiWelcome(id) {
  if (ai.role !== "host" || !ai.engine || ai.readyPeers.size < ai.chain.length || ai.chain.includes(id)) return;
  sendTo(id, { t: "ai-ready-all", model: ai.model });
  if (ai.visibility === "all" && ai.transcript.length) sendTo(id, { t: "ai-history", items: ai.transcript.slice(-20) });
  offerRedealForNewcomers();
}

// a device whose tab got reloaded comes back with a new peer id: put it back in its slot
function aiRejoin(newId, name) {
  if (ai.role !== "host" || !ai.plan?.has(name)) return;
  const i = ai.chainNames.indexOf(name);
  if (i < 0 || ai.chain[i] === newId || ai.chain.includes(newId)) return;
  const oldId = ai.chain[i];
  ai.chain[i] = newId;
  ai.readyPeers.delete(oldId);
  const { msg } = ai.plan.get(name);
  const fresh = { ...msg, next: i + 1 < ai.chain.length ? ai.chain[i + 1] : "host", host: peer.id };
  if (i > 0) sendTo(ai.chain[i - 1], { t: "ai-next", next: newId });
  sendTo(newId, fresh);
  ai.fed = null; ckptClear(true);           // its fresh engine holds nothing: re-prefill next time
  log("room", `${name} came back — reloading its layers`);
  aiStatus(`${name} reconnected, reloading its layers…`);
  $("ai-row").style.display = ai.readyPeers.size >= ai.chain.length ? "flex" : "none";
}
function aiMaybeReady() {
  if (ai.role !== "host" || !ai.engine) return;
  if (ai.readyPeers.size < ai.chain.length) return;
  const n = ai.chain.length + 1;
  ai.degraded = false;
  ai.busy = false;
  showRedeal(false);
  aiStatus(`cluster online · ${n} device${n > 1 ? "s" : ""}, ${ai.cfg.num_hidden_layers} layers split ${n} ways`);
  clearInterval(ai.progTimer);
  $("ai-panel").classList.add("online");
  $("ai-row").style.display = "flex";
  $("chat-tools").hidden = false;
  $("new-chat").hidden = false;
  $("mode-bar").hidden = false;
  setAfterAnswer(false, !!ai.conv.turns.length);
  emptyText("The model is ready. Ask anything.");
  sysNote(`Model ready on ${n} device${n > 1 ? "s" : ""}`);
  $("ai-prompt").focus();
  broadcastAll({ t: "ai-ready-all", model: ai.model });
  pushMap(0, null, false, true);
  offerRedealForNewcomers();
  setTimeout(nextQueued, 0);
  saveHost();
  mascot("Cluster online! Ask anything. Everyone in the room can.");
  codeRoleChanged();
}

// ---- laps ----
// A lap is one frame's trip round the chain. Its waiter resolves with the returned hidden
// state(s), or rejects on timeout or as soon as a device in the chain leaves.
function lapWait(key, ms, what) {
  return new Promise((res, rej) => {
    const timer = setTimeout(() => { ai.waiters.delete(key); rej(new Error(`pipeline timeout (${what})`)); }, ms);
    ai.waiters.set(key, {
      res: (h) => { clearTimeout(timer); res(h); },
      rej: (e) => { clearTimeout(timer); rej(e); },
    });
  });
}
function failWaiters(err) { for (const [k, w] of ai.waiters) { ai.waiters.delete(k); w.rej(err); } }
function lapDone(key, h) { const w = ai.waiters.get(key); if (w) { ai.waiters.delete(key); w.res(h); } }
// send a frame to the first device of the chain; a pending reset or rollback rides with it,
// so it reaches every device strictly before the frame it applies to
function sendChain(msg) {
  ai.frames = (ai.frames || 0) + 1;
  ai.hostAmax = Math.max(0.9 * (ai.hostAmax || 0), wireStats.lastMax || 0);
  const ctl = ai.pendingCtl; ai.pendingCtl = {};
  sendHidden(ai.chain[0], { ...msg, ...ctl });
}
// forget the conversation state on every device: here now, on the chain with the next frame
function resetState() {
  try { ai.engine.reset?.(); } catch {}
  ai.pos = 0;
  ai.fed = [];
  // a pending rollback (the last answer's final verify rejected drafts) must still reach the chain:
  // the save that rides with it records that answer's end state, and without the rollback a
  // worker would save its state with the rejected columns in it (the host saved its rolled-back
  // state), so a later resume of that checkpoint would run the worker's layers on a corrupt state
  const { sv, dp, rb } = ai.pendingCtl || {};
  ai.pendingCtl = ai.chain.length ? { ...(rb != null ? { rb } : {}), ...(sv != null ? { sv } : {}), ...(dp != null ? { dp } : {}), reset: 1 } : {};
}
// ---- checkpoints (?ckpt): the room's state after an answer, saved on every device ----
function ckptClear(tellChain = false) {   // engines rebuilt or in an unknown state: nothing saved is usable
  const keys = ai.ckpt ? ai.ckpt.items.map((x) => x.key) : [];
  for (const k of keys) { try { ai.engine?.dropSlot?.(k); } catch {} }
  if (tellChain && keys.length && ai.chain.length) ai.pendingCtl = { ...ai.pendingCtl, dp: [DROP_ALL] };
  ai.ckpt = new PrefixIndex(1 << 30); ai.ckptN = ai.ckptN || 0;
}
function ckptSave() {
  if (!CKPT_MAX || !ai.fed?.length || !ai.engine?.saveSlot) return;
  if (!ai.ckpt) ckptClear();
  const drop = [];
  while (ai.ckpt.items.length >= CKPT_MAX) {
    const old = ai.ckpt.items.reduce((a, b) => (a.t < b.t ? a : b));
    ai.ckpt.remove(old.key); ai.engine.dropSlot(old.key); drop.push(old.key);
  }
  const key = ai.ckptN = (ai.ckptN || 0) % 65534 + 1;   // slot numbers ride the frame header (u16)
  ai.engine.saveSlot(key);
  ai.ckpt.add(ai.fed.slice(), key);
  if (ai.chain.length) ai.pendingCtl = { ...ai.pendingCtl, sv: key, ...(drop.length ? { dp: drop } : {}) };
}
// resume from the longest checkpoint that is a prefix of ids, if it beats what the caches hold
function ckptResume(ids, reused) {
  if (!CKPT_MAX || !ai.ckpt) return reused;
  const b = ai.ckpt.best(ids);
  if (!b || b.n <= reused) return reused;
  ai.engine.loadSlot(b.key);
  ai.pos = b.n; ai.fed = ids.slice(0, b.n);
  if (ai.chain.length) { const { reset, ...rest } = ai.pendingCtl || {}; ai.pendingCtl = { ...rest, ld: b.key }; }
  return b.n;
}

// Speculative drafting reads the draft block's own KV cache, which prefill fills with the trunk's
// final hidden state at every prompt position. Solo prefill does it inside the engine; in a room
// the returned hidden states come back from the chain, so the host fills it as they arrive
// (roadmap 25: +18–45% tokens per lap after a prompt). Drafts only change speed, never output.
// ?fill=0 turns it off for A/B runs.
const FILL_DRAFTS = new URLSearchParams(location.search).get("fill") !== "0";
const MTP_REFILL = new URLSearchParams(location.search).get("mtprefill") !== "0";
const PRE_DRAFT = new URLSearchParams(location.search).get("predraft") !== "0";
const DRAFT_VOCAB = (() => { const v = new URLSearchParams(location.search).get("draftvocab"); return v === null ? 65536 : parseInt(v, 10) || 0; })();
const MTP_BATCH = new URLSearchParams(location.search).get("mtpbatch") !== "0";   // ?mtpbatch=0: one draft-cache row per submit, for A/B
const GPU_SAMPLE = new URLSearchParams(location.search).get("gpusample") !== "0";   // on by default; see the engine options in aiLoadShard
const ARGMAX_WIDE = (new URLSearchParams(location.search).get("argmaxwide") ?? (GPU_SAMPLE ? "1" : "0")) === "1";
function fillDrafts(h, ids, i0, basePos, n) {
  if (!FILL_DRAFTS || !ai.engine?.mtp) return;
  const dim = ai.engine.dims.dim, E = ai.engine;
  // the whole round in one batched pass: its returned hiddens go into the engine's batch columns
  if (MTP_BATCH && E._mtpFillBatch && E.B && n > 1 && n <= (E.NC || 4)) {
    for (let c = 0; c < n; c++) E.device.queue.writeBuffer(E.B.x.buf, c * E.B.x.stride, h.subarray(c * dim, (c + 1) * dim));
    E._mtpFillBatch(ids, i0, basePos, n);
    return;
  }
  for (let c = 0; c < n; c++) {
    const next = ids[i0 + c + 1];
    if (next === undefined) break;
    ai.engine.setHidden(h.subarray(c * dim, (c + 1) * dim));
    ai.engine.mtpRun(null, next, basePos + c + 1, false);   // no readback: queued, returns at once
  }
}

// run one token through the whole pipeline, returns logits (or null for a prompt token).
// fillNext: the prompt token after this one, to fill the draft cache with this position's hidden.
// desc: GPU sampling descriptor (engine.gpuDescFor(sample)): returns the sampler's candidates
// { ids, vals, bad } instead of the logits (the sampler reads either).
async function aiPipeToken(id, needLogits = true, fillNext, desc = null) {
  const pos = ai.pos;
  if (!ai.chain.length && !needLogits) {
    // solo prefill: layers only, no head, no readback; sync every 8 tokens
    ai.engine.pos = pos;
    await ai.engine.prefillToken(id);
    if (pos % 8 === 7) await ai.device.queue.onSubmittedWorkDone();
    ai.pos++; ai.fed?.push(id);
    return null;
  }
  const tHost = performance.now();
  let h = await ai.engine.embedRun(id, pos);
  if (badF32(h)) throw new Error(`NaN after HOST layers (pos ${pos}) — host GPU kernel issue`);
  if (ai.chain.length) {
    const hostMs = performance.now() - tHost;
    const returned = lapWait(pos, 30000, "token");
    sendChain({ t: "ai-hidden", pos, ...packWire(h) });
    h = await returned;
    if (badF32(h)) throw new Error(`NaN in hidden returned by peers (pos ${pos}) — check peer status lines`);
    noteLap(performance.now() - tHost, hostMs);
    ai.lastHidden = h;
    if (!needLogits && fillNext !== undefined) fillDrafts(h, [id, fillNext], 0, pos, 1);
  } // solo mode: engine holds every layer, embedRun already produced the final hidden
  ai.pos++; ai.fed?.push(id);
  if (!needLogits) return null;   // prefill: skip the head entirely
  if (desc) {
    const c = await ai.engine.headFromHiddenIds(h, desc);
    if (c.bad) throw new Error(`NaN in logits (pos ${ai.pos}) — head/lm_head kernel issue on host`);
    return c;
  }
  const logits = await ai.engine.headFromHidden(h);
  if (badF32(logits)) throw new Error(`NaN in logits (pos ${ai.pos}) — head/lm_head kernel issue on host`);
  return logits;
}

// Prefill `ids` (the part of the conversation the caches do not hold yet) from ai.pos; returns
// the logits after the last one.
const PREFILL_WINDOW = 6;
const LOOKUP = new URLSearchParams(location.search).get("lookup") !== "0";   // ?lookup=0: draft head only, for A/B
const TAIL_FRAME = new URLSearchParams(location.search).get("tail") !== "0";   // ?tail=0: old per-token tail, for A/B   // prefill rounds in flight round the chain at once
// aborted / onStatus: the caller's stop test and status line (roomGenerate passes its own). On
// abort no new round is issued, the rounds in flight are awaited, and it returns null with ai.fed
// and ai.pos matching exactly what the caches hold, so the next request reuses that prefix.
async function aiPrefill(ids, { aborted = () => ai.abort, onStatus = aiStatus, desc = null } = {}) {
  if (!ai.chain.length && ai.engine.prefillTokens && ids.length > 1) {
    // solo: batched prefill, several prompt tokens per GPU pass
    ai.engine.pos = ai.pos;
    await ai.engine.prefillTokens(ids.slice(0, -1));
    ai.pos = ai.engine.pos;
    ai.fed.push(...ids.slice(0, -1));
    return aiPipeToken(ids[ids.length - 1], true, undefined, desc);
  }
  let i = 0;
  // the hybrid engine takes any column count per frame (speculative verifies already send 2..8),
  // so the prompt's tail, last token included, goes round the chain as ONE frame instead of one
  // serial lap per token; short follow-ups become a single lap
  const flex = TAIL_FRAME && !!(ai.chain.length && ai.engine.specStep && ai.engine.embedRunBatch);
  let tailLogits = null;
  if (ai.engine.embedRunBatch && (ids.length > 5 || flex)) {
    // split: up to 16 prompt tokens per round, and several rounds in flight at once. Every device
    // runs frames in send order, so round r+1 can enter the host's layers while round r is on a
    // worker: the chain works like a pipeline instead of one device at a time.
    const hdim = ai.engine.dims.dim;
    const NC = ai.engine.NC || 4;   // columns per GPU pass
    // step down 16 -> 8 -> 4 on the tail: without this a remainder of up to NC-1 tokens costs one
    // network lap each
    const widths = [NC, ...[8, 4].filter((w) => w < NC)];
    const inflight = [];
    try {
      outer: for (const W of widths) while (ids.length - 1 - i >= W) {
        if (aborted()) break outer;
        const nChunks = Math.max(1, Math.min(Math.floor(16 / W), Math.floor((ids.length - 1 - i) / W)));
        const n = nChunks * W, basePos = ai.pos, i0 = i;
        const hb = new Float32Array(n * hdim);
        for (let c = 0; c < nChunks; c++)
          hb.set(await ai.engine.embedRunBatch(ids.slice(i + c * W, i + (c + 1) * W), basePos + c * W), c * W * hdim);
        if (badF32(hb)) throw new Error(`NaN in batched prefill (pos ${basePos})`);
        if (ai.chain.length) {
          while (inflight.length >= PREFILL_WINDOW) await inflight.shift();
          const p = lapWait("b" + basePos, 90000, "batch prefill").then((h) => fillDrafts(h, ids, i0, basePos, n));
          p.catch(() => {});
          inflight.push(p);
          sendChain({ t: "ai-hidden-b", basePos, n, ...packWire(hb) });
        }
        ai.pos = basePos + n;
        ai.fed.push(...ids.slice(i0, i0 + n));
        i += n;
        onStatus(`prefill: ${i}/${ids.length} tokens…`);
      }
      if (flex && !aborted() && i < ids.length) {
        const n = ids.length - i, basePos = ai.pos, i0 = i;   // n <= 4: what the widths above left
        const hb = await ai.engine.embedRunBatch(ids.slice(i), basePos);
        if (badF32(hb)) throw new Error(`NaN in batched prefill (pos ${basePos})`);
        const p = lapWait("b" + basePos, 90000, "prefill tail");
        p.catch(() => {});
        sendChain({ t: "ai-hidden-b", basePos, n, ...packWire(hb) });
        ai.pos = basePos + n;
        ai.fed.push(...ids.slice(i0));
        i = ids.length;
        for (const q of inflight) await q;
        const h = await p;
        if (badF32(h)) throw new Error(`NaN in hidden returned by peers (pos ${basePos})`);
        fillDrafts(h, ids, i0, basePos, n);
        const dim = ai.engine.dims.dim;
        ai.lastHidden = h.slice((n - 1) * dim, n * dim);
        if (desc) {
          tailLogits = await ai.engine.headFromHiddenIds(ai.lastHidden, desc);
          if (tailLogits.bad) throw new Error(`NaN in logits (pos ${ai.pos}) — head/lm_head kernel issue on host`);
        } else tailLogits = await ai.engine.headFromHidden(ai.lastHidden);
        if (!desc && badF32(tailLogits)) throw new Error(`NaN in logits (pos ${ai.pos}) — head/lm_head kernel issue on host`);
      }
      for (const p of inflight) await p;
    } catch (err) { failWaiters(err); throw err; }
  }
  if (aborted()) return null;
  if (tailLogits) return tailLogits;
  let logits = null;
  for (; i < ids.length; i++) {
    if (aborted()) return null;
    logits = await aiPipeToken(ids[i], i === ids.length - 1, ids[i + 1], desc);
  }
  return logits;
}

// ---- telemetry and the room map ----
// Workers report their compute per frame kind (ai-tele); the host times each lap, so what is left
// is the wire. The map shows the chain, what each device holds and how long its part takes.
function noteLap(lapMs, hostMs) {
  const L = ai.lapStat ||= { lap: 0, host: 0, n: 0 };
  L.lap = L.n ? 0.7 * L.lap + 0.3 * lapMs : lapMs;
  L.host = L.n ? 0.7 * L.host + 0.3 * hostMs : hostMs;
  L.n++;
}
function mapNodes(kind = "spec") {
  const nodes = [{ name: myName, layers: ai.layersByName?.[myName] || "", ms: ai.lapStat?.host, host: 1, amax: ai.hostAmax }];
  for (const id of ai.chain) {
    const t = ai.teleBy.get(id) || {};
    const name = conns.get(id)?.name || id;
    nodes.push({ name, layers: ai.layersByName?.[name] || "", ms: t[kind] ?? t.spec ?? t.one, amax: t.amax });
  }
  return nodes;
}
function mapStats(tps, acc) {
  const lap = ai.lapStat?.lap;
  if (!ai.chain.length || !lap) return { tps, acc };
  const gpu = mapNodes().reduce((s, x) => s + (x.ms || 0), 0);
  return { tps, acc, lap: Math.round(lap), gpu: Math.round(gpu), net: Math.max(0, Math.round(lap - gpu)) };
}
let lastMap = null, bestTps = 0;
let smLive = null;
function renderMap(nodes, st, live) {
  const el = $("swarm-map"); if (!el || !nodes?.length) return;
  lastMap = { nodes, st: { ...(lastMap?.st || {}), ...(st || {}) } };
  if (st?.tps && !live) bestTps = Math.max(bestTps, st.tps);
  el.hidden = false;
  el.classList.toggle("live", !!live);
  const lap = Math.max(120, Math.min(4000, st?.lap || 600));
  el.style.setProperty("--lap", lap + "ms");
  el.style.setProperty("--n", nodes.length);
  $("room-screen").style.setProperty("--lap", lap + "ms");
  el.querySelector(".sm-track").innerHTML = nodes.map((x, i) => `<div class="sm-node${x.host ? " host" : ""}" style="--i:${i}">
      <div class="sm-dot"></div><div class="sm-name">${esc(String(x.name))}</div>
      <div class="sm-sub">${x.host ? "embed · " : ""}${x.layers ? "L" + esc(String(x.layers)) : ""}${x.host ? " · head" : ""}</div>
      <div class="sm-ms">${x.ms ? Math.round(x.ms) + " ms" : ""}${x.amax ? ` <span class="sm-amax" title="largest activation this device sent (f16 tops out at 65504)">|x|≤${Math.round(x.amax)}</span>` : ""}</div></div>`).join('<div class="sm-link"><i></i></div>')
    + (nodes.length > 1 ? '<div class="sm-link back"><i></i></div>' : "");
  // the layer band: one lane per device (its name, its layers, its ms), stacked. Every lane spans the
  // whole model (one column per layer, or per few for deep models) and fills the columns its device
  // holds, in its colour; the sweep runs down the staircase. Lanes thin out as devices join.
  const spans = nodes.map((x, i) => { const m = /^(\d+)\D+(\d+)$/.exec(String(x.layers || "")); return m ? { i, name: x.name, lo: +m[1], hi: +m[2] + 1 } : null; }).filter(Boolean);
  const total = spans.reduce((t, x) => Math.max(t, x.hi), 0);
  const strip = el.querySelector(".sm-strip");
  const sig = spans.map((x) => `${x.i}:${x.name}:${x.lo}-${x.hi}`).join(",");
  if (strip.dataset.sig !== sig) {
    strip.dataset.sig = sig;
    const n = Math.min(total, 64), per = total / Math.max(1, n);
    const sorted = spans.sort((x, y) => x.lo - y.lo);
    strip.innerHTML = sorted.map((sp) => {
      let cells = "", c = Math.floor(sp.lo / per);
      const c0 = c;
      for (; c < n && c * per < sp.hi; c++) cells += `<i style="--c:${c};grid-column:${c + 1}"></i>`;
      const dev = [...conns.values()].find((e) => e.name === sp.name)?.meta;
      const icon = iconFor(sp.name === myName ? myMeta : dev || {});
      return `<div class="sm-half" data-name="${esc(String(sp.name))}" style="--sw:${devColor(sp.name)};--c0:${c0}" title="${esc(String(sp.name))}: layers ${sp.lo + 1}–${sp.hi}"><p class="hl"><i class="act" aria-hidden="true"></i>${icon}<b>${esc(String(sp.name))}</b><span class="lr">layers ${sp.lo + 1}–${sp.hi}</span><span class="lms"></span></p><div class="cells">${cells}</div></div>`;
    }).join("");
    strip.style.setProperty("--cells", n);
    el.style.setProperty("--lanes", Math.max(1, sorted.length));
    el.toggleAttribute("data-many", sorted.length > 4);
    // folded: the same split as a row of small blocks
    miniSpans = sorted.map((sp) => ({ name: sp.name, lo: sp.lo, hi: sp.hi })); miniTotal = total;
    paintMini(true);
  }
  // each lane's own time per token (what its GPU spends on its layers)
  for (const lane of strip.querySelectorAll(".sm-half")) {
    const x = nodes.find((y) => String(y.name) === lane.dataset.name);
    lane.querySelector(".lms").textContent = x?.ms ? `${Math.round(x.ms)} ms` : "";
  }
  const agm = $("ag-m");
  if (agm) agm.textContent = `${shortName(ai.model || $("ai-model").value)} on ${nodes.length} device${nodes.length > 1 ? "s" : ""}`;
  el.querySelector(".sm-model").textContent = shortName(ai.model || $("ai-model").value);
  // the device cards say which layers they hold, in the strip's colours
  for (const card of document.querySelectorAll("#peers .peer-card")) {
    const k = nodes.findIndex((x) => x.name === card.dataset.name);
    card.style.setProperty("--sw", devColor(card.dataset.name));
    card.style.setProperty("--k", Math.max(0, k));
    card.classList.toggle("holds", k >= 0 && !!nodes[k].layers);
    card.querySelector(".play").textContent = k >= 0 && nodes[k].layers ? `layers ${humanRange(nodes[k].layers)}` : "";
  }
  const S = lastMap.st;
  $("sm-tps").textContent = S?.tps ? S.tps.toFixed(1) : "-";
  $("sm-lap").textContent = S?.lap ? String(Math.round(S.lap)) : "-";
  (smLive ||= liveWords(el.querySelector(".sm-lt")))(!!live);   // "Twinkling…", "Weaving…" while writing
  el.querySelector(".sm-live").title = live ? "the room is writing an answer" : "waiting for a question";
  const bits = [];
  if (st?.tps) bits.push(`${st.tps.toFixed(1)} tok/s`);
  if (st?.lap) bits.push(DEV ? `lap ${st.lap} ms = GPUs ${st.gpu} + wire ${st.net}` : `${st.lap} ms per word`);
  if (st?.acc != null && DEV) bits.push(`${Math.round(st.acc * 100)}% of drafts accepted`);
  el.querySelector(".sm-meta").textContent = bits.join(" · ") || `${nodes.length} device${nodes.length > 1 ? "s" : ""}`;
  el.querySelector(".sm-meta").title = `${nodes.length} device${nodes.length > 1 ? "s" : ""}: every token takes a lap through all of them`;
  deviceMark();
}
// The folded band's split: one segment per device, as wide as its share of the layers (at least a few
// px, so a phone holding one or two layers never drops out of the bar), in the device's colour, with a
// thin tick between its layers when they are wide enough to show. Redrawn when the bar changes width.
let miniSpans = [], miniTotal = 0, miniW = -1;
function paintMini(force = false) {
  const mini = $("swarm-map").querySelector(".sm-mini");
  const w = mini.clientWidth;
  if (!w || (!force && w === miniW)) return;
  miniW = w;
  if (!miniTotal) { mini.innerHTML = ""; return; }
  // the bar in order: each device's span, and any layers not dealt yet
  const parts = [];
  let at = 0;
  miniSpans.map((sp, k) => ({ sp, k })).sort((a, b) => a.sp.lo - b.sp.lo).forEach(({ sp, k }) => {
    if (sp.lo > at) parts.push({ lo: at, hi: sp.lo });
    if (sp.hi > Math.max(sp.lo, at)) parts.push({ lo: Math.max(sp.lo, at), hi: sp.hi, sp, k });
    at = Math.max(at, sp.hi);
  });
  if (at < miniTotal) parts.push({ lo: at, hi: miniTotal });
  const room = w - 2 * (parts.length - 1);   // the bar less the gaps between segments
  mini.innerHTML = parts.map((p) => {
    const n = p.hi - p.lo, px = room * n / miniTotal;
    const range = n > 1 ? `layers ${p.lo + 1}\u2013${p.hi}` : `layer ${p.lo + 1}`;
    // a tick between layers when each layer gets at least 5 px
    const ticks = n > 1 && px / n >= 5 ? ` mini-ticks" style="--n:${n};` : `" style="`;
    return p.sp
      ? `<i class="seg${ticks}flex-grow:${n};--sw:${devColor(p.sp.name)};--k:${p.k}" title="${esc(String(p.sp.name))}: ${range}"></i>`
      : `<i class="seg none${ticks}flex-grow:${n}" title="${range}: not dealt"></i>`;
  }).join("");
  mini.title = miniSpans.map((sp) => `${sp.name}: layers ${sp.lo + 1}\u2013${sp.hi}`).join(" \u00b7 ");
}
if (typeof ResizeObserver === "function") new ResizeObserver(() => paintMini()).observe($("swarm-map").querySelector(".sm-mini"));
// The band folds to one line (the model, its state, a thin bar of the split). Each viewer's choice is
// kept in this browser, separately for Chat and Code.
const bandMode = () => ($("chatpane").classList.contains("code-mode") ? "code" : "chat");
function bandFolded() {
  let v = null;
  try { v = localStorage.getItem("pooled-band-" + bandMode()); } catch {}
  return v ? v === "folded" : bandMode() === "code" || innerWidth < 820;   // Code, and phones, start with it folded
}
function bandFold(on, save = false) {
  const el = $("swarm-map"), b = $("band-toggle");
  el.classList.toggle("folded", on);
  b.setAttribute("aria-expanded", String(!on));
  const t = on ? "Show the layers" : "Hide the layers";
  b.setAttribute("aria-label", t); b.dataset.tip = t;
  if (save) try { localStorage.setItem("pooled-band-" + bandMode(), on ? "folded" : "open"); } catch {}
}
$("band-toggle").addEventListener("click", () => bandFold(!$("swarm-map").classList.contains("folded"), true));
bandFold(bandFolded());
new MutationObserver(() => bandFold(bandFolded())).observe($("chatpane"), { attributes: true, attributeFilter: ["class"] });
// Chat | Code sits in the room bar at 820px and wider (the same node, moved), in its own strip on phones
{
  const home = document.querySelector(".mode-row"), wide = matchMedia("(min-width: 821px)");
  const place = () => { const b = $("mode-bar"); if (wide.matches) { if (b.parentNode !== $("room-badge").parentNode) $("room-badge").after(b); } else if (b.parentNode !== home) home.append(b); };
  place(); wide.addEventListener("change", place);
}
// phones: the chat stays on the newest message when the screen shrinks (the keyboard opens), if
// the reader was at the bottom; while typing, the header chips, the band and Chat | Code step aside
{
  const out = $("ai-output");
  let atEnd = true;
  out.addEventListener("scroll", () => { atEnd = out.scrollHeight - out.scrollTop - out.clientHeight < 40; }, { passive: true });
  const stick = () => { if (atEnd) out.scrollTop = out.scrollHeight; };
  new ResizeObserver(stick).observe(out);
  const coarse = matchMedia("(pointer: coarse)");
  const kbd = () => {
    const a = document.activeElement, typing = a && (a.id === "ai-prompt" || a.id === "code-prompt" || a.id === "ed-text");
    document.body.classList.toggle("kbd", !!(typing && coarse.matches && (visualViewport?.height ?? innerHeight) < 600));
    // how much of the page the keyboard covers where the browser does not shrink the page for it
    // (iOS Safari): Code on a phone lifts its prompt by that much
    const vv = visualViewport, kb = vv && coarse.matches && typing ? Math.max(0, Math.round(innerHeight - vv.height - vv.offsetTop)) : 0;
    document.documentElement.style.setProperty("--kb", kb + "px");
    stick();
  };
  visualViewport?.addEventListener("resize", kbd);
  visualViewport?.addEventListener("scroll", kbd);
  document.addEventListener("focusin", kbd);
  document.addEventListener("focusout", () => setTimeout(kbd, 0));
}
// a token came out: a sweep runs along the layer strip and through the device cards. At most one
// sweep per lap; tokens that come faster ride along with the one running.
let pulseAt = 0, bandTokens = 0;
function mapPulse() {
  bandTokens++;
  const tk = $("sm-tok"); if (tk) tk.textContent = bandTokens.toLocaleString("en-US");
  const now = performance.now(), rs = $("room-screen");
  const lap = parseFloat(rs.style.getPropertyValue("--lap")) || 600;
  if (now - pulseAt < Math.min(lap, 900) || document.hidden || compute.isOpen) return;
  pulseAt = now;
  rs.classList.remove("sweep");
  requestAnimationFrame(() => rs.classList.add("sweep"));
}
// a quiet line in the chat: the model is ready, someone joined
function sysNote(text, kind = "") {
  const o = $("ai-output"); if (!o) return;
  const n = document.createElement("div");
  n.className = "sys" + (kind ? " " + kind : "");
  n.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" aria-hidden="true"><circle cx="3.4" cy="3.4" r="1.8"/><circle cx="10.2" cy="3.4" r="1.99"/><circle cx="18.5" cy="3.4" r="2.38"/><circle cx="3.4" cy="10.2" r="1.99"/><circle cx="10.2" cy="10.2" r="2.38"/><circle cx="18.5" cy="10.2" r="2.94"/><circle cx="3.4" cy="18.5" r="2.38"/><circle cx="10.2" cy="18.5" r="2.94"/><circle cx="18.5" cy="18.5" r="3.9"/></svg><span></span>';
  n.querySelector("span").textContent = text;
  o.appendChild(n);
  if (o.style.display === "block") scrollChat();
}
let mapAt = 0;
function pushMap(tps, acc, live, force) {
  const now = performance.now();
  if (!force && now - mapAt < 800) return;
  mapAt = now;
  const nodes = mapNodes(), st = mapStats(tps, acc);
  renderMap(nodes, st, live);
  broadcastAll({ t: "ai-map", nodes, st, live: live ? 1 : 0 });
}
// ms per layer for every device in the chain, from this answer's verify laps (host: its own share
// of each lap; workers: their ai-tele reports); the speed split deals by these
function noteSpeeds() {
  const put = (name, ms, n) => { if (ms > 0 && n > 0) { const v = ms / n, o = ai.msPerLayer.get(name); ai.msPerLayer.set(name, o ? 0.5 * o + 0.5 * v : v); } };
  put(myName, ai.lapStat?.host, ai.layersN?.[myName]);
  for (const id of ai.chain) { const name = conns.get(id)?.name || id; put(name, ai.teleBy.get(id)?.spec, ai.layersN?.[name]); }
}
// worker: EMA of compute ms per frame kind, reported to the host at most every 700 ms
function teleNote(kind, ms) {
  const T = ai.tele ||= { at: 0, k: {} };
  T.amax = Math.max(T.amax || 0, wireStats.lastMax || 0);
  const k = T.k[kind] ||= { ema: ms, n: 0 };
  k.ema = k.n ? 0.7 * k.ema + 0.3 * ms : ms; k.n++;
  const now = performance.now();
  if (now - T.at > 700 && ai.hostId) {
    T.at = now;
    sendTo(ai.hostId, { t: "ai-tele", k: Object.fromEntries(Object.entries(T.k).map(([a, b]) => [a, Math.round(b.ema * 10) / 10])), amax: Math.round(T.amax * 10) / 10 });
    T.amax = 0;
  }
}

// who sees the chat: the host's dropdown. The full message goes to the screens allowed to see
// the text, the hidden stand-in (same type, `hidden: true`) to the others, so every screen still
// locks and unlocks its Send box with the answer.
function sendChat(msg, askerId) {
  const { full, hidden } = chatRecipients(ai.visibility || "all", askerId, [...conns.keys()]);
  for (const id of full) sendTo(id, msg);
  if (msg.t !== "ai-token") for (const id of hidden) sendTo(id, { t: msg.t, name: msg.name, stats: msg.stats, asker: msg.asker, ctx: msg.ctx, hidden: true });
}

// answer length the host picks; ?maxnew=N overrides it (tests)
const ANSWER_LEN = { short: 150, normal: MAX_NEW, long: 1200 };
const MAXNEW_PARAM = Math.max(0, parseInt(new URLSearchParams(location.search).get("maxnew"), 10) || 0);

// The room's generation core, shared by the chat (aiGenerate) and Code mode (roomApi.generate):
// prefill ids, reusing whatever the caches or a checkpoint already hold, then decode until a stop
// token, maxNew, a full context or an abort. UI-free; host only; the caller holds the lock
// (ai.busy). On failure it leaves the room clean (the next call resets) and rethrows.
//   onToken(id, drafted)  per emitted token, in order (drafted: 0 sampled, 1 draft head, 2 lookup)
//   stop                  Set of ids that end the answer (not emitted, not piped round the chain)
//   sample(logits) -> id  a wrapper may mask logits first (the tool-name constraint)
//   signal                AbortSignal; ai.abort (the Stop button) works too
// -> { tokens, reason: "stop"|"max"|"ctx"|"abort", reused, prefilled, count, tps, acc, copied,
//      tPre, tDecode, preFrames, stats }
async function roomGenerate(ids, { onToken = () => {}, stop, maxNew = MAX_NEW, sample = pickSampler(ai.settings.sampling), signal, onStatus = () => {} } = {}) {
  if (!ai.engine) throw new Error("the model is not loaded");
  // a device in the chain is gone: its frames would go nowhere and wait out the lap timeouts
  if (ai.degraded) throw new Error("a device left: re-deal the layers first");
  const aborted = () => ai.abort || !!signal?.aborted;
  const eos = (t) => stop.has(t);
  const tokens = [];
  let count = 0, capped = false, acc = null, copied = 0, first = null;
  let tPre = 0, tDecode = 0, reused = 0, prefilled = 0, preFrames = 0;
  // GPU sampling (?gpusample=1): the head's top-k / argmax runs on the GPU and "logits" below are
  // the sampler's candidates. Only for a sampler that says it reads them (.gpu); a wrapper that
  // masks logits has no .gpu and keeps the full-logits path. specStep checks the same itself.
  const desc = ai.engine.gpuDescFor?.(sample) || null;
  try {
    reused = ckptResume(ids, reusablePrefix(ai.fed, ids));
    // Continue after a cap that landed on a written token: the caches hold the whole open answer,
    // so there is nothing to prefill, and the next token was already sampled when it stopped
    const pend = ai.pending; ai.pending = null;
    if (!reused && pend && pend.at === ai.pos && ai.fed?.length === ids.length && ai.fed.every((t, i) => t === ids[i])) {
      reused = ids.length; first = pend.next;
    }
    if (!reused) resetState();
    const rest = ids.slice(reused);
    // a follow-up's first token needs a draft-cache row too: the trunk hidden at the position
    // before it is still in the engine when the last answer ended on a speculative step
    if (reused && rest.length && FILL_DRAFTS && ai.engine.mtp && ai.xAt === ai.pos) ai.engine.mtpRun(null, rest[0], ai.pos, false);
    ai.xAt = null;
    prefilled = rest.length;
    maxNew = Math.min(maxNew, ctxMax() - ids.length);
    onStatus(reused ? `prefill: ${rest.length} new tokens (${reused} already in the room's caches)…` : `prefill: ${rest.length} tokens…`);
    const t0Pre = performance.now();
    ai.frames = 0;
    let logits = rest.length ? await aiPrefill(rest, { aborted, onStatus, desc }) : null;
    tPre = performance.now() - t0Pre;
    if (prefilled) compute.pass(prefilled);
    preFrames = ai.frames;

    const t0 = performance.now();
    const emit = (tok, drafted) => {
      tokens.push(tok);
      count++;
      onToken(tok, drafted);
      mapPulse(); compute.pass(1);
      const tps = count / ((performance.now() - t0) / 1000);
      onStatus(`generating… ${count} tok · ${tps.toFixed(1)} tok/s`);
    };
    if (!logits && first == null) { /* stopped during prefill */ }
    else if (ai.engine.mtp && ai.engine.specStep) {
      // speculative decoding: the model's own draft head proposes up to K tokens,
      // one batched trunk pass verifies them (byte-identical to plain decoding)
      const spec = ai.chain.length ? {
        runTrunk: async (tokens, pos) => {
          const tLap = performance.now();
          const n = tokens.length, hdim = ai.engine.dims.dim, NC = ai.engine.NC || 4;
          const hb = new Float32Array(n * hdim);
          for (let c = 0; c < n; c += NC) {
            const m = Math.min(NC, n - c);
            hb.set(await ai.engine.embedRunBatch(tokens.slice(c, c + m), pos + c, { base: c, total: n }), c * hdim);
          }
          if (badF32(hb)) throw new Error(`NaN after HOST layers (pos ${pos})`);
          const hostMs = performance.now() - tLap;
          const returned = lapWait("b" + pos, 90000, "verify");
          sendChain({ t: "ai-hidden-b", basePos: pos, n: tokens.length, spec: 1, ...packWire(hb) });
          const h = await returned;
          if (badF32(h)) throw new Error(`NaN in hidden returned by peers (pos ${pos})`);
          noteLap(performance.now() - tLap, hostMs);
          return h;
        },
        // the rollback rides on the next frame (sendChain), strictly before it on every device
        onReject: async (k) => { ai.pendingCtl = { rb: k }; },
      } : {};
      if (ai.chain.length && ai.lastHidden && first == null) ai.engine.setHidden(ai.lastHidden);
      ai.engine.pos = ai.pos;
      // draft depth: pick by MEASURED tokens/sec per depth (K=3 warm-up, probe
      // 5 and 7 once, keep the best, re-probe now and then). Deep chains only
      // pay when the network round-trip dominates the lap; a lap-time
      // threshold can't tell GPU time from RTT and gets stuck deep.
      const kc = { cand: [3, 5, 7], ema: {}, n: {}, step: 0, used: {} };
      const pickK = () => {
        if (!ai.chain.length) return 3;
        kc.step++;
        if (kc.step <= 3) return 3;
        const untried = kc.cand.find((k) => !kc.n[k]);
        if (untried) return untried;
        let best = 3;
        for (const k of kc.cand) if (kc.ema[k] > kc.ema[best]) best = k;
        if (kc.step % 16 === 0) { const alt = kc.cand.filter((k) => k !== best); return alt[(kc.step / 16) % alt.length | 0]; }
        return best;
      };
      const st0 = { ...ai.engine.mtp.stats };
      // the first answer token is sampled here; specStep treats it as already chosen for this
      // position and returns only the tokens after it, so it has to be emitted (or end the
      // answer) before the loop, or the reply starts one word late
      let next = first ?? sample(logits), done = false, pendTok = null;
      if (eos(next)) done = true; else emit(next, false);
      while (!done && count < maxNew && !aborted()) {
        // a speculative step touches positions pos .. pos+K (K drafts verified in one pass) and
        // drafts one more; shrink K near the end of the context and stop before it overflows
        let K = pickK();
        const roomLeft = ctxMax() - ai.engine.pos - 2;
        if (roomLeft < 1) { capped = true; break; }
        // never draft past the answer cap: every token a step writes into the caches is then an
        // emitted one, so a capped answer is still a prefix of the next turn and nothing re-prefills
        K = Math.min(K, roomLeft, maxNew - count);
        const tStep = performance.now();
        // prompt lookup first: if the text is repeating something in the context, verify what
        // followed it last time (free to guess, up to 7 at once); otherwise the draft head
        // a lookup run that was accepted in full is probably a copy in progress (code being edited,
        // a file quoted back): let the next one run long, up to what one verify can take
        const lkMax = ai.lkFull ? (ai.engine.maxDrafts || 7) : 7;
        const lk = LOOKUP && ai.engine.specStepDrafts && ai.fed ? lookupDrafts([...ai.fed, next], Math.min(lkMax, roomLeft, maxNew - count)) : [];
        const viaLookup = lk.length >= 2;
        const toks = viaLookup ? await ai.engine.specStepDrafts(next, sample, lk, spec) : await ai.engine.specStep(next, sample, K, spec);
        if (viaLookup) copied += toks.length - 1;
        ai.lkFull = viaLookup && toks.length === lk.length + 1;
        // specStep wrote `next` and the accepted drafts; its last token is the next `next`
        ai.fed.push(next, ...toks.slice(0, -1));
        const tps = toks.length / ((performance.now() - tStep) / 1000);
        if (!viaLookup) {
          kc.ema[K] = kc.n[K] ? 0.6 * kc.ema[K] + 0.4 * tps : tps;
          kc.n[K] = (kc.n[K] || 0) + 1; kc.used[K] = (kc.used[K] || 0) + toks.length;
        }
        for (let j = 0; j < toks.length; j++) {
          const tk = toks[j];
          if (eos(tk)) { done = true; break; }
          if (count >= maxNew) { done = true; capped = true; if (j === toks.length - 1) pendTok = tk; break; }
          emit(tk, j < toks.length - 1 ? (viaLookup ? 2 : 1) : 0);   // all but the last were drafts the trunk accepted (2: from lookup)
        }
        next = toks[toks.length - 1];
        const d = ai.engine.mtp.stats.drafts - st0.drafts;
        acc = d ? (ai.engine.mtp.stats.accepted - st0.accepted) / d : null;
        if (ai.chain.length) pushMap(count / ((performance.now() - t0) / 1000), acc, true);
      }
      if (!done && count >= maxNew) capped = true;
      ai.pos = ai.engine.pos;
      ai.xAt = ai.pos;   // specStep left the trunk hidden at ai.pos - 1 in the engine
      if (pendTok != null && !aborted()) ai.pending = { next: pendTok, at: ai.pos };
      const st = ai.engine.mtp.stats;
      if (st.drafts) crumb(`spec: ${st.accepted}/${st.drafts} drafts accepted${ai.lapStat ? ` · lap ${Math.round(ai.lapStat.lap)}ms` : ""}`
        + (ai.chain.length ? ` · K tok/s ${kc.cand.map((k) => `${k}:${kc.ema[k] ? kc.ema[k].toFixed(1) : "-"}`).join(" ")} · tokens by K ${JSON.stringify(kc.used)}` : ""));
    } else {
      // plain decoding. An end token is not piped through the chain: the next turn's template
      // writes <|im_end|> itself, so both paths leave the caches holding exactly prompt + answer
      for (let i = 0; i < maxNew && !aborted(); i++) {
        const next = i === 0 && first != null ? first : sample(logits);
        if (eos(next)) break;
        emit(next, false);
        if (ai.pos >= ctxMax() - 1) { capped = true; break; }   // no position left for another token
        logits = await aiPipeToken(next, true, undefined, desc);
        if (ai.chain.length) pushMap(count / ((performance.now() - t0) / 1000), null, true);
      }
      if (count >= maxNew) {
        capped = true;
        // for Continue: the chosen id (sample reads logits or GPU candidates alike)
        if (logits && !aborted() && ai.pos < ctxMax() - 1) ai.pending = { next: sample(logits), at: ai.pos };
      }
    }
    tDecode = performance.now() - t0;
  } catch (err) {
    ai.fed = null;            // the caches are in an unknown state: the next request starts clean
    ai.pendingCtl = {};
    ckptClear(true);
    throw err;
  }
  const secs = tDecode / 1000, tps = count / Math.max(secs, 1e-3);
  const full = capped && ai.pos >= ctxMax() - 2;
  const stats = `${count} tok · ${tps.toFixed(1)} tok/s · ${ai.chain.length + 1} device${ai.chain.length ? "s" : ""}`
    + (acc != null ? ` · ${Math.round(acc * 100)}% drafts accepted` : "")
    + (copied ? ` · ${copied} tok by lookup` : "")
    + (aborted() ? " · stopped" : "")
    + (capped ? (full ? ` · stopped: context full (${ctxMax()} tokens)` : ` · stopped at ${count} tokens`) : "");
  if (ai.chain.length && count) { pushMap(tps, acc, false, true); noteSpeeds(); }
  else if (count > 8) lastSoloTps = Math.max(lastSoloTps, tps);
  ckptSave();   // this answer's end state, on every device, for a later regenerate or branch
  const reason = aborted() ? "abort" : capped ? (full ? "ctx" : "max") : "stop";
  return { tokens, reason, reused, prefilled, count, tps, acc, copied, tPre, tDecode, preFrames, stats, capped };
}

// mode: "ask" a new question, or "continue" the last answer (it stopped at the length cap)
async function aiGenerate(textArg, who, askerId = peer.id, mode = "ask") {
  const cont = mode === "continue";
  const lastTurn = ai.conv.turns[ai.conv.turns.length - 1];
  if (cont && lastTurn?.role !== "assistant") return;
  const text = cont ? "(continue)" : (textArg ?? $("ai-prompt").value).trim();
  const asker = who || myName;
  if (!text || ai.busy || !ai.engine) return;
  if (ai.degraded) {
    if (askerId === peer.id) toast("a device left: re-deal the layers first");
    else sendTo(askerId, { t: "ai-busy", why: "a device left the room; the host has to re-deal the layers first" });
    return;
  }
  ai.busy = "gen";
  ai.abort = false;
  ai.askerId = askerId;
  ai.lastAsker = askerId;
  setBusyUI(true, true);
  const S = specials(ai.tok);
  const persona = PERSONAS[ai.settings.persona] || PERSONAS.default;
  const thinking = !!ai.settings.thinking && S.think !== undefined;
  const sample = pickSampler(ai.settings.sampling);
  const stop = new Set([S.imEnd, S.eot]);

  setAfterAnswer(false, false);
  if (!cont) chatUser(asker, text);
  const mid = ai.msgSeq = (ai.msgSeq || 0) + 1;
  chatBotStart(mid);
  sendChat({ t: "ai-genstart", name: asker, text, asker: askerId, cont: cont ? 1 : 0, mid }, askerId);
  mascot("Thinking… every word is taking a lap through the room.");

  const answer = [];          // sampled ids of this answer, verbatim, for the next turn's history
  let reply = "", failed = null, stats = "", capped = false, dropped = 0, r = null, inGen = false;
  try {
    // the conversation with this question, trimmed to fit
    const fit = fitContext(ai.tok, { system: persona.system, turns: cont ? [...ai.conv.turns.slice(0, -1), { ...lastTurn, open: true }] : [...ai.conv.turns, { role: "user", text, name: asker }], thinking }, ctxMax(), MIN_ROOM);
    dropped = fit.dropped;
    ai.conv.turns = fit.turns;
    const cap = thinking ? MAX_NEW_THINKING : (ANSWER_LEN[ai.settings.length] ?? MAX_NEW);
    const onToken = (tok, drafted) => {
      const piece = ai.tok.decode([tok]);
      answer.push(tok);
      reply += piece;
      chatBotPiece(piece, drafted);
      sendChat({ t: "ai-token", text: piece, d: drafted || 0 }, askerId);
    };
    inGen = true;   // from here roomGenerate cleans up after itself on failure
    r = await roomGenerate(fit.ids, { onToken, stop, maxNew: MAXNEW_PARAM || cap, sample, onStatus: aiStatus });
    capped = r.capped;
    stats = r.stats + (dropped ? ` · ${dropped} oldest exchange${dropped > 1 ? "s" : ""} forgotten to fit` : "");
  } catch (err) {
    failed = err;
    if (!inGen) { ai.fed = null; ai.pendingCtl = {}; ckptClear(true); }
    stats = "failed: " + err.message;
    aiStatus("generation failed: " + err.message);
  }
  // the answer (even a partial one) joins the history, so the next turn reads what was said
  const tail = ai.conv.turns[ai.conv.turns.length - 1];
  if (tail?.role === "user") ai.conv.turns.push({ role: "assistant", ids: answer });
  else if (tail?.open) { tail.ids = [...tail.ids, ...answer]; delete tail.open; }
  const ctx = { used: ai.fed ? ai.pos : 0, max: ctxMax() };
  if (failed) chatBotEnd(reply ? null : "⚠ " + failed.message, stats);
  else chatBotEnd(null, stats);
  const canContinue = capped && !failed && !ai.abort;
  sendChat({ t: "ai-gendone", stats, ctx, failed: failed ? 1 : 0, capped: canContinue ? 1 : 0 }, askerId);   // unlocks every send box
  setAfterAnswer(canContinue, !failed);
  if (cont && ai.transcript.length) { const t = ai.transcript[ai.transcript.length - 1]; t.reply += reply; t.stats = stats; }
  else ai.transcript.push({ name: asker, text, reply, stats, mid });
  if (ai.transcript.length > 50) ai.transcript.shift();
  setCtx(ctx.used, ctx.max);
  saveHost();
  if (!failed) aiStatus(`ready — prefill ${r.prefilled} tok in ${(r.tPre / 1000).toFixed(1)}s${ai.chain.length ? ` / ${r.preFrames} frame${r.preFrames === 1 ? "" : "s"}` : ""}${r.reused ? ` (${r.reused} reused)` : ""}, ${stats}`);
  mascot("Done. Anyone in the room can ask the next one.");
  ai.busy = false;
  ai.abort = false;
  setBusyUI(false);
  setTimeout(nextQueued, 0);
  if (ai.degraded) showRedeal(true);
}

// Continue / Regenerate the last answer: the host, or whoever asked it. Regenerate drops the last
// exchange from the conversation and asks it again; the caches no longer match, so it
// re-prefills (with "exact" sampling it gives the same answer, which is the point of exact).
function setAfterAnswer(canContinue, ok) {
  $("continue-btn").hidden = !canContinue;
  $("regen-btn").hidden = !ok;
}
function aiCommand(cmd, from) {
  if (ai.role !== "host") return;
  if (ai.busy || !ai.engine || ai.degraded || ai.readyPeers.size < ai.chain.length) {
    const why = ai.degraded ? "a device left: re-deal the layers first" : ai.busy === "code" ? "the host's agent is working, try again when it is done" : "the room is busy, try again in a moment";
    if (from === peer.id) toast(why); else sendTo(from, { t: "ai-busy", why });
    return;
  }
  const byAsker = from === ai.lastAsker || from === peer.id;
  if (!byAsker) { sendTo(from, { t: "ai-busy", why: "only the host or whoever asked can do that" }); return; }
  const turns = ai.conv.turns;
  if (cmd === "continue") { aiGenerate(null, from === peer.id ? myName : conns.get(from)?.name, from, "continue"); return; }
  if (cmd === "regen" && turns.length >= 2 && turns[turns.length - 1].role === "assistant") {
    const q = turns[turns.length - 2];
    ai.conv.turns = turns.slice(0, -2);
    ai.transcript.pop();
    broadcastAll({ t: "ai-regen" });
    markReplaced();
    aiGenerate(q.text, q.name || (from === peer.id ? myName : conns.get(from)?.name), from);
  }
}
function markReplaced() {
  const ms = [...document.querySelectorAll("#ai-output .m")];
  for (const m of ms.slice(-2)) m.classList.add("replaced");
}

// Start a new conversation: the next question prefills from scratch on every device.
function aiNewChat() {
  if (ai.role !== "host" || ai.busy === "gen" || ai.busy === "code") return;
  ai.conv = { turns: [] };
  ai.fed = null;
  ai.transcript = [];
  clearChat();
  setAfterAnswer(false, false);
  broadcastAll({ t: "ai-reset", by: myName });
  setCtx(0);
  toast("new chat: the room forgot the conversation");
  saveHost();
}
function clearChat() {
  $("ai-output").innerHTML = "";
  botEl = null;
  setCtx(0);
}

// ---- worker ----
// Frames run strictly one after another in arrival order (the transport delivers them in send
// order), so several prefill rounds can be queued here while the GPU works. Control that rides
// on a frame (reset, rollback) applies before it, and goes on down the chain with it.
async function workerFrame(d) {
  if (!ai.engine) return;
  const ctl = {};
  // order matters: a pending rollback belongs to the answer that just ended, the save records
  // that answer's final state, and only then may the state be reset or replaced by a checkpoint
  // (also before a reset: the save may record the state first, and the host saved its own after
  // its rollback)
  if (d.rb != null) { ai.engine.restoreDN?.(d.rb); ctl.rb = d.rb; }
  if (d.sv != null) { ai.engine.saveSlot?.(d.sv); ctl.sv = d.sv; }
  if (d.dp != null) { for (const k of [].concat(d.dp)) k === DROP_ALL ? ai.engine.dropAllSlots?.() : ai.engine.dropSlot?.(k); ctl.dp = d.dp; }
  if (d.reset) { ai.engine.reset?.(); ctl.reset = 1; }
  if (d.ld != null) { ai.engine.loadSlot?.(d.ld); ctl.ld = d.ld; }
  const t0 = performance.now();
  if (d.t === "ai-hidden-b") {
    // n hiddens in, my layers (batched), n hiddens on
    const xs = unpackWire(d);
    const nTok = d.n || 4;
    const wdim = ai.engine.dims.dim;
    const hb = new Float32Array(nTok * wdim);
    const NC = ai.engine.NC || 4;
    for (let c = 0; c < nTok; c += NC) {
      const m = Math.min(NC, nTok - c);
      hb.set(await ai.engine.runHiddenBatch(xs.subarray(c * wdim, (c + m) * wdim), d.basePos + c, d.spec ? { base: c, total: nTok } : false), c * wdim);
    }
    if (badF32(hb)) { aiStatus(`⚠ NaN in batched prefill on this device`); sendTo(ai.hostId, { t: "ai-error", message: "NaN in batched prefill" }); }
    teleNote(d.spec ? "spec" : "pre", performance.now() - t0);
    compute.pass(nTok, performance.now() - t0);
    // the verify flag travels with the frame: every device snapshots its recurrent state per
    // column, or a later rollback on it restores a stale snapshot
    const bmsg = { basePos: d.basePos, n: nTok, ...(d.spec ? { spec: 1 } : {}), ...packWire(hb) };
    if (ai.next === "host") sendHidden(ai.hostId, { t: "ai-hiddenret-b", ...bmsg });
    else sendHidden(ai.next, { t: "ai-hidden-b", ...bmsg, ...ctl });
  } else {
    // one token: run my layers, forward along the chain
    const hin = unpackWire(d);
    if (badF32(hin)) { aiStatus(`⚠ NaN ARRIVED at this device (pos ${d.pos}) — upstream peer broken`); }
    const h = await ai.engine.runHidden(hin, d.pos);
    if (badF32(h)) { aiStatus(`⚠ NaN PRODUCED by this device (pos ${d.pos}, layers ${ai.range[0]}–${ai.range[1] - 1}) — GPU kernel issue here`); sendTo(ai.hostId, { t: "ai-error", message: `NaN produced on worker layers ${ai.range[0]}–${ai.range[1] - 1}` }); }
    teleNote("one", performance.now() - t0);
    compute.pass(1, performance.now() - t0);
    const msg = { pos: d.pos, ...packWire(h) };
    if (ai.next === "host") sendHidden(ai.hostId, { t: "ai-hiddenret", ...msg });
    else sendHidden(ai.next, { t: "ai-hidden", ...msg, ...ctl });
    if (d.pos % 8 === 0) aiStatus(`serving layers ${ai.range[0]}–${ai.range[1] - 1} — pos ${d.pos}`);
  }
}

// the host's tab closed: the room is over for everyone else
// A host tab that reloads can resume the room (it keeps the conversation in localStorage), so the
// others wait a minute and keep knocking before calling the room over.
const HOST_WAIT_MS = 60000;
function hostGone() {
  if (ai.role === "host") return;
  failWaiters(new Error("the host left"));
  codeRoleChanged();
  $("ai-row").style.display = "none";
  $("room-over").hidden = false;
  $("room-over-why").textContent = "The host's tab closed. Waiting a minute in case it comes back…";
  aiStatus("the host left; waiting for it to come back…");
  const t0 = Date.now();
  clearInterval(hostGone.timer);
  hostGone.timer = setInterval(() => {
    if (conns.has(PREFIX + roomCode)) { clearInterval(hostGone.timer); return; }
    if (Date.now() - t0 > HOST_WAIT_MS) {
      clearInterval(hostGone.timer);
      ai.engine = null;
      $("room-over-why").textContent = "The host didn't come back. The host holds the conversation and the model's first and last layers, so this room can't answer any more.";
      aiStatus("the host left; this room is over");
      mascot("The host left. Start a new room?");
      return;
    }
    const conn = peer.connect(PREFIX + roomCode, { reliable: true });
    conn.on("open", () => {
      if (conns.has(PREFIX + roomCode)) { try { conn.close(); } catch {} return; }
      clearInterval(hostGone.timer);
      wire(conn, "host", undefined, true);
      conn.send({ t: "hello", name: myName, meta: myMeta, v: PROTOCOL, back: 1 });
      ai.hostId = PREFIX + roomCode;
      $("room-over").hidden = true;
      aiStatus("the host is back; waiting for it to deal the layers…");
      toast("the host is back");
    });
    conn.on("error", () => {});
  }, 3000);
}

// ---- the host's side of resuming: what it keeps, and picking the room back up after a reload ----
function saveHost() {
  if (!isHost || !roomCode) return;   // from the moment the room exists, not only once a model runs
  try {
    localStorage.setItem(HOST_KEY, JSON.stringify({ code: roomCode, name: myName, model: ai.model || null, turns: ai.conv.turns,
      transcript: ai.transcript.slice(-20), settings: ai.settings, peers: ai.chainNames || [], split: $("ai-split").value, t: Date.now() }));
  } catch {}
}
addEventListener("pagehide", saveHost);   // stamp the saved room as the tab unloads, so a reload can go straight back in
function savedHost() {
  try { const r = JSON.parse(localStorage.getItem(HOST_KEY) || localStorage.getItem(OLD_HOST_KEY) || "null"); return r && Date.now() - r.t < 15 * 60 * 1000 ? r : null; } catch { return null; }
}
function resumeHost(r) {
  ai.conv = { turns: Array.isArray(r.turns) ? r.turns : [] };
  ai.transcript = Array.isArray(r.transcript) ? r.transcript : [];
  if (r.settings) {
    ai.settings = { ...ai.settings, ...r.settings };
    for (const [id, k] of [["ai-persona", "persona"], ["ai-sampling", "sampling"], ["ai-length", "length"]]) if (ai.settings[k]) $(id).value = ai.settings[k];
    $("ai-thinking").checked = !!ai.settings.thinking;
  }
  if (r.split) $("ai-split").value = r.split;
  for (const it of ai.transcript) { chatUser(it.name, it.text); chatBotStart(it.mid); botEl.pieces = [{ t: it.reply || "", d: 0 }]; chatBotEnd(null, it.stats); }
  if (!r.model || !MODELS[r.model]) return;
  setModelValue(r.model); modelTouched = true;
  // start the model again once the devices that held layers are back, or after 25 s regardless
  const want = new Set(r.peers || []), t0 = Date.now();
  aiStatus(want.size ? `resumed: waiting for ${[...want].join(", ")} to come back…` : "resumed: loading the model again…");
  const tick = setInterval(() => {
    const back = [...conns.values()].filter((c) => want.has(c.name)).length;
    if (back >= want.size || Date.now() - t0 > 25000) {
      clearInterval(tick);
      log("room", `resumed room ${roomCode}: ${back} of ${want.size} devices back, dealing the layers again; the conversation continues`);
      aiStart(r.model);
    }
  }, 500);
}

// ---- messages: worker, guest and host ----
// Messages only the host sends: a device ignores them from anyone else (a guest cannot rewrite the
// room's layers, chat or state), and the host ignores them altogether.
const FROM_HOST = new Set(["ai-layers", "ai-ready-all", "ai-reset", "ai-redeal", "ai-degraded", "ai-map", "ai-genstart",
  "ai-token", "ai-gendone", "ai-history", "ai-reacts", "ai-queue", "ai-queued", "ai-regen", "ai-hostprog", "ai-next",
  "ai-visibility", "ai-style", "ai-busy", "ai-wait"]);
async function aiOnData(from, d) {
  if (d.t.startsWith("ai-code") || d.t.startsWith("ai-pv")) { codeOnData(from, d); return; }
  const e = conns.get(from);
  if (FROM_HOST.has(d.t)) {
    if (ai.role === "host") return;
    if (from !== (ai.hostId || PREFIX + roomCode)) return;
  }
  if (d.t === "ai-load" && ai.role === "host") return;
  // returned hidden states are only accepted from the end of the chain
  if ((d.t === "ai-hiddenret" || d.t === "ai-hiddenret-b") && from !== ai.chain[ai.chain.length - 1]) return;
  switch (d.t) {
    case "ai-start-req":
      setModelValue(d.model);   // every screen shows the model that was actually started
      if (d.boss !== peer.id && conns.has(d.boss)) ai.hostId = d.boss;   // the device dealing the layers runs the room
      $("ai-start").disabled = true; $("ai-model").disabled = true;
      if (d.boss !== peer.id) { aiLoading(true, `starting ${MODELS[d.model]?.label.split("·")[0].trim()}`); $("ldg-sub").textContent = `${d.by} pressed start`; $("ldg-fill").style.width = "0%"; }
      if (d.boss === peer.id) { toast(`${d.by} started ${MODELS[d.model]?.label.split("·")[0].trim()}`); aiStart(d.model); }
      else aiStatus(`${d.by} started the model…`);
      break;
    case "ai-next": ai.next = d.next; ensureLink(d.next); break;
    case "ai-layers":
      ai.layersByName = d.by; loadCardRender();
      if (ai.role === "worker" && !d.by[myName]) {   // not in this deal: ask-only guest, GPU memory freed
        ai.role = "guest"; ai.range = null; ai.engine = null;
        try { ai.device?.destroy(); } catch {}
        ai.device = null;
      }
      break;
    case "ai-reset":   // the host started a new chat
      clearChat();
      toast(`${d.by || "the host"} started a new chat`);
      break;
    case "ai-redeal":
      $("chat-tools").hidden = true;
      $("ai-panel").classList.remove("online");
      $("ai-row").style.display = "none";
      $("room-over").hidden = true;
      setModelValue(d.model);
      aiLoading(true, "re-dealing the layers");
      $("ldg-sub").textContent = `${d.by} is re-dealing the layers over the devices in the room`;
      aiStatus(`${d.by} is re-dealing the layers…`);
      break;
    case "ai-degraded":
      aiStatus(`${d.why} — waiting for the host to re-deal the layers`);
      toast(d.why);
      break;
    case "ai-load": {
      setModelValue(d.model);
      ai.role = "worker";
      ai.next = d.next;
      ai.hostId = d.host;
      ai.q = Promise.resolve();
      ai.wsrc = MODELS[d.model]?.gguf && d.inv ? weightSources(MODELS[d.model].gguf, d.inv) : null;
      ensureLink(d.next);   // open the link to my chain neighbour while the weights download
      try {
        await aiLoadShard(d.model || "smollm-135m", d.range, false, false, d.ctx || maxSeqFor(d.model));
        if (!(await ensureLink(d.next))) throw new Error("could not connect to the next device in the chain");
        aiStatus(`layers ${d.range[0]}–${d.range[1] - 1} ready · syncing with the room…`);
        aiLoading(true, `layers ${d.range[0]}–${d.range[1] - 1} ready`);
        $("ldg-sub").textContent = "syncing with the rest of the room";
        $("ldg-fill").style.width = "100%";
        sendTo(ai.hostId, { t: "ai-ready" });
      } catch (err) {
        aiLoading(false);
        aiStatus("failed: " + err.message);
        sendTo(ai.hostId, { t: "ai-error", message: err.message });
      }
      break;
    }
    case "ai-hostprog": {
      const now = Date.now();
      ai.prog = { ...(d.all || {}), [myName]: Math.round(ai.myPct || 0) };
      ai.progAt = ai.progAt || {};
      for (const nm of Object.keys(d.all || {})) if (nm !== myName) ai.progAt[nm] = now;
      loadCardRender();
      break;
    }
    case "ai-progress":
      if (e?.card) { e.card.querySelector(".bw").textContent = "dl " + d.pct + "%"; peerStatus(e.card, "loading " + d.pct + "%"); }
      ai.prog = ai.prog || {}; ai.progAt = ai.progAt || {};
      ai.prog[e?.name || from] = d.pct; ai.progAt[e?.name || from] = Date.now(); loadCardRender();
      break;
    case "ai-ready":
      if (ai.role !== "host" || !ai.chain.includes(from)) break;
      ai.readyPeers.add(from);
      if (e?.card) { e.card.querySelector(".bw").textContent = "ready"; peerStatus(e.card, "ready", true); }
      aiMaybeReady();
      break;
    case "ai-error":
      aiStatus(`peer ${e?.name || from} failed: ${d.message}`);
      if (ai.role === "host" && ai.chain.includes(from)) failWaiters(new Error(`${e?.name || from}: ${d.message}`));
      break;
    case "ai-tele": if (ai.role === "host") { ai.teleBy.set(from, { ...(d.k || {}), amax: +d.amax || 0 }); } break;
    case "ai-inv-req": cachedRanges(d.url).then((have) => sendTo(from, { t: "ai-inv", url: d.url, have })); break;
    case "ai-inv": if (ai.invWait && ai.invWait.url === d.url && Array.isArray(d.have)) ai.invWait.inv[from] = d.have.slice(0, 20000); break;
    case "ai-wget": if (PEER_WEIGHTS) serveWeight(from, d); else sendTo(from, { t: "ai-wpart", id: d.id, miss: 1 }); break;
    case "ai-wpart": onWeightPart(d); break;
    case "ai-map": renderMap(d.nodes, d.st, d.live); break;
    case "ai-hidden-b":
    case "ai-hidden":
      if (ai.role !== "worker") break;
      ai.q = ai.q.then(() => workerFrame(d)).catch((err) => {
        aiStatus("⚠ " + err.message);
        sendTo(ai.hostId, { t: "ai-error", message: err.message });
      });
      break;
    case "ai-hiddenret-b": lapDone("b" + d.basePos, unpackWire(d)); break;
    case "ai-hiddenret": lapDone(d.pos, unpackWire(d)); break;
    case "ai-visibility":
      ai.visibility = d.mode;
      toast(d.mode === "all" ? "the host shows the chat to everyone" : d.mode === "host" ? "the host keeps the chat private" : "the host shows each answer to whoever asked");
      codeRoleChanged();   // Code is shared with every member only when everyone sees the answers
      break;
    case "ai-style":
      toast(`answers now: ${PERSONAS[d.persona]?.label || d.persona}${d.thinking ? " · thinking first" : ""}`);
      break;
    case "ai-regen": markReplaced(); break;
    case "ai-react": case "ai-reacts": break;   // reactions were removed; an older device may still send them
    case "ai-typing":
      if (ai.role === "host") {
        if (ai.visibility !== "all") break;
        const name = conns.get(from)?.name || "someone";
        for (const id of conns.keys()) if (id !== from) sendTo(id, { t: "ai-typing", name });
        showTyping(name);
      } else showTyping(d.name || "someone");
      break;
    case "ai-cmd": aiCommand(d.cmd, from); break;
    case "ai-genstart":
      setAfterAnswer(false, false);
      if (!d.cont)
      chatUser(d.name, d.hidden ? "asked something (the host keeps the chat private)" : d.text);
      chatBotStart(d.hidden ? null : d.mid);
      setBusyUI(true, d.asker === peer.id);
      mascot(`${d.name} asked something. Thinking…`);
      break;
    case "ai-token": chatBotPiece(d.text, d.d); mapPulse(); break;
    case "ai-gendone":
      chatBotEnd(d.hidden ? "answer hidden by the host" : null, d.stats);
      setBusyUI(false);
      setAfterAnswer(!!d.capped && !d.hidden, !d.failed && !d.hidden);
      if (d.ctx) setCtx(d.ctx.used, d.ctx.max);
      mascot("Your turn. Ask anything.");
      break;
    case "ai-history":
      for (const it of d.items || []) {
        chatUser(it.name, it.text);
        chatBotStart(it.mid);
        botEl.pieces = [{ t: it.reply || "", d: 0 }];
        chatBotEnd(null, it.stats);
      }
      break;
    case "ai-ready-all":
      aiLoading(false);
      $("ai-panel").classList.add("online");
      if (ai.role !== "host" && ai.role !== "worker") ai.role = "guest";
      if (ai.role !== "host") ai.hostId = from;
      if (MODELS[d.model]) { setModelValue(d.model); ai.model = d.model; }
      $("ai-row").style.display = "flex";
      $("chat-tools").hidden = false;
      $("mode-bar").hidden = false;   // Chat | Code for every device, not only the host (a phone guest had no way to Code)
      emptyText("The model is ready. Ask anything.");
      sysNote("Model ready");
      aiStatus(ai.range ? `cluster online · serving layers ${ai.range[0]}–${ai.range[1] - 1}` : "cluster online · this device asks, the others think");
      mascot("Cluster online! Type a question, the whole room answers.");
      codeRoleChanged();
      break;
    case "ai-ask":
      if (ai.role !== "host") break;
      aiAsk(String(d.text || "").slice(0, 8000), String(e?.name || "guest"), from);
      break;
    case "ai-queued":
      toast(d.pos === 1 ? "queued: yours is next" : `queued: ${d.pos - 1} question${d.pos > 2 ? "s" : ""} ahead of yours`);
      break;
    case "ai-queue": showQueue(d.n); break;
    case "ai-stop":
      if (ai.role === "host" && ai.busy === "gen" && from === ai.askerId) { ai.abort = true; aiStatus(`${e?.name || "the asker"} pressed stop…`); }
      break;
    case "ai-busy": toast(d.why || "the room is still answering, try again in a moment"); break;
  }
}

// ---- Code mode: the surface room/code.js sees (docs/design/harness-app.md A.2) ----
// The agent drives the room through roomApi only. A code run holds the room's lock (ai.busy =
// "code") for all its steps, so a chat question cannot reset the caches between two of them;
// questions asked meanwhile queue as usual and run when the lock is released.
//
// ?mock=code on localhost (tests only): no model needed. roomApi.ready() is true, the lock works
// without an engine, and room/code.js takes its model from window.__pooledMock.model.
const MOCK = new URLSearchParams(location.search).get("mock") === "code" && ["127.0.0.1", "localhost"].includes(location.hostname);
// Code messages only the host sends, and the ones members send it: the preview's file requests, and
// driving the shared agent (a request, Stop and an approval for the member's own request, New task,
// a project, the auto-approve box, and "send me the session" on opening Code). room/code.js checks
// each against its own state (who asked the run, the project list) and caps every field.
const CODE_FROM_HOST = new Set(["ai-code-start", "ai-code-tok", "ai-code-live", "ai-code-tool", "ai-code-note", "ai-code-done", "ai-code-files", "ai-code-history",
  "ai-code-projects", "ai-code-msg", "ai-pv", "ai-pv-blob", "ai-pv-stop"]);
const CODE_TO_HOST = new Set(["ai-pv-want", "ai-code-ask", "ai-code-stop", "ai-code-approve", "ai-code-cmd", "ai-code-sync"]);
const CODE_DRIVE = new Set(["ai-code-ask", "ai-code-cmd", "ai-code-sync"]);   // these load Code on a host that has not opened it
const CODE_MSG_MAX = 12000;   // a member's message, serialized (a request is at most 4000 characters)
const codeHandlers = new Map();   // message type -> fn(from, d)
const codeJoin = [], codeRole = [], codeStop = [];
let codeLoad = null, codeQ = Promise.resolve(), lockKind = null;

function roomLock(kind = "code") {
  if (ai.busy || ai.degraded || !(ai.engine || MOCK)) return false;
  ai.busy = lockKind = kind;
  ai.abort = false;
  setBusyUI(true, true);
  return true;
}
function roomUnlock() {
  if (!lockKind || ai.busy !== lockKind) return;
  ai.busy = false; ai.abort = false; lockKind = null;
  setBusyUI(false);
  setTimeout(nextQueued, 0);
  if (ai.degraded) showRedeal(true);
}
const codeHost = () => ai.role === "host";
const codeHostId = () => (codeHost() ? peer?.id : ai.hostId || PREFIX + roomCode);
// the room's visibility setting applies to the agent too: "host" and "asker" (the asker being the
// host) keep it on the host's screen. No stand-in messages: hidden screens get nothing.
function codeRecipients() { return chatRecipients(ai.visibility || "all", peer?.id, [...conns.keys()]).full; }
function sendCode(msg) { for (const id of codeRecipients()) sendTo(id, msg); }
function codeRoleChanged() {
  const r = { role: ai.role, hostId: codeHostId() };
  for (const fn of codeRole) { try { fn(r); } catch (err) { console.error(err); } }
}
function codeWelcome(id) {
  if (!codeHost() || !codeRecipients().includes(id)) return;
  for (const fn of codeJoin) { try { fn(id); } catch (err) { console.error(err); } }
}
// Load room/code.js once, on first use (the Code tab, or a host's code message arriving at a peer),
// so the chat page's load cost does not change.
function loadCode() {
  if (MOCK && isHost && !ai.role) { ai.role = "host"; ai.hostId = peer?.id; }
  return codeLoad ||= import("./room/code.js").then((m) => m.initCode?.(roomApi, { mock: MOCK ? window.__pooledMock : null }))
    .catch((err) => { codeLoad = null; throw err; });
}
// In arrival order, even across the first message's lazy load.
function codeOnData(from, d) {
  if (CODE_FROM_HOST.has(d.t)) {
    if (codeHost() || from !== codeHostId()) return;
  } else if (CODE_TO_HOST.has(d.t)) {
    // only from a device that said hello (a room member), only when the room shows it Code
    if (!codeHost() || !members.has(from)) return;
    if (!codeRecipients().includes(from)) {
      if (d.t === "ai-code-ask") sendTo(from, { t: "ai-code-msg", text: `only ${myName} uses Code in this room (Room settings: who sees answers)`, err: true });
      return;
    }
    if (d.t !== "ai-pv-want" && JSON.stringify(d).length > CODE_MSG_MAX) return;
  } else return;
  codeQ = codeQ.then(async () => {
    if (!codeHandlers.has(d.t) && (CODE_FROM_HOST.has(d.t) || CODE_DRIVE.has(d.t))) await loadCode();
    await codeHandlers.get(d.t)?.(from, d);
  }).catch((err) => console.error("code message", d.t, err));
}
// Chat is the room's first tab; Code is one click away (no switch on its own when the model is ready)
let simReady = false;
// initCode returns { show(mode) }; the Chat tab is handled by code.js once it is loaded
document.addEventListener("click", (e) => { if (e.target.closest?.("#mode-code") && $("mode-code").getAttribute("aria-selected") !== "true") window.pooledSparkle?.($("mode-code")); }, true);   // switching to Code sparkles (site/js/sparkle.js); capture: before the tab flips
// A tab opened before a deploy has the old modules in memory; Code mode's newer files can then fail to
// link against them. Say so plainly: a host reloads (it goes straight back into its room), a guest is asked to.
const staleModule = (err) => err instanceof SyntaxError || /binding name|export named|does not provide an export|not found in module|Failed to fetch dynamically imported module|error loading dynamically imported module/i.test(err?.message || "");
function codeLoadFailed(err) {
  if (staleModule(err)) {
    let tried = false; try { tried = sessionStorage.getItem("pooled-stale-reload") === "1"; sessionStorage.setItem("pooled-stale-reload", "1"); } catch {}
    if (isHost && !tried) { toast("Pooled was just updated: reloading to open Code…"); setTimeout(() => location.reload(), 900); return; }
    toast("Pooled was just updated. Reload this page to open Code.");
    return;
  }
  toast("Code mode failed to load: " + err.message);
}
document.addEventListener("click", (e) => { if (e.target.closest?.("#mode-code")) loadCode().then((c) => { try { sessionStorage.removeItem("pooled-stale-reload"); } catch {} c?.show?.("code"); }).catch(codeLoadFailed); });

const roomApi = {
  myId: () => peer?.id,
  name: () => myName,
  role: () => ai.role,                                  // "host" | "worker" | "guest" | undefined
  ready: () => MOCK || simReady || (!!ai.engine && !ai.degraded),   // host: can generate now (simReady: ?sim=1 pictures only)
  tok: () => ai.tok,
  chatTemplate: () => ai.tok?.chatTemplate || ai.G?.meta?.["tokenizer.chat_template"] || "",
  maxSeq: () => ctxMax(),
  generate: roomGenerate,
  lock: roomLock, unlock: roomUnlock,
  busy: () => ai.busy,
  // ai.abort ends the step in flight after its lap; code.js aborts its run's controller in onStop
  stop: () => {
    if (ai.busy === "code") { ai.abort = true; aiStatus("stopping after this lap…"); }
    for (const fn of codeStop) { try { fn(); } catch (err) { console.error(err); } }
  },
  onStop: (fn) => { codeStop.push(fn); },
  status: (text) => aiStatus(text),
  setCtx: (used, max) => setCtx(used, max ?? ctxMax()),
  // messaging
  send: (id, msg) => sendTo(id, msg),
  broadcast: (msg) => sendCode(msg),
  recipients: () => codeRecipients(),
  hostId: () => codeHostId(),
  nameOf: (id) => (id === peer?.id ? myName : conns.get(id)?.name || members.get(id)?.name || ""),
  visibility: () => ai.visibility || "all",   // "all": Code is shared, every member can drive it
  peers: () => [...conns.keys()],
  channel: (id) => conns.get(id)?.conn?.dataChannel || null,   // for bufferedAmount back-pressure (ai-pv-blob)
  on: (type, fn) => { codeHandlers.set(type, fn); },
  onPeerJoin: (fn) => { codeJoin.push(fn); },
  onRole: (fn) => { codeRole.push(fn); },
  load: loadCode,
  mock: MOCK,
};
if (MOCK) window.__pooledMock = { model: null, api: roomApi };

$("ai-start").addEventListener("click", aiStartAnywhere);
$("ai-redeal").addEventListener("click", aiRedeal);
$("ai-split").addEventListener("change", () => {
  if (ai.role === "host" && ai.engine) showRedeal(true, $("ai-split").value === "speed" ? "re-deal to put the layers on the fastest devices (measured on the answers so far)" : "re-deal to split by memory again");
});
$("ai-visibility").addEventListener("change", (e) => {
  ai.visibility = e.target.value;
  broadcastAll({ t: "ai-visibility", mode: ai.visibility });
  toast(ai.visibility === "all" ? "everyone sees the chat" : ai.visibility === "host" ? "only you see the chat" : "each answer goes to whoever asked");
});
// answer style: persona, sampling, thinking. Takes effect on the next question; a new system
// prompt changes the conversation's first tokens, so that question re-prefills from scratch.
for (const [k, v] of Object.entries(PERSONAS)) $("ai-persona").add(new Option(v.label, k));
for (const [k, v] of Object.entries(SAMPLING)) $("ai-sampling").add(new Option(v.label, k));
function styleChanged() {
  ai.settings = { persona: $("ai-persona").value, sampling: $("ai-sampling").value, thinking: $("ai-thinking").checked, length: $("ai-length").value };
  broadcastAll({ t: "ai-style", ...ai.settings });
  saveHost();
}
for (const id of ["ai-persona", "ai-sampling", "ai-thinking", "ai-length"]) $(id).addEventListener("change", styleChanged);
// Room settings: every select in the sheet shows as a segmented control (or chips, for the answer
// styles) with plain labels and a line of help for the chosen option. The select stays the source of
// truth: a click sets it and fires its change, so everything that listens to it works as before.
const SEG_LABEL = {
  "ai-visibility": { all: "Everyone", host: "Only me", asker: "Whoever asked" },
  "ai-length": { short: "Short", normal: "Normal", long: "Long" },
  "ai-sampling": { creative: "Creative", focused: "Focused", exact: "Exact" },
  "ai-split": { memory: "By memory", speed: "For speed" },
  "ai-persona": { default: "Plain", concise: "Concise", eli5: "Like I'm five", pirate: "Pirate", haiku: "Haiku", swarm: "The room speaks" },
};
const SEG_HELP = {
  "ai-visibility": { all: "Everyone in the room sees the questions and the answers.", host: "Only this device sees the text. Every device still helps write it.", asker: "Each answer goes to whoever asked it. Every device still helps write it." },
  "ai-length": { short: "About a paragraph at most (150 tokens).", normal: "A few paragraphs (400 tokens).", long: "Room for long answers and code (1,200 tokens)." },
  "ai-sampling": { creative: "Varied wording: ask twice, get two different answers.", focused: "Steadier wording, fewer surprises.", exact: "Always the likeliest word: the same question gets the same answer." },
  "ai-split": { memory: "Every device holds some layers, sized by the memory it gives.", speed: "The fastest devices hold the layers, with the fewest hops. Takes effect when the layers are dealt again." },
};
const segLabel = (id, o) => SEG_LABEL[id]?.[o.value] || o.text.replace(/\s*\(.*\)$/, "").replace(/^./, (c) => c.toUpperCase());
function buildSegs() {
  for (const seg of document.querySelectorAll(".seg[data-for]")) {
    const id = seg.dataset.for, sel = $(id);
    seg.innerHTML = [...sel.options].map((o) => `<button type="button" role="radio" data-v="${esc(o.value)}" aria-checked="false" tabindex="-1" title="${esc(o.text)}">${esc(segLabel(id, o))}</button>`).join("");
  }
  syncSegs();
}
function syncSegs() {
  for (const seg of document.querySelectorAll(".seg[data-for]")) {
    const sel = $(seg.dataset.for);
    for (const b of seg.children) { const on = b.dataset.v === sel.value; b.setAttribute("aria-checked", String(on)); b.tabIndex = on ? 0 : -1; b.disabled = sel.disabled; }
    const help = document.querySelector(`.set-help[data-help="${seg.dataset.for}"]`);
    if (help) help.textContent = SEG_HELP[seg.dataset.for]?.[sel.value] || "";
  }
}
function segPick(b) {
  const seg = b.closest(".seg[data-for]"), sel = $(seg.dataset.for);
  if (sel.value !== b.dataset.v) { sel.value = b.dataset.v; sel.dispatchEvent(new Event("change", { bubbles: true })); }
  syncSegs();
}
document.addEventListener("click", (e) => { const b = e.target.closest?.(".seg[data-for] > button"); if (b) segPick(b); });
document.addEventListener("keydown", (e) => {
  const b = e.target.closest?.(".seg[data-for] > button");
  if (!b || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) return;
  const all = [...b.parentElement.children], k = all.indexOf(b), d = e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 1;
  const nb = all[(k + d + all.length) % all.length];
  e.preventDefault(); nb.focus(); segPick(nb);
});
document.addEventListener("change", (e) => { if (e.target.closest?.("#room-menu")) syncSegs(); });
buildSegs();
$("room-menu").addEventListener("toggle", () => { if ($("room-menu").open) syncSegs(); });
$("menu-close").addEventListener("click", () => { $("room-menu").open = false; $("room-menu").querySelector("summary").focus(); });
$("cache-clear").addEventListener("click", async (ev) => {
  ev.preventDefault();
  try { await caches.delete("swarmllm-weights-v1"); weightCache = null; toast("cached weights cleared"); } catch { toast("could not clear the cache"); }
});
$("new-chat").addEventListener("click", aiNewChat);
$("draft-view").addEventListener("click", () => setDraftView(!draftView));
$("export-chat").addEventListener("click", () => { $("room-menu").open = false; exportChat(); });
for (const [id, cmd] of [["continue-btn", "continue"], ["regen-btn", "regen"]])
  $(id).addEventListener("click", () => { setAfterAnswer(false, false); if (ai.role === "host") aiCommand(cmd, peer.id); else if (ai.hostId) sendTo(ai.hostId, { t: "ai-cmd", cmd }); });
// Questions asked while the room is answering wait in the host's queue and run in order, one
// generation at a time (every device is busy with every token). At most QUEUE_MAX waiting, two
// per device.
const QUEUE_MAX = 10;
function aiAsk(text, name, from) {
  if (!text) return;
  if (ai.degraded || !ai.engine) {
    const why = ai.degraded ? "a device left: the host has to re-deal the layers before the next question" : "the model is still loading";
    if (from === peer.id) toast(why); else sendTo(from, { t: "ai-busy", why });
    return;
  }
  if (!ai.busy && !ai.queue?.length) { aiGenerate(text, name, from); return; }
  ai.queue ||= [];
  if (ai.queue.length >= QUEUE_MAX || ai.queue.filter((q) => q.from === from).length >= 2) {
    setTimeout(nextQueued, 0);
    const why = "the queue is full, try again after this answer";
    if (from === peer.id) toast(why); else sendTo(from, { t: "ai-busy", why });
    return;
  }
  ai.queue.push({ text, name, from });
  setTimeout(nextQueued, 0);   // idle with a queue: start the oldest
  const pos = ai.queue.length;
  if (from === peer.id) toast(pos === 1 ? "queued: yours is next" : `queued: ${pos - 1} ahead of yours`);
  else sendTo(from, { t: "ai-queued", pos });
  broadcastAll({ t: "ai-queue", n: ai.queue.length }); showQueue(ai.queue.length);
}
function nextQueued() {
  if (ai.role !== "host" || ai.busy || ai.degraded || !ai.engine || !ai.queue?.length) return;
  const q = ai.queue.shift();
  broadcastAll({ t: "ai-queue", n: ai.queue.length }); showQueue(ai.queue.length);
  aiGenerate(q.text, q.name, q.from);
}
function showQueue(n) { $("queue-note").textContent = n ? `${n} queued` : ""; }
function aiSubmit() {
  const text = $("ai-prompt").value.trim();
  if (!text) return;
  if (ai.role === "host") { $("ai-prompt").value = ""; growPrompt(); aiAsk(text, myName, peer.id); return; }
  const hostId = ai.hostId;
  if (!conns.has(hostId)) { toast("not connected to the host"); return; }
  $("ai-prompt").value = ""; growPrompt();
  sendTo(hostId, { t: "ai-ask", text, name: myName });
}
function aiStop() {
  if (ai.role === "host") {
    if (ai.busy === "gen") { ai.abort = true; aiStatus("stopping after this lap…"); }
    else if (ai.busy === "code") roomApi.stop();
  }
  else if (ai.hostId) sendTo(ai.hostId, { t: "ai-stop" });
  $("ai-send").disabled = true;
}
// the prompt box grows with its text; Enter sends and Shift+Enter is a new line (phones: the
// keyboard's return key is a new line, the Send button sends)
function growPrompt() {
  if (!$("ai-prompt").value) queueMicrotask(sendLabel); const p = $("ai-prompt"); p.style.height = "auto"; p.style.height = Math.min(p.scrollHeight, 160) + "px"; }
// the button stops while it says Stop; Enter always sends (or queues) the text, never stops
$("ai-send").addEventListener("click", () => { if ($("ai-send").classList.contains("stop")) aiStop(); else aiSubmit(); });
$("ai-prompt").addEventListener("input", () => { growPrompt(); sendLabel(); noteTyping(); });
$("ai-prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing && !myMeta?.phone) { e.preventDefault(); aiSubmit(); }
});
// (Code mode handles its own Esc: one in a field there backs out of the field, not the run)
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !e.defaultPrevented && !e.target?.closest?.("#code-pane") && $("ai-send").classList.contains("stop")) aiStop(); });
mascot("Hi! Create a room, or type a friend's code to join one.");

// ---- ?sim=1 on localhost: made-up devices, loading, chat and passes, for looking at the UI
// without a GPU (the visual checks use it). It paints; it never loads or runs a model.
if (new URLSearchParams(location.search).get("sim") === "1" && ["127.0.0.1", "localhost"].includes(location.hostname)) {
  const fake = { "MacBook Air": { ua: "Mac", webgpu: true, contribGB: 12 }, "Desktop PC": { ua: "Device", webgpu: true, contribGB: 12 }, "Pixel 8": { ua: "Android", webgpu: true, phone: true, contribGB: 2 }, "iPad": { ua: "iPad", webgpu: true, contribGB: 6 } };
  const names = () => [myName, ...[...conns.values()].map((c) => c.name)];
  const deal = () => {
    const rank = (nm) => { const k = Object.keys(fake).indexOf(nm); return k < 0 ? 9 : k; };
    const ns = names().sort((a, b) => rank(a) - rank(b) || a.localeCompare(b)), L = 40, by = {};
    const gb = ns.map((nm) => fake[nm]?.contribGB || 4), sum = gb.reduce((a, b) => a + b, 0);
    let lo = 0;
    ns.forEach((nm, i) => { const hi = i === ns.length - 1 ? L : Math.max(lo + 1, Math.round(lo + L * gb[i] / sum)); by[nm] = `${lo}–${hi - 1}`; lo = hi; });
    return by;
  };
  let passTimer = 0;
  window.__pooledSim = {
    devices() {
      for (const card of document.querySelectorAll("#peers .peer-card")) {
        const nm = card.dataset.name, m = fake[nm] || { ua: "Device", webgpu: true, contribGB: 8 };
        if (card.classList.contains("self")) Object.assign(myMeta, m);
        for (const [id, e] of conns) if (e.name === nm) { e.meta = { ...e.meta, ...m }; if (members.has(id)) members.get(id).meta = e.meta; }
        paintCard(card, nm, m, card.classList.contains("self"));
      }
      selfStepper();
      updateCluster();
    },
    // loading(p, { cache: true }) pictures a load from the browser cache; { early: true } one that
    // started a few seconds ago (no time left yet)
    loading(p = 0.4, { cache = false, early = false } = {}) {
      this.devices();
      setModelValue("qwen3.6-35b-moe"); ai.model = "qwen3.6-35b-moe";
      ai.layersByName = deal(); ai.cfg = { num_hidden_layers: 40 };
      const by = ai.layersByName, mine = /^(\d+)\D+(\d+)$/.exec(by[myName]);
      ai.range = mine ? [+mine[1], +mine[2] + 1] : null;
      ai.prog = Object.fromEntries(names().map((nm, i) => [nm, Math.round(Math.min(100, p * 100 * (1 + i * 0.6)))]));
      ai.myPct = ai.prog[myName];
      aiLoading(true, `Loading Qwen3.6 35B MoE`);
      const total = 3.1 * 2 ** 30, done = p * total, now = performance.now();
      ai.netBytes = cache ? 0 : done; ai.peerBytes = 0; cacheHits = cache ? done : 0;
      Object.assign(eta, { t0: now - (early ? 6000 : 48000), t: now, done, rate: (total - done) / 128, hist: [] });
      aiProgress(done, total);
      for (const card of document.querySelectorAll("#peers .peer-card")) peerStatus(card, `${ai.prog[card.dataset.name] ?? 0}%`);
      loadCardRender();
    },
    ready() {
      if (!ai.layersByName) this.loading(1);
      const mine = /^(\d+)\D+(\d+)$/.exec(ai.layersByName[myName] || "");
      ai.range = mine ? [+mine[1], +mine[2] + 1] : null;
      aiLoading(false);
      $("ai-panel").classList.add("online");
      $("ai-row").style.display = "flex"; $("chat-tools").hidden = false; $("mode-bar").hidden = false;
      emptyText("The model is ready. Ask anything.");
      for (const card of document.querySelectorAll("#peers .peer-card")) peerStatus(card, "ready", true);
      const ns = names();
      renderMap(Object.keys(ai.layersByName).map((nm, i) => ({ name: nm, layers: ai.layersByName[nm], host: i === 0 ? 1 : 0, ms: 18 + i * 9 })), { tps: 21.4, lap: 64 }, false);
      aiStatus("cluster online");
      simReady = true; setCtx(1846, 32768);
    },
    chat() {
      if (!$("ai-panel").classList.contains("online")) this.ready();
      const ns = names();
      clearChat();
      sysNote(`Model ready on ${ns.length} devices`);
      chatUser(ns[0], "what is Pooled?");
      chatBotStart(1);
      botEl.pieces = [{ t: "Pooled runs one open AI model across the devices in this room. Each one holds some of my layers, and **every word I write passes through all of them**, right here in your browser tabs.", d: 0 }];
      chatBotEnd(null, "21.4 tok/s · 3 devices");
      sysNote(`${ns[ns.length - 1]} joined with 2 GB`, "join");
      chatUser(ns[ns.length - 1], "can it write code?");
      chatBotStart(2);
      chatBotPiece("Yes. Open **Code** and tell me what to build. I write the files, run them, and you watch it ", 0);
      setCtx(1846, 32768);
      renderMap(lastMap.nodes, { tps: 21.4, lap: 64 }, true);
      this.pulse(6);
      loadCode().then((c) => c?.show?.("chat"));
    },
    // a question sent, no token yet: the working line in the answer's place
    waiting() {
      if (!$("ai-panel").classList.contains("online")) this.ready();
      chatUser(myName, "write a haiku about the sea");
      chatBotStart(3);
      renderMap(lastMap.nodes, { tps: 21.4, lap: 64 }, true);
    },
    // deal the layers again over the devices now in the room (after more tabs joined)
    reset() {
      ai.layersByName = deal();
      const mine = /^(\d+)\D+(\d+)$/.exec(ai.layersByName[myName] || "");
      ai.range = mine ? [+mine[1], +mine[2] + 1] : null;
      renderMap(Object.keys(ai.layersByName).map((nm, i) => ({ name: nm, layers: ai.layersByName[nm], host: i === 0 ? 1 : 0, ms: 18 + i * 9 })), { tps: 21.4, lap: 64 }, false);
    },
    // Code mode's context meter (a scripted model has no token count of its own)
    codeCtx(used = 5400, max = 32768) { return loadCode().then((c) => c?.ctx?.(used, max)); },
    pulse(n = 1) { for (let i = 0; i < n; i++) setTimeout(() => { pulseAt = 0; mapPulse(); }, i * 250); },
    idle() { $("ai-panel").classList.remove("online", "loading"); ai.range = null; },
    compute(on) { on ? compute.open() : compute.close(); },
    passes(on) {
      clearInterval(passTimer); passTimer = 0;
      if (on) passTimer = setInterval(() => compute.pass(1, 14 + Math.random() * 8), 90);
    },
  };
}
