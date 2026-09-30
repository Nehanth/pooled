// Pooled room: signaling, WebRTC mesh, layer assignment, weight streaming and the
// generation loop (prefill, decode, speculative verify). Served with p2p.html at /room.
// The inference engine (engine/engine.js, engine/qwen35.js and their WGSL kernels, ~250 KB) is not
// part of the join screen's module graph: loadEngine() imports it when this device enters a room,
// and aiLoadShard waits for it. The join screen works as soon as the lobby modules below are in.
import { argmax } from "./engine/sampling.js";
let autotuneCoop, makeTokenizer, DenseEngine, fetchModelShard, shardTensorNames, gpuSelfTest, kernelMicroTests, Qwen35Engine;
let engineLoad = null;
function loadEngine() {
  return engineLoad ||= Promise.all([import("./engine/engine.js"), import("./engine/qwen35.js")]).then(([e, q]) => {
    ({ autotuneCoop, makeTokenizer, DenseEngine, fetchModelShard, shardTensorNames, gpuSelfTest, kernelMicroTests } = e);
    ({ Qwen35Engine } = q);
  }, (err) => { engineLoad = null; throw new Error("couldn't load the inference engine (" + (err?.message || err) + "). Check the connection and try again"); });
}
import { f32ToF16, f16ToF32, parseGGUFHeader, ggufWeights, ggufShardBytes, GGML_EMBED, GGML_OUTPUT, GGML_FINAL_NORM,
  ggmlLayerNames, qwen35Weights, qwen35ShardBytes, qwen35MtpBytes, qwen35LayerNames, qwen35NamesFor, tokenizerFromGGUF, gpuUploadEntry, streamEntryToGPU }
  from "./engine/gguf.js";
import { roomQwen35Options, roomEngineFlags, applyRoomFlags } from "./engine/preset.js";   // no imports of its own; the engine itself loads late (loadEngine)
import { WIRE_F16, badF32, f32ToB64, packF16, unpackF16, asU16, packWire, unpackWire, asF32, b64ToF32, wireStats } from "./room/wire.js";
import { esc, md, mdChat } from "./room/markdown.js";
import { pickSampler, SAMPLING } from "./room/sampling.js";
import { chatRecipients } from "./room/visibility.js";
import { validateApiAsk, apiPrompt, apiRun, AnswerCache, API_LIMITS, pieceDecoder, helloMeta, withStyle, apiPrompt2, apiRun2, TurnCache, EncodeCache } from "./room/api.js";
import { tokenTexts } from "./harness/model-common.js";
import { PrefixIndex, pinSplit } from "./harness/prefix.js";
import { CkptStore } from "./room/ckpt-store.js";
import { MODELS, NEED_GB, FILE_GB, PICKER, MAX_SEQ, MAX_NEW, MAX_NEW_THINKING, MIN_ROOM, maxSeqFor, kvBytesPerLayerPos, kvModeFor, kvForLoad } from "./room/models.js";
// the context window of the loaded engine (per model: room/models.js CTX; 2048 for the small ones)
const ctxMax = () => ai.engine?.maxSeq || MAX_SEQ;
// ?ckpt=N: keep the room's state after the last N answers on every device (GPU copies), so a
// regenerate, an edited question or a branch resumes from the longest saved turn instead of
// prefilling the whole conversation again. 0 turns it off.
const CKPT_MAX = Math.max(0, parseInt(new URLSearchParams(location.search).get("ckpt") ?? "2", 10) || 0);
// ...and a copy of each on disk (OPFS, room/ckpt-store.js), so a device that reloads reads its part
// back instead of the room prefilling the whole conversation again. ?ckptdisk=0: GPU copies only.
const ckptDisk = CKPT_MAX && new URLSearchParams(location.search).get("ckptdisk") !== "0" && globalThis.navigator?.storage?.getDirectory
  ? new CkptStore() : null;
import { makeLink, attachWire, wireReady, sendFrame, setKeepalive, PROTOCOL, DROP_ALL } from "./room/transport.js";
import { peerErrorText, peerErrorLoud, FetchError, joinStep, versionMismatch } from "./room/errors.js";
import { PERSONAS, specials, fitContext, reusablePrefix, templateProfile } from "./room/conversation.js";
import { PING_MS, lastHeard, isSilentGone, midLoad, uniqueName, quietNamesake, staleNamesakes, renameTo, NAME_PROBE_MS } from "./room/liveness.js";
import { stopsStart, stopWhen, stopReason, loadKey as shardKey, onLoadRequest, onLoadError, freeOnStartFailed } from "./room/startstop.js";
import { planSplit, planForSpeed, phonesToLeaveOut, isPhoneMeta, ladder, bestFit, codeFromLocation, pickModelHost } from "./room/plan.js";
import { measureCopyGBps } from "./room/gpuspeed.js";
import { qrSVG } from "./room/qr.js";
import { lookupDrafts } from "./room/lookup.js";
import { drawCard } from "./room/card.js";
import { probe as preflight, deviceKind } from "./room/preflight.js";
import { computeScreen } from "./room/compute.js";
import { CACHE_NAME, PREFIX as CACHE_PREFIX, cacheKey, cachedModels, deleteModel } from "./room/weightcache.js";
import { working, liveWords } from "./room/working.js";
import { attachBrowserWeightCache, convertedBytes, clearConverted, convertedByModel, deleteConverted, modelOf } from "./room/convertedcache.js";

// Hidden-state transport (room/transport.js). ?wire=off falls back to PeerJS messages;
// ?wire=slice uses one sliced channel; ?wire=stripeN spreads slices over N peer connections.
const WIRE = (new URLSearchParams(location.search).get("wire") || "stripe4").toLowerCase();
const WIRE_STRIPES = WIRE === "off" ? 0 : WIRE.startsWith("stripe") ? Math.max(1, Math.min(8, parseInt(WIRE.slice(6), 10) || 1)) : 1;
// ?ka=ms: keep-alive period on a wire link while frames flow (keeps a phone's Wi-Fi out of power
// save between laps; room/transport.js), ?ka=0 turns it off
{ const ka = new URLSearchParams(location.search).get("ka"); if (ka != null) setKeepalive(parseInt(ka, 10) || 0); }
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
// sw: a device's colour for the dot (joined / left); kind "error" stays twice as long, to be read
function toast(text, { sw = null, kind = "" } = {}) {
  const box = $("toasts"), t = document.createElement("div");
  t.className = "toast" + (kind ? " " + kind : "");
  if (sw) t.style.setProperty("--sw", sw);
  t.textContent = text;
  box.appendChild(t);
  while (box.children.length > 3) box.firstElementChild.remove();
  // gone once its fade-out ends (p2p.html: toastout at 3.6 s, 7.8 s for .error), so a faded toast
  // never keeps its space or one of the three slots; the timer is a fallback (reduced motion, no animation)
  t.addEventListener("animationend", (e) => { if (e.animationName === "toastout") t.remove(); });
  setTimeout(() => t.remove(), kind === "error" ? 8400 : 4200);
}
// someone joined or left: a toast, but not for the devices already here when this tab came in
let roomSince = Infinity;
const presence = (name, joined) => { if (performance.now() - roomSince > 2500) toast(`${name} ${joined ? "joined" : "left"}`, { sw: joined ? devColor(name) : "var(--faint)", kind: "presence" }); };
function mascot() {}
const PREFIX = "pooled-room-";   // PeerJS id prefix (was "swarmllm-room-" before the rename; PROTOCOL did not change)
const NAME_KEY = "pooled-name";   // sessionStorage: this tab's device name, so a reload rejoins under the same name
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
        // how fast this GPU moves memory (room/gpuspeed.js), which picks the model host among
        // computers; phones skip it (they never host over a computer). ?gbps=N pins it, 0 = unknown
        const pin = new URLSearchParams(location.search).get("gbps");
        meta.gbps = pin !== null ? Math.max(0, +pin || 0) : meta.canMeasure ? await measureCopyGBps(a) : 0;
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
  $("join-gpu").querySelector("details").open = !!v.detail;   // the remedy is the useful part: show it
  $("join-gpu").hidden = false;
  $("join-pledge").classList.add("no-gpu");
  $("ap-no").textContent = v.line;
  if (v.detail) { const d = document.createElement("span"); d.className = "ap-no-d"; d.textContent = v.detail; $("ap-no").append(" ", d); }
});
// the least any model in the picker needs (the 1.7B's 4 GB)
const SMALLEST_NEED = Math.min(...PICKER.map((k) => NEED_GB[k]));
// probe once at load; fill the contribution selector, unless an amount was already chosen there
// (typed, stepped or filled in while the probe ran: it takes up to ~0.5 s with the copy timing)
let joinGbChosen = false;
for (const [id, ev] of [["join-gb", "input"], ["gb-minus", "click"], ["gb-plus", "click"]]) $(id).addEventListener(ev, () => { joinGbChosen = true; });
const metaPromise = (async () => {
  const m = await probeGPU();
  if (m.webgpu && m.budgetGB) m.contribGB = Math.max(0.2, Math.round(m.budgetGB * 0.5 * 10) / 10);
  m.phone = m.ua === "iPhone" || m.ua === "Android";
  if (m.phone) { m.contribGB = 0.5; $("join-gb").min = "0.5"; $("join-gb").step = "0.5"; }
  // a laptop gives at least what the smallest model needs, when its probe allows, so one laptop can
  // run the 1.7B alone (half of maxBufferSize is 2 GB on a typical laptop, and 2 GB runs nothing)
  else if (m.contribGB) m.contribGB = Math.max(1, Math.round(m.contribGB), Math.min(SMALLEST_NEED, Math.floor(m.budgetGB)));
  if (m.contribGB && !joinGbChosen) $("join-gb").value = m.contribGB;
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
  // an API client (`pooled serve`): angle brackets, code talking to the room
  api: '<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path d="M6 1.8v2.9M10 1.8v2.9M4.3 4.7h7.4v2.4a3.7 3.7 0 0 1-7.4 0zM8 10.8v3.4" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>',   // a plug: an API client (pooled serve)
};
const iconFor = (meta) => meta.api ? ICONS.api : meta.phone || /iPhone|Android$/.test(meta.ua || "") ? ICONS.phone : /Mac|iPad/.test(meta.ua || "") ? ICONS.laptop : ICONS.desk;
// "0–19" (what the deal sends) -> "1–20", the way people count layers
const humanRange = (r) => { const m = /^(\d+)\D+(\d+)$/.exec(String(r || "")); return m ? `${+m[1] + 1}\u2013${+m[2] + 1}` : String(r || ""); };
// One colour per device, everywhere (chips, pool bar, loading rows, band, Lend screen): given once,
// in join order, to each device that can hold layers. A device that only asks is grey everywhere.
const SWATCH = ["#2A45E0", "#2B2F3C", "#7C8FFF", "#5E616B", "#B9C6FF", "#1C33B8",
  // devices 7 to 16: slate, sky, royal and navy blue, plus violets (#320578, #7E4FFB, #5D4DA4, #7F73C3,
  // #5912CA sit at OKLCH hue 289, past the tokens' 267-275), each picked to be as far as possible
  // from every colour before it, counting a lightness step of 0.12 (OKLab) or a hue/chroma step of 0.12 as
  // one unit: below one, a 6 px dot or a thin bar reads the same (the old shades sat at 0.25-0.67)
  "#8C939B", "#127ABE", "#320578", "#7E4FFB", "#5D4DA4", "#114B75", "#79B1E0", "#7F73C3", "#4074FB", "#5912CA"];
// past 16 devices: shades generated in a blue-to-slate range (hue 222-232), so no two neighbours match
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
      ${!self && meta.api && isHost ? '<button class="api-kick" type="button">Disconnect</button>' : ""}
    </div>`;
  paintCard(card, name, meta, self);
  $("peers").appendChild(card);
  if (devSlots.size) orderCards();
  card.querySelector(".pchip").addEventListener("click", (e) => { e.stopPropagation(); chipPop(card); });
  if (!self) card.querySelector(".bw-btn").addEventListener("click", () => bwTest(id));
  card.querySelector(".api-kick")?.addEventListener("click", () => { chipPop(null); apiKick(id); });
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
  if (!$("card").hidden) closeCard();
  if (!$("leave").hidden) closeLeave();
});
document.addEventListener("pointerdown", (e) => {
  const t = e.target.closest?.("[data-tip]"); if (!t) return;
  t.classList.add("tip-off");
  t.addEventListener("pointerleave", () => t.classList.remove("tip-off"), { once: true });
});
addEventListener("resize", () => chipPop(null));
$("peers").addEventListener("scroll", () => { chipPop(null); peersEdge(); }, { passive: true });
// more chips than the header has room for: fade the side(s) that hide some, and let a mouse wheel
// scroll the row sideways (only a trackpad or a drag could reach the last chips before)
function peersEdge() {
  const b = $("peers"), max = b.scrollWidth - b.clientWidth;
  b.classList.toggle("fade-l", max > 1 && b.scrollLeft > 1);
  b.classList.toggle("fade-r", max > 1 && b.scrollLeft < max - 1);
}
$("peers").addEventListener("wheel", (e) => {
  const b = $("peers");
  if (e.ctrlKey || Math.abs(e.deltaY) <= Math.abs(e.deltaX) || b.scrollWidth <= b.clientWidth) return;
  e.preventDefault();
  b.scrollLeft += e.deltaY * (e.deltaMode === 1 ? 16 : 1);
}, { passive: false });
if (typeof ResizeObserver === "function") new ResizeObserver(peersEdge).observe($("peers"));
new MutationObserver(peersEdge).observe($("peers"), { childList: true, subtree: true, characterData: true });
// what a card says about its device (the sim hook repaints with made-up devices)
function paintCard(card, name, meta, self) {
  card.querySelector(".pname").textContent = name;
  card.querySelector(".pn").textContent = name;
  if (self) card.querySelector(".pn").insertAdjacentHTML("beforeend", " <small>(you)</small>");
  card.querySelector(".pic").innerHTML = card.querySelector(".pic2").innerHTML = iconFor(meta);
  card.querySelector(".dot").className = "dot " + (self || meta.webgpu ? "ok" : "warn");
  card.querySelector(".pkind").textContent = meta.api ? `API client${meta.client ? " \u00b7 " + meta.client : ""}` : !meta.ua || meta.ua === "Device" ? "Computer" : meta.ua;
  // the GPU name is missing when the browser hides adapter info (and on sim devices): show only what we know
  const kind = !meta.ua || meta.ua === "Device" ? "Computer" : meta.ua;
  card.querySelector(".peer-gpu").textContent = meta.webgpu === false ? `${kind} · no WebGPU` : meta.gpu ? `${kind} · ${meta.gpu}` : kind;
  const budget = meta.budgetGB || meta.maxBufGB;
  setBuf(card, meta.api ? "no layers" : meta.webgpu === false ? "no WebGPU" : meta.contribGB ? lends(meta.contribGB) : (budget ? budget + " GB" : "-"));
  card.querySelector(".cg").textContent = meta.api ? "API" : meta.webgpu === false ? "chat only" : meta.contribGB ? meta.contribGB + " GB" : "";
  card.querySelector(".pchip").title = meta.api ? `${name}: an API client (pooled serve) asking through this room` : meta.webgpu === false ? `${name}: this device can ask but can't hold model layers` : `${name}: ${card.querySelector(".buf").textContent.replace(/[\u2212+]/g, "").trim()}`;
  card.style.setProperty("--sw", devColor(name));
  peerStatus(card, meta.api ? "API client" : meta.webgpu === false ? "chat only" : self ? "this device" : "connected");
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
  // a radio group: one tab stop (the picked row), arrows move the pick. The rows are re-rendered
  // on every change, so the focus follows the picked row when it was in the group.
  const had = el.contains(document.activeElement);
  el.innerHTML = (none ? '<p class="ai-nogpu">Needs a device with WebGPU</p>' : "") + ladder(PICK_NEED, pledged).map((x) => {
    const gb = `<span class="nd">${NEED_GB[x.key] ?? ""} GB</span>`;
    const fig = x.ok ? `${gb}<b>fits</b>` : none ? gb : `<span class="more">needs ${x.short} GB more</span>`;
    // with no device that can hold layers, nothing reads as picked: there is nothing to start yet
    const sel = !none && x.key === $("ai-model").value;
    return `<button type="button" role="radio" class="rung${x.ok ? " ok" : " short"}${sel ? " sel" : ""}" data-k="${x.key}" aria-checked="${sel}" tabindex="${sel ? 0 : -1}"${x.ok ? "" : ` title="${giveFor(x.short) ? "Raise This device gives, or invite a device" : "Invite a device to fit this"}"`}><span class="rn">${esc(shortName(x.key))}</span><span class="fig">${fig}</span></button>`;
  }).join("");
  if (!el.querySelector(".rung.sel")) el.querySelector(".rung")?.setAttribute("tabindex", "0");
  if (had) el.querySelector('.rung[tabindex="0"]')?.focus({ preventScroll: true });
}
function pickRung(b) { if (!b || $("ai-model").disabled) return; setModelValue(b.dataset.k); modelTouched = true; updateCluster(); }
$("ai-ladder").addEventListener("click", (e) => pickRung(e.target.closest(".rung")));
$("ai-ladder").addEventListener("keydown", (e) => {
  const rs = [...$("ai-ladder").querySelectorAll(".rung")], i = rs.indexOf(e.target.closest(".rung"));
  const step = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[e.key];
  const to = step ? (i + step + rs.length) % rs.length : e.key === "Home" ? 0 : e.key === "End" ? rs.length - 1 : -1;
  if (i < 0 || to < 0) return;
  e.preventDefault(); pickRung(rs[to]);
});
// how much this device would give to cover `short` GB, or 0 when it can't: needs WebGPU, and stays
// within what its probe allows (a phone keeps its small default; a laptop goes up to its budget)
function giveFor(short) {
  if (!(short > 0) || !myMeta.webgpu || myMeta.phone) return 0;
  const v = Math.ceil((myMeta.contribGB + short) * 10 - 1e-6) / 10;
  const cap = Math.min(64, Math.max(SMALLEST_NEED, Math.floor(myMeta.budgetGB || 0)));
  return v <= cap ? Math.ceil(v) : 0;
}
$("ai-give").addEventListener("click", (e) => lendGB(+e.currentTarget.dataset.gb));
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
  // short, and this device alone can close the gap: offer that one tap next to the disabled Start
  const give = giveFor(need - pledged);
  $("ai-give").hidden = ok || !give || ai.busy || !!ai.engine;
  // what pressing Start costs this device: about its share of the weights file (layers are dealt by
  // memory given), so a phone on mobile data sees ~0.2 GB and a laptop alone the whole file
  const file = FILE_GB[$("ai-model").value];
  const mine = ok && file && myMeta.webgpu && myMeta.contribGB ? file * Math.min(1, myMeta.contribGB / pledged) : 0;
  const size = (gb) => gb < 1 ? `${Math.max(10, Math.round(gb * 1024 / 10) * 10)} MB` : `${gb.toFixed(1)} GB`;
  $("ap-note").textContent = mine
    ? (mine >= file * 0.98 ? `Start downloads the whole ${size(file)} model to this device, once. It stays cached for next time.`
      : `Start downloads about ${size(mine)} to this device (its share of ${size(file)}), once. It stays cached for next time.`)
    : ok && file && !myMeta.webgpu ? "This device only chats, so it downloads no layers."
    : "Each device downloads only its own layers, once. They stay cached for next time.";
  // every device here is chat only: say why Start is off, right under it
  $("ai-why").hidden = pledged > 0 || ai.busy || !!ai.engine;
  // and say so beside the stepper, which is the one-tap fix (not a second device)
  $("ap-me-hint").hidden = ok || !give;
  $("ap-me-hint").textContent = `Raise this to ${give} GB to fit ${shortName($("ai-model").value)}.`;
  if (give) { $("ai-give").textContent = `Give ${give} GB from this device`; $("ai-give").dataset.gb = give; }
  // why Start is off, for screen readers (sighted users see it in the rows and the pool card)
  $("start-why").textContent = ok ? "" : !(pledged > 0) ? "No device with WebGPU yet. Invite one to start a model."
    : `${shortName($("ai-model").value)} needs ${need} GB and the room has ${has} GB, ${+(need - pledged).toFixed(1)} GB short. Invite a device or give more memory.`;
  if (ok) $("ai-start").removeAttribute("aria-describedby"); else $("ai-start").setAttribute("aria-describedby", "start-why");
  if (ok && !wasReady) { $("ai-start").classList.remove("unlocked"); void $("ai-start").offsetWidth; $("ai-start").classList.add("unlocked"); }
  wasReady = ok;
}
$("ai-model").addEventListener("change", () => { modelTouched = true; updateCluster(); });
function updateCluster() {
  const all = [myMeta, ...[...members.values()].map(m => m.meta).filter((m) => !m?.api)];
  const apis = members.size + 1 - all.length;
  const gpus = all.filter(m => m && m.webgpu).length;
  // only devices with WebGPU hold layers; the others join as ask-only guests
  const pledged = all.reduce((s, m) => s + (m?.webgpu ? m?.contribGB || 0 : 0), 0);
  updateNeed(pledged);
  const mem = all.reduce((s, m) => s + (m?.budgetGB || m?.maxBufGB || 0), 0);
  $("cluster-summary").textContent = DEV
    ? `${all.length} device${all.length > 1 ? "s" : ""}${apis ? ` \u00b7 ${apis} API client${apis > 1 ? "s" : ""}` : ""} \u00b7 ${gpus} WebGPU \u00b7 ${pledged.toFixed(1)} GB pledged`
    : `${all.length} device${all.length > 1 ? "s" : ""}${apis ? ` \u00b7 ${apis} API client${apis > 1 ? "s" : ""}` : ""} \u00b7 ${+pledged.toFixed(1)} GB pooled`;
  $("hdr-sum").innerHTML = `<b>${+pledged.toFixed(1)} GB</b> pooled`;
  $("peers-n").textContent = String(all.length);
  renderPool(pledged);
}
// the device list scrolls past four and a half rows (p2p.html #ap-devs): fade its bottom while more rows are below
function devsEdge() { const l = $("ap-devs"); l.classList.toggle("more", l.scrollTop < l.scrollHeight - l.clientHeight - 1); }
$("ap-devs").addEventListener("scroll", devsEdge, { passive: true });
if (typeof ResizeObserver === "function") new ResizeObserver(devsEdge).observe($("ap-devs"));
// The model card's side: what the room pools (one segment per device, in its colour, with a tick at
// each model's need), what this device lends (+/-), and the invite (QR, code, copy link).
function renderPool(pledged) {
  const devs = [{ name: myName, meta: myMeta }, ...[...members.values()].map((m) => ({ name: m.name || "device", meta: m.meta || {} }))].sort((x, y) => bySlot(x.name, y.name))
    .filter((d) => d.meta?.webgpu && d.meta?.contribGB && !d.meta?.api);
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
  devsEdge();
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
  loadEngine().catch(() => {});   // fetch the engine while the room fills; aiLoadShard reports a failure when it needs it
  $("join-screen").style.display = "none";
  $("room-screen").style.display = "flex";
  $("room-badge").style.display = "";
  document.body.classList.add("in-room");
  roomSince = performance.now();
  $("compute-open").hidden = false;
  $("room-badge").textContent = roomCode;
  $("room-h").textContent = `Room ${roomCode}`;
  $("side-code").textContent = roomCode;
  $("side-code").addEventListener("click", openShare);
  $("ap-qr").innerHTML = qrSVG(roomLink(), { size: 112 });
  // Chat | Code shows from the lobby on, so a visitor who came for Code sees where it is; Code stays
  // off (codeGate) until a model is running (?mock=code: at once, there is no model)
  if (isHost) $("host-controls").hidden = false;
  $("mode-bar").hidden = false; codeGate();
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
  const entry = { conn, name: name || conn.peer, meta: meta || {}, rtt: null, card: null, link: makeLink(), stripes: [], seen: performance.now() };
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
    if (!e && isHost && !roster.has(conn.peer)) return;   // already dropped (dropStaleNamesake)
    peerGone(conn.peer, e);
  });
  conn.on("error", () => {});
  return entry;
}

// a link is gone (closed, or dropped as silent): on the host the device left; workers wait for the roster
function peerGone(id, e) {
  conns.delete(id);
  if (isHost) {
    dropCard(id); members.delete(id); roster.delete(id); broadcastRoster();
    log("room", `${e?.meta?.api ? "API client " : ""}${e?.name || id} left`);
    apiPeerGone(id);
    aiPeerLeft(id, e?.name);
  } else if (id === PREFIX + roomCode) { log("room", "lost the link to the host"); hostGone(); }
  updateCluster();
}

function ensureCard(id, name, meta) {
  let card = cards.get(id);
  if (!card) {
    card = peerCard(id, name || id, meta || {}, false);
    cards.set(id, card);
    updateCluster();
    const who = meta?.api ? `API client ${name || id}` : name || id;
    log("room", `${who} joined`);
    const noted = $("ai-output").style.display === "block";
    if (noted) sysNote(`${who} joined${meta?.contribGB && meta?.webgpu ? ` with ${meta.contribGB} GB` : ""}`, "join");
    // the chat on screen says it in the log already: no toast on top of that same line
    if (!(noted && $("ai-output").getClientRects().length)) presence(name || id, true);
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
window.pooledDebug = window.swarmDebug = () => [...conns].map(([id, e]) => ({ id, name: e.name, chans: e.link?.chans.filter((c) => c.readyState === "open").length ?? 0, sent: e.link?.sent ?? 0, recv: e.link?.recv ?? 0, ka: e.link?.kaSent ?? 0 }));
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
  if (e) e.seen = performance.now();   // any message counts as a sign of life (the ping loop drops silent links)
  if (!d || typeof d.t !== "string") return;
  if (d.t.startsWith("ai-")) { aiOnData(from, d); return; }
  switch (d.t) {
    case "hello":
      // one protocol per room: a tab from an older or newer deploy is told to reload
      if (d.v !== PROTOCOL) { versionRefused(from, d); break; }
      versionSeen.delete(from);   // back on the same version (a reload): its bye counts again
      // a peer picks its own name: keep it a short plain string (it is also escaped wherever it is shown)
      d.name = String(d.name ?? from).replace(/[\u0000-\u001f\u007f<>"'`&]/g, "").trim().slice(0, 40) || String(from).slice(0, 8);
      // an API client (`pooled serve`, cli/): an ask-only guest with no layers; the host may refuse it
      d.meta = helloMeta(d.meta, isHost);   // on a guest, meta.api is the host saying it serves API clients
      // names are the room's keys (colours, layers, load progress, the plan): the host makes a taken one
      // unique ("laptop 2"), and the device takes the name the roster gives it
      if (isHost) {
        // the name is held by a device that has been quiet for a second: ping it and decide in a moment
        const quiet = !probedHellos.has(d) && quietNamesake(d.name, from, roster, heardOf, performance.now());
        if (quiet) {
          const since = performance.now();
          sendTo(quiet, { t: "ping", ts: since });
          probedHellos.set(d, since);
          setTimeout(() => { if (conns.get(from) === e) onData(from, d); }, NAME_PROBE_MS);
          break;
        }
        if (probedHellos.has(d)) dropStaleNamesake(d.name, from, probedHellos.get(d));
        d.name = uniqueName(d.name, from, myName, roster);
      }
      e.name = d.name; e.meta = d.meta;
      if (d.meta?.api && isHost && !apiWelcome(from, d)) break;
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
      if (versionSeen.has(from)) break;   // a version mismatch this tab already explained in its own words
      toast(d.reason);
      log("room", d.reason);
      if (from === PREFIX + roomCode) { $("room-over").hidden = false; $("room-over-h").textContent = "Room over"; $("room-over-why").textContent = d.reason; }
      break;
    case "roster": {
      // the host's view of the room: draw a card per device, no mesh connections
      // colours follow the host's order (the host first, then join order), so a device has the
      // same colour on every screen (they used to go by first-seen order, which put "me" first)
      const rename = renameTo(d.members, peer.id, myName, from === PREFIX + roomCode);
      if (rename) renameSelf(rename);
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
      apiPanel();   // the API clients in the roster: the Serve API dot and list
      break;
    }
    case "ping": sendTo(from, { t: "pong", ts: d.ts }); break;
    case "pong": {
      e.missed = 0;
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

// A peer on another protocol version (its hello): say which side is older and who should reload,
// here and (as the bye reason) on its screen. The link stays up but the peer never joins the room.
const versionSeen = new Set();
function versionRefused(from, d) {
  const theyHost = !isHost && from === PREFIX + roomCode;
  const name = String(d.name ?? "").replace(/[\u0000-\u001f\u007f<>"'`&]/g, "").trim().slice(0, 40) || "A device";
  const { local, remote } = versionMismatch({ mine: PROTOCOL, theirs: d.v, name, theyHost, me: myName, iAmHost: isHost });
  versionSeen.add(from);
  sendTo(from, { t: "bye", reason: remote });
  toast(local, { kind: "error" });
  log("room", local);
  if (theyHost) { $("room-over").hidden = false; $("room-over-h").textContent = "Different version"; $("room-over-why").textContent = local; }
}

// the host: a device asking for the name of a quiet one is pinged first (room/liveness.js); if the
// quiet one did not answer by the time the hello is looked at again, its old link is dropped
const probedHellos = new WeakMap();   // hello message -> when its namesake was pinged
const heardOf = (id) => lastHeard(conns.get(id));
function dropStaleNamesake(name, id, since) {
  const now = performance.now();
  for (const rid of staleNamesakes(name, id, roster, heardOf, since)) {
    const e = conns.get(rid);
    log("room", `${name} is back under a new link: dropping its old one, silent for ${e ? Math.round((now - lastHeard(e)) / 1000) : "?"} s`);
    peerGone(rid, e);
    try { e?.conn.close(); } catch {}
  }
}
// a device: the host listed this device under another name (the one asked for was taken)
function renameSelf(name) {
  toast(`${myName} was taken in this room: this device is ${name}`);
  log("room", `the name ${myName} is taken here: this device is ${name}`);
  myName = name;
  $("name-input").value = name;
  if (!VQ.get("embed")) try { sessionStorage.setItem(NAME_KEY, name); } catch {}
  const card = document.querySelector(".peer-card.self");
  if (card) { card.dataset.name = name; paintCard(card, name, myMeta, true); }
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
// A device that vanished without a clean close is dropped once its link has been silent too long
// (room/liveness.js: longer for phones and the host, never while it loads its layers).
// The host also drops an API client (`pooled serve`) that stopped answering: a bridge that was killed
// (kill -9, a crash, a laptop lid) never says "leaving", and its data channel can take over a minute
// to close, while its running answer holds the room. Counted in pings, not wall time, so a host tab
// that was busy or throttled for a while does not drop a live bridge (the interval fires once after).
const API_MISSED_PINGS = 6;   // ~15 s
let pingAt = performance.now();
setInterval(() => {
  const now = performance.now(), late = now - pingAt > 2 * PING_MS;
  pingAt = now;
  const starting = ai.busy === true && !$("ai-panel").classList.contains("online");
  for (const [id, e] of [...conns]) {
    if (e.meta?.api) {
      if (!isHost) continue;
      e.missed = (e.missed || 0) + 1;
      if (e.missed > API_MISSED_PINGS) {
        log("room", `API client ${e.name || id} stopped answering; dropping it`);
        try { e.conn.close(); } catch {}
      }
      continue;
    }
    if (late) { e.seen = now; continue; }
    const loading = ai.loadingShard || (isHost && midLoad({ starting, inChain: ai.chain.includes(id), ready: ai.readyPeers.has(id) }));
    const heard = lastHeard(e);
    if (isSilentGone({ now, heard, toHost: id === PREFIX + roomCode, phone: isPhoneMeta(e.meta), loading })) {
      log("room", `${e.name || id} stopped answering for ${Math.round((now - heard) / 1000)} s: dropping it`);
      try { e.conn.close(); } catch {}
    }
  }
  broadcastAll({ t: "ping", ts: now });
}, PING_MS);

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
  // on a phone the status line sits below the fold: bring it to where the user is looking
  $("join-status").scrollIntoView({ block: "center", behavior: "smooth" });
}
// --- join / create ---
async function start(create, resume = null) {
  myName = resume?.name || $("name-input").value.trim() || (create ? "host" : "peer") + "-" + rand(2);
  if (!VQ.get("embed")) try { sessionStorage.setItem(NAME_KEY, myName); } catch {}   // a virtual device's iframe shares the tab's storage
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
  // PeerJS is a deferred script from cdn.jsdelivr.net (p2p.html): without it the page still renders, so say why nothing connects
  if (typeof Peer !== "function") { joinFailed("couldn't load the connection library from cdn.jsdelivr.net (offline, or blocked by an extension or network). Reload to try again"); return; }
  // host claims the well-known id for the code; joiners get random ids
  peer = new Peer(create ? PREFIX + code : undefined, { debug: 1, config: ICE, ...SIGNAL_OPTS });

  peer.on("open", () => {
    isHost = create;
    roomCode = code;
    if (create) { enterRoom(); if (resume) resumeHost(resume); return; }
    // joiner: connect to host
    $("join-status").textContent = "Reaching the other devices…";
    const conn = peer.connect(PREFIX + code, { reliable: true });
    // "still connecting" after a few seconds; more time while the two devices are still finding a path
    const t0 = performance.now(), me = peer;
    const timeout = setInterval(() => {
      // failed already (peer.on("error")), or a newer Join press owns the screen now
      if (peer !== me || !$("join-btn").disabled) { clearInterval(timeout); return; }
      const step = joinStep(performance.now() - t0, conn.peerConnection?.iceConnectionState);
      if (step.fail) { clearInterval(timeout); try { conn.close(); } catch {} joinFailed(step.fail); }   // closed, so a late open can't pull a failed join into the room
      else if (step.status) $("join-status").textContent = step.status;
    }, 1000);
    conn.on("open", () => {
      clearInterval(timeout);
      wire(conn, "host", undefined, true);
      let died = null;
      if (!VQ.get("embed")) try { const c = JSON.parse(localStorage.getItem("pooled-crumb") || "null"); if (c && Date.now() - c.t < 10 * 60 * 1000) died = { during: c.s, ago: Math.round((Date.now() - c.t) / 1000) }; } catch {}
      conn.send({ t: "hello", name: myName, meta: myMeta, died, v: PROTOCOL });
      // put the room in the address bar (a typed code never was), so a reload joins it again like a link,
      // under the same name: the host re-seats a device's layers by name (aiRejoin)
      if (!VQ.get("embed")) try { history.replaceState(history.state, "", roomLink()); } catch {}
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
      // the host says it answers API clients (docs/protocol.md "API clients"); not part of myMeta,
      // which the roster shows everyone
      // api: 2 = it also answers v2 asks (tools, structured output); ctx: its context size now
      conn.send({ t: "hello", name: myName, meta: isHost ? { ...myMeta, api: 2, ctx: ctxMax() } : myMeta, v: PROTOCOL });
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
    if ($("room-screen").style.display === "flex") {   // in the room already: not a join failure
      // a link to a device that left (the 3 s host-return retries, a chain neighbour): those callers
      // say what it means themselves, and a line per retry would flood the log
      if (err.type === "peer-unavailable") { console.warn("peer", err.message); return; }
      const text = peerErrorText(err.type, { inRoom: true });
      if (text === peerErrorShown.text && Date.now() - peerErrorShown.t < 30000) return;   // "network" then "disconnected" read the same
      peerErrorShown.text = text; peerErrorShown.t = Date.now();
      log("room", text);
      if (peerErrorLoud(err.type)) toast(text, { kind: "error" });
      return;
    }
    joinFailed(peerErrorText(err.type));
  });
}

const peerErrorShown = { text: "", t: 0 };
let wakeLock = null, awakeVideo = null;
let awakeMode = null;
let overClosed = false;   // the Room over card was closed with the room still over (see roomOver)   // how this screen stays on: "lock" (the Wake Lock API), "video", "none"; null before the first try
function awakeStatus(s) { const el = $("awake"); if (el && myMeta?.phone) el.textContent = s; }
async function keepAwake() {
  // 1. the real API (iOS 16.4+, must be called from a tap)
  try {
    if (!wakeLock && navigator.wakeLock) {
      wakeLock = await navigator.wakeLock.request("screen");
      wakeLock.addEventListener("release", () => { wakeLock = null; awakeMode = awakeVideo && !awakeVideo.paused ? "video" : "none"; awakeStatus("screen lock: released"); compute.refresh(); });
      awakeMode = "lock";
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
    if (!wakeLock) { awakeMode = "video"; awakeStatus("screen stays awake (video) \u2713"); }
  } catch (e) {
    // the setting is Auto-Lock on an iPhone, the screen timeout (Settings > Display) on Android
    if (!wakeLock) awakeMode = "none";
    if (!wakeLock) awakeStatus(`This screen can\u2019t stay awake on its own: ${myMeta?.ua === "iPhone" ? "set Auto-Lock to Never" : "set the screen timeout to its longest (Settings \u203a Display)"}`);
  }
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
    bytes: ai.range || mineDeal ? ai.shardBytes || 0 : 0,   // this device's share of the weights, once its load has started
    awake: awakeMode, ios: myMeta?.ua === "iPhone" || myMeta?.ua === "iPad",
    over: roomOver(),
  };
}
// the room ended, or the host is gone and may come back: the Room over card says which. Closing the
// card doesn't bring the room back, so it still counts until the host is connected again.
function roomOver() {
  const card = { final: $("room-over-h").textContent === "Room over", why: $("room-over-why").textContent };
  if (!$("room-over").hidden) return card;
  if (overClosed && !card.final && conns.has(PREFIX + roomCode)) overClosed = false;
  return overClosed ? card : null;
}
$("room-over-close").addEventListener("click", () => { overClosed = true; });
// Serve API: the header's black button opens the dark page, this device's layers and passes beside
// the endpoint and how to connect (the API half is apiPanel's, below)
const compute = computeScreen({ state: computeState, keepAwake, newRoom: () => $("room-over-new").click(), onShow: apiShown });
$("compute-open").addEventListener("click", () => compute.open());
// the button's mark says whether this device is working ("on" while it holds layers: the dots breathe),
// its tooltip which layers; its name stays "Serve API"
function deviceMark() {
  const s = computeState(), b = $("compute-open");
  const on = s.lo != null && s.hi != null && s.hi > s.lo && s.phase !== "idle";   // lo == hi holds nothing
  const tip = on ? (s.phase === "loading" ? `This device is loading layers ${s.lo + 1}\u2013${s.hi}` : `This device holds layers ${s.lo + 1}\u2013${s.hi}`) : "Use this room from your own tools";
  if (b.dataset.tip === tip && b.classList.contains("on") === on) return;
  b.classList.toggle("on", on);
  b.dataset.tip = tip;
}
setInterval(deviceMark, 1000);
$("create-btn").addEventListener("click", () => { keepAwake(); start(true); });
// (auto-rejoin removed: the user prefers to see what happened)
// Join only with a whole code: the greyed button and Enter in a short code do nothing but put the cursor back
const codeOk = () => /^[A-Z0-9]{4,6}$/i.test($("code-input").value.trim());
$("join-btn").addEventListener("click", () => { if (!codeOk()) { $("code-input").focus(); return; } keepAwake(); start(false); });
$("code-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && codeOk()) start(false); });
const codeReady = () => {
  $("join-btn").classList.toggle("ready", codeOk());
  $("code-input").parentElement.classList.toggle("full", $("code-input").value.length >= 4);
};
// four boxes, four characters: letters and digits only; the fourth one hands off to Join (on a
// phone that also closes the keyboard), so no box waits for a fifth
$("code-input").addEventListener("input", (e) => {
  const el = e.target, v = el.value.toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 4);
  if (el.value !== v) el.value = v;
  codeReady();
  if (v.length === 4 && e.isTrusted && document.activeElement === el) { $("join-btn").focus(); el.parentElement.scrollIntoView({ block: "nearest" }); }
  else codeInView();
});
// a phone's keyboard shrinks the screen after the boxes took focus: keep the four boxes whole in view,
// not half under the header or the keyboard (the input is an overlay the browser scrolls to by its caret)
const codeInView = () => { if (document.activeElement === $("code-input")) $("code-input").parentElement.scrollIntoView({ block: "nearest" }); };
visualViewport?.addEventListener("resize", codeInView);
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
// the overlays (Invite, Room card, Room over, Leave) are modal: while one is open, the page behind it is
// inert, so Tab cycles inside the sheet and nothing behind the blur takes focus or clicks
const overlays = [...document.querySelectorAll(".overlay")];
let modalReturn = null;   // what had focus before an overlay that opened on its own took it
function syncModal() {
  const open = overlays.some((o) => !o.hidden);
  for (const el of document.querySelectorAll("body > header, #join-screen, #room-screen")) el.inert = open;
  // the compute screen (a lending device's screen) sits above the overlays: while it is up, they wait
  // underneath, out of the Tab order, and take focus only once it closes
  const computing = $("compute-screen")?.hidden === false;
  for (const o of overlays) o.inert = computing;
  // one that opened on its own (Room over) takes focus from the page it now covers
  const ae = document.activeElement;
  // (focus left on a sheet that just closed counts as outside, e.g. Invite closed over Room over)
  if (open && !computing && !overlays.some((o) => !o.hidden && o.contains(ae))) {
    if (ae && ae !== document.body && !ae.closest?.(".overlay")) modalReturn = ae;
    overlays.find((o) => !o.hidden).querySelector("button")?.focus({ preventScroll: true });
  }
  // and gives it back when it closes (host back, a redeal, its close button), unless focus moved on
  if (!open) {
    const back = modalReturn; modalReturn = null;
    if (back?.isConnected && (!ae || ae === document.body || ae.closest?.(".overlay"))) back.focus({ preventScroll: true });
  }
}
for (const o of [...overlays, $("compute-screen")].filter(Boolean)) new MutationObserver(syncModal).observe(o, { attributes: true, attributeFilter: ["hidden"] });
// a closed sheet hands focus back to its opener, or, if another sheet is still up, leaves it for later
function focusBack(el) { if (overlays.some((o) => !o.hidden)) modalReturn = el; else el.focus({ preventScroll: true }); }
function closeShare() { $("share").hidden = true; syncModal(); if (document.body.classList.contains("in-room")) focusBack($("share-btn")); }
$("share-close").addEventListener("click", closeShare);
$("share").addEventListener("click", (e) => { if (e.target === $("share")) closeShare(); });
$("room-over-close").addEventListener("click", () => { $("room-over").hidden = true; });
$("share-copy").addEventListener("click", copyRoomLink);
$("share-native").addEventListener("click", () => navigator.share?.({ title: "Join my Pooled room", text: `Room ${roomCode}: add this device to the AI model we run together`, url: roomLink() }).catch(() => {}));
// the logo leads home. In a room it asks first: it sits in the thumb's corner on a phone, and
// leaving ends the room for everyone (the host) or takes this device's layers with it
const logoLink = document.querySelector(".logo a");
logoLink.addEventListener("click", (e) => {
  if (!document.body.classList.contains("in-room")) return;
  if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;   // a new tab or window keeps the room open: no need to ask
  e.preventDefault();
  $("leave-h").textContent = `Leave room ${roomCode}?`;
  $("leave-why").textContent = isHost
    ? (conns.size ? "The room ends for the other devices: this tab holds the conversation and the model's first and last layers." : "The room and its model close.")
    : `If this device holds some of the model's layers, the room has to re-deal them. You can join again with the code ${roomCode}.`;
  $("leave").hidden = false;
  $("leave-stay").focus({ preventScroll: true });
});
// (syncModal first: the header stays inert until it runs, and an inert logo would not take focus)
function closeLeave() { $("leave").hidden = true; syncModal(); focusBack(logoLink); }
$("leave-stay").addEventListener("click", closeLeave);
$("leave").addEventListener("click", (e) => { if (e.target === $("leave")) closeLeave(); });
$("leave-go").addEventListener("click", () => { location.href = logoLink.href; });
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
else try { const n = sessionStorage.getItem(NAME_KEY); if (n) $("name-input").value = n; } catch {}   // this tab's name from before a reload
if (+VQ.get("vgb") > 0) { $("join-gb").value = +VQ.get("vgb"); joinGbChosen = true; }
if (backAsHost && Date.now() - backAsHost.t < 60000 && !(linkCode && linkCode !== backAsHost.code)) {
  metaPromise.then(() => { if (!peer) start(true, backAsHost); });
} else if (linkCode) {
  $("code-input").value = linkCode; codeReady();
  joinWait(true, `Joining room ${linkCode}`);
  $("join-status").textContent = "Checking this device\u2026";
  metaPromise.then(() => { if (!peer) start(false); });
} else if (window.pooledEarly) {
  // a tap on Start a room or Join before this module (the whole engine) had loaded: p2p.html's early
  // script kept it and showed the wait state; run it now instead of dropping it
  const early = window.pooledEarly;
  if (early === "create") { keepAwake(); start(true); }
  else if (codeOk()) { keepAwake(); start(false); }
  else joinWait(false);
}
window.pooledWired = true;

// ================= distributed inference =================

// ---- on-disk cache of weight ranges (Cache API): a second start skips the download ----
let weightCache = null, cacheHits = 0;
// The names "swarmllm-weights-v1", "https://weights.swarmllm.ai/" (a cache key namespace, never
// fetched) and the "x-swarm-len" header are from before the rename to Pooled. They stay so weights
// people already downloaded keep working; the Cache API is per site, so pooled.run starts empty anyway.
// Keys and per-model bookkeeping live in room/weightcache.js.
async function getWeightCache() {
  if (weightCache !== null) return weightCache;
  try { weightCache = await caches.open(CACHE_NAME); } catch { weightCache = false; }
  return weightCache;
}
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
  let r;
  try { r = await fetch(url, { headers: { Range: `bytes=${lo}-${hi}` } }); }
  catch { throw new FetchError(0, url); }   // offline, CORS or a blocked host: no status to go on
  if (r.status !== 206) throw new FetchError(r.status, url);
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
  const prefix = CACHE_PREFIX + encodeURIComponent(url) + "/";
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

// ---- converted weights on disk (room/convertedcache.js, OPFS): a second load of the same layers skips
// the CPU conversion (K-quant -> Q8, BF16/Q5_0 -> f32, the embedding's repack). Keyed by model URL,
// its pinned revision, the GGUF header and engine/gguf.js itself, so any of those changing starts
// fresh. ?wcache=0 turns it off (A/B); ?wcacheverify=1 also checks each entry's payload hash.
const WCACHE = new URLSearchParams(location.search).get("wcache") !== "0";
const WCACHE_VERIFY = new URLSearchParams(location.search).get("wcacheverify") === "1";
async function useConvertedCache(G, url) {
  if (!WCACHE) { G.entryCache = null; return null; }
  return attachBrowserWeightCache(G, url, { srcUrl: new URL("./engine/gguf.js", import.meta.url).href, verify: WCACHE_VERIFY });
}
async function convertedSummary(c, t0) {
  if (!c) return;
  await c.flush();   // background writes done before the engine is built
  if (!(c.stats.hit || c.stats.write || c.stats.full)) return;
  const msg = `${c.summary()}; this device's layers loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s`;
  console.info(msg);
  if (c.stats.hit) log("room", `${myName}: ${msg}`);
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
  gone: new Set(),       // the host: chain ids whose link closed, so a same-id rejoin is re-seated (aiRejoin)
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
  settings: { persona: "default", sampling: "creative", thinking: false, length: "normal", apiAllow: true },
  apis: new Map(),       // host: API clients (`pooled serve`): peer id -> { name, client, answered }
  apiKicked: new Set(),  // host: API clients disconnected this session (a reconnect is refused)
  apiCache: new AnswerCache(8),   // host: text -> sampled ids of recent API answers (room/api.js), v1 asks
  apiTurns: new TurnCache(),      // host: (history, answer) -> sampled ids of recent API answers, v2 asks
  apiEnc: new EncodeCache(),      // host: text -> ids, for rendering v2 asks (clients resend their whole history)
  apiProf: null,                  // host: the loaded model's template profile (room/conversation.js), for v2 asks
  apiRun: null,          // host: the API request being answered { rid, from, ac }
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
  // the bubble stays out of the live region while tokens stream in (#ai-output is role=log);
  // chatBotEnd swaps in a fresh bubble so a screen reader announces the finished answer once
  m.innerHTML = `<div class="who"><span class="wn"></span><span class="wd" aria-hidden="true"><i></i><i></i><i></i></span></div><div class="bubble" aria-hidden="true"></div>`;
  m.querySelector(".wn").textContent = shortName(ai.model || $("ai-model").value) || "room";
  // until the first token: the working line (the first piece replaces it)
  const n = Object.keys(ai.layersByName || {}).length;
  m.querySelector(".bubble").append(working({ lead: n ? `Reading your message on ${n} device${n > 1 ? "s" : ""}` : "" }));
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
  // a new bubble node (not the streamed one un-hidden) is what the log announces
  const fresh = document.createElement("div"); fresh.className = "bubble";
  botEl.querySelector(".bubble").replaceWith(fresh);
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
  if (!navigator.clipboard) { toast("this browser can't copy here"); return Promise.resolve(false); }
  return navigator.clipboard.writeText(text).then(() => {
    if (!btn) { toast(`${what} copied`); return true; }
    btn.classList.add("done"); btn.setAttribute("aria-label", "Copied");
    clearTimeout(btn._t); btn._t = setTimeout(() => { btn.classList.remove("done"); btn.setAttribute("aria-label", `Copy the ${what}`); }, 1600);
    return true;
  }, () => { toast("couldn't copy"); return false; });
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
  $("card-close").focus({ preventScroll: true });
}
function closeCard() { $("card").hidden = true; syncModal(); focusBack($("room-menu").querySelector("summary")); }
async function cardBlob() { return new Promise((res) => $("card-canvas").toBlob(res, "image/png")); }
$("card-btn").addEventListener("click", () => { $("room-menu").open = false; openCard(); });
$("card-close").addEventListener("click", closeCard);
$("card").addEventListener("click", (e) => { if (e.target === $("card")) closeCard(); });
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

async function aiLoadShard(modelKey, range, hasEmbed, hasHead, ctx = maxSeqFor(modelKey), kv = kvModeFor(modelKey, KV_ASK)) {
  const M = MODELS[modelKey];
  ai.shardBytes = 0;   // until this load's first progress says how big the new range is
  aiLoading(true, `loading layers ${range[0]}\u2013${range[1] - 1} of ${M.label.split("\u00b7")[0].trim()}`);
  aiStatus("requesting GPU\u2026");
  mascot("Grabbing my slice of the model… hang tight.");
  // a previous attempt in this tab still owns its weights: release them first, or the
  // second load doubles GPU memory and every buffer after the limit comes back invalid
  if (ai.device) { try { ai.device.destroy(); } catch {} ai.device = null; ai.engine = null; }
  ai.firstGpuError = null;
  ai.peerBytes = 0; ai.netBytes = 0; cacheHits = 0;   // per load: a count left from an earlier load in this tab mislabels the status
  if (!Qwen35Engine) { aiStatus("loading the inference engine\u2026"); await loadEngine(); }
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("This browser has no WebGPU, so this device can't hold layers. Open the room in a recent Chrome, Edge or Safari, or run the model from another device");
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
    if (hasEmbed || hasHead) { ai.tok = makeTokenizer(await (await fetch(M.tok)).json()); apiModelLoaded(); }
  }

  const onProg = (done, total) => {
    aiProgress(done, total);
    // name where the bytes are coming from right now: the network, devices in the room, or this device's cache
    const gb = (b) => (b / 2 ** 30).toFixed(1) + " GB";
    aiStatus(ai.netBytes ? `downloading weights\u2026${cacheHits ? ` (${gb(cacheHits)} was already on this device)` : ""}`
      : ai.peerBytes ? `getting weights from devices in the room\u2026`
      : cacheHits ? `loading weights from this device's cache\u2026` : `downloading weights\u2026`);
    ai.myPct = total ? done / total * 100 : 0;
    ai.shardBytes = total;   // the lending screen shows how much this device holds
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
    if (ai.startFailed) throw new Error(ai.startFailed);   // the start was stopped (aiStartStopped): no point loading the rest
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
      apiModelLoaded();
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
    const wc = await useConvertedCache(G, M.gguf), tw = performance.now();
    const weights = await qwen35Weights(G, rangeBytesOf(M.gguf), opts, (done) => onProg(done, total),
      (e, name) => gpuUploadEntry(ai.device, e, name === GGML_EMBED));   // straight to the GPU, RAM stays flat
    await convertedSummary(wc, tw);
    aiStatus("building GPU pipelines (compiling shaders)\u2026");
    ai.engine = await Qwen35Engine.create({
      device: ai.device, meta: G.meta, weights, vocab: G.tensors[GGML_EMBED]?.shape?.[0],
      layerRange: range, hasEmbed, hasHead, maxSeq: ctx,
      coopWG: ai.tune?.wg, coopRows: ai.tune?.rows,
      // the room's settings (engine/preset.js: 16 batch columns, the small draft head, fused kernels, GPU
      // sampling, and the ?flags that change them). The benchmarks and profilers build their engines from
      // the same preset, so their numbers come from these settings. Prefill options are not set there:
      // every device takes the engine's defaults, so host and workers agree.
      ...roomQwen35Options(location.search),
      // ?kv=q8 on the host: int8 KV cache. The host decides for every device and sends its choice with
      // ai-load (room/models.js kvModeFor, kvForLoad), so this overrides the preset's own ?kv reading.
      kvQ8: kv === "q8",
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
    const wc = await useConvertedCache(G, M.gguf), tw = performance.now();
    const weights = await ggufWeights(G, rangeBytesOf(M.gguf), opts, (done) => onProg(done, total),
      (e, name) => gpuUploadEntry(ai.device, e, name === GGML_EMBED));
    await convertedSummary(wc, tw);
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
  // batched draft-cache fill and refill, next step's first draft in the verify's pass (engine/preset.js)
  applyRoomFlags(ai.engine, location.search);
  ai.range = range;
  ai.model = modelKey;
  // the load card stays up (its "Starting" mark once every device has its layers) until the room
  // is online: the host takes it down in aiMaybeReady, a worker on ai-ready-all. Taking it down here
  // showed the model picker for the seconds between this device's layers and the room's.
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
    ai.startFailed = null;
    ai.readyPeers = new Set();
    ai.ckptHeld = new Map();
    ai.teleBy = new Map();
    const modelKey = $("ai-model").value;
    const M = MODELS[modelKey];
    // context for this room: the model's default, or ?ctx=N up to its cap (room/models.js CTX); every device builds its engine with it
    const ROOM_CTX = maxSeqFor(modelKey, +new URLSearchParams(location.search).get("ctx") || 0);
    const ROOM_KV = kvModeFor(modelKey, KV_ASK);   // KV cache format for every device: f16, or int8 with ?kv=q8
    // devices without WebGPU join as ask-only guests: they get the chat, not layers
    ai.chain = [...conns.keys()].filter((id) => conns.get(id)?.meta?.webgpu).sort();
    ai.leftOut = new Set();
    ai.plan = new Map();                      // name -> load message, so a reloaded device can be re-seated
    ai.gone = new Set();                      // chain ids whose link closed (aiPeerLeft), for a same-id rejoin
    ai.chainNames = ai.chain.map((id) => conns.get(id)?.name || id);
    const n = ai.chain.length + 1;
    let L, layerBytes, embedBytes, cfg = null;
    if (M.kind === "qwen35") {
      aiStatus("reading model index… (11 MB)");
      ai.G = await fetchGGUFHeader(M.gguf);
      ai.GModel = modelKey;
      L = ai.G.meta["qwen35.block_count"] - (ai.G.meta["qwen35.nextn_predict_layers"] || 0);
      layerBytes = qwen35ShardBytes(ai.G, { lo: 0, hi: 4, hasEmbed: false, hasHead: false }) / 4
        + ROOM_CTX * kvBytesPerLayerPos(ai.G.meta, ROOM_KV);   // the attention layers' KV cache at this room's context
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
      const sp = planForSpeed(L, caps.map((c) => Math.floor(c / layerBytes)), [ai.msPerLayer.get(myName), ...ai.chain.map((id) => ai.msPerLayer.get(nameOf(id)))],
        [isPhoneMeta(myMeta), ...ai.chain.map((id) => isPhoneMeta(conns.get(id)?.meta))]);
      const keep = sp.used.filter((i) => i > 0).map((i) => i - 1);
      ai.leftOut = new Set(ai.chain.filter((_, i) => !keep.includes(i)));
      ai.chain = keep.map((i) => ai.chain[i]);
      ai.chainNames = ai.chain.map(nameOf);
      assigned = sp.used.map((i) => sp.assigned[i]);
      ranges = sp.used.map((i) => sp.ranges[i]);
      caps = sp.used.map((i) => caps[i]);
    } else {
      // by memory, but phones hold layers only when the computers cannot hold the model
      // (room/plan.js phonesToLeaveOut); ?phonelayers=1 deals them layers anyway
      const nameOf = (id) => conns.get(id)?.name || id;
      const out = PHONE_LAYERS ? [] : phonesToLeaveOut(L, caps.map((c) => c / layerBytes), [false, ...ai.chain.map((id) => isPhoneMeta(conns.get(id)?.meta))]);
      if (out.length) {
        ai.leftOut = new Set(out.map((i) => ai.chain[i - 1]));
        ai.chain = ai.chain.filter((id) => !ai.leftOut.has(id));
        ai.chainNames = ai.chain.map(nameOf);
        caps = caps.filter((_, i) => !out.includes(i));
        log("room", `${[...ai.leftOut].map(nameOf).join(", ")} ask${ai.leftOut.size > 1 ? "" : "s"} without holding layers: the computers hold the whole model, and a phone's layer would slow every token`);
      }
      ({ assigned, ranges } = planSplit(L, caps));
    }
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
        t: "ai-load", v: PROTOCOL, model: modelKey, range: ranges[i + 1], ctx: ROOM_CTX, kv: ROOM_KV,
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
    log("room", `${M.label} — layer split ${$("ai-split").value === "speed" ? "for speed" : "by pledge"}${ROOM_KV === "q8" ? ", int8 KV" : ""}: ${splitDesc}`);
    ai.loadingShard = true;
    try { await aiLoadShard(modelKey, ranges[0], true, true, ROOM_CTX, ROOM_KV); } finally { ai.loadingShard = false; }
    if (ai.startFailed) throw new Error(ai.startFailed);   // a device failed to load its layers while this one loaded
    if (ai.degraded) aiLoading(false);        // a device left while this one loaded: the Re-deal button is on the panel
    aiStatus(n === 1
      ? `solo: all ${L} layers local — ready`
      : `layers ${ranges[0][0]}–${ranges[0][1] - 1} ready · syncing with ${ai.chain.length} device${ai.chain.length > 1 ? "s" : ""}…`);
    ai.fed = [];                              // fresh engines everywhere: nothing cached yet
    ai.pendingCtl = {};
    ckptClear();
    ai.ckptRestoring = true;                  // ...except checkpoints saved to disk before a reload
    const back = (await ckptRestore().finally(() => { ai.ckptRestoring = false; })).length;
    if (back) log("room", `${back} saved checkpoint${back > 1 ? "s" : ""} read back from disk`);
    aiMaybeReady();
  } catch (err) {
    aiStartStopped(stopReason(ai.startFailed, err));   // a device that failed first is the reason, not what this load hit after
  }
}
// A start that cannot finish: this device failed, or a device in the chain could not load its layers
// (ai-error with load). Every screen goes back to the model picker with Start on, so the room can try
// again (the other screens are told with ai-start-failed). While this device still loads its own layers
// the load stops at its next tensor (the pacer) and aiStart's catch lands here.
function aiStartStopped(why) {
  ai.startFailed = null;
  clearInterval(ai.progTimer);
  aiLoading(false);
  ai.engine = null;
  ai.chain = []; ai.chainNames = []; ai.plan = null;   // nothing to re-seat or re-deal
  failWaiters(new Error(why));
  $("ai-panel").classList.remove("online");
  aiStatus("failed: " + why);
  ai.busy = false;
  $("ai-start").disabled = false;
  $("ai-model").disabled = false;
  updateCluster();
  broadcastAll({ t: "ai-start-failed", why });
}
function aiLoadFailed(from, name, message) {
  const why = `${name} couldn't load its layers (${message})`;
  toast(`${why}: back to the model picker`);
  if (stopWhen(ai.loadingShard) === "defer") { ai.startFailed = why; aiStatus(`${why}: stopping the start…`); }
  else aiStartStopped(why);
}

// a worker whose start was stopped: its layers are not needed, free them (the next ai-load deals afresh)
function workerStopped() {
  if (ai.role !== "worker") return;
  ai.role = null; ai.range = null; ai.engine = null;
  try { ai.device?.destroy(); } catch {}
  ai.device = null;
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
  ai.gone.add(id);   // it may come back under the same peer id (the ping loop dropped a live but stalled tab): aiRejoin
  // left before the room came online: drop the load card so the panel's Re-deal button shows
  // (while this device still loads, aiStart does it once its layers are in)
  if (!ai.loadingShard && !$("ai-panel").classList.contains("online")) aiLoading(false);
  // with disk copies the index stays: a rejoin keeps what the device reads back (ckptRejoin, then
  // ckptPrune on its ai-ready), a re-deal clears it and reads it back (aiStart)
  ai.fed = null; if (!ckptDisk) ckptClear();
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
  sendTo(id, { t: "ai-ready-all", model: ai.model, label: MODELS[ai.model]?.label, ctx: ctxMax() });
  if (ai.visibility === "all" && ai.transcript.length) sendTo(id, { t: "ai-history", items: ai.transcript.slice(-20) });
  offerRedealForNewcomers();
}

// a device that left comes back to its slot: a reloaded tab with a new peer id, or a tab the ping loop
// dropped while it was stalled (suspended, a long task), which reconnects under the same peer id
function aiRejoin(newId, name) {
  if (ai.role !== "host" || !ai.plan?.has(name)) return;
  const i = ai.chainNames.indexOf(name);
  if (i < 0) return;
  if (ai.chain[i] === newId ? !ai.gone.has(newId) : ai.chain.includes(newId)) return;
  const oldId = ai.chain[i];
  ai.chain[i] = newId;
  ai.readyPeers.delete(oldId);
  ai.gone.delete(oldId); ai.gone.delete(newId);
  const { msg } = ai.plan.get(name);
  const fresh = { ...msg, next: i + 1 < ai.chain.length ? ai.chain[i + 1] : "host", host: peer.id };
  if (i > 0) sendTo(ai.chain[i - 1], { t: "ai-next", next: newId });
  sendTo(newId, fresh);
  ai.fed = null; ckptRejoin();              // its fresh engine holds only what it reads back from disk
  log("room", `${name} came back — reloading its layers`);
  aiStatus(`${name} reconnected, reloading its layers…`);
  $("ai-row").style.display = ai.readyPeers.size >= ai.chain.length ? "flex" : "none";
}
function aiMaybeReady() {
  if (ai.role !== "host" || !ai.engine || ai.ckptRestoring) return;   // (aiStart calls it again once the restore is done)
  ckptPrune();
  if (ai.readyPeers.size < ai.chain.length) return;
  const n = ai.chain.length + 1;
  ai.degraded = false;
  ai.busy = false;
  showRedeal(false);
  aiStatus(`cluster online · ${n} device${n > 1 ? "s" : ""}, ${ai.cfg.num_hidden_layers} layers split ${n} ways`);
  clearInterval(ai.progTimer);
  if ($("load-card").classList.contains("on")) aiLoading(false);   // (a device back from a reload also lands here, with the chat up)
  $("ai-panel").classList.add("online");
  $("ai-row").style.display = "flex";
  $("chat-tools").hidden = false;
  $("new-chat").hidden = false;
  $("mode-bar").hidden = false;
  setAfterAnswer(false, !!ai.conv.turns.length);
  emptyText("The model is ready. Ask anything.");
  sysNote(`Model ready on ${n} device${n > 1 ? "s" : ""}`);
  if (!matchMedia("(pointer: coarse)").matches) $("ai-prompt").focus();   // touch: the keyboard opens when the user taps the prompt
  broadcastAll({ t: "ai-ready-all", model: ai.model, label: MODELS[ai.model]?.label, ctx: ctxMax() });
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
    const timer = setTimeout(() => { ai.waiters.delete(key); rej(new Error(`pipeline timeout (${what}): a device in the room stopped answering. Check that every device's tab is open with its screen on, then ask again`)); }, ms);
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
  // the chain saves with this frame: only now is it a checkpoint every device has, so only now
  // does the host's own copy go to disk (a reloaded host must not index a slot no worker saved)
  const saved = ctl.sv != null && ai.ckpt?.items.find((x) => x.key === ctl.sv);
  if (saved) ckptPersist(ctl.sv, saved.ids, saved.pin);
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
// (tellChain: the states are bad, so the disk copies go too; without it the engines were rebuilt,
// and the copies on disk are what a restore reads back)
function ckptClear(tellChain = false) {   // engines rebuilt or in an unknown state: nothing saved is usable
  const keys = ai.ckpt ? ai.ckpt.items.map((x) => x.key) : [];
  for (const k of keys) { try { ai.engine?.dropSlot?.(k); } catch {} }
  if (tellChain && keys.length && ai.chain.length) ai.pendingCtl = { ...ai.pendingCtl, dp: [DROP_ALL] };
  if (tellChain) ckptForget([DROP_ALL]);
  ai.ckpt = new PrefixIndex(1 << 30); ai.ckptN = ai.ckptN || 0;
}
// pin: this save is the system prompt + tools (Code mode, issue #73). It is kept apart from the
// CKPT_MAX answer checkpoints, never evicted by them, and replaces the previous pinned one.
function ckptSave(pin = false) {
  if (!CKPT_MAX || !ai.fed?.length || !ai.engine?.saveSlot) return;
  if (!ai.ckpt) ckptClear();
  // a worker applies sv before dp, so a save riding with DROP_ALL would be gone at once on every
  // worker: skip it (the host would otherwise index a slot the chain does not have)
  if (ai.chain.length && [].concat(ai.pendingCtl?.dp ?? []).includes(DROP_ALL)) return;
  // a save still waiting for its frame never reached the chain (no frame went out since, e.g. Stop
  // before the first prefill round): the frame header carries one save, so this one supersedes it.
  // Forget it here too, or a later resume could load a slot no worker saved. Its drops stay pending.
  const prev = ai.chain.length ? ai.pendingCtl?.sv : null;
  if (prev != null) {
    // a pinned save superseded at the same tokens (Stop right after it) stays pinned
    const p = ai.ckpt.items.find((x) => x.key === prev);
    if (p?.pin && p.ids.length === ai.fed.length && p.ids.every((t, i) => t === ai.fed[i])) pin = true;
    ai.ckpt.remove(prev); try { ai.engine.dropSlot(prev); } catch {}
    ckptForget([prev]);   // the host's copy is only written once the frame goes out: none should exist, but never keep one
  }
  // the caches hold exactly a checkpoint already on every device (Stop right after the pinned
  // save, solo; or a regenerate stopped before its first frame): keep that one instead of a
  // second slot with the same state, which would only evict an answer checkpoint
  const same = ai.ckpt.items.find((x) => x.ids.length === ai.fed.length && x.ids.every((t, i) => t === ai.fed[i]));
  if (same && (same.pin || !pin)) {
    same.t = ++ai.ckpt.clock;
    // the superseded save must not go out any more: no new sv replaces it here, and the workers
    // would save (and copy to disk) a slot the host no longer indexes
    if (prev != null) { const { sv, ...rest } = ai.pendingCtl; ai.pendingCtl = rest; }
    return;
  }
  const drop = [];
  if (same) { ai.ckpt.remove(same.key); ai.engine.dropSlot(same.key); drop.push(same.key); }   // re-saved pinned below
  if (pin) for (const old of ai.ckpt.pinned()) { ai.ckpt.remove(old.key); ai.engine.dropSlot(old.key); drop.push(old.key); }
  else while (ai.ckpt.unpinned().length >= CKPT_MAX) {
    const old = ai.ckpt.unpinned().reduce((a, b) => (a.t < b.t ? a : b));
    ai.ckpt.remove(old.key); ai.engine.dropSlot(old.key); drop.push(old.key);
  }
  if (drop.length) ckptForget(drop);
  const key = ai.ckptN = (ai.ckptN || 0) % 65534 + 1;   // slot numbers ride the frame header (u16)
  ai.engine.saveSlot(key);
  ai.ckpt.add(ai.fed.slice(), key, { pin });
  if (!ai.chain.length) ckptPersist(key, ai.fed, pin);   // solo: no chain to wait for (sendChain does it otherwise)
  if (ai.chain.length) {
    const dp = [...new Set([...[].concat(ai.pendingCtl?.dp ?? []), ...drop])];
    ai.pendingCtl = { ...ai.pendingCtl, sv: key, ...(dp.length ? { dp } : {}) };
  }
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
// where this device's copy of a slot lives on disk: the room, the model and its layers
function ckptWhere(slot) { return { room: roomCode, model: ai.model, sig: ai.engine.stateSignature(), slot }; }
// copy slot `key` (just saved on the GPU) to disk, in the background; ids: the host's tokens for it,
// pin: it is the pinned system prompt checkpoint (a reloaded host indexes it pinned again)
function ckptPersist(key, ids, pin = false) {
  if (!ckptDisk || !ai.engine?.exportSlot || !ai.engine.stateSignature || !roomCode) return;
  const E = ai.engine, where = ckptWhere(key);
  // read when the write's turn comes: a slot dropped or an engine replaced by then is skipped
  ckptDisk.put(where, () => (ai.engine === E ? E.exportSlot(key) : Promise.reject(new Error("engine replaced"))), ids ? { ids: Array.from(ids), ...(pin ? { pin: true } : {}) } : {})
    .catch(() => false);
}
function ckptForget(keys) {
  if (!ckptDisk || !roomCode || !keys?.length) return;
  ckptDisk.drop(roomCode, keys.includes(DROP_ALL) ? "all" : keys).catch(() => {});
}
// After this device's layers load (a reload, the host resuming its room, a re-deal): read the copies
// of its part back into GPU slots. The host also gets its index back (the tokens of each checkpoint);
// a worker holds the slots for when the host asks (ld) and lists them in its ai-ready, so the host
// forgets any it lacks (ckptPrune). -> the slots read back.
async function ckptRestore() {
  if (!ckptDisk || !ai.engine?.importState || !ai.engine.stateSignature || !roomCode) return [];
  const E = ai.engine, host = ai.role === "host", got = [];
  try {
    const have = (await ckptDisk.list(ckptWhere(0))).filter((c) => !host || Array.isArray(c.meta.ids));
    // the host indexes the newest CKPT_MAX answer checkpoints, oldest first so they keep their age
    // order (ckptSave evicts the oldest), and the newest pinned one (the system prompt), pinned again
    // so answer saves still never evict it. A worker does not know which is pinned: it keeps one
    // more for the pinned one and one more in case the host's copy of the newest is gone.
    const pinned = host ? have.filter((c) => c.meta.pin).slice(0, 1) : [];
    for (const c of host ? [...pinned, ...have.filter((c) => !c.meta.pin).slice(0, CKPT_MAX).reverse()] : have.slice(0, CKPT_MAX + 2)) {
      const st = await ckptDisk.get(ckptWhere(c.slot));
      if (!st || ai.engine !== E) continue;
      try { E.importState(st); E.saveSlot(c.slot); } catch { continue; }   // another shape: not ours after all
      got.push(c.slot);
      ai.ckptN = Math.max(ai.ckptN || 0, c.slot);   // never hand out a number a device may still have on disk
      if (host) { if (!ai.ckpt) ckptClear(); ai.ckpt.add(st.meta.ids, c.slot, { pin: !!st.meta.pin }); }
    }
  } catch {}
  if (got.length && ai.engine === E) { E.reset?.(); if (host) { ai.pos = 0; ai.fed = []; } }
  return got;
}
// Host: forget the checkpoints a device that just loaded its layers does not hold (it lists the slots
// it read back in ai-ready; a device without disk copies holds none), so a resume never asks it
// for a slot it lacks. Runs once the host's own restore is done (aiMaybeReady).
function ckptPrune() {
  if (!ai.ckptHeld?.size) return;
  const drop = [], reports = [...ai.ckptHeld].filter(([id]) => ai.chain.includes(id));
  for (const [, slots] of reports) {
    const held = new Set(Array.isArray(slots) ? slots : []);
    for (const x of ai.ckpt?.items.slice() || []) if (!held.has(x.key)) {
      ai.ckpt.remove(x.key); try { ai.engine?.dropSlot?.(x.key); } catch {}
      drop.push(x.key);
    }
  }
  // a device reads back more than the host indexes (a worker keeps CKPT_MAX + 1, and copies the host
  // lost): a slot it holds that the index does not name would sit on its GPU for good, so the chain
  // drops it too. Slot numbers go on past it, so no new save can land on a number still being dropped.
  const kept = new Set((ai.ckpt?.items || []).map((x) => x.key));
  for (const [, slots] of reports) for (const k of Array.isArray(slots) ? slots : []) {
    if (!Number.isInteger(k) || kept.has(k) || drop.includes(k)) continue;
    drop.push(k); ai.ckptN = Math.max(ai.ckptN || 0, k);
  }
  ai.ckptHeld.clear();
  if (!drop.length) return;
  ckptForget(drop);
  const dp = [...new Set([...[].concat(ai.pendingCtl?.dp ?? []), ...drop])];
  ai.pendingCtl = { ...ai.pendingCtl, dp };
}
// A device of the chain came back with a fresh engine that reads its copies back from disk
// (ckptRestore). The checkpoints it can have are those whose save went out on a frame before it
// left: keep those, forget one still waiting to go out (no device saved it; on the fresh engine it
// would save nothing useful). Without disk copies nothing saved survives it: drop everything.
function ckptRejoin() {
  if (!ckptDisk) { ckptClear(true); return; }
  const { sv, ...rest } = ai.pendingCtl || {};
  if (sv == null) return;
  ai.ckpt?.remove(sv);
  try { ai.engine?.dropSlot?.(sv); } catch {}
  ai.pendingCtl = rest;
}

// Speculative drafting reads the draft block's own KV cache, which prefill fills with the trunk's
// final hidden state at every prompt position. Solo prefill does it inside the engine; in a room
// the returned hidden states come back from the chain, so the host fills it as they arrive
// (roadmap 25: +18–45% tokens per lap after a prompt). Drafts only change speed, never output.
// ?fill=0 turns it off for A/B runs.
const FILL_DRAFTS = new URLSearchParams(location.search).get("fill") !== "0";
// ?phonelayers=1: phones hold layers even when the computers can hold the model (room/plan.js
// phonesToLeaveOut), for A/B and demos
const PHONE_LAYERS = new URLSearchParams(location.search).get("phonelayers") === "1";
// ?split=memory|speed picks the layer split at load (test harnesses pin "memory"; the default is speed)
{ const sp = new URLSearchParams(location.search).get("split"); if (sp === "memory" || sp === "speed") $("ai-split").value = sp; }
const MTP_BATCH = roomEngineFlags(location.search).mtpBatchFill;   // ?mtpbatch=0: one draft-cache row per submit, for A/B
const KV_ASK = roomQwen35Options(location.search).kvQ8 ? "q8" : null;   // ?kv=q8 (engine/preset.js): int8 KV cache, host only; see aiLoadShard
// the host's share of each lap in one submit (engine hostFuse, engine/preset.js); ?hostfuse=0 for A/B, same output
const HOST_FUSE = roomQwen35Options(location.search).hostFuse;
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
// ahead (plain decoding in a chain, see roomGenerate): { h, t0, defer, onSent }. h: the host's
// layers for id already ran (engine.headAhead, started at t0), so the lap starts with the send;
// onSent: runs while the other devices work; defer: returns null without running the head (the
// next headAhead runs it in the same submit as the next token's layers).
async function aiPipeToken(id, needLogits = true, fillNext, desc = null, ahead = null) {
  const pos = ai.pos;
  if (!ai.chain.length && !needLogits) {
    // solo prefill: layers only, no head, no readback; sync every 8 tokens
    ai.engine.pos = pos;
    await ai.engine.prefillToken(id);
    if (pos % 8 === 7) await ai.device.queue.onSubmittedWorkDone();
    ai.pos++; ai.fed?.push(id);
    return null;
  }
  const tHost = ahead?.t0 ?? performance.now();
  let h = ahead?.h || await ai.engine.embedRun(id, pos);
  if (badF32(h)) throw new Error(`NaN after HOST layers (pos ${pos}) — host GPU kernel issue`);
  if (ai.chain.length) {
    const hostMs = performance.now() - tHost;
    const returned = lapWait(pos, 30000, "token");
    sendChain({ t: "ai-hidden", pos, ...packWire(h) });
    ahead?.onSent?.();
    h = await returned;
    if (badF32(h)) throw new Error(`NaN in hidden returned by peers (pos ${pos}) — check peer status lines`);
    noteLap(performance.now() - tHost, hostMs);
    ai.lastHidden = h;
    if (!needLogits && fillNext !== undefined) fillDrafts(h, [id, fillNext], 0, pos, 1);
  } // solo mode: engine holds every layer, embedRun already produced the final hidden
  ai.pos++; ai.fed?.push(id);
  if (!needLogits || ahead?.defer) return null;   // prefill: skip the head entirely
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
  // Code, and phones (upright, or on their side: a short touch screen), start with it folded
  return v ? v === "folded" : bandMode() === "code" || innerWidth < 820 || matchMedia("(max-height: 500px) and (pointer: coarse)").matches;
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
// Chat | Code sits in the room bar at 820px and wider and on short touch screens (a phone in
// landscape, where its own strip costs too much height); in its own strip on phones held upright.
{
  const home = document.querySelector(".mode-row"), wide = matchMedia("(min-width: 821px), (max-height: 500px) and (pointer: coarse)");
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
    const a = document.activeElement, typing = a && (a.id === "ai-prompt" || a.id === "code-prompt" || a.id === "ed-text"
      || (!!a.closest?.("#code-pane") && a.matches("textarea, input[type=text]")));   // e.g. a rejection's reason
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
// pin: the length of the prompt's fixed start (Code mode: the system prompt + tools). When the
// caches do not hold it yet, the prefill pauses there and saves a pinned checkpoint on every
// device, so a later prompt that changes after it (a compacted agent conversation) resumes there.
// pinTag: whose fixed start it is (API asks: a hash of it; Code mode: none). A pinned checkpoint
// another tag used in the last PIN_KEEP_MS is not replaced (two agents in one room would otherwise
// keep evicting each other's system prompt): the new request just runs without pinning.
const PIN_KEEP_MS = 600000;
async function roomGenerate(ids, { onToken = () => {}, stop, maxNew = MAX_NEW, sample = pickSampler(ai.settings.sampling), signal, onStatus = () => {}, pin = 0, pinTag = "" } = {}) {
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
    // the fixed start first, then its checkpoint, then the rest (the same tokens at the same
    // positions, so the answer is the same; the head's logits after the first part are unused)
    // Stopped during the first part: nothing more is sent, and the end-of-answer save below keeps
    // what the caches hold, as for any stop during a prefill.
    const pinned = ai.ckpt?.pinned?.()[0];
    const tag = pinTag || "code";
    if (pinned && ai.pinInfo && reused >= pinned.ids.length && pinned.ids.length <= ids.length && pinned.ids.every((t, i) => t === ids[i])) ai.pinInfo.hitAt = performance.now();
    if (pin && pinned && ai.pinInfo && ai.pinInfo.tag !== tag && performance.now() - ai.pinInfo.hitAt < PIN_KEEP_MS) pin = 0;
    const cut = CKPT_MAX && ai.engine.saveSlot ? pinSplit(reused, pin, ids.length) : 0;
    let logits = null;
    if (!cut) logits = rest.length ? await aiPrefill(rest, { aborted, onStatus, desc }) : null;
    else {
      const part = await aiPrefill(ids.slice(reused, cut), { aborted, onStatus, desc });
      // the whole fixed start is in the caches (also when Stop came during its last lap): pin it,
      // or the next request, which reuses it from the plain end save, would never pin it
      if (ai.fed?.length === cut) { ckptSave(true); ai.pinInfo = { tag, hitAt: performance.now() }; }
      if (part && !aborted()) logits = await aiPrefill(ids.slice(cut), { aborted, onStatus, desc });
    }
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
        // pre: { hs, t0 } when the engine already ran the host's layers with the drafts (hostFuse)
        runTrunk: async (tokens, pos, pre = null) => {
          const tLap = pre?.t0 ?? performance.now();
          const n = tokens.length, hdim = ai.engine.dims.dim, NC = ai.engine.NC || 4;
          const hb = pre?.hs || new Float32Array(n * hdim);
          if (!pre) for (let c = 0; c < n; c += NC) {
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
        preTrunk: true,
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
      // ahead (greedy GPU sampling in a chain, HOST_FUSE): from the second lap on, the head of the
      // hidden the chain returned and the host's layers on its pick are one submit (engine
      // headAhead), the hidden goes out before the token is shown, and a pick that is not piped (a
      // stop token, the cap, an abort) has its layers undone (dropAhead)
      const ahead = HOST_FUSE && ai.chain.length > 0 && desc?.kind === "greedy" && !!ai.engine.canHeadAhead?.();
      let deferred = false;   // ahead: the head of ai.lastHidden has not run yet
      try {
        for (let i = 0; i < maxNew && !aborted(); i++) {
          let next, pre = null;
          const tLap = performance.now();
          if (i === 0 && first != null) next = first;
          else if (!deferred) next = sample(logits);
          else {
            const r = await ai.engine.headAhead(ai.lastHidden, ai.pos, desc);
            if (r.cands.bad) throw new Error(`NaN in logits (pos ${ai.pos}) — head/lm_head kernel issue on host`);
            logits = r.cands; deferred = false;
            next = sample(r.cands);
            if (r.h && next === r.cands.ids[0]) pre = r.h;
          }
          if (eos(next)) break;
          if (ai.pos >= ctxMax() - 1) { emit(next, false); capped = true; break; }   // no position left for another token
          if (ahead) {
            if (pre) ai.engine.keepAhead(); else ai.engine.dropAhead();
            logits = null; deferred = true;
            await aiPipeToken(next, true, undefined, desc, { h: pre, t0: tLap, defer: true, onSent: () => emit(next, false) });
          } else {
            emit(next, false);
            logits = await aiPipeToken(next, true, undefined, desc);
          }
          if (ai.chain.length) pushMap(count / ((performance.now() - t0) / 1000), null, true);
        }
      } finally { if (ahead) ai.engine.dropAhead(); }   // a pick that was not piped: its layers undone
      if (count >= maxNew) {
        capped = true;
        // for Continue: the chosen id (sample reads logits or GPU candidates alike)
        if (deferred && !aborted() && ai.pos < ctxMax() - 1) {
          logits = await ai.engine.headFromHiddenIds(ai.lastHidden, desc);
          if (logits.bad) throw new Error(`NaN in logits (pos ${ai.pos}) — head/lm_head kernel issue on host`);
        }
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
  ai.lastWasApi = false;
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
    // a character split across tokens goes out whole, never as two U+FFFD
    const pieces = pieceDecoder(ai.tok);
    const show = (piece, drafted) => {
      if (!piece) return;
      reply += piece;
      chatBotPiece(piece, drafted);
      sendChat({ t: "ai-token", text: piece, d: drafted || 0 }, askerId);
    };
    const onToken = (tok, drafted) => {
      answer.push(tok);
      show(pieces.push(tok), drafted);
    };
    inGen = true;   // from here roomGenerate cleans up after itself on failure
    try { r = await roomGenerate(fit.ids, { onToken, stop, maxNew: MAXNEW_PARAM || cap, sample, onStatus: aiStatus }); }
    finally { show(pieces.flush(), 0); }
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
  if (ai.lastWasApi) { const why = "the last answer went to an API client: ask again instead"; if (from === peer.id) toast(why); else sendTo(from, { t: "ai-busy", why }); return; }
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
  if (d.sv != null) { ai.engine.saveSlot?.(d.sv); ckptPersist(d.sv); ctl.sv = d.sv; }
  if (d.dp != null) { for (const k of [].concat(d.dp)) k === DROP_ALL ? ai.engine.dropAllSlots?.() : ai.engine.dropSlot?.(k); ckptForget([].concat(d.dp)); ctl.dp = d.dp; }
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
  $("room-over-h").textContent = "Host reconnecting";
  $("room-over-why").textContent = "The host's tab closed. Waiting a minute in case it comes back…";
  const was = $("ai-status").textContent;
  aiStatus("the host left; waiting for it to come back…");
  const t0 = Date.now();
  clearInterval(hostGone.timer);
  hostGone.timer = setInterval(() => {
    if (conns.has(PREFIX + roomCode)) { clearInterval(hostGone.timer); return; }
    if (Date.now() - t0 > HOST_WAIT_MS) {
      clearInterval(hostGone.timer);
      ai.engine = null;
      $("room-over-h").textContent = "Room over";
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
      // layers to deal only if a model was running; otherwise the card goes back to what it said
      aiStatus(ai.role ? "the host is back; waiting for it to deal the layers…" : was);
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
      transcript: ai.transcript.filter((t) => !t.api).slice(-20), settings: ai.settings, peers: ai.chainNames || [], split: $("ai-split").value, ckptN: ai.ckptN || 0, t: Date.now() }));
  } catch {}
}
addEventListener("pagehide", saveHost);   // stamp the saved room as the tab unloads, so a reload can go straight back in
function savedHost() {
  try { const r = JSON.parse(localStorage.getItem(HOST_KEY) || localStorage.getItem(OLD_HOST_KEY) || "null"); return r && Date.now() - r.t < 15 * 60 * 1000 ? r : null; } catch { return null; }
}
function resumeHost(r) {
  ai.conv = { turns: Array.isArray(r.turns) ? r.turns : [] };
  ai.transcript = Array.isArray(r.transcript) ? r.transcript : [];
  ai.ckptN = Number.isInteger(r.ckptN) ? r.ckptN : 0;   // slot numbers go on from here: a device's old copy is never mistaken for a new one
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
  "ai-visibility", "ai-style", "ai-busy", "ai-wait", "ai-start-failed"]);
async function aiOnData(from, d) {
  if (d.t.startsWith("ai-code") || d.t.startsWith("ai-pv")) { codeOnData(from, d); return; }
  const e = conns.get(from);
  if (FROM_HOST.has(d.t)) {
    if (ai.role === "host") return;
    if (from !== (ai.hostId || PREFIX + roomCode)) return;
  }
  if (d.t === "ai-load" && ai.role === "host") return;
  // layers dealt by a host on another protocol would fail as NaNs or timeouts: refuse them out loud
  if (d.t === "ai-load" && d.v != null && d.v !== PROTOCOL) {
    const { local, remote } = versionMismatch({ mine: PROTOCOL, theirs: d.v, theyHost: true, me: myName });
    toast(local, { kind: "error" }); aiStatus(local);
    sendTo(from, { t: "ai-error", message: remote });
    return;
  }
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
      // the host re-seats this device while its load of the same layers still runs (it dropped this tab
      // and it reconnected): keep that load, it reports ai-ready to the host when it is in
      const loadKey = shardKey(d.model, d.range);
      if (onLoadRequest({ loadingShard: ai.loadingShard, role: ai.role, currentKey: ai.loadKey, key: loadKey }) === "keep") {
        ai.next = d.next; ai.hostId = d.host; ensureLink(d.next);
        break;
      }
      setModelValue(d.model);
      ai.role = "worker";
      ai.next = d.next;
      ai.hostId = d.host;
      ai.q = Promise.resolve();
      ai.startFailed = null;
      ai.wsrc = MODELS[d.model]?.gguf && d.inv ? weightSources(MODELS[d.model].gguf, d.inv) : null;
      ensureLink(d.next);   // open the link to my chain neighbour while the weights download
      try {
        ai.loadingShard = true; ai.loadKey = loadKey;
        // d.kv: the host's KV format (a host without it: this device's own ?kv=, as before)
        try { await aiLoadShard(d.model || "smollm-135m", d.range, false, false, d.ctx || maxSeqFor(d.model), kvForLoad(d.model, d.kv, KV_ASK)); } finally { ai.loadingShard = false; }
        if (ai.startFailed) throw new Error(ai.startFailed);
        if (!(await ensureLink(d.next))) throw new Error("could not connect to the next device in the chain");
        const slots = await ckptRestore();   // this device's part of the room's checkpoints, if it saved any before a reload
        aiStatus(`layers ${d.range[0]}–${d.range[1] - 1} ready · syncing with the room…`);
        $("ldg-title").textContent = `layers ${d.range[0]}–${d.range[1] - 1} ready`;   // the card stays up as it is (Starting) until ai-ready-all
        $("ldg-sub").textContent = "syncing with the rest of the room";
        $("ldg-fill").style.width = "100%";
        sendTo(ai.hostId, { t: "ai-ready", slots });   // the host forgets the checkpoints not in slots
      } catch (err) {
        if (onLoadError(ai.startFailed) === "stopped") { workerStopped(); break; }   // the host stopped the start (ai-start-failed): this load stopped with it
        aiLoading(false);
        aiStatus("failed: " + err.message);
        toast(`This device couldn't load its layers (${err.message}). Press Start to try again.`);
        ai.engine = null;
        $("ai-start").disabled = false; $("ai-model").disabled = false; updateCluster();
        sendTo(ai.hostId, { t: "ai-error", message: err.message, load: 1 });   // load: the host stops the start
      }
      break;
    }
    case "ai-start-failed":   // the host stopped the start: back to the picker, and a load still running here stops (the pacer)
      ai.startFailed = d.why || "the start was stopped";
      if (freeOnStartFailed(ai.loadingShard)) workerStopped();   // (a load in flight stops at its next tensor, and its catch does this)
      aiLoading(false);
      $("ai-panel").classList.remove("online");
      $("ai-row").style.display = "none";
      $("ai-start").disabled = false; $("ai-model").disabled = false; updateCluster();
      if (!/^failed:/.test($("ai-status").textContent)) toast(`The model didn't start: ${d.why}. Press Start to try again.`);
      aiStatus("the model didn't start: " + d.why);
      break;
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
      (ai.ckptHeld ||= new Map()).set(from, d.slots);   // what it read back from disk (none from an older build)
      if (e?.card) { e.card.querySelector(".bw").textContent = "ready"; peerStatus(e.card, "ready", true); }
      aiMaybeReady();
      break;
    case "ai-error":
      aiStatus(`peer ${e?.name || from} failed: ${d.message}`);
      if (ai.role === "host" && ai.chain.includes(from)) {
        failWaiters(new Error(`${e?.name || from}: ${d.message}`));
        // a device could not load its layers while the room was starting: the start cannot finish
        if (stopsStart({ load: d.load, inChain: true, starting: ai.busy === true, online: $("ai-panel").classList.contains("online") })) aiLoadFailed(from, e?.name || from, d.message);
      }
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
      setAfterAnswer(!!d.capped && !d.hidden && !d.api, !d.failed && !d.hidden && !d.api);
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
      if (d.api) { apiAsk(from, d); break; }
      aiAsk(String(d.text || "").slice(0, 8000), String(e?.name || "guest"), from);
      break;
    case "ai-queued":
      toast(d.pos === 1 ? "queued: yours is next" : `queued: ${d.pos - 1} question${d.pos > 2 ? "s" : ""} ahead of yours`);
      break;
    case "ai-queue": showQueue(d.n); break;
    case "ai-stop":
      if (ai.role === "host" && d.rid != null) { apiStop(from, String(d.rid)); break; }
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
  "ai-code-projects", "ai-code-msg", "ai-pv", "ai-pv-blob", "ai-pv-stop", "ai-code-share"]);
const CODE_TO_HOST = new Set(["ai-pv-want", "ai-code-ask", "ai-code-stop", "ai-code-approve", "ai-code-cmd", "ai-code-sync", "ai-code-share-ask"]);
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
  const p = codeLoad ||= import("./room/code.js").then((m) => m.initCode?.(roomApi, { mock: MOCK ? window.__pooledMock : null }))
    .catch((err) => { codeLoad = null; throw err; });
  codeGate();   // a Code session reached this device (a host's code message): its tab turns on
  return p;
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
// Code runs on the room's model: until one is online its tab is shown but off, and a tap says why
function codeGate() {
  const off = !MOCK && !codeLoad && !$("ai-panel").classList.contains("online") && $("mode-code").getAttribute("aria-selected") !== "true";
  $("mode-code").setAttribute("aria-disabled", String(off));
  $("mode-code").title = off ? "Start a model first: Code mode runs on the room's model" : "";
}
new MutationObserver(codeGate).observe($("ai-panel"), { attributes: true, attributeFilter: ["class"] });
document.addEventListener("click", (e) => {
  if (!e.target.closest?.("#mode-code") || $("mode-code").getAttribute("aria-disabled") !== "true") return;
  e.preventDefault(); e.stopImmediatePropagation();
  toast("Start a model first: Code mode runs on the room's model.");
}, true);
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
  model: () => ai.model || $("ai-model").value,   // the room's model key (Code's empty state sizes its example to it)
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
  ai.settings = withStyle(ai.settings, { persona: $("ai-persona").value, sampling: $("ai-sampling").value, thinking: $("ai-thinking").checked, length: $("ai-length").value });
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
  "ai-split": { memory: "Every computer holds layers, sized by the memory it gives. A phone holds layers only when the computers can't fit the model.", speed: "The fastest devices hold the layers, with the fewest hops; the rest join to ask. Before the first answer measures them, computers go before phones. Takes effect when the layers are dealt again." },
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
$("room-menu").addEventListener("toggle", () => {
  if (!$("room-menu").open) { if (cacheArmed) cacheDisarm(); return; }
  syncSegs();
  renderCacheModels();
  // the room card pictures a finished answer's speed: not offered before one, and Share only where the browser can
  $("card-btn").hidden = !(bestTps || lastMap?.st?.tps || lastSoloTps);
  $("card-share").hidden = !navigator.canShare;
  $("export-chat").disabled = !document.querySelector("#ai-output .m.user");   // nothing asked yet: nothing to save
});
$("menu-close").addEventListener("click", () => { $("room-menu").open = false; $("room-menu").querySelector("summary").focus(); });
// two steps: the first click says how much would go, a second one within a few seconds deletes it
let cacheArmed = 0;
function cacheLabel(t, sub) { const a = $("cache-clear"); a.childNodes[1].textContent = t; a.querySelector("small").textContent = sub; }
const cacheDisarm = () => { cacheArmed = 0; cacheLabel("Clear cached weights", "Frees this device's disk; the next start downloads again"); };
async function cachedBytes() {
  const c = await getWeightCache(); let n = 0;
  if (c) for (const k of await c.keys()) n += +((await c.match(k))?.headers.get("x-swarm-len") || 0);
  try { n += await convertedBytes(await navigator.storage.getDirectory()); } catch { /* no OPFS */ }
  return n;
}
$("cache-clear").addEventListener("click", async (ev) => {
  ev.preventDefault();
  if (!cacheArmed) {
    const n = await cachedBytes().catch(() => 0);
    if (!n) { toast("no cached weights on this device"); return; }
    cacheLabel(`Clear ${fmtBytes(n)}?`, "Press again to delete them; the next start downloads them again");
    const t = cacheArmed = setTimeout(() => { if (cacheArmed === t) cacheDisarm(); }, 6000);
    return;
  }
  cacheDisarm();
  try { await navigator.storage.getDirectory().then(clearConverted); } catch { /* no OPFS */ }
  try { await caches.delete(CACHE_NAME); weightCache = null; toast("cached weights cleared"); } catch { toast("could not clear the cache"); }
  renderCacheModels();
});
// One row per cached model with its size and a Delete button (two steps, like Clear above), so one
// model's weights can go while the others stay. Filled each time the menu opens.
// A newer render (menu reopened, a delete finished) wins over one still reading the cache.
let cacheRenderGen = 0;
async function renderCacheModels() {
  const gen = ++cacheRenderGen, ul = $("cache-models"), c = await getWeightCache();
  const list = c ? await cachedModels(c, MODELS).catch(() => []) : [];
  // converted weights (room/convertedcache.js) count toward their model's row and go with it
  const opfs = await navigator.storage?.getDirectory?.().catch(() => null);
  const conv = opfs ? await convertedByModel(opfs) : new Map();
  for (const g of list) g.bytes += conv.get(modelOf(g.url)) || 0;
  if (gen !== cacheRenderGen) return;
  ul.replaceChildren(); ul.hidden = !list.length;
  if (!list.length) return;
  for (const g of list) {
    const li = document.createElement("li"), name = document.createElement("span"), b = document.createElement("button");
    name.textContent = g.label.split("\u00b7")[0].trim();
    const size = document.createElement("small"); size.textContent = fmtBytes(g.bytes); name.append(" ", size);
    const what = name.firstChild.textContent;
    const disarm = () => { armed = 0; b.textContent = "Delete"; b.setAttribute("aria-label", `Delete the cached weights of ${what}`); };
    let armed = 0; b.type = "button"; disarm();
    b.addEventListener("click", async () => {
      if (!armed) {
        b.textContent = `Delete ${fmtBytes(g.bytes)}?`;
        b.setAttribute("aria-label", `Press again to delete ${fmtBytes(g.bytes)} of cached weights of ${what}`);
        const t = armed = setTimeout(() => { if (armed === t) disarm(); }, 6000);
        return;
      }
      armed = 0; b.disabled = true;
      try { const r = await deleteModel(c, g.url); if (opfs) r.bytes += await deleteConverted(opfs, g.url); toast(`deleted ${fmtBytes(r.bytes)} of cached weights`); } catch { toast("could not delete those weights"); }
      renderCacheModels();
    });
    li.append(name, b); ul.append(li);
  }
  // what the browser counts for this whole site (weights plus a little else), to check the sizes against
  const est = await navigator.storage?.estimate?.().catch(() => null);
  if (gen !== cacheRenderGen) return;
  if (est?.usage) { const li = document.createElement("li"); li.className = "cache-est"; li.textContent = `This site uses ${fmtBytes(est.usage)} of this device's storage`; ul.append(li); }
}
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
  if (q.api) apiGenerate(q); else aiGenerate(q.text, q.name, q.from);
}
function showQueue(n) { $("queue-note").textContent = n ? `${n} queued` : ""; }

// ---- API clients: `pooled serve` (cli/) turns the room into a local OpenAI / Anthropic endpoint ----
// The bridge joins as an ask-only guest with no layers (meta.api, webgpu false) and sends whole
// conversations (ai-ask {api: 1, rid, system, messages, params}); the host answers each with
// roomGenerate on its own ids, never touching the chat's conversation (room/api.js,
// docs/design/serve.md, docs/protocol.md "API clients").
// host: a hello from an API client. false = refused (told why with a bye)
function apiWelcome(from, d) {
  const why = !ai.settings.apiAllow ? "the host does not allow API clients in this room"
    : ai.apiKicked.has(from) ? "the host disconnected this API client" : null;
  if (why) { apiBye(from, why); return false; }
  const known = ai.apis.get(from);
  ai.apis.set(from, { name: d.name, client: d.meta.client, answered: known?.answered || 0, tool: known?.tool });
  apiPanel();
  return true;
}
function apiBye(id, reason) {
  sendTo(id, { t: "bye", reason });
  setTimeout(() => { try { conns.get(id)?.conn.close(); } catch {} }, 400);   // after the bye is out
}
// host: Disconnect on the card or the panel, or API clients switched off
function apiKick(id, reason = "the host disconnected this API client") {
  if (!ai.apis.has(id)) return;
  ai.apiKicked.add(id);
  apiPeerGone(id);
  apiBye(id, reason);
  log("room", `disconnected API client ${conns.get(id)?.name || id}`);
}
// host: an API client left (or was disconnected): its queued asks go, its running one stops
function apiPeerGone(id) {
  if (!ai.apis.has(id)) return;
  ai.apis.delete(id);
  if (ai.queue?.length) {
    const n = ai.queue.length;
    ai.queue = ai.queue.filter((q) => !(q.api && q.from === id));
    if (ai.queue.length !== n) { broadcastAll({ t: "ai-queue", n: ai.queue.length }); showQueue(ai.queue.length); }
  }
  if (ai.apiRun?.from === id) ai.apiRun.ac.abort();
  apiPanel();
}
function apiAsk(from, d) {
  const rid = typeof d.rid === "string" ? d.rid.slice(0, API_LIMITS.rid) : "";
  const busy = (why, code, extra = {}) => sendTo(from, { t: "ai-busy", why, rid, code, ...extra });
  if (!ai.apis.has(from)) return busy("this device did not join as an API client", "bad");
  if (!ai.settings.apiAllow) return busy("the host does not allow API clients in this room", "off");
  const v = validateApiAsk(d, { profile: d.api === 2 && ai.tok ? apiProfile() : null });
  if (v.err) return busy(v.err, v.code);
  // the program behind the bridge (its User-Agent, read by cli/lib/http.js: Continue, OpenAI/Python...), for the Serve API panel
  const tool = v.req.params.client !== "API" ? v.req.params.client : null, known = ai.apis.get(from);
  if (tool && known.tool !== tool) { known.tool = tool; apiPanel(); }
  if (ai.degraded) return busy("a device left the room; the host has to re-deal the layers first", "degraded");
  if (!ai.engine || ai.readyPeers.size < ai.chain.length) return busy("the model is still loading", "loading");
  const entry = { api: v.req, name: ai.apis.get(from).name, from };
  if (!ai.busy && !ai.queue?.length) { apiGenerate(entry); return; }
  ai.queue ||= [];
  if (ai.queue.length >= QUEUE_MAX || ai.queue.filter((q) => q.from === from).length >= 2) {
    setTimeout(nextQueued, 0);
    return busy("the room's queue is full, try again after this answer", "queue");
  }
  ai.queue.push(entry);
  setTimeout(nextQueued, 0);
  sendTo(from, { t: "ai-queued", pos: ai.queue.length, rid });
  broadcastAll({ t: "ai-queue", n: ai.queue.length }); showQueue(ai.queue.length);
}
// the asker's stop: the running answer ends after the lap in flight; a queued one is dropped
function apiStop(from, rid) {
  if (ai.apiRun && ai.apiRun.rid === rid && ai.apiRun.from === from) { ai.apiRun.ac.abort(); aiStatus("the API client stopped its request…"); return; }
  const i = ai.queue?.findIndex((q) => q.api && q.from === from && q.api.rid === rid) ?? -1;
  if (i >= 0) {
    ai.queue.splice(i, 1);
    sendTo(from, { t: "ai-busy", rid, code: "gone", why: "removed from the queue" });
    broadcastAll({ t: "ai-queue", n: ai.queue.length }); showQueue(ai.queue.length);
  }
}
async function apiGenerate({ api: req, name, from }) {
  const rid = req.rid;
  if (!conns.has(from) || !ai.apis.has(from)) { setTimeout(nextQueued, 0); return; }   // left while it waited
  if (ai.busy || !ai.engine) { (ai.queue ||= []).unshift({ api: req, name, from }); return; }
  if (ai.degraded) { sendTo(from, { t: "ai-busy", rid, code: "degraded", why: "a device left the room; the host has to re-deal the layers first" }); return; }
  const v2 = req.api === 2;
  let prompt;
  try { prompt = v2 ? apiPrompt2(ai.tok, req, ctxMax(), { profile: apiProfile(), cache: ai.apiTurns, encoder: ai.apiEnc, model: ai.model || "" }) : apiPrompt(ai.tok, req, ctxMax(), ai.apiCache); }
  catch (err) { prompt = { err: err.message, code: "bad" }; }
  if (prompt.err) { sendTo(from, { t: "ai-busy", rid, code: prompt.code, why: prompt.err, n: prompt.n, max: prompt.max }); setTimeout(nextQueued, 0); return; }
  const client = req.params.client;
  ai.busy = "gen";
  ai.abort = false;
  ai.askerId = from;
  ai.lastWasApi = true;
  const run = ai.apiRun = { rid, from, ac: new AbortController() };
  apiPanel();
  setBusyUI(true, true);
  setAfterAnswer(false, false);
  // the screens: the room's visibility applies as for any guest, except that the asker (the bridge)
  // gets its own full stream below, never a stand-in; under "asker" only the host's screen shows it
  const { full, hidden } = chatRecipients(ai.visibility || "all", from, [...conns.keys()]);
  const toScreens = (msg) => {
    for (const id of full) if (id !== from) sendTo(id, msg);
    if (msg.t !== "ai-token") for (const id of hidden) if (id !== from) sendTo(id, { t: msg.t, name: msg.name, stats: msg.stats, asker: msg.asker, ctx: msg.ctx, api: 1, hidden: true });
  };
  const label = `${name} · ${client} (API)`;
  // the question the screens show: the last real user message (not tool results)
  const q = v2 ? [...req.messages].reverse().find((m) => m.role === "user" && !m.aside) : null;
  const last = v2 ? (q ? q.text : "(tool results)") : req.messages[req.messages.length - 1].text;
  const shown = last.length > API_LIMITS.shown ? last.slice(0, API_LIMITS.shown) + "…" : last;
  chatUser(label, shown);
  const mid = ai.msgSeq = (ai.msgSeq || 0) + 1;
  chatBotStart(mid);
  toScreens({ t: "ai-genstart", name: label, text: shown, asker: from, cont: 0, mid, api: 1 });
  sendTo(from, { t: "ai-genstart", rid, api: v2 ? 2 : 1, client, promptTokens: prompt.ids.length, model: ai.model, name: label, asker: from, mid, ...(v2 ? { style: prompt.profile.style } : {}) });
  let raw = "";
  const screen = (piece, d) => { raw += piece; chatBotPiece(piece, d); toScreens({ t: "ai-token", text: piece, d: d || 0 }); };
  const gen = (ids, o) => roomGenerate(ids, { ...o, maxNew: MAXNEW_PARAM ? Math.min(MAXNEW_PARAM, o.maxNew) : o.maxNew, onStatus: aiStatus });
  const res = v2
    ? await apiRun2({
      tok: ai.tok, req, prompt, cache: ai.apiTurns, ctxMax: ctxMax(), fallback: pickSampler(ai.settings.sampling), signal: run.ac.signal,
      tt: apiTokenTexts(), log: (m) => log("api", m),
      // the system prompt + tools, when long, is kept as the room's pinned checkpoint (tagged, so two
      // clients with different tools do not keep replacing each other's)
      generate: (ids, o) => gen(ids, { ...o, ...(prompt.systemLen >= 2048 ? { pin: prompt.systemLen, pinTag: pinTagOf(ids, prompt.systemLen) } : {}) }),
      send: (msg) => { sendTo(from, msg); apiScreenMsg(msg, screen); },
    })
    : await apiRun({
      tok: ai.tok, req, prompt, cache: ai.apiCache, ctxMax: ctxMax(), fallback: pickSampler(ai.settings.sampling), signal: run.ac.signal,
      generate: gen,
      send: (msg) => sendTo(from, msg),
      onPiece: screen,
    });
  if (res.err) aiStatus("generation failed: " + res.err);
  const stats = (res.err ? "failed: " + res.err : res.stats) + " · via API · not part of this chat's memory";
  const ctx = { used: ai.fed ? ai.pos : 0, max: ctxMax() };
  chatBotEnd(res.err && !raw ? "⚠ " + res.err : null, stats);
  toScreens({ t: "ai-gendone", stats, ctx, failed: res.err ? 1 : 0, capped: 0, api: 1 });
  sendTo(from, { t: "ai-gendone", rid, api: v2 ? 2 : 1, reason: res.reason, stopSeq: res.stopSeq || undefined, usage: res.usage, reused: res.reused, stats, ctx, failed: res.err ? 1 : 0, err: res.err || undefined,
    ...(v2 ? { calls: res.calls, ...(res.open ? { open: res.open } : {}) } : {}) });
  ai.transcript.push({ name: label, text: shown, reply: raw, stats, mid, api: 1 });
  if (ai.transcript.length > 50) ai.transcript.shift();
  const c = ai.apis.get(from); if (c) c.answered++;
  apiPanel();
  setCtx(ctx.used, ctx.max);
  if (!res.err) aiStatus(`ready — API answer for ${name}: ${res.usage.in} prompt tok${res.reused ? ` (${res.reused} reused)` : ""}, ${res.stats}`);
  ai.busy = false;
  ai.abort = false;
  if (ai.apiRun === run) ai.apiRun = null;
  setBusyUI(false);
  setTimeout(nextQueued, 0);
  if (ai.degraded) showRedeal(true);
}

// v2 asks: the loaded model's template profile (tool format, thinking rules), computed once per model
function apiProfile() {
  if (!ai.apiProf) {
    ai.apiProf = templateProfile(roomApi.chatTemplate(), ai.tok);
    if (!ai.apiProf.known) log("api", "this model's chat template keeps reasoning in an unrecognized way: past reasoning follows the last user query");
  }
  return ai.apiProf;
}
let apiTT = null, apiTTFor = null;
function apiTokenTexts() { if (apiTTFor !== ai.tok) { apiTT = tokenTexts(ai.tok); apiTTFor = ai.tok; } return apiTT; }
// a model was loaded: every cached id and the profile belong to the old one
function apiModelLoaded() {
  ai.apiCache = new AnswerCache(8);
  ai.apiTurns.clear();
  ai.apiEnc.clear();
  ai.apiProf = null;
}
function pinTagOf(ids, n) { let h = 0; for (let i = 0; i < n; i++) h = (Math.imul(h, 31) + ids[i]) | 0; return n + ":" + (h >>> 0).toString(36); }
// what a v2 answer looks like on the room's screens: reasoning in a think block (as the chat renders
// raw answers), content as it is, a tool call as one compact line "→ name(args…)" (200 characters at most)
function apiScreenMsg(msg, screen) {
  const st = apiScreenMsg.st && apiScreenMsg.st.rid === msg.rid ? apiScreenMsg.st : (apiScreenMsg.st = { rid: msg.rid, think: false, n: 0 });
  if (msg.t === "ai-token") {
    if (msg.th) { if (!st.think) { st.think = true; screen("<think>\n", 0); } screen(msg.text, msg.d); return; }
    if (st.think) { st.think = false; screen("\n</think>\n\n", 0); }
    screen(msg.text, msg.d);
  } else if (msg.t === "ai-call") {
    if (st.think) { st.think = false; screen("\n</think>\n\n", 0); }
    if (msg.name != null) { st.n = 0; screen(`\n→ ${msg.name}(`, 0); }
    else if (msg.a != null) {
      const room = 200 - st.n;
      if (room > 0) { const a = msg.a.slice(0, room); st.n += a.length; screen(a + (msg.a.length > room ? "…" : ""), 0); }
    } else if (msg.end) screen(")\n", 0);
  }
}

// Serve API (the header's black button, the dark page's API half): the command that serves this room on
// the user's own computer, its two base URLs, short examples, who is connected now (the host can
// Disconnect each), and the host's switch for API clients
const API_BASE = "http://127.0.0.1:8080";
const API_EXAMPLES = {
  curl: `curl ${API_BASE}/v1/chat/completions \\
  -H 'content-type: application/json' \\
  -d '{"model": "pooled",
    "messages": [{"role": "user", "content": "Hi"}]}'`,
  py: `# pip install openai
from openai import OpenAI

client = OpenAI(base_url="${API_BASE}/v1", api_key="pooled")
r = client.chat.completions.create(
    model="pooled", messages=[{"role": "user", "content": "Hi"}])
print(r.choices[0].message.content)`,
  js: `// npm install openai
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "${API_BASE}/v1", apiKey: "pooled" });
const r = await client.chat.completions.create({
  model: "pooled", messages: [{ role: "user", content: "Hi" }] });
console.log(r.choices[0].message.content);`,
  anth: `# pip install anthropic
from anthropic import Anthropic

client = Anthropic(base_url="${API_BASE}", api_key="pooled")
r = client.messages.create(model="pooled", max_tokens=400,
    messages=[{"role": "user", "content": "Hi"}])
print(r.content[0].text)`,
};
// one line under each example, in its card: how to stream (the install line is in the code)
const API_EX_HINTS = {
  curl: 'Add <code>"stream": true</code> to stream tokens.',
  py: 'Pass <code>stream=True</code> to stream tokens.',
  js: 'Pass <code>stream: true</code> to stream tokens.',
  anth: 'The same room through the Messages API. <code>stream=True</code> streams.',
};
let apiTab = "curl";
function apiCommand() {
  return `npx @pooled/cli serve ${roomCode || "CODE"}${SIGNAL ? ` --signal ${SIGNAL}` : ""}`;
}
// the API clients in this room: the host's own map (with how many answers each got), or the roster's
function apiClients() {
  return isHost ? [...ai.apis.entries()] : [...members.entries()].filter(([, m]) => m.meta?.api).map(([id, m]) => [id, { name: m.name, client: m.meta.client }]);
}
function apiShowTab(key) {
  apiTab = key;
  for (const b of document.querySelectorAll(".api-tabs [role=tab]")) {
    const on = b.id === `api-t-${key}`;
    b.setAttribute("aria-selected", String(on)); b.tabIndex = on ? 0 : -1;
    if (on) $("api-code").setAttribute("aria-labelledby", b.id);
  }
  // comments muted, the base URL in the accent: the one thing to change if --port moves it
  const esc = (t) => t.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  $("api-code").innerHTML = API_EXAMPLES[key].split("\n").map((l) => /^\s*(#|\/\/)/.test(l) ? `<span class="c">${esc(l)}</span>`
    : esc(l).replaceAll(API_BASE, `<span class="u">${API_BASE}</span>`)).join("\n");
  $("api-code").scrollLeft = 0;
  $("api-ex-hint").innerHTML = API_EX_HINTS[key];
  apiFade();
}
// a fade at the code's (and the command's) right edge while more of it is out of view
function apiFade() {
  for (const c of [$("api-code"), $("api-cmd")]) c.parentElement.classList.toggle("more", c.scrollWidth - c.clientWidth - c.scrollLeft > 2);
  // the command is a Tab stop only when it scrolls (a keyboard can then reach its end)
  $("api-cmd").tabIndex = $("api-cmd").scrollWidth - $("api-cmd").clientWidth > 2 ? 0 : -1;
}
// what a screen reader hears: a client coming or going, or a copy (not every request)
function apiSay(t) {
  const el = $("api-live"); if (!el) return;
  el.textContent = ""; setTimeout(() => { el.textContent = t; }, 60);
}
let apiSeen = null;   // the client ids the panel last showed, to announce who came and went
// an API client by what the person knows: the tool (from its User-Agent) once it has asked, else the bridge's name
const apiLabel = (c) => c.tool || c.name;
// where focus goes when the control it was on goes away: the command's Copy, or with the steps
// folded "How to connect", or Back
const apiHome = () => [$("api-copy"), $("api-how").querySelector("summary"), $("compute-exit")].find((el) => el.getClientRects().length);
function apiPanel() {
  if (!$("api-panel")) return;
  const list = apiClients(), running = isHost && ai.apiRun && ai.apis.get(ai.apiRun.from);
  $("compute-open").classList.toggle("live", list.length > 0);
  if (!compute.isOpen) return;
  if ($("api-cmd").textContent !== apiCommand()) { $("api-cmd").textContent = apiCommand(); apiFade(); }
  const st = $("api-status"), off = isHost && !ai.settings.apiAllow;
  st.className = "api-status " + (off ? "off" : list.length ? "live" : "wait");
  st.querySelector("span").textContent = off ? "API clients are off in this room"
    : running ? `Answering ${apiLabel(running)}`
    : list.length === 1 ? `${apiLabel(list[0][1])} is connected`
    : list.length ? `${list.length} API clients connected`
    : "Waiting for a client to connect";
  $("api-sub").textContent = off ? "Turn on Allow API clients below to let tools in."
    : list.length ? "" : "Run step 1. Your tool shows up here when it connects.";
  // beside this device's layers and passes: what the room is serving ("Serving Continue via API · 12 requests answered")
  const answered = isHost ? list.reduce((t, [, c]) => t + (c.answered || 0), 0) : null;
  $("cs-api").hidden = !list.length;
  if (list.length) {
    const who = list.length === 1 ? apiLabel(list[0][1]) : `${list.length} API clients`;
    const t = $("cs-api-t");
    t.replaceChildren(running ? "Answering " : "Serving ", Object.assign(document.createElement("b"), { textContent: who }), " via API");
    if (answered != null) { const n = document.createElement("span"); n.className = "n"; n.textContent = answered; t.append(" · ", n, ` request${answered === 1 ? "" : "s"} answered`); }
  }
  // step 1 done: the host sees a tick on it while a client is connected
  $("api-panel").classList.toggle("done", isHost && list.length > 0);
  $("api-clients-wrap").hidden = !list.length;
  // with a client connected the serving view leads and the steps fold under "How to connect" (once, as it
  // connects: the reader can open them again; never shut under the focus)
  const how = $("api-how"), was = how.classList.contains("flat");
  how.classList.toggle("flat", !list.length);
  if (!list.length) how.open = true;
  else if (was && !how.contains(document.activeElement)) how.open = false;
  const ids = list.map(([id]) => id);
  if (apiSeen) {
    const came = list.filter(([id]) => !apiSeen.includes(id)), went = apiSeen.length - (ids.length - came.length);
    if (came.length) apiSay(came.length === 1 ? `${apiLabel(came[0][1])} connected` : `${came.length} API clients connected`);
    else if (went > 0) apiSay(list.length ? `An API client left. ${list.length} connected` : "The API client left. Waiting for a client");
  }
  apiSeen = ids;
  // the list is rebuilt: a focused Disconnect gets its focus back on the new button (or a steady control, if that client left)
  const had = $("api-clients").contains(document.activeElement) ? document.activeElement.dataset.id || "" : null;
  $("api-clients").replaceChildren(...list.map(([id, c]) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="ic">${ICONS.api}</span><span class="nm"><b></b><small></small></span>`;
    li.querySelector("b").textContent = apiLabel(c);
    // one line: "curl · 3 answered" (the count on the host, which has it); the bridge's own name and
    // user agent only in the tooltip, they mean nothing to most people
    const sm = li.querySelector("small");
    if (isHost) { sm.innerHTML = `· <span class="n"></span> answered`; sm.querySelector(".n").textContent = c.answered; } else sm.remove();
    li.querySelector(".nm").title = [c.name, c.client].filter(Boolean).join(" · ");
    if (isHost) {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = "Disconnect"; b.dataset.id = id; b.setAttribute("aria-label", `Disconnect ${apiLabel(c)}`);
      b.addEventListener("click", () => { apiKick(id); if (!$("compute-screen").contains(document.activeElement)) apiHome()?.focus({ preventScroll: true }); });
      li.appendChild(b);
    }
    return li;
  }));
  if (had != null) (had && [...$("api-clients").querySelectorAll("button")].find((b) => b.dataset.id === had) || apiHome())?.focus({ preventScroll: true });
  if ($("api-allow")) { $("api-allow").checked = !!ai.settings.apiAllow; $("api-allow-row").hidden = !isHost; }
}
// the dark page opened (or closed): the API half starts from what is true now, and announces only what changes after
function apiShown(open) {
  apiSeen = null;
  if (!open || !$("api-panel")) return;
  $("room-menu").open = false; chipPop(null);
  apiShowTab(apiTab); apiPanel(); requestAnimationFrame(apiFade);
}
if ($("api-panel")) {
  for (const b of $("api-panel").querySelectorAll(".icon-act")) b.innerHTML = COPY_SVG;
  $("api-code-copy").insertAdjacentHTML("beforeend", '<span class="lb">Copy</span>');
  $("api-code").addEventListener("scroll", apiFade, { passive: true });
  $("api-cmd").addEventListener("scroll", apiFade, { passive: true });
  addEventListener("resize", () => { if (compute.isOpen) apiFade(); });
  $("api-how").addEventListener("toggle", () => { if ($("api-how").open) apiFade(); });
  $("api-copy").addEventListener("click", () => copyText(apiCommand(), "command"));
  for (const b of $("api-panel").querySelectorAll(".api-url .icon-act")) b.addEventListener("click", () => copyText(b.dataset.copy, b.dataset.what, b).then((ok) => ok && apiSay(`${b.dataset.what} copied`)));
  $("api-code-copy").addEventListener("click", (e) => copyText(API_EXAMPLES[apiTab], "example", e.currentTarget).then((ok) => ok && apiSay("Example copied")));
  const tabs = [...$("api-panel").querySelectorAll(".api-tabs [role=tab]")];
  for (const b of tabs) {
    b.addEventListener("click", () => apiShowTab(b.id.slice(6)));
    b.addEventListener("keydown", (e) => {
      const i = tabs.indexOf(b), j = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1 : e.key === "Home" ? 0 : e.key === "End" ? tabs.length - 1 : null;
      if (j == null) return;
      e.preventDefault();
      const t = tabs[(j + tabs.length) % tabs.length]; apiShowTab(t.id.slice(6)); t.focus();
    });
  }
  $("api-allow").addEventListener("change", (e) => {
    ai.settings.apiAllow = e.target.checked;
    if (!ai.settings.apiAllow) for (const id of [...ai.apis.keys()]) apiKick(id, "the host does not allow API clients in this room");
    toast(ai.settings.apiAllow ? "API clients can join this room" : "API clients are off in this room");
    saveHost(); apiPanel();
  });
}
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
    // Serve API: made-up API clients, e.g. apis([{ tool: "Continue", answered: 12 }]); [] for none.
    // The host keeps them in its own map; as a guest (guest()) they are in the roster
    apis(list = [{ tool: "Continue", answered: 12 }]) {
      for (const k of [...ai.apis.keys(), ...members.keys()]) if (k.startsWith("sim-api-")) { ai.apis.delete(k); members.delete(k); }
      list.forEach((c, i) => {
        const e = { name: `pooled serve ${String(roomCode).toLowerCase()}`, client: c.client || "OpenAI API", answered: c.answered || 0, tool: c.tool };
        if (isHost) ai.apis.set(`sim-api-${i}`, e); else members.set(`sim-api-${i}`, { name: e.name, meta: { api: 1, client: e.client } });
      });
      apiPanel();
    },
    guest() { isHost = false; apiPanel(); },
    passes(on) {
      clearInterval(passTimer); passTimer = 0;
      if (on) passTimer = setInterval(() => compute.pass(1, 14 + Math.random() * 8), 90);
    },
  };
}
