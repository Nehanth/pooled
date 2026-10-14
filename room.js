// Pooled room: signaling, WebRTC mesh, layer assignment, weight streaming and the
// generation loop (prefill, decode, speculative verify). Served with p2p.html at /room.
// The inference engine (engine/engine.js, engine/qwen35.js and their WGSL kernels, ~250 KB) is not
// part of the join screen's module graph: loadEngine() imports it when this device enters a room,
// and aiLoadShard waits for it. The join screen works as soon as the lobby modules below are in.
import { createModelLoader } from "./room/model-loader.js";
import { tabDiskChoice, saveTabDiskChoice, tabDiskMeta, tabExpertBytes, flashDiskBytes, removeFlashCopy, TAB_DISK_MODELS } from "./room/flashdisk.js";
import { createPipeline } from "./room/pipeline.js";
import { createGenerator } from "./engine/generate.js";
import { argmax } from "./engine/sampling.js";
let autotuneCoop, makeTokenizer, DenseEngine, fetchModelShard, shardTensorNames, gpuSelfTest, kernelMicroTests, Qwen35Engine;
let engineLoad = null;
function loadEngine() {
  return engineLoad ||= Promise.all([import("./engine/engine.js"), import("./engine/qwen35.js")]).then(([e, q]) => {
    ({ autotuneCoop, makeTokenizer, DenseEngine, fetchModelShard, shardTensorNames, gpuSelfTest, kernelMicroTests } = e);
    ({ Qwen35Engine } = q);
  }, (err) => { engineLoad = null; throw new Error("couldn't load the inference engine (" + (err?.message || err) + "). Check the connection and try again"); });
}
import { GGML_EMBED, GGML_OUTPUT,
  ggmlLayerNames, qwen35ShardBytes, qwen35MtpBytes, qwen4expConfig, qwen4expShardBytes }
  from "./engine/gguf.js";
import { roomQwen35Options, roomEngineFlags, applyRoomFlags } from "./engine/preset.js";   // no imports of its own; the engine itself loads late (loadEngine)
import { WIRE_F16, f32ToB64, packF16, unpackF16, asU16, asF32, b64ToF32, wireStats } from "./room/wire.js";
import { esc, md, mdChat } from "./room/markdown.js";
import { pickSampler, SAMPLING } from "./room/sampling.js";
import { chatRecipients } from "./room/visibility.js";
import { validateApiAsk, apiPrompt, apiRun, AnswerCache, API_LIMITS, pieceDecoder, helloMeta, withStyle, apiPrompt2, apiRun2, TurnCache, EncodeCache } from "./room/api.js";
import { tokenTexts } from "./harness/model-common.js";
import { CkptStore } from "./room/ckpt-store.js";
import { SHAPE, CTX, MODELS, DISK_COPY, NEED_GB, NEED_MIN_GB, FILE_GB, PICKER, MAX_SEQ, MAX_NEW, MAX_NEW_THINKING, MIN_ROOM, maxSeqFor, ctxForBinding, kvBytesPerLayerPos, kvModeFor, kvForLoad, hostHeldBytes, denseKvBytesPerLayerPos, roomBytes, pickCtx, ctxK, ctxShortNote, needText, expertsOf, modelLoadOptions, ngramOf, flashLayerCost, flashTabHostMargin,
  flashLongMeta, flashRoomFormat, flashReadyFields, readyCtxShort, flashDealReserve } from "./room/models.js";
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
import { makeLink, attachWire, holdWire, wireReady, sendFrame, setKeepalive, PROTOCOL, DUP_SLICES } from "./room/transport.js";
import { peerErrorText, peerErrorLoud, FetchError, joinStep, versionMismatch } from "./room/errors.js";
import { turnFrom, iceConfig, shareQuery, linkPath, linkRelayProtocol, normTurn, TURN_KEY, wantDefaultRelay, fetchRelay, markAuto, swapRelayServers, refreshInMs, isRelayServer, weightsOverLink, probeUdp, networkAdvice, WORK_DOCS } from "./room/ice.js";
import { PERSONAS, specials, fitContext, templateProfile } from "./room/conversation.js";
import { PING_MS, lastHeard, isSilentGone, midLoad, uniqueName, quietNamesake, staleNamesakes, renameTo, NAME_PROBE_MS,
  makeLiveness, heard as hbHeard, arm as hbArm, disarm as hbDisarm, forget as hbForget, tick as hbTick, deadAfter, suspectBack, STALL_MS } from "./room/liveness.js";
import { stopsStart, stopWhen, stopReason, loadKey as shardKey, onLoadRequest, onLoadError, freeOnStartFailed } from "./room/startstop.js";
import { verifyHost, hostAllowed, saneMeta, startBusy, startFrom, joinOrder, START_PENDING_MS } from "./room/authz.js";
import { isPhoneMeta, ladder, codeFromLocation, pickModelHost, roomFit, dealRoom, shortNote, shortBy, gbUp, specWithOffload, dealOffloads, parseForce, minFitPledge, diskShares, PAGE_SHORT_FIX, layersTotal, dealWithHostMargin } from "./room/plan.js";
import { measureCopyGBps } from "./room/gpuspeed.js";
import { qrSVG } from "./room/qr.js";
import { diskCopyOf, rowChipHTML, rowTitle, poolGpuLine, poolStoreLine, poolFix, devStoreTag, devStoreTitle, allowLabel } from "./room/diskpick.js";
import { showDiskPick, initDiskPick, refreshDiskPick, diskShortNote, diskStateOf, storedNow } from "./room/diskpick-ui.js";
import { connectDiskStore } from "./room/diskstore.js";
import { roomCtxPlan, hostMarginFor, ctxOptions, prefillSeconds, durationWords } from "./room/ctxpick.js";
import { DENSE_SPEC_V } from "./room/lookup.js";
import { DraftModel } from "./room/draftmodel.js";
import { drawCard } from "./room/card.js";
import { probe as preflight, deviceKind } from "./room/preflight.js";
import { pledgeRule, pledgeGB, dealPledgeBytes, afterLoadDeath, offloadFor, ngramFor, storageOf } from "./room/pledge.js";
import { computeScreen } from "./room/compute.js";
import { CACHE_NAME, PREFIX as CACHE_PREFIX, cacheKey, cachedModels, deleteModel } from "./room/weightcache.js";
import { working, liveWords } from "./room/working.js";
import { partsFromProg, partsFromSnapshot, progressTracker, etaTracker, roomProgress, packParts, unpackParts, wholePct, stepLine, rowVerb,
  deviceVerb, headline, sentence, footer, eyebrow, stoppedText, stopWho, firstStart, WORDS_NOTE, cardRows, cardSub, diskPart, partsEnded, slimLine, redealGuess, redealWait } from "./room/loadsteps.js";
import { addable, addWhen, wouldHelp } from "./room/autoadd.js";
import { fmtTps, speedLine, bandLine, parseStats, SpeedMeter, pushHistory, speedup, gainText, warmupAnswers, barScale, paramsOf, paramsText, expertsAt, expertsText, expertsHead, expertsCost, tokenShares, pushEvent, onlineText, fmtGB } from "./room/speedo.js";
import { TeleMeter, expertCache, expertCacheOf, expertStoreOf } from "./room/tele.js";
import { serverList, parseServer, openPeer, FALLBACK_ERRORS, reconnectDelay } from "./room/signal.js";
import { convertedBytes, clearConverted, convertedByModel, deleteConverted, modelOf } from "./room/convertedcache.js";
import { resumableGenerate, waitForRoom, linkSilent, backFromAway, sameShard, guestResume, GUEST_KEY, REJOIN_GRACE_MS, LINK_SILENT_MS } from "./room/resume.js";
import { GpuWaker } from "./room/gpuwake.js";
import { randomCode, parseCode, formatCode, keyFromHash, keyFragment, validKey, makeGate, restoreGate, saveGate, decide as gateDecide,
  enqueue as gateEnqueue, allow as gateAllow, deny as gateDeny, withdraw as gateWithdraw, requestLine, DENIED_TEXT,
  hostWantsAuth, startAuth, decideAuth } from "./room/joingate.js";
import { AUTH_V, linkFingerprints, joinerStart, joinerProof, joinerCheckAdmit, meshProof, meshCheck, validMeshKey } from "./room/chanauth.js";

// Hidden-state transport (room/transport.js). ?wire=off falls back to PeerJS messages;
// ?wire=slice uses one sliced channel; ?wire=stripeN spreads slices over N peer connections.
const WIRE = (new URLSearchParams(location.search).get("wire") || "stripe4").toLowerCase();
const WIRE_STRIPES = WIRE === "off" ? 0 : WIRE.startsWith("stripe") ? Math.max(1, Math.min(8, parseInt(WIRE.slice(6), 10) || 1)) : 1;
// frames of up to this many slices go out twice, on two associations (a lost packet then costs
// nothing); ?wiredup=0 turns it off
const WIRE_DUP = (() => { const v = parseInt(new URLSearchParams(location.search).get("wiredup"), 10); return v >= 0 ? Math.min(v, 64) : DUP_SLICES; })();
// ?ka=ms: keep-alive period on a wire link while frames flow (keeps a phone's Wi-Fi out of power
// save between laps; room/transport.js), ?ka=0 turns it off
{ const ka = new URLSearchParams(location.search).get("ka"); if (ka != null) setKeepalive(parseInt(ka, 10) || 0); }
// Signaling (room/signal.js): the public PeerJS cloud by default; a deployment can list fallbacks in
// window.POOLED_SIGNAL_SERVERS, tried in order when one is down; ?signal=host:port (or a comma list)
// wins over both (the emulator and big rooms point it at our own PeerServer). See
// docs/self-host-signaling.md.
const SIGNAL = new URLSearchParams(location.search).get("signal");
// GPU wake (room/gpuwake.js): a phone worker asks the host (hello meta `wake`) for an `ai-wake` at
// the start of every decode lap and keeps its GPU busy until the frame arrives, so its layers do not
// run on a clocked-down GPU. ?wake=0 turns it off on this device (as a worker it does not ask, as the
// host it sends none); ?wake=1 asks for it on any device; ?wake=keep also spins from the moment this
// worker sends its own frame on (no host signal needed). ?wakems caps one spin (default 50 ms).
const WAKE = new URLSearchParams(location.search).get("wake") || "";
const WAKE_MAX_MS = Math.max(1, parseInt(new URLSearchParams(location.search).get("wakems"), 10) || 50);
const PAGE_SECURE = location.protocol === "https:";
const SIGNALS = serverList({ query: SIGNAL, configured: window.POOLED_SIGNAL_SERVERS, pageSecure: PAGE_SECURE });
// what a page with no ?signal= would try first: the invite link names the server only when it differs
const SIGNAL_FIRST = serverList({ configured: window.POOLED_SIGNAL_SERVERS, pageSecure: PAGE_SECURE })[0].spec;
let signalServer = null;   // the server this tab registered on ({ spec, label, opts })
// the relay this tab uses, and what the network allows (room/ice.js; see keepRelayFresh)
let relayOn = false;     // a TURN server is in this tab's ICE configuration
let relayFrom = null;    // "yours" (?turn= / Network box), "site" (window.TURN_SERVERS), "default" (/api/turn)
let udpProbe = null;     // probeUdp(): { udpOut, host } or null
let udpSeen = null;      // its answer, once in
let relayTimer = null;
let relayGot = null;     // the site's relay as last fetched: { relay, at } (defaultRelay)

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
// Who runs the model is checked here, not taken from whoever says so (room/authz.js): when each device was
// first seen, the creator's roster order (join order), and when this device joined
const seenAt = new Map();
let rosterOrder = [], joinedAt = 0;
// a guest: the metas the pick is checked with, from the creator's roster only (not the hellos and pledges on
// this device's own links). The creator: each device's join number (seq), given once and kept across relinks
// and a resume
let authzView = new Map();
const seqOf = new Map();
let nextSeq = 0;
let roomCode = null;
// Joining (room/joingate.js). Host: the gate (invite key, passes, Ask before new devices join) and the
// links waiting in its lobby. Guest: the invite key from the link it opened, the pass the host gave it,
// and where it is in getting in ("wait": linked, the host hasn't answered; "lobby": the host is asked;
// "in"; "out": refused, no knocking).
let gate = null;
const lobbyConns = new Map();   // host: peer id -> { conn, hello, buf, stripes }
let joinKey = "", myPass = "", admission = null, afterAdmit = null;
// Proving who is on a link (room/chanauth.js, docs/protocol.md "Proving who is on a link"): the invite
// key and passes never cross a link; each side proves them with an HMAC bound to the link's DTLS
// fingerprints. meshKey: the room's key for links between devices (the host makes it with its gate and
// hands it out in admit); every such link, and every stripe, proves it before it carries anything.
// Devices from before the proofs (docs/protocol.md "Compatibility"): a host still takes their raw key or
// pass for one release, with a warning, and links from devices its roster lists as theirs without a proof
// (LEGACY_AUTH; ?legacyauth=0 or window.POOLED_LEGACY_AUTH = false: never). A joining device never sends
// its raw key or pass: an older host waits for Allow instead. ?legacyauth=1 (LEGACY_SEND) restores that,
// and is unsafe: anyone in the middle of signaling can pose as an older host and read the key
let meshKey = null;
const LEGACY_Q = new URLSearchParams(location.search).get("legacyauth");
const LEGACY_AUTH = LEGACY_Q !== "0" && window.POOLED_LEGACY_AUTH !== false;
const LEGACY_SEND = LEGACY_Q === "1";
const MESH_WAIT_MS = 10000;
const HOST_HELLO_WAIT_MS = 2000;
// The breadcrumb the previous page of this tab left (room.js crumb): what it was doing when it was
// last heard from, so a tab iOS killed can say so when it rejoins. Read once per page load, and its
// "loading" mark consumed at once, so one kill is reported once (not again on a later reconnect).
const diedCrumb = (() => {
  try {
    const c = JSON.parse(localStorage.getItem("pooled-crumb") || "null");
    if (c?.loading) localStorage.setItem("pooled-crumb", JSON.stringify({ ...c, loading: undefined }));
    return c && Date.now() - c.t < 10 * 60 * 1000 ? c : null;
  } catch { return null; }
})();
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
                 webgpu: false, gpu: "no WebGPU", maxBufGB: 0,
                 // dense verify frames this device's engine handles (room/lookup.js chainDenseSpec); ?dspec=0
                 // reports none, as a device from before them would (tests)
                 dspec: new URLSearchParams(location.search).get("dspec") === "0" ? 0 : DENSE_SPEC_V };
  if (navigator.gpu) {
    try {
      const a = await navigator.gpu.requestAdapter();
      if (a) {
        meta.webgpu = true;
        const info = a.info || {};
        meta.gpu = [...new Set([info.vendor, info.architecture || info.device].filter(Boolean))].join(" ") || "GPU";
        meta.maxBufGB = +(a.limits.maxBufferSize / 2 ** 30).toFixed(1);
        // the largest buffer this device can bind (what aiLoadShardIn asks for; phones capped at 256 MB):
        // the host keeps the room's context within every device's limit (room/models.js ctxForBinding)
        meta.maxBindMB = Math.floor(Math.min(a.limits.maxStorageBufferBindingSize, a.limits.maxBufferSize, meta.ua === "iPhone" || meta.ua === "Android" ? 256 * 2 ** 20 : Infinity) / 2 ** 20);
        // flashLong: this tab loads Qwen3.8-Flash-Next with QSA when the host asks (ai-load.flash; room/models.js
        // flashLongMeta); ?flong=0 leaves it out, as a tab from before it (tests)
        if (new URLSearchParams(location.search).get("flong") !== "0") meta.flashLong = flashLongMeta(meta.maxBindMB);
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
// the least any model in the picker needs (the 1.7B's 4 GB, at its 8K fallback context)
const SMALLEST_NEED = Math.min(...PICKER.map((k) => NEED_MIN_GB[k] ?? NEED_GB[k]));
// probe once at load; fill the contribution selector, unless an amount was already chosen there
// (typed, stepped or filled in while the probe ran: it takes up to ~0.5 s with the copy timing)
let joinGbChosen = false;
for (const [id, ev] of [["join-gb", "input"], ["gb-minus", "click"], ["gb-plus", "click"]]) $(id).addEventListener(ev, () => { joinGbChosen = true; });
// (this tab's disk tier, below setTabDisk: whether it can have one; a worker's question left unanswered, after which
// this page offers no disk until its user answers in Room settings)
let tabDiskCan = false, tabDiskDeclined = false;
const metaPromise = (async () => {
  const m = await probeGPU();
  if (m.webgpu && m.budgetGB) m.contribGB = Math.max(0.2, Math.round(m.budgetGB * 0.5 * 10) / 10);
  m.phone = m.ua === "iPhone" || m.ua === "Android";
  if (m.webgpu && WAKE !== "0" && (m.phone || WAKE === "1" || WAKE === "keep")) m.wake = 1;
  // phones and tablets lend at most what their browser tab survives (room/pledge.js, #207)
  const rule = pledgeRule(m.ua, navigator.deviceMemory);
  if (rule.capped) {
    m.pledgeMax = rule.max;
    m.contribGB = rule.def;
    $("join-gb").min = String(rule.min); $("join-gb").step = String(rule.step); $("join-gb").max = String(rule.max);
    $("join-gb").title = rule.why;
    if (m.webgpu) { $("ap-cap").textContent = rule.why; $("ap-cap").hidden = false; }
  }
  // a laptop gives at least what the smallest model needs, when its probe allows, so one laptop can
  // run the 1.7B alone (half of maxBufferSize is 2 GB on a typical laptop, and 2 GB runs nothing)
  else if (m.contribGB) m.contribGB = Math.max(1, Math.round(m.contribGB), Math.min(SMALLEST_NEED, Math.floor(m.budgetGB)));
  if (m.contribGB && !joinGbChosen) $("join-gb").value = m.contribGB;
  tabDiskCan = !!(m.webgpu && !m.phone && navigator.storage?.getDirectory);
  setTabDisk(m);
  return m;
})();
// Expert offload's disk tier in this tab (room/flashdisk.js; room/pledge.js tabDiskGB): a computer's tab with WebGPU and
// OPFS can hold Qwen3.8-Flash-Next layers past its pledge with their experts on its disk, read when a token picks them,
// and the n-gram table there too. Nothing is written until this tab's user says yes: the host asks when it starts
// Flash (aiStart), a worker when a deal gives it disk layers (the loader's askDisk); the answer is remembered for this
// site (localStorage) and changed in Room settings, where "Delete saved copy" deletes the copy. meta.tabDisk says what
// the room may deal it: { ramGB, models, ask } until its user answers, without ask after a yes, nothing after a no.
// ?tabram=G: the RAM its spare and RAM cache take (default 3, held to what the device can spare; 0 turns the tier
// off); ?tabdisk=0 says no for this page load (the URL can never say yes: room links keep their query).
const tabDiskNow = () => tabDiskChoice({ search: location.search, storage: globalThis.localStorage });
function setTabDisk(m = myMeta) {
  const choice = tabDiskNow();
  const t = tabDiskCan && !(tabDiskDeclined && choice !== "allow") ? tabDiskMeta(m, { search: location.search, choice, opfs: true, deviceMemory: navigator.deviceMemory }) : null;
  if (t) m.tabDisk = t; else delete m.tabDisk;
  return t;
}
// tell the room this tab's disk tier changed (the pledge message carries it: room/pledge.js tabDiskGB reads it)
function tabDiskChanged() {
  setTabDisk();
  renderTabDisk();
  if (myMeta.webgpu) broadcastAll({ t: "pledge", gb: myMeta.contribGB, tabDisk: myMeta.tabDisk || null });
  updateCluster();   // the picker: Flash's fit (this tab's disk tier counts or not) and its consent line
}
// a peer's tabDisk from a pledge message: only the fields room/pledge.js reads, typed
function cleanTabDisk(t) {
  if (!t || typeof t !== "object") return undefined;
  const models = Array.isArray(t.models) ? t.models.filter((x) => typeof x === "string").slice(0, 8) : [];
  return +t.ramGB > 0 && models.length ? { ramGB: Math.min(64, +t.ramGB), models, ...(t.ask ? { ask: true } : {}) } : undefined;
}
// The question itself (p2p.html #disk-ask): "Save Qwen3.8 Flash to this device?", in the picker's words (it downloads
// N GB once and keeps it on this device's disk). Download and save / Not now; the answer is remembered for this site.
// host: this device starts the model; else the room's host asks for this device's disk. Unanswered for DISK_ASK_MS: no (for this deal only, not remembered).
// -> Promise<boolean>
const DISK_ASK_MS = 120e3;
let diskAsking = null;
function askTabDisk({ bytes, model, host = false }) {
  if (diskAsking) return diskAsking;
  const M = MODELS[model];
  const name = M ? shortName(model) : "this model";   // the picker's name (room/diskpick.js)
  if (bytes > 0) lc.diskHint = bytes;
  // under a load card (a worker the deal gives disk layers, the host mid-start): the question is part of the card, and
  // the room's card shows it waits on this device's answer; no sheet over it
  if ($("load-card").classList.contains("on") && lc.phase !== "stopped" && lc.phase !== "ready") {
    diskAsking = new Promise((resolve) => {
      const done = (v, remember) => {
        clearTimeout(timer);
        lcAsk(null);
        diskAsking = null;
        if (remember) saveTabDiskChoice(v ? "allow" : "deny", globalThis.localStorage);
        else if (!v) tabDiskDeclined = true;
        tabDiskChanged();
        log("room", v ? `${myName} lets models use ${fmtBytes(bytes)} of its disk` : `${myName} keeps models off its disk${remember ? "" : " (no answer)"}`);
        resolve(v);
      };
      const timer = setTimeout(() => done(false, false), DISK_ASK_MS);
      lcAsk({ text: host ? `Save ${fmtBytes(bytes)} of ${name} on this computer, once?` : `Save this computer's ${fmtBytes(bytes)} part of ${name} here, once?`,
        sub: `${host ? "It's" : "The room waits for your answer (up to 2 min). It's"} kept on this computer's disk so later starts are quick. Delete it any time from the model picker.`,
        resolve: (v) => done(v, true) });
    });
    return diskAsking;
  }
  $("disk-ask-h").textContent = `Save ${name} to this device?`;
  $("disk-ask-why").textContent = (host ? `It downloads ${fmtBytes(bytes)} once and keeps it on this device's disk, then reads from it while it runs. `
    : `To help the room run ${name}, this device downloads ${fmtBytes(bytes)} once and keeps it on its disk. `)
    + "Delete it any time from the model picker or Room settings.";
  $("disk-ask").hidden = false;
  $("disk-ask-allow").focus({ preventScroll: true });
  diskAsking = new Promise((resolve) => {
    const done = (v, remember) => {
      clearTimeout(timer);
      $("disk-ask-allow").onclick = $("disk-ask-no").onclick = null;
      $("disk-ask").hidden = true;
      diskAsking = null;
      if (remember) saveTabDiskChoice(v ? "allow" : "deny", globalThis.localStorage);
      else if (!v) tabDiskDeclined = true;
      tabDiskChanged();
      log("room", v ? `${myName} lets models use ${fmtBytes(bytes)} of its disk` : `${myName} keeps models off its disk${remember ? "" : " (no answer)"}`);
      resolve(v);
    };
    const timer = setTimeout(() => done(false, false), DISK_ASK_MS);
    $("disk-ask-allow").onclick = () => done(true, true);
    $("disk-ask-no").onclick = () => done(false, true);
  });
  return diskAsking;
}

// Start, pressed on this device while its disk was not asked yet, said "saves N GB here, once" (room/diskpick-ui.js):
// the press is the yes, remembered as the question's would be (the picker's disk control can still say no)
function diskYesByStart(bytes) {
  ai.pressedHere = false; if (bytes > 0) lc.diskHint = bytes;
  saveTabDiskChoice("allow", globalThis.localStorage); tabDiskDeclined = false; tabDiskChanged();
  log("room", `${myName} lets models use ${fmtBytes(bytes)} of its disk (Start)`);
  return Promise.resolve(true);
}

// Start's label says what it saves here ("saves 92.4 GB here, once": room/diskpick-ui.js): only then is the press a yes
const startSaysSave = () => / saves \S+ \S+ here, once/.test($("ai-start")?.textContent || "");

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
      <div class="pheld" hidden></div>
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
  if (ai.heldGB || ai.outWhy) paintHeld();
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
const shortName = (key) => diskCopyOf(key)?.name || (MODELS[key]?.label || key).split("\u00b7")[0].trim();
// the picker's rows: the three models and the models kept on disk (room/models.js DISK_COPY: Qwen3.8 Flash, which the
// room never picks by itself and which asks before it stores anything: room/diskpick-ui.js, room/flashdisk.js). The
// others stay in the (?dev=1) select for tests, never in the rows.
const DEV = document.documentElement.classList.contains("dev");
const PICK_NEED = Object.fromEntries([...PICKER, ...Object.keys(DISK_COPY)].map((k) => [k, NEED_GB[k]]));
function addModelOption(key) {
  const sel = $("ai-model");
  if (!MODELS[key] || [...sel.options].some((o) => o.value === key)) return;
  sel.add(new Option(shortName(key), key));
}
if (DEV) Object.keys(MODELS).forEach(addModelOption);
// set the picker to a model, adding it when another device started one the picker does not list
function setModelValue(key) { if (!MODELS[key]) return; addModelOption(key); $("ai-model").value = key; }
// How much more a device could still lend: its kind's cap (phones), the cap it reported itself and
// what its memory probe allows, less what it lends now
function spareGBOf(meta) {
  const rule = pledgeRule(meta?.ua, NaN);
  const ceil = Math.min(rule.capped ? rule.max : 64, meta?.pledgeMax || 64, meta?.budgetGB ? Math.max(SMALLEST_NEED, Math.floor(meta.budgetGB)) : 64);
  return Math.max(0, +(ceil - pledgeGB(meta)).toFixed(1));
}
// The picker's context choice per model (Qwen3.8 Flash: room/ctxpick.js ctxOptions), remembered for this room in this
// tab (sessionStorage, per room code): the longest context the room may open at. Start deals with it (aiStart:
// roomCtxPlan's choice, as roomFitFor), and the devices load the room's context from ai-load. 0: none (the longest proven
// context the room holds). ?ctx=N overrides it (an explicit ask, tried alone).
const CTX_URL_ASK = +new URLSearchParams(location.search).get("ctx") || 0;
const CTX_PICK_KEY = "pooled-ctx";
function ctxChoiceFor(key) {
  try { return +(JSON.parse(sessionStorage.getItem(CTX_PICK_KEY) || "{}")[`${roomCode}|${key}`]) || 0; } catch { return 0; }
}
function setCtxChoice(key, c) {
  try {
    const all = JSON.parse(sessionStorage.getItem(CTX_PICK_KEY) || "{}");
    all[`${roomCode}|${key}`] = c;
    sessionStorage.setItem(CTX_PICK_KEY, JSON.stringify(Object.fromEntries(Object.entries(all).slice(-16))));
  } catch {}
}
// Whether this room's pledges hold `key` the way the host will deal it (room/plan.js roomFit): each
// device's pledge (held to its kind's cap) in whole layers with their KV cache at the room's context,
// the model host's (room/plan.js pickModelHost) less the embedding and the head. The context is the
// one the host will pick (room/models.js pickCtx): the model's default, or its fallback (the 1.7B: 8K
// for 16K) when only that fits; a short room is short for the fallback. null for a model without a
// SHAPE (room/models.js) or a room with no device that can hold layers.
function roomFitFor(key) {
  const devs = [{ id: peer?.id || "self", name: myName, meta: myMeta }, ...[...members].map(([id, m]) => ({ id, name: m.name || "device", meta: m.meta || {} }))]
    .filter((d) => d.meta?.webgpu && !d.meta?.api);
  const ask = CTX_URL_ASK, kvAsk = roomQwen35Options(location.search).kvQ8 ? "q8" : null;
  if (!devs.length || !roomBytes(key, maxSeqFor(key, ask), kvModeFor(key, kvAsk))) return null;
  const hid = pickModelHost(devs);
  devs.sort((a, b) => (b.id === hid) - (a.id === hid));
  const pledges = devs.map((d) => dealPledgeBytes(d.meta, Infinity, key)), metas = devs.map((d) => d.meta);
  // One plan with Start (room/ctxpick.js roomCtxPlan, aiStart calls it too): the context (the picker's choice, then the
  // ladder below it; Flash: the room's K / V format and the longest context its devices take), each layer's bytes there
  // (Flash: room/models.js flashLayerCost; roomBytes is memoized per context), and a tab model host's Flash load margin
  // (room/ctxpick.js hostMarginFor; a room node host has its load check instead)
  const margin = hostMarginFor(key, devs[0].meta);
  const plan = roomCtxPlan(key, { metas, pledges, ask, choice: ctxChoiceFor(key), kvAsk, margin }), kv = plan.kv;
  const fitAt = (c, pl = pledges) => plan.fitAt(c, pl).fit;
  const pick = plan;
  const fit = fitAt(pick.ctx), spare = devs.map((d) => spareGBOf(d.meta));
  const out = { fit, devs, ctx: pick.ctx, want: pick.want, fellBack: pick.fellBack, kv,
    // the picker's context choices (Flash): each one's need here, and why it is off (room/ctxpick.js ctxOptions)
    ctxOpts: ask ? [] : ctxOptions(key, { metas, pledges, margin, kvAsk }),
    shortGB: gbUp(shortBy(fit, spare)), note: shortNote(shortName(key), fit, devs.map((d) => d.name), spare, PAGE_SHORT_FIX) };
  // a model kept on disk (room/models.js DISK_COPY): its need is the GPU memory the planner itself needs from these
  // devices (room/plan.js minFitPledge over roomFit, with their offload and disk tiers), not its size. needGB, with
  // needDisk when the rest streams from disk there; the picker's line, the pool's line and the row all use it.
  if (diskCopyOf(key)) {
    // stores: some device here can store it (room/pledge.js storageOf, read-only; offloadFor's rules): the one test for
    // "no device can store it" that the row, the line under Start, the picker's note and the pool all read. No numbers
    // change with it
    out.stores = devs.some((d, i) => pledges[i] > 0 && !!storageOf(d.meta, key).kind);
    const has = pledges.reduce((a, b) => a + b, 0), m = minFitPledge(pledges, (pl) => fitAt(pick.ctx, pl).fits);
    if (m) {
      const needGB = fit.fits ? Math.min(gbUp(m.bytes), +(has / 2 ** 30).toFixed(1)) : gbUp(m.bytes);
      const at = fitAt(pick.ctx, m.scaled);
      Object.assign(out, { needGB, needDisk: !!(at.disk || at.offload), shortGB: fit.fits ? 0 : Math.max(0.1, +(needGB - has / 2 ** 30).toFixed(1)) });
    }
    // share: what each tab here would copy to its disk (name -> bytes), from the deal the host would make now (dealRoom,
    // as aiStart calls it, the split by memory); only tabs copy (a room node reads its own files). null when it doesn't fit
    if (fit.fits && SHAPE[key]?.exp) {
      const rb = roomBytes(key, pick.ctx, kv), metas = devs.map((d) => d.meta);
      // (Flash: each layer's own cost and a tab host's load margin, as aiStart deals it)
      const dealX = (x) => dealRoom({ L: rb.L, ...(rb.layerCost ? { layerCost: rb.layerCost } : { layerBytes: rb.layerBytes }), hostBytes: rb.hostBytes + x, pledges,
        phone: metas.map(isPhoneMeta), phoneLayers: new URLSearchParams(location.search).get("phonelayers") === "1",
        off: offloadFor(metas, rb.experts || rb.expertBytes, key), ngram: ngramFor(metas, ngramOf(key)) });
      const deal = margin ? dealWithHostMargin(dealX, ...margin) : dealX(0);
      if (deal.fit.fits) {
        const sh = diskShares(deal, devs.length, SHAPE[key].exp);
        out.share = new Map(devs.map((d, i) => [d.name, d.meta?.tabDisk ? sh[i] : null]));
      }
    }
  }
  return out;
}
// the ladder with each row's fit from the real deal where the model has a SHAPE (NEED_GB otherwise)
function fitLadder(pledged) {
  return ladder(PICK_NEED, pledged).map((x) => {
    const f = roomFitFor(x.key);
    return f ? { ...x, ok: f.fit.fits, short: f.shortGB, f } : x;
  });
}
let ladderHTML = "";
function renderLadder(pledged, rows = fitLadder(pledged)) {
  const el = $("ai-ladder"); if (!el) return;
  const none = !(pledged > 0);
  // a radio group: one tab stop (the picked row), arrows move the pick. The rows are re-rendered
  // on every change, so the focus follows the picked row when it was in the group.
  const had = el.contains(document.activeElement);
  const html = (none ? '<p class="ai-nogpu">Needs a device with WebGPU</p>' : "") + rows.map((x) => {
    // fits only at the model's fallback context: say which, with what it needs there
    const back = x.ok && x.f?.fellBack;
    const gb = `<span class="nd">${back ? `${NEED_MIN_GB[x.key] ?? ""} GB · ${ctxK(x.f.ctx)}` : `${NEED_GB[x.key] ?? ""} GB`}</span>`;
    const fig = x.ok ? `${gb}<b>fits</b>` : none ? gb : `<span class="more">${x.short} GB short</span>`;
    // with no device that can hold layers, nothing reads as picked: there is nothing to start yet
    const sel = !none && x.key === $("ai-model").value;
    // a model kept on disk (room/diskpick.js): its size and where it runs from as the chip; picked, the row takes the
    // Code tab's selected look (black, a soft blue glow, stars twinkling: site/js/sparkle.js).
    // Its fit chip is the other rows' ("fits" / "N GB short"), from the same roomFitFor as the pool's line: the one
    // fit answer every visitor sees (the need line above the rows is the developer's)
    const dk = diskCopyOf(x.key);
    // (no device here can store it, roomFitFor stores: the chip says so, not the whole model's GPU gap, and the
    // "runs from disk" tag steps aside for it)
    const noStore = !none && !x.ok && x.f?.stores === false;
    if (dk) return `<button type="button" role="radio" class="rung disk${x.ok ? " ok" : " short"}${sel ? " sel" : ""}" data-k="${x.key}" aria-checked="${sel}" tabindex="${sel ? 0 : -1}" title="${esc(x.ok ? rowTitle(dk) : noStore ? "Needs a device that can store it" : x.f?.note || "Invite a device to fit this")}"><span class="rn">${esc(dk.name)}</span><span class="fig">${noStore ? '<span class="more">needs disk</span>' : `${rowChipHTML(dk)}${none ? "" : x.ok ? "<b>fits</b>" : `<span class="more">${x.short} GB short</span>`}`}</span></button>`;
    return `<button type="button" role="radio" class="rung${x.ok ? " ok" : " short"}${sel ? " sel" : ""}" data-k="${x.key}" aria-checked="${sel}" tabindex="${sel ? 0 : -1}"${back ? ` title="${esc(ctxShortNote(shortName(x.key), x.f.ctx, x.f.want))}"` : x.ok ? "" : ` title="${esc(x.f?.note || (giveFor(x.short) ? "Raise This device gives, or invite a device" : "Invite a device to fit this"))}"`}><span class="rn">${esc(shortName(x.key))}</span><span class="fig">${fig}</span></button>`;
  }).join("");
  // the same rows: leave them be (the twinkling stars keep their rhythm through the room's updates)
  if (html === ladderHTML && el.childElementCount) return;
  el.innerHTML = ladderHTML = html;
  el.querySelectorAll(".rung.disk").forEach((b) => window.pooledTwinkles?.(b));
  if (!el.querySelector(".rung.sel")) el.querySelector(".rung")?.setAttribute("tabindex", "0");
  if (had) el.querySelector('.rung[tabindex="0"]')?.focus({ preventScroll: true });
}
function pickRung(b) { if (!b || $("ai-model").disabled) return; setModelValue(b.dataset.k); modelTouched = true; updateCluster(); }
// The context choice under the rows (a model with choices: Qwen3.8 Flash, room/ctxpick.js ctxOptions): each proven length
// with the GPU memory it needs in this room and about how long reading a full paste takes there; one this room can't
// hold is off and says why. The picked one is the context the room opens at (roomFitFor's: the choice, or the longest
// that fits). The same numbers as the need line, the pool and Start (roomCtxPlan). Room settings has the same choice
// (#ai-ctx-set) for a running room, where a change loads the model again (ctxChange).
let ctxHTML = "";
function ctxOptionWords(key, o, batch) {
  if (!o.fits) return o.why;
  const t = durationWords(prefillSeconds(key, o.ctx, { batch }));
  return `${o.needGB} GB${t ? ` · a full paste ${t}` : ""}`;
}
function renderCtx(row) {
  const el = $("ai-ctx"), key = $("ai-model").value, opts = row?.f?.ctxOpts || [];
  // (prompt passes run batched unless a device reads experts from its disk: room/flash.js, room-node shard.js)
  const batch = !row?.f?.fit?.disk;
  let html = "";
  if (opts.length) {
    const cur = row.f.ctx;
    html = `<p>Context: how much text it reads at once</p><div class="cx" role="radiogroup" aria-label="Context length">${opts.map((o) =>
      `<button type="button" role="radio" data-c="${o.ctx}" aria-checked="${o.ctx === cur}" class="${o.ctx === cur ? "sel" : ""}"${o.fits ? "" : " disabled"} title="${esc(`${o.label} context: ${ctxOptionWords(key, o, batch)}`)}">${o.label}<small>${esc(ctxOptionWords(key, o, batch))}</small></button>`).join("")}</div>`;
  }
  el.hidden = !html;
  if (html !== ctxHTML) el.innerHTML = ctxHTML = html;
  // room settings: the same choice, for a running room too
  const set = $("ai-ctx-set");
  $("set-ctx").hidden = !opts.length && !(ai.engine && CTX[ai.model]?.ladder);
  if (opts.length && document.activeElement !== set) {
    const sh = opts.map((o) => `<option value="${o.ctx}"${o.fits ? "" : " disabled"}>${o.label}${o.fits ? "" : ` (${esc(o.why)})`}</option>`).join("");
    if (set.dataset.sh !== sh) { set.innerHTML = sh; set.dataset.sh = sh; }
    set.value = String(ai.engine && ai.model === key ? ai.roomCtx || row.f.ctx : row.f.ctx);
  }
}
// A context choice. Before Start: remembered and the picker follows. While the model runs (this device hosts it): the
// room's context changes only by loading the model again on every device, so it says that and asks first; a no keeps it.
async function ctxChange(key, c) {
  if (!(c > 0)) return;
  if (ai.engine && ai.role === "host" && ai.model === key && c !== ai.roomCtx) {
    const ok = window.confirm(`Change ${shortName(key)}'s context to ${ctxK(c)}? The model loads again on every device; the chat stays and is read again on the next question.`);
    if (!ok) { $("ai-ctx-set").value = String(ai.roomCtx); return; }
    setCtxChoice(key, c);
    log("room", `${myName} changes the context to ${ctxK(c)}: loading the model again`);
    aiRedeal(true);
    return;
  }
  if (ai.engine && ai.role !== "host") { toast("Only the device that started the model can change its context"); return; }
  setCtxChoice(key, c);
  updateCluster();
}
$("ai-ctx").addEventListener("click", (e) => { const b = e.target.closest("button[data-c]"); if (b && !b.disabled) ctxChange($("ai-model").value, +b.dataset.c); });
$("ai-ctx-set").addEventListener("change", (e) => ctxChange(ai.engine ? ai.model : $("ai-model").value, +e.target.value));
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
  const cap = Math.min(myMeta.pledgeMax || 64, Math.max(SMALLEST_NEED, Math.floor(myMeta.budgetGB || 0)));
  return v <= cap ? Math.ceil(v) : 0;
}
$("ai-give").addEventListener("click", (e) => lendGB(+e.currentTarget.dataset.gb));
// -> the picker's rows (fitLadder), which renderPool reads too: one roomFitFor per model and room update, so the rows,
// the need line and the pool's line can't drift apart
function updateNeed(pledged) {
  const rows = fitLadder(pledged);
  if (!modelTouched && !ai.engine && !ai.busy && !$("ai-model").disabled) {
    // never a model that stores a copy on disk: that one is only picked by hand (it asks first)
    const auto = rows.filter((x) => !diskCopyOf(x.key)), fits = auto.filter((x) => x.ok);
    $("ai-model").value = (fits.length ? fits[fits.length - 1] : auto[0] || rows[0]).key;
  }
  renderLadder(pledged, rows);
  const row = rows.find((x) => x.key === $("ai-model").value);
  renderCtx(row);
  // a disk model's need is the planner's (roomFitFor needGB), else the model's size
  const need = row?.f?.needGB ?? (NEED_GB[$("ai-model").value] || 1);
  // fits: every device's layers within its pledge, the host's with the embedding and the head
  // (roomFitFor); a model without a SHAPE falls back to the sum of the pledges against NEED_GB
  const ok = row ? row.ok : pledged >= need;
  const shortGB = row ? row.short : Math.max(0, +(need - pledged).toFixed(1));
  // a disk model no device here can store (roomFitFor stores): short of a device that can, not of the whole model's
  // GPU figure, so no GB gap is said for it anywhere (the row, under Start, the picker's note, the pool)
  const noStore = !ok && !!diskCopyOf($("ai-model").value) && row?.f?.stores === false;
  const note = noStore ? diskShortNote($("ai-model").value, shortGB, false)
    : row?.f?.note || (ok ? "" : `This room is ${shortGB} GB short for ${shortName($("ai-model").value)}. ${PAGE_SHORT_FIX}.`);
  $("need-fill").style.width = Math.min(100, ok ? 100 : noStore ? 0 : pledged / (pledged + shortGB) * 100).toFixed(1) + "%";
  const has = +pledged.toFixed(1);
  // a model with a fallback context says both needs; a room that holds it only there says so
  // (a disk model: on GPUs, and whether the rest streams from disk; no device here can store it: a device that can)
  const needs = noStore ? "a device that can store it on disk"
    : row?.f?.needGB != null ? `${row.f.needGB} GB on GPUs${row.f.needDisk ? ", the rest streamed from disk" : ""}`
    : NEED_GB[$("ai-model").value] ? needText($("ai-model").value) : `${need} GB`;
  const back = ok && row?.f?.fellBack ? ` ${ctxShortNote(shortName($("ai-model").value), row.f.ctx, row.f.want)}.` : "";
  $("need-text").textContent = ok
    ? `Needs ${needs}. The room has ${has} GB.${back}`
    : `Needs ${needs}. The room has ${has} GB${noStore ? "" : `, ${shortGB} GB short`}.`;
  $("ai-need").classList.toggle("ok", ok);
  if (!ai.busy && !ai.engine) $("ai-start").disabled = !ok;
  // short: say by how much and who could give more, right under Start (a disk model: in GPU memory, and the fix its
  // disk answer leaves: room/diskpick.js shortNote)
  $("ai-short").hidden = ok || !(pledged > 0) || ai.busy || !!ai.engine;
  // written only when it changed: it is a live region (role=status), and this runs on every room update
  const shortText = diskCopyOf($("ai-model").value) ? diskShortNote($("ai-model").value, shortGB, row?.f?.stores !== false) : note;
  if ($("ai-short").textContent !== shortText) $("ai-short").textContent = shortText;
  // short, and this device alone can close the gap: offer that one tap next to the disabled Start
  const meI = row?.f ? row.f.devs.findIndex((d) => d.meta === myMeta) : -1;
  const give = giveFor(meI >= 0 && row.f.fit.raise[meI] < Infinity ? gbUp(row.f.fit.raise[meI]) : row?.f ? 0 : need - pledged);
  $("ai-give").hidden = ok || !give || ai.busy || !!ai.engine;
  // short because this device's disk storage is off (not asked, or a no): turn it on right here, under Start, not only
  // in the right column (room/diskpick-ui.js initDiskPick wires it)
  const dk = diskCopyOf($("ai-model").value), dState = dk ? diskStateOf($("ai-model").value) : null;
  const canAllow = !ok && (dState === "deny" || (dState === "ask" && row?.f?.stores === false));
  $("ai-allow-disk").hidden = !canAllow || ai.busy || !!ai.engine;
  if (canAllow) $("ai-allow-disk").textContent = allowLabel(dk);
  // what pressing Start costs this device: about its share of the weights file (layers are dealt by
  // memory given), so a phone on mobile data sees ~0.2 GB and a laptop alone the whole file
  const file = FILE_GB[$("ai-model").value];
  const mine = ok && file && myMeta.webgpu && myMeta.contribGB ? file * Math.min(1, myMeta.contribGB / pledged) : 0;
  const size = (gb) => gb < 1 ? `${Math.max(10, Math.round(gb * 1024 / 10) * 10)} MB` : `${gb.toFixed(1)} GB`;
  // (a disk model, Qwen3.8 Flash: showDiskPick below writes its own note, pickNote, over this one)
  $("ap-note").textContent = mine
    ? (mine >= file * 0.98 ? `Start downloads the whole ${size(file)} model to this device, once. It stays cached for next time.`
      : `Start downloads about ${size(mine)} to this device (its share of ${size(file)}), once. It stays cached for next time.`)
    : ok && file && !myMeta.webgpu ? "This device only asks questions, so it doesn't load any of the model."
    : "Each device gets only its own part of the model, once. It stays saved for next time.";
  // every device here is chat only: say why Start is off, right under it
  $("ai-why").hidden = pledged > 0 || ai.busy || !!ai.engine;
  // and say so beside the stepper, which is the one-tap fix (not a second device)
  $("ap-me-hint").hidden = ok || !give;
  $("ap-me-hint").textContent = `Raise this to ${give} GB to fit ${shortName($("ai-model").value)}.`;
  if (give) { $("ai-give").textContent = `Give ${give} GB from this device`; $("ai-give").dataset.gb = give; }
  // why Start is off, for screen readers (sighted users see it in the rows and the pool card)
  $("start-why").textContent = ok ? "" : !(pledged > 0) ? "No device with WebGPU yet. Invite one to start a model."
    : note;
  if (ok) $("ai-start").removeAttribute("aria-describedby"); else $("ai-start").setAttribute("aria-describedby", "start-why");
  // a disk model: "This device, for <model>" in the right column. Its Disk row says the RAM the tab's disk tier would
  // read it with (room/flashdisk.js tabDiskMeta: TAB_RAM_GB held to tabRamCap, whatever the stored answer); the
  // picker's note reads whether some device here can store it (roomFitFor stores)
  const tier = tabDiskCan ? tabDiskMeta(myMeta, { search: location.search, opfs: true, deviceMemory: navigator.deviceMemory }) : null;
  showDiskPick($("ai-model").value, { ramGB: tier?.ramGB ?? null, stores: row?.f?.stores !== false, needGB: row?.f?.needGB ?? null, fits: !!ok, shareBytes: row?.f?.share?.get(myName) ?? null, devs: row?.f?.devs?.length ?? 1 });
  if (ok && !wasReady) { $("ai-start").classList.remove("unlocked"); void $("ai-start").offsetWidth; $("ai-start").classList.add("unlocked"); }
  wasReady = ok;
  return rows;
}
$("ai-model").addEventListener("change", () => { modelTouched = true; updateCluster(); });
initDiskPick({ toast: (t) => toast(t) });
// the picker's consent, stored copy and delete are this tab's disk tier's (room/flashdisk.js, below askTabDisk and
// freeTabDisk): the same answer, the same bytes, the same guard
connectDiskStore({
  able: () => tabDiskCan,
  can: () => tabDiskCan && !!tabDiskMeta(myMeta, { search: location.search, opfs: true, deviceMemory: navigator.deviceMemory }),
  changed: (key) => {
    tabDiskDeclined = false;
    tabDiskChanged();
    log("room", `${myName} lets models use up to ${fmtBytes(diskCopyOf(key)?.diskBytes || 0)} of its disk`);
    toast("big models may use this device's disk");
  },
  free: () => freeTabDisk(),
});
function updateCluster() {
  const all = [myMeta, ...[...members.values()].map(m => m.meta).filter((m) => !m?.api)];
  const apis = members.size + 1 - all.length;
  const gpus = all.filter(m => m && m.webgpu).length;
  // only devices with WebGPU hold layers; the others join as ask-only guests
  const pledged = all.reduce((s, m) => s + (m?.webgpu ? m?.contribGB || 0 : 0), 0);
  const rows = updateNeed(pledged);
  const mem = all.reduce((s, m) => s + (m?.budgetGB || m?.maxBufGB || 0), 0);
  $("cluster-summary").textContent = DEV
    ? `${all.length} device${all.length > 1 ? "s" : ""}${apis ? ` \u00b7 ${apis} API client${apis > 1 ? "s" : ""}` : ""} \u00b7 ${gpus} WebGPU \u00b7 ${pledged.toFixed(1)} GB pledged`
    : `${all.length} device${all.length > 1 ? "s" : ""}${apis ? ` \u00b7 ${apis} API client${apis > 1 ? "s" : ""}` : ""} \u00b7 ${+pledged.toFixed(1)} GB pooled`;
  $("hdr-sum").innerHTML = `<b>${+pledged.toFixed(1)} GB</b><span class="hw"> pooled</span>`;
  $("peers-n").textContent = String(all.length);
  renderPool(pledged, rows);
}
// the device list scrolls past four and a half rows (p2p.html #ap-devs): fade its bottom while more rows are below
function devsEdge() { const l = $("ap-devs"); l.classList.toggle("more", l.scrollTop < l.scrollHeight - l.clientHeight - 1); }
$("ap-devs").addEventListener("scroll", devsEdge, { passive: true });
if (typeof ResizeObserver === "function") new ResizeObserver(devsEdge).observe($("ap-devs"));
// The model card's side: what the room pools (one segment per device, in its colour, with a tick at
// each model's need), what this device lends (+/-), and the invite (QR, code, copy link).
// rows: updateNeed's (fitLadder): each model's roomFitFor, computed once for the picker and this side alike
let storeHTML = "";
function renderPool(pledged, rows = []) {
  const devs = [{ name: myName, meta: myMeta }, ...[...members.values()].map((m) => ({ name: m.name || "device", meta: m.meta || {} }))].sort((x, y) => bySlot(x.name, y.name))
    .filter((d) => d.meta?.webgpu && d.meta?.contribGB && !d.meta?.api);
  // each model's need: a disk model's from the planner (roomFitFor needGB), so the tick, the line and the picker agree
  const had = new Map(rows.map((x) => [x.key, x.f ?? null]));
  const fitOf = new Map(Object.keys(PICK_NEED).map((k) => [k, had.has(k) ? had.get(k) : roomFitFor(k)]));
  const needs = Object.entries(PICK_NEED).map(([k, gb]) => [k, fitOf.get(k)?.needGB ?? gb]).sort((a, b) => a[1] - b[1]);
  const sel = $("ai-model").value, dk = diskCopyOf(sel), F = dk ? fitOf.get(sel) : null;
  // a disk model picked (room/diskpick.js pool words): the GPU bar shows only its tick, at the GPU memory the planner
  // needs when the rest is stored on some device's disk; with no device that can store it there is no GPU figure to
  // show (roomFitFor's would be the whole model on GPUs, which is not what it needs)
  const dkTick = dk && F?.needDisk && F?.needGB != null ? F.needGB : null;
  const top = (dk ? Math.max(pledged, dkTick || 0) : Math.max(pledged, ...needs.map((x) => x[1]))) * 1.06 || 1;
  $("ap-total").textContent = `${+pledged.toFixed(1)} GB`;
  const meter = $("ap-meter");
  meter.querySelector(".apm-fill").innerHTML = devs.map((d) => `<i style="--sw:${devColor(d.name)};width:${(d.meta.contribGB / top * 100).toFixed(2)}%" title="${esc(String(d.name))}: ${d.meta.contribGB} GB"></i>`).join("");
  // a tick where each model starts to fit; the picked one says what it needs
  const fitsK = (k, gb) => { const f = fitOf.get(k); return f ? f.fit.fits : pledged >= gb; };
  // a disk model's tick says GPU memory, or, with no device here to store it, that it needs one
  const tickTitle = (k, gb) => !diskCopyOf(k) ? `${shortName(k)} needs ${gb} GB`
    : fitOf.get(k)?.stores === false ? `${shortName(k)} needs a device that stores it`
    : fitOf.get(k)?.needDisk ? `${shortName(k)} needs ${gb} GB of GPU memory` : `${shortName(k)} needs more GPU memory`;
  const ticks = dk ? (dkTick != null ? [[sel, dkTick]] : []) : needs;
  meter.querySelector(".apm-ticks").innerHTML = ticks.map(([k, gb]) => `<span class="${fitsK(k, gb) ? "ok" : ""}${k === sel ? " sel" : ""}${gb / top > 0.6 ? " r" : gb / top < 0.3 ? " l" : ""}" style="left:${(gb / top * 100).toFixed(2)}%" title="${esc(tickTitle(k, gb))}"></span>`).join("");
  // what the selected model needs, as its own line under the meter (not a label hanging off its tick)
  const selNeed = needs.find(([k]) => k === sel);
  $("ap-need").hidden = !selNeed;
  // a disk model: what each device brings beyond GPU memory (room/pledge.js storageOf, offloadFor's rules, read-only)
  const st = dk ? new Map(devs.map((d) => [d, storageOf(d.meta, sel)])) : null;
  if (selNeed && dk) {
    const all = [...st.values()], nDisk = all.filter((x) => x.kind === "disk").length, nRam = all.filter((x) => x.kind === "ram").length;
    // whether any device can store it is roomFitFor's stores (the row, under Start and the picker read the same)
    const fits = fitsK(sel, selNeed[1]), w = { needGB: F?.needGB ?? null, needDisk: !!F?.needDisk, fits, shortGB: F?.shortGB ?? 0, nStore: F?.stores === false ? 0 : nDisk + nRam };
    const g = poolGpuLine(dk, w), sl = poolStoreLine(dk, { ...w, nDisk, nRam, state: diskStateOf(sel) });
    const chip = (c) => c ? `<span>${esc(c)}</span>` : "";
    // the words in an <em> of their own, so a long line wraps beside its chip (p2p.html #ap-need em)
    $("ap-need").innerHTML = `<em>${esc(g.text)}${g.num != null ? ` <b>${g.num} GB</b>` : ""}</em>${chip(g.chip)}`; $("ap-need").classList.toggle("ok", g.ok);
    $("ap-need").classList.add("fl");
    $("ap-store-n").textContent = `${nDisk + nRam} of ${devs.length} can store it`;
    $("ap-smeter").querySelector(".apm-fill").innerHTML = devs.map((d) => { const x = st.get(d);
      return `<i class="${x.kind === "disk" ? (x.ask ? "ask" : "") : x.kind === "ram" ? "ram" : "no"}" style="--sw:${devColor(d.name)}"></i>`; }).join("");
    // written only when it changed: it is a live region, and renderPool runs on every room update
    const sh = `<em>${esc(sl.text)}</em>${chip(sl.chip)}`;
    if (sh !== storeHTML) { $("ap-store").innerHTML = storeHTML = sh; $("ap-store").classList.toggle("ok", sl.ok); }
    const fix = poolFix(diskStateOf(sel), w);
    $("ap-store-fix").textContent = fix; $("ap-store-fix").hidden = !fix;
  } else if (selNeed) {
    // the verdict is the row's (fitsK: roomFitFor's fit), the figures the planner's
    const [k, gb] = selNeed, f = fitOf.get(k), fits = fitsK(k, gb), short = f ? f.shortGB : +Math.max(0, gb - pledged).toFixed(1);
    $("ap-need").classList.remove("fl");
    $("ap-need").innerHTML = `${esc(shortName(k))} needs <b>${gb} GB</b>${f?.needGB != null ? " on GPUs" : ""}<span>${fits ? "fits" : `${short} GB short`}</span>`; $("ap-need").classList.toggle("ok", fits);
  }
  const flash = !!(dk && selNeed);
  for (const id of ["ap-gpu-k", "ap-store-k", "ap-smeter", "ap-store"]) $(id).hidden = !flash;
  if (!flash) $("ap-store-fix").hidden = true;
  $("ap-devs").innerHTML = devs.map((d) => { const x = flash ? st.get(d) : null;
    return `<li style="--sw:${devColor(d.name)}"${x ? ` title="${esc(devStoreTitle(String(d.name), d.meta.contribGB, x, F?.share?.get(d.name), d.name === myName && diskStateOf(sel) === "deny"))}"` : ""}><i></i><span>${esc(String(d.name))}${d.name === myName ? " <small>(this device)</small>" : ""}</span><b>${d.meta.contribGB} GB${x ? `<small class="dk"> · ${esc(devStoreTag(x, F?.share?.get(d.name), d.name === myName && diskStateOf(sel) === "deny", dk, d.name === myName))}</small>` : ""}</b></li>`; }).join("")
    || '<li class="none">No device with WebGPU yet</li>';
  devsEdge();
  const can = !!myMeta.webgpu;
  $("ap-step").hidden = !can; $("ap-no").hidden = can;
  if (can) { if (document.activeElement !== $("ap-gb")) $("ap-gb").value = myMeta.contribGB; $("ap-minus").disabled = myMeta.contribGB <= lendMin(); $("ap-plus").disabled = myMeta.contribGB >= lendMax(); }
}
const lendRule = () => pledgeRule(myMeta.ua, navigator.deviceMemory);
const lendMin = () => (myMeta.phone ? 0.5 : lendRule().min);
const lendMax = () => myMeta.pledgeMax || 64;
const lendStep = () => (myMeta.phone ? 0.5 : lendRule().step);
function selfSteps(card) { const s = card.querySelectorAll(".gbstep .step"); if (s.length) { s[0].disabled = myMeta.contribGB <= lendMin(); s[1].disabled = myMeta.contribGB >= lendMax(); } }
// lend a different amount: this device's card, the room's total, and every other device hear it
function lendGB(v) {
  v = Math.round(Math.min(lendMax(), Math.max(lendMin(), v)) * 10) / 10;
  if (!myMeta.webgpu || v === myMeta.contribGB) return;
  myMeta.contribGB = v;
  const selfCard = document.querySelector(".peer-card.self");
  if (selfCard) { setLends(selfCard, v); selfSteps(selfCard); }
  updateCluster(); broadcastAll({ t: "pledge", gb: v });
}
$("ap-minus").addEventListener("click", () => lendGB(myMeta.contribGB - lendStep()));
$("ap-plus").addEventListener("click", () => lendGB(myMeta.contribGB + lendStep()));
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
  $("room-badge").textContent = formatCode(roomCode);
  $("room-badge").setAttribute("aria-label", `Room ${spokenCode(roomCode)}: invite a device`);
  $("room-h").textContent = `Room ${formatCode(roomCode)}`;
  $("side-code").textContent = formatCode(roomCode);
  $("side-code").addEventListener("click", openShare);
  $("ap-qr").innerHTML = qrSVG(roomLink(), { size: 112 });
  // Chat | Code shows from the lobby on, so a visitor who came for Code sees where it is; Code stays
  // off (codeGate) until a model is running (?mock=code: at once, there is no model)
  if (isHost) $("host-controls").hidden = false;
  $("mode-bar").hidden = false; codeGate();
  peerCard("self", myName, myMeta, true);
  updateCluster();
  log("room", `${formatCode(roomCode)}: type this code on your other devices${isHost && gate?.ask ? " (you let each new device in), or open the invite link (no asking)" : ""}`);
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
    buf.addEventListener("click", (e) => { const b = e.target.closest(".step"); if (b) lendGB(myMeta.contribGB + +b.dataset.d * lendStep()); });
  }
  selfSteps(selfCard);
}

// --- connection wiring ---
// hold: "mesh" (a link to another device: held until it proves the room's mesh key) or "host" (this
// device's link to the room's host: held until the host let it in and proved what this device proved);
// stripesLater: the side that dialed opens its stripes once the hold is over
function wire(conn, name, meta, initiator = false, { hold = null, stripesLater = false } = {}) {
  // a link this device opened (or took) to a device in the room's roster: its name and meta from the
  // roster until its hello says (the hello can be lost: see the roster message)
  const known = members.get(conn.peer);
  const entry = { conn, name: name || known?.name || conn.peer, meta: meta || known?.meta || {}, rtt: null, card: null, link: makeLink({ dup: WIRE_DUP }), stripes: [], seen: performance.now(), path: null, initiator, stripesLater: !!(hold && stripesLater) };
  if (hold) holdLink(entry, hold);
  const prev = conns.get(conn.peer);
  conns.set(conn.peer, entry);
  if (prev && prev.conn !== conn) retire(prev);   // a new link to a device we already had one to
  watchLink(conn, () => linkDied(entry));
  if (WIRE_STRIPES > 0) {
    attachWire(entry.link, conn, (m) => onData(conn.peer, m));
    // extra associations for striping: the side that dialed opens them, the other side accepts
    // them in peer.on("connection") by label and attaches its end of the wire channel
    if (initiator && !entry.stripesLater) for (let i = 1; i < WIRE_STRIPES; i++) dialStripe(entry, conn.peer);
  }
  notePath(entry);

  conn.on("data", (d) => onData(conn.peer, d));
  let done = false;   // once per link: a dropLink and PeerJS's own close later
  const onClose = () => {
    const e = conns.get(conn.peer);
    if (done) return;
    if (e && e.conn !== conn) return;   // an older link to the same device
    if (!e && isHost && !roster.has(conn.peer)) return;   // already dropped (dropStaleNamesake)
    done = true;
    peerGone(conn.peer, e);
  };
  conn.on("close", onClose);
  entry.drop = onClose;   // dropLink: the close handling at once, even when PeerJS never emits "close"
  conn.on("error", () => {});
  return entry;
}
// ---- held links (room/chanauth.js) ----
// What a held link sends waits (up to 256 messages) and so does what this device sends it, until it
// proves the mesh key ("mesh") or the host lets this device in ("host"). Pings pass, so it is not
// dropped as silent meanwhile.
const PASS_HELD = new Set(["ping", "pong", "leaving", "mesh", "hello", "auth-proof", "bye"]);
const HOST_HANDSHAKE = new Set(["hello", "auth", "lobby", "admit", "bye"]);
const HELD_MAX = 256;
function holdLink(e, kind) {
  e.hold = { kind, q: [], out: [] };
  if (kind !== "mesh") return;
  e.hold.timer = setTimeout(async () => {
    if (!e.hold || conns.get(e.conn.peer) !== e) return;
    const v = await meshVerdict(e.conn, e.initiator ? "dial" : "accept", null);
    if (!e.hold) return;
    if (v === "legacy") { releaseLink(e, "legacy"); return; }
    log("room", `${e.name}: no proof that this link belongs to the room in ${MESH_WAIT_MS / 1000} s; closing it`);
    dropLink(e.conn.peer, "closed an unproved link");
  }, MESH_WAIT_MS);
}
function releaseLink(e, why = "ok") {
  const h = e?.hold;
  if (!h) return;
  e.hold = null; clearTimeout(h.timer);
  const id = e.conn.peer;
  if (why === "legacy") log("room", `${e.name}: linked without a proof (it runs an older Pooled)`);
  if (h.kind === "host" && e.stripesLater) for (let i = 1; i < WIRE_STRIPES; i++) dialStripe(e, id);
  for (const m of h.out) sendTo(id, m);
  for (const m of h.q) onData(id, m);
}
// this end's mesh message for a link (or a stripe): its proof, or none: 1 without a mesh key
async function meshHello(conn, role) {
  const dialer = role === "dial" ? peer.id : conn.peer, acceptor = role === "dial" ? conn.peer : peer.id;
  const p = meshKey ? await meshProof(meshKey, { fps: linkFingerprints(conn), dialer, acceptor, role }) : null;
  return p ? { t: "mesh", v: AUTH_V, p } : { t: "mesh", v: AUTH_V, none: 1 };
}
function sendMesh(conn, role) { meshHello(conn, role).then((m) => { try { conn.send(m); } catch {} }).catch(() => {}); }
// the other end's mesh message (null: none came) -> "ok" | "bad" | "legacy" (a device the host listed
// as one from before the proofs, while they are allowed) | "wait" (the roster doesn't list it yet)
async function meshVerdict(conn, role, d) {
  if (!meshKey) return "ok";   // a room whose host hands out no mesh key (an older host): nothing to check
  if (d?.t === "mesh" && d.p) {
    const dialer = role === "dial" ? peer.id : conn.peer, acceptor = role === "dial" ? conn.peer : peer.id;
    return (await meshCheck(meshKey, { fps: linkFingerprints(conn), dialer, acceptor, role: role === "dial" ? "accept" : "dial" }, d.p)) ? "ok" : "bad";
  }
  if (!LEGACY_AUTH) return "bad";
  const m = isHost ? (roster.has(conn.peer) ? roster.get(conn.peer) : null) : members.get(conn.peer);
  if (!m) return "wait";
  return m.a ? "bad" : "legacy";
}
async function meshIn(e, d) {
  const h = e.hold;
  if (!h || h.kind !== "mesh") return;
  const v = await meshVerdict(e.conn, e.initiator ? "dial" : "accept", d);
  if (e.hold !== h) return;
  if (v === "wait") { h.waitOn = d; return; }   // asked again when the roster comes
  if (v === "ok" || v === "legacy") { releaseLink(e, v); return; }
  log("room", `${e.name}: the link didn't prove it belongs to this room (someone else, or someone in the middle); closing it`);
  dropLink(e.conn.peer, "closed an unproved link");
}
// a stripe joins its link's wire once it proved the mesh key. attach(held): held is its wire channel,
// opened at once (room/transport.js holdWire): the other end may finish its check first and send
// slices straight away, which wait there, in order, until this end's check passes
function proveStripe(e, sc, role, attach) {
  if (!meshKey) { attach(null); sendMesh(sc, role); return; }
  const held = holdWire(sc);
  let done = false;
  const finish = (v) => {
    if (done) return;
    done = true; clearTimeout(timer);
    if (conns.get(sc.peer) !== e) { held?.drop(); try { sc.close(); } catch {} return; }
    if (v === "ok" || v === "legacy") attach(held);
    else { held?.drop(); log("room", `a stripe to ${e.name} didn't prove it belongs to this room; closed it`); try { sc.close(); } catch {} }
  };
  const timer = setTimeout(() => meshVerdict(sc, role, null).then((v) => finish(v === "legacy" ? v : "bad")), MESH_WAIT_MS);
  sc.on("data", (d) => { if (!done && d?.t === "mesh") meshVerdict(sc, role, d).then((v) => { if (v !== "wait") finish(v); }); });
  sendMesh(sc, role);
  meshVerdict(sc, role, null).then((v) => { if (v === "legacy") finish(v); });
}

// Close a link that stopped carrying anything (a locked phone, a tab iOS suspended): the data channel
// itself only says so when ICE gives up, ~30 s later, or never. Runs the close handling at once even
// when PeerJS does not emit "close" for a channel that is already dead.
function dropLink(id, why) {
  const e = conns.get(id);
  if (!e) return;
  log("room", `${e.name || id}: ${why}`);
  try { e.conn.close(); } catch {}
  for (const sc of e.stripes || []) { try { sc.close(); } catch {} }
  if (conns.get(id) === e) e.drop();
}
// a link that is not usable any more: closed, its channel not open, or silent too long
function linkDead(e, limit = LINK_SILENT_MS) {
  if (!e) return true;
  const dc = e.conn?.dataChannel;
  if (e.conn && e.conn.open === false) return true;
  if (dc && (dc.readyState === "closed" || dc.readyState === "closing")) return true;
  return linkSilent(Math.max(lastHeard(e), visibleSince), performance.now(), limit);
}
// When this tab is visible: the time it last became visible (silence while it was hidden is not
// the other side's fault: its own timers and data were frozen); when it went hidden.
// (performance.now(), the clock of e.seen and the ping loop)
let visibleSince = performance.now(), hiddenAt = null;
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") { hiddenAt = performance.now(); return; }
  const away = backFromAway(hiddenAt, performance.now());
  visibleSince = performance.now(); hiddenAt = null;
  if (!away || !peer) return;
  // back from a lock or the background: the signaling link is probably gone, and the data links may be
  // too. Reconnect the first now; give the others a few pings' time to show they are alive.
  signalBack();
  const back = visibleSince;
  setTimeout(() => {
    for (const [id, e] of [...conns]) if (lastHeard(e) < back) dropLink(id, "no answer after this screen came back: reconnecting");
  }, 8000);
});
// the signaling server link (PeerJS's websocket): iOS closes it when the screen locks. Without it this
// device can't open new links (to the host after a lock, or to a new chain neighbour).
function signalBack() {
  if (peer && peer.disconnected && !peer.destroyed) { try { peer.reconnect(); } catch {} }
}

// --- dead links ---
// A network that stops passing packets for longer than ICE's write timeout (~15 s: a frozen
// Wi-Fi, a laptop lid, a phone switching networks) kills the link's candidate pairs for good:
// Chrome reports connectionState "failed", but iceConnectionState stays "disconnected", SCTP
// still says "connected" and every data channel stays "open", so PeerJS never closes the
// connection. Nothing sent on it arrives again, and without this the room would wait on it
// forever (every answer timing out). So a failed link is replaced: the side that dialed it dials
// a new one to the same device (same peer id, same layers, a fresh wire); the other side waits
// RELINK_WAIT_MS for that, then closes the link (the device left, as before).
const RELINK_WAIT_MS = 45000;
function watchLink(conn, onDead) {
  const pc = conn.peerConnection;
  if (!pc) return;
  let fired = false;
  const check = () => {
    if (fired || !conn.open) return;
    if (pc.connectionState === "failed" || pc.iceConnectionState === "failed") { fired = true; onDead(); }
  };
  pc.addEventListener("connectionstatechange", check);
  pc.addEventListener("iceconnectionstatechange", check);
}
function linkDied(entry) {
  const id = entry.conn.peer;
  if (conns.get(id) !== entry) return;
  log("room", `the link to ${entry.name} went down (no packets for too long)${entry.initiator ? "; reconnecting" : ""}`);
  chainLinkLost(id, entry.name);
  linkState(id, entry.name, false);
  if (!entry.initiator) {
    setTimeout(() => { if (conns.get(id) === entry) { log("room", `${entry.name} did not reconnect`); try { entry.conn.close(); } catch {} } }, RELINK_WAIT_MS);
    return;
  }
  relink(entry, id, 0);
}
// dial a replacement link; wire() swaps it in and retires the dead one
function relink(entry, id, tries) {
  if (conns.get(id) !== entry || !peer || peer.destroyed) return;
  if (tries >= 8) { log("room", `could not reconnect to ${entry.name}`); try { entry.conn.close(); } catch {} return; }
  const c = peer.connect(id, { reliable: true });
  // PeerJS returns nothing while it is cut off from the signaling server: try again later
  if (!c) { setTimeout(() => relink(entry, id, tries + 1), 2000 * Math.min(tries + 1, 4)); return; }
  let done = false;
  const retry = () => { if (done) return; done = true; try { c.close(); } catch {} setTimeout(() => relink(entry, id, tries + 1), 2000 * Math.min(tries + 1, 4)); };
  const to = setTimeout(retry, 20000);
  c.on("open", () => {
    if (done) return;
    done = true; clearTimeout(to);
    if (conns.get(id) !== entry) { try { c.close(); } catch {} return; }
    linkOpened(c, { name: entry.name, meta: entry.meta, extra: { back: 1 } });
    log("room", `reconnected to ${entry.name}`);
  });
  c.on("error", () => {});
  c.on("close", retry);
}
// a replaced link: close it and its stripes quietly (its close handlers see it is not current)
function retire(old) {
  for (const s of old.stripes) try { s.close(); } catch {}
  try { old.conn.close(); } catch {}
  chainLinkLost(old.conn.peer, old.name);
  linkState(old.conn.peer, old.name, true);
}
// Host: links in the chain that are down and being replaced, "reporter|peer" -> { name, at }. A
// question waits for them (up to RELINK_WAIT_MS) instead of sending frames into a dead link.
// Workers report their own links with ai-linklost {up}.
const linksDown = new Map();
function linkState(id, name, up) {
  if (ai.role === "worker") { const h = ai.hostId || PREFIX + roomCode; if (id !== h) sendTo(h, { t: "ai-linklost", name: String(name || ""), up: up ? 1 : 0 }); return; }
  if (ai.role !== "host" || (!up && !ai.chain?.includes(id))) return;
  noteLink(peer.id + "|" + id, name, up);
}
function noteLink(key, name, up) {
  if (up) linksDown.delete(key); else linksDown.set(key, { name, at: performance.now() });
}
async function linksUp() {
  const live = () => { for (const [k, v] of linksDown) if (performance.now() - v.at > RELINK_WAIT_MS + 5000) linksDown.delete(k); return [...linksDown.values()]; };
  const t0 = performance.now();
  for (let d = live(); d.length && performance.now() - t0 < RELINK_WAIT_MS; d = live()) {
    aiStatus(`reconnecting to ${[...new Set(d.map((x) => x.name))].join(", ")}…`);
    await new Promise((r) => setTimeout(r, 250));
  }
}
// host: frames in flight on a lost link are gone, so a lap waiting on them fails now instead of
// timing out, and the next question prefills from scratch
// (a worker whose link to another device dropped tells the host with ai-linklost, see linkState)
function chainLinkLost(id, name) {
  if (ai.role !== "host" || !ai.engine || !ai.chain?.includes(id)) return;
  if (!ai.waiters.size && ai.fed == null) return;
  failWaiters(new Error(`the link to ${name || "a device"} dropped; ask again`));
  ai.fed = null; ckptClear(true);
}

// One extra association for the wire. A stripe can fail on its own on a bad network (its ICE
// times out while the main link survives): the dialing side opens a new one a few times, so the
// link does not stay on fewer associations for the rest of the session.
function dialStripe(entry, id, tries = 0) {
  const sc = peer.connect(id, { reliable: true, label: "stripe" });
  if (!sc) { if (tries < 4) setTimeout(() => { if (conns.get(id) === entry && entry.conn.open) dialStripe(entry, id, tries + 1); }, 2000 * (tries + 1)); return; }
  let opened = false;
  sc.on("open", () => { opened = true; proveStripe(entry, sc, "dial", (held) => { wireStripe(entry.link, sc, held, (m) => onData(id, m)); watchLink(sc, () => sc.close()); }); });
  sc.on("error", () => {});
  sc.on("close", () => {
    entry.stripes = entry.stripes.filter((c) => c !== sc);
    if (conns.get(id) !== entry || !peer || peer.destroyed || tries >= 4) return;
    setTimeout(() => { if (conns.get(id) === entry && entry.conn.open) dialStripe(entry, id, opened ? 0 : tries + 1); }, 2000 * (tries + 1));
  });
  entry.stripes.push(sc);
}
// direct or through the TURN relay: read once the link has settled, logged, and shown in pooledDebug()
function notePath(entry) {
  setTimeout(async () => {
    if (conns.get(entry.conn.peer) !== entry) return;
    await pathOf(entry.conn.peer);
    if (entry.path === "relay") log("room", `link to ${entry.name} goes through the relay (TURN${entry.via ? " over " + entry.via.toUpperCase() : ""})${PEER_WEIGHTS ? "; model weights never go over it" : ""}`);
  }, 3000);
}
// this link's path now ("direct", "relay", or null before ICE picked one), read from its stats
async function pathOf(id) {
  const e = conns.get(id), pc = e?.conn?.peerConnection;
  if (!e) return null;
  if (e.path) return e.path;
  try { const st = pc ? await pc.getStats() : null; e.path = linkPath(st); e.via = linkRelayProtocol(st); } catch {}
  return e.path;
}

// a link is gone (closed, or dropped as silent): on the host the device left; workers wait for the roster
// (e.dead: it was dropped for silence, "stopped responding")
// the host this device followed left the room (the creator's roster dropped it): free for the next Start. A device
// that only asked (a guest, no layers) goes back to the picker instead of a stale online screen; a creator that
// left ends the room as before (hostGone)
function followedHostLeft() {
  const was = ai.dealer; ai.pressedHere = false;
  ai.dealer = null; ai.dealerFresh = false; ai.dealerActive = false; ai.startAt = null; clearTimeout(ai.startWait); ai.startWait = null;
  if (was === PREFIX + roomCode || ai.role === "host" || ai.engine || ai.loadingShard) return;
  if (ai.hostId === was) ai.hostId = null;
  aiLoading(false);
  $("ai-panel").classList.remove("online");
  $("ai-start").disabled = false; $("ai-model").disabled = false; updateCluster();
  aiStatus("the model's host left the room: press Start to start it again");
}
function peerGone(id, e) {
  conns.delete(id);
  hbForget(liveness, id);
  // links to or from it that were down and being replaced: nothing will replace them now, so a
  // question must not wait for them (linksUp)
  for (const [k, v] of linksDown) if (k.endsWith("|" + id) || k.startsWith(id + "|") || (e?.name && v.name === e.name)) linksDown.delete(k);
  if (isHost) {
    dropCard(id); members.delete(id); roster.delete(id); broadcastRoster();
    if (ai.dealer === id) followedHostLeft();   // the member it followed as the model host left
    const verb = e?.dead ? "stopped responding" : "left";
    log("room", `${e?.meta?.api ? "API client " : ""}${e?.name || id} ${verb}`);
    apiPeerGone(id);
    aiPeerLeft(id, e?.name, verb);
  } else if (id === PREFIX + roomCode) {
    if (admission === "wait" || admission === "lobby") refused("Lost the link to the room's host before it let this device in. Try again.");
    else if (admission !== "out") { log("room", "lost the link to the host"); hostGone(); }
  }
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
    if (performance.now() - roomSince > 2500) speedEvent(`${who} joined`);
    mascot(`${name || id} joined! ${members.size + 1} devices in the room.`);
  }
  const e = conns.get(id);
  if (e) e.card = card;
  return card;
}
function dropCard(id) { const c = cards.get(id); if (c) { c.remove(); cards.delete(id); presence(c.dataset.name || id, false); speedEvent(`${c.dataset.name || id} left`); } }
// open a data link to a chain neighbour if we do not have one yet; resolves when it is up
const linked = (id) => { const e = conns.get(id); return !!e && !e.hold; };   // up, and proved
function ensureLink(id, timeoutMs = 60000) {
  if (id && conns.has(id) && linkDead(conns.get(id))) dropLink(id, "stale link replaced");
  if (!id || id === "host" || linked(id)) return Promise.resolve(true);
  if (!ensureLink.pending.has(id) && !conns.has(id)) { ensureLink.pending.add(id); meshConnect(id); }
  return new Promise((res) => {
    const t0 = performance.now();
    const t = setInterval(() => {
      if (linked(id)) { clearInterval(t); ensureLink.pending.delete(id); res(true); }
      else if (performance.now() - t0 > timeoutMs) { clearInterval(t); ensureLink.pending.delete(id); res(false); }
    }, 100);
  });
}
ensureLink.pending = new Set();

function sendTo(id, obj) {
  const e = conns.get(id);
  if (e?.hold && !PASS_HELD.has(obj?.t)) { if (e.hold.out.length < HELD_MAX) e.hold.out.push(obj); return; }
  e?.conn.send(obj);
}
// debug: per-peer wire state (channels open, frames sent/received) — `pooledDebug()` in the console (`swarmDebug()` still works)
window.pooledDebug = window.swarmDebug = () => [...conns].map(([id, e]) => ({ id, name: e.name, chans: e.link?.chans.filter((c) => c.readyState === "open").length ?? 0, sent: e.link?.sent ?? 0, recv: e.link?.recv ?? 0, ka: e.link?.kaSent ?? 0, dups: e.link?.dups ?? 0, skipped: e.link?.skipped ?? 0, path: e.path, via: e.via || null }));
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
  if (e?.hold) { sendTo(id, msg); return; }
  if (e?.link && wireReady(e.link) && sendFrame(e.link, msg)) return;
  sendTo(id, msg);
}
function broadcastAll(obj) { for (const [id] of conns) sendTo(id, obj); }

// bandwidth test state
const bwRecv = new Map(); // fromId -> {bytes, t0}

function onData(from, d) {
  const seenE = conns.get(from); if (seenE) seenE.seen = performance.now();   // any message counts as a sign of life (the ping loop drops silent links)
  // binary chunk = bandwidth test payload
  if (d instanceof ArrayBuffer || ArrayBuffer.isView(d)) {
    const st = bwRecv.get(from);
    if (st) st.bytes += d.byteLength || d.length;
    return;
  }
  const e = conns.get(from);
  if (!d || typeof d.t !== "string") return;
  // a held link: only what proves it gets through, and pings (room/chanauth.js)
  if (e?.hold) {
    if (e.hold.kind === "mesh" && d.t === "mesh") { meshIn(e, d); return; }
    // once the host's admit is in (its proof being checked), everything but pings waits behind it, so the
    // room's messages count in the order the host sent them (ai-ready-all comes right behind admit)
    if (!(d.t === "ping" || d.t === "pong" || (!e.hold.admitting && (d.t === "leaving" || (e.hold.kind === "host" && HOST_HANDSHAKE.has(d.t)))))) {
      // a device from before the proofs says hello without auth, and sends no mesh message either
      if (e.hold.kind === "mesh" && d.t === "hello" && !(+d.auth >= AUTH_V)) meshIn(e, d);
      if (e.hold.q.length < HELD_MAX) { e.hold.q.push(d); return; }
      // more than a link can have to say before it is proved: start over rather than lose a message
      e.hold.q = [];
      dropLink(from, "too many messages while the link was being proved; closed it");
      return;
    }
  }
  if (d.t === "mesh") return;   // a proof on a link that needs none, or is proved already
  if (d.t.startsWith("ai-")) { aiOnData(from, d); return; }
  switch (d.t) {
    case "hello":
      // one protocol per room: a tab from an older or newer deploy is told to reload
      if (d.v !== PROTOCOL) { versionRefused(from, d); break; }
      // the host's hello: one from before the gate (no gate: 1) lets every device in, so go in now
      if (!isHost && from === PREFIX + roomCode) {
        if (e) e.hostHello = d;
        hostAsks = !!d.gate && !!d.ask;
        // a host from before the gate: nothing to prove, and it never got this device's key or pass.
        // Holding one, it goes in on trust (as for a typed code), and says so
        if (!d.gate && (joinKey || myPass) && admission === "wait") { log("room", TOFU_HOST); toast(TOFU_HOST); }
        if (!d.gate) releaseLink(e);
        if (!d.gate && admission === "wait") guestIn();
      }
      versionSeen.delete(from);   // back on the same version (a reload): its bye counts again
      // a peer picks its own name: keep it a short plain string (it is also escaped wherever it is shown)
      d.name = String(d.name ?? from).replace(/[\u0000-\u001f\u007f<>"'`&]/g, "").trim().slice(0, 40) || String(from).slice(0, 8);
      // an API client (`pooled serve`, cli/): an ask-only guest with no layers; the host may refuse it
      d.meta = helloMeta(d.meta, isHost);   // on a guest, meta.api is the host saying it serves API clients
      // names are the room's keys (colours, layers, load progress, the plan): the host makes a taken one
      // unique ("laptop 2"), and the device takes the name the roster gives it
      if (isHost) {
        // the name is held by a device that has been quiet for a second: ping it and decide in a moment
        // a device that says it is back (back: 1) under a new link: its old link may still look alive
        // (its close not seen yet), so probe it whatever its silence, or the room lists it twice and a
        // re-deal deals both
        const quiet = !probedHellos.has(d) && (quietNamesake(d.name, from, roster, heardOf, performance.now())
          || (d.back ? [...roster].find(([rid, m]) => rid !== from && m.name === d.name)?.[0] || null : null));
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
      if (!seenAt.has(from)) seenAt.set(from, performance.now());
      if (d.meta?.api && isHost && !apiWelcome(from, d)) break;
      members.set(from, { ...members.get(from), name: d.name, meta: d.meta });
      ensureCard(from, d.name, d.meta);
      if (isHost) {
        // the model runs on another device (biggestPeerId): say which, so this one takes its ai-ready-all
        // when it links in (welcomeFar)
        // (only a host this creator checked on a Start it saw, room/authz.js: a newcomer takes no other)
        if (ai.role !== "host" && ai.dealerFresh && ai.dealer && ai.dealer !== peer.id && conns.has(ai.dealer) && ai.dealer !== from) sendTo(from, { t: "ai-modelhost", id: ai.dealer });
        if (!seqOf.has(from)) seqOf.set(from, ++nextSeq);
        roster.set(from, { name: d.name, meta: d.meta, seq: seqOf.get(from), ...(+d.auth >= AUTH_V ? { a: 1 } : {}) }); broadcastRoster();
        if (!aiLoadDeath(from, d)) aiRejoin(from, d.name);
        if (d.died?.during && d.died.ago > 2) log("room", `${d.name} came back: its tab was killed ${d.died.ago} s ago while ${d.died.during}. Phones kill background tabs; keep the screen on.`);
        if (ai.visibility !== "all") sendTo(from, { t: "ai-visibility", mode: ai.visibility });
        // a device back (a reload rejoins by itself with back: 1) that holds no layers is welcomed too:
        // without ai-ready-all it showed the model picker as if no model ran (it keeps its own chat)
        aiWelcome(from, { history: !d.back });
        codeWelcome(from);
      }
      break;
    case "leaving": {   // the tab is closing: treat the link as gone now instead of waiting for ICE to time out
      // (and for PeerJS's close, which can come late or never: until then the device would stay listed,
      // and come back beside itself under a new link)
      const e = conns.get(from);
      try { e?.conn.close(); } catch {}
      if (e && conns.get(from) === e) e.drop?.();
      break;
    }
    case "auth":    // the host's challenge: prove what this device holds, bound to this link
      if (isHost || from !== PREFIX + roomCode || !e?.jauth || e.jauth.hn) break;
      joinerProof(e.jauth, d, { fps: linkFingerprints(e.conn), me: peer.id, host: from }).then((m) => {
        if (m && conns.get(from) === e) try { e.conn.send(m); } catch {}
      }).catch((err) => console.warn("auth", err));
      break;
    case "admit":   // the host let this device in: keep the pass it gave, for coming back
      if (isHost || from !== PREFIX + roomCode) break;
      if (e?.hold) e.hold.admitting = true;   // what comes next waits behind it while its proof is checked
      hostAdmitted(e, from, d).catch((err) => { console.warn("admit", err); if (e?.hold) { e.hold.admitting = false; try { e.conn.close(); } catch {} } });
      break;
    case "lobby":   // the host was asked: wait for Allow or Deny
      if (isHost || from !== PREFIX + roomCode) break;
      if (e?.jauth) e.jauth.lobbied = true;
      inLobby(e?.jauth?.sas, !!e?.jauth?.proved, !!e?.oldHost);
      break;
    case "bye":
      // the host said no (Deny, a full lobby, a tab too old to wait) before this device got in: back
      // to the join screen with its reason, and no knocking on the host again
      if (!isHost && from === PREFIX + roomCode && admission !== "in") { refused(d.reason); break; }
      if (versionSeen.has(from)) break;   // a version mismatch this tab already explained in its own words
      toast(d.reason);
      log("room", d.reason);
      if (from === PREFIX + roomCode) {
        $("room-over").hidden = false; $("room-over-h").textContent = "Room over"; $("room-over-why").textContent = d.reason;
        // the host closed the room for good (pooled host q): its link closing next is not a host to wait for
        if (d.closed) { admission = "out"; clearInterval(hostGone.timer); aiStatus("the host closed the room"); }
      }
      break;
    case "roster": {
      if (from === PREFIX + roomCode && Array.isArray(d.members)) {
        rosterOrder = joinOrder(d.members);
        authzView = new Map(d.members.filter((m) => m?.id && !m.meta?.api).map((m) => [m.id, m.meta || {}]));
      }
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
        members.set(m.id, { name: m.name, meta: m.meta, ...(m.a ? { a: 1 } : {}) });
        if (!seenAt.has(m.id)) seenAt.set(m.id, performance.now());
        const c = ensureCard(m.id, m.name, m.meta);
        if (m.meta?.contribGB) setLends(c, m.meta.contribGB);
        const ce = conns.get(m.id); if (ce) ce.meta = m.meta;
        // the room's names are the host's (it makes them unique): a link's name comes from here too, not
        // only from the hello on it. That hello is the first message the other side sends when the link
        // opens, and WebRTC can lose it (a fresh link in Chromium drops the accepting side's first message
        // now and then): a joiner's link to the room's creator then kept its placeholder "host", and a
        // model host that is not the creator dealt with the creator under that name, so the creator found
        // no reason under its own ("Not needed" missing: room_state leftout-guest's flake)
        if (ce && from === PREFIX + roomCode && !isHost && m.name) ce.name = m.name;
      }
      for (const id of [...members.keys()]) if (!seen.has(id)) { members.delete(id); dropCard(id); }
      if (ai.dealer && ai.dealer !== peer.id && !seen.has(ai.dealer)) followedHostLeft();
      // links that waited to see whether the roster lists their device as one from before the proofs
      if (from === PREFIX + roomCode) for (const [, c] of conns) if (c.hold?.waitOn) { const w = c.hold.waitOn; c.hold.waitOn = null; meshIn(c, w); }
      // this device runs the model but not the room: a device it has no link to can't ask it yet
      for (const id of seen) if (!members.get(id)?.meta?.api) welcomeFar(id);
      updateCluster();
      apiPanel();   // the API clients in the roster: the Serve API dot and list
      break;
    }
    case "ping": sendTo(from, { t: "pong", ts: d.ts }); break;
    case "pong": {
      e.missed = 0;
      e.rtt = Math.round(performance.now() - d.ts);
      if (Number.isFinite(d.ts)) e.pongFor = Math.max(e.pongFor || 0, d.ts);   // drop detection: a ping sent then was answered
      if (e.card) e.card.querySelector(".rtt").textContent = e.rtt + " ms";
      break;
    }
    case "pledge": {
      // tabDisk (when the message has it): the device's disk tier changed (its user said yes or no: room/pledge.js tabDiskGB)
      const upd = (meta) => {
        const m = { ...meta, contribGB: d.gb };
        if ("tabDisk" in d) { const t = cleanTabDisk(d.tabDisk); if (t) m.tabDisk = t; else delete m.tabDisk; }
        // local: a room node with the disk tier downloaded a model after it joined (packages/room-node announceLocal):
        // the models whose files are all on its disk now (meta.local, room/pledge.js diskFor)
        if (m.disk && Array.isArray(d.local)) m.local = d.local.filter((k) => typeof k === "string" && k.length > 0 && k.length <= 64).slice(0, 32);
        return m;
      };
      if (e) { e.meta = upd(e.meta); if (e.card) setLends(e.card, d.gb); }
      if (members.has(from)) members.get(from).meta = upd(members.get(from).meta);
      if (isHost && roster.has(from)) { roster.get(from).meta = upd(roster.get(from).meta); broadcastRoster(); }
      if ("tabDisk" in d && cleanTabDisk(d.tabDisk) && !cleanTabDisk(d.tabDisk).ask) ai.noDisk?.delete(e?.name || from);   // it said yes after all
      updateCluster();
      break;
    }
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

// ---- joining: the guest's side (room/joingate.js; docs/protocol.md "Joining a room") ----
let hostAsks = false;   // guest: the host asks before new devices join (its hello), for the Invite sheet
function guestIn() {
  if (admission === "in" || admission === "out") return;
  admission = "in";
  joinWait(false);
  $("jw-cancel").hidden = true;
  const f = afterAdmit; afterAdmit = null;
  f?.();
}
// the host has been asked about this device: the waiting screen (or, for a device already in the room
// whose link came back without a pass the host knows, the room's own card)
function inLobby(sas = null, linkFailed = false, oldHost = false) {
  if (admission === "in") {
    $("room-over").hidden = false;
    $("room-over-h").textContent = "Waiting for the host";
    $("room-over-why").textContent = "This device's link to the room came back, and the host has to let it in again.";
    return;
  }
  if (admission !== "wait") return;
  admission = "lobby";
  joinWait(true, "Waiting for the host to let you in");
  $("join-status").textContent = `The host of room ${formatCode(roomCode)} sees \u201c${myName} wants to join\u201d.`;
  // the six digits the host sees beside the request: the same on both screens only when nobody sits in
  // the middle of the link (room/chanauth.js sasOf)
  if (sas || linkFailed || oldHost) {
    const c = document.createElement("span");
    c.className = "jw-sas";
    c.textContent = (oldHost ? " " + OLD_HOST : linkFailed ? " This invite link didn't check out with the host, so it has to let you in." : "") + (sas ? ` Check that it shows the code ${sas}.` : "");
    $("join-status").append(c);
  }
  $("jw-cancel").hidden = false;
}
// the host's admit on a link to it: when this device proved its invite key or pass, the host must have
// proved it back (room/chanauth.js joinerCheckAdmit) before anything from the room counts
async function hostAdmitted(e, from, d) {
  const st = e?.jauth;
  const v = st ? await joinerCheckAdmit(st, d) : "ok";
  // nothing proved back (a host restarted without its saved passes, a link from an earlier room, Allow
  // after the lobby, an older host): in on trust, as for a typed code, and said out loud
  if (v === "tofu") { const w = TOFU_HOST + (st.sas ? ` (code ${st.sas})` : ""); log("room", w); toast(w); }
  if (v === "bad") {
    console.warn("auth: the host's proof is missing or wrong");
    if (e) { e.authFailed = true; try { e.conn.close(); } catch {} }
    if (admission !== "in") refused(UNVERIFIED_HOST);
    else { admission = "out"; clearInterval(hostGone.timer); log("room", UNVERIFIED_HOST); toast(UNVERIFIED_HOST, { kind: "error" }); }
    return;
  }
  if (e && conns.get(from) !== e) return;
  if (e) e.jauth = null;
  if (validKey(d.pass)) { myPass = d.pass; keepPass(roomCode, myPass); }
  if (validMeshKey(d.mk)) meshKey = d.mk;
  releaseLink(e);
  if (admission !== "in") { if (admission === "lobby") toast("The host let you in"); guestIn(); }
  else if (!$("room-over").hidden && $("room-over-h").textContent === "Waiting for the host") $("room-over").hidden = true;
}
const UNVERIFIED_HOST = "Couldn't verify this room's host: its proof of the invite key is wrong, so someone may be in the middle of the connection. Try again later, or join with the room code and check the six digits with the host.";
const TOFU_HOST = "Couldn't verify this room's host with the invite link (fine if it restarted or runs an older Pooled; otherwise check the six digits with the host).";
const OLD_HOST = "This room's host runs an older Pooled: ask them to let you in, or to update.";
// the host said no, or went away, before this device got in
function refused(reason) {
  admission = "out";
  $("jw-cancel").hidden = true;
  const p = peer; peer = null;
  try { p?.destroy(); } catch {}
  conns.clear();
  joinFailed(String(reason || DENIED_TEXT).slice(0, 300));
  $("join-status").classList.add("refused");
}
// passes this tab was given, per room code (sessionStorage: this tab only; not a virtual device's frame)
const PASS_KEY = "pooled-passes";
function storedPass(code) {
  if (VQ.get("embed")) return "";
  try { const p = JSON.parse(sessionStorage.getItem(PASS_KEY) || "{}")[code]; return validKey(p) ? p : ""; } catch { return ""; }
}
function keepPass(code, pass) {
  if (VQ.get("embed")) return;
  try {
    const all = JSON.parse(sessionStorage.getItem(PASS_KEY) || "{}");
    all[code] = pass;
    sessionStorage.setItem(PASS_KEY, JSON.stringify(Object.fromEntries(Object.entries(all).slice(-8))));
  } catch {}
}

// ---- joining: the host's side ----
// Ask before new devices join, on by default (?ask=0 turns it off for this room: the e2e tests that
// join by typed code or ?code= links)
const ASK_DEFAULT = new URLSearchParams(location.search).get("ask") !== "0";
const LOBBY_BUF = 64;   // messages a link may send between its hello and the host's answer
// A link the host did not open: it says hello, and the gate decides. Until it is let in the link is
// not in `conns`, so nothing the room sends (roster, chat, layers, pings) reaches it and nothing it
// sends reaches the room; it only gets its pongs. Messages after its hello wait (up to LOBBY_BUF)
// and are handled once it is in.
function gateConn(conn) {
  const id = conn.peer;
  const L = { conn, hello: null, buf: [], stripes: [], auth: null };
  const prev = lobbyConns.get(id);
  if (prev && prev.conn !== conn) {
    try { prev.conn.close(); } catch {}
    // its request goes too: an Allow must answer the link whose name and code it showed, not a newer
    // one under the same id (the new link asks again, with its own six digits)
    if (gate && gateWithdraw(gate, id)) joinRequests();
  }
  lobbyConns.set(id, L);
  const onMsg = (d) => {
    if (!d || typeof d.t !== "string") return;   // binary (bandwidth tests): not from a device in the lobby
    if (d.t === "ping") { try { conn.send({ t: "pong", ts: d.ts }); } catch {} return; }
    if (d.t === "leaving") { try { conn.close(); } catch {} return; }
    const broke = (err) => { console.warn("gate", err); gateRefuse(L, "The host couldn't check this device. Try again."); };
    // its proofs, for the challenge this host sent (once)
    if (d.t === "auth-proof" && L.auth) { const p = L.auth; L.auth = null; gateProof(L, p, d).catch(broke); return; }
    if (L.hello) { if (L.buf.length < LOBBY_BUF) L.buf.push(d); return; }
    if (d.t !== "hello") return;
    L.hello = d;
    gateHello(L, d).catch(broke);
  };
  L.onMsg = onMsg;
  conn.on("data", onMsg);
  conn.on("close", () => {
    if (lobbyConns.get(id) !== L) return;
    lobbyConns.delete(id);
    for (const s of L.stripes) try { s.close(); } catch {}
    if (gate && gateWithdraw(gate, id)) { log("room", `${cleanName(L.hello?.name, id)} stopped waiting to join`); joinRequests(); }
  });
  conn.on("error", () => {});
}
const cleanName = (n, id) => String(n ?? id).replace(/[\u0000-\u001f\u007f<>"'`&]/g, "").trim().slice(0, 40) || String(id).slice(0, 8);
async function gateHello(L, d) {
  const id = L.conn.peer;
  const name = cleanName(d.name, id);
  if (d.v !== PROTOCOL) {   // told which side should reload, as on any link (versionRefused)
    const { local, remote } = versionMismatch({ mine: PROTOCOL, theirs: d.v, name, theyHost: false, me: myName, iAmHost: true });
    toast(local, { kind: "error" }); log("room", local);
    gateRefuse(L, remote);
    return;
  }
  const meta = helloMeta(d.meta, true);
  // API clients: the Allow API clients switch comes first (no point asking about one it would refuse)
  if (meta?.api && !ai.settings.apiAllow) { gateRefuse(L, "the host does not allow API clients in this room"); return; }
  // it speaks the proofs (room/chanauth.js): challenge it and decide on its answer (gateProof)
  if (hostWantsAuth(d)) {
    const { pending, msg } = startAuth(gate, d);
    L.auth = pending;
    try { L.conn.send(msg); } catch {}
    return;
  }
  const r = await gateDecide(gate, id, d);
  if (lobbyConns.get(id) !== L) return;   // it left, or a newer link from it took over, while the hash ran
  if (r.legacy) {
    log("room", `${name} runs an older Pooled that sent its ${r.via === "key" ? "invite key" : "pass"} the old way, unprotected: ask it to reload or update`);
    toast(`${name} runs an older Pooled: ask it to reload or update`);
  }
  gateAnswer(L, d, name, meta, r);
}
// the device's proofs -> in (the host's proof back), held for Allow (with the six digits), or a bye
async function gateProof(L, pending, proof) {
  const id = L.conn.peer, d = L.hello, name = cleanName(d.name, id);
  const r = await decideAuth(gate, id, d, pending, proof, { fps: linkFingerprints(L.conn), me: peer.id, peer: id });
  if (lobbyConns.get(id) !== L) return;
  if (r.failed) log("room", `${name}: its invite link or pass didn't check out on this link (an old link, or someone in the middle of the connection)`);
  gateAnswer(L, d, name, helloMeta(d.meta, true), r);
}
function gateAnswer(L, d, name, meta, r) {
  const id = L.conn.peer;
  if (r.kind === "admit") { gateAdmit(L, r.pass, r.via, r.hp); return; }
  if (r.kind === "refuse") {
    const why = meta?.api && !d.join ? "This room's host asks before new devices join, and this pooled serve is older. Update it (npx @pooled/cli@latest) and start it with the room's invite link." : r.reason;
    if (!d.join) log("room", `${name} runs an older Pooled that can't wait to be let in: told it to reload`);
    gateRefuse(L, why);
    return;
  }
  // ask the host. A device from before the proofs that showed the raw invite key (which this host
  // doesn't take, ?legacyauth=0) is marked, so the prompt says why it asks about one that came by a link
  if (r.oldRaw) { log("room", `${name} runs an older Pooled and can't prove the invite key: it waits for you to let it in (ask it to update)`); }
  gateEnqueue(gate, id, name, meta, Date.now(), r.sas, !!r.oldRaw);
  try { L.conn.send({ t: "lobby" }); } catch {}
  log("room", `${name} is waiting to join`);
  joinRequests(true);
}
// into the room: the link becomes a room link (wire), then its hello and whatever it sent meanwhile
// The admit carries the pass to come back with, the host's proof of the secret the device proved (via +
// hp: the device checks it), and the room's mesh key for its links to the other devices
function gateAdmit(L, pass, via, hp = null) {
  const id = L.conn.peer;
  lobbyConns.delete(id);
  L.conn.off("data", L.onMsg);
  const entry = wire(L.conn);
  // (an API client never links to other devices: it gets no mesh key, so a client the host disconnects
  // can't use one to dial the room's devices)
  const mk = meshKey && !L.hello?.meta?.api ? { mk: meshKey } : {};
  try { L.conn.send({ t: "admit", ...(pass ? { pass } : {}), ...(hp ? { via, hp } : {}), ...mk }); } catch {}
  if (via === "key") log("room", `${cleanName(L.hello?.name, id)} came in with the invite link`);
  onData(id, L.hello);
  // after its hello: the roster then says whether it proves its links (its stripes do too)
  for (const sc of L.stripes) if (sc.open) acceptStripe(entry, sc);
  for (const m of L.buf) onData(id, m);
  saveHost();
}
function gateRefuse(L, reason) {
  const id = L.conn.peer;
  if (lobbyConns.get(id) === L) lobbyConns.delete(id);
  try { L.conn.send({ t: "bye", reason }); } catch {}
  setTimeout(() => { try { L.conn.close(); } catch {} }, 400);   // after the bye is out
}
// a stripe's end of the wire: the channel it opened while it proved the mesh key (held), or a new one
function wireStripe(link, sc, held, onFrame) { if (held) held.join(link, onFrame); else attachWire(link, sc, onFrame); }
// a stripe someone else opened for a link: it joins the link's wire once it proved the mesh key
function acceptStripe(e, sc) {
  proveStripe(e, sc, "accept", (held) => {
    if (e.stripes.includes(sc)) { held?.drop(); return; }
    wireStripe(e.link, sc, held, (m) => onData(sc.peer, m)); e.stripes.push(sc);
    sc.on("close", () => { e.stripes = e.stripes.filter((c) => c !== sc); });
    watchLink(sc, () => sc.close());   // the dialing side opens a new one
  });
}
// the host's answer to a request (the Allow / Deny prompt)
async function answerJoin(id, yes) {
  const L = lobbyConns.get(id);
  if (yes) {
    const r = await gateAllow(gate, id);
    if (!r) return;
    if (L && lobbyConns.get(id) === L) { log("room", `let ${r.req.name} in`); gateAdmit(L, r.pass, "allowed"); }
  } else {
    const r = gateDeny(gate, id);
    if (r) log("room", `did not let ${r.name} in`);
    if (L) gateRefuse(L, DENIED_TEXT);
  }
  joinRequests();
}

// The host's prompt: the oldest request, Allow / Deny, and how many more wait. It never takes focus by
// itself (the host may be typing a question); a screen reader hears each new request at once, and Tab
// reaches the prompt right after the header. Answering one with the keyboard puts focus on the next
// one's Allow, and after the last back where it was before.
let jrShown = null, jrReturn = null;
function joinRequests(announce = false) {
  const box = $("join-reqs");
  const q = gate ? gate.lobby : [];
  const head = q[0];
  if (!head) {
    const hadFocus = box.contains(document.activeElement);
    box.hidden = true; jrShown = null;
    if (hadFocus) { const back = jrReturn; jrReturn = null; (back?.isConnected ? back : $("share-btn"))?.focus({ preventScroll: true }); }
    return;
  }
  const line = requestLine(head.name, head.meta);
  const changed = jrShown !== head.id;
  const hadFocus = box.contains(document.activeElement);
  jrShown = head.id;
  $("jr-line").textContent = line;
  // the six digits the device's waiting screen shows too: the same only when nobody sits in the middle
  // of the link (room/chanauth.js sasOf; a device from before them has none)
  $("jr-sas").textContent = head.sas ? `Check that its screen shows ${head.sas}` : "";
  $("jr-sas").hidden = !head.sas;
  $("jr-sub").textContent = head.old ? "An older Pooled device that can't prove the invite link. Ask it to update, or Allow it if you expect it."
    : head.meta?.api ? "An API client (pooled serve) that typed the room code. Once in, it can ask the model and see the chat."
    : "It typed the room code. Once in, it can hold layers and see the chat.";
  $("jr-more").textContent = q.length > 1 ? `${q.length - 1} more waiting` : "";
  box.hidden = false;
  if (changed && hadFocus) $("jr-allow").focus({ preventScroll: true });
  if (announce) {
    const n = q.length;
    $("jr-live").textContent = "";
    setTimeout(() => { $("jr-live").textContent = `${requestLine(q[n - 1].name, q[n - 1].meta)}. Allow or Deny it under the header${n > 1 ? `; ${n} requests waiting` : ""}.`; }, 50);
    if (document.visibilityState === "hidden") toast(requestLine(q[n - 1].name, q[n - 1].meta));
  }
}
$("join-reqs").addEventListener("focusin", (e) => { if (!jrReturn && e.relatedTarget && !$("join-reqs").contains(e.relatedTarget)) jrReturn = e.relatedTarget; });
$("jr-allow").addEventListener("click", () => { if (jrShown) answerJoin(jrShown, true); });
$("jr-deny").addEventListener("click", () => { if (jrShown) answerJoin(jrShown, false); });
// Room settings: Ask before new devices join (the host's gate; saved with the room)
function askSwitch() { if ($("ask-join")) $("ask-join").checked = !!gate?.ask; }
$("ask-join")?.addEventListener("change", (e) => {
  if (!gate) return;
  gate.ask = e.target.checked;
  toast(gate.ask ? "New devices wait until you let them in" : "Anyone with the room code can join now");
  log("room", gate.ask ? "asking before new devices join" : "not asking before new devices join: the room code is enough");
  saveHost();
});

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
  // a: 1 = it proves its links (room/chanauth.js); a link from one without it runs an older Pooled
  const members = [{ id: peer.id, name: myName, meta: myMeta, seq: 0, ...(meshKey ? { a: 1 } : {}) },
    ...[...roster.entries()].map(([id, m]) => ({ id, ...m }))];
  broadcastAll({ t: "roster", members });
}

function meshConnect(targetId) {
  const conn = peer.connect(targetId, { reliable: true });
  conn.on("open", () => linkOpened(conn));
}
// a link this device dialed is open: to the room's host it goes through the gate (joinHello, held until
// the host lets it in); to any other device it proves the room's mesh key first
function linkOpened(conn, { name, meta, extra = {} } = {}) {
  if (!isHost && conn.peer === PREFIX + roomCode) {
    wire(conn, name || "host", meta, true, { hold: "host", stripesLater: true });
    joinHello(conn).then((f) => conn.send(helloFor(conn.peer, { ...extra, ...f })));
    return;
  }
  wire(conn, name, meta, true, { hold: meshKey ? "mesh" : null });
  conn.send(helloFor(conn.peer, extra));
  sendMesh(conn, "dial");
}
// the hello this device sends on a link it opened (auth: it proves its links, room/chanauth.js)
function helloFor(id, extra = {}) {
  return { t: "hello", name: myName, meta: myMeta, v: PROTOCOL, auth: AUTH_V, ...extra };
}
// What the hello to the room's host adds: this device can wait in the lobby (join: 1), and what it can
// prove, never the secret itself (jc, kc, pid: room/chanauth.js joinerStart). Holding the invite key or
// a pass it first waits briefly for the host's hello: a host with the gate from before the proofs (no
// auth) gets the raw key and pass as it used to, with a warning (?legacyauth=0: never; it waits for
// Allow instead).
async function joinHello(conn) {
  const e = conns.get(conn.peer);
  const key = validKey(joinKey) ? joinKey : null, pass = validKey(myPass) ? myPass : null;
  if (key || pass) {
    const t0 = performance.now();
    while (conns.get(conn.peer) === e && !e?.hostHello && performance.now() - t0 < HOST_HELLO_WAIT_MS) await new Promise((r) => setTimeout(r, 25));
  }
  const hh = e?.hostHello;
  if (hh && hh.gate && !(+hh.auth >= AUTH_V) && (key || pass)) {
    if (LEGACY_SEND) {
      log("room", "this room's host runs an older Pooled: the invite key goes to it the old way, unprotected (?legacyauth=1)");
      return { join: 1, ...(pass ? { pass } : {}), ...(key ? { key } : {}) };
    }
    // it can't check a proof, and the raw key would go to whoever answers as that host: wait for Allow
    log("room", OLD_HOST + " (This device doesn't send it the invite key: it can't check it safely.)");
    toast(OLD_HOST);
    if (e) { e.jauth = await joinerStart({}); e.oldHost = true; }
    return { join: 1, ...(e?.jauth?.helloFields || {}) };
  }
  const st = await joinerStart({ key, pass });
  if (e) e.jauth = st;
  return { join: 1, ...st.helloFields };
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
      e.dead = true;
      try { e.conn.close(); } catch {}
    }
  }
  broadcastAll({ t: "ping", ts: now });
}, PING_MS);

// --- drop detection (room/liveness.js) ---
// While an answer runs, the host pings every device in the chain twice a second and treats
// anything it hears from one (a pong, any message, any wire slice) as a sign of life. A device
// silent past deadAfter(rtt) (3.5-5 s) is held: the host says so at once, the answer waits
// for it (Stop gives up) and a new question waits for it. A frozen device that comes back finishes
// the answer and keeps its place and layers (no re-deal); a link ICE gives up on (~15 s) fails the
// answer (chainLinkLost) and is redialed; a device that stays silent is dropped by the ping loop
// above (SILENT_MS, longer for a phone), which puts the room in the degraded / re-deal state.
// ?hb=0 turns it off.
const liveness = makeLiveness();
const HB_ON = new URLSearchParams(location.search).get("hb") !== "0";
let hbLast = 0;
function hbLoop() {
  const now = performance.now();
  const stalled = hbLast && now - hbLast > STALL_MS;   // this tab did not run: its inbox is stale
  hbLast = now;
  if (isHost && !stalled) hbSuspects(now);
  const on = HB_ON && isHost && ai.role === "host" && (ai.busy === "gen" || ai.busy === "code") && ai.chain.length > 0 && !ai.degraded;
  if (!on) { hbDisarm(liveness); hbTick(liveness, now, []); return; }
  hbArm(liveness, now);
  const ids = ai.chain.filter((id) => conns.has(id));
  for (const id of ids) hbHeard(liveness, id, Math.max(0, lastHeard(conns.get(id))));
  const r = hbTick(liveness, now, ids, (id) => conns.get(id)?.rtt);
  if (r.ping) for (const id of ids) sendTo(id, { t: "ping", ts: now });
  for (const { id, silentMs, limitMs } of r.dead) {
    const e = conns.get(id);
    if (!e || e.suspect) continue;
    e.suspect = { since: now - silentMs };
    console.warn(`[room] ${e.name} silent ${silentMs} ms (limit ${limitMs} ms, rtt ${e.rtt ?? "?"} ms): holding the answer for it`);
    log("room", `${e.name} stopped responding; waiting for it`);
    noteLink(peer.id + "|" + id, e.name, false);   // a new question waits for it (linksUp)
    // the answer is held, not failed: frames on a frozen link are late, not lost, and a device that
    // comes back finishes it. Stop gives up now; a link ICE gives up on fails it (chainLinkLost).
    aiStatus(`${e.name} stopped responding; waiting for it (Stop gives up)…`);
    toast(`${e.name} stopped responding; waiting for it`);
  }
}
// held devices: back once they answer a ping again. One that stays silent is dropped by the ping
// loop above (SILENT_MS for a computer, longer for a phone), which takes the departure path.
function hbSuspects(now) {
  for (const [id, e] of conns) {
    if (!e.suspect) continue;
    // back only on a round trip: a pong to a ping sent after the silence began (a late packet
    // still draining from before the freeze does not count)
    if (now - (e.suspect.pinged || 0) >= 500) { e.suspect.pinged = now; sendTo(id, { t: "ping", ts: now }); }
    if (suspectBack(e.pongFor || 0, e.suspect.since)) {
      e.suspect = null;
      noteLink(peer.id + "|" + id, e.name, true);
      log("room", `${e.name} is responding again`);
      toast(`${e.name} is back`);
    }
  }
}
setInterval(hbLoop, 250);
// tests: the longest silence seen per chain device while answering, and the limits in force
window.pooledLiveness = () => ({ armed: liveness.armed,
  maxSilence: Object.fromEntries([...liveness.maxSilence].map(([id, ms]) => [conns.get(id)?.name || id, Math.round(ms)])),
  limit: Object.fromEntries([...liveness.maxSilence.keys()].map((id) => [conns.get(id)?.name || id, deadAfter(conns.get(id)?.rtt)])) });

const stepGB = (d) => { const i = $("join-gb"); const lo = parseFloat(i.min) || 1; const st = parseFloat(i.step) || 1; i.value = Math.min(parseFloat(i.max) || 64, Math.max(lo, (parseFloat(i.value) || lo) + d * st)); };
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
    const m = /^(.*room )([A-Z0-9]{3}-[A-Z0-9]{3}|[A-Z0-9]{4,6})(.*)$/.exec(text), el = $("jw-t");
    if (m) { const b = document.createElement("b"); b.textContent = m[2]; el.replaceChildren(m[1], b, m[3]); } else el.textContent = text;
  }
}
// --- network: the optional TURN relay (room/ice.js) ---
function storedTurn() { try { return localStorage.getItem(TURN_KEY); } catch { return null; } }
function turnConfig() { return turnFrom(new URLSearchParams(location.search), storedTurn()); }
function turnForm() {
  const t = turnConfig();
  const note = (text) => { $("turn-note").textContent = text; };
  if (t?.from === "url") note("set by this page's link (?turn=)");
  else if (t?.urls?.length) note("relay saved");
  let saved = null; try { saved = JSON.parse(storedTurn() || "null"); } catch {}
  if (saved) { $("turn-url").value = [].concat(saved.urls || []).join(", "); $("turn-user").value = saved.username || ""; $("turn-cred").value = saved.credential || ""; $("turn-force").checked = !!saved.force; }
  $("turn-save").addEventListener("click", () => {
    const n = normTurn({ urls: $("turn-url").value, username: $("turn-user").value.trim(), credential: $("turn-cred").value, force: $("turn-force").checked });
    if (!n || !n.urls.length) { note(n?.bad ? `not a relay URL: ${n.bad[0]} (want turn:host:port or turns:host:port)` : "enter the relay's URL first"); return; }
    try { localStorage.setItem(TURN_KEY, JSON.stringify({ urls: n.urls, username: n.username, credential: n.credential, force: n.force })); } catch { note("this browser won't save it (private window?)"); return; }
    note(n.bad ? `saved; ignored ${n.bad.join(", ")}` : "saved");
  });
  $("turn-clear").addEventListener("click", () => {
    try { localStorage.removeItem(TURN_KEY); } catch {}
    for (const id of ["turn-url", "turn-user", "turn-cred"]) $(id).value = "";
    $("turn-force").checked = false;
    note("removed");
  });
}
if ($("join-net")) turnForm();

function joinFailed(text) {
  joinWait(false);
  $("join-status").classList.remove("signal-down");
  $("join-status").textContent = text;
  $("create-btn").disabled = $("join-btn").disabled = false;
  // on a phone the status line sits below the fold: bring it to where the user is looking
  $("join-status").scrollIntoView({ block: "center", behavior: "smooth" });
}
// --- join / create ---
// from: where in the server list to start (a joiner that found no room on one server tries the next)
async function start(create, resume = null, from = 0) {
  if (resume?.guest) { $("name-input").value = resume.name; if (resume.gb) $("join-gb").value = resume.gb; }
  myName = resume?.name || $("name-input").value.trim() || (create ? "host" : "peer") + "-" + rand(2);
  if (!VQ.get("embed")) try { sessionStorage.setItem(NAME_KEY, myName); } catch {}   // a virtual device's iframe shares the tab's storage
  const code = resume?.code || (create ? randomCode() : parseCode($("code-input").value));
  if (!code) { $("join-status").textContent = "Enter a room code"; return; }
  // what gets this device in: the invite key when it came by that room's link (or pasted it), and
  // the pass the host gave this tab before (a reload, or joining again after the room was over)
  if (!create) {
    joinKey = code === linkCode && linkKey ? linkKey : code === pastedLink.code ? pastedLink.key : "";
    myPass = resume?.pass || storedPass(code);
    admission = "wait";
  }
  $("create-btn").disabled = $("join-btn").disabled = true;
  joinWait(true, create ? (resume ? `Opening room ${formatCode(code)} again` : "Opening your room") : `Joining room ${formatCode(code)}`);
  $("join-status").classList.remove("signal-down");
  $("join-status").textContent = "Connecting…";
  myMeta = await metaPromise;
  const gbIn = parseFloat($("join-gb").value);
  myMeta.contribGB = Math.min(lendMax(), Math.max(lendMin(), gbIn > 0 ? gbIn : (myMeta.contribGB || 1)));
  // killed while loading layers last time (the breadcrumb below): come back with the smallest share
  if (myMeta.pledgeMax && diedCrumb?.loading) myMeta.contribGB = lendMin();

  // STUN for hole-punching; a TURN relay when there is one (room/ice.js: ?turn=, the Network box
  // under the join form, window.TURN_SERVERS, else the site's default relay from /api/turn with
  // credentials that expire). ICE prefers direct candidates, so the relay only carries traffic when
  // a direct path is impossible (a work network that blocks UDP), or when ?relay=1 forces it.
  const turn = turnConfig();
  if (turn?.bad) log("room", `ignored relay URL${turn.bad.length > 1 ? "s" : ""} ${turn.bad.join(", ")} (want turn:host:port or turns:host:port)`);
  const extra = Array.isArray(window.TURN_SERVERS) ? window.TURN_SERVERS : [];
  const params = new URLSearchParams(location.search);
  // is UDP getting out of this network? (answers in a few seconds; used to explain a failed join)
  udpProbe = probeUdp(window.RTCPeerConnection);
  const auto = wantDefaultRelay(params, location.hostname, turn, extra) ? await defaultRelay() : null;
  const ICE = iceConfig(turn, [...extra, ...markAuto(auto?.iceServers)], { force: params.get("relay") === "1" });
  relayFrom = turn?.urls?.length ? "yours" : extra.length ? "site" : auto ? "default" : null;
  relayOn = ICE.iceServers.some(isRelayServer);
  if (auto) keepRelayFresh(ICE, auto.ttl);
  // PeerJS is a deferred script from cdn.jsdelivr.net (p2p.html): without it the page still renders, so say why nothing connects
  if (typeof Peer !== "function") { joinFailed("couldn't load the connection library from cdn.jsdelivr.net (offline, or blocked by an extension or network). Reload to try again"); return; }
  // a host coming back after a reload goes to the server its room was on first: its guests are there
  let servers = SIGNALS;
  const was = resume?.signal && !SIGNAL ? parseServer(resume.signal, PAGE_SECURE) : null;
  if (was) servers = [was, ...SIGNALS.filter((x) => x.spec !== was.spec)];
  // host claims the well-known id for the code; joiners get random ids. The first server that answers
  // wins; one that is down or unreachable hands over to the next (room/signal.js).
  let got;
  try {
    got = await openPeer(Peer, create ? PREFIX + code : undefined, { debug: 1, config: ICE }, servers, {
      from,
      onTry: (s, i, err) => {
        $("join-status").textContent = err ? `${servers[i - 1].label} isn’t answering; trying ${s.label}…` : servers.length > 1 || from ? `Connecting to ${s.label}…` : "Connecting…";
        if (err) log("room", `signaling: ${servers[i - 1].label} failed (${err.type || err.message}); trying ${s.label}`);
      },
    });
  } catch (err) {
    // resuming: the old tab's id is still registered until the signaling server notices it left
    if (resume && err.type === "unavailable-id" && (resume.tries = (resume.tries || 0) + 1) < 30) {
      $("join-status").textContent = `waiting for room ${code} to be free again (the old tab is still registered)…`;
      setTimeout(() => start(true, resume), 3000);
      return;
    }
    if (err.type === "signaling-down") { signalingDown(err.tried); return; }
    joinFailed(err.type === "unavailable-id" ? "that code is already hosting a room: press Join instead"
      : "error: " + (err.type || err.message));
    return;
  }
  peer = got.peer;
  signalServer = got.server;
  if (got.index > 0 || from) log("room", `signaling on ${signalServer.label}`);
  watchSignaling(peer);
  networkNote(create);

  isHost = create;
  roomCode = code;
  seenAt.clear(); rosterOrder = []; authzView = new Map(); joinedAt = performance.now(); ai.dealer = null; ai.dealerFresh = false; ai.dealerActive = false; ai.startAt = null;
  seqOf.clear(); nextSeq = 0;
  if (create && resume?.seqs) for (const [id, n] of resume.seqs) if (typeof id === "string" && Number.isFinite(+n)) { seqOf.set(id, +n); nextSeq = Math.max(nextSeq, +n); }
  // the host's gate: a reloaded host keeps its invite key (links already shared keep working), the
  // passes it gave out and the Ask setting; a room saved by an older build gets a new one
  if (create) { gate = resume ? restoreGate(resume.gate, { ask: ASK_DEFAULT, legacy: LEGACY_AUTH }) : makeGate({ ask: ASK_DEFAULT, legacy: LEGACY_AUTH }); meshKey = gate.mk; askSwitch(); }
  let joinTimer = null;
  if (create) { enterRoom(); if (resume) resumeHost(resume); }
  else {
    // joiner: connect to host
    $("join-status").textContent = "Reaching the other devices…";
    const conn = peer.connect(PREFIX + code, { reliable: true });
    // "still connecting" after a few seconds; more time while the two devices are still finding a path
    const t0 = performance.now(), me = peer;
    joinTimer = setInterval(() => {
      // failed already (peer.on("error")), or a newer Join press owns the screen now
      if (peer !== me || !$("join-btn").disabled) { clearInterval(joinTimer); return; }
      const pc = conn.peerConnection, ice = pc?.iceConnectionState;
      const step = joinStep(performance.now() - t0, ice);
      if (step.fail) {
        clearInterval(joinTimer);
        try { conn.close(); } catch {}   // closed, so a late open can't pull a failed join into the room
        // the host answered (or ICE got as far as checking): the room exists, the path is what failed,
        // and a relay (TURN) server gets around that (room/ice.js)
        const found = !!pc?.remoteDescription || ice === "checking" || ice === "failed" || ice === "disconnected";
        joinFailed(!found ? step.fail : pathFailText(relayOn, relayFrom, udpSeen));
        if (found) docsLink();
        if (found && $("join-net")) $("join-net").open = true;
      } else if (step.status) $("join-status").textContent = step.status;
    }, 1000);
    conn.on("open", () => {
      clearInterval(joinTimer);
      let died = null;
      if (!VQ.get("embed") && diedCrumb) { const c = diedCrumb; died = { during: c.s, ago: Math.round((Date.now() - c.t) / 1000), at: c.t, loading: !!c.loading }; }
      linkOpened(conn, { name: "host", extra: { died, ...(resume?.guest ? { back: 1 } : {}) } });
      // into the room once the host lets this device in (admit), or at once when its hello shows a host
      // from before the gate (guestIn); until then the host holds it in its lobby (the waiting screen)
      $("join-status").textContent = "Waiting for the host…";
      admission = "wait";
      afterAdmit = () => {
        // put the room in the address bar (a typed code never was), so a reload joins it again like a link,
        // under the same name: the host re-seats a device's layers by name (aiRejoin). Without the key:
        // this tab has its pass now, and the address bar is no place for a secret
        if (!VQ.get("embed")) try { history.replaceState(history.state, "", roomLink()); } catch {}
        enterRoom();
        saveGuest();
        if (resume?.guest) {
          const what = died ? `This tab was reloaded ${died.ago} s ago while ${died.during.slice(0, 80)}${myMeta?.phone ? " (iOS reloads a page that uses too much memory, or one left in the background)" : ""}.` : "This tab was reloaded.";
          toast(`${what} Back in room ${formatCode(code)}.`);
          log("room", `${what} Rejoined room ${code} as ${myName}; the host puts this device back in its slot.`);
        }
      };
      // a host that never says hello at all: go in as before rather than wait forever (not with an
      // invite key or a pass: that host would have to prove it holds them)
      setTimeout(() => {
        const he = conns.get(conn.peer);
        if (admission === "wait" && peer === me && he && !he.hostHello && !he.jauth?.proved && !joinKey && !myPass) { releaseLink(he); guestIn(); }
      }, 15000);
    });
  }

  peer.on("connection", (conn) => {
    conn.on("open", () => {
      if (conn.label === "stripe") {   // extra association for the hidden-state wire, not a new peer
        // (the host: one from a device still in its lobby waits there, and is attached when it is let in)
        const L = isHost && lobbyConns.get(conn.peer);
        if (L) { L.stripes.push(conn); return; }
        const e = conns.get(conn.peer);
        if (e) acceptStripe(e, conn);
        else try { conn.close(); } catch {}
        return;
      }
      // the host: a new link waits at the gate (its hello decides) instead of joining the room at once;
      // a device: a link from another device of the room is held until it proves the room's mesh key
      if (isHost) gateConn(conn); else { wire(conn, undefined, undefined, false, { hold: meshKey ? "mesh" : null }); sendMesh(conn, "accept"); }
      // the host says it answers API clients (docs/protocol.md "API clients"); not part of myMeta,
      // which the roster shows everyone
      // api: 2 = it also answers v2 asks (tools, structured output); ctx: its context size now
      // gate: 1 = this host holds new devices until it lets them in (admit / lobby); ask: whether it asks
      // auth: this device proves its links, and (the host) proves the invite key and passes back
      // instead of reading them (room/chanauth.js)
      conn.send({ t: "hello", name: myName, meta: isHost ? { ...myMeta, api: 2, ctx: ctxMax() } : myMeta, v: PROTOCOL, auth: AUTH_V,
        ...(isHost ? { gate: 1, ask: gate?.ask ? 1 : 0 } : {}) });
    });
  });

  peer.on("error", (err) => {
    if ($("room-screen").style.display === "flex") {   // in the room already: not a join failure
      // losing the signaling server is shown by watchSignaling, which reconnects; the room keeps going
      if (FALLBACK_ERRORS.has(err.type)) { console.warn("signaling", err.type); return; }
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
    // no such room on this server: the host may have fallen back to a later one in the list
    if (!create && err.type === "peer-unavailable" && got.index + 1 < servers.length) {
      clearInterval(joinTimer);
      const next = servers[got.index + 1];
      log("room", `no room ${code} on ${signalServer.label}; looking on ${next.label}`);
      const p = peer; peer = null; try { p.destroy(); } catch {}
      start(false, null, got.index + 1);
      return;
    }
    clearInterval(joinTimer);
    joinFailed(FALLBACK_ERRORS.has(err.type) ? `Lost the signaling server (${signalServer.label}) while joining. Try again.` : peerErrorText(err.type));
  });
}

const peerErrorShown = { text: "", t: 0 };

// --- the relay this tab uses, and what the network allows (room/ice.js) ---
// the default relay's credentials expire: fetch new ones before they do, so links made later in a
// long session (a device joining, a chain relink, a stripe) still authenticate. Links already up
// keep their allocation.
// the site's relay (/api/turn), reused while less than half its lifetime has passed: start() runs
// again on a resume retry (every 3 s) or a fallback to the next signaling server
async function askRelay() {
  const relay = await fetchRelay(window.fetch?.bind(window), window.POOLED_TURN_ENDPOINT || undefined);
  if (relay) relayGot = { relay, at: performance.now() };
  return relay;
}
async function defaultRelay() {
  if (relayGot && performance.now() - relayGot.at < relayGot.relay.ttl * 500) return relayGot.relay;
  return askRelay();
}
function keepRelayFresh(cfg, ttl) {
  clearTimeout(relayTimer);
  relayTimer = setTimeout(async () => {
    const fresh = await askRelay();
    if (fresh) swapRelayServers(cfg, fresh.iceServers);
    keepRelayFresh(cfg, fresh ? fresh.ttl : ttl / 4);   // failed: try again sooner
  }, refreshInMs(ttl));
}
// why a join that found the room still failed, from what this tab knows about its network
function pathFailText(relay, from, udp) {
  const blocked = udp && !udp.udpOut;
  if (relay && from === "yours") return "Found the room, but could not connect, not even through the relay (TURN) server. Check its address and password under Network, or try another network.";
  if (relay) return "Found the room, but could not connect, not even through the relay (TURN) server. This network may block it too (some work networks only let web traffic out). Try another network, or set up your own relay under Network.";
  return blocked
    ? "Found the room, but this network blocks direct (UDP) connections and there is no relay (TURN) server to go around it. Add one under Network below, or try another network (a phone hotspot works)."
    : "Found the room, but these two devices can't reach each other (a strict firewall or mobile network on one side). A relay (TURN) server gets around that: add one under Network below. Or put both on the same Wi-Fi, or try another network.";
}
// a "Rooms at work" link after the join status text
function docsLink() {
  const a = document.createElement("a");
  a.href = WORK_DOCS; a.target = "_blank"; a.rel = "noopener";
  a.textContent = " Rooms at work";
  $("join-status").append(" See", a, ".");
}
// debug / tests: what this tab knows about its network (`pooledNet()` in the console)
window.pooledNet = () => ({ relay: relayOn, from: relayFrom, udp: udpSeen });
// once the probe answers: say which relay this tab has, and warn when the network looks closed
function networkNote(host) {
  if (relayOn) log("room", relayFrom === "default" ? "relay (TURN) ready: links go direct when they can, through the relay when a network blocks that" : "relay (TURN) set: links go direct when they can, through the relay when they can't");
  Promise.resolve(udpProbe).then((u) => {
    udpSeen = u;
    const adv = networkAdvice(u, relayOn);
    if (!adv) return;
    log("room", adv.text + (adv.level === "warn" ? ` (${WORK_DOCS})` : ""));
    if (adv.level === "warn" && host) toast(adv.text, { kind: "error" });
  }).catch(() => {});
}

// No signaling server answered: say what that means (the room can't be found or opened, a running
// room would be fine) and what to do, on the join screen.
function signalingDown(tried) {
  const names = tried.filter((x, i) => tried.indexOf(x) === i).join(", ");
  joinFailed(`Can’t reach the signaling server${tried.length > 1 ? "s" : ""} (${names}). Devices use ${tried.length > 1 ? "them" : "it"} only to find each other, and ${tried.length > 1 ? "they" : "it"} may be down or blocked on this network. Rooms already running are not affected. Try again in a minute, or `);
  const a = document.createElement("a");
  a.href = "https://github.com/Nehanth/pooled/blob/main/docs/self-host-signaling.md";
  a.target = "_blank"; a.rel = "noopener";
  a.textContent = "run your own signaling server";
  $("join-status").append(a, ".");
  $("join-status").classList.add("signal-down");
}

// In a room, the signaling server can drop (the cloud restarts, the network blips). The links already
// open are direct and keep working; only new devices can't find the room. Say so and reconnect with
// backoff (PeerJS keeps our id: reconnect() re-registers it) until the server is back.
function watchSignaling(p) {
  let tries = 0, timer = null;
  const again = () => {
    timer = null;
    if (p !== peer || p.destroyed || !p.disconnected) return;
    try { p.reconnect(); } catch {}
    timer = setTimeout(again, reconnectDelay(tries++));
  };
  p.on("disconnected", () => {
    if (p !== peer || p.destroyed) return;
    signalNote(true);
    if (!timer) timer = setTimeout(again, reconnectDelay(tries++));
  });
  p.on("open", () => {
    clearTimeout(timer); timer = null; tries = 0;
    if (p === peer && signalNote.on) { signalNote(false); toast("signaling is back: new devices can join again"); }
  });
}
function signalNote(on) {
  signalNote.on = on;
  const el = $("signal-note");
  if (!el) return;
  el.hidden = !on;
  if (on) el.textContent = `Lost the signaling server (${signalServer?.label || "PeerJS"}). The devices here keep working; new devices can’t join until it’s back. Reconnecting…`;
}

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
// Whether this device is kept awake while it holds layers (Wake Lock, or the silent video on older
// iOS). No header icon (the owner found it noise); the lend screen and the side hint still warn
// when the screen may sleep. window.pooledAwake() is for tests and debugging.
function holdsLayers() { return !!ai.engine && (ai.role === "host" || ai.role === "worker"); }
window.pooledAwake = () => ({ holds: holdsLayers(), awake: !!wakeLock || !!(awakeVideo && !awakeVideo.paused) });
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
    // why this device holds no layers (the host's deal: not needed, a pledge under one layer, or it
    // joined after the start), and what it holds of what it pledged
    out: ai.outWhy?.[myName] || null, phone: !!myMeta?.phone || isPhoneMeta(myMeta),
    held: ai.heldGB?.[myName] ?? null, pledge: myMeta?.webgpu ? pledgeGB(myMeta) : null,
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
// A guest tab that reloads (the user, or iOS after killing it for memory) walks back into its room under
// the same name, so the host puts it back in its slot (aiRejoin); the tab says what happened.
function saveGuest() {
  if (isHost || !roomCode) return;
  if (admission !== "in") return;   // only a device the host let in comes back by itself
  try { sessionStorage.setItem(GUEST_KEY, JSON.stringify({ code: roomCode, name: myName, gb: myMeta?.contribGB, pass: myPass || undefined, t: Date.now() })); } catch {}
}
setInterval(() => { if (peer && !isHost && roomCode && document.visibilityState === "visible") saveGuest(); }, 10000);
addEventListener("pagehide", saveGuest);
// Join only with a whole code: the greyed button and Enter in a short code do nothing but put the cursor back
// a whole code: six characters, or four (rooms opened before codes had six, and their links)
const codeOk = () => !!parseCode($("code-input").value);
$("join-btn").addEventListener("click", () => { if (!codeOk()) { $("code-input").focus(); return; } keepAwake(); start(false); });
$("code-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && codeOk()) start(false); });
// six boxes in two groups of three behind one input: the boxes show its characters, the box the next
// one goes in has the ring (the input's own text and caret are invisible)
const codeReady = () => {
  const v = $("code-input").value, box = $("code-input").parentElement, slots = box.querySelectorAll("i");
  slots.forEach((el, i) => { el.textContent = v[i] || ""; el.classList.toggle("on", i < v.length); el.classList.toggle("cur", i === Math.min(v.length, slots.length - 1)); });
  $("join-btn").classList.toggle("ready", codeOk());
  box.classList.toggle("full", v.length >= slots.length);
};
// letters and digits of the room alphabet only (no I, L, O, U, 0 or 1: no code has them); the sixth
// hands off to Join (on a phone that also closes the keyboard), so no box waits for a seventh
$("code-input").addEventListener("input", (e) => {
  const el = e.target, v = el.value.toUpperCase().replace(/[^A-HJKMNP-TV-Z2-9]/g, "").slice(0, 6);
  if (el.value !== v) el.value = v;
  codeReady();
  if (v.length === 6 && e.isTrusted && document.activeElement === el) { $("join-btn").focus(); el.parentElement.scrollIntoView({ block: "nearest" }); }
  else codeInView();
});
// a pasted invite link (or code with its dash): the code fills the boxes, and the link's key goes with the join
$("code-input").addEventListener("paste", (e) => {
  const text = (e.clipboardData?.getData("text") || "").trim();
  let code = "", key = "";
  if (/^https?:\/\//i.test(text)) {
    try { const u = new URL(text); code = codeFromLocation(u.pathname, u.search, u.hash); key = keyFromHash(u.hash); } catch {}
  } else code = parseCode(text);
  if (!code) return;
  e.preventDefault();
  $("code-input").value = code;
  if (key) { pastedLink.code = code; pastedLink.key = key; }
  $("code-input").dispatchEvent(new Event("input"));
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
  const path = (location.pathname.startsWith("/r/") ? "/room" : location.pathname) + "?" + q + (isHost ? keyFragment(gate?.key) : "");   // this computer's own devices: the key, no asking
  const box = document.createElement("div");
  box.className = "vdev";
  box.innerHTML = `<iframe title="virtual device ${virtualN}" src="${esc(path)}" allow="clipboard-write"></iframe><button type="button" title="close this virtual device">\u00d7</button>`;
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
  // a room on anything but the page's usual first server: the link names it, so joiners look there
  if (location.pathname === "/room" || location.pathname.startsWith("/r/")) {
    // (dev=0: a signal= link would otherwise open the page in dev mode, see p2p.html)
    const sig = signalServer && signalServer.spec !== SIGNAL_FIRST ? "?signal=" + encodeURIComponent(signalServer.spec) + (DEV ? "" : "&dev=0") : "";
    return `${location.origin}/r/${roomCode}${sig}${inviteKey()}`;
  }
  const q = shareQuery(location.search); q.set("code", roomCode); q.delete("ask");   // never a relay password in a link
  if (signalServer && (q.has("signal") || signalServer.spec !== SIGNAL_FIRST)) {
    if (!q.has("signal") && !DEV) q.set("dev", "0");
    q.set("signal", signalServer.spec);
  }
  return `${location.origin}${location.pathname}?${q}${inviteKey()}`;
}
// the host's links carry the invite key (a device that opens one is let in without asking); a guest's
// don't: it doesn't know the key, and the host is asked about whoever it invites
function inviteKey() { return isHost && gate ? keyFragment(gate.key) : ""; }
// a code for a screen reader, one character at a time ("4 T K, G 9 P")
function spokenCode(c) { return formatCode(c).split("-").map((g) => g.split("").join(" ")).join(", "); }
function copyRoomLink() {
  const url = roomLink();
  if (!navigator.clipboard) { toast("room code: " + formatCode(roomCode)); return; }
  navigator.clipboard.writeText(url).then(() => toast("join link copied")).catch(() => { navigator.clipboard.writeText(formatCode(roomCode)); toast("room code copied"); });
}
function openShare() {
  const url = roomLink();
  $("share-qr").innerHTML = qrSVG(url, { size: 220 });
  $("share-url").textContent = url;
  $("share-code").textContent = formatCode(roomCode);
  $("share-code").setAttribute("aria-label", `Room code ${spokenCode(roomCode)}`);
  // who needs the host's OK: a typed code does (when the host asks), the link doesn't
  const asks = isHost ? !!gate?.ask : hostAsks;
  $("share-or").textContent = asks ? (isHost ? "or type the room code (you let the device in)" : "or type the room code (the host lets the device in)") : "or type the room code";
  $("share-k").textContent = isHost || !asks ? "Open the link or scan the code on another device. It joins this room and adds its memory."
    : "Open the link or scan the code on another device. The host lets it in, then it adds its memory.";
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
$("share-native").addEventListener("click", () => navigator.share?.({ title: "Join my Pooled room", text: `Room ${formatCode(roomCode)}: add this device to the AI model we run together`, url: roomLink() }).catch(() => {}));
// the logo leads home. In a room it asks first: it sits in the thumb's corner on a phone, and
// leaving ends the room for everyone (the host) or takes this device's layers with it
const logoLink = document.querySelector(".logo a");
logoLink.addEventListener("click", (e) => {
  if (!document.body.classList.contains("in-room")) return;
  if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;   // a new tab or window keeps the room open: no need to ask
  e.preventDefault();
  $("leave-h").textContent = `Leave room ${formatCode(roomCode)}?`;
  $("leave-why").textContent = isHost
    ? (conns.size ? "The room ends for the other devices: this tab holds the conversation and the model's first and last layers." : "The room and its model close.")
    : `If this device holds some of the model's layers, the room has to re-deal them. You can join again with the code ${formatCode(roomCode)}.`;
  $("leave").hidden = false;
  $("leave-stay").focus({ preventScroll: true });
});
// (syncModal first: the header stays inert until it runs, and an inert logo would not take focus)
function closeLeave() { $("leave").hidden = true; syncModal(); focusBack(logoLink); }
$("leave-stay").addEventListener("click", closeLeave);
$("leave").addEventListener("click", (e) => { if (e.target === $("leave")) closeLeave(); });
$("leave-go").addEventListener("click", () => { location.href = logoLink.href; });
// waiting for the host to let this device in: give up, back to an empty join screen (the host's prompt goes)
$("jw-cancel").addEventListener("click", () => { try { broadcastAll({ t: "leaving" }); } catch {} location.href = location.pathname.startsWith("/r/") ? "/room" : location.pathname; });
$("room-over-new").addEventListener("click", () => { location.href = location.pathname.startsWith("/r/") ? "/room" : location.pathname.replace(/\?.*$/, ""); });
// A host that reloads its tab goes straight back into its room (no note on the join screen): only on a
// real reload of this tab, and only while the guests are still waiting for it (HOST_WAIT_MS).
const reloaded = (() => { try { return performance.getEntriesByType("navigation")[0]?.type === "reload"; } catch { return false; } })();
const backAsHost = reloaded ? savedHost() : null;
// a link with a room code fills it in and joins once the GPU probe is done
const linkCode = codeFromLocation(location.pathname, location.search, location.hash);
// and the invite key in its fragment (#k=, never sent to a server): the host lets this device in without asking
const linkKey = linkCode ? keyFromHash(location.hash) : "";
// an invite link pasted into the code boxes: its code fills them, its key goes with the join
const pastedLink = { code: "", key: "" };
// a virtual device (an iframe the host added, see addVirtual): its name, pledge and a compact page
const VQ = new URLSearchParams(location.search);
if (VQ.get("embed") === "1") document.documentElement.classList.add("embed");
if (VQ.get("vname")) $("name-input").value = VQ.get("vname").slice(0, 20);
else try { const n = sessionStorage.getItem(NAME_KEY); if (n) $("name-input").value = n; } catch {}   // this tab's name from before a reload
if (+VQ.get("vgb") > 0) { $("join-gb").value = +VQ.get("vgb"); joinGbChosen = true; }
let backAsGuest = null;
try { backAsGuest = reloaded && !VQ.get("embed") ? guestResume(JSON.parse(sessionStorage.getItem(GUEST_KEY) || "null"), { linkCode }) : null; } catch {}
if (backAsHost && Date.now() - backAsHost.t < 60000 && !(linkCode && linkCode !== backAsHost.code)) {
  metaPromise.then(() => { if (!peer) start(true, backAsHost); });
} else if (backAsGuest) {
  $("code-input").value = backAsGuest.code; codeReady();
  joinWait(true, `Joining room ${formatCode(backAsGuest.code)} again`);
  $("join-status").textContent = "This tab was reloaded: rejoining\u2026";
  metaPromise.then(() => { if (!peer) start(false, { ...backAsGuest, guest: true }); });
} else if (linkCode) {
  $("code-input").value = linkCode; codeReady();
  joinWait(true, `Joining room ${formatCode(linkCode)}`);
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
// ?rawcache=0: no copy of fetched weights in Cache Storage (a test machine short of disk)
async function getWeightCache() {
  if (weightCache !== null) return weightCache;
  if (new URLSearchParams(location.search).get("rawcache") === "0") return weightCache = false;
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
      // streamed: the parts go to the reader as they arrive, with flow control (peerGet), so a
      // phone never holds a whole 150 MB range in JS (#207). A failure after this point surfaces
      // in the reader, marked retryNet: the loaders then fetch the range from the network.
      const body = await peerGet(src, url, lo, hi);
      const resp = new Response(body, { status: 200, headers: { "content-type": "application/octet-stream", "x-swarm-len": String(hi - lo + 1) } });
      if (c && !myMeta?.phone) storeRange(c, key, resp.clone(), hi - lo + 1);
      return resp;
    } catch (err) {
      crumb(`peer weights from ${conns.get(src)?.name || src} failed (${err.message}); using the network`);
      // relayed: nothing else from that device either (one check, not one per range)
      if (err.relayed) { for (const [k, v] of ai.wsrc.map) if (v === src) ai.wsrc.map.delete(k); }
      else ai.wsrc.map.delete(lo + "-" + hi);
    }
  }
  ai.netBytes = (ai.netBytes || 0) + (hi - lo + 1);
  let r;
  try { r = await fetch(url, { headers: { Range: `bytes=${lo}-${hi}` } }); }
  catch { throw new FetchError(0, url); }   // offline, CORS or a blocked host: no status to go on
  if (r.status !== 206) throw new FetchError(r.status, url);
  // phones skip the store (no spare RAM for the copy); Cache API refuses 206s, so store as a plain 200
  if (c && !myMeta?.phone) storeRange(c, key, r.clone(), hi - lo + 1);
  return r;
}
// Store a range in the weight cache as it streams (no whole-range copy in JS). A body that errors
// makes put() fail and nothing is stored; one that ends short is stored but never trusted: every
// read checks the length (rangeFetch below keys on x-swarm-len, the loaders on the bytes they got).
function storeRange(c, key, r, len) {
  try {
    let got = 0;
    const counted = r.body.pipeThrough(new TransformStream({ transform(ch, ctl) { got += ch.byteLength; ctl.enqueue(ch); } }));
    c.put(key, new Response(counted, { status: 200, headers: { "content-type": "application/octet-stream", "x-swarm-len": String(len) } }))
      .then(() => { if (got !== len) return c.delete(key); ai.cachedBytes = (ai.cachedBytes || 0) + len; }, () => {});
  } catch {}
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
// A range from a device in the room, streamed with flow control: the requester says how much it
// lets the sender run ahead (win) and acks what its reader has taken; the sender waits for the acks.
// Before this, the whole range (up to 160 MB for a MoE expert tensor) was collected in one JS buffer
// on the phone, and a prefetched range kept a second one alive: that killed iPhone tabs (#207).
const W_PART = 64 * 1024;
const W_WIN = 8 * 2 ** 20;
const wGets = new Map();   // request id -> { ctl, got, len, acked, timer, first, fail, body }
let wSeq = 0;
// -> a ReadableStream of the range, once its first part is here (rejects on a miss or a silent source)
async function peerGet(src, url, lo, hi) {
  if (!(await ensureLink(src, 10000))) throw new Error("no link");
  // a relayed link costs the relay's owner per GB: weights come from the network instead (room/ice.js)
  if (!weightsOverLink(await pathOf(src), relayOn)) { const e = new Error("the link goes through the relay"); e.relayed = true; throw e; }
  const len = hi - lo + 1, id = `${peer.id}:${++wSeq}`;
  return new Promise((res, rej) => {
    const w = { got: 0, len, acked: 0, timer: null, ctl: null, first: { res, rej } };
    const fail = (err) => {
      clearTimeout(w.timer); wGets.delete(id);
      sendTo(src, { t: "ai-wack", id, cancel: 1 });
      err.retryNet = true;   // the loaders fetch the range from the network instead
      if (w.first) { w.first.rej(err); w.first = null; } else try { w.ctl.error(err); } catch {}
    };
    w.fail = fail;
    // Called by the stream after every read and every enqueue while there is room in the queue
    // (it returns at once, so it never blocks a later call). Acks what the reader took; with the
    // queue empty it acks everything, so the sender is never held back while the reader waits.
    // Silence counts only while the reader waits on an empty queue (a prefetched range may sit
    // unread for a while, and that is fine).
    const pull = () => {
      const queued = Math.max(0, W_WIN - w.ctl.desiredSize);
      const taken = w.got - queued;
      if (taken > w.acked && (taken - w.acked >= W_WIN / 4 || !queued)) { w.acked = taken; sendTo(src, { t: "ai-wack", id, got: taken }); }
      clearTimeout(w.timer);
      if (!queued && w.got < w.len) w.timer = setTimeout(() => fail(new Error("stalled")), 15000);
    };
    const body = new ReadableStream({
      start(ctl) { w.ctl = ctl; },
      pull,
      cancel() { clearTimeout(w.timer); wGets.delete(id); sendTo(src, { t: "ai-wack", id, cancel: 1 }); },
    }, new ByteLengthQueuingStrategy({ highWaterMark: W_WIN }));
    w.body = body;
    wGets.set(id, w);
    sendTo(src, { t: "ai-wget", id, url, lo, hi, win: W_WIN });
  });
}
function onWeightPart(d) {
  const w = wGets.get(d.id); if (!w) return;
  if (d.miss) { w.fail(new Error("not cached there")); return; }
  if (d.data) {
    const part = d.data instanceof Uint8Array ? d.data : new Uint8Array(d.data);
    if (d.off !== w.got || w.got + part.length > w.len) { w.fail(new Error(`part out of order at ${d.off}/${w.got}`)); return; }
    w.got += part.length;
    ai.peerBytes = (ai.peerBytes || 0) + part.length;
    clearTimeout(w.timer);
    w.ctl.enqueue(part);
    if (w.first) { w.first.res(w.body); w.first = null; }
  }
  if (d.done) {
    if (w.got !== w.len) { w.fail(new Error(`short: ${w.got}/${w.len}`)); return; }
    clearTimeout(w.timer); wGets.delete(d.id);
    if (w.first) { w.first.res(w.body); w.first = null; }
    w.ctl.close();
  }
}
// serve a cached range to a device in the room, 64 KB at a time, straight from the cache's body
// stream (not the whole range in memory), minding the channel's buffer and the requester's window
const wServes = new Map();   // request id -> { acked, cancel, wake }
function onWeightAck(d) {
  const s = wServes.get(d.id); if (!s) return;
  if (d.cancel) s.cancel = true; else s.acked = Math.max(s.acked, +d.got || 0);
  const wake = s.wake; s.wake = null; wake?.();
}
// a device asks for a range: serve it, or say "miss" (it then uses the network): with peer weights
// off, or over a relayed link, which the serving side refuses too (an older tab asks without checking)
async function answerWget(from, d) {
  if (!PEER_WEIGHTS || !weightsOverLink(await pathOf(from), relayOn)) { sendTo(from, { t: "ai-wpart", id: d.id, miss: 1 }); return; }
  return serveWeight(from, d);
}
async function serveWeight(from, d) {
  const e = conns.get(from); if (!e) return;
  const c = await getWeightCache();
  const hit = c && Number.isInteger(d.lo) && Number.isInteger(d.hi) ? await c.match(cacheKey(d.url, d.lo, d.hi)).catch(() => null) : null;
  const len = d.hi - d.lo + 1;
  if (!hit || hit.headers.get("x-swarm-len") !== String(len)) { sendTo(from, { t: "ai-wpart", id: d.id, miss: 1 }); return; }
  const win = +d.win > 0 ? +d.win : Infinity;   // (a requester from before flow control sends no window)
  const s = { acked: 0, cancel: false, wake: null };
  wServes.set(d.id, s);
  const reader = hit.body.getReader();
  try {
    let off = 0, pend = new Uint8Array(0);
    for (;;) {
      const { value, done } = await reader.read();
      let buf = done ? pend : pend.length ? concat(pend, value) : value;
      let o = 0;
      while (buf.length - o >= W_PART || (done && o < buf.length)) {
        while (off - s.acked >= win && !s.cancel && conns.has(from)) {
          const t0 = Date.now();
          await new Promise((r) => { s.wake = r; setTimeout(r, 1000); });
          if (Date.now() - t0 >= 1000 && off - s.acked >= win) s.idle = (s.idle || 0) + 1; else s.idle = 0;
          if (s.idle > 120) return;   // the requester stopped reading for 2 minutes: give up on it
        }
        if (s.cancel || !conns.has(from)) return;
        const n = Math.min(W_PART, buf.length - o);
        e.conn.send({ t: "ai-wpart", id: d.id, off, data: buf.subarray(o, o + n) });
        o += n; off += n;
        while (e.conn.dataChannel && e.conn.dataChannel.bufferedAmount > 4 * 2 ** 20) await new Promise((r) => setTimeout(r, 10));
      }
      pend = buf.subarray(o);
      if (done) break;
    }
    sendTo(from, { t: "ai-wpart", id: d.id, done: 1 });
    ai.servedBytes = (ai.servedBytes || 0) + off;
  } finally { wServes.delete(d.id); reader.cancel().catch(() => {}); }
}
const concat = (a, b) => { const m = new Uint8Array(a.length + b.length); m.set(a); m.set(b, a.length); return m; };

let ai = {
  dealer: null,          // the model host this device checked (room/authz.js): ai-load and FROM_HOST come only from it
  dealerFresh: false,    // that host was checked on a Start this device saw (the creator names only such a host in ai-modelhost)
  dealerActive: false,   // that host has shown it deals (ai-load, ai-layers, ai-hostprog, ai-ready-all): busy for other Starts
  startAt: null,         // when this device followed a Start (a second one meanwhile is refused); null once it fails
  startWait: null,       // this device pressed Start on another device's behalf: the timer that gives up
  visibility: "all",   // who sees the chat: all | host | asker (room/visibility.js)
  engine: null, tok: null, cfg: null, device: null,
  role: null,            // "host" | "worker" | "guest"
  gone: new Set(),       // the host: chain ids whose link closed, so a same-id rejoin is re-seated (aiRejoin)
  chain: [],             // host: worker peer ids in pipeline order
  next: null,            // worker: peer id to forward hidden to, or "host"
  readyPeers: new Set(),
  relinks: new Map(),    // host: device id -> deadline, while the device before it opens a fresh link to it (aiRejoin)
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
  recovering: null,      // host: a run is waiting for a dropped device to come back or a re-deal ({ kind, since })
  held: null,            // worker: { model, range, ctx } of the layers this engine holds (a device back from a lock keeps them)
};

const modelLoader = createModelLoader({
  state: ai, rangeFetch, getWeightCache, getMeta: () => myMeta,
  options: modelLoadOptions(location.search), getEnginePreset: () => roomQwen35Options(location.search),
  hooks: { onStatus: aiStatus, crumb, onModelLoaded: apiModelLoaded,
    onCacheHit(msg) { log("room", `${myName}: ${msg}`); },
    // this tab's disk (room/flashdisk.js tabDiskGate): what it offers and its user's answer; a worker asks when a deal
    // needs its disk and its user was not asked yet
    tabDisk: () => ({ tabDisk: myMeta.tabDisk || null, choice: tabDiskDeclined && tabDiskNow() !== "allow" ? "deny" : tabDiskNow() }),
    // (this device pressed Start, whose label said what it saves here: that press was the yes)
    askDisk: ({ model, bytes }) => (ai.pressedHere ? diskYesByStart(bytes) : askTabDisk({ model, bytes, host: ai.role === "host" })) },
});
const { fetchModelHeader, fetchGGUFHeader } = modelLoader;

function aiStatus(s) { $("ai-status").textContent = s; crumb(s); }
// breadcrumb: if iOS kills the tab, the reloaded page can say where it died
// loading: this tab was loading its layers (a kill then means the share was too big for it, #207)
var crumbLoading = false;   // var: crumb can run before the rest of the module is initialised
function crumb(s) { try { localStorage.setItem("pooled-crumb", JSON.stringify({ s, t: Date.now(), mem: performance.memory?.usedJSHeapSize, loading: crumbLoading || undefined })); } catch {} }
// (crumb is kept in localStorage for debugging, not shown on the join screen)
// The load card (p2p.html #load-card): from the Start press to the first question, one card that tells one story.
// It shows the moment Start is pressed (or a Start this device follows), says how long until chat as soon as that
// is worth saying (room/loadsteps.js etaTracker), turns into "ready" in the same place, and goes away when the first
// question is sent (aiOut) or on New chat. The card's top stays put (only its bottom grows) and the composer sits
// under it the whole time: a question typed while it loads is sent when the model is ready (lc.queued).
// One device alone: two rows at most, "Saving to this computer (once)" (the model and its word list as one figure,
// the one Start promised) and "Loading onto the graphics chip"; with one part only, no rows, its bytes under the bar.
// Several: a row per device, one bar and a short state each. The parts come from today's sequential onProgress
// (loadProg -> partsFromProg) or a concurrent loader's parts (onProgress info.parts, onParts: loadParts ->
// partsFromSnapshot): the same card. A start that fails stops in place (lcStop): what is saved stays saved, Try
// again goes on from there. A disk question during the load is asked in the card (askTabDisk), not over it.
const lcFresh = () => ({ t0: performance.now(), prog: progressTracker(), eta: etaTracker(), adapt: {}, phase: "pressed", parts: false,
  mine: null, left: null, roomBest: 0, why: "", host: null, redeal: false, paced: false, took: null, sentAt: 0,
  paintAt: 0, raf: 0, timer: 0, sig: "", struct: "", gb: {}, gbAt: 0, modelKey: null, expect: false, diskHint: 0, queued: false, asking: false,
  adding: [], guess: null, rateAt: {}, peerEta: {} });
let lc = lcFresh();
// the card is up and loading (not ready, not stopped): the composer queues instead of sending
const lcLoading = () => $("load-card").classList.contains("on") && (lc.phase === "pressed" || lc.phase === "run");
// the composer under the card: there from the Start press; while it loads, a question waits for ready (aiSubmit)
function composerLock(on) {
  const p = $("ai-prompt");
  p.disabled = false; $("ai-send").disabled = false;
  if (!on) lc.queued = false;
  if (on) p.placeholder = $("load-card").classList.contains("slim") ? "Type your next question" : "Type your first question"; else promptHint();
  sendLabel();
}
// modelKey: the model this card starts (a follower knows it from the Start message, not from its own picker)
function aiLoading(show, title, modelKey = null) {
  $("ai-loading").style.display = "none";   // (the old sidebar bar: one bar on screen, the card's)
  if (title) $("ldg-title").textContent = title;
  const card = $("load-card"), was = card.classList.contains("on");
  // a load under a conversation (a re-deal: a device added or gone) is a slim line over the chat, which stays; the
  // full card is for a first Start (and a start that stopped: lcStop)
  card.classList.toggle("slim", !!show && !!$("ai-output").querySelector(".m"));
  $("ai-panel").classList.toggle("loading", !!show);
  card.classList.toggle("on", !!show);
  $("ai-empty").style.display = show ? "none" : "";
  if (show) {
    // a fresh card for a new Start (or Try again); a load that starts under a card already up (the host's own after
    // its press, a re-deal mid-load) keeps its bar's best and its clock (lcNewLoad)
    // (and no other device's numbers from the last start: the room's bar starts from 0)
    if (!was || lc.phase === "stopped" || lc.phase === "ready") {
      cancelAnimationFrame(lc.raf); clearTimeout(lc.timer);
      const queued = lc.queued && lc.phase === "stopped";   // (Try again keeps a question that was waiting)
      lc = lcFresh(); lc.queued = queued; ai.partsW = {}; ai.leftBy = {}; ai.prog = {}; ai.askBy = {};
      lcAsk(null);
    }
    if (modelKey && MODELS[modelKey]) lc.modelKey = modelKey;
    if (!lc.modelKey) lc.modelKey = $("ai-model")?.value || ai.model || null;
    lc.expect = lcExpect();
    if (/re-deal/.test(title || "")) lc.redeal = true;
    card.classList.remove("stopped", "ready");
    $("ai-row").style.display = "flex";
    if (lc.phase === "pressed" || lc.phase === "run") composerLock(true);
    lcPaint(true);
  } else {
    const wasReady = lc.phase === "ready";
    lc.phase = "off"; cancelAnimationFrame(lc.raf); clearTimeout(lc.timer);
    card.classList.remove("stopped", "ready", "slim");
    lcAsk(null);
    composerLock(false);
    // (no live-looking message box under the picker after a Start that went nowhere: only a room that is up chats)
    $("ai-row").style.display = $("ai-panel").classList.contains("online") ? "flex" : "none";
    if (wasReady) emptyText("The model is ready. Ask anything.");
  }
  if (show) speed.since = 0;   // the next answers warm up (room/speedo.js warmupAnswers)
  if (show && /re-deal/.test(title || "")) { speed.redeal = true; speedEvent("Sharing out the model again"); }
}
// What the card will show, drawn grey from the press so it doesn't grow from nothing: this computer alone starting
// Qwen3.8 Flash for the first time (its disk may be used and the copy is not all there) has two rows
function lcExpect() {
  const k = lc.modelKey;
  if (!diskCopyOf(k) || lcDevices().length > 1) return false;
  if (tabDiskNow() !== "allow" && !/ saves /.test($("ai-start")?.textContent || "")) return false;
  const st = storedNow(k), need = lc.diskHint || diskCopyOf(k).diskBytes || 0;
  return !(st > 0 && need > 0 && st >= need * 0.99);
}
// a load starts on this device (aiLoadShardIn): its parts from scratch (its own bar too: a re-deal's new total is not
// compared with the old one); the room's bar keeps its best (lc.roomBest) and the clock runs from the press
function lcNewLoad() { delete lc.rateAt[myName]; lc.adapt = {}; lc.prog = progressTracker(); lc.eta = etaTracker(); lc.parts = false; lc.mine = null; lc.left = null; lc.host = null; lc.dealt = true; lc.gb = {}; lc.gbAt = 0; }
// every device has its layers and the room is up: the card says so where it stands, and the composer opens. A chat
// that already has turns (a re-deal mid-chat, a late joiner added) goes straight back to its conversation
function lcReady() {
  if (!$("load-card").classList.contains("on") || lc.phase === "ready") return;
  const queued = lc.queued;
  ai.pressedHere = false;
  if (ai.conv?.turns?.length) aiLoading(false);
  else {
    lc.phase = "ready"; lc.took = (performance.now() - lc.t0) / 1000;
    $("load-card").classList.add("ready");
    $("ai-row").style.display = "flex";
    composerLock(false);
    lcPaint(true);
  }
  // the question typed while it loaded: sent now (after the caller has put the room online)
  if (queued) setTimeout(() => { if ($("ai-panel").classList.contains("online") && $("ai-prompt").value.trim()) aiSubmit(); }, 0);
}
// the start stopped (failed): the card stays with what happened and what to do; the bar stays where it got to
function lcStop(why, plain = false) {
  const card = $("load-card");
  if (!card.classList.contains("on")) { card.classList.add("on"); $("ai-panel").classList.add("loading"); $("ai-empty").style.display = "none"; }
  const queued = lc.queued;
  lc.phase = "stopped"; lc.why = String(why || ""); lc.plain = !!plain; ai.pressedHere = false;
  card.classList.add("stopped"); card.classList.remove("ready", "slim");
  lcAsk(null);
  $("ai-row").style.display = "none";
  composerLock(false); lc.queued = queued;
  lcPaint(true);
}
$("lc-retry")?.addEventListener("click", () => { $("ai-start").disabled = false; $("ai-model").disabled = false; $("ai-start").click(); });
$("lc-pick")?.addEventListener("click", () => { lc.queued = false; aiLoading(false); updateCluster(); });
// The disk question inside the card (askTabDisk while the card is up): q { text, sub, resolve(yes) } or null to clear
function lcAsk(q) {
  const box = $("lc-ask"); if (!box) return;
  const was = lc.asking;
  lc.asking = !!q;
  if (!q) { box.hidden = true; $("lc-ask-yes").onclick = $("lc-ask-no").onclick = null; }
  else {
    $("lc-ask-q").textContent = q.text; $("lc-ask-h").textContent = q.sub;
    box.hidden = false;
    $("lc-ask-yes").onclick = () => q.resolve(true); $("lc-ask-no").onclick = () => q.resolve(false);
    $("lc-ask-yes").focus({ preventScroll: true });
  }
  // the room hears that this device waits on its user (its row: "asked to save its part")
  if (was !== lc.asking && ai.role === "worker" && ai.hostId) sendTo(ai.hostId, { t: "ai-progress", pct: Math.round(ai.myPct || 0), asking: lc.asking ? 1 : 0, ...(ai.partsW?.[myName] ? { parts: ai.partsW[myName], left: lc.left } : {}) });
  if (was !== lc.asking) lcPaint(true);
}

// this device's parts into the card: lc.mine (its bar, never backwards) and lc.left (its time left)
function lcPush(s) {
  lc.mine = lc.prog.push(s);
  lc.left = lc.eta.push(performance.now(), lc.mine.parts, s);
  rateSeen(myName, lc.mine.done);
  if (lc.phase === "pressed") lc.phase = "run";
  ai.partsW = ai.partsW || {}; ai.leftBy = ai.leftBy || {};
  ai.partsW[myName] = lcWire(); ai.leftBy[myName] = lc.left;
}
// what peers hear about this device's parts (ai-progress, ai-hostprog): packed parts and its bytes
const lcWire = () => (lc.mine ? { p: packParts(lc.mine.parts), db: [Math.round(lc.mine.done), Math.round(lc.mine.total)] } : null);
// How fast each device loaded, bytes a second, by name: measured over each load (10 s or more of it), kept for the
// session, so the offer to add a device that loaded here before says how long it takes (room/loadsteps.js redealGuess)
const loadRates = {};
function rateSeen(nm, done) {
  if (!nm || !(+done >= 0)) return;
  const now = performance.now(), r = lc.rateAt[nm];
  if (!r || done < r.d) { lc.rateAt[nm] = { t: now, d: +done }; return; }
  if (now - r.t >= 10000 && done > r.d) loadRates[nm] = (done - r.d) / ((now - r.t) / 1000);
}
// onParts (contract): a loader that runs the parts at the same time says where each one is (room/model-loader.js
// loadShard's onParts, or the same snapshot as onProgress's info.parts: loadProg); from then on the card uses those
function loadParts(snap) {
  if (!snap || !Array.isArray(snap.parts)) return;
  lc.parts = true;
  lcPush(partsFromSnapshot(snap));
  if (ai.role === "worker" && performance.now() - lc.sentAt > 250) { lc.sentAt = performance.now(); sendTo(ai.hostId, { t: "ai-progress", pct: Math.round(ai.myPct || 0), ...(ai.step?.[myName] ? { step: ai.step[myName], gpu: Math.round(ai.gpuPct || 0) } : {}), parts: ai.partsW[myName], left: lc.left }); }
  lcPaint();
}
// A peer's parts from a message: { p, db } checked (room/loadsteps.js unpackParts), or null
function partsWireIn(w) {
  if (!w || typeof w !== "object") return null;
  const parts = unpackParts(w.p);
  const db = Array.isArray(w.db) && w.db.length === 2 && w.db.every((x) => Number.isFinite(+x) && +x >= 0) ? [+w.db[0], +w.db[1]] : null;
  return parts ? { p: packParts(parts), db } : null;
}

// The devices on the card: the ones the deal gave layers (before the deal: every device here with a GPU), each
// with its parts, bytes, time left and percent. A peer on an older build has only its percent (old)
function lcDevices() {
  const by = ai.layersByName || {};
  let names = [myName, ...[...conns.values()].filter((c) => !c.meta?.api).map((c) => c.name)].filter(Boolean);
  if (lc.dealt && Object.keys(by).length) names = names.filter((nm) => by[nm]);
  else names = names.filter((nm) => nm === myName ? myMeta.webgpu !== false : [...conns.values()].some((c) => c.name === nm && c.meta?.webgpu));
  if (!names.length) names = [myName];
  return [...new Set(names)].sort(bySlot).map((nm) => {
    if (nm === myName) return { name: nm, you: true, parts: lc.mine?.parts || [], done: lc.mine?.done, total: lc.mine?.total, left: lc.left, pct: lc.mine?.pct ?? 0, asking: lc.asking };
    const w = ai.partsW?.[nm], parts = w ? unpackParts(w.p) : null, pct = (ai.prog || {})[nm] ?? 0;
    return { name: nm, parts: parts || [], done: w?.db?.[0], total: w?.db?.[1], left: ai.leftBy?.[nm] ?? null, pct, old: !parts && pct > 0, step: (ai.step || {})[nm] || null, asking: !!ai.askBy?.[nm] };
  });
}
const lcKey = () => (lc.phase === "ready" && ai.model) || lc.modelKey || $("ai-model")?.value || ai.model;
const lcModel = () => { const k = lcKey(); return MODELS[k] ? shortName(k) : "the model"; };
const lcFlash = () => lcKey() === "qwen3.8-flash";
// Paint the card: at most every 250 ms (a part's state or the phase changing paints at once), its GB figures
// rewritten at most once a second
function lcPaint(now = false) {
  if (!$("load-card").classList.contains("on")) return;
  const devs = lcDevices();
  const sig = lc.phase + "|" + lc.asking + "|" + devs.map((d) => d.name + ":" + (d.asking ? "?" : "") + (d.parts || []).map((p) => p.state).join(",")).join(";");
  if (sig !== lc.sig) now = true;
  if (!now) {
    if (lc.raf || lc.timer) return;
    const wait = Math.max(0, 250 - (performance.now() - lc.paintAt));
    const go = () => { lc.raf = requestAnimationFrame(() => { lc.raf = 0; lcPaintNow(); }); };
    if (wait > 0) lc.timer = setTimeout(() => { lc.timer = 0; go(); }, wait); else go();
    return;
  }
  cancelAnimationFrame(lc.raf); clearTimeout(lc.timer); lc.raf = lc.timer = 0;
  lcPaintNow(sig);
}
// a GB figure held back by the once-a-second rule is painted when its second is up (a load that then goes quiet
// doesn't leave an old figure under a newer bar)
function lcGbLater() {
  if (lc.gbTimer) return;
  lc.gbTimer = setTimeout(() => { lc.gbTimer = 0; lcPaint(true); }, Math.max(50, 1000 - (performance.now() - lc.gbAt)));
}
function lcPaintNow(sigIn) {
  const card = $("load-card"); if (!card.classList.contains("on")) return;
  const devs = lcDevices(), solo = devs.length <= 1, phase = lc.phase;
  lc.sig = sigIn || phase + "|" + lc.asking + "|" + devs.map((d) => d.name + ":" + (d.asking ? "?" : "") + (d.parts || []).map((p) => p.state).join(",")).join(";");
  lc.paintAt = performance.now();
  const gbNow = lc.paintAt - lc.gbAt > 1000; if (gbNow) lc.gbAt = lc.paintAt;
  const by = Object.fromEntries(devs.map((d) => [d.name, d]));
  const room = roomProgress(by, lc.roomBest);
  lc.roomBest = room.pct;
  const me = devs.find((d) => d.you), myParts = me?.parts || [];
  const single = !lcFlash() || (myParts.length === 1);
  // first start: this load (or another device's) saves to its disk; only on a tab that may use its disk
  const firsts = devs.map((d) => firstStart(d.parts || [])).filter((x) => x != null);
  const first = single ? null : firsts.some(Boolean) ? true : firsts.length && tabDiskNow() === "allow" ? false : null;
  const wordsFailed = myParts.some((p) => p.id === "words" && p.state === "fail");
  const model = lcModel();
  const ready = phase === "ready", stopped = phase === "stopped";
  const set = (sel, text) => { const el = card.querySelector(sel); if (el && el.textContent !== text) el.textContent = text; };
  set(".lc-eyebrow", eyebrow(phase, { model, first }));
  const slim = card.classList.contains("slim");
  // slim (a re-deal under the chat): one line, "Adding pc to the model · about 2 min left"; the guess from before it
  // started counts down until the devices' own estimates come in
  const guess = lc.guess == null ? null : Math.max(0, lc.guess - (performance.now() - lc.t0) / 1000);
  set(".lc-hl", slim ? slimLine(room, phase, { adding: lc.adding, guess }) : headline(room, phase, { model, why: lc.why, me: myName }));
  // the sentence: a stopped start says what happened and what to do
  let sent;
  if (stopped) {
    const who = stopWho(lc.why), d = by[who && by[who] ? who : myName];
    const sv = diskPart(d?.parts || [], d?.you ? lc.diskHint : 0);
    sent = stoppedText(lc.why, { me: myName, save: sv && sv.total > 0 ? sv : null, plain: lc.plain });
    card.querySelector(".lc-sent").title = lc.why;
  } else {
    sent = sentence(room, phase, devs, { host: lc.host, model, redeal: lc.redeal && !lc.mine, paced: lc.paced, single, first, asking: lc.asking,
      cached: lcFlash() && tabDiskNow() === "allow" ? null : false });
    card.querySelector(".lc-sent").removeAttribute("title");
  }
  set(".lc-sent", sent);
  // the room's bar
  const pct = ready ? 100 : room.pct;
  card.querySelector(".lc-bar .fill").style.width = pct + "%";
  set(".lc-bar .pct", ready ? "100%" : pct + "%");
  card.querySelector(".lc-bar").setAttribute("aria-valuenow", String(pct));
  // the rows: this computer alone, at most two (cardRows), grey placeholders from the press; several devices, one
  // row each with one bar. Ready: no rows (the footer says it in one line)
  let rows = [];
  if (!ready && solo) {
    const parts = !myParts.length && lc.expect ? [{ id: "save", done: 0, total: null, state: "wait" }, { id: "gpu", done: 0, total: null, state: "wait" }] : myParts;
    rows = cardRows(parts, { single, hint: lc.diskHint }).map((r) => {
      // GB text at most once a second (the bar under it moves every paint)
      if (r.cls === "run") { if (gbNow || !lc.gb[r.k]) lc.gb[r.k] = r.value; else if (r.value !== lc.gb[r.k]) { r.value = lc.gb[r.k]; lcGbLater(); } }
      return r;
    });
  } else if (!ready) {
    rows = devs.map((d) => {
      const dp = d.parts || [];
      const ended = !!dp.length && partsEnded(dp);
      let pctD = d.old || !dp.length ? Math.round(d.pct || 0) : Number.isFinite(d.done) && d.total > 0 ? wholePct(d.done, d.total) : 0;
      if (!ended) pctD = Math.min(99, pctD);
      const verb = stopped ? "stopped" : d.old ? rowVerb(d.step) : deviceVerb(dp, { roomReady: false, slowest: room.slowest, self: d.name, asking: d.asking, cached: lcFlash() ? null : false });
      return { k: d.name, you: !!d.you, verb, pct: ended ? 100 : pctD, label: `${ended ? 100 : pctD}%`, done: ended };
    });
  }
  // with no rows, this computer's bytes under the bar ("3.1 of 11.8 GB")
  const sub = card.querySelector(".lc-sub");
  if (sub) {
    sub.hidden = ready || stopped || rows.length > 0 || !solo;
    let t = sub.hidden ? "" : cardSub(myParts);
    if (t) { if (gbNow || !lc.gb.sub) lc.gb.sub = t; else if (t !== lc.gb.sub) { t = lc.gb.sub; lcGbLater(); } }
    if (sub.textContent !== t) sub.textContent = t;
  }
  const struct = (solo ? "s|" : "d|") + rows.map((r) => r.k).join(";");
  const box = $("lc-rows");
  if (struct !== lc.struct) {
    lc.struct = struct;
    box.innerHTML = solo
      ? rows.map((r) => `<div class="lc-part" data-k="${r.k}"><span class="l"></span><span class="v"></span><div class="bar"><i></i></div></div>`).join("")
      : rows.map((r) => `<div class="lc-dev${r.you ? " you" : ""}" data-k="${esc(r.k)}" style="--sw:${devColor(r.k)}"><div class="lc-row"><i class="sw"></i><div class="n"><span class="nm">${esc(r.k)}${r.you ? " <small>(you)</small>" : ""}</span><span class="lr"></span></div><div class="bar"><div class="fill"></div></div><div class="pct"></div></div></div>`).join("");
  }
  const els = box.children;
  rows.forEach((r, i) => {
    const el = els[i]; if (!el) return;
    if (solo) {
      el.className = `lc-part ${r.cls}`;
      el.querySelector(".l").textContent = r.label;
      el.querySelector(".v").textContent = r.value;
      el.querySelector(".bar i").style.width = r.pct + "%";
      return;
    }
    el.classList.toggle("done", r.done);
    el.querySelector(".lr").textContent = r.verb;
    el.querySelector(".fill").style.width = r.pct + "%";
    el.querySelector(".pct").textContent = r.label;
  });
  // the footer, or the note when part of the model is read online; the buttons when it stopped
  const note = card.querySelector(".lc-foot");
  if (wordsFailed && !ready && !stopped) {
    note.classList.add("box");
    if (!note.querySelector("b")) note.innerHTML = `<b>${esc(WORDS_NOTE[0])}</b>${esc(WORDS_NOTE[1])}`;
  } else {
    note.classList.remove("box");
    const t = footer(phase, { solo, first, took: lc.took, wordsFailed, saved: diskPart(myParts)?.done || 0 });
    if (note.textContent !== t || note.querySelector("b")) note.textContent = t;
  }
  $("lc-actions").hidden = !stopped;
}
// the hidden sidebar bar's numbers (the end-to-end tests read #ldg-sub as a trace); the card is lcPaint's
const fmtBytes = (b) => b >= 2 ** 30 ? (b / 2 ** 30).toFixed(1) + " GB" : (b > 0 ? Math.max(1, Math.round(b / 2 ** 20)) : 0) + " MB";
const loadPct = wholePct;
function aiProgress(done, total, note) {
  const pct = loadPct(done, total);
  $("ldg-fill").style.width = pct + "%";
  $("ldg-sub").textContent = `${(done / 2 ** 20).toFixed(0)} MB of ${(total / 2 ** 20).toFixed(0)} MB · ${pct}%` + (note ? " · " + note : "");
}
function emptyText(s) { $("ai-empty-t").textContent = s; }
// the transcript shows: the first question was sent (from any device), so a load card still saying "ready" goes
function aiOut() { if (lc.phase === "ready") aiLoading(false); const o = $("ai-output"); o.style.display = "block"; $("ai-empty").style.display = "none"; return o; }

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
  // one steady line (no verbs rotating every two seconds); the device count only when there is more than one. The first
  // question since the model went ready can take a while (the model warms up on it): after 4 s with no word, it says so
  // once, in plain words, here in the bubble (no panel, no number)
  m.querySelector(".bubble").append(working({ steady: true, lead: n > 1 ? `Reading your message on ${n} devices` : "Reading your message",
    then: speed.firstWait ? { after: 4000, text: "The first answer takes a little longer" } : null }));
  m.classList.add("live");
  m.pieces = [];
  // the first answer of a model that reads from disk is slow to start: after 10 s of nothing, say that it is expected
  const k = ai.model || $("ai-model").value;
  if (!ai.answered && diskCopyOf(k)) setTimeout(() => {
    const v = m.querySelector(".wk-v");
    if (v && m.isConnected && !m.pieces.length && m.classList.contains("live"))
      v.textContent = n > 1 ? "Thinking\u2026 the first answer takes a little while." : "Thinking\u2026 on one computer the first answer can take a minute.";
  }, 10000);
  o.appendChild(m); scrollChat();
  botEl = m;
  speedStart();
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
  if (!note && !ai.answered) { ai.answered = true; promptHint(); }
  // a finished answer in a background tab: say so in the tab title until the tab is looked at
  if (!note && document.hidden) { document.title = "\u2713 answer ready \u00b7 pooled"; }
  // under the answer: a copy icon (answers only, not notes), then the numbers
  const acts = document.createElement("div");
  acts.className = "m-acts";
  if (!note && botEl.pieces.length) acts.innerHTML = `<button type="button" class="copy-ans icon-act" aria-label="Copy the answer" title="Copy">${COPY_SVG}</button>`;
  // the numbers ("24 tok · 18.3 tok/s · 2 devices") stay in the tooltip; the line says only what a person would want
  // to know: that older messages were forgotten, the room carried on after a device left, or the answer failed
  if (stats) { const s = document.createElement("div"); s.className = "stats"; s.textContent = statsNote(stats); s.title = stats; acts.appendChild(s); }
  if (acts.childElementCount) botEl.appendChild(acts);
  if (!note && botEl.dataset.mid && readAloud && botEl.pieces.length) speak(botEl.pieces.map((p) => p.t).join(""));
  speedDone(stats, botEl.dataset.mid ?? null);
  botEl = null;
}
// a device's share of the model in words, for its card: "half", "a third", "about 40%"
function shareWords(f) {
  const near = (x) => Math.abs(f - x) < 0.03;
  return near(0.5) ? "half" : near(1 / 3) ? "a third" : near(2 / 3) ? "two thirds" : near(0.25) ? "a quarter" : near(0.75) ? "three quarters" : `about ${Math.max(1, Math.round(f * 100))}%`;
}
// the model's size, short: "125B model" (room/speedo.js paramsOf; what B means is in the tooltip, paramsText); ""
// without a size
function sizeText(p) {
  const m = /^(\d+(?:\.\d+)?)([BMT])$/.exec(p?.total || "");
  return m ? `${m[1]}${m[2]} model` : "";
}
// What the line under an answer shows of its stats (engine/generate.js; room/speedo.js parseStats reads the same
// string, so it is not changed): nothing for an ordinary answer, a plain note when something happened
function statsNote(stats) {
  const s = String(stats || "");
  if (/^failed:/.test(s)) return s;
  const notes = [];
  if (/forgotten to fit/.test(s)) notes.push("Forgot the oldest messages to make room");
  if (/carried on after/.test(s)) notes.push("Kept going after a device left");
  if (/via API/.test(s)) notes.push("Answered for a connected app, not part of this chat's memory");
  return notes.join(" · ");
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
    code: formatCode(roomCode), nodes, tps, acc: lastMap?.st?.acc, lap: lastMap?.st?.lap, date: new Date().toISOString().slice(0, 10) });
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
  const lines = [`# Pooled room ${formatCode(roomCode || "")}`, "", `_${new Date().toISOString().slice(0, 16).replace("T", " ")} · ${MODELS[ai.model || $("ai-model").value]?.label || ""}_`, ""];
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
  // under a load card that is still loading: Send keeps the question for when the model is ready (aiSubmit)
  if (lcLoading()) {
    if (!typed) lc.queued = false;
    b.classList.remove("stop"); b.textContent = lc.queued ? "Sends when ready" : "Send"; b.disabled = lc.queued;
    return;
  }
  if (b.textContent === "Sends when ready") b.disabled = false;
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
  el.innerHTML = `<i style="--p:${pct}" aria-hidden="true"></i>${pct}% of chat memory used`;
  el.title = `${used.toLocaleString("en-US")} of ${max.toLocaleString("en-US")} tokens`;
  el.classList.toggle("warn", used > max * 0.8);
}

// Main and draft loads share the room's pacing and cancellation policy.
// Only the GPU step is paced (room/loadsteps.js): a device still saving to its disk is not waited on, and a device's
// GPU percent (ai.gpuProg) is compared, not its whole load's, or the one holding the n-gram table (28.8 GB more to
// save) would hold the others back. A peer on an older build sends no step: its one percent is its GPU percent.
function slowestLoadProgress() {
  const now = Date.now();
  let m = Infinity;
  for (const [nm, all] of Object.entries(ai.prog || {})) {
    const step = (ai.step || {})[nm], pct = (ai.gpuProg || {})[nm] ?? all;
    if (nm === myName || pct >= 100 || (step && step !== "gpu")) continue;
    if (now - ((ai.progAt || {})[nm] || 0) > 30000) continue;     // silent for 30 s: don't wait on it
    m = Math.min(m, pct);
  }
  return m;
}
async function aiLoadPace() {
  if (ai.startFailed) throw new Error(ai.startFailed);   // the start was stopped (aiStartStopped): no point loading the rest
  while (ai.gpuPct < 100 && ai.gpuPct > slowestLoadProgress() + 4) {
    aiStatus("Waiting a moment so every device finishes together");
    if (!lc.paced) { lc.paced = true; lcPaint(true); }
    await new Promise((r) => setTimeout(r, 250));
  }
  if (lc.paced) { lc.paced = false; lcPaint(true); }
}

// what a person sees when this computer's GPU fails the quick check before a load (the self-test, the kernel tests): what
// it means for them, in plain words; the test's own detail (err.detail) goes to the console and the room's log
const GPU_CHECK_FAILED = "This computer's graphics chip didn't pass a quick check, so it can't hold part of the model.";
// the same, as the host reads it about another device (its log and toast: "pc couldn't load its part of the model (…)"), with the detail
const gpuCheckFailedFor = (err) => `its graphics chip didn't pass a quick check${err?.detail ? ` (${err.detail})` : ""}`;
// the breadcrumbs written while this runs say "loading": if iOS kills the tab now, the host learns
// from the hello after the reload that this share was too big for it (aiLoadDeath)
async function aiLoadShard(...args) {
  crumbLoading = true;
  try { return await aiLoadShardIn(...args); }
  finally {
    crumbLoading = false;
    try { const c = JSON.parse(localStorage.getItem("pooled-crumb") || "null"); if (c?.loading) { delete c.loading; localStorage.setItem("pooled-crumb", JSON.stringify(c)); } } catch {}
  }
}
// offload: this device's part of the deal's expert offload (offMsg), Qwen3.8-Flash-Next on a tab: room/model-loader.js
// flash: Qwen3.8-Flash-Next's format from the host (ai-load.flash: QSA), or null for the dense engine
async function aiLoadShardIn(modelKey, range, hasEmbed, hasHead, ctx = maxSeqFor(modelKey), kv = kvModeFor(modelKey, KV_ASK), offload = null, flash = null) {
  const M = MODELS[modelKey];
  ai.shardBytes = 0;   // until this load's first progress says how big the new range is
  aiLoading(true, `loading layers ${range[0]}\u2013${range[1] - 1} of ${M.label.split("\u00b7")[0].trim()}`);
  lcNewLoad();   // the card: this device's parts from scratch (its bar keeps its best, its clock runs from the press)
  aiStatus("Getting this computer ready\u2026");
  mascot("Grabbing my slice of the model… hang tight.");
  // a previous attempt in this tab still owns its weights: release them first, or the
  // second load doubles GPU memory and every buffer after the limit comes back invalid
  // (and a Qwen3.8-Flash-Next engine's n-gram workers, whose OPFS locks the new load's table needs: room/flash.js close)
  closeEngine();
  if (ai.device) { try { ai.waker?.destroy(); ai.device.destroy(); } catch {} ai.device = null; ai.engine = null; }
  ai.held = null; ai.waker = null; ai.draft = null;
  ai.firstGpuError = null;
  ai.peerBytes = 0; ai.netBytes = 0; cacheHits = 0;   // per load: a count left from an earlier load in this tab mislabels the status
  if (!Qwen35Engine) { aiStatus("Getting this computer ready\u2026"); await loadEngine(); }
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
  aiStatus("Checking the graphics chip\u2026");
  const tAdapter = await navigator.gpu.requestAdapter();   // an adapter gives out one device only
  const tdev = await tAdapter.requestDevice();               // throwaway: its test buffers die with it
  const st = await gpuSelfTest(tdev);
  if (!st.ok) log("room", `${myName} GPU self-test: ${st.detail}`);
  // (the detail goes to the console and the room's log; the person sees what it means for this computer)
  if (!st.ok) { console.error(`GPU self-test failed on this device: ${st.detail}`); throw Object.assign(new Error(GPU_CHECK_FAILED), { detail: `GPU self-test: ${st.detail}` }); }
  const mt = await kernelMicroTests(tdev);
  if (!mt.ok) log("room", `${myName} kernels: ${mt.detail}`);
  if (!mt.ok) { console.error(`GPU kernel check failed on this device: ${mt.firstFail} (${mt.detail})`); throw Object.assign(new Error(GPU_CHECK_FAILED), { detail: `GPU kernel: ${mt.firstFail}` }); }
  try { tdev.destroy(); } catch {}
  ai.device.lost.then((l) => crumb("GPU device lost: " + l.reason + " " + l.message));
  aiStatus("Checking the graphics chip\u2026");
  ai.tune = await autotuneCoop(ai.device).catch(() => ({ wg: 256, rows: 4 }));
  crumb(`autotune: WG=${ai.tune.wg} ROWS=${ai.tune.rows}`);
  const isPhone = myMeta?.phone;
  ai.myPct = 0; ai.gpuPct = 0;
  ai.prog = { [myName]: 0 }; ai.progAt = { [myName]: Date.now() }; ai.step = {}; ai.gpuProg = {}; ai.partsW = {}; ai.leftBy = {};
  const streamOpts = { pace: isPhone ? 300 : 0, staging: isPhone ? 2 * 2 ** 20 : 8 * 2 ** 20 };
  if (M.cfg) {
    ai.cfg = await (await fetch(M.cfg)).json();
    if (hasEmbed || hasHead) { ai.tok = makeTokenizer(await (await fetch(M.tok)).json()); apiModelLoaded(); }
  }

  if (ai.role === "host") {
    clearInterval(ai.progTimer);
    ai.progTimer = setInterval(() => { if (ai.role === "host") broadcastAll({ t: "ai-hostprog", all: ai.prog || {}, steps: ai.step || {}, gpu: ai.gpuProg || {}, parts: ai.partsW || {}, left: ai.leftBy || {}, ask: ai.askBy || {}, at: Date.now() }); }, 600);
  }
  const onProg = loadProg;
  // every device (host included, even when its weights come from cache) keeps within a few
  // percent of the slowest device, so the bars climb together and the room finishes as one
  ai.loadParts = null;
  await modelLoader.loadShard({ modelKey, range, hasEmbed, hasHead, ctx, kv, streamOpts, onProgress: onProg, pace: aiLoadPace, offload, flash,
    onParts: (s) => { ai.loadParts = s; } },   // onParts (contract): the load's parts for the load card (loadProg renders it)
    { makeTokenizer, DenseEngine, Qwen35Engine, fetchModelShard, shardTensorNames });
  if (ai.peerBytes) log("room", `${myName}: ${(ai.peerBytes / 2 ** 20).toFixed(1)} MB of weights came from devices in the room, ${((ai.netBytes || 0) / 2 ** 20).toFixed(1)} MB from the network`);
  // batched draft-cache fill and refill, next step's first draft in the verify's pass (engine/preset.js)
  applyRoomFlags(ai.engine, location.search);
  ai.range = range;
  ai.model = modelKey;
  // the load card stays up (its "Starting" mark once every device has its layers) until the room
  // is online: the host takes it down in aiMaybeReady, a worker on ai-ready-all. Taking it down here
  // showed the model picker for the seconds between this device's layers and the room's.
}

// A load's progress (room/model-loader.js loadShard's onProgress). info: the step it is on (Qwen3.8 Flash: room/loadsteps.js,
// done / total then the whole load's; { step, stepNo, steps, stepDone, stepTotal, line }); without it, the one step
// of every other model, as before. The step line replaces the status while a step runs; the technical line (what is
// copied where) goes to the breadcrumbs and the card's tooltip. Peers hear the whole load's percent, the step, and the
// GPU step's own percent (the pacing compares that: slowestLoadProgress).
function loadProg(done, total, info = null) {
  info = info && typeof info === "object" ? info : null;
  aiProgress(done, total);
  const step = info?.step || "gpu";
  if (info?.line) crumb(info.line);
  if (info?.step) $("ai-status").textContent = stepLine(step, info);
  else {
    // name where the bytes are coming from right now: the network, devices in the room, or this device's cache
    const gb = (b) => (b / 2 ** 30).toFixed(1) + " GB";
    aiStatus(ai.netBytes ? `downloading the model\u2026${cacheHits ? ` (${gb(cacheHits)} was already on this computer)` : ""}`
      : ai.peerBytes ? `copying the model from devices in the room\u2026`
      : cacheHits ? `loading the model from this computer's cache\u2026` : `downloading the model\u2026`);
  }
  // the card's parts (room/loadsteps.js partsFromProg), unless a loader that says them itself (onParts) took over
  // (a concurrent loader sends its parts with each onProgress too, info.parts: those, whatever order its steps run in)
  if (Array.isArray(info?.parts)) { lc.parts = true; lcPush(partsFromSnapshot({ parts: info.parts })); }
  else if (!lc.parts) lcPush(partsFromProg(lc.adapt, done, total, info));
  ai.myPct = loadPct(done, total);
  // the GPU step's own percent, for the pacing (aiLoadPace): the loader's gpuPct (100 once the GPU step is done and a copy
  // goes on, room/flashload.js); without it, 0 while saving to the disk, which is never paced
  ai.gpuPct = info?.gpuPct ?? (step !== "gpu" ? 0 : info?.stepTotal ? info.stepDone / info.stepTotal * 100 : ai.myPct);
  if (!info || step === "gpu") ai.shardBytes = info?.stepTotal || total;   // the lending screen shows how much this device holds
  ai.prog = ai.prog || {}; ai.progAt = ai.progAt || {}; ai.step = ai.step || {}; ai.gpuProg = ai.gpuProg || {};
  ai.prog[myName] = Math.round(ai.myPct); ai.progAt[myName] = Date.now();
  ai.step[myName] = info?.step || null; ai.gpuProg[myName] = Math.round(ai.gpuPct);
  // parts / left: the card's view of this device (an older host ignores them); pct, step, gpu as before (the pacer)
  if (ai.role === "worker") {
    lc.sentAt = performance.now();
    sendTo(ai.hostId, { t: "ai-progress", pct: Math.round(ai.myPct), ...(info?.step ? { step: info.step, gpu: Math.round(ai.gpuPct) } : {}), ...(ai.partsW?.[myName] ? { parts: ai.partsW[myName], left: lc.left } : {}) });
  }
  lcPaint();
}

// ---- host ----
// Drop this tab's engine. What dropping one does not free with its GPU device: a Qwen3.8-Flash-Next engine's n-gram
// table reader (OPFS workers holding their files' locks, room/flash.js close), so every path that lets go of
// ai.engine comes through here. Every other engine has no close
// what an ai-load says about a device's expert offload (room/plan.js dealRoom offload; the room node's offloadMsg):
// the layers, the GPU cache, the RAM it parks, and with the disk tier [diskLo, hi) and the RAM cache. diskOnly: a tab
// (it reads every offloaded layer's experts from OPFS)
const offMsg = (o) => (o ? { lo: o.lo, hi: o.hi, vramBytes: o.vramBytes, ramBytes: o.ramBytes, ...(o.ramSpare > 0 ? { ramSpare: o.ramSpare } : {}),
  ...(Number.isInteger(o.diskLo) ? { diskLo: o.diskLo, diskRamBytes: o.diskRamBytes } : {}), ...(o.diskOnly ? { diskOnly: true } : {}) } : null);
function closeEngine() { const e = ai.engine; ai.engine = null; try { e?.close?.()?.catch?.(() => {}); } catch {} ai.answered = false; promptHint(); }
// The prompt's placeholder: a model read from disk (room/diskpick.js) is slow to its first answer, so until one has come
// (since the model came online) the box says so; then the plain "Message the room". Only the placeholder: the status
// lines and the speed numbers are the loading screen's.
const PROMPT_PH = "Message the room";
function promptHint() {
  if (lcLoading()) return;   // (under the load card: composerLock's "Type your first question")
  const k = ai.model || $("ai-model").value;
  $("ai-prompt").placeholder = $("ai-panel").classList.contains("online") && !ai.answered && diskCopyOf(k)
    ? `Ask ${shortName(k)}. The first answer takes a little while.` : PROMPT_PH;
}

// the model host (room/plan.js pickModelHost): the strongest device, a computer before a phone,
// whoever pressed Start. The room's creator stays the PeerJS hub either way.
function biggestPeerId() {
  if (roomView().get(PREFIX + roomCode)?.fixedHost) return PREFIX + roomCode;   // a pooled host runs its own model
  return pickModelHost([...roomView()].map(([id, meta]) => ({ id, meta: saneMeta(meta) }))) || peer.id;
}
// The facts the model host is checked with (room/authz.js): the devices in the creator's roster with the metas
// it lists (not API clients). The creator: its own roster. A guest: the roster the creator sent, never the
// hellos and pledges on its own links
function roomView() {
  if (isHost) {
    const v = new Map([[peer.id, myMeta]]);
    for (const [id, m] of roster) if (!m.meta?.api) v.set(id, m.meta || {});
    return v;
  }
  const v = new Map(authzView);
  if (!v.has(peer.id)) v.set(peer.id, myMeta);
  return v;
}
const joinSeq = () => (isHost ? [peer.id, ...joinOrder([...roster].map(([id, m]) => ({ id, seq: m.seq })))] : rosterOrder);
// does `boss` run the model, by this device's own pick over the devices `ids` names (room/authz.js)? On a Start
// this device is in the room for, devices it has seen for a while can't be left out
function checkHost(boss, ids, from) {
  const now = performance.now();
  const r = verifyHost({ boss, ids: Array.isArray(ids) ? ids : null, view: roomView(), creator: PREFIX + roomCode, from, order: joinSeq(),
    seenFor: (id) => now - (id === peer.id ? joinedAt : seenAt.get(id) ?? now) });
  if (!r.ok) log("room", `not following ${members.get(boss)?.name || boss} as the model host: ${r.why}`);
  return r;
}
const dealerHere = () => !!ai.dealer && roomView().has(ai.dealer);
const startRunning = () => ai.busy || !!ai.engine || !!ai.loadingShard || ai.role === "worker" || $("ai-panel").classList.contains("online");
function aiStartAnywhere() {
  const model = $("ai-model").value;
  const boss = biggestPeerId();
  const ids = [...roomView().keys()];
  const busy = startBusy({ boss, self: peer.id, dealer: ai.dealer, dealerHere: dealerHere(), dealing: ai.dealerActive, running: startRunning(), pendingSince: ai.startAt, now: performance.now() });
  if (busy === "replay") return;
  if (busy) { aiStatus(`can't start: ${busy}`); return; }
  ai.dealer = boss; ai.dealerFresh = true; ai.dealerActive = false; ai.startAt = performance.now();
  // this device is the pick: it starts, and tells the room so every device checks it and follows (the creator
  // relays it to the devices this one has no link to)
  // (the card shows now, not after the model's header is fetched: no dead click; the press is this device's disk consent)
  if (boss === peer.id) { broadcastAll({ t: "ai-start-req", model, boss, by: myName, ids }); ai.pressedHere = startSaysSave(); aiLoading(true, `starting ${MODELS[model].label.split("·")[0].trim()}`, model); aiStart(model); return; }
  ai.hostId = boss;   // its ai-layers and ai-ready-all are the ones this device listens to (FROM_HOST)
  // nobody follows it (their view differs, or another Start is in progress): give the button back. Any sign of
  // the host dealing (ai-layers, ai-hostprog, ai-ready-all) stops this
  followWait(START_PENDING_MS);
  ai.pressedHere = startSaysSave();   // (its disk's yes, if the host deals it a part that saves: diskYesByStart)
  $("ai-start").disabled = true; $("ai-model").disabled = true;
  aiLoading(true, `starting ${MODELS[model].label.split("·")[0].trim()}`, model);
  lc.host = conns.get(boss)?.name || members.get(boss)?.name || "The biggest device"; lcPaint(true);
  aiStatus(`asked ${conns.get(boss)?.name || "the biggest device"} to start ${MODELS[model].label.split("·")[0].trim()}…`);
  broadcastAll({ t: "ai-start-req", model, boss, by: myName, ids });
}
// a Start this device follows (its own, or one the creator relayed): give up when its host shows no sign of
// dealing in time, so a host that never starts doesn't hold the room
function followWait(ms) {
  clearTimeout(ai.startWait);
  ai.startWait = setTimeout(() => startRefused(`${members.get(ai.dealer)?.name || "the host"} didn't start within ${Math.round(ms / 1000)} s`), ms);
}
// this device's Start was refused (ai-start-refused), or a Start it follows went nowhere: the picker comes back
function startRefused(why) {
  clearTimeout(ai.startWait); ai.startWait = null; ai.pressedHere = false;
  if (ai.loadingShard || ai.engine || ai.role === "worker" || ai.dealerActive || $("ai-panel").classList.contains("online")) return;
  // (the host it asked for stays its named host: if that one deals after all, this device still takes part)
  ai.dealer = null; ai.dealerFresh = false; ai.startAt = null;
  aiLoading(false);
  $("ai-start").disabled = false; $("ai-model").disabled = false; updateCluster();
  aiStatus("the model didn't start: " + why);
}
// why a start stops when the room's pledges cannot hold the model (room/plan.js shortNote)
// What a deal stores on the host tab's disk (its offloaded layers' experts in OPFS, the n-gram table when it holds that
// layer), bytes: the number its question shows
function hostDiskBytes(deal, modelKey) {
  if (!TAB_DISK_MODELS.includes(modelKey) || !ai.G) return 0;
  const o = deal.offload?.[0], layers = o && Number.isInteger(o.diskLo) ? Array.from({ length: o.hi - o.diskLo }, (_, i) => o.diskLo + i) : [];
  return (layers.length ? tabExpertBytes(ai.G, layers) : 0) + (deal.ngram?.at === 0 ? deal.ngram.bytes : 0);
}
function shortWhy(M, fit, names, metas) {
  return shortNote(M.label.split("\u00b7")[0].trim(), fit, names, metas.map(spareGBOf), PAGE_SHORT_FIX);
}
async function aiStart(modelArg) {
  if (ai.engine || ai.busy) return;
  ai.busy = true;
  ai.dealing = true;   // until this function ends: a re-deal asked meanwhile waits for it (aiAutoRedeal)
  if (typeof modelArg === "string") setModelValue(modelArg);
  $("ai-start").disabled = true;
  $("ai-model").disabled = true;
  try {
    ai.role = "host";
    ai.dealer = peer.id; ai.dealerFresh = true; ai.dealerActive = true;
    ai.degraded = false;
    ai.startFailed = null;
    ai.readyPeers = new Set(); ai.batchOf = new Map();
    ai.ckptHeld = new Map();
    ai.relinks = new Map();
    ai.teleBy = new Map();
    ai.lapStat = null;                        // a new chain: lap timeouts start from the fixed fallbacks again
    const modelKey = $("ai-model").value;
    const M = MODELS[modelKey];
    // context for this room: the model's default, or ?ctx=N up to its cap (room/models.js CTX); every
    // device builds its engine with it. Without ?ctx= a room whose pledges hold the model only at its
    // fallback (the 1.7B: 8K for 16K) opens it there (pickCtx, below)
    const CTX_ASK = CTX_URL_ASK;
    let ROOM_CTX = maxSeqFor(modelKey, CTX_ASK);
    const ROOM_KV = kvModeFor(modelKey, KV_ASK);   // KV cache format for every device: f16, or int8 with ?kv=q8
    // a model host that is not the room's creator (biggestPeerId) has links only to the creator and to
    // devices it dialed: link every device first, so the deal, ai-layers and ai-ready-all reach them all
    // (a device the deal leaves out would otherwise sit on its Loading card, and could not ask)
    if (!isHost) { aiStatus("linking the devices in the room…"); await linkMembers(); }
    // devices without WebGPU join as ask-only guests: they get the chat, not layers
    // (a device whose tab was killed twice while loading its layers stays a guest: aiLoadDeath)
    ai.chain = [...conns.keys()].filter((id) => conns.get(id)?.meta?.webgpu && conns.get(id)?.conn?.open !== false && !ai.dropped?.has(conns.get(id)?.name)).sort();
    ai.leftOut = new Set();
    ai.plan = new Map();                      // name -> load message, so a reloaded device can be re-seated
    ai.gone = new Set();                      // chain ids whose link closed (aiPeerLeft), for a same-id rejoin
    ai.chainNames = ai.chain.map((id) => conns.get(id)?.name || id);
    const n = ai.chain.length + 1;
    // layerAt(ctx): one layer's bytes with its KV cache at that context, or (Qwen3.8-Flash-Next) each layer's own bytes
    // at that context (room/models.js flashLayerCost: room/plan.js takes either)
    let L, layerBytes, layerAt, embedBytes, cfg = null, experts = null, hostMargin = null;
    if (M.kind === "qwen35") {
      aiStatus("Looking up the model…");
      ai.G = await fetchModelHeader(M);
      ai.GModel = modelKey;
      // one attention layer's K (or V) cache is a single GPU buffer: hold the context to what the
      // smallest binding limit in the room fits (a device from before maxBindMB counts as WebGPU's 128 MiB)
      const bindMin = Math.min(...[myMeta, ...ai.chain.map((id) => conns.get(id)?.meta)].map((m) => (m?.maxBindMB || 128) * 2 ** 20));
      const fitCtx = ctxForBinding(ai.G.meta, ROOM_CTX, ROOM_KV, bindMin);
      if (fitCtx < ROOM_CTX) { log("room", `context ${ROOM_CTX} needs bigger GPU buffers than a device here allows: using ${fitCtx}`); ROOM_CTX = fitCtx; }
      L = ai.G.meta["qwen35.block_count"] - (ai.G.meta["qwen35.nextn_predict_layers"] || 0);
      const w = qwen35ShardBytes(ai.G, { lo: 0, hi: 4, hasEmbed: false, hasHead: false }) / 4;
      layerAt = (c) => w + c * kvBytesPerLayerPos(ai.G.meta, ROOM_KV);   // the attention layers' KV cache at the room's context
      embedBytes = hostHeldBytes("qwen35", { embed: ai.G.tensors[GGML_EMBED]?.byteLength || 0, out: ai.G.tensors[GGML_OUTPUT]?.byteLength || 0, mtp: qwen35MtpBytes(ai.G) });
      experts = expertsOf(ai.G, L);   // what each layer's experts park (null for a dense model): ExpertStore's sizes
    } else if (M.kind === "qwen4exp") {
      // Qwen3.8-Flash-Next: each layer's own weights and its state at the context (room/models.js flashLayerCost, the
      // memory model the picker, the room node and its load check share), the embedding and the head's HC mixer +
      // output.weight on the host
      aiStatus("Looking up the model…");
      ai.G = await fetchModelHeader(M);
      ai.GModel = modelKey;
      L = qwen4expConfig(ai.G.meta).L;
      const G = ai.G;
      // (kv: the room's K / V format, room/ctxpick.js roomCtxPlan: f16 when every device loads it)
      layerAt = (cx, kv = "f16") => flashLayerCost(G, cx, { kv: kv === "f32" ? "f32" : "f16" });
      embedBytes = qwen4expShardBytes(ai.G, { lo: 0, hi: 0, hasEmbed: true, hasHead: true, qsa: false });
      // this tab's load margin past its layers: the biggest layer's upload staging and scratch, which no load check
      // here counts (only its other weights when it offloads every layer it holds: room/plan.js dealWithHostMargin)
      hostMargin = [flashTabHostMargin(G), flashTabHostMargin(G, true)];
      experts = expertsOf(ai.G, L);   // expert offload: a room node parks them (or reads them from disk), a tab reads them from OPFS
    } else {
      cfg = await (await fetch(M.cfg)).json();
      L = cfg.num_hidden_layers;
    }

    // real per-shard byte costs (gguf: from the file's own index)
    if (M.kind === "gguf") {
      aiStatus("Looking up the model…");
      ai.G = await fetchGGUFHeader(M.gguf, false);
      ai.GModel = modelKey;
      // one layer's weights plus its f32 KV cache at this room's context (engine/dense.js: 64 MB a
      // layer for the 1.7B at 8k, more than its weights), and what the host holds besides its layers
      const kvDim = cfg.num_key_value_heads * (cfg.head_dim || cfg.hidden_size / cfg.num_attention_heads);
      const w = Object.values(ggmlLayerNames(0)).reduce((s, nm) => s + (ai.G.tensors[nm]?.byteLength || 0), 0);
      layerAt = (c) => w + c * denseKvBytesPerLayerPos(kvDim);
      embedBytes = hostHeldBytes("gguf", { embed: ai.G.tensors[GGML_EMBED]?.byteLength || 0, out: ai.G.tensors[GGML_OUTPUT]?.byteLength || 0 });
    } else if (M.kind === "safetensors") {
      const d = cfg.hidden_size;
      const kvDim = cfg.num_key_value_heads * ((cfg.head_dim || d / cfg.num_attention_heads));
      layerAt = (c) => (2 * d * d + 2 * kvDim * d + 3 * cfg.intermediate_size * d) * 4 + c * denseKvBytesPerLayerPos(kvDim);
      embedBytes = cfg.vocab_size * d * 4;
    }
    // what each device lends, held to its kind's cap (phones: room/pledge.js) and to a share this host
    // lowered after the device's tab was killed while loading (aiLoadDeath). A pledge is a promise:
    // no device is dealt more layers than fit in it (the host's also pays for the embedding, the
    // head and the draft block), and a room whose pledges can't hold the model does not start.
    ai.shareCap ??= new Map();
    const nameOf = (id) => conns.get(id)?.name || id;
    const pledgeOf = (m, name) => dealPledgeBytes(m, ai.shareCap.get(name), modelKey);   // (64 GB a device, 128 for Flash)
    const pledges = [pledgeOf(myMeta, myName), ...ai.chain.map((id) => pledgeOf(conns.get(id)?.meta, nameOf(id)))];
    // expert offload: a room node with a discrete GPU (meta.offload, meta.ramGB) holds layers past its pledge with
    // their experts in its RAM, when the pledges alone can't hold the model (room/plan.js); a tab only with the OPFS disk
    // tier (meta.tabDisk: Qwen3.8-Flash-Next, once its user says yes), its offloaded layers' experts read from its disk.
    // With the model key a node started with --disk that has this model's files (room/pledge.js diskFor) may also
    // leave some layers' experts on disk when parking can't hold the model
    // (a device that refused the offload it was dealt, ai.noDisk: a tab keeping the model off its disk, an older node,
    // counts without any offload: no disk tier, nothing parked)
    const metasNow = () => [myMeta, ...ai.chain.map((id) => { const m = conns.get(id)?.meta; return m && ai.noDisk?.has(nameOf(id)) ? { ...m, tabDisk: undefined, offload: undefined, ramGB: undefined } : m; })];
    const ctxCap = ROOM_CTX < maxSeqFor(modelKey, CTX_ASK) ? ROOM_CTX : Infinity;   // (a qwen35 room's binding limit)
    let roomKv = null;
    const planNow = () => {
      const metas = metasNow();
      // the context: the picker's choice (or the model's default), then the ladder below it when only that fits these
      // pledges; Flash also within its K / V format's and its devices' limits. One plan with the picker (room/ctxpick.js
      // roomCtxPlan, as roomFitFor), on this header's costs (the picker's come from SHAPE: the same numbers)
      const costs = (c, kv) => ({ L, layers: layerAt(c, kv), hostBytes: embedBytes, experts });
      const plan = roomCtxPlan(modelKey, { metas, pledges, ask: CTX_ASK, choice: ctxChoiceFor(modelKey), kvAsk: KV_ASK, costs, margin: hostMargin, cap: ctxCap, wrapOff: withSplit });
      roomKv = plan.kv;
      // fastest devices first (measured ms per layer from earlier answers), fewest hops, or by memory;
      // either way devices that are not needed (phones while the computers hold the model) and devices
      // whose pledge is under one layer stay in the room as ask-only guests (room/plan.js dealRoom)
      const deal = plan.fitAt(plan.ctx, pledges, { mode: $("ai-split").value === "speed" ? "speed" : "memory",
        ms: [ai.msPerLayer.get(myName), ...ai.chain.map((id) => ai.msPerLayer.get(nameOf(id)))],
        phone: [isPhoneMeta(myMeta), ...ai.chain.map((id) => isPhoneMeta(conns.get(id)?.meta))], phoneLayers: PHONE_LAYERS,
        // Qwen3.8-Flash-Next: the device dealt the n-gram layer must have its table on disk (null for every other model)
        ngram: ngramFor(metas, ngramOf(modelKey)) }).deal;
      return { pick: plan, deal };
    };
    ai.moe = !!experts;                       // the speed panel: a MoE model's cards say where the experts are
    let { pick, deal } = planNow();
    // This tab's own disk (room/flashdisk.js): when the deal stores anything on it (offloaded layers' experts, the n-gram
    // table) and its user was not asked yet, ask once now (remembered); on a no, deal again without it
    const myDisk = deal.fit.fits && myMeta.tabDisk?.ask ? hostDiskBytes(deal, modelKey) : 0;
    const pressed = ai.pressedHere; ai.pressedHere = false;
    if (deal.fit.fits) { const hb = hostDiskBytes(deal, modelKey); if (hb > 0) lc.diskHint = hb; }
    if (myDisk > 0 && pressed) await diskYesByStart(myDisk);   // the press said what it saves: no question
    else if (myDisk > 0) {
      aiStatus("Waiting for you to say if the model may be saved on this computer");
      if (!(await askTabDisk({ model: modelKey, bytes: myDisk, host: true }))) {
        ({ pick, deal } = planNow());
        // "Not now" and the room can't run it without this disk: back to the picker quietly (no "N GB short" error);
        // it shows disk storage off, with Allow under Start
        if (!deal.fit.fits) { aiStartStopped("the host kept the model off its disk", { quiet: true }); return; }
      }
    }
    ROOM_CTX = pick.ctx;
    layerBytes = layerAt(ROOM_CTX, roomKv);
    if (typeof layerBytes !== "number") layerBytes = layersTotal(layerBytes, L) / L;   // (per-layer costs: their average, for the cards)
    ai.ctxWant = pick.fellBack ? pick.want : 0;
    ai.ctxNote = pick.fellBack ? ctxShortNote(shortName(modelKey), pick.ctx, pick.want) : "";
    ai.ctxK = ctxK(pick.ctx);
    if (ai.ctxNote) log("room", ai.ctxNote);
    ai.layerGB = layerBytes / 2 ** 30;
    if (!deal.fit.fits || !deal.used.length) throw Object.assign(new Error(shortWhy(M, deal.fit, [myName, ...ai.chain.map(nameOf)], [myMeta, ...ai.chain.map((id) => conns.get(id)?.meta)])), { plain: true });
    ai.outWhy = Object.fromEntries(Object.entries(deal.out).map(([i, why]) => [nameOf(ai.chain[i - 1]), why]));
    for (const i of Object.keys(deal.out)) ai.leftOut.add(ai.chain[i - 1]);
    if (Object.keys(deal.out).length) log("room", `${Object.keys(ai.outWhy).join(", ")} ask${Object.keys(ai.outWhy).length > 1 ? "" : "s"} without holding layers (${[...new Set(Object.values(ai.outWhy))].map((w) => w === "small" ? "a pledge under one layer" : "the others hold the whole model").join("; ")})`);
    ai.chain = deal.used.slice(1).map((i) => ai.chain[i - 1]);
    ai.chainNames = ai.chain.map(nameOf);
    const { assigned, ranges } = deal;
    ai.layersN = Object.fromEntries([[myName, assigned[0]], ...ai.chain.map((id, i) => [nameOf(id), assigned[i + 1]])]);
    // what each device holds of its pledge (the host's includes the embedding and the head)
    ai.heldGB = Object.fromEntries([myName, ...ai.chainNames].map((nm, k) => [nm, deal.held[k] / 2 ** 30]));
    log("room", `memory per device (of its pledge): ${deal.used.map((i, k) => `${[myName, ...ai.chain.map(nameOf)][k]} ${(deal.held[k] / 2 ** 30).toFixed(2)} of ${(pledges[i] / 2 ** 30).toFixed(1)} GB`).join(" · ")}`);
    if (deal.ngram) log("room", `the n-gram table (${(deal.ngram.bytes / 2 ** 30).toFixed(1)} GB, layer ${deal.ngram.layer}): ${[myName, ...ai.chainNames][deal.ngram.at]} reads it (from its own copy when it has one, else by HTTP range: its log says which)${deal.ngram.checked ? "" : "; whether it has the disk for it is not reported"}`);

    ai.deferred = [];
    // what every device already has cached, so each one can take its missing ranges from the room.
    // The host itself is never a source: it is loading its own layers and serving the whole room,
    // so a device missing a range goes to the model host instead of queueing behind it.
    const inv = M.gguf && conns.size ? await gatherInventory(M.gguf) : {};
    ai.wsrc = M.gguf ? weightSources(M.gguf, inv) : null;
    // who offloads (engine/generate.js: speculation stays off then, room/plan.js specWithOffload)
    ai.offloadBy = Object.fromEntries((deal.offload || []).map((o, k) => [k ? nameOf(ai.chain[k - 1]) : myName, o]).filter(([, o]) => o));
    if (dealOffloads(ai.offloadBy)) log("room", specWithOffload(true, OFFLOAD_SPEC) ? "speculative decoding stays on with expert offload (?offspec=1)"
      : "speculative decoding is off while experts are offloaded: plain decoding is faster there (?offspec=1 turns it on)");
    const offTab = (o) => `layers ${o.lo}\u2013${o.hi - 1} with their experts read from this device's disk (${(o.vramBytes / 2 ** 30).toFixed(1)} GB of GPU cache, ${(o.diskRamBytes / 2 ** 30).toFixed(1)} GB RAM cache, ${o.slots} experts per layer)`;
    const myOff = deal.offload?.[0] || null;
    if (myOff) log("room", `${myName} offloads ${offTab(myOff)}`);
    // Qwen3.8-Flash-Next's format (ai-load.flash): QSA when every dealt device loads it (meta.flashLong), else none
    const flash = flashRoomFormat(modelKey, [myMeta, ...ai.chain.map((id) => conns.get(id)?.meta)]);
    ai.roomModel = modelKey; ai.roomCtx = ROOM_CTX; ai.roomFlash = flash;
    if (M.kind === "qwen4exp") log("room", flash ? `Qwen3.8 Flash with QSA (${flash.kv} K/V) at ${ROOM_CTX} positions` : `Qwen3.8 Flash dense at ${ROOM_CTX} positions (a device here runs an older Pooled)`);
    ai.chain.forEach((id, i) => {
      const o = deal.offload?.[i + 1];
      const disk = o && Number.isInteger(o.diskLo);
      if (o?.diskOnly) log("room", `${nameOf(id)} offloads ${offTab(o)}`);
      else if (o) log("room", `${nameOf(id)} offloads layers ${o.lo}–${o.hi - 1} with their experts in RAM (${(o.ramBytes / 2 ** 30).toFixed(1)} GB parked, ${(o.vramBytes / 2 ** 30).toFixed(1)} GB of GPU cache)`
        + (disk ? `; layers ${o.diskLo}–${o.hi - 1} read theirs from disk (${(o.diskRamBytes / 2 ** 30).toFixed(1)} GB RAM cache)` : ""));
      const msg = {
        t: "ai-load", v: PROTOCOL, model: modelKey, range: ranges[i + 1], ctx: ROOM_CTX, kv: ROOM_KV,
        next: i + 1 < ai.chain.length ? ai.chain[i + 1] : "host",
        host: peer.id,
        inv,
        // the same shape as the room node's (packages/room-node/roomnode.js offloadMsg), disk tier included
        ...(o ? { offload: { ...offMsg(o), ...(flashDealReserve(modelKey) > 0 ? { batchReserve: flashDealReserve(modelKey) } : {}) } } : {}),
        ...(flash ? { flash } : {}),
      };
      ai.plan.set(conns.get(id)?.name || id, { msg, small: false });
      sendTo(id, msg);
    });
    ai.layersByName = Object.fromEntries([[myName, `${ranges[0][0]}–${ranges[0][1] - 1}`], ...ai.chain.map((id, i) => [conns.get(id)?.name || id, `${ranges[i + 1][0]}–${ranges[i + 1][1] - 1}`])]);
    broadcastAll({ t: "ai-layers", by: ai.layersByName, held: ai.heldGB, out: ai.outWhy });
    paintHeld();
    const splitDesc = [`you ${assigned[0]}+embed`, ...ai.chain.map((id, i) =>
      `${conns.get(id)?.name || id} ${assigned[i + 1]}`)].join(" · ");
    log("room", `${M.label} — layer split ${$("ai-split").value === "speed" ? "for speed" : "by pledge"}${ROOM_KV === "q8" ? ", int8 KV" : ""}: ${splitDesc}`);
    ai.loadingShard = true;
    try { await aiLoadShard(modelKey, ranges[0], true, true, ROOM_CTX, ROOM_KV, offMsg(myOff), flash); } finally { ai.loadingShard = false; }
    if (ai.startFailed) throw new Error(ai.startFailed);   // a device failed to load its layers while this one loaded
    if (n > 1) await aiLoadDraft(modelKey);
    if (ai.redealPending) { ai.busy = false; ai.dealing = false; aiAutoRedeal(ai.redealWhy); return; }   // a device died (or stayed away) while loading: deal again
    if (ai.degraded) aiLoading(false);        // a device left while this one loaded: the Re-deal button is on the panel
    // (detail for the room's log; the load card's sentence says this moment in plain words)
    log("room", n === 1
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
    if (err?.detail) log("room", `${myName}: ${err.detail}`);   // (the GPU check's detail: the message says what it means)
    aiStartStopped(stopReason(ai.startFailed, err), { failed: true, plain: !ai.startFailed && (!!err?.plain || err?.message === GPU_CHECK_FAILED) });   // a device that failed first is the reason, not what this load hit after
  } finally {
    ai.dealing = false;
    // a re-deal asked for after this device's load (the rejoin grace ran out while the deal was still
    // reading back checkpoints): deal again now, unless the room came whole meanwhile
    if (ai.redealPending && ai.role === "host" && ai.degraded && missingNames().length) { ai.busy = false; aiAutoRedeal(ai.redealWhy); }
    else if (!ai.degraded) ai.redealPending = false;
  }
}
// A start that cannot finish: this device failed, or a device in the chain could not load its layers
// (ai-error with load). Every screen goes back to the model picker with Start on, so the room can try
// again (the other screens are told with ai-start-failed). While this device still loads its own layers
// the load stops at its next tensor (the pacer) and aiStart's catch lands here.
// quiet: this device's own choice stopped it (a "Not now" to the disk question): no "failed" line here
// failed: a load failed (this device's, or one the host heard of): the load card stays where it got to and says what
// happened, with Try again and Pick another model (lcStop). Without it (a Start that went nowhere): the picker
function aiStartStopped(why, { quiet = false, failed = false, plain = false } = {}) {
  ai.startFailed = null; ai.pressedHere = false;
  ai.dealer = null; ai.dealerFresh = false; ai.dealerActive = false; ai.startAt = null;   // free for the next Start (room/authz.js)
  ai.redealPending = false;   // nothing to deal again: the room is stopped
  clearInterval(ai.progTimer);
  if (failed) lcStop(why, plain); else aiLoading(false);
  closeEngine();
  ai.chain = []; ai.chainNames = []; ai.plan = null;   // nothing to re-seat or re-deal
  ai.heldGB = null; ai.outWhy = {}; paintHeld();
  failWaiters(new Error(why));
  $("ai-panel").classList.remove("online");
  aiStatus(quiet ? "" : "failed: " + why);
  ai.busy = false;
  $("ai-start").disabled = false;
  $("ai-model").disabled = false;
  updateCluster();
  broadcastAll({ t: "ai-start-failed", why });
}
function aiLoadFailed(from, name, message) {
  const why = `${name} couldn't load its part of the model (${message})`;
  toast(`The model stopped: ${why}`);
  if (stopWhen(ai.loadingShard) === "defer") { ai.startFailed = why; aiStatus(`${why}: stopping the start…`); }
  else aiStartStopped(why, { failed: true });
}

// a worker whose start was stopped: its layers are not needed, free them (the next ai-load deals afresh)
function workerStopped() {
  if (ai.role !== "worker") return;
  closeEngine();
  ai.role = null; ai.range = null;
  try { ai.device?.destroy(); } catch {}
  ai.device = null;
}

// Deal the layers again over whoever is in the room now: after a device left (the room is
// degraded) or to bring in devices that joined after the start. Cached ranges reload in seconds;
// the conversation is kept and re-prefilled on the next question.
async function aiRedeal(force = false) {
  if (ai.role !== "host" || (!force && (ai.busy === "gen" || ai.busy === "code"))) return;
  clearTimeout(ai.idleRedeal);
  if (ai.loadingShard) { toast("wait for this device's layers to finish loading, then re-deal"); return; }
  $("chat-tools").hidden = true;
  const model = ai.model || $("ai-model").value;
  // who this adds (the slim line over the chat names them) and how long that took them before, if they loaded here
  // (added on its own: the devices it chose; else the ones that are not phones: a phone stays out while computers hold the model)
  const spare = sparePeers(), comp = spare.filter((id) => !isPhoneMeta(conns.get(id)?.meta));
  const adding = ai.addPending?.length ? ai.addPending.slice() : (comp.length ? comp : spare).map((id) => conns.get(id)?.name || id);
  const guess = redealGuess((comp.length ? comp : spare).filter((id) => adding.includes(conns.get(id)?.name || id)).map(joinerLoad));
  ai.redealAt = Date.now(); clearTimeout(ai.addTimer); ai.addPending = [];
  failWaiters(new Error("re-dealing the layers"));
  ckptClear();
  closeEngine(); ai.busy = false; ai.fed = null;
  $("ai-panel").classList.remove("online");
  $("ai-row").style.display = "none";
  showRedeal(false);
  broadcastAll({ t: "ai-redeal", by: myName, model, adding, ...(guess != null ? { guess: Math.round(guess) } : {}) });
  codeRoleChanged();
  aiLoading(true, "re-dealing the layers");
  lc.adding = adding; lc.guess = guess; lcPaint(true);
  await aiStart(model);
}
// label: the button's words (devices that joined late: "Add pc to the model"); any other reason keeps the default
const REDEAL = "Re-deal the layers";
function showRedeal(on, why, label = REDEAL, { skip = false } = {}) {
  const b = $("ai-redeal");
  b.hidden = !on || ai.role !== "host";
  if (on) b.textContent = label;
  if (why) $("redeal-why").textContent = why;
  $("redeal-why").hidden = b.hidden;
  $("ai-redeal-skip").hidden = b.hidden || !skip;
}
// what a device that joined late would load, and how fast it loaded here before (null: it hasn't): redealGuess's input
function joinerLoad(id) {
  const e = conns.get(id), nm = e?.name || id;
  return { bytes: pledgeGB(e?.meta, ai.shareCap?.get(nm), ai.model) * 2 ** 30, rate: loadRates[nm] || 0 };
}
// devices with a GPU that are in the room but hold no layers (joined after the start)
// (a device the deal left out by name is still left out under a new id after a reload, not "late")
function sparePeers() { return [...conns.keys()].filter((id) => conns.get(id)?.meta?.webgpu && !ai.chain.includes(id) && !ai.leftOut?.has(id) && !["unneeded", "small"].includes(ai.outWhy?.[conns.get(id)?.name])); }
function offerRedealForNewcomers() {
  if (ai.role !== "host" || !ai.engine || ai.degraded) return;
  const spare = sparePeers();
  const n = spare.length, names = spare.map((id) => conns.get(id)?.name || id);
  // (an honest wait: the PC took 100 s to load its part of Qwen3.8 Flash, not "a few seconds")
  const wait = redealWait(redealGuess(spare.map(joinerLoad)));
  const auto = autoAddTick();
  if (n && auto.length) showRedeal(true, `${andList(auto)} joined. Adding ${auto.length > 1 ? "them" : "it"} to the model when the room is free: ${wait}, and the chat stays.`, "Add now", { skip: true });
  else if (n) showRedeal(true, `${names.join(", ")} ${n > 1 ? "are" : "is"} here but not helping yet. Add ${n > 1 ? "them" : "it"} to share the work and speed up answers. Takes ${wait}; the chat stays.`,
    `Add ${n > 1 ? `${n} devices` : names[0]} to the model`);
  // every screen says who waits for a re-deal (their cards, and their own Lend screen)
  if (spare.length) { ai.outWhy = outNow(); paintHeld(); broadcastAll({ t: "ai-out", out: ai.outWhy }); }
}
const andList = (ns) => (ns.length > 2 ? `${ns.slice(0, -1).join(", ")} and ${ns[ns.length - 1]}` : ns.join(" and "));
// A device that joins while the model runs is added on its own (room/autoadd.js): a few seconds after the newest join,
// once the room is idle, at most one re-deal every 90 s and the same device at most twice. Looks again while it has to
// wait; returns the names it is about to add ([] for none: the offer's button stays for those)
function autoAddTick() {
  clearTimeout(ai.addTimer); ai.addTimer = 0;
  if (ai.role !== "host") return (ai.addPending = []);
  const now = Date.now(), spare = sparePeers();
  ai.joinSeen ??= new Map(); ai.addSkip ??= new Set(); ai.addCount ??= new Map();
  const devs = spare.map((id) => {
    const m = conns.get(id)?.meta || {}, nm = conns.get(id)?.name || id;
    if (!ai.joinSeen.has(nm)) ai.joinSeen.set(nm, now);
    return { name: nm, webgpu: !!m.webgpu, api: !!m.api, phone: isPhoneMeta(m), gb: pledgeGB(m, ai.shareCap?.get(nm), ai.model) };
  });
  const help = wouldHelp({ split: $("ai-split").value, offload: Object.keys(ai.offloadBy || {}).length > 0,
    unneeded: [...conns.values()].some((c) => ai.outWhy?.[c.name] === "unneeded"), chain: ai.chain?.length || 0 });
  const names = help ? addable(devs, { layerGB: ai.layerGB || 0, skip: ai.addSkip, added: ai.addCount }) : [];
  const w = addWhen({ on: autoAddOn(), host: true, online: !!ai.engine && $("ai-panel").classList.contains("online"), degraded: !!ai.degraded,
    busy: !!ai.busy, loading: !!ai.loadingShard || !!ai.dealing, queued: ai.queue?.length || 0, names,
    joinedAt: Math.max(0, ...names.map((nm) => ai.joinSeen.get(nm) || 0)), lastAt: ai.redealAt || 0, now });
  ai.addPending = w.go || w.wait != null ? names : [];
  if (w.go) {
    for (const nm of names) ai.addCount.set(nm, (ai.addCount.get(nm) || 0) + 1);
    log("room", `adding ${names.join(", ")} to the model (joined while it ran)`);
    setTimeout(() => aiRedeal(), 0);
  } else if (w.wait != null) ai.addTimer = setTimeout(offerRedealForNewcomers, w.wait);   // (which looks again, and repaints the offer)
  return ai.addPending;
}
function autoAddOn() { return $("ai-autoadd")?.checked !== false; }
// why each device with a GPU holds no layers, by name: "unneeded" (the others hold the model),
// "small" (its pledge is under one layer), "late" (it joined after the start and waits for a re-deal)
function outNow() {
  const out = { ...(ai.outWhy || {}) };
  for (const id of sparePeers()) out[conns.get(id)?.name || id] = "late";
  return out;
}
// what a device the deal left out says in its own chat (the Serve API screen: room/compute.js lendStatus)
function outNote(why) {
  if (why === "small") return "Not holding layers: what this device lends is less than one layer of this model. You can still ask.";
  if (why === "late") return "This device joined after the start: it starts helping once the room adds it (about a minute or two to load its part). You can still ask.";
  return `Not needed for this model: the ${isPhoneMeta(myMeta) ? "computers" : "other devices"} hold it. You can still ask.`;
}
// the device cards: what each holds of its pledge, or why it holds nothing
const OUT_TEXT = { unneeded: "not needed: the others hold the model", small: "pledge under one layer", late: "joined after the start: not helping yet" };
function paintHeld() {
  for (const card of document.querySelectorAll("#peers .peer-card")) {
    const nm = card.dataset.name, why = ai.outWhy?.[nm], held = ai.heldGB?.[nm];
    const el = card.querySelector(".pheld"); if (!el) continue;
    const meta = nm === myName ? myMeta : [...members.values()].find((m) => m.name === nm)?.meta;
    el.textContent = why === "unneeded" && isPhoneMeta(meta) ? "not needed: the computers hold the model"
      : why ? OUT_TEXT[why] || "" : held != null ? `uses ${held.toFixed(1)} of ${+pledgeGB(meta, Infinity, $("ai-model")?.value).toFixed(1)} GB pledged` : "";
    el.hidden = !el.textContent;
  }
  speedDeal();
}

// a device in the chain left: every lap in flight fails now instead of timing out, and the room
// waits for a re-deal
function aiPeerLeft(id, name, verb = "left") {
  if (ai.role !== "host") return;
  if (!ai.chain.includes(id)) { offerRedealForNewcomers(); if (!sparePeers().length && !ai.degraded) showRedeal(false); return; }
  const layers = ai.layersByName?.[name];
  const why = `${name || "a device"} ${verb}${layers ? ` (layers ${layers})` : ""}`;
  ai.degraded = true;
  ai.readyPeers.delete(id);
  ai.gone.add(id);   // it may come back under the same peer id (the ping loop dropped a live but stalled tab): aiRejoin
  // an answer or a Code run holds the room: it waits for the device (or a re-deal) and carries on
  // (roomGenerate's recovery); marked here, before codeRoleChanged, so Code does not stop the run
  const running = ai.busy === "gen" || ai.busy === "code";
  if (running && !ai.recovering) ai.recovering = { kind: ai.busy, since: Date.now(), pending: true };
  // left before the room came online: drop the load card so the panel's Re-deal button shows
  // (while this device still loads, aiStart does it once its layers are in)
  if (!ai.loadingShard && !$("ai-panel").classList.contains("online")) aiLoading(false);
  // with disk copies the index stays: a rejoin keeps what the device reads back (ckptRejoin, then
  // ckptPrune on its ai-ready), a re-deal clears it and reads it back (aiStart)
  ai.fed = null; if (!ckptDisk) ckptClear();
  failWaiters(new Error(why));
  // (under a load card the composer stays, off: nothing jumps)
  $("ai-row").style.display = ai.engine || $("load-card").classList.contains("on") ? "flex" : "none";
  const auto = autoRedealOn(), secs = Math.round(REJOIN_GRACE_MS / 1000);
  aiStatus(running ? `${why}: waiting for it to come back, then this ${ai.busy === "code" ? "Code run" : "answer"} carries on` : `${why}: waiting for it to come back${auto ? ` (re-dealing without it in ${secs} s)` : ""}, or re-deal the layers`);
  showRedeal(true, `${why}. It goes back into its slot if it returns${auto ? `; otherwise the layers are dealt again over the devices still here in ${secs} s (experimental)` : ""}. Or re-deal now; the chat stays.`);
  broadcastAll({ t: "ai-degraded", why: `${why}: waiting for it to come back` });
  codeRoleChanged();
  // idle room: re-deal on its own after the grace period (a run in progress does this in roomRecover).
  // A device that left while the deal loads counts too: the grace can run out while this device still
  // loads its own layers (a slow host: the 35B MoE on a PC takes minutes), or after, with the start's
  // lock (busy) still held because the room never came whole. That used to skip the re-deal for good
  // and leave the room waiting forever; now aiAutoRedeal deals again once this device's layers are in
  clearTimeout(ai.idleRedeal);
  if (auto && !running) ai.idleRedeal = setTimeout(() => {
    if (ai.role !== "host" || !ai.degraded || ai.busy === "gen" || ai.busy === "code" || ai.recovering || !autoRedealOn() || !missingNames().length) return;
    const why = `${missingNames().join(", ")} did not come back in ${secs} s: re-dealing the layers (experimental auto re-deal)`;
    log("room", why);
    const had = ai.redealPending;
    aiAutoRedeal(why);
    if (ai.redealPending && !had) ai.redealPending = "grace";   // (dropped if it comes back before then: aiRejoin)
  }, REJOIN_GRACE_MS);
}
// chain devices that are not in the room now (by name, for the screen)
function missingNames() { return ai.chain.map((id, i) => (conns.has(id) ? null : ai.chainNames?.[i] || id)).filter(Boolean); }
// the host's setting: re-deal on its own when a device does not come back (experimental; default on)
function autoRedealOn() { return $("ai-autoredeal")?.checked !== false; }

// a newcomer while the room is online gets the chat as a guest, and the conversation so far
// The room's state for a device that says hello (it joins, reloads, or comes back on a new link), sent
// on every hello and not only once: its screen shows the chat from ai-ready-all, so a device that
// missed it (a replacement link that came in before the host saw the old one close: the host had no
// departure to re-seat, #265) would keep a hidden input. A device in the chain that is being re-seated
// (aiRejoin: ai-load) gets it from aiMaybeReady once the chain is whole again.
function aiWelcome(id, { history = true } = {}) {
  if (ai.role !== "host" || !ai.engine || ai.readyPeers.size < ai.chain.length || relinking()) { offerRedealForNewcomers(); return; }
  const inChain = ai.chain.includes(id);
  if (inChain && !ai.readyPeers.has(id)) return;
  if (ai.layersByName) sendTo(id, { t: "ai-layers", by: ai.layersByName, held: ai.heldGB, out: outNow() });
  sendTo(id, { t: "ai-ready-all", model: ai.model, label: MODELS[ai.model]?.label, ctx: ctxMax(), ...(ai.ctxWant ? { ctxWant: ai.ctxWant } : {}), out: outNow() });
  if (!inChain && history && ai.visibility === "all" && ai.transcript.length) sendTo(id, { t: "ai-history", items: ai.transcript.slice(-20) });
  offerRedealForNewcomers();
}

// links from this device (a model host that is not the room's creator) to every device in the
// room's roster it has none to; API clients talk to the creator only
async function linkMembers(ms = 15000) {
  const ids = [...members].filter(([id, m]) => !conns.has(id) && !m.meta?.api).map(([id]) => id);
  await Promise.all(ids.map((id) => ensureLink(id, ms)));
}
// A device that joins (or reloads into) a room whose model host is not the room's creator only links
// to the creator. The model host learns of it from the roster: it links to it and welcomes it, as the
// creator would (aiWelcome), with the layer map, so a device left out of the deal sees why.
function welcomeFar(id) {
  if (isHost || ai.role !== "host" || !ai.engine || conns.has(id) || welcomeFar.pending.has(id)) return;
  welcomeFar.pending.add(id);
  ensureLink(id, 20000).then((ok) => {
    welcomeFar.pending.delete(id);
    if (!ok || ai.role !== "host" || !ai.engine) return;
    if (ai.layersByName) sendTo(id, { t: "ai-layers", by: ai.layersByName, held: ai.heldGB, out: outNow() });
    aiWelcome(id);
  });
}
welcomeFar.pending = new Set();

// A device in the chain comes back from a tab that was killed while it loaded its layers (the
// "loading" breadcrumb in its hello: iOS closes a Safari tab that goes over its memory budget, #207).
// Putting it back in its slot would load the same layers and get it killed again, so the room is
// re-dealt: with a smaller share for it, or without it (it stays as a guest that can ask) when it
// was down to one layer already or was killed twice (room/pledge.js afterLoadDeath).
// Returns true when it handled the device (aiRejoin must not re-seat it).
function aiLoadDeath(newId, d) {
  const name = d.name;
  if (ai.role !== "host" || !d.died?.loading || !ai.plan?.has(name) || !ai.chainNames?.includes(name)) return false;
  ai.loadDeaths ??= new Map(); ai.shareCap ??= new Map(); ai.dropped ??= new Set(); ai.deathsSeen ??= new Set();
  // the same kill again (the device's hello on another link, or a reconnect): already handled
  const kill = name + "@" + (d.died.at ?? d.died.ago);
  if (ai.deathsSeen.has(kill)) return true;
  ai.deathsSeen.add(kill);
  const deaths = (ai.loadDeaths.get(name) || 0) + 1;
  ai.loadDeaths.set(name, deaths);
  const meta = conns.get(newId)?.meta;
  const r = afterLoadDeath({ layers: ai.layersN?.[name] || 1, layerGB: ai.layerGB || 0.5, gb: pledgeGB(meta, ai.shareCap.get(name), ai.GModel), deaths });
  let why;
  if (r.drop) {
    ai.dropped.add(name);
    why = `${name}'s browser closed its tab while it loaded its layers${deaths > 1 ? " again" : ""}: re-dealing without it (it can still ask)`;
    sendTo(newId, { t: "ai-share", drop: true, why: "This device's browser closed the tab while it loaded its layers, so the room runs without it. You can still ask questions." });
  } else {
    ai.shareCap.set(name, r.gb);
    why = `${name}'s browser closed its tab while it loaded its layers: re-dealing with a smaller share for it (${r.gb} GB)`;
    sendTo(newId, { t: "ai-share", gb: r.gb, why: `This device's browser closed the tab while it loaded its layers, so it now holds less of the model (${r.gb} GB).` });
  }
  log("room", why);
  sysNote(why);
  // the link from before the reload may not have timed out yet: close it, so the re-deal can't pick it
  for (const [id, e] of conns) if (id !== newId && e.name === name) try { e.conn.close(); } catch {}
  aiAutoRedeal(why);
  return true;
}
// re-deal on its own (after a load death); waits for this device's own layers when they are still loading
function aiAutoRedeal(why) {
  if (ai.role !== "host") return;
  if (ai.loadingShard || ai.dealing || ai.busy === "gen" || ai.busy === "code") { ai.redealWhy = why; ai.redealPending = true; return; }
  ai.redealPending = false;
  aiStatus(why);
  setTimeout(() => aiRedeal(), 500);   // after the old link's close has run
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
  clearTimeout(ai.idleRedeal);
  if (ai.redealPending === "grace" && !missingNames().length) ai.redealPending = false;   // back after all: no re-deal
  const { msg } = ai.plan.get(name);
  const fresh = { ...msg, next: i + 1 < ai.chain.length ? ai.chain[i + 1] : "host", host: peer.id };
  // the device before it opens a fresh link and says so (ai-linked): until then the chain is not whole,
  // even with every device ready (the first frames of a resumed answer would go down the dead link)
  if (i > 0) { sendTo(ai.chain[i - 1], { t: "ai-next", next: newId, relink: 1 }); ai.relinks.set(newId, Date.now() + RELINK_MS); setTimeout(aiMaybeReady, RELINK_MS + 100); }
  sendTo(newId, fresh);
  ai.fed = null; ckptRejoin();              // its fresh engine holds only what it reads back from disk
  log("room", `${name} came back into its slot`);
  aiStatus(`${name} reconnected, getting its layers back…`);
  $("ai-row").style.display = ai.readyPeers.size >= ai.chain.length ? "flex" : "none";
}
const RELINK_MS = 15000;   // how long the host waits for an ai-linked (an older device never sends one)
function relinking() {
  for (const [id, until] of ai.relinks) if (Date.now() > until || !ai.chain.includes(id)) ai.relinks.delete(id);
  return ai.relinks.size > 0;
}
function aiMaybeReady() {
  if (ai.role !== "host" || !ai.engine || ai.ckptRestoring) return;   // (aiStart calls it again once the restore is done)
  ckptPrune();
  if (ai.readyPeers.size < ai.chain.length || relinking()) return;
  if (!ai.chain.every((id) => conns.has(id) && ai.readyPeers.has(id))) return;   // (a device that left is not ready)
  const n = ai.chain.length + 1;
  ai.degraded = false;
  clearTimeout(ai.idleRedeal);
  if (!ai.recovering) ai.busy = false;   // a run waiting in roomRecover keeps the room's lock
  showRedeal(false);
  aiStatus(`cluster online · ${n} device${n > 1 ? "s" : ""}, ${ai.cfg.num_hidden_layers} layers split ${n} ways`);
  clearInterval(ai.progTimer);
  // the load card says "ready" where it stands and stays until the first question (aiOut); a device back from a
  // reload also lands here, with the chat up and no card
  if ($("load-card").classList.contains("on")) lcReady();
  const wasOnline = $("ai-panel").classList.contains("online");
  $("ai-panel").classList.add("online");
  if (!wasOnline) speedReady();   // speedReady (C) (aiMaybeReady runs again after a relink, re-deal or recovery)
  promptHint();
  $("ai-row").style.display = "flex";
  $("chat-tools").hidden = false;
  $("new-chat").hidden = false;
  $("mode-bar").hidden = false;
  setAfterAnswer(false, !!ai.conv.turns.length);
  if (lc.phase !== "ready") emptyText("The model is ready. Ask anything.");
  sysNote(`Model ready on ${n} device${n > 1 ? "s" : ""}`);
  if (ai.ctxNote) sysNote(ai.ctxNote);
  if (!matchMedia("(pointer: coarse)").matches) $("ai-prompt").focus();   // touch: the keyboard opens when the user taps the prompt
  broadcastAll({ t: "ai-ready-all", model: ai.model, label: MODELS[ai.model]?.label, ctx: ctxMax(), ...(ai.ctxWant ? { ctxWant: ai.ctxWant } : {}), out: outNow() });
  pushMap(0, null, false, true);
  offerRedealForNewcomers();
  if (!ai.recovering) setTimeout(nextQueued, 0);
  saveHost();
  keepAwake();
  mascot("Cluster online! Ask anything. Everyone in the room can.");
  codeRoleChanged();
}

// the slowest round trip to a device in the chain (pongs), for lap timeouts
function chainRtt() { return Math.max(0, ...ai.chain.map((id) => conns.get(id)?.rtt || 0)); }
// a decode lap starts: the workers that asked for it (a phone) wake their GPU now, so it is clocked
// up when this lap's frame reaches them (room/gpuwake.js). A hint only: a worker that misses it or
// gets it late is just slower, and it is ignored once its frame has arrived.
function wakeChain(pos) {
  if (WAKE === "0" || !ai.chain.length) return;
  for (const id of ai.chain) if (conns.get(id)?.meta?.wake) sendTo(id, { t: "ai-wake", pos });
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

const PREFILL_WINDOW = 6;
const LOOKUP = new URLSearchParams(location.search).get("lookup") !== "0";   // ?lookup=0: draft head only, for A/B
// ?densespec=0: dense models (no draft head) decode plainly in a room, no lookup drafts (A/B)
const DENSE_SPEC = new URLSearchParams(location.search).get("densespec") !== "0";
// speculation while a room node in the deal offloads experts (room/plan.js specWithOffload): off by default
// there (plain decoding is faster), ?offspec=1 forces it on, ?offspec=0 off
const OFFLOAD_SPEC = parseForce(new URLSearchParams(location.search).get("offspec"));
// a disk-only tab host + a room node that parks (room/plan.js splitBySpeed): ?flashsplit=0 keeps the split by memory,
// ?hostlayers=K gives the host K layers (tests)
const SPLIT_Q = new URLSearchParams(location.search);
const withSplit = (off) => (!off ? off : { ...off, ...(SPLIT_Q.get("flashsplit") === "0" ? { speedSplit: false } : {}), ...(+SPLIT_Q.get("hostlayers") > 0 ? { hostLayers: +SPLIT_Q.get("hostlayers") } : {}) });
// ?draft=qwen3-0.6b (experimental, off by default): the host also loads that small model whole and it
// drafts when prompt lookup finds nothing (room/draftmodel.js); ?draftk=N drafts per lap (default 4).
// Same tokenizer as the Qwen3 1.7B / 4B. Exact like lookup: the verify decides every token.
const DRAFT_MODEL = new URLSearchParams(location.search).get("draft");
const DRAFT_K = Math.max(1, Math.min(7, parseInt(new URLSearchParams(location.search).get("draftk"), 10) || 4));
const DRAFT_FOR = { "qwen3-0.6b": ["qwen3-1.7b", "qwen3-4b"] };
async function aiLoadDraft(modelKey) {
  ai.draft = null;
  if (!DRAFT_MODEL || !DRAFT_FOR[DRAFT_MODEL]?.includes(modelKey) || !ai.engine || ai.engine.specStep) return;
  try {
    aiStatus(`loading the draft model (${DRAFT_MODEL})\u2026`);
    const M = MODELS[DRAFT_MODEL];
    const e = await modelLoader.loadDraft({ modelKey: DRAFT_MODEL, pace: aiLoadPace }, { DenseEngine });
    ai.draft = new DraftModel(e, { argmax });
    log("room", `draft model ${M.label} loaded on the host: it drafts ${DRAFT_K} tokens per lap when prompt lookup finds nothing`);
  } catch (err) { ai.draft = null; log("room", `the draft model did not load (${err?.message || err}); prompt lookup only`); }
}
const TAIL_FRAME = new URLSearchParams(location.search).get("tail") !== "0";   // ?tail=0: old per-token tail, for A/B   // prefill rounds in flight round the chain at once

// Engines stay lazy-loaded; the browser owns delivery, storage and UI observers.
const pipeline = createPipeline({
  state: ai, transport: { sendHidden, sendTo, chainRtt },
  options: { checkpointMax: CKPT_MAX, fillDrafts: FILL_DRAFTS, mtpBatch: MTP_BATCH, tailFrame: TAIL_FRAME, prefillWindow: PREFILL_WINDOW },
  checkpointStore: ckptDisk, getRoomCode: () => roomCode,
  hooks: { onStatus: aiStatus, computePass: compute.pass, teleNote, wakeChain, keepWarm,
    onFrame(d) { ai.waker?.stop(); ai.lastFramePos = d.t === "ai-hidden" ? d.pos : d.basePos; } },
});
const { failWaiters, ckptClear, ckptRestore, ckptPrune, ckptRejoin } = pipeline;
const roomGenerateOnce = createGenerator({
  state: ai, pipeline,
  options: { checkpointMax: CKPT_MAX, fillDrafts: FILL_DRAFTS, lookup: LOOKUP, hostFuse: HOST_FUSE, denseSpec: DENSE_SPEC, draftK: DRAFT_K, offloadSpec: OFFLOAD_SPEC },
  hooks: { computePass: compute.pass, mapPulse, pushMap, crumb, noteSpeeds, wakeChain, chainRtt, log,
    getPeerMeta: (id) => conns.get(id)?.meta,
    onSoloSpeed(tps) { lastSoloTps = Math.max(lastSoloTps, tps); } },
});
// ---- telemetry and the room map ----
// Workers report their compute per frame kind (ai-tele); the host times each lap, so what is left
// is the wire. The map shows the chain, what each device holds and how long its part takes.
function mapNodes(kind = "spec") {
  const nodes = [{ name: myName, layers: ai.layersByName?.[myName] || "", ms: ai.lapStat?.host, host: 1, amax: ai.hostAmax, exp: expertsAt(ai.moe, ai.offloadBy?.[myName]), xc: expertCacheOf(expertStoreOf(ai.engine)) }];
  for (const id of ai.chain) {
    const t = ai.teleBy.get(id) || {};
    const name = conns.get(id)?.name || id;
    nodes.push({ name, layers: ai.layersByName?.[name] || "", ms: t[kind] ?? t.spec ?? t.one, amax: t.amax, exp: expertsAt(ai.moe, ai.offloadBy?.[name]), xc: t.xc || null });
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
  // "Qwen3 1.7B · 8K context" when the room's memory was short for the model's default context
  el.querySelector(".sm-model").textContent = shortName(ai.model || $("ai-model").value) + (ai.ctxNote ? ` · ${ai.ctxK} context` : "");
  el.querySelector(".sm-model").title = ai.ctxNote || "";
  // the device cards say which layers they hold, in the strip's colours
  for (const card of document.querySelectorAll("#peers .peer-card")) {
    const k = nodes.findIndex((x) => x.name === card.dataset.name);
    card.style.setProperty("--sw", devColor(card.dataset.name));
    card.style.setProperty("--k", Math.max(0, k));
    card.classList.toggle("holds", k >= 0 && !!nodes[k].layers);
    // (which layers: the developer view and the tooltip; the card says it in words)
    const holds = k >= 0 && nodes[k].layers, play = card.querySelector(".play");
    play.textContent = !holds ? "" : DEV ? `layers ${humanRange(nodes[k].layers)}` : nodes.filter((x) => x.layers).length > 1 ? "holds part of the model" : "holds the whole model";
    play.title = holds ? `layers ${humanRange(nodes[k].layers)}` : "";
  }
  const S = lastMap.st;
  $("sm-tps").textContent = S?.tps ? S.tps.toFixed(1) : "-";
  $("sm-lap").textContent = S?.lap ? String(Math.round(S.lap)) : "-";
  (smLive ||= liveWords(el.querySelector(".sm-lt")))(!!live);   // a steady "Writing…" while writing
  paintLivePill();   // ("Reading…" before the answer's first word)
  el.querySelector(".sm-live").title = live ? "the room is writing an answer" : "waiting for a question";
  // (the head's line is the last answer's speed, speedMeta; the developer view's lap and drafts go to its tooltip only)
  const bits = [];
  if (st?.lap && DEV) bits.push(`lap ${st.lap} ms = GPUs ${st.gpu} + wire ${st.net}`);
  if (st?.acc != null && DEV) bits.push(`${Math.round(st.acc * 100)}% of drafts accepted`);
  speed.devTip = bits.join(" · ");
  speedMeta();
  deviceMark();
  // the speed panel: "online · 2 devices" once per deal (the first map after a re-deal, or a new count)
  if (speed.redeal || nodes.length !== speed.lastN) { speed.redeal = false; speed.lastN = nodes.length; speedEvent(onlineText(nodes.length)); }
  renderSpeed();
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
  // folded everywhere unless this viewer opened it ("Show the layers"): at ready the band is one line, never a panel
  // that opens by itself. (The developer view, ?dev=1, keeps the old default: open on a desktop's Chat, for its numbers.)
  if (v) return v === "folded";
  return DEV ? bandMode() === "code" || innerWidth < 820 || matchMedia("(max-height: 500px) and (pointer: coarse)").matches : true;
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
  speedTok();
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
// worker: EMA of compute ms per frame kind, reported to the host at most every 700 ms (room/tele.js: the same
// meter a room node worker reports with), with the expert cache's hits and misses when this engine offloads
function teleNote(kind, ms, n = 1) {
  const T = ai.tele ||= new TeleMeter();
  T.note(kind, ms, n, wireStats.lastMax || 0);
  const now = performance.now();
  if (T.due(now) && ai.hostId) sendTo(ai.hostId, T.report(now, expertStoreOf(ai.engine)));
}

// ---- the speed panel (room/speedo.js): this answer's tok/s, live and big; the recent answers' speeds,
// marked by how many devices held the model; the room's events (a device joins, the re-deal, online);
// and a card per device: its layers, its GB, where its experts are (MoE), its share of each token's time.
// Every screen times the tokens it is shown; a finished answer's speed is the host's (its stats line).
// (since / count: the answers since the last load and the device count they ran on, so the first ones after a load,
// a re-deal or a change of count are marked warm and never compared: room/speedo.js speedup)
const speed = { meter: new SpeedMeter(), hist: [], events: [], asked: new Set(), lastN: 0, redeal: false, at: 0, idle: 0, seq: 0, last: null, reading: false,
  firstWait: false, devTip: "", since: 0, count: 0 };
// The band's pill while the room works on an answer: "Reading…" until its first word is on screen (the model is still
// reading the message), "Writing…" after; renderMap's liveWords sets "Writing…" / "Ready", this corrects the first
function paintLivePill() {
  const m = $("swarm-map"), t = m?.querySelector(".sm-lt");
  if (t && m.classList.contains("live")) t.textContent = speed.reading ? "Reading\u2026" : "Writing\u2026";
}
function speedEvent(text) {
  speed.events = pushEvent(speed.events, text, Date.now());
  const ol = $("sp-events"); if (!ol) return;
  ol.innerHTML = speed.events.map((e, i) => `<li${i === speed.events.length - 1 ? ' class="now"' : ""}>${esc(e.text)}</li>`).join("");
}
// The model went ready (the host's aiMaybeReady, a device's first ai-ready-all): nothing about speed until a real
// answer. The panel (only open when this viewer asked for it) keeps its readout and chart out of sight (quiet), the
// head's line stays empty, the band folds unless this viewer opened it, and the first question gets its plain wait words.
function speedReady() {
  speed.firstWait = true;
  if (!speed.hist.length) $("speedo")?.classList.add("quiet");
  if (!speed.last) speedMeta();
  bandFold(bandFolded());
}
// the band head's one line: the last real answer's speed in words ("Last answer: about 3 words a second"), 12px,
// muted; "" before one. Its tooltip has the numbers ("1.5 tok/s · 24 tokens · 1 device"), and the dev view's lap.
function speedMeta() {
  const el = $("swarm-map")?.querySelector(".sm-meta"); if (!el) return;
  const L = speed.last;
  const gain = L ? gainText(speedup(speed.hist)) : "";
  el.textContent = bandLine(L) + (gain ? ` · ${gain}` : "");
  const tip = L ? `${fmtTps(L.tps)} tok/s` + (L.count ? ` · ${L.count} tokens` : "") + (L.devices ? ` · ${L.devices} device${L.devices === 1 ? "" : "s"}` : "") : "";
  el.title = [tip, DEV ? speed.devTip : ""].filter(Boolean).join("\n");
}
function speedStart() {
  speed.meter.start();
  clearTimeout(speed.idle);
  $("speedo")?.classList.add("live");
  speed.reading = true;
  speedShow(0, { live: true });
  paintLivePill();
}
// The readout: one plain line ("Writing about 3 words a second"), nothing until the rate settles; the exact tok/s in
// its tooltip. (It was a 64px tok/s number: loud, and a unit most people don't know.)
function speedShow(tps, { live = false, devices = 0, count = 0 } = {}) {
  const el = $("sp-tps"); if (!el) return;
  const line = speedLine(tps, { live, devices });
  el.textContent = line;
  el.title = tps > 0 ? `${fmtTps(tps)} tok/s` + (count ? ` · ${count} tokens` : "") + (devices ? ` · ${devices} device${devices === 1 ? "" : "s"}` : "") : "";
  // (while live the pill above says "Reading…" / "Writing…"; before an answer the chat says "The model is ready. Ask
  // anything.": no "Waiting for a question" here)
  $("sp-unit").textContent = "";
}
// a token on screen (mapPulse): the readout at most every 250 ms; 2.5 s without one ends the live state
// (an answer that never gets its end here: Code's turns, a dropped link)
function speedTok() {
  speed.firstWait = false;
  if (!speed.meter.on) speedStart();
  if (speed.reading) { speed.reading = false; paintLivePill(); }
  const now = performance.now();
  speed.meter.tok(now);
  clearTimeout(speed.idle);
  speed.idle = setTimeout(() => speedDone(null, null, true), 2500);
  if (now - speed.at < 250) return;
  speed.at = now;
  const r = speed.meter.rate(now);
  if (r > 0) { $("speedo")?.classList.remove("quiet"); speedShow(r, { live: true }); }   // (a real answer, two words or more, is running: the opened panel's readout may show)
}
// an answer ended (chatBotEnd): its speed from the host's stats line, else this screen's own timing
function speedDone(stats, key = null, own = false) {
  clearTimeout(speed.idle);
  speed.reading = false;
  speed.firstWait = false;
  const avg = speed.meter.end(), st = parseStats(stats);
  $("speedo")?.classList.remove("live");
  const tps = st ? (st.count >= 2 ? st.tps : 0) : avg;   // a one-token answer has no speed worth showing
  if (st && st.count >= 2) {
    if (st.devices !== speed.count) { speed.count = st.devices; speed.since = 0; }
    const warm = speed.since++ < warmupAnswers(st.devices);
    speed.hist = pushHistory(speed.hist, { key: key ?? "a" + ++speed.seq, tps: st.tps, devices: st.devices, warm, model: ai.model || null });
    speed.last = st;
  } else if (own && avg > 0 && speed.meter.n >= 2) speed.last = { count: speed.meter.n, tps: avg, devices: lastMap?.nodes?.length || 1 };   // (an answer with no stats line ever, e.g. Code's turns: this screen's own timing; a failed or stopped one leaves it)
  if (speed.last) $("speedo")?.classList.remove("quiet");   // the head line has a speed: so may the opened panel
  speedMeta();
  if (tps > 0) speedShow(tps, { devices: st?.devices || 0, count: st?.count || 0 });
  else speedShow(speed.last?.tps || 0, { devices: speed.last?.devices || 0 });   // a one-token, failed or stopped answer: the last one's speed, never "Writing…" with nothing running
  paintHistory();
}
function paintHistory() {
  const ol = $("sp-hist"); if (!ol) return;
  const h = speed.hist, hs = barScale(h), cur = h.length ? h[h.length - 1].devices : 0;
  ol.innerHTML = h.map((x, i) => {
    const step = i === 0 || h[i - 1].devices !== x.devices;
    return `<li class="bar${x.devices === cur ? " cur" : ""}${step ? " step" : ""}${x.warm ? " warm" : ""}" data-tps="${x.tps}" data-n="${x.devices}" style="--h:${hs[i].toFixed(3)}" title="${fmtTps(x.tps)} tok/s on ${x.devices} device${x.devices === 1 ? "" : "s"}${x.warm ? " (warming up: not compared)" : ""}">`
      + `<i></i>${step ? `<span class="nl">${x.devices} device${x.devices === 1 ? "" : "s"}</span>` : ""}</li>`;
  }).join("");
  ol.classList.toggle("empty", !h.length);
  if (h.length) $("speedo")?.classList.remove("quiet");   // the first real answer: the readout and the chart may show
  $("sp-gain").textContent = gainText(speedup(h));
}
// the model, its size, and the device cards; from the deal (ai.layersByName, ai.heldGB, ai.outWhy) and the
// last map (each device's ms per lap and, for a MoE model, where its experts are)
function renderSpeed() {
  const box = $("speedo"); if (!box) return;
  const key = ai.model || $("ai-model").value, M = MODELS[key];
  $("sp-model").textContent = shortName(key);
  // "125B model"; its size in words, the active count and the quantization in the tooltip
  const pa = M ? paramsOf(M.label, M.gguf || "", key) : null;
  $("sp-params").textContent = pa ? sizeText(pa) : "";
  $("sp-params").title = pa ? `${String(pa.total).replace(/B$/, " billion").replace(/M$/, " million").replace(/T$/, " trillion")} parameters: the model's size\n${paramsText(pa)}` : "";
  const nodes = lastMap?.nodes || [], st = lastMap?.st || {};
  const by = ai.layersByName || {}, holders = Object.keys(by);
  const shares = tokenShares(nodes, st.lap);
  const names = [...holders];
  for (const c of document.querySelectorAll("#peers .peer-card")) if (!names.includes(c.dataset.name)) names.push(c.dataset.name);
  const solo = holders.length === 1;
  const totalL = holders.reduce((t, k) => { const m = /^(\d+)\D+(\d+)$/.exec(by[k]); return m ? Math.max(t, +m[2] + 1) : t; }, 0);
  $("sp-devs").innerHTML = names.map((nm) => {
    const meta = metaOf(nm) || {}, x = nodes.find((y) => y.name === nm), r = by[nm];
    const n = /^(\d+)\D+(\d+)$/.exec(String(r || "")), nL = n ? +n[2] - +n[1] + 1 : 0;
    const gb = ai.heldGB?.[nm], why = ai.outWhy?.[nm];
    const part = shares?.parts.find((p) => p.name === nm);
    const role = !r ? (meta.api ? "connected app" : why === "late" ? "waits for the model to be shared out again" : "only asking questions") : holders[0] === nm ? (solo ? "runs the whole model" : "started the room") : "holds part of the model";
    // in plain words on the card: how much of the model it holds and its GB; which layers, where its experts are, the
    // cache's hit rate, what a token costs it and its ms of each token go to the card's tooltip (the room's numbers
    // stay one hover away, not on the screen)
    const lines = [], tip = [];
    const share = totalL ? nL / totalL : 0;
    if (r) {
      // (one device alone: its role already says "runs the whole model"; only its GB, if known)
      if (!solo) lines.push(`<span>${share >= 0.995 ? "all of the model" : `holds ${shareWords(share)} of the model`}</span>` + (gb != null ? `<span>${fmtGB(gb)}</span>` : ""));
      else if (gb != null) lines.push(`<span>${fmtGB(gb)}</span>`);
      tip.push(`layers ${humanRange(r)} (${Math.round(100 * share)}% of the model)`);
    } else lines.push(`<span class="dim">${why === "late" ? "joined after the start: asks until the model is shared out again" : meta.api ? "asks through this room" : meta.webgpu === false ? "this browser can't hold part of the model: it asks" : "the others hold the model: it asks"}</span>`);
    // experts (MoE): where they are, the hit rate and the cost per token, in the tooltip; on the card only that a tab
    // keeps part of its share on its disk
    const xe = r ? x?.exp ?? (ai.role === "host" ? expertsAt(ai.moe, ai.offloadBy?.[nm]) : null) : null;
    const head = expertsHead(xe, x?.xc), ex = expertsText(xe), cost = expertsCost(x?.xc);
    for (const t of [head, ex, head && cost]) if (t) tip.push(t);
    if (xe && /disk/.test(xe.where || "")) lines.push(`<span class="dim">keeps part of its share on this computer's disk</span>`);
    let time = "";
    if (part?.timed) {
      time = `<div class="dv-bar"><i style="width:${part.pct.toFixed(1)}%"></i></div><div class="dv-t"><span>${Math.round(part.pct)}% of the work</span></div>`;
      tip.push(`${Math.round(part.ms)} ms of each ${Math.round(shares.lap)} ms token`);
    }
    return `<li class="dv${r ? " holds" : " ask"}" data-name="${esc(nm)}" style="--sw:${devColor(nm)}"${tip.length ? ` title="${esc(tip.join("\n"))}"` : ""}>`
      + `<div class="dv-h"><i class="sw" aria-hidden="true"></i>${iconFor(meta)}<b>${esc(nm)}</b></div>`
      + `<div class="dv-role">${esc(role)}${nm === myName ? " · you" : ""}</div>`
      + lines.map((l) => `<div class="dv-l">${l}</div>`).join("") + time + `</li>`;
  }).join("");
  // one token's time, end to end: each device's part in its colour, then the wire (and any device without timing)
  // (developer view only, ?dev=1: one token's milliseconds round the room)
  const sh = $("sp-share");
  if (shares && DEV) {
    sh.hidden = false;
    sh.innerHTML = `<div class="sh-cap">one token, ${Math.round(shares.lap)} ms round the room</div><div class="sh-bar">`
      + shares.parts.filter((p) => p.timed).map((p) => `<i style="flex-grow:${p.ms.toFixed(2)};--sw:${devColor(p.name)}" title="${esc(p.name)}: ${Math.round(p.ms)} ms"></i>`).join("")
      + (shares.rest.ms > 0 ? `<i class="wire" style="flex-grow:${shares.rest.ms.toFixed(2)}" title="${esc(shares.rest.label)}: ${Math.round(shares.rest.ms)} ms"></i>` : "")
      + `</div><div class="sh-key">` + shares.parts.filter((p) => p.timed).map((p) => `<span style="--sw:${devColor(p.name)}"><i></i>${esc(p.name)} ${Math.round(p.pct)}%</span>`).join("")
      + (shares.rest.ms > 0 ? `<span class="wire"><i></i>${esc(shares.rest.label)} ${Math.round(shares.rest.pct)}%</span>` : "") + `</div>`;
  } else { sh.hidden = true; sh.innerHTML = ""; }
}
// the deal changed (who holds layers, who asks): the cards, an event for a device that asks only, and the
// band itself on a screen that has the deal but no map yet (a guest that joined a room already online)
function speedDeal() {
  for (const [nm, why] of Object.entries(ai.outWhy || {})) {
    if ((why === "unneeded" || why === "small") && !speed.asked.has(nm)) { speed.asked.add(nm); speedEvent(`${nm} is only asking questions`); }
  }
  if (!lastMap && ai.layersByName && $("ai-panel").classList.contains("online")) {
    renderMap(Object.keys(ai.layersByName).map((nm, i) => ({ name: nm, layers: ai.layersByName[nm], host: i === 0 ? 1 : 0 })), null, false);
    return;
  }
  renderSpeed();
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
//
// A device in the chain that drops mid-answer does not fail it: roomRecover waits for the device to
// come back into its slot (or re-deals without it, experimental) and the answer carries on from the
// last emitted token (room/resume.js). -> the result also says how many times it `resumed`.
async function roomGenerate(ids, opts = {}) {
  const kind = ai.busy;   // "gen" (chat) or "code": the lock the caller holds, kept through a recovery
  const aborted = () => ai.abort || !!opts.signal?.aborted;
  const status = opts.onStatus || (() => {});
  try {
    return await resumableGenerate(roomGenerateOnce, ids, { maxNew: MAX_NEW, ...opts }, {
      aborted,
      recover: ({ err }) => roomRecover(err, kind, status, aborted),
      onResume: ({ emitted }) => {
        const s = emitted ? `the room is whole again: carrying on after ${emitted} token${emitted === 1 ? "" : "s"}` : "the room is whole again: starting over";
        status(s); aiStatus(s); log("room", s); toast(emitted ? "Everyone's back, carrying on" : "Everyone's back, starting the answer again");
      },
    }).then((r) => (r.resumed ? { ...r, stats: r.stats + ` · carried on after ${r.resumed > 1 ? r.resumed + " drops" : "a device dropped"}` } : r));
  } finally {
    if (ai.recovering?.pending) ai.recovering = null;
  }
}
// Wait until the room can generate again after a chain device dropped: it comes back into its slot
// (aiRejoin, then its ai-ready), or after REJOIN_GRACE_MS the layers are re-dealt over the devices
// still here (when the host's "re-deal on its own" is on). The caller's lock (ai.busy) is held
// throughout, so no queued question or Code request slips in between.
async function roomRecover(err, kind, status, aborted) {
  if (ai.role !== "host") throw err;
  ai.recovering = { kind, since: Date.now() };
  const why = err?.message || "a device left";
  log("room", `the ${kind === "code" ? "Code run" : "answer"} is waiting: ${why}`);
  toast(`Waiting for everyone to be back, then this ${kind === "code" ? "Code run" : "answer"} carries on`);
  const say = (s) => { status(s); aiStatus(s); };
  try {
    await waitForRoom({
      ready: () => !!ai.engine && !ai.degraded && !ai.loadingShard && ai.readyPeers.size >= ai.chain.length && ai.chain.every((id) => conns.has(id)) && !relinking(),
      gone: missingNames,
      redeal: async () => {
        log("room", `${missingNames().join(", ")} did not come back in ${Math.round(REJOIN_GRACE_MS / 1000)} s: re-dealing the layers (experimental auto re-deal)`);
        await aiRedeal(true);
        if (!ai.engine) throw new Error("the automatic re-deal failed: " + ($("ai-status").textContent || "no model"));
      },
      autoRedeal: autoRedealOn,
      status: say,
      aborted,
    });
  } finally {
    ai.recovering = null;
    // the lock stays with the run (aiStart and aiMaybeReady may have touched it during a re-deal)
    ai.busy = kind;
    if (kind === "code") lockKind = kind;
    setBusyUI(true, true);
  }
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
    if (askerId === peer.id) toast("A device left. Share out the model again first.");
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
    await linksUp();   // a link in the chain is being replaced: frames sent now would be lost
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
  if (ai.recovering?.pending) ai.recovering = null;
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
  if (lc.phase === "ready") aiLoading(false);   // New chat: the ready card goes too
  $("ai-output").innerHTML = "";
  botEl = null;
  setCtx(0);
}

// GPU wake on a worker (room/gpuwake.js): spin until the next frame arrives (aiOnData stops it)
function gpuWake() {
  if (!ai.engine || !ai.device || !myMeta.wake) return;
  try { (ai.waker ||= new GpuWaker(ai.device)).wake(WAKE_MAX_MS); } catch { myMeta.wake = 0; }
}
// ?wake=keep: also spin from the moment this worker has sent frame `p` on, unless the next one is already here
function keepWarm(p) { if (WAKE === "keep" && ai.lastFramePos === p) gpuWake(); }

// the host's tab closed: the room is over for everyone else
// A host tab that reloads can resume the room (it keeps the conversation in localStorage), so the
// others wait a minute and keep knocking before calling the room over.
const HOST_WAIT_MS = 60000;
function hostGone() {
  if (ai.role === "host" || admission === "out") return;
  failWaiters(new Error("the host left"));
  codeRoleChanged();
  $("ai-row").style.display = "none";
  // this device was the one away (screen locked, Safari in the background) if it came back just now
  const wasAway = performance.now() - visibleSince < 20000;
  $("room-over").hidden = false;
  $("room-over-h").textContent = wasAway ? "Reconnecting" : "Host reconnecting";
  $("room-over-why").textContent = wasAway ? "This screen was locked or in the background, so the room lost its link to this device. Reconnecting; its layers come back into their slot…" : "The host's tab closed. Waiting a minute in case it comes back…";
  const was = $("ai-status").textContent;
  aiStatus(wasAway ? "reconnecting to the room…" : "the host left; waiting for it to come back…");
  const t0 = Date.now();
  clearInterval(hostGone.timer);
  hostGone.timer = setInterval(() => {
    if (conns.has(PREFIX + roomCode)) { clearInterval(hostGone.timer); return; }
    if (Date.now() - t0 > HOST_WAIT_MS) {
      clearInterval(hostGone.timer);
      closeEngine();
      $("room-over-h").textContent = "Room over";
      $("room-over-why").textContent = "The host didn't come back. The host holds the conversation and the model's first and last layers, so this room can't answer any more.";
      aiStatus("the host left; this room is over");
      mascot("The host left. Start a new room?");
      return;
    }
    signalBack();   // a new link needs the signaling server
    if (peer.disconnected) return;
    const conn = peer.connect(PREFIX + roomCode, { reliable: true });
    conn.on("open", () => {
      if (conns.has(PREFIX + roomCode)) { try { conn.close(); } catch {} return; }
      clearInterval(hostGone.timer);
      linkOpened(conn, { name: "host", extra: { back: 1 } });
      ai.hostId = PREFIX + roomCode;
      $("room-over").hidden = true;
      // layers to deal only if a model was running; otherwise the card goes back to what it said
      aiStatus(wasAway ? (ai.engine ? "back in the room; rejoining the chain…" : "back in the room") : ai.role ? "the host is back; waiting for it to deal the layers…" : was);
      toast(wasAway ? "back in the room" : "the host is back");
      saveGuest();
    });
    conn.on("error", () => {});
  }, 3000);
}

// ---- the host's side of resuming: what it keeps, and picking the room back up after a reload ----
function saveHost() {
  if (!isHost || !roomCode) return;   // from the moment the room exists, not only once a model runs
  try {
    localStorage.setItem(HOST_KEY, JSON.stringify({ code: roomCode, name: myName, signal: signalServer?.spec || null, model: ai.model || null, turns: ai.conv.turns,
      transcript: ai.transcript.filter((t) => !t.api).slice(-20), settings: ai.settings, peers: ai.chainNames || [], split: $("ai-split").value, ckptN: ai.ckptN || 0,
      gate: gate ? saveGate(gate) : null, seqs: [...seqOf], selfHosted: ai.role === "host", t: Date.now() }));
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
  // the model ran on another device: this creator doesn't start it on itself (the room's pick decides: press Start)
  if (r.selfHosted === false) { aiStatus("resumed: the model ran on another device; press Start to start it again"); return; }
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
  "ai-visibility", "ai-style", "ai-busy", "ai-wait", "ai-start-failed", "ai-share", "ai-wake", "ai-out"]);
// a host showing it deals (room/authz.js): this device follows it from now on, busy for any other Start
const HOST_SIGNS = new Set(["ai-layers", "ai-hostprog", "ai-ready-all", "ai-redeal", "ai-map"]);
function hostDealing(from) {
  if (!ai.dealer || !dealerHere()) ai.dealer = from;
  if (from !== ai.dealer) return;
  ai.dealerActive = true;
  clearTimeout(ai.startWait); ai.startWait = null;
}
async function aiOnData(from, d) {
  if (d.t.startsWith("ai-code") || d.t.startsWith("ai-pv")) { codeOnData(from, d); return; }
  const e = conns.get(from);
  if (FROM_HOST.has(d.t)) {
    if (ai.role === "host") return;
    // the host this device follows while it is in the room; before it follows one, the creator or the host the
    // creator named in ai-modelhost (room/authz.js)
    if (!hostAllowed({ from, dealer: ai.dealer, dealerHere: dealerHere(), named: ai.hostId || PREFIX + roomCode }).take) return;
    if (HOST_SIGNS.has(d.t)) hostDealing(from);
  }
  if (d.t === "ai-load" && ai.role === "host") return;
  if (d.t === "ai-load") {
    const g = hostAllowed({ from, host: d.host, dealer: ai.dealer, dealerHere: dealerHere(), named: ai.hostId || PREFIX + roomCode });
    // refused out loud: the host's start fails at once instead of waiting on this device
    if (!g.take) { log("room", `refused layers from ${e?.name || from}: ${g.why}`); sendTo(from, { t: "ai-error", load: 1, message: `${myName} refused the layers: ${g.why}` }); return; }
    hostDealing(from);
  }
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
      // every device checks the pick itself, with the creator's facts (room/authz.js); the creator relays a request
      // it checked to the devices the requester has no link to yet (rq: who pressed Start, for their answers)
      {
        if (!MODELS[d.model]) break;
        // only the creator's (its own, or one it checked and relays): a member's copy sent here directly is ignored
        if (!startFrom({ from, creator: PREFIX + roomCode, isCreator: isHost })) break;
        const busy = startBusy({ boss: d.boss, self: peer.id, dealer: ai.dealer, dealerHere: dealerHere(), dealing: ai.dealerActive, running: startRunning(), pendingSince: ai.startAt, now: performance.now() });
        if (busy === "replay") break;   // the host this device follows already: nothing to do
        const r = busy ? { ok: false, why: busy } : checkHost(d.boss, d.ids, from);
        if (!r.ok) { if (busy) log("room", `ignored ${d.by || "a device"}'s Start: ${busy}`); sendTo(from, { t: "ai-start-refused", by: myName, why: r.why, ...(d.rq ? { rq: d.rq } : {}) }); break; }
      }
      if (isHost) {
        const to = [...conns.keys()].filter((id) => id !== from && !conns.get(id)?.meta?.api);
        ai.relay = { rq: from, boss: d.boss, to: new Set(to) };
        for (const id of to) sendTo(id, { ...d, rq: from });
      }
      ai.dealer = d.boss; ai.dealerFresh = true; ai.dealerActive = false; ai.startAt = performance.now();
      if (d.boss !== peer.id) followWait(START_PENDING_MS);
      setModelValue(d.model);   // every screen shows the model that was actually started
      // the device dealing the layers runs the room: listen to it, even before this device has a link to
      // it (a model host that is not the room's creator links every device before it deals)
      if (d.boss !== peer.id) ai.hostId = d.boss;
      $("ai-start").disabled = true; $("ai-model").disabled = true;
      if (d.boss !== peer.id) { aiLoading(true, `starting ${MODELS[d.model]?.label.split("·")[0].trim()}`, d.model); lc.host = conns.get(d.boss)?.name || members.get(d.boss)?.name || String(d.by || "The host").slice(0, 40); lcPaint(true); }
      else if (!ai.engine && !ai.busy) aiLoading(true, `starting ${MODELS[d.model]?.label.split("·")[0].trim()}`, d.model);
      if (d.boss === peer.id) { toast(`${d.by} started ${MODELS[d.model]?.label.split("·")[0].trim()}`); aiStart(d.model); }
      else aiStatus(`${d.by} started the model…`);
      break;
    case "ai-start-refused": {   // a device didn't follow a Start (room/authz.js)
      // the creator: an answer to a Start it relayed goes to whoever pressed it
      // (named by the link it came on, never by its own text; boss: 1 when the device that would run the model refused)
      if (isHost && d.rq && ai.relay?.rq === d.rq && ai.relay.to.has(from) && conns.has(d.rq)) {
        sendTo(d.rq, { t: "ai-start-refused", fwd: 1, boss: from === ai.relay.boss ? 1 : 0, by: conns.get(from)?.name || "a device", why: String(d.why || "").slice(0, 120) });
        break;
      }
      // a Start this device waits on: only the creator answers. It gives up when the creator's own check refused it
      // or the device that would run it did; another device's refusal is a note (the timer still runs)
      if (ai.startWait && from === PREFIX + roomCode) {
        const why = `${String(d.by || "a device").slice(0, 40)}: ${String(d.why || "refused").slice(0, 120)}`;
        if (!d.fwd || d.boss) startRefused(why); else log("room", `${why} (it doesn't follow this Start)`);
      }
      break;
    }
    case "ai-modelhost":   // the room's creator: the model runs on another device, which links to this one
      if (isHost || from !== PREFIX + roomCode || ai.role === "host" || ai.role === "worker" || typeof d.id !== "string") break;
      ai.hostId = d.id;
      // the creator checked it on a Start it saw: follow it (busy for a Start naming another device) until it shows it
      // deals, or for START_PENDING_MS
      if (!dealerHere()) { ai.dealer = d.id; ai.dealerActive = false; ai.startAt = performance.now(); followWait(START_PENDING_MS); }
      break;
    case "ai-next":
      // relink: the device after this one came back under the same id; the old link to it is dead
      if (d.relink && conns.has(d.next)) dropLink(d.next, "came back: opening a fresh link");
      ai.next = d.next;
      ensureLink(d.next).then((ok) => { if (d.relink) sendTo(ai.hostId || from, { t: "ai-linked", next: d.next, ok }); });
      break;
    case "ai-linked":   // worker -> host: its fresh link to a device that came back is up
      if (ai.role !== "host" || !ai.chain.includes(from)) break;
      ai.relinks.delete(d.next);
      aiMaybeReady();
      break;
    case "ai-layers":
      ai.layersByName = d.by; lc.dealt = true; lc.host = null; lcPaint(true);
      ai.heldGB = d.held && typeof d.held === "object" ? d.held : null;
      ai.outWhy = d.out && typeof d.out === "object" ? d.out : {};
      paintHeld();
      if (ai.role === "worker" && !d.by[myName]) {   // not in this deal: ask-only guest, GPU memory freed
        closeEngine();
        ai.role = "guest"; ai.range = null; ai.held = null;
        try { ai.device?.destroy(); } catch {}
        ai.device = null;
      }
      break;
    case "ai-out":   // the host: why devices hold no layers (a newcomer waits for a re-deal)
      ai.outWhy = d.out && typeof d.out === "object" ? d.out : {};
      paintHeld();
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
      lc.adding = Array.isArray(d.adding) ? d.adding.filter((x) => typeof x === "string").slice(0, 16).map((x) => x.slice(0, 40)) : [];
      lc.guess = Number.isFinite(+d.guess) && +d.guess > 0 ? Math.min(+d.guess, 3 * 3600) : null; lcPaint(true);
      $("ldg-sub").textContent = `${d.by} is re-dealing the layers over the devices in the room`;
      aiStatus(`${d.by} is re-dealing the layers…`);
      break;
    case "ai-share":   // the host lowered this device's share (or left it out) after its tab was killed while loading
      if (!d.drop && d.gb > 0) {
        myMeta.contribGB = Math.max(0.1, Math.min(myMeta.contribGB || d.gb, d.gb));
        const selfCard = document.querySelector(".peer-card.self");
        if (selfCard) { setLends(selfCard, myMeta.contribGB); selfSteps(selfCard); }
        updateCluster();
      }
      log("room", d.why); toast(d.why); aiStatus(d.why);
      break;
    case "ai-linklost": {   // a worker's link to another device in the chain dropped (up: 0, it is being replaced) or is back (up: 1)
      if (ai.role !== "host" || !ai.chain.includes(from)) break;
      const nm = String(d.name || "").replace(/[\u0000-\u001f\u007f<>"'`&]/g, "").slice(0, 40);
      if (d.up || !ai.chainNames?.includes(nm)) noteLink(from + "|" + nm, nm, true);
      else { noteLink(from + "|" + nm, nm, false); chainLinkLost(from, nm); }
      break;
    }
    case "ai-degraded":
      aiStatus(`${d.why} — waiting for the host to re-deal the layers`);
      toast(d.why);
      break;
    case "ai-load": {
      // the host re-seats this device while its load of the same layers still runs (it dropped this tab
      // and it reconnected): keep that load, it reports ai-ready to the host when it is in
      const loadKey = shardKey(d.model, d.range, d.flash);
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
      // back in my slot after a lock or a lost link, with the same layers still on the GPU: no reload
      const keep = ai.engine && sameShard(ai.held, d) && ai.held.kv === kvForLoad(d.model, d.kv, KV_ASK);
      try {
        if (keep) {
          ai.q = Promise.resolve(); ai.loadKey = loadKey;
          try { ai.engine.reset?.(); ai.engine.dropAllSlots?.(); } catch {}
          log("room", `back in the room: layers ${d.range[0]}–${d.range[1] - 1} are still loaded, no download`);
        } else {
          ai.loadingShard = true; ai.loadKey = loadKey;
          // d.kv: the host's KV format (a host without it: this device's own ?kv=, as before)
          try { await aiLoadShard(d.model || "smollm-135m", d.range, false, false, d.ctx || maxSeqFor(d.model), kvForLoad(d.model, d.kv, KV_ASK), d.offload || null, d.flash || null); } finally { ai.loadingShard = false; }
          if (ai.startFailed) throw new Error(ai.startFailed);
          ai.held = { model: d.model, range: [d.range[0], d.range[1]], ctx: d.ctx, kv: kvForLoad(d.model, d.kv, KV_ASK), offload: d.offload || null, flash: d.flash || null };
        }
        keepAwake();
        if (!(await ensureLink(d.next))) throw new Error("could not connect to the next device in the chain");
        const slots = await ckptRestore();   // this device's part of the room's checkpoints, if it saved any before a reload
        aiStatus(`layers ${d.range[0]}–${d.range[1] - 1} ready · syncing with the room…`);
        $("ldg-title").textContent = `layers ${d.range[0]}–${d.range[1] - 1} ready`;   // the card stays up as it is (Starting) until ai-ready-all
        $("ldg-sub").textContent = "syncing with the rest of the room";
        $("ldg-fill").style.width = "100%";
        // slots: the host forgets the checkpoints not in them; ckpt: this device applies sv / ld / dp
        // (a room node hosting a dense model checks it: tabs from before the dense engine had slots ignore them)
        // ctx, fsig (Flash): the positions this engine holds and its state signature; batch (Qwen3.8-Flash-Next): whether
        // this device runs prompt frames in batched passes (room/flash.js); a host narrows its frames for a device that
        // does not (room/pipeline.js chainSlow). Older hosts ignore them
        sendTo(ai.hostId, { t: "ai-ready", slots, ckpt: ai.engine?.saveSlot ? 1 : 0, ...flashReadyFields(ai.engine, d.model), ...(ai.engine?.batch !== undefined ? { batch: ai.engine.batch ? 1 : 0 } : {}) });
      } catch (err) {
        if (onLoadError(ai.startFailed) === "stopped") { workerStopped(); break; }   // the host stopped the start (ai-start-failed): this load stopped with it
        closeEngine();
        if (err.noDisk) {
          // not a failure to report: this device kept the model off its disk, and the host deals again without it
          const why = "this device keeps the model off its disk: the host is dealing the layers again without it";
          aiStatus(why); toast(why); log("room", `${myName}: ${err.message}`);
          $("ldg-sub").textContent = "the host is dealing the layers again without this device's disk";
        } else {
          lcStop(err.message, err.message === GPU_CHECK_FAILED);   // the card stays where it got to; the host's ai-start-failed then names the device
          aiStatus("failed: " + err.message);
          if (err.detail) log("room", `${myName}: ${err.detail}`);
          toast(err.message === GPU_CHECK_FAILED ? err.message : `The model stopped: this device couldn't load its part (${err.message})`);
          $("ai-start").disabled = false; $("ai-model").disabled = false; updateCluster();
        }
        // load: the host stops the start; noDisk: this device refused to store the model on its disk (its user said no,
        // or it never offered it): the host deals again without its disk
        sendTo(ai.hostId, { t: "ai-error", message: err.message === GPU_CHECK_FAILED ? gpuCheckFailedFor(err) : err.message, load: 1, ...(err.noDisk ? { nodisk: 1 } : {}) });
      }
      break;
    }
    case "ai-start-failed":   // the host stopped the start: back to the picker, and a load still running here stops (the pacer)
      ai.dealer = null; ai.dealerFresh = false; ai.dealerActive = false; ai.startAt = null; clearTimeout(ai.startWait); ai.startWait = null;
      ai.heldGB = null; ai.outWhy = {}; paintHeld();
      ai.startFailed = d.why || "the start was stopped";
      if (freeOnStartFailed(ai.loadingShard)) workerStopped();   // (a load in flight stops at its next tensor, and its catch does this)
      // the card stays where the load got to, with what happened (lcStop); a device that had no card: the picker
      if ($("load-card").classList.contains("on")) lcStop(ai.startFailed); else aiLoading(false);
      $("ai-panel").classList.remove("online");
      $("ai-row").style.display = "none";
      $("ai-start").disabled = false; $("ai-model").disabled = false; updateCluster();
      if (!/^failed:/.test($("ai-status").textContent)) toast(`The model stopped: ${d.why}`);
      aiStatus("the model didn't start: " + d.why);
      break;
    case "ai-hostprog": {
      const now = Date.now();
      ai.prog = { ...(d.all || {}), [myName]: Math.round(ai.myPct || 0) };
      // each device's step and GPU percent (room/loadsteps.js; none from an older host)
      ai.step = { ...(d.steps && typeof d.steps === "object" ? d.steps : {}), [myName]: (ai.step || {})[myName] ?? null };
      ai.gpuProg = { ...(d.gpu && typeof d.gpu === "object" ? d.gpu : {}), [myName]: Math.round(ai.gpuPct || 0) };
      ai.progAt = ai.progAt || {};
      for (const nm of Object.keys(d.all || {})) if (nm !== myName) ai.progAt[nm] = now;
      // each device's parts and time left for the card (none from an older host): this device's own stay its own
      const pw = {}, lb = {};
      if (d.parts && typeof d.parts === "object") for (const [nm, w] of Object.entries(d.parts)) { const v = partsWireIn(w); if (v && nm !== myName) pw[nm] = v; }
      if (d.left && typeof d.left === "object") for (const [nm, v] of Object.entries(d.left)) if (nm !== myName && (v == null || Number.isFinite(+v))) lb[nm] = v == null ? null : +v;
      ai.partsW = { ...pw, ...(ai.partsW?.[myName] ? { [myName]: ai.partsW[myName] } : {}) };
      ai.leftBy = { ...lb, [myName]: lc.left };
      ai.askBy = {};
      if (d.ask && typeof d.ask === "object") for (const [nm, v] of Object.entries(d.ask)) if (nm !== myName && v) ai.askBy[nm] = true;
      if (lc.host && Object.keys(pw).length) lc.host = null;   // the host shows it deals: the card shows the devices
      lcPaint();
      break;
    }
    case "ai-progress":
    {
      // d.step / d.gpu: the step it is on and that step's GPU percent (room/loadsteps.js); an older peer sends neither
      const step = /^(save|words|gpu)$/.test(d.step || "") ? d.step : null, nm = e?.name || from;
      if (e?.card) { e.card.querySelector(".bw").textContent = `${rowVerb(step)} ${d.pct}%`; peerStatus(e.card, `${rowVerb(step)} ${d.pct}%`); }
      ai.prog = ai.prog || {}; ai.progAt = ai.progAt || {}; ai.step = ai.step || {}; ai.gpuProg = ai.gpuProg || {};
      ai.prog[nm] = d.pct; ai.progAt[nm] = Date.now(); ai.step[nm] = step;
      if (Number.isFinite(+d.gpu)) ai.gpuProg[nm] = +d.gpu; else delete ai.gpuProg[nm];
      // its parts and time left for the load card (room/loadsteps.js; an older peer sends neither: its pct stands in)
      ai.partsW = ai.partsW || {}; ai.leftBy = ai.leftBy || {};
      const pw = partsWireIn(d.parts);
      if (pw) ai.partsW[nm] = pw; else delete ai.partsW[nm];
      ai.leftBy[nm] = d.left == null || !Number.isFinite(+d.left) ? null : +d.left;
      if (pw?.db) rateSeen(nm, pw.db[0]);
      // a peer that sends only its percent (a room node, `pooled join`): its time left from that percent of what it was
      // dealt (ai.heldGB), so the card and the re-deal's slim line have a real estimate for it too
      const held = +ai.heldGB?.[nm] * 2 ** 30, pct = +d.pct;
      if (!pw && ai.leftBy[nm] == null && held > 0 && Number.isFinite(pct)) {
        const done = Math.max(0, Math.min(100, pct)) / 100 * held;
        ai.leftBy[nm] = (lc.peerEta[nm] ||= etaTracker()).push(performance.now(), [{ id: "gpu", done, total: held, state: pct >= 100 ? "done" : "run" }]);
        rateSeen(nm, done);
      }
      ai.askBy = ai.askBy || {}; ai.askBy[nm] = !!d.asking;   // its user is being asked about its disk (lcAsk)
      lcPaint(!!d.asking);
    }
      break;
    case "ai-ready": {
      if (ai.role !== "host" || !ai.chain.includes(from)) break;
      // a worker whose engine holds fewer positions than the room's context (an older build loads Flash at 2048)
      const short = readyCtxShort(ai.roomModel, ai.roomCtx, d, ai.roomFlash);
      if (short) { log("room", `${e?.name || from}: ${short}`); aiLoadFailed(from, e?.name || from, short); break; }
      ai.readyPeers.add(from);
      (ai.ckptHeld ||= new Map()).set(from, d.slots);   // what it read back from disk (none from an older build)
      // a Flash worker without batched passes (batch: 0, or a build before them that does not say): narrow prompt frames
      if (MODELS[ai.model]?.kind === "qwen4exp") (ai.batchOf ||= new Map()).set(from, d.batch === 1);
      if (e?.card) { e.card.querySelector(".bw").textContent = "ready"; peerStatus(e.card, "ready", true); }
      aiMaybeReady();
      break;
    }
    case "ai-error":
      aiStatus(`peer ${e?.name || from} failed: ${d.message}`);
      // a worker refused the offload it was dealt: a tab keeping the model off its disk (nodisk), or a room node from
      // before Flash offload (@pooled/cli 0.4.1's shard.js: "no expert offload yet"). Deal again once without its
      // offload (aiStart's metasNow), not a failed start; a second refusal from it stops the start as any error does
      const oldNode = /has no expert offload yet/.test(String(d.message || ""));
      if ((d.nodisk || oldNode) && ai.role === "host" && ai.chain.includes(from) && !ai.noDisk?.has(e?.name || from)) {
        const nm = e?.name || from;
        (ai.noDisk ||= new Set()).add(nm);
        const why = d.nodisk ? `${nm} keeps the model off its disk` : `${nm} can't offload this model's experts (an older pooled)`;
        log("room", `${why}: dealing the layers again without its offload`);
        failWaiters(new Error(`${nm}: ${d.message}`));
        aiAutoRedeal(`${why}: dealing the layers again…`);
        break;
      }
      if (ai.role === "host" && ai.chain.includes(from)) {
        failWaiters(new Error(`${e?.name || from}: ${d.message}`));
        // a device could not load its layers while the room was starting: the start cannot finish
        if (stopsStart({ load: d.load, inChain: true, starting: ai.busy === true, online: $("ai-panel").classList.contains("online") })) aiLoadFailed(from, e?.name || from, d.message);
      }
      break;
    case "ai-tele": if (ai.role === "host") { ai.teleBy.set(from, { ...(d.k || {}), amax: +d.amax || 0, xc: d.exp ? expertCache(d.exp, ai.teleBy.get(from)?.xc) : null }); ai.teleLog?.push({ k: d.k, exp: d.exp }); } break;
    case "ai-inv-req": cachedRanges(d.url).then((have) => sendTo(from, { t: "ai-inv", url: d.url, have })); break;
    case "ai-inv": if (ai.invWait && ai.invWait.url === d.url && Array.isArray(d.have)) ai.invWait.inv[from] = d.have.slice(0, 20000); break;
    case "ai-wget": answerWget(from, d); break;
    case "ai-wpart": onWeightPart(d); break;
    case "ai-wack": onWeightAck(d); break;
    case "ai-map": renderMap(d.nodes, d.st, d.live); break;
    case "ai-wake":
      // the frame of this lap may have overtaken its wake (they ride different channels)
      if (ai.role === "worker" && d.pos > (ai.lastFramePos ?? -1)) gpuWake();
      break;
    case "ai-hidden-b":
    case "ai-hidden":
    case "ai-hiddenret-b":
    case "ai-hiddenret": pipeline.handleFrame(from, d); break;
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
    case "ai-ready-all": {
      const wasOnline = $("ai-panel").classList.contains("online");   // (a device back on a dropped link hears it again)
      // the load card says "ready" in place until the first question; a device holding no layers (left out, a guest)
      // goes straight to the chat, where the note says why it holds none
      if ($("load-card").classList.contains("on") && !wasOnline && ai.range && lc.phase !== "stopped") lcReady(); else aiLoading(false);
      if (d.out && typeof d.out === "object") { ai.outWhy = d.out; paintHeld(); }
      $("ai-panel").classList.add("online");
      if (!wasOnline) speedReady();   // speedReady (C) (the host sends ai-ready-all again on every hello)
      if (ai.role !== "host" && ai.role !== "worker") ai.role = "guest";
      if (ai.role !== "host") ai.hostId = from;
      if (MODELS[d.model]) { setModelValue(d.model); ai.model = d.model; }
      promptHint();
      // the host opened the model at its fallback context: the room's memory is short for the default
      ai.ctxNote = d.ctxWant > d.ctx && MODELS[d.model] ? ctxShortNote(shortName(d.model), d.ctx, d.ctxWant) : "";
      ai.ctxK = d.ctx > 0 ? ctxK(d.ctx) : "";
      $("ai-row").style.display = "flex";
      $("chat-tools").hidden = false;
      $("mode-bar").hidden = false;   // Chat | Code for every device, not only the host (a phone guest had no way to Code)
      if (lc.phase !== "ready") emptyText("The model is ready. Ask anything.");
      if (!wasOnline) { sysNote("Model ready"); if (ai.ctxNote) sysNote(ai.ctxNote); }
      // left out of the deal: say why, so a device with no layers doesn't look broken
      if (!ai.range && ai.outWhy?.[myName]) { const t = outNote(ai.outWhy[myName]); emptyText(t); if (!wasOnline) sysNote(t); }
      aiStatus(ai.range ? `cluster online · serving layers ${ai.range[0]}–${ai.range[1] - 1}` : "cluster online · this device asks, the others think");
      mascot("Cluster online! Type a question, the whole room answers.");
      codeRoleChanged();
      break;
    }
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
  if (ai.recovering?.pending) ai.recovering = null;   // the run ended before it needed the room again
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
  // Code mode on the serve v2 core (harness/core-model.js): the same template profile and token
  // texts the API path uses (one mask cache for both), and which model the cached ids belong to
  profile: () => (ai.tok ? apiProfile() : null),
  tokenTexts: () => apiTokenTexts(),
  modelKey: () => ai.model || "",
  recovering: () => !!ai.recovering,   // a device dropped mid-run: the run waits for it (or a re-deal) and carries on
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
// ?dev=1 test hook (packages/room-node/test/flash_tab_gold.mjs): the model host generates from token ids through the
// room (prefill, decode, every device's layers), greedy, no stop token, and fingerprints every step's logits (FNV-1a
// over their bits: two runs with the same fingerprint computed the same logits, offloaded or resident). Also the
// expert store's stats on this device. -> { tokens, tps, tPre, bits, experts }
if (DEV) window.__pooledGen = async (ids, maxNew) => {
  if (ai.role !== "host" || !ai.engine) throw new Error("not the model host, or no model loaded");
  if (ai.busy) throw new Error("the room is busy");
  ai.busy = "gen";
  let x = 0x811c9dc5;
  const sample = (lg) => {
    const u = new Uint32Array(lg.buffer, lg.byteOffset, lg.length);
    let best = 0;
    for (let i = 0; i < lg.length; i++) { x = Math.imul(x ^ u[i], 0x01000193) >>> 0; if (lg[i] > lg[best]) best = i; }
    return best;
  };
  const st = ai.engine.experts, s0 = st ? { ...st.stats, t1: st.t1 ? { ...st.t1.stats } : null } : null;
  ai.lapLog = []; ai.teleLog = [];   // every lap (lap ms, host ms) and every worker report of this run
  try {
    const r = await roomGenerate(ids, { maxNew, sample, stop: new Set() });
    const laps = ai.lapLog, tele = ai.teleLog;
    const d = st ? Object.fromEntries(["cuts", "lookups", "hits", "misses", "diskReads", "diskBytes", "diskWaitMs", "t1Copies"].map((k) => [k, st.stats[k] - s0[k]])) : null;
    if (d && st.t1) d.t1Hits = st.t1.stats.hits - s0.t1.hits;
    return { tokens: r.tokens, tps: r.tps, tPre: r.tPre, bits: x.toString(16), experts: d, summary: st?.summary?.() || null,
      reused: r.reused, ckpt: { on: typeof ai.engine?.saveSlot === "function", slots: ai.ckpt?.items?.length ?? null }, laps, tele };
  } finally { ai.busy = false; ai.lapLog = ai.teleLog = null; }
};
if (DEV) window.__pooledInvite = () => (roomCode ? roomLink() : null);
// (?dev=1&stalls=MS: log this page's event-loop stalls past MS, for the Flash room's slow laps)
if (DEV && +SPLIT_Q.get("stalls") > 0) { let last = performance.now(); setInterval(() => { const now = performance.now(); if (now - last > 50 + +SPLIT_Q.get("stalls")) console.error(`event loop stalled ${(now - last - 50).toFixed(0)} ms at ${new Date().toISOString().slice(11, 23)}`); last = now; }, 50); }

$("ai-start").addEventListener("click", aiStartAnywhere);
$("ai-redeal").addEventListener("click", () => aiRedeal());
// "Not now" on a device about to be added on its own: it waits for the button instead
$("ai-redeal-skip").addEventListener("click", () => {
  for (const nm of ai.addPending || []) (ai.addSkip ??= new Set()).add(nm);
  clearTimeout(ai.addTimer); ai.addPending = [];
  offerRedealForNewcomers();
});
// adding devices that join while the model runs: on unless this device's host turned it off (remembered here)
try { if (localStorage.getItem("pooled-autoadd") === "off") $("ai-autoadd").checked = false; } catch {}
$("ai-autoadd").addEventListener("change", (e) => {
  try { localStorage.setItem("pooled-autoadd", e.target.checked ? "on" : "off"); } catch {}
  if (!e.target.checked) { clearTimeout(ai.addTimer); ai.addPending = []; }
  offerRedealForNewcomers();
});
// auto re-deal (experimental): the host's choice, remembered on this device
try { if (localStorage.getItem("pooled-autoredeal") === "off") $("ai-autoredeal").checked = false; } catch {}
$("ai-autoredeal").addEventListener("change", (e) => {
  try { localStorage.setItem("pooled-autoredeal", e.target.checked ? "on" : "off"); } catch {}
  if (!e.target.checked) clearTimeout(ai.idleRedeal);
  toast(e.target.checked ? "Shares out the model again by itself if a device doesn't come back (experimental)" : "Waits for a device that dropped, or for you to share out the model again");
});
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
  renderTabDisk(true);
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
// This device's disk for big models (room/flashdisk.js): the switch is its user's answer (remembered for this site), and
// "Delete saved copy" deletes the Flash copy (experts and n-gram table) with its size; two steps, like Clear above.
// measure: also read how much the copy takes (the menu opening; a walk of the OPFS files)
let tabDiskArmed = 0, tabDiskStored = 0;
const tabDiskFreeLabel = (t, sub) => { const a = $("tabdisk-free"); a.childNodes[1].textContent = t; a.querySelector("small").textContent = sub; };
async function renderTabDisk(measure = false) {
  const on = tabDiskCan;
  $("tabdisk-set").hidden = !on;
  if (!on) { $("tabdisk-line").hidden = true; return; }
  const choice = tabDiskNow();
  $("tabdisk-allow").checked = choice === "allow";
  // the size is the picker's (DISK_COPY diskBytes), and so are the answer's words (room/diskpick.js DISK_STATE)
  const fl = DISK_COPY["qwen3.8-flash"];
  $("d-tabdisk").textContent = `${fl.name} runs from disk: it saves up to ${fmtBytes(fl.diskBytes)} to this device once. `
    + (choice === "allow" ? "Allowed." : choice === "deny" ? "Not allowed." : "Not asked: you are asked when the model needs it.");
  if (measure) tabDiskStored = (await flashDiskBytes().catch(() => ({ total: 0 }))).total;
  $("tabdisk-line").hidden = !tabDiskStored;
  if (!tabDiskArmed) tabDiskFreeLabel("Delete saved copy", `Deletes ${fl.name}'s copy on this device (${fmtBytes(tabDiskStored)})`);
}
$("tabdisk-allow").addEventListener("change", (ev) => {
  saveTabDiskChoice(ev.target.checked ? "allow" : "deny", globalThis.localStorage);
  tabDiskDeclined = false;
  tabDiskChanged();
  toast(ev.target.checked ? "big models may use this device's disk" : "big models stay off this device's disk (Delete saved copy deletes what is there)");
});
$("tabdisk-free").addEventListener("click", async (ev) => {
  ev.preventDefault();
  if (!tabDiskArmed) {
    if (!tabDiskStored) { toast("no model copy on this device's disk"); return; }
    tabDiskFreeLabel(`Delete ${fmtBytes(tabDiskStored)}?`, "Press again; the next start that needs it copies it again");
    const t = tabDiskArmed = setTimeout(() => { if (tabDiskArmed === t) { tabDiskArmed = 0; renderTabDisk(); } }, 6000);
    return;
  }
  tabDiskArmed = 0;
  const r = await freeTabDisk();
  if (r.refused) toast(r.refused);
});
// Delete the Flash copy on this device (Room settings' "Delete saved copy", the picker's Delete), after
// both presses. -> { removed, kept } or { refused: why it did not run }
async function freeTabDisk() {
  // the running model reads the copy (open files), and a load may be copying into it or reading from it: stop first
  if (ai.loadingShard || ai.engine?.experts || (ai.model && TAB_DISK_MODELS.includes(ai.model) && ai.engine)) {
    renderTabDisk();
    return { refused: ai.loadingShard ? "this device is loading its layers: wait for it, or stop the model, then free the space" : "the model is using this copy: stop it (or leave the room) first" };
  }
  const r = await removeFlashCopy().catch(() => ({ removed: 0, kept: tabDiskStored }));
  toast(r.kept ? `freed ${fmtBytes(r.removed)}; ${fmtBytes(r.kept)} is in use and stays` : `freed ${fmtBytes(r.removed)} of this device's disk`);
  await renderTabDisk(true);
  refreshDiskPick();
  return r;
}
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
  // whatever throws in here, the room must not stay busy: the asker gets a failed gendone and the
  // lock, the Stop button and the queue are released
  let answered = false, toScreens = () => {};
  try {
    apiPanel();
    setBusyUI(true, true);
    setAfterAnswer(false, false);
    // the screens: the room's visibility applies as for any guest, except that the asker (the bridge)
    // gets its own full stream below, never a stand-in; under "asker" only the host's screen shows it
    const { full, hidden } = chatRecipients(ai.visibility || "all", from, [...conns.keys()]);
    toScreens = (msg) => {
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
    answered = true;
    ai.transcript.push({ name: label, text: shown, reply: raw, stats, mid, api: 1 });
    if (ai.transcript.length > 50) ai.transcript.shift();
    const c = ai.apis.get(from); if (c) c.answered++;
    apiPanel();
    setCtx(ctx.used, ctx.max);
    if (!res.err) aiStatus(`ready — API answer for ${name}: ${res.usage.in} prompt tok${res.reused ? ` (${res.reused} reused)` : ""}, ${res.stats}`);
  } catch (e) {
    const err = String(e?.message || e).slice(0, 300);
    log("api", `API answer failed: ${err}`);
    aiStatus("generation failed: " + err);
    if (!answered) sendTo(from, { t: "ai-gendone", rid, api: v2 ? 2 : 1, reason: "error", usage: { in: 0, out: 0 }, failed: 1, err });
    try { chatBotEnd("⚠ " + err, "failed · via API"); toScreens({ t: "ai-gendone", stats: "failed · via API", ctx: { used: ai.fed ? ai.pos : 0, max: ctxMax() }, failed: 1, capped: 0, api: 1 }); } catch {}
  } finally {
    ai.busy = false;
    ai.abort = false;
    if (ai.apiRun === run) ai.apiRun = null;
    setBusyUI(false);
    setTimeout(nextQueued, 0);
  }
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
// the host's command carries the invite link (its key lets the client in without asking); a guest's has
// the code, and the host is asked
function apiCommand() {
  const room = isHost && roomCode && gate ? `"${roomLink()}"` : formatCode(roomCode || "") || "CODE";
  return `npx @pooled/cli serve ${room}${SIGNAL ? ` --signal ${SIGNAL}` : ""}`;
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
  // while the load card loads, the question waits and is sent when the model is ready (lcReady)
  if (lcLoading()) { lc.queued = true; sendLabel(); return; }
  if (!$("ai-panel").classList.contains("online") || (ai.role === "host" && !ai.engine)) return;
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
      lc.dealt = true; lc.phase = "run"; lc.t0 = now - (early ? 6000 : 48000);
      lc.mine = lc.prog.push(partsFromProg({}, done, total)); lc.left = early ? null : 128;
      ai.partsW = {}; ai.leftBy = { [myName]: lc.left };
      for (const nm of names()) if (nm !== myName) ai.leftBy[nm] = early ? null : 128 + 30 * names().indexOf(nm);
      aiProgress(done, total);
      for (const card of document.querySelectorAll("#peers .peer-card")) peerStatus(card, `${ai.prog[card.dataset.name] ?? 0}%`);
      lcPaint(true);
    },
    ready() {
      if (!ai.layersByName) this.loading(1);
      const mine = /^(\d+)\D+(\d+)$/.exec(ai.layersByName[myName] || "");
      ai.range = mine ? [+mine[1], +mine[2] + 1] : null;
      aiLoading(false);
      $("ai-panel").classList.add("online");
      speedReady();   // speedReady (C)
      promptHint();
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
