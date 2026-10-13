// A load as parts that can run side by side, and the words the load card says about them (room.js lcPaint).
// Qwen3.8 Flash on a tab whose user allowed its disk has three parts: "save" (its offloaded layers' routed experts
// copied to this computer, room/flashdisk.js tabExpertTier), "words" (the n-gram table copied there,
// room/model-loader.js saveNgramEarly) and "gpu" (its layers onto the GPU). Today's loader runs them one after another
// and says so through onProgress(done, total, info) (partsFromProg); a loader that runs them at the same time says
// so through onParts(snapshot) (partsFromSnapshot, the contract in room/model-loader.js). Both come out as
// { parts, done, total } and the card draws the same thing from either.
// Pure (no DOM, no clock: `now` is passed in): tests/unit/loadsteps_test.js. The visible words are plain on purpose
// (no "experts", "OPFS", "n-gram", "kernels"): the detail goes to the console and tooltips.

// the parts in the order today's loader runs them (and the card lists them)
export const PART_ORDER = ["save", "words", "gpu"];
// parts that must be done / skip / fail before chat opens. #388's rule: all three. Removing "words" later lets the
// word list finish after ready (a one-line change; the card already handles a running part that does not gate)
export const GATE = new Set(["save", "words", "gpu"]);
const STATES = new Set(["wait", "run", "done", "skip", "fail"]);
const ENDED = new Set(["done", "skip", "fail"]);
const DISK = new Set(["save", "words"]);
const GiB = 2 ** 30;

// -> the plan: [{ id, bytes }] in PART_ORDER, without the parts that have nothing to do (0 bytes)
// (room/model-loader.js sizes today's sequential load with it)
export function planSteps({ saveBytes = 0, wordsBytes = 0, gpuBytes = 0 } = {}) {
  const by = { save: +saveBytes || 0, words: +wordsBytes || 0, gpu: +gpuBytes || 0 };
  return PART_ORDER.filter((id) => by[id] > 0).map((id) => ({ id, bytes: by[id] }));
}
// Where a sequential load is when step `stepId` has done `stepDone` of its bytes (loadTracker's arithmetic)
function where(plan, stepId, stepDone = 0) {
  const total = plan.reduce((a, s) => a + s.bytes, 0);
  let done = 0, stepNo = 0;
  const at = PART_ORDER.indexOf(stepId);
  for (let i = 0; i < plan.length; i++) {
    const s = plan[i], k = PART_ORDER.indexOf(s.id);
    if (k < at) { done += s.bytes; stepNo = i + 1; }
    else if (k === at) { done += Math.max(0, Math.min(s.bytes, +stepDone || 0)); stepNo = i + 1; break; }
    else break;
  }
  if (!stepNo && plan.length) stepNo = 1;
  return { done, total, pct: total ? Math.min(100, Math.floor(done / total * 100)) : 0, stepNo, steps: plan.length };
}
// the plan without step `id` (it turned out to have nothing to do: the copy was on this computer already)
export const drop = (plan, id) => plan.filter((s) => s.id !== id);
// Today's sequential load as it goes (room/model-loader.js): at(stepId, fraction 0..1 of that step) -> { done, total,
// pct, stepNo, steps } whose pct and done never go back. drop(id): a step that never reported any bytes leaves the
// plan; one that did (a copy that failed partway) keeps the bytes it did and ends there
export function loadTracker(plan) {
  let p = plan.slice(), best = 0, bestDone = 0;
  const did = {};
  return {
    get plan() { return p; },
    at(stepId, frac) {
      const s = p.find((x) => x.id === stepId);
      const mine = s ? Math.max(0, Math.min(1, +frac || 0)) * s.bytes : 0;
      if (s) did[stepId] = Math.max(did[stepId] || 0, mine);
      const o = where(p, stepId, mine);
      best = Math.max(best, o.pct); bestDone = Math.max(bestDone, Math.min(o.done, o.total));
      return { ...o, pct: best, done: Math.max(o.done, Math.min(bestDone, o.total)) };
    },
    drop(id) {
      const d = did[id] || 0;
      p = d > 0 ? p.map((s) => (s.id === id ? { ...s, bytes: Math.min(s.bytes, d) } : s)) : drop(p, id);
    },
  };
}

// ---- the two inputs, one shape: { parts: [{ id, done, total, state }], done, total } ----

// Today's onProgress(done, total, info) as parts. prev: an object this keeps its memory in between calls (start each
// load with a fresh {}). info { step, stepNo, steps, stepDone, stepTotal } (room/model-loader.js qwen4exp); without
// it (every other model) the load is one "gpu" part. done / total are onProgress's own, exact: the bar uses them.
// The parts before the running one: done when they ended with all their bytes, fail (keeping the bytes they did)
// when the loader dropped them partway, skip when they never ran. The ones after it: wait (total not known yet) as
// many as the plan has left (info.steps - info.stepNo; "gpu" is always last), skip for the rest.
export function partsFromProg(prev, done, total, info) {
  done = Math.max(0, +done || 0); total = Math.max(0, +total || 0);
  if (!info || !PART_ORDER.includes(info.step)) {
    return { parts: [{ id: "gpu", done, total, state: total > 0 && done >= total ? "done" : "run" }], done, total };
  }
  const st = prev || {};
  st.known ||= {}; st.ended ||= {};
  const X = info.step, stepDone = Math.max(0, +info.stepDone || 0), stepTotal = Math.max(0, +info.stepTotal || 0);
  if (st.cur && st.cur !== X && !st.ended[st.cur]) {
    // the step before this one just ended: the plan's bytes before X are what the earlier steps finished with
    const before = Math.max(0, done - stepDone);
    const others = Object.entries(st.ended).reduce((a, [, e]) => a + e.total, 0);
    const fin = Math.max(0, before - others), had = st.known[st.cur] || 0;
    st.ended[st.cur] = fin + Math.max(1, had * 1e-6) < had ? { state: "fail", done: fin, total: fin } : { state: "done", done: had, total: had };
  }
  st.cur = X;
  st.known[X] = stepTotal;
  const at = PART_ORDER.indexOf(X);
  const later = Math.max(0, (+info.steps || 0) - (+info.stepNo || 0));
  const after = PART_ORDER.slice(at + 1);
  const waiting = new Set(after.slice(Math.max(0, after.length - later)));
  const parts = PART_ORDER.map((id, k) => {
    if (k < at) { const e = st.ended[id]; return e ? { id, ...e } : { id, done: 0, total: 0, state: "skip" }; }
    if (k === at) return { id, done: Math.min(stepDone, stepTotal || stepDone), total: stepTotal, state: stepTotal > 0 && stepDone >= stepTotal && id === "gpu" ? "done" : "run" };
    return waiting.has(id) ? { id, done: 0, total: null, state: "wait" } : { id, done: 0, total: 0, state: "skip" };
  });
  return { parts, done, total };
}

// A concurrent loader's onParts(snapshot) as the same shape: done / total summed over the parts that are not skip;
// a fail counts as total = done (it keeps what it did, the rest is read some other way)
export function partsFromSnapshot(snap) {
  const parts = (Array.isArray(snap?.parts) ? snap.parts : []).filter((p) => p && PART_ORDER.includes(p.id) && STATES.has(p.state))
    .map((p) => ({ id: p.id, done: Math.max(0, +p.done || 0), total: p.total == null ? null : Math.max(0, +p.total || 0), state: p.state }));
  let done = 0, total = 0;
  for (const p of parts) {
    if (p.state === "skip") continue;
    const t = p.total ?? 0, d = Math.min(p.done, t || p.done);
    if (p.state === "fail") { done += p.done; total += p.done; continue; }
    done += Math.min(d, t); total += t;
  }
  return { parts, done, total };
}

// A whole percent, rounded down: 99.6% is not "100" (the card says ready only at the end)
export const wholePct = (done, total) => (total > 0 ? (done >= total ? 100 : Math.max(0, Math.floor(done / total * 100))) : 0);

// One device's bar: push({ parts, done, total }) from either adapter -> { parts, done, total, pct }, whose pct and
// done never go back (through a skip, a fail, or a re-deal that starts a new load with a bigger total)
export function progressTracker() {
  let best = 0, bestDone = 0;
  return {
    push({ parts = [], done = 0, total = 0 } = {}) {
      const pct = wholePct(done, total);
      best = Math.max(best, pct); bestDone = Math.max(bestDone, +done || 0);
      return { parts, done: bestDone, total, pct: best };
    },
  };
}

// The room's bar over every device: byDevice { name -> { parts?, done?, total?, left?, pct? } } -> { done, total,
// pct, left, slowest }. done / total: the bytes of the devices that report bytes; a peer on an older build has only
// its pct and is left out of them (the room's % then averages over the devices with bytes; with none, over the pcts).
// left: the most any running device has left, null while one running device has no estimate. slowest: that device
// (or, with an unknown one, the running device furthest behind). best: the room's pct so far (it never goes back).
export function roomProgress(byDevice = {}, best = 0) {
  let done = 0, total = 0;
  const pcts = [], running = [];
  for (const [name, d] of Object.entries(byDevice)) {
    const bytes = Number.isFinite(d?.done) && Number.isFinite(d?.total) && d.total > 0;
    const pct = bytes ? wholePct(d.done, d.total) : Math.max(0, Math.min(100, +d?.pct || 0));
    if (bytes) { done += Math.min(d.done, d.total); total += d.total; } else pcts.push(pct);
    const ended = d?.parts?.length ? d.parts.every((p) => ENDED.has(p.state)) : bytes ? d.done >= d.total : pct >= 100;
    if (!ended) running.push({ name, pct, left: d?.left == null || !Number.isFinite(+d.left) ? null : Math.max(0, +d.left) });
  }
  const unknown = running.filter((r) => r.left == null);
  const slow = unknown.length ? unknown.reduce((a, r) => (r.pct < a.pct ? r : a)) : running.length ? running.reduce((a, r) => (r.left > a.left ? r : a)) : null;
  const pct = Math.max(+best || 0, total > 0 ? wholePct(done, total) : pcts.length ? Math.floor(pcts.reduce((a, b) => a + b, 0) / pcts.length) : 0);
  return { done, total, pct, left: unknown.length ? null : slow ? slow.left : 0, slowest: slow?.name ?? null };
}

// The time left on one device: push(now ms, parts, agg { done, total }?) -> seconds, or null until it is worth
// saying (20 s since the first byte and 1 GiB in). Each running part's rate over the last 20 s; the time is the
// longer of the slowest gating part alone and every remaining byte over the summed rate (the parts share one
// network link). A part waiting with an unknown total (today's sequential loader) counts as agg.total - agg.done.
// What it shows moves only by at least max(20%, 120 s) or after 10 s held, and never goes up by less than 60 s.
export function etaTracker({ windowMs = 20000, minMs = 20000, minBytes = GiB, holdMs = 10000 } = {}) {
  const hist = new Map();   // part id -> [{ t, d }]
  let t0 = null, d0 = null, shown = null, setAt = 0;
  return {
    push(now, parts = [], agg = null) {
      const sumDone = parts.reduce((a, p) => a + (p.state === "skip" ? 0 : +p.done || 0), 0);
      if (d0 == null) d0 = sumDone;
      if (t0 == null && sumDone > d0) t0 = now;
      let gate = 0, rem = 0, rate = 0, unknownWait = false;
      for (const p of parts) {
        if (p.state === "wait" && p.total == null) { unknownWait = true; continue; }
        if (p.state !== "run" && p.state !== "wait") { hist.delete(p.id); continue; }
        const r = Math.max(0, (+p.total || 0) - (+p.done || 0));
        rem += r;
        if (p.state !== "run") continue;
        const h = hist.get(p.id) || [];
        if (h.length && p.done < h[h.length - 1].d) h.length = 0;   // a new load of this part
        h.push({ t: now, d: +p.done || 0 });
        while (h.length > 2 && now - h[0].t > windowMs) h.shift();
        hist.set(p.id, h);
        const dt = (h[h.length - 1].t - h[0].t) / 1000, rr = dt > 0 ? (h[h.length - 1].d - h[0].d) / dt : 0;
        rate += rr;
        if (GATE.has(p.id) && rr > 0) gate = Math.max(gate, r / rr);
      }
      if (unknownWait && agg && agg.total > 0) {
        // sequential: what is left is the whole load's remainder (the running part's own remainder is inside it)
        rem = Math.max(rem, (+agg.total || 0) - (+agg.done || 0));
      }
      if (t0 == null || now - t0 < minMs || sumDone - d0 < minBytes || !(rate > 0)) return shown == null ? null : Math.max(0, shown - (now - setAt) / 1000);
      const raw = Math.max(gate, rem / rate);
      if (shown == null) { shown = raw; setAt = now; return raw; }
      const cur = Math.max(0, shown - (now - setAt) / 1000);
      const diff = raw - cur;
      const big = Math.abs(diff) >= Math.max(0.2 * cur, 120), held = now - setAt >= holdMs;
      if ((big || held) && !(diff > 0 && diff < 60)) { shown = raw; setAt = now; return raw; }
      return cur;
    },
  };
}

// ---- wire: a device's parts in an ai-progress / ai-hostprog message ----
// -> [[id, done, total (-1: not known), state]]
export const packParts = (parts = []) => parts.map((p) => [p.id, Math.round(+p.done || 0), p.total == null ? -1 : Math.round(+p.total || 0), p.state]);
// a peer's packed parts, checked (anything malformed is left out); null when there is nothing usable
export function unpackParts(arr) {
  if (!Array.isArray(arr)) return null;
  const out = [];
  for (const a of arr.slice(0, PART_ORDER.length)) {
    if (!Array.isArray(a) || !PART_ORDER.includes(a[0]) || !STATES.has(a[3])) continue;
    const d = +a[1], t = +a[2];
    if (!Number.isFinite(d) || !Number.isFinite(t)) continue;
    out.push({ id: a[0], done: Math.max(0, d), total: t < 0 ? null : t, state: a[3] });
  }
  return out.length ? out : null;
}

// ---- the words ----

// how long is left, said plainly: "about 9 min", "about a minute", "under a minute"
export function leftText(s) {
  s = Math.max(0, +s || 0);
  if (s >= 90) return `about ${Math.round(s / 60)} min`;
  if (s >= 45) return "about a minute";
  return "under a minute";
}
// a part's bytes, one unit when both have it: "22.3 of 63.6 GB", "300 MB of 1.2 GB"
export function stepBytes(done, total) {
  const f = (b) => (b >= 2 ** 30 ? [(b / 2 ** 30).toFixed(1), "GB"] : [String(b > 0 ? Math.max(1, Math.round(b / 2 ** 20)) : 0), "MB"]);
  const [d, du] = f(+done || 0), [t, tu] = f(+total || 0);
  return du === tu || d === "0" ? `${d} of ${t} ${tu}` : `${d} ${du} of ${t} ${tu}`;
}
const LINES = {
  save: "Saving the model to this computer (first time only)",
  words: "Saving the model's word list to this computer (first time only)",
  gpu: "Putting the model on the graphics chip",
};
// the step a sequential load is on, in a line (room.js #ai-status and the breadcrumbs; never drawn on the card)
export const stepLine = (id) => LINES[id] || LINES.gpu;
// (the peer card in the room: "saving" while a device writes to its disk, "loading" after; deviceVerb on the card)
export const rowVerb = (id) => (DISK.has(id) ? "saving" : "loading");

const LONG = { save: "Saving to this computer", words: "Saving the word list", gpu: "Loading the graphics chip" };
const SHORT = { save: "Saving", words: "Word list", gpu: "Graphics chip" };
// One part's row on the card -> { label, value, cls, pct }. solo: this computer alone (long label, its GB); else a
// meter under a device's row (short label, a %). single: the model loads in one part (every model but Flash)
export function partRow(p, solo = true, { single = false } = {}) {
  const label = single && p.id === "gpu" ? (solo ? "Downloading the model" : "Loading") : (solo ? LONG : SHORT)[p.id] || LONG.gpu;
  const pct = p.state === "done" ? 100 : p.total > 0 ? wholePct(p.done, p.total) : 0;
  if (p.state === "skip") return { label, value: "", cls: "skip", pct: 0 };
  if (p.state === "wait") return { label, value: "waiting", cls: "wait", pct: 0 };
  if (p.state === "done") return { label, value: solo ? "\u2713 done" : "\u2713", cls: "done", pct: 100 };
  if (p.state === "fail") return p.id === "words" ? { label, value: solo ? "read online instead" : "online", cls: "fb", pct: 100 } : { label, value: "stopped", cls: "fail", pct };
  return { label, value: solo ? (p.total > 0 ? stepBytes(p.done, p.total) : "starting") : `${pct}%`, cls: "run", pct };
}

const runs = (parts, id) => parts.some((p) => p.id === id && p.state === "run");
const stateOf = (parts, id) => parts.find((p) => p.id === id)?.state;
const allEnded = (parts) => parts.length > 0 && parts.every((p) => ENDED.has(p.state));
const allWait = (parts) => !parts.length || parts.every((p) => p.state === "wait" || p.state === "skip") && !parts.some((p) => p.done > 0);
// was this device's start a second one (both copies already on it)?
export const cachedStart = (parts) => parts.length > 1 && stateOf(parts, "save") === "skip" && stateOf(parts, "words") === "skip";
// did this load save anything to the disk (a first start)? null while not known
export function firstStart(parts) {
  if (!parts?.length || parts.length === 1) return null;
  return parts.some((p) => DISK.has(p.id) && p.state !== "skip");
}

// A device's row verb: "getting ready", "saving and loading", "saving to its disk", "finishing its save", "loading",
// "loading from its disk", "waiting for pc", "✓ ready"
export function deviceVerb(parts = [], { roomReady = false, slowest = null, cached = null, self = null } = {}) {
  if (allEnded(parts)) return roomReady ? "✓ ready" : slowest && slowest !== self ? `waiting for ${slowest}` : "✓ ready";
  if (allWait(parts)) return "getting ready";
  const disk = runs(parts, "save") || runs(parts, "words"), gpu = runs(parts, "gpu");
  if (disk && gpu) return "saving and loading";
  if (disk) return stateOf(parts, "gpu") === "done" ? "finishing its save" : "saving to its disk";
  if (gpu) return (cached ?? cachedStart(parts)) ? "loading from its disk" : "loading";
  return "getting ready";
}

const DISKFULL = /quota|disk.?full|no space|out of (disk )?space|not enough (disk|storage)/i;
// who a stop names: "pc couldn't load its part of the model (...)" -> "pc"
export const stopWho = (why) => /^(.+?) couldn't load its part/.exec(String(why || ""))?.[1] || null;
export const stopIsDisk = (why) => DISKFULL.test(String(why || ""));

// The card's big line. room { left }, phase "pressed" | "run" | "ready" | "stopped"; ctx { model, why, me }
export function headline(room = {}, phase = "run", { model = "The model", why = "", me = null } = {}) {
  if (phase === "ready") return `${model} is ready`;
  if (phase === "stopped") {
    if (!stopIsDisk(why)) return "The start stopped";
    const who = stopWho(why);
    return !who || who === me ? "This computer ran out of disk space" : `${who} ran out of disk space`;
  }
  if (phase === "pressed" || room.left == null) return "Getting ready…";
  return `Ready to chat in ${leftText(room.left)}`;
}

const names = (devices) => devices.length > 2 ? `${devices.length} devices` : devices.map((d) => d.name).join(" and ");
// The line under it. devices [{ name, you, parts, done, total }]; ctx { host (a follower before the deal), redeal,
// paced (the pacer holds this device), single (the model loads in one part), cached (this computer has the copy) }
export function sentence(room = {}, phase = "run", devices = [], ctx = {}) {
  const solo = devices.length <= 1;
  if (phase === "ready") return solo ? "Running on this computer. Ask it anything below." : `Running on ${names(devices)} together. Ask it anything below.`;
  if (ctx.host && !devices.some((d) => d.parts?.length)) return `${ctx.host} is getting the model ready.`;
  if (ctx.redeal && phase === "pressed") return "Sharing out the model again. Nothing is lost.";
  if (!solo) {
    if (phase === "pressed" || devices.every((d) => allWait(d.parts || []))) return `${names(devices)} are checking what's already saved.`;
    const me = devices.find((d) => d.you);
    if (ctx.paced && room.slowest) return `Waiting for ${room.slowest} to catch up.`;
    if (me && allEnded(me.parts || []) && room.slowest && room.slowest !== me.name) return `Your part is ready. Waiting for ${room.slowest} to finish.`;
    return ctx.single ? "Each device downloads its own part and loads it."
      : ctx.first === false ? "Each device loads its own part from its disk." : "Each device saves its own part and loads it.";
  }
  const parts = devices[0]?.parts || [];
  let s;
  if (phase === "pressed" || allWait(parts)) s = "Checking this computer and what's already saved.";
  else {
    const disk = runs(parts, "save") || runs(parts, "words"), gpu = runs(parts, "gpu"), g = stateOf(parts, "gpu");
    if (disk && gpu) s = "Saving the model to this computer and loading it at the same time.";
    else if (disk && g === "wait") s = "Saving the model to this computer first. This happens once.";
    else if (disk && g === "done") s = "The graphics chip is loaded. Finishing the save to this computer.";
    else if (gpu && (ctx.single || parts.length === 1)) s = "Downloading the model and loading it onto the graphics chip.";
    else if (gpu && parts.some((p) => DISK.has(p.id) && (p.state === "done" || p.state === "fail"))) s = "Saved. Now loading it onto the graphics chip.";
    else if (gpu && (ctx.cached ?? cachedStart(parts))) s = "Loading it from this computer onto the graphics chip.";
    else if (gpu) s = "Loading it onto the graphics chip.";
    else s = "Almost ready.";
  }
  if (phase !== "pressed" && room.left == null && (devices[0]?.done > 0) && !allEnded(parts)) s += " Working out how long this takes…";
  return s;
}

// The quiet line at the bottom. first: this start saves to the disk (true), it is a second start (false), or not known
// (null); took: seconds from Start to ready
export function footer(phase = "run", { solo = true, first = null, took = null, wordsFailed = false } = {}) {
  if (phase === "stopped") return "";
  if (phase === "ready") {
    if (took == null) return "";
    const t = took >= 90 ? `${Math.round(took / 60)} min` : `${Math.max(1, Math.round(took))} s`;
    return first ? `Took ${t} · next start about a minute` : `Took ${t}`;
  }
  if (wordsFailed) return "";   // the note box says it
  if (!first) return "";
  return solo ? "The saving happens once. Next time it starts in about a minute."
    : "Each device saves only its own part, once. Next time it starts in about a minute.";
}
export const WORDS_NOTE = ["The word list didn't fit on this computer's disk.", " It's read online instead, so answers may start a moment later. Nothing to do."];

// What a stopped start says under its headline. save: the stopping device's save part ({ done, total } bytes) or null
// plain: why is already said in plain words (the room can't hold the model, the graphics chip check); else the
// technical why stays in the tooltip and this says which device could not load its part
export function stoppedText(why, { me = null, save = null, plain = false } = {}) {
  const who = stopWho(why);
  if (stopIsDisk(why) && save?.total > 0) {
    const gb = (b) => (b / GiB).toFixed(1);
    const x = save.done, y = save.total;
    return `${gb(x)} of ${gb(y)} GB is saved and stays saved. Free up about ${Math.ceil((y - x) / GiB + 2)} GB, then Try again continues from there.`;
  }
  if (who) return `${who === me ? "This computer" : who} couldn't load its part.`;
  const s = String(why || "").trim();
  return plain && s ? s.replace(/^./, (c) => c.toUpperCase()).replace(/([^.])$/, "$1.") : "This computer couldn't load its part.";
}
// the eyebrow over the headline
export function eyebrow(phase, { model = "the model", first = null } = {}) {
  if (phase === "stopped") return `${model} stopped`;
  if (phase === "ready") return first === true ? "First start \u00b7 saved for next time" : first === false ? "Started from what's saved" : "Started";
  if (first === true) return `Starting ${model} · first start`;
  if (first === false) return `Starting ${model} · saved from last time`;
  return `Starting ${model}`;
}
