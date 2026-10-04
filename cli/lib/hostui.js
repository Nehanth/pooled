// pooled host in a terminal: the screen's state, what each key does, and what it draws. Pure (no GPU,
// no network, no terminal), so it is unit tested; cli/lib/hostrun.js wires it to the room node.
//
// The room opens first, so devices can join while the host still chooses. Then, for whatever the
// flags did not already say: the model (a picker), how much this computer lends (a pledge prompt),
// and the room itself: every device and its pledge, who waits to join, and whether the pledges hold
// the model, with the room page's own math (room/plan.js roomFit, room/models.js roomBytes). Start
// stays off until they do.
//
// lib: what the room node exports (MODELS, FILES, NEED_GB, roomBytes, roomFit, shortNote, gbUp,
// pledgeGB, nodeCtxFor, and pickCtx / ctxShortNote for a model with a fallback context), passed in so
// tests can use the repo's modules directly.

import { mkStyle, visible, clip, wrap, width, padEnd, padStart, gb, gbNum, header, table, label, I, progressRow, clock, upFor } from "./style.js";

const GiB = 2 ** 30;
const r1 = (x) => Math.round(x * 10) / 10;
export const shortLabel = (lib, key) => (lib.MODELS[key]?.label || key).split("·")[0].trim();

// what the room lists a device as
export function deviceKind(meta = {}, self = false) {
  if (self) return "this computer";
  if (meta?.api) return "API";
  if (meta?.native === "node-dawn") return "CLI";
  if (meta?.ua === "iPhone" || meta?.ua === "Android") return "phone";
  if (meta?.ua === "iPad" || meta?.ua === "Android tablet") return "tablet";
  return "browser tab";
}

// How much a model needs in this room: the whole deal's bytes at the context the node opens it with
// (roomBytes), else room/models.js NEED_GB for a model without a SHAPE
const needAt = (lib, key, ctx) => { const rb = lib.roomBytes(key, ctx, "f16"); return rb ? r1((rb.L * rb.layerBytes + rb.hostBytes) / GiB) : null; };
export function modelNeedGB(lib, key, ctxAsk = 0) {
  return needAt(lib, key, lib.nodeCtxFor(key, ctxAsk)) ?? lib.NEED_GB[key] ?? null;
}
// The context a node falls back to for `key` when the room is short for its default (room/models.js
// pickCtx: the 1.7B opens at 8K when 16K doesn't fit), and what the model needs there; null when it
// has none (or --ctx was given) -> { ctx, needGB } | null
export function modelFallback(lib, key, ctxAsk = 0) {
  if (!lib.pickCtx) return null;
  const want = lib.nodeCtxFor(key, ctxAsk);
  const p = lib.pickCtx(key, { want, ask: ctxAsk, fitsAt: () => false });
  return p.ctx < want ? { ctx: p.ctx, needGB: needAt(lib, key, p.ctx) } : null;
}
const kOf = (n) => `${Math.round(n / 1024)}K`;
// "6 GB" or "6 GB (4 GB at 8K)": a model's need in words, whole GB
export function needWords(lib, key, ctxAsk = 0, needGB = modelNeedGB(lib, key, ctxAsk)) {
  if (needGB == null) return "";
  const fb = modelFallback(lib, key, ctxAsk);
  return `${Math.max(1, Math.round(needGB))} GB` + (fb?.needGB ? ` (${Math.max(1, Math.round(fb.needGB))} GB at ${kOf(fb.ctx)})` : "");
}

// Whether the room's pledges hold `model`, as the room page decides it (room.js roomFitFor): devices
// host first ({ name, meta, self }), each pledge through room/pledge.js pledgeGB.
// spareGB: how much more each device could lend (the host's own headroom; others unknown: 0).
// The context is the one the node will open the room with (room/models.js pickCtx, as the room node
// and the room page pick it): the model's default, or its fallback when only that fits (fellBack, with
// ctxNote saying so); a room short even there is short for the fallback (shortGB, note).
// Expert offload (lib.offloadFor, room/pledge.js): a device whose meta offers RAM (meta.offload, meta.ramGB) counts
// with the layers it holds that way when the pledges alone fall short (room/plan.js roomFit); offload: it fits so.
// -> { fits, needGB, minGB, haveGB, shortGB, note, ctx, want, fellBack, ctxNote, offload }
export function roomFitNow(lib, { model, devices, ctxAsk = 0, spareGB = [] }) {
  const pl = devices.map((d) => +lib.pledgeGB(d.meta) || 0);
  const haveGB = r1(pl.reduce((a, b) => a + b, 0));
  const needGB = modelNeedGB(lib, model, ctxAsk);
  const want = lib.nodeCtxFor(model, ctxAsk);
  const label = shortLabel(lib, model);
  if (!lib.roomBytes(model, want, "f16")) {
    const fits = needGB != null && haveGB >= needGB;
    const shortGB = fits ? 0 : r1((needGB || 0) - haveGB);
    return { fits, needGB, minGB: null, haveGB, shortGB, ctx: want, want, fellBack: false, ctxNote: "",
      note: fits ? "" : `This room is ${shortGB} GB short for ${label}: add a device or raise a pledge.` };
  }
  const off = (rb) => (lib.offloadFor ? lib.offloadFor(devices.map((d) => d.meta), rb.experts || rb.expertBytes || 0) : null);
  const fitAt = (c) => { const rb = lib.roomBytes(model, c, "f16"); return lib.roomFit(rb.L, pl.map((g) => g * GiB), rb.layerBytes, rb.hostBytes, off(rb)); };
  const pick = lib.pickCtx ? lib.pickCtx(model, { want, ask: ctxAsk, fitsAt: (c) => fitAt(c).fits }) : { ctx: want, want, fellBack: false };
  const fb = modelFallback(lib, model, ctxAsk);
  const base = { needGB, minGB: fb?.needGB ?? null, haveGB, ctx: pick.ctx, want, fellBack: !!pick.fellBack,
    ctxNote: pick.fellBack && lib.ctxShortNote ? lib.ctxShortNote(label, pick.ctx, want) : "" };
  const fit = fitAt(pick.ctx);
  if (fit.fits) return { fits: true, ...base, shortGB: 0, note: "", offload: !!fit.offload };
  const shortGB = lib.gbUp(lib.shortBy ? lib.shortBy(fit, spareGB) : fit.short);
  const note = lib.shortNote(label, fit, devices.map((d) => d.name), spareGB).replace(/\. Add a device/, ": add a device");
  return { fits: false, ...base, shortGB, note };
}

// The picker's rows: every model a node can host, smallest first (by its download: what it needs
// depends on its context, which is longer on some), with its download and its need.
// pulled: Set of model keys on disk; pledgeGB: what this computer lends (for "fits here alone");
// maxGB: the most it can lend (hereGB: the whole GB "Run it here" lends, null when it can't hold it);
// ramGB / offGB: expert offload (hereGB); hereOff: "Run it here" keeps some of the model's experts in RAM
export function modelRows(lib, { keys, pulled = new Set(), pledgeGB = 0, ctxAsk = 0, maxGB = 0, ramGB = 0, offGB = 0 }) {
  return keys.map((key) => {
    const needGB = modelNeedGB(lib, key, ctxAsk);
    const alone = roomFitNow(lib, { model: key, devices: [{ name: "this computer", meta: selfMeta(pledgeGB, ramGB) }], ctxAsk });
    const here = hereGB(lib, key, { ctxAsk, maxGB: Math.max(maxGB, pledgeGB), ramGB, offGB });
    return { key, label: lib.MODELS[key].label, fileBytes: lib.FILES?.[key]?.bytes || null, pulled: pulled.has(key), needGB, fitsAlone: alone.fits,
      minNeed: modelFallback(lib, key, ctxAsk),
      hereGB: here, hereOff: here != null && !!roomFitNow(lib, { model: key, devices: [{ name: "this computer", meta: selfMeta(here, ramGB) }], ctxAsk }).offload };
  }).sort((a, b) => (a.fileBytes ?? Infinity) - (b.fileBytes ?? Infinity) || (a.needGB ?? 99) - (b.needGB ?? 99));
}
// this computer's meta for a fit check: its pledge, and the RAM it offers for experts (none: 0)
const selfMeta = (gb, ramGB = 0) => ({ contribGB: gb, webgpu: true, ...(ramGB > 0 ? { offload: true, ramGB } : {}) });
// "Run it here": the smallest whole GB, up to maxGB, at which this computer holds the model alone
// (the room page's math: roomFit with the embedding and head on this device) at its default context,
// else at its fallback (the 1.7B: 16K on 6 GB, else 8K on 4 GB) -> GB | null.
// ramGB (expert offload, a discrete GPU's --ram): when no pledge up to maxGB holds a MoE model whole, it runs here
// with the experts it can't hold in RAM, lending offGB (the memory rule's default: all the GPU it lends, the
// expert cache gets the rest) when that fits.
export function hereGB(lib, key, { ctxAsk = 0, maxGB = 0, ramGB = 0, offGB = maxGB } = {}) {
  const need = modelNeedGB(lib, key, ctxAsk), min = modelFallback(lib, key, ctxAsk)?.needGB;
  const at = (g, ram = 0) => roomFitNow(lib, { model: key, devices: [{ name: "this computer", meta: selfMeta(g, ram) }], ctxAsk });
  for (let g = Math.max(1, Math.floor(need || 1)); g <= Math.floor(maxGB); g++) { const f = at(g); if (f.fits && !f.fellBack) return g; }
  if (min) for (let g = Math.max(1, Math.floor(min)); g <= Math.floor(maxGB); g++) if (at(g).fits) return g;
  const og = Math.floor(Math.min(offGB || maxGB, maxGB) * 2) / 2;
  if (ramGB > 0 && og >= 1 && at(og, ramGB).fits) return og;
  return null;
}

// the model to preselect: the largest this computer can hold alone, else the smallest
export function recommendModel(rows) {
  const fits = rows.filter((r) => r.hereGB != null || r.fitsAlone);
  return (fits.length ? fits[fits.length - 1] : rows[0])?.key || null;
}

// How much to lend by default, as the room page does (half the GPU's memory), at least what the
// smallest model needs, and never more than the memory rule's most (lend.js memoryRule with "max").
// mem: detectMemory(); maxGB: the most this computer can lend. -> { def, max, totalGB }
export function pledgeDefaults(mem, { maxGB, ruleGB, smallestNeedGB = 4 }) {
  const totalGB = mem?.totalGB > 0 ? mem.totalGB : null;
  const max = Math.max(1, Math.min(64, Math.floor(maxGB || ruleGB || 1)));
  const half = totalGB ? Math.round(totalGB / 2) : ruleGB || 1;
  const def = Math.max(1, Math.min(max, Math.max(half, Math.ceil(smallestNeedGB))));
  return { def, max, totalGB };
}

// ---------------- the state and its keys ----------------
// state: {
//   step: "confirm" | "pick" | "how" | "pledge" | "room" | "starting" | "online",
//   how: 0 "Pool with devices" | 1 "Run it here" (the cursor on the "how" step); flags.mode: --pool / --here
//   model, rows, sel, pledge: { gb, max, totalGB, typed }, fixed: { model, pledge } (given as flags),
//   devices: [{ id, name, kind, gb, self, pct, range }], lobby: [{ id, line }],
//   dl: { key, state: "none" | "ask" | "running" | "done" | "error" | "stream", done, total, bps, error },
//   fit, flags: { start, wait, chat }, notice, split, code, link
// }
export function initialState({ rows, model = null, pledge, fixedPledge = false, flags = {}, pulled = new Set(), code = "", link = "", yes = false, noPull = false }) {
  const s = {
    step: "pick", rows, model, sel: 0, how: 0, pledge: { ...pledge, typed: "" }, fixed: { model: !!model, pledge: !!fixedPledge },
    devices: [], lobby: [], dl: { key: null, state: "none", done: 0, total: 0, bps: null, error: null },
    fit: null, flags: { start: !!flags.start, wait: flags.wait || 0, chat: !!flags.chat, mode: flags.mode === "here" || flags.mode === "pool" ? flags.mode : null, splitGiven: !!flags.splitGiven },
    notice: "", split: null, splitMode: flags.split === "memory" ? "memory" : "speed", code, link, yes, noPull,
  };
  const rec = model || recommendModel(rows);
  s.sel = Math.max(0, rows.findIndex((r) => r.key === rec));
  if (model) {
    if (!pulled.has(model)) {
      if (noPull) s.dl = { ...s.dl, key: model, state: "stream" };
      else if (yes) s.dl = { ...s.dl, key: model, state: "running" };
      else s.dl = { ...s.dl, key: model, state: "ask" };
    } else s.dl = { ...s.dl, key: model, state: "done" };
    // (initialState has no effects: hostrun applies the pledge and split of a --here / --pool start)
    if (s.dl.state === "ask") s.step = "confirm";
    else s.step = nextAfterModel(s, []);
  }
  return s;
}

// whether Start can go now -> { ok, why }
export function canStart(s) {
  if (!s.model) return { ok: false, why: "choose a model first" };
  if (s.step === "starting" || s.step === "online") return { ok: false, why: "" };
  if (s.dl.key === s.model && s.dl.state === "running") return { ok: false, why: "waiting for the download" };
  if (s.dl.key === s.model && s.dl.state === "error") return { ok: false, why: "the download failed: m picks the model again to retry" };
  if (s.dl.key === s.model && s.dl.state === "ask") return { ok: false, why: "" };
  if (!s.fit) return { ok: false, why: "" };
  if (!s.fit.fits) return { ok: false, why: s.fit.note };
  return { ok: true, why: "" };
}
// --start / --wait N: start by itself once it can (and N devices are in)
export function autoStart(s) {
  if (!(s.flags.start || s.flags.wait > 0) || s.step !== "room") return false;
  if (s.flags.wait > 0 && s.devices.length < s.flags.wait) return false;
  return canStart(s).ok;
}

// a download the screen started ended (pullWithProgress's result: { ok } | { aborted } | { error }).
// A notice that waited on it ("waiting for the download") goes with it.
export function pullDone(s, key, r = {}) {
  const t = { ...s };
  if (t.dl.key === key) {
    if (r.ok) t.dl = { ...t.dl, state: "done" };
    else if (r.aborted) { if (t.dl.state === "running") t.dl = { ...t.dl, state: "none" }; }
    else t.dl = { ...t.dl, state: "error", error: r.error?.message || String(r.error || "failed") };
  }
  if (/^waiting for the download/.test(t.notice || "")) t.notice = "";
  return t;
}

// after the model: "How do you want to run it?" (step "how"), unless --here / --pool said
function nextAfterModel(t, fx) {
  if (!t.flags.mode) { t.how = 0; return "how"; }
  return runMode(t, t.flags.mode, fx);
}
const afterPledge = (t) => (t.fixed.pledge || t.pledgeDone ? "room" : "pledge");
// the choice on "how" -> the next step (t changes in place; fx gets the pledge and split to set)
//   pool: the lend question, then the room panel waiting for devices, spread across all of them
//   here: lend what the model needs (never more than this computer can), start once it can
function runMode(t, mode, fx) {
  if (mode === "pool") {
    if (!t.flags.splitGiven) { t.splitMode = "memory"; fx.push({ do: "split", mode: "memory" }); }
    return afterPledge(t);
  }
  const row = t.rows.find((r) => r.key === t.model);
  if (!row?.hereGB) { t.notice = hereWhy(row, t.pledge); return "how"; }
  const gb = t.fixed.pledge && t.pledge.gb >= row.hereGB ? t.pledge.gb : row.hereGB;
  t.pledge = { ...t.pledge, gb, typed: "" }; t.pledgeDone = true;
  fx.push({ do: "pledge", gb });
  if (!t.flags.splitGiven) { t.splitMode = "speed"; fx.push({ do: "split", mode: "speed" }); }
  t.flags = { ...t.flags, start: true };
  return "room";
}
// why "Run it here" is off for a model
export const hereWhy = (row, pledge) => `needs ${Math.max(1, Math.round(row?.minNeed?.needGB || row?.needGB || 0))} GB${row?.minNeed ? ` (at ${kOf(row.minNeed.ctx)})` : ""}, this computer has ${pledge?.max ?? 0} GB to lend`;

// one key -> { state, fx: [effects] }. keys: "up" | "down" | "left" | "right" | "enter" | "backspace"
// | "esc" | a single character. Effects: { do: "pull", key } | { do: "stream", key } | { do: "model", key }
// | { do: "pledge", gb } | { do: "start" } | { do: "redeal" } | { do: "allow", id } | { do: "deny", id }
// | { do: "chat" } | { do: "quit" }
export function reduce(s, key) {
  const fx = [];
  const t = { ...s, notice: "" };
  const k = key.length === 1 ? key.toLowerCase() : key;
  if (k === "q" && t.step !== "pledge") return { state: t, fx: [{ do: "quit" }] };
  switch (t.step) {
    case "confirm": {
      if (k === "y" || k === "enter") { t.dl = { ...t.dl, state: "running", done: 0 }; fx.push({ do: "pull", key: t.dl.key }); t.step = nextAfterModel(t, fx); }
      else if (k === "n") { t.dl = { ...t.dl, state: "stream" }; fx.push({ do: "stream", key: t.dl.key }); t.step = nextAfterModel(t, fx); }
      break;
    }
    case "how": {
      if (k === "up") t.how = 0;
      else if (k === "down") t.how = 1;
      else if (k === "esc") { t.step = "pick"; t.sel = Math.max(0, t.rows.findIndex((r) => r.key === t.model)); }
      else if (k === "enter") t.step = runMode(t, t.how === 1 ? "here" : "pool", fx);
      break;
    }
    case "pick": {
      // the list stops at its ends (it does not wrap: ↑ on the smallest model stays there)
      if (k === "up") t.sel = Math.max(0, t.sel - 1);
      else if (k === "down") t.sel = Math.min(t.rows.length - 1, t.sel + 1);
      else if (k === "esc" && t.model) t.step = "room";
      else if (k === "enter") {
        const row = t.rows[t.sel];
        t.model = row.key;
        fx.push({ do: "model", key: row.key });
        if (row.pulled) t.dl = { key: row.key, state: "done", done: 0, total: 0, bps: null, error: null };
        else if (t.noPull) { t.dl = { key: row.key, state: "stream", done: 0, total: 0, bps: null, error: null }; fx.push({ do: "stream", key: row.key }); }
        else if (!(t.dl.key === row.key && (t.dl.state === "running" || t.dl.state === "done"))) {
          // picking a model that needs a download is the go-ahead for it (the row says how big)
          t.dl = { key: row.key, state: "running", done: 0, total: row.fileBytes || 0, bps: null, error: null };
          fx.push({ do: "pull", key: row.key });
        }
        t.step = nextAfterModel(t, fx);
      }
      break;
    }
    case "pledge": {
      const p = { ...t.pledge };
      const set = (v) => { p.gb = Math.max(1, Math.min(p.max, v)); p.typed = ""; };
      if (k === "left" || k === "down") set((p.typed ? +p.typed : p.gb) - 1);
      else if (k === "right" || k === "up") set((p.typed ? +p.typed : p.gb) + 1);
      else if (/^[0-9]$/.test(k) && p.typed.length < 3) p.typed += k;
      else if (k === "backspace") p.typed = p.typed.slice(0, -1);
      else if (k === "esc") { p.typed = ""; if (t.pledgeDone) t.step = "room"; }
      else if (k === "enter") {
        if (p.typed) {
          const v = +p.typed;
          if (!(v >= 1)) { t.notice = "lend at least 1 GB"; p.typed = ""; t.pledge = p; break; }
          if (v > p.max) t.notice = `this computer can lend at most ${p.max} GB: lending ${p.max}`;
          set(v);
        }
        t.pledgeDone = true;
        fx.push({ do: "pledge", gb: p.gb });
        t.step = "room";
      } else if (k === "q") return { state: t, fx: [{ do: "quit" }] };
      t.pledge = p;
      break;
    }
    case "room": {
      if (k === "m") { t.step = "pick"; t.sel = Math.max(0, t.rows.findIndex((r) => r.key === t.model)); }
      else if (k === "p" || k === "l") { t.step = "pledge"; t.pledge = { ...t.pledge, typed: "" }; }
      else if (k === "i") fx.push({ do: "copy" });
      else if (k === "s") { t.splitMode = t.splitMode === "memory" ? "speed" : "memory"; fx.push({ do: "split", mode: t.splitMode }); }
      else if (k === "a" && t.lobby.length) fx.push({ do: "allow", id: t.lobby[0].id });
      else if (k === "d" && t.lobby.length) fx.push({ do: "deny", id: t.lobby[0].id });
      else if (k === "enter") {
        const c = canStart(t);
        if (c.ok) { t.step = "starting"; fx.push({ do: "start" }); }
        else t.notice = c.why || "not yet";
      }
      break;
    }
    case "starting": {
      if (k === "a" && t.lobby.length) fx.push({ do: "allow", id: t.lobby[0].id });
      else if (k === "d" && t.lobby.length) fx.push({ do: "deny", id: t.lobby[0].id });
      break;
    }
    case "online": {
      if (k === "c") fx.push({ do: "chat" });
      else if (k === "i") fx.push({ do: "copy" });
      else if (k === "s") { t.splitMode = t.splitMode === "memory" ? "speed" : "memory"; fx.push({ do: "split", mode: t.splitMode }); t.notice = "r rebalances the room with the new split"; }
      else if (k === "enter" || k === "r") fx.push({ do: "redeal" });
      else if (k === "a" && t.lobby.length) fx.push({ do: "allow", id: t.lobby[0].id });
      else if (k === "d" && t.lobby.length) fx.push({ do: "deny", id: t.lobby[0].id });
      break;
    }
  }
  return { state: t, fx };
}

// ---------------- drawing ----------------
// The look (docs: the CLI design spec, "minimal") lives in lib/style.js: S = style() gives the roles
// (ink, ink2, ink3, acc, err, pill, bar, mark, keys). colors() is kept for older callers.
export function colors(on) {
  const w = (a, b = 0) => (s) => (on ? `\x1b[${a}m${s}\x1b[${b || (a === 1 || a === 2 ? 22 : 39)}m` : String(s));
  return { on, dim: w(2), bold: w(1), green: w(32), yellow: w(33), red: w(31), cyan: w(36), magenta: w(35) };
}
export { visible, clip, wrap };
const pad = padEnd;
const gbText = (b) => gb(b);
export const fmtCode = (x) => (String(x || "").length === 6 ? `${x.slice(0, 3)}-${x.slice(3)}` : String(x || ""));

// one picker row in words: downloaded / "1.8 GB download", "needs 6 GB", which group it is in;
// min: "4 GB at 8K" for a model with a fallback context
export function pickerRow(r, { rec = null } = {}) {
  const have = r.pulled ? "downloaded" : r.fileBytes ? `${gbText(r.fileBytes)} download` : "download";
  const need = r.needGB != null ? `${Math.max(1, Math.round(r.needGB))} GB` : "";
  const min = r.minNeed?.needGB ? `${Math.max(1, Math.round(r.minNeed.needGB))} GB at ${kOf(r.minNeed.ctx)}` : "";
  return { have, need, min, alone: !!r.fitsAlone, rec: r.key === rec };
}

// the plain (no escapes) style: tests, and callers that pass nothing
const PLAIN = mkStyle({ depth: "none", plain: true });

// A split costs a network round per token. When the lends split a model that this computer could
// hold alone by lending more, say so before Start, with a rough speed from the links' round trips
// (informational: the user decides how much each computer lends).
// -> { alone: { gb } | null, devices, hopMs, tps } | null (no split, or nothing to say)
export function splitAdvice(s, lib) {
  if (!s.model || !s.fit?.fits || !lib) return null;
  const gpu = s.devices.filter((d) => d.gb > 0);
  if (gpu.length < 2) return null;
  const fitsOn = (d, g) => roomFitNow(lib, { model: s.model, devices: [{ name: d.name, meta: { ...d.meta, webgpu: true, contribGB: g } }] }).fits;
  if (gpu.some((d) => fitsOn(d, d.gb))) return null;   // one device holds it: the deal puts it all there
  const self = gpu.find((d) => d.self);
  let alone = null;
  if (self) for (let g = Math.ceil(self.gb); g <= (s.pledge.max || 0); g++) if (fitsOn(self, g)) { alone = { gb: g }; break; }
  const rtts = gpu.map((d) => d.rtt).filter((x) => Number.isFinite(x) && x > 0);
  const hopMs = rtts.length ? Math.round(rtts.reduce((a, b) => a + b, 0) / rtts.length / 2) : null;
  // each token goes around the ring: one hop per device, plus a little compute
  const tps = hopMs ? Math.max(1, Math.round(1000 / (gpu.length * hopMs + 12))) : null;
  return { alone, devices: gpu.length, hopMs, tps };
}
export function splitLines(adv, { model, selfName = "this computer", gb: now } = {}) {
  if (!adv) return [];
  const L = [];
  if (adv.alone) L.push(`${model} would run on ${selfName} alone if it lends ${adv.alone.gb} GB (now ${now} GB): l lends more.`);
  L.push(`Split across ${adv.devices} devices, each token waits for the network: ${adv.tps ? `expect roughly ${adv.tps} tok/s` : "expect it to be slower"}.`);
  return L;
}
// the network's share of an online split room
export const hopHint = (devices, hopMs) => (devices > 1 && hopMs ? `each token crosses the network ${devices === 2 ? "twice" : `${devices} times`}; ~${hopMs} ms per hop` : "");

// the model's name and the rest of its label ("Qwen3.8 27B", "Q4")
const labelParts = (lib, key) => {
  const full = lib?.MODELS?.[key]?.label || key || "";
  const [name, ...rest] = full.split("·").map((x) => x.trim());
  return { name, rest: rest.join(" · ") };
};

// the header's third line: the model and what the room is doing
function statusText(s, S, { lib, spin, now }) {
  const { name, rest } = labelParts(lib, s.model);
  const tail = (t) => S.ink3(` · ${rest ? `${rest} · ` : ""}`) + t;
  if (!s.model) return S.ink3("choosing a model");
  if (s.step === "online") {
    const n = s.devices.length;
    return name + tail(`${S.acc(S.g.live)} online` + S.ink3(` · ${n} device${n === 1 ? "" : "s"}${s.onlineAt ? ` · up ${upFor(now - s.onlineAt)}` : ""}${s.tps ? ` · ${s.tps.toFixed(1)} tok/s` : ""}`));
  }
  if (s.step === "starting") return name + tail(`${spin} loading` + (s.startedAt ? S.ink3(` · ${clock(now - s.startedAt)}`) : ""));
  if (s.dl.key === s.model && s.dl.state === "running") return name + tail(`${spin} downloading`);
  if (canStart(s).ok) return name + tail(S.ink3("ready to start"));
  return name + tail(S.ink3("waiting for devices"));
}

// What Start would deal, as the room page does it (room/plan.js dealRoom): device index (host first,
// devices that lend) -> [lo, hi) | null (not needed). null when it can't be worked out (no fit yet).
export function dealPreview(s, lib) {
  if (!s.model || !s.fit?.fits || !lib?.dealRoom || !lib.roomBytes) return null;
  const rb = lib.roomBytes(s.model, s.fit.ctx || lib.nodeCtxFor(s.model, s.ctxAsk || 0), "f16");
  if (!rb) return null;
  const gpu = s.devices.filter((d) => d.gb > 0);
  const deal = lib.dealRoom({ L: rb.L, layerBytes: rb.layerBytes, hostBytes: rb.hostBytes, pledges: gpu.map((d) => d.gb * GiB),
    mode: s.splitMode === "memory" ? "memory" : "speed", phone: gpu.map((d) => d.kind === "phone") });
  if (!deal?.used?.length) return null;
  const out = new Map();
  gpu.forEach((d, i) => { const k = deal.used.indexOf(i); out.set(d, k >= 0 ? deal.ranges[k] : null); });
  return out;
}
// one device holds the model by its own pledge: spreading it is a choice, not a need
const onePledgeHolds = (s, lib) => !!(lib && s.model && s.devices.some((d) => d.gb > 0 && roomFitNow(lib, { model: s.model, devices: [{ name: d.name, meta: { ...d.meta, webgpu: true, contribGB: d.gb } }] }).fits));

// a device's GPU as the room reports it ("nvidia nvidia-gb10", "apple m3-pro") -> "NVIDIA GB10", "Apple M3 Pro"
const VENDOR = { nvidia: "NVIDIA", amd: "AMD", apple: "Apple", intel: "Intel", qualcomm: "Qualcomm", arm: "Arm" };
export function gpuLabel(g) {
  let t = String(g || "").trim().split(/\s+/).filter(Boolean);
  if (!t.length) return "";
  const v = t[0].toLowerCase();
  if (t.length > 1 && t[1].toLowerCase().startsWith(v + "-")) t = [t[0], t[1].slice(v.length + 1), ...t.slice(2)];
  const words = t.join(" ").split(/[\s-]+/).map((w, i) => (i === 0 && VENDOR[w.toLowerCase()]) || (/^[a-z]*\d/i.test(w) ? w.toUpperCase() : w.charAt(0).toUpperCase() + w.slice(1)));
  return words.join(" ");
}

// the table of devices: DEVICE, GPU (70 columns and up), LENDS or LAYERS, and a state
function deviceTable(s, S, { W, online, preview = null }) {
  const wide = W >= 69;
  const cols = [{ h: "DEVICE", w: 12 }];
  if (wide) cols.push({ h: "GPU", w: 14 });
  cols.push(online ? { h: "LAYERS", w: 6 } : { h: "LENDS", w: 5, align: "r" });
  if (preview) cols.push({ h: "HOLDS", w: 6 });
  cols.push({ h: "", w: 20 });
  const cut = (x, n) => (width(x) > n ? [...x].slice(0, n - 1).join("") + "…" : x);
  const rows = s.devices.map((d) => {
    const gpuName = cut(d.self && s.gpuName ? s.gpuName : gpuLabel(d.meta?.gpu) || (d.kind === "phone" ? "phone" : ""), 14);
    const chatOnly = d.gb == null;
    const name = d.self ? S.bold(cut(d.name, 12)) : chatOnly ? S.ink3(cut(d.name, 12)) : cut(d.name, 12);
    const mid = online ? (chatOnly || !d.range ? S.ink3(S.g.none) : String(d.range).replace(/[–-]/, S.g.dash)) : (chatOnly ? S.ink3(S.g.none) : gbNum(d.gb));
    const state = d.self ? S.ink3("this computer") : chatOnly ? S.ink3("chat only") : online ? "" : S.ink3("joined");
    const r = [name];
    if (wide) r.push(chatOnly ? S.ink3(gpuName || d.kind) : gpuName);
    r.push(mid);
    if (preview) {
      const rg = preview.get(d);
      r.push(rg ? `${rg[0]}${S.g.dash}${rg[1] - 1}` : S.ink3(S.g.none));
      r.push(rg === null && !chatOnly ? S.ink3(d.self ? "this computer · not needed" : "not needed") : state);
    } else r.push(state);
    return r;
  });
  // who waits in the lobby: table rows too; keys on the first only
  s.lobby.slice(0, 3).forEach((l, i) => {
    const who = String(l.line || "").replace(/ wants to join.*$/, "");
    const r = [cut(who, 12)];
    if (wide) r.push(S.ink3(cut(String(l.line || "").match(/\(([^)]*)\)/)?.[1] || "", 14)));
    r.push("");
    if (preview) r.push("");
    // the code the joining screen shows too (room/chanauth.js sasOf): the same only on a link nobody sits in the middle of
    r.push(S.acc("wants to join") + (l.sas ? S.ink3(` · code ${l.sas}`) : "") + (i === 0 ? "  " + S.ink2("a") + S.ink3(" allow  ") + S.ink2("d") + S.ink3(" deny") : ""));
    rows.push(r);
  });
  const out = table(S, cols, rows);
  if (s.lobby.length > 3) out.push(I + S.ink3(`and ${s.lobby.length - 3} more`));
  return out;
}

// the screen for a state -> lines (each cut to width - 1)
//   S: style() (plain when left out); spin: the spinner's frame; events: [{ t, text }] newest first
export function render(s, { width: cols = 80, S = PLAIN, lib, spin = "", events = [], now = Date.now() } = {}) {
  const W = Math.max(40, cols) - 1;
  const L = [];
  const lbl = (k) => labelParts(lib, k).name;
  spin ||= S.spin(0);
  const push = (...ls) => { for (const l of ls) L.push(l); };
  const blank = () => { if (L.length && L[L.length - 1] !== "") L.push(""); };

  // the header, fixed for the whole session: the mark, the room code and its link, the status
  const link = s.link || "";
  const shortLink = link.replace(/^https:\/\//, "");
  push("", ...header(S, cols, [
    S.bold("pooled host"),
    s.code ? S.pill(fmtCode(s.code)) + "  " + S.ink2(S.link(link, cols >= 100 ? link : shortLink)) : "",
    statusText(s, S, { lib, spin, now }),
  ]));
  blank();
  if (s.gpu && (s.step === "pick" || s.step === "how" || s.step === "pledge" || s.step === "confirm")) push(label(S, "gpu") + s.gpu.replace(/ · /, S.ink3(" · ")));
  if (s.gpu && (s.step === "pick" || s.step === "confirm")) blank();

  if (s.step === "confirm") {
    const row = s.rows.find((r) => r.key === s.dl.key);
    push(label(S, "model") + lbl(s.dl.key));
    blank();
    push(I + `${lbl(s.dl.key)} is not downloaded${row?.fileBytes ? ` (${gbText(row.fileBytes)})` : ""}.`);
    push(I + "Download it now?");
    blank();
    push(I + S.keys([["y", "download", "primary"], ["n", "stream from Hugging Face instead"], ["q", "quit"]], { width: W - 2 }));
    return L.map((l) => clip(l.replace(/ +$/, ""), W));
  }

  if (s.step === "pick") {
    push(...(s.devices.length > 1 || s.lobby.length ? [...deviceTable(s, S, { W, online: false }), ""] : []));
    push(I + "Which model?");
    const rec = recommendModel(s.rows);
    const nameW = Math.max(...s.rows.map((r) => width(lbl(r.key)))) + 2;
    // "(4 GB at 8K)" after a need with a fallback context, the column kept for the others
    const minW = Math.max(0, ...s.rows.map((r) => { const m = pickerRow(r).min; return m ? width(m) + 3 : 0; }));
    // one list, smallest first; how to run it (pooled or here) is the next question
    blank();
    s.rows.forEach((r, i) => {
      const on = i === s.sel, x = pickerRow(r, { rec });
      const name = on ? `${S.acc(S.g.sel)} ${S.bold(pad(lbl(r.key), nameW))}` : `  ${pad(lbl(r.key), nameW)}`;
      push(I + name + S.ink3("needs ") + padStart(x.need, 5) + S.ink3(pad(x.min ? ` (${x.min})` : "", minW)) + "   " + (r.pulled ? S.ink2(x.have) : S.ink3(x.have)));
    });
    blank();
    const k = [[S.g.up, "choose"], ["enter", "host it", "primary"]];
    if (s.model) k.push(["esc", "back"]);
    k.push(["q", "quit"]);
    push(I + S.keys(k, { width: W - 2 }));
    if (s.notice) push(I + S.ink3(s.notice));
    return L.map((l) => clip(l.replace(/ +$/, ""), W));
  }

  if (s.step === "how") {
    const row = s.rows.find((r) => r.key === s.model);
    push(label(S, "model") + lbl(s.model) + S.ink3(row ? ` · ${pickerRow(row).have}` : ""));
    blank();
    push(I + `How do you want to run ${lbl(s.model)}?`);
    blank();
    const choices = [
      ["Pool with devices", "other devices join and each holds a part", true],
      ["Run it here", row?.hereGB ? `all of it here, lending ${row.hereGB} GB${row.hereOff ? " (some experts in RAM: slower)" : ""}; others can chat` : hereWhy(row, s.pledge), !!row?.hereGB],
    ];
    const nameW = Math.max(...choices.map((c) => width(c[0]))) + 2;
    choices.forEach(([name, what, ok], i) => {
      const on = i === s.how;
      const n = on ? `${S.acc(S.g.sel)} ${ok ? S.bold(pad(name, nameW)) : S.ink3(pad(name, nameW))}` : `  ${ok ? pad(name, nameW) : S.ink3(pad(name, nameW))}`;
      push(I + n + S.ink3(what));
    });
    blank();
    push(I + S.keys([[S.g.up, "choose"], ["enter", "go", "primary"], ["esc", "back"], ["q", "quit"]], { width: W - 2 }));
    if (s.notice) push(I + S.ink3(s.notice));
    return L.map((l) => clip(l.replace(/ +$/, ""), W));
  }

  if (s.step === "pledge") {
    if (s.model) push(label(S, "model") + lbl(s.model) + S.ink3(s.dl.key === s.model && s.dl.state === "done" ? " · downloaded" : ""));
    blank();
    push(I + "How much GPU memory should this computer lend?");
    blank();
    const shown = s.pledge.typed ? S.bold(s.pledge.typed) + S.rev(" ") : S.bold(`${s.pledge.gb} GB`);
    const total = s.pledge.totalGB ? Math.round(s.pledge.totalGB) : s.pledge.max;
    push(I + S.bar(s.pledge.gb / (total || 1), W < 69 ? 16 : 32) + "  " + shown + S.ink3(` of ${total} GB`));
    const row = s.rows.find((r) => r.key === s.model);
    const sentence = [];
    if (s.pledge.totalGB) sentence.push(`Leaves ${Math.max(0, Math.round(s.pledge.totalGB - s.pledge.gb))} GB for this computer.`);
    if (row?.needGB != null) {
      const need = Math.max(1, Math.round(row.needGB)), mn = row.minNeed?.needGB ? Math.max(1, Math.round(row.minNeed.needGB)) : null;
      const k = row.minNeed ? kOf(row.minNeed.ctx) : "";
      sentence.push(need <= s.pledge.gb ? `The model needs ${need} GB: this computer can hold it alone.`
        : mn && mn <= s.pledge.gb ? `The model needs ${need} GB, or ${mn} GB with a ${k} context: this computer can hold it alone at ${k}.`
        : `The model needs ${need} GB${mn ? ` (${mn} GB at ${k})` : ""}, so other devices have to lend at least ${(mn || need) - s.pledge.gb} GB more.`);
    }
    for (const l of wrap(sentence.join(" "), W - 2)) push(I + S.ink3(l));
    blank();
    push(I + S.keys([[S.g.lr, "1 GB"], ["type", "a number"], ["enter", "lend", "primary"], ...(s.pledgeDone ? [["esc", "back"]] : [])], { width: W - 2 }));
    if (s.notice) push(I + S.ink3(s.notice));
    return L.map((l) => clip(l.replace(/ +$/, ""), W));
  }

  // room / starting / online
  const online = s.step === "online";
  if (s.step === "starting") {
    const dealt = s.devices.filter((d) => d.gb > 0 && d.range);
    const rows = (dealt.length ? dealt : s.devices.filter((d) => d.gb > 0)).map((d) => ({ d, pct: d.pct ?? 0 })).sort((a, b) => a.pct - b.pct);
    const out = table(S, [{ h: "DEVICE", w: 12 }, { h: "LAYERS", w: 6 }, { h: "", w: 40 }], rows.map(({ d, pct }) => {
      const range = String(d.range || S.g.none).replace(/[–-]/, S.g.dash);
      if (pct >= 100) return [S.ink2(d.name), S.ink2(range), S.ink2("loaded")];
      return [d.self ? S.bold(d.name) : d.name, range, S.bar(pct / 100, 20) + "  " + padStart(`${Math.round(pct)}%`, 4) + (d.self && s.load?.total ? "  " + S.ink3(loadNote(s.load)) : "")];
    }));
    push(...out);
  } else push(...deviceTable(s, S, { W, online, preview: s.step === "room" ? dealPreview(s, lib) : null }));
  blank();

  if (!online && s.step === "room") {
    const d = s.dl;
    if (d.key === s.model && d.state === "running") push(progressRow(S, "download", { done: d.done, total: d.total, bps: d.bps, cols: 20 }));
    else if (d.key === s.model && d.state === "stream") push(label(S, "download") + S.ink3("not downloaded: streams from Hugging Face on each start"));
    else if (d.key === s.model && d.state === "error") push(label(S, "download") + S.err(`download failed: ${d.error}`));
    if (s.fit) {
      // a model with a fallback context: both needs, the bar against the one the room would open
      const have = Math.round(s.fit.haveGB), need = Math.max(1, Math.round(s.fit.needGB || 0));
      const fb = s.fit.minGB ? modelFallback(lib, s.model, s.ctxAsk || 0) : null, mn = fb ? Math.max(1, Math.round(s.fit.minGB)) : null;
      const barTo = mn && (s.fit.fellBack || !s.fit.fits) ? mn : need;
      push(label(S, "memory") + S.bar(Math.min(1, have / barTo), W < 69 ? 12 : 20) + "  " + `${have} GB lent` + S.ink3(` · ${need} GB needed${mn ? ` (${mn} GB at ${kOf(fb.ctx)})` : ""}`));
    }
    const gpuN = s.devices.filter((x) => x.gb > 0).length;
    if (gpuN > 1) push(label(S, "split") + (s.splitMode === "memory" ? "across all devices" : "fastest first") + S.ink3(" · s changes it"));
    if (s.code && (!s.fit?.fits || gpuN <= 1)) push(label(S, "invite") + S.bold(`pooled join ${fmtCode(s.code)}`) + S.ink3(" on the other computer"));
    const notes = [];
    if (s.fit && !s.fit.fits) notes.push(`Needs ${Math.max(1, Math.round(s.fit.shortGB))} GB more: one more device, or press l to lend more.`);
    if (s.fit?.fits && s.fit.fellBack) notes.push(`${s.fit.ctxNote}. Lend ${Math.max(1, Math.round(s.fit.needGB - s.fit.haveGB))} GB more (l) or add a device for ${kOf(s.fit.want)}.`);
    const self = s.devices.find((x) => x.self);
    notes.push(...splitLines(splitAdvice(s, lib), { model: lbl(s.model), selfName: self?.name, gb: s.pledge.gb }));
    if (s.splitMode === "memory" && gpuN > 1 && s.fit?.fits && onePledgeHolds(s, lib)) notes.push("Spreading over the network is slower per token; it uses less memory on each device.");
    if (notes.length) { blank(); for (const n of notes) for (const l of wrap(n, W - 2)) push(I + S.ink3(l)); }
  }
  if (online) {
    if (s.ctxNote) push(label(S, "context") + S.ink2(s.ctxNote));
    push(label(S, "chat") + "press c, or " + S.bold(`pooled chat ${fmtCode(s.code)}`) + S.ink3(" on any computer"));
    push(label(S, "api") + S.bold(`pooled serve ${fmtCode(s.code)}`));
    const gpu = s.devices.filter((d) => d.gb > 0 && d.range);
    const rtts = gpu.map((d) => d.rtt).filter((x) => Number.isFinite(x) && x > 0);
    const hint = hopHint(gpu.length, rtts.length ? Math.round(rtts.reduce((a, b) => a + b, 0) / rtts.length / 2) : null);
    if (hint) push(label(S, "network") + S.ink3(hint));
  }
  if (events.length && s.step !== "pick") {
    blank();
    for (const e of events.slice(0, 3)) push(I + S.ink3(`${new Date(e.t).toTimeString().slice(0, 5)}  ${e.text}`));
  }
  if (s.notice && s.notice !== s.fit?.note) { blank(); for (const l of wrap(s.notice, W - 2)) push(I + (/fail/i.test(s.notice) ? S.err(l) : S.ink3(l))); }
  blank();
  if (s.step === "room") {
    const ok = canStart(s).ok;
    push(I + S.keys(W < 69 ? [["enter", "start", ok ? "primary" : "off"], ["i", "copy invite"], ["q", "quit"]]
      : [["enter", "start", ok ? "primary" : "off"], ["i", "copy invite"], ["m", "model"], ["l", "lend"], ...(s.devices.filter((x) => x.gb > 0).length > 1 ? [["s", "split"]] : []), ["q", "quit"]], { width: W - 2 }));
  } else if (s.step === "starting") push(I + S.keys([["q", "cancel and close the room"]], { width: W - 2 }));
  else push(I + S.keys([["c", "chat here", "primary"], ["i", "copy invite"], ["r", "rebalance"], ["q", "close room"]], { width: W - 2 }));
  return L.map((l) => clip(l.replace(/ +$/, ""), W));
}
const loadNote = (ld) => (ld.from && ld.from !== "disk" && ld.fetched < ld.total ? `${gbText(ld.fetched)} of ${gbText(ld.total)} from ${ld.from}` : ld.fetched < ld.total ? "from disk" : "onto the GPU");

// the room's devices for the screen, from the node: host first
export function devicesFrom(node, lib, { pct = new Map(), ranges = null } = {}) {
  const out = [{ id: "self", name: node.name, kind: deviceKind(node.meta, true), gb: +lib.pledgeGB(node.meta) || 0, self: true, meta: node.meta, pct: pct.get(node.name) ?? null, range: ranges?.[node.name] || null, rtt: null }];
  for (const [id, e] of node.conns) {
    if (!e?.meta || e.meta.api) continue;
    const gpu = !!e.meta.webgpu;
    out.push({ id, name: e.name || id, kind: deviceKind(e.meta), gb: gpu ? +lib.pledgeGB(e.meta) || 0 : null, self: false, meta: e.meta, pct: pct.get(e.name) ?? null, range: ranges?.[e.name] || null,
      rtt: Number.isFinite(e.rtt) ? e.rtt : null });
  }
  return out;
}
