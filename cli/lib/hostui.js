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
// pledgeGB, nodeCtxFor), passed in so tests can use the repo's modules directly.

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
export function modelNeedGB(lib, key, ctxAsk = 0) {
  const rb = lib.roomBytes(key, lib.nodeCtxFor(key, ctxAsk), "f16");
  return rb ? r1((rb.L * rb.layerBytes + rb.hostBytes) / GiB) : lib.NEED_GB[key] ?? null;
}

// Whether the room's pledges hold `model`, as the room page decides it (room.js roomFitFor): devices
// host first ({ name, meta, self }), each pledge through room/pledge.js pledgeGB.
// spareGB: how much more each device could lend (the host's own headroom; others unknown: 0).
// -> { fits, needGB, haveGB, shortGB, note }
export function roomFitNow(lib, { model, devices, ctxAsk = 0, spareGB = [] }) {
  const pl = devices.map((d) => +lib.pledgeGB(d.meta) || 0);
  const haveGB = r1(pl.reduce((a, b) => a + b, 0));
  const needGB = modelNeedGB(lib, model, ctxAsk);
  const rb = lib.roomBytes(model, lib.nodeCtxFor(model, ctxAsk), "f16");
  const label = shortLabel(lib, model);
  if (!rb) {
    const fits = needGB != null && haveGB >= needGB;
    const shortGB = fits ? 0 : r1((needGB || 0) - haveGB);
    return { fits, needGB, haveGB, shortGB, note: fits ? "" : `This room is ${shortGB} GB short for ${label}: add a device or raise a pledge.` };
  }
  const fit = lib.roomFit(rb.L, pl.map((g) => g * GiB), rb.layerBytes, rb.hostBytes);
  if (fit.fits) return { fits: true, needGB, haveGB, shortGB: 0, note: "" };
  const shortGB = lib.gbUp(lib.shortBy ? lib.shortBy(fit, spareGB) : fit.short);
  const note = lib.shortNote(label, fit, devices.map((d) => d.name), spareGB).replace(/\. Add a device/, ": add a device");
  return { fits: false, needGB, haveGB, shortGB, note };
}

// The picker's rows: every model a node can host, smallest first, with its download and its need.
// pulled: Set of model keys on disk; pledgeGB: what this computer lends (for "fits here alone")
export function modelRows(lib, { keys, pulled = new Set(), pledgeGB = 0, ctxAsk = 0 }) {
  return keys.map((key) => {
    const needGB = modelNeedGB(lib, key, ctxAsk);
    const alone = roomFitNow(lib, { model: key, devices: [{ name: "this computer", meta: { contribGB: pledgeGB, webgpu: true } }], ctxAsk });
    return { key, label: lib.MODELS[key].label, fileBytes: lib.FILES?.[key]?.bytes || null, pulled: pulled.has(key), needGB, fitsAlone: alone.fits };
  }).sort((a, b) => (a.needGB ?? 99) - (b.needGB ?? 99));
}

// the model to preselect: the largest this computer holds alone, else the smallest
export function recommendModel(rows) {
  const fits = rows.filter((r) => r.fitsAlone);
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
//   step: "confirm" | "pick" | "pledge" | "room" | "starting" | "online",
//   model, rows, sel, pledge: { gb, max, totalGB, typed }, fixed: { model, pledge } (given as flags),
//   devices: [{ id, name, kind, gb, self, pct, range }], lobby: [{ id, line }],
//   dl: { key, state: "none" | "ask" | "running" | "done" | "error" | "stream", done, total, bps, error },
//   fit, flags: { start, wait, chat }, notice, split, code, link
// }
export function initialState({ rows, model = null, pledge, fixedPledge = false, flags = {}, pulled = new Set(), code = "", link = "", yes = false, noPull = false }) {
  const s = {
    step: "pick", rows, model, sel: 0, pledge: { ...pledge, typed: "" }, fixed: { model: !!model, pledge: !!fixedPledge },
    devices: [], lobby: [], dl: { key: null, state: "none", done: 0, total: 0, bps: null, error: null },
    fit: null, flags: { start: !!flags.start, wait: flags.wait || 0, chat: !!flags.chat }, notice: "", split: null, code, link, yes, noPull,
  };
  const rec = model || recommendModel(rows);
  s.sel = Math.max(0, rows.findIndex((r) => r.key === rec));
  if (model) {
    s.step = s.fixed.pledge ? "room" : "pledge";
    if (!pulled.has(model)) {
      if (noPull) s.dl = { ...s.dl, key: model, state: "stream" };
      else if (yes) s.dl = { ...s.dl, key: model, state: "running" };
      else { s.dl = { ...s.dl, key: model, state: "ask" }; s.step = "confirm"; }
    } else s.dl = { ...s.dl, key: model, state: "done" };
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

const nextAfterModel = (s) => (s.fixed.pledge ? "room" : s.pledgeDone ? "room" : "pledge");

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
      if (k === "y" || k === "enter") { t.dl = { ...t.dl, state: "running", done: 0 }; fx.push({ do: "pull", key: t.dl.key }); t.step = nextAfterModel(t); }
      else if (k === "n") { t.dl = { ...t.dl, state: "stream" }; fx.push({ do: "stream", key: t.dl.key }); t.step = nextAfterModel(t); }
      break;
    }
    case "pick": {
      if (k === "up") t.sel = (t.sel - 1 + t.rows.length) % t.rows.length;
      else if (k === "down") t.sel = (t.sel + 1) % t.rows.length;
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
        t.step = nextAfterModel(t);
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
      else if (k === "p") { t.step = "pledge"; t.pledge = { ...t.pledge, typed: "" }; }
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
      else if (k === "enter" || k === "r") fx.push({ do: "redeal" });
      else if (k === "a" && t.lobby.length) fx.push({ do: "allow", id: t.lobby[0].id });
      else if (k === "d" && t.lobby.length) fx.push({ do: "deny", id: t.lobby[0].id });
      break;
    }
  }
  return { state: t, fx };
}

// ---------------- drawing ----------------
// c: colors ({ dim, bold, green, yellow, red, cyan, inv } functions; identity without color)
export function colors(on) {
  const w = (a, b = 0) => (s) => (on ? `\x1b[${a}m${s}\x1b[${b || (a === 1 || a === 2 ? 22 : 39)}m` : String(s));
  return { on, dim: w(2), bold: w(1), green: w(32), yellow: w(33), red: w(31), cyan: w(36), magenta: w(35) };
}
// the visible width of a string with escape codes, and a cut to `width` columns that keeps them balanced
export const visible = (s) => String(s).replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "");
export function clip(s, width) {
  s = String(s);
  if (visible(s).length <= width) return s;
  let out = "", n = 0;
  for (let i = 0; i < s.length;) {
    const m = /^\x1b\[[0-9;?]*[A-Za-z]/.exec(s.slice(i));
    if (m) { out += m[0]; i += m[0].length; continue; }
    if (n >= width - 1) { out += "…"; break; }
    out += s[i]; n++; i++;
  }
  return out + (/\x1b\[/.test(s) ? "\x1b[0m" : "");
}
// words to lines of at most w columns
export function wrap(text, w) {
  const out = [];
  let line = "";
  for (const word of String(text).split(/\s+/).filter(Boolean)) {
    if (line && line.length + 1 + word.length > w) { out.push(line); line = word; }
    else line = line ? `${line} ${word}` : word;
  }
  if (line) out.push(line);
  return out;
}
const pad = (s, n) => s + " ".repeat(Math.max(0, n - visible(s).length));
const gbText = (b) => (b >= GiB ? `${(b / GiB).toFixed(1)} GB` : `${Math.round(b / 2 ** 20)} MB`);
function bar(pct, w, c) {
  const f = Math.round((Math.max(0, Math.min(100, pct)) / 100) * w);
  return c.green("█".repeat(f)) + c.dim("░".repeat(w - f));
}
export const fmtCode = (x) => (String(x || "").length === 6 ? `${x.slice(0, 3)}-${x.slice(3)}` : String(x || ""));

// the screen for a state -> lines (each cut to width)
export function render(s, { width = 80, c = colors(false), lib, spin = "" } = {}) {
  const L = [];
  const W = Math.max(40, width) - 1;
  const label = (k) => (lib ? shortLabel(lib, k) : k);
  L.push(`${c.bold("pooled host")} ${c.dim("·")} room ${c.bold(c.cyan(fmtCode(s.code)))}${s.model ? ` ${c.dim("·")} ${label(s.model)}` : ""}`);
  if (s.link) L.push(`${c.dim("invite")}  ${s.link}`);
  L.push(c.dim("─".repeat(Math.min(W, 72))));
  const dlLine = () => {
    const d = s.dl;
    if (!d.key || d.key !== s.model) return null;
    if (d.state === "running") {
      const pct = d.total ? Math.floor((d.done / d.total) * 100) : 0;
      const eta = d.total && d.bps ? Math.max(0, Math.round((d.total - d.done) / d.bps)) : null;
      return `${c.dim("download")} ${bar(pct, 20, c)} ${String(pct).padStart(3)}%  ${gbText(d.done)} / ${d.total ? gbText(d.total) : "?"}${d.bps ? `  ${gbText(d.bps)}/s` : ""}${eta != null ? `  ${eta >= 60 ? `${Math.floor(eta / 60)}m${String(eta % 60).padStart(2, "0")}s` : `${eta}s`} left` : ""}`;
    }
    if (d.state === "error") return c.red(`download failed: ${d.error}`);
    if (d.state === "stream") return c.dim("not downloaded: each device streams its layers from Hugging Face");
    return null;
  };
  if (s.step === "confirm") {
    const row = s.rows.find((r) => r.key === s.dl.key);
    L.push(`${c.bold(s.dl.key)} is not downloaded${row?.fileBytes ? ` (${gbText(row.fileBytes)})` : ""}. Download it now? ${c.bold("[Y/n]")}`);
    L.push(c.dim("  n streams this computer's layers from Hugging Face on every start instead"));
  } else if (s.step === "pick") {
    L.push(`${c.bold("Choose a model")} ${c.dim("↑/↓ then Enter")}`);
    const rec = recommendModel(s.rows);
    for (let i = 0; i < s.rows.length; i++) {
      const r = s.rows[i], on = i === s.sel;
      const have = r.pulled ? c.green("✓ downloaded") : r.fileBytes ? c.dim(`${gbText(r.fileBytes)} download`) : c.dim("download");
      const need = r.needGB != null ? `needs ${String(r.needGB).padStart(4)} GB` : "";
      const tag = r.key === rec ? c.cyan("recommended") : r.fitsAlone ? c.dim("fits here") : c.dim("needs more devices");
      L.push(`${on ? c.cyan("❯") : " "} ${pad(on ? c.bold(label(r.key)) : label(r.key), 18)} ${pad(have, 18)} ${pad(need, 15)} ${tag}`);
    }
    L.push(c.dim(`  "needs": all devices together, at the room's context. this computer lends ${s.pledge.gb} GB${s.model ? " · Esc: back" : ""}`));
  } else if (s.step === "pledge") {
    const shown = s.pledge.typed ? `${s.pledge.typed}${c.dim("_")}` : c.bold(`${s.pledge.gb}`);
    L.push(`${c.bold("How much memory does this computer lend?")} ${c.dim("←/→ or type, then Enter")}`);
    L.push(`  ${c.cyan("◀")} ${shown} GB ${c.cyan("▶")}   ${c.dim(`of ${s.pledge.totalGB ? `${Math.round(s.pledge.totalGB)} GB on this GPU` : "this GPU"}; at most ${s.pledge.max} GB`)}`);
    if (s.model) L.push(c.dim(`  ${label(s.model)} needs ${s.rows.find((r) => r.key === s.model)?.needGB ?? "?"} GB over the whole room`));
  }
  if (s.step === "room" || s.step === "starting" || s.step === "online" || ((s.step === "pick" || s.step === "pledge" || s.step === "confirm") && s.devices.length > 1)) {
    if (s.step === "room" || s.step === "starting" || s.step === "online") {
      const have = s.dl.key === s.model && s.dl.state === "done" ? c.green("✓ downloaded") : "";
      L.push(`${c.dim("model  ")} ${s.model ? c.bold(label(s.model)) : c.dim("none yet")}${have ? `  ${have}` : ""}${s.step === "room" ? c.dim("   m: change") : ""}`);
      const dl = dlLine(); if (dl) L.push(`${" ".repeat(8)}${dl}`);
      L.push(`${c.dim("lending")} ${c.bold(`${s.pledge.gb} GB`)}${s.pledge.totalGB ? c.dim(` of ${Math.round(s.pledge.totalGB)} GB`) : ""}${s.step === "room" ? c.dim("   p: change") : ""}`);
    }
    L.push("");
    L.push(c.bold(`Devices (${s.devices.length})`));
    for (const d of s.devices) {
      const dot = d.pct != null && d.pct < 100 ? c.yellow("◐") : s.step === "online" && d.range ? c.green("●") : c.dim("●");
      let right = "";
      if (s.step === "starting" && d.pct != null) right = `${bar(d.pct, 14, c)} ${String(d.pct).padStart(3)}%`;
      else if (d.range) right = c.dim(`layers ${d.range}`);
      L.push(`  ${dot} ${pad(d.self ? c.bold(d.name) : d.name, 18)} ${pad(c.dim(d.kind), 14)} ${pad(d.gb != null ? `${d.gb} GB` : c.dim("chat only"), 8)} ${right}`);
    }
    if (s.lobby.length) {
      L.push("");
      L.push(c.yellow(c.bold(`Waiting to join (${s.lobby.length})`)));
      s.lobby.slice(0, 3).forEach((r, i) => L.push(`  ${c.yellow("?")} ${r.line}${i === 0 ? c.dim("   a: allow  d: deny") : ""}`));
      if (s.lobby.length > 3) L.push(c.dim(`  and ${s.lobby.length - 3} more`));
    }
  }
  if (s.step === "room" && s.model && s.fit) {
    L.push("");
    const pct = s.fit.needGB ? Math.min(100, (s.fit.haveGB / s.fit.needGB) * 100) : 0;
    L.push(`${c.dim("room   ")} ${bar(s.fit.fits ? 100 : pct, 20, c)} ${s.fit.haveGB} GB pledged, ${label(s.model)} needs ${s.fit.needGB} GB`);
    if (!s.fit.fits) for (const l of wrap(s.fit.note, W)) L.push(c.yellow(l));
  }
  if (s.step === "online" && s.split) { L.push(""); L.push(`${c.green("● online")}  ${c.dim(s.split)}`); }
  if (s.step === "starting") { L.push(""); L.push(`${spin || "…"} ${c.bold("loading the model")} ${c.dim("each device loads its layers")}`); }
  if (s.notice && s.notice !== s.fit?.note) for (const l of wrap(s.notice, W)) L.push(c.yellow(l));
  // the keys, last
  L.push("");
  const keys = [];
  if (s.step === "room") {
    const cs = canStart(s);
    if (cs.ok) keys.push(c.green(c.bold("Enter: start")));
    else if (s.flags.start || s.flags.wait) keys.push(c.dim(s.flags.wait > 1 ? `starts by itself with ${s.flags.wait} devices` : "starts by itself once it fits"));
    else keys.push(c.dim("Enter: start (when the room fits)"));
    keys.push("m: model", "p: pledge");
  }
  if (s.step === "online") keys.push(c.bold("c: chat here"), "Enter: re-deal");
  keys.push("q: quit");
  L.push(keys.join(c.dim("  ·  ")));
  return L.map((l) => clip(l.replace(/ +$/, ""), W));
}

// the room's devices for the screen, from the node: host first
export function devicesFrom(node, lib, { pct = new Map(), ranges = null } = {}) {
  const out = [{ id: "self", name: node.name, kind: deviceKind(node.meta, true), gb: +lib.pledgeGB(node.meta) || 0, self: true, meta: node.meta, pct: pct.get(node.name) ?? null, range: ranges?.[node.name] || null }];
  for (const [id, e] of node.conns) {
    if (!e?.meta || e.meta.api) continue;
    const gpu = !!e.meta.webgpu;
    out.push({ id, name: e.name || id, kind: deviceKind(e.meta), gb: gpu ? +lib.pledgeGB(e.meta) || 0 : null, self: false, meta: e.meta, pct: pct.get(e.name) ?? null, range: ranges?.[e.name] || null });
  }
  return out;
}
