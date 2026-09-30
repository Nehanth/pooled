// pooled join / pooled host: lend this computer's GPU to a room from the terminal, on the headless
// room node (packages/room-node: Pooled's engine on Dawn, the room protocol over node-datachannel).
// Foreground only: it runs until Ctrl-C, which leaves the room and frees the GPU.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import os from "node:os";
import { parseLendArgs, parseGb, detectMemory, memoryRule, afterCheck, formatStatus, tpsFromStats, explainError, versionFromBye,
  hostable, ctxNote, autoRedeal, fmtGb, fmtCode, passCounter, deviceName, UsageError, HELP_JOIN, HELP_HOST, needsRoom, loadText } from "./lend.js";
import { dawnLoader, quietLoader, driverLog } from "./dawn.js";
import { cleanText } from "./common.js";
import { roomCodeFrom, roomKeyFrom } from "./room.js";
import { modelsDir, ensureModelsDir, modelState, fmtBytes, resolveModel } from "./cache.js";
import { roomFitNow, devicesFrom } from "./hostui.js";
import { askLine, askYesNo, colorOn, termCaps, liveRegion, keysOf as KEYS } from "./tui.js";
import { style, detectTheme, gbNum } from "./style.js";
import { joinScreen } from "./joinui.js";
import * as HOSTUI from "./hostui.js";

const ROOM_URL = "https://pooled.run/r/";

// the room node: the bundle built into dist/ (npm run build; what the npm package ships), else the
// package in a Pooled checkout
export async function loadRoomNode() {
  const tries = [new URL("../dist/room-node.js", import.meta.url), new URL("../../packages/room-node/index.js", import.meta.url)];
  let last = null;
  for (const u of tries) {
    try { return await import(u.href); }
    catch (e) { if (e?.code !== "ERR_MODULE_NOT_FOUND" || !String(e.message).includes(u.pathname.split("/").pop())) throw e; last = e; }
  }
  const e = new Error("this copy of pooled has no room node built in");
  e.type = "room-node-missing"; e.cause = last;
  e.hint = "In a Pooled checkout, build it first: (cd cli && npm ci && npm run build)";
  throw e;
}

// a test allocation: how many GB the GPU really gives (buffers of up to 1 GB, destroyed at once)
async function testAlloc(gb) {
  const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!a) return null;
  const chunk = Math.min(a.limits.maxBufferSize, 2 ** 30);
  const dev = await a.requestDevice({ requiredLimits: { maxBufferSize: chunk } });
  const bufs = [], want = gb * 2 ** 30;
  let got = 0;
  try {
    while (got < want) {
      const size = Math.min(chunk, Math.ceil((want - got) / 4) * 4);
      dev.pushErrorScope("out-of-memory");
      let b = null;
      try { b = dev.createBuffer({ size, usage: GPUBufferUsage.STORAGE }); } catch {}
      const err = await dev.popErrorScope();
      if (err || !b) { try { b?.destroy(); } catch {} break; }
      bufs.push(b); got += size;
    }
  } finally {
    for (const b of bufs) try { b.destroy(); } catch {}
    try { dev.destroy(); } catch {}
  }
  return got / 2 ** 30;
}

// the terminal: log lines, and one status line kept at the bottom (a TTY), or plain lines
function makeOut({ jsonLog, quiet }) {
  // a status line redrawn in place needs a terminal that takes escapes (not TERM=dumb)
  const tty = !!process.stderr.isTTY && process.env.TERM !== "dumb" && !jsonLog;
  let status = "", shown = false;
  // screen: a whole live screen (pooled join at a terminal) instead of one status line
  let screen = null, region = null, lastS = null;
  const clear = () => { if (region) { region.clear(); return; } if (tty && shown) { process.stderr.write("\r\x1b[K"); shown = false; } };
  const draw = () => { if (region) { if (lastS) region.render(screen(lastS)); return; } if (tty && status) { process.stderr.write("\r\x1b[K" + status); shown = true; } };
  const line = (msg, level) => {
    msg = String(msg).split("\n").map((l) => cleanText(l)).join("\n  ");
    if (jsonLog) { process.stderr.write(JSON.stringify({ t: new Date().toISOString(), level, msg }) + "\n"); return; }
    if (region) { region.log(`${screenStyle.ink3(new Date().toTimeString().slice(0, 8))} ${level === "error" ? screenStyle.err(msg) : screenStyle.ink3(msg)}`); return; }
    clear();
    process.stderr.write(`${new Date().toTimeString().slice(0, 8)} ${msg}\n`);
    draw();
  };
  let screenStyle = null;
  return {
    tty,
    // lines(s) -> the screen for a status state; ST: its style
    useScreen(lines, ST) { if (!tty) return; screen = lines; screenStyle = ST; region = liveRegion(process.stderr); },
    redraw() { if (region && lastS) region.render(screen(lastS)); },
    close() { if (region) { region.close(); region = null; screen = null; } },
    log: (msg, level = "info") => { if (!quiet || level === "error") line(msg, level); },
    print: (text) => {
      if (jsonLog) { process.stderr.write(JSON.stringify({ t: new Date().toISOString(), level: "info", msg: String(text).replace(/\n\s*/g, " | ") }) + "\n"); return; }
      if (region) { region.log(text); return; }
      clear(); process.stderr.write(text + "\n"); draw();
    },
    hint: (text) => { if (jsonLog) process.stderr.write(JSON.stringify({ t: new Date().toISOString(), level: "hint", msg: text }) + "\n"); else process.stderr.write(`  ${text}\n`); },
    status(s) {
      if (jsonLog) { process.stderr.write(JSON.stringify({ t: new Date().toISOString(), level: "status", ...s }) + "\n"); return; }
      if (region) { lastS = s; status = ""; region.render(screen(s)); return; }
      const text = formatStatus(s, tty ? (process.stderr.columns || 100) - 1 : 0);
      if (tty) { status = text; draw(); }
      else process.stderr.write(`${new Date().toTimeString().slice(0, 8)} ${text}\n`);
    },
    done() { clear(); status = ""; },
  };
}

function fail(out, err, ctx) {
  const x = explainError(err, ctx);
  out.done();
  out.log(`pooled: ${x.message}`, "error");
  if (x.hint) out.hint(x.hint);
  if (process.env.POOLED_DEBUG && err?.stack) process.stderr.write(err.stack + "\n");
  return x.code;
}

// GPU + memory, before anything joins: -> { rn, pledge, rule, adapterName }
export async function prepare(opts, out) {
  const rn = await loadRoomNode();
  // the GPU driver's own stderr lines are hidden unless --verbose (POOLED_VERBOSE=1)
  const verbose = opts.verbose || !!process.env.POOLED_VERBOSE;
  const loader = verbose ? dawnLoader(opts.cmd) : quietLoader(dawnLoader(opts.cmd));
  await rn.setupNode({ webgpu: loader });
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw Object.assign(new Error("no WebGPU adapter"), { type: "no-adapter", driver: verbose ? "" : driverLog() });
  const info = adapter.info || {};
  const adapterName = [...new Set([info.vendor, info.architecture || info.device].filter(Boolean))].join(" ") || "GPU";
  const mem = detectMemory({
    run: (cmd, args) => execFileSync(cmd, args, { encoding: "utf8", timeout: 5000, stdio: ["ignore", "pipe", "ignore"] }),
    read: (p) => readFileSync(p, "utf8"), totalmem: os.totalmem, freemem: os.freemem,
  });
  let rule = memoryRule(mem, opts.gb, { maxBufGB: adapter.limits.maxBufferSize / 2 ** 30 });
  if (rule.low) throw Object.assign(new Error(rule.why), { type: "low-memory" });
  // a discrete GPU's own memory: allocate it for a moment to make sure it is there. Not on unified
  // memory: the OS's numbers are the memory itself there, and touching 64 GB of it takes ~20 s (GB10)
  if (opts.check && !opts.gb?.gb && mem.kind === "discrete") {
    const t0 = Date.now();
    const got = await testAlloc(rule.gb).catch(() => null);
    if (Date.now() - t0 > 3000) out.log(`test allocation of ${rule.gb} GB took ${((Date.now() - t0) / 1000).toFixed(1)} s`);
    rule = afterCheck(rule, got);
    if (rule.low) throw Object.assign(new Error(rule.why), { type: "low-memory" });
  }
  return { rn, loader, rule, adapterName, mem };
}

function header(out, { title, adapterName, mem, rule, hosting = false }) {
  const gpu = mem.kind === "discrete" ? `${mem.name} · ${fmtGb(mem.totalGB)} GB` : mem.kind === "unified" ? `${mem.name || adapterName} · ${fmtGb(mem.totalGB)} GB unified memory` : adapterName;
  out.print(`${title}
  GPU      ${gpu}
  lending  ${rule.gb} GB  (${rule.why}${rule.why.startsWith("--gb") || rule.why.includes("--gb sets it") ? "" : "; --gb to change"})
  ${hosting
    ? "every device holding layers sees the hidden states of what is asked here (they carry the prompts\n  and answers), and whoever is in can ask: share the invite link only with people you trust"
    : "whoever the host lets in can use what this computer lends, and it sees the room's hidden states\n  (they carry the prompts and answers): lend to rooms you trust"}
  Ctrl-C leaves the room and frees the GPU`);
}

// one line to pick a number of GB: ←/→ (or ↑/↓) step by 1, digits type it, Enter takes it -> GB | null (Ctrl-C)
function pledgePrompt({ def, max, totalGB }) {
  const ST = style({ stream: process.stderr });
  let gb = def, typed = "";
  const total = Math.round(totalGB || max);
  const draw = () => process.stderr.write(`\r\x1b[K  ${ST.ink3("lend".padEnd(10))}${ST.bar(gb / (total || 1), 20)}  ${typed ? ST.bold(typed) + ST.rev(" ") : ST.bold(`${gb} GB`)}${ST.ink3(` of ${total} GB`)}   ${ST.keys([[ST.g.lr, "1 GB"], ["type", "a number"], ["enter", "lend", "primary"]])}`);
  return new Promise((resolve) => {
    const done = (v) => { process.stdin.off("data", on); try { process.stdin.setRawMode(false); } catch {} process.stdin.pause(); process.stderr.write("\n"); resolve(v); };
    const on = (b) => {
      for (const k of KEYS(b)) {
        if (k === "ctrl-c") return done(null);
        if (k === "left" || k === "down") { gb = Math.max(1, (typed ? +typed : gb) - 1); typed = ""; }
        else if (k === "right" || k === "up") { gb = Math.min(max, (typed ? +typed : gb) + 1); typed = ""; }
        else if (/^[0-9]$/.test(k) && typed.length < 3) typed += k;
        else if (k === "backspace") typed = typed.slice(0, -1);
        else if (k === "enter") { if (typed) gb = Math.max(1, Math.min(max, +typed)); return done(gb); }
      }
      draw();
    };
    try { process.stdin.setRawMode(true); } catch {}
    process.stdin.resume(); process.stdin.on("data", on);
    draw();
  });
}

// the same as a typed line (TERM=dumb) -> GB | null
async function pledgeLine({ def, max }) {
  for (;;) {
    const a = await askLine(`  lend how many GB? 1-${max} (Enter: ${def}): `);
    if (a == null) return null;
    if (!a.trim()) return def;
    const n = Number(a.trim());
    if (Number.isFinite(n) && n >= 1) return Math.min(max, n);
    process.stderr.write(`  type a number from 1 to ${max}\n`);
  }
}

// leave the room within about a second: the room node says "leaving" at once, and closing its links
// (one being dialed to a host that is gone can take seconds) must not hold the process
export const CLOSE_WAIT_MS = 800;
export async function closeSoon(node, ms = CLOSE_WAIT_MS) {
  if (!node) return;
  let t;
  await Promise.race([Promise.resolve().then(() => node.close()).catch(() => {}), new Promise((r) => { t = setTimeout(r, ms); })]);
  clearTimeout(t);
}

// ---------------- pooled join ----------------
async function runJoin(opts, out) {
  let code = opts.code;
  // one name for the whole run, the same after a restart (the host re-seats a device by its name)
  const name = opts.name || deviceName(os.hostname());
  let hostName = null;   // the host this run joined: a rejoin only goes back to a host of that name
  const { rn, loader, rule, adapterName, mem } = await prepare(opts, out);
  let ST = null, spinAt = 0;
  if (out.tty) {
    // a terminal: the join screen (lib/joinui.js), redrawn in place
    await detectTheme();
    ST = style({ stream: process.stderr });
    process.stderr.write(`\n  ${ST.ink3("Every device in a room sees what is asked there. Lend to rooms you trust.")}\n`);
    out.useScreen((st) => joinScreen(st, { S: ST, cols: process.stderr.columns || 80, spin: ST.spin(spinAt) }), ST);
  } else header(out, { title: `pooled join · room ${fmtCode(code)}`, adapterName, mem, rule });
  let pass = null;   // what the host gave this device when it let it in: back in without asking
  const passes = passCounter();
  const S = { code, phase: "connecting", devices: null, range: null, model: null, tps: null, passes: 0, pct: null, load: null, tries: 0, signaling: true, answering: false,
    dl: null, hostName: null, modelLabel: null, lobbyAt: null, onlineAt: null, L: null,
    you: { name: opts.name || deviceName(os.hostname()), gpu: mem.kind === "unknown" ? adapterName : (mem.name || adapterName), gb: rule.gb, totalGB: mem.totalGB } };
  let node = null, leaving = false, ended = null, rejoinP = null;
  const finish = (x) => { if (!ended) { ended = x; wake(x); } };
  let wake = () => {};
  const endP = new Promise((r) => { wake = r; });
  let hostGone = false, hinted = false, prompting = false;
  const tick = () => {
    if (prompting) return;
    if (node && S.phase !== "rejoining" && S.phase !== "leaving" && S.phase !== "connecting") {
      const st = node.status();
      S.devices = st.devices.length || null; S.range = st.range; S.model = st.model; S.passes = passes(node, st.passes); S.signaling = st.signaling;
      S.phase = hostGone ? "hostgone" : node.admission === "lobby" ? "lobby" : st.loading ? "loading" : st.degraded ? "degraded" : st.online ? (!st.range ? "guest" : S.answering ? "answering" : "online") : st.range ? "ready" : "waiting";
      S.hostName = node.hostName || S.hostName;
      const hm = node.conns.get(node.ai.hostId)?.meta || {};
      const mk = S.model || (hm.model && rn.MODELS[hm.model] ? hm.model : null);
      if (mk) S.modelLabel = rn.MODELS[mk].label.split("·")[0].trim();
      // the host's model (from its hello, which may come a moment after the link opens, or change
      // when the host picks another): download it now, before the host deals, unless --no-pull
      const hostKey = hm.model && rn.MODELS[hm.model] ? hm.model : null;
      if (hostKey && pledged && !opts.noPull && !onDisk.has(hostKey) && pulling?.key !== hostKey) ensurePulled(hostKey).catch(() => {});
      if (S.phase === "lobby") S.lobbyAt ||= Date.now();
      if (S.phase === "online" || S.phase === "answering") S.onlineAt ||= Date.now();
      S.you.gb = node.pledgeGB || S.you.gb;
    }
    out.status(S);
  };
  // the host's model not on this computer: download it whole first (as pooled pull), so this load and
  // every later one reads from disk; --no-pull streams this device's layers from Hugging Face instead.
  // A deal that comes while it downloads waits for it (the room node's beforeLoad)
  let pulling = null, pledged = null;
  const onDisk = new Set();
  const ensurePulled = async (key) => {
    if (opts.noPull || onDisk.has(key) || !rn.MODELS[key] || !rn.FILES?.[key]) return;
    if (modelState(opts.modelDir, key, rn.MODELS, rn.FILES, rn.LOCAL).pulled) { onDisk.add(key); return; }
    await pledged;
    if (pulling?.key !== key) pulling = { key, p: pullFirst(key) };
    await pulling.p;
  };
  const pullFirst = async (key) => {
    const { pullWithProgress } = await import("./pullrun.js");
    const home = opts.modelDir.startsWith(os.homedir()) ? "~" + opts.modelDir.slice(os.homedir().length) : opts.modelDir;
    const say = `${rn.MODELS[key].label.split("·")[0].trim()} isn't downloaded yet (${fmtBytes(rn.FILES[key].bytes)}). Downloading it into ${home} so this and later joins load from disk.`;
    let r;
    if (out.tty) {
      // the join screen shows the download row (the same row as pooled pull)
      out.log(say);
      S.dl = { state: "running", done: 0, total: rn.FILES[key].bytes, bps: null };
      r = await pullWithProgress(rn, key, opts.modelDir, { quiet: true, onProgress: (p) => { S.dl = { state: "running", done: p.done, total: p.total || S.dl.total, bps: p.bps }; } });
      S.dl = null;
    } else {
      prompting = true; out.done();
      out.print(say);
      r = await pullWithProgress(rn, key, opts.modelDir, { quiet: opts.jsonLog });
      prompting = false;
    }
    if (r.ok) { onDisk.add(key); out.log(`${key} downloaded: loading from disk`); }
    else if (!r.aborted) out.log(`download failed (${cleanText(r.error?.message || "", 200)}): streaming the layers from Hugging Face instead; pooled pull ${key} resumes it`, "error");
    tick();
  };
  const joinOnce = () => rn.joinRoom(code, { pledgeGB: lendGB, name, signal: opts.signal, modelDir: opts.modelDir, setup: { webgpu: loader },
    expectHost: hostName, key: opts.key, pass, log: (m) => out.log(m), beforeLoad: ensurePulled });
  const attach = (n) => {
    n.on("loadprogress", (pct) => { S.pct = pct; });
    n.on("loadstat", (x) => { S.load = x; });
    n.on("loaded", (x) => {
      S.pct = null; S.load = null; out.log(`holding layers ${x.range[0]}-${x.range[1] - 1} of ${x.model} (loaded in ${x.s.toFixed(1)} s)`);
      // streamed from Hugging Face: say once how to make the next join start from disk
      if (!hinted && opts.noPull && !modelState(opts.modelDir, x.model, rn.MODELS, rn.FILES, rn.LOCAL).pulled) { hinted = true; out.hint(`pooled pull ${x.model} makes the next join start faster (it reads the layers from disk)`); }
    });
    n.on("hostgone", () => { hostGone = true; S.answering = false; });
    n.on("lobby", () => { tick(); });
    n.on("admitted", () => { if (n.pass) pass = n.pass; tick(); });
    n.on("back", () => { hostGone = false; });
    n.on("signaling", (up) => { S.signaling = up; });
    n.on("chat", (d) => {
      if (d.t === "ai-genstart") S.answering = true;
      if (d.t === "ai-gendone") { S.answering = false; const t = tpsFromStats(d.stats); if (t != null) S.tps = t; }
    });
    n.on("version", (v) => { if (v.theyHost) finish({ type: "version", theirs: v.theirs, theyHost: true }); });
    n.on("members", () => { if (!hostName && n.hostName) hostName = n.hostName; });
    n.on("otherhost", (x) => finish({ type: "other-host", ...x }));
    n.on("bye", (reason) => {
      const v = versionFromBye(reason, rn.PROTOCOL);
      finish(v ? { type: "version", theirs: v.theirs, theyHost: true } : Object.assign(new Error(String(reason || "")), { type: "kicked" }));
    });
    // the host closed the room (pooled host q): over, no knocking and no rejoin
    n.on("closed", () => finish({ type: "host-closed" }));
    n.on("roomover", () => { if (!leaving && !rejoinP) rejoinP = rejoin().finally(() => { rejoinP = null; }); });
  };
  // the host did not come back within a minute: start over (join again) with backoff, for --wait
  const rejoin = async () => {
    const old = node; node = null;
    hostName ||= old?.hostName || null;
    // never heard the host's name: there is nothing to tell its room from another under this code
    if (!hostName) { await old?.close().catch(() => {}); finish({ type: "room-over" }); return; }
    S.phase = "rejoining"; S.range = null; S.tries = 0; S.devices = null; S.signaling = true; S.answering = false;
    await old?.close().catch(() => {});
    const t0 = Date.now();
    out.log(`trying to join room ${code} again for up to ${Math.round(opts.waitMs / 60000)} min (only while its host is ${hostName})`);
    while (!leaving && !ended && Date.now() - t0 < opts.waitMs) {
      await new Promise((r) => setTimeout(r, rn.reconnectDelay(S.tries)));
      if (leaving || ended) return;
      S.tries++;
      try {
        node = await joinOnce(); hostGone = false; attach(node); S.phase = "waiting"; S.tries = 0;
        if (node.otherHost) finish({ type: "other-host", ...node.otherHost }); else out.log(`back in room ${code}`);
        return;
      }
      catch (e) {
        if (e?.type === "unavailable-id") continue;
        if (e?.code !== "room-not-found" && e?.type !== "signaling-down") out.log(`rejoin: ${cleanText(e?.message || e, 200)}`);
      }
    }
    if (!leaving) finish({ type: "room-over" });
  };
  // a terminal and no --gb: ask how much to lend before knocking, so the host's "wants to join"
  // line (and its Allow prompt) shows the amount this device really lends, not the default
  let lendGB = rule.gb;
  if ((opts.interactive || opts.lines) && !opts.gb) {
    prompting = true; out.done();
    out.print(`  room     ${fmtCode(code)}`);
    const max = Math.max(rule.gb, memoryRule(mem, { max: true }).gb || rule.gb);
    const gb = opts.interactive ? await pledgePrompt({ def: rule.gb, max, totalGB: mem.totalGB }) : await pledgeLine({ def: rule.gb, max });
    if (gb == null) { out.done(); return 130; }
    lendGB = gb; S.you.gb = gb;
    if (gb !== rule.gb) out.log(`lending ${gb} GB`);
    prompting = false;
  }
  pledged = Promise.resolve();
  const bye = async (sig) => {
    if (leaving) { out.done(); process.exit(130); }
    leaving = true; S.phase = "leaving"; tick();
    out.log(`leaving room ${fmtCode(code)} and freeing the GPU${sig ? " (again to quit at once)" : ""}`);
    await closeSoon(node);
    out.done();
    process.exit(0);
  };
  process.on("SIGINT", () => bye(true)); process.on("SIGTERM", () => bye(false));
  // the spinner redraws the join screen at a terminal
  if (out.tty) setInterval(() => { spinAt++; if (!prompting) tick(); }, 80).unref?.();
  node = await joinOnce();
  attach(node);
  S.phase = "waiting";
  out.log(`reached room ${fmtCode(code)} as ${node.name}${node.server && node.server.spec !== "cloud" ? ` (signaling: ${node.server.label})` : ""}`);
  // the host's hello (its name and model) comes right after the link opens
  await new Promise((r) => setTimeout(r, 400));
  const hm = node.conns.get(node.ai.hostId)?.meta || {};
  const hostModel = hm.model && rn.MODELS[hm.model] ? hm.model : null;
  // q leaves (Ctrl-C too): raw keys once the questions are done
  if (out.tty && process.stdin.isTTY) {
    try { process.stdin.setRawMode(true); process.stdin.resume(); process.stdin.on("data", (b) => { for (const k of KEYS(b)) if (k === "q" || k === "ctrl-c") bye(true); }); } catch {}
  }
  if (hostModel) await ensurePulled(hostModel);
  const timer = setInterval(tick, 1000);
  // plain (non-TTY) output: a status line when the phase changes, and once a minute while it runs
  if (!out.tty) {
    let last = "", lastAt = 0;
    const plain = out.status; out.status = (s) => { const k = `${s.phase}|${s.devices}|${s.range}`; if (k !== last || Date.now() - lastAt > 60000) { last = k; lastAt = Date.now(); plain(s); } };
  }
  tick();
  const r = await endP;
  clearInterval(timer);
  if (!r.ok) await closeSoon(node);
  out.done();
  if (r.ok) return 0;
  if (r.type === "host-closed") { out.print("The host closed the room."); return 0; }
  return fail(out, r, { code, cmd: "join", mine: rn.PROTOCOL });
}

// ---------------- pooled host ----------------
async function runHost(opts, out, prepared = null) {
  const { rn, loader, rule, adapterName, mem } = prepared || await prepare(opts, out);
  if (!hostable(rn.MODELS).includes(opts.model)) throw new UsageError(`unknown model "${opts.model}"; one of: ${hostable(rn.MODELS).join(", ")}`);
  // not on this computer: download it first (--no-pull streams this computer's layers instead)
  if (!modelState(opts.modelDir, opts.model, rn.MODELS, rn.FILES, rn.LOCAL).pulled) {
    if (opts.noPull) out.log(`${opts.model} is not downloaded: streaming this computer's layers from Hugging Face (--no-pull)`);
    else {
      const { pullWithProgress } = await import("./pullrun.js");
      out.log(`${opts.model} is not downloaded: pulling it (${fmtBytes(rn.FILES?.[opts.model]?.bytes)}) into ${opts.modelDir}; --no-pull streams instead`);
      const ac = new AbortController();
      const onInt = () => ac.abort();
      process.on("SIGINT", onInt);
      const r = await pullWithProgress(rn, opts.model, opts.modelDir, { signal: ac.signal, quiet: opts.jsonLog });
      process.off("SIGINT", onInt);
      if (r.aborted) { out.log(`download paused: pooled pull ${opts.model} resumes it`); return 130; }
      if (!r.ok) throw r.error;
      out.log(`${opts.model} downloaded`);
    }
  }
  const cn = ctxNote(rn, opts.model, opts.ctx);
  if (cn) out.log(`--ctx ${opts.ctx}: ${cn}`);
  const node = await rn.createRoom({ model: opts.model, pledgeGB: rule.gb, name: opts.name, signal: opts.signal, modelDir: opts.modelDir, ctx: opts.ctx || 0,
    gate: true, ask: !opts.allowAll, setup: { webgpu: loader }, log: (m) => out.log(m), split: opts.split, ...(opts.roomCode ? { code: opts.roomCode } : {}) });
  const code = node.code;
  // the invite link: its #k= key lets a device in without asking (a room node from before the gate has none)
  const link = `${ROOM_URL}${code}${node.inviteFragment || ""}`;
  header(out, { title: `pooled host · room ${fmtCode(code)} · ${rn.MODELS[opts.model].label}`, adapterName, mem, rule, hosting: true });
  out.print(`  invite   ${link}
  join     pooled join "${link}"
  chat     pooled chat "${link}"      (your own tools: pooled serve "${link}")
  ${opts.allowAll ? `--allow-all: anyone with the code ${fmtCode(code)} comes in without asking` : `with the code ${fmtCode(code)} alone, a device waits until you let it in${process.stdin.isTTY ? " (a allows, d denies)" : ""}`}`);
  const passes = passCounter();
  const S = { code, hosting: true, lobby: 0, phase: "waiting", devices: 1, range: null, embed: true, model: opts.model, tps: null, passes: 0, signaling: true };
  let leaving = false, solo = 0, starting = null;
  node.on("prefill", (x) => { if (x.count && x.tDecode) S.tps = x.count / (x.tDecode / 1000); if (!node.ai.chain.length) solo += x.count + (x.prefilled ? 1 : 0); });
  node.on("signaling", (up) => { S.signaling = up; });
  node.on("version", (v) => out.log(`${v.name || "a device"} can't join: ${v.theirs > rn.PROTOCOL ? `it runs a newer Pooled (protocol ${v.theirs}, this pooled ${rn.PROTOCOL}); update this one: npx @pooled/cli@latest host` : `it runs an older Pooled (protocol ${v.theirs}, this pooled ${rn.PROTOCOL}); it should reload`}`));
  const tick = () => {
    const st = node.status();
    S.devices = st.devices.length; S.range = st.range; S.lobby = node.waitingJoins().length; S.passes = passes(node, st.passes) + solo; S.signaling = st.signaling;
    if (!leaving) S.phase = st.loading || (node.ai.starting && !st.online) ? "loading" : st.degraded ? "degraded" : st.online ? (node.ai.busy ? "answering" : "online") : "waiting";
    out.status(S);
    // --devices N: the room went on with fewer (a device stayed away past the rejoin grace, and the
    // room re-dealt without it) and N devices are in again: deal again so the newcomers hold layers
    // (a host at a terminal presses Enter instead)
    const peers = node.gpuPeers();
    if (!starting && !leaving && autoRedeal(opts.devices, { online: st.online, busy: !!node.ai.busy, starting: !!node.ai.starting,
      chain: node.ai.chain, peers, dealt: node.ai.dealtPeers })) {
      out.log(`${peers.length + 1} devices in the room again`);
      deal(true);
    }
  };
  const deal = (auto = false) => {
    if (starting || leaving) return;
    const again = !!node.ai.engine;
    if (again) { if (!auto) out.log("re-dealing the layers over the devices in the room"); }
    else if (!(opts.devices > node.gpuPeers().length + 1)) out.log(`dealing the layers over ${node.gpuPeers().length + 1} device(s)`);
    starting = (again ? node.redeal() : node.start(opts.model, { minDevices: opts.devices || 1 }))
      .then(() => out.log(`room online: ${node.status().split?.join(" · ") || "ready"}`))
      .catch((e) => { const x = explainError(e, { code, cmd: "host" }); out.log(`couldn't start: ${x.message}${x.hint ? ` ${x.hint}` : ""}`, "error"); if (!out.tty) bye(false, 1); })
      .finally(() => { starting = null; });
  };
  let exitCode = 0;
  // leaving ends the process (bye): Ctrl-C, q, SIGTERM, or a failed start without a terminal
  async function bye(sig, codeOut = 0) {
    if (leaving) { out.done(); process.exit(130); }
    leaving = true; exitCode = codeOut; S.phase = "leaving"; out.status(S);
    out.log(`closing room ${fmtCode(code)} and freeing the GPU${sig ? " (again to quit at once)" : ""}`);
    await closeSoon(node);
    if (process.stdin.isTTY) { try { process.stdin.setRawMode(false); } catch {} }
    out.done();
    process.exit(exitCode);
  }
  process.on("SIGINT", () => bye(true)); process.on("SIGTERM", () => bye(false));
  // who gets in: a device with the code alone waits in the lobby until the host answers (a / d here)
  const tty = !!process.stdin.isTTY;
  node.on("joinrequest", (r) => {
    out.log(tty ? `${r.line}: press a to let it in, d to turn it away`
      : `${r.line}: it waits (no terminal here to ask; give it the invite link, or start pooled host with --allow-all)`);
  });
  const answerJoin = async (yes) => {
    const w = node.waitingJoins();
    if (!w.length) { out.log("nobody is waiting to join"); return; }
    const r = yes ? await node.allowJoin(w[0].id) : node.denyJoin(w[0].id);
    // (the node logs who it let in or turned away)
    if (r && w.length > 1) out.log(`${w.length - 1} more waiting: ${w[1].line} (a / d)`);
  };
  // a terminal: Enter deals (again), a / d answer a join request, q or Ctrl-C leaves. Raw mode, so
  // typing doesn't scribble over the status line
  if (tty) {
    process.stdin.setRawMode(true); process.stdin.resume();
    process.stdin.on("data", (b) => {
      const k = b.toString();
      if (k === "\u0003" || k === "q") bye(true);
      else if (k === "a" || k === "A") answerJoin(true).catch((e) => out.log(`couldn't let it in: ${e.message}`, "error"));
      else if (k === "d" || k === "D") answerJoin(false);
      else if ((k === "\r" || k === "\n") && !opts.devices) deal();
    });
  }
  // --deny-unknown: a device with the code alone is turned away (the invite link still lets it in)
  if (opts.denyUnknown) node.on("joinrequest", (r) => { node.denyJoin(r.id); });
  // deal by itself (no terminal, --start, --wait N) once N devices are in and their pledges hold the
  // model, by the room page's math (room/plan.js roomFit); a terminal otherwise waits for Enter
  const auto = !tty || opts.start || opts.devices > 0;
  const lib = { MODELS: rn.MODELS, NEED_GB: rn.NEED_GB, roomBytes: rn.roomBytes, roomFit: rn.roomFit, shortNote: rn.shortNote, shortBy: rn.shortBy,
    gbUp: rn.gbUp, pledgeGB: rn.pledgeGB, nodeCtxFor: rn.nodeCtxFor };
  let lastWhy = "";
  const maybeDeal = () => {
    if (!auto || starting || leaving || node.ai.engine || node.ai.starting) return;
    const devs = devicesFrom(node, lib).filter((d) => d.gb != null);
    const f = roomFitNow(lib, { model: opts.model, devices: devs, ctxAsk: opts.ctx || 0 });
    const want = opts.devices || 1;
    const why = devs.length < want ? `waiting for ${want} devices (this one included; ${devs.length} in)` : !f.fits ? `${f.note} Waiting for devices.` : "";
    if (!why) { deal(); return; }
    if (why !== lastWhy) { lastWhy = why; out.log(why); }
  };
  if (!auto) out.log("waiting for devices: press Enter to deal the layers (Enter again re-deals after more join), q to quit");
  maybeDeal();
  const timer = setInterval(() => { maybeDeal(); tick(); }, 1000);
  if (!out.tty) {
    let last = "", lastAt = 0;
    const plain = out.status; out.status = (s) => { const k = `${s.phase}|${s.devices}|${s.range}`; if (k !== last || Date.now() - lastAt > 60000) { last = k; lastAt = Date.now(); plain(s); } };
  }
  tick();
  return new Promise(() => { void timer; });   // until bye()
}

// pooled host at a terminal that takes no escapes (TERM=dumb): the choices the screen would ask
// for, as numbered lines and typed answers; then the plain runHost. -> false when Ctrl-C / Ctrl-D
async function askHostLines(opts, rn) {
  const keys = hostable(rn.MODELS);
  if (!opts.modelGiven) {
    process.stderr.write("Models:\n");
    keys.forEach((k, i) => {
      const st = modelState(opts.modelDir, k, rn.MODELS, rn.FILES, rn.LOCAL);
      process.stderr.write(`  ${i + 1}) ${k.padEnd(18)} ${rn.MODELS[k].label.padEnd(26)} ${st.pulled ? "downloaded" : `${fmtBytes(rn.FILES?.[k]?.bytes)} download`}\n`);
    });
    const def = Math.max(0, keys.indexOf(opts.model));
    for (;;) {
      const a = await askLine(`Model 1-${keys.length} (Enter: ${def + 1}): `);
      if (a == null) return false;
      const t = a.trim();
      if (!t) { opts.model = keys[def]; break; }
      const n = Number(t);
      if (Number.isInteger(n) && n >= 1 && n <= keys.length) { opts.model = keys[n - 1]; break; }
      const m = resolveModel(t, keys);
      if (!m.error) { opts.model = m.key; break; }
      process.stderr.write(`  type a number from 1 to ${keys.length}\n`);
    }
    opts.modelGiven = true;
  }
  if (!opts.gb) {
    for (;;) {
      const a = await askLine("GB of memory to lend (Enter: the memory rule): ");
      if (a == null) return false;
      if (!a.trim()) break;
      try { opts.gb = parseGb(a.trim()); break; } catch (e) { process.stderr.write(`  ${e.message}\n`); }
    }
  }
  if (!opts.noPull && !opts.yes && !modelState(opts.modelDir, opts.model, rn.MODELS, rn.FILES, rn.LOCAL).pulled) {
    const y = await askYesNo(`${opts.model} is not downloaded (${fmtBytes(rn.FILES?.[opts.model]?.bytes)}). Download it now?`);
    if (y == null) return false;
    if (!y) opts.noPull = true;
  }
  if (!opts.devices && !opts.start) opts.start = true;
  return true;
}

// argv after "join" / "host" -> exit status
export async function lendMain(cmd, argv, { version = "" } = {}) {
  let opts;
  try { opts = parseLendArgs(cmd, argv); }
  catch (e) {
    if (!(e instanceof UsageError)) throw e;
    process.stderr.write((e.lines || [`pooled ${cmd}: ${e.message}`]).join("\n") + "\n");
    return 2;
  }
  if (opts.help) { process.stdout.write(cmd === "join" ? HELP_JOIN : HELP_HOST); return 0; }
  if (opts.model === "list") {
    const rn = await loadRoomNode().catch(() => null);
    if (!rn) { process.stderr.write("pooled host: this copy of pooled has no room node built in\n"); return 1; }
    for (const k of hostable(rn.MODELS)) process.stdout.write(`${k.padEnd(18)} ${rn.MODELS[k].label}\n`);
    return 0;
  }
  // pooled host 35b: a part of a model's name picks it (the list comes with the room node)
  if (cmd === "host" && opts.modelGiven && opts.model !== "list") {
    const rn = await loadRoomNode().catch(() => null);
    const keys = rn ? hostable(rn.MODELS) : null;
    if (keys && !keys.includes(opts.model)) {
      const m = resolveModel(opts.model, keys);
      if (m.error) { process.stderr.write(`pooled host: ${m.error}; one of: ${keys.join(", ")}\n`); return 2; }
      opts.model = m.key;
    }
  }
  // the one folder pooled keeps models in (made on first use); --models / POOLED_MODELS override it
  opts.modelDir = modelsDir({ flag: opts.modelDir });
  try { ensureModelsDir(opts.modelDir); } catch (e) { process.stderr.write(`pooled ${cmd}: can't make ${opts.modelDir}: ${e.message}\n`); return 1; }
  // a terminal on both ends and no --json-log: the interactive screens (a flag given skips its
  // question). A terminal that takes no escapes (TERM=dumb) gets the same questions as plain lines
  const caps = termCaps();
  opts.interactive = caps.ansi && !opts.jsonLog;
  opts.lines = caps.tty && !caps.ansi && !opts.jsonLog;
  if (cmd === "join" && !opts.code) {
    if (!opts.interactive && !opts.lines) { process.stderr.write(needsRoom("join") + "\n"); return 2; }
    for (;;) {
      const a = await askLine("Room code or invite link: ");
      if (a == null) return 130;
      const t = a.trim().replace(/^["']|["']$/g, "");
      const code = roomCodeFrom(t);
      if (code) { opts.code = code; opts.key = roomKeyFrom(t); break; }
      if (t) process.stderr.write(`  "${cleanText(t, 60)}" is not a room code (like 4TK-G9P) or a room link\n`);
    }
  }
  const out = makeOut(opts);
  try {
    if (cmd === "join") return await runJoin(opts, out);
    if (opts.interactive) {
      const prepared = await prepare(opts, out);
      const { runHostInteractive } = await import("./hostrun.js");
      return await runHostInteractive(opts, { prepared, version });
    }
    if (opts.lines) {
      const rn = await loadRoomNode();
      if (!(await askHostLines(opts, rn))) return 130;
    }
    return await runHost(opts, out);
  }
  catch (e) {
    if (e instanceof UsageError) { out.done(); process.stderr.write(`pooled ${cmd}: ${e.message}\n`); return 2; }
    return fail(out, e, { code: opts.code || opts.roomCode || "", cmd });
  }
}
