// pooled host in a terminal: the room opens at once, then the screen (cli/lib/hostui.js) asks for
// what the flags did not say (model, pledge), shows the room live (devices, pledges, who waits to
// join, whether the pledges hold the model), starts it, and offers chat right there.
import os from "node:os";
import { hostable, memoryRule, fmtCode } from "./lend.js";
import { modelState } from "./cache.js";
import { initialState, reduce, render, roomFitNow, modelRows, recommendModel, pledgeDefaults, devicesFrom, autoStart, colors, modelNeedGB } from "./hostui.js";
import { liveRegion, keysOf, colorOn } from "./tui.js";
import { pullWithProgress } from "./pullrun.js";
import { cleanText } from "./common.js";

const SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const ROOM_URL = "https://pooled.run/r/";

// prepared: lendrun's prepare() result ({ rn, loader, rule, mem, adapterName }); version: for chat
export async function runHostInteractive(opts, { prepared, version = "" }) {
  const { rn, loader, rule, mem } = prepared;
  const lib = { MODELS: rn.MODELS, FILES: rn.FILES, NEED_GB: rn.NEED_GB, roomBytes: rn.roomBytes, roomFit: rn.roomFit, shortNote: rn.shortNote,
    shortBy: rn.shortBy, gbUp: rn.gbUp, pledgeGB: rn.pledgeGB, nodeCtxFor: rn.nodeCtxFor };
  const dir = opts.modelDir;
  const keys = hostable(rn.MODELS);
  const pulled = new Set(keys.filter((k) => modelState(dir, k, rn.MODELS, rn.FILES, rn.LOCAL).pulled));
  const c = colors(colorOn(process.stderr));
  const maxGB = memoryRule(mem, { max: true }, { maxBufGB: 0 }).gb || rule.gb;
  const smallest = Math.min(...keys.map((k) => modelNeedGB(lib, k, opts.ctx || 0) || 99));
  const pd = pledgeDefaults(mem, { maxGB, ruleGB: rule.gb, smallestNeedGB: Math.min(smallest, 4) });
  const pledge0 = opts.gbGiven ? rule.gb : pd.def;
  const rowsFor = () => modelRows(lib, { keys, pulled, pledgeGB: S?.pledge.gb ?? pledge0, ctxAsk: opts.ctx || 0 });
  let S = null;
  const rows0 = rowsFor();
  const model0 = opts.modelGiven ? opts.model : recommendModel(rows0);

  const region = liveRegion(process.stderr);
  let chatting = false;
  const stamp = () => c.dim(new Date().toTimeString().slice(0, 8));
  // the room's log lines scroll above the screen; while the chat has the terminal they wait
  const held = [];
  const log = (m) => { const l = `${stamp()} ${c.dim(cleanText(String(m), 400))}`; if (chatting) held.push(l); else region.log(l); };

  region.render([`${c.dim(SPIN[0])} opening a room on this computer…`]);
  const node = await rn.createRoom({ model: model0, pledgeGB: pledge0, name: opts.name, signal: opts.signal, modelDir: dir, ctx: opts.ctx || 0,
    gate: true, ask: !opts.allowAll, setup: { webgpu: loader }, log, ...(opts.roomCode ? { code: opts.roomCode } : {}) });
  node.setPledge(pledge0);
  const code = node.code;
  const link = `${ROOM_URL}${code}${node.inviteFragment || ""}`;
  region.clear();
  // the lines to copy, once, above the live screen
  process.stderr.write([
    `${c.bold("pooled host")} ${c.dim("·")} room ${c.bold(c.cyan(fmtCode(code)))}`,
    `  ${c.dim("gpu")}     ${mem.kind === "discrete" ? `${mem.name} · ${Math.round(mem.totalGB)} GB` : mem.kind === "unified" ? `${mem.name || prepared.adapterName} · ${Math.round(mem.totalGB)} GB unified memory` : prepared.adapterName}`,
    `  ${c.dim("invite")}  ${link}`,
    `  ${c.dim("join")}    pooled join "${link}"`,
    `  ${c.dim("chat")}    pooled chat "${link}"`,
    `  ${c.dim(opts.allowAll ? `--allow-all: anyone with the code ${fmtCode(code)} comes in without asking` : opts.denyUnknown ? `--deny-unknown: a device with the code alone is turned away; the link lets it in` : `a device with the code ${fmtCode(code)} alone waits until you let it in (a / d)`)}`,
    `  ${c.dim("every device holding layers sees what is asked here: share the link only with people you trust")}`,
    "",
  ].join("\n") + "\n");

  S = initialState({ rows: rows0, model: opts.modelGiven ? opts.model : null, pledge: { gb: pledge0, max: Math.max(pd.max, pledge0), totalGB: pd.totalGB },
    fixedPledge: opts.gbGiven, flags: { start: opts.start, wait: opts.devices || 0, chat: opts.chat }, pulled, code, link: "", yes: opts.yes, noPull: opts.noPull });
  S.pledgeDone = opts.gbGiven;

  // ---- effects
  let pullAbort = null, starting = null, leaving = false, spinAt = 0, chatOnce = false;
  const pct = new Map();
  const doPull = (key) => {
    pullAbort?.abort();
    const ac = new AbortController(); pullAbort = ac;
    S.dl = { key, state: "running", done: 0, total: rn.FILES?.[key]?.bytes || 0, bps: null, error: null };
    log(`downloading ${key} to ${dir.replace(os.homedir(), "~")}/${key} (devices can join meanwhile)`);
    pullWithProgress(rn, key, dir, { quiet: true, signal: ac.signal,
      onProgress: (p) => { if (S.dl.key === key) S.dl = { ...S.dl, done: p.done, total: p.total || S.dl.total, bps: p.bps ?? S.dl.bps }; } })
      .then((r) => {
        if (pullAbort === ac) pullAbort = null;
        if (r.ok) { pulled.add(key); S.rows = rowsFor(); if (S.dl.key === key) S.dl = { ...S.dl, state: "done" }; log(`${key} downloaded`); }
        else if (r.aborted) { if (S.dl.key === key && S.dl.state === "running") S.dl = { ...S.dl, state: "none" }; }
        else { if (S.dl.key === key) S.dl = { ...S.dl, state: "error", error: r.error.message }; log(`download failed: ${r.error.message}`); }
        refresh();
      });
  };
  const doStart = () => {
    if (starting) return;
    pct.clear();
    log(`starting ${S.model} over ${S.devices.filter((d) => d.gb != null).length} device(s)`);
    S.step = "starting";
    const again = !!node.ai.engine;
    starting = (again ? node.redeal() : node.start(S.model, { minDevices: 1 }))
      .then(() => {
        S.step = "online";
        S.split = node.status().split?.join(" · ") || "";
        log(`room online: ${S.split}`);
        if (S.flags.chat && !chatOnce) { chatOnce = true; setTimeout(() => doChat(), 50); }
      })
      .catch((e) => { S.step = "room"; S.notice = `couldn't start: ${cleanText(e?.message || e, 200)}`; log(S.notice); })
      .finally(() => { starting = null; refresh(); });
  };
  async function doChat() {
    if (chatting || leaving) return;
    chatting = true;
    stopKeys();
    region.close();
    process.stderr.write(c.dim("chat with the room here: /exit (or Ctrl-D) goes back to the room screen\n"));
    const had = new Set(process.listeners("SIGINT"));
    try {
      const { chatMain } = await import("./chatrun.js");
      const { Peer } = await rn.setupNode({ webgpu: loader });
      await chatMain([link, "--name", `${node.name} chat`, ...(opts.signal ? ["--signal", opts.signal] : [])], { version, embedded: true, Peer });
    } catch (e) { log(`chat: ${cleanText(e?.message || e, 200)}`); }
    for (const f of process.listeners("SIGINT")) if (!had.has(f)) process.off("SIGINT", f);
    chatting = false;
    if (!leaving) { process.stderr.write("\n"); for (const l of held.splice(0)) region.log(l); startKeys(); refresh(true); }
  }
  async function bye(code = 0) {
    if (leaving) { region.close(); process.exit(130); }
    leaving = true;
    pullAbort?.abort();
    stopKeys();
    region.log(`${stamp()} closing room ${fmtCode(node.code)} and freeing the GPU`);
    region.close();
    try { await node.close(); } catch {}
    process.exit(code);
  }
  const run = (fx) => {
    for (const f of fx) {
      if (f.do === "quit") bye(0);
      else if (f.do === "pull") doPull(f.key);
      else if (f.do === "stream") log(`${f.key}: not downloading; each start streams this computer's layers from Hugging Face`);
      else if (f.do === "model") { if (!node.ai.engine) node.ai.model = f.key; S.rows = rowsFor(); }
      else if (f.do === "pledge") { node.setPledge(f.gb); S.rows = rowsFor(); }
      else if (f.do === "start" || f.do === "redeal") doStart();
      else if (f.do === "allow") node.allowJoin(f.id)?.catch?.((e) => log(`couldn't let it in: ${e.message}`));
      else if (f.do === "deny") node.denyJoin(f.id);
      else if (f.do === "chat") doChat();
    }
  };

  // ---- the room, as the screen shows it
  function refresh(force = false) {
    if (chatting || leaving) return;
    const ranges = node.ai.layersByName || null;
    S.devices = devicesFrom(node, lib, { pct, ranges: S.step === "online" || S.step === "starting" ? ranges : null });
    S.lobby = node.waitingJoins().map((r) => ({ id: r.id, line: r.line }));
    if (opts.denyUnknown) for (const r of S.lobby) node.denyJoin(r.id);
    if (opts.denyUnknown) S.lobby = [];
    const gpu = S.devices.filter((d) => d.gb != null);
    S.fit = S.model ? roomFitNow(lib, { model: S.model, devices: gpu, ctxAsk: opts.ctx || 0, spareGB: [Math.max(0, S.pledge.max - S.pledge.gb)] }) : null;
    if (S.step === "online" && !node.ai.online && !starting) S.notice = "a device left: the room waits for it (Enter re-deals without it)";
    if (S.step === "online" && node.ai.online) S.split = node.status().split?.join(" · ") || S.split;
    if (autoStart(S)) { S.step = "starting"; doStart(); }
    S.link = link;
    region.render(render(S, { width: process.stderr.columns || 80, c, lib, spin: c.cyan(SPIN[spinAt % SPIN.length]) }));
    void force;
  }
  node.on("progress", (p) => { if (p?.name) pct.set(p.name, p.pct); });
  node.on("loadprogress", (p) => pct.set(node.name, p));
  node.on("members", () => refresh());
  node.on("joinrequests", () => refresh());
  node.on("joinrequest", (r) => { if (!opts.denyUnknown) log(`${r.line}: press a to let it in, d to turn it away`); });
  node.on("version", (v) => log(`${v.name || "a device"} can't join: it runs ${v.theirs > rn.PROTOCOL ? "a newer" : "an older"} Pooled (protocol ${v.theirs}, this pooled ${rn.PROTOCOL})`));

  // ---- keys
  const onData = (b) => {
    for (const k of keysOf(b)) {
      if (k === "ctrl-c") { bye(0); return; }
      const r = reduce(S, k);
      S = r.state;
      run(r.fx);
    }
    refresh();
  };
  function startKeys() { try { process.stdin.setRawMode(true); } catch {} process.stdin.resume(); process.stdin.on("data", onData); }
  function stopKeys() { process.stdin.off("data", onData); try { process.stdin.setRawMode(false); } catch {} process.stdin.pause(); }
  process.on("SIGINT", () => { if (!chatting) bye(0); });   // (in the chat, Ctrl-C stops an answer)
  process.on("SIGTERM", () => bye(0));
  if (S.dl.state === "running") doPull(S.dl.key);
  startKeys();
  setInterval(() => { spinAt++; refresh(); }, 120).unref?.();
  refresh();
  return new Promise(() => {});   // until bye()
}
