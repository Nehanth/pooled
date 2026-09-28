// Code mode (docs/design/harness-app.md E): the host's agent controller and the peers' view.
// room.js imports this lazily (the Code tab, or a code message arriving) and calls
// initCode(roomApi, { mock }) once; it returns { show(mode) }.
//
// One shared agent session per room, run on the model host (the device that samples every token).
// Anyone in the room can drive it, like Chat: a member's request goes to the host as ai-code-ask,
// queues behind the current run, and its bubble carries the asker's name. Who may do what:
//   request, new task, switch or create a project saved in the browser: any member who sees Code
//     (the room's "who sees answers" is Everyone; with "Only me" / "Whoever asked" Code is the host's)
//   approve / reject an edit, Stop: the member who asked that request, or the host
//   open a folder from disk, save in the editor, drive a folder project: the host only (the files
//     are on the host's disk, and what the agent reads there would reach the asker's screen)
// Projects live in the host's browser (OPFS) or on its disk; members see them through the timeline,
// the file tree and the preview.
//
// Host: a project (OPFS scratch folder or a picked folder, harness/projects.js), a PreviewServer
// over it, the 8 tools, and an Agent over the room's model (harness/room-model.js; with
// ?mock=code, window.__pooledMock.model instead). A run holds the room's lock for all its steps.
// Everything the timeline shows is an ai-code-* message: the host renders it, keeps it for late
// joiners and broadcasts it (visibility rules apply, room.js sendCode).
// Peer: renders those messages, and mirrors the served ports through PreviewSubscriber; each
// peer runs the preview itself, straight away (sandboxed as on the host; anyone in the room can drive the agent).
import { codeUI } from "./code-ui.js";
import { Agent, briefCall } from "../harness/agent.js";
import { codingTools } from "../harness/codetools.js";
import { PreviewServer } from "../harness/preview.js";
import { previewTools } from "../harness/preview-tools.js";
import { runJsTool, runJsAvailable } from "../harness/run-js.js";
import { mountPreview, openPreviewTab } from "../harness/preview-frame.js";
import { PreviewPublisher, PreviewSubscriber } from "../harness/preview-sync.js";
import { lineDiff } from "../harness/diff.js";
import { listProjects, createProject, openProject, openFolder, canOpenFolder, saveSession, loadSession, slugify } from "../harness/projects.js";
import { roomModel } from "../harness/room-model.js";
import { detectStyle } from "../harness/tools.js";
import { normPath, riskyPath } from "../harness/workspace.js";
import { CODE_SYSTEM } from "../harness/code-prompt.js";

const $ = (id) => document.getElementById(id);
const str = (v, n) => String(v ?? "").slice(0, n);
const cap = (s, n) => (s.length > n ? s.slice(0, n) + `\n…(${s.length - n} chars cut)` : s);
const HIST = 50, TOK_MS = 50, EDGE = 50;
const QUEUE_MAX = 6, ASK_MAX = 4000;   // requests waiting on the host (two per member), a request's length
// tools whose results are the project's own content: for a folder on disk they stay on the host
const READS = new Set(["read_file", "search", "list_dir"]);
const WORDS = new Set(("a an the me my us our please build make create write code develop implement simple small little basic new "
  + "game app application website site page web in with using for of to and that js javascript html css plain canvas").split(" "));
// "build a tetris game" -> "tetris"
export function projectName(text) {
  const keep = (text.toLowerCase().match(/[a-z0-9]+/g) || []).filter((w) => !WORDS.has(w)).slice(0, 2);
  return keep.join(" ") || "project";
}
// the approval card's diff: lineDiff rows as [op, text] / [" ", "", skip]. The path is the one
// written (normalised). Too long to diff: the first and last EDGE lines, and (host only) the
// whole proposed file for "view full file".
export function makeDiff({ path, before, after, error }) {
  try { path = normPath(path); } catch { /* the tool reports it */ }
  const lines = after ? after.split("\n").length - (after.endsWith("\n") ? 1 : 0) : 0;
  const rows = lineDiff(before ?? "", after ?? "", { max: 400 });
  const d = { path, isNew: before == null, lines, add: 0, del: 0, rows: null };
  if (rows) {
    d.rows = rows.map((r) => (r.skip ? [" ", "", r.skip] : [r.op, r.text]));
    for (const r of rows) { if (r.op === "+") d.add++; else if (r.op === "-") d.del++; }
  } else {
    d.add = lines;
    const all = (after ?? "").split("\n");
    if (all.length && all[all.length - 1] === "") all.pop();
    d.head = all.slice(0, EDGE);
    d.tail = all.length > 2 * EDGE ? all.slice(-EDGE) : [];
    d.full = after ?? "";
  }
  if (error) d.error = error;
  return d;
}
// ai-code-tool's diff is at most ~4 KB on the wire (a long file's edges ~8 KB); never the full file
function wireDiff(d) {
  if (!d) return d;
  const { full, ...w } = d;
  if (w.head) { w.head = w.head.map((t) => t.slice(0, 160)); w.tail = w.tail.map((t) => t.slice(0, 160)); }
  if (!w.rows) return w;
  const rows = w.rows.map(([o, t, s]) => (s ? [o, "", s] : [o, t.slice(0, 160)]));
  let k = rows.length;
  while (k > 0 && JSON.stringify(rows.slice(0, k)).length > 3800) k = Math.floor(k * 0.8);
  return { ...w, rows: rows.slice(0, k), more: rows.length - k };
}

// the eval suite (tests/eval/, not deployed) runs only on a development host
const DEV_HOST = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(globalThis.location?.hostname || "");
const EVAL = DEV_HOST ? new URLSearchParams(globalThis.location?.search || "").get("eval") : null;

export async function initCode(api, { mock = null } = {}) {
  const ui = codeUI({ onMode: (m) => { if (m === "code") entered(); } });
  const isHost = () => api.role() === "host" || (!api.role() && !!api.myId() && api.myId() === api.hostId());
  const hostName = () => (isHost() ? api.name?.() : api.nameOf?.(api.hostId())) || "the host";
  // the room shares Code with every member only when everyone sees the answers
  const shared = () => (api.visibility?.() || "all") === "all";
  let sid = crypto.getRandomValues(new Uint32Array(1))[0].toString(36);

  // ================================================================ host
  let project = null, server = null, publisher = null, tools = [], agent = null, agentSrc = null, agentStyle = null, model = null;
  let sessionJson = null, hist = [], tree = [], running = false, ctrl = null, allowTask = false;
  let userAuto = null;   // the user's own tick of "auto-approve edits", kept across projects
  let mid = "", toolN = 0;
  let queue = [], asker = null, retry = 0;   // requests waiting: { text, name, from }; asker: who asked the current run
  let remoteAns = null;                      // the asker's answer to the approval card on screen: { mid, i, res }
  const callIdx = new Map();

  // render, remember for late joiners, broadcast
  function emit(msg, wire = msg) {
    ui.apply(msg);
    record(wire);
    api.broadcast(wire);
  }
  function record(m) {
    if (m.t === "ai-code-live") return;   // live typing is not history: the finished call's card is
    if (m.t === "ai-code-tok") {
      const last = hist[hist.length - 1];
      if (last?.t === m.t && last.mid === m.mid && last.step === m.step) { last.text += m.text; return; }
    } else if (m.t === "ai-code-tool") {
      const j = hist.findIndex((x) => x.t === m.t && x.mid === m.mid && x.i === m.i);
      if (j >= 0) { hist[j] = { ...hist[j], ...m }; return; }
    }
    hist.push({ ...m });
    if (hist.length > 4 * HIST) hist.splice(0, hist.length - 4 * HIST);
  }
  const note = (text, err = false) => emit({ t: "ai-code-note", mid, text, ...(err ? { err: true } : {}) });
  // model text, coalesced so a peer gets ~20 messages a second, not one per token
  let tokBuf = "", tokStep = 0, tokTimer = 0;
  function flushTok() {
    clearTimeout(tokTimer); tokTimer = 0;
    if (tokBuf) { const text = tokBuf; tokBuf = ""; emit({ t: "ai-code-tok", mid, step: tokStep, text }); }
  }
  function tok(step, text) {
    if (step !== tokStep) flushTok();
    tokStep = step; tokBuf += text;
    tokTimer ||= setTimeout(flushTok, TOK_MS);
  }
  // the tool call the model is typing, sent as deltas (a new call resets), at most every TOK_MS
  let liveRaw = null, liveSent = 0, liveN = 0, liveStep = 0, liveTimer = 0;
  function flushLive() {
    clearTimeout(liveTimer); liveTimer = 0;
    if (liveRaw == null || liveRaw.length <= liveSent) return;
    emit({ t: "ai-code-live", mid, step: liveStep, n: liveN, reset: liveSent === 0, text: liveRaw.slice(liveSent) });
    liveSent = liveRaw.length;
  }
  function live(step, raw) {
    if (raw == null) {
      if (liveRaw != null) { flushLive(); emit({ t: "ai-code-live", mid, step: liveStep, n: liveN, end: true }); }
      liveRaw = null; liveSent = 0; return;
    }
    // a new call: the text typed before it goes out first, so it stays above the call's card
    if (liveRaw == null || raw.length < liveSent || step !== liveStep) { flushTok(); liveN++; liveSent = 0; }
    liveRaw = raw; liveStep = step;
    liveTimer ||= setTimeout(flushLive, TOK_MS);
  }
  const tool = (i, fields, wire) => {
    const base = { t: "ai-code-tool", mid, i, ...fields };
    emit(base, wire ? { ...base, ...wire } : base);
  };

  async function sendFiles() {
    if (!project) return;
    try { tree = (await project.ws.walk(500)).slice(0, 500); } catch { tree = []; }
    ui.tree(tree);
    api.broadcast({ t: "ai-code-files", tree });
  }

  // ---- projects
  let projList = [];
  async function refreshProjects() {
    projList = (await listProjects().catch(() => [])).slice(0, 50);
    fillProjects(projList, project?.id || "");
    $("code-proj-kind").textContent = project ? (project.kind === "folder" ? "folder on disk · edits ask first" : "saved in this browser") : "";
    $("code-newtask").disabled = !project;
    driverNote();
    sendProjects();
  }
  function fillProjects(list, cur, { guest = false } = {}) {
    const sel = $("code-proj-select");
    sel.replaceChildren(new Option(list.length ? "Open a project…" : "No projects yet", ""));
    const n = new Map();
    for (const p of list) n.set(p.name, (n.get(p.name) || 0) + 1);
    for (const p of list) {
      // two projects with one name: the later ones carry their number (opfs:counter-2 is "counter (2)")
      const k = n.get(p.name) > 1 && new RegExp(`^opfs:${slugify(p.name)}-(\\d+)$`).exec(p.id)?.[1];
      const o = new Option(p.name + (k ? ` (${k})` : "") + (p.kind === "folder" ? " (folder)" : ""), p.id);
      o.disabled = guest && p.kind === "folder";   // a folder on the host's disk: the host opens it
      sel.add(o);
    }
    sel.value = cur;
  }
  // members see the host's projects, the one open and the auto-approve box
  const projMsg = () => ({ t: "ai-code-projects", list: projList.map(({ id, name, kind }) => ({ id, name, kind })), cur: project?.id || "",
    kind: project?.kind || "", auto: $("code-auto").checked });
  function sendProjects(to = null) { if (!isHost()) return; if (to) api.send(to, projMsg()); else api.broadcast(projMsg()); }
  // another project: requests queued for this one are dropped (except when the queue's first
  // request is what made the project)
  function closeProject({ keepQueue = false } = {}) {
    if (running) ctrl?.abort();
    if (!keepQueue) for (const q of queue.splice(0)) tell(q.from, "the project changed: your queued request was dropped", true);
    server?.close(); publisher?.close();   // in this order: the server's stops reach the members (ai-pv-stop) before the publisher unsubscribes
    for (const port of [...ui.ports.keys()]) ui.dropPort(port);
    project = server = publisher = agent = model = null; agentSrc = null; tools = [];
  }
  // keepAuto: the project pump() made for the first request keeps the box as the user left it
  async function useProject(p, { keepAuto = false } = {}) {
    if (!p) return;
    closeProject({ keepQueue: keepAuto });
    project = p;
    server = new PreviewServer(p.ws);
    // run_js only when the snippet runs on the isolated preview host (a loop there cannot freeze the room)
    tools = [...codingTools(p.ws, { server }), ...previewTools(server), ...(runJsAvailable() ? [runJsTool(server)] : [])];
    publisher = new PreviewPublisher(server, { send: api.send, broadcast: api.broadcast, channel: api.channel });
    server.onUpdate(portUpdate);
    const saved = await loadSession(p.id).catch(() => null);
    sessionJson = saved?.agent || null;
    hist = Array.isArray(saved?.hist) ? saved.hist : [];
    sid = crypto.getRandomValues(new Uint32Array(1))[0].toString(36);
    // scratch files only live here (on unless the user turned it off); a real folder asks per edit
    if (!keepAuto || p.kind === "folder") $("code-auto").checked = p.kind === "opfs" && userAuto !== false;
    ui.clear();
    for (const m of hist) ui.apply(m);
    if (!hist.length) ui.placeholder(`project <b>${escapeHTML(p.name)}</b> is empty<br>ask for something to build`);
    await sendFiles();
    api.broadcast({ t: "ai-code-history", sid, items: hist.slice(-HIST), tree });
    refreshProjects();
    ctxMeter();
  }
  const escapeHTML = (s) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
  function portUpdate(u) {
    if (u.stopped) { ui.dropPort(u.port); return; }
    if (ui.ports.has(u.port)) return;   // the mounted frame follows its own updates
    const port = u.port, src = server;
    ui.portTab(port, { closable: true, mount: (el) => mountPreview(el, src, port, { onLog: (e) => ui.logRow(port, e), onStatus: (s) => ui.status(port, s) }) });
    ui.activate(port);
    ui.outTab("preview");
  }
  ui.onClosePort((port) => { if (isHost()) server?.stop(port); else ui.dropPort(port); });
  ui.onReload(async (port) => {
    const P = ui.ports.get(port);
    if (isHost() && server) { const s = await server.refresh(port); if (!s) P?.mount.reload(); }
    else P?.mount.reload();
  });
  ui.onOpen((port, path) => { openPreviewTab(isHost() ? server : sub, port, path); });
  // the Files view: the host opens a file to edit it (a very long one read-only); a peer sees the
  // files of a served preview, read-only
  const EDIT_MAX = 300000;
  ui.onFile(async (path) => {
    if (isHost() && project) {
      try {
        const text = await project.ws.read(path);
        if (text.length > EDIT_MAX) ui.openFile(path, cap(text, EDIT_MAX), { readOnly: true, label: `${path} · too long to edit here` });
        else ui.openFile(path, text, { readOnly: false });
      } catch (e) { ui.viewFile(`(${e.message})`, path); }
      return;
    }
    for (const s of (sub?.ports() || []).map((p) => sub.snapshot(p.port))) {
      const rel = s.dir ? (path.startsWith(s.dir + "/") ? path.slice(s.dir.length + 1) : null) : path;
      const f = rel != null && s.files.get(rel);
      if (f) { ui.openFile(path, cap(new TextDecoder().decode(f.bytes), 200000), { readOnly: true }); return; }
    }
    ui.viewFile("(the file stays on the host; files of a served preview show here)", path);
  });
  // Save in the editor: into the project (a served preview reloads by itself), a line in the
  // timeline for everyone, and the agent hears about it with the next request
  const handEdits = new Set();
  ui.onSave(async (path, text) => {
    if (!isHost() || !project) throw new Error("only the host can save");
    await project.ws.write(path, text);
    handEdits.add(path);
    note(`${api.name?.() || "the host"} edited ${path} by hand` + (server?.servedPorts(path).length ? " · the preview reloaded" : ""));
    save();
    sendFiles();
  });
  // the agent wrote a file that is open in the editor: show the new text
  async function refreshOpen() {
    const p = ui.openPath;
    if (!p || !project) return;
    try { ui.fileChanged(p, await project.ws.read(p)); } catch {}
  }

  // the host's project controls while the agent works: they say why nothing happens (members hear the same)
  const busyNote = () => localNote("the agent is working: try again when it is done", true);
  $("code-proj-select").addEventListener("change", async (e) => {
    const id = e.target.value;
    if (!isHost()) { if (id && id !== peerProj.cur) askHost({ t: "ai-code-cmd", cmd: "open", id }); e.target.value = peerProj.cur; return; }
    if (!id || id === project?.id) return;
    if (running) { e.target.value = project?.id || ""; busyNote(); return; }
    try { await useProject(await openProject(id)); } catch (err) { localNote(err.message, true); refreshProjects(); }
  });
  const newName = $("code-new-name");
  $("code-new").addEventListener("click", () => {
    if (running) { busyNote(); return; }
    const on = newName.hidden;
    newName.hidden = !on; $("code-proj-select").hidden = on;
    if (on) { newName.value = ""; newName.focus(); }
  });
  newName.addEventListener("keydown", async (e) => {
    if (e.key === "Escape") { newName.hidden = true; $("code-proj-select").hidden = false; return; }
    if (e.key !== "Enter" || !newName.value.trim()) return;
    const name = newName.value.trim().slice(0, 40);
    newName.hidden = true; $("code-proj-select").hidden = false;
    if (!isHost()) { askHost({ t: "ai-code-cmd", cmd: "new", name }); return; }
    try { await useProject(await createProject(name)); } catch (err) { localNote("could not create the project: " + err.message, true); }
  });
  $("code-open").dataset.can = canOpenFolder() ? "1" : "";
  $("code-open").addEventListener("click", async () => {
    if (!isHost()) return;
    if (running) { busyNote(); return; }
    try { const p = await openFolder(); if (p) await useProject(p); } catch (err) { localNote("could not open the folder: " + err.message, true); }
  });
  $("code-newtask").addEventListener("click", () => { if (!isHost()) askHost({ t: "ai-code-cmd", cmd: "newtask" }); else if (running) busyNote(); else newTask(); });
  function newTask(by = null) {
    if (running || !project) return false;
    agent?.reset(); sessionJson = null;
    if (model?.stats) model.stats.last = null;   // the meter measures the fresh conversation, not the last run's prompt
    mid = "";
    note(`new task${by ? ` (${by})` : ""}: the agent starts fresh · files and previews stay`);
    save();
    ctxMeter();
    return true;
  }
  $("code-auto").addEventListener("change", (e) => {
    if (!isHost()) { askHost({ t: "ai-code-cmd", cmd: "auto", on: e.target.checked }); e.target.checked = peerProj.auto; return; }
    if (project?.kind !== "folder") userAuto = e.target.checked;
    sendProjects();
  });
  $("pv-to-agent").addEventListener("click", () => { ui.tab("agent"); $("code-prompt").value = "Fix the errors in the preview console"; grow(); $("code-prompt").focus(); });

  // a line only this screen sees (not part of the session)
  function localNote(text, err = false) { ui.apply({ t: "ai-code-note", text, err }); }
  function save() {
    if (!project) return;
    saveSession(project.id, { v: 1, agent: agent ? agent.toJSON() : sessionJson, hist: hist.slice(-4 * HIST) }).catch((e) => console.warn("code session not saved", e));
  }
  function ctxMeter() {
    if (!agent || !model?.count) { ui.ctx(0); return; }   // a scripted model has no token count to show
    const max = api.maxSeq(), last = model?.stats?.last;
    let used;
    try { used = last ? last.prompt + last.generated : agent._size(); } catch { used = 0; }
    ui.ctx(used, max);
  }

  // ---- the agent
  function ensureAgent() {
    // rebuilt when the model's tool format changes too (a re-deal to another model)
    const style = mock?.model ? "xml" : detectStyle(api.chatTemplate());
    const src = mock?.model || "room";
    if (agent && agentSrc === src && agentStyle === style) return;
    model = mock?.model ? (typeof mock.model === "function" ? { generate: mock.model } : mock.model) : roomModel(api, { tools, style, maxNew: 8192, sampling: style === "json" ? "exact" : "focused" });
    const json = agent ? agent.toJSON() : sessionJson;
    agent = Agent.from(json, {
      generate: model.generate, tools, style, system: CODE_SYSTEM, maxSteps: 30, approve, onEvent,
      budget: model.budget || Infinity, count: model.count || null,
      usage: model.stats ? () => model.stats.last : null, idsFor: model.idsFor || null, adopt: model.adopt || null, idsTag: model.idsTag || null,
    });
    agentSrc = src; agentStyle = style;
  }
  async function approve(call, info) {
    const i = callIdx.get(call);
    const diff = info && !info.error ? makeDiff(info) : null;   // a call that fails changes nothing: no diff on its card
    // a file of a folder on disk that can run commands (package.json, a script, a dotfile) always
    // asks, whatever auto-approve and "Allow edits for this task" say
    if (diff && project?.kind === "folder" && riskyPath(diff.path)) diff.risky = true;
    // a failing edit is not worth a question (the tool returns the error to the model), nor is a
    // write that changes nothing
    const same = diff && diff.rows && !diff.isNew && !diff.add && !diff.del;
    const auto = !diff?.risky && ($("code-auto").checked || allowTask || info?.error || same);
    tool(i, { state: auto ? "approved" : "pending", diff }, { diff: wireDiff(diff) });
    if (auto) return true;
    let off = null;
    const stopped = new Promise((r) => { const f = () => r({ ok: false, reason: "stopped" }); ctrl.signal.addEventListener("abort", f, { once: true }); off = () => ctrl?.signal.removeEventListener("abort", f); });
    // the host answers on its own screen; a member who asked answers on theirs (ai-code-approve)
    const remote = asker && asker !== api.myId() ? new Promise((res) => { remoteAns = { mid, i, res }; }) : null;
    const v = await Promise.race([ui.ask(mid, i, { risky: !!diff?.risky }), stopped, ...(remote ? [remote] : [])]);
    off?.(); ui.cancelAsk(mid, i); remoteAns = null;
    if (v === "all") allowTask = true;
    const ok = v === true || v === "all";
    tool(i, { state: ok ? "approved" : v?.reason === "stopped" ? "stopped" : "declined" });
    return ok || v;
  }
  function onEvent(e) {
    switch (e.type) {
      case "text": tok(e.step, e.text); break;
      case "call-live": live(e.step, e.raw); break;
      case "tool-start": {
        flushTok(); live(e.step, null);
        const i = ++toolN;
        callIdx.set(e.call, i);
        tool(i, { step: e.step, name: str(e.call.name || /<function=([^>\s]+)>/.exec(e.call.raw || "")?.[1] || /"name"\s*:\s*"([^"]+)"/.exec(e.call.raw || "")?.[1] || "tool call", 60), brief: str(briefCall(e.call), 200), state: "running" });
        break;
      }
      case "tool": {
        const i = callIdx.get(e.call), r = String(e.result);
        const state = r === "declined by the user: stopped" ? "stopped" : /^declined by the user/.test(r) ? "declined" : /^error/.test(r) ? "error" : "done";
        // a folder on disk: what the agent read stays on the host (peers see its size only)
        const wire = project?.kind === "folder" && READS.has(e.call.name) && state === "done"
          ? `(${r.split("\n").length} lines · a folder on disk: the output stays on the host)` : cap(r, 600);
        tool(i, { state, result: cap(r, 4000), ms: e.ms }, { result: wire });
        if (state === "done" && ["write_file", "edit_file"].includes(e.call.name)) { sendFiles(); refreshOpen(); }
        ctxMeter();
        break;
      }
      case "compacted":
        if (e.tier < 4) note(`older steps shortened to fit the context (${e.before} → ${e.after} tokens)`);
        break;
      case "usage": ctxMeter(); break;
      case "limit": note(`stopped after ${e.steps} steps`); break;
      case "stuck": note("stopped: the same tool call failed three times in a row" + (api.peers().length ? ". With other devices in the room, the split model may be producing bad output: try it on one device, or re-deal" : ". Try rephrasing the request, or a bigger model"), true); break;
    }
  }
  function stats(r, t0, gen0) {
    const s = model?.stats, gen = s ? s.generated - gen0 : 0, secs = ((Date.now() - t0) / 1000).toFixed(1);
    const parts = [`${r.steps} step${r.steps === 1 ? "" : "s"}`, `${r.calls} tool call${r.calls === 1 ? "" : "s"}`];
    if (gen) parts.push(`${gen} tok`, `${(s.tps || 0).toFixed(1)} tok/s`);
    parts.push(`${api.peers().length + 1} device${api.peers().length ? "s" : ""}`, `${secs} s`);
    if (r.reason !== "done") parts.push(r.reason);
    return parts.join(" · ");
  }
  // the Send button: the host queues its own request; a member sends it to the host
  function submit() {
    const box = $("code-prompt"), text = box.value.trim();
    if (!text) return;
    if (isHost()) {
      if (EVAL != null && /^\/eval\b/.test(text)) { if (running) return; box.value = ""; grow(); return runEval(text.slice(5).trim() || EVAL); }
      box.value = ""; grow();
      request(text, api.name?.() || "host", api.myId());
      return;
    }
    if (!shared()) { localNote(`only ${hostName()} uses Code in this room (Room settings: who sees answers)`, true); return; }
    if (peerProj.kind === "folder") { localNote(`${hostName()} has a folder from their disk open: only they can send requests to it`, true); return; }
    box.value = ""; grow();
    askHost({ t: "ai-code-ask", text: str(text, ASK_MAX) });
  }
  // a line for whoever asked: on this screen, or sent to that member's screen
  function tell(from, text, err = false) {
    if (from === api.myId()) localNote(text, err);
    else api.send(from, { t: "ai-code-msg", text: str(text, 300), ...(err ? { err: true } : {}) });
  }
  // the host takes a request (its own, or a member's) into the queue; the queue runs in order
  function request(text, name, from) {
    if (!api.ready()) { tell(from, "the model is not loaded yet: pick a model in Chat and press Start", true); return; }
    if (from !== api.myId() && project?.kind === "folder") { tell(from, `${hostName()} has a folder from their disk open: only they can send requests to it`, true); return; }
    if (queue.length >= QUEUE_MAX || queue.filter((q) => q.from === from).length >= 2) { tell(from, "the queue is full: send again after the current request", true); return; }
    queue.push({ text: str(text, ASK_MAX), name: str(name, 40), from });
    const ahead = queue.length - 1 + (running ? 1 : 0);
    if (ahead) tell(from, `queued: ${ahead} request${ahead === 1 ? "" : "s"} ahead of yours`);
    pump();
  }
  async function pump() {
    if (running || !queue.length) return;
    clearTimeout(retry); retry = 0;
    const q = queue[0];
    running = true;
    let wait = false;
    try {
      if (!api.ready()) {
        for (const x of queue.splice(0)) tell(x.from, "the model is not loaded: the request was dropped", true);
        return;
      }
      if (!project) await useProject(await createProject(projectName(q.text)), { keepAuto: true });
      if (!api.lock("code")) {
        // a chat answer holds the room: try again when it is done
        if (api.busy()) { wait = true; return; }
        queue.shift();
        tell(q.from, "the room cannot run the agent right now (a device left? re-deal first)", true);
        return;
      }
      queue.shift();
      await runOne(q);
    } catch (err) {
      localNote(err.message, true);
    } finally {
      running = false; ctrl = null; asker = null;
      ui.running(false);
      if (wait) retry = setTimeout(pump, 700);
      else if (queue.length) setTimeout(pump, 0);
    }
  }
  // one request, holding the room's lock (released here)
  async function runOne({ text, name, from }) {
    ensureAgent();
    mid = "m" + Date.now().toString(36);
    toolN = 0; callIdx.clear(); allowTask = false;
    ctrl = new AbortController();
    asker = from;
    ui.running(true, true);
    emit({ t: "ai-code-start", sid, mid, name, from, text });
    const t0 = Date.now(), gen0 = model?.stats?.generated || 0;
    let r;
    // files the user edited by hand since the agent's last turn: it is told, so it reads them first
    const told = handEdits.size ? `\n\n(I edited ${[...handEdits].join(", ")} by hand since your last turn: read ${handEdits.size === 1 ? "it" : "them"} before changing ${handEdits.size === 1 ? "it" : "them"}.)` : "";
    handEdits.clear();
    try { r = await agent.run(text + told, { signal: ctrl.signal }); }
    catch (err) { console.error(err); r = { steps: 0, calls: 0, reason: "error" }; note("error: " + err.message, true); }
    // a live chunk still waiting for its timer belongs to this run: dropped, so it never lands after its end
    finally { flushTok(); clearTimeout(liveTimer); liveTimer = 0; liveRaw = null; liveSent = 0; api.unlock(); }
    if (r.reason === "stopped") note("stopped");
    if (r.reason === "context") note(r.text, true);
    emit({ t: "ai-code-done", mid, steps: r.steps, reason: r.reason, stats: stats(r, t0, gen0) });
    save();
    ctxMeter();
  }
  // ?eval=all (or ?eval=tetris,todo): "/eval [ids]" in the prompt runs the eval suite
  // (tests/eval/) on the room's model, each task in a fresh in-memory project; the records and
  // trajectories download as .jsonl at the end
  async function runEval(spec) {
    if (!api.lock("code")) { localNote("the room is busy: try again when it is done", true); return; }
    running = true; ui.running(true); ctrl = new AbortController();
    const lines = [];
    try {
      let TASKS, byId, S;
      try { ({ TASKS, byId } = await import("../tests/eval/tasks/index.js")); S = await import("../tests/eval/suite.js"); }
      catch { throw new Error("the eval suite is not deployed here (run it from a local checkout)"); }
      const tasks = !spec || spec === "all" ? TASKS : spec.split(",").map((id) => byId(id.trim())).filter(Boolean);
      const style = detectStyle(api.chatTemplate());
      localNote(`eval: ${tasks.length} task${tasks.length === 1 ? "" : "s"} on ${api.peers().length + 1} device(s)`);
      const recs = await S.runSuite(tasks, {
        model: "room", signal: ctrl.signal, root: document.body,
        makeModel: ({ tools }) => ({ ...roomModel(api, { tools, style, maxNew: 8192, sampling: style === "json" ? "exact" : "focused" }), style }),
        onResult: ({ rec, trajectory }) => {
          lines.push(JSON.stringify(rec), JSON.stringify({ trajectory }));
          localNote(`${rec.ok ? "PASS" : "FAIL"} ${rec.id} · ${rec.reason} · ${rec.steps} steps · ${rec.generated} tok · ${(rec.ms / 1000).toFixed(0)} s`, !rec.ok);
        },
      });
      localNote(S.summary(recs));
    } catch (err) { localNote("eval failed: " + err.message, true); }
    finally {
      api.unlock(); running = false; ctrl = null; ui.running(false);
      if (lines.length) {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "application/x-ndjson" }));
        a.download = `eval-room-${new Date().toISOString().slice(0, 16).replace(/:/g, "")}.jsonl`;
        a.click();
      }
    }
  }
  api.onStop(() => ctrl?.abort());
  $("code-send").addEventListener("click", submit);
  // Stop: the host stops the run; the member who asked it asks the host to
  const stopRun = () => { if (isHost()) api.stop(); else if (peerRun) askHost({ t: "ai-code-stop", mid: peerRun.mid }); };
  $("code-stop").addEventListener("click", stopRun);
  const grow = () => { const p = $("code-prompt"); p.style.height = "auto"; p.style.height = Math.min(p.scrollHeight, 160) + "px"; };
  $("code-prompt").addEventListener("input", grow);
  $("code-prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing && !matchMedia("(pointer: coarse)").matches) { e.preventDefault(); submit(); }
  });
  // Esc stops the run, but not from a field or the approval buttons (Esc there backs out of them)
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || ui.mode !== "code" || e.defaultPrevented) return;
    if (!(isHost() ? running && ctrl : peerRun?.from === api.myId())) return;
    if (e.target?.closest?.("input, textarea, select, .cm-approve")) return;
    stopRun();
  });
  api.on("ai-pv-want", (from, d) => { if (isHost()) publisher?.onWant(from, d); });
  api.onPeerJoin((id) => {
    if (!isHost()) return;
    sendProjects(id);
    if (!hist.length && !server?.ports().length) return;
    api.send(id, { t: "ai-code-history", sid, items: hist.slice(-HIST), tree, ...(running && asker ? { run: { mid, from: asker } } : {}) });
    publisher?.helloTo(id);
  });

  // ---- a member's requests and answers (room.js lets these through only from members who see
  // Code; they are typed and capped here, and nothing in them runs: text goes to the agent as a
  // request, as the host's own would, and ids are looked up in the host's list)
  const MEMBER = api.nameOf ? (id) => str(api.nameOf(id) || "guest", 40) : () => "guest";
  api.on("ai-code-ask", (from, d) => {
    if (!isHost()) return;
    const text = str(d.text, ASK_MAX).trim();
    if (text) request(text, MEMBER(from), from);
  });
  api.on("ai-code-stop", (from, d) => {
    if (!isHost() || !running || !ctrl || from !== asker || str(d.mid, 40) !== mid) return;
    note(`${MEMBER(from)} pressed stop`);
    api.stop();
  });
  api.on("ai-code-approve", (from, d) => {
    if (!isHost() || !remoteAns || from !== asker || str(d.mid, 40) !== remoteAns.mid || d.i >>> 0 !== remoteAns.i) return;
    const v = d.v === "yes" ? true : d.v === "all" ? "all" : d.v === "no" ? { ok: false, reason: str(d.reason, 300).trim() } : null;
    if (v != null) remoteAns.res(v);
  });
  api.on("ai-code-cmd", async (from, d) => {
    if (!isHost()) return;
    const who = MEMBER(from);
    if (running || queue.length) { tell(from, "the agent is working: try again when it is done", true); return; }
    try {
      if (d.cmd === "newtask") { if (!newTask(who)) tell(from, "no project open yet", true); }
      else if (d.cmd === "auto") {
        if (project?.kind === "folder") { tell(from, `edits to ${hostName()}'s folder always ask ${hostName()}`, true); return; }
        $("code-auto").checked = userAuto = !!d.on;
        note(`${who} turned auto-approve ${d.on ? "on" : "off"}`);
        sendProjects();
      } else if (d.cmd === "open") {
        const id = str(d.id, 120), p = projList.find((x) => x.id === id);
        if (!p || p.kind !== "opfs") { tell(from, "that project cannot be opened from here", true); return; }
        if (id !== project?.id) { await useProject(await openProject(id)); note(`${who} opened the project ${p.name}`); }
      } else if (d.cmd === "new") {
        const name = str(d.name, 40).replace(/[\u0000-\u001f\u007f]/g, "").trim();
        if (!name) return;
        await useProject(await createProject(name));
        note(`${who} started the project ${name}`);
      }
    } catch (err) { tell(from, err.message, true); }
  });
  // a member opened Code: the projects, and the session so far
  const synced = new Map();
  api.on("ai-code-sync", (from) => {
    if (!isHost() || Date.now() - (synced.get(from) || 0) < 2000) return;
    synced.set(from, Date.now());
    sendProjects(from);
    api.send(from, { t: "ai-code-history", sid, items: hist.slice(-HIST), tree, ...(running && asker ? { run: { mid, from: asker } } : {}) });
    publisher?.helloTo(from);
  });

  // ================================================================ peer
  let sub = null;
  let peerRun = null;                                  // the run going on: { mid, from }
  const askers = new Map();                            // mid -> { from, name }, for the approval wait line
  let peerProj = { list: [], cur: "", kind: "", auto: true };
  let pending = 0;                                     // a request sent, no word back yet
  // a message to the host; a host on an older Pooled ignores it, so say so if nothing comes back
  function askHost(msg) {
    const id = api.hostId();
    if (!id || !api.peers().includes(id)) { localNote("not connected to the host", true); return; }
    api.send(id, msg);
    if (msg.t !== "ai-code-ask") return;
    clearTimeout(pending);
    pending = setTimeout(() => localNote(`${hostName()} did not pick up the request: if their tab runs an older Pooled, reload both pages`, true), 8000);
  }
  const heard = () => { clearTimeout(pending); pending = 0; };
  const mine = (m) => !!m && m === api.myId();
  function peerRunning() { ui.running(!!peerRun, mine(peerRun?.from)); }
  function peerView() {
    if (sub) return sub;
    sub = new PreviewSubscriber({ send: (m) => api.send(api.hostId(), m), hostId: () => api.hostId() });
    sub.onUpdate((u) => {
      if (u.stopped) { ui.dropPort(u.port); return; }
      if (ui.ports.has(u.port)) return;
      const port = u.port;
      ui.portTab(port, { closable: false, mount: (el) => mountPreview(el, sub, port, { autorun: true, onLog: (e) => ui.logRow(port, e), onStatus: (s) => ui.status(port, s) }) });
      if (ui.activePort == null) ui.activate(port);
    });
    return sub;
  }
  // peers take the host's messages as untrusted data: typed and capped here, textContent in the UI
  function clean(d) {
    const o = { t: d.t, mid: str(d.mid, 40) };
    if ("step" in d) o.step = d.step >>> 0;
    if ("i" in d) o.i = d.i >>> 0;
    for (const k of ["sid", "name", "from", "text", "brief", "state", "result", "stats", "reason"]) if (k in d) o[k] = str(d[k], k === "text" ? 8000 : k === "result" ? 4000 : 400);
    if ("ms" in d) o.ms = d.ms >>> 0;
    if ("steps" in d) o.steps = d.steps >>> 0;
    if (d.err) o.err = true;
    if ("n" in d) o.n = d.n >>> 0;
    if (d.reset) o.reset = true;
    if (d.end) o.end = true;
    if (d.diff && typeof d.diff === "object") {
      const x = d.diff;
      const lines = (a) => (Array.isArray(a) ? a.slice(0, 50).map((t) => str(t, 200)) : null);
      o.diff = { path: str(x.path, 300), isNew: !!x.isNew, lines: x.lines >>> 0, add: x.add >>> 0, del: x.del >>> 0, more: x.more >>> 0,
        rows: Array.isArray(x.rows) ? x.rows.slice(0, 400).map((r) => [str(r?.[0], 1), str(r?.[1], 200), r?.[2] >>> 0]) : null,
        head: lines(x.head), tail: lines(x.tail), risky: !!x.risky };
    }
    return o;
  }
  // who a run's approval waits for, on screens that cannot answer it
  ui.onWaitText((m) => {
    const a = isHost() ? { name: null } : askers.get(m);
    const host = hostName();
    return a?.name && !mine(a.from) && a.from !== api.hostId() ? `waiting for ${a.name} or ${host} to approve` : `waiting for ${host} to approve`;
  });
  function track(m) {
    if (m.t === "ai-code-start") {
      askers.set(m.mid, { from: m.from || "", name: m.name || "" });
      if (askers.size > 200) askers.delete(askers.keys().next().value);
      peerRun = { mid: m.mid, from: m.from || "" };
      if (mine(m.from)) heard();
    } else if (m.t === "ai-code-done" && peerRun?.mid === m.mid) peerRun = null;
  }
  const peerMsg = (d) => {
    if (isHost()) return;
    peerView();
    setChrome();
    ui.poke();
    const m = clean(d);
    if (m.t === "ai-code-start" && m.sid && m.sid !== sid) { sid = m.sid; }
    // an answered approval: this screen's buttons go (whoever answered)
    if (m.t === "ai-code-tool" && m.state && m.state !== "pending") ui.cancelAsk(m.mid, m.i);
    track(m);
    ui.apply(m);
    if (m.t === "ai-code-start" || m.t === "ai-code-done") peerRunning();
    // the member who asked answers its own approvals
    if (m.t === "ai-code-tool" && m.state === "pending" && peerRun?.mid === m.mid && mine(peerRun.from)) {
      const i = m.i, at = m.mid;
      ui.ask(at, i, { risky: !!m.diff?.risky }).then((v) => {
        if (v?.reason === "approval card missing") return;
        api.send(api.hostId(), { t: "ai-code-approve", mid: at, i, v: v === true ? "yes" : v === "all" ? "all" : "no", reason: str(v?.reason, 300) });
      });
    }
  };
  for (const t of ["ai-code-start", "ai-code-tok", "ai-code-live", "ai-code-tool", "ai-code-note", "ai-code-done"]) api.on(t, (from, d) => peerMsg(d));
  api.on("ai-code-files", (from, d) => { if (!isHost() && Array.isArray(d.tree)) ui.tree(d.tree.slice(0, 500).map((p) => str(p, 300))); });
  api.on("ai-code-history", (from, d) => {
    if (isHost()) return;
    peerView(); setChrome(); ui.poke();
    sid = str(d.sid, 40);
    ui.clear();
    peerRun = null;
    for (const it of (Array.isArray(d.items) ? d.items : []).slice(-HIST)) if (it && typeof it === "object") { const m = clean(it); track(m); ui.apply(m); }
    // a run cut from the history's window is still going: the host says whose it is
    if (d.run && typeof d.run === "object") peerRun = { mid: str(d.run.mid, 40), from: str(d.run.from, 80) };
    if (!$("code-log").children.length) placeholderFor(false);
    if (Array.isArray(d.tree)) ui.tree(d.tree.slice(0, 500).map((p) => str(p, 300)));
    peerRunning();
  });
  // the host's projects: a member can open (not a folder), start one, and tick auto-approve
  api.on("ai-code-projects", (from, d) => {
    if (isHost()) return;
    const list = (Array.isArray(d.list) ? d.list : []).slice(0, 50).filter((p) => p && typeof p === "object")
      .map((p) => ({ id: str(p.id, 120), name: str(p.name, 60), kind: p.kind === "folder" ? "folder" : "opfs" }));
    peerProj = { list, cur: str(d.cur, 120), kind: d.kind === "folder" ? "folder" : d.kind === "opfs" ? "opfs" : "", auto: !!d.auto };
    fillProjects(list, peerProj.cur, { guest: true });
    $("code-auto").checked = peerProj.auto;
    $("code-auto").disabled = peerProj.kind === "folder";
    $("code-newtask").disabled = !peerProj.cur;
    $("code-proj-kind").textContent = peerProj.kind === "folder" ? `a folder on ${hostName()}'s disk` : peerProj.kind ? `saved in ${hostName()}'s browser` : "";
    setChrome();
  });
  api.on("ai-code-msg", (from, d) => { if (isHost()) return; heard(); localNote(str(d.text, 300), !!d.err); });
  api.on("ai-pv", (from, d) => { if (!isHost()) { peerView().onManifest(from, d); ui.poke(); } });
  api.on("ai-pv-blob", (from, d) => { if (!isHost()) peerView().onBlob(from, d); });
  api.on("ai-pv-stop", (from, d) => { if (!isHost()) peerView().onStop(from, d); });

  // ================================================================ both
  // a note only when this screen can't drive (a private room, or a folder from the host's disk)
  function driverNote() {
    if (isHost()) { ui.driverNote(api.peers().length && project?.kind === "folder" ? "This project is a folder on your disk: only you can send requests to it." : ""); return; }
    const host = hostName();
    ui.driverNote(!shared() ? `Only ${host} uses Code in this room. You see nothing of it.`
      : peerProj.kind === "folder" ? `${host} has a folder from their disk open: only ${host} can drive it. You see what it does, live.`
      : "");   // the usual case needs no note: anyone can ask, the agent works where it always does
  }
  function setChrome() {
    const host = isHost();
    ui.setHost(host, { canDrive: host || (shared() && peerProj.kind !== "folder") });
    driverNote();
  }
  function placeholderFor(host) {
    if (host) {
      ui.placeholder(api.ready()
        ? "<b>Code mode</b>: the room's model writes a web app, serves it on a port and fixes its own errors.<br>Ask for something to build, like “a tetris game”."
        : "<b>Code mode</b> runs on the room's model.<br>Pick a model in Chat and press Start, then ask for something to build.");
    } else ui.placeholder(shared()
      ? `<b>Code mode</b>: ask for something to build, like “a tetris game”.<br>The agent runs on ${escapeHTML(hostName())}'s device; everyone in the room sees it work, live`
      : `only ${escapeHTML(hostName())} uses Code in this room`);
  }
  function entered() {
    const host = isHost();
    setChrome();
    if (host) {
      refreshProjects();
      if (!project && !hist.length) placeholderFor(true);
    } else {
      if (!$("code-log").children.length) placeholderFor(false);
      if (api.hostId() && api.peers().includes(api.hostId())) api.send(api.hostId(), { t: "ai-code-sync" });
      peerRunning();
    }
    // not on touch screens: a focused prompt opens the keyboard (and the keyboard layout) before anyone asked to type
    if (!$("code-row").hidden && !matchMedia("(pointer: coarse)").matches) setTimeout(() => $("code-prompt").focus(), 0);
  }
  api.onRole(() => {
    const host = isHost();
    setChrome();
    if (!host && running) ctrl?.abort();
    // a device left mid-run: the next step would wait out the lap timeouts, so stop here
    else if (running && !api.ready() && !ctrl?.signal.aborted) { note("a device left: stopped · re-deal the layers, then send again", true); ctrl?.abort(); }
  });
  setChrome();
  return { show: (m) => ui.show(m), ctx: (used, max) => ui.ctx(used, max) };
}
