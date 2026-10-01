// No GPU, no network, no OpenClaw install: what 0.2 adds. The state file (invite key, passes),
// onboarding with a scripted prompter (host: the gate's key and the model download; join: a pasted
// link, the lobby, the pass kept), the join gate on asks (lobby, denied), /pooled, the pull lock and
// the download, and the warm-up.
//   node --test packages/openclaw/test/*_test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createHash } from "node:crypto";
import { EventEmitter } from "node:events";
import { setStateDir, roomFile, loadRoomState, saveHostGate, savedGate, saveJoinState, joinState } from "../src/state.js";
import { runSetup, deps, knock, hostKey, nonInteractive } from "../src/setup.js";
import { parseRoom, transport, bridgeFor, ensureOnline, admitted, PooledError } from "../src/pool.js";
import { runPooledCommand, statusText } from "../src/commands.js";
import { tryLock, lock, lockPath } from "../src/pulllock.js";
import { download, pullLine, pullState, downloadingMessage } from "../src/download.js";
import { fixedPart, replayBody, remember, prewarm, prewarmFile } from "../src/prewarm.js";

const tmp = (p) => fs.mkdtempSync(path.join(os.tmpdir(), `pooled-oc-${p}-`));
setStateDir(tmp("state"));
const KEY = "abcdefghijklmnopqrstuv", PASS = "PASSpassPASSpassPASSpass";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const mode = (f) => fs.statSync(f).mode & 0o777;

// ---------------- state ----------------
test("state: the host's gate and a joined room's key and pass live in <state>/pooled/room.json (0600), merged", () => {
  assert.equal(saveHostGate("4TKG9P", { ask: true, key: KEY, passes: {} }), true);
  assert.equal(saveHostGate("4TKG9P", { ask: true, key: KEY, passes: {} }), false, "unchanged: not written again");
  assert.equal(savedGate("4TKG9P").key, KEY);
  assert.equal(savedGate("ZZZZZZ"), null, "another code: a new room");
  saveJoinState("K7QXAB", { key: KEY });
  saveJoinState("K7QXAB", { pass: PASS, host: "mac", key: null });
  assert.deepEqual([joinState("K7QXAB").key, joinState("K7QXAB").pass, joinState("K7QXAB").host], [KEY, PASS, "mac"]);
  assert.equal(mode(roomFile()), 0o600);
  assert.equal(loadRoomState().host.code, "4TKG9P", "one file holds both");
  assert.equal(hostKey("4TKG9P"), KEY, "the link shown in onboarding is the gateway's");
  const k2 = hostKey("NEWNEW");
  assert.equal(k2.length >= 22, true); assert.equal(hostKey("NEWNEW"), k2);
});

test("parseRoom: links, dashed and lower-case codes; a key only from #k=", () => {
  assert.deepEqual(parseRoom(`https://pooled.run/r/4TKG9P#k=${KEY}`), { code: "4TKG9P", key: KEY });
  assert.deepEqual(parseRoom(" 4tk-g9p "), { code: "4TKG9P", key: null });
  assert.deepEqual(parseRoom("https://pooled.run/room?code=ABCD"), { code: "ABCD", key: null });
  assert.equal(parseRoom("I0O1LU").code, "", "not the room page's alphabet");
  assert.equal(parseRoom("hello").code, "");
});

// ---------------- onboarding ----------------
// a prompter that answers from a list, and keeps what it showed
function prompter(answers) {
  const shown = { notes: [], progress: [], selects: [] };
  const next = (kind, params) => {
    const a = answers.shift();
    if (a === undefined) throw new Error(`no answer for ${kind}: ${params.message}`);
    if (typeof a === "function") return a(params);
    if (kind === "text" && params.validate) { const bad = params.validate(String(a)); if (bad) throw new Error(`rejected ${a}: ${bad}`); }
    return a;
  };
  return {
    shown,
    select: async (p) => { shown.selects.push(p); return next("select", p); },
    text: async (p) => next("text", p),
    confirm: async (p) => next("confirm", p),
    note: async (m, t) => { shown.notes.push(`${t || ""}\n${m}`); },
    progress: (label) => { shown.progress.push(label); return { update: (m) => shown.progress.push(m), stop: (m) => shown.progress.push(`stop: ${m || ""}`) }; },
  };
}
const MEM = () => ({ mem: { kind: "discrete", name: "RTX 5070", totalGB: 12 }, def: 6, max: 10, label: "RTX 5070 · 12 GB" });

class FakeBridge extends EventEmitter {
  static how = "in"; static made = [];
  constructor(o) { super(); Object.assign(this, o); this.pass = null; this.waiting = false; this.connected = false; this.kicked = null; this.ready = false; FakeBridge.made.push(this); }
  async connect() {
    const how = FakeBridge.how;
    if (how === "hang") { this.waiting = true; this.emit("lobby"); return new Promise(() => {}); }
    if (how === "refused") { this.kicked = "The host didn't let this device in."; throw new Error(`the host of room ${this.code} said: ${this.kicked}`); }
    if (how === "lobby") { this.waiting = true; this.emit("lobby"); await sleep(20); this.waiting = false; }
    this.sentPass = this.pass;
    if (!this.pass) this.pass = PASS;
    this.hostName = "mac"; this.hostMeta = { api: 2, ctx: 65536, model: "qwen3.6-35b-moe" }; this.connected = true; this.ready = true;
  }
  async leave() { this.left = true; } destroy() { this.destroyed = true; }
}

test("onboarding (host): the GPU's default pledge, the model list, 'don't download', the gate; the link carries the kept key", async () => {
  deps.memoryDefaults = MEM;
  const models = tmp("models");
  const p = prompter(["host", "8", "qwen3-1.7b", false, "stream", 1, true]);
  const res = await runSetup({ prompter: p, config: { plugins: { entries: { pooled: { config: { modelDir: models } } } } } });
  const sel = p.shown.selects;
  const hints = sel[1].options.map((o) => o.hint).join("\n");
  assert.match(hints, /1\.7 GB download · needs about [\d.]+ GB across the room \([\d.]+ GB at 8k\) · 16k context · fits on this machine alone · small: slow turns and tool loops in OpenClaw/);
  assert.equal(sel[1].initialValue, "qwen3.6-35b-moe", "8 GB holds neither big model alone: the MoE is still the one for OpenClaw");
  assert.match(hints, /128k context · needs more devices · recommended for OpenClaw/);
  assert.match(p.shown.notes.join("\n"), /Small models struggle with OpenClaw's long prompts and tools: expect slow turns and tool loops\. Use the 35B MoE if your devices can hold it\./);
  assert.match(p.shown.notes.join("\n"), /global OpenClaw settings/);
  assert.equal(sel[2].initialValue, false, "the trimmed settings are opt-in");
  assert.match(sel[3].message, /not downloaded yet \(1\.7 GB/);
  const c = res.configPatch.plugins.entries.pooled.config;
  assert.deepEqual([c.mode, c.model, c.pledgeGB, c.minDevices, c.modelDir, c.pull, c.ask], ["host", "qwen3-1.7b", 8, 1, models, false, undefined]);
  assert.match(c.code, /^[A-HJKMNP-TV-Z2-9]{6}$/);
  const key = savedGate(c.code).key;
  const note = p.shown.notes.join("\n");
  assert.ok(note.includes(`https://pooled.run/r/${c.code}#k=${key}`), note);
  assert.match(note, /\/pooled allow/);
  assert.ok(res.notes[0].includes(`#k=${key}`));
  assert.deepEqual(res.configPatch.tools.byProvider.pooled.allow, ["read", "write", "edit", "ls"], "the 1.7B gets the file tools only");
  assert.equal(res.configPatch.tools.toolSearch, undefined, "no global change unless chosen");
  assert.equal(res.configPatch.agents.defaults.compaction, undefined);
  // the same room again keeps its code (and so its link)
  const p2 = prompter(["host", "8", "qwen3-1.7b", true, "later", 1, false]);
  const res2 = await runSetup({ prompter: p2, config: { plugins: { entries: { pooled: { config: { ...c } } } } } });
  assert.equal(res2.configPatch.plugins.entries.pooled.config.code, c.code);
  assert.equal(res2.configPatch.plugins.entries.pooled.config.ask, false);
  assert.equal(p2.shown.selects[1].initialValue, "qwen3.6-35b-moe", "an earlier small pick is not preselected again");
  // opted in: the two global settings, next to the 1.7B's file tools
  assert.equal(res2.configPatch.tools.toolSearch, false);
  assert.deepEqual(res2.configPatch.tools.byProvider.pooled.allow, ["read", "write", "edit", "ls"]);
  assert.deepEqual(res2.configPatch.agents.defaults.compaction, { memoryFlush: { enabled: false } });
  assert.ok(res2.configPatch.agents.defaults.models["pooled/qwen3-1.7b"], "the model entry stays");
});

test("onboarding (host): the 35B MoE is preselected when this device holds it, else the 27B; no small-model warning for them", async () => {
  const models = tmp("models");
  for (const [gb, want] of [[40, "qwen3.6-35b-moe"], [22, "qwen3.8-27b"]]) {
    deps.memoryDefaults = () => ({ mem: { kind: "unified", name: "M5", totalGB: 64 }, def: gb, max: 48, label: "M5 · 64 GB" });
    const p = prompter(["host", String(gb), want, "stream", 1, true]);
    const res = await runSetup({ prompter: p, config: { plugins: { entries: { pooled: { config: { modelDir: models } } } } } });
    assert.equal(p.shown.selects[1].initialValue, want, `${gb} GB`);
    assert.ok(!p.shown.notes.join("\n").includes("Small models"), "no warning");
    assert.equal(res.configPatch.tools, undefined, "big models keep OpenClaw's tools");
  }
});

test("onboarding (join): a pasted link; waiting for the host's Allow is shown; the pass and the host's model are kept", async () => {
  deps.memoryDefaults = MEM; deps.Bridge = FakeBridge; FakeBridge.how = "lobby"; FakeBridge.made = [];
  const p = prompter(["join", `https://pooled.run/r/K7QXAB#k=${KEY}`, "6"]);
  const res = await runSetup({ prompter: p, config: {} });
  assert.equal(FakeBridge.made[0].key, KEY, "knocks with the link's key");
  assert.match(p.shown.progress.join("\n"), /Waiting for .* to let this device in/);
  assert.match(p.shown.progress.at(-1), /In Pooled room K7Q-XAB \(host: mac\), running Qwen3\.6 35B MoE/);
  assert.equal(joinState("K7QXAB").pass, PASS);
  const m = res.configPatch.models.providers.pooled.models[0];
  assert.deepEqual([m.id, m.name, m.contextWindow], ["room", "Pooled room K7Q-XAB (Qwen3.6 35B MoE)", 65536]);
  assert.ok(!JSON.stringify(res.configPatch).includes(KEY) && !JSON.stringify(res.configPatch).includes(PASS), "no secrets in openclaw.json");
  // again later: the kept pass goes along (in without a new Allow)
  FakeBridge.how = "in";
  await runSetup({ prompter: prompter(["join", "K7Q-XAB", "6"]), config: {} });
  assert.equal(FakeBridge.made.at(-1).sentPass, PASS);
  assert.equal(FakeBridge.made.at(-1).key, KEY, "the key kept from the link");
});

test("onboarding (join): a host whose context is shorter than OpenClaw's prompt (pooled host's 8k default) is called out", async () => {
  deps.memoryDefaults = MEM; deps.Bridge = class extends FakeBridge {
    async connect() { await super.connect(); this.hostName = "spark"; this.hostMeta = { api: 2, ctx: 8192, model: "qwen3-1.7b" }; }
  };
  const p = prompter(["join", "K7Q-XAB", "6", false]);
  const res = await runSetup({ prompter: p, config: {} });
  const note = p.shown.notes.join("\n");
  assert.match(note, /spark runs this room with a 8192-token context/);
  assert.match(note, /spark runs this room with a small model\. Small models struggle with OpenClaw/);
  assert.equal(res.configPatch.tools.toolSearch, undefined, "declined: OpenClaw's settings stay");
  assert.match(note, /`pooled host qwen3-1\.7b --ctx 16384`/);
  assert.equal(res.configPatch.models.providers.pooled.models[0].contextWindow, 8192, "OpenClaw is told the truth (it fails fast)");
  FakeBridge.how = "in"; deps.Bridge = FakeBridge;
  const p2 = prompter(["join", "K7Q-XAB", "6"]);   // the 35B at 64k: no note
  await runSetup({ prompter: p2, config: {} });
  assert.ok(!p2.shown.notes.join("\n").includes("too short"), p2.shown.notes.join("\n"));
});

test("onboarding (join): turned away is an error; no answer yet is saved for the gateway to retry", async () => {
  deps.memoryDefaults = MEM; deps.Bridge = FakeBridge;
  FakeBridge.how = "refused";
  await assert.rejects(runSetup({ prompter: prompter(["join", "ZXCVBN", "6"]), config: {} }), /turned this device away/);
  FakeBridge.how = "hang"; deps.waitMs = 50;
  const p = prompter(["join", "ZXCVBN", "6"]);
  const res = await runSetup({ prompter: p, config: {} });
  assert.match(p.shown.progress.at(-1), /has not let this device in yet: the gateway asks again/);
  assert.equal(res.configPatch.plugins.entries.pooled.config.code, "ZXCVBN");
  assert.ok(FakeBridge.made.at(-1).destroyed, "the knock is dropped");
  deps.waitMs = 180000;
});

test("onboarding (non-interactive): POOLED_LINK keeps its key in the state file; a host gets its key made", async () => {
  const cfg = await nonInteractive({}, { POOLED_LINK: `https://pooled.run/r/QWERTY#k=${KEY}` });
  assert.equal(cfg.plugins.entries.pooled.config.code, "QWERTY");
  assert.equal(joinState("QWERTY").key, KEY);
  const h = await nonInteractive({}, { POOLED_CODE: "ASDFGH" });
  assert.ok(savedGate("ASDFGH")?.key);
  assert.equal(h.agents.defaults.model.primary, "pooled/qwen3-1.7b");
});

// ---------------- asks through the gate ----------------
const joinedRoom = (node, extra = {}) => ({ s: { mode: "join", signal: null, waitSeconds: 1 }, code: "K7QXAB", key: null, node: { hosting: () => false, log() {}, name: "pc (OpenClaw)", ...node },
  P: { Bridge: FakeBridge, setupNode: async () => ({ Peer: null }) }, bridge: null, bridgeTry: null, ...extra });

test("join: an ask while the host hasn't let this device in says so after a while, instead of hanging; refused says so", async () => {
  const r = joinedRoom({ admission: "lobby" });
  const e = await transport(r, { lobbyMs: 60 }).catch((x) => x);
  assert.ok(e instanceof PooledError); assert.equal(e.code, "lobby");
  assert.match(e.message, /waiting for the host of Pooled room K7Q-XAB to let this device in.*\/pooled allow.*invite link/);
  r.refused = "The host didn't let this device in.";
  assert.equal((await admitted(r).catch((x) => x)).code, "denied");
});

test("join: the ask link presents this device's pass (one device for the host); a bridge held in the lobby says so and is reused", async () => {
  FakeBridge.made = []; FakeBridge.how = "in";
  const r = joinedRoom({ admission: "in", pass: PASS });
  const t = await transport(r, { lobbyMs: 60 });
  assert.equal(FakeBridge.made[0].sentPass, PASS);
  assert.equal(t.hostMeta.ctx, 65536);
  FakeBridge.made = []; FakeBridge.how = "hang";
  const r2 = joinedRoom({ admission: "in" });
  assert.equal((await bridgeFor(r2, { lobbyMs: 40 }).catch((x) => x)).code, "lobby");
  assert.equal((await bridgeFor(r2, { lobbyMs: 40 }).catch((x) => x)).code, "lobby");
  assert.equal(FakeBridge.made.length, 1, "still the same knock");
  FakeBridge.how = "refused";
  const r3 = joinedRoom({ admission: "in" });
  assert.equal((await bridgeFor(r3).catch((x) => x)).code, "denied");
});

test("host: an ask while the model downloads says how far it is", async () => {
  const r = { s: { model: "qwen3-1.7b", waitSeconds: 1 }, code: "4TKG9P", node: { ai: { online: false } },
    pull: { key: "qwen3-1.7b", state: "running", done: 2 ** 30 * 0.8, total: 2 ** 30 * 1.8, bps: 50 * 2 ** 20 } };
  const e = await ensureOnline(r).catch((x) => x);
  assert.equal(e.code, "downloading");
  assert.match(e.message, /downloading Qwen3 1\.7B for room 4TK-G9P: 44% \(819 MB of 1\.8 GB\), about 20s left/);
});

// ---------------- /pooled ----------------
function hostRoom() {
  const waiting = [{ id: "a", name: "phone", line: "phone wants to join (iPhone, 0.5 GB)" }, { id: "b", name: "otter", line: "otter wants to join (Mac, 8 GB)" }];
  const r = {
    s: { mode: "host", model: "qwen3-1.7b" }, code: "4TKG9P", link: `https://pooled.run/r/4TKG9P#k=${KEY}`, persisted: 0,
    node: { gate: {}, ai: {}, waitingJoins: () => waiting.slice(), allowJoin: async (id) => { const i = waiting.findIndex((w) => w.id === id); return i < 0 ? null : waiting.splice(i, 1)[0]; },
      denyJoin: (id) => { const i = waiting.findIndex((w) => w.id === id); return i < 0 ? null : waiting.splice(i, 1)[0]; }, setPledge: (gb) => gb },
    status: () => ({ mode: "host", online: false, devices: [{ name: "gb10 (OpenClaw)", gb: 2, self: true }], pledgedGB: 2, waiting: waiting.slice() }),
  };
  r.persist = () => { r.persisted++; };
  return r;
}
test("/pooled: the room, the link, and Allow / Deny for devices waiting to join", async () => {
  const r = hostRoom(), get = async () => r;
  const st = await runPooledCommand("", { getRoom: get });
  assert.match(st, /Pooled room 4TK-G9P · this gateway hosts it/);
  assert.ok(st.includes(r.link));
  assert.match(st, /1\. phone wants to join .*\n  2\. otter wants to join/);
  assert.match(st, /short|add a device/i, "says the room can't hold the model yet");
  assert.equal(await runPooledCommand("allow 2", { getRoom: get }), "Let in: otter");
  assert.equal(await runPooledCommand("deny", { getRoom: get }), "Turned away: phone");
  assert.equal(await runPooledCommand("allow", { getRoom: get }), "Nobody is waiting to join.");
  assert.equal(r.persisted, 2, "the gate is saved after each");
  assert.match(await runPooledCommand("link", { getRoom: get }), /#k=/);
  assert.match(await runPooledCommand("pledge 12", { getRoom: get }), /now lends 12 GB/);
  assert.match(await runPooledCommand("pledge x", { getRoom: get }), /0\.5 to 64/);
  assert.match(await runPooledCommand("", { getRoom: async () => null }), /isn't running in this gateway/);
  const joined = { ...hostRoom(), s: { mode: "join" }, status: () => ({ mode: "join", admission: "lobby", devices: [] }) };
  assert.match(await runPooledCommand("allow", { getRoom: async () => joined }), /Only the room's host/);
  assert.match(statusText(joined), /Waiting for the host to let this device in/);
});

test("/pooled allow all lets every waiting device in", async () => {
  const r = hostRoom();
  assert.equal(await runPooledCommand("allow all", { getRoom: async () => r }), "Let in: phone, otter");
});

// ---------------- the model cache ----------------
test("pull lock: one writer per model; a dead holder's lock is taken over", async () => {
  const dir = tmp("lock");
  const rel = tryLock(dir, "m");
  assert.ok(rel);
  assert.equal(tryLock(dir, "m"), null, "held by a live process");
  rel();
  assert.ok(!fs.existsSync(lockPath(dir, "m")));
  fs.writeFileSync(lockPath(dir, "m"), JSON.stringify({ pid: 4194301, host: os.hostname(), at: new Date().toISOString() }));
  const rel2 = tryLock(dir, "m", { isAlive: () => false });
  assert.ok(rel2, "stale: taken over"); rel2();
  // waiting for another holder
  const held = tryLock(dir, "m");
  let waited = false;
  const p = lock(dir, "m", { pollMs: 10, onWait: () => { waited = true; } });
  await sleep(30); held();
  (await p)();
  assert.ok(waited);
});

test("download: into <dir>/<model>/ with progress, checked against its SHA-256; a failure is a state, not a throw", async () => {
  const dir = tmp("dl");
  const data = Buffer.alloc(300000, 7);
  const sha = createHash("sha256").update(data).digest("hex");
  const models = { MODELS: { tiny: { gguf: "https://x/tiny.gguf" } }, FILES: { tiny: { bytes: data.length, sha256: sha } } };
  const fetch = async (url, o = {}) => {
    const from = +(/bytes=(\d+)-/.exec(o.headers?.range || "")?.[1] || 0);
    const body = data.subarray(from);
    return new Response(new ReadableStream({ start(c) { for (let i = 0; i < body.length; i += 65536) c.enqueue(body.subarray(i, i + 65536)); c.close(); } }),
      { status: from ? 206 : 200, headers: { "content-length": String(body.length), ...(from ? { "content-range": `bytes ${from}-${data.length - 1}/${data.length}` } : {}) } });
  };
  const seen = [];
  const st = await download(dir, "tiny", { fetch, models, onChange: (s) => seen.push(s.state) });
  assert.equal(st.state, "done", st.error);
  assert.equal(fs.statSync(path.join(dir, "tiny", "tiny.gguf")).size, data.length);
  assert.ok(seen.includes("running"));
  assert.ok(!fs.existsSync(lockPath(dir, "tiny")), "the lock is released");
  const bad = await download(tmp("dl2"), "tiny", { models, fetch: async () => new Response("no", { status: 404 }) });
  assert.equal(bad.state, "error"); assert.match(pullLine(bad), /download failed: .*404/);
  assert.match(pullLine({ ...pullState("qwen3-1.7b"), state: "running", done: 2 ** 29, total: 2 ** 30, bps: 2 ** 27 }), /^50% \(512 MB of 1\.0 GB\), about 4s left$/);
  assert.match(downloadingMessage({ ...pullState("qwen3-1.7b"), state: "waiting" }, "4TK-G9P"), /waiting for another download/);
});

// ---------------- warm-up ----------------
test("warm-up: the last system prompt and tools are kept (0600) and replayed once for a one-token answer", async () => {
  assert.equal(fixedPart({ system: "s", messages: [] }), null, "a v1 ask has no tools to pin");
  const body = { api: 2, system: "You are OpenClaw.", messages: [{ role: "user", text: "secret question" }], tools: [{ name: "read", parameters: {} }], params: { maxTokens: 4096, toolChoice: "auto", thinking: false, client: "OpenClaw" } };
  const f = fixedPart(body);
  assert.deepEqual(Object.keys(f).sort(), ["api", "params", "system", "tools"]);
  assert.ok(!JSON.stringify(f).includes("secret question"), "the conversation itself is not kept");
  assert.equal(remember(body, "qwen3-1.7b"), true);
  assert.equal(remember(body, "qwen3-1.7b"), false, "unchanged: not written again");
  assert.equal(mode(prewarmFile()), 0o600);
  const asked = [];
  const r = { node: { request: (b, h, { rid }) => { asked.push(b); setTimeout(() => { h({ t: "ai-genstart", rid, promptTokens: 17400 }); h({ t: "ai-gendone", rid, reason: "length" }); }, 5); } } };
  const logs = [];
  const got = await prewarm(r, { log: (m) => logs.push(m) });
  assert.equal(got.promptTokens, 17400);
  const b = asked[0];
  assert.deepEqual([b.messages.length, b.params.maxTokens, b.params.toolChoice, b.tools.length, b.system], [1, 1, "auto", 1, "You are OpenClaw."]);
  assert.match(logs[0], /warmed up OpenClaw's system prompt and tools: 17400 tokens/);
  assert.equal(replayBody(null), null);
});

// ---------------- review fixes ----------------
test("the invite key never reaches chat text or logs: waiting / memory / degraded name /pooled link instead", async () => {
  const link = `https://pooled.run/r/4TKG9P#k=${KEY}`;
  const base = { code: "4TKG9P", link, shareLink: "https://pooled.run/r/4TKG9P", s: { model: "qwen3-1.7b", waitSeconds: 0, minDevices: 2, ctx: null } };
  const node = { ai: { online: false }, status: () => ({ devices: [{ name: "a", gb: 1 }], pledgedGB: 1 }), waitingJoins: () => [] };
  const e1 = await ensureOnline({ ...base, node }, { waitMs: 0 }).catch((x) => x);
  assert.equal(e1.code, "waiting");
  assert.ok(!e1.message.includes(KEY) && /\/pooled link/.test(e1.message), e1.message);
  const e2 = await ensureOnline({ ...base, s: { ...base.s, minDevices: 1 }, node }, { waitMs: 0 }).catch((x) => x);
  assert.equal(e2.code, "memory");
  assert.ok(!e2.message.includes(KEY), e2.message);
  const deg = { ai: { online: true, degraded: true, engine: {} }, whole: () => false, missingNames: () => ["mac"] };
  const e3 = await ensureOnline({ ...base, node: deg }, { waitMs: 0 }).catch((x) => x);
  assert.equal(e3.code, "degraded");
  assert.ok(!e3.message.includes(KEY), e3.message);
  const { busyMessage } = await import("../src/stream.js");
  assert.ok(!busyMessage({ code: "degraded" }, base).includes(KEY));
});

test("/pooled allow with a word that isn't a number or all lets nobody in", async () => {
  const r = hostRoom();
  assert.match(await runPooledCommand("allow everyone", { getRoom: async () => r }), /takes a number/);
  assert.equal(r.node.waitingJoins().length, 2);
});

test("download: a model key that isn't a known model never becomes a path", async () => {
  const dir = tmp("badkey");
  const st = await download(dir, "../../escape", { fetch: async () => { throw new Error("no fetch expected"); } });
  assert.equal(st.state, "error");
  assert.match(st.error, /unknown model/);
  assert.ok(!fs.existsSync(path.join(dir, "..", "..", "escape")));
});

test("pull lock: a stale lock another process already took over is left alone", () => {
  const dir = tmp("lock2");
  fs.mkdirSync(path.join(dir, "m"), { recursive: true });
  fs.writeFileSync(lockPath(dir, "m"), JSON.stringify({ pid: 4194301, host: os.hostname(), at: "x" }));
  const fresh = JSON.stringify({ pid: 4194302, host: os.hostname(), at: "y" });
  // between our read of the dead holder's lock and our rm, another process takes it over
  const rel = tryLock(dir, "m", { isAlive: (pid) => { if (pid === 4194301) { fs.writeFileSync(lockPath(dir, "m"), fresh); return false; } return true; } });
  assert.equal(rel, null, "the new holder keeps it");
  assert.equal(fs.readFileSync(lockPath(dir, "m"), "utf8"), fresh);
});

test("webgpu (optional) didn't install: the GPU path says how to add it, as an install error", async () => {
  const { dawnFor } = await import("../src/runtime.js");
  const e = await dawnFor(async () => { throw Object.assign(new Error("no webgpu"), { type: "dawn-missing" }); })().catch((x) => x);
  assert.equal(e.code, "install");
  assert.match(e.message, /npm install webgpu@\d/);
  const b = await dawnFor(async () => { throw Object.assign(new Error("dawn.node: bad ELF"), { type: "dawn-broken", hint: "reinstall it" }); })().catch((x) => x);
  assert.equal(b.code, "install");
  assert.match(b.message, /bad ELF\. reinstall it/);
});
