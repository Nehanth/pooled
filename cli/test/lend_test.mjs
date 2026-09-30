// pooled join / pooled host without a GPU or a network: argument parsing, the memory rule, the
// status line, error messages, and loading Dawn lazily (cli/lib/lend.js, cli/lib/dawn.js).
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseLendArgs, parseGb, autoRedeal, detectMemory, memoryRule, afterCheck, formatStatus, tpsFromStats, passCounter, explainError,
  versionAdvice, versionFromBye, hostable, deviceName, UsageError, HELP_JOIN, HELP_HOST, DESK_MAX_GB } from "../lib/lend.js";
import { dawnLoader, dawnPackages, DAWN_VERSION } from "../lib/dawn.js";

const GB = 2 ** 30;
const MODELS = { "qwen3-1.7b": { kind: "gguf" }, "qwen3.6-35b-moe": { kind: "qwen35" }, "smollm-135m": { kind: "safetensors" } };

test("join: a code or a link, and the defaults", () => {
  const o = parseLendArgs("join", ["k7qx"]);
  assert.equal(o.code, "K7QX");
  assert.equal(o.gb, null);
  assert.equal(o.check, true);
  assert.equal(o.waitMs, 10 * 60000);
  assert.equal(parseLendArgs("join", ["https://pooled.run/room/AB2C"]).code, "AB2C");
  assert.equal(parseLendArgs("join", ["https://pooled.run/room?code=zz9z"]).code, "ZZ9Z");
  const p = parseLendArgs("join", ["ABCD", "--gb", "12", "--name", "mac", "--signal", "127.0.0.1:9000", "--models", "/m", "--no-check", "--wait", "0.5", "--json-log"]);
  assert.deepEqual(p.gb, { gb: 12 });
  assert.equal(p.name, "mac"); assert.equal(p.signal, "127.0.0.1:9000"); assert.equal(p.modelDir, "/m");
  assert.equal(p.check, false); assert.equal(p.waitMs, 30000); assert.equal(p.jsonLog, true);
  assert.equal(parseLendArgs("join", ["--help"]).help, true);
  assert.equal(parseLendArgs("join", ["-h"]).help, true);
});

test("join: bad arguments are usage errors that say what is wrong", () => {
  const bad = (argv, re) => assert.throws(() => parseLendArgs("join", argv), (e) => e instanceof UsageError && re.test(e.message), argv.join(" "));
  assert.equal(parseLendArgs("join", []).code, null);   // no code: a terminal asks for it, a script gets the usage error
  assert.equal(parseLendArgs("join", []).askCode, true);
  bad(["not a code!"], /"not a code!" is not a room code/);
  assert.throws(() => parseLendArgs("join", ["not a code!"]), (e) => e.lines.length === 1 && e.lines[0] === 'pooled join: "not a code!" is not a room code (like 4TK-G9P) or an invite link.');
  // an unknown option: one line with the nearest real one, and where the rest are
  assert.throws(() => parseLendArgs("join", ["ABCD", "--gbb", "4"]), (e) => e.lines.join("\n") === 'pooled join: unknown option "--gbb". Did you mean "--gb"?\nRun pooled join --help for all options.');
  assert.throws(() => parseLendArgs("host", ["--strat"]), (e) => e.lines[0] === 'pooled host: unknown option "--strat". Did you mean "--start"?');
  bad(["ABCD", "EFGH"], /one room code/);
  bad(["ABCD", "--gb", "lots"], /--gb must be a number/);
  bad(["ABCD", "--gb", "0.5"], /at least 1/);
  bad(["ABCD", "--gb", "-3"], /--gb/);
  bad(["ABCD", "--gb=-3"], /--gb must be a number/);
  bad(["ABCD", "--wait", "soon"], /--wait/);
  bad(["ABCD", "--frobnicate"], /frobnicate/);
  bad(["ABCD", "--name", "\u0007"], /--name/);
});

test("host: model, devices, code, ctx", () => {
  const o = parseLendArgs("host", [], { models: MODELS });
  assert.equal(o.model, "qwen3-1.7b");
  assert.equal(o.devices, undefined);
  const p = parseLendArgs("host", ["--model", "qwen3.6-35b-moe", "--devices", "3", "--code", "k7qx", "--ctx", "32768", "--gb", "max"], { models: MODELS });
  assert.equal(p.model, "qwen3.6-35b-moe"); assert.equal(p.devices, 3); assert.equal(p.roomCode, "K7QX"); assert.equal(p.ctx, 32768);
  assert.deepEqual(p.gb, { max: true });
  assert.equal(parseLendArgs("host", ["--model", "list"], { models: MODELS }).model, "list");
  const bad = (argv, re) => assert.throws(() => parseLendArgs("host", argv, { models: MODELS }), (e) => e instanceof UsageError && re.test(e.message), argv.join(" "));
  bad(["--model", "gpt-9"], /unknown model "gpt-9"; one of: qwen3-1.7b, qwen3.6-35b-moe/);
  bad(["--model", "smollm-135m"], /unknown model/);   // no loader for safetensors in the node
  bad(["--devices", "0"], /--devices/);
  bad(["--devices", "2.5"], /--devices/);
  bad(["--code", "IL01"], /without I, L, O, U, 0 or 1/);
  bad(["--ctx", "12"], /--ctx/);
  bad(["ABCD"], /unknown model "ABCD"/);   // the positional is the model now
  bad(["qwen3-1.7b", "--model", "qwen3.6-35b-moe"], /two models/);
  bad(["qwen3-1.7b", "extra"], /one model/);
  assert.deepEqual(hostable(MODELS), ["qwen3-1.7b", "qwen3.6-35b-moe"]);
});

test("--gb: numbers, max, and the per-device cap", () => {
  assert.equal(parseGb(undefined), null);
  assert.deepEqual(parseGb("MAX"), { max: true });
  assert.deepEqual(parseGb("7.5"), { gb: 7.5 });
  assert.deepEqual(parseGb("500"), { gb: DESK_MAX_GB });
});

test("the help texts name every option", () => {
  for (const o of ["--gb", "--name", "--signal", "--no-pull", "--no-check", "--wait", "--json-log", "--quiet", "--verbose", "--help"]) assert.ok(HELP_JOIN.includes(o), o);
  for (const o of ["--model", "--gb", "--devices", "--code", "--ctx", "--name", "--signal", "--no-pull", "--no-check", "--json-log", "--verbose"]) assert.ok(HELP_HOST.includes(o), o);
  // where the models are kept is pooled pull's business (--models / POOLED_MODELS are advanced overrides)
  assert.ok(!HELP_JOIN.includes("--models") && !HELP_HOST.includes("--models"));
});

test("the help texts say what a device in a room sees", () => {
  assert.match(HELP_JOIN, /hidden states/); assert.match(HELP_HOST, /hidden states/);
  assert.match(HELP_JOIN, /only a host of the same name/);
});

test("deviceName: the same for a hostname every run, different across hostnames, never the hostname itself", () => {
  assert.equal(deviceName("mac-studio"), deviceName("mac-studio"));
  assert.match(deviceName("mac-studio", "darwin"), /^mac-[a-z]{3}$/);
  assert.match(deviceName("spark", "linux"), /^linux-[a-z]{3}$/);
  assert.match(deviceName("desk", "win32"), /^pc-[a-z]{3}$/);
  assert.match(deviceName("x", "aix"), /^node-[a-z]{3}$/);
  assert.notEqual(deviceName("mac-studio"), deviceName("spark"));
  assert.ok(!deviceName("alice-laptop").includes("alice"));
  assert.match(deviceName("", "darwin"), /^mac-[a-z]{3}$/);
});

// ---------------- memory ----------------
const noRun = () => { throw new Error("ENOENT"); };
const noRead = () => { throw new Error("ENOENT"); };

test("detectMemory: NVIDIA discrete, several GPUs, GB10 unified, AMD sysfs, Apple silicon, unknown", () => {
  const smi = (out) => (cmd) => { assert.equal(cmd, "nvidia-smi"); return out; };
  let m = detectMemory({ platform: "linux", run: smi("NVIDIA GeForce RTX 5070, 12227, 11530\n"), read: noRead, totalmem: () => 64 * GB, freemem: () => 40 * GB });
  assert.equal(m.kind, "discrete"); assert.equal(m.name, "NVIDIA GeForce RTX 5070");
  assert.ok(Math.abs(m.freeGB - 11530 / 1024) < 1e-9);
  m = detectMemory({ platform: "win32", run: smi("NVIDIA A, 8192, 2000\r\nNVIDIA B, 24576, 20000\r\n"), read: noRead, totalmem: () => 0, freemem: () => 0 });
  assert.equal(m.name, "NVIDIA B"); assert.equal(m.gpus, 2);
  m = detectMemory({ platform: "linux", run: smi("NVIDIA GB10, [N/A], [N/A]\n"), read: noRead, totalmem: () => 121 * GB, freemem: () => 92 * GB });
  assert.equal(m.kind, "unified"); assert.equal(m.totalGB, 121); assert.equal(m.availGB, 92);
  const sys = { "/sys/class/drm/card1/device/mem_info_vram_total": String(16 * GB), "/sys/class/drm/card1/device/mem_info_vram_used": String(2 * GB) };
  m = detectMemory({ platform: "linux", run: noRun, read: (p) => { if (p in sys) return sys[p]; throw new Error("ENOENT"); }, totalmem: () => 0, freemem: () => 0 });
  assert.equal(m.kind, "discrete"); assert.equal(m.freeGB, 14); assert.equal(m.source, "amdgpu sysfs");
  m = detectMemory({ platform: "darwin", arch: "arm64", run: noRun, read: noRead, totalmem: () => 128 * GB, freemem: () => 1 * GB });
  assert.equal(m.kind, "unified"); assert.equal(m.totalGB, 128); assert.equal(m.availGB, undefined);   // macOS free memory leaves out the cache
  assert.equal(detectMemory({ platform: "darwin", arch: "x64", run: noRun, read: noRead, totalmem: () => 0, freemem: () => 0 }).kind, "unknown");
  assert.equal(detectMemory({ platform: "linux", run: noRun, read: noRead, totalmem: () => 0, freemem: () => 0 }).kind, "unknown");
});

test("memoryRule: discrete lends free less 1.5 GB, unified total less max(8 GB, 35%), capped at 64", () => {
  let r = memoryRule({ kind: "discrete", name: "RTX 5070", totalGB: 12, freeGB: 11.3 });
  assert.equal(r.gb, 9.5); assert.match(r.why, /11.3 GB free on RTX 5070, less 1.5 GB kept free/);
  // Mac Studio 128 GB: 128 - 44.8 = 83.2 -> capped at 64
  r = memoryRule({ kind: "unified", name: "Apple silicon", totalGB: 128 });
  assert.equal(r.gb, 64); assert.match(r.why, /less 44.8 GB kept for the system, capped at 64 GB per device/);
  // a 16 GB MacBook: 16 - 8 = 8
  assert.equal(memoryRule({ kind: "unified", totalGB: 16 }).gb, 8);
  // a 24 GB one: 24 - 8.4 = 15.6 -> 15.5
  assert.equal(memoryRule({ kind: "unified", totalGB: 24 }).gb, 15.5);
  // GB10 with most of its memory in use: what is free now, less 2 GB
  r = memoryRule({ kind: "unified", totalGB: 121, availGB: 30 });
  assert.equal(r.gb, 28); assert.match(r.why, /30 GB is free right now, so 28 GB/);
  // max keeps only a margin
  assert.equal(memoryRule({ kind: "discrete", name: "x", freeGB: 11.3 }, { max: true }).gb, 10.5);
  assert.equal(memoryRule({ kind: "unified", totalGB: 16 }, { max: true }).gb, 12);
});

test("memoryRule: --gb wins (and says when it is more than looks free); too little is 'low'; unknown falls back", () => {
  let r = memoryRule({ kind: "discrete", name: "x", freeGB: 6 }, { gb: 12 });
  assert.equal(r.gb, 12); assert.equal(r.over, true); assert.match(r.why, /more than the 4.5 GB that looks free/);
  r = memoryRule({ kind: "discrete", name: "x", freeGB: 20 }, { gb: 12 });
  assert.equal(r.over, false); assert.equal(r.why, "--gb 12");
  r = memoryRule({ kind: "discrete", name: "x", freeGB: 2 });
  assert.equal(r.low, true); assert.equal(r.gb, 0);
  r = memoryRule({ kind: "unknown", why: "no counters" }, null, { maxBufGB: 4 });
  assert.equal(r.gb, 2); assert.match(r.why, /couldn't read this GPU's memory \(no counters\): lending 2 GB; --gb sets it/);
  assert.equal(memoryRule({ kind: "unknown", why: "?" }, { gb: 5 }).gb, 5);
});

test("afterCheck: a test allocation that came up short lowers the pledge", () => {
  const rule = { gb: 20, why: "w", low: false, over: false };
  assert.equal(afterCheck(rule, null), rule);
  assert.equal(afterCheck(rule, 20), rule);
  const r = afterCheck(rule, 12.3);
  assert.equal(r.gb, 11); assert.match(r.why, /gave only 12.3 GB in a test allocation, so 11 GB/);
  assert.equal(afterCheck(rule, 1.5).low, true);
});

// ---------------- status ----------------
test("formatStatus: room, phase, devices, layers, tok/s, passes", () => {
  assert.equal(formatStatus({ code: "K7QX", phase: "online", devices: 3, range: [20, 40], model: "qwen3.6-35b-moe", tps: 24.06, passes: 1204 }),
    "room K7QX · online · 3 devices · layers 20-39 of qwen3.6-35b-moe · 24.1 tok/s · 1204 passes");
  assert.equal(formatStatus({ code: "K7QX", phase: "waiting", devices: 1, passes: 0 }), "room K7QX · waiting for the host to deal layers · 1 device · 0 passes");
  assert.equal(formatStatus({ code: "K7QX", phase: "ready", devices: 2, range: [20, 40], model: "qwen3.6-35b-moe", passes: 0 }), "room K7QX · layers loaded: waiting for the rest of the room · 2 devices · layers 20-39 of qwen3.6-35b-moe · 0 passes");
  assert.equal(formatStatus({ code: "K7QX", phase: "waiting", hosting: true, devices: 1, model: "qwen3-1.7b", passes: 1 }), "room K7QX · waiting for devices · 1 device · qwen3-1.7b · 1 pass");
  assert.equal(formatStatus({ code: "K7QX", phase: "loading", pct: 45, devices: 2, passes: 0 }), "room K7QX · loading layers 45% · 2 devices · 0 passes");
  assert.equal(formatStatus({ code: "K7QX", phase: "online", hosting: true, embed: true, devices: 2, range: [0, 14], model: "qwen3-1.7b", passes: 9 }),
    "room K7QX · online · 2 devices · layers 0-13 + embed/head of qwen3-1.7b · 9 passes");
  assert.match(formatStatus({ code: "K7QX", phase: "rejoining", tries: 3, passes: 0 }), /rejoining \(try 3\)/);
  assert.match(formatStatus({ code: "K7QX", phase: "online", signaling: false, passes: 0 }), /signaling down \(links still up\)$/);
  assert.match(formatStatus({ code: "K7QX", phase: "hostgone", passes: 0 }), /lost the host: knocking/);
  assert.match(formatStatus({ code: "K7QX", phase: "guest", devices: 3, passes: 0 }), /without layers \(the host re-deals to include this device\)/);
  const narrow = formatStatus({ code: "K7QX", phase: "online", devices: 3, range: [20, 40], model: "qwen3.6-35b-moe", tps: 24, passes: 5 }, 30);
  assert.equal(narrow.length, 30); assert.ok(narrow.endsWith("…"));
});

test("passCounter keeps counting over a re-deal (the node's counter restarts) and a new node", () => {
  const c = passCounter(), a = {}, b = {};
  assert.equal(c(a, 0), 0); assert.equal(c(a, 36), 36);
  assert.equal(c(a, 0), 36); assert.equal(c(a, 16), 52);
  assert.equal(c(b, 5), 57);
});

test("tpsFromStats reads the host's answer stats", () => {
  assert.equal(tpsFromStats("48 tok · 21.3 tok/s · 2 devices · prompt 20 tok"), 21.3);
  assert.equal(tpsFromStats("failed: the host left"), null);
  assert.equal(tpsFromStats(undefined), null);
});

// ---------------- errors ----------------
test("errors: each says what happened and what to do", () => {
  const ex = (err, ctx = {}) => explainError(err, { code: "K7QX", mine: 4, ...ctx });
  let x = ex(Object.assign(new Error("no WebGPU adapter"), { type: "no-adapter" }));
  assert.match(x.message, /No WebGPU adapter/); assert.match(x.hint, /Metal.*Vulkan.*D3D12/); assert.match(x.hint, /pooled serve works without a GPU/);
  x = ex(Object.assign(new Error("1.2 GB free on RTX, less 1.5 GB kept free"), { type: "low-memory" }));
  assert.match(x.message, /^Not enough GPU memory to lend: 1.2 GB free/); assert.match(x.hint, /--gb N/);
  x = ex(Object.assign(new Error("no room K7QX"), { code: "room-not-found" }));
  assert.equal(x.message, "No room K7QX."); assert.match(x.hint, /still open/);
  x = ex(Object.assign(new Error("no signaling server answered"), { type: "signaling-down", tried: ["0.peerjs.com"] }));
  assert.match(x.message, /signaling server \(0.peerjs.com\)/); assert.match(x.hint, /--signal/);
  x = ex(Object.assign(new Error("peer error: unavailable-id"), { type: "unavailable-id" }), { cmd: "host" });
  assert.match(x.message, /Room code K7QX is taken/);
  x = ex({ type: "version", theirs: 5, theyHost: true });
  assert.match(x.message, /newer version of Pooled \(protocol 5, this pooled 4\)\. Update: npx @pooled\/cli@latest join K7QX/);
  x = ex({ type: "version", theirs: 3, theyHost: true });
  assert.match(x.message, /older version of Pooled \(protocol 3, this pooled 4\)\. Ask the host to reload/);
  x = ex(new Error("GPU error: Out of memory while creating buffer"));
  assert.match(x.message, /ran out of memory/); assert.match(x.hint, /--gb N/);
  x = ex(Object.assign(new Error("go away\x1b[2J"), { type: "kicked" }));
  assert.ok(!/\x1b/.test(x.message));
  x = ex({ type: "other-host", was: "host-ab", now: "stranger" });
  assert.match(x.message, /Room K7QX now has another host \(stranger, not host-ab\).*pooled left it/); assert.match(x.hint, /pooled join K7QX/);
  x = ex({ type: "room-over" });
  assert.match(x.message, /Room K7QX is over/);
  for (const e of [new Error("x"), { type: "no-adapter" }, { type: "version", theirs: 9 }]) assert.equal(ex(e).code, 1);
});

test("version: advice both ways, and reading the host's bye", () => {
  assert.match(versionAdvice({ mine: 4, theirs: "x" }), /different version/);
  assert.match(versionAdvice({ mine: 4, theirs: 3, theyHost: false }), /A device in this room is on an older version.*Ask them/);
  // a browser host's bye (room/errors.js versionLine, for the reader: "protocol <host's>, this tab <ours>")
  assert.deepEqual(versionFromBye("This room's host is on a newer version of Pooled (protocol 5, this tab 4). Reload this page to update, then join again.", 4), { theirs: 5 });
  assert.deepEqual(versionFromBye("This room's host is on an older version of Pooled (protocol 3, this tab 4). Ask the host to reload the page, then join again.", 4), { theirs: 3 });
  // a room node host's bye
  assert.deepEqual(versionFromBye("node-abc speaks room protocol 5, this device 4: reload the older one", 4), { theirs: 5 });
  assert.equal(versionFromBye("the host closed the room", 4), null);
});

// ---------------- Dawn, lazily ----------------
test("dawn: per-OS package first, then webgpu; a clear message when neither is installed", async () => {
  assert.deepEqual(dawnPackages("linux", "x64"), ["@pooled/dawn-linux-x64", "webgpu"]);
  assert.deepEqual(dawnPackages("darwin", "arm64"), ["@pooled/dawn-darwin-universal", "webgpu"]);
  const fake = { create: () => ({}), globals: { GPUBufferUsage: {} } };
  const tried = [];
  let got = await dawnLoader("join", { resolve: (n) => { tried.push(n); if (n === "webgpu") return "/x/webgpu/index.js"; throw new Error("MODULE_NOT_FOUND"); }, load: async () => fake })();
  assert.equal(got.create, fake.create);
  assert.equal(tried.length, 2); assert.equal(tried[1], "webgpu");
  await assert.rejects(dawnLoader("join", { resolve: () => { throw new Error("MODULE_NOT_FOUND"); } })(), (e) => {
    assert.equal(e.type, "dawn-missing");
    assert.match(e.message, /pooled join needs Dawn/);
    assert.ok(e.hint.includes(`webgpu@${DAWN_VERSION}`)); assert.match(e.hint, /npx -p @pooled\/cli -p webgpu@0.6.1 pooled join <CODE>/);
    return true;
  });
  await assert.rejects(dawnLoader("host", { resolve: () => "/x", load: async () => { throw new Error("libvulkan.so.1: cannot open shared object file\nmore"); } })(), (e) => {
    assert.equal(e.type, "dawn-broken"); assert.match(e.message, /did not load: libvulkan.so.1/); assert.match(e.hint, /glibc 2.38/);
    return true;
  });
  const x = explainError(Object.assign(new Error("m"), { type: "dawn-missing", hint: "h" }));
  assert.equal(x.hint, "h");
});

test("Dawn installs with the package but never blocks it: webgpu is an optional dependency", async () => {
  const { readFileSync } = await import("node:fs");
  const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
  assert.equal(pkg.dependencies.webgpu, undefined);          // a failed Dawn install must not fail npm i
  assert.equal(pkg.optionalDependencies?.webgpu, "0.6.1");   // the engine is tested on this release (dawn.js DAWN_VERSION)
  assert.equal(pkg.peerDependencies?.webgpu, undefined);
});

test("autoRedeal: --devices re-deals when N are back, only for devices the last deal did not see", () => {
  const base = { online: true, busy: false, starting: false };
  // the room re-dealt without the mac (it was away); the mac is back under the same id
  assert.equal(autoRedeal(2, { ...base, chain: [], peers: ["mac"], dealt: new Set() }), true);
  // ... but not while an answer runs, a deal loads, or the room is not online
  assert.equal(autoRedeal(2, { ...base, busy: true, chain: [], peers: ["mac"], dealt: new Set() }), false);
  assert.equal(autoRedeal(2, { ...base, starting: true, chain: [], peers: ["mac"], dealt: new Set() }), false);
  assert.equal(autoRedeal(2, { ...base, online: false, chain: [], peers: ["mac"], dealt: new Set() }), false);
  // the last deal saw it and left it out (a phone, a load death): no loop
  assert.equal(autoRedeal(2, { ...base, chain: [], peers: ["phone"], dealt: new Set(["phone"]) }), false);
  // enough devices hold layers already, or not enough are in the room, or no --devices
  assert.equal(autoRedeal(2, { ...base, chain: ["mac"], peers: ["mac", "pc"], dealt: new Set(["mac"]) }), false);
  assert.equal(autoRedeal(3, { ...base, chain: [], peers: ["mac"], dealt: new Set() }), false);
  assert.equal(autoRedeal(undefined, { ...base, chain: [], peers: ["mac"], dealt: new Set() }), false);
});

test("join security: a quoted invite link gives its key; six-character codes grouped or not; host --allow-all and --code 4TK-G9P", () => {
  const KEY = "AbCdEfGhIjKlMnOpQrStUv";
  const j = parseLendArgs("join", [`https://pooled.run/r/4TKG9P#k=${KEY}`]);
  assert.equal(j.code, "4TKG9P"); assert.equal(j.key, KEY);
  assert.equal(parseLendArgs("join", ["4tk-g9p"]).key, null);
  assert.equal(parseLendArgs("join", ["4tk-g9p"]).code, "4TKG9P");
  assert.throws(() => parseLendArgs("join", ["4TKG9"]), (e) => e instanceof UsageError && /like 4TK-G9P/.test(e.lines[0]));
  const h = parseLendArgs("host", ["--allow-all", "--code", "4tk-g9p"], { models: MODELS });
  assert.equal(h.allowAll, true); assert.equal(h.roomCode, "4TKG9P");
  assert.equal(parseLendArgs("host", [], { models: MODELS }).allowAll, false);
  assert.throws(() => parseLendArgs("host", ["--code", "4TKG9"], { models: MODELS }), UsageError);
  assert.match(HELP_HOST, /--allow-all/); assert.match(HELP_HOST, /press a to allow/);
  assert.match(HELP_JOIN, /invite link/);
  assert.equal(formatStatus({ code: "4TKG9P", phase: "lobby", passes: 0 }), "room 4TK-G9P · waiting for the host to let you in · 0 passes");
  assert.equal(formatStatus({ code: "4TKG9P", phase: "online", hosting: true, devices: 1, lobby: 2, passes: 0 }), "room 4TK-G9P · online · 1 device · 2 waiting to join · 0 passes");
});

test("leaving: a room node whose close hangs (a link dialed to a gone host) still lets pooled join exit within a second", async () => {
  const { closeSoon, CLOSE_WAIT_MS } = await import("../lib/lendrun.js");
  assert.ok(CLOSE_WAIT_MS <= 1000);
  let closed = 0;
  const t0 = Date.now();
  await closeSoon({ close: () => { closed++; return new Promise(() => {}); } }, 50);
  assert.equal(closed, 1); assert.ok(Date.now() - t0 < 500);
  await closeSoon({ close: async () => { throw new Error("already closed"); } });
  await closeSoon(null);
});
