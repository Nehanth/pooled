// pooled host's terminal screen (cli/lib/hostui.js): the model list, pledge defaults, the room's fit
// with the room page's math, Start gating, the keys, and flags that skip every question.
import { test } from "node:test";
import assert from "node:assert/strict";
import { MODELS, FILES, NEED_GB, roomBytes } from "../../room/models.js";
import { roomFit, shortNote, shortBy, gbUp } from "../../room/plan.js";
import { pledgeGB } from "../../room/pledge.js";
import { nodeCtxFor } from "../../packages/room-node/roomnode.js";
import { modelRows, recommendModel, pledgeDefaults, roomFitNow, initialState, reduce, canStart, autoStart, render, deviceKind, clip, visible, colors, modelNeedGB } from "../lib/hostui.js";
import { keysOf } from "../lib/tui.js";
import { parseLendArgs } from "../lib/lend.js";

const lib = { MODELS, FILES, NEED_GB, roomBytes, roomFit, shortNote, shortBy, gbUp, pledgeGB, nodeCtxFor };
const KEYS = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3.8-27b", "qwen3.6-35b-moe"];
const dev = (name, gb, extra = {}) => ({ name, meta: { webgpu: true, contribGB: gb, ...extra } });

test("the model list: smallest first, sizes, downloaded or not, and the recommendation for this GPU", () => {
  const rows = modelRows(lib, { keys: KEYS, pulled: new Set(["qwen3-1.7b"]), pledgeGB: 32 });
  assert.deepEqual(rows.map((r) => r.key), ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3.8-27b", "qwen3.6-35b-moe"]);
  const moe = rows.find((r) => r.key === "qwen3.6-35b-moe");
  assert.equal(moe.fileBytes, 20836243072); assert.equal(moe.pulled, false);
  assert.equal(moe.needGB, modelNeedGB(lib, "qwen3.6-35b-moe"));
  assert.ok(moe.needGB > 20 && moe.needGB < 26, `the MoE at the node's context: ${moe.needGB}`);
  assert.equal(rows.find((r) => r.key === "qwen3-1.7b").pulled, true);
  assert.equal(recommendModel(rows), "qwen3.6-35b-moe", "32 GB holds the MoE alone");
  assert.equal(recommendModel(modelRows(lib, { keys: KEYS, pledgeGB: 8 })), "qwen3-4b", "8 GB: the largest that fits");
  assert.equal(recommendModel(modelRows(lib, { keys: ["qwen3-1.7b", "qwen3.6-35b-moe"], pledgeGB: 1 })), "qwen3-1.7b", "nothing fits: the smallest");
});

test("pledge defaults: half the GPU as the room page, at least the smallest model's need, within the most it can lend", () => {
  assert.deepEqual(pledgeDefaults({ kind: "unified", totalGB: 128 }, { maxGB: 83, ruleGB: 76, smallestNeedGB: 4 }), { def: 64, max: 64, totalGB: 128 });
  assert.deepEqual(pledgeDefaults({ kind: "discrete", totalGB: 12 }, { maxGB: 11, ruleGB: 10, smallestNeedGB: 3.8 }), { def: 6, max: 11, totalGB: 12 });
  assert.equal(pledgeDefaults({ kind: "discrete", totalGB: 6 }, { maxGB: 5, ruleGB: 4.5, smallestNeedGB: 3.8 }).def, 4, "half is 3: raised to what the smallest model needs");
  assert.equal(pledgeDefaults({ kind: "discrete", totalGB: 4 }, { maxGB: 3, ruleGB: 2.5, smallestNeedGB: 3.8 }).def, 3, "never more than it can lend");
  assert.deepEqual(pledgeDefaults({ kind: "unknown" }, { maxGB: 0, ruleGB: 8 }), { def: 8, max: 8, totalGB: null });
});

test("a short room: the same sentence and number as the room page; pledges that fit clear it", () => {
  const moe = "qwen3.6-35b-moe";
  const one = roomFitNow(lib, { model: moe, devices: [dev("spark", 12)], spareGB: [0] });
  assert.equal(one.fits, false);
  assert.ok(one.shortGB > 9 && one.shortGB < 15, `short ${one.shortGB}`);
  assert.match(one.note, /^This room is [\d.]+ GB short for Qwen3.6 35B MoE: add a device or raise a pledge/);
  // the room page's own computation, byte for byte
  const rb = roomBytes(moe, nodeCtxFor(moe), "f16");
  assert.equal(one.shortGB, gbUp(roomFit(rb.L, [12 * 2 ** 30], rb.layerBytes, rb.hostBytes).short));
  const two = roomFitNow(lib, { model: moe, devices: [dev("spark", 12), dev("mac", 16, { native: "node-dawn" })] });
  assert.equal(two.fits, true); assert.equal(two.haveGB, 28); assert.equal(two.note, "");
  // a phone counts as the room page counts it: capped at 1 GB whatever it says
  const ph = roomFitNow(lib, { model: "qwen3-1.7b", devices: [dev("pc", 2), dev("iphone", 8, { ua: "iPhone" })] });
  assert.equal(ph.haveGB, 3);
});

test("Start stays off until the pledges fit and the download is done; --start and --wait start by themselves", () => {
  const rows = modelRows(lib, { keys: KEYS, pulled: new Set(["qwen3-1.7b"]), pledgeGB: 12 });
  let s = initialState({ rows, model: "qwen3.6-35b-moe", pledge: { gb: 12, max: 60, totalGB: 128 }, fixedPledge: true, pulled: new Set(["qwen3.6-35b-moe"]), flags: { start: true } });
  assert.equal(s.step, "room");
  s.devices = [{ name: "spark", gb: 12 }];
  s.fit = roomFitNow(lib, { model: s.model, devices: [dev("spark", 12)] });
  assert.equal(canStart(s).ok, false); assert.match(canStart(s).why, /GB short/);
  assert.equal(autoStart(s), false);
  let r = reduce(s, "enter");
  assert.equal(r.state.step, "room"); assert.deepEqual(r.fx, []); assert.match(r.state.notice, /short/);
  s.devices.push({ name: "mac", gb: 16 });
  s.fit = roomFitNow(lib, { model: s.model, devices: [dev("spark", 12), dev("mac", 16)] });
  assert.equal(canStart(s).ok, true); assert.equal(autoStart(s), true);
  r = reduce(s, "enter");
  assert.equal(r.state.step, "starting"); assert.deepEqual(r.fx, [{ do: "start" }]);
  // --wait 3: two devices are not enough even though they fit
  assert.equal(autoStart({ ...s, flags: { wait: 3 } }), false);
  assert.equal(autoStart({ ...s, flags: { wait: 2 } }), true);
  // no flags: never by itself
  assert.equal(autoStart({ ...s, flags: {} }), false);
  // a download in progress holds Start
  assert.equal(canStart({ ...s, dl: { key: s.model, state: "running" } }).ok, false);
});

test("keys: pick a model, a download starts when it is not here; pledge by arrows or typing; m / p go back", () => {
  const rows = modelRows(lib, { keys: ["qwen3-1.7b", "qwen3.6-35b-moe"], pulled: new Set(["qwen3-1.7b"]), pledgeGB: 32 });
  let s = initialState({ rows, pledge: { gb: 32, max: 60, totalGB: 128 } });
  assert.equal(s.step, "pick");
  assert.equal(s.rows[s.sel].key, "qwen3.6-35b-moe", "the recommendation is preselected");
  let r = reduce(s, "up"); assert.equal(r.state.rows[r.state.sel].key, "qwen3-1.7b");
  r = reduce(r.state, "down");
  r = reduce(r.state, "enter");
  assert.deepEqual(r.fx, [{ do: "model", key: "qwen3.6-35b-moe" }, { do: "pull", key: "qwen3.6-35b-moe" }]);
  assert.equal(r.state.step, "pledge"); assert.equal(r.state.dl.state, "running");
  s = r.state;
  r = reduce(s, "right"); assert.equal(r.state.pledge.gb, 33);
  r = reduce(r.state, "left"); r = reduce(r.state, "left"); assert.equal(r.state.pledge.gb, 31);
  for (const k of ["9", "9"]) r = reduce(r.state, k);
  assert.equal(r.state.pledge.typed, "99");
  r = reduce(r.state, "enter");
  assert.equal(r.state.pledge.gb, 60, "held to the most it can lend"); assert.match(r.state.notice, /at most 60/);
  assert.deepEqual(r.fx, [{ do: "pledge", gb: 60 }]); assert.equal(r.state.step, "room");
  r = reduce(r.state, "p"); assert.equal(r.state.step, "pledge");
  r = reduce(r.state, "esc"); assert.equal(r.state.step, "room");
  r = reduce(r.state, "m"); assert.equal(r.state.step, "pick");
  r = reduce(r.state, "up"); r = reduce(r.state, "enter");
  assert.deepEqual(r.fx, [{ do: "model", key: "qwen3-1.7b" }], "already here: no download");
  assert.equal(r.state.step, "room", "the pledge was answered once: back to the room");
  // the lobby: a / d answer the oldest request
  r.state.lobby = [{ id: "p1", line: "otter wants to join (Mac, 8 GB)" }];
  assert.deepEqual(reduce(r.state, "a").fx, [{ do: "allow", id: "p1" }]);
  assert.deepEqual(reduce(r.state, "d").fx, [{ do: "deny", id: "p1" }]);
  assert.deepEqual(reduce(r.state, "q").fx, [{ do: "quit" }]);
  // online: c chats here
  assert.deepEqual(reduce({ ...r.state, step: "online" }, "c").fx, [{ do: "chat" }]);
});

test("a model given but not downloaded: [Y/n] in the screen, -y downloads, --no-pull streams", () => {
  const rows = modelRows(lib, { keys: KEYS, pledgeGB: 32 });
  const ask = initialState({ rows, model: "qwen3.6-35b-moe", pledge: { gb: 32, max: 60 } });
  assert.equal(ask.step, "confirm");
  const lines = render(ask, { lib }).join("\n");
  assert.match(lines, /qwen3\.6-35b-moe is not downloaded \(19\.4 GB\)\. Download it now\? \[Y\/n\]/);
  let r = reduce(ask, "enter");
  assert.deepEqual(r.fx, [{ do: "pull", key: "qwen3.6-35b-moe" }]); assert.equal(r.state.step, "pledge");
  r = reduce(ask, "n");
  assert.deepEqual(r.fx, [{ do: "stream", key: "qwen3.6-35b-moe" }]); assert.equal(r.state.dl.state, "stream");
  assert.equal(initialState({ rows, model: "qwen3.6-35b-moe", pledge: { gb: 32 }, yes: true }).dl.state, "running");
  assert.equal(initialState({ rows, model: "qwen3.6-35b-moe", pledge: { gb: 32 }, noPull: true }).dl.state, "stream");
});

test("every choice as a flag: no question at all (pooled host qwen3.6-35b-moe --gb 64 --start --yes)", () => {
  const MODELS2 = { "qwen3-1.7b": { kind: "gguf" }, "qwen3.6-35b-moe": { kind: "qwen35" } };
  const o = parseLendArgs("host", ["qwen3.6-35b-moe", "--gb", "64", "--start", "--yes", "--allow-all", "--chat", "--name", "spark", "--wait", "2"], { models: MODELS2 });
  assert.equal(o.modelGiven, true); assert.equal(o.gbGiven, true); assert.equal(o.start, true); assert.equal(o.yes, true);
  assert.equal(o.allowAll, true); assert.equal(o.chat, true); assert.equal(o.name, "spark"); assert.equal(o.devices, 2);
  assert.throws(() => parseLendArgs("host", ["--allow-all", "--deny-unknown"], { models: MODELS2 }), /opposite/);
  assert.equal(parseLendArgs("host", ["-y", "--no-pull", "--deny-unknown"], { models: MODELS2 }).denyUnknown, true);
  const rows = modelRows(lib, { keys: KEYS, pledgeGB: 64 });
  const s = initialState({ rows, model: o.model, pledge: { gb: 64, max: 64 }, fixedPledge: o.gbGiven, yes: o.yes, flags: { start: o.start } });
  // no picker, no pledge question, no [Y/n]: straight to the room, downloading, and it starts by itself
  assert.equal(s.step, "room"); assert.equal(s.dl.state, "running");
  s.devices = [{ name: "spark", gb: 64 }];
  s.fit = roomFitNow(lib, { model: s.model, devices: [dev("spark", 64)] });
  assert.equal(autoStart(s), false, "not before the download is done");
  s.dl = { ...s.dl, state: "done" };
  assert.equal(autoStart(s), true);
});

test("the screen: fits 80 columns, colors only when on, the room's devices, the lobby and the keys", () => {
  const rows = modelRows(lib, { keys: KEYS, pulled: new Set(["qwen3.6-35b-moe"]), pledgeGB: 12 });
  const s = initialState({ rows, model: "qwen3.6-35b-moe", pledge: { gb: 12, max: 60, totalGB: 128 }, fixedPledge: true, pulled: new Set(["qwen3.6-35b-moe"]), code: "4TKG9P" });
  s.link = "https://pooled.run/r/4TKG9P#k=AbCdEfGhIjKlMnOpQrStUv";
  s.devices = [{ name: "spark", kind: "this computer", gb: 12, self: true }, { name: "node-abc", kind: "CLI", gb: 4 }, { name: "iphone", kind: "phone", gb: 1 }];
  s.lobby = [{ id: "x", line: "otter wants to join (Mac, 8 GB)" }];
  s.fit = roomFitNow(lib, { model: s.model, devices: [dev("spark", 12), dev("node-abc", 4), dev("iphone", 1, { ua: "iPhone" })] });
  const plain = render(s, { width: 80, lib });
  assert.ok(plain.every((l) => visible(l).length <= 79), plain.find((l) => visible(l).length > 79));
  const text = plain.join("\n");
  assert.match(text, /room 4TK-G9P · Qwen3\.6 35B MoE/);
  assert.match(text, /invite\s+https:\/\/pooled\.run\/r\/4TKG9P#k=/);
  assert.match(text, /Devices \(3\)/); assert.match(text, /node-abc\s+CLI\s+4 GB/); assert.match(text, /iphone\s+phone/);
  assert.match(text, /otter wants to join \(Mac, 8 GB\)\s+a: allow  d: deny/);
  assert.match(text, /This room is [\d.]+ GB short for Qwen3\.6 35B MoE/);
  assert.match(text, /Enter: start \(when the room fits\)/);
  assert.doesNotMatch(text, /\x1b\[/, "no color codes when color is off");
  const colored = render(s, { width: 80, lib, c: colors(true) });
  assert.match(colored.join(""), /\x1b\[/);
  assert.ok(colored.every((l) => visible(l).length <= 79));
  // fits: Enter starts
  s.fit = { fits: true, needGB: 22.8, haveGB: 30, shortGB: 0, note: "" };
  assert.match(render(s, { lib }).join("\n"), /^Enter: start  ·  m: model/m);
  assert.equal(clip("\x1b[1mabcdef\x1b[22m", 4).replace(/\x1b\[[0-9;]*m/g, ""), "abc…");
  assert.equal(deviceKind({ native: "node-dawn" }), "CLI"); assert.equal(deviceKind({ ua: "iPhone" }), "phone");
  assert.equal(deviceKind({ ua: "Mac" }), "browser tab"); assert.equal(deviceKind({}, true), "this computer");
});

test("keys from the terminal: arrows, Enter, Backspace, Ctrl-C and pasted digits", () => {
  assert.deepEqual(keysOf(Buffer.from("\x1b[A\x1b[B\x1b[C\x1b[D\r\x7f\x03")), ["up", "down", "right", "left", "enter", "backspace", "ctrl-c"]);
  assert.deepEqual(keysOf(Buffer.from("24\r")), ["2", "4", "enter"]);
  assert.deepEqual(keysOf(Buffer.from("\x1bOA")), ["up"]);
  assert.deepEqual(keysOf(Buffer.from("\x1b")), ["esc"]);
});
