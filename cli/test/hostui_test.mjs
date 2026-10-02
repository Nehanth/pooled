// pooled host's terminal screen (cli/lib/hostui.js): the model list, pledge defaults, the room's fit
// with the room page's math, Start gating, the keys, and flags that skip every question.
import { test } from "node:test";
import assert from "node:assert/strict";
import { MODELS, FILES, NEED_GB, roomBytes, pickCtx, ctxShortNote } from "../../room/models.js";
import { roomFit, shortNote, shortBy, gbUp } from "../../room/plan.js";
import { pledgeGB } from "../../room/pledge.js";
import { nodeCtxFor } from "../../packages/room-node/roomnode.js";
import { modelRows, recommendModel, pledgeDefaults, roomFitNow, initialState, reduce, canStart, autoStart, render, deviceKind, clip, visible, colors, modelNeedGB } from "../lib/hostui.js";
import { width } from "../lib/style.js";
import { keysOf } from "../lib/tui.js";
import { parseLendArgs } from "../lib/lend.js";

const lib = { MODELS, FILES, NEED_GB, roomBytes, roomFit, shortNote, shortBy, gbUp, pledgeGB, nodeCtxFor, pickCtx, ctxShortNote };
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
  let s = initialState({ rows, model: "qwen3.6-35b-moe", pledge: { gb: 12, max: 60, totalGB: 128 }, fixedPledge: true, pulled: new Set(["qwen3.6-35b-moe"]), flags: { start: true, mode: "pool" } });
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
  assert.equal(r.state.step, "how", "then: how to run it"); assert.equal(r.state.dl.state, "running");
  assert.equal(r.state.how, 0, "Pool with devices is preselected");
  r = reduce(r.state, "enter");
  assert.deepEqual(r.fx, [{ do: "split", mode: "memory" }], "pooled: spread across the devices");
  assert.equal(r.state.step, "pledge");
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
  r = reduce(r.state, "enter");
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
  assert.match(lines, /Qwen3\.6 35B MoE is not downloaded \(19\.4 GB\)\.\n  Download it now\?/);
  assert.match(lines, /y download · n stream from Hugging Face instead/);
  let r = reduce(ask, "enter");
  assert.deepEqual(r.fx, [{ do: "pull", key: "qwen3.6-35b-moe" }]); assert.equal(r.state.step, "how");
  r = reduce(ask, "n");
  assert.deepEqual(r.fx, [{ do: "stream", key: "qwen3.6-35b-moe" }]); assert.equal(r.state.dl.state, "stream");
  assert.equal(initialState({ rows, model: "qwen3.6-35b-moe", pledge: { gb: 32 }, yes: true }).dl.state, "running");
  assert.equal(initialState({ rows, model: "qwen3.6-35b-moe", pledge: { gb: 32 }, noPull: true }).dl.state, "stream");
});

test("every choice as a flag: no question at all (pooled host qwen3.6-35b-moe --pool --gb 64 --start --yes)", () => {
  const MODELS2 = { "qwen3-1.7b": { kind: "gguf" }, "qwen3.6-35b-moe": { kind: "qwen35" } };
  const o = parseLendArgs("host", ["qwen3.6-35b-moe", "--pool", "--gb", "64", "--start", "--yes", "--allow-all", "--chat", "--name", "spark", "--wait", "2"], { models: MODELS2 });
  assert.equal(o.modelGiven, true); assert.equal(o.gbGiven, true); assert.equal(o.start, true); assert.equal(o.yes, true);
  assert.equal(o.allowAll, true); assert.equal(o.chat, true); assert.equal(o.name, "spark"); assert.equal(o.devices, 2);
  assert.throws(() => parseLendArgs("host", ["--allow-all", "--deny-unknown"], { models: MODELS2 }), /opposite/);
  assert.equal(parseLendArgs("host", ["-y", "--no-pull", "--deny-unknown"], { models: MODELS2 }).denyUnknown, true);
  const rows = modelRows(lib, { keys: KEYS, pledgeGB: 64 });
  const s = initialState({ rows, model: o.model, pledge: { gb: 64, max: 64 }, fixedPledge: o.gbGiven, yes: o.yes, flags: { start: o.start, mode: o.mode } });
  // no picker, no pledge question, no [Y/n]: straight to the room, downloading, and it starts by itself
  assert.equal(s.step, "room"); assert.equal(s.dl.state, "running");
  s.devices = [{ name: "spark", gb: 64 }];
  s.fit = roomFitNow(lib, { model: s.model, devices: [dev("spark", 64)] });
  assert.equal(autoStart(s), false, "not before the download is done");
  s.dl = { ...s.dl, state: "done" };
  assert.equal(autoStart(s), true);
});

test("the screen: fits 58, 80 and 110 columns in every style, the room's devices, the lobby and the keys", async () => {
  const rows = modelRows(lib, { keys: KEYS, pulled: new Set(["qwen3.6-35b-moe"]), pledgeGB: 12 });
  const s = initialState({ rows, model: "qwen3.6-35b-moe", pledge: { gb: 12, max: 60, totalGB: 128 }, fixedPledge: true, pulled: new Set(["qwen3.6-35b-moe"]), code: "4TKG9P", flags: { mode: "pool", splitGiven: true } });
  s.link = "https://pooled.run/r/4TKG9P#k=AbCdEfGhIjKlMnOpQrStUv";
  s.devices = [{ name: "spark", kind: "this computer", gb: 12, self: true }, { name: "node-abc", kind: "CLI", gb: 4 }, { name: "iphone", kind: "phone", gb: 1 }];
  s.lobby = [{ id: "x", line: "otter wants to join (Mac, 8 GB)" }];
  s.fit = roomFitNow(lib, { model: s.model, devices: [dev("spark", 12), dev("node-abc", 4), dev("iphone", 1, { ua: "iPhone" })] });
  const plain = render(s, { width: 80, lib });
  assert.ok(plain.every((l) => width(l) <= 79), plain.find((l) => width(l) > 79));
  const text = plain.join("\n");
  assert.match(text, /pooled host\n  \[4TK-G9P\]  pooled\.run\/r\/4TKG9P#k=/);
  assert.match(text, /Qwen3\.6 35B MoE · Q4 · waiting for devices/);
  assert.match(text, /DEVICE\s+GPU\s+LENDS/); assert.match(text, /node-abc\s+4 GB\s+joined/); assert.match(text, /spark\s+12 GB\s+this computer/);
  assert.match(text, /otter\s+Mac, 8 GB\s+wants to join  a allow  d deny/);
  assert.match(text, /memory    ━+─*  17 GB lent · 23 GB needed/);
  assert.match(text, /invite    pooled join 4TK-G9P on the other computer/);
  assert.match(text, /Needs \d+ GB more: one more device, or press l to lend more\./);
  assert.match(text, /enter start · i copy invite · m model · l lend · s split · q quit/);
  assert.doesNotMatch(text, /\x1b\[/, "no escapes in the plain style");
  assert.doesNotMatch(text, /[✓❯█░◐]/);
  const { mkStyle } = await import("../lib/style.js");
  for (const [depth, theme] of [["truecolor", "dark"], ["256", "light"], ["none", "dark"]]) {
    for (const w of [58, 80, 110]) {
      const lines = render(s, { width: w, lib, S: mkStyle({ depth, theme }) });
      assert.ok(lines.every((l) => width(l) <= w - 1), `${depth} ${w}: ${lines.find((l) => width(l) > w - 1)}`);
      assert.doesNotMatch(lines.join(""), /\x1b\[(31|32|33|35|36)m/, "no red/green/yellow/magenta/cyan");
      assert.doesNotMatch(lines.join(""), /[✓❯█░◐]/);
    }
  }
  assert.match(render(s, { width: 80, lib, S: mkStyle({ depth: "truecolor" }) }).join(""), /\x1b\[48;2;42;69;224m/, "the pill in the brand blue");
  // fits: enter is the primary key
  s.fit = { fits: true, needGB: 22.8, haveGB: 30, shortGB: 0, note: "" };
  assert.match(render(s, { lib }).join("\n"), /ready to start/);
  assert.doesNotMatch(render(s, { lib }).join("\n"), /invite    pooled join/, "the invite row only while short (or alone)");
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

test("a split that one computer could avoid: say so before Start, with a speed from the links", async () => {
  const { splitAdvice, splitLines, hopHint, pickerRow } = await import("../lib/hostui.js");
  const model = "qwen3-1.7b";
  const devs = [{ name: "spark", self: true, gb: 3, meta: { webgpu: true, contribGB: 3 }, rtt: null },
    { name: "mac", gb: 3, meta: { webgpu: true, contribGB: 3 }, rtt: 60 }];
  const s = { model, devices: devs, pledge: { gb: 3, max: 64 } };
  s.fit = roomFitNow(lib, { model, devices: devs });
  assert.equal(s.fit.fits, true, "3 + 3 GB hold the 1.7B (16k context)");
  const adv = splitAdvice(s, lib);
  assert.ok(adv.alone.gb > 3 && adv.alone.gb <= 6, `alone at ${adv.alone.gb} GB`);
  assert.equal(adv.hopMs, 30); assert.equal(adv.tps, Math.round(1000 / (2 * 30 + 12)));
  const lines = splitLines(adv, { model: "Qwen3 1.7B", selfName: "spark", gb: 3 });
  assert.equal(lines[0], `Qwen3 1.7B would run on spark alone if it lends ${adv.alone.gb} GB (now 3 GB): l lends more.`);
  assert.equal(lines[1], `Split across 2 devices, each token waits for the network: expect roughly ${adv.tps} tok/s.`);
  // no round trips yet: no number; one device holding it all: nothing to say
  assert.match(splitLines(splitAdvice({ ...s, devices: devs.map((d) => ({ ...d, rtt: null })) }, lib), {})[1], /expect it to be slower/);
  const big = devs.map((d) => (d.self ? { ...d, gb: 8, meta: { webgpu: true, contribGB: 8 } } : d));
  assert.equal(splitAdvice({ ...s, devices: big, fit: roomFitNow(lib, { model, devices: big }) }, lib), null);
  assert.equal(hopHint(2, 30), "each token crosses the network twice; ~30 ms per hop");
  assert.equal(hopHint(1, 30), "");
  // the picker in plain words
  assert.deepEqual(pickerRow({ key: "a", pulled: true, needGB: 4.1, fitsAlone: true }, { rec: "a" }), { have: "downloaded", need: "4 GB", min: "", alone: true, rec: true });
  assert.equal(pickerRow({ key: "c", needGB: 5.5, minNeed: { ctx: 8192, needGB: 3.8 } }).min, "4 GB at 8K", "a fallback context: its need there");
  assert.equal(pickerRow({ key: "b", pulled: false, fileBytes: 2 ** 30 * 18.6, needGB: 70.1, fitsAlone: false }).have, "18.6 GB download");
});

test("the host screen: the header once (pill, link, status), the picker as one plain list", () => {
  const rows = modelRows(lib, { keys: KEYS, pledgeGB: 12 });
  const s = initialState({ rows, pledge: { gb: 12, max: 14, totalGB: 16 }, code: "9PFZ8T" });
  Object.assign(s, { link: "https://pooled.run/r/9PFZ8T#k=AbCdEfGhIjKlMnOpQrStUv", gpu: "RTX 5070 Ti · 16 GB", devices: [] });
  const L = render(s, { width: 80, lib }).map(visible);
  assert.equal(L.filter((l) => l.includes("9PF-Z8T")).length, 1);
  assert.equal(L.filter((l) => l.includes("9PFZ8T#k=")).length, 1);
  assert.ok(L.includes("  gpu       RTX 5070 Ti · 16 GB"));
  const text = L.join("\n");
  assert.match(text, /Which model\?\n\n    Qwen3 0\.6B/, "one list, smallest first, no groups");
  assert.doesNotMatch(text, /Fits on this computer|Needs another device/);
  assert.match(text, /\n    Qwen3\.8 27B\s+needs 20 GB\s+15\.0 GB download/);
  assert.match(text, /\n    Qwen3 1\.7B\s+needs  6 GB \(4 GB at 8K\)   1\.7 GB download/, "the 1.7B: 16K, and its 8K fallback");
  assert.match(text, /› Qwen3 4B\s+needs  5 GB/, "the recommended model is preselected, not labelled");
  assert.doesNotMatch(text, /recommended|"needs"/);
  assert.match(text, /↑↓ choose · enter host it · q quit/);
  for (const l of L) assert.ok(l.length <= 79, l);
  const w110 = render(s, { width: 110, lib }).map(visible).join("\n");
  assert.match(w110, /https:\/\/pooled\.run\/r\/9PFZ8T#k=/, "the whole link from 100 columns");
});

test("split: s toggles fastest first / across all devices, the rows show what Start would deal, the choice reaches the node", async () => {
  const { dealRoom } = await import("../../room/plan.js");
  const { parseLendArgs } = await import("../lib/lend.js");
  const L2 = { ...lib, dealRoom };
  const model = "qwen3-1.7b";
  const rows = modelRows(lib, { keys: KEYS, pulled: new Set([model]), pledgeGB: 8 });
  let s = initialState({ rows, model, pledge: { gb: 8, max: 60, totalGB: 128 }, fixedPledge: true, pulled: new Set([model]), code: "4TKG9P", flags: { mode: "pool", splitGiven: true } });
  s.devices = [{ name: "spark", kind: "this computer", gb: 8, self: true, meta: { webgpu: true, contribGB: 8 } }, { name: "mac", kind: "CLI", gb: 8, meta: { webgpu: true, contribGB: 8 } }];
  s.fit = roomFitNow(lib, { model, devices: [dev("spark", 8), dev("mac", 8)] });
  assert.equal(s.splitMode, "speed", "fastest first by default, as the room page");
  let text = render(s, { width: 80, lib: L2 }).join("\n");
  assert.match(text, /HOLDS/);
  assert.match(text, /spark\s+8\.0 GB\s+0–27\s+this computer|spark\s+8 GB\s+0–27\s+this computer/);
  assert.match(text, /mac\s+8 GB\s+—\s+not needed/);
  assert.match(text, /split     fastest first · s changes it/);
  assert.match(text, /s split/);
  assert.doesNotMatch(text, /Spreading over the network/);
  const r = reduce(s, "s");
  assert.deepEqual(r.fx, [{ do: "split", mode: "memory" }]);
  s = r.state;
  text = render(s, { width: 80, lib: L2 }).join("\n");
  assert.match(text, /split     across all devices/);
  assert.match(text, /spark\s+8 GB\s+0–1\d\s+this computer/);
  assert.match(text, /mac\s+8 GB\s+1\d–27\s+joined/);
  assert.match(text, /Spreading over the network is slower per token; it uses less memory on each\s+device\./);
  assert.equal(reduce(s, "s").state.splitMode, "speed");
  // online: s changes it for the next rebalance
  const on = reduce({ ...s, step: "online" }, "s");
  assert.deepEqual(on.fx, [{ do: "split", mode: "speed" }]); assert.match(on.state.notice, /rebalances/);
  // --split, with friendly names
  assert.equal(parseLendArgs("host", []).split, "speed");
  assert.equal(parseLendArgs("host", ["--split", "spread"]).split, "memory");
  assert.equal(parseLendArgs("host", ["--split", "memory"]).split, "memory");
  assert.equal(parseLendArgs("host", ["--split", "fastest"]).split, "speed");
  assert.throws(() => parseLendArgs("host", ["--split", "sideways"]), /--split is "speed"/);
  assert.equal(initialState({ rows, pledge: { gb: 8, max: 60 }, flags: { split: "memory" } }).splitMode, "memory");
});

test("picker: models in size order; ↑ on the first and ↓ on the last stay put (no wrap)", () => {
  const rows = modelRows(lib, { keys: [...KEYS].reverse(), pledgeGB: 8 });
  assert.deepEqual(rows.map((r) => r.fileBytes), [...rows.map((r) => r.fileBytes)].sort((a, b) => a - b), "smallest download first, whatever order the keys came in");
  assert.deepEqual(rows.map((r) => r.key), KEYS, "0.6B, 1.7B, 4B, 27B, MoE (the 1.7B's 16k context needs more than the 4B's 4k: it is still listed by its size)");
  let s = { ...initialState({ rows, pledge: { gb: 8, max: 16, totalGB: 16 } }), sel: 0 };
  s = reduce(s, "up").state;
  assert.equal(s.sel, 0, "↑ at the top stops");
  for (let i = 0; i < rows.length + 2; i++) s = reduce(s, "down").state;
  assert.equal(s.sel, rows.length - 1, "↓ at the bottom stops");
  // the screen lists them in the same order the cursor walks
  const shown = render(s, { lib }).filter((l) => /needs/.test(l)).map((l) => l.replace(/^[\s›]+/, "").split(/\s{2,}/)[0]);
  assert.deepEqual(shown, rows.map((r) => lib.MODELS[r.key].label.split("·")[0].trim()));
});

test("the download ends: \"waiting for the download\" goes, Start can go", async () => {
  const { pullDone } = await import("../lib/hostui.js");
  const rows = modelRows(lib, { keys: KEYS, pledgeGB: 8 });
  let s = initialState({ rows, model: "qwen3-1.7b", pledge: { gb: 8, max: 16, totalGB: 16 }, fixedPledge: true, yes: true, flags: { mode: "pool" } });
  assert.equal(s.dl.state, "running");
  s.fit = roomFitNow(lib, { model: "qwen3-1.7b", devices: [dev("me", 8)] });
  s = reduce(s, "enter").state;
  assert.equal(s.notice, "waiting for the download");
  assert.match(render(s, { lib }).join("\n"), /waiting for the download/);
  s = pullDone(s, "qwen3-1.7b", { ok: true });
  assert.equal(s.dl.state, "done"); assert.equal(s.notice, "");
  assert.doesNotMatch(render(s, { lib }).join("\n"), /waiting for the download/);
  assert.equal(canStart(s).ok, true);
  // a failed one says so instead; another model's download leaves this one alone
  const f = pullDone({ ...s, dl: { ...s.dl, state: "running" }, notice: "waiting for the download" }, "qwen3-1.7b", { error: new Error("HTTP 503") });
  assert.equal(f.dl.state, "error"); assert.equal(f.dl.error, "HTTP 503"); assert.equal(f.notice, "");
  assert.equal(pullDone(s, "qwen3-4b", { ok: true }).dl.key, "qwen3-1.7b");
});

test("how to run it: Pool with devices first and preselected; Run it here lends what the model needs and starts", async () => {
  const { hereGB } = await import("../lib/hostui.js");
  const { mkStyle } = await import("../lib/style.js");
  const rows = modelRows(lib, { keys: KEYS, pulled: new Set(["qwen3-1.7b"]), pledgeGB: 8, maxGB: 14 });
  const r17 = rows.find((r) => r.key === "qwen3-1.7b"), moe = rows.find((r) => r.key === "qwen3.6-35b-moe");
  assert.equal(r17.hereGB, hereGB(lib, "qwen3-1.7b", { maxGB: 14 }));
  assert.ok(r17.hereGB >= 5 && r17.hereGB <= 6, `the 1.7B alone at ${r17.hereGB} GB (16k context)`);
  const at = (g) => roomFitNow(lib, { model: "qwen3-1.7b", devices: [dev("me", g)] });
  assert.ok(at(r17.hereGB).fits && !at(r17.hereGB).fellBack && at(r17.hereGB - 1).fellBack, "the least that holds it at 16K");
  // a computer that can lend 4 GB at most runs it here at 8K; 3 GB can't, and says what 8K needs
  assert.equal(hereGB(lib, "qwen3-1.7b", { maxGB: 4 }), 4);
  assert.equal(hereGB(lib, "qwen3-1.7b", { maxGB: 3 }), null);
  const { hereWhy } = await import("../lib/hostui.js");
  assert.equal(hereWhy(r17, { max: 3 }), "needs 4 GB (at 8K), this computer has 3 GB to lend");
  assert.equal(moe.hereGB, null, "14 GB can't hold the MoE");
  let s = initialState({ rows, pledge: { gb: 8, max: 14, totalGB: 16 }, code: "9PFZ8T" });
  s.gpu = "RTX 5070 Ti · 16 GB";
  s.sel = rows.indexOf(r17);
  let r = reduce(s, "enter");
  assert.equal(r.state.step, "how"); assert.equal(r.state.how, 0);
  const text = render(r.state, { width: 80, lib }).join("\n");
  assert.match(text, /model     Qwen3 1\.7B · downloaded\n\n  How do you want to run Qwen3 1\.7B\?\n\n  › Pool with devices +other devices join and each holds a part\n/);
  assert.match(text, new RegExp(`\\n    Run it here +all of it here, lending ${r17.hereGB} GB; others can chat\\n`));
  assert.match(text, /↑↓ choose · enter go · esc back · q quit/);
  for (const [depth, theme] of [["truecolor", "dark"], ["256", "light"], ["none", "dark"]]) for (const w of [58, 80, 110]) {
    const lines = render(r.state, { width: w, lib, S: mkStyle({ depth, theme }) });
    assert.ok(lines.every((l) => width(l) <= w - 1), `${depth} ${w}`);
  }
  assert.equal(reduce(r.state, "up").state.how, 0, "no wrap");
  r = reduce(r.state, "down"); assert.equal(r.state.how, 1);
  assert.equal(reduce(r.state, "down").state.how, 1);
  assert.equal(reduce(r.state, "esc").state.step, "pick");
  r = reduce(r.state, "enter");
  assert.equal(r.state.step, "room"); assert.equal(r.state.pledge.gb, r17.hereGB); assert.equal(r.state.pledgeDone, true);
  assert.equal(r.state.splitMode, "speed"); assert.equal(r.state.flags.start, true, "starts once it can");
  assert.deepEqual(r.fx, [{ do: "pledge", gb: r17.hereGB }, { do: "split", mode: "speed" }]);
  // a model this computer can't hold: Run it here is off, with why
  s.sel = rows.indexOf(moe);
  r = reduce(s, "enter"); r = reduce(r.state, "down");
  assert.match(render(r.state, { lib }).join("\n"), /Run it here +needs 23 GB, this computer has 14 GB to lend/);
  r = reduce(r.state, "enter");
  assert.equal(r.state.step, "how"); assert.match(r.state.notice, /needs 23 GB, this computer has 14 GB to lend/);
  r = reduce(r.state, "up"); r = reduce(r.state, "enter");
  assert.equal(r.state.step, "pledge"); assert.equal(r.state.splitMode, "memory");
});

test("--here / --pool skip the question; --split given wins; --here and --pool together are an error", () => {
  const MODELS2 = { "qwen3-1.7b": { kind: "gguf" }, "qwen3.6-35b-moe": { kind: "qwen35" } };
  assert.equal(parseLendArgs("host", ["1.7b", "--here"], { models: MODELS2 }).mode, "here");
  assert.equal(parseLendArgs("host", ["--pool"], { models: MODELS2 }).mode, "pool");
  assert.equal(parseLendArgs("host", [], { models: MODELS2 }).mode, null);
  assert.throws(() => parseLendArgs("host", ["--here", "--pool"], { models: MODELS2 }), /opposite/);
  assert.equal(parseLendArgs("host", ["--split", "speed"], { models: MODELS2 }).splitGiven, true);
  const rows = modelRows(lib, { keys: KEYS, pulled: new Set(["qwen3-1.7b"]), pledgeGB: 8, maxGB: 14 });
  const here = initialState({ rows, model: "qwen3-1.7b", pledge: { gb: 8, max: 14 }, pulled: new Set(["qwen3-1.7b"]), flags: { mode: "here" } });
  assert.equal(here.step, "room"); assert.equal(here.flags.start, true); assert.equal(here.pledge.gb, rows.find((r) => r.key === "qwen3-1.7b").hereGB);
  const pool = initialState({ rows, model: "qwen3-1.7b", pledge: { gb: 8, max: 14 }, pulled: new Set(["qwen3-1.7b"]), flags: { mode: "pool" } });
  assert.equal(pool.step, "pledge"); assert.equal(pool.splitMode, "memory");
  const kept = initialState({ rows, model: "qwen3-1.7b", pledge: { gb: 8, max: 14 }, pulled: new Set(["qwen3-1.7b"]), flags: { mode: "pool", split: "speed", splitGiven: true } });
  assert.equal(kept.splitMode, "speed");
  // no flag: the question
  assert.equal(initialState({ rows, model: "qwen3-1.7b", pledge: { gb: 8, max: 14 }, pulled: new Set(["qwen3-1.7b"]) }).step, "how");
});

test("key hints never break inside a word: at 58, 60 and 80 columns they drop whole hints (never the primary one, q last)", async () => {
  const { mkStyle, visible: vis, width: wd } = await import("../lib/style.js");
  const { joinScreen, lendRow } = await import("../lib/joinui.js");
  const S = mkStyle({ depth: "truecolor", theme: "dark" });
  const whole = (line, hints) => vis(line).trim().split(" · ").every((h) => hints.includes(h));
  // pooled join's lend line (it wrapped as "en / ter lend" in a 0.3.1 capture)
  for (const cols of [58, 60, 80]) {
    const l = lendRow(S, { gb: 32, total: 128, cols });
    assert.ok(wd(l) <= cols - 1, `${cols}: ${vis(l)}`);
    const keys = vis(l).split(/GB {3}/)[1] || "";
    assert.ok(!keys || keys.split(" · ").every((h) => ["←→ 1 GB", "type a number", "enter lend"].includes(h)), `${cols}: ${keys}`);
    assert.match(vis(l), /enter lend/, "the primary hint stays");
    // the join screen's footer
    for (const phase of ["lobby", "online"]) {
      const L = joinScreen({ code: "4TKG9P", phase, you: { name: "spark", gb: 32 } }, { S, cols });
      assert.ok(L.every((x) => wd(x) <= cols - 1));
    }
  }
  // the host's footers: every step, and the room with many hints
  const rows = modelRows(lib, { keys: KEYS, pulled: new Set(["qwen3-1.7b"]), pledgeGB: 8, maxGB: 14 });
  const HINTS = ["↑↓ choose", "enter host it", "enter go", "esc back", "q quit", "←→ 1 GB", "type a number", "enter lend", "enter start", "i copy invite", "m model", "l lend", "s split",
    "c chat here", "r rebalance", "q close room", "y download", "n stream from Hugging Face instead", "q cancel and close the room"];
  for (const cols of [58, 60, 80]) {
    let s = initialState({ rows, pledge: { gb: 8, max: 14, totalGB: 16 }, code: "4TKG9P" });
    const states = [s];
    s = reduce(s, "enter").state; states.push(s);                 // how
    s = reduce(s, "enter").state; states.push(s);                 // pledge
    s = reduce(s, "enter").state;                                 // room
    s.devices = [{ name: "spark", gb: 8, self: true, meta: { webgpu: true, contribGB: 8 } }, { name: "mac", gb: 8, meta: { webgpu: true, contribGB: 8 } }];
    states.push(s, { ...s, step: "online" }, { ...s, step: "starting" });
    for (const st of states) {
      const L = render(st, { width: cols, lib, S });
      assert.ok(L.every((x) => wd(x) <= cols - 1), `${st.step} ${cols}`);
      const foot = L.filter((x) => / · |^ {2}\S+ \S/.test(vis(x))).pop();
      assert.ok(whole(foot, HINTS), `${st.step} at ${cols}: "${vis(foot)}"`);
      if (st.step === "room") assert.match(vis(foot), /q quit/, "q stays");
    }
  }
});

test("the 1.7B: 16K when the room holds it, 8K when it is short for 16K, an asked --ctx stays", () => {
  const m = "qwen3-1.7b";
  const f8 = roomFitNow(lib, { model: m, devices: [dev("laptop", 4)] });
  assert.equal(f8.fits, true, "one laptop lending 4 GB starts it alone");
  assert.equal(f8.ctx, 8192); assert.equal(f8.want, 16384); assert.equal(f8.fellBack, true);
  assert.equal(f8.ctxNote, "Qwen3 1.7B · 8K context: the room's memory is short for 16K");
  assert.equal(Math.round(f8.needGB), 6); assert.equal(Math.round(f8.minGB), 4);
  const f16 = roomFitNow(lib, { model: m, devices: [dev("laptop", 6)] });
  assert.deepEqual([f16.fits, f16.ctx, f16.fellBack, f16.ctxNote], [true, 16384, false, ""]);
  // the reported room (a laptop 3 GB + a phone 1 GB): 8K, within both pledges
  const rep = roomFitNow(lib, { model: m, devices: [dev("laptop", 3), dev("iphone", 1, { ua: "iPhone" })] });
  assert.deepEqual([rep.fits, rep.ctx, rep.fellBack], [true, 8192, true]);
  // short for both: short for 8K, the least it could start with
  const f3 = roomFitNow(lib, { model: m, devices: [dev("laptop", 3)], spareGB: [5] });
  assert.equal(f3.fits, false); assert.equal(f3.ctx, 8192);
  assert.match(f3.note, /^This room is 0\.\d GB short for Qwen3 1\.7B: add a device or raise a pledge: laptop could give 0\.\d GB more\.$/);
  // --ctx 16384 is taken as asked: no fallback, short on 4 GB
  const asked = roomFitNow(lib, { model: m, devices: [dev("laptop", 4)], ctxAsk: 16384 });
  assert.deepEqual([asked.fits, asked.ctx, asked.fellBack], [false, 16384, false]);
  // --ctx 8192: 8K on 4 GB, not a fallback (nothing to say)
  const a8 = roomFitNow(lib, { model: m, devices: [dev("laptop", 4)], ctxAsk: 8192 });
  assert.deepEqual([a8.fits, a8.ctx, a8.fellBack, a8.ctxNote], [true, 8192, false, ""]);
  // the room screen says both needs and why the context is 8K; online, the context line
  const rows = modelRows(lib, { keys: KEYS, pulled: new Set([m]), pledgeGB: 4 });
  const s = initialState({ rows, model: m, pledge: { gb: 4, max: 4, totalGB: 8 }, fixedPledge: true, pulled: new Set([m]), code: "4TKG9P", flags: { mode: "pool" } });
  s.step = "room";
  s.devices = [{ name: "laptop", kind: "this computer", gb: 4, self: true, meta: { webgpu: true, contribGB: 4 } }];
  s.fit = f8;
  const text = render(s, { width: 140, lib }).map(visible).join("\n");
  assert.match(text, /4 GB lent · 6 GB needed \(4 GB at 8K\)/);
  assert.match(text, /Qwen3 1\.7B · 8K context: the room's memory is short for 16K\. Lend 2 GB more \(l\) or add a device for 16K\./);
  const on = render({ ...s, step: "online", ctxNote: f8.ctxNote }, { width: 100, lib }).map(visible).join("\n");
  assert.match(on, /context +Qwen3 1\.7B · 8K context: the room's memory is short for 16K/);
  assert.doesNotMatch(render({ ...s, fit: f16 }, { width: 100, lib }).map(visible).join("\n"), /8K context/);
});

test("expert offload: a 12 GB GPU with RAM to spare runs the 35B --here, its experts in RAM; without RAM it can't", async () => {
  const { hereGB } = await import("../lib/hostui.js");
  const { offloadFor } = await import("../../room/pledge.js");
  const olib = { ...lib, offloadFor };
  // the RTX 5070 PC: lends 10 GB by the memory rule (11 at most), 48 GB of its 64 GB of RAM (total less 16)
  assert.equal(hereGB(olib, "qwen3.6-35b-moe", { maxGB: 11 }), null, "no RAM: as before");
  assert.equal(hereGB(lib, "qwen3.6-35b-moe", { maxGB: 11, ramGB: 48, offGB: 10 }), null, "a lib without offloadFor: as before");
  assert.equal(hereGB(olib, "qwen3.6-35b-moe", { maxGB: 11, ramGB: 48, offGB: 10 }), 10, "the memory rule's 10 GB, the experts it can't hold in RAM");
  assert.equal(hereGB(olib, "qwen3.6-35b-moe", { maxGB: 11, ramGB: 4, offGB: 10 }), null, "4 GB of RAM can't take the experts");
  // a model it holds whole is still dealt whole, at the least that holds it
  assert.equal(hereGB(olib, "qwen3-1.7b", { maxGB: 11, ramGB: 48, offGB: 10 }), hereGB(lib, "qwen3-1.7b", { maxGB: 11 }));
  const rows = modelRows(olib, { keys: KEYS, pledgeGB: 10, maxGB: 11, ramGB: 48, offGB: 10 });
  const moe = rows.find((r) => r.key === "qwen3.6-35b-moe"), r17 = rows.find((r) => r.key === "qwen3-1.7b");
  assert.equal(moe.hereGB, 10); assert.equal(moe.hereOff, true); assert.equal(r17.hereOff, false);
  // the room's fit: a device that offers RAM counts with it (the room page's roomFit with offload)
  const f = roomFitNow(olib, { model: "qwen3.6-35b-moe", devices: [dev("pc", 10, { offload: true, ramGB: 48 })] });
  assert.equal(f.fits, true); assert.equal(f.offload, true);
  assert.equal(roomFitNow(olib, { model: "qwen3.6-35b-moe", devices: [dev("pc", 10)] }).fits, false);
  // two devices whose pledges hold it: no offload
  const two = roomFitNow(olib, { model: "qwen3.6-35b-moe", devices: [dev("a", 14), dev("pc", 12, { offload: true, ramGB: 48 })] });
  assert.equal(two.fits, true); assert.equal(two.offload, false);
});
