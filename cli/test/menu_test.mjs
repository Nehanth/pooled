// pooled with no command (cli/lib/menu.js) and the terminal toolkit under it (cli/lib/tui.js): the
// menu by arrows, Enter and number keys; the same menu as numbered lines on TERM=dumb; no colors
// with NO_COLOR; a TERM without terminfo (xterm-ghostty over ssh) works like any other. And the
// fd-2 swap that hides the GPU driver's own stderr lines (cli/lib/dawn.js).
import { test } from "node:test";
import assert from "node:assert/strict";
import { PassThrough } from "node:stream";
import { spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { choose, termCaps, colorOn } from "../lib/tui.js";
import { menuMain, MENU } from "../lib/menu.js";
import { loadText } from "../lib/lend.js";

// a fake terminal: keys go in, what is drawn comes out
function fakeTTY({ tty = true } = {}) {
  const input = new PassThrough(), output = new PassThrough();
  let text = "";
  output.on("data", (b) => { text += b.toString(); });
  if (tty) {
    input.isTTY = true; output.isTTY = true; output.columns = 80;
    input.raw = false; input.setRawMode = (on) => { input.raw = on; return input; };
  }
  return { input, output, text: () => text, send: (s) => setImmediate(() => input.write(s)) };
}
const SGR = /\x1b\[[0-9;]*m/;
const ESC = /\x1b/;
const roomCodeFrom = (s) => (/^[A-Z2-9]{3}-?[A-Z2-9]{3}$/i.test(String(s).split("#")[0].split("/").pop()) ? "4TKG9P" : null);

for (const TERM of ["xterm-256color", "xterm-ghostty", "tmux-256color", "some-unknown-term", undefined]) {
  test(`menu: arrows and Enter pick an item (TERM=${TERM})`, async () => {
    const t = fakeTTY();
    t.send("\x1b[B\x1b[B\r");   // down, down, Enter
    const i = await choose({ title: "t", items: MENU, input: t.input, output: t.output, env: { TERM } });
    assert.equal(i, 2);
    assert.equal(t.input.raw, false, "raw mode is off again");
    assert.match(t.text(), /\x1b\[\?25l/, "the menu is drawn in place (plain ANSI, no terminfo)");
    assert.match(t.text(), /Chat with a room\n/, "the choice stays on screen");
  });
}

test("menu: a number key picks at once; q, Esc and Ctrl-C leave", async () => {
  for (const [keys, want] of [["4", 3], ["1", 0], ["q", null], ["\x1b", null], ["\x03", null], ["\x1b[A\r", 4]]) {
    const t = fakeTTY();
    t.send(keys);
    assert.equal(await choose({ title: "t", items: MENU, input: t.input, output: t.output, env: { TERM: "xterm" } }), want, JSON.stringify(keys));
  }
});

test("menu with NO_COLOR: no colors (bold and dim only), still drawn in place", async () => {
  const t = fakeTTY();
  t.send("\r");
  await choose({ title: "t", items: MENU, input: t.input, output: t.output, env: { TERM: "xterm-256color", NO_COLOR: "1" } });
  assert.ok(!/\x1b\[(3\d|4\d|9\d)[;m]/.test(t.text()), JSON.stringify(t.text().slice(0, 200)));
  assert.match(t.text(), /\x1b\[\?25l/);
});

test("menu with TERM=dumb: numbered lines and a typed number, no escapes at all", async () => {
  const t = fakeTTY();
  t.send("3\n");
  const i = await choose({ title: "What now?", items: MENU, input: t.input, output: t.output, env: { TERM: "dumb" } });
  assert.equal(i, 2);
  assert.ok(!ESC.test(t.text()), JSON.stringify(t.text()));
  assert.match(t.text(), /  1\) Host a room /);
  assert.match(t.text(), /  5\) Models /);
  // Enter takes the default, a bad answer asks again
  const u = fakeTTY();
  u.send("9\n"); setTimeout(() => u.input.write("\n"), 30);
  assert.equal(await choose({ title: "t", items: MENU, input: u.input, output: u.output, env: { TERM: "dumb" } }), 0);
  assert.match(u.text(), /type a number from 1 to 5/);
});

test("termCaps: a terminal that takes escapes unless TERM=dumb; colors unless NO_COLOR", () => {
  const tty = { isTTY: true, setRawMode() {} }, out = { isTTY: true };
  assert.deepEqual(termCaps({ input: tty, output: out, env: { TERM: "xterm-ghostty" } }), { tty: true, ansi: true, keys: true, color: true });
  assert.deepEqual(termCaps({ input: tty, output: out, env: {} }), { tty: true, ansi: true, keys: true, color: true });
  assert.deepEqual(termCaps({ input: tty, output: out, env: { TERM: "dumb" } }), { tty: true, ansi: false, keys: false, color: false });
  assert.equal(termCaps({ input: tty, output: out, env: { TERM: "xterm", NO_COLOR: "1" } }).color, false);
  assert.equal(termCaps({ input: tty, output: out, env: { TERM: "xterm", NO_COLOR: "1" } }).ansi, true);
  assert.equal(termCaps({ input: {}, output: out, env: {} }).tty, false);
  assert.equal(colorOn({ isTTY: true }, { NO_COLOR: "" }), true, "an empty NO_COLOR does not count");
});

test("pooled menu: each item runs its flow; chat and serve ask for the room first", async () => {
  for (const [keys, want] of [["1", ["host", []]], ["2", ["join", []]]]) {
    const t = fakeTTY(); const ran = [];
    t.send(keys);
    await menuMain({ version: "0.3.1", input: t.input, output: t.output, env: { TERM: "xterm-ghostty" }, run: async (c, a) => { ran.push([c, a]); return 0; } });
    assert.deepEqual(ran, [want]);
    assert.match(t.text().replace(/\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)/g, ""), /pooled 0\.3\.1\n.*Run one AI model across several computers\.\n.*pooled\.run\n\n  What do you want to do\?/);
  }
  for (const [key, cmd] of [["3", "chat"], ["4", "serve"]]) {
    const t = fakeTTY(); const ran = [];
    t.send(key); setTimeout(() => t.input.write("nope\n"), 30); setTimeout(() => t.input.write("4TK-G9P\n"), 60);
    await menuMain({ input: t.input, output: t.output, env: { TERM: "xterm" }, roomCodeFrom, run: async (c, a) => { ran.push([c, a]); return 0; } });
    assert.deepEqual(ran, [[cmd, ["4TK-G9P"]]], cmd);
    assert.match(t.text(), /"nope" is not a room code/);
  }
  // models: the list, then download one
  const t = fakeTTY(); const ran = [];
  t.send("5"); setTimeout(() => t.input.write("1"), 30); setTimeout(() => t.input.write("\x1b[B\r"), 60);
  const models = { pulled: [{ key: "qwen3-0.6b", hint: "" }], notPulled: [{ key: "qwen3-1.7b", hint: "" }, { key: "qwen3-4b", hint: "" }] };
  await menuMain({ input: t.input, output: t.output, env: { TERM: "xterm" }, models, run: async (c, a) => { ran.push([c, a]); return 0; } });
  assert.deepEqual(ran, [["list", []], ["pull", ["qwen3-4b"]]]);
  // q leaves with 0 and runs nothing
  const q = fakeTTY(); const none = [];
  q.send("q");
  assert.equal(await menuMain({ input: q.input, output: q.output, env: { TERM: "xterm" }, run: async (c) => { none.push(c); } }), 0);
  assert.deepEqual(none, []);
});

test("pooled menu on TERM=dumb: numbered lines, then the chosen flow", async () => {
  const t = fakeTTY(); const ran = [];
  t.send("1\n");
  await menuMain({ input: t.input, output: t.output, env: { TERM: "dumb" }, run: async (c, a) => { ran.push([c, a]); return 0; } });
  assert.deepEqual(ran, [["host", []]]);
  assert.ok(!ESC.test(t.text()));
});

test("the load status says what it is doing", () => {
  assert.equal(loadText({ from: "Hugging Face", fetched: 312e6, total: 900e6, bps: 18e6 }, 20), "downloading 312 of 900 MB · 18 MB/s · from Hugging Face");
  assert.equal(loadText({ from: "Hugging Face", fetched: 900e6, total: 900e6, bps: 18e6 }, 99), "loading onto the GPU");
  assert.equal(loadText({ from: "disk", fetched: 450e6, total: 900e6 }, 50), "loading onto the GPU 50% (from disk)");
  assert.equal(loadText(null, 12), "loading layers 12%");
});

test("the GPU driver's own stderr lines are caught; ours still show (pipe, file)", () => {
  const code = `
    import fs from "node:fs";
    import { quietStderr, driverLog } from ${JSON.stringify(new URL("../lib/dawn.js", import.meta.url).href)};
    process.stderr.write("ours before\\n");
    const v = quietStderr(() => { fs.writeSync(2, "MESA: error: Opening /dev/dri/card0 failed\\n"); return 7; });
    process.stderr.write("ours after " + v + "\\n");
    console.log(JSON.stringify(driverLog()));`;
  if (process.platform === "win32") return;
  // stderr as a shell pipe (| cat) and as a file (Node's own stdio "pipe" is a socket, which can't be
  // reopened: there it runs without catching)
  const dir = mkdtempSync(path.join(os.tmpdir(), "pooled-quiet-"));
  const js = path.join(dir, "q.mjs"), err = path.join(dir, "err.txt");
  writeFileSync(js, code);
  const piped = spawnSync("sh", ["-c", `"${process.execPath}" "${js}" 2>&1 >/dev/null | cat`], { encoding: "utf8" });
  assert.equal(piped.stdout, "ours before\nours after 7\n");
  const r = spawnSync("sh", ["-c", `"${process.execPath}" "${js}" 2>"${err}"`], { encoding: "utf8" });
  assert.equal(r.status, 0);
  assert.equal(readFileSync(err, "utf8"), "ours before\nours after 7\n");
  assert.equal(JSON.parse(r.stdout), "MESA: error: Opening /dev/dri/card0 failed\n");
  rmSync(dir, { recursive: true, force: true });
});
