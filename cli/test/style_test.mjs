// The look (cli/lib/style.js, cli/lib/joinui.js): widths that skip escapes and links, clip() that keeps
// them balanced, the palette per depth, and pooled join's screen.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkStyle, width, clip, colorOn, colorDepth, stripAnsi, progressRow } from "../lib/style.js";
import { joinScreen } from "../lib/joinui.js";

test("widths: escapes and OSC 8 links take no columns; code points count once", () => {
  const S = mkStyle({ depth: "truecolor" });
  const l = S.link("https://pooled.run/r/9PFZ8T#k=AbCdEfGhIjKlMnOpQrStUv", "pooled.run/r/9PFZ8T");
  assert.equal(width(l), "pooled.run/r/9PFZ8T".length);
  assert.equal(width(S.pill("9PF-Z8T")), 9);
  assert.equal(width("━━━●·"), 5);
  const c = clip(S.acc("x") + l + " and more text here", 10);
  assert.equal(stripAnsi(c), "xpooled.r…");
  assert.match(c, /\x1b\]8;;\x1b\\$/, "an open link is closed");
  assert.doesNotMatch(c, /\x1b\[0m/, "no full reset mid-line");
});

test("colors: NO_COLOR, TERM=dumb, FORCE_COLOR, CLICOLOR; truecolor from COLORTERM", () => {
  const tty = { isTTY: true }, pipe = { isTTY: false };
  assert.equal(colorOn(tty, {}), true);
  assert.equal(colorOn(tty, { NO_COLOR: "1" }), false);
  assert.equal(colorOn(tty, { TERM: "dumb" }), false);
  assert.equal(colorOn(tty, { CLICOLOR: "0" }), false);
  assert.equal(colorOn(pipe, {}), false);
  assert.equal(colorOn(pipe, { FORCE_COLOR: "1" }), true);
  assert.equal(colorOn(pipe, { CLICOLOR_FORCE: "1" }), true);
  assert.equal(colorDepth({ COLORTERM: "truecolor" }), "truecolor");
  assert.equal(colorDepth({}), "256");
  assert.match(mkStyle({ depth: "256", theme: "light" }).acc("x"), /\x1b\[38;5;26m/);
  assert.match(mkStyle({ depth: "truecolor", theme: "dark" }).acc("x"), /\x1b\[38;2;127;147;255m/);
  assert.equal(mkStyle({ depth: "none" }).acc("x"), "\x1b[1mx\x1b[22m", "NO_COLOR: bold instead of the accent");
  assert.equal(mkStyle({ depth: "none", plain: true }).acc("x"), "x", "no escapes at all without a terminal");
  assert.equal(mkStyle({ ascii: true, depth: "none", plain: true }).bar(0.5, 4), "==--");
});

test("the progress row: fixed columns", () => {
  const S = mkStyle({ depth: "none", plain: true });
  assert.equal(progressRow(S, "download", { done: 0.41 * 15 * 2 ** 30, total: 15 * 2 ** 30, bps: 48e6 }),
    "  download  ━━━━━━━━────────────   41%   6.1 GB of 15.0 GB  48 MB/s  3 min left");
});

test("pooled join's screen: waiting, downloading, in the room", () => {
  const S = mkStyle({ depth: "none", plain: true });
  const base = { code: "9PFZ8T", hostName: "desk", modelLabel: "Qwen3.8 27B", you: { name: "otter", gpu: "Apple M2", gb: 8, totalGB: 16 } };
  const now = 1e9;
  let t = joinScreen({ ...base, phase: "lobby", lobbyAt: now - 8000 }, { S, now }).join("\n");
  assert.match(t, /pooled join\n  \[9PF-Z8T\]  desk's room · Qwen3\.8 27B\n  . waiting for desk to let you in · 0:08/);
  assert.match(t, /you       otter · Apple M2 · lends 8 GB of 16 GB/);
  assert.match(t, /An invite link skips this step/);
  t = joinScreen({ ...base, phase: "waiting", dl: { state: "running", done: 1.7 * 2 ** 30, total: 4.1 * 2 ** 30, bps: 31e6 } }, { S, now }).join("\n");
  assert.match(t, /downloading the model/);
  assert.match(t, /download  ━+─+   41%\s+1\.7 GB of 4\.1 GB\s+31 MB\/s/);
  assert.match(t, /load      waits for the download/);
  t = joinScreen({ ...base, phase: "online", devices: 3, range: [34, 48], passes: 38, onlineAt: now - 4 * 60000 }, { S, now }).join("\n");
  assert.match(t, /● in the room · 3 devices · up 4m/);
  assert.match(t, /lends 8 GB · layers 34–47/);
  assert.match(t, /Keep this open/);
  assert.match(t, /q leave the room/);
  for (const w of [58, 80]) for (const l of joinScreen({ ...base, phase: "online", devices: 3 }, { S: mkStyle({ depth: "truecolor" }), cols: w, now })) assert.ok(width(l) <= w - 1);
});
