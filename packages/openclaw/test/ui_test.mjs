// The plugin's look (src/ui.js): the CLI's design in OpenClaw's prompts, plain text for hosted wizards
// and the chat, NO_COLOR as attributes only.
import { test } from "node:test";
import assert from "node:assert/strict";
import { promptUI, block, introLines, modelOptions, progressText, deviceBlock, memoryLine, downloadLine, noticeText, isNoticeText, padLabels, PLAIN, stripAnsi } from "../src/ui.js";
import { width } from "../../../cli/lib/style.js";

const tty = (columns = 100) => ({ isTTY: true, columns });
const clack = () => { const out = []; return { out, plain: async (m) => out.push(m), note: async (m, t) => out.push(`[note ${t}]\n${m}`), select() {}, text() {} }; };
const SGR = /\x1b\[(\d+(?:;\d+)*)m/g;

test("ui: OpenClaw's terminal prompter gets the color look; a hosted wizard (openUrl / deviceCode) or a pipe gets plain text", () => {
  const env = { COLORTERM: "truecolor", TERM: "xterm-256color" };
  assert.equal(promptUI(clack(), { stream: tty(), env }).terminal, true);
  assert.equal(promptUI({ ...clack(), openUrl() {}, deviceCode() {} }, { stream: tty(), env }).terminal, false, "hosted");
  assert.equal(promptUI(clack(), { stream: { isTTY: false }, env }).terminal, false, "not a terminal");
  assert.equal(promptUI({ note() {} }, { stream: tty(), env }).terminal, false, "no plain()");
});

test("ui: the intro block: the mark, the tagline, the GPU row on the CLI's grid; one accent, no other hues", async () => {
  const p = clack();
  const ui = promptUI(p, { stream: tty(), env: { COLORTERM: "truecolor" } });
  await block(p, ui, introLines(ui.S, ui.cols, { label: "NVIDIA GB10 · 122 GB unified memory" }), "Pooled");
  const text = stripAnsi(p.out[0]).split("\n");
  assert.match(text[1], /^│ {2}· · • {5}Pooled$/);
  assert.match(text[2], /^│ {2}· • ● {5}Peer-to-peer inference engine for your claw$/);
  assert.match(text[5], /^│ {2}gpu {7}NVIDIA GB10 · 122 GB unified memory$/, "values at column 12 behind the guide bar");
  for (const m of p.out[0].matchAll(SGR)) assert.ok(!/^(31|32|33|35|36)$/.test(m[1]), `no red/green/yellow/magenta/cyan: ${m[1]}`);
  assert.ok(p.out[0].includes("\x1b[38;2;127;147;255m●"), "the last dot of the mark in the accent");
});

test("ui: NO_COLOR keeps the layout with bold / dim / reverse only; a hosted wizard gets a note with no escapes", async () => {
  const p = clack();
  const ui = promptUI(p, { stream: tty(), env: { NO_COLOR: "1" } });
  await block(p, ui, [...introLines(ui.S, ui.cols, { label: "M5 · 64 GB" }), ui.S.pill("4TK-G9P")], "Pooled");
  for (const m of p.out[0].matchAll(SGR)) assert.match(m[1], /^(1|2|7|22|27)$/, "attributes only");
  assert.ok(p.out[0].includes("\x1b[7m\x1b[1m 4TK-G9P "), "the pill in reverse video");
  const h = { ...clack(), openUrl() {} };
  const hu = promptUI(h, { stream: tty(), env: {} });
  await block(h, hu, [...introLines(hu.S, hu.cols, { label: "M5 · 64 GB" }), "", hu.S.pill("4TK-G9P")], "Pooled");
  assert.equal(h.out[0], "[note Pooled]\nPeer-to-peer inference engine for your claw\npooled.run\n\ngpu       M5 · 64 GB\n\n[4TK-G9P]");
});

test("ui: option labels line up; the model rows put the need and the download in columns", () => {
  const o = padLabels([{ value: 1, label: "Start a room" }, { value: 2, label: "Join a room" }]);
  assert.equal(width(o[0].label), width(o[1].label));
  const rows = [{ key: "a", name: "Qwen3 1.7B", needGB: 5.5, pulled: true, fileBytes: 1.8e9, fitsAlone: true, small: true },
    { key: "b", name: "Qwen3.6 35B MoE", needGB: 22.8, pulled: false, fileBytes: 21e9, fitsAlone: false }];
  const m = modelOptions(PLAIN, rows, { recommended: "b" });
  assert.equal(m[0].label, "Qwen3 1.7B       needs  6 GB   downloaded");
  assert.equal(m[1].label, "Qwen3.6 35B MoE  needs 23 GB   19.6 GB download");
  assert.equal(m[1].hint, "recommended · needs another device");
});

test("ui: a download in one line: a thin bar, the percent, size, speed and time left", () => {
  const st = { state: "running", done: 0.4 * 2 ** 30 * 1.8, total: 2 ** 30 * 1.8, bps: 48e6 };
  assert.match(stripAnsi(progressText(PLAIN, "Qwen3 1.7B", st, { cols: 100 })), /^Qwen3 1\.7B {2}━{8}─{12} {3}40% {2}737 MB of 1\.8 GB {2}48 MB\/s {2}\d+ s left$/);
  assert.equal(progressText(PLAIN, "Qwen3 1.7B", { state: "done", total: 2 ** 30 * 1.8 }), "Qwen3 1.7B downloaded · 1.8 GB · checked");
  assert.match(downloadLine(st), /^download {2}━{8}─{12} {2}40% {2}737 MB of 1\.8 GB {2}48 MB\/s {2}\d+ s left$/);
  assert.equal(downloadLine({ state: "done" }), null);
});

test("ui: the chat's device table is a code block on one grid; the memory row is the CLI's", () => {
  const b = deviceBlock([{ name: "gb10 (OpenClaw)", gpu: "NVIDIA GB10", gb: 16, range: [0, 24], self: true }, { name: "macbook", gpu: "Apple M3 Pro", gb: 10, range: [24, 40] }],
    { waiting: [{ name: "otter" }], extra: [memoryLine(26, 22.8)] });
  assert.deepEqual(b, [
    "```text",
    "DEVICE           GPU             LENDS  HOLDS",
    "gb10 (OpenClaw)  NVIDIA GB10     16 GB  0–23    this device",
    "macbook          Apple M3 Pro    10 GB  24–39",
    "otter                                           wants to join",
    "",
    "memory    ━━━━━━━━━━━━━━━━━━━━  26 GB lent · 23 GB needed",
    "```",
  ]);
});

test("ui: a notice is markdown: Pooled, the room code as a pill, what happened, then plain sentences", () => {
  const t = noticeText("waiting", "1 of 2 devices are in the room. Open the invite link, then ask again.", "4TKG9P");
  assert.equal(t, "**Pooled** · `4TK-G9P` · waiting for devices\n\n1 of 2 devices are in the room. Open the invite link, then ask again.");
  assert.equal(noticeText("setup", "Pooled: Pooled is not set up on this machine: run `openclaw onboard`"), "**Pooled** · not set up\n\nPooled is not set up on this machine: run `openclaw onboard`.");
  assert.ok(isNoticeText(t) && isNoticeText("⚠️ Pooled: room 4TK-G9P is waiting") && !isNoticeText("Pooled rooms are great"));
});
