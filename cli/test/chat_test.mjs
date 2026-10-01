// pooled chat without a room: arguments, the conversation it sends, what the terminal shows.
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseChatArgs, parseLine, History, Renderer, statusLine, tokPerSec, explainChatError, termText, UsageError, DEFAULT_MAX_TOKENS } from "../lib/chat.js";
import { Ask, Collector } from "../lib/answer.js";
import { askBody } from "../lib/common.js";

const KEY = "AbCdEfGhIjKlMnOpQrStUv";

test("arguments: a code (six grouped, four still) or a quoted invite link with its key; a prompt makes it one-shot", () => {
  let o = parseChatArgs(["4tk-g9p"]);
  assert.equal(o.code, "4TKG9P"); assert.equal(o.key, null); assert.equal(o.prompt, null);
  assert.equal(o.maxTokens, DEFAULT_MAX_TOKENS); assert.equal(o.thinking, false);
  o = parseChatArgs([`https://pooled.run/r/4TKG9P#k=${KEY}`, "why", "is the sky blue?", "--think", "--max-tokens", "300", "--temperature", "0"]);
  assert.equal(o.code, "4TKG9P"); assert.equal(o.key, KEY); assert.equal(o.prompt, "why is the sky blue?");
  assert.equal(o.thinking, true); assert.equal(o.maxTokens, 300); assert.equal(o.temperature, 0);
  assert.equal(parseChatArgs(["ABCD"]).code, "ABCD");
  assert.equal(parseChatArgs(["-h"]).help, true);
  for (const bad of [[], ["ABCDE"], ["ABCD", "--max-tokens", "0"], ["ABCD", "--temperature", "9"], ["ABCD", "--nope"], ["ABCD", "--name", ""]])
    assert.throws(() => parseChatArgs(bad), UsageError, bad.join(" "));
});

test("lines: /commands, questions, a path that starts with a slash is a question", () => {
  assert.equal(parseLine("   "), null);
  assert.deepEqual(parseLine("hello"), { text: "hello" });
  assert.deepEqual(parseLine("/exit"), { cmd: "exit", arg: "" });
  assert.deepEqual(parseLine("/bye"), { cmd: "exit", arg: "" });
  assert.deepEqual(parseLine("/clear"), { cmd: "clear", arg: "" });
  assert.deepEqual(parseLine("/think On"), { cmd: "think", arg: "on" });
  assert.deepEqual(parseLine("/frobnicate"), { cmd: "unknown", arg: "frobnicate" });
  assert.deepEqual(parseLine("/etc/hosts: what is it?"), { text: "/etc/hosts: what is it?" });
});

test("history: every turn sends the whole conversation, the same prefix each time (the room's cache reuses it)", () => {
  const h = new History("Be brief.");
  h.add("user", "Name a planet.");
  const r1 = h.request({ maxTokens: 64, hostMeta: { api: 2, ctx: 4096 } });
  assert.equal(r1.system, "Be brief.");
  assert.deepEqual(r1.messages, [{ role: "user", text: "Name a planet." }]);
  assert.equal(r1.maxTokens, 64); assert.equal(r1.thinking, false); assert.equal(r1.client, "pooled chat");
  h.settle("Mars.");
  h.add("user", "How far is it?");
  const r2 = h.request({ thinking: true });
  assert.deepEqual(r2.messages.map((m) => [m.role, m.text]), [["user", "Name a planet."], ["assistant", "Mars."], ["user", "How far is it?"]]);
  assert.equal(r2.thinking, true);
  const b1 = askBody(r1, true), b2 = askBody(r2, true);
  assert.deepEqual(b2.messages.slice(0, 1), b1.messages, "turn 2 starts with turn 1's messages");
  assert.equal(b2.api, 2);
  assert.deepEqual(askBody(r2, false).messages.length, 3, "an older host gets the v1 shape");
  // a failed or empty answer: the question goes too, so the next one is not two questions in a row
  h.add("user", "and Venus?");
  h.settle("");
  assert.equal(h.length, 3);
  h.clear();
  assert.equal(h.length, 0);
  assert.throws(() => h.request(), /at least one user message/);
});

// the room's messages for an answer, through the same Ask pooled serve uses, into the renderer
function play(msgs, { color = false } = {}) {
  let out = "";
  const R = new Renderer({ write: (s) => { out += s; }, color });
  const col = new Collector({ id: "r" });
  const h = new History(); h.add("user", "q");
  const ask = new Ask({ req: h.request({ maxTokens: 100 }), meta: {}, v2: true, encoders: [col, R] });
  let res = null;
  for (const m of msgs) res = ask.feed(m) || res;
  R.end();
  return { out, R, res, col };
}

test("rendering: reasoning dimmed under 'thinking', then the answer; control characters from the room never reach the terminal", () => {
  const { out, R, res } = play([
    { t: "ai-genstart", api: 2, promptTokens: 12 },
    { t: "ai-token", text: "Let me see", th: 1 }, { t: "ai-token", text: ".", th: 1 },
    { t: "ai-token", text: "\n\nHi" }, { t: "ai-token", text: " \x1b]52;c;ZXZpbA==\x07there\x1b[2J" },
    { t: "ai-gendone", api: 2, reason: "stop", usage: { in: 12, out: 5 }, reused: 8, stats: "5 tok · 42.5 tok/s" },
  ], { color: true });
  assert.equal(out, "\x1b[2mthinking\n\x1b[0m\x1b[2mLet me see\x1b[0m\x1b[2m.\x1b[0m\n\nHi ]52;c;ZXZpbA==there[2J\n");
  assert.equal(R.answer, "Hi ]52;c;ZXZpbA==there[2J");
  assert.equal(R.promptTokens, 12);
  assert.equal(res.answer.reused, 8);
  assert.ok(!/\x1b(?!\[2m|\[0m)/.test(out), "no escape from the room");
  // no reasoning: just the answer, no colors when off
  const p = play([{ t: "ai-genstart", api: 2, promptTokens: 3 }, { t: "ai-token", text: "OK" }, { t: "ai-gendone", api: 2, reason: "stop", usage: { in: 3, out: 1 } }]);
  assert.equal(p.out, "OK\n");
  assert.equal(termText("a\r\nb\u0007\tc\u009b"), "a\nb\tc");
});

test("the status line: room, model, tokens, tok/s, the cache reuse; tok/s from the host's stats or the stream", () => {
  const answer = { usage: { in: 230, out: 212 }, reused: 180, reason: "stop" };
  assert.equal(statusLine({ code: "4TKG9P", model: "Qwen3 1.7B", answer, tps: 41.23 }),
    "room 4TK-G9P · Qwen3 1.7B · 212 tokens · 41.2 tok/s · 180 of 230 prompt tokens reused");
  assert.equal(statusLine({ code: "ABCD", model: null, answer: { usage: { in: 9, out: 1 }, reused: 0 }, stopped: true }),
    "room ABCD · 1 token (stopped) · 9 prompt tokens");
  assert.match(statusLine({ code: "ABCD", answer: { usage: { in: 9, out: 64 }, reason: "max" } }), /cut at --max-tokens/);
  assert.equal(tokPerSec("48 tok · 21.3 tok/s · 2 devices"), 21.3);
  assert.equal(tokPerSec("", { tokens: 11, tFirst: 1000, tEnd: 2000 }), 10);
  assert.equal(tokPerSec("", { tokens: 1, tFirst: 1000, tEnd: 2000 }), null);
});

test("errors say what happened: no room, no model, API clients off, an older host, Deny", () => {
  const x = (m) => explainChatError(new Error(m), { code: "4TKG9P" }).message;
  assert.equal(x("no room 4TKG9P (is the host page open?)"), "No room 4TK-G9P.");
  assert.equal(x("no model started yet"), "Room 4TK-G9P has no model started yet.");
  assert.equal(x("the host of room 4TKG9P said: the host does not allow API clients in this room"), "The host of room 4TK-G9P does not allow API clients.");
  assert.match(x("the host of room 4TKG9P runs an older Pooled; reload the host page"), /runs an older Pooled/);
  assert.match(x("the host of room 4TKG9P said: This room's host asks before new devices join, and this tab runs an older Pooled that can't wait for that."), /too old/);
  assert.equal(x("the host of room 4TKG9P said: The host didn't let this device in."), "The host of room 4TK-G9P didn't let this client in.");
  assert.ok(explainChatError(new Error("no model started yet"), { code: "4TKG9P" }).hint);
});
