// pooled chat: the parts with no network and no terminal, so they can be unit tested: argument
// parsing, the conversation (the whole of it goes to the room each turn, so the host's cache reuses
// the earlier turns), what the terminal shows (answer, dimmed reasoning, the status line) and what an
// error tells the person at the terminal. cli/lib/chatrun.js runs them against a room.
import { parseArgs } from "node:util";
import { roomCodeFrom, roomKeyFrom } from "./room.js";
import { withDefaults, finishRequest, cleanText, cleanLabel } from "./common.js";

export const DEFAULT_MAX_TOKENS = 8192;

export const HELP_CHAT = `Usage
  pooled chat <ROOM CODE | "room link"> [prompt] [options]

  Talk to a Pooled room's model from the terminal. Joins the room as an ask-only client (no layers,
  no GPU needed here), streams each answer as it comes, and keeps the conversation: every turn sends
  the whole of it, so the room reuses what it already computed for the earlier turns.

  With a prompt (or with input that is not a terminal: echo "..." | pooled chat CODE), it asks once,
  prints the answer on stdout and exits, for scripts.

  Getting in: with the room's invite link (in quotes: "https://pooled.run/r/4TKG9P#k=..."), the host
  lets this client in at once. With the code alone (4TK-G9P), a host that asks before new devices
  join sees "pooled chat ... wants to join (API client)" and this waits until it presses Allow. The
  host's "Allow API clients" must be on either way.

In the chat
  /clear            forget the conversation (start over)
  /think [on|off]   let the model think before it answers (shown dimmed); no argument toggles
  /help             this list
  /exit, /bye       leave (Ctrl-D too)
  Ctrl-C            stop the answer being written; at the prompt, leave

Options
  --system <text>   a system prompt for the conversation
  --think           start with thinking on (default off)
  --max-tokens <n>  the most one answer may be (default ${DEFAULT_MAX_TOKENS}; the room's context bounds it too)
  --temperature <x> sampling temperature (default: the room's)
  --name <s>        how the room shows this client (default: "pooled chat" and 4 random letters)
  --signal <spec>   PeerJS signaling server, as the room page's ?signal= (default: PeerJS cloud)
  --wait <s>        without a terminal: how long to wait for the room's model to be started
                    (default 120)
  --no-color        no dimming (also NO_COLOR=1)
  -h, --help        this help

  What you type goes to the room's host (another person's computer or browser) and, under the
  room's settings, to the other screens in the room. Every device holding layers sees the hidden
  states of it. Chat in rooms you trust.
`;

export class UsageError extends Error {}

// argv after "chat" -> options, or throws UsageError
export function parseChatArgs(argv) {
  let r;
  try {
    r = parseArgs({ args: argv, allowPositionals: true, strict: true, options: {
      system: { type: "string" }, think: { type: "boolean" }, "max-tokens": { type: "string" }, temperature: { type: "string" },
      name: { type: "string" }, signal: { type: "string" }, wait: { type: "string" }, "no-color": { type: "boolean" },
      help: { type: "boolean", short: "h" },
    } });
  } catch (e) { throw new UsageError(e.message.replace(/^.*?: /, "")); }
  const o = r.values, pos = r.positionals;
  if (o.help) return { help: true };
  const code = roomCodeFrom(pos[0]);
  if (!code) throw new UsageError(`give a room code (six letters and digits like 4TK-G9P; older rooms have four) or a room link${pos[0] ? `, not "${pos[0]}"` : ""}`);
  const out = { code, key: roomKeyFrom(pos[0]), prompt: pos.length > 1 ? pos.slice(1).join(" ") : null,
    system: o.system ?? "", thinking: !!o.think, maxTokens: DEFAULT_MAX_TOKENS, temperature: null,
    name: o.name != null ? cleanLabel(o.name) : null, signal: o.signal || null, waitMs: 120000,
    color: !o["no-color"] && !process.env.NO_COLOR };
  if (o.name != null && !out.name) throw new UsageError("--name must not be empty");
  if (o["max-tokens"] != null) {
    const n = Number(o["max-tokens"]);
    if (!Number.isInteger(n) || n < 1 || n > 65536) throw new UsageError("--max-tokens must be a whole number from 1 to 65536");
    out.maxTokens = n;
  }
  if (o.temperature != null) {
    const t = Number(o.temperature);
    if (!Number.isFinite(t) || t < 0 || t > 2) throw new UsageError("--temperature must be a number from 0 to 2");
    out.temperature = t;
  }
  if (o.wait != null) {
    const w = Number(o.wait);
    if (!Number.isFinite(w) || w < 0) throw new UsageError("--wait must be a number of seconds");
    out.waitMs = w * 1000;
  }
  return out;
}

// a line typed at the prompt -> { cmd, arg } for a /command, { text } for a question, or null (blank)
export function parseLine(line) {
  const s = String(line ?? "").trim();
  if (!s) return null;
  if (s.startsWith("/")) {
    const [c, ...rest] = s.slice(1).split(/\s+/);
    const cmd = { exit: "exit", bye: "exit", quit: "exit", q: "exit", clear: "clear", reset: "clear", think: "think", help: "help", "?": "help" }[c.toLowerCase()];
    if (cmd) return { cmd, arg: rest.join(" ").trim().toLowerCase() };
    // "/path/to/file is ..." is a question, not a command
    if (!/^[a-z?]+$/i.test(c)) return { text: s };
    return { cmd: "unknown", arg: c };
  }
  return { text: s };
}

// The conversation. Each turn sends all of it: the host renders the same prefix again, and its
// cache reuses the tokens it computed for the earlier turns (the status line shows how many).
export class History {
  constructor(system = "") { this.system = system; this.turns = []; }
  add(role, text) { this.turns.push({ role, text: String(text ?? "") }); }
  // an answer cut short (Ctrl-C): kept, so the next question reads in context; an empty one is dropped
  // together with its question, which then did not happen as far as the model is concerned
  settle(answer, { stopped = false } = {}) {
    void stopped;
    if (answer && answer.trim()) this.add("assistant", answer);
    else if (this.turns.at(-1)?.role === "user") this.turns.pop();
  }
  clear() { this.turns = []; }
  get length() { return this.turns.length; }
  // -> the internal request (common.js) for the next answer, checked like pooled serve checks one
  request({ maxTokens = DEFAULT_MAX_TOKENS, thinking = false, temperature = null, client = "pooled chat", hostMeta = null } = {}) {
    const req = withDefaults({ api: "chat", stream: true, client, system: this.system, messages: this.turns.map((t) => ({ ...t })),
      maxTokens, thinking, temperature });
    return finishRequest(req, { hostMeta });
  }
}

// text from the room for the terminal: newlines and tabs kept, every other control character out (a
// hostile host could otherwise send escape sequences: clear the screen, write the clipboard, fake a prompt)
export const termText = (s) => String(s ?? "").replace(/\r\n?/g, "\n").replace(/[\u0000-\u0008\u000b-\u001f\u007f-\u009f]/g, "");

export const DIM = "\x1b[2m", RESET = "\x1b[0m";

// The answer as it streams: reasoning dimmed (under "thinking"), then the answer. An Encoder for
// answer.js's Ask: start / think / text / done / error. write(s) writes to the terminal.
export class Renderer {
  constructor({ write, color = true, showThinking = true } = {}) {
    this.write = write; this.color = color; this.showThinking = showThinking;
    this.mode = null;          // null (nothing yet) | "think" | "text"
    this.answer = ""; this.thought = ""; this.t0 = null; this.tFirst = null; this.tokens = 0; this.promptTokens = 0; this.col = 0;
  }
  dim(s) { return this.color ? DIM + s + RESET : s; }
  out(s) { if (!s) return; this.write(s); const i = s.lastIndexOf("\n"); this.col = i < 0 ? this.col + s.length : s.length - i - 1; }
  start(promptTokens) { this.promptTokens = promptTokens; this.t0 ??= Date.now(); }
  think(t) {
    t = termText(t);
    this.tokens++; this.tFirst ??= Date.now();
    this.thought += t;
    if (!this.showThinking) return;
    if (this.mode !== "think") { this.out(this.dim("thinking\n")); this.mode = "think"; t = t.replace(/^\n+/, ""); }
    this.out(this.dim(t));
  }
  text(t) {
    t = termText(t);
    this.tokens++; this.tFirst ??= Date.now();
    if (this.mode !== "text") {
      t = t.replace(/^\n+/, "");   // the model's newlines after its reasoning
      if (!t) return;              // (keep "mode" until real text comes)
      if (this.mode === "think") this.out(this.col ? "\n\n" : "\n");
      this.mode = "text";
    }
    this.answer += t;
    this.out(t);
  }
  // an answer ends with the cursor at the start of a line
  end() { if (this.col) this.out("\n"); }
  callStart() {} callArgs() {} callEnd() {} done() {} error() {} keepAlive() {}
}

// the line under an answer: "room 4TK-G9P · Qwen3 1.7B · 212 tokens · 41.2 tok/s · 180 of 230 prompt tokens reused"
export function statusLine({ code, model, answer, tps = null, stopped = false }) {
  const parts = [`room ${fmtCode(code)}`];
  if (model) parts.push(model);
  const out = answer?.usage?.out ?? 0;
  parts.push(`${out} token${out === 1 ? "" : "s"}${stopped ? " (stopped)" : answer?.reason === "max" ? " (cut at --max-tokens)" : answer?.reason === "ctx" ? " (the room's context is full)" : ""}`);
  if (tps != null && Number.isFinite(tps) && tps > 0) parts.push(`${tps.toFixed(1)} tok/s`);
  const pin = answer?.usage?.in || 0, reused = answer?.reused || 0;
  if (pin) parts.push(reused ? `${reused} of ${pin} prompt tokens reused` : `${pin} prompt tokens`);
  return parts.join(" · ");
}
// tok/s: the host's own figure from its stats ("48 tok · 21.3 tok/s · ..."), else from the stream here
// (first token to last: the decode, without the prefill)
export function tokPerSec(stats, { tokens = 0, tFirst = null, tEnd = null } = {}) {
  const m = /(\d+(?:\.\d+)?) tok\/s/.exec(String(stats || ""));
  if (m) return +m[1];
  if (tokens > 1 && tFirst != null && tEnd > tFirst) return (tokens - 1) / ((tEnd - tFirst) / 1000);
  return null;
}
export const fmtCode = (c) => (String(c || "").length === 6 ? `${c.slice(0, 3)}-${c.slice(3)}` : String(c || ""));

// an error from joining or asking -> { message, hint? } for the person at the terminal
export function explainChatError(err, { code = "" } = {}) {
  const msg = cleanText(err?.message || err || "unknown error", 400);
  const room = fmtCode(code);
  if (/^no room /.test(msg)) return { message: `No room ${room}.`, hint: "Check the code, and that the host's page (or pooled host) is still open." };
  if (/could not reach the signaling server/.test(msg)) return { message: msg.charAt(0).toUpperCase() + msg.slice(1) + ".", hint: "Devices use it only to find each other; try again in a minute, or pass --signal." };
  if (/too old to wait|can't wait for that/.test(msg)) return { message: `The host of room ${room} asks before new devices join, and this pooled is too old for that.`, hint: "Update it: npx @pooled/cli@latest" };
  if (/older Pooled|protocol \d+/i.test(msg)) return { message: `The host of room ${room} runs an older Pooled that can't take API clients like this one.`, hint: "Ask the host to reload the room page (or update pooled host)." };
  if (/does not allow API clients|API clients/i.test(msg)) return { message: `The host of room ${room} does not allow API clients.`, hint: "The host can turn \"Allow API clients\" on in the room's settings." };
  if (/didn't let this device in/.test(msg)) return { message: `The host of room ${room} didn't let this client in.` };
  if (/several devices waiting/.test(msg)) return { message: `The host of room ${room} has several devices waiting to join already.`, hint: "Try again in a minute." };
  if (/no model|not ready|not started/i.test(msg)) return { message: `Room ${room} has no model started yet.`, hint: "The host starts it on the room page (or pooled host deals the layers); then ask again." };
  return { message: msg };
}
