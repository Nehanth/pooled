// A small terminal toolkit for pooled (no dependencies): colors that respect NO_COLOR, key names
// from raw stdin, a live region at the bottom of the terminal that redraws in place (only when it
// changed, so it does not flicker) while log lines scroll above it, and a menu.
//
// Nothing here reads terminfo: the few escapes used (colors, cursor up, clear to the end of the
// screen, hide the cursor) are plain ANSI that every terminal pooled meets understands, so a TERM
// the computer has no terminfo entry for (xterm-ghostty over ssh) works like xterm. TERM=dumb gets
// no escapes at all: line-by-line prompts and plain log lines instead.
import { visible, clip, width, padEnd, stripAnsi, style, I, colorOn as colorOnStyle } from "./style.js";

// color on `stream`: not for NO_COLOR, TERM=dumb, FORCE_COLOR=0, CLICOLOR=0 or a non-terminal
// (FORCE_COLOR / CLICOLOR_FORCE turn it on)
export const colorOn = (stream = process.stdout, env = process.env) => colorOnStyle(stream, env);

// What the terminal can do, from isTTY and the environment only.
//   tty: both ends are a terminal; ansi: it takes cursor movement (a live screen); keys: raw keys
//   (arrows); color: colors on `output`
export function termCaps({ input = process.stdin, output = process.stderr, env = process.env } = {}) {
  const tty = !!input.isTTY && !!output.isTTY;
  const ansi = tty && env.TERM !== "dumb";
  return { tty, ansi, keys: ansi && typeof input.setRawMode === "function", color: colorOn(output, env) };
}

// raw stdin bytes -> key names (several keys can come in one chunk: pasted digits)
export function keysOf(buf) {
  const s = buf.toString("utf8"), out = [];
  for (let i = 0; i < s.length;) {
    const rest = s.slice(i);
    const esc = /^\x1b(?:\[|O)([ABCDHF])/.exec(rest);
    if (esc) { out.push({ A: "up", B: "down", C: "right", D: "left", H: "home", F: "end" }[esc[1]]); i += esc[0].length; continue; }
    const other = /^\x1b\[[0-9;]*[~A-Za-z]/.exec(rest);
    if (other) { i += other[0].length; continue; }
    const ch = s[i++];
    if (ch === "\r" || ch === "\n") out.push("enter");
    else if (ch === "\x7f" || ch === "\b") out.push("backspace");
    else if (ch === "\x1b") out.push("esc");
    else if (ch === "\x03") out.push("ctrl-c");
    else if (ch === "\x04") out.push("ctrl-d");
    else if (ch === "\t") out.push("tab");
    else if (ch >= " ") out.push(ch);
  }
  return out;
}

// The live region: render(lines) draws it (again), log(text) prints a line above it.
export function liveRegion(stream = process.stderr) {
  let shown = 0, last = [], hidden = false;
  const cols = () => stream.columns || 80;
  const erase = () => {
    if (!shown) return "";
    // back to the region's first line, then clear to the end of the screen
    const s = (shown > 1 ? `\x1b[${shown - 1}A` : "") + "\r\x1b[J";
    shown = 0;
    return s;
  };
  const draw = (lines) => {
    const w = cols() - 1;
    const cut = lines.map((l) => clip(l, w));
    shown = cut.length;
    return cut.join("\n");
  };
  return {
    render(lines) {
      if (!hidden) { stream.write("\x1b[?25l"); hidden = true; }
      const same = lines.length === last.length && lines.every((l, i) => l === last[i]);
      if (same && shown) return;
      last = lines;
      stream.write(erase() + draw(lines));
    },
    log(text) {
      const body = String(text).split("\n").map((l) => clip(l, cols() - 1)).join("\n");
      stream.write(erase() + body + "\n" + (last.length ? draw(last) : ""));
    },
    // leave the region on screen as it is and put the cursor under it
    close() {
      if (shown) stream.write("\n");
      shown = 0; last = [];
      if (hidden) { stream.write("\x1b[?25h"); hidden = false; }
    },
    clear() { stream.write(erase()); last = []; if (hidden) { stream.write("\x1b[?25h"); hidden = false; } },
    get lines() { return last; },
  };
}
export { visible, clip };

// one line of text from the terminal (readline, so editing keys work) -> string | null (Ctrl-C / Ctrl-D)
// terminal: readline's line editing (it draws with escapes; off for TERM=dumb, where the terminal's
// own line discipline echoes what is typed)
export async function askLine(question, { input = process.stdin, output = process.stderr, terminal = !!input.isTTY && process.env.TERM !== "dumb" } = {}) {
  const readline = await import("node:readline");
  const rl = readline.createInterface({ input, output, terminal });
  return new Promise((resolve) => {
    let done = false;
    const end = (v) => { if (done) return; done = true; rl.close(); resolve(v); };
    rl.on("SIGINT", () => { output.write("\n"); end(null); });
    rl.on("close", () => end(null));
    rl.question(question, (a) => end(a));
  });
}

// [Y/n] from the terminal -> true | false | null (Ctrl-C)
export async function askYesNo(question, { def = true } = {}) {
  const a = await askLine(`${question} ${def ? "[Y/n]" : "[y/N]"} `);
  if (a == null) return null;
  const t = a.trim().toLowerCase();
  return t ? t.startsWith("y") : def;
}

// A menu: ↑/↓ (or j/k) and Enter, or a number key, in a terminal that takes escapes; numbered lines
// and a typed number otherwise (TERM=dumb). items: [{ label, hint }]
//   top: lines above the question (the menu's header); extra: { key: value } more keys that end it
// -> index | a value of `extra` | null (q, Esc, Ctrl-C)
export function menuLines(S, { top = [], title, items, sel, keys, tail = [], cols = 80 }) {
  const labelW = Math.max(...items.map((it) => width(it.label))) + 2;
  const L = [...top];
  if (title) L.push(I + title, "");
  items.forEach((it, i) => {
    const on = i === sel;
    L.push(I + (on ? `${S.acc(S.g.sel)} ${S.bold(padEnd(it.label, labelW))}` : `  ${padEnd(it.label, labelW)}`) + S.ink3(it.hint || ""));
  });
  L.push("", I + S.keys(keys));
  if (tail.length) L.push("", ...tail);
  return L.map((l) => clip(l, Math.max(20, cols) - 1));
}
export async function choose({ title, items, def = 0, input = process.stdin, output = process.stderr, env = process.env, top = [], keys = null, tail = [], extra = {} }) {
  const caps = termCaps({ input, output, env });
  const S = style({ stream: output, env, depth: caps.color ? null : "none" });
  const n = items.length;
  const labelW = Math.max(...items.map((it) => width(it.label)));
  if (!caps.keys) {
    output.write(`${stripAnsi(title || "")}\n`);
    items.forEach((it, i) => output.write(`  ${i + 1}) ${it.label.padEnd(labelW)}  ${it.hint || ""}`.trimEnd() + "\n"));
    for (;;) {
      const a = await askLine(`Choose 1-${n} (Enter: ${def + 1}, q: quit): `, { input, output, terminal: false });
      if (a == null) return null;
      const t = a.trim().toLowerCase();
      if (!t) return def;
      if (t === "q" || t === "quit" || t === "exit") return null;
      if (extra[t] !== undefined) return extra[t];
      const k = Number(t);
      if (Number.isInteger(k) && k >= 1 && k <= n) return k - 1;
      output.write(`  type a number from 1 to ${n}\n`);
    }
  }
  const region = liveRegion(output);
  let sel = def;
  keys ||= [[S.g.up, "choose"], ["enter", "go", "primary"], ["q", "quit"]];
  const draw = () => region.render(menuLines(S, { top, title, items, sel, keys, tail, cols: output.columns || 80 }));
  return new Promise((resolve) => {
    const done = (v) => {
      input.off("data", on);
      try { input.setRawMode(false); } catch {}
      input.pause();
      region.clear();
      if (typeof v === "number") output.write(`${I}${S.acc(S.g.sel)} ${items[v].label}\n`);
      resolve(v);
    };
    const on = (b) => {
      for (const k of keysOf(b)) {
        if (k === "ctrl-c" || k === "ctrl-d" || k === "esc" || k === "q") return done(null);
        if (extra[k] !== undefined) return done(extra[k]);
        if (k === "up" || k === "k") sel = (sel + n - 1) % n;
        else if (k === "down" || k === "j" || k === "tab") sel = (sel + 1) % n;
        else if (k === "home") sel = 0;
        else if (k === "end") sel = n - 1;
        else if (k === "enter") return done(sel);
        else if (/^[1-9]$/.test(k) && +k <= n) { sel = +k - 1; return done(sel); }
      }
      draw();
    };
    try { input.setRawMode(true); } catch {}
    input.resume();
    input.on("data", on);
    draw();
  });
}
