// A small terminal toolkit for pooled host / join (no dependencies): colors that respect NO_COLOR,
// key names from raw stdin, and a live region at the bottom of the terminal that redraws in place
// (only when it changed, so it does not flicker) while log lines scroll above it.
import { visible, clip } from "./hostui.js";

export const colorOn = (stream = process.stdout, env = process.env) =>
  !!stream.isTTY && env.NO_COLOR == null && env.TERM !== "dumb" && env.FORCE_COLOR !== "0";

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
export async function askLine(question, { input = process.stdin, output = process.stderr } = {}) {
  const readline = await import("node:readline");
  const rl = readline.createInterface({ input, output, terminal: !!input.isTTY });
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
