// The chat prompt (pooled chat, and c in pooled host): read a line, run it, read the next.
//
// While a line runs (an answer streams to the terminal) there is no readline at all: a live readline
// redraws its prompt on every keypress and every terminal resize ("\r" + ">>> " + what was typed),
// which wrote over the line being streamed ("def add_numbers(a, b):" came out as ">>> _numbers(a,
// b):"). So the terminal is in raw mode without echo while the answer streams, and only this reads
// it: Ctrl-C stops the answer, anything else typed is kept and put back on the next prompt. The
// prompt comes back only after the answer and its status line.
import readline from "node:readline";
import { keysOf } from "./tui.js";

// onLine(line) -> Promise (runs one line); onStop(): Ctrl-C while one runs; onEnd(): Ctrl-C or Ctrl-D
// at the prompt. -> { close() }
export function repl({ input = process.stdin, output = process.stdout, prompt = ">>> ", onLine, onStop = () => {}, onEnd = () => {}, historySize = 200 }) {
  const history = [];
  let typed = "", closed = false, rl = null;
  const tty = !!input.isTTY;

  // while a line runs: raw, no echo; Ctrl-C stops it, the rest waits for the next prompt
  const hold = (b) => {
    for (const k of keysOf(b)) {
      if (k === "ctrl-c") onStop();
      else if (k === "backspace") typed = typed.slice(0, -1);
      else if (k.length === 1) typed += k;
      else if (k === "enter" && typed) typed += " ";
    }
  };
  const holdOn = () => { if (!tty) return; try { input.setRawMode(true); } catch {} input.on("data", hold); input.resume(); };
  const holdOff = () => { if (!tty) return; input.off("data", hold); try { input.setRawMode(false); } catch {} input.pause(); };

  const ask = () => new Promise((resolve) => {
    rl = readline.createInterface({ input, output, prompt, terminal: tty, history, historySize, removeHistoryDuplicates: true });
    let done = false;
    const end = (v) => { if (done) return; done = true; const r = rl; rl = null; r.close(); resolve(v); };
    rl.on("line", (l) => end(l));
    rl.on("SIGINT", () => { output.write("\n"); end(null); });
    rl.on("close", () => end(null));
    rl.prompt();
    if (typed) { rl.write(typed); typed = ""; }
  });

  (async () => {
    while (!closed) {
      const line = await ask();
      if (closed) return;
      if (line == null) { closed = true; onEnd(); return; }
      holdOn();
      try { await onLine(line); } catch {}
      holdOff();
    }
  })();
  return {
    close() { closed = true; holdOff(); try { rl?.close(); } catch {} },
    get typed() { return typed; },
  };
}
