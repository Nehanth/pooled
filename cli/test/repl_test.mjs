// The chat prompt (cli/lib/repl.js): an answer streaming to the terminal is never written over by
// the prompt, whatever is typed or however the window is resized meanwhile.
import { test } from "node:test";
import assert from "node:assert/strict";
import { PassThrough } from "node:stream";
import { repl } from "../lib/repl.js";

function fakeTTY() {
  const input = new PassThrough(), output = new PassThrough();
  let text = "";
  output.on("data", (b) => { text += b.toString(); });
  input.isTTY = true; output.isTTY = true; output.columns = 80; output.rows = 24;
  input.setRawMode = (on) => { input.raw = on; return input; };
  return { input, output, text: () => text };
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

test("a code block streams intact while keys arrive and the window is resized; typed keys come back on the next prompt", async () => {
  const t = fakeTTY();
  const answer = ["```python\n", "def ", "add_", "numbers(a, b):\n", "    return ", "a + b\n", "```\n"];
  let streamed = "", ended = false, stops = 0;
  const lines = [];
  const r = repl({ input: t.input, output: t.output, prompt: ">>> ",
    onLine: async (line) => {
      lines.push(line);
      const from = t.text().length;
      for (const [i, chunk] of answer.entries()) {
        t.output.write(chunk);
        if (i === 2) { t.input.write("xy"); t.output.emit("resize"); }
        if (i === 4) t.input.write("\x1b[A");   // an arrow key mid-stream: ignored, never a redraw
        await sleep(5);
      }
      t.output.write("room 4TK-G9P · 12 tok/s\n");
      streamed = t.text().slice(from);
    },
    onStop: () => { stops++; },
    onEnd: () => { ended = true; } });
  await sleep(20);
  t.input.write("write add in python\r");
  await sleep(150);
  assert.deepEqual(lines, ["write add in python"]);
  assert.equal(streamed, answer.join("") + "room 4TK-G9P · 12 tok/s\n", "nothing but the answer and its status line while it streamed");
  assert.ok(!streamed.includes(">>>"));
  // the next prompt, with what was typed during the answer
  assert.match(t.text().slice(t.text().lastIndexOf(">>> ")).replace(/\x1b\[[0-9;]*[A-Za-z]/g, ""), /^>>> xy/);
  // Ctrl-C during an answer stops it; at the prompt it leaves
  const t2 = fakeTTY(); let stop2 = 0, end2 = false;
  repl({ input: t2.input, output: t2.output, onLine: async () => { t2.input.write("\x03"); await sleep(20); }, onStop: () => { stop2++; }, onEnd: () => { end2 = true; } });
  await sleep(10); t2.input.write("hi\r"); await sleep(60);
  assert.equal(stop2, 1); assert.equal(end2, false);
  t2.input.write("\x03"); await sleep(20);
  assert.equal(end2, true);
  r.close();
  assert.equal(stops, 0); assert.equal(ended, false);
});
