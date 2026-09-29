// harness/cards.js and the error recovery around it (docs/design/harness-light.md A.1, A.3, A.4):
// which card a call earns, one card per step, no repeat while in context, the repeat
// short-circuit, the empty-answer nudge, forgiving edit_file, the rewrite note.
import { MemoryWorkspace } from "../../harness/workspace.js";
import { codingTools } from "../../harness/codetools.js";
import { Agent } from "../../harness/agent.js";
import { CARDS, pickCard, hint } from "../../harness/cards.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const call = (name, args = {}) => `<tool_call>\n<function=${name}>\n` + Object.entries(args).map(([k, v]) => `<parameter=${k}>\n${v}\n</parameter>\n`).join("") + "</function>\n</tool_call>";
function scripted(replies, seen = []) {
  let i = 0;
  return async function* ({ turns }) { seen.push(turns.map((t) => t.text)); yield replies[i++] ?? "done"; };
}
const T = (ws) => Object.fromEntries(codingTools(ws).map((t) => [t.name, t]));
const count = (s, x) => s.split(x).length - 1;

Deno.test("pickCard: each trigger", () => {
  eq(pickCard({ call: { name: "read_file" }, result: "x", repeat: true }), "loop");
  eq(pickCard({ call: { name: "write_file", arguments: {}, salvage: { lines: 3 } }, result: "wrote a" }), "parts");
  eq(pickCard({ call: { error: "unterminated <tool_call>", open: true }, result: "error: your answer was cut at 4096 tokens before the call was complete" }), "parts");
  eq(pickCard({ call: { error: "unterminated <tool_call>", open: true }, result: "error: your answer ended in the middle of a tool call." }), "format");
  eq(pickCard({ call: { name: "nope" }, result: "error: there is no tool called nope; the tools are a" }), "format");
  eq(pickCard({ call: { name: "edit_file" }, result: "error: old not found in a.js" }), "edit");
  eq(pickCard({ call: { name: "edit_file" }, result: "error: no such file: a.js; use write_file to create it" }), null);
  eq(pickCard({ call: { name: "write_file", arguments: { content: "x\n".repeat(160) } }, result: "wrote a.js (160 lines, 1 KB)" }), "parts");
  eq(pickCard({ call: { name: "write_file", arguments: { content: "x\n".repeat(160), append: true } }, result: "appended to a.js" }), null);
  eq(pickCard({ call: { name: "write_file", arguments: { content: "x" } }, result: "wrote a.js (50 lines, 1 KB) · 49 of 50 old lines unchanged" }), "rewrite");
  eq(pickCard({ call: { name: "serve" }, result: "serving . on :5173\nloaded in 5 ms · 1 error:\n[0.1s] error game.js:3 x" }), "errors");
  eq(pickCard({ call: { name: "serve" }, result: "serving . on :5173\nloaded in 5 ms · no errors" }), null);
  eq(pickCard({ call: { name: "read_file" }, result: "1|x" }), null);
  // app code run with run_js instead of written into the page; a short probe is fine
  const game = "const c = document.getElementById('g');\n" + "let x = 1;\n".repeat(20) + "requestAnimationFrame(loop);";
  eq(pickCard({ call: { name: "run_js", arguments: { code: game } }, result: "ok in 5 ms (no output)" }), "scratch");
  eq(pickCard({ call: { name: "run_js", arguments: { code: "console.log(document.title)" } }, result: "ok in 5 ms\nx" }), null);
  eq(pickCard({ call: { name: "run_js", arguments: { code: game } }, result: "error: x is not defined" }), "runjs");
  for (const [id, t] of Object.entries(CARDS)) ok(t.length <= 245, `card ${id}: ${t.length} chars`);
});

Deno.test("agent: a failed edit gets the edit card once; the card is not repeated while in context", async () => {
  const ws = new MemoryWorkspace({ "a.js": "let a = 1;\n" }), ev = [];
  const bad = call("edit_file", { path: "a.js", old: "zzz", new: "y" }), bad2 = call("edit_file", { path: "a.js", old: "qqq", new: "y" });
  const A = new Agent({ generate: scripted([bad, bad2, "gave up"]), tools: codingTools(ws), onEvent: (e) => e.type === "card" && ev.push(e.id) });
  await A.run("edit");
  eq(ev, ["edit"]);
  ok(A.turns[2].text.includes(hint("edit")), A.turns[2].text);
  ok(!A.turns[4].text.includes("hint:"), A.turns[4].text);
  eq(A.reqs[1].cards, { edit: 1 });
});

Deno.test("agent: the same read again with nothing changed is answered as a repeat; a write in between makes it fresh", async () => {
  const ws = new MemoryWorkspace({ "a.js": "let a = 1;\n" }), runs = [];
  const tools = codingTools(ws).map((t) => (t.name === "read_file" ? { ...t, run: (a) => { runs.push(a.path); return t.run(a); } } : t));
  const rd = call("read_file", { path: "a.js" });
  const A = new Agent({ generate: scripted([rd, rd, call("write_file", { path: "a.js", content: "let a = 2;" }), rd, "ok"]), tools });
  const r = await A.run("look");
  eq(r.reason, "done");
  eq(runs, ["a.js", "a.js", "a.js"], "a read runs again to check the file (it can change outside the agent)");
  ok(A.turns[4].text.includes("1|let a = 1; (same call as step 1; nothing changed)" + hint("loop")), A.turns[4].text);
  // a model that keeps repeating is stopped as stuck instead of burning the steps
  const B = new Agent({ generate: scripted(Array(8).fill(rd)), tools: codingTools(ws) });
  const rb = await B.run("look");
  eq([rb.reason, rb.steps], ["stuck", 4]);
});

Deno.test("agent: a repeated read after the file changed outside the agent gets the new content, not a stale answer", async () => {
  const ws = new MemoryWorkspace({ "a.js": "let a = 1;\n" });
  const rd = call("read_file", { path: "a.js" });
  // the user saves the file in the editor while the model writes its second answer
  const gen = scripted([rd, rd, "ok"]);
  let n = 0;
  const A = new Agent({ generate: (x) => { if (++n === 2) ws.files.set("a.js", "let a = 2;\n"); return gen(x); }, tools: codingTools(ws) });
  const r = await A.run("look");
  eq(r.reason, "done");
  ok(A.turns[4].text.includes("1|let a = 2;"), A.turns[4].text);
  ok(!A.turns[4].text.includes("same call as step"), A.turns[4].text);
  ok(!A.turns[4].text.includes(hint("loop")), "no loop card for a changed file");
  // a mutating call repeated with nothing changed is still not run twice
  const writes = [];
  const tools = codingTools(ws).map((t) => (t.name === "write_file" ? { ...t, run: (a) => { writes.push(a.path); return t.run(a); } } : t));
  const wr = call("write_file", { path: "b.js", content: "x" });
  const B = new Agent({ generate: scripted([wr, wr, "ok"]), tools });
  await B.run("write");
  eq(writes, ["b.js"]);
});

Deno.test("agent: a repeated serve with the same errors is a repeat even though its load time differs", async () => {
  let n = 0, err = "boom";
  const serve = { name: "serve", description: "s", parameters: { type: "object", properties: {} }, mutates: false,
    run: async () => `serving . on :5173\nloaded in ${++n * 7} ms · 1 error:\n[0.${n}s] error a.js:1 ${err}` };
  const sv = call("serve", {});
  const A = new Agent({ generate: scripted(Array(8).fill(sv)), tools: [serve] });
  const r = await A.run("go");
  eq([r.reason, r.steps, n], ["stuck", 4, 4], "each serve ran, and the same errors three times is stuck");
  ok(A.turns[4].text.includes("(same call as step 1; nothing changed)"), A.turns[4].text);
  // other errors on the second serve: fresh, not a repeat
  n = 0;
  const gen = scripted([sv, sv, "ok"]);
  const B = new Agent({ generate: (x) => { if (n === 1) err = "other"; return gen(x); }, tools: [serve] });
  await B.run("go");
  ok(B.turns[4].text.includes("error a.js:1 other") && !B.turns[4].text.includes("same call as step"), B.turns[4].text);
});

Deno.test("agent: a repeated serve of the same broken page is stuck at step 4 though its counts, ×N and more: cursor change", async () => {
  // real serve output: the error count, the fold count and the preview_logs cursor all grow on each reload
  let n = 0, err = "boom";
  const serve = { name: "serve", description: "s", parameters: { type: "object", properties: {} }, mutates: false,
    run: async () => { n++; return `serving . on :5173 (index.html, 2 files, 1 KB)\nloaded in ${n * 13} ms · ${n * 40} errors, ${n} warning${n > 1 ? "s" : ""}:\n` +
      `[0.${n}s] error game.js:12 ${err}\n  at tick (game.js:12) ×${n * 40}\n[0.${n}s] warn game.js:3 slow ×${n}\nmore: preview_logs since=${n * 57}`; } };
  const sv = call("serve", {});
  const A = new Agent({ generate: scripted(Array(30).fill(sv)), tools: [serve] });
  const r = await A.run("go");
  eq([r.reason, r.steps, n], ["stuck", 4, 4], "the same broken page served again is a loop, not 24 serves");
  ok(A.turns[4].text.includes("(same call as step 1; nothing changed)"), A.turns[4].text);
  // a different error on the next serve: fresh, not a repeat
  n = 0;
  const gen = scripted([sv, sv, "ok"]);
  const B = new Agent({ generate: (x) => { if (n === 1) err = "other"; return gen(x); }, tools: [serve] });
  await B.run("go");
  ok(B.turns[4].text.includes("error game.js:12 other") && !B.turns[4].text.includes("same call as step"), B.turns[4].text);
});

Deno.test("agent: an empty answer gets one nudge; a second one ends the request", async () => {
  const seen = [];
  const A = new Agent({ generate: scripted(["", "Done: nothing to do."], seen), tools: codingTools(new MemoryWorkspace()) });
  const r = await A.run("hi");
  eq([r.reason, r.text, r.steps], ["done", "Done: nothing to do.", 2]);
  eq(seen[1].slice(-1), ["(empty answer: call a tool or say you are done)"]);
  const B = new Agent({ generate: scripted(["", " "]), tools: codingTools(new MemoryWorkspace()) });
  eq((await B.run("hi")).steps, 2);
});

Deno.test("agent: one card per step, the most urgent; errors card at most once per request", async () => {
  const ws = new MemoryWorkspace({ "a.js": "x\n" }), ev = [];
  const serve = { name: "serve", description: "s", parameters: { type: "object", properties: { port: { type: "integer" } } }, mutates: false,
    run: async ({ port }) => `serving . on :${port}\nloaded in 5 ms · 1 error:\n[0.1s] error a.js:1 boom` };
  const two = call("edit_file", { path: "a.js", old: "zzz", new: "y" }) + "\n" + call("serve", { port: 1 });
  const A = new Agent({ generate: scripted([two, call("serve", { port: 2 }), call("serve", { port: 3 }), "ok"]), tools: [...codingTools(ws), serve], onEvent: (e) => e.type === "card" && ev.push(e.id) });
  await A.run("go");
  eq(ev, ["edit", "errors"], "step 1: edit wins over errors; step 2: errors; step 3: errors already sent");
  eq(count(A.turns[2].text, "hint:"), 1);
});

Deno.test("edit_file: trailing spaces, smart quotes and indentation still match; ambiguity refuses; misses name the closest lines", async () => {
  const src = "function f() {\n\tif (x) {\n\t\treturn 'a';   \n\t}\n}\n";
  let ws = new MemoryWorkspace({ "a.js": src }), t = T(ws);
  eq(await t.edit_file.run({ path: "a.js", old: "\t\treturn ‘a’;", new: "\t\treturn 'b';" }), "edited a.js line 3 (1 -> 1 lines, matched ignoring whitespace)");
  eq(await ws.read("a.js"), src.replace("return 'a';   ", "return 'b';"));
  ws = new MemoryWorkspace({ "a.js": src }); t = T(ws);
  const r = await t.edit_file.run({ path: "a.js", old: "  if (x) {\n    return 'a';\n  }", new: "  if (x) {\n    return 'c';\n  }" });
  eq(r, "edited a.js lines 2-4 (3 -> 3 lines, matched ignoring indentation)");
  eq(await ws.read("a.js"), "function f() {\n\tif (x) {\n\t  return 'c';\n\t}\n}\n", "new text re-based on the file's indent");
  const pv = await t.edit_file.preview({ path: "a.js", old: "\t  return 'c';   ", new: "\t  return 'd';" });
  eq(pv.after, "function f() {\n\tif (x) {\n\t  return 'd';\n\t}\n}\n", "the approval preview shows what run writes");
  ws = new MemoryWorkspace({ "d.js": "a\n  b\nc\na\n\tb\nc\n" }); t = T(ws);
  ok(/appears more than once in d\.js \(ignoring indentation\)/.test(await t.edit_file.run({ path: "d.js", old: "a\nb\nc", new: "z" })));
  const big = Array.from({ length: 60 }, (_, i) => `line ${i + 1};`).join("\n") + "\n";
  ws = new MemoryWorkspace({ "b.js": big }); t = T(ws);
  const miss = await t.edit_file.run({ path: "b.js", old: "line 40;\nline 41 changed;\nline 42;", new: "x" });
  ok(miss.startsWith("error: old not found in b.js; lines 40-42 are closest:\n40|line 40;\n41|line 41;\n42|line 42;"), miss);
  eq(await ws.read("b.js"), big);
});

Deno.test("write_file: rewriting a long file that barely changed says so (and earns the rewrite card)", async () => {
  const body = Array.from({ length: 50 }, (_, i) => `line ${i}`).join("\n") + "\n";
  const ws = new MemoryWorkspace({ "g.js": body }), t = T(ws);
  const r = await t.write_file.run({ path: "g.js", content: body.replace("line 7", "line seven") });
  ok(/^wrote g\.js \(50 lines, \d+ B\) · 49 of 50 old lines unchanged$/.test(r), r);
  eq(pickCard({ call: { name: "write_file", arguments: { content: "" } }, result: r }), "rewrite");
  eq(await t.write_file.run({ path: "g.js", content: "new\n" }), "wrote g.js (1 lines, 4 B)");
  ok(!/unchanged/.test(await t.write_file.run({ path: "n.js", content: body })), "a new file has no note");
});

Deno.test("review fixes: run_js card, garbage calls, duplicates in one answer, repeats after compaction, blank lines in old", async () => {
  eq(pickCard({ call: { name: "run_js" }, result: "error in 3 ms\nError: x" }), "runjs");
  eq(pickCard({ call: { name: "run_js" }, result: "ok in 3 ms\n[0.1s] error x" }), null, "(ok only when nothing errored)");
  eq(pickCard({ call: { name: "write_file", garbage: true, error: "x", open: true }, result: "error: x" }), null, "no card for the engine's garbage");
  // an answer the adapter stopped as garbage: well-formed calls in it do not run
  const ws = new MemoryWorkspace({ "a.js": "keep\n" });
  const w = call("write_file", { path: "a.js", content: "zz" });
  const G = new Agent({ generate: scripted([w, "ok"]), tools: codingTools(ws), usage: () => ({ reason: "garbage", generated: 40, prompt: 9, forced: 20 }) });
  await G.run("go");
  eq(await ws.read("a.js"), "keep\n");
  ok(/not run: 20 tokens were forced/.test(G.turns[2].text) && !/hint:/.test(G.turns[2].text), G.turns[2].text);
  // the same append twice in one answer runs once
  const ws2 = new MemoryWorkspace({ "b.js": "1\n" });
  const ap = call("write_file", { path: "b.js", content: "2", append: "true" });
  const D = new Agent({ generate: scripted([ap + "\n" + ap, "ok"]), tools: codingTools(ws2) });
  await D.run("go");
  eq(await ws2.read("b.js"), "1\n2");
  ok(/skipped: the same call/.test(D.turns[2].text), D.turns[2].text);
  // a repeat whose earlier result was compacted away gets the result again
  const ws3 = new MemoryWorkspace({ "c.js": "let c = 3;\n" }), rd = call("read_file", { path: "c.js" });
  const E = new Agent({ generate: scripted([rd, rd, "ok"]), tools: codingTools(ws3) });
  const gen = E.generate;
  let n = 0;
  E.generate = (o) => { if (n++ === 1) o.turns[2].text = "<tool_response>\n(output of read_file dropped; run it again if needed)\n</tool_response>"; return gen(o); };
  await E.run("look");
  ok(E.turns[4].text.includes("1|let c = 3;\n(same call as step 1; nothing changed)"), E.turns[4].text);
  // edit_file ignoring indentation with a blank line inside old
  const ws4 = new MemoryWorkspace({ "e.js": "function f() {\n\tlet a = 1;\n\n\treturn a;\n}\n" }), t = T(ws4);
  eq(await t.edit_file.run({ path: "e.js", old: "  let a = 1;\n\n  return a;", new: "  let a = 2;\n\n  return a;" }), "edited e.js lines 2-4 (3 -> 3 lines, matched ignoring indentation)");
  eq(await ws4.read("e.js"), "function f() {\n\tlet a = 2;\n\n\treturn a;\n}\n");
});
