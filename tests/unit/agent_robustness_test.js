// Code mode agent robustness (roadmap/28-code-mode-agent-robustness.md, issue #74): the call
// formats room models write besides the one the prompt asks for, the stop for a model that keeps
// repeating a failure (with a warning one step before), and first-error-first preview feedback.
// The model outputs below are shaped like what Qwen 3.6 / Qwen3 1.7B / distilled models wrote.
import { MemoryWorkspace } from "../../harness/workspace.js";
import { codingTools } from "../../harness/codetools.js";
import { Agent, pageError, PAGE_ERR, bareFunctionCalls } from "../../harness/agent.js";
import { parseCallBody, splitCallBody, normalizeXmlCall, ToolCallParser } from "../../harness/tools.js";
import { fold } from "../../harness/preview-tools.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const call = (name, args = {}) => `<tool_call>\n<function=${name}>\n` + Object.entries(args).map(([k, v]) => `<parameter=${k}>\n${v}\n</parameter>\n`).join("") + "</function>\n</tool_call>";
function scripted(replies, seen = []) {
  let i = 0;
  return async function* ({ turns }) { seen.push(turns.map((t) => t.text)); const r = replies[i++] ?? "done"; for (let k = 0; k < r.length; k += 7) yield r.slice(k, k + 7); };
}
const schemas = Object.fromEntries(codingTools(new MemoryWorkspace({})).map((t) => [t.name, t.parameters]));
const schemaFor = (n) => schemas[n];
const parseAll = (text) => { const P = new ToolCallParser({ schemaFor }); const r = P.feed(text), e = P.end(); return { text: r.text + e.text, calls: [...r.calls, ...e.calls] }; };

Deno.test("formats: a ```json fence inside <tool_call> (Qwen3 1.7B)", () => {
  const r = parseAll('I will create the page.\n<tool_call>\n```json\n{"name": "write_file", "arguments": {"path": "index.html", "content": "<h1>Hi</h1>\\n"}}\n```\n</tool_call>');
  eq(r.calls, [{ name: "write_file", arguments: { path: "index.html", content: "<h1>Hi</h1>\n" } }]);
  eq(r.text, "I will create the page.\n");
});

Deno.test("formats: JSON arguments inside <function=NAME> instead of <parameter=> blocks", () => {
  eq(parseCallBody('\n<function=serve>\n{"dir": ".", "entry": "index.html"}\n</function>\n', schemaFor), { name: "serve", arguments: { dir: ".", entry: "index.html" } });
  eq(parseCallBody('<function=read_file>\n{"arguments": {"path": "game.js", "start_line": 40}}\n</function>', schemaFor), { name: "read_file", arguments: { path: "game.js", start_line: 40 } });
  eq(parseCallBody("<function=list_dir>\n</function>", schemaFor), { name: "list_dir", arguments: {} }, "no arguments stays no arguments");
});

Deno.test("formats: <function name=\"x\"> / <parameter name=\"p\"> / <invoke> spellings", () => {
  const attr = '<tool_call>\n<function name="edit_file">\n<parameter name="path">\ngame.js\n</parameter>\n<parameter name="old">\nlet speed = 1;\n</parameter>\n<parameter name="new">\nlet speed = 2;\n</parameter>\n</function>\n</tool_call>';
  eq(parseAll(attr).calls, [{ name: "edit_file", arguments: { path: "game.js", old: "let speed = 1;", new: "let speed = 2;" } }]);
  eq(parseAll('<tool_call>\n<function="read_file">\n<parameter="path">\nindex.html\n</parameter>\n</function>\n</tool_call>').calls, [{ name: "read_file", arguments: { path: "index.html" } }]);
  // a value that quotes such a tag mid-line is the value's own text
  eq(normalizeXmlCall('<function=write_file>\n<parameter=content>\nuse <parameter name="x"> here\n</parameter>\n</function>'),
    '<function=write_file>\n<parameter=content>\nuse <parameter name="x"> here\n</parameter>\n</function>');
  // no <tool_call> at all, Anthropic-style blocks (distilled models)
  const byName = new Map(Object.entries(schemas).map(([name, parameters]) => [name, { name, parameters }]));
  const bare = 'Let me check the files.\n<function_calls>\n<invoke name="list_dir">\n<parameter name="path">\n.\n</parameter>\n</invoke>\n</function_calls>';
  eq(bareFunctionCalls(bare, byName), [{ name: "list_dir", arguments: { path: "." } }]);
});

Deno.test("formats: several calls in one <tool_call> block run as several calls", () => {
  // two <function> blocks, the </tool_call> of the first forgotten (Qwen 3.6 writing two files)
  const two = "<tool_call>\n<function=write_file>\n<parameter=path>\nindex.html\n</parameter>\n<parameter=content>\n<canvas></canvas>\n</parameter>\n</function>\n<tool_call>\n"
    + "<function=write_file>\n<parameter=path>\ngame.js\n</parameter>\n<parameter=content>\nconst c = 1;\n</parameter>\n</function>\n</tool_call>";
  eq(parseAll(two).calls.map((c) => [c.name, c.arguments.path, c.arguments.content]), [["write_file", "index.html", "<canvas></canvas>"], ["write_file", "game.js", "const c = 1;"]]);
  // a JSON array, and JSON objects one per line (Qwen3 1.7B)
  eq(parseAll('<tool_call>\n[{"name": "read_file", "arguments": {"path": "a.js"}}, {"name": "read_file", "arguments": {"path": "b.js"}}]\n</tool_call>').calls.map((c) => c.arguments.path), ["a.js", "b.js"]);
  eq(parseAll('<tool_call>\n{"name": "write_file", "arguments": {"path": "a.css", "content": "p { color: red; }"}}\n{"name": "serve", "arguments": {}}\n</tool_call>').calls.map((c) => c.name), ["write_file", "serve"]);
  // one call whose value mentions the tags inline, or one JSON call with braces in a string, stays one call
  eq(splitCallBody("<function=write_file>\n<parameter=content>\nx</function><function=y>\n</parameter>\n</function>").length, 1);
  eq(splitCallBody('{"name": "write_file", "arguments": {"content": "} {"}}').length, 1);
  // a JSON call cut before </tool_call> with a second object still counts both
  eq(parseAll('<tool_call>\n{"name": "list_dir", "arguments": {}}\n{"name": "read_file", "arguments": {"path": "x"}}').calls.map((c) => c.name), ["list_dir", "read_file"]);
});

Deno.test("formats: a malformed call's error is short", () => {
  const r = parseCallBody('{"name": "write_file", "arguments": {"path": "a.js", "content": ' + '"x" + '.repeat(200) + "}}");
  ok(r.error && r.error.length <= 170, r.error);
});

Deno.test("loops: a model repeating a failing edit is warned once, then stopped with a message naming it", async () => {
  const ws = new MemoryWorkspace({ "game.js": "function drop() {\n  piece.y += 1;\n}\n" });
  // the model keeps "fixing" a line that is not in the file, with small variations each time
  const edit = (n) => call("edit_file", { path: "game.js", old: `  piece.y = piece.y + ${n};`, new: "  piece.y += 2;" });
  const seen = [];
  const A = new Agent({ generate: scripted([edit(1), edit(2), edit(3), edit(4), "done"], seen), tools: codingTools(ws) });
  const r = await A.run("make pieces fall faster");
  eq(r.reason, "stuck");
  eq(r.steps, 3);
  ok(/^Stopped: edit_file game\.js .* failed 3 times in a row: error: old not found in game\.js/.test(r.text), r.text);
  ok(/note: this failed the same way last step \(error: old not found in game\.js.*\)\. Do something different: the task stops if it fails a third time\./.test(seen[2].at(-1)), seen[2].at(-1));
  ok(!/note: this failed/.test(seen[1].at(-1)), "no warning after the first failure");
});

Deno.test("loops: writing the same content again and again stops too", async () => {
  const ws = new MemoryWorkspace({ "index.html": "<h1>hi</h1>\n" });
  const w = call("write_file", { path: "index.html", content: "<h1>hi</h1>\n" });
  const rd = call("read_file", { path: "index.html" });
  // a read in between each time, so it is never the plain same-call repeat of the step before
  const A = new Agent({ generate: scripted([w, rd, w, rd, w, rd, "done"]), tools: codingTools(ws) });
  const r = await A.run("fix the heading");
  eq([r.reason, r.steps], ["stuck", 5]);
  ok(/write_file index\.html .*: unchanged: index\.html already has exactly this content/.test(r.text), r.text);
});

Deno.test("loops: the page showing the same first error after every change is warned about, then stopped", async () => {
  let rev = 0;
  const log = [];
  const tools = [
    { name: "edit_file", mutates: true, parameters: { type: "object", properties: { path: { type: "string" }, old: { type: "string" }, new: { type: "string" } } },
      run: async (a) => { log.push(a.new); return `edited ${a.path} line 12 (1 -> 1 lines)`; } },
    { name: "serve", mutates: false, parameters: { type: "object", properties: {} },
      // the line number moves as the edits shift the file; the error is the same
      run: async () => `serving . on :5173 (index.html, 2 files, 3.1 KB)\nloaded in 40 ms · 1 error:\n[0.1s] error game.js:${40 + ++rev}:9 TypeError: Cannot read properties of undefined (reading 'shape')` },
  ];
  const step = (n) => call("edit_file", { path: "game.js", old: `x${n}`, new: `y${n}` }) + "\n" + call("serve");
  const seen = [];
  const A = new Agent({ generate: scripted([1, 2, 3, 4, 5, 6, 7].map(step), seen), tools });
  const r = await A.run("fix the crash");
  eq(r.reason, "stuck");
  eq(r.steps, PAGE_ERR.stop);
  ok(/^Stopped: the page showed the same error after 4 changes: game\.js:45:9 TypeError: Cannot read properties of undefined \(reading 'shape'\)$/.test(r.text), r.text);
  const warned = seen.findIndex((s) => /note: the page showed this same error after each of your last 2 changes/.test(s.at(-1)));
  eq(warned, PAGE_ERR.warn, "warned right after the third sighting");
  eq(seen.at(-1).filter((t) => /note: the page showed this same error/.test(t)).length, 1, "sent once");
});

Deno.test("loops: a page error that changes (progress) or a check without a change does not count", async () => {
  let k = 0;
  const errs = ["game.js:3 ReferenceError: ctx is not defined", "game.js:9 TypeError: board is undefined", "game.js:20 TypeError: piece is null", "game.js:31 RangeError: bad row", "game.js:44 TypeError: x is not a function", "game.js:50 SyntaxError: missing )"];
  const tools = [
    { name: "edit_file", mutates: true, parameters: { type: "object", properties: { old: { type: "string" } } }, run: async () => "edited game.js line 3 (1 -> 1 lines)" },
    { name: "serve", mutates: false, parameters: { type: "object", properties: {} }, run: async () => `serving . on :5173 (index.html, 2 files, 1 KB)\nloaded in 5 ms · 1 error:\n[0.1s] error ${errs[k++ % errs.length]}` },
  ];
  const step = (n) => call("edit_file", { old: `v${n}` }) + "\n" + call("serve");
  const A = new Agent({ generate: scripted([1, 2, 3, 4, 5, 6].map(step)), tools });
  eq((await A.run("fix it")).reason, "done");
  eq(pageError("serve", "serving . on :5173\nloaded in 5 ms · no errors"), "");
  eq(pageError("read_file", "[0.1s] error x"), null);
  eq(pageError("preview_logs", "(rev 2, current)\n[1.0s] error game.js:9:2 tick failed ×12\nnext: since=13"), "game.js:9:2 tick failed");
});

Deno.test("preview feedback: errors first, repeats folded into their first copy", () => {
  const e = (seq, level, text, line) => ({ seq, level, text, src: "game.js", line, col: 1, t: 100 + seq, rev: 1 });
  const lines = [e(1, "warn", "slow frame", 0), e(2, "error", "ReferenceError: grid is not defined", 12), e(3, "error", "TypeError: cannot draw", 80), e(4, "error", "ReferenceError: grid is not defined", 12), e(5, "error", "TypeError: cannot draw", 80)];
  const rows = fold(lines.filter((x) => x.level === "error"), { all: true });
  eq(rows.map((r) => [r.e.text, r.n]), [["ReferenceError: grid is not defined", 2], ["TypeError: cannot draw", 2]]);
  eq(fold(lines.filter((x) => x.level === "error")).length, 4, "without all, only neighbours fold");
});

Deno.test("loops: the same successful call over and over is stopped as a repeat, not as a failure", async () => {
  const ws = new MemoryWorkspace({ "index.html": "<h1>hi</h1>\n" });
  const rd = call("read_file", { path: "index.html" });
  const r = await new Agent({ generate: scripted([rd, rd, rd, rd, "done"]), tools: codingTools(ws) }).run("look");
  eq([r.reason, r.text], ["stuck", "Stopped: read_file index.html was repeated 3 times with nothing changed"]);
});
