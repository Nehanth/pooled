// harness/codetools.js: line-range reads, output caps, append, old/new edits, preview();
// harness/workspace.js: bytes, remove, watch; harness/diff.js: lineDiff.
import { MemoryWorkspace, watch } from "../../harness/workspace.js";
import { codingTools, MAX_LINES, MAX_CHARS, MAX_HITS, MAX_ENTRIES } from "../../harness/codetools.js";
import { previewTools } from "../../harness/preview-tools.js";
import { runJsTool } from "../../harness/run-js.js";
import { CARDS } from "../../harness/cards.js";
import { PreviewServer } from "../../harness/preview.js";
import { toolsSystemPrompt, toolsSystemPromptExact } from "../../harness/tools.js";
import { compileSchema, SCHEMA_CAPS } from "../../harness/jsonschema.js";
import { grammarNodeCount, GrammarConstraint } from "../../harness/constrain.js";
import { lineDiff } from "../../harness/diff.js";
import { CODE_SYSTEM } from "../../harness/code-prompt.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const T = (ws) => Object.fromEntries(codingTools(ws).map((t) => [t.name, t]));
const lines = (n, f = (i) => "line " + i) => Array.from({ length: n }, (_, i) => f(i + 1)).join("\n") + "\n";

Deno.test("read_file: 200 lines per call, numbered N|text, with the continuation hint", async () => {
  const t = T(new MemoryWorkspace({ "big.js": lines(412) }));
  const r = (await t.read_file.run({ path: "big.js" })).split("\n");
  eq(r.length, MAX_LINES + 1); eq(r[0], "1|line 1"); eq(r[199], "200|line 200");
  eq(r[200], "(lines 1-200 of 412; read on with start_line=201)");
  const r2 = (await t.read_file.run({ path: "big.js", start_line: 401 })).split("\n");
  eq(r2, ["401|line 401", ...Array.from({ length: 11 }, (_, i) => `${402 + i}|line ${402 + i}`)], "the tail needs no hint");
  eq(await t.read_file.run({ path: "big.js", start_line: 41, end_line: 43 }), "41|line 41\n42|line 42\n43|line 43", "an explicit range needs no hint");
  eq(await t.read_file.run({ path: "big.js", start_line: 500 }), "error: big.js has 412 lines");
});

Deno.test("read_file: 8,000 chars per call and long lines cut", async () => {
  const t = T(new MemoryWorkspace({ "wide.js": lines(150, () => "x".repeat(99)), "min.js": "y".repeat(5000) + "\nend\n", "e.txt": "" }));
  const r = await t.read_file.run({ path: "wide.js" });
  ok(r.length <= MAX_CHARS + 80, r.length);
  ok(r.endsWith("(lines 1-77 of 150; read on with start_line=78)"), r.slice(-60));
  const m = await t.read_file.run({ path: "min.js" });
  ok(m.startsWith("1|" + "y".repeat(1000) + "…(4000 chars cut)\n2|end"), m.slice(990, 1040));
  eq(await t.read_file.run({ path: "e.txt" }), "(e.txt is empty)");
});

Deno.test("list_dir and search caps", async () => {
  const files = {};
  for (let i = 0; i < 250; i++) files[`many/f${String(i).padStart(3, "0")}.js`] = `const hit = ${i}; // ${"z".repeat(300)}\n`;
  const t = T(new MemoryWorkspace(files));
  const l = (await t.list_dir.run({ path: "many" })).split("\n");
  eq(l.length, MAX_ENTRIES + 1); eq(l.at(-1), "(+50 more)");
  const s = (await t.search.run({ pattern: "hit =" })).split("\n");
  eq(s.length, MAX_HITS + 1); eq(s.at(-1), "(+220 more; narrow the pattern or path)");
  ok(s[0].startsWith("many/f000.js:1: const hit = 0;") && s[0].endsWith("…(158 chars cut)"), s[0]);
});

Deno.test("edit_file: old/new, 0 and 2 matches, lines reported; preview before/after", async () => {
  const ws = new MemoryWorkspace({ "g.js": "a\nb\nc\nb\n", "h.js": "one\ntwo\n" }), t = T(ws);
  eq(await t.edit_file.run({ path: "g.js", old: "zzz", new: "y" }), "error: old not found in g.js; read_file it again and copy the text exactly (whitespace included)");
  eq(await t.edit_file.run({ path: "g.js", old: "b", new: "y" }), "error: old appears 2 times in g.js; include more surrounding lines so it is unique");
  eq(await t.edit_file.run({ path: "nope.js", old: "a", new: "b" }), "error: no such file: nope.js; use write_file to create it");
  const p = await t.edit_file.preview({ path: "h.js", old: "two", new: "2\n2b" });
  eq(p, { path: "h.js", before: "one\ntwo\n", after: "one\n2\n2b\n" });
  eq(await ws.read("h.js"), "one\ntwo\n", "preview does not write");
  eq(await t.edit_file.run({ path: "h.js", old: "two", new: "2\n2b" }), "edited h.js lines 2-3 (1 -> 2 lines)");
  eq(await ws.read("h.js"), "one\n2\n2b\n");
  ok((await t.edit_file.preview({ path: "g.js", old: "b", new: "y" })).error.includes("2 times"));
});

Deno.test("write_file: create, replace, append; preview", async () => {
  const ws = new MemoryWorkspace(), t = T(ws);
  eq(await t.write_file.preview({ path: "a.js", content: "1\n" }), { path: "a.js", before: null, after: "1\n" });
  eq(await t.write_file.run({ path: "a.js", content: "1\n" }), "wrote a.js (1 lines, 2 B)");
  eq(await t.write_file.preview({ path: "a.js", content: "2\n", append: true }), { path: "a.js", before: "1\n", after: "1\n2\n" });
  eq(await t.write_file.run({ path: "a.js", content: "2\n", append: true }), "appended to a.js (now 2 lines, 4 B)");
  eq(await t.write_file.run({ path: "n/b.js", content: "x", append: true }), "wrote n/b.js (1 lines, 1 B)", "append to a new file creates it");
  eq(await t.write_file.run({ path: "a.js", content: "y".repeat(3000) }), "wrote a.js (1 lines, 2.9 KB)");
});

Deno.test("workspace: bytes, remove, watch", async () => {
  const mem = new MemoryWorkspace({ "a.txt": "hé" }), ws = watch(mem), seen = [];
  eq([...(await ws.readBytes("a.txt"))], [104, 195, 169]);
  const off = ws.onChange((e) => seen.push(e));
  ok(watch(ws) === ws, "watching twice returns the same view");
  await ws.writeBytes("img/x.png", new Uint8Array([1, 2, 3]));
  eq([...(await ws.readBytes("img/x.png"))], [1, 2, 3]);
  await ws.write("./b.txt", "b");
  await ws.remove("img");
  eq(await ws.exists("img/x.png"), false);
  eq(seen, [{ path: "img/x.png", kind: "write" }, { path: "b.txt", kind: "write" }, { path: "img", kind: "remove" }]);
  off(); await ws.write("c.txt", "c"); eq(seen.length, 3);
  eq(await mem.read("c.txt"), "c", "writes land in the wrapped workspace");
  let threw = false; try { await ws.remove("nope"); } catch { threw = true; } ok(threw);
});

Deno.test("lineDiff: add, remove, change, folding, cap", () => {
  eq(lineDiff("a\nb\nc\n", "a\nb\nc\n"), []);
  eq(lineDiff(null, "x\ny\n"), [{ op: "+", text: "x" }, { op: "+", text: "y" }]);
  eq(lineDiff("a\nb\nc", "a\nB\nc\nd"), [{ op: " ", text: "a" }, { op: "-", text: "b" }, { op: "+", text: "B" }, { op: " ", text: "c" }, { op: "+", text: "d" }]);
  const big = lines(1000), edited = big.replace("line 500\n", "line five hundred\n");
  const d = lineDiff(big, edited);
  eq(d.map((r) => r.op + (r.skip ?? r.text)).join(","), " 496, line 497, line 498, line 499,-line 500,+line five hundred, line 501, line 502, line 503, 497");
  eq(lineDiff("", lines(500)), null, "over max rows");
  eq(lineDiff("", lines(500), { max: 600 }).length, 500);
});

Deno.test("prompt size: the 8 code tools in the xml block plus the Code system prompt stay under 3,700 chars", () => {
  const ws = watch(new MemoryWorkspace()), s = new PreviewServer(ws);
  const tools = [...codingTools(ws, { server: s }), ...previewTools(s), runJsTool(s)].map(({ name, description, parameters }) => ({ name, description, parameters }));
  eq(tools.map((t) => t.name), ["list_dir", "read_file", "search", "edit_file", "write_file", "serve", "preview_logs", "run_js"]);
  const system = CODE_SYSTEM;   // harness/code-prompt.js (harness-light A.2)
  const p = toolsSystemPrompt(tools, { style: "xml", system });
  console.log(`system prompt + tool block: ${p.length} chars`);
  ok(p.length <= 3700, `system prompt + tool block is ${p.length} chars`);
  for (const [id, t] of Object.entries(CARDS)) ok(t.length <= 245, `card ${id} is ${t.length} chars (~70 tokens max)`);
  s.close();
});

// Code mode's core path (harness/core-model.js, ?hcore=1): the template's own tool block, and the strict
// grammar, which enforces `required` and every declared type while the model writes a call
Deno.test("hcore: the template-worded system text stays under 3,900 chars; every tool compiles into the strict grammar under both call styles", () => {
  const ws = watch(new MemoryWorkspace()), s = new PreviewServer(ws);
  const tools = [...codingTools(ws, { server: s }), ...previewTools(s), runJsTool(s)].map(({ name, description, parameters }) => ({ name, description, parameters }));
  for (const style of ["xml", "json"]) {
    const p = toolsSystemPromptExact(tools, { style, system: CODE_SYSTEM });
    ok(p.length <= 3900, `${style}: the system turn is ${p.length} chars`);
    ok(p.includes(CODE_SYSTEM), style + ": Code's own system text is in it");
    for (const t of tools) compileSchema(t.parameters);
    const n = grammarNodeCount(tools, { style });
    ok(n > 0 && n < SCHEMA_CAPS.nodes, `${style}: ${n} grammar nodes`);
    new GrammarConstraint(tools, { vocabSize: 8, tokenText: () => "", style, mode: "auto", maskCache: new Map() });
  }
  s.close();
});
Deno.test("hcore: required parameters are declared as the tools need them (the grammar enforces them)", () => {
  const ws = watch(new MemoryWorkspace()), s = new PreviewServer(ws);
  const req = Object.fromEntries([...codingTools(ws, { server: s }), ...previewTools(s), runJsTool(s)].map((t) => [t.name, t.parameters.required || []]));
  eq(req, { list_dir: [], read_file: ["path"], search: ["pattern"], edit_file: ["path", "old", "new"], write_file: ["path", "content"], serve: [], preview_logs: [], run_js: ["code"] });
  // every required name is a declared property, and write_file's append stays optional
  for (const t of [...codingTools(ws, { server: s }), ...previewTools(s), runJsTool(s)]) for (const r of t.parameters.required || []) ok(r in t.parameters.properties, `${t.name}.${r}`);
  s.close();
});

Deno.test("read_file: a binary file is named, not dumped", async () => {
  const ws = new MemoryWorkspace({ "a.txt": "hi\n" });
  await ws.writeBytes("img.png", new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0, 0, 0, 13, 0xff, 0xfe]));
  eq(await T(ws).read_file.run({ path: "img.png" }), "(img.png is a binary file, 10 B)");
});

Deno.test("write_file / edit_file: the card's path is the written path; long paths refused", async () => {
  const ws = new MemoryWorkspace({ "a.js": "x\n" }), t = T(ws);
  eq((await t.write_file.preview({ path: ".\\\\sub//./b.js", content: "y" })).path, "sub/b.js");
  eq(await t.write_file.run({ path: "./sub//b.js", content: "y" }), "wrote sub/b.js (1 lines, 1 B)");
  const long = "./".repeat(140) + "a/".repeat(110) + "c.js";
  ok((await t.write_file.preview({ path: long, content: "z" })).error, "preview reports it");
  ok(/path is \d+ characters \(max 200\)/.test(await t.write_file.run({ path: long, content: "z" }).catch((e) => e.message)));
});

// a FileSystemDirectoryHandle in memory, enough for DirWorkspace
function fakeDir() {
  const dir = (m = new Map()) => ({
    kind: "directory", m,
    async getDirectoryHandle(n, { create } = {}) { let h = m.get(n); if (!h && create) m.set(n, h = dir()); if (!h || h.kind !== "directory") throw new DOMException(n, "NotFoundError"); return h; },
    async getFileHandle(n, { create } = {}) { let h = m.get(n); if (!h && create) m.set(n, h = file()); if (!h || h.kind !== "file") throw new DOMException(n, "NotFoundError"); return h; },
    async removeEntry(n) { if (!m.delete(n)) throw new DOMException(n, "NotFoundError"); },
    async *entries() { yield* m.entries(); },
  });
  const file = () => { let data = new Uint8Array(); return { kind: "file",
    async getFile() { return { text: async () => new TextDecoder().decode(data), arrayBuffer: async () => data.slice().buffer }; },
    async createWritable() { let next; return { write: async (v) => { next = typeof v === "string" ? new TextEncoder().encode(v) : new Uint8Array(v); }, close: async () => { data = next; } }; } }; };
  return dir();
}

Deno.test("DirWorkspace private (a folder on disk): hidden and secret files do not exist for the agent", async () => {
  const { DirWorkspace } = await import("../../harness/workspace.js");
  const root = fakeDir(), open = new DirWorkspace(root);
  for (const [p, v] of [[".env", "OPENAI_API_KEY=sk-1"], [".git/config", "[core]"], ["keys/id_rsa", "k"], ["src/a.js", "let a;"], ["server.pem", "p"]]) await open.write(p, v);
  const ws = watch(new DirWorkspace(root, { private: true })), t = T(ws);
  eq(await ws.walk(), ["src/a.js"]);
  eq((await ws.list("")).map((e) => e.name), ["keys", "src"]);
  eq(await ws.exists(".env"), false);
  ok(/off limits/.test(await t.read_file.run({ path: ".env" }).catch((e) => e.message)));
  const pv = await t.write_file.preview({ path: ".git/config", content: "[core]\n\tfsmonitor = x" });
  ok(/off limits/.test(pv.error), "no approval card for it: " + JSON.stringify(pv));
  ok(/off limits/.test(await t.write_file.run({ path: ".git/hooks/pre-commit", content: "x" }).catch((e) => e.message)));
  eq(await t.search.run({ pattern: "sk-|core" }), "no matches");
  eq(await open.read(".git/config"), "[core]", "untouched");
});

Deno.test("search: long lines are tested only up to MAX_LINE characters", async () => {
  const ws = new MemoryWorkspace({ "a.js": "x".repeat(5000) + "NEEDLE\nNEEDLE here\n" });
  eq(await T(ws).search.run({ pattern: "NEEDLE" }), "a.js:2: NEEDLE here");
});
