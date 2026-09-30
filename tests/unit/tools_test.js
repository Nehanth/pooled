// harness/tools.js: tool prompts, streaming tool-call parsing (both Qwen formats), results.
import { detectStyle, toolsSystemPrompt, toolResponses, renderCalls, parseCallBody, ToolCallParser } from "../../harness/tools.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

const TOOLS = [
  { name: "read_file", description: "Read a file", parameters: { type: "object", properties: { path: { type: "string" }, max_lines: { type: "integer" } }, required: ["path"] } },
  { name: "run", description: "Run a command", parameters: { type: "object", properties: { cmd: { type: "string" }, dry: { type: "boolean" } } } },
];
const schemaFor = (n) => TOOLS.find((t) => t.name === n)?.parameters;

Deno.test("detectStyle reads the chat template", () => {
  eq(detectStyle("{{- '<tool_call>\\n<function=' + tool_call.name }}"), "xml");
  eq(detectStyle("{{- '<tool_call>\\n{\"name\": \"' }}"), "json");
  eq(detectStyle(undefined), "json");
});

Deno.test("toolsSystemPrompt keeps the system text and lists every tool", () => {
  const p = toolsSystemPrompt(TOOLS, { system: "You are Tabby." });
  ok(p.startsWith("You are Tabby.\n\n# Tools"), p.slice(0, 40));
  ok(p.includes('"name":"read_file"') && p.includes('"name":"run"'));
  ok(p.includes("<tool_call>\n{\"name\": <function-name>"));
  const x = toolsSystemPrompt(TOOLS, { style: "xml", system: "You are Tabby." });
  ok(x.startsWith("# Tools") && x.endsWith("\n\nYou are Tabby.") && x.includes("<function=example_function_name>"), "xml: tools first, system appended");
  eq(toolsSystemPrompt([], { system: "x" }), "x");
});

Deno.test("parseCallBody: JSON and XML", () => {
  eq(parseCallBody('\n{"name": "read_file", "arguments": {"path": "a.js"}}\n'), { name: "read_file", arguments: { path: "a.js" } });
  eq(parseCallBody('{"name": "run", "arguments": "{\\"cmd\\": \\"ls\\"}"}'), { name: "run", arguments: { cmd: "ls" } });
  eq(parseCallBody("\n<function=read_file>\n<parameter=path>\nsrc/a b.js\n</parameter>\n<parameter=max_lines>\n40\n</parameter>\n</function>\n", schemaFor),
    { name: "read_file", arguments: { path: "src/a b.js", max_lines: 40 } });
  eq(parseCallBody("<function=run>\n<parameter=cmd>\necho 'a\nb'\n</parameter>\n<parameter=dry>\ntrue\n</parameter>\n</function>", schemaFor),
    { name: "run", arguments: { cmd: "echo 'a\nb'", dry: true } });
  eq(parseCallBody("<function=read_file>\n<parameter=path>\na.js\n<parameter=max_lines>\n3\n</function>", schemaFor),
    { name: "read_file", arguments: { path: "a.js", max_lines: 3 } }, "missing </parameter> tolerated");
  ok(parseCallBody("{not json").error, "malformed JSON is an error, not a throw");
  ok(parseCallBody('{"arguments": {}}').error, "a call needs a name");
});

Deno.test("ToolCallParser streams text and calls, never showing half a tag", () => {
  const full = "Let me look.\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.js\"}}\n</tool_call>\n<tool_call>\n<function=run>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>";
  for (const step of [1, 3, 7, 1000]) {
    const P = new ToolCallParser({ schemaFor });
    let text = "", calls = [];
    for (let i = 0; i < full.length; i += step) {
      const r = P.feed(full.slice(i, i + step));
      ok(!r.text.includes("<"), `step ${step}: leaked a tag fragment: ${JSON.stringify(r.text)}`);
      text += r.text; calls.push(...r.calls);
    }
    const e = P.end(); text += e.text; calls.push(...e.calls);
    eq(text, "Let me look.\n", `step ${step} text`);
    eq(calls, [{ name: "read_file", arguments: { path: "a.js" } }, { name: "run", arguments: { cmd: "ls" } }], `step ${step} calls`);
  }
});

Deno.test("ToolCallParser: plain answers pass through; unterminated calls are reported", () => {
  const P = new ToolCallParser();
  eq(P.feed("a < b and <tool").text, "a < b and ");
  eq(P.feed("s> done").text, "<tools> done");
  eq(P.end().text, "");
  const Q = new ToolCallParser({ schemaFor });
  Q.feed("<tool_call>\n<function=run>\n<parameter=cmd>\nls\n</parameter>\n</function>");
  eq(Q.end().calls, [{ name: "run", arguments: { cmd: "ls" } }], "closing tag missing at EOS still yields the call");
  const R = new ToolCallParser();
  R.feed("<tool_call>\n{\"name\": \"x\"");
  ok(R.end().calls[0].error);
});

Deno.test("toolResponses and renderCalls round-trip through the parser", () => {
  eq(toolResponses(["ok", { lines: 3 }]), "<tool_response>\nok\n</tool_response>\n<tool_response>\n{\"lines\":3}\n</tool_response>");
  const calls = [{ name: "read_file", arguments: { path: "a.js", max_lines: 5 } }];
  for (const style of ["json", "xml"]) {
    const P = new ToolCallParser({ schemaFor });
    const r = P.feed(renderCalls(calls, style)); const e = P.end();
    eq([...r.calls, ...e.calls], calls, style);
  }
});

Deno.test("ToolCallParser: a call cut before </function> is an error, not a truncated write", () => {
  const P = new ToolCallParser({ schemaFor });
  P.feed("<tool_call>\n<function=write_file>\n<parameter=path>\ngame.js\n</parameter>\n<parameter=content>\nfunction a() {\n  retu");
  const c = P.end().calls[0];
  ok(c.error && c.open && !c.name, JSON.stringify(c));
});

Deno.test("a call that ends after </parameter> without </function> still counts", () => {
  const P = new ToolCallParser();
  P.feed("<tool_call>\n<function=read_file>\n<parameter=path>\ngame.js\n</parameter>\n");
  const e = P.end();
  if (e.calls.length !== 1 || e.calls[0].name !== "read_file" || e.calls[0].arguments.path !== "game.js") throw new Error(JSON.stringify(e));
});

Deno.test("xml: tool-call tags inside a line of a value are the value's text; a missing </parameter> still ends at a line-start <parameter=", () => {
  const doc = "Calls look like <function=x> with <parameter=y> and </function>.";
  eq(parseCallBody(`<function=write_file>\n<parameter=path>\nREADME.md\n</parameter>\n<parameter=content>\n${doc}\n</parameter>\n</function>`).arguments, { path: "README.md", content: doc });
  eq(parseCallBody("<function=write_file>\n<parameter=path>\na.md\n<parameter=content>\nhi\n</parameter>\n</function>").arguments, { path: "a.md", content: "hi" });
});

Deno.test("JSON calls from small models: extra/missing braces, a stray { before arguments, no </tool_call>", () => {
  const eq = (a, b) => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(JSON.stringify(a) + " != " + JSON.stringify(b)); };
  eq(parseCallBody('{"name": "write_file", "arguments": {"path": "a.html", "content": "<p>{x}</p>\\n"}}}'), { name: "write_file", arguments: { path: "a.html", content: "<p>{x}</p>\n" } });
  eq(parseCallBody('{"name": "serve", {"arguments": {"dir": ".", "entry": "index.html"}}}'), { name: "serve", arguments: { dir: ".", entry: "index.html" } });
  eq(parseCallBody('{"name": "read_file", "arguments": {"path": "x.js"'), { name: "read_file", arguments: { path: "x.js" } });
  const P = new ToolCallParser();
  P.feed('<tool_call>\n{"name": "list_dir", "arguments": {"path": "."}}}');
  eq(P.end().calls, [{ name: "list_dir", arguments: { path: "." } }]);
});

// ---- the API path (docs/design/serve.md): exact template text, coercion, the streaming parser ----
import { pyJSON, toolsSystemPromptExact, callBodyText, callSeparator, coerce, CallStream } from "../../harness/tools.js";

Deno.test("pyJSON writes what the templates' tojson writes (Python separators, key order, non-ASCII as is)", () => {
  eq(pyJSON({ b: 1, a: [true, null, "x, y: z"], "ü": { e: 1.5, s: "a\"b\n" } }), '{"b": 1, "a": [true, null, "x, y: z"], "ü": {"e": 1.5, "s": "a\\"b\\n"}}');
  eq(pyJSON([]), "[]"); eq(pyJSON({}), "{}"); eq(pyJSON("é"), '"é"'); eq(pyJSON({ u: undefined, n: NaN }), '{"n": null}');
});
Deno.test("toolsSystemPromptExact: the Qwen3 and Qwen3.5+ tool blocks (byte-exact tests: api_host_test renderApi)", () => {
  const j = toolsSystemPromptExact(TOOLS, { style: "json", system: "S" });
  ok(j.startsWith("S\n\n# Tools\n\nYou may call one or more functions") && j.includes('\n{"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object"'), j.slice(0, 200));
  ok(j.endsWith('<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'));
  const x = toolsSystemPromptExact(TOOLS, { style: "xml", system: "S", prefix: "Reasoning effort is set to low." });
  ok(x.startsWith("Reasoning effort is set to low.\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{") && x.endsWith("</IMPORTANT>\n\nS"));
  eq(toolsSystemPromptExact([], { system: "S", prefix: "P" }), "P\n\nS");
});
Deno.test("callBodyText / callSeparator: past calls as the templates render them", () => {
  eq(callBodyText({ name: "f", arguments: { a: "x y", n: 2, o: { k: [1] } } }, "json"), '\n{"name": "f", "arguments": {"a": "x y", "n": 2, "o": {"k": [1]}}}\n');
  eq(callBodyText({ name: "f", arguments: { a: "x\ny", n: 2, o: { k: [1] }, e: "high" } }, "xml"),
    "\n<function=f>\n<parameter=a>\nx\ny\n</parameter>\n<parameter=n>\n2\n</parameter>\n<parameter=o>\n{\"k\": [1]}\n</parameter>\n<parameter=e>\nhigh\n</parameter>\n</function>\n");
  eq([callSeparator(0, "", "json"), callSeparator(0, "hi", "json"), callSeparator(1, "", "json")], ["", "\n", "\n"]);
  eq([callSeparator(0, "  ", "xml"), callSeparator(0, "hi", "xml"), callSeparator(1, "hi", "xml")], ["", "\n\n", "\n"]);
});
Deno.test("coerce: vLLM's order (null, integer, number, boolean, object, array, string), type lists and anyOf", () => {
  eq(coerce("5", { type: "integer" }), 5);
  eq(coerce("5.5", { type: "number" }), 5.5);
  eq(coerce("5.5", { type: "integer" }), "5.5", "not an integer: the text");
  eq(coerce("True", { type: "boolean" }), true);
  eq(coerce("null", { type: ["string", "null"] }), null);
  eq(coerce("nullish", { type: ["string", "null"] }), "nullish");
  eq(coerce("12", { anyOf: [{ type: "string" }, { type: "integer" }] }), 12, "integer before string");
  eq(coerce('{"a": 1}', { type: "object" }), { a: 1 });
  eq(coerce("[1, 2]", { type: ["array", "string"] }), [1, 2]);
  eq(coerce("[1, 2", { type: ["array", "string"] }), "[1, 2");
  eq(coerce("high", { enum: ["high", "low"] }), "high");
  eq(coerce("42", undefined), 42, "no schema: a JSON value when it is one");
  eq(coerce("hello", undefined), "hello");
  eq(coerce("7", { $ref: "#/$defs/N" }, { $defs: { N: { type: "integer" } } }), 7, "refs resolve against the tool's schema");
});

// the CallStream contract, at every split point of the text
const SCHEMA_TOOLS = [
  { name: "get_weather", parameters: { type: "object", properties: { city: { type: "string" }, unit: { type: "string", enum: ["c", "f"] }, days: { type: "integer" }, opts: { type: "object" }, tags: { type: "array" }, maybe: { type: ["string", "null"] } }, required: ["city"] } },
  { name: "noop", parameters: { type: "object", properties: {} } },
];
export function streamAll(text, style, { step = 0, tools = SCHEMA_TOOLS, allowed = null, constrained = true } = {}) {
  const S = new CallStream({ style, tools, allowed, constrained });
  const ev = [];
  if (step <= 0) ev.push(...S.push(text));
  else for (let i = 0; i < text.length; i += step) ev.push(...S.push(text.slice(i, i + step)));
  ev.push(...S.end());
  const frags = [];
  for (const e of ev) if (e.t === "args") frags[e.i] = (frags[e.i] || "") + e.a;
  for (const e of ev) if (e.t === "end") { if ((frags[e.i] || "") !== e.args) throw new Error(`fragments ${frags[e.i]} != args ${e.args}`); if (e.mismatch) throw new Error("mismatch on " + text); }
  const names = ev.filter((e) => e.t === "call").map((e) => e.name);
  return { text: ev.filter((e) => e.t === "text").map((e) => e.text).join(""), calls: S.calls, open: S.open, names, ev };
}
function everySplit(text, style, want, opts = {}) {
  for (let step = 1; step <= Math.min(text.length, 40); step++) {
    const r = streamAll(text, style, { ...opts, step });
    eq({ text: r.text, calls: r.calls.map((c) => ({ name: c.name, args: JSON.parse(c.args) })), open: r.open && r.open.name }, want, `step ${step}`);
    ok(!/<tool_c|<too$|<functio$/.test(r.text) || want.text.includes("<tool"), "no half tag leaked: " + r.text);
  }
}
Deno.test("CallStream (xml): content, calls and typed values; string-capable values raw, enums unquoted; fragments = final args", () => {
  const text = "Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis \"FR\"\nline 2\n</parameter>\n<parameter=unit>\nc\n</parameter>\n<parameter=days>\n3\n</parameter>\n"
    + "<parameter=opts>\n{\"a\": [1, 2]}\n</parameter>\n<parameter=tags>\n[\"x\"]\n</parameter>\n<parameter=maybe>\nnull\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=noop>\n</function>\n</tool_call>";
  everySplit(text, "xml", { text: "Let me check.", calls: [{ name: "get_weather", args: { city: "Paris \"FR\"\nline 2", unit: "c", days: 3, opts: { a: [1, 2] }, tags: ["x"], maybe: null } }, { name: "noop", args: {} }], open: null });
  const r = streamAll(text, "xml");
  eq(r.calls[0].args.slice(0, 44), '{"city": "Paris \\"FR\\"\\nline 2", "unit": "c"', "raw strings become JSON strings, as written");
  const i = r.ev.findIndex((e) => e.t === "args" && e.a.includes("Paris"));
  ok(i > 0 && i < r.ev.findIndex((e) => e.t === "args" && e.a.includes("unit")), "the string value streams before the call ends");
});
Deno.test("CallStream (json): the arguments object streams as written; the name once complete", () => {
  const text = "Sure\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo \\\"x\\\" }\", \"days\": 2}}\n</tool_call>\n<tool_call>\n{\"name\": \"noop\", \"arguments\": {}}\n</tool_call>";
  everySplit(text, "json", { text: "Sure", calls: [{ name: "get_weather", args: { city: "Tokyo \"x\" }", days: 2 } }, { name: "noop", args: {} }], open: null });
  eq(streamAll(text, "json").calls[0].args, '{"city": "Tokyo \\"x\\" }", "days": 2}');
});
Deno.test("CallStream: plain text passes; a partial tag at the end is text; whitespace around calls is dropped", () => {
  everySplit("Just an answer: a < b, <b>bold</b> and <tool nothing.\n", "xml", { text: "Just an answer: a < b, <b>bold</b> and <tool nothing.\n", calls: [], open: null });
  everySplit("x <tool_c", "json", { text: "x <tool_c", calls: [], open: null });
  everySplit("\n\n<tool_call>\n<function=noop>\n</function>\n</tool_call>\n\n", "xml", { text: "", calls: [{ name: "noop", args: {} }], open: null });
});
Deno.test("CallStream: an answer cut inside a call leaves it open; before its name nothing of it shows", () => {
  everySplit("ok\n<tool_call>\n<function=get_weather>\n<parameter=city>\nPar", "xml", { text: "ok", calls: [], open: "get_weather" });
  everySplit("ok\n<tool_call>\n<function=get_wea", "xml", { text: "ok", calls: [], open: null });
  everySplit('<tool_call>\n{"name": "get_weather", "arguments": {"city": "P', "json", { text: "", calls: [], open: "get_weather" });
});
Deno.test("CallStream: the lazy bare trigger (xml), only at a line start and only for a declared name", () => {
  everySplit("ok\n<function=noop>\n</function>\n</tool_call>", "xml", { text: "ok", calls: [{ name: "noop", args: {} }], open: null });
  everySplit("see <function=noop> in docs", "xml", { text: "see <function=noop> in docs", calls: [], open: null });
  everySplit("a\n<function=other>\nx", "xml", { text: "a\n<function=other>\nx", calls: [], open: null });
});
Deno.test("CallStream without the grammar: near misses are parsed whole; undeclared or broken calls stay content", () => {
  everySplit("<tool_call>\n```json\n{\"name\": \"noop\", \"arguments\": {}}\n```\n</tool_call>", "json", { text: "", calls: [{ name: "noop", args: {} }], open: null }, { constrained: false });
  everySplit("<tool_call>\n<function name=\"noop\">\n</function>\n</tool_call>", "xml", { text: "", calls: [{ name: "noop", args: {} }], open: null }, { constrained: false });
  everySplit("text <tool_call>\n<function=nope>\n</function>\n</tool_call> after", "xml", { text: "text <tool_call>\n<function=nope>\n</function>\n</tool_call> after", calls: [], open: null }, { constrained: false });
  everySplit("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"A\"}}\n</tool_call>", "json", { text: "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"A\"}}\n</tool_call>", calls: [], open: null }, { allowed: ["noop"] });
});
Deno.test("CallStream: a value containing \"\\n</parameter>\" ends there (known limit: no lookahead)", () => {
  const S = new CallStream({ style: "xml", tools: SCHEMA_TOOLS, constrained: false });
  const ev = [...S.push("<tool_call>\n<function=get_weather>\n<parameter=city>\nA\n</parameter>\nB\n</parameter>\n</function>\n</tool_call>"), ...S.end()];
  eq(JSON.parse(S.calls[0].args).city, "A", "what follows the first closer is lost");
  ok(ev.find((e) => e.t === "end").mismatch, "and without the grammar the stray text makes the streamed arguments suspect: flagged");
});
