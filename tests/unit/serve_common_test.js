// cli/lib/common.js and cli/lib/answer.js: the internal request every API adapter produces, its
// normalization and checks, the v1 / v2 ask bodies, how an answer ends, and the bridge's Ask.
import { normalizeMessages, finishRequest, needsV2, askBody, outcome, parseArgs, toolText, normTool, checkToolChoice, checkFormat, blob, ids, id24, withDefaults,
  ApiError, LIMITS, OLD_HOST_MSG, IMAGE_PLACEHOLDER } from "../../cli/lib/common.js";
import { Ask, Collector, TOKEN_SLACK } from "../../cli/lib/answer.js";
import { sseSequence } from "../../cli/lib/sse.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ":\n" + ja + "\n!=\n" + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const throwsKind = (f, kind, re, m) => { try { f(); } catch (e) { ok(e instanceof ApiError && e.kind === kind && re.test(e.message), m + ": " + e.kind + " " + e.message); return; } throw new Error(m + ": no error"); };
const U = (text, o = {}) => ({ role: "user", text, ...o });
const A = (text, o = {}) => ({ role: "assistant", text, ...o });
const T = (text, id) => ({ role: "tool", text, ...(id ? { id } : {}) });
const SYS = (text) => ({ role: "system", text });

// the two shapes Claude Code 2.1.285 sent in one session (captured): the environment block as a
// system message right after the first user message, before and after the first tool round trip
const REMINDER = "<system-reminder>\nAttribution for git commits ...\n</system-reminder>";
const ENV = "# Environment\nYou have been invoked in the following environment:\n - Platform: linux";
Deno.test("normalizeMessages: a mid-conversation system message folds into the user turn before it, so both Claude Code steps render the same first turn", () => {
  const step1 = normalizeMessages([U(REMINDER + "\n\nRead a.txt"), SYS(ENV)]);
  const step2 = normalizeMessages([U(REMINDER + "\n\nRead a.txt"), SYS(ENV), A("Reading.", { calls: [{ id: "toolu_01A", name: "Read", args: { file_path: "a.txt" } }] }), T("1\thello", "toolu_01A")]);
  eq(step1.messages[0], step2.messages[0], "the same first user turn");
  eq(step1.messages[0].text, REMINDER + "\n\nRead a.txt\n\n" + ENV);
  eq(step2.messages.map((m) => m.role), ["user", "assistant", "tool"]);
});
Deno.test("normalizeMessages: leading system joins system; after tool results a system note is an aside; with nothing before it, it folds forward", () => {
  const r = normalizeMessages([SYS("a"), SYS("b"), U("q"), A("", { calls: [{ id: "1", name: "f", args: {} }] }), T("r", "1"), SYS("note"), U("next")], { system: "s0" });
  eq(r.system, "s0\n\na\n\nb");
  eq(r.messages, [U("q"), A("", { calls: [{ id: "1", name: "f", args: {} }] }), T("r", "1"), U("note", { aside: true }), U("next")]);
  eq(normalizeMessages([A("hi"), SYS("x"), U("q")]).messages, [A("hi"), U("x\n\nq")], "forward, into the next user turn");
});
Deno.test("normalizeMessages: consecutive turns merge; tool results go in call order", () => {
  const r = normalizeMessages([U("a"), U("b"), A("x", { reasoning: "" }), A("y", { reasoning: "r2", calls: [{ id: "c1", name: "f", args: {} }, { id: "c2", name: "g", args: {} }] }), T("from g", "c2"), T("orphan", "zz"), T("from f", "c1")]);
  eq(r.messages[0], U("a\n\nb"));
  eq(r.messages[1], A("x\n\ny", { reasoning: "r2", calls: [{ id: "c1", name: "f", args: {} }, { id: "c2", name: "g", args: {} }] }));
  eq(r.messages.slice(2).map((m) => m.text), ["from f", "from g", "orphan"]);
});
Deno.test("parseArgs, toolText, normTool, checkToolChoice, checkFormat", () => {
  const logs = [];
  eq([parseArgs('{"a": 1}'), parseArgs({ b: 2 }), parseArgs(""), parseArgs("[1]", (m) => logs.push(m)), parseArgs("{bad", (m) => logs.push(m))], [{ a: 1 }, { b: 2 }, {}, {}, {}]);
  eq(logs.length, 2, "bad arguments are logged");
  eq(toolText([{ type: "text", text: "a" }, { type: "image", source: {} }, { type: "text", text: "b" }]), "a" + IMAGE_PLACEHOLDER + "b");
  eq(toolText("x"), "x"); eq(toolText(null), "");
  eq(normTool({ name: "f" }), { name: "f", description: "", parameters: { type: "object", properties: {} } });
  throwsKind(() => normTool({ name: "a b" }), "bad", /tool names/, "name");
  throwsKind(() => normTool({ name: "f", parameters: [] }), "bad", /JSON schema object/, "params");
  throwsKind(() => normTool({ name: "f", parameters: { x: "y".repeat(LIMITS.schemaChars) } }), "bad", /characters/, "size");
  const tools = [{ name: "f" }, { name: "g" }];
  checkToolChoice("auto", null, null); checkToolChoice("none", null, null); checkToolChoice({ name: "g" }, tools, ["f", "g"]);
  throwsKind(() => checkToolChoice("required", null, null), "bad", /needs tools/, "required");
  throwsKind(() => checkToolChoice({ name: "h" }, tools, null), "bad", /not in tools/, "named");
  throwsKind(() => checkToolChoice("auto", tools, ["h"]), "bad", /allowed tool h/, "allowed");
  eq(checkFormat({ type: "schema", schema: { type: "object" }, name: "x" }), { type: "schema", schema: { type: "object" }, name: "x" });
  throwsKind(() => checkFormat({ type: "schema" }), "bad", /schema object/, "no schema");
});
Deno.test("blob: pooled1. + base64url, round trip; foreign strings are not ours", () => {
  const s = blob.encode("Thinking… 🦀\nline");
  ok(s.startsWith("pooled1.") && /^[A-Za-z0-9_.-]+$/.test(s), s);
  eq(blob.decode(s), "Thinking… 🦀\nline");
  eq([blob.decode("EqQBCkYIBxgC"), blob.decode("pooled1.@@"), blob.decode(null)], [null, null, null]);
  ok(/^call_[0-9A-Za-z]{24}$/.test(ids.call()) && /^toolu_[0-9A-Za-z]{24}$/.test(ids.toolu()) && /^resp_/.test(ids.resp()) && id24("x").length === 25);
});
const base = (over = {}) => withDefaults({ api: "openai", stream: false, maxTokens: 100, messages: [U("hi")], ...over });
Deno.test("finishRequest: v1 hosts get plain asks only; the byte cap; the early context check", () => {
  const plain = finishRequest(base(), { hostMeta: { api: 1 } });
  eq(needsV2(plain), false);
  const tools = base({ tools: [{ name: "f", description: "", parameters: { type: "object" } }] });
  throwsKind(() => finishRequest(tools, { hostMeta: { api: 1 } }), "bad", new RegExp(OLD_HOST_MSG.replace(/[()]/g, "\\$&")), "old host");
  eq(finishRequest(tools, { hostMeta: { api: 2 } }).tools.length, 1);
  throwsKind(() => finishRequest(base({ messages: [U("q"), A("prefill")] }), {}), "bad", /last message/, "prefill");
  throwsKind(() => finishRequest(base({ toolChoice: "required" }), { hostMeta: { api: 2 } }), "bad", /needs tools/, "choice");
  throwsKind(() => finishRequest(base({ messages: [U("x".repeat(LIMITS.totalChars + 1))] }), {}), "bad", /characters/, "chars");
  // multi-byte text: characters under the limit, bytes over the room's message cap
  const big = "€".repeat(Math.ceil(LIMITS.askBytes / 3) + 10);   // 3 bytes, 1 character each
  throwsKind(() => finishRequest(base({ messages: [U(big)] }), { hostMeta: { api: 2 } }), "toolarge", /bytes/, "bytes");
  throwsKind(() => finishRequest(base({ messages: [U("x".repeat(9000))] }), { hostMeta: { api: 2, ctx: 1024 } }), "ctx", /maximum context length is 1024/, "ctx openai");
  throwsKind(() => finishRequest(base({ api: "anthropic", messages: [U("x".repeat(9000))] }), { hostMeta: { api: 2, ctx: 1024 } }), "ctx", /^prompt is too long: \d+ tokens > 1024 maximum$/, "ctx anthropic");
  eq(finishRequest(base({ messages: [U("x".repeat(8000))] }), { hostMeta: { api: 2, ctx: 1024 } }).messages.length, 1, "8 characters a token is the line");
});
Deno.test("askBody: v1 as today; v2 with tools, calls, reasoning, asides and the new params (call ids stay here)", () => {
  const r = finishRequest(base({ tools: [{ name: "f", description: "d", parameters: { type: "object" } }], toolChoice: { name: "f" }, parallel: false, effort: "low", format: { type: "json" },
    messages: [U("q"), A("", { reasoning: "r", calls: [{ id: "call_x", name: "f", args: { a: 1 } }] }), T("out", "call_x"), U("see", { aside: true })] }), { hostMeta: { api: 2 } });
  eq(askBody(r, true), { api: 2, system: "", messages: [U("q"), A("", { calls: [{ name: "f", args: { a: 1 } }], reasoning: "r" }), T("out"), U("see", { aside: true })],
    tools: [{ name: "f", description: "d", parameters: { type: "object" } }],
    params: { maxTokens: 100, stop: [], thinking: false, client: "", toolChoice: { name: "f" }, parallel: false, format: { type: "json" }, effort: "low" } });
  eq(askBody(finishRequest(base({ system: "s", temperature: 0 }), {}), false), { system: "s", messages: [U("hi")], params: { maxTokens: 100, temperature: 0, stop: [], thinking: false, client: "" } });
});
Deno.test("outcome: tool / stop / stop_seq / length / ctx", () => {
  const a = (reason, calls = []) => ({ reason, calls });
  eq([outcome(a("stop", [{}]), base()), outcome(a("stop", [{}]), base({ toolChoice: { name: "f" } })), outcome(a("stop"), base()), outcome(a("stop_seq"), base()), outcome(a("max", [{}]), base()), outcome(a("ctx"), base())],
    ["tool", "stop", "stop", "stop_seq", "length", "ctx"]);
});
Deno.test("sseSequence numbers events strictly increasing", () => {
  const ev = sseSequence();
  eq(ev("response.created", { a: 1 }), 'event: response.created\ndata: {"type":"response.created","sequence_number":0,"a":1}\n\n');
  ok(ev("x").includes('"sequence_number":1') && ev.next() === 2);
});

// ---- the bridge's side of one ask (answer.js) ----
const REQ = (over = {}) => withDefaults({ api: "openai", maxTokens: 20, stop: [], tools: [{ name: "f", description: "", parameters: {} }, { name: "g", description: "", parameters: {} }], ...over });
function run(msgs, { v2 = true, req = REQ() } = {}) {
  const seen = [];
  const rec = { start: (n) => seen.push(["start", n]), think: (t) => seen.push(["think", t]), text: (t) => seen.push(["text", t]), callStart: (i, id, n) => seen.push(["call", i, id, n]),
    callArgs: (i, a) => seen.push(["args", i, a]), callEnd: (i, a) => seen.push(["end", i, a]), done() {}, error() {}, keepAlive() {} };
  const c = new Collector({ id: "r", created: 1, model: "m" });
  let n = 0;
  const ask = new Ask({ req, meta: {}, v2, encoders: [c, rec], idFor: (i) => `id${i}` + (n++ ? "" : ""), log: () => {} });
  let out = null;
  for (const d of msgs) { out = ask.feed(d); if (out) break; }
  return { out, seen, answer: c.answer };
}
const GS = { t: "ai-genstart", promptTokens: 7, api: 2 };
Deno.test("Ask: v2 messages become encoder calls and one Answer; gendone's calls count", () => {
  const { out, seen, answer } = run([GS, { t: "ai-token", text: "hm", th: 1 }, { t: "ai-token", text: "Hi" }, { t: "ai-call", i: 0, name: "f" }, { t: "ai-call", i: 0, a: '{"a": ' }, { t: "ai-call", i: 0, a: "1}" }, { t: "ai-call", i: 0, end: 1 },
    { t: "ai-gendone", reason: "stop", usage: { in: 7, out: 5, think: 1 }, reused: 3, calls: [{ name: "f", args: '{"a": 1}' }] }]);
  eq(seen, [["start", 7], ["think", "hm"], ["text", "Hi"], ["call", 0, "id0", "f"], ["args", 0, '{"a": '], ["args", 0, "1}"], ["end", 0, '{"a": 1}']]);
  eq(out.answer, answer);
  eq([answer.text, answer.think, answer.calls, answer.usage, answer.reused, answer.open], ["Hi", "hm", [{ id: "id0", name: "f", args: '{"a": 1}' }], { in: 7, out: 5, think: 1 }, 3, null]);
});
Deno.test("Ask: an open call, and the final arguments win over streamed ones that differ", () => {
  const open = run([GS, { t: "ai-call", i: 0, name: "g" }, { t: "ai-call", i: 0, a: '{"x' }, { t: "ai-gendone", reason: "max", usage: { in: 7, out: 20 }, calls: [], open: { i: 0, name: "g" } }]);
  eq([open.answer.reason, open.answer.calls, open.answer.open], ["max", [], { id: "id0", name: "g", args: '{"x' }]);
  const diff = run([GS, { t: "ai-call", i: 0, name: "f" }, { t: "ai-call", i: 0, a: "{}" }, { t: "ai-call", i: 0, end: 1 }, { t: "ai-gendone", reason: "stop", usage: {}, calls: [{ name: "f", args: '{"b": 2}' }] }]);
  eq(diff.answer.calls, [{ id: "id0", name: "f", args: '{"b": 2}' }]);
  const cap = run([GS, { t: "ai-gendone", reason: "stop", usage: {}, calls: [{ name: "f", args: "{}" }, { name: "g", args: "{}" }] }], { req: REQ({ parallel: false }) });
  eq(cap.answer.calls.length, 1, "parallel false: never more than one");
});
Deno.test("Ask: the room is checked: undeclared tools, calls out of order, floods, a host that fell back to v1", () => {
  const bad = (msgs, re, m, o) => { const r = run(msgs, o); ok(r.out?.error && re.test(r.out.error.message), m + ": " + JSON.stringify(r.out)); };
  bad([GS, { t: "ai-call", i: 0, name: "nope" }], /not a declared tool/, "undeclared");
  bad([GS, { t: "ai-call", i: 1, name: "f" }], /out of order/, "order");
  bad([GS, { t: "ai-call", i: 0, a: "{}" }], /not open/, "args first");
  bad([GS, { t: "ai-call", i: 99, name: "f" }], /out of range/, "index");
  bad([GS, { t: "ai-gendone", reason: "stop", usage: {}, calls: [{ name: "nope", args: "{}" }] }], /bad calls/, "final calls");
  bad([{ t: "ai-genstart", promptTokens: 1 }], /older Pooled; retry/, "genstart without api 2");
  const flood = [GS, { t: "ai-call", i: 0, name: "f" }];
  for (let k = 0; k < 20 + TOKEN_SLACK + 5; k++) flood.push({ t: "ai-call", i: 0, a: "x" });
  const r = run(flood);
  eq([r.out.answer.reason, r.out.answer.usage.out], ["max", 20 + TOKEN_SLACK], "fragments count toward max_tokens");
  const v1 = run([{ t: "ai-genstart", promptTokens: 2 }, { t: "ai-call", i: 0, name: "f" }, { t: "ai-token", text: "ok" }, { t: "ai-gendone", reason: "stop", usage: { in: 2, out: 1 } }], { v2: false });
  eq([v1.answer.text, v1.answer.calls], ["ok", []], "a v1 ask ignores ai-call");
});
