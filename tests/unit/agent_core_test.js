// harness/agent.js on the core path (?hcore=1, harness/core-model.js): the Agent's behaviours that
// carry over from the legacy path, run on scriptedCore (scripted raw answers through CallStream):
// structured history, salvage of a cut write, open calls paired with their error, the bare-call
// fallback (with the recorded Qwen3 1.7B / Qwen3.6 slips), garbage, model errors, abort, the
// stuck guard, idle-done, compaction of results arrays, sessions, and switching paths both ways.
import { Agent, stubResultList, legacyText, stripStrayTags, bareFromText } from "../../harness/agent.js";
import { scriptedCore, toMessages } from "../../harness/core-model.js";
import { MemoryWorkspace } from "../../harness/workspace.js";
import { codingTools } from "../../harness/codetools.js";
import { toolResponses } from "../../harness/tools.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

const call = (name, args = {}) => `<tool_call>\n<function=${name}>\n` + Object.entries(args).map(([k, v]) => `<parameter=${k}>\n${v}\n</parameter>\n`).join("") + "</function>\n</tool_call>";
// each ask streams the next reply in pieces; onPiece(k, n) runs before piece k of reply n
function scripted(replies, { seen = [], onPiece } = {}) {
  let i = 0;
  return async function* ({ system, turns, signal }) {
    seen.push({ system, turns: turns.map((t) => ({ ...t })) });
    const n = i++, r = replies[n] ?? "done";
    for (let k = 0; k < r.length; k += 5) {
      onPiece?.(k, n);
      if (signal?.aborted) return;
      yield r.slice(k, k + 5);
      await null;
    }
  };
}
const core = (replies, o = {}) => scriptedCore(scripted(replies, o), { style: o.style || "xml", ...(o.count ? { count: o.count } : {}) });
const tools = (log = []) => [
  { name: "read_file", description: "read", parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] }, mutates: false,
    run: async (a) => { log.push(["read", a.path]); return a.path === "big" ? "x".repeat(10000) : `contents of ${a.path}`; } },
  { name: "write_file", description: "write", parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" }, append: { type: "boolean" } }, required: ["path", "content"] }, mutates: true,
    preview: async (a) => ({ path: a.path, before: null, after: a.content }),
    run: async (a) => { log.push(["write", a.path, a.content]); return `wrote ${a.path}`; } },
  { name: "edit_file", description: "edit", parameters: { type: "object", properties: { path: { type: "string" }, old: { type: "string" }, new: { type: "string" } }, required: ["path", "old", "new"] }, mutates: true,
    run: async (a) => { log.push(["edit", a.path]); return `error: old not found in ${a.path}`; } },
  { name: "serve", description: "serve", parameters: { type: "object", properties: { port: { type: "integer" } } }, mutates: false, run: async () => "serving :5173 (index.html)\nloaded in 12 ms · no errors" },
];

Deno.test("hcore agent: read, write, answer; history is structured and renders as v2 messages", async () => {
  const log = [], seen = [];
  const A = new Agent({ model: core(["I'll look.\n" + call("read_file", { path: "a.js" }), call("write_file", { path: "b.js", content: "x = 1;" }), "Done: wrote b.js."], { seen }), tools: tools(log), system: "You are Tabby." });
  const r = await A.run("make b.js");
  eq(r, { text: "Done: wrote b.js.", steps: 3, calls: 2, reason: "done" });
  eq(log, [["read", "a.js"], ["write", "b.js", "x = 1;"]]);
  ok(seen[0].system.includes("<tools>") && seen[0].system.endsWith("You are Tabby."), "the template's tool block, then Code's system text");
  eq(A.system, seen[0].system);
  eq(A.turns[1], { role: "assistant", text: "I'll look.", sampled: [{ name: "read_file", args: '{"path": "a.js"}' }], req: 1 });
  eq(A.turns[2].results, ["contents of a.js"]);
  eq(A.turns[2].text, toolResponses(["contents of a.js"]));
  const { messages } = toMessages(A.turns);
  eq(messages.map((m) => m.role), ["user", "assistant", "tool", "assistant", "tool", "assistant"]);
  eq(messages[3].calls, [{ name: "write_file", args: { path: "b.js", content: "x = 1;" } }]);
});

Deno.test("hcore agent: a bare JSON call (Qwen3 1.7B) runs, with the hint, and leaves history once", async () => {
  const log = [];
  const reply = "Let me write it:\n```json\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"a.js\", \"content\": \"x\"}}\n```";
  const ev = [];
  const A = new Agent({ model: core([reply, "ok"], { style: "json" }), tools: tools(log), onEvent: (e) => ev.push(e) });
  const r = await A.run("go");
  eq(r.reason, "done");
  eq(log, [["write", "a.js", "x"]]);
  eq(A.turns[1].text, "Let me write it:", "the call and its fence are not left in the content");
  eq(A.turns[1].sampled, [{ name: "write_file", args: '{"path":"a.js","content":"x"}' }]);
  ok(/hint: this ran, but write tool calls as <tool_call>\n\{"name"/.test(A.turns[2].results[0]), A.turns[2].results[0]);
  eq(ev.filter((e) => e.type === "bare").map((e) => e.n), [1]);
});

Deno.test("hcore agent: Qwen3.6's garbled opener (baseline add-feature) runs the call and is taken out of history; bare tags run too", async () => {
  const log = [];
  const garbled = "Now I understand the app.\n\n<toolly_call>\n<function=read_file>\n<parameter=path>\napp.js\n</parameter>\n</function>\n</tool_call>";
  const tags = "<write_file>\n<path>a.html</path>\n<content>\n<h1>x</h1>\n</content>\n</write_file>";
  const A = new Agent({ model: core([garbled, tags, "ok"]), tools: tools(log) });
  await A.run("go");
  eq(log, [["read", "app.js"], ["write", "a.html", "<h1>x</h1>"]]);
  eq(A.turns[1].text, "Now I understand the app.");
  eq(A.turns[3].text, "", "the bare tags are not left as content");
  ok(A.turns[4].results[0].includes("hint: this ran"), A.turns[4].results[0]);
  eq(stripStrayTags("x\n\n<toolly_call>\n</tool_call>"), "x");
  eq(bareFromText("no calls here", new Map([["read_file", {}]])), { calls: [], text: "no calls here" });
});

Deno.test("hcore agent: a write_file cut by the cap keeps its complete lines; history holds what ran", async () => {
  const log = [];
  const cut = "<tool_call>\n<function=write_file>\n<parameter=path>\ngame.js\n</parameter>\n<parameter=content>\nconst a = 1;\nfunction b() {\n  retu";
  const A = new Agent({ model: core([cut, "ok"]), tools: tools(log), usage: () => ({ reason: "max", generated: 4096, prompt: 10 }) });
  eq((await A.run("go")).reason, "done");
  eq(log, [["write", "game.js", "const a = 1;\nfunction b() {\n"]]);
  eq(A.turns[1].sampled, [{ name: "write_file", args: JSON.stringify({ path: "game.js", content: "const a = 1;\nfunction b() {\n", append: false }) }]);
  ok(/first 2 lines of game\.js[\s\S]*function b\(\) \{[\s\S]*append: true/.test(A.turns[2].results[0]), A.turns[2].results[0]);
});

Deno.test("hcore agent: another cut call is not run; it stays in history with its error as its own result", async () => {
  const log = [];
  const cut = "Fixing.\n<tool_call>\n<function=edit_file>\n<parameter=path>\ngame.js\n</parameter>\n<parameter=old>\nfunction a() {\n  retu";
  const A = new Agent({ model: core([call("read_file", { path: "x" }) + "\n" + cut, "ok"]), tools: tools(log), usage: () => ({ reason: "max", generated: 4096, prompt: 10 }) });
  await A.run("go");
  eq(log, [["read", "x"]], "no truncated edit");
  eq(A.turns[1].sampled.map((c) => c.name), ["read_file", "edit_file"]);
  eq(JSON.parse(A.turns[1].sampled[1].args), { path: "game.js", old: "function a() {\n  retu" });
  eq(A.turns[2].results.length, 2, "one result per call");
  ok(/^error: your answer was cut at 4096 tokens/.test(A.turns[2].results[1]), A.turns[2].results[1]);
  ok(!/Write the call again/.test(A.turns[2].results[1]), "no format advice: the format was not the problem");
  const { messages } = toMessages(A.turns);
  eq(messages.slice(1, 4).map((m) => [m.role, m.calls?.length ?? 0]), [["assistant", 2], ["tool", 0], ["tool", 0]]);
  // ended mid-call for another reason: the reason and the call's tail are named
  const B = new Agent({ model: core([cut, "ok"]), tools: tools(), usage: () => ({ reason: "stop", generated: 30, prompt: 10 }) });
  await B.run("go");
  ok(/ended in the middle of a tool call \(stop after 30 tokens\)\. The call ended with: ".*retu"/.test(B.turns[2].results[0]), B.turns[2].results[0]);
});

// a model object on the core interface, answering with fixed Answers (or throwing)
const fixed = (answers, extra = {}) => {
  let i = 0;
  return { style: "xml", systemText: (s) => s, stats: { last: null }, async ask({ on = () => {} } = {}) { const a = answers[i++] ?? { text: "done", calls: [], open: null, reason: "stop" }; if (a instanceof Error) { on({ t: "text", text: "Start" }); throw a; } if (a.text) on({ t: "text", text: a.text }); return a; }, ...extra };
};

Deno.test("hcore agent: garbage runs nothing; its errors go back as the user's text (no call to pair with)", async () => {
  const log = [];
  const A = new Agent({ model: fixed([{ text: "", calls: [{ name: "write_file", arguments: { path: "a", content: "zz" }, sampled: '{"path": "a", "content": "zz"}' }], open: null, reason: "garbage" }, { text: "ok", calls: [], open: null, reason: "stop" }]), tools: tools(log), usage: () => ({ reason: "garbage", forced: 3, forcedAll: 30, generated: 40 }) });
  await A.run("go");
  eq(log, [], "a garbage write never runs");
  eq(A.turns[1].sampled, []);
  ok(/stopped and its calls were not run: the room's engine is producing garbage/.test(A.turns[2].results[0]), A.turns[2].results[0]);
  eq(toMessages(A.turns).messages.map((m) => m.role), ["user", "assistant", "user", "assistant"]);
});

Deno.test("hcore agent: a model failure is thrown (the run shows an error, never a quiet done), the turn is closed", async () => {
  const A = new Agent({ model: fixed([new Error("a device left"), { text: "fine", calls: [], open: null, reason: "stop" }]), tools: tools() });
  let threw = null;
  try { await A.run("one"); } catch (e) { threw = e.message; }
  eq(threw, "a device left");
  eq(A.turns.map((t) => [t.role, t.text]), [["user", "one"], ["assistant", "Start"]]);
  await A.run("two");
  eq(A.turns.map((t) => t.role), ["user", "assistant", "user", "assistant"]);
});

Deno.test("hcore agent: Stop mid-answer keeps the text, drops the calls that never ran; an abort the room reports counts too", async () => {
  const ac = new AbortController(), log = [];
  const reply = "Let me write it now.\n" + call("write_file", { path: "a.js", content: "x" });
  const A = new Agent({ model: core([reply, "again"], { onPiece: (k, n) => { if (n === 0 && k >= 60) ac.abort(); } }), tools: tools(log) });
  eq((await A.run("go", { signal: ac.signal })).reason, "stopped");
  eq(log, []);
  eq(A.turns[1], { role: "assistant", text: "Let me write it now.", sampled: [], req: 1 });
  eq((await A.run("again")).reason, "done");
  // the room's Stop (ai.abort) ends the answer with reason abort while the run's own signal is untouched
  const B = new Agent({ model: fixed([{ text: "partial", calls: [{ name: "serve", arguments: {}, sampled: "{}" }], open: null, reason: "abort" }]), tools: tools(log) });
  eq((await B.run("go")).reason, "stopped");
  eq(B.turns[1].sampled, []);
});

Deno.test("hcore agent: the same failing edit is warned once, then stopped", async () => {
  const edit = (n) => call("edit_file", { path: "game.js", old: `piece.y = ${n};`, new: "x" });
  const seen = [];
  const A = new Agent({ model: core([edit(1), edit(2), edit(3), edit(4), "done"], { seen }), tools: tools() });
  const r = await A.run("fix");
  eq([r.reason, r.steps], ["stuck", 3]);
  const last = seen[2].turns.at(-1);
  ok(/note: this failed the same way last step/.test(last.results.at(-1)) && last.text.includes("note: this failed"), last.text);
});

Deno.test("hcore agent: served clean and nothing changed: done (idle)", async () => {
  const ev = [];
  const A = new Agent({ model: core([call("write_file", { path: "index.html", content: "<h1>x</h1>" }), call("serve"), call("serve"), "never"]), tools: tools(), onEvent: (e) => ev.push(e) });
  const r = await A.run("page");
  eq([r.reason, r.steps, r.text], ["done", 3, "Done: the page is served with no errors."]);
  ok(ev.some((e) => e.type === "done" && e.idle));
  eq(A.turns.at(-2).results.length, 1, "the idle step's results are a results turn too");
});

Deno.test("hcore agent: compaction stubs results arrays (text kept in step); sizes count the calls as the template writes them", async () => {
  const replies = ["a", "b", "c", "d"].map((p) => call("read_file", { path: p })).concat(["done"]);
  const ts = tools();
  ts[0].run = async (a) => a.path.repeat(1000);
  const ev = [];
  let budget = Infinity;
  const gen = scripted(replies);
  const A = new Agent({ model: scriptedCore(async function* (o) { if (A.turns.length === 7) budget = A._size() + 30; yield* gen(o); }, { style: "xml", count: (t) => t.length }), tools: ts, budget: () => budget, onEvent: (e) => ev.push(e) });
  await A.run("read four files");
  eq(ev.find((e) => e.type === "compacted")?.tier, 1);
  const res = A.turns.filter((t) => t.results);
  eq(res[0].results, ["(output of read_file a dropped; run it again if needed)"]);
  eq(res[0].text, toolResponses(res[0].results));
  ok(res[2].results[0].startsWith("ccc") && res[3].results[0].startsWith("ddd"));
  eq(stubResultList(["short", "y".repeat(300)], ["a", "b"]), ["short", "(output of b dropped; run it again if needed)"]);
  // a call's size is its rendered body (an XML value unescaped), not the JSON text
  const t = { role: "assistant", text: "", sampled: [{ name: "write_file", args: JSON.stringify({ path: "a", content: "l1\n\"q\"\n" }) }] };
  eq(A._turnSize(t), A.count("") + A.count("\n<function=write_file>\n<parameter=path>\na\n</parameter>\n<parameter=content>\nl1\n\"q\"\n\n</parameter>\n</function>\n") + 4);
});

Deno.test("hcore agent: sessions keep each answer's exact ids under the same tokenizer only", async () => {
  const store = new WeakMap();
  const model = (tag) => ({ ...fixed([{ text: "hi", calls: [], open: null, reason: "stop", ids: { ids: [5, 6], thinkEnd: 2 } }]),
    setIds: (t, v) => store.set(t, v), idsOf: (t) => store.get(t) || null, idsTag: () => tag, adoptIds: (t, v) => store.set(t, v) });
  const A = new Agent({ model: model("core:a"), tools: tools() });
  await A.run("q");
  const json = JSON.parse(JSON.stringify(A.toJSON()));
  eq([json.tok, json.turns[1].ids, json.turns[1].idsEnd], ["core:a", [5, 6], 2]);
  const B = Agent.from(json, { model: model("core:a"), tools: tools() });
  eq(store.get(B.turns[1]), { ids: [5, 6], thinkEnd: 2 });
  const C = Agent.from(json, { model: model("core:b"), tools: tools() });
  eq(store.get(C.turns[1]) ?? null, null, "another tokenizer re-renders from the text");
  eq(C.turns.map((t) => t.text), ["q", "hi"]);
});

Deno.test("hcore agent: a session switches paths both ways (hcore -> legacy -> hcore) and every call stays paired with its result", async () => {
  const log = [];
  const A = new Agent({ model: core([call("read_file", { path: "a.js" }), "read it"]), tools: tools(log) });
  await A.run("one");
  // legacy renders the core turns with their calls as markup, in the model's format
  const seen = [];
  const L = Agent.from(JSON.parse(JSON.stringify(A.toJSON())), { generate: scripted([call("write_file", { path: "b.js", content: "y" }), "wrote it"], { seen }), tools: tools(log), style: "xml" });
  await L.run("two");
  eq(seen[0].turns[1].text, legacyText(A.turns[1], "xml"));
  ok(seen[0].turns[1].text.includes("<function=read_file>") && seen[0].turns[1].text.includes("a.js"), seen[0].turns[1].text);
  // and back: the legacy raw turn is parsed into its call, the results split per call
  const seen2 = [];
  const H = Agent.from(JSON.parse(JSON.stringify(L.toJSON())), { model: core(["three done"], { seen: seen2 }), tools: tools(log) });
  await H.run("three");
  const { messages } = toMessages(seen2[0].turns, { style: "xml", tools: tools() });
  eq(messages.map((m) => m.role), ["user", "assistant", "tool", "assistant", "user", "assistant", "tool", "assistant", "user"]);
  eq(messages[5].calls, [{ name: "write_file", args: { path: "b.js", content: "y" } }]);
  eq(log.map((x) => x[0]), ["read", "write"]);
});

Deno.test("hcore agent: live call text is append-only JSON-shaped (code-ui parseLive) and ends with null", async () => {
  const ev = [];
  const A = new Agent({ model: core([call("write_file", { path: "a.js", content: "line1\nline2" }), "ok"]), tools: tools(), onEvent: (e) => { if (e.type === "call-live") ev.push(e.raw); } });
  await A.run("go");
  const live = ev.filter((x) => x != null);
  ok(live.length > 1, "streams");
  for (let i = 1; i < live.length; i++) ok(live[i].startsWith(live[i - 1]), "append-only");
  eq(live.at(-1), '{"name": "write_file", "arguments": {"path": "a.js", "content": "line1\\nline2"}');
  eq(ev.at(-1), null);
  // a real codingTools workspace, end to end
  const ws = new MemoryWorkspace({});
  const B = new Agent({ model: core([call("write_file", { path: "x.txt", content: "hello" }), "ok"]), tools: codingTools(ws) });
  await B.run("go");
  eq(await ws.read("x.txt"), "hello");
});
