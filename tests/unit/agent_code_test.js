// harness/agent.js for Code mode: cancellation, approval with a reason, the result cap,
// compaction tiers, and saving/restoring a session (docs/design/harness-app.md A.4, F.3).
import { Agent, capResult, CONTEXT_FULL } from "../../harness/agent.js";
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

const call = (name, args = {}) => `<tool_call>\n<function=${name}>\n` + Object.entries(args).map(([k, v]) => `<parameter=${k}>\n${v}\n</parameter>\n`).join("") + "</function>\n</tool_call>";
// each call yields the next reply in pieces; onPiece(k, callIndex) runs before piece k
function scripted(replies, { seen = [], onPiece } = {}) {
  let i = 0;
  return async function* ({ turns, signal }) {
    seen.push(turns.map((t) => ({ role: t.role, text: t.text })));
    const n = i++, r = replies[n] ?? "done";
    for (let k = 0; k < r.length; k += 5) {
      onPiece?.(k, n);
      if (signal?.aborted) return;
      yield r.slice(k, k + 5);
      await null;
    }
  };
}
const tools = (log = []) => [
  { name: "read_file", description: "read", parameters: { type: "object", properties: { path: { type: "string" } } }, mutates: false,
    run: async (a, ctx) => { log.push(["read", a.path, ctx.step, !!ctx.signal]); return a.path === "big" ? "x".repeat(10000) : `contents of ${a.path}`; } },
  { name: "write_file", description: "write", parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } } }, mutates: true,
    preview: async (a) => ({ path: a.path, before: null, after: a.content }),
    run: async (a) => { log.push(["write", a.path]); return `wrote ${a.path}`; } },
  { name: "serve", description: "serve", parameters: { type: "object", properties: { port: { type: "integer" } } }, mutates: false, run: async () => "serving on :5173" },
];

Deno.test("agent: abort mid-generation keeps the partial text and runs no tool", async () => {
  const ac = new AbortController(), log = [], events = [];
  const reply = "Let me write it now.\n" + call("write_file", { path: "a.js", content: "x" });
  const A = new Agent({ generate: scripted([reply], { onPiece: (k) => { if (k >= 30) ac.abort(); } }), tools: tools(log), onEvent: (e) => events.push(e.type) });
  const r = await A.run("go", { signal: ac.signal });
  eq(r.reason, "stopped");
  eq(log, [], "no tool ran");
  eq(A.turns.map((t) => t.role), ["user", "assistant"]);
  ok(A.turns[1].text.startsWith("Let me write"), A.turns[1].text);
  ok(events.includes("stopped"));
  // the next request follows an assistant turn
  const B = await A.run("again");
  eq(B.reason, "done");
  eq(A.turns.map((t) => t.role), ["user", "assistant", "user", "assistant"]);
});

Deno.test("agent: abort before any text takes the request back", async () => {
  const ac = new AbortController();
  const A = new Agent({ generate: scripted(["hello"], { onPiece: () => ac.abort() }), tools: tools() });
  const r = await A.run("go", { signal: ac.signal });
  eq(r.reason, "stopped");
  eq(A.turns, []);
  ac.abort();
  const A2 = new Agent({ generate: scripted(["x"]), tools: tools() });
  eq((await A2.run("go", { signal: ac.signal })).reason, "stopped");
  eq(A2.turns, [], "an already aborted signal never reaches the model");
});

Deno.test("agent: abort during tools closes the step with an assistant turn", async () => {
  const ac = new AbortController();
  const ts = tools();
  ts[0].run = async () => { ac.abort(); return "read"; };
  const A = new Agent({ generate: scripted([call("read_file", { path: "a" }) + call("read_file", { path: "b" })]), tools: ts });
  const r = await A.run("go", { signal: ac.signal });
  eq(r.reason, "stopped");
  eq(A.turns.map((t) => t.role), ["user", "assistant", "user", "assistant"]);
  ok(A.turns[2].text.includes("(stopped by the user)"), "the second call did not run");
});

Deno.test("agent: tool-start comes before approval; approve gets the preview; a reason goes back", async () => {
  const order = [], seen = [], log = [];
  const A = new Agent({
    generate: scripted([call("write_file", { path: "a.js", content: "hi" }), "ok"], { seen }), tools: tools(log),
    onEvent: (e) => { if (e.type === "tool-start" || e.type === "tool") order.push(e.type); },
    approve: async (c, info) => { order.push("approve"); eq(info, { path: "a.js", before: null, after: "hi" }); return { ok: false, reason: "use b.js" }; },
  });
  await A.run("write");
  eq(order, ["tool-start", "approve", "tool"]);
  eq(log, []);
  ok(seen[1][2].text.includes("declined by the user: use b.js"), seen[1][2].text);
});

Deno.test("agent: tools get { signal, step }; results are capped head and tail; events carry ms", async () => {
  const log = [], ev = [];
  const A = new Agent({ generate: scripted([call("read_file", { path: "big" }), "ok"]), tools: tools(log), maxResultChars: 1000, onEvent: (e) => { if (e.type === "tool") ev.push(e); } });
  await A.run("read");
  eq(log, [["read", "big", 1, false]]);
  ok(ev[0].result.length < 1100 && ev[0].result.includes("(9000 chars cut)"), ev[0].result.slice(0, 80));
  ok(Number.isFinite(ev[0].ms));
  eq(capResult("abcdef", 10), "abcdef");
});

Deno.test("agent: usage after each step; ContextFull from the model ends the run", async () => {
  const ev = [];
  const A = new Agent({ generate: scripted(["hi"]), tools: tools(), usage: () => ({ prompt: 10, reused: 4, generated: 1, tps: 3 }), onEvent: (e) => ev.push(e) });
  await A.run("x");
  eq(ev.find((e) => e.type === "usage"), { type: "usage", step: 1, prompt: 10, reused: 4, generated: 1, tps: 3, forced: 0 });
  const gen = async function* () { const e = new Error("full"); e.name = "ContextFull"; throw e; };
  const B = new Agent({ generate: gen, tools: tools() });
  const r = await B.run("y");
  eq(r, { text: CONTEXT_FULL, steps: 1, calls: 0, reason: "context" });
  eq(B.turns, [], "the request was taken back");
});

// count = characters (exact sizes); the system prompt counts too
const sized = (A) => A.count(A.system) + A.turns.reduce((n, t) => n + A.count(t.text) + 4, 0);

Deno.test("agent compaction 1: old tool results become stubs naming the call; the last two stay", async () => {
  const replies = ["a", "b", "c", "d"].map((p) => call("read_file", { path: p })).concat(["done"]);
  const ts = tools();
  ts[0].run = async (a) => a.path.repeat(1000);
  const seen = [], ev = [];
  let budget = Infinity;
  const A = new Agent({ generate: scripted(replies, { seen, onPiece: () => {} }), tools: ts, count: (t) => t.length, budget: () => budget, onEvent: (e) => ev.push(e) });
  // over budget from the 5th step on: four results in the conversation, two of them old
  A.generate = ((g) => async function* (o) { if (o.turns.length === 7) budget = sized(A) + 30; yield* g(o); })(A.generate);
  await A.run("read four files");
  const c = ev.find((e) => e.type === "compacted");
  eq(c.tier, 1);
  const res = A.turns.filter((t) => t.calls);
  ok(res[0].text.includes("(output of read_file a dropped; run it again if needed)"), res[0].text);
  ok(res[1].text.includes("(output of read_file b dropped"), res[1].text);
  ok(c.after < c.before - 1800, JSON.stringify(c));
  ok(res[2].text.includes("ccc") && res[3].text.includes("ddd"), "the last two steps keep their output");
});

Deno.test("agent compaction 2-3: earlier requests fold into one line, then drop; 4: context full", async () => {
  const ts = tools();
  const ev = [];
  let budget = Infinity;
  const replies = [call("write_file", { path: "index.html", content: "x".repeat(900) }), call("serve", { port: 5173 }), "Built it: " + "y".repeat(300),
    call("write_file", { path: "game.js", content: "z".repeat(900) }), "Added the game.", "third answer"];
  const A = new Agent({ generate: scripted(replies), tools: ts, count: (t) => t.length, budget: () => budget, onEvent: (e) => ev.push(e) });
  await A.run("build a page");
  await A.run("add a game");
  const size2 = sized(A);
  // tier 2: fold (tool outputs are short here, so tier 1 frees little)
  budget = size2 - 10;
  await A.run("third");
  const c = ev.filter((e) => e.type === "compacted");
  eq(c[0].tier, 2, JSON.stringify(c));
  const fold = A.turns.find((t) => t.folded);
  ok(fold.text.startsWith("[earlier: wrote index.html; served :5173]\nBuilt it: yyy"), fold.text);
  eq(A.turns[0], { role: "user", text: "build a page", req: 1 });
  // tier 3: drop whole earlier requests, never the current one
  budget = A.count(A.system) + 60;
  A.generate = scripted(["fourth answer"]);
  const r = await A.run("fourth");
  eq(r.reason, "done");
  eq(A.turns.map((t) => t.text), ["fourth", "fourth answer"]);
  ok(ev.some((e) => e.type === "compacted" && e.tier === 3));
  // tier 4: the current request alone does not fit
  budget = A.count(A.system) + 5;
  const r5 = await A.run("a request that is far too long for what is left");
  eq(r5.reason, "context");
  eq(A.turns.map((t) => t.text), [], "taken back; earlier requests had to go too");
});

Deno.test("agent: toJSON / from round trip keeps turns, request state and sampled ids", async () => {
  const ids = new Map();
  const A = new Agent({ generate: scripted([call("read_file", { path: "a" }), "done"]), tools: tools(), idsFor: (t) => ids.get(t) });
  await A.run("go");
  ids.set(A.turns[1].text, [1, 2, 3]);
  const json = JSON.parse(JSON.stringify(A.toJSON()));
  eq(json.v, 1);
  eq(json.turns[1].ids, [1, 2, 3]);
  const adopted = [];
  const B = Agent.from(json, { generate: scripted(["again"]), tools: tools(), adopt: (t, i) => adopted.push([t, i]) });
  eq(adopted, [[A.turns[1].text, [1, 2, 3]]]);
  eq(B.turns, A.turns);
  eq(B.reqs, A.reqs);
  await B.run("more");
  eq(B.turns[B.turns.length - 2].req, 2, "request numbering continues");
  B.reset();
  eq(B.turns, []);
});

Deno.test("agent: a call cut by the answer cap is not run; the model is told why", async () => {
  const log = [];
  const cut = "<tool_call>\n<function=edit_file>\n<parameter=path>\ngame.js\n</parameter>\n<parameter=old>\nfunction a() {\n  retu";
  const A = new Agent({ generate: scripted([cut, "ok"]), tools: tools(log), usage: () => ({ reason: "max", generated: 4096, prompt: 10 }) });
  const r = await A.run("go");
  eq(r.reason, "done");
  eq(log, [], "no truncated edit");
  ok(/cut at 4096 tokens[\s\S]*hint: Write long files in parts[\s\S]*append: true/.test(A.turns[2].text), A.turns[2].text);
});

Deno.test("agent: a write_file cut by the answer cap keeps its complete lines and says where to continue", async () => {
  const log = [];
  const cut = "<tool_call>\n<function=write_file>\n<parameter=path>\ngame.js\n</parameter>\n<parameter=content>\nconst a = 1;\nfunction b() {\n  retu";
  const A = new Agent({ generate: scripted([cut, "ok"]), tools: tools(log), usage: () => ({ reason: "max", generated: 4096, prompt: 10 }) });
  const r = await A.run("go");
  eq(r.reason, "done");
  eq(log, [["write", "game.js"]], "the complete lines were written");
  ok(/first 2 lines of game\.js[\s\S]*function b\(\) \{[\s\S]*append: true/.test(A.turns[2].text), A.turns[2].text);
});

Deno.test("agent: stopped requests fold later instead of filling the context for good", async () => {
  const ac = new AbortController(), ev = [];
  let budget = Infinity;
  const A = new Agent({ generate: scripted(["y".repeat(900), "second"], { onPiece: (k, n) => { if (n === 0 && k >= 800) ac.abort(); } }), tools: tools(),
    count: (t) => t.length, budget: () => budget, onEvent: (e) => ev.push(e) });
  eq((await A.run("first", { signal: ac.signal })).reason, "stopped");
  budget = A.count(A.system) + 400;
  const r = await A.run("next");
  eq(r.reason, "done", JSON.stringify(ev.filter((e) => e.type === "compacted")));
  ok(!A.turns.some((t) => !t.folded && t.text.startsWith("yyyy")), "the stopped request was folded or dropped");
  ok(ev.some((e) => e.type === "compacted" && (e.tier === 2 || e.tier === 3)));
});

Deno.test("agent: a model error or the step limit never leaves two user turns in a row", async () => {
  let n = 0;
  const failing = async function* () { if (n++ === 0) { yield "Starting"; throw new Error("a device left"); } yield "fine"; };
  const A = new Agent({ generate: failing, tools: tools() });
  let threw = null;
  try { await A.run("one"); } catch (e) { threw = e.message; }
  eq(threw, "a device left");
  await A.run("two");
  eq(A.turns.map((t) => t.role), ["user", "assistant", "user", "assistant"]);
  const B = new Agent({ generate: scripted([call("read_file", { path: "a" }), "after"]), tools: tools(), maxSteps: 1 });
  eq((await B.run("loop")).reason, "limit");
  eq(B.turns[B.turns.length - 1].role, "assistant");
  await B.run("next");
  const roles = B.turns.map((t) => t.role);
  ok(roles.every((r, i) => i === 0 || !(r === "user" && roles[i - 1] === "user")), roles.join(","));
});

Deno.test("agent: saved ids are replayed only under the same tokenizer", async () => {
  const A = new Agent({ generate: scripted(["hello"]), tools: tools(), idsFor: () => [7, 8], idsTag: () => "qwen:151k" });
  await A.run("hi");
  const json = A.toJSON();
  eq(json.tok, "qwen:151k");
  const same = [], other = [];
  Agent.from(json, { generate: scripted([]), tools: tools(), adopt: (t, i) => same.push(i), idsTag: () => "qwen:151k" });
  Agent.from(json, { generate: scripted([]), tools: tools(), adopt: (t, i) => other.push(i), idsTag: () => "llama:128k" });
  eq(same, [[7, 8]]);
  eq(other, [], "another model re-encodes the text");
});

Deno.test("agent: a write_file that ends early for any reason keeps its lines; the error names the reason", async () => {
  const log = [];
  const cut = "<tool_call>\n<function=write_file>\n<parameter=path>\ngame.js\n</parameter>\n<parameter=content>\nconst a = 1;\nlet b";
  const A = new Agent({ generate: scripted([cut, "ok"]), tools: tools(log), usage: () => ({ reason: "stop", generated: 700, prompt: 10 }) });
  await A.run("go");
  eq(log, [["write", "game.js"]]);
  const B = new Agent({ generate: scripted(["<tool_call>\n<function=edit_file>\n<parameter=path>\nx\n</parameter>\n<parameter=old>\nab", "ok"]), tools: tools([]), usage: () => ({ reason: "stop", generated: 12, prompt: 10 }) });
  await B.run("go");
  ok(/ended in the middle of a tool call \(stop after 12 tokens\)/.test(B.turns[2].text), B.turns[2].text);
  // many positions forced by the call grammar: say the engine may be the problem
  const C = new Agent({ generate: scripted(["<tool_call>\n<function=edit_file>\n<parameter=path>\nx", "ok"]), tools: tools([]), usage: () => ({ reason: "stop", generated: 30, prompt: 10, forced: 20 }) });
  await C.run("go");
  ok(/20 tokens were forced by the call format/.test(C.turns[2].text), C.turns[2].text);
});

Deno.test("agent: the same failing call three steps in a row stops the run as stuck", async () => {
  const bad = "<tool_call>\n<function=write_file>\n<parameter=path>";
  const A = new Agent({ generate: scripted([bad, bad, bad, bad, bad, "ok"]), tools: tools([]), usage: () => ({ reason: "stop", generated: 207, prompt: 10 }) });
  const r = await A.run("go");
  eq(r.reason, "stuck");
  eq(r.steps, 3);
});

Deno.test("agent: a <function=> call with a garbled <tool_call> opener still runs", async () => {
  const log = [];
  // what Qwen 3.6 wrote on a follow-up request when its earlier calls had been re-tokenized as text
  const garbled = "<tool_tool_calls>\n<function=write_file>\n<parameter=path>\nstyle.css\n</parameter>\n<parameter=content>\nbody { background: #e8e8f0; }\n</parameter>\n</function>\n</tool_call>";
  const A = new Agent({ generate: scripted([garbled, "done"]), tools: tools(log) });
  const r = await A.run("use lighter colors");
  eq(r.reason, "done");
  eq(log, [["write", "style.css"]]);
  ok(/hint: this ran/.test(A.turns[2].text), A.turns[2].text);
});
Deno.test("agent: a known tool written as bare tags (no <tool_call>) runs", async () => {
  const log = [];
  const bare = "<write_file>\n<path>index.html</path>\n<content>\n<!DOCTYPE html>\n<h1>hello</h1>\n</content>\n</write_file>";
  const A = new Agent({ generate: scripted([bare, "done"]), tools: tools(log) });
  const r = await A.run("go");
  eq(r.reason, "done");
  eq(log, [["write", "index.html"]]);
  ok(/hint: this ran/.test(A.turns[2].text), A.turns[2].text);
});
