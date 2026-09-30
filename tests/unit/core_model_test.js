// harness/core-model.js: Code mode on the serve v2 core, in process (?hcore=1). toMessages (Agent
// turns -> v2 messages, legacy raw turns included), the request it builds (what validateApiAsk would
// accept), ask() over a byte-level tokenizer with a scripted generate (events, calls, open calls,
// exact ids that the next step replays, the pin, garbage, errors, ContextFull, abort), and recorded
// model outputs (tests/fixtures/api) replayed through it, the malformed and cut ones included.
import { coreModel, coreRequest, toMessages, parseLegacyAnswer, turnResults, scriptedCore } from "../../harness/core-model.js";
import { validateApiAsk } from "../../room/api.js";
import { templateProfile, renderApi } from "../../room/conversation.js";
import { makeTokenizer } from "../../engine/tokenizer.js";
import { toolResponses, pyJSON } from "../../harness/tools.js";
import { scripted, xmlCall } from "../scripted-model.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

const FX = new URL("../fixtures/api/", import.meta.url);
const readFx = (name) => JSON.parse(Deno.readTextFileSync(new URL(name, FX)));
const TPL = { json: Deno.readTextFileSync(new URL("templates/qwen3-1.7b.jinja", FX)), xml: Deno.readTextFileSync(new URL("templates/qwen3.6-35b-moe.jinja", FX)) };
// a byte-level tokenizer.json (no merges) with Qwen's specials as added tokens (as api_host_test)
function qwenByteTok() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const vocab = {};
  cs.forEach((c, i) => { vocab[String.fromCharCode(c)] = i; });
  const added = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<tool_call>", "</tool_call>", "<think>", "</think>", "<tool_response>", "</tool_response>"].map((content, k) => ({ id: 256 + k, content, special: true }));
  return makeTokenizer({ model: { vocab, merges: [] }, added_tokens: added });
}
const BT = qwenByteTok();
const NV = 265;
const TAG_RE = /<tool_call>|<\/tool_call>|<think>|<\/think>|<tool_response>|<\/tool_response>|<\|im_end\|>/g;
// an answer written as text, the tags as their special ids (as a model samples them)
function answerIds(text) {
  const out = [];
  let at = 0;
  for (const m of text.matchAll(TAG_RE)) { out.push(...BT.encode(text.slice(at, m.index)), BT.vocab[m[0]]); at = m.index + m[0].length; }
  out.push(...BT.encode(text.slice(at)));
  return out;
}
// a host whose model prefers the next scripted id at each step (the grammar may overrule it), then
// the end token; seen: every call's { ids, pin, pinTag, maxNew }
function fakeHost(style, scripts, { maxSeq = 1 << 16, fail = null, nanFrom = -1 } = {}) {
  const seen = [];
  let k = 0;
  const host = {
    seen, overruled: 0,
    tok: () => BT, chatTemplate: () => TPL[style], maxSeq: () => maxSeq,
    async generate(ids, { onToken, stop, maxNew, sample, signal, pin, pinTag }) {
      seen.push({ ids: ids.slice(), pin, pinTag, maxNew });
      if (fail) throw new Error(fail);
      const want = [...(scripts[k++] ?? []), BT.vocab["<|im_end|>"]];
      let n = 0;
      for (const w of want) {
        if (signal?.aborted) return { reason: "abort", reused: 0 };
        if (n >= maxNew) return { reason: "max", reused: 0 };
        const lg = new Float32Array(NV).fill(0);
        if (nanFrom >= 0 && n >= nanFrom) lg.fill(NaN); else lg[w] = 10;
        const t = sample(lg);
        if (t !== w) host.overruled++;
        if (stop.has(t)) return { reason: "stop", reused: 0, prefilled: ids.length, tps: 7 };
        n++; onToken(t, 0);
      }
      return { reason: "stop", reused: 0, prefilled: ids.length, tps: 7 };
    },
  };
  return host;
}
const TOOLS = [
  { name: "read_file", description: "Read a file.", parameters: { type: "object", properties: { path: { type: "string" }, start_line: { type: "integer" } }, required: ["path"] } },
  { name: "write_file", description: "Write a file.", parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" }, append: { type: "boolean" } }, required: ["path", "content"] } },
  { name: "serve", description: "Serve.", parameters: { type: "object", properties: { port: { type: "integer" } } } },
];
// (the JSON layout the template shows and the grammar holds a call to: ", " and ": ")
const JCALL = (name, args) => `<tool_call>\n{"name": "${name}", "arguments": ${pyJSON(args)}}\n</tool_call>`;
const sorted = (v) => (Array.isArray(v) ? v.map(sorted) : v && typeof v === "object" ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, sorted(v[k])])) : v);
const XCALL = (name, args) => `<tool_call>\n<function=${name}>\n${Object.entries(args).map(([k, v]) => `<parameter=${k}>\n${typeof v === "string" ? v : JSON.stringify(v)}\n</parameter>\n`).join("")}</function>\n</tool_call>`;
const events = () => { const ev = []; return { ev, on: (e) => ev.push(e) }; };

// ---- toMessages ----

Deno.test("core: toMessages maps every turn kind, pairs results with calls, merges user turns", () => {
  const turns = [
    { role: "user", text: "build it", req: 1 },
    { role: "assistant", text: "Reading.", sampled: [{ name: "read_file", args: '{"path": "a.js"}' }, { name: "serve", args: "{}" }], req: 1 },
    { role: "user", text: toolResponses(["1|x", "serving"]), results: ["1|x", "serving"], calls: ["read_file a.js", "serve"], req: 1 },
    { role: "assistant", text: "", sampled: [], req: 1 },
    { role: "user", text: "(empty answer: call a tool or say you are done)", req: 1 },
    { role: "assistant", text: "Done.", req: 1 },
    { role: "user", text: "more", req: 2 },
    { role: "assistant", text: "(stopped by the user)", req: 2 },
    { role: "user", text: "again", req: 3 },
    { role: "user", text: "(and this)", req: 3 },
  ];
  const { messages, src } = toMessages(turns);
  eq(messages, [
    { role: "user", text: "build it" },
    { role: "assistant", text: "Reading.", calls: [{ name: "read_file", args: { path: "a.js" } }, { name: "serve", args: {} }] },
    { role: "tool", text: "1|x" }, { role: "tool", text: "serving" },
    { role: "assistant", text: "", calls: [] },
    { role: "user", text: "(empty answer: call a tool or say you are done)" },
    { role: "assistant", text: "Done." },
    { role: "user", text: "more" },
    { role: "assistant", text: "(stopped by the user)" },
    { role: "user", text: "again\n\n(and this)" },
  ]);
  eq(src[1], turns[1], "an assistant message knows its turn (for its exact ids)");
  // the request passes the host's own validation (the adapter builds it already normalized)
  const req = coreRequest({ system: "S", tools: TOOLS, messages });
  const v = validateApiAsk(req, { profile: templateProfile(TPL.xml, BT) });
  ok(!v.err, v.err);
  eq(sorted(v.req), sorted(req), "coreRequest is exactly what validateApiAsk normalizes it to");
});

Deno.test("core: results after an answer with no calls go as the user's text (never a tool result without a call)", () => {
  const { messages } = toMessages([
    { role: "user", text: "q" },
    { role: "assistant", text: "garbage happened", sampled: [] },
    { role: "user", text: toolResponses(["error: stopped"]), results: ["error: stopped"] },
  ]);
  eq(messages.map((m) => m.role), ["user", "assistant", "user"]);
  eq(messages[2].text, "<tool_response>\nerror: stopped\n</tool_response>");
});

Deno.test("core: legacy raw turns (sessions saved by the old path) convert: XML, JSON, joined results, a cut call, unparseable markup", () => {
  // Qwen3.6 as the legacy path kept it (baseline trajectory, add-feature)
  const xml = "Now I understand the app.\n\n<tool_call>\n<function=edit_file>\n<parameter=path>\napp.js\n</parameter>\n<parameter=old>\na\n</parameter>\n<parameter=new>\nb\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=serve>\n</function>\n</tool_call>";
  const json = "Let me write it.\n<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"index.html\", \"content\": \"<h1>x</h1>\\n\"}}\n</tool_call>";
  const cut = "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"a.js\", \"content\": \"line1\\nli";
  const { messages } = toMessages([
    { role: "user", text: "q" },
    { role: "assistant", text: xml },
    { role: "user", text: toolResponses(["edited app.js", "serving :5173"]) },
    { role: "assistant", text: json },
    { role: "user", text: toolResponses(["wrote index.html"]) },
    { role: "assistant", text: cut },
    { role: "user", text: toolResponses(["error: your answer was cut at 12 tokens before the call was complete"]) },
    { role: "assistant", text: "I wrote <tool_call> garbage </tool_call>" },
  ], { style: "xml", tools: [] });
  eq(messages[1], { role: "assistant", text: "Now I understand the app.", calls: [{ name: "edit_file", args: { path: "app.js", old: "a", new: "b" } }, { name: "serve", args: {} }] });
  eq(messages.slice(2, 4).map((m) => [m.role, m.text]), [["tool", "edited app.js"], ["tool", "serving :5173"]]);
  eq(messages[4], { role: "assistant", text: "Let me write it.", calls: [{ name: "write_file", args: { path: "index.html", content: "<h1>x</h1>\n" } }] });
  eq(messages[6], { role: "assistant", text: "", calls: [{ name: "write_file", args: { path: "a.js", content: "line1\nli" } }] }, "a cut call keeps what it had");
  eq(messages[7].role, "tool");
  eq(messages[8], { role: "assistant", text: "I wrote <tool_call> garbage </tool_call>", calls: [] }, "markup that is no call stays text");
  eq(parseLegacyAnswer(xml, "xml"), parseLegacyAnswer(xml, "xml"), "memoized, same result");
  eq(turnResults({ role: "user", text: toolResponses(["a", "b\nc"]) }), ["a", "b\nc"]);
  eq(turnResults({ role: "user", text: "plain" }), null);
});

// ---- ask() ----

for (const style of ["json", "xml"]) {
  Deno.test(`core ${style}: ask streams content and calls, returns the sampled args, pins the system turn`, async () => {
    const CALL = style === "json" ? JCALL : XCALL;
    const host = fakeHost(style, [answerIds(`I'll read it.\n\n${CALL("read_file", { path: "a.js" })}\n${CALL("serve", {})}`)]);
    const m = coreModel(host, { maxNew: 500 });
    eq(m.style, style);
    const { ev, on } = events();
    const turns = [{ role: "user", text: "fix a.js" }];
    const res = await m.ask({ system: "You are Tabby.", tools: TOOLS, turns, on });
    eq(host.overruled, 0, "a well-formed answer is the model's own");
    eq(res.text, "I'll read it.");
    eq(res.calls.map((c) => [c.name, c.arguments]), [["read_file", { path: "a.js" }], ["serve", {}]]);
    eq(res.calls.map((c) => JSON.parse(c.sampled)), [{ path: "a.js" }, {}]);
    eq([res.reason, res.open], ["stop", null]);
    ok(res.ids?.ids.length > 0, "exact ids come back");
    // events: content, then per call its name, argument fragments and end
    eq(ev.filter((e) => e.t === "text").map((e) => e.text).join(""), "I'll read it.");
    eq(ev.filter((e) => e.t === "call").map((e) => [e.i, e.name]), [[0, "read_file"], [1, "serve"]]);
    eq(ev.filter((e) => e.t === "args" && e.i === 0).map((e) => e.a).join(""), res.calls[0].sampled, "fragments join to the sampled args");
    eq(ev.filter((e) => e.t === "end").map((e) => e.i), [0, 1]);
    ok(ev.some((e) => e.t === "delta"), "raw pieces for the delta event");
    // the pin: the system turn, tagged as Code's
    const req = coreRequest({ system: "You are Tabby.", tools: TOOLS, messages: toMessages(turns).messages });
    const sys = renderApi(BT, req, templateProfile(TPL[style], BT)).systemLen;
    eq([host.seen[0].pin, host.seen[0].pinTag], [sys, "code"]);
    ok(host.seen[0].maxNew <= 500, "maxNew " + host.seen[0].maxNew);
    // the system text the template writes, with Code's system prompt in it
    ok(m.systemText("You are Tabby.", TOOLS).includes("You are Tabby.") && m.systemText("You are Tabby.", TOOLS).includes("<tools>"));
    eq([m.stats.calls, m.stats.last.reason, m.stats.last.prompt, m.stats.last.tps], [1, "stop", host.seen[0].ids.length, 7]);
  });

  Deno.test(`core ${style}: the next step's prompt extends this one's with the exact sampled ids (and after the ids are dropped, still renders)`, async () => {
    const CALL = style === "json" ? JCALL : XCALL;
    const a1 = answerIds(`Reading.\n${CALL("read_file", { path: "a.js" })}`);
    const host = fakeHost(style, [a1, answerIds("Done.")]);
    const m = coreModel(host);
    const turns = [{ role: "user", text: "q" }];
    const r1 = await m.ask({ system: "S", tools: TOOLS, turns });
    const t = { role: "assistant", text: r1.text, sampled: r1.calls.map((c) => ({ name: c.name, args: c.sampled })) };
    m.setIds(t, r1.ids);
    turns.push(t, { role: "user", text: toolResponses(["1|x"]), results: ["1|x"] });
    const r2 = await m.ask({ system: "S", tools: TOOLS, turns });
    eq(r2.text, "Done.");
    const p1 = host.seen[0].ids, p2 = host.seen[1].ids;
    eq(p2.slice(0, p1.length), p1, "step 2 starts with step 1's prompt");
    eq(p2.slice(p1.length, p1.length + a1.length), a1, "then exactly the ids step 1 sampled");
    eq(p2[p1.length + a1.length], BT.vocab["<|im_end|>"]);
    eq(m.stats.last.exact, 1, "one assistant turn replayed from its ids");
    // a changed model (a re-deal) drops them: the turn renders from its text and calls
    const m2 = coreModel(host);
    eq(m2.idsOf(t), null);
  });
}

Deno.test("core: a call cut by the length cap is open, with the arguments it had (recorded Qwen3.6 and 1.7B outputs)", async () => {
  for (const [name, style] of [["qwen3.6-35b-moe-truncated", "xml"], ["qwen3-1.7b-truncated", "json"]]) {
    const fx = readFx(name + ".json");
    const ids = answerIds(fx.texts.filter((t) => t !== "<|im_end|>").join(""));
    const host = fakeHost(style, [ids]);
    const m = coreModel(host, { maxNew: ids.length - 5 });
    const res = await m.ask({ system: "", tools: fx.req.tools, turns: [{ role: "user", text: "q" }] });
    eq([res.reason, res.calls.length, res.open?.name], ["max", 0, "write_file"], name);
    ok(res.open.args.startsWith("{") && res.open.args.includes('"path"'), name + ": " + res.open.args.slice(0, 60));
    eq(res.ids, null, "a cut answer has no exact ids");
  }
});

Deno.test("core: recorded well-formed answers replay through the grammar unchanged (Qwen3 1.7B, Qwen3.6, Qwen3.5 2B)", async () => {
  for (const name of ["qwen3-1.7b-write", "qwen3-1.7b-parallel", "qwen3-1.7b-text_then_call", "qwen3.6-35b-moe-write", "qwen3.6-35b-moe-parallel", "qwen3.6-35b-moe-text_then_call", "qwen3.5-2b-write"]) {
    const fx = readFx(name + ".json");
    const style = fx.profile.style;
    const host = fakeHost(style, [answerIds(fx.texts.filter((t) => t !== "<|im_end|>").join(""))]);
    const m = coreModel(host);
    const res = await m.ask({ system: fx.req.system || "", tools: fx.req.tools, turns: [{ role: "user", text: "q" }] });
    eq(host.overruled, 0, name + ": every token allowed");
    ok(res.calls.length >= 1 && res.open === null && res.reason === "stop", name + " " + JSON.stringify(res).slice(0, 200));
  }
});

Deno.test("core: malformed answers are held to the format: Qwen3.6's garbled opener, JSON with a bad escape, a bare JSON object", async () => {
  // the opener Qwen3.6 wrote in the baseline (add-feature-0): "<toolly_call>" is plain text the
  // grammar does not let become a call, so it stays content (the Agent's bare fallback runs it)
  const garbled = "Adding the button.\n\n<toolly_call>\n<function=read_file>\n<parameter=path>\napp.js\n</parameter>\n</function>\n</tool_call>";
  // (what the grammar forces there is up to it; this fake model then goes on with its script, so only
  // the shape is checked: every call names a tool and has its required arguments, typed)
  const h1 = fakeHost("xml", [answerIds(garbled)]);
  const r1 = await coreModel(h1).ask({ system: "", tools: TOOLS, turns: [{ role: "user", text: "q" }] });
  for (const c of r1.calls) ok(c.name === "read_file" && typeof c.arguments.path === "string" && (c.arguments.start_line === undefined || Number.isInteger(c.arguments.start_line)), JSON.stringify(r1.calls));
  eq(r1.calls.length, 1, "the line-initial <function= starts the call (the XML lazy trigger)");
  eq(r1.text, "Adding the button.\n\n<toolly_call>", "the garbled opener is content (the Agent strips it from history)");
  // a JSON call whose string has an invalid escape: the grammar keeps the arguments valid JSON
  const bad = JCALL("write_file", { path: "a.js", content: "x" }).replace('"x"', '"a\\qb"');
  const h2 = fakeHost("json", [answerIds(bad)]);
  const r2 = await coreModel(h2).ask({ system: "", tools: TOOLS, turns: [{ role: "user", text: "q" }] });
  ok(h2.overruled > 0, "the bad escape was overruled");
  eq(r2.calls.length, 1);
  ok(typeof r2.calls[0].arguments.content === "string" && JSON.parse(r2.calls[0].sampled).path === "a.js", r2.calls[0].sampled);
  // a bare JSON object with no <tool_call>: text (the Agent's bare fallback picks it up)
  const h3 = fakeHost("json", [answerIds('{"name": "serve", "arguments": {}}')]);
  const r3 = await coreModel(h3).ask({ system: "", tools: TOOLS, turns: [{ role: "user", text: "q" }] });
  eq([r3.calls.length, r3.text], [0, '{"name": "serve", "arguments": {}}']);
});

Deno.test("core: an engine fault is garbage (not run); a model failure throws with what was said; a full context throws ContextFull; Stop is abort", async () => {
  const h = fakeHost("xml", [answerIds(XCALL("write_file", { path: "a", content: "x".repeat(40) }))], { nanFrom: 40 });
  const r = await coreModel(h).ask({ system: "", tools: TOOLS, turns: [{ role: "user", text: "q" }] });
  eq(r.reason, "garbage");
  let err = null;
  try { await coreModel(fakeHost("xml", [], { fail: "a device left" })).ask({ system: "", tools: TOOLS, turns: [{ role: "user", text: "q" }] }); } catch (e) { err = e; }
  ok(err && err.message === "a device left" && err.partialText === "", String(err));
  err = null;
  try { await coreModel(fakeHost("xml", [], { maxSeq: 300 })).ask({ system: "", tools: TOOLS, turns: [{ role: "user", text: "q".repeat(400) }] }); } catch (e) { err = e; }
  ok(err?.name === "ContextFull" && err.max === 300, String(err));
  const ac = new AbortController();
  const host = fakeHost("xml", [answerIds("a long answer ".repeat(10))]);
  const g = host.generate;
  host.generate = (ids, o) => { ac.abort(); return g(ids, o); };
  const r2 = await coreModel(host).ask({ system: "", tools: TOOLS, turns: [{ role: "user", text: "q" }], signal: ac.signal });
  eq(r2.reason, "abort");
});

Deno.test("core: a template without a tool-call format is refused (the room keeps the legacy path for it)", async () => {
  const host = fakeHost("xml", []);
  host.chatTemplate = () => "{% for m in messages %}{{ m.content }}{% endfor %}";
  const m = coreModel(host);
  eq(m.tools, false);
  let err = null;
  try { await m.ask({ system: "", tools: TOOLS, turns: [{ role: "user", text: "q" }] }); } catch (e) { err = e; }
  ok(/no tool-call format/.test(err?.message), String(err));
});

Deno.test("core: the host's profile and token texts are used when it has them (one mask cache with the API path)", async () => {
  const host = fakeHost("json", [answerIds("hi")]);
  let pc = 0, tc = 0;
  const tt = (id) => BT.decode([id]);
  host.profile = () => { pc++; return templateProfile(TPL.json, BT); };
  host.tokenTexts = () => { tc++; return tt; };
  const m = coreModel(host);
  await m.ask({ system: "", tools: TOOLS, turns: [{ role: "user", text: "q" }] });
  ok(pc >= 1 && tc >= 1, `profile ${pc}, tokenTexts ${tc}`);
});

Deno.test("core: scriptedCore runs scripted raw text through CallStream into the same answer shape", async () => {
  const m = scriptedCore(scripted(["Looking.\n" + xmlCall("read_file", { path: "a.js" }), "<tool_call>\n<function=write_file>\n<parameter=path>\nb.js\n</parameter>\n<parameter=content>\nline"]));
  const { ev, on } = events();
  const r = await m.ask({ system: "", tools: TOOLS, turns: [{ role: "user", text: "q" }], on });
  eq([r.text, r.calls.map((c) => [c.name, c.arguments]), r.reason], ["Looking.", [["read_file", { path: "a.js" }]], "stop"]);
  ok(ev.some((e) => e.t === "call") && ev.some((e) => e.t === "end"));
  const r2 = await m.ask({ system: "", tools: TOOLS, turns: [{ role: "user", text: "q" }] });
  eq([r2.calls.length, r2.open?.name], [0, "write_file"]);
  ok(r2.open.args.startsWith('{"path": "b.js", "content": "line'), r2.open.args);
});
