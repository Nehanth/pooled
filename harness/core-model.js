// Code mode on the serve v2 core, in process ("hcore"; docs/design/harness-core.md). Every model
// call goes through the same code `pooled serve` answers API clients with (room/api.js): the chat
// template's own tool prompt and rendering (room/conversation.js renderApi, templateProfile), the
// strict tool-call grammar (harness/constrain.js GrammarConstraint), the streaming call parser
// (harness/tools.js CallStream) and the think split. No HTTP, no wire message: one Chat
// Completions-shaped call,
//   ask({ system, tools, turns, signal, on }) -> { text, calls, open, reason, err, ids }
// over a host that says where the model runs:
//   { tok(), chatTemplate(), maxSeq(), generate(ids, {onToken, stop, maxNew, sample, signal, pin, pinTag}),
//     modelKey()?, profile()?, tokenTexts()?, vocabSize()? }
// room.js roomApi is one (the room's model), harness/engine-gen.js engineHost another (one engine:
// the eval and tests).
//
// History is structured (harness/agent.js): an assistant turn is { text, sampled: [{ name, args }] }
// (args: the JSON text as sampled), a tool-results turn { results: [...] }. toMessages() maps turns
// to v2 messages, so the same session renders right for a JSON-style model and an XML one; older
// turns that hold raw call markup (sessions saved before, or by the legacy path) are parsed once.
// Each answer's exact sampled ids are kept per turn object (idsOf / setIds): the next step replays
// them instead of re-tokenizing the text, so the prompt extends what the caches hold.
//
// On-event stream: { t: "delta", text } raw decoded pieces | { t: "text", text } content |
// { t: "think", text } reasoning | { t: "call", i, name } | { t: "args", i, a } | { t: "end", i }.
import { apiPrompt2, apiRun2, EncodeCache } from "../room/api.js";
import { templateProfile } from "../room/conversation.js";
import { pickSampler } from "../room/sampling.js";
import { CallStream, toolsSystemPromptExact, callBodyText, parseLooseJSON } from "./tools.js";
import { compileSchema, SCHEMA_CAPS } from "./jsonschema.js";
import { grammarNodeCount } from "./constrain.js";
import { tokenTexts, ContextFull } from "./model-common.js";

export { ContextFull };

const MARGIN = 16;   // positions kept free past the answer (the template's end tokens), as the legacy adapters
const safeParse = (s) => { try { const v = JSON.parse(s); return v && typeof v === "object" && !Array.isArray(v) ? v : {}; } catch { return {}; } };
const looseArgs = (s) => { try { const v = parseLooseJSON(String(s || "").trim() || "{}", { open: true }); return v && typeof v === "object" && !Array.isArray(v) ? v : {}; } catch { return {}; } };

// token texts, one table per tokenizer (the grammar's mask cache is keyed by this function)
const ttMemo = new WeakMap();
export const tokenTextsFor = (tok) => { let f = ttMemo.get(tok); if (!f) { f = tokenTexts(tok); ttMemo.set(tok, f); } return f; };

// ---- history: Agent turns -> v2 messages ----------------------------------------------------------
const RESP_RE = /<tool_response>\n([\s\S]*?)\n<\/tool_response>/g;
const legacyMemo = new Map();   // raw assistant text -> { text, calls } (bounded)
// a raw assistant answer with call markup in it, as the legacy path kept it: its content and calls
export function parseLegacyAnswer(raw, style = "json", tools = [], other = false) {
  const key = style + "\u0000" + raw;
  const hit = legacyMemo.get(key);
  if (hit) return hit;
  // any name the text calls counts (a tool may have been renamed or removed since)
  const names = new Set(tools.map((t) => t.name));
  for (const m of raw.matchAll(/<function=([^>\s<]+)>|"name"\s*:\s*"([^"\\]{1,128})"/g)) names.add(m[1] || m[2]);
  const S = new CallStream({ style, tools, allowed: [...names], constrained: false });
  let text = "";
  for (const e of [...S.push(raw), ...S.end()]) if (e.t === "text") text += e.text;
  const calls = S.calls.map((c) => ({ name: c.name, args: safeParse(c.args) }));
  if (S.open) calls.push({ name: S.open.name, args: looseArgs(S.open.args) });
  let v = { text, calls };
  // saved under a model with the other call style (a re-deal): parse it in that one
  if (!calls.length && !other) { const w = parseLegacyAnswer(raw, style === "json" ? "xml" : "json", tools, true); if (w.calls.length) v = w; }
  legacyMemo.set(key, v);
  if (legacyMemo.size > 256) legacyMemo.delete(legacyMemo.keys().next().value);
  return v;
}
const hasMarkup = (s) => /<tool_call>|<function=|<\/function>|"name"\s*:/.test(s);
// the results of a tool-results turn: its `results`, or (a legacy turn) its <tool_response> blocks
export function turnResults(t) {
  if (Array.isArray(t.results)) return t.results;
  if (t.role !== "user" || !t.text.startsWith("<tool_response>")) return null;
  const out = [];
  for (const m of t.text.matchAll(RESP_RE)) out.push(m[1]);
  return out.length ? out : null;
}
// -> { messages, src }: src[j] is the turn assistant message j came from (for its exact ids)
export function toMessages(turns, { style = "json", tools = [] } = {}) {
  const messages = [], src = [];
  const push = (m, t = null) => {
    const last = messages[messages.length - 1];
    if (m.role === "user" && last?.role === "user") { last.text += "\n\n" + m.text; return; }
    messages.push(m); src.push(t);
  };
  for (const t of turns) {
    if (t.role === "assistant") {
      if (Array.isArray(t.sampled)) push({ role: "assistant", text: t.text || "", calls: t.sampled.map((c) => ({ name: c.name, args: typeof c.args === "string" ? looseArgs(c.args) : c.args || {} })) }, t);
      else if (hasMarkup(t.text || "")) { const p = parseLegacyAnswer(t.text, style, tools); push({ role: "assistant", text: p.text, calls: p.calls }, null); }
      else push({ role: "assistant", text: t.text || "" }, null);
      continue;
    }
    const res = turnResults(t);
    const last = messages[messages.length - 1];
    // results pair with the calls before them; with none (an answer whose calls could not run) the
    // block goes as the user's text, so there is never a <tool_response> without a call
    if (res && last?.role === "assistant" && last.calls?.length) { for (const r of res) push({ role: "tool", text: String(r) }); }
    else push({ role: "user", text: t.text });
  }
  return { messages, src };
}

// The v2 ask Code mode makes, already in the shape room/api.js validateApiAsk normalizes one to (it
// is built here, not validated per step; a unit test keeps the two in step): every tool allowed, any
// number of calls (the grammar's MAX_CALLS), free text before them, sampled with the room's preset.
export function coreRequest({ system = "", tools = [], messages, maxNew = 8192, thinking = false }) {
  return { api: 2, rid: "code", system, tools: tools.length ? tools : null, messages,
    params: { maxTokens: maxNew, temperature: null, topK: null, stop: [], thinking: !!thinking, thinkBudget: null, client: "Code",
      toolChoice: "auto", allowed: null, parallel: true, maxCalls: null, format: null, effort: null } };
}

// ---- the adapter ----------------------------------------------------------------------------------
// maxNew: the answer cap; sampling: room/sampling.js preset (default: "exact" for the JSON style,
// "focused" for XML, as the legacy path); pinTag: whose pinned checkpoint the system prompt is.
// The room keeps another tag's pin used in the last 10 minutes (room.js PIN_KEEP_MS): with a
// `pooled serve` client active, Code runs unpinned and a compaction re-prefills from the start.
export function coreModel(host, { maxNew = 8192, sampling = null, thinking = false, pinTag = "code" } = {}) {
  const ids = new WeakMap();       // turn object -> { ids, thinkEnd, epoch }
  const counts = new Map();        // text -> tokens, LRU
  const enc = new EncodeCache(8 << 20);
  const stats = { calls: 0, reused: 0, prefilled: 0, generated: 0, tps: 0, forced: 0, last: null };
  let lastTok = null, lastKey = null, profile = null, epoch = 0, grammarOk = null;
  const tok = () => {
    const T = host.tok();
    if (!T) throw new Error("the model is not loaded");
    const key = host.modelKey?.() ?? "";
    if (T !== lastTok || key !== lastKey) { lastTok = T; lastKey = key; profile = null; counts.clear(); enc.clear(); grammarOk = null; epoch++; }
    return T;
  };
  const prof = () => { const T = tok(); return (profile ||= host.profile?.() ?? templateProfile(host.chatTemplate(), T)); };
  const tt = () => host.tokenTexts?.() ?? tokenTextsFor(tok());
  const defs = (tools) => (tools || []).map(({ name, description, parameters }) => ({ name, description: description || "", parameters: parameters || { type: "object", properties: {} } }));
  // the tools compile, and fit the grammar together (once per tool set and model): a tool the
  // grammar cannot take fails here with its name, not as every step's "could not build the grammar"
  const checkTools = (tools) => {
    const sig = JSON.stringify(tools);
    if (grammarOk === sig) return;
    for (const t of tools) compileSchema(t.parameters);
    const n = grammarNodeCount(tools, { style: prof().style });
    if (n > SCHEMA_CAPS.nodes) throw new Error(`Code's tools are too large for the tool-call grammar (${n} nodes)`);
    grammarOk = sig;
  };
  const m = {
    get style() { return prof().style; },
    // the model's chat template has a tool-call format (Code mode needs one; else the legacy path)
    get tools() { return !!prof().tools; },
    // the system turn as the template writes it (for counting and the eval's trajectories)
    systemText: (system, tools) => toolsSystemPromptExact(defs(tools), { style: prof().style, system }),
    // one sampled call as the template renders it in history (for counting)
    callText(c) {
      const style = prof().style;
      return callBodyText({ name: c.name, arguments: style === "json" ? String(c.args ?? "{}") : looseArgs(c.args) }, style);
    },
    budget: () => host.maxSeq() - (Math.min(maxNew, Math.floor(host.maxSeq() / 4)) + 64),
    count(text) {
      const T = tok();
      let n = counts.get(text);
      if (n !== undefined) { counts.delete(text); counts.set(text, n); return n; }
      n = enc.encode(T, text).length;
      counts.set(text, n);
      if (counts.size > 256) counts.delete(counts.keys().next().value);
      return n;
    },
    stats,
    reset() { lastTok = null; tok(); },
    // exact ids per assistant turn (the Agent calls setIds when the turn is exactly what was sampled)
    setIds(turn, v) { if (v?.ids?.length) ids.set(turn, { ids: Array.from(v.ids), thinkEnd: v.thinkEnd || 0, epoch }); },
    idsOf(turn) { const v = ids.get(turn); return v && v.epoch === epoch ? v : null; },
    // which tokenizer saved ids belong to (sessions): its vocabulary size and how a fixed probe encodes
    idsTag() { const T = tok(); let vs = 0; for (const v of Object.values(T.vocab || {})) if (v >= vs) vs = v + 1; return `core:${vs}:` + T.encode("Tabby ids · fn(x) => 1024 ✓").join(","); },
    adoptIds(turn, v) { tok(); m.setIds(turn, v); },   // v: { ids, thinkEnd } as idsOf gave it

    async ask({ system = "", tools = [], turns, signal = null, on = () => {} }) {
      const T = tok(), P = prof();
      if (!P.tools) throw new Error("this model's chat template has no tool-call format (Code mode's core path needs one)");
      const td = defs(tools);
      checkTools(td);
      const { messages, src } = toMessages(turns, { style: P.style, tools: td });
      const req = coreRequest({ system, tools: td, messages, maxNew, thinking });
      const maxSeq = host.maxSeq();
      const turnIds = (j) => { const t = src[j]; const v = t && m.idsOf(t); return v ? { ids: v.ids, thinkEnd: v.thinkEnd, reasoned: false } : null; };
      const prompt = apiPrompt2(T, req, maxSeq, { profile: P, encoder: enc, turnIds });
      if (prompt.err) throw new ContextFull(prompt.n, maxSeq);
      const send = (x) => {
        if (x.t === "ai-token") on(x.th ? { t: "think", text: x.text } : { t: "text", text: x.text });
        else if (x.t === "ai-call") {
          if (x.name != null) on({ t: "call", i: x.i, name: x.name });
          else if (x.a != null) on({ t: "args", i: x.i, a: x.a });
          else if (x.end) on({ t: "end", i: x.i });
        }
      };
      const res = await apiRun2({
        tok: T, req, prompt, send, onPiece: (p) => on({ t: "delta", text: p }), cache: null,
        generate: (x, o) => host.generate(x, { ...o, pin: prompt.systemLen, pinTag }),
        fallback: pickSampler(sampling ?? (P.style === "json" ? "exact" : "focused")),
        ctxMax: maxSeq - MARGIN, signal, tt: tt(), vocabSize: host.vocabSize?.() || 0, garbage: "mass",
      });
      const generated = res.usage?.out || 0, reused = res.reused || 0;
      const reason = res.garbage ? "garbage" : res.reason;
      stats.calls++; stats.reused += reused; stats.prefilled += prompt.ids.length - reused; stats.generated += generated;
      stats.tps = res.gen?.tps || 0; stats.forced += res.forcedFree || 0;
      stats.last = { reason, prompt: prompt.ids.length, reused, prefilled: prompt.ids.length - reused, generated, tps: res.gen?.tps || 0,
        forced: res.forcedFree || 0, forcedAll: res.forced || 0, tDecode: res.gen?.tDecode ?? null, stats: res.stats || "", exact: prompt.exact };
      // a generation failure (the model is not loaded, a device left and the room could not recover)
      // is the Agent's error path, never an ordinary answer
      if (res.reason === "error" && !res.garbage) {
        const e = new Error(res.err || "the model failed");
        e.partialText = res.text || "";
        throw e;
      }
      return {
        text: res.text || "",
        calls: res.calls.map((c) => ({ name: c.name, arguments: safeParse(c.args), sampled: c.args })),
        open: res.open ? { name: res.open.name, args: res.openArgs ?? "" } : null,
        reason, err: res.err, raw: res.raw,
        ids: res.ids ? { ids: res.ids, thinkEnd: res.thinkEnd } : null,
      };
    },
  };
  return m;
}

// A scripted model on the same interface, for tests and ?mock=code: textGen({ system, turns, signal })
// yields raw answer text (tests/scripted-model.js), which runs through CallStream (no grammar, no
// tokens) into the same events and Answer. { style, budget, count } as the mock gives them.
export function scriptedCore(textGen, { style = "xml", budget = null, count = null } = {}) {
  const stats = { calls: 0, reused: 0, prefilled: 0, generated: 0, tps: 0, forced: 0, last: null };
  const defs = (tools) => (tools || []).map(({ name, description, parameters }) => ({ name, description: description || "", parameters: parameters || { type: "object", properties: {} } }));
  return {
    style, tools: true, stats, scripted: true,
    systemText: (system, tools) => toolsSystemPromptExact(defs(tools), { style, system }),
    callText: (c) => callBodyText({ name: c.name, arguments: style === "json" ? String(c.args ?? "{}") : looseArgs(c.args) }, style),
    ...(budget ? { budget } : {}), ...(count ? { count } : {}),
    setIds() {}, idsOf: () => null,
    async ask({ system = "", tools = [], turns, signal = null, on = () => {} }) {
      const S = new CallStream({ style, tools: defs(tools), constrained: false });
      let text = "", raw = "";
      const fire = (evs) => {
        for (const e of evs) {
          if (e.t === "text") { text += e.text; on({ t: "text", text: e.text }); }
          else if (e.t === "call") on({ t: "call", i: e.i, name: e.name });
          else if (e.t === "args") on({ t: "args", i: e.i, a: e.a });
          else if (e.t === "end") on({ t: "end", i: e.i });
        }
      };
      for await (const d of textGen({ system: toolsSystemPromptExact(defs(tools), { style, system }), turns, signal })) {
        raw += d;
        on({ t: "delta", text: d });
        fire(S.push(d));
        if (signal?.aborted) break;
      }
      fire(S.end());
      const reason = signal?.aborted ? "abort" : "stop";
      stats.calls++;
      stats.last = { reason, prompt: 0, reused: 0, prefilled: 0, generated: 0, tps: 0, forced: 0 };
      return {
        text, raw, reason, err: null, ids: null,
        calls: S.calls.map((c) => ({ name: c.name, arguments: safeParse(c.args), sampled: c.args })),
        open: S.open ? { name: S.open.name, args: S.open.args } : null,
      };
    },
  };
}
