// The Code mode agent loop: ask the model, run the tools it calls, hand back the results, repeat
// until it answers without calling a tool. The model is any function
//   generate({ system, turns, signal }) -> async iterable of text deltas
// (the room, a single engine, or a script in tests), so the loop knows nothing about GPUs.
//
// turns: [{ role: "user" | "assistant", text }]. Tool results go back as one user turn of
// <tool_response> blocks, as the Qwen templates expect. Assistant turns keep the model's raw
// text (tool-call markup included) so the history the model sees is exactly what it wrote.
// Turns also carry bookkeeping the model ignores: `req` (which request they belong to) and, on a
// tool-result turn, `calls` (what each result was, for the compaction stubs).
//
// Tool: { name, description, parameters, mutates, run(args, { signal, step }) -> string,
//         preview?(args) -> { path, before, after } }   (shown to approve() for mutating tools)
// Events (onEvent): step, delta (raw streamed text), text (visible text), call-live, tool-start,
// tool, usage, compacted, trimmed, stopped, stuck, done, limit, card (a recovery note was added,
// harness/cards.js), bare (hcore: calls written outside the call format were run).
//
// Two model paths (room/code.js ?hcore=0|1):
//   generate (legacy): raw text in, parsed here with ToolCallParser; assistant turns keep the raw
//     text, tool results are one user turn of <tool_response> blocks
//   model (hcore, harness/core-model.js): model.ask() runs the serve v2 core (the template's tool
//     prompt, the strict grammar, CallStream) and returns content and calls. Assistant turns are
//     { text, sampled: [{ name, args }] } (args: the JSON text as sampled), tool-result turns also
//     carry `results` (the array; `text` stays the <tool_response> blocks, for sizes and checks).
//     Either path renders the other's turns (toMessages / _legacyTurns).
import { toolsSystemPrompt, toolResponses, ToolCallParser, parseCallBody, parseLooseJSON, normalizeXmlCall, renderCalls } from "./tools.js";
import { pickCard, hint, PRIORITY, MAX_PER } from "./cards.js";
import { fixArgs, fixToolName } from "./argfix.js";

const STOPPED = "(stopped by the user)";
const EMPTY = "(empty answer: call a tool or say you are done)";
const LIVE = new Set(["preview_logs", "run_js"]);   // results that can change with time: a repeat is not a loop
// a clean serve: the page loaded with no errors (harness/preview-tools.js)
const CLEAN = /^serving [^\n]*\n(?:loaded in \d+ ms · no errors|still loading after 2 s · no errors)/;
// a read run again gives the same result: for serve, ignoring what changes on every reload of the
// same page (harness/preview-tools.js): its load time, the log timestamps, the error / warning counts,
// the ×N fold counts (a loop throwing each frame) and the "more: preview_logs since=N" cursor
export const timeless = (s) => s
  .replace(/loaded in \d+ ms/g, "loaded")
  .replace(/^((?:loaded|still loading after 2 s) · )(.*)$/m, (_, a, counts) => a + counts.replace(/\d+ (error|warning)s?\b/g, "$1s"))
  .replace(/^\[\d+(?:\.\d+)?s\] /gm, "")
  .replace(/ ×\d+$/gm, "")
  .replace(/^more: preview_logs since=\d+$/gm, "more: preview_logs");
const sameResult = (name, a, b) => a === b || (name === "serve" && timeless(a) === timeless(b));
// said after a clean serve to a small (JSON-style) model, which otherwise tends to write the same files again
export const SERVED_OK = "\nnext: the page loads with no errors. If it does what was asked, reply with one short line saying what you built (no tool call). Otherwise fix it with edit_file.";
export const CONTEXT_FULL = "context full: start a new task (the files are kept)";
// the same failure this many steps in a row stops the request (one step before, the model is warned)
export const STUCK_AFTER = 3;
// the same page error seen by this many checks, each after a change: a note at WARN, a stop at STOP
export const PAGE_ERR = { warn: 3, stop: 5 };

// the first line of a result, short enough for a note or the stop message
export const firstLine = (s, max = 160) => { const l = String(s ?? "").split("\n")[0].trim(); return l.length > max ? l.slice(0, max) + "…" : l; };
// the first page error a serve / preview_logs result reports ("game.js:41:5 ReferenceError: ctx is
// not defined"), or null; "no errors" results give "". preview_logs can list older pages' logs too:
// only its "first error:" line or the lines under the current page's "(rev N, current)" count.
const ERR_LINE = /^(?:first error: )?\[[\d.]+s\] error (.*?)(?: ×\d+)?$/m;
export function pageError(name, result) {
  if (name !== "serve" && name !== "preview_logs") return null;
  let text = String(result ?? "");
  if (name === "preview_logs") {
    const first = /^first error: .*$/m.exec(text), cur = text.lastIndexOf(", current)\n");
    if (first) text = first[0];
    else if (cur >= 0) text = text.slice(cur);
    else return null;
  }
  const m = ERR_LINE.exec(text);
  if (m) return firstLine(m[1]);
  return CLEAN.test(text) ? "" : null;
}

// head and tail of a long tool result (errors are usually at the end, headers at the start)
export function capResult(s, max) {
  if (s.length <= max) return s;
  const cut = s.length - max, head = Math.ceil(max * 0.6), tail = max - head;
  return s.slice(0, head) + `\n…(${cut} chars cut)…\n` + s.slice(s.length - tail);
}
// one line naming a call, e.g. "read_file game.js 1 200"
export function briefCall(c) {
  const vals = Object.entries(c.arguments || {}).map(([k, v]) => {
    if (k === "append") return v === true || v === "true" ? "(append)" : null;   // write_file's flag, not a bare 'true'
    const s = typeof v === "string" ? v : JSON.stringify(v);
    return s.includes("\n") || s.length > 40 ? null : s;
  }).filter((v) => v != null && v !== "");
  return [c.name || "?", ...vals].join(" ");
}

// A tool-result turn in its compacted form: each <tool_response> over 200 characters becomes a stub
// naming its call (calls[k], the turn's call briefs), shorter ones stay. It depends on nothing but
// its arguments and a stub is short, so compacting a compacted turn changes nothing: the turn then
// renders the same on every later step.
export function stubResults(text, calls) {
  let k = 0;
  return text.replace(/<tool_response>\n([\s\S]*?)\n<\/tool_response>/g, (m, body) => {
    const name = calls?.[k++];
    if (body.length <= 200) return m;
    return `<tool_response>\n(output of ${name || "this call"} dropped; run it again if needed)\n</tool_response>`;
  });
}

// The same for a turn's results array (hcore): a result over 200 characters becomes the stub.
export function stubResultList(results, calls) {
  return results.map((r, k) => (r.length <= 200 ? r : `(output of ${calls?.[k] || "this call"} dropped; run it again if needed)`));
}
// an hcore assistant turn as legacy text: its content, then its calls as markup in `style`
export function legacyText(t, style = "xml") {
  if (!t.sampled?.length) return t.text;
  const calls = t.sampled.map((c) => ({ name: c.name, arguments: looseObject(c.args) }));
  return (t.text ? t.text + (t.text.trim() ? "\n\n" : "") : "") + renderCalls(calls, style);
}
// a content's trailing call-tag fragments ("<toolly_call>", "</tool_call>"), taken off
export function stripStrayTags(s) {
  for (let t = s; ;) { const u = t.replace(/\s*<\/?[\w.]*tool[\w.]*>?\s*$/, ""); if (u === t) return t; t = u; }
}
const looseObject = (s) => {
  if (s && typeof s === "object") return s;
  try { const v = parseLooseJSON(String(s || "").trim() || "{}", { open: true }); return v && typeof v === "object" && !Array.isArray(v) ? v : {}; } catch { return {}; }
};

export class Agent {
  // budget: tokens the conversation may take (a number, or a function when the context can change,
  // e.g. roomModel().budget); count: text -> tokens (default ~3.5 characters per token).
  // approve(call, info) -> true | false | { ok: false, reason }: info is tool.preview(args).
  // usage() -> { prompt, reused, generated, tps, reason } after each step (roomModel's stats.last),
  // optional; reason "max" / "ctx" means the answer hit the length cap.
  // idsFor(text) / adopt(text, ids): the model's exact sampled ids, for toJSON / from; idsTag() names
  // the tokenizer they belong to, so a session saved under another model re-encodes its text.
  // coach: short "what to do next" notes after a clean serve (default: JSON-style models, the small ones)
  constructor({ generate = null, model = null, tools, style = "xml", system = "", maxSteps = 24, approve = async () => true, onEvent = () => {},
    budget = Infinity, count = null, maxResultChars = 6000, usage = null, idsFor = null, adopt = null, idsTag = null, coach = null }) {
    this.generate = generate; this.model = model; this.tools = tools; this.maxSteps = maxSteps;
    this.style = style = model ? model.style : style;
    this.coach = coach ?? style === "json";
    this.budget = typeof budget === "function" ? budget : () => budget;
    this.count = count || ((t) => Math.ceil(t.length / 3.5));
    this.approve = approve; this.onEvent = onEvent; this.maxResultChars = maxResultChars;
    this.usage = usage; this.idsFor = idsFor; this.adopt = adopt; this.idsTag = idsTag;
    this.toolDefs = tools.map(({ name, description, parameters }) => ({ name, description, parameters }));
    this.systemRaw = system;
    // the system turn the model sees (for sizes and the eval's trajectories): the template's tool
    // block (hcore) or Code's own wording (legacy)
    if (model) { this.system = model.systemText(system, this.toolDefs); this.idsTag = idsTag ?? (model.idsTag ? () => model.idsTag() : null); }
    else this.system = toolsSystemPrompt(this.toolDefs, { style, system });
    this.byName = new Map(tools.map((t) => [t.name, t]));
    this.reset();
  }
  reset() { this.turns = []; this.reqs = {}; this.req = 0; }

  // Run one user request to the end. -> { text, steps, calls, reason: "done"|"stopped"|"limit"|"context"|"stuck" }
  async run(userText, { signal } = {}) {
    const req = ++this.req;
    const R = this.reqs[req] = { calls: [], done: false, answer: "", cards: {} };
    this.turns.push({ role: "user", text: userText, req });
    let calls = 0, shown = "", failRun = 0, lastFail = "", empty = 0, mut = 0;
    let pageErr = { sig: "", n: 0, mut: -1 };   // the page error the checks keep reporting, and after how many changes
    let prevStep = new Map();   // the last step's calls: name+args -> { step, result, mut }
    let cleanAt = -1;           // mut when a serve last loaded with no errors (-1: not since the last change)
    const fails = new Map();    // a failed call (name+args) -> { n: times, mut } since the last change
    // never leave two user turns in a row: an untouched request is taken back, anything else gets
    // a closing assistant turn
    const close = () => {
      const last = this.turns[this.turns.length - 1];
      if (last?.role !== "user") return;
      if (last.req === req && !last.calls) { this.turns.pop(); delete this.reqs[req]; }
      else this.turns.push({ role: "assistant", text: STOPPED, req });
    };
    // a request that ended any way but "done" is still finished history: compaction may fold it
    const finish = () => { R.done = true; R.answer ||= shown.trim(); };
    const full = (step) => {
      finish(); close();
      this.onEvent({ type: "done", step, reason: "context" });
      return { text: CONTEXT_FULL, steps: step, calls, reason: "context" };
    };
    const stopped = (step) => {
      finish(); close();
      this.onEvent({ type: "stopped", step });
      return { text: shown.trim(), steps: step, calls, reason: "stopped" };
    };
    for (let step = 1; step <= this.maxSteps; step++) {
      if (signal?.aborted) return stopped(step - 1);
      this.onEvent({ type: "step", step });
      if (this._compact(req) === "full") return full(step);
      let found = [], u = null;
      shown = "";
      if (this.model) {
        // hcore: one answer from the serve v2 core (harness/core-model.js), calls already parsed
        let live = null, res = null, failed = null;
        const on = (ev) => {
          if (ev.t === "delta") this.onEvent({ type: "delta", text: ev.text, step });
          else if (ev.t === "text") { shown += ev.text; this.onEvent({ type: "text", text: ev.text, step }); }
          // the call being typed, as append-only JSON-shaped text (room/code-ui.js parseLive reads
          // it the same for both call styles and on older peers); null once it is complete
          else if (ev.t === "call") { live = `{"name": ${JSON.stringify(ev.name)}, "arguments": `; this.onEvent({ type: "call-live", raw: live, step }); }
          else if (ev.t === "args") { if (live != null) { live += ev.a; this.onEvent({ type: "call-live", raw: live, step }); } }
          else if (ev.t === "end") { live = null; this.onEvent({ type: "call-live", raw: null, step }); }
        };
        try { res = await this.model.ask({ system: this.systemRaw, tools: this.toolDefs, turns: this.turns, signal, on }); }
        catch (err) { failed = err; }
        if (live != null) this.onEvent({ type: "call-live", raw: null, step });
        // what was said stays; calls that never ran do not (no call in history without its result)
        const keep = () => { if (shown.trim()) this.turns.push({ role: "assistant", text: shown, sampled: [], req }); };
        if (failed?.name === "ContextFull") { keep(); return full(step); }
        if (failed && !signal?.aborted) {
          // the model failed (a device left, the model unloaded): keep what it said and close the
          // turn, so the next request does not follow a dangling user turn
          keep(); finish(); close();
          throw failed;
        }
        if (signal?.aborted || !res || res.reason === "abort") { keep(); return stopped(step); }
        u = this.usage?.() ?? this.model.stats?.last ?? null;
        found = res.calls.map((c) => ({ name: c.name, arguments: c.arguments, sampled: c.sampled }));
        let text = res.text;
        if (res.reason === "garbage") {
          // the engine's logits were not a model's (NaN, or almost no mass on anything the format
          // allows): none of the calls runs (a forced write_file would overwrite a file)
          if (res.open) found.push({ name: res.open.name, arguments: {}, sampled: "{}" });
          for (const c of found) {
            c.garbage = true; c.open = true;
            c.error = `this answer was stopped and its calls were not run: the room's engine is producing garbage (the model's output stopped following the call format${u?.forcedAll ? ` after ${u.forcedAll} forced tokens` : ""}; try again, or reload the model)`;
          }
        } else if (res.open) found.push(this._openCall(res.open, u));
        else if (!found.length) {
          // no call, but a known tool written outside the call format (a bare JSON object, bare
          // tags, a <function=...> block with no opener): run it, say how to write it next time
          const b = bareFromText(text, this.byName);
          if (b.calls.length) {
            for (const c of b.calls) { c.bare = true; c.sampled = JSON.stringify(c.arguments); }
            found.push(...b.calls); text = b.text;
            this.onEvent({ type: "bare", n: b.calls.length, step });
            try { console.warn(`[code] ${b.calls.length} call(s) written outside the call format`); } catch { /* no console */ }
          }
        }
        // a garbled call opener left at the end of the content ("<toolly_call>", seen from Qwen3.6): the
        // call after it ran (the grammar's line-initial <function= trigger), so it is noise the model
        // would copy on its next answer
        if (found.length && !found[0].bare) text = stripStrayTags(text);
        if (u?.forced > 8 && res.reason !== "garbage") for (const c of found) if (c.error) c.error += ` (the call format forced ${u.forced} of the model's own tokens)`;
        // small-model slips: arguments under other names or types (for running them; history keeps
        // what was sampled)
        for (const c of found) this._fix(c);
        const turn = { role: "assistant", text, sampled: found.filter((c) => !c.garbage).map((c) => ({ name: c.name, args: c.sampled })), req };
        this.turns.push(turn);
        // the turn is exactly what was sampled: its ids replay next step (the prompt extends the caches)
        if (res.ids && !res.open && text === res.text && found.length === res.calls.length && found.every((c) => !c.garbage && !c.bare)) this.model.setIds?.(turn, res.ids);
      } else {
        const P = new ToolCallParser({ schemaFor: (n) => this.byName.get(n)?.parameters });
        let raw = "";
        try {
          for await (const d of this.generate({ system: this.system, turns: this._legacyTurns(), signal })) {
            raw += d;
            this.onEvent({ type: "delta", text: d, step });
            const r = P.feed(d);
            shown += r.text; found.push(...r.calls);
            if (r.text) this.onEvent({ type: "text", text: r.text, step });
            // the tool call being typed, so the UI can show code as it is written (null once it is complete)
            if (P.inCall) this.onEvent({ type: "call-live", raw: P.buf, step });
            else if (r.calls.length) this.onEvent({ type: "call-live", raw: null, step });
          }
        } catch (err) {
          if (err?.name === "ContextFull") {
            if (raw) this.turns.push({ role: "assistant", text: raw, req });
            return full(step);
          }
          if (!signal?.aborted) {
            // the model failed (a device left): keep what it said and close the turn, so the next
            // request does not follow a dangling user turn
            if (raw) this.turns.push({ role: "assistant", text: raw, req });
            finish(); close();
            throw err;
          }
        }
        if (signal?.aborted) {
          // keep what was said (the tool calls in it do not run)
          if (raw) this.turns.push({ role: "assistant", text: raw, req });
          return stopped(step);
        }
        const e = P.end();
        u = this.usage?.();
        // the call grammar forced most of a call's tokens (the adapter ended the answer): the calls are
        // the grammar's shape around garbage logits, not the model's, so none of them runs (a forced
        // write_file would overwrite a file, auto-approved in a scratch project)
        if (u?.reason === "garbage") {
          for (const c of [...found, ...e.calls]) {
            c.garbage = true;
            c.error = `this answer was stopped and its calls were not run: ${u.forced} tokens were forced by the call format, so the room's engine is producing garbage (try again, or reload the model)`;
            c.open = true;   // (no "write the call again" advice: the format was not the problem)
          }
        }
        // a call left open by the length cap: its last value is a fragment, so say why instead of running it
        else {
          for (let i = 0; i < e.calls.length; i++) {
            const c = e.calls[i];
            if (!c.open) continue;
            // a write_file cut mid-content: keep its complete lines instead of losing the whole answer,
            // and tell the model exactly where to pick up (otherwise it retries the same long write and
            // is cut at the same place again)
            const part = salvageWrite(c.raw, (n) => this.byName.get(n)?.parameters);
            if (part) { e.calls[i] = part; continue; }
            if (u && (u.reason === "max" || u.reason === "ctx")) c.error = `your answer was cut at ${u.generated} tokens before the call was complete`;
            else {
              // ended mid-call for another reason (end of turn, a stop, a device hiccup): say which, so it can be traced
              const tail = String(c.raw || "").slice(-160).replace(/\s+/g, " ").trim();
              c.error = `your answer ended in the middle of a tool call${u ? ` (${u.reason || "stop"} after ${u.generated} tokens)` : ""}.${tail ? ` The call ended with: "${tail}"` : ""}`;
              try { console.warn("[code] call ended early", u, JSON.stringify(String(c.raw || "").slice(-300))); } catch {}
            }
          }
        }
        // many tokens forced by the call grammar: the logits were not the model's (a misbehaving engine)
        if (u?.forced > 8 && u.reason !== "garbage") for (const c of [...found, ...e.calls]) if (c.error) c.error += ` (${u.forced} tokens were forced by the call format: the room's engine may be misbehaving)`;
        shown += e.text; found.push(...e.calls);
        // no <tool_call> at all, but the answer wrote a known tool as bare tags
        // (<write_file><path>a</path><content>..</content></write_file>, seen from Qwen 3.6): run those
        if (!found.length) {
          const bare = bareCalls(shown, this.byName);
          // or as a bare JSON object ({"name": "write_file", "arguments": {...}}), seen from Qwen3 1.7B
          if (!bare.length) bare.push(...bareJsonCalls(shown, this.byName));
          // or <function=NAME> / <invoke name="NAME"> blocks whose <tool_call> opener came out garbled
          // ("<tool_tool_calls>") or missing
          if (!bare.length) bare.push(...bareFunctionCalls(shown, this.byName));
          if (bare.length) { for (const c of bare) c.bare = true; found.push(...bare); }
        }
        // small-model slips: a tool's other name, arguments under other names or types
        for (const c of found) this._fix(c);
        if (e.text) this.onEvent({ type: "text", text: e.text, step });
        this.turns.push({ role: "assistant", text: raw, req });
      }
      if (u) this.onEvent({ type: "usage", step, prompt: u.prompt, reused: u.reused, generated: u.generated, tps: u.tps, forced: u.forced || 0 });
      if (!found.length && !shown.trim() && !empty++ && step < this.maxSteps) {
        // an empty answer: one nudge, then a second empty answer ends the request
        this.turns.push({ role: "user", text: EMPTY, req });
        continue;
      }
      if (!found.length) {
        R.done = true; R.answer = shown.trim();
        this.onEvent({ type: "done", step });
        return { text: shown.trim(), steps: step, calls, reason: "done" };
      }
      const results = [], briefs = [], reps = [], again = [], cur = new Map(), seen = new Set(), cleanBefore = cleanAt;
      for (const c of found) {
        calls++;
        R.calls.push({ name: c.name, arguments: c.arguments });
        briefs.push(briefCall(c));
        // the same call as last step with nothing changed since: answer from memory. A call that
        // only reads (read_file, list_dir, search, serve) is run again anyway, since the files can change
        // outside the agent (the user's editor, another program on a folder on disk), and is a
        // repeat only when its result is the same (serve: apart from its timings); a mutating call is not run twice.
        const key = c.error || LIVE.has(c.name) ? null : c.name + "\u0000" + JSON.stringify(c.arguments || {});
        const prev = key && prevStep.get(key);
        // the same call twice in one answer (e.g. a forced second call): run it once
        const dup = c.error ? null : c.name + "\u0000" + JSON.stringify(c.arguments || {});
        const twice = !!dup && seen.has(dup);
        if (dup) seen.add(dup);
        let rep = !!prev && prev.mut === mut && !signal?.aborted, fresh = null;
        if (rep && !twice && this.byName.get(c.name) && !this.byName.get(c.name).mutates) {
          fresh = await this._runCall(c, step, signal);   // (its tool-start / tool events go out here)
          rep = sameResult(c.name, fresh, prev.result) && !signal?.aborted;
        }
        let r;
        if (signal?.aborted) r = STOPPED;
        else if (twice) {
          r = "skipped: the same call as the one before it in this answer";
          this.onEvent({ type: "tool-start", call: c, step });
          this.onEvent({ type: "tool", call: c, result: r, step, ms: 0 });
        } else if (rep) {
          // its result is still in the prompt: point at it; else (compacted away) give it again
          const inPrompt = this.turns.some((t) => t.role === "user" && t.text.includes(prev.result));
          r = inPrompt ? `${prev.result.split("\n")[0]} (same call as step ${prev.step}; nothing changed)`
            : `${prev.result}\n(same call as step ${prev.step}; nothing changed)`;
          if (fresh == null) { this.onEvent({ type: "tool-start", call: c, step }); this.onEvent({ type: "tool", call: c, result: r, step, ms: 0 }); }
        } else {
          r = fresh ?? await this._runCall(c, step, signal);
          if (this.byName.get(c.name)?.mutates && !/^(error|declined|unchanged)/.test(r)) mut++;
          if (c.name === "serve" && CLEAN.test(r)) cleanAt = mut;
        }
        // the same failing call again with other calls in between (read_file, edit_file, read_file,
        // the same edit_file...): not caught as a repeat of the last step, so counted here
        let n = 0;
        if (key && /^(?:error|unchanged)/.test(r)) { const f = fails.get(key); n = f && f.mut === mut ? f.n + 1 : 1; fails.set(key, { n, mut }); }
        again.push(n);
        reps.push(rep);
        if (key) cur.set(key, rep ? prev : { step, result: r, mut });
        results.push(r);
      }
      prevStep = cur;
      const plain = results.slice();   // the stuck check below compares results without their cards
      // the page was served clean and this step changed nothing (the same calls again, a write of
      // the same content): the task is done, so end here instead of looping to the stuck guard
      const idle = plain.length && plain.every((r, i) => reps[i] || /^unchanged/.test(r) || (found[i].name === "serve" && CLEAN.test(r))
        || (found[i].name === "preview_logs" && /^no new logs/.test(r)));
      if (cleanBefore >= 0 && cleanBefore === mut && idle && !signal?.aborted) {
        this._pushResults(results, briefs, req);
        const text = shown.trim() || "Done: the page is served with no errors.";
        this.turns.push({ role: "assistant", text, req });
        R.done = true; R.answer = text;
        this.onEvent({ type: "done", step, idle: true });
        return { text, steps: step, calls, reason: "done" };
      }
      this._card(R, found, results, reps.map((x, i) => x || again[i] >= 2), step);
      // the same failure three steps in a row: the model (or the room) is stuck, so stop and say so
      // instead of burning the context on retries. "unchanged" (a write of what the file already
      // has) did nothing either, so it counts as a failure here.
      const failed = plain.length && plain.every((r, i) => reps[i] || /^(?:error|unchanged)/.test(r));
      const sig = failed ? plain.map((r) => r.replace(/\d+/g, "#").slice(0, 80)).join("|") : "";
      failRun = failed && (sig === lastFail || failRun === 0) ? failRun + 1 : failed ? 1 : 0;
      lastFail = sig;
      // or one failing call made a third time with nothing changed in between
      const same = plain.length && plain.every((r, i) => again[i] >= STUCK_AFTER);
      // the page keeps reporting the same first error although the model keeps changing files
      let bumped = false;
      for (let i = 0; i < found.length; i++) {
        const pe = pageError(found[i].name, plain[i]);
        if (pe === null || reps[i]) continue;
        const k = pe.replace(/\d+/g, "#");
        if (!pe) pageErr = { sig: "", n: 0, mut };
        else if (k !== pageErr.sig) pageErr = { sig: k, n: 1, mut, text: pe };
        else if (mut > pageErr.mut) { pageErr = { ...pageErr, n: pageErr.n + 1, mut, text: pe }; bumped = true; }
      }
      const pageStuck = pageErr.n >= PAGE_ERR.stop && pageErr.mut === mut;
      // one step before the stop, say what is being repeated so the model can change course
      const last = results.length - 1;
      if (!pageStuck && failRun === STUCK_AFTER - 1 && !same) {
        results[last] += `\nnote: this failed the same way last step (${firstLine(plain.find((r) => /^(?:error|unchanged)/.test(r)) ?? plain[0], 100)}). Do something different: the task stops if it fails a third time.`;
      } else if (!pageStuck && pageErr.n === PAGE_ERR.warn && bumped) {
        results[last] += `\nnote: the page showed this same error after each of your last ${pageErr.n - 1} changes: ${pageErr.text}. Those changes did not fix it: read_file the lines it names and fix the cause there.`;
      }
      this._pushResults(results, briefs, req);
      if (signal?.aborted) return stopped(step);
      if (failRun >= STUCK_AFTER || same || pageStuck) {
        const f = plain.findIndex((r) => /^(?:error|unchanged)/.test(r)), i = Math.max(0, f), n = same ? STUCK_AFTER : failRun;
        const why = pageStuck ? `the page showed the same error after ${pageErr.n - 1} changes: ${pageErr.text}`
          : f < 0 ? `${briefCall(found[i])} was repeated ${n} times with nothing changed`
          : `${briefCall(found[i])} failed ${n} times in a row: ${firstLine(plain[i])}`;
        finish(); close();
        this.onEvent({ type: "stuck", step, error: results[0], why });
        return { text: `Stopped: ${why}`, steps: step, calls, reason: "stuck" };
      }
    }
    finish(); close();
    this.onEvent({ type: "limit", steps: this.maxSteps });
    return { text: `(stopped after ${this.maxSteps} steps)`, steps: this.maxSteps, calls, reason: "limit" };
  }

  async _runCall(c, step, signal) {
    let result;
    const t0 = Date.now();
    this.onEvent({ type: "tool-start", call: c, step });
    if (c.error) result = c.open && !/^unterminated/.test(c.error) ? `error: ${c.error}` : `error: ${c.error}. Write the call again in the format the system prompt shows.`;
    else {
      const t = this.byName.get(c.name);
      if (!t) result = `error: there is no tool called ${c.name}; the tools are ${[...this.byName.keys()].join(", ")}`;
      else {
        let verdict = true;
        if (t.mutates) {
          let info = null;
          try { info = (await t.preview?.(c.arguments || {})) ?? null; } catch { info = null; }
          verdict = await this.approve(c, info);
        }
        const ok = verdict === true || (verdict && typeof verdict === "object" && verdict.ok !== false);
        if (!ok) result = "declined by the user" + (verdict?.reason ? `: ${verdict.reason}` : "");
        else {
          try { result = String(await t.run(c.arguments || {}, { signal, step })); }
          catch (err) { result = `error: ${err.message}`; }
          if (c.salvage && !/^error/.test(result)) result += `\nThe answer was cut, so only the first ${c.salvage.lines} lines of ${c.arguments.path} were saved. The last saved line is:\n${c.salvage.last}\nContinue with write_file append: true from the line after it.`;
        }
      }
    }
    if (c.bare && this.style === "xml") result += "\nhint: this ran, but write tool calls as <tool_call>\n<function=NAME>\n<parameter=NAME>\nvalue\n</parameter>\n</function>\n</tool_call>";
    else if (c.bare) result += "\nhint: this ran, but write tool calls as <tool_call>\n{\"name\": \"NAME\", \"arguments\": {...}}\n</tool_call>";
    result = capResult(result, this.maxResultChars);
    if (this.coach && c.name === "serve" && CLEAN.test(result)) result += SERVED_OK;
    this.onEvent({ type: "tool", call: c, result, step, ms: Date.now() - t0 });
    return result;
  }

  // at most one card per step: the most urgent one earned, unless its last copy is still in the
  // prompt or it was sent MAX_PER times this request
  _card(R, found, results, reps, step) {
    const cards = (R.cards ||= {});
    let best = -1, id = null;
    for (let i = 0; i < results.length; i++) {
      const k = results[i] === STOPPED ? null : pickCard({ call: found[i], result: results[i], repeat: reps[i] });
      if (!k || (cards[k] || 0) >= (MAX_PER[k] ?? 2) || (id && PRIORITY.indexOf(k) >= PRIORITY.indexOf(id))) continue;
      if (this.turns.some((t) => t.text.includes(hint(k, this.style)))) continue;
      best = i; id = k;
    }
    if (best < 0) return;
    results[best] += hint(id, this.style);
    cards[id] = (cards[id] || 0) + 1;
    this.onEvent({ type: "card", id, step });
  }

  // a call's tool name and arguments, repaired in place (harness/argfix.js)
  _fix(c) {
    if (c.error || c.garbage) return;
    const name = fixToolName(c.name, (n) => this.byName.has(n));
    const t = this.byName.get(name);
    if (!t) return;
    if (name !== c.name) { c.asked = c.name; c.name = name; }
    c.arguments = fixArgs(c.arguments ?? {}, t.parameters);
  }

  // a step's tool results as one user turn (hcore: the array too, for the core's tool messages)
  _pushResults(results, briefs, req) {
    const t = { role: "user", text: toolResponses(results), req, calls: briefs };
    if (this.model) t.results = results.slice();
    this.turns.push(t);
  }
  // a call left open (the length cap, or the answer ended mid-call), hcore: a write_file keeps its
  // complete lines and runs (the model is told where to pick up); anything else becomes a call with
  // what arguments it had and an error result, so history shows what the model wrote and why it
  // did not run
  _openCall(open, u) {
    if (open.name === "write_file") {
      const part = salvageArgs(open.args, this.byName.get("write_file")?.parameters);
      if (part) return { ...part, sampled: JSON.stringify(part.arguments) };
    }
    const args = looseObject(open.args);
    const c = { name: open.name, arguments: args, sampled: JSON.stringify(args), open: true, raw: open.args };
    if (u && (u.reason === "max" || u.reason === "ctx")) c.error = `your answer was cut at ${u.generated} tokens before the call was complete`;
    else {
      const tail = String(open.args || "").slice(-160).replace(/\s+/g, " ").trim();
      c.error = `your answer ended in the middle of a tool call${u ? ` (${u.reason || "stop"} after ${u.generated} tokens)` : ""}.${tail ? ` The call ended with: "${tail}"` : ""}`;
      try { console.warn("[code] call ended early", u, JSON.stringify(String(open.args || "").slice(-300))); } catch { /* no console */ }
    }
    return c;
  }
  // the turns as the legacy path renders them: hcore assistant turns get their calls back as markup
  // in the model's format (a session switched back to ?hcore=0 keeps working)
  _legacyTurns() {
    if (!this.turns.some((t) => Array.isArray(t.sampled))) return this.turns;
    return this.turns.map((t) => (Array.isArray(t.sampled) ? { ...t, text: legacyText(t, this.style) } : t));
  }
  _turnSize(t) {
    if (!Array.isArray(t.sampled)) return this.count(t.text) + 4;
    if (!this.model?.callText) return this.count(legacyText(t, this.style)) + 4;
    // the calls as the template renders them (an XML value is its text, not escaped JSON)
    return this.count(t.text) + t.sampled.reduce((n, c) => n + this.count(this.model.callText(c)), 0) + 4;
  }
  _size() { return this.count(this.system) + this.turns.reduce((n, t) => n + this._turnSize(t), 0); }

  // Past the budget, compact down to 60% of it. Every compaction changes the middle of the prompt,
  // so everything after the first changed turn is prefilled again (the system prompt + tools stay
  // cached: the model adapters checkpoint them, issue #73). It should be rare and free a lot:
  // 1. stub old tool results, 2. fold finished requests into one line each, 3. drop the oldest
  // folded requests, 4. give up ("full"). The current request's own turns are only ever stubbed.
  // Each step works oldest first and produces a stable form (stubResults, _foldLine: functions of
  // the turn alone), so a compacted turn renders the same on every later step and the turns
  // before the first changed one (the `at` of the "compacted" event) keep their cached prefix.
  _compact(cur) {
    const B = this.budget();
    const before = this._size();
    if (before <= B) return null;
    const target = B * 0.6;
    let tier = 0, at = Infinity;
    // 1. tool results older than the last 2 steps, oldest first
    const res = this.turns.map((t, i) => i).filter((i) => this.turns[i].role === "user" && this.turns[i].text.startsWith("<tool_response>"));
    let cut = 0;
    for (const i of res.slice(0, -2)) {
      if (this._size() <= target) break;
      const t = this.turns[i];
      if (Array.isArray(t.results)) {
        const r = stubResultList(t.results, t.calls);
        if (r.some((x, k) => x !== t.results[k])) { t.results = r; t.text = toolResponses(r); cut++; tier = 1; at = Math.min(at, i); }
        continue;
      }
      const text = stubResults(t.text, t.calls);
      if (text !== t.text) { t.text = text; cut++; tier = 1; at = Math.min(at, i); }
    }
    if (cut) this.onEvent({ type: "trimmed", turns: cut });
    // 2. fold earlier finished requests, oldest first
    const earlier = [...new Set(this.turns.map((t) => t.req))].filter((r) => r !== cur && this.reqs[r]);
    for (const r of earlier) {
      if (this._size() <= target) break;
      const idx = this.turns.map((t, i) => (t.req === r ? i : -1)).filter((i) => i >= 0);
      if (idx.length <= 2 && this.turns[idx[idx.length - 1]]?.folded) continue;
      const user = this.turns[idx[0]];
      const line = { role: "assistant", text: this._foldLine(r), req: r, folded: true };
      this.turns.splice(idx[0], idx.length, { ...user }, line);
      tier = 2; at = Math.min(at, idx[0] + 1);
    }
    // 3. drop the oldest earlier requests whole
    for (const r of earlier) {
      if (this._size() <= target) break;
      const i = this.turns.findIndex((t) => t.req === r);
      if (i >= 0) at = Math.min(at, i);
      this.turns = this.turns.filter((t) => t.req !== r);
      delete this.reqs[r];
      tier = 3;
    }
    const after = this._size();
    if (tier) this.onEvent({ type: "compacted", tier, before, after, at });
    if (after > B) { this.onEvent({ type: "compacted", tier: 4, before, after }); return "full"; }
    return tier;
  }
  // "[earlier: wrote index.html, game.js; edited game.js; served :5173; 3 other calls]" + the answer
  _foldLine(r) {
    const R = this.reqs[r], wrote = [], edited = [], served = [];
    let other = 0;
    for (const c of R.calls) {
      const p = c.arguments?.path;
      if (c.name === "write_file" && p) { if (!wrote.includes(p)) wrote.push(p); }
      else if (c.name === "edit_file" && p) { if (!edited.includes(p)) edited.push(p); }
      else if (c.name === "serve") { const s = ":" + (c.arguments?.port || 5173); if (!served.includes(s)) served.push(s); }
      else other++;
    }
    const parts = [];
    if (wrote.length) parts.push("wrote " + wrote.slice(0, 12).join(", ") + (wrote.length > 12 ? ` (+${wrote.length - 12})` : ""));
    if (edited.length) parts.push("edited " + edited.slice(0, 12).join(", ") + (edited.length > 12 ? ` (+${edited.length - 12})` : ""));
    if (served.length) parts.push("served " + served.join(", "));
    if (other) parts.push(`${other} other call${other > 1 ? "s" : ""}`);
    return `[earlier: ${parts.join("; ") || "no tool calls"}]\n` + (R.answer || "").slice(0, 600);
  }

  // Session state. Assistant turns carry the ids they were sampled as when the model knows them.
  toJSON() {
    const tag = this._tag();
    return {
      v: 1, req: this.req, reqs: this.reqs, ...(tag ? { tok: tag } : {}),
      turns: this.turns.map((t) => {
        const o = { ...t };
        if (t.role === "assistant" && this.model) { const v = this.model.idsOf?.(t); if (v) { o.ids = Array.from(v.ids); o.idsEnd = v.thinkEnd; } }
        else if (t.role === "assistant" && !Array.isArray(t.sampled)) { const ids = this.idsFor?.(t.text); if (ids) o.ids = Array.from(ids); }
        return o;
      }),
    };
  }
  _tag() { try { return this.idsTag?.() ?? null; } catch { return null; } }
  static from(json, opts) {
    const a = new Agent(opts);
    if (json?.v !== 1) return a;
    a.req = json.req || 0; a.reqs = json.reqs || {};
    const same = (json.tok ?? null) === a._tag();   // ids from another tokenizer (or an untagged save) are noise
    a.turns = (json.turns || []).map(({ ids, idsEnd, ...t }) => {
      if (same && ids && t.role === "assistant") {
        if (a.model) { if (Array.isArray(t.sampled)) a.model.adoptIds?.(t, { ids, thinkEnd: idsEnd || 0 }); }
        else if (!Array.isArray(t.sampled)) a.adopt?.(t.text, ids);
      }
      return t;
    });
    return a;
  }
}

// A write_file call cut by the length cap: its content up to the last complete line, as a call that
// can run (appending when the model asked to append), with where it stopped. null when the cut
// call is anything else or has no complete line yet.
export function salvageWrite(raw, schemaFor = () => null) {
  if (!raw) return null;
  let c = parseCallBody(raw, schemaFor);
  // a JSON call cut inside its content string: close the string where it was cut
  if (c?.error && /^\s*\{/.test(raw)) {
    try { const o = parseLooseJSON(raw.trim(), { open: true }); c = { name: o?.name, arguments: fixArgs(o?.arguments ?? o?.parameters ?? {}, schemaFor("write_file")) }; } catch { return null; }
  }
  return salvageCall(c);
}
// hcore: an open write_file's arguments as CallStream streamed them (JSON text, cut anywhere)
export function salvageArgs(args, schema = null) {
  if (!args) return null;
  let o;
  try { o = parseLooseJSON(String(args).trim(), { open: true }); } catch { return null; }
  if (!o || typeof o !== "object" || Array.isArray(o)) return null;
  return salvageCall({ name: "write_file", arguments: fixArgs(o, schema) });
}
function salvageCall(c) {
  const a = c?.arguments;
  if (c?.name !== "write_file" || !a || typeof a.path !== "string" || !a.path.trim() || typeof a.content !== "string") return null;
  const cut = a.content.lastIndexOf("\n");
  if (cut < 0) return null;
  const content = a.content.slice(0, cut + 1), lines = content.split("\n").length - 1;
  const last = content.slice(0, -1).split("\n").pop();
  return { name: "write_file", arguments: { path: a.path.trim(), content, append: a.append === true }, salvage: { lines, last } };
}

// Calls written as bare JSON objects outside <tool_call> (in a ```json fence or plain text):
// {"name": "write_file", "arguments": {...}}. Only known tool names (or their usual other names)
// with an arguments object count.
export function bareJsonCalls(text, byName, spans = null) {
  const out = [];
  if (!text || !byName?.size) return out;
  const re = /\{\s*"(?:name|tool|function)"\s*:/g;
  let m;
  while ((m = re.exec(text))) {
    let o;
    try { o = parseLooseJSON(text.slice(m.index)); } catch { continue; }
    const c = parseCallBody(JSON.stringify(o));
    if (c.error) continue;
    const name = fixToolName(c.name, (n) => byName.has(n));
    if (!byName.has(name) || !c.arguments || typeof c.arguments !== "object") continue;
    out.push({ name, arguments: c.arguments });
    spans?.push([m.index, jsonEnd(text, m.index)]);
    re.lastIndex = Math.max(re.lastIndex, jsonEnd(text, m.index));   // (not the objects inside this one)
  }
  return out;
}

// the index just past the JSON object that starts at `at` (strings respected), or the text's end
function jsonEnd(text, at) {
  let depth = 0, inStr = false, esc = false;
  for (let i = at; i < text.length; i++) {
    const c = text[i];
    if (inStr) { if (esc) esc = false; else if (c === "\\") esc = true; else if (c === '"') inStr = false; continue; }
    if (c === '"') inStr = true;
    else if (c === "{" || c === "[") depth++;
    else if ((c === "}" || c === "]") && --depth === 0) return i + 1;
  }
  return text.length;
}

// Calls written as bare tags named after a known tool, children named after its parameters:
// <write_file>\n<path>a.html</path>\n<content>\n...\n</content>\n</write_file>. Only known
// tools and parameters count, so HTML the model merely quotes is left alone.
// Calls written as XML <function=NAME> ... </function> blocks without a well-formed <tool_call> around
// them (the opener garbled, "<tool_tool_calls>", seen from Qwen 3.6 on a follow-up request): parsed
// like a call body. Only known tools, and only complete blocks.
export function bareFunctionCalls(text, byName, spans = null) {
  const out = [];
  if (!text || !byName?.size) return out;
  text = normalizeXmlCall(text);
  const re = /<function=([^>\s]+)>[\s\S]*?<\/function>/g;
  let m;
  while ((m = re.exec(text))) {
    const c = parseCallBody(m[0], (n) => byName.get(n)?.parameters);
    if (c.error) continue;
    const name = fixToolName(c.name, (n) => byName.has(n));
    if (!byName.has(name)) continue;
    out.push({ name, arguments: c.arguments || {} });
    spans?.push([m.index, m.index + m[0].length]);
  }
  return out;
}

export function bareCalls(text, byName, spans = null) {
  const out = [];
  if (!text || !byName?.size) return out;
  const names = [...byName.keys()].map((n) => n.replace(/[^\w-]/g, "")).join("|");
  const re = new RegExp("<(" + names + ")>([\\s\\S]*?)</\\1>", "g");
  let m;
  while ((m = re.exec(text))) {
    const t = byName.get(m[1]), props = t?.parameters?.properties || {}, args = {};
    let found = 0;
    for (const p of Object.keys(props)) {
      const pm = new RegExp("<" + p + ">\\n?([\\s\\S]*?)\\n?</" + p + ">").exec(m[2]);
      if (!pm) continue;
      const ty = props[p]?.type;
      args[p] = ty === "integer" || ty === "number" ? Number(pm[1].trim()) : ty === "boolean" ? pm[1].trim() === "true" : pm[1];
      found++;
    }
    const req = t?.parameters?.required || [];
    if (found && req.every((r) => r in args)) { out.push({ name: m[1], arguments: args }); spans?.push([m.index, m.index + m[0].length]); }
  }
  return out;
}

// hcore: the calls a text answer wrote outside the call format (bare tags, a bare JSON object, a
// <function=...> block with its opener garbled or missing), found as the legacy path finds them,
// and the text with them taken out (with the fences and stray call tags left around them), so the
// history does not show them twice. -> { calls, text }
export function bareFromText(text, byName) {
  if (!text || !byName?.size) return { calls: [], text };
  let spans = [], src = text;
  let calls = bareCalls(src, byName, spans);
  if (!calls.length) { spans = []; calls = bareJsonCalls(src, byName, spans); }
  if (!calls.length) { spans = []; src = normalizeXmlCall(text); calls = bareFunctionCalls(src, byName, spans); }
  if (!calls.length) return { calls: [], text };
  let rest = "", at = 0;
  for (const [a, b] of spans.sort((x, y) => x[0] - y[0])) { if (a < at) continue; rest += src.slice(at, a); at = b; }
  rest += src.slice(at);
  rest = rest.replace(/```[\w-]*\s*```/g, "").replace(/^[ \t]*<\/?[\w.]*tool[\w.]*>[ \t]*$/gm, "").replace(/\n{3,}/g, "\n\n").trim();
  return { calls, text: rest };
}
