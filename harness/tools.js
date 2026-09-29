// Tool calling for the Qwen chat template: how tools are described to the model, how its tool
// calls are recognised in the streamed answer, and how results go back. DOM-free and
// engine-free so the harness (and the unit tests) can use it anywhere.
//
// Two formats exist in the Qwen family, and a model is trained on exactly one of them:
//   "json" (Qwen2.5 / Qwen3, Hermes-style):
//       <tool_call>
//       {"name": "read_file", "arguments": {"path": "src/a.js"}}
//       </tool_call>
//   "xml" (Qwen3-Coder, Qwen3.5 and later):
//       <tool_call>
//       <function=read_file>
//       <parameter=path>
//       src/a.js
//       </parameter>
//       </function>
//       </tool_call>
// Results go back in a user turn as <tool_response> ... </tool_response> blocks, one per call,
// in call order. detectStyle() reads the GGUF's own chat template (tokenizer.chat_template) so the
// prompt matches what the model was trained on; the parser accepts both formats either way, plus
// the near misses room models write: a ```json fence inside the call, JSON arguments inside
// <function=NAME>, <function name="x"> / <parameter name="p"> / <invoke> spellings, and several
// calls in one <tool_call> block (splitCallBody).

// tools: [{ name, description, parameters: JSON schema object }]

export function detectStyle(chatTemplate = "") {
  return /<function=|<parameter=/.test(chatTemplate) ? "xml" : "json";
}

const fnJSON = (t) => JSON.stringify({ type: "function", function: { name: t.name, description: t.description || "", parameters: t.parameters || { type: "object", properties: {} } } });

// The system prompt with the tool block appended, as the Qwen templates render it.
export function toolsSystemPrompt(tools, { style = "json", system = "" } = {}) {
  if (!tools || !tools.length) return system;
  const list = "# Tools\n\nYou have access to the following functions:\n\n<tools>\n" + tools.map(fnJSON).join("\n") + "\n</tools>";
  if (style === "xml") {   // Qwen3.5+ templates: tool block first, the caller's system text appended
    return list + "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
      + "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
      + "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
      + "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
      + "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
      + "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"
      + (system ? "\n\n" + system : "");
  }
  return (system ? system + "\n\n" : "") + list + "\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    + "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
}

// The text of a tool-results user turn: one <tool_response> block per result, in call order.
export function toolResponses(results) {
  return results.map((r) => "<tool_response>\n" + (typeof r === "string" ? r : JSON.stringify(r)) + "\n</tool_response>").join("\n");
}

// An assistant turn's tool calls rendered back in the model's format (for history rebuilt from
// structured calls; a live session should keep the sampled ids instead, see room/conversation.js).
export function renderCalls(calls, style = "json") {
  return calls.map((c) => style === "xml"
    ? "<tool_call>\n<function=" + c.name + ">\n" + Object.entries(c.arguments || {}).map(([k, v]) =>
      "<parameter=" + k + ">\n" + (typeof v === "string" ? v : JSON.stringify(v)) + "\n</parameter>\n").join("") + "</function>\n</tool_call>"
    : "<tool_call>\n" + JSON.stringify({ name: c.name, arguments: c.arguments || {} }) + "\n</tool_call>").join("\n");
}

// Parse one <tool_call> body (the text between the tags) in either format.
// Returns { name, arguments } or { error, raw } when the model produced something malformed.
export function parseCallBody(body, schemaFor = () => null) {
  const raw = body;
  const b = unfence(normalizeXmlCall(body).trim());
  const fm = /^<function=([^>\s]+)>([\s\S]*?)(?:<\/function>\s*)?$/.exec(b);
  if (fm) {
    const name = fm[1], args = {};
    const props = schemaFor(name)?.properties || {};
    // a missing </parameter> before the next parameter (on a line of its own) or </function> is tolerated
    // (seen in the wild); a <parameter= inside a line is the value's own text
    const re = /<parameter=([^>\s]+)>\n?([\s\S]*?)(?:\n?<\/parameter>|\n(?=<parameter=)|\n?$)/g;
    let m;
    while ((m = re.exec(fm[2]))) args[m[1]] = coerce(m[2], props[m[1]]);
    // no <parameter=> at all but a JSON object: <function=serve>\n{"dir": "."}\n</function>
    if (!Object.keys(args).length && /^\s*\{/.test(fm[2])) {
      try { const o = parseLooseJSON(unfence(fm[2].trim())); if (o && typeof o === "object" && !Array.isArray(o)) return { name, arguments: o.arguments ?? o.parameters ?? o }; } catch { /* no arguments */ }
    }
    return { name, arguments: args };
  }
  try {
    let o = parseLooseJSON(b);
    if (Array.isArray(o) && o.length === 1) o = o[0];   // [{"name": ...}]
    if (o && typeof o.function === "object" && o.function) o = o.function;   // OpenAI's {"type": "function", "function": {...}}
    const name = o && [o.name, o.tool, o.function, o.tool_name].find((v) => typeof v === "string");
    if (!name) return { error: "tool call has no name", raw };
    const k = ["arguments", "parameters", "args", "input", "params"].find((x) => o[x] !== undefined);
    let a;
    if (k) a = o[k];
    else {
      // the arguments written next to the name: {"name": "serve", "dir": "."}
      a = {};
      for (const [x, v] of Object.entries(o)) if (!["name", "tool", "function", "tool_name", "type", "id"].includes(x)) a[x] = v;
    }
    if (typeof a === "string") { try { a = parseLooseJSON(a); } catch { /* leave as text */ } }
    return { name: name.trim(), arguments: a ?? {} };
  } catch (e) {
    const msg = String(e.message || e);
    return { error: "tool call is not valid JSON: " + (msg.length > 120 ? msg.slice(0, 120) + "…" : msg), raw };
  }
}

// The XML call spellings other models use, as the Qwen one: <function name="x"> / <invoke name="x">
// for <function=x>, <parameter name="p"> for <parameter=p>, and a quoted <function="x">. Only tags
// at the start of a line (or right after the opener) change, so a value that merely quotes one is
// left alone.
export function normalizeXmlCall(text) {
  if (!/<(?:function|invoke|parameter)[\s="']/.test(text)) return text;
  return text
    .replace(/(^|\n)([ \t]*)<(?:function|invoke)\s+name\s*=\s*["']?([^"'>\s]+)["']?\s*>/g, "$1$2<function=$3>")
    .replace(/(^|\n)([ \t]*)<\/invoke>/g, "$1$2</function>")
    .replace(/(^|\n)([ \t]*)<function=["']([^"'>\s]+)["']\s*>/g, "$1$2<function=$3>")
    .replace(/(^|\n)([ \t]*)<parameter\s+name\s*=\s*["']?([^"'>\s]+)["']?\s*>/g, "$1$2<parameter=$3>")
    .replace(/(^|\n)([ \t]*)<parameter=["']([^"'>\s]+)["']\s*>/g, "$1$2<parameter=$3>");
}

// a call body wrapped in a Markdown fence (```json ... ```), as small models write it
const unfence = (s) => s.replace(/^```[\w-]*[ \t]*\n?/, "").replace(/\n?```\s*$/, "");

// One <tool_call> body that holds several calls, as separate bodies: <function=a>..</function>
// <function=b>..</function> (a stray <tool_call> between them, the closer forgotten, is dropped),
// a JSON array of calls, or JSON objects one after another. A body with one call comes back as is.
export function splitCallBody(body) {
  const b = normalizeXmlCall(body);
  // (both tags on lines of their own: a value that merely contains them stays one call)
  const between = /(?<=(?:^|\n)[ \t]*<\/function>)[ \t]*\n\s*(?:<\/?tool_call>\s*)*(?=<function=[^>\s<]+>)/;
  if (between.test(b)) return b.split(new RegExp(between.source, "g")).filter((s) => s.trim());
  const t = unfence(b.trim());
  if (!t.startsWith("{") && !t.startsWith("[")) return [body];
  const named = (o) => o && typeof o === "object" && [o.name, o.tool, o.function?.name ?? o.function].some((v) => typeof v === "string");
  try {
    const o = JSON.parse(t);
    if (Array.isArray(o) && o.length > 1 && o.every(named)) return o.map((x) => JSON.stringify(x));
    return [body];
  } catch { /* maybe several objects */ }
  const parts = [];
  let at = 0;
  while (at < t.length) {
    const s = t.indexOf("{", at);
    if (s < 0 || t.slice(at, s).trim()) break;
    const e = objectEnd(t, s);
    if (e < 0) break;
    parts.push(t.slice(s, e));
    at = e;
  }
  if (parts.length > 1 && !t.slice(at).trim()) {
    try { if (parts.every((p) => named(JSON.parse(p)))) return parts; } catch { /* not clean JSON */ }
  }
  return [body];
}

// the index just past the JSON object that starts at `at` (strings respected), or -1 when it never closes
function objectEnd(text, at) {
  let depth = 0, inStr = false, esc = false;
  for (let i = at; i < text.length; i++) {
    const c = text[i];
    if (inStr) { if (esc) esc = false; else if (c === "\\") esc = true; else if (c === '"') inStr = false; continue; }
    if (c === '"') inStr = true;
    else if (c === "{" || c === "[") depth++;
    else if ((c === "}" || c === "]") && --depth === 0) return i + 1;
  }
  return -1;
}

// JSON.parse, then the near misses small models write for a call: extra or missing closing braces,
// and a stray "{" before "arguments" ({"name": "x", {"arguments": {...}}}). Strings are respected.
// Also: raw newlines / tabs inside strings (escaped), trailing commas, and with { open: true } a
// string the answer ended inside (closed there, for salvaging a cut write).
export function parseLooseJSON(s, { open = false } = {}) {
  try { return JSON.parse(s); } catch (first) {
    let t = s.trim().replace(/("name"\s*:\s*"[^"]*"\s*,\s*)\{\s*(?="(?:arguments|parameters)"\s*:)/, "$1");
    // balance braces outside strings: drop unmatched closers, close what is still open
    let out = "", depth = 0, inStr = false, esc = false;
    const stack = [];
    for (const c of t) {
      if (inStr) {
        if (esc) { esc = false; out += c; continue; }
        if (c === "\\") { esc = true; out += c; continue; }
        if (c === '"') inStr = false;
        out += c === "\n" ? "\\n" : c === "\r" ? "\\r" : c === "\t" ? "\\t" : c;
        continue;
      }
      if (c === '"') { inStr = true; out += c; continue; }
      if (c === "{" || c === "[") { depth++; stack.push(c === "{" ? "}" : "]"); }
      if (c === "}" || c === "]") {
        if (depth === 0) continue;
        depth--; stack.pop();
        out = out.replace(/,\s*$/, "");   // a trailing comma
      }
      out += c;
      if (depth === 0 && out.trim().startsWith("{") && (c === "}")) break;   // one object: ignore what follows it
    }
    if (inStr) {
      if (!open) throw first;
      if (esc) out = out.slice(0, -1);   // a cut escape
      out += '"';
    }
    out = out.replace(/,\s*$/, "").replace(/(?:,|:)\s*$/, (m) => (m.trim() === ":" ? ": null" : ""));
    while (stack.length) out += stack.pop();
    return JSON.parse(out);
  }
}

// XML parameters are text; turn them into the schema's type when it says number / boolean /
// object / array (the Qwen3-Coder parser does the same), else keep the string.
function coerce(text, schema) {
  const t = schema?.type;
  if (t === "string" || !t) {
    if (!t) { try { const v = JSON.parse(text); if (typeof v !== "string") return v; } catch { /* text */ } }
    return text;
  }
  if (t === "integer" || t === "number") { const n = Number(text.trim()); return Number.isFinite(n) ? n : text; }
  if (t === "boolean") { const s = text.trim().toLowerCase(); return s === "true" ? true : s === "false" ? false : text; }
  try { return JSON.parse(text); } catch { return text; }
}

// Streaming recogniser: feed() the decoded answer text as it grows (deltas); get back the text
// that is safe to show (never a half-typed "<tool_c") and any tool calls completed so far.
// end() flushes the rest; an unterminated <tool_call> at the end is reported as an error call.
export class ToolCallParser {
  constructor({ schemaFor = () => null } = {}) {
    this.buf = ""; this.inCall = false; this.calls = []; this.schemaFor = schemaFor;
  }
  feed(delta) {
    this.buf += delta;
    let text = "";
    const calls = [];
    for (;;) {
      if (!this.inCall) {
        if (this.eatNL && this.buf) { if (this.buf[0] === "\n") this.buf = this.buf.slice(1); this.eatNL = false; }   // the newline between calls
        const i = this.buf.indexOf("<tool_call>");
        if (i < 0) {
          // hold back a tail that could be the start of the tag
          const keep = partialTagTail(this.buf, "<tool_call>");
          text += this.buf.slice(0, this.buf.length - keep);
          this.buf = this.buf.slice(this.buf.length - keep);
          break;
        }
        text += this.buf.slice(0, i);
        this.buf = this.buf.slice(i + "<tool_call>".length);
        this.inCall = true;
      } else {
        const j = this.buf.indexOf("</tool_call>");
        if (j < 0) break;
        for (const body of splitCallBody(this.buf.slice(0, j))) {
          const c = parseCallBody(body, this.schemaFor);
          calls.push(c); this.calls.push(c);
        }
        this.buf = this.buf.slice(j + "</tool_call>".length);
        this.inCall = false; this.eatNL = true;
      }
    }
    return { text, calls };
  }
  end() {
    const r = { text: "", calls: [] };
    if (this.inCall) {
      // a model that stops right after </function> without closing the call still meant it; without
      // </function> the answer was cut mid-call (length cap) and its last value is a fragment
      // also accept a call that ends right after a closed parameter (seen from Qwen: no </function>)
      const buf = normalizeXmlCall(this.buf);
      let done = /<\/function>/.test(buf) || /^\s*<function=[^>\s]+>[\s\S]*<\/parameter>\s*$/.test(buf);
      // a JSON call the model ended without </tool_call>: complete if it parses (loosely)
      if (!done && /^\s*\{/.test(this.buf)) { try { const o = parseLooseJSON(this.buf.trim()); done = !!o && typeof o.name === "string" && (o.arguments !== undefined || o.parameters !== undefined); } catch { /* still open */ } }
      const cs = done ? splitCallBody(this.buf).map((b) => parseCallBody(b, this.schemaFor)) : [{ error: "unterminated <tool_call>", raw: this.buf, open: true }];
      r.calls.push(...cs); this.calls.push(...cs);
    } else r.text = this.buf;
    this.buf = ""; this.inCall = false;
    return r;
  }
}

function partialTagTail(s, tag) {
  for (let k = Math.min(tag.length - 1, s.length); k > 0; k--) if (tag.startsWith(s.slice(s.length - k))) return k;
  return 0;
}

