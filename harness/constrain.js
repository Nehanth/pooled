// Tool-call constraint while sampling (docs/design/harness-light.md B.1): decode freely, but once
// a <tool_call> starts, only tokens that keep the call well-formed are allowed. XML style (Qwen3.5+)
// runs a small character automaton over the whole call:
//
//   FREE --"<tool_call>"--> "\n<function=" NAME(tool)">" BODY
//   BODY --"\n<parameter=" (an unused param of fn)">\n" VALUE "</parameter>"--> BODY
//   BODY --"\n</function>" (only when the required params are given)--> "\n</tool_call>" FREE
//
// After </tool_call> the text is free again (the parser ignores trailing text; a stop, a second
// <tool_call> or an invented "\n<tool_response>" are all fine there).
// VALUE is free text except: stop tokens, <tool_call> and </tool_call> anywhere, and <function=,
// </function>, <parameter= at the start of a line (those mean a missing </parameter>; elsewhere
// they can be legitimate file content); integer / number params take digits, boolean params
// true / false. A token is allowed iff running its characters from the current
// state never rejects, so tokens may cross state boundaries (">\n", "</parameter>\n<").
// JSON style (Code mode uses it for Qwen3 1.7B) only limits the "name" string to declared tools.
//
// Masks are a function of the automaton state (not of the text inside a value), computed with one
// vocabulary scan per distinct state and cached per (tokenizer, tools) across steps and requests:
// literal states keep a short allow list, values a short deny list. Outside calls only one thing is
// masked: after "<tool" the tag must go on to <tool_call> (or <tool_response>), never a garbled one.
//
//   const C = new ToolCallConstraint(tools, { vocabSize, tokenText, style, stops, thinking })
//   C.allowed() -> null (anything) | { allow: Int32Array } | { deny: Int32Array }   (sorted ids)
//   C.mask(logits) -> logits;  C.push(tokenText);  C.setText(answerSoFar)  (base for push)
// tokenText(id) is the decoded text of that one token; stops are the end-of-turn ids.

import { compileSchema, stringCapable, SchemaError, SCHEMA_CAPS } from "./jsonschema.js";

const OPEN = "<tool_call>", THINK_END = "</think>", CLOSE_P = "</parameter>";
const FORBID = new Set(["<tool_call>", "</tool_call>", "\n<function=", "\n</function>", "\n<parameter="]);
const PREFIXES = new Set();
for (const p of [CLOSE_P, ...FORBID]) for (let k = 1; k < p.length; k++) PREFIXES.add(p.slice(0, k));
// free text that has written this much of "<tool_call>" ("<tool") is a tag being opened: from there
// only <tool_call> or <tool_response> may follow, so a garbled opener ("<tool_tool_calls>", which the
// parser does not see as a call) cannot be written at all
const GUARD = 5, OPENERS = [OPEN, "<tool_response>"];
const DIGIT = /[0-9]/;
const MAX_KEYS = 512;

const shared = new WeakMap();   // tokenText fn -> Map(signature -> Map(key -> mask))
const has = (a, id) => { let lo = 0, hi = a.length - 1; while (lo <= hi) { const m = (lo + hi) >> 1; if (a[m] === id) return true; if (a[m] < id) lo = m + 1; else hi = m - 1; } return false; };
export const maskHas = (m, id) => !m || (m.allow ? has(m.allow, id) : !has(m.deny, id));

// how far s ends in a prefix of "<tool_call>" (or </think>); the patterns have no inner '<'
const tail = (m, ch, pat) => (ch === pat[m] ? m + 1 : ch === pat[0] ? 1 : 0);

export class ToolCallConstraint {
  // With API options (mode, format, ...) the whole answer follows the strict grammar below
  // (GrammarConstraint); without them this is Code mode's automaton, unchanged.
  constructor(tools, opts) {
    if (opts && (opts.mode !== undefined || opts.format != null || opts.strict)) return new GrammarConstraint(tools, opts);
    const { vocabSize, tokenText, style = "xml", stops = [], thinking = false } = opts;
    this.vocabSize = vocabSize; this.tokenText = tokenText; this.style = style;
    this.stops = new Set(stops); this.thinking = thinking;
    this.fns = new Map();
    for (const t of tools) {
      const props = t.parameters?.properties || {}, names = Object.keys(props);
      const ty = names.map((n) => ({ integer: "i", number: "n", boolean: "b" })[props[n]?.type] || "s");
      let req = 0;
      for (const r of t.parameters?.required || []) if (names.includes(r)) req |= 1 << names.indexOf(r);
      this.fns.set(t.name, { names, ty, req });
    }
    const sig = style + "|" + vocabSize + "|" + [...this.stops].join(",") + "|"
      + [...this.fns].map(([n, f]) => n + ":" + f.names.map((p, i) => p + f.ty[i]).join(",") + ":" + f.req).join(";");
    let byTok = shared.get(tokenText);
    if (!byTok) shared.set(tokenText, (byTok = new Map()));
    if (!byTok.has(sig)) byTok.set(sig, new Map());
    this.cache = byTok.get(sig);
    this.opts = new Map();   // literal options per (tag, fn, given)
    this.setText("");
  }

  // ---- state
  _start() { return this.style === "xml" ? { k: this.thinking ? "T" : "F", m: 0 } : ""; }
  // the answer so far: later push()es extend it. Incremental when it extends the last text.
  setText(t) {
    const b = this.base;
    let from = null;
    if (b && t.startsWith(b.text)) from = b;
    else if (this.clean && t.startsWith(this.clean.text)) from = this.clean;
    let st = from ? from.st : this._start();
    st = this._feed(st, t.slice(from ? from.text.length : 0), true);
    this.base = { text: t, st };
    if (!t.endsWith("�")) this.clean = this.base;
    this.st = st;
  }
  push(s) { this.st = this._feed(this.st, s, true); }
  get text() { return this.base.text; }
  // inside a tool call (the next token is masked)
  get inCall() { return this.style === "xml" ? this.st.k !== "F" && this.st.k !== "T" : this._jsonSlot(this.st) != null; }
  set text(t) { this.setText(t); }   // (older callers)

  // feed text; `lenient`: a character the automaton rejects drops the call back to free text
  _feed(st, s, lenient) {
    if (this.style !== "xml") { const t = st + s; return t.length > 4096 ? t.slice(-2048) : t; }
    for (let i = 0; i < s.length && st; i++) {
      const n = this._step(st, s[i]);
      st = n || (lenient ? { k: "F", m: 0 } : null);
    }
    return st;
  }

  _lit(tag, fn, given, typed, p) { return { k: "L", tag, fn, given, typed, p }; }
  _options(st) {
    const key = st.tag + "|" + st.fn + "|" + st.given + "|" + st.p;
    let o = this.opts.get(key);
    if (o) return o;
    const f = this.fns.get(st.fn);
    o = [];
    if (st.tag === "open") o.push(["\n<function=", () => this._lit("name", "", 0, "")]);
    else if (st.tag === "name") for (const n of this.fns.keys()) o.push([n + ">", () => this._lit("body", n, 0, "")]);
    else if (st.tag === "body") {
      if (f.names.some((_, i) => !(st.given & (1 << i)))) o.push(["\n<parameter=", () => this._lit("pname", st.fn, st.given, "")]);
      if ((st.given & f.req) === f.req) o.push(["\n</function>", () => this._lit("close", "", 0, "")]);
    } else if (st.tag === "pname") {
      f.names.forEach((n, i) => {
        if (st.given & (1 << i)) return;
        const g = st.given | (1 << i), ty = f.ty[i];
        o.push([n + ">\n", () => (ty === "s" ? { k: "V", fn: st.fn, given: g, pm: "" }
          : ty === "b" ? this._lit("bool", st.fn, g, "") : { k: "N", fn: st.fn, given: g, ty, d: 0, sign: 0 })]);
      });
    } else if (st.tag === "bool" || st.tag === "vclose") {
      const vals = st.tag === "bool" ? ["true", "false"] : [""];
      for (const v of vals) for (const c of ["\n" + CLOSE_P, CLOSE_P]) o.push([v + c, () => this._lit("body", st.fn, st.given, "")]);
    } else if (st.tag === "close") o.push(["\n</tool_call>", () => ({ k: "F", m: 0 })]);
    this.opts.set(key, o);
    return o;
  }

  // one character: the next state, or null when the call cannot continue with it
  _step(st, ch) {
    switch (st.k) {
      case "T": { const m = tail(st.m, ch, THINK_END); return m === THINK_END.length ? { k: "F", m: 0 } : m === st.m && m === 0 ? st : { k: "T", m }; }
      case "F": { const m = tail(st.m, ch, OPEN); return m === OPEN.length ? this._lit("open", "", 0, "") : m === st.m && m === 0 ? st : { k: "F", m }; }
      case "L": {
        const typed = st.typed + ch;
        let live = false;
        for (const [s, to] of this._options(st)) {
          if (s === typed) return to();
          if (s.startsWith(typed)) live = true;
        }
        return live ? { ...st, typed } : null;
      }
      case "V": {
        if (isSym(ch)) return null;   // no tag token inside a value
        if (!st.pm && ch !== "<" && ch !== "\n") return st;
        const s = st.pm + ch;
        if (s === CLOSE_P) return this._lit("body", st.fn, st.given, "");
        if (FORBID.has(s)) return null;
        let pm = "";
        for (let k = 0; k < s.length; k++) if (PREFIXES.has(s.slice(k))) { pm = s.slice(k); break; }
        return pm === st.pm ? st : { ...st, pm };
      }
      case "N": {
        if (DIGIT.test(ch) || (st.ty === "n" && ch === "." && st.d)) return st.d ? st : { ...st, d: 1 };
        if (ch === "-" && !st.d && !st.sign) return { ...st, sign: 1 };
        if (!st.d) return null;
        return this._step(this._lit("vclose", st.fn, st.given, ""), ch);
      }
    }
    return null;
  }

  _key(st) {
    if (this.style !== "xml") { const s = this._jsonSlot(st); return s ? "J|" + s.typed : null; }
    if (st.k === "F") return st.m >= GUARD ? "F|" + st.m : null;
    if (st.k === "T") return null;
    if (st.k === "L") return `L|${st.tag}|${st.fn}|${st.given}|${st.typed}`;
    if (st.k === "V") return `V|${st.fn}|${st.given}|${st.pm}`;
    return `N|${st.fn}|${st.given}|${st.ty}|${st.d}|${st.sign}`;
  }
  _jsonSlot(t) {
    const call = t.lastIndexOf(OPEN);
    if (call < 0 || t.indexOf("</tool_call>", call) >= 0) return null;
    const j = /"name"\s*:\s*"([^"]*)$/.exec(t.slice(call));
    return j ? { typed: j[1] } : null;
  }
  _okJson(typed, w) {
    for (const n of this.fns.keys()) {
      const r = (n + "\"").slice(typed.length);
      if ((n + "\"").startsWith(typed) && (r.startsWith(w) || w.startsWith(r))) return true;
    }
    return false;
  }

  // the mask for the current state (cached), or null when anything goes
  allowed() {
    const key = this._key(this.st);
    if (key == null) return null;
    let m = this.cache.get(key);
    if (m !== undefined) { this.cache.delete(key); this.cache.set(key, m); return m; }
    m = this._scan();
    this.cache.set(key, m);
    if (this.cache.size > MAX_KEYS) this.cache.delete(this.cache.keys().next().value);
    return m;
  }
  _scan() {
    const st = this.st, json = this.style !== "xml";
    const slot = json ? this._jsonSlot(st) : null;
    const allow = [], deny = [], first = new Map();
    if (st.k === "F") {
      // free text ending in "<tool" (a tag being opened): the tag can only become <tool_call> (or the
      // <tool_response> the adapter cuts at). "<tool" then a token not starting with "_" is still
      // free ("<toolbar>"); stop tokens stay allowed.
      const pre = OPEN.slice(0, st.m);
      for (let id = 0; id < this.vocabSize; id++) {
        const w = this.tokenText(id);
        let ok = this.stops.has(id);
        if (!ok && w) {
          const t = pre + w;
          ok = OPENERS.some((o) => o.startsWith(t) || t.startsWith(o)) || (st.m === GUARD && w[0] !== "_");
        }
        (ok ? allow : deny).push(id);
      }
      if (!allow.length) return null;
      return allow.length <= deny.length ? { allow: Int32Array.from(allow) } : { deny: Int32Array.from(deny) };
    }
    for (let id = 0; id < this.vocabSize; id++) {
      let ok;
      if (this.stops.has(id)) ok = false;
      else {
        const w = this.tokenText(id);
        if (!w) ok = false;
        else if (json) ok = this._okJson(slot.typed, w);
        else if (st.k === "V" && !st.pm && w.indexOf("<") < 0 && w.indexOf("\n") < 0) ok = true;   // the common case: plain text
        else {
          // most tokens fail on their first character: step each distinct first character once
          const c = w[0];
          let s1 = first.get(c);
          if (s1 === undefined) first.set(c, (s1 = this._step(st, c)));
          ok = s1 !== null && (w.length === 1 || this._feed(s1, w.slice(1), false) !== null);
        }
      }
      (ok ? allow : deny).push(id);
    }
    if (!allow.length) return null;   // no token fits (odd tokenizer): don't wedge the decoder
    return allow.length <= deny.length ? { allow: Int32Array.from(allow) } : { deny: Int32Array.from(deny) };
  }

  // mask logits in place; this.forced: the unmasked argmax was not allowed
  mask(logits) {
    const m = this.allowed();
    this.forced = false;
    if (!m) return logits;
    let b = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[b]) b = i;
    this.forced = !maskHas(m, b);
    if (m.allow) {
      const keep = Array.from(m.allow, (i) => logits[i]);
      logits.fill(-Infinity);
      m.allow.forEach((i, k) => { logits[i] = keep[k]; });
    } else for (const i of m.deny) logits[i] = -Infinity;
    return logits;
  }
}

// ---------------------------------------------------------------------------------------------
// The strict grammar for API answers (`pooled serve`, docs/design/serve.md "Constraint modes").
// The whole answer, not just the inside of a call, follows it:
//
//   [S0 "<think>"] T ... "</think>"  ANSWER
//   ANSWER, by mode:
//     auto      free text; "<tool_call>" (or, XML, a line-initial "<function=" followed by the start
//               of a declared name: the lazy bare trigger) starts a call; after "</tool_call>" only
//               whitespace, another call or the end (no free text after calls)
//     none      free text; "<tool_call>" and the bare trigger are banned
//     required  whitespace, then a call; more calls or the end after it
//     {name}    as required with the name fixed; the end after it
//     format    whitespace, a JSON value of the schema, the end   (auto + format: a call or the value)
//   CALL, XML: "\n<function=" NAME ">" ( "\n<parameter=" P ">\n" VALUE ("\n")? "</parameter>" )*
//              "\n</function>" "\n</tool_call>"     declared names (narrowed by `allowed`), each
//              parameter at most once, required ones before </function>; VALUE is raw text for a
//              string-capable schema, one of the raw enum strings, else a JSON value of the schema
//   CALL, JSON: "\n{\"name\": \"" NAME "\", \"arguments\": " JSON-VALUE(parameters) "}\n</tool_call>"
//   T (reasoning) is free; in the forcing modes (required, named, format only) its end of turn is
//   banned. For XML profiles a "<tool_call>" in T closes the reasoning (Qwen3.6 opens calls there);
//   for the JSON style it is a draft inside the reasoning and changes nothing.
//   After the allowed number of calls (max_tool_calls; 1 when parallel_tool_calls is false or the
//   tool is named; at most 16) only the end of turn is left, so the answer ends on an end token.
//
// JSON values (harness/jsonschema.js node table): declared keys in any order, required ones before
// "}", strings with valid escapes and no raw control characters, the JSON number grammar, depth
// <= 32. Whitespace per gap: at most one space, or one "\n" followed by at most 4 x (depth + 1)
// spaces (pretty-printed JSON passes, newline floods do not).
//
// Masks are cached per state key in one LRU per tokenizer with a byte budget (MASK_BYTES), keyed by
// a hash of everything that shapes the grammar. Before scanning the vocabulary for a new state,
// mask() tries the model's own top candidates: when at least k of the top 64 are allowed (k = the
// sampler's top-k, 1 when greedy), masking just those is exact, and no scan is needed.
// ---------------------------------------------------------------------------------------------

export const MASK_BYTES = 64 << 20;
export const MAX_CALLS = 16;
const THINK_OPEN = "<think>", RESPONSE = "<tool_response>", FN = "<function=", CLOSE_TC = "</tool_call>";
// Tags that are the model's own special tokens are atomic symbols in the grammar (a private-use
// character each), so structure can only be written with the real token: a tag spelled out in text
// pieces ("<" "tool" "_call" ">") is not one (models copy spelled tags badly, and the parser, which
// reads decoded text, would see a call the grammar did not).
export const TAG_SYMBOLS = { "<tool_call>": "\uE000", "</tool_call>": "\uE001", "<think>": "\uE002", "</think>": "\uE003", "<tool_response>": "\uE004", "</tool_response>": "\uE005" };
const isSym = (ch) => ch >= "\uE000" && ch <= "\uE0FF";
const SYM_RE = /[\uE000-\uE0FF]/;
const NAME_RE = /^[A-Za-z0-9_.:-]{1,128}$/, PARAM_RE = /^[^<>\n\r]{1,128}$/;

// one LRU of masks per tokenizer (tokenText function), bounded in bytes
class MaskLRU {
  constructor(budget) { this.budget = budget; this.bytes = 0; this.m = new Map(); }
  get(k) { const v = this.m.get(k); if (v !== undefined) { this.m.delete(k); this.m.set(k, v); } return v; }
  set(k, v) {
    const old = this.m.get(k);
    if (old !== undefined) { this.bytes -= sizeOf(old); this.m.delete(k); }
    this.m.set(k, v); this.bytes += sizeOf(v);
    while (this.bytes > this.budget && this.m.size > 1) { const [k0, v0] = this.m.entries().next().value; this.m.delete(k0); this.bytes -= sizeOf(v0); }
  }
  get size() { return this.m.size; }
}
const sizeOf = (m) => 64 + (m ? (m.allow || m.deny).length * 4 : 0);
const lrus = new WeakMap();
export function maskCacheFor(tokenText, budget = MASK_BYTES) {
  let c = lrus.get(tokenText);
  if (!c) lrus.set(tokenText, (c = new MaskLRU(budget)));
  return c;
}

// a 64-bit-ish hash of a string, for cache signatures
export function hash64(s) {
  let h1 = 0x811c9dc5, h2 = 0x9e3779b9;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 16777619);
    h2 = Math.imul(h2 ^ c, 2246822519) ^ (h2 >>> 13);
  }
  return (h1 >>> 0).toString(36) + (h2 >>> 0).toString(36) + s.length.toString(36);
}

// merge a compiled schema table into `dst`, returning the new root id
function appendTable(dst, { nodes, root }) {
  const off = dst.length;
  const r = (id) => (id == null ? id : id + off);
  for (const n of nodes) {
    const c = { ...n };
    if (n.k === "obj") { c.props = n.props.map(([k, id]) => [k, r(id)]); c.free = r(n.free); }
    else if (n.k === "arr") { c.items = r(n.items); c.prefix = n.prefix.map(r); }
    else if (n.k === "u") for (const k of ["str", "num", "bool", "null", "obj", "arr"]) c[k] = r(n[k]);
    dst.push(c);
  }
  return root + off;
}

const WS = (c) => c === " " || c === "\n";
const DIG = (c) => c >= "0" && c <= "9";
const HEX = (c) => DIG(c) || (c >= "a" && c <= "f") || (c >= "A" && c <= "F");
const tailM = (m, ch, pat) => (ch === pat[m] ? m + 1 : ch === pat[0] ? 1 : 0);

export class GrammarConstraint {
  constructor(tools, { vocabSize, tokenText, style = "xml", stops = [], thinking = false, thinkInPrompt = false, mode = "auto",
    allowed = null, maxCalls = null, parallel = true, format = null, maskCache = null, caps = SCHEMA_CAPS, tags = null } = {}) {
    // tags: { "<tool_call>": id, ... } the special ids of the tags (none: tags are plain text)
    this.idSym = new Map(); this.special = new Set();
    for (const [tag, id] of Object.entries(tags || {})) if (Number.isInteger(id) && TAG_SYMBOLS[tag]) { this.idSym.set(id, TAG_SYMBOLS[tag]); this.special.add(tag); }
    const sym = (tag) => (this.special.has(tag) ? TAG_SYMBOLS[tag] : tag);
    this.S = { open: sym(OPEN), close: sym(CLOSE_TC), think: sym(THINK_OPEN), thinkEnd: sym(THINK_END) };
    // the grammar's view of a token: its text, or the symbol of the tag it is
    this.tt = (id) => this.idSym.get(id) ?? tokenText(id);
    this.vocabSize = vocabSize; this.tokenText = this.tt; this.style = style === "json" ? "json" : "xml";
    this.stops = new Set(stops); this.thinking = !!thinking; this.thinkInPrompt = !!thinkInPrompt;
    this.strict = true;
    tools = tools || [];
    this.named = mode && typeof mode === "object" ? String(mode.name) : null;
    this.mode = this.named ? "named" : ["auto", "none", "required"].includes(mode) ? mode : "auto";
    this.nodes = [];
    this.fns = new Map();
    for (const t of tools) this._addTool(t, caps);
    let names = [...this.fns.keys()];
    if (allowed) names = names.filter((n) => allowed.includes(n));
    if (this.named) names = names.filter((n) => n === this.named);
    this.names = names;
    const canCall = this.mode !== "none" && names.length > 0;
    this.limit = !canCall ? 0 : this.named || parallel === false ? 1 : Math.max(1, Math.min(MAX_CALLS, maxCalls ?? MAX_CALLS));
    this.fmtRoot = null;
    if (format) {
      const schema = format.type === "schema" ? format.schema : { type: "object" };
      this.fmtRoot = appendTable(this.nodes, compileSchema(schema, { caps }));
    }
    if (this.nodes.length > caps.nodes) throw new SchemaError(`the tool schemas are too large together (more than ${caps.nodes} nodes)`);
    // what the answer starts with (after the reasoning)
    const forcedCall = canCall && (this.mode === "required" || this.mode === "named");
    this.start = forcedCall ? "call" : this.fmtRoot != null ? (canCall && this.mode === "auto" ? "union" : "json") : "free";
    this.forcing = this.start === "call" || this.start === "json";
    this.canCall = canCall;
    this.bare = this.style === "xml" && names.length > 0;   // (tracked in none mode too, to ban it)
    const sig = JSON.stringify([this.style, vocabSize, [...this.stops].sort(), [...this.idSym].sort(), this.mode, this.named, names, this.limit, this.thinking, this.thinkInPrompt, this.start,
      tools.map((t) => [t.name, t.parameters ?? null]), format ?? null]);
    this.sig = hash64(sig);
    this.cache = maskCache || maskCacheFor(tokenText);
    this.opts = new Map();
    this.broken = false;
    this.forced = false; this.free = false; this.bad = false; this.mass = 1;
    this.setText("");
  }

  _addTool(t, caps) {
    if (!t || typeof t.name !== "string" || !t.name) return;
    const params = t.parameters && typeof t.parameters === "object" ? t.parameters : { type: "object", properties: {} };
    const props = params.properties && typeof params.properties === "object" ? params.properties : {};
    const names = Object.keys(props);
    const f = { names, kinds: [], nodes: [], enums: [], req: "", freeParams: false, args: null };
    const req = new Set(Array.isArray(params.required) ? params.required : []);
    f.req = names.map((n) => (req.has(n) ? "1" : "0")).join("");
    if (this.style === "json") {
      f.args = appendTable(this.nodes, compileSchema(params, { caps }));
    } else {
      names.forEach((n, i) => {
        const s = props[n];
        const sc = stringCapable(s, params);
        let kind = "s", node = null, en = null;
        if (sc === "only") {
          const vals = s && Array.isArray(s.enum) ? s.enum : s && s.const !== undefined ? [s.const] : null;
          if (vals && vals.every((v) => typeof v === "string") && vals.length <= caps.enum && vals.every((v) => !/<\/parameter>|\n<(?:parameter=|\/function>)/.test(v))) { kind = "e"; en = [...new Set(vals)]; }
        } else if (sc === "none") {
          kind = "j"; node = appendTable(this.nodes, compileSchema(s, { caps, doc: params }));
        }
        f.kinds[i] = kind; f.nodes[i] = node; f.enums[i] = en;
      });
      f.freeParams = !names.length && params.additionalProperties !== false;
    }
    this.fns.set(t.name, f);
  }

  // ---- states
  _answerStart() {
    if (this.start === "free") return { k: "F", t: "", ls: 1, fm: "" };
    return { k: "W", n: 0, to: this.start };
  }
  _initial() {
    if (!this.thinking) return this._answerStart();
    return this.thinkInPrompt ? { k: "T", m: 0, o: 0 } : { k: "S0", typed: "" };
  }
  setText(t) {
    const b = this.base;
    let from = null;
    if (b && t.startsWith(b.text)) from = b;
    else if (this.clean && t.startsWith(this.clean.text)) from = this.clean;
    let st = from ? from.st : this._initial();
    st = this._feed(st, t.slice(from ? from.text.length : 0), true);
    this.base = { text: t, st };
    if (!t.endsWith("�")) this.clean = this.base;
    this.st = st;
  }
  push(s) { this.st = this._feed(this.st, s, true); }
  get text() { return this.base.text; }
  get inCall() { const k = this.st.k; return k === "L" || k === "V" || k === "PN" || (k === "J" && this.st.then !== "fmt"); }
  get state() { return this.st; }
  // the answer may end here (an end-of-turn token is allowed)
  endOk(st = this.st) {
    switch (st.k) {
      case "S0": return !this.forcing && st.typed === "";
      case "T": return !this.forcing;
      case "F": return true;
      case "AC": return !st.typed.trim();
      case "E": return true;
      case "J": return st.then === "fmt" && this._jEnd(st.fr);
    }
    return false;
  }

  _feed(st, s, lenient) {
    for (let i = 0; i < s.length && st; i++) {
      const n = this._step(st, s[i]);
      if (n) { st = n; continue; }
      if (!lenient) return null;
      // text the grammar rejects (it was not sampled through this constraint): free from here
      this.broken = true;
      st = { k: "F", t: "", ls: s[i] === "\n" ? 1 : 0, fm: "" };
    }
    return st;
  }

  _lit(tag, o = {}) { return { k: "L", tag, fn: "", given: "", typed: "", nc: 0, p: -1, ...o }; }
  _afterCall(nc) { return nc >= this.limit ? { k: "E", n: 0 } : { k: "AC", typed: "", nc }; }
  _paramStart(st, i) {
    const f = this.fns.get(st.fn);
    const given = st.given.slice(0, i) + "1" + st.given.slice(i + 1);
    const base = { fn: st.fn, given, nc: st.nc };
    const kind = f.kinds[i];
    if (kind === "e") return this._lit("enum", { ...base, p: i });
    if (kind === "j") return { k: "J", fr: [{ t: "v", n: f.nodes[i], w: 0 }], then: "xp", ...base };
    return { k: "V", pm: "", ...base };
  }
  _options(st) {
    const key = st.tag + "|" + st.fn + "|" + st.given + "|" + st.p + "|" + st.nc + "|" + (st.bare ? 1 : 0);
    let o = this.opts.get(key);
    if (o) return o;
    const f = this.fns.get(st.fn);
    const base = { fn: st.fn, given: st.given, nc: st.nc };
    o = [];
    switch (st.tag) {
      case "ropen": o.push([this.S.open, () => this._lit(this.style === "xml" ? "open" : "jopen", { nc: st.nc })]); break;
      case "open": o.push(["\n" + FN, () => this._lit("name", { nc: st.nc })]); break;
      case "name":
        for (const n of this.names) o.push([n + ">", () => this._lit("body", { fn: n, given: "0".repeat(this.fns.get(n).names.length), nc: st.nc, bare: st.bare })]);
        break;
      case "body": {
        const left = f.names.some((_, i) => st.given[i] !== "1");
        if (left || f.freeParams) o.push(["\n<parameter=", () => (f.freeParams ? { k: "PN", typed: "", ...base } : this._lit("pname", { ...base, bare: st.bare }))]);
        let reqOk = true;
        for (let i = 0; i < f.req.length; i++) if (f.req[i] === "1" && st.given[i] !== "1") { reqOk = false; break; }
        if (reqOk) o.push(["\n</function>", () => this._lit("close", { nc: st.nc })]);
        break;
      }
      case "pname":
        f.names.forEach((n, i) => { if (st.given[i] !== "1") o.push([n + ">\n", () => this._paramStart(st, i)]); });
        break;
      case "enum":
        for (const v of f.enums[st.p]) for (const c of ["\n" + CLOSE_P, CLOSE_P]) o.push([v + c, () => this._lit("body", base)]);
        break;
      case "vclose": for (const c of ["\n" + CLOSE_P, CLOSE_P]) o.push([c, () => this._lit("body", base)]); break;
      case "close": o.push(["\n" + this.S.close, () => this._afterCall(st.nc + 1)]); break;
      case "jopen": o.push(["\n{\"name\": \"", () => this._lit("jname", { nc: st.nc })]); break;
      case "jname":
        for (const n of this.names) o.push([n + "\", \"arguments\": ", () => ({ k: "J", fr: [{ t: "v", n: this.fns.get(n).args, w: 0 }], then: "jc", fn: n, given: "", nc: st.nc })]);
        break;
      case "jclose": o.push(["}\n" + this.S.close, () => this._afterCall(st.nc + 1)]); break;
      case "pnl": o.push(["\n", () => ({ k: "V", pm: "", ...base })]); break;
    }
    this.opts.set(key, o);
    return o;
  }

  // one character: the next state, or null when the grammar does not allow it
  _step(st, ch) {
    switch (st.k) {
      case "S0": {
        if (ch === this.S.think && this.special.has(THINK_OPEN)) return { k: "T", m: 0, o: 0 };
        const t = st.typed + ch;
        if (t === THINK_OPEN) return this.special.has(THINK_OPEN) ? null : { k: "T", m: 0, o: 0 };
        if (THINK_OPEN.startsWith(t)) return { k: "S0", typed: t };
        return this._feed(this._answerStart(), t, false);   // no reasoning after all
      }
      case "T": {
        const callsHere = this.style === "xml" && this.canCall;
        if (isSym(ch)) {
          if (ch === this.S.thinkEnd) return this._answerStart();
          if (callsHere && ch === this.S.open) return this._lit("open", { nc: 0 });   // a call opened inside the reasoning closes it
          return st;
        }
        // the tags spelled out in text: the trigger when they are not special tokens, else banned
        const m = tailM(st.m, ch, THINK_END);
        if (m === THINK_END.length) return this.special.has(THINK_END) ? null : this._answerStart();
        let o = 0;
        if (callsHere) {
          o = tailM(st.o, ch, OPEN);
          if (o === OPEN.length) return this.special.has(OPEN) ? null : this._lit("open", { nc: 0 });
        }
        return m === st.m && o === st.o ? st : { k: "T", m, o };
      }
      case "F": return this._stepFree(st, ch);
      case "W": {
        if (WS(ch) && st.n < 3) return { ...st, n: st.n + 1 };
        if (st.to !== "json" && this.S.open.startsWith(ch)) return this.S.open.length === 1 ? this._lit(this.style === "xml" ? "open" : "jopen", { nc: 0 }) : { ...this._lit("ropen", { nc: 0 }), typed: ch };
        if (st.to !== "call") { const j = this._jStart(this.fmtRoot, ch, 0); if (j) return { k: "J", fr: j, then: "fmt", fn: "", given: "", nc: 0 }; }
        return null;
      }
      case "L": {
        const typed = st.typed + ch;
        let live = false;
        for (const [s, to] of this._options(st)) {
          if (s === typed) return to();
          if (s.startsWith(typed)) live = true;
        }
        return live ? { ...st, typed } : null;
      }
      case "V": {
        if (!st.pm && ch !== "<" && ch !== "\n") return st;
        const s = st.pm + ch;
        if (s === CLOSE_P || s === "\n" + CLOSE_P) return this._lit("body", { fn: st.fn, given: st.given, nc: st.nc });
        if (FORBID.has(s)) return null;
        let pm = "";
        for (let k = 0; k < s.length; k++) if (PREFIXES.has(s.slice(k)) || ("\n" + CLOSE_P).startsWith(s.slice(k))) { pm = s.slice(k); break; }
        return pm === st.pm ? st : { ...st, pm };
      }
      case "PN": {
        if (ch === ">") return st.typed && PARAM_RE.test(st.typed) ? { k: "L", tag: "pnl", fn: st.fn, given: st.given, typed: "", nc: st.nc, p: -1 } : null;
        if (ch === "<" || ch === "\n" || ch === "\r" || isSym(ch) || st.typed.length >= 128) return null;
        return { ...st, typed: st.typed + ch };
      }
      case "J": {
        const r = this._jStep(st.fr, ch);
        if (!r) return null;
        if (r.fr) return { ...st, fr: r.fr };
        // the value is complete; r.rest: a character that ended a number, still to be read
        const next = st.then === "xp" ? this._lit("vclose", { fn: st.fn, given: st.given, nc: st.nc })
          : st.then === "jc" ? this._lit("jclose", { nc: st.nc }) : { k: "E", n: 0 };
        return r.rest == null ? next : this._step(next, r.rest);
      }
      case "AC": {
        const typed = st.typed + ch;
        for (const o of [this.S.open, "\n" + this.S.open, "\n\n" + this.S.open]) {
          if (o === typed) return this._lit(this.style === "xml" ? "open" : "jopen", { nc: st.nc });
          if (o.startsWith(typed)) return { ...st, typed };
        }
        return null;
      }
      case "E": return WS(ch) && st.n < 2 ? { k: "E", n: st.n + 1 } : null;
    }
    return null;
  }

  _stepFree(st, ch) {
    // the "<tool_call>" opener (and a garbled "<tool...>" guard, as in Code mode)
    const open = () => (this.mode === "none" ? null : !this.canCall ? { k: "F", t: "", ls: 0, fm: "" } : this._lit(this.style === "xml" ? "open" : "jopen", { nc: 0 }));
    if (isSym(ch)) return ch === this.S.open ? open() : st.t || st.fm || st.ls ? { k: "F", t: "", ls: 0, fm: "" } : st;
    let t = st.t + ch;
    if (t === OPEN) return this.special.has(OPEN) ? null : open();   // spelled out when it is a token: no
    if (t === RESPONSE) t = "";
    else if (this.mode === "none" && this.names.length && t === OPEN.slice(0, -1)) return null;   // none: no "<tool_call" at all (not even "<tool_call >")
    else if (!OPEN.startsWith(t) && !RESPONSE.startsWith(t)) {
      if (st.t.length >= GUARD && !(st.t === OPEN.slice(0, GUARD) && ch !== "_")) return null;   // "<tool_x": a garbled opener
      t = ch === "<" ? "<" : "";
    }
    // the lazy bare trigger: a line-initial "<function=" then the start of a declared name
    let fm = "";
    if (this.bare && (st.fm || (st.ls && ch === "<"))) {
      const f = st.fm + ch;
      if (this.mode === "none" && f === FN.slice(0, -1)) return null;   // none: no line-initial "<function" either
      if (FN.startsWith(f)) fm = f;
      else if (f.startsWith(FN)) {
        const x = f.slice(FN.length);
        if (x && this.names.some((n) => n.startsWith(x))) {
          if (this.mode === "none") return null;
          return { ...this._lit("name", { nc: 0, bare: 1 }), typed: x };
        }
      }
    }
    const ls = ch === "\n" ? 1 : 0;
    return t === st.t && fm === st.fm && ls === st.ls ? st : { k: "F", t, ls, fm };
  }

  // ---- JSON values. A frame stack; the top frame is the value being written.
  //   {t:"v", n, w}             a value of node n, after the gap w
  //   {t:"o", n, g, ph, w, kv}  an object: g given keys ("0101"), ph "k0" | "k" | "c" | "cv" | "a"
  //   {t:"k", n, typed, g}      a declared key being written (typed: its text so far)
  //   {t:"ks", e, u}            a free key string
  //   {t:"a", n, i, ph, w}      an array, ph "v0" | "v" | "a"
  //   {t:"s", n, e, u, typed}   a string (typed: an enum value's text so far)
  //   {t:"d", int, ph}          a number
  //   {t:"l", n, typed}         true / false / null / an enum literal
  // w (a whitespace gap): 0 nothing yet, 1 one space, 2 + s a newline and s spaces
  _node(n) { return this.nodes[n]; }
  _depth(fr) { let d = 0; for (const f of fr) if (f.t === "o" || f.t === "a") d++; return d; }
  _gap(w, ch, depth) {
    if (ch === " ") { if (w === 0) return 1; if (w >= 2 && w - 2 < 4 * (depth + 1)) return w + 1; return -1; }
    if (ch === "\n") return w === 0 ? 2 : -1;
    return -1;
  }
  _litOpts(nd) {
    if (!nd) return [];
    if (nd.k === "bool") return ["true", "false"];
    if (nd.k === "null") return ["null"];
    if (nd.k === "lit") return nd.texts;
    if (nd.k === "any") return ["true", "false", "null"];
    if (nd.k === "u") return [...(nd.bool != null ? ["true", "false"] : []), ...(nd.null != null ? ["null"] : []), ...nd.lits];
    return [];
  }
  // the frames a value of node n starting with ch opens (depth: containers around it), or null
  _jStart(n, ch, depth) {
    let nd = this._node(n);
    if (!nd) return null;
    if (nd.k === "u") {
      const pick = ch === '"' ? nd.str : ch === "-" || DIG(ch) ? nd.num : ch === "t" || ch === "f" ? nd.bool : ch === "n" ? nd.null : ch === "{" ? nd.obj : ch === "[" ? nd.arr : null;
      if (pick != null) { n = pick; nd = this._node(n); }
      else {
        const lits = nd.lits.filter((x) => x[0] === ch);
        return lits.length ? [{ t: "l", n, typed: ch }] : null;
      }
    }
    const any = nd.k === "any";
    if (ch === "{" && (nd.k === "obj" || any)) return depth >= SCHEMA_CAPS.depth ? null : [{ t: "o", n, g: nd.k === "obj" ? "0".repeat(nd.props.length) : "", ph: "k0", w: 0, kv: -1 }];
    if (ch === "[" && (nd.k === "arr" || any)) return depth >= SCHEMA_CAPS.depth ? null : [{ t: "a", n, i: 0, ph: "v0", w: 0 }];
    if (ch === '"' && (nd.k === "str" || any)) return [{ t: "s", n, e: 0, u: 0, typed: "" }];
    if ((ch === "-" || DIG(ch)) && (nd.k === "num" || nd.k === "int" || any)) {
      const int = nd.k === "int";
      return [{ t: "d", int, ph: ch === "-" ? "m" : ch === "0" ? "z" : "i" }];
    }
    const opts = this._litOpts(nd);
    if (opts.some((o) => o[0] === ch)) return [{ t: "l", n, typed: ch }];
    return null;
  }
  _jEnd(fr) {   // could the value end here (a top-level number or literal that is complete)?
    if (fr.length !== 1) return false;
    const f = fr[0];
    if (f.t === "d") return f.ph === "z" || f.ph === "i" || f.ph === "f" || f.ph === "x";
    if (f.t === "l") return this._litOpts(this._node(f.n)).includes(f.typed);
    return false;
  }
  // the value on top finished: the parent moves on. -> { fr } or { done: true }
  _jPop(fr) {
    const rest = fr.slice(0, -1);
    if (!rest.length) return { done: true };
    const p = rest[rest.length - 1];
    if (p.t === "o") rest[rest.length - 1] = { ...p, ph: "a", w: 0, kv: -1 };
    else if (p.t === "a") rest[rest.length - 1] = { ...p, ph: "a", w: 0, i: p.i + 1 };
    return { fr: rest };
  }
  _jStep(fr, ch) {
    const top = fr[fr.length - 1], depth = this._depth(fr);
    const put = (f) => { const c = fr.slice(); c[c.length - 1] = f; return { fr: c }; };
    const open = (frames) => ({ fr: fr.slice(0, -1).concat(frames) });
    switch (top.t) {
      case "v": {
        const g = this._gap(top.w, ch, depth);
        if (g >= 0) return put({ ...top, w: g });
        const s = this._jStart(top.n, ch, depth);
        return s ? open(s) : null;
      }
      case "s": {
        const nd = this._node(top.n);
        if (nd && nd.k === "str" && nd.en) {
          const t = top.typed + ch;
          let live = false, done = false;
          for (const v of nd.en) { const x = JSON.stringify(v).slice(1); if (x === t) done = true; else if (x.startsWith(t)) live = true; }
          if (done) return this._jPop(fr);
          return live ? put({ ...top, typed: t }) : null;
        }
        return this._strStep(top, ch, () => this._jPop(fr), put);
      }
      case "d": {
        const ph = top.ph, int = top.int;
        let nx = null;
        if (ph === "m") nx = ch === "0" ? "z" : DIG(ch) ? "i" : null;
        else if (ph === "z" || ph === "i") nx = ph === "i" && DIG(ch) ? "i" : !int && ch === "." ? "p" : !int && (ch === "e" || ch === "E") ? "e" : "END";
        else if (ph === "p") nx = DIG(ch) ? "f" : null;
        else if (ph === "f") nx = DIG(ch) ? "f" : ch === "e" || ch === "E" ? "e" : "END";
        else if (ph === "e") nx = ch === "+" || ch === "-" ? "s" : DIG(ch) ? "x" : null;
        else if (ph === "s") nx = DIG(ch) ? "x" : null;
        else if (ph === "x") nx = DIG(ch) ? "x" : "END";
        if (nx == null) return null;
        if (nx !== "END") return nx === ph ? { fr } : put({ ...top, ph: nx });
        // the number ended before ch: the parent reads ch
        const r = this._jPop(fr);
        if (r.done) return { done: true, rest: ch };
        return this._jStep(r.fr, ch);
      }
      case "l": {
        const opts = this._litOpts(this._node(top.n));
        const t = top.typed + ch;
        if (opts.some((o) => o.startsWith(t))) {
          if (opts.includes(t) && !opts.some((o) => o !== t && o.startsWith(t))) return this._jPop(fr);
          return put({ ...top, typed: t });
        }
        if (opts.includes(top.typed)) {   // a literal that is a prefix of another ended before ch
          const r = this._jPop(fr);
          if (r.done) return { done: true, rest: ch };
          return this._jStep(r.fr, ch);
        }
        return null;
      }
      case "o": return this._objStep(fr, top, ch, depth, put, open);
      case "k": {
        const nd = this._node(top.n);
        const t = top.typed + ch;
        let live = false;
        for (let i = 0; i < nd.props.length; i++) {
          if (top.g[i] === "1") continue;
          const x = JSON.stringify(nd.props[i][0]).slice(1);
          if (x === t) {
            const c = fr.slice(0, -1), p = c[c.length - 1];
            c[c.length - 1] = { ...p, ph: "c", w: 0, kv: i, g: p.g.slice(0, i) + "1" + p.g.slice(i + 1) };
            return { fr: c };
          }
          if (x.startsWith(t)) live = true;
        }
        return live ? put({ ...top, typed: t }) : null;
      }
      case "ks":
        return this._strStep(top, ch, () => { const c = fr.slice(0, -1), p = c[c.length - 1]; c[c.length - 1] = { ...p, ph: "c", w: 0, kv: -1 }; return { fr: c }; }, put);
      case "a": {
        const nd = this._node(top.n);
        const itemNode = nd.k === "any" ? top.n : top.i < nd.prefix.length ? nd.prefix[top.i] : nd.items;
        if (top.ph === "v0" || top.ph === "v") {
          const g = this._gap(top.w, ch, depth);
          if (g >= 0) return put({ ...top, w: g });
          if (top.ph === "v0" && ch === "]") return this._jPop(fr);
          const s = this._jStart(itemNode, ch, depth);
          return s ? { fr: fr.slice(0, -1).concat([{ ...top, ph: "a", w: 0 }], s) } : null;
        }
        // ph "a": after an item
        const g = this._gap(top.w, ch, depth - 1);
        if (g >= 0) return put({ ...top, w: g });
        if (ch === ",") return put({ ...top, ph: "v", w: 0 });
        if (ch === "]") return this._jPop(fr);
        return null;
      }
    }
    return null;
  }
  _strStep(top, ch, close, put) {
    if (top.u) return HEX(ch) ? put({ ...top, u: top.u - 1 }) : null;
    if (top.e) {
      if (ch === "u") return put({ ...top, e: 0, u: 4 });
      return '"\\/bfnrt'.includes(ch) ? put({ ...top, e: 0 }) : null;
    }
    if (ch === '"') return close();
    if (ch === "\\") return put({ ...top, e: 1 });
    if (ch < " " || isSym(ch)) return null;
    return put(top);
  }
  _objStep(fr, top, ch, depth, put) {
    const nd = this._node(top.n);
    const free = nd.k === "any" || (nd.k === "obj" && !nd.props.length && nd.free != null);
    const reqOk = () => nd.k !== "obj" || nd.req.every((i) => top.g[i] === "1");
    const left = () => free || (nd.k === "obj" && nd.props.some((_, i) => top.g[i] !== "1"));
    const g = this._gap(top.w, ch, top.ph === "a" ? depth - 1 : depth);
    switch (top.ph) {
      case "k0": case "k":
        if (g >= 0) return put({ ...top, w: g });
        if (top.ph === "k0" && ch === "}" && reqOk()) return this._jPop(fr);
        if (ch === '"' && left()) return { fr: fr.concat([free ? { t: "ks", e: 0, u: 0 } : { t: "k", n: top.n, typed: "", g: top.g }]) };
        return null;
      case "c":
        if (ch === ":") return put({ ...top, ph: "cv", w: 0 });
        return null;
      case "cv": {
        if (g >= 0) return put({ ...top, w: g });
        const vn = nd.k === "any" ? top.n : free ? nd.free : nd.props[top.kv][1];
        const s = this._jStart(vn, ch, depth);
        return s ? { fr: fr.slice(0, -1).concat([{ ...top, ph: "a", w: 0 }], s) } : null;
      }
      case "a":
        if (g >= 0) return put({ ...top, w: g });
        if (ch === "," && left()) return put({ ...top, ph: "k", w: 0 });
        if (ch === "}" && reqOk()) return this._jPop(fr);
        return null;
    }
    return null;
  }

  // ---- masks
  // the cache key of a state's mask, or null when every token is allowed there
  _key(st) {
    let k;
    switch (st.k) {
      case "T": if (!this.forcing && !st.m && !st.o) return null; k = `T|${st.m}|${st.o}`; break;
      case "F": if (st.t.length < GUARD && !st.fm) return null; k = `F|${st.t}|${st.fm}`; break;
      case "S0": k = "S0|" + st.typed; break;
      case "W": k = `W|${st.n}|${st.to}`; break;
      case "L": k = `L|${st.tag}|${st.fn}|${st.given}|${st.typed}|${st.nc}|${st.p}|${st.bare ? 1 : 0}`; break;
      case "V": k = `V|${st.fn}|${st.given}|${st.pm}|${st.nc}`; break;
      case "PN": k = `P|${st.fn}|${st.given}|${st.typed}|${st.nc}`; break;
      case "J": k = `J|${st.then}|${st.fn}|${st.given}|${st.nc}|` + JSON.stringify(st.fr); break;
      case "AC": k = `A|${st.typed}|${st.nc}`; break;
      case "E": k = `E|${st.n}`; break;
    }
    return this.sig + "\u0001" + k;
  }
  // a state whose text is the model's own choice (a value's contents): forced tokens there are
  // what the garbage guard counts
  _freeValue(st) {
    if (st.k === "V") return true;
    if (st.k !== "J") return false;
    const top = st.fr[st.fr.length - 1];
    if (top.t === "s") { const nd = this._node(top.n); return !(nd && nd.k === "str" && nd.en); }
    return top.t === "d" || top.t === "ks";
  }
  accepts(id, st = this.st) {
    if (this.stops.has(id)) return this.endOk(st);
    const w = this.tokenText(id);
    return !!w && this._feed(st, w, false) !== null;
  }
  allowed() {
    const key = this._key(this.st);
    if (key == null) return null;
    let m = this.cache.get(key);
    if (m !== undefined) return m;
    m = this._scan(this.st);
    this.cache.set(key, m);
    return m;
  }
  _scan(st) {
    const allow = [], deny = [], first = new Map();
    const endOk = this.endOk(st);
    const plainV = st.k === "V" && !st.pm;
    const top = st.k === "J" ? st.fr[st.fr.length - 1] : null;
    const plainS = !!top && top.t === "s" && !top.e && !top.u && !(this._node(top.n)?.en);
    for (let id = 0; id < this.vocabSize; id++) {
      let ok;
      if (this.stops.has(id)) ok = endOk;
      else {
        const w = this.tokenText(id);
        if (!w) ok = false;
        else if (plainV && w.indexOf("<") < 0 && w.indexOf("\n") < 0 && !SYM_RE.test(w)) ok = true;
        else if (plainS && !/["\\\u0000-\u001f\uE000-\uE0FF]/.test(w)) ok = true;
        else {
          const c = w[0];
          let s1 = first.get(c);
          if (s1 === undefined) first.set(c, (s1 = this._step(st, c)));
          ok = s1 !== null && (w.length === 1 || this._feed(s1, w.slice(1), false) !== null);
        }
      }
      (ok ? allow : deny).push(id);
    }
    if (!allow.length) return null;
    return allow.length <= deny.length ? { allow: Int32Array.from(allow) } : { deny: Int32Array.from(deny) };
  }

  // Mask logits in place for a sampler that takes the top k (1: greedy). Sets:
  //   forced  the model's own top token was not allowed
  //   free    this position is inside a value's contents (what the garbage guard counts)
  //   mass    the allowed share of the probability (computed only when forced in a free value)
  //   bad     the logits hold NaN or +Infinity
  mask(logits, k = 1) {
    this.forced = false; this.bad = false; this.mass = 1;
    const key = this._key(this.st);
    this.free = this._freeValue(this.st);
    if (key == null) return logits;
    let m = this.cache.get(key);
    // the top candidates (64, the most any sampler here takes), and the argmax
    const K = 64, ids = new Int32Array(K).fill(-1), vals = new Float32Array(K).fill(-Infinity);
    let minAt = 0, bad = false;
    for (let i = 0; i < logits.length; i++) {
      const v = logits[i];
      if (v !== v || v === Infinity) { bad = true; continue; }
      if (v > vals[minAt]) {
        ids[minAt] = i; vals[minAt] = v;
        let mn = 0; for (let j = 1; j < K; j++) if (vals[j] < vals[mn]) mn = j;
        minAt = mn;
      }
    }
    this.bad = bad;
    let best = -1, bv = -Infinity;
    for (let j = 0; j < K; j++) if (ids[j] >= 0 && vals[j] > bv) { bv = vals[j]; best = ids[j]; }
    if (m === undefined) {
      // fast path: enough of the top candidates are allowed -> masking just those is exact
      let pass = 0, cut = Infinity;
      const okc = new Uint8Array(K), good = [];
      for (let j = 0; j < K; j++) {
        if (ids[j] < 0) { cut = -Infinity; continue; }
        if (vals[j] < cut) cut = vals[j];
        if (this.accepts(ids[j])) { okc[j] = 1; pass++; good.push(vals[j]); }
      }
      good.sort((a, b) => b - a);
      // exact only when the k-th allowed candidate beats everything outside the candidates (ties at
      // the cut could be picked instead)
      if (pass >= Math.max(1, k) && good[Math.max(1, k) - 1] > cut) {
        // (tokens below the candidates stay unmasked: a top-k sampler with k <= pass never reaches them)
        this.forced = best >= 0 && !this.accepts(best);
        if (this.forced && this.free) this.mass = massOf(logits, null, ids, okc);
        for (let j = 0; j < K; j++) if (ids[j] >= 0 && !okc[j]) logits[ids[j]] = -Infinity;
        return logits;
      }
      m = this._scan(this.st);
      this.cache.set(key, m);
    }
    if (!m) return logits;
    this.forced = best >= 0 && !maskHas(m, best);
    if (this.forced && this.free) this.mass = massOf(logits, m);
    if (m.allow) {
      const keep = Array.from(m.allow, (i) => logits[i]);
      logits.fill(-Infinity);
      m.allow.forEach((i, j) => { logits[i] = keep[j]; });
    } else for (const i of m.deny) logits[i] = -Infinity;
    return logits;
  }
}

// the probability share of the allowed tokens (before masking)
function massOf(logits, m, ids = null, okc = null) {
  let mx = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
  if (!Number.isFinite(mx)) return 0;
  let tot = 0;
  for (let i = 0; i < logits.length; i++) { const v = logits[i]; if (v > -Infinity) tot += Math.exp(v - mx); }
  let ok = 0;
  if (m) {
    if (m.allow) for (const i of m.allow) ok += Math.exp(logits[i] - mx);
    else { let no = 0; for (const i of m.deny) if (logits[i] > -Infinity) no += Math.exp(logits[i] - mx); ok = tot - no; }
  } else if (ids) for (let j = 0; j < ids.length; j++) if (okc[j]) ok += Math.exp(logits[ids[j]] - mx);
  return tot > 0 ? ok / tot : 0;
}
