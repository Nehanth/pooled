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
  constructor(tools, { vocabSize, tokenText, style = "xml", stops = [], thinking = false }) {
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
