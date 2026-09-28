// harness/constrain.js: the tool-call grammar while sampling (docs/design/harness-light.md B.1).
import { ToolCallConstraint, maskHas } from "../../harness/constrain.js";
import { constrainedSampler } from "../../harness/model-common.js";
import { ToolCallParser } from "../../harness/tools.js";
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

const tools = [
  { name: "read_file", parameters: { properties: { path: { type: "string" }, start_line: { type: "integer" } }, required: ["path"] } },
  { name: "write_file", parameters: { properties: { path: { type: "string" }, content: { type: "string" }, append: { type: "boolean" } }, required: ["path", "content"] } },
  { name: "list_dir", parameters: { properties: {} } },
];
// single characters plus nasty multi-character tokens that cross state boundaries
const CHARS = [..."abcdefghijklmnopqrstuvwxyz_0123456789 .-<>/=\n\"{}:"];
const MULTI = ["<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>", ">\n", "</parameter>\n<", "\n</",
  "\n</parameter>\n", "</parameter>\n</function>\n</tool_call>", "read", "_file", "write", "list_dir>", "path", "content", "append", "start_line",
  "true", "false", "12", "hello world", "<div>", "</div>", "{\"name\": \"", "read_file\"", "", "<|im_end|>"];
const VOCAB = [...CHARS, ...MULTI];
const STOP = VOCAB.indexOf("<|im_end|>");
const mk = (style = "xml", extra = {}) => new ToolCallConstraint(tools, { vocabSize: VOCAB.length, tokenText: tt, style, stops: [STOP], ...extra });
const tt = (i) => VOCAB[i];
const words = (m) => (m ? VOCAB.filter((_, i) => maskHas(m, i)).sort() : null);
const after = (text, extra) => { const C = mk("xml", extra); C.setText(text); return C; };

Deno.test("free text outside tool calls; stop allowed there", () => {
  const C = mk();
  eq(C.allowed(), null);
  C.push("hello <too"); eq(C.allowed(), null);
});

Deno.test("after \"<tool\" the tag can only become <tool_call> (or <tool_response>)", () => {
  // Qwen 3.6 wrote "<tool_tool_calls>" once its earlier calls were spelled out in text: the parser
  // never saw a call and the call's text was shown. From "<tool" on, only the real openers can follow.
  const C = after("I'll edit it.\n<tool");
  const w = words(C.allowed());
  ok(w.includes("_") && w.includes("<|im_end|>"), "_ (toward _call>) and the stop");
  ok(w.includes("a") && w.includes(" "), "\"<toolbar\" or \"<tool \" are still free text");
  const C2 = after("I'll edit it.\n<tool_");
  eq(words(C2.allowed()), ["c", "r", "<|im_end|>"].sort(), "only toward call> or response>");
  ok(!words(C2.allowed()).includes("_file"), "no \"<tool__file\"");
  const C3 = after("x <tool_call");
  eq(words(C3.allowed()), [">", ">\n", "<|im_end|>"].sort(), "then the closing >");
  // and once it is written in pieces, the call is constrained as usual
  const C4 = after("x <tool_call>");
  eq(words(C4.allowed()), ["\n"], "the call proper");
});

Deno.test("xml: the whole call is constrained, step by step", () => {
  const C = after("I'll look.\n<tool_call>");
  eq(words(C.allowed()), ["\n"].sort(), "only the newline before <function=");
  C.push("\n<function=");
  eq(words(C.allowed()), ["l", "r", "w", "read", "write", "list_dir>"].sort(), "declared names");
  C.push("read_file>");
  eq(words(C.allowed()), ["\n"], "path is required: no </function> yet");
  C.push("\n<parameter=");
  eq(words(C.allowed()), ["p", "s", "path", "start_line"].sort());
  C.push("path>");
  eq(words(C.allowed()), ["\n", "\n</", "\n</parameter>\n"].sort(), "the newline, then the value (may be empty)");
  C.push("\nsrc/a.js");
  const v = C.allowed();
  ok(v.deny, "values are a deny list");
  ok(!maskHas(v, STOP), "no end of turn inside a value");
  for (const w of ["<tool_call>", "</tool_call>"]) ok(!maskHas(v, VOCAB.indexOf(w)), w + " cannot appear in a value");
  for (const w of ["<function=", "</function>", "<parameter="]) ok(maskHas(v, VOCAB.indexOf(w)), w + " is fine mid-line (docs, regexes)");
  const nl = after("<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/a.js\n").allowed();
  for (const w of ["<function=", "</function>", "<parameter=", "<tool_call>"]) ok(!maskHas(nl, VOCAB.indexOf(w)), w + " cannot start a line of a value");
  ok(maskHas(nl, VOCAB.indexOf("<div>")) && maskHas(nl, VOCAB.indexOf("</parameter>")), "other tags can");
  ok(!maskHas(nl, STOP), "still no end of turn");
  for (const w of ["<div>", "</div>", "</parameter>", "\n</parameter>\n", "</parameter>\n<", "hello world", "\n</"]) ok(maskHas(v, VOCAB.indexOf(w)), w + " is fine");
  ok(maskHas(v, VOCAB.indexOf("</parameter>\n</function>\n</tool_call>")), "closing the value gives path, so the call may end in the same token");
});

Deno.test("xml: required params gate </function>, given params are not offered again, typed values", () => {
  let C = after("<tool_call>\n<function=read_file>\n<parameter=path>\na.js\n</parameter>");
  eq(words(C.allowed()), ["\n", "\n</"].sort(), "path given: </function> is open");
  C.push("\n<parameter=");
  eq(words(C.allowed()), ["s", "start_line"].sort(), "path was given");
  C.push("start_line>\n");
  eq(words(C.allowed()), ["-", ..."0123456789", "12"].sort(), "integer: digits");
  C.push("12");
  eq(words(C.allowed()), [..."0123456789", "<", "\n", "</parameter>", "\n</", "\n</parameter>\n", "</parameter>\n<", "</parameter>\n</function>\n</tool_call>", "12"].sort(), "more digits or the closer");
  C = after("<tool_call>\n<function=write_file>\n<parameter=append>\n");
  eq(words(C.allowed()), ["f", "t", "true", "false"].sort(), "boolean");
  C = after("<tool_call>\n<function=write_file>\n<parameter=path>\na\n</parameter>\n");
  ok(!maskHas(C.allowed(), VOCAB.indexOf("\n</")), "content still missing");
  C = after("<tool_call>\n<function=list_dir>");
  eq(words(C.allowed()), ["\n", "\n</"].sort(), "no params: only </function>");
});

Deno.test("xml: after </tool_call> the text is free again (a stop after a newline is not masked)", () => {
  const C = after("<tool_call>\n<function=list_dir>\n</function>\n</tool_call>");
  eq(C.allowed(), null);
  C.push("\n");
  eq(C.allowed(), null, "a newline then <|im_end|> or an invented <tool_response> must not force a second call");
  C.push("<tool_call>\n<function=");
  ok(C.allowed().allow.length > 1, "a second call");
});

Deno.test("streaming token boundaries: any split of the same text gives the same state", () => {
  const text = "ok\n<tool_call>\n<function=write_file>\n<parameter=path>\ng.js\n</parameter>\n<parameter=content>\nlet a = '<div>';\n</parameter>\n";
  const ref = words(after(text).allowed());
  const rnd = mulberry(7);
  for (let r = 0; r < 50; r++) {
    const C = mk();
    let i = 0;
    while (i < text.length) { const n = 1 + Math.floor(rnd() * 6); C.push(text.slice(i, i + n)); i += n; }
    eq(words(C.allowed()), ref, "split " + r);
    // setText incrementally, and after a rewind (a speculative step that was rejected)
    const D = mk();
    for (let k = 1; k <= text.length; k += 3) D.setText(text.slice(0, k));
    D.push("x</param"); D.setText(text);
    eq(words(D.allowed()), ref, "setText " + r);
  }
});

Deno.test("thinking: a <tool_call> inside the think block is not a call", () => {
  const C = after("hmm <tool_call> maybe", { thinking: true });
  eq(C.allowed(), null);
  C.push("</think>\n\n<tool_call>");
  eq(words(C.allowed()), ["\n"].sort());
});

Deno.test("text the automaton rejects (e.g. sampled unconstrained) drops back to free text", () => {
  const C = after("<tool_call>\n<function=nope>");
  eq(C.allowed(), null);
});

Deno.test("random sampling through the masks always yields valid calls (2,000 runs per tool)", () => {
  const rnd = mulberry(1);
  const parsed = { read_file: 0, write_file: 0, list_dir: 0 };
  for (const t of tools) {
    for (let run = 0; run < 2000; run++) {
      const C = after(`<tool_call>\n<function=${t.name}>`);
      let text = `<tool_call>\n<function=${t.name}>`, ended = false;
      for (let s = 0; s < 300 && !ended; s++) {
        const m = C.allowed();
        if (!m) { ok(text.includes("</tool_call>"), "constrained all the way: " + JSON.stringify(text)); ended = true; break; }   // free again after the call
        const ids = VOCAB.map((_, i) => i).filter((i) => maskHas(m, i));
        // favour multi-character tokens so runs finish
        const w = ids.map((i) => (VOCAB[i].length > 1 ? 6 : 1)), tot = w.reduce((a, b) => a + b, 0);
        let x = rnd() * tot, id = ids[0];
        for (let k = 0; k < ids.length; k++) { x -= w[k]; if (x <= 0) { id = ids[k]; break; } }
        if (id === STOP) { ended = true; break; }
        C.push(VOCAB[id]); text += VOCAB[id];
      }
      if (!ended) continue;   // ran out of steps mid-value: fine, it was never invalid
      const P = new ToolCallParser({ schemaFor: (n) => tools.find((x) => x.name === n)?.parameters });
      const r = P.feed(text), e = P.end();
      ok(!r.text.trim() && !e.text.trim(), "nothing outside calls: " + JSON.stringify(text));
      const calls = [...r.calls, ...e.calls];
      ok(calls.length >= 1, "at least one call");
      for (const c of calls) {
        ok(!c.error, c.error + " " + JSON.stringify(text));
        const def = tools.find((x) => x.name === c.name);
        ok(def, "declared name " + c.name);
        for (const q of def.parameters.required || []) ok(q in c.arguments, `required ${q} in ${JSON.stringify(text)}`);
        for (const [k, v] of Object.entries(c.arguments)) {
          const ty = def.parameters.properties[k]?.type;
          ok(ty, "declared param " + k);
          if (ty === "integer") ok(Number.isInteger(v), `${k}=${JSON.stringify(v)}`);
          if (ty === "boolean") ok(typeof v === "boolean", `${k}=${JSON.stringify(v)}`);
        }
        parsed[c.name]++;
      }
      for (const m of text.matchAll(/<function=[^>]+>([\s\S]*?)<\/function>/g)) {
        const names = [...m[1].matchAll(/(?:^|\n)<parameter=([^>]+)>/g)].map((x) => x[1]);
        eq(names.length, new Set(names).size, "no duplicated params");
      }
    }
  }
  for (const [n, k] of Object.entries(parsed)) ok(k > 200, `${n}: ${k} complete calls`);
});

Deno.test("json: the name string is limited", () => {
  const C = mk("json");
  C.push("<tool_call>\n{\"name\": \"");
  eq(words(C.allowed()), ["l", "r", "w", "read", "write", "read_file\""].sort());
  C.push("read_file\"");
  eq(C.allowed(), null);
});

Deno.test("mask(): allow lists and deny lists, forced, cache shared across instances and bounded", () => {
  const C = after("<tool_call>\n<function=");
  const lg = new Float32Array(VOCAB.length).fill(1);
  lg[VOCAB.indexOf("hello world")] = 5;   // the model wanted prose
  C.mask(lg);
  ok(C.forced, "forced");
  eq(VOCAB.filter((_, i) => lg[i] > -Infinity), ["l", "r", "w", "read", "write", "list_dir>"]);
  ok(after("<tool_call>\n<function=").allowed() === C.allowed(), "the mask is cached per tokenizer + tools");
  const V = after("<tool_call>\n<function=read_file>\n<parameter=path>\n");
  const lv = new Float32Array(VOCAB.length).fill(0);
  V.mask(lv);
  ok(!V.forced, "a plain value is not forced");
  eq(lv[STOP], -Infinity);
  ok(V.cache.size <= 512, "bounded: " + V.cache.size);
});

Deno.test("constrainedSampler counts forced positions of kept tokens; garbage past the gate", () => {
  const argmax = (lg) => { let b = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[b]) b = i; return b; };
  const cs = constrainedSampler(argmax, tools, { tokenText: tt, vocabSize: VOCAB.length, stops: [STOP] });
  cs.setText("");
  const garbage = () => { const lg = new Float32Array(VOCAB.length).fill(0); lg[VOCAB.indexOf("hello world")] = 9; return lg; };
  const want = (w) => { const lg = new Float32Array(VOCAB.length).fill(0); lg[VOCAB.indexOf(w)] = 9; return lg; };
  let out = "";
  const step = (lg) => { out += tt(cs.sample(lg)); cs.keep(1); cs.setText(out); };
  for (const w of ["<tool_call>", "\n"]) step(want(w));
  eq(cs.forced, 0);
  for (let k = 0; k < 5; k++) step(garbage());
  ok(cs.forced >= 4, "garbage logits are forced: " + cs.forced + " " + JSON.stringify(out));
  ok(!cs.garbage, "not yet past the gate");
  for (let k = 0; k < 20 && !cs.garbage; k++) step(garbage());
  ok(cs.garbage, "a call forcing most of its tokens is garbage");
  cs.setText(""); eq(cs.forced, 0, "reset per answer"); ok(!cs.garbage);

  // a speculative step: 3 columns sampled, only the first kept; the rejected ones are not counted
  out = "<tool_call>\n";
  cs.setText(out);
  cs.sample(want("<function=")); cs.sample(garbage()); cs.sample(garbage());
  cs.keep(1);
  eq(cs.forced, 0, "forced columns after a rejected draft do not count");
  cs.sample(garbage()); cs.keep(1);
  eq(cs.forced, 1, "the next step's columns start fresh");
});

function mulberry(a) {
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
