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

// ---- the strict grammar for API answers (GrammarConstraint: mode / format given) ----
import { GrammarConstraint, maskCacheFor, MASK_BYTES } from "../../harness/constrain.js";
import { SchemaError } from "../../harness/jsonschema.js";

const APITOOLS = [
  { name: "get_weather", parameters: { type: "object", properties: { city: { type: "string" }, unit: { type: "string", enum: ["c", "f"] }, days: { type: "integer" }, opts: { type: "object", properties: { a: { type: "boolean" }, n: { $ref: "#/$defs/N" } }, required: ["a"], additionalProperties: false }, tags: { type: "array", items: { type: "string" } } }, required: ["city"], $defs: { N: { type: "number" } } } },
  { name: "noop", parameters: { type: "object", properties: {} } },
];
const JV = [...new Set([...CHARS, ..."ABCTUVWXYZ,[]\\\t", ...MULTI, "<think>", "</think>", "get_weather>", "noop>", "\n\n", "  ", "    ", "\"city\"", "Paris", "\n  ", "\n\n\n", "True", "null", "\"a\"", "<function", "<tool.call>"])];
const JSTOP = JV.indexOf("<|im_end|>");
const jt = (i) => JV[i];
const G = (o = {}, tools = APITOOLS) => new GrammarConstraint(tools, { vocabSize: JV.length, tokenText: jt, stops: [JSTOP], style: "xml", mode: "auto", maskCache: maskCacheFor(jt, MASK_BYTES), ...o });
const at = (text, o, tools) => { const C = G(o, tools); C.setText(text); return C; };
const allows = (C, w) => C.accepts(JV.indexOf(w));
const feedOk = (C, s) => { for (const ch of s) { const n = C._step(C.st, ch); if (!n) return false; C.st = n; } return true; };
const valid = (text, o, tools) => { const C = G(o, tools); return feedOk(C, text) && C; };

Deno.test("strict: auto is free text until a call; after </tool_call> only whitespace, another call or the end", () => {
  const C = at("Sure, here it is.");
  eq(C.allowed(), null, "free text is not masked");
  ok(allows(C, "<|im_end|>"), "the end is allowed in free text");
  const A = at("<tool_call>\n<function=noop>\n</function>\n</tool_call>");
  ok(allows(A, "<|im_end|>") && allows(A, "\n") && allows(A, "<tool_call>"), "end, newline, another call");
  ok(!allows(A, "hello world") && !allows(A, "a"), "no free text after calls");
  ok(valid("x\n<tool_call>\n<function=noop>\n</function>\n</tool_call>\n\n<tool_call>\n<function=noop>\n</function>\n</tool_call>"), "two calls");
});
Deno.test("strict: required and named force a call right away; the end is banned until then", () => {
  const R = at("", { mode: "required" });
  ok(!allows(R, "<|im_end|>") && !allows(R, "hello world") && allows(R, "<tool_call>") && allows(R, "\n"), "whitespace then <tool_call>");
  ok(!valid("hi<tool_call>", { mode: "required" }), "no text first");
  const N = at("<tool_call>\n<function=", { mode: { name: "noop" } });
  eq(words(N.allowed()).filter((w) => JV.includes(w)).length > 0, true);
  ok(allows(N, "n") && allows(N, "noop>") && !allows(N, "g") && !allows(N, "get_weather>"), "the name is fixed");
  const after = at("<tool_call>\n<function=noop>\n</function>\n</tool_call>", { mode: { name: "noop" } });
  ok(allows(after, "<|im_end|>") && !allows(after, "<tool_call>"), "named: one call, then the end");
  const one = at("<tool_call>\n<function=noop>\n</function>\n</tool_call>", { parallel: false });
  ok(!allows(one, "<tool_call>") && allows(one, "<|im_end|>"), "parallel false: one call");
  const two = at("<tool_call>\n<function=noop>\n</function>\n</tool_call>\n<tool_call>\n<function=noop>\n</function>\n</tool_call>", { maxCalls: 2 });
  ok(!allows(two, "<tool_call>") && allows(two, "<|im_end|>"), "maxCalls");
  const al = at("<tool_call>\n<function=", { allowed: ["noop"] });
  ok(!allows(al, "g") && allows(al, "n"), "allowed narrows the names");
});
Deno.test("strict: none bans the opener, the bare trigger and near spellings; tools are otherwise free text", () => {
  const C = at("x <tool_cal", { mode: "none" });
  ok(!allows(C, "l") && !allows(C, "<|im_end|>") === false, "no <tool_call");
  ok(!allows(at("x ", { mode: "none" }), "<tool_call>"), "the added token too");
  ok(!allows(at("x\n", { mode: "none" }), "<function"), "no line-initial <function");
  ok(allows(at("see ", { mode: "none" }), "<function"), "mid-line is prose");
});
Deno.test("strict: the lazy bare trigger commits only on a declared name's start", () => {
  const C = at("ok\n<function=");
  eq(C.state.k, "F", "\"<function=\" alone is still free text");
  ok(allows(C, "g") && allows(C, "x"), "anything may follow");
  const D = at("ok\n<function=ge");
  eq([D.state.k, D.state.tag], ["L", "name"], "committed: a call named get_...");
  ok(!allows(D, "x") && allows(D, "t"), "and it must finish the name");
  eq(at("see <function=ge").state.k, "F", "only at a line start");
});
Deno.test("strict: thinking. XML closes the reasoning at <tool_call>; JSON keeps drafts inside it; forcing bans the end there", () => {
  const X = at("I will call it <tool_call>", { thinking: true, thinkInPrompt: true });
  eq([X.state.k, X.state.tag], ["L", "open"], "xml: a call inside the reasoning closes it");
  const J = new GrammarConstraint(APITOOLS, { vocabSize: JV.length, tokenText: jt, stops: [JSTOP], style: "json", mode: "auto", thinking: true });
  J.setText("<think>\nmaybe <tool_call>\n{\"name\": \"x\"}");
  eq(J.state.k, "T", "json: a draft in the reasoning is reasoning");
  J.setText("<think>\nok\n</think>\n\n<tool_call>");
  eq([J.state.k, J.state.tag], ["L", "jopen"]);
  const F = at("hmm", { thinking: true, thinkInPrompt: true, mode: "required" });
  ok(!allows(F, "<|im_end|>") && allows(F, "a"), "forcing: no end inside the reasoning");
  const S0 = G({ thinking: true, mode: "required" });
  ok(allows(S0, "<think>") && allows(S0, "<tool_call>") && !allows(S0, "a"), "thinking, the model opens the block (or skips it and calls)");
});
Deno.test("strict: typed XML values (integers, enums unquoted, objects by schema), required params before </function>", () => {
  ok(valid("<tool_call>\n<function=get_weather>\n<parameter=unit>\nc\n</parameter>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"), "any order, enum raw");
  ok(!valid("<tool_call>\n<function=get_weather>\n<parameter=unit>\n\"c\"\n</parameter>"), "a quoted enum is not the value");
  ok(!valid("<tool_call>\n<function=get_weather>\n<parameter=days>\nthree"), "integer");
  ok(valid("<tool_call>\n<function=get_weather>\n<parameter=days>\n-12\n</parameter>"), "integer ok");
  ok(!valid("<tool_call>\n<function=get_weather>\n<parameter=days>\n1.5"), "no fraction for an integer");
  ok(!valid("<tool_call>\n<function=get_weather>\n<parameter=unit>\nc\n</parameter>\n</function>"), "city is required");
  ok(!valid("<tool_call>\n<function=get_weather>\n<parameter=city>\na\n</parameter>\n<parameter=city>"), "no parameter twice");
  ok(valid("<tool_call>\n<function=get_weather>\n<parameter=opts>\n{\"n\": 1e3, \"a\": true}\n</parameter>\n<parameter=city>\nx\n</parameter>\n</function>\n</tool_call>"), "object keys any order, $ref number");
  ok(!valid("<tool_call>\n<function=get_weather>\n<parameter=opts>\n{\"n\": 1}"), "required key a missing: no }");
  ok(!valid("<tool_call>\n<function=get_weather>\n<parameter=opts>\n{\"z\": 1"), "additionalProperties false");
  ok(!valid("<tool_call>\n<function=get_weather>\n<parameter=opts>\n{\"a\": True"), "JSON booleans only");
  ok(valid("<tool_call>\n<function=get_weather>\n<parameter=tags>\n[\"x\", \"y\\n\\u00e9\"]\n</parameter>"), "string arrays with escapes");
  ok(!valid("<tool_call>\n<function=get_weather>\n<parameter=tags>\n[1"), "items are strings");
  ok(!valid("<tool_call>\n<function=get_weather>\n<parameter=tags>\n[\"a\\q"), "bad escape");
});
Deno.test("strict: whitespace in JSON: one space, or a newline and indentation; no floods", () => {
  const fmt = { format: { type: "schema", schema: { type: "object", properties: { a: { type: "array", items: { type: "integer" } }, b: { type: "string" } }, required: ["a"] } } };
  ok(valid('{\n  "a": [\n    1,\n    2\n  ],\n  "b": "x"\n}', fmt), "pretty-printed");
  ok(valid('{"a": [1, 2], "b": "x"}', fmt), "compact with spaces");
  ok(!valid('{\n\n"a"', fmt), "no blank lines");
  ok(!valid('{"a":  [', fmt), "two spaces in a gap");
  ok(!valid('{\n' + " ".repeat(9) + '"a"', fmt), "indentation past 4 x (depth + 1)");
  ok(!valid('{"a":\t[', fmt), "no tabs");
  const C = at('{"a": [1]}', fmt);
  ok(allows(C, "<|im_end|>") && allows(C, "\n") && !allows(C, "a"), "the value, then the end");
  ok(!allows(at('{"a": [1]', fmt), "<|im_end|>"), "not before it is complete");
});
Deno.test("strict: minItems / maxItems bound array lengths", () => {
  const fmt = (a) => ({ format: { type: "schema", schema: { type: "object", properties: { a: { type: "array", items: { type: "integer" }, ...a } }, required: ["a"] } } });
  const m = fmt({ minItems: 1, maxItems: 3 });
  ok(!valid('{"a": []', m), "not fewer than minItems");
  ok(valid('{"a": [1]}', m) && valid('{"a": [1, 2, 3]}', m), "1 to 3 items");
  ok(!valid('{"a": [1, 2, 3,', m), "no fourth item");
  ok(!valid('{"a": [1, 2]', fmt({ minItems: 3 })) && valid('{"a": [1, 2, 3, 4]}', fmt({ minItems: 3 })), "a minimum alone");
  ok(valid('{"a": []}', fmt({ maxItems: 0 })) && !valid('{"a": [1', fmt({ maxItems: 0 })), "maxItems 0: only []");
  ok(valid('{"a": [1, 2, 3]}', fmt({ minItems: 5, maxItems: 2 })), "an impossible pair is ignored");
});
Deno.test("strict: format only, and format with tools (a call or the value)", () => {
  const f = { format: { type: "json" } };
  const C = at("", f);
  ok(allows(C, "{") && !allows(C, "hello world") && !allows(C, "<|im_end|>") && !allows(C, "[") , "json_object: an object");
  const U = at("", { format: { type: "json" } }, APITOOLS);
  ok(allows(U, "{") && allows(U, "<tool_call>") && !allows(U, "a"), "auto + format: either");
  const T = at("\n", { format: { type: "json" }, thinking: true, thinkInPrompt: true }, []);
  ok(!allows(T, "<|im_end|>"), "format only: no end while thinking");
});
Deno.test("strict: schema caps throw (the host answers bad)", () => {
  let err = null;
  try { G({}, [{ name: "t", parameters: { type: "object", properties: { e: { enum: Array.from({ length: 1001 }, (_, i) => i) } } } }]); } catch (e) { err = e; }
  ok(err instanceof SchemaError, String(err));
});
Deno.test("strict: masks live in one LRU per tokenizer, bounded in bytes", () => {
  const cache = maskCacheFor((i) => JV[i], 64 * 40);
  for (const t of ["a", "<tool_call>\n", "<tool_call>\n<function=", "<tool_call>\n<function=noop>", '{"a"']) {
    const C = new GrammarConstraint(APITOOLS, { vocabSize: JV.length, tokenText: (i) => JV[i], stops: [JSTOP], mode: "auto", maskCache: cache, format: t.startsWith("{") ? { type: "json" } : null });
    C.setText(t); C.allowed();
  }
  ok(cache.bytes <= 64 * 40 || cache.size === 1, "evicted to the budget: " + cache.bytes);
  const A = at("<tool_call>\n<function="), B = at("<tool_call>\n<function=");
  ok(A.allowed() === B.allowed(), "shared across requests with the same tools");
});
Deno.test("strict: the candidate fast path gives what the full mask gives (10,000 random trials)", () => {
  const rnd = mulberry(11);
  const texts = ["", "<tool_call>\n", "<tool_call>\n<function=get_weather>\n<parameter=", "<tool_call>\n<function=get_weather>\n<parameter=days>\n1", "<tool_call>\n<function=get_weather>\n<parameter=opts>\n{\"a\": ", "ok\n<function=g"];
  for (let trial = 0; trial < 10000; trial++) {
    const text = texts[trial % texts.length];
    const k = [1, 3, 8][trial % 3];
    const lg = Float32Array.from(JV, () => rnd() * 10 - 5);
    for (let j = 0; j < 3; j++) lg[Math.floor(rnd() * JV.length)] = 8 + rnd();
    const fast = at(text, { maskCache: new Map() });
    const a = fast.mask(Float32Array.from(lg), k);
    const full = at(text); const m = full.allowed();
    const b = Float32Array.from(lg);
    if (m) { if (m.allow) { const keep = new Set(m.allow); for (let i = 0; i < b.length; i++) if (!keep.has(i)) b[i] = -Infinity; } else for (const i of m.deny) b[i] = -Infinity; }
    const top = (x) => [...x.keys()].filter((i) => x[i] > -Infinity).sort((p, q) => x[q] - x[p] || p - q).slice(0, k);
    eq(top(a), top(b), `trial ${trial} (${JSON.stringify(text)}, k ${k})`);
  }
});
Deno.test("strict sampler: the garbage guard counts forced values only; a named call with enums is not garbage", () => {
  const argmaxS = (lg) => { let b = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[b]) b = i; return b; };
  argmaxS.gpu = { kind: "greedy" };
  const mk2 = () => constrainedSampler(argmaxS, APITOOLS, { tokenText: jt, vocabSize: JV.length, stops: [JSTOP], style: "xml", mode: { name: "get_weather" } });
  const want = (w) => { const lg = new Float32Array(JV.length).fill(0); lg[JV.indexOf("hello world")] = 9; lg[JV.indexOf(w)] = 5; return lg; };
  // the model keeps wanting prose, but every forced token is a literal or a name: not garbage
  let cs = mk2(), out = "";
  cs.setText("");
  for (let s = 0; s < 40; s++) { const t = cs.sample(want("<parameter=")); if (t === JSTOP) break; out += jt(t); cs.keep(1); cs.setText(out); }
  ok(cs.forced > 5 && !cs.garbage, `forced ${cs.forced} literals: not garbage (${JSON.stringify(out)})`);
  // forced inside a free value (the model wants a raw newline flood in a JSON string): garbage
  cs = constrainedSampler(argmaxS, APITOOLS, { tokenText: jt, vocabSize: JV.length, stops: [JSTOP], style: "xml", mode: "auto" });
  out = "<tool_call>\n<function=get_weather>\n<parameter=tags>\n[\"";
  cs.setText(out);
  const flood = () => { const lg = new Float32Array(JV.length).fill(0); lg[JV.indexOf("\n\n\n")] = 9; lg[JV.indexOf("\t")] = 8; lg[JV.indexOf("a")] = 1; return lg; };
  for (let s = 0; s < 30 && !cs.garbage; s++) { out += jt(cs.sample(flood())); cs.keep(1); cs.setText(out); }
  ok(cs.garbage, "forced free values past the gate are garbage");
  cs = constrainedSampler(argmaxS, APITOOLS, { tokenText: jt, vocabSize: JV.length, stops: [JSTOP], style: "xml", mode: "auto" });
  cs.setText("<tool_call>\n");
  const nan = new Float32Array(JV.length).fill(NaN);
  cs.sample(nan); cs.keep(1);
  ok(cs.garbage, "NaN logits are garbage at once");
});
