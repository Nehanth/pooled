// harness/model-common.js encodeTurn: an assistant turn re-tokenized from its text (its sampled ids
// lost: a new tokenizer, a restored session) keeps the tags the model writes as single added tokens.
// Plain encode spells "<tool_call>" out in byte pieces, and the model then copies that spelling on
// the next request ("<tool_tool_calls>"), which no longer parses as a call.
import { makeTokenizer } from "../../engine/tokenizer.js";
import { encodeTurn } from "../../harness/model-common.js";
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

// a byte-level tokenizer.json with no merges (one token per byte) and Qwen's tool/think added tokens
function tinyTok() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const vocab = {};
  cs.forEach((c, i) => { vocab[String.fromCharCode(c)] = i; });
  const added = ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<think>", "</think>", "<tool_response>", "</tool_response>"]
    .map((content, i) => ({ id: 1000 + i, content }));
  return makeTokenizer({ model: { vocab, merges: [] }, added_tokens: added });
}

Deno.test("encodeTurn: tool and think tags become their single added tokens", () => {
  const tok = tinyTok();
  const text = "I'll edit it.\n<tool_call>\n<function=edit_file>\n<parameter=path>\nstyle.css\n</parameter>\n</function>\n</tool_call>";
  const plain = tok.encode(text), ids = encodeTurn(tok, text);
  ok(!plain.includes(tok.vocab["<tool_call>"]), "plain encode spells the tag out (the bug this guards against)");
  ok(ids.includes(tok.vocab["<tool_call>"]) && ids.includes(tok.vocab["</tool_call>"]), "the call's tags are single tokens");
  ok(tok.decode(ids) === text, "the ids decode back to the same text");
  ok(ids.length < plain.length, "shorter than the spelled-out form");
});

Deno.test("encodeTurn: text without tags is unchanged; other specials stay text", () => {
  const tok = tinyTok();
  const t = "plain answer, no calls";
  ok(JSON.stringify(encodeTurn(tok, t)) === JSON.stringify(tok.encode(t)));
  // a chat-template token written in a file's content must not become a control token
  const c = "<|im_end|> in a string";
  ok(!encodeTurn(tok, c).includes(tok.vocab["<|im_end|>"]), "<|im_end|> is not turned into the control token");
  const th = "<think>\nhmm\n</think>\n\nok";
  const ids = encodeTurn(tok, th);
  ok(ids[0] === tok.vocab["<think>"] && ids.includes(tok.vocab["</think>"]) && tok.decode(ids) === th);
});
