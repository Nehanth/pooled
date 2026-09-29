// engine/tokenizer.js: the pre-tokenizer split must be the model's own (Qwen's, not GPT-2's), or
// code, newlines, digits and non-English text get different token ids than the model trained on.
// tokenizer_llamacpp.json holds llama.cpp's ids for a corpus of such prompts on four vocabs
// (rebuild it with scripts/tokenizer-fixture.js); our ids must match them exactly.
import { makeTokenizer, preSplitter } from "../../engine/tokenizer.js";

const fixture = JSON.parse(Deno.readTextFileSync(new URL("./tokenizer_llamacpp.json", import.meta.url)));
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };

// tokenizer.json pre_tokenizer blocks as the models in room/models.js ship them
const QWEN3_PRE = { type: "Sequence", pretokenizers: [
  { type: "Split", pattern: { Regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, behavior: "Isolated", invert: false },
  { type: "ByteLevel", add_prefix_space: false, trim_offsets: false, use_regex: false },
] };
const SMOLLM_PRE = { type: "Sequence", pretokenizers: [
  { type: "Digits", individual_digits: true },
  { type: "ByteLevel", add_prefix_space: false, trim_offsets: true, use_regex: true },
] };

function check(name, tj) {
  const tok = makeTokenizer(tj), ids = fixture.vocabs[name].ids;
  fixture.corpus.forEach((text, i) => {
    eq(tok.encode(text), ids[i], `${name}: ${JSON.stringify(text)}`);
    eq(tok.decode(ids[i]), text, `${name} round trip`);
  });
}
const model = (name) => ({ model: { vocab: fixture.vocabs[name].vocab, merges: fixture.vocabs[name].merges } });

for (const name of Object.keys(fixture.vocabs)) {
  Deno.test(`tokenizer: ids match llama.cpp on the ${name} vocab (GGUF tokenizer.ggml.pre)`, () => {
    check(name, { ...model(name), pre: fixture.vocabs[name].pre });
  });
}

Deno.test("tokenizer: Qwen3 tokenizer.json split (with its (?i:) group) matches llama.cpp", () => {
  check("qwen2", { ...model("qwen2"), pre_tokenizer: QWEN3_PRE });
});

Deno.test("tokenizer: SmolLM tokenizer.json split (Digits, then GPT-2) matches llama.cpp", () => {
  check("starcoder", { ...model("starcoder"), pre_tokenizer: SMOLLM_PRE });
});

Deno.test("tokenizer: the old GPT-2 split gives different ids on the Qwen vocab", () => {
  // guards the fixture: if every prompt split the same both ways, the tests above would prove nothing
  const tok = makeTokenizer(model("qwen2")), ids = fixture.vocabs.qwen2.ids;
  const differ = fixture.corpus.filter((t, i) => JSON.stringify(tok.encode(t)) !== JSON.stringify(ids[i]));
  if (differ.length < 10) throw new Error(`only ${differ.length} prompts differ`);
});

Deno.test("preSplitter: the tricky pieces, by hand", () => {
  const qwen = preSplitter({ pre: "qwen2" }), q35 = preSplitter({ pre: "qwen35" });
  const gpt2 = preSplitter({}), smol = preSplitter({ pre_tokenizer: SMOLLM_PRE });
  eq(qwen("x = 12345;"), ["x", " =", " ", "1", "2", "3", "4", "5", ";"], "digits one at a time");
  eq(gpt2("x = 12345;"), ["x", " =", " 12345", ";"]);
  eq(qwen("f(foo).bar"), ["f", "(foo", ").", "bar"], "one punct char joins the word after");
  eq(gpt2("f(foo).bar"), ["f", "(", "foo", ").", "bar"]);
  eq(qwen("{\n\treturn"), ["{\n", "\treturn"], "newline stays with the punct, tab joins the word");
  eq(qwen("a\n\n  b"), ["a", "\n\n", " ", " b"], "a newline run is its own piece");
  eq(qwen("I'M DON'T"), ["I", "'M", " DON", "'T"], "contractions ignore case");
  eq(gpt2("I'M"), ["I", "'", "M"]);
  eq(qwen("e\u0301t\u00e9"), ["e", "\u0301t\u00e9"], "qwen2: a combining mark starts a new word");
  eq(q35("e\u0301t\u00e9"), ["e\u0301t\u00e9"], "qwen35: marks count as letters");
  eq(smol("a  12"), ["a", "  ", "1", "2"], "Digits splits first, then GPT-2 on the rest");
  eq(preSplitter({ pre_tokenizer: { type: "Sequence", pretokenizers: [] } })("a 12"), ["a", " 12"], "no Split: GPT-2");
  eq(preSplitter({ pre: "llama-bpe" })("a 12"), ["a", " 12"], "unknown GGUF pre: GPT-2");
});
