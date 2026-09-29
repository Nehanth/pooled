// harness/engine-model.js with a fake engine and a word-level tokenizer: prefix reuse between
// calls, stop tokens, and the tool-name constraint applied while sampling.
import { engineModel } from "../../harness/engine-model.js";
import { ContextFull } from "../../harness/model-common.js";
import { Agent } from "../../harness/agent.js";
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

const WORDS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "<function=", "rm_rf>", "read_file>", "<parameter=", "path>", "</parameter>", "</function>", "</tool_call>", "hello", "\n"];
const CHARS = [..."abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;!?'\"-_/<>=(){}[]#\n"];
const VOCAB = [...WORDS, ...CHARS.filter((c) => !WORDS.includes(c))];
const ID = Object.fromEntries(VOCAB.map((w, i) => [w, i]));
const tok = {
  vocab: ID,
  encode(s) {   // greedy longest match
    const out = [];
    for (let i = 0; i < s.length;) {
      let best = null;
      for (const w of VOCAB) if (s.startsWith(w, i) && (!best || w.length > best.length)) best = w;
      if (!best) throw new Error("untokenizable: " + s[i]);
      out.push(ID[best]); i += best.length;
    }
    return out;
  },
  decode: (ids) => ids.map((i) => VOCAB[i]).join(""),
};
// the fake model always prefers script[k] (logit 10) and likes read_file> (5) second
function fakeEngine(script) {
  const e = { maxSeq: 4096, pos: 0, mtp: null, dims: { vocab: VOCAB.length }, step: 0, prefilled: 0, resets: 0,
    reset() { this.pos = 0; this.resets++; }, async prefillTokens(ids) { this.pos += ids.length; this.prefilled += ids.length; },
    async forwardToken() {
      this.pos++;
      const lg = new Float32Array(VOCAB.length);
      lg[ID["read_file>"]] = 5;
      lg[ID[script[Math.min(this.step, script.length - 1)]]] = 10;
      this.step++;
      return lg;
    } };
  return e;
}
const collect = async (it) => { let s = ""; for await (const d of it) s += d; return s; };
const CALL = ["<tool_call>", "\n", "<function=", "rm_rf>", "\n", "</function>", "\n", "</tool_call>", "<|im_end|>"];
const tools = [{ name: "read_file", parameters: { properties: { path: {} } } }];

Deno.test("unconstrained: the model's own choice", async () => {
  const m = engineModel(fakeEngine(CALL), tok, { spec: false });
  eq(await collect(m.generate({ turns: [{ role: "user", text: "hi" }] })), "<tool_call>\n<function=rm_rf>\n</function>\n</tool_call>");
});

Deno.test("with tools: an undeclared function name cannot be sampled", async () => {
  const m = engineModel(fakeEngine(CALL), tok, { spec: false, tools });
  eq(await collect(m.generate({ turns: [{ role: "user", text: "hi" }] })), "<tool_call>\n<function=read_file>\n</function>\n</tool_call>");
});

Deno.test("second call prefills only the new tokens and keeps the sampled ids", async () => {
  const E = fakeEngine(["hello", "<|im_end|>"]);
  const m = engineModel(E, tok, { spec: false });
  const a = await collect(m.generate({ system: "s", turns: [{ role: "user", text: "hi" }] }));
  eq(a, "hello");
  E.step = 0;
  const pre = E.prefilled;
  await collect(m.generate({ system: "s", turns: [{ role: "user", text: "hi" }, { role: "assistant", text: a }, { role: "user", text: "more" }] }));
  eq(E.resets, 1, "no reset on the follow-up");
  const first = tok.encode("<|im_start|>system\ns<|im_end|>\n").length;
  ok(m.stats.reused >= first + 5, `reused ${m.stats.reused} tokens (system prompt alone is ${first})`);
  eq(m.stats.calls, 2);
});

Deno.test("a conversation past the context throws ContextFull, and the agent ends the request with reason context", async () => {
  const E = fakeEngine(["hello", "<|im_end|>"]);
  E.maxSeq = 40;
  const m = engineModel(E, tok, { spec: false });
  let err = null;
  try { await collect(m.generate({ system: "s", turns: [{ role: "user", text: "x".repeat(60) }] })); } catch (e) { err = e; }
  ok(err instanceof ContextFull, "threw " + err);
  eq([err.name, err.max], ["ContextFull", 40]);
  ok(err.tokens > 40, "tokens " + err.tokens);
  eq(E.prefilled, 0, "nothing was prefilled");
  const A = new Agent({ generate: m.generate, tools: [] });
  const r = await A.run("x".repeat(60));
  eq(r.reason, "context");
});
