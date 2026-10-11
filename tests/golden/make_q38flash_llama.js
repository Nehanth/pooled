// Goldens for Qwen3.8-Flash-Next (the design note, roadmap 33, section 5): greedy continuations from a running llama-server that
// knows the qwen4exp arch (the unsloth b10840 build on the GB10; mainline needs >= 6c84c7d), same bartowski
// Q4_0 GGUF, prompts sent as token ids from the engine's own tokenizer so both sides see the same prompt.
//   llama-server -m models/q38flash/Qwen3.8-Flash-Next-Q4_0-00001-of-00003.gguf -ngl 99 -fa on -c 9216 -np 1 \
//     -ot "per_layer_token_embd=CPU" --port 8297
//   deno run --allow-read --allow-net --allow-write tests/golden/make_q38flash_llama.js [http://127.0.0.1:8297] [n=64]
// Cases: test_moe's three chat prompts; "eos-mid", an <|endoftext|> in the middle of the prompt (the n-gram
// hash resets there); and "ctx4k" / "ctx8k", past QSA's 2051-cell budget, for the sparse-attention phase.
// Each case keeps its prompt ids, the greedy text and ids, and per step the gap between the top two
// log-probabilities ("gap", nats): a step under 0.1 is a near tie that the engine may legitimately flip
// ("nearTie", the first such step, or -1).
import { openGGUF } from "../load_model.js";
const URL_ = Deno.args[0] || "http://127.0.0.1:8297", N = +(Deno.args[1] || 64);
const PATH = new URL("../../models/q38flash/Qwen3.8-Flash-Next-Q4_0-00001-of-00003.gguf", import.meta.url).pathname;
const model = openGGUF(PATH, { cache: false });
const tok = model.tokenizer(), V = tok.vocab;
const chat = (q) => [V["<|im_start|>"], ...tok.encode("user\n" + q), V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n"), V["<think>"], ...tok.encode("\n\n"), V["</think>"], ...tok.encode("\n\n")];
const QS = { "two-sum": "Write the Python code for two sum. Code only.", "hash-map": "Explain what a hash map is in two sentences.", "bash": "Write a bash one-liner that counts lines in all .js files." };

// a long document for the 4K/8K cases: the roadmap notes in name order, cut to a token budget at a line end
const doc = (budget) => {
  const dir = new URL("../../roadmap/", import.meta.url);
  const names = [...Deno.readDirSync(dir)].map((e) => e.name).filter((n) => n.endsWith(".md")).sort();
  const lines = names.flatMap((n) => Deno.readTextFileSync(new URL(n, dir)).split("\n"));
  let text = "", n = 0;
  for (const l of lines) { const k = tok.encode(l + "\n").length; if (n + k > budget) break; text += l + "\n"; n += k; }
  return text;
};
const long = (budget) => chat("Here are some design notes.\n\n" + doc(budget) + "\nWhich of these notes is about running models from disk, and what does it propose? Answer in three sentences.");

const cases = Object.fromEntries(Object.entries(QS).map(([k, q]) => [k, { q, prompt: chat(q) }]));
cases["eos-mid"] = { q: "<|endoftext|> mid-prompt, then: Write a haiku about the sea.",
  prompt: [...tok.encode("The quick brown fox jumps over the lazy dog."), V["<|endoftext|>"], ...chat("Write a haiku about the sea.")] };
cases["ctx4k"] = { q: "roadmap notes (~4K tokens), which note is about disk", prompt: long(3900) };
cases["ctx8k"] = { q: "roadmap notes (~8K tokens), which note is about disk", prompt: long(7900) };

const out = {};
for (const [name, { q, prompt }] of Object.entries(cases)) {
  const r = await fetch(URL_ + "/completion", { method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt, n_predict: N, temperature: 0, top_k: 1, cache_prompt: false, return_tokens: true, n_probs: 2, post_sampling_probs: false }) });
  const j = await r.json();
  const gap = (j.completion_probabilities || []).map((p) => { const t = p.top_logprobs || []; return t.length > 1 ? +(t[0].logprob - t[1].logprob).toFixed(3) : null; });
  out[name] = { q, text: j.content, ids: j.tokens, nearTie: gap.findIndex((g) => g !== null && g < 0.1), gap, prompt };
  console.log(name, `${prompt.length} prompt tokens, near tie at ${out[name].nearTie}:`, JSON.stringify(j.content));
}
const p = new URL("./q38flash_llama_greedy.json", import.meta.url);
Deno.writeTextFileSync(p, "{\n" + Object.entries({ _source: "unsloth llama.cpp b10840-mix-d5c17a0 CUDA llama-server on the GB10, greedy, bartowski Qwen3.8-Flash-Next-Q4_0 (3-file split), -fa on, per_layer_token_embd on the CPU", ...out }).map(([k, v]) => ` ${JSON.stringify(k)}: ${JSON.stringify(v)}`).join(",\n") + "\n}\n");
