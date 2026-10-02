// Goldens for MODEL=122b test_moe.js: greedy continuations from a running llama-server (llama.cpp CUDA, same GGUF) on
// test_moe's three chat prompts, sent as token ids from the engine's own tokenizer so both sides see the same prompt.
//   deno run --allow-read --allow-net --allow-write tests/golden/make_q122_llama.js [http://127.0.0.1:8199] [n=40]
import { openGGUF, Q122_PATH } from "../load_model.js";
const URL_ = Deno.args[0] || "http://127.0.0.1:8199", N = +(Deno.args[1] || 40);
const model = openGGUF(Q122_PATH, { cache: false });
const tok = model.tokenizer(), V = tok.vocab;
const chat = (q) => [V["<|im_start|>"], ...tok.encode("user\n" + q), V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n"), V["<think>"], ...tok.encode("\n\n"), V["</think>"], ...tok.encode("\n\n")];
const QS = { "two-sum": "Write the Python code for two sum. Code only.", "hash-map": "Explain what a hash map is in two sentences.", "bash": "Write a bash one-liner that counts lines in all .js files." };
const out = {};
for (const [name, q] of Object.entries(QS)) {
  const r = await fetch(URL_ + "/completion", { method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt: chat(q), n_predict: N, temperature: 0, top_k: 1, cache_prompt: false, return_tokens: true }) });
  const j = await r.json();
  out[name] = { q, text: j.content, ids: j.tokens };
  console.log(name, JSON.stringify(j.content));
}
const p = new URL("./q122_llama_greedy.json", import.meta.url);
Deno.writeTextFileSync(p, "{\n" + Object.entries({ _source: "llama.cpp 749f688 CUDA llama-server, greedy, bartowski Qwen_Qwen3.5-122B-A10B-Q4_0 (2-file split)", ...out }).map(([k, v]) => ` ${JSON.stringify(k)}: ${JSON.stringify(v)}`).join(",\n") + "\n}\n");
