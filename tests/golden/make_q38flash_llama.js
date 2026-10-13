// Goldens for Qwen3.8-Flash-Next (roadmap/33-qwen38-flash.md, section 5): greedy continuations from a running llama-server that
// knows the qwen4exp arch (the unsloth b10840 build on the GB10; mainline needs >= 6c84c7d), same bartowski
// Q4_0 GGUF, prompts sent as token ids from the engine's own tokenizer so both sides see the same prompt.
//   llama-server -m models/q38flash/Qwen3.8-Flash-Next-Q4_0-00001-of-00003.gguf -ngl 99 -fa on -c 33792 -np 1 \
//     -ot "per_layer_token_embd=CPU" --port 8297
//   deno run --allow-read --allow-net --allow-write tests/golden/make_q38flash_llama.js [http://127.0.0.1:8297] [n=64] [cases]
// Cases: test_moe's three chat prompts; "eos-mid", an <|endoftext|> in the middle of the prompt (the n-gram
// hash resets there); "ctx2k", just under the dense path's 2048 cap (prompt + ids); "ctx4k" / "ctx8k", past
// QSA's 2051-cell budget, for the sparse-attention phase; and "ctx2k_b" (prompt and continuation stay under 2051
// cells: dense, at long positions) / "ctx2k1" (just past 2051: QSA drops a few blocks), which split a long-context
// mismatch into position drift vs the QSA selection (tests/test_q38flash_batch.js); "ctx32k" (~31.6K prompt tokens,
// the long-context target: QSA keeps 2048 of ~32K cells, the wide prefill's 1024-column passes). [cases] (comma separated)
// regenerates only those and keeps the file's others.
// Each case keeps its prompt ids, the greedy text and ids, and per step the gap between the top two
// log-probabilities ("gap", nats). A step under 0.2 is a near tie that the engine may legitimately flip:
// "nearTies" lists every one as [step, gap], "nearTie" is the first (or -1), so a test compares tokens only up
// to there, or up to the first gap under the engine's own measured noise.
// "top" keeps llama.cpp's top TOPK (5) per step as [id, logprob] pairs, for the backend near-tie rule
// (tests/golden/q38flash_rule.js): a step where the engine picks another token passes when that token is in this
// list within 0.5 nats of llama.cpp's top and the engine's own top-2 gap there is under 0.5.
// llama.cpp is not reproducible on the long prompts (CUDA, > ~2K tokens: the same request gives other logprobs and,
// near a tie, other ids; measured 2026-10-12, docs/kernels.md): each case runs REPEAT (3) times, the first run is
// the golden, and when another run's ids differ, "runs" keeps those runs' { ids, top } (the rule may take llama.cpp's
// top-k from any run whose ids agree with the golden's up to the step). "reproducible": all REPEAT runs gave the same ids.
// A case already in the file keeps its prompt ids (the long cases' prompts are built from roadmap/*.md, which
// changes; regenerating must not move the prompt under the other goldens). FRESH=1 rebuilds them.
import { openGGUF } from "../load_model.js";
const URL_ = Deno.args[0] || "http://127.0.0.1:8297", N = +(Deno.args[1] || 64), ONLY = Deno.args[2]?.split(",") || null;
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
cases["ctx2k"] = { q: "roadmap notes (~1.9K tokens), which note is about disk", prompt: long(1880) };
cases["ctx4k"] = { q: "roadmap notes (~4K tokens), which note is about disk", prompt: long(3900) };
cases["ctx8k"] = { q: "roadmap notes (~8K tokens), which note is about disk", prompt: long(7900) };
cases["ctx2k_b"] = { q: "roadmap notes (~2K tokens, prompt + 64 <= 2051 cells), which note is about disk", prompt: long(1950) };
cases["ctx2k1"] = { q: "roadmap notes (~2.1K tokens, just past 2051 cells), which note is about disk", prompt: long(2070) };
cases["ctx32k"] = { q: "roadmap notes (~32K tokens), which note is about disk", prompt: long(31600) };

const NEAR_TIE = 0.2;
const nearTies = (gap) => {
  const ties = gap.flatMap((g, i) => (g !== null && g < NEAR_TIE ? [[i, g]] : []));
  return { nearTie: ties.length ? ties[0][0] : -1, nearTies: ties };
};
const TOPK = 5, REPEAT = +(Deno.env.get("REPEAT") || 3);
const p = new URL("./q38flash_llama_greedy.json", import.meta.url);
let old = {};
try { old = JSON.parse(Deno.readTextFileSync(p)); } catch { /* first run */ }
const out = ONLY ? Object.fromEntries(Object.entries(old).filter(([k]) => !k.startsWith("_"))) : {};
for (const [name, c] of Object.entries(cases)) {
  if (ONLY && !ONLY.includes(name)) continue;
  const q = c.q, prompt = Deno.env.get("FRESH") !== "1" && old[name]?.prompt ? old[name].prompt : c.prompt;
  const ask = async () => (await fetch(URL_ + "/completion", { method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt, n_predict: N, temperature: 0, top_k: 1, cache_prompt: false, return_tokens: true, n_probs: TOPK, post_sampling_probs: false }) })).json();
  const topOf = (j) => (j.completion_probabilities || []).map((p) => (p.top_logprobs || []).slice(0, TOPK).map((t) => [t.id, +t.logprob.toFixed(3)]));
  const j = await ask();
  const cp = j.completion_probabilities || [];
  const gap = cp.map((p) => { const t = p.top_logprobs || []; return t.length > 1 ? +(t[0].logprob - t[1].logprob).toFixed(3) : null; });
  const top = topOf(j), runs = [];
  for (let k = 1; k < REPEAT; k++) {
    const jr = await ask();
    if (jr.tokens.some((x, i) => x !== j.tokens[i]) || jr.tokens.length !== j.tokens.length) runs.push({ ids: jr.tokens, top: topOf(jr) });
  }
  out[name] = { q, text: j.content, ids: j.tokens, ...nearTies(gap), gap, top, reproducible: runs.length ? false : REPEAT, ...(runs.length ? { runs } : {}), prompt };
  console.log(name, `${prompt.length} prompt tokens, near tie at ${out[name].nearTie}, ${runs.length ? `${runs.length} of ${REPEAT - 1} reruns differ (first at ${runs.map((r) => r.ids.findIndex((x, i) => x !== j.tokens[i])).join(",")})` : `reproducible x${REPEAT}`}:`, JSON.stringify(j.content));
}
Deno.writeTextFileSync(p, "{\n" + Object.entries({ _source: `unsloth llama.cpp b10840-mix-d5c17a0 CUDA llama-server on the GB10, greedy, bartowski Qwen3.8-Flash-Next-Q4_0 (3-file split), -fa on, per_layer_token_embd on the CPU, n_probs ${TOPK}`, ...out }).map(([k, v]) => ` ${JSON.stringify(k)}: ${JSON.stringify(v)}`).join(",\n") + "\n}\n");
