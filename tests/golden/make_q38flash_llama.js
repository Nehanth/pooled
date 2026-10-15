// Goldens for Qwen3.8-Flash-Next (roadmap/33-qwen38-flash.md, section 5): greedy continuations from a running llama-server that
// knows the qwen4exp arch (mainline >= 6c84c7d; the unsloth b10840 build made the first ones), same bartowski
// Q4_0 GGUF, prompts sent as token ids from the engine's own tokenizer so both sides see the same prompt.
//   llama-server -m models/q38flash/Qwen3.8-Flash-Next-Q4_0-00001-of-00003.gguf -ngl 99 -fa on -c 131584 -np 1 \
//     -ot "per_layer_token_embd=CPU" --fit off --port 8297
//   BUILD="<llama.cpp build>" deno run --allow-read --allow-net --allow-write --allow-env tests/golden/make_q38flash_llama.js [http://127.0.0.1:8297] [n=64] [cases]
// Each case records the llama.cpp build that made it ("build", from BUILD; the cases made before the field: unsloth
// b10840). Since 2026-10-15 every case is from mainline llama.cpp (f2cc728, CUDA 13.0 / its CCCL 3.0): b10840's QSA graph
// (build_qsa_top_k, a 2051-cell top-k over per-cell copies of the block scores, CUB's non-deterministic top-k) is not
// the model's selection past ~2051 cells (llama.cpp issue #28497); mainline's build_qsa_sel (the top 512 whole blocks
// and the own tail block, an exact top-k) is what engine/ and tests/reference/qsa_ref.mjs follow (docs/kernels.md).
// Cases: test_moe's three chat prompts; "eos-mid", an <|endoftext|> in the middle of the prompt (the n-gram
// hash resets there); "ctx2k", just under the dense path's 2048 cap (prompt + ids); "ctx4k" / "ctx8k", past
// QSA's 2051-cell budget, for the sparse-attention phase; and "ctx2k_b" (prompt and continuation stay under 2051
// cells: dense, at long positions) / "ctx2k1" (just past 2051: QSA drops a few blocks), which split a long-context
// mismatch into position drift vs the QSA selection (tests/test_q38flash_batch.js). "ctx16k" / "ctx32k" (roadmap
// then docs notes, ~16K / ~32K tokens) and "needle30k" (a passphrase sentence planted ~2.5K tokens into ~30K
// tokens of notes, asked for at the end; the case keeps "needle", the passphrase, "found", whether llama.cpp's
// answer has it, and "foundRuns", the same for each of its REPEAT runs, golden first) are the long-context goldens (tests/test_q38flash_batch.js LONG=1); they need -c >= 32.5K (-c 9216
// was enough before them). [cases] (comma separated) regenerates only those and keeps the file's others.
// DRY=1 prints the picked cases' prompt sizes (built fresh, and as stored in the file) and exits.
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
// the 16K / 32K cases need more text than roadmap/ holds (~30K tokens): roadmap/*.md then docs/*.md, each in name order
const docLong = (budget, needle = null, needleAt = 0) => {
  const lines = [];
  // (64K / 128K need more: docs/archive, docs/research and the docs site after them; the shorter cases' text is unchanged)
  for (const d of ["roadmap", "docs", "docs/archive", "docs/research", "docs-site/src/content/docs"]) {
    const dir = new URL(`../../${d}/`, import.meta.url);
    const names = [...Deno.readDirSync(dir)].map((e) => e.name).filter((n) => n.endsWith(".md") || n.endsWith(".mdx")).sort();
    for (const n of names) lines.push(...Deno.readTextFileSync(new URL(n, dir)).split("\n"));
  }
  let text = "", n = 0, planted = needle === null;
  for (const l of lines) {
    if (!planted && n >= needleAt) { text += needle + "\n"; n += tok.encode(needle + "\n").length; planted = true; }
    const k = tok.encode(l + "\n").length; if (n + k > budget) break; text += l + "\n"; n += k;
  }
  if (n < budget - 400) throw new Error(`docLong: only ${n} tokens of text for a ${budget}-token budget`);
  return text;
};
const longer = (budget) => chat("Here are some design notes.\n\n" + docLong(budget) + "\nWhich of these notes is about running models from disk, and what does it propose? Answer in three sentences.");
// needle in a haystack: one sentence with a made-up passphrase planted ~NEEDLE_AT tokens into the notes, asked for at the end
const NEEDLE = { pass: "amber-falcon-4127", sentence: "Note for whoever reads this far: the passphrase for the staging room is amber-falcon-4127. Keep it in mind." };
const needle = (budget, at) => chat("Here are some design notes.\n\n" + docLong(budget, NEEDLE.sentence, at) + "\nSomewhere in these notes there is a passphrase for the staging room. What is it? Reply with the passphrase, then one sentence on where in the notes it appeared.");

// each case is built only when picked ([cases]): the long ones tokenize ~80K tokens of notes, and docLong throws when
// roadmap/ + docs/ no longer hold enough text
const cases = Object.fromEntries(Object.entries(QS).map(([k, q]) => [k, () => ({ q, prompt: chat(q) })]));
cases["eos-mid"] = () => ({ q: "<|endoftext|> mid-prompt, then: Write a haiku about the sea.",
  prompt: [...tok.encode("The quick brown fox jumps over the lazy dog."), V["<|endoftext|>"], ...chat("Write a haiku about the sea.")] });
cases["ctx2k"] = () => ({ q: "roadmap notes (~1.9K tokens), which note is about disk", prompt: long(1880) });
cases["ctx4k"] = () => ({ q: "roadmap notes (~4K tokens), which note is about disk", prompt: long(3900) });
cases["ctx8k"] = () => ({ q: "roadmap notes (~8K tokens), which note is about disk", prompt: long(7900) });
cases["ctx2k_b"] = () => ({ q: "roadmap notes (~2K tokens, prompt + 64 <= 2051 cells), which note is about disk", prompt: long(1950) });
cases["ctx2k1"] = () => ({ q: "roadmap notes (~2.1K tokens, just past 2051 cells), which note is about disk", prompt: long(2070) });
// long context (tests/test_q38flash_batch.js LONG=1): ~16K and ~32K tokens of notes, and a needle in ~30K
cases["ctx16k"] = () => ({ q: "roadmap + docs notes (~16K tokens), which note is about disk", prompt: longer(15900) });
cases["ctx32k"] = () => ({ q: "roadmap + docs notes (~32K tokens), which note is about disk", prompt: longer(32300) });
cases["needle30k"] = () => ({ q: `needle: "${NEEDLE.sentence}" planted ~2.5K tokens into ~30K tokens of notes, asked at the end`, prompt: needle(30000, 2500), needle: NEEDLE.pass });
// rooms at 64K / 128K (rooms long-context PR 5): prompt + 64 ids within the room's context; a needle ~8K tokens into
// ~120K. Need -c >= 131584
const NEEDLE2 = { pass: "cobalt-heron-8830", sentence: "Note for whoever reads this far: the passphrase for the demo room is cobalt-heron-8830. Keep it in mind." };
const needle2 = (budget, at) => chat("Here are some design notes.\n\n" + docLong(budget, NEEDLE2.sentence, at) + "\nSomewhere in these notes there is a passphrase for the demo room. What is it? Reply with the passphrase, then one sentence on where in the notes it appeared.");
cases["ctx64k"] = () => ({ q: "roadmap + docs notes (~64K tokens), which note is about disk", prompt: longer(65300) });
cases["ctx128k"] = () => ({ q: "roadmap + docs notes (~128K tokens), which note is about disk", prompt: longer(130800) });
cases["needle120k"] = () => ({ q: `needle: "${NEEDLE2.sentence}" planted ~8K tokens into ~120K tokens of notes, asked at the end`, prompt: needle2(120000, 8000), needle: NEEDLE2.pass });
const picked = Object.keys(cases).filter((name) => !ONLY || ONLY.includes(name));

const p = new URL("./q38flash_llama_greedy.json", import.meta.url);
let old = {};
try { old = JSON.parse(Deno.readTextFileSync(p)); } catch { /* first run */ }
if (Deno.env.get("DRY") === "1") {   // prompt sizes only, no server: a fresh build next to the stored prompt (what the tests run)
  for (const name of picked) {
    const st = old[name]?.prompt;
    let fresh;
    try {
      const f = cases[name]().prompt;
      fresh = `${f.length}${st ? (f.length === st.length && f.every((x, i) => x === st[i]) ? " (same ids as stored)" : " (ids differ from stored)") : ""}`;
    } catch (e) { fresh = `(cannot build: ${e.message})`; }
    console.log(name, `fresh ${fresh} prompt tokens, stored ${st ? st.length : "none"} (the tests run the stored ids)`);
  }
  Deno.exit(0);
}
const NEAR_TIE = 0.2;
const nearTies = (gap) => {
  const ties = gap.flatMap((g, i) => (g !== null && g < NEAR_TIE ? [[i, g]] : []));
  return { nearTie: ties.length ? ties[0][0] : -1, nearTies: ties };
};
const TOPK = 5, REPEAT = +(Deno.env.get("REPEAT") || 3);
const B10840 = "unsloth llama.cpp b10840-mix-d5c17a0 CUDA";
const BUILD = Deno.env.get("BUILD") || (() => { throw new Error("BUILD=<the llama-server's build> (recorded per case)"); })();
for (const v of Object.values(old)) if (v && typeof v === "object" && !Array.isArray(v) && !v.build) v.build = B10840;
const out = ONLY ? Object.fromEntries(Object.entries(old).filter(([k]) => !k.startsWith("_"))) : {};
for (const name of picked) {
  const keep = Deno.env.get("FRESH") !== "1" && old[name]?.prompt;
  const c = keep ? { q: old[name].q, prompt: old[name].prompt, ...(old[name].needle ? { needle: old[name].needle } : {}) } : cases[name]();
  const q = c.q, prompt = c.prompt;
  const ask = async () => (await fetch(URL_ + "/completion", { method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt, n_predict: N, temperature: 0, top_k: 1, cache_prompt: false, return_tokens: true, n_probs: TOPK, post_sampling_probs: false }) })).json();
  const topOf = (j) => (j.completion_probabilities || []).map((p) => (p.top_logprobs || []).slice(0, TOPK).map((t) => [t.id, +t.logprob.toFixed(3)]));
  const j = await ask();
  const cp = j.completion_probabilities || [];
  const gap = cp.map((p) => { const t = p.top_logprobs || []; return t.length > 1 ? +(t[0].logprob - t[1].logprob).toFixed(3) : null; });
  const top = topOf(j), runs = [], foundRuns = c.needle ? [j.content.includes(c.needle)] : null;
  for (let k = 1; k < REPEAT; k++) {
    const jr = await ask();
    if (foundRuns) foundRuns.push(jr.content.includes(c.needle));
    if (jr.tokens.some((x, i) => x !== j.tokens[i]) || jr.tokens.length !== j.tokens.length) runs.push({ ids: jr.tokens, top: topOf(jr) });
  }
  out[name] = { q, build: BUILD, ...(c.needle ? { needle: c.needle, found: j.content.includes(c.needle), foundRuns } : {}), text: j.content, ids: j.tokens, ...nearTies(gap), gap, top, reproducible: runs.length ? false : REPEAT, ...(runs.length ? { runs } : {}), prompt };
  console.log(name, `${prompt.length} prompt tokens, near tie at ${out[name].nearTie}, ${runs.length ? `${runs.length} of ${REPEAT - 1} reruns differ (first at ${runs.map((r) => r.ids.findIndex((x, i) => x !== j.tokens[i])).join(",")})` : `reproducible x${REPEAT}`}:`, JSON.stringify(j.content));
}
Deno.writeTextFileSync(p, "{\n" + Object.entries({ _source: `llama.cpp CUDA llama-server on the GB10 (each case's "build"), greedy, bartowski Qwen3.8-Flash-Next-Q4_0 (3-file split), -fa on, per_layer_token_embd on the CPU, n_probs ${TOPK}`, ...out }).map(([k, v]) => ` ${JSON.stringify(k)}: ${JSON.stringify(v)}`).join(",\n") + "\n}\n");
