// DP4a wide prefill accuracy evaluation (pre-registered; docs/bench-log.md 2026-10-01 "DP4a accuracy evaluation").
// Decides whether DP4A_DEFAULT.dense (engine/qwen35.js) may be true: the dp4a prefill GEMM against the f32 wide GEMM
// (the dense default) and llama.cpp (CPU llama-server, the reference), on the same token ids.
//
// What dp4a changes: only the wide prefill's projections (prompt chunks of >= 64 tokens, BN = 64). Decode is the same
// kernels in both modes, so every difference below comes from the KV cache / DeltaNet state the prefill wrote.
//
// Distribution metrics (tests/golden/dp4a_eval_prompts.json: 15 prompts, chat template, thinking off):
//   buckets short (< 64 tokens: never reaches the wide GEMM, a control), mid (300-700), 2k, 8k.
//   llama.cpp's greedy continuation of GEN = 256 tokens with its top 20 log-probabilities at every position
//   (n_probs) is the reference; ours is teacher-forced on llama.cpp's tokens, so every position is comparable:
//     KL(llama || ours) over llama's top 20, both renormalized over those 20 ids (nats, mean over positions)
//     top-1 agreement: our argmax == llama.cpp's greedy token
//     top-5 overlap: |our top 5 ∩ llama.cpp's top 5| / 5
//     greedy agreement: first position where our argmax leaves llama.cpp's greedy sequence (== the length of a free
//       greedy run's common prefix with llama.cpp's, since every earlier token was the same)
//     also KL(f32 || dp4a) over the f32 run's own top 20, and #302's max |Δ logprob| over llama's top 20 (first position)
// Downstream (built here, deterministic): 40 items scored exactly, greedy, every prompt >= 64 tokens:
//   20 lookups in an 80-row employee table (~2.3K tokens), 10 arithmetic word problems (3-shot), 10 "what does this
//   Python print" (4-shot); llama.cpp, f32 and dp4a each answer every item.
//
// PRE-REGISTERED DECISION RULE (fixed before any measurement; not changed after):
//   For each model with the dp4a path (27B q38 and Qwen3.5-2B; both must pass), positions = all GEN teacher-forced
//   positions of every prompt >= 64 tokens (mid + 2k + 8k):
//   R1  mean KL_dp4a <= 2 x mean KL_f32                                     (KL vs llama.cpp)
//   R2  top1_f32 - top1_dp4a <= 0.5 percentage points
//   R3  per bucket (mid, 2k, 8k): mean KL_dp4a <= max(3 x KL_f32, KL_f32 + 0.01 nats)
//   R4  downstream: correct_f32 - correct_dp4a <= max(2, 1.96 * sqrt(b + c)), b / c = items only f32 / only dp4a got right
//   All pass -> DP4A_DEFAULT.dense = true (NVIDIA only via dp4aAutoDevice; the FXC/compile fallback unchanged).
//   Reported, not gated: top-5 overlap, greedy agreement, KL(f32 || dp4a), short bucket (must be identical: control).
//
//   MODEL=27b|2b  MODES=wide,dp4a  GEN=256  ONLY=name,name  LLAMA_URL=http://127.0.0.1:8091 (refresh the llama.cpp
//   golden first; GOLD_ONLY=1 then exits before the GPU)  OUT=results.json
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env --allow-net --allow-run --allow-write=$HOME/.cache/swarmllm-weights,golden,/tmp eval_dp4a.js
//   llama.cpp: llama-server -m <model.gguf> -c 9216 -t 20 -np 1 --port 8091 (CPU build)
import { Qwen35Engine } from "../engine/qwen35.js";
import { argmax } from "../engine/engine.js";
import { openGGUF, gpuDevice, watchGpuErrors, trunkLayers, Q38_PATH } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "27b"), GEN = +env("GEN", 256), LLAMA = env("LLAMA_URL", ""), NP = 20;
const MODES = env("MODES", "wide,dp4a").split(","), ONLY = env("ONLY", "") ? env("ONLY", "").split(",") : null;
const PATHS = { "27b": Q38_PATH, "2b": new URL("../models/q35-2b/model.gguf", import.meta.url).pathname };
const GOLD = new URL(`./golden/dp4a_eval_${MODEL}_llama.json`, import.meta.url);
const model = openGGUF(PATHS[MODEL]);
const G = model.G, L = trunkLayers(G), nBlk = G.meta["qwen35.block_count"];
const tok = model.tokenizer(), V = tok.vocab;
const STOP = new Set([V["<|im_end|>"], V["<|endoftext|>"]].filter((x) => x !== undefined));
const chat = (u) => [V["<|im_start|>"], ...tok.encode("user\n" + u), V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n"), V["<think>"], ...tok.encode("\n\n"), V["</think>"], ...tok.encode("\n\n")];
const hash = (ids) => { let h = 2166136261; for (const t of ids) { h ^= t; h = Math.imul(h, 16777619) >>> 0; } return h.toString(16); };

// ---- prompts ----
const fixture = JSON.parse(await Deno.readTextFile(new URL("./golden/dp4a_eval_prompts.json", import.meta.url)));
const PROMPTS = fixture.map((p) => ({ ...p, ids: chat(p.text) })).filter((p) => !ONLY || ONLY.includes(p.name));

// ---- downstream items (deterministic) ----
function downstream() {
  let s = 302; const rnd = () => { s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const pick = (a) => a[Math.floor(rnd() * a.length)];
  const FIRST = ["Ava", "Liam", "Noah", "Emma", "Mia", "Lucas", "Sofia", "Mateo", "Hana", "Kenji", "Priya", "Arjun", "Olga", "Ivan", "Fatima", "Omar", "Chloe", "Felix", "Nora", "Elias"];
  const LAST = ["Lindqvist", "Okafor", "Moreau", "Tanaka", "Novak", "Haddad", "Silva", "Kowalski", "Brennan", "Iyer", "Fischer", "Rossi", "Nakamura", "Petrov", "Mendes", "Larsen", "Dubois", "Kaur", "Walsh", "Yilmaz"];
  const DEPT = ["Finance", "Legal", "Research", "Sales", "Support", "Logistics", "Design", "Security"];
  const CITY = ["Porto", "Osaka", "Calgary", "Tallinn", "Nairobi", "Lyon", "Bergen", "Cusco", "Hobart", "Gdansk", "Accra", "Ghent", "Busan", "Leeds", "Cebu", "Graz"];
  const rows = [], names = new Set(), idsUsed = new Set();
  while (rows.length < 80) {
    const name = `${pick(FIRST)} ${pick(LAST)}`; if (names.has(name)) continue;
    let id; do id = String(10000 + Math.floor(rnd() * 90000)); while (idsUsed.has(id));
    names.add(name); idsUsed.add(id); rows.push({ name, dept: pick(DEPT), id, city: pick(CITY) });
  }
  const table = "Employee directory:\n" + rows.map((r) => `Name: ${r.name} | Department: ${r.dept} | Employee ID: ${r.id} | City: ${r.city}`).join("\n");
  const items = [];
  for (let i = 0; i < 20; i++) {
    const r = rows[Math.floor(rnd() * rows.length)];
    if (i % 10 < 7) items.push({ name: `lookup-id-${i}`, task: "lookup", text: `${table}\n\nWhat is the employee ID of ${r.name}? Answer with the number only.`, answer: r.id, kind: "id", max: 16 });
    else items.push({ name: `lookup-city-${i}`, task: "lookup", text: `${table}\n\nIn which city does ${r.name} work? Answer with the city name only.`, answer: r.city, kind: "city", max: 16 });
  }
  const MATH_SHOT = `Solve each problem. Show brief working and end with "Answer: <number>".

Problem: A farmer has 12 rows of 15 apple trees. 37 trees are cut down. How many trees are left?
Working: 12 x 15 = 180 trees. 180 - 37 = 143.
Answer: 143

Problem: A car uses 6 liters of fuel per 100 km. How many liters does it need for a 450 km trip?
Working: 450 / 100 = 4.5. 4.5 x 6 = 27.
Answer: 27

Problem: Anna buys 3 notebooks at $4.50 each and a pen for $2.25. She pays with a $20 bill. How much change does she get?
Working: 3 x 4.50 = 13.50. 13.50 + 2.25 = 15.75. 20 - 15.75 = 4.25.
Answer: 4.25

Problem: `;
  const MATH = [
    ["A bakery makes 24 muffins per tray. It bakes 7 trays in the morning and 5 trays in the afternoon, then sells 250 muffins. How many muffins are left?", 38],
    ["Tom reads 18 pages per day. His book has 423 pages. After 15 days, how many pages are left to read?", 153],
    ["A train travels at 84 km/h for 2.5 hours and then at 60 km/h for 1.75 hours. How many kilometers does it travel in total?", 315],
    ["A shirt costs $40. It is discounted by 25%, and then a further 10% is taken off the discounted price. What is the final price in dollars?", 27],
    ["Three boxes hold 17, 29 and 46 marbles. All the marbles are shared equally among 4 bags. How many marbles go in each bag?", 23],
    ["A rectangle has a perimeter of 54 cm and a length of 17 cm. What is its area in square centimeters?", 170],
    ["Sara earns $15 per hour for the first 40 hours of a week and 1.5 times that rate for every hour after 40. She works 46 hours. How many dollars does she earn?", 735],
    ["A tank holds 1200 liters. It is 35% full, and then 260 liters are added. How many more liters are needed to fill it?", 520],
    ["The sum of three consecutive even numbers is 138. What is the largest of the three?", 48],
    ["A school has 560 students. 3/8 of them take music, and 40% of the music students also take art. How many students take both?", 84],
  ];
  MATH.forEach(([q, a], i) => items.push({ name: `math-${i}`, task: "math", text: MATH_SHOT + q, answer: String(a), kind: "num", max: 160 }));
  const CODE_SHOT = `For each Python program, give its exact output. Reply with the output only.

Program:
print([n * 2 for n in range(4)])
Output:
[0, 2, 4, 6]

Program:
s = "abcdef"
print(s[1:4], len(s))
Output:
bcd 6

Program:
d = {"a": 1, "b": 2}
d["c"] = d["a"] + d["b"]
print(sorted(d.values()))
Output:
[1, 2, 3]

Program:
total = 0
for i in range(1, 5):
    total += i * i
print(total)
Output:
30

Program:
`;
  const CODE = [
    ["x = [3, 1, 4, 1, 5, 9, 2, 6]\nprint(sorted(x)[2:5])", "[2, 3, 4]"],
    ["s = 'hello world'\nprint(s[::-1].title())", "Dlrow Olleh"],
    ["d = {}\nfor w in 'the cat and the hat and the bat'.split():\n    d[w] = d.get(w, 0) + 1\nprint(d['the'], d['and'], len(d))", "3 2 5"],
    ["def f(n):\n    return 1 if n < 2 else n * f(n - 1)\nprint(f(6) // f(4))", "30"],
    ["a = [i * i for i in range(10) if i % 3 == 0]\nprint(sum(a), a[-1])", "126 81"],
    ["print(list(zip('abc', [1, 2, 3, 4]))[1:])", "[('b', 2), ('c', 3)]"],
    ["x = 7\nfor i in range(3):\n    x = x * 2 - i\nprint(x)", "52"],
    ["words = ['banana', 'kiwi', 'apple', 'fig']\nprint(max(words, key=len), min(words))", "banana apple"],
    ["n = 0\ni = 1\nwhile i < 100:\n    i *= 3\n    n += 1\nprint(n, i)", "5 243"],
    ["print('-'.join(str(i % 4) for i in range(2, 9)))", "2-3-0-1-2-3-0"],
  ];
  CODE.forEach(([c, a], i) => items.push({ name: `code-${i}`, task: "code", text: CODE_SHOT + c + "\nOutput:", answer: a, kind: "exact", max: 32 }));
  return items.map((it) => ({ ...it, ids: chat(it.text) }));
}
function score(it, text) {
  const t = text.trim();
  if (it.kind === "id") return (t.match(/\d{5}/) || [""])[0] === it.answer;
  if (it.kind === "city") return t.toLowerCase().includes(it.answer.toLowerCase());
  if (it.kind === "num") { const ns = t.replace(/(\d),(\d)/g, "$1$2").match(/-?\d+(\.\d+)?/g); return !!ns && Math.abs(+ns.at(-1) - +it.answer) < 1e-6; }
  const line = t.replace(/```[a-z]*\n?/g, "").replace(/`/g, "").split("\n").map((l) => l.trim()).find((l) => l) || "";
  return line === it.answer;
}
const ITEMS = downstream().filter((p) => !ONLY || ONLY.includes(p.name));

// ---- llama.cpp reference ----
async function llama(ids, n, probs) {
  const r = await fetch(`${LLAMA}/completion`, { method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt: ids, n_predict: n, temperature: 0, top_k: 1, samplers: ["top_k"], cache_prompt: false, return_tokens: true, ignore_eos: probs, ...(probs ? { n_probs: NP } : {}) }) });
  if (!r.ok) throw new Error(`llama-server ${r.status}: ${await r.text()}`);
  return r.json();
}
let gold = {}; try { gold = JSON.parse(await Deno.readTextFile(GOLD)); } catch { /* none yet */ }
gold.prompts ??= {}; gold.downstream ??= {};
if (LLAMA) {
  const t0 = performance.now();
  for (const p of PROMPTS) {
    const j = await llama(p.ids, GEN, true);
    const top = j.completion_probabilities.map((c) => c.top_logprobs.map((e) => [e.id, Math.round(e.logprob * 1e4) / 1e4]));
    gold.prompts[p.name] = { n: p.ids.length, h: hash(p.ids), tokens: j.tokens, top };
    console.log(`llama.cpp ${p.name} (${p.ids.length} tokens): ${JSON.stringify(tok.decode(j.tokens.slice(0, 24)))}... ${((performance.now() - t0) / 1e3).toFixed(0)} s`);
    await Deno.writeTextFile(GOLD, JSON.stringify(gold) + "\n");
  }
  for (const it of ITEMS) {
    const j = await llama(it.ids, it.max, false), text = tok.decode(j.tokens.filter((t) => !STOP.has(t)));
    gold.downstream[it.name] = { h: hash(it.ids), tokens: j.tokens, ok: score(it, text) };
    console.log(`llama.cpp ${it.name}: ${score(it, text) ? "ok " : "BAD"} ${JSON.stringify(text.slice(-60))}`);
  }
  await Deno.writeTextFile(GOLD, JSON.stringify(gold) + "\n");
  console.log(`llama.cpp golden written to ${GOLD.pathname}`);
  if (env("GOLD_ONLY", "") === "1") Deno.exit(0);
}
for (const p of PROMPTS) if (gold.prompts[p.name]?.h !== hash(p.ids)) throw new Error(`no llama.cpp golden for ${p.name} with these ids (run with LLAMA_URL)`);

// ---- our engine ----
const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: L < nBlk });
const maxLen = Math.max(...PROMPTS.map((p) => p.ids.length), ...ITEMS.map((i) => i.ids.length));
const maxSeq = Math.ceil((maxLen + GEN + 64) / 256) * 256;
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq, batchCols: 16, coopRowsB: 1, prefillUbatch: 256, prefillDp4a: true });
if (MODES.includes("dp4a") && !eng.dp4aCfg) { console.log("SKIP: prefillDp4a unavailable on this device"); Deno.exit(0); }
console.log(`${MODEL}: ${L} layers, ubatch ${eng.ubatch}, wide BN ${eng.wideCfg?.BN}, dp4a tile ${JSON.stringify(eng.dp4aCfg)}, maxSeq ${maxSeq}`);
const setMode = (m) => { eng.reset(); eng.prefillWide = m !== "narrow"; eng.prefillDp4a = m === "dp4a"; };

// top-k ids of a logits row (descending) and its log-sum-exp
function topk(lg, k) {
  const ids = new Int32Array(k).fill(-1), v = new Float64Array(k).fill(-Infinity);
  let m = -Infinity; for (let i = 0; i < lg.length; i++) if (lg[i] > m) m = lg[i];
  let z = 0;
  for (let i = 0; i < lg.length; i++) {
    const x = lg[i]; z += Math.exp(x - m);
    if (x > v[k - 1]) { let j = k - 1; while (j > 0 && v[j - 1] < x) { v[j] = v[j - 1]; ids[j] = ids[j - 1]; j--; } v[j] = x; ids[j] = i; }
  }
  return { ids: Array.from(ids), lz: m + Math.log(z) };
}
// KL(p || q) with p, q given as log-weights over the same ids, each renormalized over those ids
function klRenorm(lp, lq) {
  const lse = (a) => { const m = Math.max(...a); return m + Math.log(a.reduce((s, x) => s + Math.exp(x - m), 0)); };
  const zp = lse(lp), zq = lse(lq); let kl = 0;
  for (let i = 0; i < lp.length; i++) { const a = lp[i] - zp; kl += Math.exp(a) * (a - (lq[i] - zq)); }
  return Math.max(0, kl);
}
const f32Top = {};   // prompt -> per position [ids, logits] of the first mode (the f32 reference for KL(f32 || dp4a))
async function evalPrompt(p, mode) {
  const g = gold.prompts[p.name], ref = g.tokens.slice(0, GEN);
  setMode(mode);
  const t0 = performance.now();
  await eng.prefillTokens(p.ids.slice(0, -1));
  let lg = await eng.forwardToken(p.ids.at(-1));
  const tPre = performance.now() - t0;
  const pos = [], keep = mode === MODES[0] ? (f32Top[p.name] = []) : null;
  let diverge = -1;
  for (let i = 0; i < ref.length; i++) {
    const tk = topk(lg, NP), top = g.top[i];
    const kl = klRenorm(top.map((e) => e[1]), top.map((e) => lg[e[0]]));
    const top1 = tk.ids[0] === ref[i];
    if (!top1 && diverge < 0) diverge = i;
    const l5 = new Set(top.slice(0, 5).map((e) => e[0])), ov5 = tk.ids.slice(0, 5).filter((t) => l5.has(t)).length / 5;
    const r = { kl, top1, ov5 };
    if (i === 0) r.maxLp = Math.max(...top.map(([id, l]) => Math.abs(lg[id] - tk.lz - l)));
    if (keep) keep.push([tk.ids, tk.ids.map((t) => lg[t])]);
    else if (f32Top[p.name]) { const [fi, fl] = f32Top[p.name][i]; r.klF = klRenorm(fl, fi.map((t) => lg[t])); r.top1F = fi[0] === tk.ids[0]; }
    pos.push(r);
    if (i + 1 < ref.length) lg = await eng.forwardToken(ref[i]);
  }
  return { pos, diverge: diverge < 0 ? ref.length : diverge, tPre };
}
async function evalItem(it, mode) {
  setMode(mode);
  await eng.prefillTokens(it.ids.slice(0, -1));
  let next = argmax(await eng.forwardToken(it.ids.at(-1)));
  const out = [];
  while (!STOP.has(next) && out.length < it.max) { out.push(next); if (out.length < it.max) next = argmax(await eng.forwardToken(next)); }
  const text = tok.decode(out);
  return { ok: score(it, text), text };
}

const mean = (a) => a.length ? a.reduce((s, x) => s + x, 0) / a.length : NaN;
const R = { model: MODEL, gen: GEN, modes: MODES, prompts: {}, downstream: {} };
for (const p of PROMPTS) {
  R.prompts[p.name] = { bucket: p.bucket, n: p.ids.length };
  for (const m of MODES) {
    const r = await evalPrompt(p, m);
    R.prompts[p.name][m] = { kl: mean(r.pos.map((x) => x.kl)), top1: mean(r.pos.map((x) => +x.top1)), ov5: mean(r.pos.map((x) => x.ov5)), diverge: r.diverge, maxLp0: r.pos[0].maxLp,
      klF: r.pos[0].klF === undefined ? undefined : mean(r.pos.map((x) => x.klF)), top1F: r.pos[0].top1F === undefined ? undefined : mean(r.pos.map((x) => +x.top1F)), kls: r.pos.map((x) => x.kl), top1s: r.pos.map((x) => +x.top1), prefillS: r.tPre / 1e3 };
    const q = R.prompts[p.name][m];
    console.log(`${p.name.padEnd(18)} ${String(p.ids.length).padStart(5)} ${m.padEnd(5)} KL ${q.kl.toFixed(5)} top1 ${(100 * q.top1).toFixed(1)}% top5 ${(100 * q.ov5).toFixed(1)}% greedy==llama ${q.diverge} maxLp0 ${q.maxLp0.toFixed(3)}`
      + (q.klF !== undefined ? ` KL(f32||dp4a) ${q.klF.toFixed(5)} top1==f32 ${(100 * q.top1F).toFixed(1)}%` : "") + ` prefill ${q.prefillS.toFixed(1)} s`);
  }
}
for (const it of ITEMS) {
  R.downstream[it.name] = { task: it.task, llama: gold.downstream[it.name]?.h === hash(it.ids) ? gold.downstream[it.name].ok : null };
  for (const m of MODES) { const r = await evalItem(it, m); R.downstream[it.name][m] = r.ok; R.downstream[it.name][m + "Text"] = r.text; }
  console.log(`${it.name.padEnd(16)} llama ${R.downstream[it.name].llama} ${MODES.map((m) => `${m} ${R.downstream[it.name][m]}`).join(" ")}  ${JSON.stringify(R.downstream[it.name][MODES.at(-1) + "Text"].slice(-50))}`);
}

// ---- summary and the pre-registered rule ----
const [F, D] = ["wide", "dp4a"];
const pool = (bk, m, f) => PROMPTS.filter((p) => bk.includes(p.bucket)).flatMap((p) => R.prompts[p.name][m][f]);
const S = {};
for (const m of MODES) {
  S[m] = {};
  for (const [nm, bk] of [["gated", ["mid", "2k", "8k"]], ["short", ["short"]], ["mid", ["mid"]], ["2k", ["2k"]], ["8k", ["8k"]]]) {
    const ps = PROMPTS.filter((p) => bk.includes(p.bucket)).map((p) => R.prompts[p.name][m]);
    S[m][nm] = { kl: mean(pool(bk, m, "kls")), top1: 100 * mean(pool(bk, m, "top1s")), ov5: 100 * mean(ps.map((x) => x.ov5)), diverge: mean(ps.map((x) => x.diverge)),
      klF: ps[0]?.klF === undefined ? undefined : mean(ps.map((x) => x.klF)), top1F: ps[0]?.top1F === undefined ? undefined : 100 * mean(ps.map((x) => x.top1F)) };
  }
  S[m].downstream = Object.values(R.downstream).filter((x) => x[m]).length;
}
S.llamaDownstream = Object.values(R.downstream).filter((x) => x.llama).length;
console.log("\nbucket   mode  KL(llama||ours)  top1%   top5%  greedy==llama  KL(f32||dp4a)  top1==f32%");
for (const nm of ["short", "mid", "2k", "8k", "gated"]) for (const m of MODES) {
  const s = S[m][nm];
  console.log(`${nm.padEnd(8)} ${m.padEnd(5)} ${s.kl.toFixed(5).padStart(15)} ${s.top1.toFixed(2).padStart(7)} ${s.ov5.toFixed(2).padStart(7)} ${s.diverge.toFixed(1).padStart(14)} ${s.klF === undefined ? "" : s.klF.toFixed(5).padStart(14)} ${s.top1F === undefined ? "" : s.top1F.toFixed(2).padStart(11)}`);
}
console.log(`downstream correct of ${ITEMS.length}: llama.cpp ${S.llamaDownstream}, ${MODES.map((m) => `${m} ${S[m].downstream}`).join(", ")}`);
for (const t of ["lookup", "math", "code"]) console.log(`  ${t}: llama.cpp ${Object.values(R.downstream).filter((x) => x.task === t && x.llama).length}, ${MODES.map((m) => `${m} ${Object.values(R.downstream).filter((x) => x.task === t && x[m]).length}`).join(", ")}`);
if (S[F] && S[D] && !ONLY) {
  const b = Object.values(R.downstream).filter((x) => x[F] && !x[D]).length, c = Object.values(R.downstream).filter((x) => !x[F] && x[D]).length;
  const g = S[F].gated, h = S[D].gated;
  const rules = {
    R1: [h.kl <= 2 * g.kl, `KL dp4a ${h.kl.toFixed(5)} <= 2 x f32 ${g.kl.toFixed(5)}`],
    R2: [g.top1 - h.top1 <= 0.5, `top1 f32 ${g.top1.toFixed(2)} - dp4a ${h.top1.toFixed(2)} = ${(g.top1 - h.top1).toFixed(2)} pt <= 0.5`],
    R3: [["mid", "2k", "8k"].every((k) => S[D][k].kl <= Math.max(3 * S[F][k].kl, S[F][k].kl + 0.01)),
      ["mid", "2k", "8k"].map((k) => `${k} ${S[D][k].kl.toFixed(5)} <= ${Math.max(3 * S[F][k].kl, S[F][k].kl + 0.01).toFixed(5)}`).join(", ")],
    R4: [S[F].downstream - S[D].downstream <= Math.max(2, 1.96 * Math.sqrt(b + c)), `f32 ${S[F].downstream} - dp4a ${S[D].downstream} <= max(2, 1.96 sqrt(${b}+${c}))`],
  };
  for (const [k, [ok, s]] of Object.entries(rules)) console.log(`${k} ${ok ? "PASS" : "FAIL"}: ${s}`);
  R.rules = Object.fromEntries(Object.entries(rules).map(([k, [ok, s]]) => [k, { ok, s }]));
  R.pass = Object.values(rules).every(([ok]) => ok);
  console.log(R.pass ? `DP4A EVAL ${MODEL}: PASS (all four rules)` : `DP4A EVAL ${MODEL}: FAIL`);
}
R.summary = S;
if (env("OUT", "")) await Deno.writeTextFile(env("OUT", ""), JSON.stringify(R) + "\n");
console.log(`GPU errors ${errors.count}`);
Deno.exit(errors.count ? 1 : 0);
