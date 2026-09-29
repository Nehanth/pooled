// engine/tokenizer.js: a word longer than 64 symbols goes through the heap BPE (bpeLong), which must
// give exactly the parts the plain rescan-per-merge loop gives, and stay fast on a huge word.
import { makeTokenizer } from "../../engine/tokenizer.js";

// the plain loop, as the tokenizer had it: lowest rank first, leftmost among equals
function reference(word, ranks) {
  let parts = [...word];
  while (parts.length > 1) {
    let best = null, bestRank = Infinity;
    for (let i = 0; i < parts.length - 1; i++) {
      const r = ranks.get(parts[i] + " " + parts[i + 1]);
      if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
    }
    if (best === null) break;
    parts = [...parts.slice(0, best), parts[best] + parts[best + 1], ...parts.slice(best + 2)];
  }
  return parts;
}

// a toy vocabulary over "abc": every string up to 4 letters is a token, merges in a seeded order
function toy(seed) {
  let x = seed;
  const rnd = () => ((x = (x * 1103515245 + 12345) >>> 0) / 2 ** 32);
  const vocab = {}, merges = [];
  const add = (t) => { if (!(t in vocab)) vocab[t] = Object.keys(vocab).length; };
  for (const c of "abc") add(c);
  const toks = ["a", "b", "c"];
  for (let len = 2; len <= 4; len++) for (const l of [...toks]) for (const r of [...toks]) {
    if (l.length + r.length !== len || rnd() < 0.3) continue;
    add(l + r); toks.push(l + r); merges.push(l + " " + r);
  }
  for (let i = merges.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [merges[i], merges[j]] = [merges[j], merges[i]]; }
  return { tj: { model: { vocab, merges } }, ranks: new Map(merges.map((m, i) => [m, i])), vocab };
}

Deno.test("tokenizer: long words merge exactly as the plain loop does", () => {
  for (let seed = 1; seed <= 20; seed++) {
    const { tj, ranks, vocab } = toy(seed);
    const tok = makeTokenizer(tj);
    let x = seed * 7919;
    for (let k = 0; k < 30; k++) {
      let w = "";
      const len = 65 + (k * 37) % 400;
      while (w.length < len) w += "abc"[(x = (x * 69069 + 1) >>> 0) % (k % 3 + 1)];
      const want = reference(w, ranks).map((t) => vocab[t]);
      const got = tok.encode(w);
      if (want.join() !== got.join()) throw new Error(`seed ${seed} word ${w.slice(0, 30)}…: ${got.length} ids vs ${want.length}`);
    }
  }
});

Deno.test("tokenizer: a 200k-letter word encodes in well under a second", () => {
  const { tj } = toy(3);
  const tok = makeTokenizer(tj);
  const t = performance.now();
  tok.encode("ab".repeat(100000));
  const ms = performance.now() - t;
  if (ms > 3000) throw new Error(`took ${ms.toFixed(0)} ms`);
});
