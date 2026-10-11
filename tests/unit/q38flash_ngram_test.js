// Qwen3.8-Flash-Next n-gram hash (tests/reference/q38flash_ngram.mjs) against the row indices llama.cpp itself
// computed (tests/fixtures/q38flash/ngram_rows.json: the I32 input of the gather on per_layer_token_embd, dumped
// from llama.cpp f2cc728 for fixed sequences, some decoded in several chunks). One wrong 64-bit multiply would
// change every row and degrade the model without breaking it, so this runs before any engine code uses the hash.
import { pleParams, ngramRows, rowsOf } from "../reference/q38flash_ngram.mjs";

const str = (x) => JSON.stringify(x, (_, v) => (typeof v === "bigint" ? v + "n" : v));
const eq = (a, b, m) => { const ja = str(a), jb = str(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

const FIX = JSON.parse(Deno.readTextFileSync(new URL("../fixtures/q38flash/ngram_rows.json", import.meta.url)));
const P = pleParams(FIX.meta);

Deno.test("q38flash n-gram: parameters from the GGUF keys", () => {
  eq([P.ngram, P.perGram, P.nHeads, P.eos, P.image, P.headDim, P.conv], [3, 8, 16, 248044, 248056, 160, 4]);
  // the table rows the heads cover: 16 contiguous ranges, offsets ascending, 320,001,446 rows in all
  for (let h = 1; h < P.nHeads; h++) eq(P.offsets[h], P.offsets[h - 1] + P.vocab[h - 1], `head ${h} offset`);
  eq(P.offsets[15] + P.vocab[15], 320001446n);
});

Deno.test("q38flash n-gram: rows equal llama.cpp's for every fixed sequence and chunking", () => {
  ok(FIX.cases.length >= 4, "fixture cases");
  for (const c of FIX.cases) {
    const prev = [];
    c.chunks.forEach((ids, i) => {
      eq(Array.from(ngramRows(P, ids, prev)), c.rows[i], `${c.name} chunk ${i}`);
      prev.push(...ids);
    });
  }
});

Deno.test("q38flash n-gram: the same rows whether a sequence is decoded at once or token by token", () => {
  const ids = FIX.cases.flatMap((c) => c.chunks.flat()), whole = Array.from(ngramRows(P, ids));
  const steps = [];
  ids.forEach((t, i) => steps.push(...ngramRows(P, [t], ids.slice(0, i))));
  eq(steps, whole);
});

Deno.test("q38flash n-gram: EOS and the sequence start cut the context", () => {
  const E = P.eos, row = (ctx) => Array.from(rowsOf(P, ctx));
  // the first token: both predecessors are missing, so they read as EOS
  eq(Array.from(ngramRows(P, [42])), row([42, E, E]));
  // an EOS predecessor cuts everything at or before it
  eq(Array.from(ngramRows(P, [42], [7, E])), row([42, E, E]));
  eq(Array.from(ngramRows(P, [42], [E, 7])), row([42, 7, E]));
  // the token's own EOS does not cut its context
  eq(Array.from(ngramRows(P, [E], [5, 7])), row([E, 7, 5]));
  // only the last ngram-1 predecessors matter
  eq(Array.from(ngramRows(P, [42], [1, 2, 3, 5, 7])), row([42, 7, 5]));
});

Deno.test("q38flash n-gram: rows stay inside each head's range", () => {
  const ids = [0, 1, 248319, 248044, 248045, 131071, 248056, 99999];
  const r = ngramRows(P, ids);
  for (let i = 0; i < r.length; i++) {
    const h = i % P.nHeads;
    ok(r[i] >= Number(P.offsets[h]) && r[i] < Number(P.offsets[h] + P.vocab[h]), `row ${r[i]} outside head ${h}`);
  }
});

Deno.test("q38flash n-gram: double-precision arithmetic would get the rows wrong", () => {
  // the reason for BigInt: token * multiplier needs up to 63 bits, a double keeps 53
  const m = FIX.meta["qwen4exp.ple.layer_multipliers"], v = FIX.meta["qwen4exp.ple.head_vocab_sizes"][0];
  const t = 248044, exact = Number((BigInt(t) * BigInt(m[0])) % BigInt(v)), approx = (t * m[0]) % v;
  ok(exact !== approx, `Number product happened to be exact (${exact})`);
});
