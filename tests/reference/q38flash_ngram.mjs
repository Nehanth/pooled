// Qwen3.8-Flash-Next (qwen4exp) n-gram hash: which rows of per_layer_token_embd a token gathers.
// Port of llama.cpp src/models/qwen4exp.cpp llm_graph_input_qwen4exp_ple::set_input (master f2cc728, lines 1224-1295).
//
// For the token at position p: ctx[0] = token p, ctx[s] = token p - s (s = 1 .. ngram - 1). A predecessor
// before the sequence start reads as EOS, and so does everything at or before an EOS predecessor (the token's
// own EOS does not cut its context). Then for n = 2 .. ngram:
//   mixed = ctx[0]*m[0] ^ ctx[1]*m[1] [^ ctx[2]*m[2]]      (uint64)
//   row[(n-2)*perGram + g] = mixed % vocab[h] + offset[h]   for g in 0 .. perGram-1, h = (n-2)*perGram + g
// The multipliers are 45-bit and token ids are below 2^18, so the products need 63 bits: BigInt, never Number.
// No file access here, so the Deno unit test can import it. engine/ngram.js (draft PR #339) has its own copy of
// this hash; nothing ties the two together except tests/fixtures/q38flash/ngram_rows.json, the rows llama.cpp
// itself computed, which both should be tested against.

// The hash constants from GGUF metadata (keys "qwen4exp.ple.*"). parseGGUFHeader returns uint64 as Number;
// these values are below 2^53, so they are exact, and they become BigInt here.
export function pleParams(meta, arch = "qwen4exp") {
  const k = (s) => meta[arch + ".ple." + s];
  const ngram = k("ngram_size"), perGram = k("heads_per_ngram");
  const nHeads = (ngram - 1) * perGram;
  const big = (a, n) => {
    if (!Array.isArray(a) || a.length < n) throw new Error(`${arch}.ple: array too short (${a && a.length} < ${n})`);
    return a.slice(0, n).map((v) => BigInt(v));
  };
  return {
    ngram, perGram, nHeads,
    eos: k("eos_token_id"),
    image: k("image_token_id") || k("eos_token_id"),
    mult: big(k("layer_multipliers"), ngram),
    offsets: big(k("head_offsets"), nHeads),
    vocab: big(k("head_vocab_sizes"), nHeads),
    headDim: meta[arch + ".embedding_length_per_layer_input"],
    conv: k("conv_kernel"),
    layers: k("layers"),
  };
}

// The hash context of one token: ctx[0..ngram-1] with the EOS cut applied. hist(s) returns the token s
// positions back, or null when there is none (before the sequence start).
function context(P, tok, hist) {
  const ctx = [tok];
  let cut = false;
  for (let s = 1; s < P.ngram; s++) {
    const t = cut ? null : hist(s);
    cut = cut || t === null || t < 0 || t === P.eos;
    ctx.push(cut ? P.eos : t);
  }
  return ctx;
}

// Rows for one token given its context: Int32Array(nHeads), head order as in the gathered embedding.
export function rowsOf(P, ctx, out = new Int32Array(P.nHeads), at = 0) {
  for (let n = 2; n <= P.ngram; n++) {
    let mixed = BigInt(ctx[0]) * P.mult[0];
    for (let j = 1; j < n; j++) mixed ^= BigInt(ctx[j]) * P.mult[j];
    mixed = BigInt.asUintN(64, mixed);
    const base = (n - 2) * P.perGram;
    for (let g = 0; g < P.perGram; g++) {
      const h = base + g;
      out[at + h] = Number(mixed % P.vocab[h] + P.offsets[h]);
    }
  }
  return out;
}

// Rows for a run of tokens that continues a sequence. ids: this chunk's tokens; prev: the tokens already in
// the sequence before it, oldest first (only the last ngram-1 matter). Returns Int32Array(ids.length * nHeads),
// token-major, the layout of llama.cpp's I32 "rows" input.
export function ngramRows(P, ids, prev = []) {
  const all = [...prev, ...ids], base = prev.length;
  const out = new Int32Array(ids.length * P.nHeads);
  for (let i = 0; i < ids.length; i++) {
    const p = base + i;
    const ctx = context(P, ids[i], (s) => (p - s >= 0 ? all[p - s] : null));
    rowsOf(P, ctx, out, i * P.nHeads);
  }
  return out;
}
