// Byte-level BPE tokenizer built from a tokenizer.json (Qwen/Llama family).

// Pre-tokenizer splits (text -> pieces that BPE runs on), written as llama.cpp's llama-vocab.cpp has
// them: tokenizer.json's (?i:'s|...) is spelled out, since JS regexes only get (?i:) in ES2025.
// Qwen's split differs from GPT-2's in ways that change ids: digits go one at a time, a leading
// punctuation char joins the word after it ("(foo", ".bar"), and newlines stay with the run before them.
const CONTRACTIONS = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])";
export const PRE_SPLITS = {
  gpt2: String.raw`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`,
  qwen2: CONTRACTIONS + String.raw`|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
  // Qwen 3.5+: combining marks (\p{M}) count as letters, so "é" spelled e + U+0301 stays one word
  qwen35: CONTRACTIONS + String.raw`|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
};
// GGUF tokenizer.ggml.pre names (llama-vocab.cpp) that use a split other than plain GPT-2
const GGUF_PRE = {
  "qwen2": "qwen2", "deepseek-r1-qwen": "qwen2", "stablelm2": "qwen2", "megrez": "qwen2",
  "qwen35": "qwen35",
  "smollm": "digits", "starcoder": "digits", "refact": "digits", "command-r": "digits",
};

// Picks the split for a tokenizer.json (its pre_tokenizer) or a GGUF (tj.pre, from tokenizerFromGGUF).
// Unknown or missing config falls back to GPT-2, which is what this file always used.
export function preSplitter(tj) {
  let src = PRE_SPLITS.gpt2, digits = false;
  if (tj.pre !== undefined) {
    const kind = GGUF_PRE[tj.pre];
    if (kind === "digits") digits = true;
    else if (kind) src = PRE_SPLITS[kind];
  } else if (tj.pre_tokenizer) {
    const steps = tj.pre_tokenizer.type === "Sequence" ? tj.pre_tokenizer.pretokenizers || [] : [tj.pre_tokenizer];
    for (const s of steps) {
      if (s.type === "Digits" && s.individual_digits) digits = true;
      const re = s.type === "Split" && s.pattern && s.pattern.Regex;
      if (re) src = re.replace("(?i:'s|'t|'re|'ve|'m|'ll|'d)", CONTRACTIONS);
    }
  }
  let re;
  try { re = new RegExp(src, "gu"); } catch { re = new RegExp(PRE_SPLITS.gpt2, "gu"); }
  // Digits (SmolLM) runs first: every digit is its own piece, and the regex splits the text between
  return digits
    ? (text) => text.split(/(\p{N})/u).flatMap((c) => (c ? c.match(re) || [] : []))
    : (text) => text.match(re) || [];
}

export function makeTokenizer(tj) {
  // special tokens (<|im_start|>, <think>, ...) live in added_tokens for
  // Qwen-family tokenizer.json files, not in model.vocab
  const vocab = { ...tj.model.vocab };
  for (const t of tj.added_tokens || []) if (t && t.content !== undefined) vocab[t.content] = t.id;
  const idToTok = {};
  for (const [t, i] of Object.entries(vocab)) idToTok[i] = t;
  const ranks = new Map();
  tj.model.merges.forEach((m, i) => ranks.set(Array.isArray(m) ? m.join(" ") : m, i));
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const byteToChar = {}, charToByte = {};
  bs.forEach((b, i) => { byteToChar[b] = String.fromCharCode(cs[i]); charToByte[String.fromCharCode(cs[i])] = b; });
  const split = preSplitter(tj);
  const enc = new TextEncoder(), dec = new TextDecoder();
  function bpe(word) {
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
  return {
    vocab,
    encode(text) {
      const ids = [];
      for (const piece of split(text)) {
        let word = "";
        for (const b of enc.encode(piece)) word += byteToChar[b];
        for (const tok of bpe(word)) ids.push(vocab[tok]);
      }
      return ids;
    },
    decode(ids) {
      const bytes = [];
      for (const id of ids) {
        const tok = idToTok[id];
        if (tok === undefined) continue;
        for (const ch of tok) { const b = charToByte[ch]; if (b !== undefined) bytes.push(b); }
      }
      return dec.decode(new Uint8Array(bytes));
    },
  };
}
