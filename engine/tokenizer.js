// Byte-level BPE tokenizer built from a tokenizer.json (Qwen/Llama family).

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
  const pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
  const enc = new TextEncoder(), dec = new TextDecoder();
  // Merges the lowest-ranked adjacent pair, leftmost first, until none is left. Short words (almost
  // all of them) rescan the list per merge; that is quadratic in the word's length, so a long run with
  // no spaces (a 32k-letter "word" took 18 s) goes through bpeLong instead, which gives the same parts.
  function bpe(word) {
    if (word.length > 64) return bpeLong(word);
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
  // The same merges with a linked list and a min-heap of candidate pairs keyed by (rank, position):
  // O(n log n). A heap entry is stale once either side changed (merged away or grown).
  function bpeLong(word) {
    const sym = [...word], n = sym.length;
    const next = new Int32Array(n), prev = new Int32Array(n), ver = new Uint32Array(n);
    for (let i = 0; i < n; i++) { next[i] = i + 1; prev[i] = i - 1; }
    const heap = [];   // [rank, i, j, ver i, ver j]
    const less = (a, b) => a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]);
    const push = (e) => {
      heap.push(e);
      for (let k = heap.length - 1; k > 0;) { const p = (k - 1) >> 1; if (!less(heap[k], heap[p])) break; [heap[k], heap[p]] = [heap[p], heap[k]]; k = p; }
    };
    const pop = () => {
      const top = heap[0], last = heap.pop();
      if (heap.length) {
        heap[0] = last;
        for (let k = 0; ;) {
          const l = 2 * k + 1, r = l + 1; let m = k;
          if (l < heap.length && less(heap[l], heap[m])) m = l;
          if (r < heap.length && less(heap[r], heap[m])) m = r;
          if (m === k) break;
          [heap[k], heap[m]] = [heap[m], heap[k]]; k = m;
        }
      }
      return top;
    };
    const cand = (i, j) => { if (i < 0 || j >= n) return; const r = ranks.get(sym[i] + " " + sym[j]); if (r !== undefined) push([r, i, j, ver[i], ver[j]]); };
    for (let i = 0; i + 1 < n; i++) cand(i, i + 1);
    while (heap.length) {
      const [, i, j, vi, vj] = pop();
      if (sym[i] === null || sym[j] === null || next[i] !== j || ver[i] !== vi || ver[j] !== vj) continue;
      sym[i] += sym[j]; sym[j] = null; ver[i]++; ver[j]++;
      next[i] = next[j]; if (next[j] < n) prev[next[j]] = i;
      cand(prev[i], i); cand(i, next[i]);
    }
    const parts = [];
    for (let i = 0; i < n; i = next[i]) parts.push(sym[i]);
    return parts;
  }
  return {
    vocab,
    encode(text) {
      const ids = [];
      for (const piece of text.match(pat) || []) {
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
