// Copy Qwen3.8-Flash-Next's n-gram table out of its GGUF into one file per hash head (engine/ngram.js layout
// "split"): ngram-00.bin ... ngram-15.bin, 1.8 GB each for the Q4_0 file, plus ngram.json (the config the readers
// need, so they do not have to parse the 11 MB GGUF header again). This is the layout the OPFS reader uses
// (engine/ngram_worker.js); a room node can read the GGUF in place instead (layout "tensor").
//   node scripts/ngram-split.mjs <shard with per_layer_token_embd.weight> <out dir> [heads, e.g. 0,15]
// llama.cpp split files keep the qwen4exp.ple.* keys only in shard 1. When the table is in another shard
// (...-0000N-of-0000M.gguf), the keys are read from shard 1 next to it (bartowski's Q4_0 has the table in shard 1).
// Writes 28.8 GB for the whole Q4_0 table. Nothing is deleted.
import fs from "node:fs/promises";
import path from "node:path";
import { parseGGUFHeader } from "../engine/gguf.js";
import { ngramConfig, splitHeads, fileReadAt, ngramFileName, NGRAM_TENSOR } from "../engine/ngram.js";

const [src, outDir, headArg] = process.argv.slice(2);
if (!src || !outDir) { console.error("usage: node scripts/ngram-split.mjs <gguf shard> <out dir> [heads]"); process.exit(2); }
// the header: grow the read until it parses
async function header(fh) {
  for (let n = 16 << 20; ; n *= 2) {
    const b = Buffer.alloc(n);
    await fh.read(b, 0, n, 0);
    try { return parseGGUFHeader(b.buffer.slice(b.byteOffset, b.byteOffset + n), { skipTokenizer: true }); } catch (e) { if (n > 256 << 20) throw e; }
  }
}
const fh = await fs.open(src, "r");
let G = await header(fh);
if (!G.tensors?.[NGRAM_TENSOR]) { console.error(`${src} has no ${NGRAM_TENSOR}: pass the shard that holds it`); process.exit(1); }
if (G.meta["qwen4exp.ple.ngram_size"] == null) {
  // a later shard of a split: its tensor offsets are its own, the metadata is shard 1's
  const first = src.replace(/-(\d{5})-of-(\d{5})\.gguf$/, "-00001-of-$2.gguf");
  if (first === src) { console.error(`${src} has no qwen4exp.ple.* keys and is not a split shard (-0000N-of-0000M.gguf)`); process.exit(1); }
  const f1 = await fs.open(first, "r");
  G = { ...G, meta: { ...(await header(f1)).meta, ...G.meta } };
  await f1.close();
  console.log(`metadata from ${path.basename(first)}`);
}
// the table is read as file 0 (fileReadAt([fh])), whatever shard number the header gives it
const cfg = { ...ngramConfig(G), shard: 0 };
const heads = headArg ? headArg.split(",").map(Number) : [...Array(cfg.nHeads).keys()];
await fs.mkdir(outDir, { recursive: true });
const outs = {};
for (const h of heads) outs[h] = await fs.open(path.join(outDir, ngramFileName(h)), "w");
const t0 = Date.now();
let last = 0;
await splitHeads(cfg, fileReadAt([fh]), async (h, b, off) => { await outs[h].write(b, 0, b.length, off); }, {
  heads, chunkRows: 1 << 18,
  onProgress: (done, total) => { if (done - last > 1e9 || done === total) { last = done; console.log(`${(done / 1e9).toFixed(1)} / ${(total / 1e9).toFixed(1)} GB`); } },
});
for (const h of heads) await outs[h].close();
const { offsets, vocab, mult, ngramSize, perNgram, width, eos, image, ggmlType, nRows } = cfg;
await fs.writeFile(path.join(outDir, "ngram.json"), JSON.stringify({ offsets, vocab, mult, ngramSize, perNgram, width, eos, image, ggmlType, nRows }, null, 1));
await fh.close();
console.log(`${heads.length} head files in ${outDir}, ${((Date.now() - t0) / 1000).toFixed(1)} s`);
