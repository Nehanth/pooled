// Spot check of the n-gram row source (engine/ngram.js) on the real Qwen3.8-Flash-Next table: shard 1 of bartowski's
// Q4_0 (per_layer_token_embd.weight, 320,001,536 x 160, 28.8 GB). No GPU.
//   deno run --allow-read --allow-env [--allow-write --allow-net] tests/ngram_real.js
//   Q38FLASH=<shard 1 path>      default ~/llm/models/q38flash/Qwen3.8-Flash-Next-Q4_0-00001-of-00003.gguf
//   SPLIT_DIR=<dir>              also split head 15 there (1.8 GB, needs --allow-write) and read it back as a split file
//   URL=<shard 1 URL>            also read rows over HTTP range (needs --allow-net) and compare them to the file's
//
// Checks: the config (16 heads, 160 values, 90-byte rows, heads inside the tensor, 16 x 160 = the hidden size);
// rows hashed from a real prompt, every head's first and last row and the padding rows, plus random rows, equal
// dequantF32 of the same bytes read straight from the file; values are finite and look like an embedding; per-token
// read time through Deno file reads (page cache state unknown: whatever the box has).
import { parseGGUFHeader, dequantF32, tokenizerFromGGUF } from "../engine/gguf.js";
import { makeTokenizer } from "../engine/engine.js";
import { ngramConfig, NgramState, NgramTable, fileReadAt, httpReadAt, splitHeads, ngramFileName } from "../engine/ngram.js";

const HOME = Deno.env.get("HOME");
const FILE = Deno.env.get("Q38FLASH") || `${HOME}/llm/models/q38flash/Qwen3.8-Flash-Next-Q4_0-00001-of-00003.gguf`;
let fails = 0;
const check = (name, ok, detail = "") => { if (!ok) fails++; console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? "  " + detail : ""}`); };
const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));

const f = await Deno.open(FILE, { read: true });
async function readRaw(off, len) { const b = new Uint8Array(len); await f.seek(off, Deno.SeekMode.Start); let got = 0; while (got < len) { const n = await f.read(b.subarray(got)); if (!n) break; got += n; } return b; }
// the header: grow the read until it parses (the tokenizer arrays make it ~11 MB)
let G = null;
for (let n = 16 << 20; !G; n *= 2) { try { G = parseGGUFHeader((await readRaw(0, n)).buffer); } catch (e) { if (n > 256 << 20) throw e; } }
const cfg = ngramConfig(G);
const T = G.tensors["per_layer_token_embd.weight"];
console.log(`table: ${T.shape.join(" x ")} type ${T.ggmlType}, ${(T.byteLength / 1e9).toFixed(2)} GB at byte ${T.byteOffset}; heads end at row ${cfg.offsets[15] + cfg.vocab[15]}`);
check("config: 16 heads x 160 values = 2560, 90-byte Q4_0 rows", cfg.nHeads === 16 && cfg.width === 160 && cfg.dim === 2560 && cfg.rowBytes === 90 && cfg.ggmlType === 2);
check("config: heads contiguous, inside the tensor", cfg.offsets.every((o, h) => h === 0 ? o === 0 : o === cfg.offsets[h - 1] + cfg.vocab[h - 1]) && cfg.offsets[15] + cfg.vocab[15] <= cfg.nRows
  && T.byteLength === cfg.nRows * cfg.rowBytes);

// rows to check: a real prompt's (with an EOS in it), head edges, the padding past the last head, random rows
const tok = makeTokenizer(tokenizerFromGGUF(G.meta));
const prompt = [...tok.encode("The capital of France is Paris. The capital of Germany is Berlin."), cfg.eos, ...tok.encode("Hello again, world!")];
const st = new NgramState(cfg);
const hashed = st.push(prompt);
const edges = [];
for (let h = 0; h < 16; h++) edges.push(cfg.offsets[h], cfg.offsets[h] + cfg.vocab[h] - 1);
for (let r = cfg.offsets[15] + cfg.vocab[15]; r < cfg.nRows; r += 13) edges.push(r);
edges.push(cfg.nRows - 1);
let seed = 7;
const rnd = () => { let t = (seed = (seed + 0x6d2b79f5) | 0); t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32; };
const random = Array.from({ length: 2000 }, () => Math.floor(rnd() * cfg.nRows));
const ids = Uint32Array.from([...hashed, ...edges, ...random]);

// the reference: each row's bytes read on their own, through dequantF32
const want = new Float32Array(ids.length * 160);
for (let i = 0; i < ids.length; i++) {
  const b = await readRaw(cfg.byteOffset + ids[i] * 90, 90);
  want.set(dequantF32({ ggmlType: 2, nElems: 160, shape: [1, 160] }, b), i * 160);
}
const table = new NgramTable(cfg, { readAt: fileReadAt([f]), layout: "tensor" });
const got = await table.rows(ids);
check(`${prompt.length} prompt tokens' rows + ${edges.length} edge / padding rows + ${random.length} random rows == dequantF32`, same(got, want), `${ids.length} rows`);

// do they look like an embedding: finite, not all zero (except padding), a sane scale
let nonFinite = 0, zeroRows = 0, sumAbs = 0, maxAbs = 0;
const nReal = hashed.length + 32;
for (let i = 0; i < nReal; i++) {
  let z = true;
  for (let j = 0; j < 160; j++) { const v = got[i * 160 + j]; if (!Number.isFinite(v)) nonFinite++; if (v) z = false; sumAbs += Math.abs(v); maxAbs = Math.max(maxAbs, Math.abs(v)); }
  if (z) zeroRows++;
}
const padZero = edges.slice(32).every((r) => { const i = ids.indexOf(r); return got.subarray(i * 160, (i + 1) * 160).every((v) => v === 0); });
console.log(`     values: mean |x| ${(sumAbs / (nReal * 160)).toExponential(3)}, max |x| ${maxAbs.toExponential(3)}, ${zeroRows} all-zero rows of ${nReal}; padding rows all zero: ${padZero}`);
check("rows finite, none all-zero", nonFinite === 0 && zeroRows === 0);
const emb = got.subarray(0, 2560);
console.log(`     first token (${prompt[0]} "${tok.decode([prompt[0]])}") rows: ${[...hashed.subarray(0, 16)].join(" ")}`);
console.log(`     its embedding [0..4] ${[...emb.subarray(0, 4)].map((v) => v.toExponential(3)).join(" ")}`);

// per-token read time: 16 rows of a fresh token each, cache off (a row cache would hide the reads)
const cold = new NgramTable(cfg, { readAt: fileReadAt([f]), layout: "tensor", cacheRows: 0 });
const s2 = new NgramState(cfg), N = 500, t0 = performance.now();
for (let i = 0; i < N; i++) await cold.rows(s2.push([Math.floor(rnd() * 248000)]));
console.log(`     ${((performance.now() - t0) / N).toFixed(3)} ms per token (16 rows, Deno seek+read, one file; page cache as found)`);

// a split head file: head 15 (the last, next to the padding)
const SPLIT = Deno.env.get("SPLIT_DIR");
if (SPLIT) {
  const path = `${SPLIT}/${ngramFileName(15)}`;
  const out = await Deno.open(path, { write: true, create: true, truncate: true });
  const t1 = performance.now();
  await splitHeads(cfg, fileReadAt([f]), async (h, b, off) => { await out.seek(off, Deno.SeekMode.Start); let w = 0; while (w < b.length) w += await out.write(b.subarray(w)); },
    { heads: [15], chunkRows: 1 << 18 });
  out.close();
  const hf = await Deno.open(path, { read: true });
  const files = []; files[15] = hf;
  const split = new NgramTable(cfg, { readAt: fileReadAt(files), layout: "split" });
  const h15 = Uint32Array.from([...ids].filter((r) => r >= cfg.offsets[15] && r < cfg.offsets[15] + cfg.vocab[15]));
  const want15 = new Float32Array(h15.length * 160);
  h15.forEach((r) => { const i = ids.indexOf(r); want15.set(got.subarray(i * 160, (i + 1) * 160), h15.indexOf(r) * 160); });
  check(`split head 15 (${((await Deno.stat(path)).size / 1e9).toFixed(2)} GB in ${((performance.now() - t1) / 1000).toFixed(1)} s): ${h15.length} rows == the tensor's`, same(await split.rows(h15), want15));
  hf.close();
  await Deno.remove(path);
}

// HTTP range against the published file: the downloaded bytes are the published bytes, and the HTTP reader works
const URL_ = Deno.env.get("URL");
if (URL_) {
  const pick = Uint32Array.from([...hashed.subarray(0, 32), ...edges.slice(0, 32)]);
  const net = new NgramTable(cfg, { readAt: httpReadAt([URL_]), layout: "tensor", inflight: 8 });
  const t2 = performance.now();
  const viaHttp = await net.rows(pick);
  const viaFile = await new NgramTable(cfg, { readAt: fileReadAt([f]), layout: "tensor" }).rows(pick);
  check(`${pick.length} rows over HTTP range == the file's`, same(viaHttp, viaFile), `${((performance.now() - t2) / 1000).toFixed(1)} s`);
}
f.close();
console.log(fails ? `NGRAM REAL: ${fails} FAILED` : "NGRAM REAL PASS");
Deno.exit(fails ? 1 : 0);
