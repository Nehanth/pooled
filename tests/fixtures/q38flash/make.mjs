// Builds tests/fixtures/q38flash/headers.json: the GGUF headers (metadata and tensor tables, no weights) of
// bartowski's Qwen3.8-Flash-Next Q4_0 (3 shards) and ggml-org's MTP Q4_0 file, so the qwen4exp loader can be
// tested without the 102 GB of weights (tests/unit/q38flash_loader_test.js).
//
//   node tests/fixtures/q38flash/make.mjs [file-or-url ...]
//
// With no arguments it range-reads the four files from Hugging Face at the pinned revisions below (a few MB
// each: the first shard and the MTP file carry the 10 MB tokenizer). A local path is read the same way.
// tokenizer.* is left out; the test writes a filler key in its place, so a rebuilt header is byte-for-byte as
// long as the real one and every tensor lands at the real file's offset.
import { openSync, readSync, closeSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const BARTOWSKI = "https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/resolve/928589fdb66c6ff07f22ac561e3fbce76553548f/Qwen3.8-Flash-Next-Q4_0/";
const GGML_ORG = "https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF/resolve/052beeaca7bec4a303e59cc7bc630c4f3a1b845d/";
const SOURCES = [
  { url: BARTOWSKI + "Qwen3.8-Flash-Next-Q4_0-00001-of-00003.gguf", size: 39853467552 },
  { url: BARTOWSKI + "Qwen3.8-Flash-Next-Q4_0-00002-of-00003.gguf", size: 39647284192 },
  { url: BARTOWSKI + "Qwen3.8-Flash-Next-Q4_0-00003-of-00003.gguf", size: 21098500928 },
  { url: GGML_ORG + "mtp-Qwen3.8-Flash-Next-Q4_0.gguf", size: 2199652640 },
];

// the first n bytes of a file or URL
async function head(src, n) {
  if (/^https?:/.test(src)) {
    const r = await fetch(src, { headers: { range: `bytes=0-${n - 1}` } });
    if (!r.ok) throw new Error(`${src}: HTTP ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
  }
  const fd = openSync(src, "r"), buf = new Uint8Array(n);
  try { return buf.subarray(0, readSync(fd, buf, 0, n, 0)); } finally { closeSync(fd); }
}

// GGUF value types (gguf.h), as engine/gguf.js numbers them
const T_U8 = 0, T_I8 = 1, T_U16 = 2, T_I16 = 3, T_U32 = 4, T_I32 = 5, T_F32 = 6, T_BOOL = 7, T_STR = 8, T_ARR = 9,
  T_U64 = 10, T_I64 = 11, T_F64 = 12;

// Reads a header keeping each value's GGUF type (engine/gguf.js parseGGUFHeader drops it), so the test can write
// the same bytes back. Throws RangeError when buf ends inside the header.
function readHeader(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let off = 0;
  const u32 = () => { const v = dv.getUint32(off, true); off += 4; return v; };
  const u64 = () => { const v = dv.getBigUint64(off, true); off += 8; if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("u64 past 2^53"); return Number(v); };
  const str = () => { const n = u64(); if (off + n > buf.length) throw new RangeError("short"); const s = new TextDecoder().decode(buf.subarray(off, off + n)); off += n; return s; };
  const value = (t) => {
    switch (t) {
      case T_U8: return dv.getUint8(off++);
      case T_I8: return dv.getInt8(off++);
      case T_U16: { const v = dv.getUint16(off, true); off += 2; return v; }
      case T_I16: { const v = dv.getInt16(off, true); off += 2; return v; }
      case T_U32: return u32();
      case T_I32: { const v = dv.getInt32(off, true); off += 4; return v; }
      case T_F32: { const v = dv.getFloat32(off, true); off += 4; return v; }
      case T_BOOL: return !!dv.getUint8(off++);
      case T_STR: return str();
      case T_U64: return u64();
      case T_I64: { const v = dv.getBigInt64(off, true); off += 8; return Number(v); }
      case T_F64: { const v = dv.getFloat64(off, true); off += 8; return v; }
      default: throw new Error("bad gguf value type " + t);
    }
  };
  if (u32() !== 0x46554747) throw new Error("not a GGUF file");
  const version = u32(), nTensors = u64(), nKV = u64();
  const kv = [], omitted = [];
  for (let i = 0; i < nKV; i++) {
    const key = str(), t = u32();
    let v;
    if (t === T_ARR) { const et = u32(), n = u64(); v = [et, Array.from({ length: n }, () => value(et))]; } else v = value(t);
    if (key.startsWith("tokenizer.")) omitted.push(key); else kv.push([key, t, v]);
  }
  const tensors = [];
  for (let i = 0; i < nTensors; i++) {
    const name = str(), nd = u32(), ne = Array.from({ length: nd }, u64);
    tensors.push([name, u32(), ne, u64()]);
  }
  const alignment = kv.find(([k]) => k === "general.alignment")?.[2] || 32;
  return { version, headerBytes: off, dataStart: Math.ceil(off / alignment) * alignment, kv, omitted, tensors };
}

async function readFile(src) {
  for (let n = 1 << 20; ; n *= 4) {
    const buf = await head(src, n);
    try { return readHeader(buf); } catch (e) { if (!(e instanceof RangeError) || buf.length < n) throw e; }
  }
}

const args = process.argv.slice(2);
const srcs = args.length ? args : SOURCES.map((s) => s.url);
const files = [];
for (const [i, src] of srcs.entries()) {
  const h = await readFile(src);
  files.push({ name: (SOURCES[i]?.url || src).split("/").pop(), url: SOURCES[i]?.url, size: SOURCES[i]?.size, ...h });
  console.log(src, `${h.tensors.length} tensors, ${h.kv.length} keys, header ${h.headerBytes} B`);
}
// one tensor per line: a 1,258-row table stays diffable without the 4x of pretty-printed JSON
const lines = ["{", `"files": [`];
files.forEach((f, i) => {
  const { tensors, ...rest } = f;
  lines.push(JSON.stringify(rest).slice(0, -1) + `, "tensors": [`);
  tensors.forEach((t, j) => lines.push(JSON.stringify(t) + (j < tensors.length - 1 ? "," : "")));
  lines.push("]}" + (i < files.length - 1 ? "," : ""));
});
lines.push("]}");
const out = fileURLToPath(new URL("./headers.json", import.meta.url));
writeFileSync(out, lines.join("\n") + "\n");
console.log("wrote", out);
