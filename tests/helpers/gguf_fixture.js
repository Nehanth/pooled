// Real GGUF headers without the weights: tests/fixtures/<model>/headers.json (written by that folder's make.mjs)
// keeps each file's metadata with its GGUF value types and its tensor table. ggufFromFixture writes one back as a
// GGUF header that engine/gguf.js parseGGUFHeader reads like the real file's first bytes.

const T_U8 = 0, T_I8 = 1, T_U16 = 2, T_I16 = 3, T_U32 = 4, T_I32 = 5, T_F32 = 6, T_BOOL = 7, T_STR = 8, T_ARR = 9,
  T_U64 = 10, T_I64 = 11, T_F64 = 12;
const SIZE = { [T_U8]: 1, [T_I8]: 1, [T_BOOL]: 1, [T_U16]: 2, [T_I16]: 2, [T_U32]: 4, [T_I32]: 4, [T_F32]: 4,
  [T_U64]: 8, [T_I64]: 8, [T_F64]: 8 };
// a key in place of the left-out tokenizer: parseGGUFHeader's skipTokenizer walks past it without decoding
export const PAD_KEY = "tokenizer.fixture_pad";

// -> { files: [{ name, url, size, version, headerBytes, dataStart, kv: [[key, type, value]], omitted, tensors:
// [[name, ggmlType, ne (ggml order), offset]] }] }
export function readFixture(model) {
  return JSON.parse(Deno.readTextFileSync(new URL(`../fixtures/${model}/headers.json`, import.meta.url)));
}

// One fixture file as GGUF header bytes (an ArrayBuffer of exactly f.headerBytes). When the file's tokenizer was
// left out, a PAD_KEY string takes its room, so dataStart and every tensor's byteOffset are the real file's.
export function ggufFromFixture(f) {
  const enc = new TextEncoder();
  const strLen = (s) => 8 + enc.encode(s).length;
  const valLen = (t, v) => t === T_STR ? strLen(v) : t === T_ARR ? 12 + v[1].reduce((n, x) => n + valLen(v[0], x), 0) : SIZE[t];
  let n = 24;
  for (const [k, t, v] of f.kv) n += strLen(k) + 4 + valLen(t, v);
  for (const [name, , ne] of f.tensors) n += strLen(name) + 4 + 8 * ne.length + 4 + 8;
  const pad = f.omitted.length ? f.headerBytes - n - (strLen(PAD_KEY) + 4 + 8) : 0;
  if (pad < 0 || (!f.omitted.length && n !== f.headerBytes)) throw new Error(`${f.name}: ${n} header bytes, the file has ${f.headerBytes}`);
  const buf = new ArrayBuffer(f.headerBytes), dv = new DataView(buf), u8 = new Uint8Array(buf);
  let off = 0;
  const u32 = (v) => { dv.setUint32(off, v, true); off += 4; };
  const u64 = (v) => { dv.setBigUint64(off, BigInt(v), true); off += 8; };
  const str = (s) => { const b = enc.encode(s); u64(b.length); u8.set(b, off); off += b.length; };
  const value = (t, v) => {
    switch (t) {
      case T_U8: case T_BOOL: dv.setUint8(off++, +v); break;
      case T_I8: dv.setInt8(off++, v); break;
      case T_U16: dv.setUint16(off, v, true); off += 2; break;
      case T_I16: dv.setInt16(off, v, true); off += 2; break;
      case T_U32: u32(v); break;
      case T_I32: dv.setInt32(off, v, true); off += 4; break;
      case T_F32: dv.setFloat32(off, v, true); off += 4; break;
      case T_STR: str(v); break;
      case T_ARR: u32(v[0]); u64(v[1].length); for (const x of v[1]) value(v[0], x); break;
      case T_U64: u64(v); break;
      case T_I64: dv.setBigInt64(off, BigInt(v), true); off += 8; break;
      case T_F64: dv.setFloat64(off, v, true); off += 8; break;
      default: throw new Error("bad gguf value type " + t);
    }
  };
  u32(0x46554747); u32(f.version); u64(f.tensors.length); u64(f.kv.length + (f.omitted.length ? 1 : 0));
  for (const [k, t, v] of f.kv) { str(k); u32(t); value(t, v); }
  if (f.omitted.length) { str(PAD_KEY); u32(T_STR); u64(pad); off += pad; }   // a zero-filled string
  for (const [name, type, ne, offset] of f.tensors) { str(name); u32(ne.length); for (const d of ne) u64(d); u32(type); u64(offset); }
  return buf;
}
