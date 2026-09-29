// room/wire.js: packWire/unpackWire, asU16 and asF32 over every payload shape PeerJS can hand
// back (typed arrays, views at odd byte offsets, bare ArrayBuffers, base64 strings), and the
// f16 range guard at its exact boundary.
import {
  packWire, unpackWire, asU16, asF32, f32ToB64, b64ToF32, packF16, unpackF16, badF32,
  WireRangeError, wireStats, F16_MAX, WIRE_F16,
} from "../../room/wire.js";

function ok(c, m) { if (!c) throw new Error(m || "assertion failed"); }
function eq(a, b, m) { const x = JSON.stringify(a), y = JSON.stringify(b); if (x !== y) throw new Error(`${m ? m + ": " : ""}${x} !== ${y}`); }
function throws(f, cls, m) { let e = null; try { f(); } catch (x) { e = x; } if (!e || (cls && !(e instanceof cls))) throw new Error("expected throw: " + m + " got " + e); }
const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));

// f16-exact values so pack/unpack is lossless
const exact = new Float32Array([0, -0, 1, -1, 0.5, 1024, -65504, 65504, 6.103515625e-5, 5.960464477539063e-8, 3.140625]);

Deno.test("wire: packWire then unpackWire is exact for f16-representable values", () => {
  ok(WIRE_F16, "the f16 wire is on");
  const w = packWire(exact);
  eq(w.enc, "f16"); ok(w.data instanceof Uint16Array && w.data.length === exact.length);
  ok(same(unpackWire(w), exact), "round trip");
  eq(wireStats.lastMax, 65504);
});

Deno.test("wire: packWire rounds to nearest f16 within half an ulp", () => {
  const f = new Float32Array(2000); for (let i = 0; i < f.length; i++) f[i] = Math.sin(i) * 300;
  const back = unpackWire(packWire(f));
  for (let i = 0; i < f.length; i++) { const ulp = 2 ** (Math.floor(Math.log2(Math.abs(f[i]) || 1)) - 10); ok(Math.abs(back[i] - f[i]) <= ulp / 2 + 1e-12, "at " + i); }
});

Deno.test("wire: the f16 range guard sits exactly at 65504", () => {
  for (const v of [65504, -65504]) packF16(new Float32Array([v]));
  for (const v of [65504.5, -65505, 1e6, Infinity, -Infinity]) throws(() => packF16(new Float32Array([1, v])), WireRangeError, String(v));
  eq(F16_MAX, 65504);
  packF16(new Float32Array(0)); eq(wireStats.lastMax, 0, "an empty frame records 0");
});

Deno.test("wire: NaN is not a range error: packF16 passes it through as an f16 NaN (callers check badF32 first)", () => {
  const f = new Float32Array([1, NaN, 2]);
  ok(badF32(f), "badF32 sees it");
  const back = unpackWire(packWire(f));
  ok(back[0] === 1 && Number.isNaN(back[1]) && back[2] === 2);
});

Deno.test("wire: asU16 accepts every binary shape and copies unaligned views", () => {
  const src = new Uint16Array([1, 2, 0xffff, 0x3c00, 7]);
  const u8 = new Uint8Array(src.buffer);
  const odd = new Uint8Array(src.byteLength + 1); odd.set(u8, 1);
  const cases = [
    ["Uint16Array (same object)", src],
    ["ArrayBuffer", src.buffer],
    ["Uint8Array", u8],
    ["view at an odd byteOffset", odd.subarray(1)],
    ["DataView", new DataView(src.buffer)],
    ["Uint16 subarray", new Uint16Array([9, ...src]).subarray(1)],
  ];
  for (const [name, x] of cases) eq(Array.from(asU16(x)), Array.from(src), name);
  ok(asU16(src) === src, "a Uint16Array is returned as is");
  const view = odd.subarray(1), r = asU16(view); r[0] = 42; eq(view[0], 1, "an unaligned view is copied, not aliased");
  for (const bad of [[1, 2], "AAA=", null, 5, { data: 1 }]) throws(() => asU16(bad), Error, JSON.stringify(bad));
  // an odd byte count cannot be f16 data: it throws (RangeError from Uint16Array) rather than truncating
  throws(() => asU16(new Uint8Array(3)), RangeError, "odd length");
});

Deno.test("wire: unpackWire decodes f16 from any binary shape", () => {
  const h = packF16(exact), bytes = new Uint8Array(h.buffer);
  const odd = new Uint8Array(bytes.length + 3); odd.set(bytes, 3);
  for (const data of [h, h.buffer, bytes, odd.subarray(3)]) ok(same(unpackWire({ enc: "f16", data }), exact), data.constructor.name);
  eq(unpackF16(new Uint16Array(0)).length, 0);
});

Deno.test("wire: asF32 and unpackWire without enc read f32 as typed array, buffer, view or base64", () => {
  const f = new Float32Array([1.5, -2.25, 1e-30, 3e38, -0]);
  const u8 = new Uint8Array(f.buffer);
  const odd = new Uint8Array(u8.length + 1); odd.set(u8, 1);
  const cases = [["Float32Array", f], ["ArrayBuffer", f.buffer], ["Uint8Array", u8], ["odd view", odd.subarray(1)], ["base64", f32ToB64(f)]];
  for (const [name, x] of cases) { ok(same(asF32(x), f), name); ok(same(unpackWire({ data: x }), f), "unpackWire " + name); }
  ok(asF32(f) === f);
  for (const bad of [[1, 2], null, 3, {}]) throws(() => asF32(bad), Error, JSON.stringify(bad));
});

Deno.test("wire: base64 round trip across the 32 KB chunk boundary and for empty/odd inputs", () => {
  for (const n of [0, 1, 8191, 8192, 8193, 20000]) {
    const f = new Float32Array(n); for (let i = 0; i < n; i++) f[i] = i * 0.37 - 100;
    const s = f32ToB64(f);
    eq(s.length, 4 * Math.ceil(n * 4 / 3), "b64 length " + n);
    ok(same(b64ToF32(s), f), "round trip " + n);
  }
  // a view: only its own bytes are encoded
  const big = new Float32Array([1, 2, 3, 4]); ok(same(b64ToF32(f32ToB64(big.subarray(1, 3))), new Float32Array([2, 3])));
  throws(() => b64ToF32(btoa("abc")), RangeError, "3 bytes is not whole floats");
  throws(() => b64ToF32("!!!"), Error, "not base64");
});
