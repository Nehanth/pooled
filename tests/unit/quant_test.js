// engine/quant.js: Q4_0 quantization (f16 block scales, 32 per block, two nibbles per byte with
// j and j+16 sharing a byte) and the Q4/Q8 dequant the GPU self-test compares against.
import { quantizeQ4, dequantQ4, dequantQ8 } from "../../engine/quant.js";
import { quantizeQ8, f16ToF32, f32ToF16 } from "../../engine/gguf.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

let seed = 7;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
const randArr = (n, s = 1) => Float32Array.from({ length: n }, () => rnd() * s);
const scaleOf = (q, b) => f16ToF32((q.scales[b >> 1] >>> ((b & 1) * 16)) & 0xFFFF);

// Q4_0: d = f16(maxv / -8), q = clamp(round(x/d) + 8, 0, 15). Every value is within d/2 except the
// side opposite the block's max, where +8 clamps to 15 (an error up to |d|); plus f16 rounding of d.
function checkQ4(x, why) {
  const n = x.length, nb = Math.ceil(n / 32);
  const q = quantizeQ4(x);
  eq(q.qs.length, nb * 16, `${why}: qs bytes`);
  eq(q.scales.length, Math.ceil(nb / 2), `${why}: packed scale words`);
  const y = dequantQ4(q, n);
  eq(y.length, n, `${why}: output length`);
  for (let b = 0; b < nb; b++) {
    let amax = 0;
    for (let i = b * 32; i < Math.min(n, b * 32 + 32); i++) amax = Math.max(amax, Math.abs(x[i]));
    const d = Math.abs(scaleOf(q, b));
    for (let i = b * 32; i < Math.min(n, b * 32 + 32); i++) {
      ok(Number.isFinite(y[i]), `${why}: y[${i}] = ${y[i]}`);
      const err = Math.abs(y[i] - x[i]);
      // |d| (clamp on the side opposite the max) plus 8 x the f16 rounding of d (matters for subnormal scales)
      ok(err <= d * 1.0001 + 8 * Math.abs(d - amax / 8) + 1e-30, `${why}: block ${b} i=${i} x=${x[i]} y=${y[i]} err=${err} d=${d}`);
    }
    // d is amax/8 rounded to f16: half an ulp relative, or 2^-25 absolute in the subnormal range
    if (amax > 0) ok(Math.abs(d - amax / 8) <= Math.max(amax / 8 * 2 ** -11, 2 ** -25), `${why}: block ${b} scale ${d} vs amax/8 ${amax / 8}`);
  }
  return { q, y };
}

Deno.test("Q4 round trip within the error bound: sizes incl. tail blocks", () => {
  for (const n of [1, 5, 16, 17, 31, 32, 33, 40, 63, 64, 65, 96, 100, 257, 1000, 4096]) checkQ4(randArr(n, 2), `n=${n}`);
});

Deno.test("Q4 round trip: value ranges", () => {
  const cases = [
    ["tiny", randArr(64, 1e-3)],
    ["large", randArr(64, 1000)],
    ["one outlier per block", Float32Array.from({ length: 64 }, (_, i) => (i % 32 === 5 ? 50 : rnd()))],
    ["all positive", Float32Array.from({ length: 64 }, () => Math.abs(rnd()) + 0.1)],
    ["all negative", Float32Array.from({ length: 64 }, () => -Math.abs(rnd()) - 0.1)],
    ["constant", new Float32Array(64).fill(3)],
    ["alternating sign", Float32Array.from({ length: 64 }, (_, i) => (i & 1 ? -1 : 1))],
    ["f16 subnormal scale", randArr(64, 1e-5)],
  ];
  for (const [why, x] of cases) checkQ4(x, why);
});

Deno.test("Q4: the block max is reproduced exactly (it maps to nibble 0)", () => {
  // the scale is chosen so the signed max lands on -8 * d exactly (up to f16 rounding of d)
  for (const m of [1, -1, 8, -0.125, 7.5]) {
    const x = new Float32Array(32); x[3] = m; x[20] = m / 4;
    const { q, y } = checkQ4(x, "max " + m);
    ok(Math.abs(y[3] - m) <= Math.abs(m) * 2 ** -10, `max ${m} -> ${y[3]}`);
    eq(q.qs[3] & 0xF, 0, "the max's nibble");
  }
});

Deno.test("Q4: an all-zero block gets scale 1 and dequantizes to zeros", () => {
  for (const n of [32, 40, 64]) {
    const x = new Float32Array(n);
    const { q, y } = checkQ4(x, "zeros " + n);
    for (let b = 0; b < Math.ceil(n / 32); b++) eq(scaleOf(q, b), 1, `zero block ${b} scale`);
    ok(y.every((v) => v === 0), "all zeros");
    ok(q.qs.every((v) => v === 0x88), "every nibble is the zero point 8");
  }
});

Deno.test("Q4: a zero block next to a non-zero one keeps its own scale", () => {
  const x = new Float32Array(96); for (let i = 32; i < 64; i++) x[i] = (i - 48) / 4;
  const { q, y } = checkQ4(x, "mixed");
  eq(scaleOf(q, 0), 1); eq(scaleOf(q, 2), 1);
  ok(y.slice(0, 32).every((v) => v === 0) && y.slice(64).every((v) => v === 0));
});

Deno.test("Q4: nibble layout, element j in the low nibble and j+16 in the high one", () => {
  // the block max is -8, so d = -8 / -8 = 1 and q = round(x) + 8
  const x = new Float32Array(32);
  x[0] = -8; x[1] = 3; x[16] = -2; x[17] = 7;
  const q = quantizeQ4(x);
  eq(scaleOf(q, 0), 1);
  eq(q.qs[0], 0 | (6 << 4), "x[0]=-8 -> 0, x[16]=-2 -> 6");
  eq(q.qs[1], 11 | (15 << 4), "x[1]=3 -> 11, x[17]=7 -> 15");
  eq(q.qs[2], 0x88, "zeros sit at the zero point");
  eq([...dequantQ4(q, 32)].filter((_, i) => [0, 1, 16, 17].includes(i)), [-8, 3, -2, 7]);
});

Deno.test("Q4: the tail block reads zeros past n and never writes past n", () => {
  const x = Float32Array.from({ length: 40 }, (_, i) => i - 20);
  const q = quantizeQ4(x);
  // positions 40..63 of block 1 were read as 0: their nibbles are the zero point
  for (let j = 8; j < 16; j++) eq(q.qs[16 + j], 0x88, `tail byte ${j}`);
  const y = dequantQ4(q, 40);
  eq(y.length, 40);
});

Deno.test("Q4 scales pack two f16 per u32, even block in the low half", () => {
  const x = new Float32Array(96);
  x[0] = -8; x[32] = -16; x[64] = -24;   // scales 1, 2, 3
  const q = quantizeQ4(x);
  eq(q.scales.length, 2);
  eq(q.scales[0] & 0xFFFF, f32ToF16(1)); eq(q.scales[0] >>> 16, f32ToF16(2)); eq(q.scales[1] & 0xFFFF, f32ToF16(3));
  eq(q.scales[1] >>> 16, 0, "the unused half of the last word stays 0");
});

Deno.test("Q8 dequant matches gguf quantizeQ8 within half a step, tail blocks included", () => {
  for (const n of [1, 31, 32, 33, 64, 100, 1000]) {
    const x = randArr(n, 4);
    const q = quantizeQ8(x);
    const y = dequantQ8(q, n);
    eq(y.length, n);
    for (let i = 0; i < n; i++) {
      const d = Math.abs(scaleOf(q, (i / 32) | 0));
      ok(Math.abs(y[i] - x[i]) <= d / 2 * 1.001 + 1e-12, `n=${n} i=${i}: ${x[i]} -> ${y[i]} (d=${d})`);
    }
  }
});

Deno.test("Q8 dequant: signed bytes and per-block scales by hand", () => {
  // 64 values, block 0 scale 0.5, block 1 scale 2; bytes 0, 1, 127, 128 (-128), 255 (-1)
  const qs = new Uint8Array(64);
  qs.set([0, 1, 127, 128, 255], 0); qs.set([0, 1, 127, 128, 255], 32);
  const scales = new Uint32Array([f32ToF16(0.5) | (f32ToF16(2) << 16)]);
  const y = dequantQ8({ qs, scales }, 64);
  eq([...y.slice(0, 5)], [0, 0.5, 63.5, -64, -0.5]);
  eq([...y.slice(32, 37)], [0, 2, 254, -256, -2]);
});
