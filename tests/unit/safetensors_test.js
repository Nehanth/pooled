// engine/safetensors.js on hand-built files: header parsing, F32/BF16 reads, the tensor names a
// layer range needs, and the weights map DenseEngine builds from them. Before this only the GPU
// e2e (SmolLM) exercised it.
import { parseSafetensors, tensorF32, shardTensorNames, weightsFromSafetensors } from "../../engine/safetensors.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const throws = (f, re, m) => { try { f(); } catch (e) { if (re && !re.test(String(e))) throw new Error((m || "wrong error") + ": " + e); return; } throw new Error((m || "expected a throw")); };

// Build a .safetensors buffer: [u64 header length][header JSON, space-padded to `align`][data].
// tensors: { name: { dtype, shape, data: Float32Array | Uint16Array } }
function build(tensors, { meta = null, align = 8, order = null } = {}) {
  const names = order || Object.keys(tensors);
  const header = {};
  if (meta) header.__metadata__ = meta;
  let off = 0;
  const blobs = [];
  for (const name of names) {
    const t = tensors[name];
    const bytes = new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength);
    header[name] = { dtype: t.dtype, shape: t.shape, data_offsets: [off, off + bytes.length] };
    blobs.push(bytes); off += bytes.length;
  }
  let hj = new TextEncoder().encode(JSON.stringify(header));
  const padded = Math.ceil(hj.length / align) * align;
  const hb = new Uint8Array(padded).fill(0x20); hb.set(hj); hj = hb;
  const buf = new ArrayBuffer(8 + hj.length + off);
  new DataView(buf).setBigUint64(0, BigInt(hj.length), true);
  const u8 = new Uint8Array(buf);
  u8.set(hj, 8);
  let o = 8 + hj.length;
  for (const b of blobs) { u8.set(b, o); o += b.length; }
  return buf;
}
const f32 = (...v) => Float32Array.from(v);
// exact bf16 of a float whose low 16 bits are zero
const bf16 = (...v) => Uint16Array.from(v, (x) => new Uint32Array(Float32Array.of(x).buffer)[0] >>> 16);

Deno.test("parseSafetensors: offsets, dtypes, shapes; __metadata__ skipped", () => {
  const buf = build({
    "a": { dtype: "F32", shape: [2, 3], data: f32(1, 2, 3, 4, 5, 6) },
    "b": { dtype: "BF16", shape: [4], data: bf16(1, -2, 0.5, 256) },
    "c": { dtype: "F32", shape: [], data: f32(42) },
  }, { meta: { format: "pt" } });
  const t = parseSafetensors(buf);
  eq(Object.keys(t).sort(), ["a", "b", "c"], "no __metadata__ entry");
  const hl = Number(new DataView(buf).getBigUint64(0, true));
  eq([t.a.dtype, t.a.shape, t.a.byteOffset, t.a.byteLength], ["F32", [2, 3], 8 + hl, 24]);
  eq([t.b.dtype, t.b.shape, t.b.byteOffset, t.b.byteLength], ["BF16", [4], 8 + hl + 24, 8]);
  eq([t.c.byteOffset, t.c.byteLength], [8 + hl + 32, 4]);
  ok(t.a.arrayBuf === buf, "tensors view the same buffer, no copy");
  eq([...tensorF32(t.a)], [1, 2, 3, 4, 5, 6]);
  eq([...tensorF32(t.b)], [1, -2, 0.5, 256]);
  eq([...tensorF32(t.c)], [42], "a scalar (shape []) has one element");
});

Deno.test("parseSafetensors: data order in the file need not match header order", () => {
  const buf = build({ "x": { dtype: "F32", shape: [2], data: f32(7, 8) }, "y": { dtype: "F32", shape: [1], data: f32(9) } }, { order: ["y", "x"] });
  const t = parseSafetensors(buf);
  eq([...tensorF32(t.x)], [7, 8]); eq([...tensorF32(t.y)], [9]);
  ok(t.y.byteOffset < t.x.byteOffset);
});

Deno.test("parseSafetensors: empty file (no tensors) and metadata only", () => {
  eq(parseSafetensors(build({})), {});
  eq(parseSafetensors(build({}, { meta: { a: "b" } })), {});
});

Deno.test("parseSafetensors: a zero-size tensor", () => {
  const t = parseSafetensors(build({ "z": { dtype: "F32", shape: [0, 4], data: f32() }, "w": { dtype: "F32", shape: [1], data: f32(3) } }));
  eq(t.z.byteLength, 0); eq(tensorF32(t.z).length, 0); eq([...tensorF32(t.w)], [3]);
});

Deno.test("parseSafetensors: malformed headers throw", () => {
  const bad = new ArrayBuffer(17);
  new DataView(bad).setBigUint64(0, 9n, true);
  new Uint8Array(bad).set(new TextEncoder().encode("{not json"), 8);
  throws(() => parseSafetensors(bad), /JSON|Unexpected|Expected/, "junk header");
  const short = new ArrayBuffer(12);
  new DataView(short).setBigUint64(0, 1000n, true);   // header runs past the end
  throws(() => parseSafetensors(short), null, "header length past the end");
  throws(() => parseSafetensors(new ArrayBuffer(4)), null, "shorter than the length word");
});

Deno.test("tensorF32: BF16 bit patterns (sign, inf, nan, subnormal) widen exactly", () => {
  const raw = Uint16Array.from([0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x0001, 0x7F7F]);
  const buf = build({ "t": { dtype: "BF16", shape: [raw.length], data: raw } });
  const v = tensorF32(parseSafetensors(buf).t);
  eq(Object.is(v[0], 0), true); eq(Object.is(v[1], -0), true);
  eq([v[2], v[3], v[4], v[5]], [1, -1, Infinity, -Infinity]);
  ok(v[6] > 0 && v[6] < 1e-38, "bf16 subnormal");
  ok(v[7] > 3e38, "bf16 max");
});

Deno.test("tensorF32: unsupported dtypes throw", () => {
  for (const dtype of ["F16", "I64", "U8", "F64"]) {
    const buf = build({ "t": { dtype, shape: [2], data: f32(1, 2) } });
    throws(() => tensorF32(parseSafetensors(buf).t), /unsupported dtype/, dtype);
  }
});

Deno.test("tensorF32: shape product decides the element count, not byteLength", () => {
  const t = parseSafetensors(build({ "t": { dtype: "F32", shape: [2, 2], data: f32(1, 2, 3, 4) } })).t;
  eq(tensorF32({ ...t, shape: [3] }).length, 3);
});

const PARTS = ["input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
  "self_attn.v_proj.weight", "self_attn.o_proj.weight", "post_attention_layernorm.weight",
  "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"];

Deno.test("shardTensorNames: [lo, hi) with and without embed/head", () => {
  const cfg = { num_hidden_layers: 30 };
  const cases = [
    // [range, hasEmbed, hasHead, extra names first, layers]
    [[0, 1], false, false, [], [0]],
    [[0, 3], true, false, ["model.embed_tokens.weight"], [0, 1, 2]],
    [[27, 30], false, true, ["model.embed_tokens.weight", "model.norm.weight"], [27, 28, 29]],   // tied head needs the embedding
    [[0, 30], true, true, ["model.embed_tokens.weight", "model.norm.weight"], Array.from({ length: 30 }, (_, i) => i)],
    [[5, 5], false, false, [], []],
    [[5, 5], true, true, ["model.embed_tokens.weight", "model.norm.weight"], []],
    [[10, 12], false, false, [], [10, 11]],
  ];
  for (const [range, e, h, extra, layers] of cases) {
    const names = shardTensorNames(cfg, range, e, h);
    const want = [...extra, ...layers.flatMap((i) => PARTS.map((p) => `model.layers.${i}.${p}`))];
    eq(names, want, `range ${range} embed=${e} head=${h}`);
    eq(new Set(names).size, names.length, "no duplicates");
  }
});

Deno.test("shardTensorNames: ranges that tile the model cover every layer tensor exactly once (1 and 5 devices)", () => {
  const L = 30;
  for (const cuts of [[0, 30], [0, 6, 12, 18, 24, 30], [0, 1, 2, 3, 4, 30], [0, 29, 30]]) {
    const all = [];
    for (let d = 0; d + 1 < cuts.length; d++) all.push(...shardTensorNames({}, [cuts[d], cuts[d + 1]], d === 0, d === cuts.length - 2));
    const layerNames = all.filter((n) => n.startsWith("model.layers."));
    eq(layerNames.length, L * PARTS.length, `cuts ${cuts}`);
    eq(new Set(layerNames).size, layerNames.length, `cuts ${cuts}: layers dealt twice`);
    ok(all.includes("model.norm.weight") && all.includes("model.embed_tokens.weight"));
  }
});

// a small model file: 3 layers, dim 4; every tensor filled with its own marker value
function tinyModel(dtype = "F32") {
  const tensors = {};
  let k = 1;
  const put = (name, shape) => {
    const n = shape.reduce((a, b) => a * b, 1), v = k++;
    tensors[name] = { dtype, shape, data: dtype === "F32" ? new Float32Array(n).fill(v) : bf16(...new Array(n).fill(v)) };
  };
  put("model.embed_tokens.weight", [8, 4]);
  put("model.norm.weight", [4]);
  for (let i = 0; i < 3; i++) for (const p of PARTS) put(`model.layers.${i}.${p}`, p.includes("norm") ? [4] : [4, 4]);
  return parseSafetensors(build(tensors));
}

Deno.test("weightsFromSafetensors: layer slots map to the right tensors (F32 and BF16)", () => {
  for (const dtype of ["F32", "BF16"]) {
    const t = tinyModel(dtype);
    const W = weightsFromSafetensors(t, { lo: 1, hi: 3, hasEmbed: false, hasHead: false });
    eq(W.layers.length, 2);
    ok(!("embed" in W) && !("finalNorm" in W), "no embed or head in a middle shard");
    const slot = { inNorm: 0, q: 1, k: 2, v: 3, o: 4, postNorm: 5, gate: 6, up: 7, down: 8 };
    for (let li = 0; li < 2; li++) {
      for (const [key, pi] of Object.entries(slot)) {
        const w = W.layers[li][key];
        eq(w.kind, "f32", `${dtype} ${key}`);
        ok(w.data instanceof Float32Array, `${dtype} ${key} data`);
        eq(w.data[0], 3 + (li + 1) * 9 + pi, `${dtype} layer ${li + 1} ${key} got the wrong tensor`);
      }
    }
  }
});

Deno.test("weightsFromSafetensors: embed/head flags", () => {
  const t = tinyModel();
  const cases = [
    [{ hasEmbed: true, hasHead: false }, true, false],
    [{ hasEmbed: false, hasHead: true }, true, true],
    [{ hasEmbed: true, hasHead: true }, true, true],
    [{ hasEmbed: false, hasHead: false }, false, false],
  ];
  for (const [flags, embed, head] of cases) {
    const W = weightsFromSafetensors(t, { lo: 0, hi: 1, ...flags });
    eq(["embed" in W, "finalNorm" in W], [embed, head], JSON.stringify(flags));
    if (embed) { eq(W.embed.data.length, 32); eq(W.embed.data[0], 1); }
    if (head) { eq(W.finalNorm.data.length, 4); eq(W.finalNorm.data[0], 2); }
  }
});

Deno.test("weightsFromSafetensors: the names shardTensorNames lists are exactly the ones it reads", () => {
  const t = tinyModel();
  for (const [lo, hi, e, h] of [[0, 3, true, true], [1, 2, false, false], [2, 3, false, true], [0, 0, true, false]]) {
    const names = shardTensorNames({}, [lo, hi], e, h);
    const only = Object.fromEntries(names.map((n) => [n, t[n]]));   // what fetchModelShard would return
    const W = weightsFromSafetensors(only, { lo, hi, hasEmbed: e, hasHead: h });
    eq(W.layers.length, hi - lo);
  }
});

Deno.test("weightsFromSafetensors: a missing tensor fails loudly", () => {
  const t = tinyModel();
  delete t["model.layers.1.mlp.up_proj.weight"];
  throws(() => weightsFromSafetensors(t, { lo: 0, hi: 2, hasEmbed: false, hasHead: false }), null, "missing up_proj");
});
