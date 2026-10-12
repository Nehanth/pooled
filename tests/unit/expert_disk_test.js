// Expert offload's disk tier (engine/expert_disk.js): one expert read from a stacked GGUF tensor and converted on its
// own must be byte for byte the slice of the whole tensor's conversion (engine/gguf.js convertEntry) at that expert,
// for every format the loader takes (Q4_0 / Q8_0 repacked, other quants requantized to Q8). Synthetic tensors, so CI
// needs no model file; with the 35B MoE in models/ it also checks sampled layers of the real file.
//   deno test --no-check --allow-read tests/unit/expert_disk_test.js
import { expertDisk, expertGeometry, expertKind } from "../../engine/expert_disk.js";
import { convertEntry, ggmlTypeBytes, parseGGUFHeader, GGML_F32, GGML_F16, GGML_Q4_0, GGML_Q4_1, GGML_Q5_0, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0 } from "../../engine/gguf.js";

const assert = (c, m) => { if (!c) throw new Error(m); };
const bytesOf = (a) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
const same = (a, b) => a.byteLength === b.byteLength && a.every((x, i) => x === b[i]);

// a file: `pad` bytes, then a stacked tensor [nExp][rows][cols] of `type` with deterministic contents. Each block's
// leading f16 (the scale, for every type here but the K-quants, where it is a sub-scale byte pair) is kept small and
// finite so the requantizing types see real numbers.
function synthFile(type, nExp, rows, cols, pad = 4096 + 18) {
  const n = nExp * rows * cols, len = ggmlTypeBytes(type, n), file = new Uint8Array(pad + len);
  let s = 12345;
  for (let i = pad; i < file.length; i++) file[i] = (s = (s * 1103515245 + 12345) >>> 0) >>> 24;
  const blk = { [GGML_Q4_0]: 18, [GGML_Q4_1]: 20, [GGML_Q5_0]: 22, [GGML_Q8_0]: 34 }[type];
  if (blk) for (let o = pad; o < file.length; o += blk) { file[o] = 0x00; file[o + 1] = 0x24; if (type === GGML_Q4_1) { file[o + 2] = 0x00; file[o + 3] = 0xa4; } }
  if (type === GGML_Q6_K) for (let o = pad; o < file.length; o += 210) { file[o + 208] = 0x00; file[o + 209] = 0x24; }
  if (type === GGML_Q5_K) for (let o = pad; o < file.length; o += 176) { file[o] = 0x00; file[o + 1] = 0x24; file[o + 2] = 0x00; file[o + 3] = 0x20; }
  return { file, info: { name: "blk.0.ffn_gate_exps.weight", shape: [nExp, rows, cols], ggmlType: type, byteOffset: pad, byteLength: len, shard: 0 } };
}
const reader = (files) => expertDisk({ readAt: async (shard, off, len) => files[shard].slice(off, off + len) });

for (const [name, type, cols] of [["Q4_0", GGML_Q4_0, 128], ["Q8_0", GGML_Q8_0, 128], ["Q4_1", GGML_Q4_1, 128], ["Q5_0", GGML_Q5_0, 128], ["Q6_K", GGML_Q6_K, 256], ["Q5_K", GGML_Q5_K, 512]]) {
  Deno.test(`readPart: every expert of a ${name} tensor equals its slice of the whole conversion`, async () => {
    const nExp = 8, rows = 24, { file, info } = synthFile(type, nExp, rows, cols);
    const whole = convertEntry({ ...info, shape: [nExp * rows, cols], nElems: nExp * rows * cols }, file.subarray(info.byteOffset, info.byteOffset + info.byteLength));
    const g = expertGeometry(info), qsB = rows * cols / (g.kind === "q4" ? 2 : 1), scB = rows * cols / 16;
    assert(whole.kind === g.kind && g.kind === expertKind(type), `${name}: kind ${whole.kind} vs ${g.kind}`);
    const R = reader([file]);
    for (let e = 0; e < nExp; e++) {
      const { qs, sc } = await R.readPart(info, e);
      assert(same(qs, bytesOf(whole.qs).subarray(e * qsB, (e + 1) * qsB)), `${name} expert ${e}: qs`);
      assert(same(sc, bytesOf(whole.scales).subarray(e * scB, (e + 1) * scB)), `${name} expert ${e}: scales`);
    }
    assert(R.stats.reads === nExp && R.stats.bytes === info.byteLength, `stats ${JSON.stringify(R.stats)}`);
  });
}

Deno.test("readPart: a tensor in another shard of a split file, and the guards", async () => {
  const a = synthFile(GGML_Q4_0, 4, 16, 64, 64), b = synthFile(GGML_Q8_0, 4, 16, 64, 640);
  const R = reader([a.file, b.file]);
  const whole = convertEntry({ ...b.info, shape: [64, 64], nElems: 64 * 64 }, b.file.subarray(640, 640 + b.info.byteLength));
  const got = await R.readPart({ ...b.info, shard: 1 }, 3);
  assert(same(got.qs, bytesOf(whole.qs).subarray(3 * 16 * 64, 4 * 16 * 64)), "shard 1, expert 3");
  let threw = 0;
  for (const f of [() => expertKind(GGML_F32), () => expertKind(GGML_F16), () => expertGeometry({ shape: [64, 64], ggmlType: GGML_Q4_0 }), () => R.readPart(a.info, 4), () => R.readPart(a.info, -1)])
    try { await f(); } catch { threw++; }
  assert(threw === 5, `guards: ${threw} of 5 threw`);
  const short = expertDisk({ readAt: async () => new Uint8Array(10) });
  let msg = "";
  try { await short.readPart(a.info, 0); } catch (e) { msg = e.message; }
  assert(/short read/.test(msg), "a short read fails loudly");
});

// The real 35B MoE file, when it is there (models/q36moe): sampled experts of sampled layers, each part
const MOE = new URL("../../models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf", import.meta.url).pathname;
let moe = false;
try { moe = Deno.statSync(MOE).isFile; } catch { moe = false; }
Deno.test({ name: "q36moe real file: readPart gives the 35B MoE's experts as their slices of the whole tensors (models/q36moe)", ignore: !moe, fn: async () => {
  const fs = await import("node:fs");
  const fd = fs.openSync(MOE, "r");
  const read = (off, len) => { const out = new Uint8Array(len); let o = 0; while (o < len) { const n = fs.readSync(fd, out, o, len - o, off + o); if (n <= 0) break; o += n; } return out; };
  try {
    const G = parseGGUFHeader(read(0, 64 << 20).buffer);
    const R = expertDisk({ readAt: async (_s, off, len) => read(off, len) });
    const types = new Set();
    for (const l of [0, 3, 13, 20, 31, 39]) for (const part of ["gate", "up", "down"]) {
      const info = { name: `blk.${l}.ffn_${part}_exps.weight`, ...G.tensors[`blk.${l}.ffn_${part}_exps.weight`] };
      const g = expertGeometry(info), whole = convertEntry({ ...info, shape: [g.nExp * g.rows, g.cols], nElems: info.nElems }, read(info.byteOffset, info.byteLength));
      const qsB = g.rows * g.cols / (g.kind === "q4" ? 2 : 1), scB = g.rows * g.cols / 16;
      types.add(info.ggmlType);
      for (const e of [0, 1, 77, 128, g.nExp - 1]) {
        const { qs, sc } = await R.readPart(info, e);
        assert(same(qs, bytesOf(whole.qs).subarray(e * qsB, (e + 1) * qsB)) && same(sc, bytesOf(whole.scales).subarray(e * scB, (e + 1) * scB)), `layer ${l} ${part} expert ${e}`);
      }
    }
    console.log(`  ggml types checked: ${[...types].join(", ")}`);
  } finally { fs.closeSync(fd); }
} });
