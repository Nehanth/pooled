// Run the JS reference (f32 activations) on the q38dump sequences, forcing only the residual at half-layer
// boundaries (hc_init, hc_combine-i, l_last-i) from llama.cpp's dump, and write every tapped tensor in q38dump's
// format so tests/test_q38flash.js LAYERS=N can compare the engine against the reference directly.
// The engine then matches the reference to ~1e-7 wherever both read the same weights (Q4_0, Q8_0, F32, BF16); where
// the engine reads a requantized copy (Q4_1 experts on layers 0-5, Q6_K attn_q, Q5_0 shared experts -> Q8) to ~5e-3.
// usage: Q38F_DIR=<model dir> node tests/reference/q38flash_refdump.mjs <dumpsDir> <outDir> <nLayers> [s0,s1,..]
//   then: LAYERS=<nLayers> Q38F_DUMPS=<outDir> deno run ... tests/test_q38flash.js
import fs from "fs"; import path from "path";
import * as R from "./ref_q38flash.mjs";
// FREE=1: no forcing at all (the reference chained on its own tensors), for tests/test_q38flash.js FREE=1
const [dumps, out, nL, seqArg] = process.argv.slice(2), FREE = process.env.FREE === "1";
const seqs = seqArg ? seqArg.split(",") : fs.readdirSync(dumps).filter((s) => /^s\d+$/.test(s)).sort();
const model = R.openModel();
for (const s of seqs) {
  let dump, seen, rec;
  const tap = (name, x) => {
    const k = seen.get(name) || 0; seen.set(name, k + 1);
    const key = `${name}.${k}`;
    if (!FREE && (name === "hc_init" || name.startsWith("hc_combine-") || name.startsWith("l_last-"))) {
      const r = dump.get(key);
      if (r) { rec.push([key, Float32Array.from(r), "LLAMA"]); return Float32Array.from(r); }   // forced: llama.cpp's
    }
    rec.push([key, x instanceof Int32Array ? Int32Array.from(x) : Float32Array.from(x)]);
    return null;
  };
  const ref = new R.Q38FlashRef(model, tap, { act: "f32" }), st = ref.newState();
  const chunks = fs.readdirSync(path.join(dumps, s)).filter((c) => /^c\d+$/.test(c)).sort((a, b) => +a.slice(1) - +b.slice(1));
  for (const c of chunks) {
    const cd = path.join(dumps, s, c), od = path.join(out, s, c);
    fs.mkdirSync(od, { recursive: true });
    const ids = fs.readFileSync(path.join(cd, "tokens.txt"), "utf8").trim().split(/[\s,]+/).map(Number);
    dump = R.loadDump(cd); seen = new Map(); rec = [];
    ref.forward(st, ids, +nL);
    const lines = [];
    for (const [key, a, op = "REF"] of rec) {
      const i32 = a instanceof Int32Array;
      fs.writeFileSync(path.join(od, key + (i32 ? ".i32" : ".f32")), new Uint8Array(a.buffer));
      lines.push([key, op, a.length, 1, 1, 1, i32 ? "i32" : "f32"].join("\t"));
    }
    // the residuals the engine is forced with stay llama.cpp's
    if (!FREE) for (const [key, a] of dump) if (/^(hc_init|hc_combine-\d+|l_last-\d+)\.\d+$/.test(key) && !rec.some(([k]) => k === key)) {
      fs.writeFileSync(path.join(od, key + ".f32"), new Uint8Array(a.buffer, a.byteOffset, a.byteLength)); lines.push([key, "LLAMA", a.length, 1, 1, 1, "f32"].join("\t"));
    }
    fs.writeFileSync(path.join(od, "shapes.tsv"), lines.join("\n") + "\n");
    fs.copyFileSync(path.join(cd, "tokens.txt"), path.join(od, "tokens.txt"));
    console.log(s, c, rec.length, "tensors");
  }
}
