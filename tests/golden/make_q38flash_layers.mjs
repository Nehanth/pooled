// Per-layer goldens for Qwen3.8-Flash-Next from a q38dump sequence (tests/reference/q38dump): for each layer's
// output (the 4-stream residual "l_last-N") and each token, the L2 norm over all streams and the first 8 values
// of stream 0, plus the top-5 logits per token. Small enough to commit; the engine's per-layer gate (roadmap 33
// PR 4) compares against it, and tests/reference/ref_q38flash.mjs reproduces the full tensors.
//   node tests/golden/make_q38flash_layers.mjs <dumps>/s0
import fs from "fs";
import path from "path";
import { loadDump } from "../reference/ref_q38flash.mjs";
const dir = path.join(process.argv[2], "c0"), dump = loadDump(dir);
const ids = fs.readFileSync(path.join(dir, "tokens.txt"), "utf8").trim().split(",").map(Number), T = ids.length;
const r6 = (v) => +v.toPrecision(6);
const layers = [];
for (let L = 0; dump.has(`l_last-${L}.0`); L++) {
  const R = dump.get(`l_last-${L}.0`), W = R.length / T;
  layers.push(Array.from({ length: T }, (_, t) => {
    let ss = 0;
    for (let i = 0; i < W; i++) ss += R[t * W + i] ** 2;
    return { norm: r6(Math.sqrt(ss)), head: Array.from(R.subarray(t * W, t * W + 8), r6) };
  }));
}
const lo = dump.get("result_output.0"), V = lo.length / T;
const top = Array.from({ length: T }, (_, t) => {
  const row = lo.subarray(t * V, (t + 1) * V);
  return [...row.keys()].sort((a, b) => row[b] - row[a]).slice(0, 5).map((i) => [i, r6(row[i])]);
});
const out = { _source: "llama.cpp f2cc728 CPU backend (q38dump: f32 KV, no flash attention, no repacking), bartowski Qwen3.8-Flash-Next-Q4_0; the reference matches these tensors to ~1e-7..1e-4 per op (tests/reference/ref_q38flash.mjs)", ids, layers, top };
fs.writeFileSync(new URL("./q38flash_layers.json", import.meta.url), "{\n" + Object.entries(out).map(([k, v]) => k === "layers" ? ` "layers": [\n${v.map((l) => "  " + JSON.stringify(l)).join(",\n")}\n ]` : ` ${JSON.stringify(k)}: ${JSON.stringify(v)}`).join(",\n") + "\n}\n");
