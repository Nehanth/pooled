// engine/wgsl/prune.js: the per-entry-point module keeps the entry point and everything it reaches
// (and only that), so Dawn compiles the same kernel without reprocessing the engine's whole module
// for every pipeline (~1.3 s a pipeline under Windows' FXC; tests/unit can't time that, the PC can:
// packages/room-node/test/compile_prof.mjs).
import { wgslDecls, pruneWGSL, moduleSet } from "../../engine/wgsl/prune.js";
import { WGSL } from "../../engine/wgsl/base.js";
import { coopWGSL } from "../../engine/wgsl/coop.js";
import { WGSL2 } from "../../engine/wgsl/qwen35.js";
import { moeWGSL } from "../../engine/wgsl/moe.js";
import { layerFuseWGSL } from "../../engine/wgsl/layer_fuse.js";
import { SILU_MUL_W_WGSL } from "../../engine/wgsl/gemm_wide.js";

const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

const SMALL = `
enable f16;
// a comment naming unused_fn /* and a /* nested */ block comment */
diagnostic(off, derivative_uniformity);
struct Shape { rows: u32, cols: array<u32, 4>, }
alias V4 = vec4<f32>;
const K: u32 = 8u;
override WG: u32 = 64u;
@group(0) @binding(0) var<storage, read_write> ys: array<f32>;
@group(0) @binding(1) var<uniform> shape: Shape;
var<workgroup> sh: array<f32, K>;
fn helper(v: V4) -> f32 { return v.x + f32(K); }
fn unused_fn() -> f32 { return 1.0; }
@group(1) @binding(0) var<storage, read> only_b: array<f32>;
const_assert K > 0u;
@compute @workgroup_size(WG)
fn a(@builtin(local_invocation_id) lid: vec3<u32>) { sh[lid.x % K] = 1.0; ys[lid.x] = helper(V4(0.0)) + f32(shape.rows); }
@compute @workgroup_size(64) fn b(@builtin(global_invocation_id) g: vec3<u32>) { ys[g.x] = only_b[g.x]; };
`;

Deno.test("pruneWGSL: an entry point keeps what it reaches, drops the rest", () => {
  const names = (src) => new Set(wgslDecls(src).map((d) => d.name).filter(Boolean));
  const a = pruneWGSL(SMALL, "a");
  const na = names(a);
  for (const n of ["Shape", "V4", "K", "WG", "ys", "shape", "sh", "helper", "a"]) ok(na.has(n), `a keeps ${n}`);
  for (const n of ["unused_fn", "only_b", "b"]) ok(!na.has(n), `a drops ${n}`);
  ok(a.includes("enable f16;") && a.includes("diagnostic(off, derivative_uniformity);") && a.includes("const_assert K > 0u;"), "directives and const_assert stay");
  ok(!a.includes("comment"), "comments go");
  const b = pruneWGSL(SMALL, "b"), nb = names(b);
  ok(nb.has("b") && nb.has("ys") && nb.has("only_b") && !nb.has("a") && !nb.has("helper") && !nb.has("Shape"), [...nb].join(","));
  const both = names(pruneWGSL(SMALL, ["a", "b"]));
  ok(both.has("a") && both.has("b") && !both.has("unused_fn"));
  let err = null;
  try { pruneWGSL(SMALL, "nope"); } catch (e) { err = e; }
  ok(err && /no entry point "nope"/.test(err.message));
});

Deno.test("pruneWGSL: every engine kernel's module is closed over its references and much smaller", () => {
  const src = WGSL + coopWGSL(256, 4, 64, 4, 4, true, true, true) + moeWGSL() + SILU_MUL_W_WGSL + WGSL2 + layerFuseWGSL();
  const decls = wgslDecls(src);
  const declared = new Set(decls.map((d) => d.name).filter(Boolean));
  const entries = [...src.matchAll(/@compute[^]*?fn\s+([A-Za-z_0-9]+)/g)].map((m) => m[1]);
  ok(entries.length > 60, `entries: ${entries.length}`);
  let total = 0;
  for (const e of entries) {
    const out = pruneWGSL(src, e, decls);
    total += out.length;
    const kept = wgslDecls(out);
    const have = new Set(kept.map((d) => d.name).filter(Boolean));
    ok(have.has(e), `${e} kept`);
    for (const d of kept) for (const r of d.refs) ok(!declared.has(r) || have.has(r), `${e}: ${d.name} references ${r}, which was dropped`);
    // the same declarations, in source order
    const order = decls.filter((d) => d.name && have.has(d.name)).map((d) => d.name);
    ok(JSON.stringify(order) === JSON.stringify(kept.filter((d) => d.name).map((d) => d.name)), `${e}: order`);
  }
  ok(total / entries.length < src.length / 10, `average ${Math.round(total / entries.length)} of ${src.length} chars`);
});

Deno.test("moduleSet: one module per distinct pruned text; prune: false keeps the whole module", () => {
  const made = [];
  const device = { createShaderModule: (d) => { made.push(d.code); return { code: d.code }; } };
  const mod = moduleSet(device, SMALL);
  const ma = mod("a"), mb = mod("b");
  ok(ma !== mb && mod("a") === ma && made.length === 2, "cached per entry");
  const whole = moduleSet(device, SMALL, { prune: false });
  ok(whole("a") === whole("b") && whole("a").code === SMALL && made.length === 3, "unpruned");
});
