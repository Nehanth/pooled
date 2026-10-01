// Every workgroup barrier in the engine's WGSL is reached by all threads of the workgroup together
// (scripts/wgsl-barrier-lint.mjs). Windows' FXC compiler (Dawn's D3D12 backend without DXC) refuses a
// kernel that breaks this with X3663 "thread sync operation found in varying flow control", so one
// such kernel fails the whole engine on that GPU: topk_b did, the first Windows test of Pooled.
import { lintSources } from "../../scripts/wgsl-barrier-lint.mjs";

const WGSL_DIR = new URL("../../engine/wgsl/", import.meta.url);
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const lint = (wgsl) => lintSources([{ file: "t.js", js: "export const W = `" + wgsl + "`;" }]);
const HEAD = `
@group(0) @binding(0) var<storage, read> xs: array<u32>;
@group(0) @binding(1) var<storage, read_write> ys: array<u32>;
@group(0) @binding(2) var<uniform> u: vec4<u32>;
var<workgroup> sh: array<u32, 64>;
fn red(t: u32, v: u32) -> u32 { sh[t] = v; workgroupBarrier(); let r = sh[0]; workgroupBarrier(); return r; }`;
const kernel = (body) => `${HEAD}
@compute @workgroup_size(64)
fn k(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
  let t = lid.x; let n = u.x;
${body}
}`;

Deno.test("engine WGSL: no barrier that only some threads reach", () => {
  const sources = [...Deno.readDirSync(WGSL_DIR)].filter((e) => e.name.endsWith(".js")).sort((a, b) => (a.name < b.name ? -1 : 1))
    .map((e) => ({ file: e.name, js: Deno.readTextFileSync(new URL(e.name, WGSL_DIR)) }));
  ok(sources.length >= 10, "found the WGSL sources");
  const found = lintSources(sources);
  ok(!found.length, "\n" + found.map((x) => `${x.file} fn ${x.fn}: \`${x.what}\`: ${x.msg}`).join("\n"));
});

Deno.test("lint: flags the shapes FXC rejects", () => {
  const cases = {
    // topk_b before the fix: a run-time, thread-dependent candidate loop (with a data-dependent
    // continue) inside the round loop around a barrier helper
    topkB: `for (var r: u32 = 0u; r < n; r++) { var b: u32 = 0u;
      for (var e: u32 = t; e < n; e += 64u) { let i = xs[e]; if (i == 0u) { continue; } b = max(b, i); }
      let w = red(t, b); if (t == 0u) { ys[r] = w; } }`,
    continueBeforeBarrier: `for (var r: u32 = 0u; r < n; r++) { if (xs[t] == 0u) { continue; } workgroupBarrier(); }`,
    breakInBarrierLoop: `for (var r: u32 = 0u; r < n; r++) { workgroupBarrier(); if (xs[r * 64u + t] == 0u) { break; } }`,
    returnBeforeBarrier: `if (t >= n) { return; } sh[t] = 1u; workgroupBarrier();`,
    barrierInIf: `if (xs[t] > 0u) { workgroupBarrier(); }`,
    helperInIf: `if (t < 32u) { let w = red(t, 1u); }`,
    threadLoopWithBarrier: `for (var i: u32 = t; i < n; i += 64u) { workgroupBarrier(); }`,
    assignedUnderThreadCond: `var m: u32 = n; if (t == 0u) { m = 3u; } for (var i: u32 = 0u; i < m; i++) { workgroupBarrier(); }`,
  };
  for (const [name, body] of Object.entries(cases)) ok(lint(kernel(body)).length > 0, `${name}: not flagged`);
});

Deno.test("lint: passes uniform shapes", () => {
  const cases = {
    // topk_b after the fix: a trip count every thread shares, the body guarded instead
    topkB: `for (var r: u32 = 0u; r < n; r++) { var b: u32 = 0u;
      for (var e0: u32 = 0u; e0 < n; e0 += 64u) { let e = e0 + t; if (e < n) { let i = xs[e]; if (i != 0u) { b = max(b, i); } } }
      let w = red(t, b); if (t == 0u) { ys[r] = w; } }`,
    treeReduce: `sh[t] = xs[t]; workgroupBarrier(); for (var s: u32 = 32u; s > 0u; s >>= 1u) { if (t < s) { sh[t] += sh[t + s]; } workgroupBarrier(); }`,
    uniformReturn: `if (wg.x >= n) { return; } workgroupBarrier();`,
    uniformBreak: `for (var r: u32 = 0u; r < 8u; r++) { if (r == n) { break; } workgroupBarrier(); }`,
    returnAfterLastBarrier: `workgroupBarrier(); if (t > 0u) { return; } ys[0] = sh[0];`,
    uniformStorageIndex: `let m = xs[wg.y]; for (var i: u32 = 0u; i < m; i++) { workgroupBarrier(); }`,
    uniformLoad: `if (t == 0u) { sh[0] = xs[0]; } let m = workgroupUniformLoad(&sh[0]); for (var i: u32 = 0u; i < m; i++) { workgroupBarrier(); }`,
    // the same names in two blocks are two variables
    shadowed: `{ let s = t; ys[s] = 1u; } { let s = n; for (var i: u32 = 0u; i < s; i++) { workgroupBarrier(); } }`,
    // a thread-dependent loop next to inline barriers (moe_router, attn_flash): FXC compiles it
    inlineBarriers: `for (var k: u32 = 0u; k < n; k++) { var b: u32 = 0u; for (var i: u32 = t; i < n; i += 64u) { if (xs[i] > b) { b = xs[i]; } } sh[t] = b; workgroupBarrier(); }`,
  };
  for (const [name, body] of Object.entries(cases)) {
    const f = lint(kernel(body));
    ok(!f.length, `${name}: flagged: ${JSON.stringify(f)}`);
  }
});

Deno.test("lint: reads the WGSL inside template literals, ${} included", () => {
  const js = "const P = 'q'; export const W = (n) => `" + HEAD + `
@compute @workgroup_size(64)
fn k_\${P}(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  \${n > 1 ? \`if (xs[t] > 0u) { return; }\` : ""}
  workgroupBarrier();
}\`;`;
  const f = lintSources([{ file: "t.js", js }]);
  ok(f.length === 1 && f[0].fn === "k__X_", JSON.stringify(f));
});
