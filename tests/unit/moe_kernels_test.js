// CPU check of the MoE expert GEMV kernels (engine/wgsl/moe.js); the harness is in moe_check.js.
// The layout sweep is split by workgroup size across this file and moe_sweep_wg{64,128}_test.js.
// No GPU.   deno test --no-check tests/unit/moe_kernels_test.js tests/unit/moe_sweep_wg*_test.js
import { moeKernelConfig, moeWGSL, MOE_DEFAULT } from "../../engine/wgsl/moe.js";
import { check, ok, sweep } from "./moe_check.js";

Deno.test("moe kernels: presets at the real Qwen3.6-35B-A3B shape (2048 / 512)", () => {
  for (const p of ["default", "legacy"]) {
    const cfg = moeKernelConfig(p, { dim: 2048, inter: 512 });
    for (const [g, d] of [["q4", "q4"], ["q8", "q8"]]) ok(check(g, d, cfg, { dim: 2048, inter: 512, nExp: 3, K: 2, C: 1 }), `${p} ${g}/${d}`);
  }
});

Deno.test("moe kernels: layout sweep with tails (dim 256, expert width 96), WG 32", () => {
  const n = sweep([32]);
  if (n < 150) throw new Error(`only ${n} layouts checked`);
});

Deno.test("moe kernels: config resolution", () => {
  const c = moeKernelConfig("default", { dim: 2048, inter: 512 });
  if (c.gu.rows !== 4 || c.dn.rows !== 16 || !c.gu.xsh || !c.dn.xsh) throw new Error("default rows / staging changed: " + JSON.stringify(c));
  const l = moeKernelConfig("legacy", { dim: 2048, inter: 512 });
  if (l.gu.rows !== 4 || l.dn.rows !== 4 || l.gu.wide || l.dn.xsh) throw new Error("legacy preset changed");
  const o = moeKernelConfig({ dn: { TPR: 8 } }, { dim: 2048, inter: 512 });
  if (o.dn.TPR !== 8 || o.dn.WG !== MOE_DEFAULT.dn.WG || o.gu.TPR !== MOE_DEFAULT.gu.TPR) throw new Error("partial override");
  // a gate/up input too wide to stage falls back to reading x from the storage buffer
  if (moeKernelConfig("default", { dim: 8192, inter: 512 }).gu.xsh) throw new Error("8192-wide x cannot fit in 16 KB with the reduction");
  for (const bad of [{ gu: { WG: 96 } }, { gu: { TPR: 512 } }, { dn: { R: 9 } }, { dn: { wide: false, TPR: 2 } }, "fast"]) {
    let threw = false; try { moeKernelConfig(bad, { dim: 2048, inter: 512 }); } catch { threw = true; }
    if (!threw) throw new Error("accepted " + JSON.stringify(bad));
  }
  if (!moeWGSL(c).includes("fn moe_dn_q8")) throw new Error("moeWGSL output");
});
