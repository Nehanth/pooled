// Dawn's options for the room node (env.js dawnFlagsFor): DAWN_OPTS as given, plus FXC's optimizer off
// on Windows (d3d_skip_shader_optimizations: the D3D12 backend's shader compiles ~2x faster).
import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { dawnFlagsFor, v8FlagsFor } from "../env.js";

const SKIP = "d3d_skip_shader_optimizations";
test("dawnFlagsFor: DAWN_OPTS elsewhere, FXC's optimizer off on Windows", () => {
  assert.deepEqual(dawnFlagsFor({}, "linux"), []);
  assert.deepEqual(dawnFlagsFor({ DAWN_OPTS: "backend=vulkan  enable-dawn-features=dump_shaders" }, "darwin"), ["backend=vulkan", "enable-dawn-features=dump_shaders"]);
  assert.deepEqual(dawnFlagsFor({}, "win32"), [`enable-dawn-features=${SKIP}`]);
  // merged into the one enable list (Dawn takes the last flag of a name)
  assert.deepEqual(dawnFlagsFor({ DAWN_OPTS: "enable-dawn-features=dump_shaders,disable_symbol_renaming" }, "win32"),
    [`enable-dawn-features=dump_shaders,disable_symbol_renaming,${SKIP}`]);
  // opt-out, and a toggle the user already named (enabled or disabled) is left alone
  assert.deepEqual(dawnFlagsFor({ POOLED_FXC_OPTIMIZE: "1" }, "win32"), []);
  assert.deepEqual(dawnFlagsFor({ DAWN_OPTS: `disable-dawn-features=${SKIP}` }, "win32"), [`disable-dawn-features=${SKIP}`]);
});

// V8's Maglev compiler crashes a Windows process that holds a Dawn device (env.js v8FlagsFor)
test("v8FlagsFor: Maglev off on Windows only", () => {
  assert.deepEqual(v8FlagsFor("win32"), ["--no-maglev"]);
  for (const p of ["linux", "darwin"]) assert.deepEqual(v8FlagsFor(p), []);
});

// setFlagsFromString("--no-maglev") after startup does stop Maglev: a hot function compiled after it
// goes to TurboFan, never Maglev (--trace-opt names each compile's target)
test("--no-maglev set at run time: nothing compiles with Maglev after it", (t) => {
  const hot = `function f(n){let s=0;for(let i=0;i<n;i++)s=(s+i*31+(s>>>3))|0;return s}
    let a=0;for(let r=0;r<3000;r++)a^=f(2000);console.log("ran",a)`;
  const run = (pre) => spawnSync(process.execPath, ["--trace-opt", "--input-type=module", "-e", pre + hot], { encoding: "utf8" }).stdout;
  const maglevF = (out) => out.split("\n").filter((l) => /JSFunction f /.test(l) && /MAGLEV/.test(l)).length;
  const before = run(""), after = run(`(await import("node:v8")).setFlagsFromString("--no-maglev");`);
  assert.match(before, /ran/); assert.match(after, /ran/);
  if (maglevF(before) === 0) return t.skip(`this V8 (Node ${process.versions.node}) did not use Maglev for the hot function`);
  assert.equal(maglevF(after), 0);
});
