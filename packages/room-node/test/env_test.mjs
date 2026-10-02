// Dawn's options for the room node (env.js dawnFlagsFor): DAWN_OPTS as given, plus FXC's optimizer off
// on Windows (d3d_skip_shader_optimizations: the D3D12 backend's shader compiles ~2x faster).
import { test } from "node:test";
import assert from "node:assert/strict";
import { dawnFlagsFor } from "../env.js";

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
