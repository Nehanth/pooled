// engine/compile.js: a pipeline that fails to compile names its kernel (the CLI's "this GPU's shader
// compiler can't build <kernel>" comes from it) and keeps the compiler's full text for --verbose.
import { compilePipeline } from "../../engine/compile.js";

const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

Deno.test("compilePipeline: passes a pipeline through, names the kernel on failure", async () => {
  const good = { createComputePipelineAsync: async (d) => ({ entry: d.compute.entryPoint }) };
  ok((await compilePipeline(good, { compute: { entryPoint: "topk_a" } })).entry === "topk_a");
  const raw = "FXC compile failed with error: E_FAIL msg: C:\\fakepath(51,3-35): error X3663: thread sync operation found in varying flow control\n  /* Generated HLSL: */\n  struct tint_struct {";
  const bad = { createComputePipelineAsync: async () => { throw new Error(raw); } };
  let err = null;
  try { await compilePipeline(bad, { compute: { entryPoint: "topk_b" } }); } catch (e) { err = e; }
  ok(err && err.shaderCompile === true && err.kernel === "topk_b" && err.raw === raw, "fields");
  ok(err.message.startsWith("shader compile failed for topk_b: FXC compile failed") && !err.message.includes("\n"), err.message);
});
