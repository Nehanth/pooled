// Shard of the MoE kernel layout sweep (see moe_check.js); a separate file so deno test --parallel
// can run it next to the other shards. No GPU.
import { sweep } from "./moe_check.js";

Deno.test("moe kernels: layout sweep with tails (dim 256, expert width 96), WG 128", () => {
  const n = sweep([128]);
  if (n < 150) throw new Error(`only ${n} layouts checked`);
});
