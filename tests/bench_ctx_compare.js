// Compare two tests/bench_ctx.js runs of the same model, usually f16 against KV=q8 (issue #71):
// per fill, how many greedy tokens agree and the int8 / f16 speed ratios.
//   deno run --allow-read bench_ctx_compare.js f16.log q8.log   (the saved output, or just its RESULT line)
import { compareRuns } from "./ctx_plan.js";

const load = (path) => {
  const line = Deno.readTextFileSync(path).split("\n").find((l) => l.startsWith("RESULT "));
  if (!line) throw new Error(`${path}: no RESULT line`);
  return JSON.parse(line.slice(7));
};
const [pa, pb] = Deno.args;
if (!pa || !pb) { console.error("usage: bench_ctx_compare.js <run a> <run b>"); Deno.exit(2); }
const a = load(pa), b = load(pb);
console.log(`${a.model}: ${a.kv || "f16"} (a) vs ${b.kv || "f16"} (b)`);
for (const r of compareRuns(a, b)) {
  console.log(`fill ${String(r.fill).padStart(6)}: ${r.sameTokens}/${r.compared} greedy tokens the same${r.identical ? " (identical)" : ""} · prefill b/a ${r.prefill ?? "-"} · plain decode b/a ${r.plain ?? "-"}`);
}
