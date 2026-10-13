// Summarize profile_token.mjs results: node summarize.mjs <json>... -> one row per arm: decode tok/s per case and summed
// over the cases (tokens / seconds), verdicts, the cut split per decode token, the read-ahead's hit count
import fs from "node:fs";
import path from "node:path";
for (const f of process.argv.slice(2)) {
  const d = JSON.parse(fs.readFileSync(f, "utf8").trim().split("\n").pop());
  if (d.error) { console.log(path.basename(f), "ERROR", d.error.split("\n")[0]); continue; }
  let n = 0, s = 0;
  const per = [], v = [];
  const P = { tokMs: 0, readbackMs: 0, diskMs: 0, planMs: 0 };
  let steps = 0, ahead = [0, 0], reads = 0;
  for (const [name, c] of Object.entries(d.cases)) {
    n += c.tps * c.decodeS; s += c.decodeS; per.push(`${name} ${c.tps.toFixed(2)}`); v.push(c.verdict);
    if (c.prof) { for (const k of Object.keys(P)) P[k] += c.prof[k] * c.prof.decodeSteps; steps += c.prof.decodeSteps; }
    ahead[0] += c.aheadUsed || 0; ahead[1] += c.aheadReads || 0; reads += c.diskReads || 0;
  }
  const ms = Object.fromEntries(Object.entries(P).map(([k, x]) => [k, +(x / steps).toFixed(1)]));
  console.log([path.basename(f, ".json"), `sum ${(n / s).toFixed(3)} tok/s`, per.join(" | "), v.join(","), JSON.stringify(ms),
    `reads ${reads} ahead ${ahead[0]}/${ahead[1]}`].join("  "));
}
