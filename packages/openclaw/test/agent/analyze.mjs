// runtasks.mjs results -> a markdown table (and summary.json): node analyze.mjs <OUT dir> [label]
// Per task, from the gateway's "[pooled] done in room" lines: model calls, tool calls, the first call's
// prefill (its time to first token), prompt tokens read vs reused and where they were resumed from.
import fs from "node:fs";
const dir = process.argv[2], label = process.argv[3] || dir;
const res = JSON.parse(fs.readFileSync(`${dir}/results.json`, "utf8"));
const rows = [];
for (const r of res) {
  const gl = fs.readFileSync(`${dir}/${r.task}.gateway.log`, "utf8").split("\n").filter((l) => /^\S+ \[pooled\] done in room/.test(l));
  const calls = gl.map((l) => {
    const m = /: (\w+), (\d+) prompt tokens \((\d+) reused\), (\d+) out, calls ([^;]*); (\d+) tok · ([\d.]+) tok\/s .*?prompt \d+ tok: (\d+) read in ([\d.]+) s(?:, \d+ from ([a-z ]+))?/.exec(l) || [];
    return { prompt: +m[2], reused: +m[3], out: +m[4], calls: m[5], tps: +m[7], prefilled: +m[8], pre: +m[9], from: m[10] || "cold" };
  });
  const sum = (f) => calls.reduce((a, c) => a + f(c), 0);
  const outT = sum((c) => c.out), decS = sum((c) => c.out / Math.max(c.tps, 0.01));
  const tools = calls.flatMap((c) => c.calls && c.calls !== "none" ? c.calls.split(",") : []);
  const fromCounts = {}; for (const c of calls) fromCounts[c.from] = (fromCounts[c.from] || 0) + 1;
  rows.push({ task: r.task, ok: r.ok, seconds: +r.seconds.toFixed(0), calls: calls.length, tools: tools.length, toolList: tools.join(" "),
    ttft: calls[0]?.pre, firstPrompt: calls[0]?.prompt, maxPrompt: Math.max(...calls.map((c) => c.prompt)),
    prefilled: sum((c) => c.prefilled), reused: sum((c) => c.reused), promptTotal: sum((c) => c.prompt), prefillS: +sum((c) => c.pre).toFixed(1),
    decodeTps: +(outT / Math.max(decS, 1e-3)).toFixed(1), outTokens: outT, from: fromCounts, check: r.check });
}
console.log(`### ${label}\n`);
console.log("| task | ok | wall s | model calls | tool calls | TTFT 1st call s | prompt tok 1st / max | prefilled / total prompt tok (reuse %) | prefill s | decode tok/s | resumed from |");
console.log("|---|---|---|---|---|---|---|---|---|---|---|");
for (const x of rows) console.log(`| ${x.task} | ${x.ok ? "PASS" : "FAIL"} | ${x.seconds} | ${x.calls} | ${x.tools} | ${x.ttft} | ${x.firstPrompt} / ${x.maxPrompt} | ${x.prefilled} / ${x.promptTotal} (${(100 * x.reused / x.promptTotal).toFixed(0)}%) | ${x.prefillS} | ${x.decodeTps} | ${Object.entries(x.from).map(([k, v]) => `${k} ${v}`).join(", ")} |`);
fs.writeFileSync(`${dir}/summary.json`, JSON.stringify(rows, null, 1));
