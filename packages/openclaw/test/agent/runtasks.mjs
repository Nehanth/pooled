// The OpenClaw agent tasks of the POC (docs/openclaw-poc.md): five real agent tasks through a running
// gateway whose default model is a Pooled room, each in a new session, checked by check.mjs.
//   read     a question that needs reading files
//   edit     a code edit that must be correct (formatMoney on negative amounts)
//   tests    run the tests, fix the source, run them again
//   feature  a new module + its tests + running them until they pass
//   long     read a ~140 KB file (~33k tokens) in chunks and write a summary of its planted incidents
// Setup: node sandbox.mjs <proj>; point OpenClaw's agents.defaults.workspace at <proj>; start the
// gateway with POOLED_DEBUG=1 and its output in GW_LOG (the per-call "[pooled] done in room" lines).
// env: OUT (results dir), PROJ, GW_LOG, CHECK (default ./check.mjs), TASKS (default all), OC (openclaw
//      command), SID (session id prefix). Then: node analyze.mjs OUT for the table.
import fs from "node:fs"; import path from "node:path"; import { execFileSync, spawnSync } from "node:child_process";
const OUT = process.env.OUT, PROJ = process.env.PROJ, GW = process.env.GW_LOG, CHECK = process.env.CHECK || new URL("./check.mjs", import.meta.url).pathname;
const SIDP = process.env.SID || "poc";
fs.mkdirSync(OUT, { recursive: true });
const PROMPTS = {
  read: "Question about the project in this workspace: how many attempts does the tax-rate HTTP client make before it gives up, and in which file and function is that limit defined and enforced? Read the source files to find out; do not guess.",
  edit: "In src/format.js, formatMoney is wrong for negative amounts: formatMoney(-1234) should return \"-$12.34\" and formatMoney(-5) should return \"-$0.05\" (positive amounts must stay exactly as they are now). Fix it.",
  tests: "Run the test suite with `npm test` in the workspace, fix the source code so that all tests pass (do not change the tests), and run the tests again to confirm.",
  feature: "Add a new module src/pricing.js that exports bulkPrice(unitPriceCents, qty): it returns the total in cents, unitPriceCents * qty, with 10% off when qty >= 10 and 20% off when qty >= 50, rounded with Math.round. Then write tests for it in test/pricing.test.js (node:test and node:assert/strict, like the existing tests), run npm test, and fix anything that fails until everything passes.",
  long: "docs/ops-journal.md is a long operations journal (about 140 KB). Read the whole file; if the read tool truncates it, keep reading with offset/limit until you reach the end. Then write SUMMARY.md in the workspace listing every incident marked [MAJOR]: its date, what happened, and its root cause, plus a one-paragraph overview of the routine work.",
};
const tasks = (process.env.TASKS || "read,edit,tests,feature,long").split(",");
const parseStats = (s) => {
  const m = /(\d+) tok · ([\d.]+) tok\/s · (\d+) devices?.*?prompt (\d+) tok: (\d+) read in ([\d.]+) s(?:, (\d+) from ([a-z ]+))?/.exec(s) || [];
  return { out: +m[1] || 0, tps: +m[2] || 0, devices: +m[3] || 0, prompt: +m[4] || 0, prefilled: +m[5] || 0, prefillS: +m[6] || 0, reused: +m[7] || 0, from: m[8] || null };
};
const summary = [];
for (const t of tasks) {
  const off = fs.existsSync(GW) ? fs.statSync(GW).size : 0;
  const sid = `${SIDP}-${t}`, t0 = Date.now();
  const r = spawnSync(process.env.OC || "openclaw", ["agent", "--agent", "main", "--session-id", sid, "--message", PROMPTS[t], "--json", "--timeout", "2400"],
    { encoding: "utf8", maxBuffer: 256 << 20, timeout: 2500000 });
  const secs = (Date.now() - t0) / 1000;
  fs.writeFileSync(`${OUT}/${t}.agent.out`, (r.stdout || "") + "\n--- stderr ---\n" + (r.stderr || ""));
  let json = null; try { json = JSON.parse(r.stdout.slice(r.stdout.indexOf("{"))); } catch {}
  const payloads = json?.result?.payloads || json?.payloads || [];
  const text = payloads.map((p) => p.text || "").join("\n");
  fs.writeFileSync(`${OUT}/${t}.answer.txt`, text);
  const gl = fs.existsSync(GW) ? fs.readFileSync(GW).subarray(off).toString() : "";
  fs.writeFileSync(`${OUT}/${t}.gateway.log`, gl);
  const calls = gl.split("\n").filter((l) => /\[pooled\] done in room/.test(l) && !l.includes("\x1b[")).map((l) => {
    const m = /: (\w+), (\d+) prompt tokens \((\d+) reused\), (\d+) out, calls ([^;]*); (.*)$/.exec(l) || [];
    return { reason: m[1], prompt: +m[2], reused: +m[3], out: +m[4], calls: m[5], ...parseStats(m[6] || ""), raw: (m[6] || l).slice(0, 300) };
  });
  const fails = gl.split("\n").filter((l) => !l.includes("\x1b[") && /request failed|Pooled:|error/i.test(l) && /pooled/i.test(l)).slice(0, 20);
  let chk = null; try { chk = JSON.parse(execFileSync("node", [CHECK, t, PROJ, `${OUT}/${t}.answer.txt`], { encoding: "utf8" })); } catch (e) { chk = { ok: false, detail: ["check crashed " + e.message] }; }
  const tools = calls.flatMap((c) => (c.calls && c.calls !== "none") ? c.calls.split(",") : []);
  const row = { task: t, ok: chk.ok, check: chk.detail, seconds: secs, exit: r.status, modelCalls: calls.length, tools,
    firstPrefillS: calls[0]?.prefillS ?? null, firstPrompt: calls[0]?.prompt ?? null, firstReused: calls[0]?.reused ?? null,
    maxPrompt: Math.max(0, ...calls.map((c) => c.prompt)), prefillTotalS: +calls.reduce((a, c) => a + c.prefillS, 0).toFixed(1),
    outTokens: calls.reduce((a, c) => a + c.out, 0), meanTps: calls.length ? +(calls.reduce((a, c) => a + c.tps, 0) / calls.length).toFixed(1) : null,
    perCall: calls.map((c) => `${c.prompt}p/${c.reused}r/${c.prefillS}s/${c.out}o/${c.tps}t/s ${c.calls}`), fails, answer: text.slice(0, 1500) };
  summary.push(row);
  fs.writeFileSync(`${OUT}/results.json`, JSON.stringify(summary, null, 1));
  console.log(new Date().toISOString(), JSON.stringify({ task: t, ok: row.ok, seconds: secs, modelCalls: row.modelCalls, tools, firstPrefillS: row.firstPrefillS, maxPrompt: row.maxPrompt, meanTps: row.meanTps }));
}
