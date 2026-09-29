#!/usr/bin/env node
// CI guard over `deno test --junit-path` output: fail when a test failed or errored, when no tests
// ran at all (a bad glob), or when a test was skipped that is not on the list below. deno test
// already exits non-zero on failures; this catches the quiet ways a suite can stop testing.
//   node scripts/check-test-report.mjs report.xml
import { existsSync, readFileSync } from "node:fs";

// Tests that may skip, by name prefix, and why. Anything else that skips fails CI.
export const MAY_SKIP = [
  ["weight cache:", "needs GGUF weights on disk and a writable cache dir; CI has neither"],
];

// -> { total, failed: [name], skipped: [name] } from a JUnit XML string
export function parseJunit(xml) {
  const out = { total: 0, failed: [], skipped: [] };
  const re = /<testcase\b([^>]*?)(\/>|>([\s\S]*?)<\/testcase>)/g;
  for (let m; (m = re.exec(xml));) {
    out.total++;
    const name = unescape(/\bname="([^"]*)"/.exec(m[1])?.[1] ?? "?");
    const body = m[3] || "";
    if (/<(failure|error)\b/.test(body)) out.failed.push(name);
    else if (/<skipped\b/.test(body)) out.skipped.push(name);
  }
  return out;
}
const unescape = (s) => s.replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");

// -> [problem] (empty when the report is fine)
export function problems(report, allow = MAY_SKIP) {
  const out = [];
  if (!report.total) out.push("no tests ran");
  for (const n of report.failed) out.push(`failed: ${n}`);
  for (const n of report.skipped) if (!allow.some(([p]) => n.startsWith(p))) out.push(`skipped without a reason in scripts/check-test-report.mjs: ${n}`);
  return out;
}

if (import.meta.main ?? process.argv[1]?.endsWith("check-test-report.mjs")) {
  const file = process.argv[2];
  if (!file) { console.error("usage: node scripts/check-test-report.mjs report.xml"); process.exit(2); }
  if (!existsSync(file)) { console.error(`::error::no test report at ${file}: deno test stopped before writing it`); process.exit(1); }
  const r = parseJunit(readFileSync(file, "utf8"));
  const bad = problems(r);
  console.log(`${r.total} tests, ${r.failed.length} failed, ${r.skipped.length} skipped (allowed: ${MAY_SKIP.map(([p]) => p).join(", ")})`);
  for (const p of bad) console.error("::error::" + p);
  process.exit(bad.length ? 1 : 0);
}
