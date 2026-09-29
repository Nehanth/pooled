// scripts/check-test-report.mjs: the CI guard over deno test's JUnit report.
import { parseJunit, problems, MAY_SKIP } from "../../scripts/check-test-report.mjs";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };

const tc = (name, body = "") => `<testcase name="${name}" classname="./t.js" time="0.001" line="1" col="1">\n${body}\n</testcase>`;
const report = (...cases) => `<?xml version="1.0" encoding="UTF-8"?>\n<testsuites name="deno test"><testsuite name="./t.js">${cases.join("\n")}</testsuite></testsuites>`;

Deno.test("check-test-report: parses passed, failed, errored and skipped cases", () => {
  const r = parseJunit(report(
    tc("a passes"),
    tc("b fails", `<failure message="Uncaught Error: x">Error: x</failure>`),
    tc("c errors", `<error message="boom"/>`),
    tc("weight cache: needs files", "<skipped/>"),
    tc("&quot;quoted&quot; &amp; &lt;tagged&gt;"),
    `<testcase name="self-closing" classname="./t.js" time="0"/>`,
  ));
  eq(r, { total: 6, failed: ["b fails", "c errors"], skipped: ["weight cache: needs files"] });
  eq(parseJunit(report(tc("&quot;q&quot; &amp; &lt;t&gt;", "<skipped/>"))).skipped, ['"q" & <t>']);
});

Deno.test("check-test-report: what fails CI, table-driven", () => {
  const cases = [
    ["all pass", report(tc("a"), tc("b")), []],
    ["allowed skip", report(tc("a"), tc("weight cache: x", "<skipped/>")), []],
    ["no tests at all", report(), ["no tests ran"]],
    ["a failure", report(tc("a", "<failure/>")), ["failed: a"]],
    ["a skip nobody allowed", report(tc("a"), tc("room: flaky one", "<skipped/>")), ["skipped without a reason in scripts/check-test-report.mjs: room: flaky one"]],
    ["the prefix must match from the start", report(tc("not weight cache: x", "<skipped/>")), ["skipped without a reason in scripts/check-test-report.mjs: not weight cache: x"]],
  ];
  for (const [what, xml, want] of cases) eq(problems(parseJunit(xml)), want, what);
});

Deno.test("check-test-report: every allowed skip says why", () => {
  for (const [prefix, why] of MAY_SKIP) if (!prefix.endsWith(":") || why.length < 10) throw new Error(`${prefix}: ${why}`);
});
