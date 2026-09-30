// No-GPU lint for the GPU test files (issue #81): a test that prints FAIL must also have a nonzero exit,
// so tests/run.sh sees the failure. run.sh also fails a test whose output has a FAIL word; that part is
// proved by `tests/run.sh selftest` (fixtures in tests/fixtures/run_sh/), run here when --allow-run is given.
import { assert, assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";

const TESTS = new URL("../", import.meta.url);
const read = (name) => Deno.readTextFileSync(new URL(name, TESTS));

// Prints FAIL: a string or template literal containing the word FAIL/FAILED (not "fail" in code).
const PRINTS_FAIL = /["'`][^"'`\n]*\bFAIL(ED)?\b/;
// Nonzero exit: Deno.exit(1 | 2 ...), or a Deno.exit(cond ? 0 : 1) either way round.
const EXITS_NONZERO = /Deno\.exit\(\s*[1-9]\s*\)|Deno\.exit\(.*\?\s*0\s*:\s*[1-9]\s*\)|Deno\.exit\(.*\?\s*[1-9]\s*:\s*0\s*\)/;

/** Returns why a test source would print FAIL and still exit 0, or "" when it is fine. */
export function lintFailExit(src) {
  if (!PRINTS_FAIL.test(src)) return "";
  return EXITS_NONZERO.test(src) ? "" : "prints FAIL but has no nonzero Deno.exit";
}

// The suites run.sh knows about, read from its quick=( ... ) style arrays.
function runShFiles() {
  const sh = read("run.sh"), out = [];
  for (const m of sh.matchAll(/^(quick|extra|q38)=\(([^)]*)\)/gm)) out.push(...m[2].trim().split(/\s+/));
  return out;
}

Deno.test("lintFailExit: flags a FAIL print with no nonzero exit", () => {
  assertEquals(lintFailExit(`console.log(ok ? "PASS" : "FAIL");`), "prints FAIL but has no nonzero Deno.exit");
  assertEquals(lintFailExit(`console.log(ok ? "PASS" : "FAIL"); Deno.exit(0);`), "prints FAIL but has no nonzero Deno.exit");
  assertEquals(lintFailExit("console.log(`${n} FAILED`);"), "prints FAIL but has no nonzero Deno.exit");
});

Deno.test("lintFailExit: accepts the exit forms the GPU tests use", () => {
  assertEquals(lintFailExit(`console.log("X FAIL"); Deno.exit(ok ? 0 : 1);`), "");
  assertEquals(lintFailExit(`console.log("X FAIL"); if (fail) Deno.exit(1);`), "");
  assertEquals(lintFailExit(`console.log("X FAIL"); Deno.exit(fail || errors.count ? 1 : 0);`), "");
  assertEquals(lintFailExit(`if (import.meta.main) Deno.exit((await run(await q38Context())) ? 0 : 1); const s = "FAIL";`), "");
  assertEquals(lintFailExit(`let fail = 0; console.log("ok");`), "");
});

Deno.test("every GPU test file in run.sh exists and exits nonzero when it prints FAIL", () => {
  const files = [...new Set([...runShFiles(), "run_q38_once.js"])];
  assert(files.length >= 20, `found only ${files.length} files in run.sh`);
  const bad = [];
  for (const f of files) {
    let src;
    try { src = read(f); } catch { bad.push(`${f}: listed in run.sh but missing`); continue; }
    const why = lintFailExit(src);
    if (why) bad.push(`${f}: ${why}`);
  }
  assertEquals(bad, []);
});

Deno.test("every tests/test_*.js exits nonzero when it prints FAIL", () => {
  const bad = [];
  for (const e of Deno.readDirSync(TESTS)) {
    if (!e.isFile || !/^test_.*\.js$/.test(e.name)) continue;
    const why = lintFailExit(read(e.name));
    if (why) bad.push(`${e.name}: ${why}`);
  }
  assertEquals(bad, []);
});

Deno.test("lintFailExit: catches a real test with its exit removed", () => {
  const src = read("test_reset.js");
  assertEquals(lintFailExit(src), "");
  assertEquals(lintFailExit(src.replace(/Deno\.exit\([^\n]*\);?/g, "")), "prints FAIL but has no nonzero Deno.exit");
});

Deno.test("run.sh selftest fixtures cover a FAIL print with exit 0", () => {
  const names = [...Deno.readDirSync(new URL("fixtures/run_sh/", TESTS))].map((e) => e.name);
  assert(names.includes("fail_print_exit0.js"), "fixture missing");
  assert(names.includes("pass_clean.js"), "fixture missing");
  // the fixture must be the bug itself: it prints FAIL and never exits nonzero
  assertEquals(lintFailExit(read("fixtures/run_sh/fail_print_exit0.js")), "prints FAIL but has no nonzero Deno.exit");
});

const canRun = Deno.permissions.querySync?.({ name: "run" }).state === "granted";
Deno.test({
  name: "tests/run.sh selftest: a printed FAIL gives a nonzero exit (needs --allow-run, no GPU)",
  ignore: !canRun,
  async fn() {
    const sh = new URL("run.sh", TESTS).pathname;
    const run = async (env) => {
      const r = await new Deno.Command("bash", { args: [sh, "selftest"], env, stdout: "piped", stderr: "piped" }).output();
      return { code: r.code, text: new TextDecoder().decode(r.stdout) + new TextDecoder().decode(r.stderr) };
    };
    const plain = await run({ STRICT: "0" });
    assertEquals(plain.code, 0, plain.text);
    assert(/ok\s+fixtures\/run_sh\/fail_print_exit0\.js -> 1/.test(plain.text), plain.text);
    const strict = await run({ STRICT: "1" });
    assertEquals(strict.code, 0, strict.text);
    assert(/ok\s+fixtures\/run_sh\/skip_print\.js -> 1/.test(strict.text), strict.text);
  },
});
