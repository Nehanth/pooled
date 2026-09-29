// room/code.js's pure exports: projectName (a new scratch project's name from the first request)
// and makeDiff (the approval card's diff). Only the GPU e2e room_synth touched this file before.
import { projectName, makeDiff } from "../../room/code.js";
import { slugify } from "../../harness/projects.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

Deno.test("projectName: table", () => {
  const cases = [
    ["build a tetris game", "tetris"],
    ["Build a Tetris game", "tetris", "case"],
    ["make me a snake game with canvas", "snake"],
    ["Please create a simple todo list app using plain JavaScript and HTML", "todo list", "keeps at most two words"],
    ["write a markdown previewer that renders tables", "markdown previewer", "the first two kept words"],
    ["build pong", "pong"],
    ["2048 clone", "2048 clone", "digits are words"],
    ["tic-tac-toe!", "tic tac", "punctuation splits words"],
    ["", "project", "empty"],
    ["!!! ??? ...", "project", "punctuation only"],
    ["build a game", "project", "only filler words"],
    ["make a new web app with html css and js", "project", "only filler words, many"],
    ["俄罗斯方块", "project", "no ASCII words"],
    ["  \n\t  ", "project", "whitespace"],
  ];
  for (const [text, want, why] of cases) eq(projectName(text), want, why || text);
});

Deno.test("projectName: always a non-empty, slug-able name", () => {
  for (const t of ["a", "the the the", "Café au lait", "x".repeat(5000), "build a 🚀 rocket game", "über cool app"]) {
    const n = projectName(t);
    ok(n.length > 0, `empty name for ${t}`);
    ok(/^[a-z0-9]+( [a-z0-9]+)?$/.test(n), `name ${JSON.stringify(n)} for ${JSON.stringify(t)}`);
    ok(slugify(n) !== "", "slug of the name");
  }
});

Deno.test("makeDiff: small edits give rows with counts", () => {
  const cases = [
    // [input, expected subset, why]
    [{ path: "a.js", before: "x\ny\n", after: "x\nz\n" }, { path: "a.js", isNew: false, lines: 2, add: 1, del: 1, rows: [[" ", "x"], ["-", "y"], ["+", "z"]] }, "one line changed"],
    [{ path: "n.js", before: null, after: "a\nb" }, { isNew: true, lines: 2, add: 2, del: 0, rows: [["+", "a"], ["+", "b"]] }, "new file (null before)"],
    [{ path: "n.js", before: undefined, after: "a\n" }, { isNew: true, lines: 1, add: 1, del: 0 }, "new file (undefined before)"],
    [{ path: "e.js", before: "", after: "a\n" }, { isNew: false, lines: 1, add: 1, del: 0 }, "an empty existing file is not new"],
    [{ path: "s.js", before: "same\n", after: "same\n" }, { isNew: false, add: 0, del: 0, rows: [] }, "no change: empty rows, not null"],
    [{ path: "c.js", before: "a\nb\n", after: "" }, { lines: 0, add: 0, del: 2, rows: [["-", "a"], ["-", "b"]] }, "cleared file"],
    [{ path: "t.js", before: "a", after: "a\n" }, { lines: 1, add: 0, del: 0, rows: [] }, "only a trailing newline added"],
    [{ path: "b.js", before: "a\n", after: "a\n\n" }, { lines: 2, add: 1, del: 0 }, "a blank line added"],
  ];
  for (const [inp, want, why] of cases) {
    const d = makeDiff(inp);
    for (const [k, v] of Object.entries(want)) eq(d[k], v, `${why}: ${k}`);
    ok(!("error" in d), `${why}: no error key`);
    ok(!("head" in d) && !("full" in d), `${why}: rows path has no head/full`);
  }
});

Deno.test("makeDiff: paths are normalised; a bad path is kept as given", () => {
  eq(makeDiff({ path: "./src//a.js", before: "", after: "x" }).path, "src/a.js");
  eq(makeDiff({ path: "src\\win\\a.js", before: "", after: "x" }).path, "src/win/a.js");
  // normPath throws on "..": makeDiff must not throw (the tool reports the error itself)
  eq(makeDiff({ path: "../escape.js", before: "", after: "x" }).path, "../escape.js");
});

Deno.test("makeDiff: error is carried", () => {
  const d = makeDiff({ path: "a.js", before: "a", after: "a", error: "old_string not found" });
  eq(d.error, "old_string not found");
  eq(d.rows, []);
});

Deno.test("makeDiff: a one-line edit in a long file folds the unchanged runs", () => {
  const A = Array.from({ length: 2000 }, (_, i) => "line " + i);
  const B = A.slice(); B[1000] = "changed";
  const d = makeDiff({ path: "long.js", before: A.join("\n") + "\n", after: B.join("\n") + "\n" });
  ok(d.rows && d.rows.length < 20, "folded to a few rows");
  eq([d.add, d.del, d.lines], [1, 1, 2000]);
  const skips = d.rows.filter((r) => r.length === 3);
  ok(skips.length === 2 && skips.every(([op, text, n]) => op === " " && text === "" && n > 900), "two skip rows as [\" \", \"\", n]");
});

Deno.test("makeDiff: too long to diff -> head/tail/full, add = lines", () => {
  const long = Array.from({ length: 2000 }, (_, i) => "l" + i).join("\n") + "\n";
  const d = makeDiff({ path: "big.js", before: null, after: long });
  eq(d.rows, null);
  eq([d.isNew, d.lines, d.add, d.del], [true, 2000, 2000, 0]);
  eq(d.head.length, 50); eq(d.head[0], "l0"); eq(d.head[49], "l49");
  eq(d.tail.length, 50); eq(d.tail[0], "l1950"); eq(d.tail[49], "l1999", "the trailing empty line is dropped");
  eq(d.full, long);
});

Deno.test("makeDiff: fallback edges by length (tail only past 2*EDGE lines)", () => {
  // a big deletion makes lineDiff give up (> 400 rows) even when the new file is short
  const before = Array.from({ length: 1000 }, (_, i) => "old" + i).join("\n");
  for (const [n, head, tail] of [[0, 0, 0], [1, 1, 0], [50, 50, 0], [100, 50, 0], [101, 50, 50], [300, 50, 50]]) {
    const after = Array.from({ length: n }, (_, i) => "new" + i).join("\n");
    const d = makeDiff({ path: "x.js", before, after });
    eq(d.rows, null, `n=${n}: rows`);
    eq([d.head.length, d.tail.length, d.lines], [head, tail, n], `n=${n}: head/tail/lines`);
    if (tail) eq(d.tail[tail - 1], "new" + (n - 1), `n=${n}: tail ends at the last line`);
    ok(d.head.length + d.tail.length <= n, `n=${n}: head and tail never overlap`);
  }
});
