// harness/projects.js slugify: a scratch project's OPFS folder name. Importing the module touches
// no browser API (IndexedDB and OPFS are only reached inside the functions).
import { slugify, PROJECTS_DIR, canOpenFolder } from "../../harness/projects.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

Deno.test("slugify: table", () => {
  const cases = [
    ["Tetris", "tetris"],
    ["Snake Game", "snake-game"],
    ["  --Hello, World!--  ", "hello-world", "trim punctuation at both ends"],
    ["a___b...c", "a-b-c", "runs of separators collapse"],
    ["Café", "cafe", "accents are stripped (NFKD)"],
    ["Ünïcödé", "unicode"],
    ["crème brûlée", "creme-brulee"],
    ["ﬁle", "file", "ligature decomposes (NFKD)"],
    ["①②", "12", "circled digits decompose (NFKD)"],
    ["Ｔｅｔｒｉｓ", "tetris", "full-width letters decompose (NFKD)"],
    ["2048", "2048"],
    ["", "project", "empty"],
    [null, "project", "null"],
    [undefined, "project", "undefined"],
    ["中文项目", "project", "no latin letters"],
    ["🚀🚀", "project", "emoji only"],
    ["---", "project", "separators only"],
    [0, "project", "0 is falsy"],
    [42, "42", "numbers are stringified"],
  ];
  for (const [inp, want, why] of cases) eq(slugify(inp), want, why || String(inp));
});

Deno.test("slugify: letters with no decomposition become separators", () => {
  // ø, ß, æ, ł have no NFKD decomposition to ASCII: they split the word (documenting today's behaviour)
  eq(slugify("Prøject"), "pr-ject");
  eq(slugify("straße"), "stra-e");
});

Deno.test("slugify: at most 40 chars and never ends in a hyphen after the cut", () => {
  eq(slugify("a".repeat(100)), "a".repeat(40));
  eq(slugify("a".repeat(39) + " b"), "a".repeat(39), "the cut lands on the hyphen, which is dropped");
  eq(slugify("a".repeat(38) + "   bcd"), "a".repeat(38) + "-b", "separator runs collapse before the cut");
  eq(slugify("a".repeat(39) + "!!! bcd"), "a".repeat(39), "a collapsed run lands on the cut and is dropped");
  eq(slugify("a".repeat(40) + "b"), "a".repeat(40));
});

Deno.test("slugify: output is always a safe folder name (sweep)", () => {
  const samples = ["", " ", "x", "X Y Z", "../../etc/passwd", "con", ".git", "a/b\\c", "tab\there", "new\nline", "\u0000nul",
    "  leading", "trailing  ", "ÀÉÎÕÜ", "mixed 中文 and latin", "a".repeat(1000), "-".repeat(50) + "x", "x" + "-".repeat(50) + "y"];
  for (const s of samples) {
    const g = slugify(s);
    ok(/^[a-z0-9]+(-[a-z0-9]+)*$/.test(g), `${JSON.stringify(s)} -> ${JSON.stringify(g)}`);
    ok(g.length >= 1 && g.length <= 40, `${JSON.stringify(s)} length ${g.length}`);
    eq(slugify(g), g, "idempotent");
  }
  eq(slugify("../../etc/passwd"), "etc-passwd", "no path traversal");
});

Deno.test("projects: constants and folder support off outside a browser", () => {
  eq(PROJECTS_DIR, "pooled-projects");
  eq(canOpenFolder(), false);
});
