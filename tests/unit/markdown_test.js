// room/markdown.js md(): a code fence shows its text as written, without the language tag
import { md } from "../../room/markdown.js";

Deno.test("md keeps fenced code verbatim and drops the language tag", () => {
  const out = md("Here:\n```js\nconst a = 2 * 3;\n\n# x\n- y\nconst s = `x`;\n  b();\n```\n**done**");
  const want = "<p>Here:</p><pre>const a = 2 * 3;\n\n# x\n- y\nconst s = `x`;\n  b();</pre><p><b>done</b></p>";
  if (out !== want) throw new Error(out);
});

Deno.test("md still renders markdown outside fences", () => {
  const out = md("# Title\n- one\n- two\nsome `code` and *it*");
  if (out !== "<h2>Title</h2><ul><li>one</li><li>two</li></ul><p>some <code>code</code> and <i>it</i></p>") throw new Error(out);
});
