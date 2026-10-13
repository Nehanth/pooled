// room/working.js: the words of the "working" line shown before a model's first token.
import { verbs, liveText, LIVE } from "../../room/working.js";

Deno.test("verbs: one word at a time, never the same twice in a row, every verb before a repeat", () => {
  const next = verbs();
  const seen = [];
  for (let i = 0; i < 64; i++) seen.push(next());
  for (let i = 1; i < seen.length; i++) if (seen[i] === seen[i - 1]) throw new Error(`repeat at ${i}: ${seen[i]}`);
  for (const v of seen) if (!/^[A-Z][a-z]+ing$/.test(v)) throw new Error(`not a single -ing word: ${v}`);
  const first = new Set(seen.slice(0, 16));
  if (first.size !== 16) throw new Error(`the first 16 are not all different: ${[...first]}`);
});


Deno.test("liveText: the band's pill says a steady \"Writing…\"; the rotating words only when asked for", () => {
  for (let k = 0; k < 12; k++) if (liveText(k) !== "Writing\u2026") throw new Error(`k=${k}: ${liveText(k)}`);
  if (liveText(1, { rotate: true }) !== LIVE[1] + "\u2026") throw new Error(liveText(1, { rotate: true }));
});
