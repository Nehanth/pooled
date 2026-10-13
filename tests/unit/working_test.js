// room/working.js: the words of the "working" line shown before a model's first token.
import { verbs, liveText, LIVE, waitText } from "../../room/working.js";

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

Deno.test("waitText: the chat's steady wait says what happens, and after 4 s on the first question, once, that it takes longer", () => {
  const lead = "Reading your message", then = { after: 4000, text: "The first answer takes a little longer" };
  const eqs = (a, b, m) => { if (a !== b) throw new Error(`${m}: ${a} != ${b}`); };
  eqs(waitText(0, { lead, then }), lead, "t=0");
  eqs(waitText(3999, { lead, then }), lead, "t=3999");
  eqs(waitText(4000, { lead, then }), then.text, "t=4000");
  eqs(waitText(60000, { lead, then }), then.text, "one swap only: it stays");
  for (const t of [0, 4000, 10000, 600000]) eqs(waitText(t, { lead }), lead, `no then, t=${t}`);
  eqs(waitText(9000, { label: "Working" }), "Working", "no lead: the label");
  const next = verbs(), all = new Set();
  for (let i = 0; i < 16; i++) all.add(next());
  for (const t of [0, 1999, 2000, 4000, 8000, 30000]) for (const o of [{ lead, then }, { lead }, {}]) {
    const w = waitText(t, o);
    if (all.has(w)) throw new Error(`a verb in steady mode at t=${t}: ${w}`);
  }
});
