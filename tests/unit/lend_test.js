import { lendStatus, lendNotes } from "../../room/compute.js";
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const serving = { phase: "serving", lo: 0, hi: 20, model: "Qwen3.5 27B", bytes: 2.4 * 2 ** 30 };

Deno.test("lend: serving shows its layers and how much it holds", () => {
  eq(lendStatus(serving), { title: "Serving", sub: "Layers 1–20 · 2.4 GB · Qwen3.5 27B" });
});
Deno.test("lend: a small shard reads in MB, and no size before the load says one", () => {
  eq(lendStatus({ ...serving, bytes: 300 * 2 ** 20 }).sub, "Layers 1–20 · 300 MB · Qwen3.5 27B");
  eq(lendStatus({ ...serving, bytes: 0 }).sub, "Layers 1–20 · Qwen3.5 27B");
});
Deno.test("lend: loading, not holding layers, standing by", () => {
  eq(lendStatus({ ...serving, phase: "loading", pct: 41.6 }).title, "Loading 42%");
  eq(lendStatus({ phase: "loading", lo: null, hi: null, model: "M" }), { title: "Loading", sub: "M" });
  eq(lendStatus({ phase: "serving", lo: null, hi: null, model: "M" }), { title: "Not holding layers", sub: "The other devices run M" });
  eq(lendStatus({ phase: "idle", lo: null, hi: null }).title, "Standing by");
});
Deno.test("lend: an ended room wins over a stale Serving", () => {
  eq(lendStatus({ ...serving, over: { final: true, why: "" } }).title, "Room over");
  eq(lendStatus({ ...serving, over: { final: true, why: "the host closed the room" } }).sub, "the host closed the room");
  eq(lendStatus({ ...serving, over: { final: false, why: "" } }).title, "Host reconnecting");
});
Deno.test("lend: no notes when all is well, idle or the room is over", () => {
  eq(lendNotes({ ...serving, awake: "lock", battery: { charging: true, level: 0.5 } }), []);
  eq(lendNotes({ phase: "idle", awake: "none" }), []);
  eq(lendNotes({ ...serving, awake: "none", over: { final: true } }), []);
  eq(lendNotes({ phase: "serving", lo: null, hi: null, awake: "none" }), [], "no notes for a device that holds no layers");
});
Deno.test("lend: a screen that can sleep says which setting to change", () => {
  const [a] = lendNotes({ ...serving, awake: "none", ios: true }), [b] = lendNotes({ ...serving, awake: "none" });
  if (!/Auto-Lock/.test(a) || !/screen timeout/.test(b)) throw new Error(a + " / " + b);
  eq(lendNotes({ ...serving, awake: null }), [], "before the first try it says nothing");
});
Deno.test("lend: battery notes, low battery names battery saver", () => {
  const [low] = lendNotes({ ...serving, battery: { charging: false, level: 0.15 } });
  if (!/15%/.test(low) || !/Battery saver/.test(low)) throw new Error(low);
  const [off] = lendNotes({ ...serving, battery: { charging: false, level: 0.8 } });
  if (!/Not charging/.test(off)) throw new Error(off);
});
Deno.test("lend: a background tab is named after a real absence only", () => {
  eq(lendNotes({ ...serving, awayMs: 1200 }), []);
  const [n] = lendNotes({ ...serving, awayMs: 42300 });
  if (!/42 s/.test(n)) throw new Error(n);
});
