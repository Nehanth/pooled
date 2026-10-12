// room/speedo.js: the speed panel's numbers and words (no DOM, times passed in).
import { fmtTps, parseStats, SpeedMeter, pushHistory, speedup, gainText, barScale, paramsOf, paramsText, expertsAt, expertsText,
  expertsHead, expertsCost, tokenShares, pushEvent, onlineText, fmtGB } from "../../room/speedo.js";
import { MODELS } from "../../room/models.js";
import { expertCache } from "../../room/tele.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const near = (a, b, m, e = 1e-6) => { if (!(Math.abs(a - b) <= e)) throw new Error((m || "mismatch") + `: ${a} != ${b}`); };

Deno.test("fmtTps: a dash for none, one decimal under 100, whole numbers above", () => {
  for (const [v, s] of [[0, "–"], [-1, "–"], [NaN, "–"], [Infinity, "–"], [undefined, "–"], [0.04, "0.0"], [7.46, "7.5"], [23.45, "23.4"], [99.94, "99.9"], [100, "100"], [142.6, "143"]]) eq(fmtTps(v), s, String(v));
});

Deno.test("parseStats: the host's stats line (engine/generate.js), with and without the extras", () => {
  eq(parseStats("9 tok · 7.5 tok/s · 1 device · 42% drafts accepted"), { count: 9, tps: 7.5, devices: 1 });
  eq(parseStats("120 tok · 18.3 tok/s · 3 devices · 12 tok by lookup · stopped at 120 tokens"), { count: 120, tps: 18.3, devices: 3 });
  eq(parseStats("24 tok · 30 tok/s · 2 devices · carried on after a device dropped"), { count: 24, tps: 30, devices: 2 });
  eq(parseStats("40 tok · 11.0 tok/s · 2 devices · via API · not part of this chat's memory"), { count: 40, tps: 11, devices: 2 });
  eq(parseStats("0 tok · 0.0 tok/s · 1 device"), { count: 0, tps: 0, devices: 1 });
  for (const s of ["failed: the model is not loaded", "", null, undefined, "failed · via API", "answer hidden by the host"]) eq(parseStats(s), null, String(s));
});

Deno.test("SpeedMeter: whole-answer rate under the window, windowed rate after, average to the end", () => {
  const m = new SpeedMeter(2000);
  eq(m.rate(0), 0, "no tokens");
  m.start();
  m.tok(1000);
  eq(m.rate(1000), 0, "one token is no rate");
  m.tok(1100);   // 1 token in 100 ms after the first
  near(m.rate(1100), 10, "two tokens 100 ms apart");
  for (let t = 1200; t <= 2000; t += 100) m.tok(t);
  near(m.rate(2000), 10, "steady 10 tok/s");
  // then it speeds up to 50 tok/s for 3 s: the window forgets the slow start
  for (let t = 2020; t <= 5000; t += 20) m.tok(t);
  near(m.rate(5000), 50, "windowed", 0.5);
  const avg = m.end();
  near(avg, (m.n - 1) / 4, "average first to last");
  eq(m.on, false);
  // spec decoding lands several at once
  const b = new SpeedMeter(1000); b.start();
  for (let lap = 0; lap < 20; lap++) b.tok(lap * 100, 4);
  near(b.rate(1900), 40, "4 tokens a 100 ms lap", 3);
});

Deno.test("pushHistory: newest last, capped, one entry per answer key, none without a speed", () => {
  let h = [];
  h = pushHistory(h, { key: "1", tps: 5, devices: 1 });
  h = pushHistory(h, { key: "2", tps: 0, devices: 1 });
  h = pushHistory(h, null);
  eq(h.length, 1);
  h = pushHistory(h, { key: "1", tps: 6, devices: 1 });   // replayed (a reload): replaced, not doubled
  eq(h, [{ key: "1", tps: 6, devices: 1 }]);
  for (let i = 0; i < 30; i++) h = pushHistory(h, { key: "k" + i, tps: i + 1, devices: 2 }, 16);
  eq(h.length, 16); eq(h[15].key, "k29");
});

Deno.test("speedup: the newest run of answers against the run before, by device count", () => {
  const H = (...xs) => xs.map(([tps, devices], i) => ({ key: i, tps, devices }));
  eq(speedup(H([5, 1])), null);
  eq(speedup(H([5, 1], [6, 1])), null, "one run");
  const g = speedup(H([5, 1], [7, 1], [6, 1], [13, 2], [11, 2]));
  eq([g.from, g.to], [1, 2]); near(g.x, 12 / 6, "medians");
  // only the run right before counts (a third device after two)
  const g3 = speedup(H([5, 1], [12, 2], [12, 2], [18, 3]));
  eq([g3.from, g3.to], [2, 3]); near(g3.x, 1.5);
  eq(gainText(g), "2.0× faster than on 1 device");
  eq(gainText({ x: 12.4, from: 1, to: 2 }), "12× faster than on 1 device");
  eq(gainText({ x: 0.75, from: 2, to: 1 }), "25% slower than on 2 devices");
  eq(gainText({ x: 1.02, from: 2, to: 3 }), "about as fast as on 2 devices");
  eq(gainText(null), "");
});

Deno.test("barScale: heights against the fastest, a sliver at least", () => {
  eq(barScale([]), []);
  eq(barScale([{ tps: 10 }, { tps: 5 }, { tps: 0.1 }]), [1, 0.5, 0.06]);
});

Deno.test("paramsOf: the size from the picker label, the active count from a MoE's file name", () => {
  eq(paramsOf(MODELS["qwen3.8-27b"].label, MODELS["qwen3.8-27b"].gguf), { total: "27B", active: "", quant: "Q4" });
  eq(paramsOf(MODELS["qwen3.6-35b-moe"].label, MODELS["qwen3.6-35b-moe"].gguf), { total: "35B", active: "3B", quant: "Q4" });
  eq(paramsOf(MODELS["qwen3.5-122b-moe"].label, MODELS["qwen3.5-122b-moe"].gguf), { total: "122B", active: "10B", quant: "Q4" });
  eq(paramsOf(MODELS["qwen3-1.7b"].label, MODELS["qwen3-1.7b"].gguf), { total: "1.7B", active: "", quant: "Q8" });
  eq(paramsOf(MODELS["smollm-135m"].label, ""), { total: "135M", active: "", quant: "bf16" });
  eq(paramsOf("Qwen3.8 Flash", ""), { total: "", active: "", quant: "" }, "no size in the name: none made up");
  eq(paramsText(paramsOf(MODELS["qwen3.6-35b-moe"].label, MODELS["qwen3.6-35b-moe"].gguf)), "35B parameters · 3B active · Q4");
  eq(paramsText({ total: "27B", active: "", quant: "" }), "27B parameters");
  eq(paramsText({ total: "", active: "", quant: "Q4" }), "Q4");
  eq(paramsText(null), "");
});

Deno.test("expertsAt / expertsText: dense none, MoE on the GPU, or parked in RAM by the deal", () => {
  eq(expertsAt(false, null), null); eq(expertsText(null), "");
  eq(expertsText(expertsAt(true, null)), "experts on GPU");
  const e = expertsAt(true, { lo: 20, hi: 40, ramBytes: 12.5 * 2 ** 30, vramBytes: 3 * 2 ** 30 });
  eq(e.where, "GPU + RAM");
  eq(expertsText(e), "experts of layers 21–40 in RAM (12.5 GB), 3.0 GB GPU cache");
});

Deno.test("expertsHead / expertsCost: an offloading device's card says its GPU cache's hit rate and what a token costs", () => {
  const off = expertsAt(true, { lo: 20, hi: 40, ramBytes: 12.5 * 2 ** 30, vramBytes: 3 * 2 ** 30 });
  eq(expertsHead(null, null), ""); eq(expertsHead(expertsAt(true, null), null), "", "on the GPU: expertsText says it");
  eq(expertsHead(off, null), "experts: GPU + RAM", "no report yet");
  const xc = expertCache({ hits: 84, misses: 16, tok: 12, all: [84, 16] });
  eq(expertsHead(off, xc), "experts: GPU + RAM, 84% hits");
  eq(expertsCost(xc), "1.3 experts from RAM per token");
  eq(expertsCost(expertCache({ hits: 9, misses: 1, tok: 1, all: [9, 1] })), "1 expert from RAM per token");
  eq(expertsCost(expertCache({ hits: 0, misses: 24, tok: 2, all: [0, 24], reads: 24, readBytes: 24 * 3 * 2 ** 20 })), "12 experts from RAM per token · 12 disk reads per token (36 MB)");
  eq(expertsCost(null), "");
});

Deno.test("tokenShares: each device's part of the lap, the rest is the wire (and devices without timing)", () => {
  eq(tokenShares([{ name: "mac", ms: 30 }], 30), null, "one device");
  eq(tokenShares([{ name: "mac", ms: 30 }, { name: "pc", ms: 10 }], 0), null, "no lap yet");
  const s = tokenShares([{ name: "mac", ms: 30 }, { name: "pc", ms: 10 }], 50);
  eq(s.parts.map((p) => [p.name, p.pct]), [["mac", 60], ["pc", 20]]);
  eq([s.rest.ms, s.rest.pct, s.rest.label], [10, 20, "wire"]);
  // a worker that sends no ai-tele (an older version): its time is in the rest, and the rest says so
  const n = tokenShares([{ name: "mac", ms: 30 }, { name: "pc", ms: undefined }], 60);
  eq(n.parts.map((p) => [p.name, p.timed, p.pct]), [["mac", true, 50], ["pc", false, 0]]);
  eq([n.rest.ms, n.rest.label, n.untimed], [30, "wire + pc", ["pc"]]);
  // EMAs can add up past the lap: shares stay within 100%
  const o = tokenShares([{ name: "a", ms: 40 }, { name: "b", ms: 30 }], 60);
  near(o.parts.reduce((t, p) => t + p.pct, 0) + o.rest.pct, 100); eq(o.rest.ms, 0);
});

Deno.test("pushEvent: newest last, capped, a repeat is one event", () => {
  let ev = [];
  ev = pushEvent(ev, onlineText(1), 1);
  ev = pushEvent(ev, "pc joined", 2);
  ev = pushEvent(ev, "pc joined", 3);
  eq(ev, [{ text: "online · 1 device", at: 1 }, { text: "pc joined", at: 3 }]);
  ev = pushEvent(ev, "", 4);
  eq(ev.length, 2);
  for (const t of ["re-dealing layers", onlineText(2), "phone joined", "phone asks only"]) ev = pushEvent(ev, t, 5);
  eq(ev.map((e) => e.text), ["re-dealing layers", "online · 2 devices", "phone joined", "phone asks only"]);
});

Deno.test("fmtGB: one decimal under 10 GB, never 0 GB", () => {
  eq([fmtGB(0.01), fmtGB(0.43), fmtGB(7.25), fmtGB(12.6)], ["<0.1 GB", "0.4 GB", "7.3 GB", "13 GB"]);
});
