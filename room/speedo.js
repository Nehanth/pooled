// The room's speed panel: how fast the room writes, answer by answer, and what each device does for it
// (renderSpeed in room.js draws it). Everything here is pure (no DOM, no clock of its own: callers pass
// performance.now()), so tests/unit/speedo_test.js covers it.
//
// The numbers come from what the room already measures: each screen times the tokens it is shown (the
// live readout); the finished answer's speed is the host's own ("24 tok · 18.3 tok/s · 2 devices", the
// stats line under every answer, engine/generate.js), so every screen's history agrees with the host's;
// per-device time is the host's lap timing (ai.lapStat) and the workers' ai-tele reports.

// tok/s as the readout prints it: one decimal under 100, none above; a dash for no speed yet
export function fmtTps(v) {
  if (!(v > 0) || !Number.isFinite(v)) return "–";
  return v >= 100 ? String(Math.round(v)) : v.toFixed(1);
}

// The speed in words, for the panel (the tok/s number stays in its tooltip): about 0.75 words a token in English text.
// wordsPerSec(tps) -> words a second; speedWords(tps) -> "about 3 words a second" ("" without a speed)
export const WORDS_PER_TOKEN = 0.75;
export const wordsPerSec = (tps) => (+tps > 0 && Number.isFinite(+tps) ? +tps * WORDS_PER_TOKEN : 0);
export function speedWords(tps) {
  const w = wordsPerSec(tps);
  if (!(w > 0)) return "";
  if (w < 0.75) return "less than a word a second";
  const n = Math.max(1, Math.round(w));
  return `about ${n} word${n === 1 ? "" : "s"} a second`;
}
// the readout's line: live "Writing about 3 words a second"; after an answer "Last answer: about 3 words a second on
// 2 devices"; "" without a speed (the panel shows nothing rather than a dash)
export function speedLine(tps, { live = false, devices = 0 } = {}) {
  const w = speedWords(tps);
  if (!w) return "";
  if (live) return `Writing ${w}`;
  return `Last answer: ${w}` + (devices > 0 ? ` on ${devices} device${devices === 1 ? "" : "s"}` : "");
}

// The band head's one calm line, from the last real answer ({ tps, devices }, two tokens or more): "Last answer: about
// 3 words a second" (" on 2 devices" with more than one); "" before there is one. Never a live number: while an answer
// is written the band's pill says "Reading…" / "Writing…".
export function bandLine(last) {
  if (!last || !(+last.tps > 0)) return "";
  const n = +last.devices || 0;
  return speedLine(+last.tps, { devices: n > 1 ? n : 0 });
}

// an answer's stats line -> { count, tps, devices }; null for one without a speed ("failed: ...", a note)
export function parseStats(stats) {
  const s = String(stats || "");
  const m = /(?:^|·\s*)(\d+) tok · (\d+(?:\.\d+)?) tok\/s(?: · (\d+) devices?)?/.exec(s);
  if (!m) return null;
  return { count: +m[1], tps: +m[2], devices: m[3] ? +m[3] : 1 };
}

// The live rate: tokens over the last `win` ms (spec decoding lands several tokens per lap, so a window,
// not the gap between two tokens), or over the whole answer while it is shorter than the window.
export class SpeedMeter {
  constructor(win = 2000) { this.win = win; this.reset(); }
  reset() { this.ts = []; this.first = 0; this.last = 0; this.n = 0; this.on = false; }
  start() { this.reset(); this.on = true; }
  tok(now, k = 1) {
    if (!this.on) this.start();
    if (!this.n) this.first = now;
    this.n += k; this.last = now;
    for (let i = 0; i < k; i++) this.ts.push(now);
    while (this.ts.length > 2 && this.ts[0] < now - this.win) this.ts.shift();
  }
  // tokens per second now; 0 until there are two tokens a few ms apart
  rate(now = this.last) {
    if (this.n < 2) return 0;
    const span = now - this.first;
    if (span < this.win) return span > 50 ? (this.n - 1) / (span / 1000) : 0;
    const recent = this.ts.filter((t) => t >= now - this.win);
    // the first token in the window opens it: the rate is the ones after it over the time since it
    return recent.length > 1 ? (recent.length - 1) / ((now - recent[0]) / 1000) : 0;
  }
  // the whole answer's average (first token to last)
  average() { const span = this.last - this.first; return this.n > 1 && span > 0 ? (this.n - 1) / (span / 1000) : 0; }
  end() { this.on = false; return this.average(); }
}

// the history of answers, newest last, at most `max`: { key, tps, devices }; an answer shown again (a
// reloaded tab replays the conversation) keeps its one entry
export function pushHistory(hist, entry, max = 16) {
  if (!entry || !(entry.tps > 0)) return hist;
  const out = entry.key != null ? hist.filter((h) => h.key !== entry.key) : hist.slice();
  out.push(entry);
  return out.slice(-max);
}

const median = (a) => { const s = [...a].sort((x, y) => x - y), k = s.length >> 1; return s.length % 2 ? s[k] : (s[k - 1] + s[k]) / 2; };
// The change since the device count changed: the median of the newest run of answers with the same count
// against the run before it. { x, from, to } (x = new / old) or null with no earlier run.
export function speedup(hist) {
  if (hist.length < 2) return null;
  const cur = hist[hist.length - 1].devices;
  let i = hist.length - 1;
  while (i >= 0 && hist[i].devices === cur) i--;
  if (i < 0) return null;
  const prev = hist[i].devices;
  let j = i;
  while (j >= 0 && hist[j].devices === prev) j--;
  const a = median(hist.slice(j + 1, i + 1).map((h) => h.tps)), b = median(hist.slice(i + 1).map((h) => h.tps));
  return a > 0 ? { x: b / a, from: prev, to: cur } : null;
}
export function gainText(g) {
  if (!g || !(g.x > 0)) return "";
  const on = `than on ${g.from} device${g.from === 1 ? "" : "s"}`;
  if (Math.abs(g.x - 1) < 0.05) return `about as fast as on ${g.from} device${g.from === 1 ? "" : "s"}`;
  return g.x > 1 ? `${g.x.toFixed(g.x >= 10 ? 0 : 1)}× faster ${on}` : `${Math.round((1 - g.x) * 100)}% slower ${on}`;
}

// bar heights for the history, 0..1 of the fastest answer shown (at least a sliver for the slow ones)
export function barScale(hist, min = 0.06) {
  const top = Math.max(0, ...hist.map((h) => h.tps));
  return hist.map((h) => (top > 0 ? Math.max(min, h.tps / top) : min));
}

// A model's size from its picker label and file name: "Qwen3.8 27B · Q4" -> 27B; a MoE's file says how many
// are active per token ("Qwen3.6-35B-A3B" -> 3B). { total, active, quant }; total "" when the label has none.
// A model whose label and file name carry no size (Qwen3.8 Flash Next) has it in SIZES, by its room/models.js key.
const SIZES = { "qwen3.8-flash": { total: "125B", active: "6B" } };   // room/models.js: 512 experts, 10 active, ~6B of 125B
export function paramsOf(label = "", url = "", key = "") {
  const parts = String(label).split("·").map((s) => s.trim());
  const m = /(?:^|\s)(\d+(?:\.\d+)?)\s?([BMT])\b/.exec(parts[0] || "");
  const a = /-A(\d+(?:\.\d+)?)B(?:\b|-|_)/i.exec(String(url).split("/").pop() || "");
  const k = SIZES[key];
  return { total: m ? m[1] + m[2] : k?.total || "", active: a ? a[1] + "B" : k?.active || "", quant: parts[1] || "" };
}
export function paramsText(p) {
  if (!p?.total) return p?.quant || "";
  return `${p.total} parameters` + (p.active ? ` · ${p.active} active` : "") + (p.quant ? ` · ${p.quant}` : "");
}

// Where a device's experts live, from what the host dealt (ai.offloadBy) for a MoE model: null for a dense
// model; { where: "GPU" } when they sit with the layers; { where: "GPU + RAM", lo, hi, ramGB, cacheGB } when it
// parks the experts of layers lo..hi-1 in system RAM with a GPU cache in front (room/plan.js dealRoom); with the
// disk tier (dealRoom's diskLo: the experts of layers diskLo..hi-1 stay in the model's files) { where: "GPU + RAM +
// disk", ..., diskLo }. The map carries it to the other screens, so a missing or odd field draws less, never NaN.
const fin = (v) => Number.isFinite(+v) && v !== null && v !== "";
export function expertsAt(moe, off) {
  if (!moe) return null;
  if (!off || typeof off !== "object") return { where: "GPU" };
  const disk = fin(off.diskLo) && fin(off.hi) && +off.diskLo < +off.hi;
  return { where: disk ? "GPU + RAM + disk" : "GPU + RAM", lo: off.lo, hi: off.hi, ramGB: (+off.ramBytes || 0) / 2 ** 30, cacheGB: (+off.vramBytes || 0) / 2 ** 30,
    ...(disk ? { diskLo: +off.diskLo } : {}) };
}
export function expertsText(e) {
  if (!e || typeof e !== "object") return "";
  if (e.where === "GPU") return "experts on GPU";
  const gb = (v) => (Number.isFinite(+v) ? (+v).toFixed(1) : "?");
  if (!fin(e.lo) || !fin(e.hi)) return `experts in ${e.where === "GPU + RAM + disk" ? "RAM and on disk" : "RAM"}, ${gb(e.cacheGB)} GB GPU cache`;
  const lo = +e.lo, hi = +e.hi, dl = fin(e.diskLo) && +e.diskLo > lo && +e.diskLo < hi ? +e.diskLo : fin(e.diskLo) && +e.diskLo <= lo ? lo : null;
  if (dl === null) return `experts of layers ${lo + 1}–${hi} in RAM (${gb(e.ramGB)} GB), ${gb(e.cacheGB)} GB GPU cache`;
  if (dl === lo) return `experts of layers ${lo + 1}–${hi} on disk, ${gb(e.cacheGB)} GB GPU cache`;
  return `experts of layers ${lo + 1}–${dl} in RAM (${gb(e.ramGB)} GB), ${dl + 1}–${hi} on disk, ${gb(e.cacheGB)} GB GPU cache`;
}
// The card's headline for a device that offloads: "experts: GPU + RAM, 84% hits", the hit rate of its GPU expert
// cache from its ai-tele (room/tele.js expertCache: the last report with lookups); without one yet, the place only.
// "" for a device that doesn't offload (expertsText says "experts on GPU" there).
export function expertsHead(e, xc) {
  if (!e || typeof e !== "object" || e.where === "GPU" || typeof e.where !== "string") return "";
  return `experts: ${e.where}` + (xc && Number.isFinite(+xc.hit) ? `, ${Math.round(+xc.hit)}% hits` : "");
}
// what each token costs the cache: "1.3 experts copied in per token" (a miss: from the parked experts, the disk
// tier's RAM cache or the file), and the disk reads when the store reads experts from disk; "" without a report
const per = (v) => (v >= 10 ? Math.round(v) : Math.round(v * 10) / 10);
export function expertsCost(xc) {
  if (!xc || typeof xc !== "object") return "";
  const bits = [], ok = (v) => v != null && Number.isFinite(+v) && +v >= 0;
  if (ok(xc.missPerTok)) bits.push(`${per(+xc.missPerTok)} expert${per(+xc.missPerTok) === 1 ? "" : "s"} copied in per token`);
  if (ok(xc.readsPerTok)) bits.push(`${per(+xc.readsPerTok)} disk read${per(+xc.readsPerTok) === 1 ? "" : "s"} per token` + (+xc.readMBPerTok > 0 ? ` (${per(+xc.readMBPerTok)} MB)` : ""));
  return bits.join(" · ");
}

// Each device's share of one token's time: its own compute per lap (nodes[i].ms: the host's from its lap
// timing, a worker's from its ai-tele: browser workers and room nodes both report it, room/tele.js) over the lap;
// what is left is the wire, plus any device that reports no timing (a worker on an older version). null without a lap or with one device (it does all of every token).
export function tokenShares(nodes, lap) {
  lap = +lap;
  if (!(lap > 0) || !Number.isFinite(lap) || !Array.isArray(nodes) || nodes.length < 2) return null;
  // ms may come from another screen's map or an older worker: a number or nothing (never a string's concatenation)
  const parts = nodes.map((x) => { const ms = +x?.ms; return { name: String(x?.name ?? ""), ms: ms > 0 && Number.isFinite(ms) ? ms : 0, timed: ms > 0 && Number.isFinite(ms) }; });
  const sum = parts.reduce((s, x) => s + x.ms, 0);
  const total = Math.max(lap, sum);   // EMAs of different lengths: the parts may add up past the lap
  for (const x of parts) x.pct = 100 * x.ms / total;
  const rest = Math.max(0, total - sum), untimed = parts.filter((x) => !x.timed).map((x) => x.name);
  return { lap: total, parts, rest: { ms: rest, pct: 100 * rest / total, label: untimed.length ? `wire + ${untimed.join(", ")}` : "wire" }, untimed };
}

// the event line: newest last, at most `max`; the same text twice in a row is one event (its time updated)
export function pushEvent(list, text, now, max = 4) {
  if (!text) return list;
  const out = list.slice();
  if (out.length && out[out.length - 1].text === text) out[out.length - 1] = { text, at: now };
  else out.push({ text, at: now });
  return out.slice(-max);
}
export const onlineText = (n) => `Running on ${n} device${n === 1 ? "" : "s"}`;

// GB as the cards print it (a tiny share, as the tests' synthetic model has, is not "0 GB")
export const fmtGB = (gb) => (gb < 0.05 ? "<0.1" : gb >= 10 ? Math.round(gb) : Math.round(gb * 10) / 10) + " GB";
