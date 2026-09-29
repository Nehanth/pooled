// tests/ctx_plan.js: the pure part of the long-context bench (tests/bench_ctx.js, issue #71):
// which fills to time, the cache size, KV bytes in f16 / int8, the bench-log rows and the
// f16 vs int8 comparison. No GPU.
import { CTX_PRESET, decodeHeadroom, parseFills, parseKV, planCtx, kvBytesPerPos, fillLabel, benchLogTable, compareRuns } from "../ctx_plan.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const throws = (f, re, m) => { try { f(); } catch (e) { if (re && !re.test(e.message)) throw new Error(`${m}: wrong error "${e.message}"`); return; } throw new Error(`${m}: did not throw`); };

Deno.test("ctx preset: 1K / 8K / 32K", () => {
  eq(CTX_PRESET, [1024, 8192, 32768]);
  eq(decodeHeadroom(32), 80);
  eq(decodeHeadroom(0), 16);
});

Deno.test("parseFills: counts and k suffixes", () => {
  eq(parseFills("1024,8k,32K"), [1024, 8192, 32768]);
  eq(parseFills(" 1024 , 4096 "), [1024, 4096], "spaces");
  eq(parseFills("2000,32000"), [2000, 32000], "plain counts");
  eq(parseFills("1024,"), [1024], "trailing comma");
  for (const bad of ["", ",", "abc", "1.5k", "-1", "0", "1", "8m", "1024,1024", "8k,1k"]) throws(() => parseFills(bad), /fill|fills/, `"${bad}"`);
});

Deno.test("parseKV: f16 by default, q8 when asked, a typo is an error", () => {
  for (const v of [undefined, null, "", "f16"]) eq(parseKV(v), "f16", String(v));
  eq(parseKV("q8"), "q8");
  for (const v of ["Q8", "int8", "q4", "f32"]) throws(() => parseKV(v), /bad KV/, v);
});

Deno.test("planCtx: preset with no CTX fits 32K plus headroom, never below the room default", () => {
  eq(planCtx({ tokens: 32 }), { fills: [1024, 8192, 32768], maxSeq: 33024, dropped: [] });
  eq(planCtx({ tokens: 32, roomDefault: 65536 }).maxSeq, 65536, "MoE room default is larger");
  eq(planCtx({ tokens: 32, roomDefault: 16384 }).maxSeq, 33024, "27B: raised to hold 32K");
});

Deno.test("planCtx: preset with a small CTX drops the fills that do not fit", () => {
  eq(planCtx({ maxSeq: 16384, tokens: 32 }), { fills: [1024, 8192], maxSeq: 16384, dropped: [32768] });
  eq(planCtx({ maxSeq: 32768, tokens: 32 }), { fills: [1024, 8192], maxSeq: 32768, dropped: [32768] }, "32768 exactly leaves no headroom");
  eq(planCtx({ maxSeq: 32768 + 80, tokens: 32 }).dropped, [], "just enough");
  throws(() => planCtx({ maxSeq: 1024, tokens: 32 }), /no fill fits/, "nothing fits");
});

Deno.test("planCtx: explicit fills", () => {
  eq(planCtx({ fills: [1024, 4096], tokens: 32 }), { fills: [1024, 4096], maxSeq: 4352, dropped: [] }, "rounded up to 256");
  eq(planCtx({ fills: [1024], tokens: 32, roomDefault: 16384 }).maxSeq, 16384);
  eq(planCtx({ fills: [1024, 4096], maxSeq: 8192, tokens: 32 }).fills, [1024, 4096]);
  throws(() => planCtx({ fills: [1024, 16384], maxSeq: 16384, tokens: 32 }), /raise CTX/, "explicit fill too big is an error, not a skip");
  const f = [1024]; planCtx({ fills: f, tokens: 32 }); eq(f, [1024], "input not mutated");
});

Deno.test("planCtx: every fill always has its decode headroom (sweep)", () => {
  for (const tokens of [1, 8, 32, 64]) for (let maxSeq = 2048; maxSeq <= 70000; maxSeq += 1531) {
    let p; try { p = planCtx({ maxSeq, tokens }); } catch { ok(maxSeq < 1024 + decodeHeadroom(tokens)); continue; }
    for (const f of p.fills) ok(f + decodeHeadroom(tokens) <= p.maxSeq, `tokens ${tokens} maxSeq ${maxSeq} fill ${f}`);
    eq([...p.fills, ...p.dropped].sort((a, b) => a - b), CTX_PRESET, "fills + dropped = preset");
  }
});

Deno.test("kvBytesPerPos: whole trunk, f16 and int8", () => {
  const meta27 = { "qwen35.attention.head_count_kv": 4, "qwen35.attention.key_length": 256, "qwen35.full_attention_interval": 4 };
  const metaMoe = { "qwen35.attention.head_count_kv": 2, "qwen35.attention.key_length": 256, "qwen35.full_attention_interval": 4 };
  // 27B: 64 layers, 16 attention layers x 4 KB = 64 KB per position (f16); int8 = 16 x 2304
  eq(kvBytesPerPos(meta27, 64), 65536);
  eq(kvBytesPerPos(meta27, 64, "q8"), 36864, "the changelog's 36 KB per token");
  eq(kvBytesPerPos(metaMoe, 40), 20480, "MoE: 10 attention layers x 2 KB");
  eq(kvBytesPerPos(metaMoe, 40, "q8"), 11520);
  eq(kvBytesPerPos(meta27, 63), 65536, "a partial group still has its attention layer");
  eq(kvBytesPerPos(meta27, 64) * 32768 / 2 ** 30, 2, "27B at 32K: 2 GB f16");
});

Deno.test("fillLabel", () => {
  eq([1024, 8192, 32768, 65536, 1000, 4000].map(fillLabel), ["1K", "8K", "32K", "64K", "1000", "4000"]);
});

const run = (kv, rows) => ({ model: "27b", kv, maxSeq: 33024, gpuErrors: 0, rows });
const row = (fill, pf, plain, ids, extra = {}) => ({ fill, prefillTokPerS: pf, prefillS: 1, plainTokPerS: plain, specTokPerS: null, acceptance: "", specIdentical: null, finite: true, plainIds: ids, ...extra });

Deno.test("benchLogTable: one markdown row per fill", () => {
  const t = benchLogTable(run("q8", [
    row(1024, 120.5, 9.1, [1], { specTokPerS: 15.2, acceptance: "85%", specIdentical: true }),
    row(32768, 80, 6.5, [1], { specIdentical: false, specTokPerS: 10, acceptance: "70%", finite: false }),
  ]), { date: "Sep 29", hardware: "GB10" });
  const lines = t.split("\n");
  eq(lines.length, 4);
  ok(lines[0].startsWith("| Date | Model |") && lines[1].startsWith("|---|"));
  eq(lines[2], "| Sep 29 | 27b | GB10 (Deno) | q8 | 1K | 120.5 | 9.1 | 15.2 (85%) | yes |");
  eq(lines[3], "| Sep 29 | 27b | GB10 (Deno) | q8 | 32K | 80 | 6.5 NaN! | 10 (70%) | **no** |");
  for (const l of lines) eq(l.split("|").length, lines[0].split("|").length, "same column count");
  const noKv = benchLogTable({ model: "moe", rows: [row(1024, 1, 1, [])] }, { runtime: "" });
  ok(noKv.split("\n")[2].includes("|  | f16 | 1K | 1 | 1 | - | - |"), "old RESULT without kv is f16, no spec is '-': " + noKv);
});

Deno.test("compareRuns: tokens that agree and speed ratios per fill", () => {
  const a = run("f16", [row(1024, 100, 10, [1, 2, 3, 4]), row(8192, 90, 9, [5, 6, 7]), row(32768, 50, 5, [1])]);
  const b = run("q8", [row(1024, 110, 9, [1, 2, 3, 4]), row(8192, 90, 12, [5, 9, 7])]);
  eq(compareRuns(a, b), [
    { fill: 1024, compared: 4, sameTokens: 4, identical: true, prefill: 1.1, plain: 0.9 },
    { fill: 8192, compared: 3, sameTokens: 1, identical: false, prefill: 1, plain: 1.333 },
  ], "32K only in a: skipped");
  const c = compareRuns(run("f16", [row(1024, 0, 10, [])]), run("q8", [row(1024, 5, 10, [1, 2])]));
  eq(c[0], { fill: 1024, compared: 0, sameTokens: 0, identical: false, prefill: null, plain: 1 }, "no tokens / zero speed");
  throws(() => compareRuns(a, { ...b, model: "moe" }), /different models/, "model mismatch");
});
