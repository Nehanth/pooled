// Pure helpers for tests/bench_ctx.js (no GPU): which context fills to time, how big the KV cache
// must be for them, what it costs in memory in f16 or int8, the bench-log rows a run prints, and
// the comparison of an f16 run with a KV=q8 run (issue #71: can int8 KV be the default for long
// documents?). Unit tests: tests/unit/ctx_plan_test.js.
import { kvBytesPerLayerPos } from "../room/models.js";

// the 1K / 8K / 32K contexts the bench log reports (roadmap/30-long-context-and-sessions.md)
export const CTX_PRESET = [1024, 8192, 32768];

// Decode runs `tokens` plain tokens past each fill, and speculative drafts reach a few past that:
// a fill closer than this to maxSeq writes past the KV cache.
export const decodeHeadroom = (tokens) => 2 * tokens + 16;

// "1024,8k,32K" -> [1024, 8192, 32768]. Throws on anything that is not a positive whole count.
export function parseFills(str) {
  const out = String(str).split(",").map((s) => s.trim()).filter(Boolean).map((s) => {
    const m = /^(\d+)([kK]?)$/.exec(s);
    const n = m ? +m[1] * (m[2] ? 1024 : 1) : NaN;
    if (!(n >= 2)) throw new Error(`bad fill "${s}": want a token count like 1024 or 8k`);
    return n;
  });
  if (!out.length) throw new Error("no fills");
  for (let i = 1; i < out.length; i++) if (out[i] <= out[i - 1]) throw new Error(`fills must increase: ${out.join(",")}`);
  return out;
}

// KV format for the bench: KV=q8 -> "q8", unset / "f16" -> "f16"; anything else is a typo.
export function parseKV(v) {
  if (v === undefined || v === null || v === "" || v === "f16") return "f16";
  if (v === "q8") return "q8";
  throw new Error(`bad KV "${v}": want f16 or q8`);
}

// Which fills to time and the cache size. With no FILLS the 1K / 8K / 32K preset; with no CTX the
// smallest multiple of 256 that holds the largest fill plus its decode headroom, and never less than
// the room default (so the 1K / 8K rows time the cache size a room really uses). An explicit CTX
// that is too small for an explicit fill is an error; with the preset, fills that do not fit are
// dropped instead (CTX=16384 times 1K and 8K only).
export function planCtx({ fills, maxSeq, tokens = 32, roomDefault = 0 } = {}) {
  const head = decodeHeadroom(tokens);
  const preset = !fills;
  let F = preset ? CTX_PRESET.slice() : fills.slice();
  if (!maxSeq) {
    const need = Math.ceil((Math.max(...F) + head) / 256) * 256;
    return { fills: F, maxSeq: Math.max(need, roomDefault), dropped: [] };
  }
  const fit = (f) => f + head <= maxSeq;
  const dropped = F.filter((f) => !fit(f));
  if (dropped.length && !preset) throw new Error(`fill ${dropped[0]} + ${head} decode positions exceeds maxSeq ${maxSeq}: raise CTX`);
  F = F.filter(fit);
  if (!F.length) throw new Error(`no fill fits in maxSeq ${maxSeq}`);
  return { fills: F, maxSeq, dropped };
}

// K+V bytes per position for the whole trunk (the attention layers only), f16 or int8.
export function kvBytesPerPos(meta, layers, kv = "f16") {
  const every = meta["qwen35.full_attention_interval"] || 1;
  return kvBytesPerLayerPos(meta, kv) * every * Math.ceil(layers / every);
}

// "1K" / "8K" / "32K" for the preset, the plain count otherwise.
export const fillLabel = (n) => (n % 1024 === 0 ? `${n / 1024}K` : String(n));

// Markdown rows for docs/bench-log.md from a bench_ctx RESULT object.
export function benchLogTable(res, { date = "", hardware = "", runtime = "Deno" } = {}) {
  const head = "| Date | Model | Hardware | KV | Context | Prefill tok/s | Plain decode tok/s | Spec decode tok/s | Spec = plain |";
  const lines = [head, "|---|---|---|---|---|---|---|---|---|"];
  for (const r of res.rows) {
    const spec = r.specTokPerS == null ? "-" : `${r.specTokPerS} (${r.acceptance})`;
    const same = r.specIdentical == null ? "-" : r.specIdentical ? "yes" : "**no**";
    lines.push(`| ${date} | ${res.model} | ${hardware}${runtime ? ` (${runtime})` : ""} | ${res.kv || "f16"} | ${fillLabel(r.fill)} | ${r.prefillTokPerS} | ${r.plainTokPerS}${r.finite === false ? " NaN!" : ""} | ${spec} | ${same} |`);
  }
  return lines.join("\n");
}

// Compare two runs of the same model and fills (typically f16 vs KV=q8): per fill, how many of
// the greedy tokens agree before the first difference, and the speed ratio b / a.
export function compareRuns(a, b) {
  if (a.model !== b.model) throw new Error(`different models: ${a.model} vs ${b.model}`);
  const byFill = new Map(b.rows.map((r) => [r.fill, r]));
  return a.rows.filter((r) => byFill.has(r.fill)).map((ra) => {
    const rb = byFill.get(ra.fill);
    const x = ra.plainIds || [], y = rb.plainIds || [];
    const n = Math.min(x.length, y.length);
    let same = 0; while (same < n && x[same] === y[same]) same++;
    const ratio = (p, q) => (p > 0 && q > 0 ? +(q / p).toFixed(3) : null);
    return { fill: ra.fill, compared: n, sameTokens: same, identical: n > 0 && same === n,
      prefill: ratio(ra.prefillTokPerS, rb.prefillTokPerS), plain: ratio(ra.plainTokPerS, rb.plainTokPerS) };
  });
}
