// Long-context benchmark: prefill speed as the context fills, and decode speed (plain and
// speculative) with 1K .. 32K+ tokens already in the cache. Also checks that speculative decoding
// stays identical to plain at every fill and that no logit goes NaN. Prints bench-log rows at the end.
//   MODEL=moe|27b  FILLS=1024,8k,32k (default: the 1K / 8K / 32K preset, tests/ctx_plan.js)  TOKENS=32
//   CTX=<maxSeq> (default: the room default, raised to hold the largest fill; with an explicit CTX
//     and no FILLS, the preset fills that do not fit are skipped)
//   KV=f16|q8: KV cache format (q8 = engine kvQ8, the room's ?kv=q8; default f16)
//   HW="GB10" DATE="Sep 29": labels for the bench-log rows
//   Compare an f16 and a q8 run: save each output, then deno run --allow-read bench_ctx_compare.js f16.log q8.log
//   MOEGROUP=U (MoE: expert-grouped prefill in U-token ubatches, a multiple of 16; 0 = off; unset = engine default 256)  MOEGROUP_UC=8
//   PREFILL_UBATCH=256 [PREFILL_TILE=json]: wide prefill (engine option prefillUbatch; see load_model.js wideOpts)
//   cd tests && MODEL=moe deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights bench_ctx.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { makeTokenizer, argmax } from "../engine/engine.js";
import { qwen35Weights, tokenizerFromGGUF } from "../engine/gguf.js";
import { openGGUF, wideOpts, wideLimits } from "./load_model.js";
import { CTX } from "../room/models.js";
import { planCtx, parseFills, parseKV, kvBytesPerPos, benchLogTable } from "./ctx_plan.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "moe");
const PATH = MODEL === "27b" ? "../models/q38/model.gguf" : "../models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf";
const ROOM_KEY = MODEL === "27b" ? "qwen3.8-27b" : "qwen3.6-35b-moe";
const N = +env("TOKENS", 32), K = +env("K", 3);
const KV = parseKV(env("KV", "f16"));
// decode runs N tokens past each fill (and spec drafts past that): a fill too close to maxSeq writes past the KV cache
// and the plain / spec comparison is meaningless (27B, CTX 16384, FILLS=16384 "spec DIFFERS" on the M5 Max)
let plan;
try { plan = planCtx({ fills: env("FILLS") ? parseFills(env("FILLS")) : null, maxSeq: +env("CTX", 0), tokens: N, roomDefault: CTX[ROOM_KEY].def }); }
catch (e) { console.error(e.message); Deno.exit(2); }
const MAXSEQ = plan.maxSeq, FILLS = plan.fills;
if (plan.dropped.length) console.log(`skipping fills ${plan.dropped.join(" / ")}: they do not fit in CTX=${MAXSEQ}`);

const model = openGGUF(PATH);   // node:fs reads through the converted-weights cache (tests/weight_cache.js; WEIGHT_CACHE=0 disables)
const readAt = model.readAt;
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, ...wideLimits(adapter) } });
let gpuErrors = 0;
device.addEventListener?.("uncapturederror", (e) => { if (gpuErrors++ < 4) console.error("GPU ERROR:", e.error?.message?.slice(0, 200)); });

const G = model.G;
const m = G.meta, nBlk = m["qwen35.block_count"], L = nBlk - (m["qwen35.nextn_predict_layers"] || 0);
const hasMtp = Object.keys(G.tensors).some((k) => k.startsWith(`blk.${nBlk - 1}.`));
const kvPerPos = kvBytesPerPos(m, L, KV);
console.log(`${MODEL}: maxSeq ${MAXSEQ}, KV cache ${(kvPerPos * MAXSEQ / 2 ** 30).toFixed(2)} GB (${KV}), fills ${FILLS.join(" / ")}`);
const tok = makeTokenizer(tokenizerFromGGUF(m));
let t0 = performance.now();
const weights = await qwen35Weights(G, (i) => readAt(i.byteOffset, i.byteLength), { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: hasMtp });
const eng = await Qwen35Engine.create({ device, meta: m, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: MAXSEQ, batchCols: 16, coopRowsB: 1,
  moeGroupPrefill: env("MOEGROUP") === undefined ? undefined : +env("MOEGROUP"), moeGroupUC: +env("MOEGROUP_UC", 8), moeGroupTiled: env("MOEGROUP_TILED", "1") === "1", kvQ8: KV === "q8", ...wideOpts() });
// the engine only takes int8 KV with flash attention and a head size that is a multiple of 32
if ((KV === "q8") !== !!eng.kvQ8) { console.error(`KV=${KV} asked, engine kvQ8 ${!!eng.kvQ8}`); Deno.exit(2); }
console.log(`moeGroupPrefill ${eng.moeGrpU || "off"}${eng.moeGrpU ? ` UC ${eng.moeGrpUC} tiled ${!!eng.moeGrpTiled}` : ""}`);
console.log(`loaded in ${((performance.now() - t0) / 1000).toFixed(0)}s, mtp ${!!eng.mtp}, attnPrefillTile ${eng.attnPrefillTile}${eng.attnPTCfg ? ` (TK ${eng.attnPTCfg.TK}, ${eng.attnPTCfg.CW} columns per workgroup)` : ""}, wide prefill ${eng.ubatch ? `U=${eng.ubatch} tile ${JSON.stringify(eng.wideCfg)}` : "off"}`);

// a long, realistic coding context: this repo's own source, tokenized until there is enough
const need = Math.max(...FILLS) + 16;
let ids = [];
for (const f of ["../engine/qwen35.js", "../room.js", "../engine/gguf.js", "../engine/wgsl/base.js", "../engine/wgsl/moe.js", "../harness/agent.js", "../room/plan.js"]) {
  // CTX_SRC=<repo dir>: read these files from another checkout, so A/B runs across branches prefill the same tokens
  const u = env("CTX_SRC", "") ? new URL(f.replace(/^\.\.\//, ""), "file://" + env("CTX_SRC", "").replace(/\/?$/, "/")) : new URL(f, import.meta.url);
  try { ids.push(...tok.encode(`\n// file: ${f}\n` + await Deno.readTextFile(u))); } catch {}
  if (ids.length >= need) break;
}
while (ids.length < need) ids = ids.concat(ids);
ids = ids.slice(0, need);

const finite = (lg) => { for (let i = 0; i < lg.length; i += 97) if (!Number.isFinite(lg[i])) return false; return true; };
eng.reset(); if (eng.mtp) eng.mtpFill = true;
let pos = 0, fail = 0;
const rows = [];
for (const fill of FILLS) {
  // prefill from the current position up to this fill (the last token goes through forwardToken)
  const chunk = ids.slice(pos, fill - 1);
  t0 = performance.now();
  if (chunk.length) await eng.prefillTokens(chunk);
  let logits = await eng.forwardToken(ids[fill - 1]);
  const pfS = (performance.now() - t0) / 1000, pfTok = chunk.length + 1;
  pos = fill;
  // plain decode from here, then the same start again with speculative decoding
  let saved = false;
  try { eng.saveSlot("fill"); saved = true; } catch {}
  let next = argmax(logits); const plain = [next]; let ok = finite(logits);
  t0 = performance.now();
  for (let i = 1; i < N; i++) { logits = await eng.forwardToken(next); ok = ok && finite(logits); next = argmax(logits); plain.push(next); }
  const plainTs = (N - 1) / ((performance.now() - t0) / 1000);
  let specTs = null, same = null, acc = "";
  if (eng.mtp && saved) {
    eng.loadSlot("fill"); eng.mtp.stats = { drafts: 0, accepted: 0 };
    next = plain[0]; const spec = [next];
    t0 = performance.now();
    while (spec.length < N) { for (const t of await eng.specStep(next, argmax, K)) spec.push(t); next = spec[spec.length - 1]; }
    specTs = (spec.length - 1) / ((performance.now() - t0) / 1000);
    same = plain.every((t, i) => spec[i] === t); acc = `${eng.mtp.stats.accepted}/${eng.mtp.stats.drafts}`;
    if (!same) fail++;
  }
  if (!ok) fail++;
  // continue filling from the prompt, not from the generated tokens
  if (saved) { eng.loadSlot("fill"); try { eng.dropSlot("fill"); } catch {} } else pos = eng.pos;
  const r = { fill, prefillTokPerS: +(pfTok / pfS).toFixed(1), prefillS: +pfS.toFixed(1), plainTokPerS: +plainTs.toFixed(2), specTokPerS: specTs && +specTs.toFixed(2), acceptance: acc, specIdentical: same, finite: ok, plainIds: plain };
  rows.push(r);
  console.log(`fill ${String(fill).padStart(6)}: prefill ${r.prefillTokPerS} tok/s (${pfTok} tok in ${r.prefillS}s) · plain ${r.plainTokPerS} tok/s · spec ${r.specTokPerS ?? "-"} tok/s ${acc} ${same === null ? "" : same ? "identical" : "DIFFERS"} ${ok ? "" : "NaN!"}`);
  console.log("  text: " + JSON.stringify(tok.decode(plain).slice(0, 80)));
}
const result = { model: MODEL, kv: KV, maxSeq: MAXSEQ, rows, gpuErrors };
console.log("RESULT " + JSON.stringify(result));
console.log("\nbench-log rows:\n" + benchLogTable(result, { date: env("DATE", new Date().toDateString().slice(4, 10)), hardware: env("HW", "") }));
console.log(fail || gpuErrors ? "CTX FAIL" : "CTX PASS");
if (fail || gpuErrors) Deno.exit(1);
