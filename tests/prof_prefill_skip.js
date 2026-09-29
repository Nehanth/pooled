// Where does prefill time go? Fresh prefills of LENS-token prompts (512 / 2048 / 8192 by default, each from an
// empty cache, like llama-bench pp), timed with whole kernel families skipped. A family's cost is
// (full time) - (time with that family skipped). Timestamp queries are slow on the GB10 (and a per-dispatch
// pass changes what is measured), so this uses the engine's skip hook instead. Outputs are wrong while
// skipping; only the timing matters. The MoE's expert kernels read the routing written by the (still running)
// router, so skipping other families can change routing a little; the router itself is its own family.
//   MODEL=27b|moe  LENS=512,2048,8192  RUNS=2 (full-speed runs per length)  SKIPS=1 (0: tok/s only)
//   PREFILL_UBATCH / ATTN_PREFILL_TILE / MOEGROUP as for bench_ctx.js (unset: engine defaults)
//   cd tests && MODEL=moe deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights prof_prefill_skip.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { makeTokenizer, argmax } from "../engine/engine.js";
import { qwen35Weights, tokenizerFromGGUF } from "../engine/gguf.js";
import { openGGUF, wideOpts, wideLimits } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "moe");
const PATH = MODEL === "27b" ? "../models/q38/model.gguf" : "../models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf";
const LENS = env("LENS", "512,2048,8192").split(",").map(Number), RUNS = +env("RUNS", 2), SKIPS = env("SKIPS", "1") !== "0";
const MAXSEQ = Math.max(...LENS) + 64;

const model = openGGUF(PATH);
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, ...wideLimits(adapter) } });
let gpuErrors = 0;
device.addEventListener?.("uncapturederror", (e) => { if (gpuErrors++ < 4) console.error("GPU ERROR:", e.error?.message?.slice(0, 200)); });
const G = model.G, m = G.meta, nBlk = m["qwen35.block_count"], L = nBlk - (m["qwen35.nextn_predict_layers"] || 0);
const hasMtp = Object.keys(G.tensors).some((k) => k.startsWith(`blk.${nBlk - 1}.`));
const tok = makeTokenizer(tokenizerFromGGUF(m));
const weights = await qwen35Weights(G, (i) => model.readAt(i.byteOffset, i.byteLength), { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: hasMtp });
const eng = await Qwen35Engine.create({ device, meta: m, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: MAXSEQ, batchCols: 16, coopRowsB: 1,
  moeGroupPrefill: env("MOEGROUP") === undefined ? undefined : +env("MOEGROUP"), ...wideOpts() });
console.log(`${MODEL}: layers ${L}, mtp ${!!eng.mtp}, gemmOn ${eng.gemmOn}, attnPrefillTile ${eng.attnPrefillTile}${eng.attnPTCfg ? ` (TK ${eng.attnPTCfg.TK})` : ""}, wide ${eng.ubatch ? `U=${eng.ubatch} ${JSON.stringify(eng.wideCfg)}` : "off"}, moeGroup ${eng.moeGrpU || "off"}`);

// same prompt recipe as bench_ctx.js (this repo's source)
let ids = [];
for (const f of ["../engine/qwen35.js", "../room.js", "../engine/gguf.js", "../engine/wgsl/base.js", "../engine/wgsl/moe.js", "../harness/agent.js", "../room/plan.js"]) {
  try { ids.push(...tok.encode(`\n// file: ${f}\n` + await Deno.readTextFile(new URL(f, import.meta.url)))); } catch {}
  if (ids.length >= MAXSEQ) break;
}
while (ids.length < MAXSEQ) ids = ids.concat(ids);

// ---- families: every dispatch goes through one of these engine methods ----
const opCat = new Map();
// built after the warm-up prefill: the batched (layerB) and wide (layerW) ops only exist after the first prefill
const catOps = () => {
  const put = (op, c) => { if (op && typeof op === "object" && (op.pipe || op.gemm) && !opCat.has(op)) opCat.set(op, c); };
  const cat = (key, full) => /router|^rs$|shRouter/i.test(key) ? "moe_router" : /^(gateUp|gu|down|gate|up|mvGate|mvUp|mvDown|shDown)$/.test(key) ? "ffn_proj"
    : full ? "attn_proj" : "dn_proj";
  const walk = (obj, full, depth = 0) => { for (const [k, v] of Object.entries(obj || {})) {
    if (Array.isArray(v)) v.forEach((o) => put(o, cat(k, full)));
    else if (v && typeof v === "object" && (v.pipe || v.gemm)) put(v, cat(k, full));
    else if (v && typeof v === "object" && depth < 1 && k === "mc") walk(v, full, depth + 1); } };
  eng.layers.forEach((Ly, i) => { walk(Ly, Ly.isFull); walk(eng.layerB?.[i], Ly.isFull); });
  // the MTP (draft) block: its batched layer sits after the model's layers in layerB; its projection is mtp.projB
  (eng.layerB || []).slice(eng.layers.length).forEach((LB) => { for (const v of Object.values(LB || {})) (Array.isArray(v) ? v : [v]).forEach((o) => put(o, "mtp_ops")); });
  if (eng.mtp) for (const v of Object.values(eng.mtp)) put(v, "mtp_ops");
};

let phase = null;   // which engine stage an unclaimed _dop op was issued from
for (const [fn, ph] of [["_encDnMid", "dn"], ["_encMoeFfn", "moe"], ["_mtpFillBatch", "mtp"], ["_encAttnGlue", "attn"], ["_encAttnCore", "attn"]]) {
  const f0 = eng[fn]; if (typeof f0 !== "function") continue;
  eng[fn] = function (...a) { const prev = phase; phase = ph; try { return f0.apply(this, a); } finally { phase = prev; } };
}
const unk = {};   // ops that no family claims, by pipeline (reported in the census)
const pipeCat = (n) => /^moe_(gs|gu|dn|comb)/.test(n) ? "moe_experts" : /^moe_route|^moe_/.test(n) ? "moe_router"
  : /^dn_/.test(n) ? "dn_core" : /^(attn|kv|flash|fa_|rope|qsplit|q_split|head_norm|sigmoid|ks_|qk)/.test(n) ? "attn_core"
  : /^(rmsnorm|add_res|silu|xpose|l2)/.test(n) ? "norms_glue" : "other:" + n;
const counts = {}; let counting = false, skip = new Set(), curOp = null;
const hit = (c) => { if (counting) counts[c] = (counts[c] || 0) + 1; return skip.has(c) || skip.has("ALL"); };
const o = { _d: eng._d, _d3: eng._d3, _dop: eng._dop, _dW: eng._dW, _dxyz: eng._dxyz, _dMC: eng._dMC, _dCol: eng._dCol, _dInd: eng._dInd };
eng._dop = function (pass, op, n) { const prev = curOp; curOp = opCat.get(op) || opCat.get(op?.base) || (phase ? "proj_other_" + phase : "proj_other"); if (counting && curOp.startsWith("proj_other")) unk[curOp + ":" + op?.pipe] = (unk[curOp + ":" + op?.pipe] || 0) + 1; try { return o._dop.call(this, pass, op, n); } finally { curOp = prev; } };
eng._d3 = function (pass, pipe, bg, wgs) { if (hit(curOp || pipeCat(pipe))) return; return o._d3.call(this, pass, pipe, bg, wgs); };
eng._dW = function (p, op, w) { if (hit(opCat.get(op) || wideCat.get(op) || "proj_other")) return; return o._dW.call(this, p, op, w); };
for (const k of ["_d", "_dxyz", "_dMC", "_dCol", "_dInd"]) eng[k] = function (pass, name, ...a) { if (hit(pipeCat(name))) return; return o[k].call(this, pass, name, ...a); };
// wide ops live in layerW: proj (attention or DeltaNet), out, gate / up / down
const wideCat = new Map();
const catWide = () => { if (!eng.layerW) return; eng.layerW.forEach((W, i) => { if (!W) return; const full = eng.layers[i].isFull;
  for (const op of W.proj || []) wideCat.set(op, full ? "attn_proj" : "dn_proj"); if (W.out) wideCat.set(W.out, full ? "attn_proj" : "dn_proj");
  for (const k of ["gate", "up", "down"]) if (W[k]) wideCat.set(W[k], "ffn_proj"); }); };
// silu_mul_w (wide dense FFN) is dispatched inline; the engine's own skip set covers it
const setSkip = (s) => { skip = new Set(s); eng.skip = skip.has("norms_glue") || skip.has("ALL") ? new Set(["silu_mul_w"]) : null; };
// encoder / queue counters
const q = device.queue, cnt = { submit: 0, write: 0, writeBytes: 0, copy: 0, enc: 0, map: 0 };
const oSub = q.submit.bind(q), oWr = q.writeBuffer.bind(q), oEnc = device.createCommandEncoder.bind(device);
q.submit = (b) => { cnt.submit++; return oSub(b); };
q.writeBuffer = (b, off, d, ...a) => { cnt.write++; cnt.writeBytes += d.byteLength ?? 0; return oWr(b, off, d, ...a); };
device.createCommandEncoder = (d) => { cnt.enc++; const e = oEnc(d); const oc = e.copyBufferToBuffer.bind(e); e.copyBufferToBuffer = (...a) => { cnt.copy++; return oc(...a); }; return e; };
const oMap = GPUBuffer.prototype.mapAsync; GPUBuffer.prototype.mapAsync = function (...a) { cnt.map++; return oMap.apply(this, a); };

// one fresh prefill of n tokens (n - 1 through prefillTokens, the last through forwardToken as bench_ctx does)
async function prefill(n, { mtp = true } = {}) {
  eng.reset(); if (eng.mtp) eng.mtpFill = mtp;
  await q.onSubmittedWorkDone();
  const t0 = performance.now();
  await eng.prefillTokens(ids.slice(0, n - 1));
  await q.onSubmittedWorkDone();
  const tPre = performance.now() - t0;
  const lg = await eng.forwardToken(ids[n - 1]);
  return { ms: performance.now() - t0, tPre, am: argmax(lg) };
}

console.log("warm-up (shader compiles)"); setSkip([]); await prefill(Math.min(600, Math.max(...LENS))); catOps(); catWide();
const out = { model: MODEL, lens: {} };
for (const n of LENS) {
  // dispatch census for this length
  for (const k in counts) delete counts[k]; for (const k in cnt) cnt[k] = 0; counting = true;
  setSkip([]); const f0 = await prefill(n); counting = false;
  const census = { unk: { ...unk },  ...cnt, dispatches: Object.values(counts).reduce((a, b) => a + b, 0), byFamily: { ...counts } };
  const full = [f0.ms];
  for (let r = 1; r < RUNS; r++) full.push((await prefill(n)).ms);
  const fullMs = Math.min(...full);
  const row = { tokps: full.map((ms) => +(n / (ms / 1000)).toFixed(1)), argmax: f0.am, census, fam: {} };
  console.log(`\n== ${MODEL} ${n} tokens: prefill ${row.tokps.join(" / ")} tok/s (${full.map((x) => (x / 1000).toFixed(2)).join(" / ")} s), argmax ${f0.am}`);
  console.log(`   census: ${census.dispatches} dispatches, ${cnt.submit} submits, ${cnt.enc} encoders, ${cnt.copy} copies, ${cnt.write} writeBuffer (${(cnt.writeBytes / 2 ** 20).toFixed(1)} MB), ${cnt.map} mapAsync`);
  if (Object.keys(unk).length) console.log("   proj_other pipes: " + JSON.stringify(unk));
  console.log("   by family: " + Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k}:${v}`).join("  "));
  if (SKIPS) {
    const fams = [...new Set([...Object.keys(counts)])].filter((k) => !k.startsWith("other:"));
    const t = async (s, opt) => { setSkip(s); const r = await prefill(n, opt); setSkip([]); return r.ms; };
    for (const f of fams) { const ms = await t([f]); row.fam[f] = +(fullMs - ms).toFixed(0); }
    const mtpOff = await t([], { mtp: false }); row.fam.mtp_fill = +(fullMs - mtpOff).toFixed(0);
    const none = await t(["ALL"], { mtp: false }); row.fam.fixed_all_skipped = +none.toFixed(0);
    const accounted = Object.entries(row.fam).filter(([k]) => k !== "fixed_all_skipped").reduce((a, [, v]) => a + v, 0) + none;
    row.fam.unaccounted = +(fullMs - accounted).toFixed(0);
    console.log(`   family ms (full ${fullMs.toFixed(0)} ms): ` + Object.entries(row.fam).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${v} (${(100 * v / fullMs).toFixed(1)}%)`).join("  "));
  }
  out.lens[n] = row;
}
console.log("RESULT " + JSON.stringify({ ...out, gpuErrors }));
if (gpuErrors) Deno.exit(1);
