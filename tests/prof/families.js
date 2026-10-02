// Kernel families, built from the engine's own pipeline list (eng.pipes), for the profilers
// (benchmarks/bench_breakdown.js skips one family at a time; tests/prof/moe_decode_prof.js times each
// pipeline). The old profilers carried hand-written name lists that fell behind the engine: new kernels
// (attn_flash, dn_pre, the *_acc GEMVs, the prefill GEMM...) were in no list, so their time landed in
// "everything else". Here every pipeline the engine created belongs to exactly one family, and a name
// that matches no rule lands in "other", which tests/unit/prof_families_test.js fails on, so a new
// kernel gets a family when it is added.

// first match wins; order matters where prefixes overlap (moe_gsort before the expert kernels)
export const FAMILIES = [
  ["GEMV (projections, LM head)", /^matvec/],
  ["prefill GEMM", /^gemm_|^quant_q8_w$/],   // quant_q8_w: the dp4a GEMM's activation quantization
  ["MoE router", /^moe_(router|route|nrt|nrt_w)$/],   // nrt: the post-attention RMSNorm fused into the router GEMV
  ["MoE sort/combine", /^moe_(combine|combw|gsort)$/],
  ["MoE experts", /^moe_(gu|dn|qx$)/],   // moe_qx: the dp4a experts' activation quantization
  ["DeltaNet core", /^dn_/],
  ["attention", /^(attn_|kv_store|qsplit|head_norm|rope_part|sigmoid_mul)/],
  ["norms + residual", /^(rmsnorm|add_res)/],
  ["SiLU gate", /^silu_mul/],
  ["sampling (argmax, top-k)", /^(argmax|topk_)/],
  ["embedding gather", /^emb_gather$/],
];

export function pipeFamily(name) {
  for (const [fam, re] of FAMILIES) if (re.test(name)) return fam;
  return "other";
}

// the engine's pipeline names (every pipeline create() made, including the optional modules it merges in)
export const pipeNames = (eng) => Object.keys(eng.pipes || {}).sort();

// { family: [pipeline names] } over every pipeline of the engine; families with no pipeline are left out
export function familiesOf(names) {
  const out = {};
  for (const n of names) (out[pipeFamily(n)] ||= []).push(n);
  return out;
}

// per-pipeline rows for a kernel-mode profile: every pipeline the engine created, dispatched or not,
// plus any dispatched name the list does not have (flagged unlisted). kern: [{ name, ms }], units: the
// number of tokens or steps the kernels were recorded over.
export function pipeRows(names, kern, units = 1) {
  const r2 = (x) => Math.round(x * 100) / 100;
  const by = new Map(names.map((n) => [n, { n: 0, ms: 0 }]));
  const listed = new Set(names);
  for (const k of kern) { const o = by.get(k.name) || by.set(k.name, { n: 0, ms: 0 }).get(k.name); o.n++; o.ms += k.ms || 0; }
  return [...by].map(([pipe, o]) => ({ pipe, family: pipeFamily(pipe), ms: r2(o.ms / units), n: r2(o.n / units), usEach: o.n ? r2(1000 * o.ms / o.n) : 0,
    ...(listed.has(pipe) ? {} : { unlisted: true }) }))
    .sort((a, b) => b.ms - a.ms || b.n - a.n || (a.pipe < b.pipe ? -1 : 1));
}
