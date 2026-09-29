// Wide prefill GEMM on Chrome's subgroup matrices (chromium-experimental-subgroup-matrix: Vulkan
// cooperative matrices / Metal simdgroup_matrix / D3D12 wave matrices). A drop-in for the tiled
// f32 GEMM in gemm_wide.js: same weight buffers (the engine's repacked Q4_0 / Q8_0 and paired f16
// scales), same output layout y[col * ys + row] (= or += with _acc), grid (dOut / BM, width / BN).
// Opt-in (engine option prefillMma), feature-detected; anything missing leaves the f32 wide GEMM.
//
// Two families, picked from GPUAdapterInfo.subgroupMatrixConfigs:
//
// "i8" (i8 x i8 -> i32, K = 32: NVIDIA / AMD / Intel on Vulkan and D3D12 list these): an MMQ-style
//   kernel, the scheme llama.cpp's CUDA backend runs for Q4_0 prefill.
//   - wmma_quant: activations -> i8 per (column, 32-block) with an f32 scale, d = amax / 127,
//     q = round-half-away(x / d) (llama.cpp's quantize_q8_1, f32 d instead of f16).
//   - Weights: Q4_0 nibbles unpack exactly to i8 (q - 8), Q8_0 is already i8; staged per K stage in
//     workgroup memory, one K = 32 MMA per quant block.
//   - A block's i32 result cannot be scaled inside a subgroup matrix (scales differ per row and per
//     column), so each block's 16 x 16 i32 tiles go through workgroup memory and every lane applies
//     w_scale * x_scale into f32 accumulators it owns (lane = row, so the final stores coalesce).
//   Numerics: activation quantization (rel ~4e-3 per GEMM on random data): prefill tolerance, not bits.
//
// "f32" (f32 x f32 -> f32, 8 x 8 x 8: Apple): weights dequantized with their scale into an f32 tile
//   in workgroup memory, activations loaded straight from the column-strided x, one accumulator
//   matrix per 8 x 8 output tile for the whole K range. f32 products and sums (the wide GEMM's
//   numerics up to summation order).
//
// Workgroup: NS = (BM / SM) * (BN / SN) subgroups, each owning an SM x SN output tile; the workgroup
// size is NS * SG with SG = the adapter's subgroupMaxSize (the extension requires a multiple of it).
// If the hardware runs smaller subgroups the tile math is wrong, so the engine self-tests the kernels.

export const WMMA_FEATURE = "chromium-experimental-subgroup-matrix";
export const WMMA_FEATURES_OPT = ["subgroups"];
export const WMMA_DEFAULT = Object.freeze({ i8: { BM: 64, BN: 64, SM: 32, SN: 32, KB: 1, PAD: 4 }, f32: { BM: 64, BN: 64, SM: 32, SN: 32, KB: 1, PAD: 4 } });
// "template": subgroupMatrixLoad<T, row_major>(p, off, stride) (current Dawn, the gpuweb proposal);
// "bool": subgroupMatrixLoad<T>(p, off, colMajor, stride) (Chromium <= 153 at least). Both are tried.
// Type parameter order: Chrome 153 (GB10) takes the proposal's subgroup_matrix_left<T, M, K>,
// right<T, K, N>, result<T, M, N> ("template"); older Dawn spelled them <T, cols, rows> ("template-km",
// "bool"). All square shapes (Apple 8x8x8) read the same either way. Tried in this order.
export const WMMA_SYNTAX = ["template", "template-km", "bool"];
const mtypes = (syntax, T, TR, M, N, K) => syntax === "template"
  ? [`subgroup_matrix_left<${T}, ${M}, ${K}>`, `subgroup_matrix_right<${T}, ${K}, ${N}>`, `subgroup_matrix_result<${TR}, ${M}, ${N}>`]
  : [`subgroup_matrix_left<${T}, ${K}, ${M}>`, `subgroup_matrix_right<${T}, ${N}, ${K}>`, `subgroup_matrix_result<${TR}, ${N}, ${M}>`];

const rng = (n) => Array.from({ length: n }, (_, i) => i);

// cfgs: GPUAdapterInfo.subgroupMatrixConfigs. prefer: "f32" | "i8" | undefined (f32 first: closer numerics).
export function pickWmmaConfig(info, prefer) {
  const cs = [...(info?.subgroupMatrixConfigs || [])];
  if (!cs.length) return { why: "adapter lists no subgroupMatrixConfigs" };
  const SG = info?.subgroupMaxSize || 0;
  if (!SG) return { why: "adapter reports no subgroupMaxSize" };
  const i8 = cs.find((c) => c.componentType === "i8" && c.resultComponentType === "i32" && c.K === 32 && c.M === 16 && c.N === 16)
    || cs.find((c) => c.componentType === "i8" && c.resultComponentType === "i32" && c.K === 32 && c.M === 16 && c.N === 8);
  const f32 = cs.find((c) => c.componentType === "f32" && c.resultComponentType === "f32" && c.M === 8 && c.N === 8 && c.K === 8);
  const order = prefer === "i8" ? [["i8", i8], ["f32", f32]] : [["f32", f32], ["i8", i8]];
  for (const [kind, c] of order) if (c) return { kind, M: c.M, N: c.N, K: c.K, SG };
  return { why: `no usable config in ${JSON.stringify(cs.map((c) => `${c.componentType}->${c.resultComponentType} ${c.M}x${c.N}x${c.K}`))}` };
}

// Resolve tiles: throws on shapes the generator cannot emit or that overflow wgMem / 1024 threads.
export function wmmaPlan(mc, opt = {}, wgMem = 16384) {
  const t = { ...WMMA_DEFAULT[mc.kind], ...(opt || {}) };
  const { BM, BN, SM, SN, KB, PAD } = t;
  for (const [k, v] of Object.entries({ BM, BN, SM, SN, KB })) if (!Number.isInteger(v) || v <= 0) throw new Error(`prefillMma.${k} must be a positive integer`);
  if (!Number.isInteger(PAD) || PAD < 0 || PAD % 4) throw new Error("prefillMma.PAD must be a multiple of 4");
  if (BM % SM || BN % SN) throw new Error("prefillMma: SM must divide BM and SN divide BN");
  if (SM % mc.M || SN % mc.N) throw new Error(`prefillMma: SM / SN must be multiples of the ${mc.M}x${mc.N} MMA`);
  if (mc.kind === "i8" && SM % mc.SG) throw new Error(`prefillMma: SM must be a multiple of the subgroup size ${mc.SG} (one row per lane)`);
  if (BN % 16) throw new Error("prefillMma: BN must be a multiple of 16");
  const NS = (BM / SM) * (BN / SN), T = NS * mc.SG;
  if (T > 1024) throw new Error(`prefillMma: ${T} threads per workgroup`);
  let smem;
  if (mc.kind === "i8") { t.WSTR = KB * 8 + PAD; smem = 4 * (BM * t.WSTR + BM * KB + BN * KB + NS * KB * SM * SN); }
  else { t.WSTR4 = KB * 8 + PAD / 4; smem = 16 * BM * t.WSTR4; }
  if (smem > wgMem) throw new Error(`prefillMma: ${smem} B of workgroup memory (limit ${wgMem})`);
  return Object.freeze({ ...t, ...mc, NS, T, smem, KS: 32 * KB });
}

const ld = (syntax, ty, arr, off, colMajor, stride) => syntax !== "bool"
  ? `subgroupMatrixLoad<${ty}, ${colMajor ? "col_major" : "row_major"}>(&${arr}, ${off}, ${stride})`
  : `subgroupMatrixLoad<${ty}>(&${arr}, ${off}, ${colMajor}, ${stride})`;
const st = (syntax, arr, off, v, colMajor, stride) => syntax !== "bool"
  ? `subgroupMatrixStore<${colMajor ? "col_major" : "row_major"}>(&${arr}, ${off}, ${v}, ${stride})`
  : `subgroupMatrixStore(&${arr}, ${off}, ${v}, ${colMajor}, ${stride})`;

// ---- i8 family ----
function i8Kernel(fmt, acc, P, syntax) {
  const { BM, BN, SM, SN, KB, WSTR, NS, T, SG, M, N } = P, WC = BN / SN, TI = SM / M, TJ = SN / N, RA = SM / SG;
  const units = BM * KB, uPer = Math.ceil(units / T), xsU = BN * KB, xsPer = Math.ceil(xsU / T);
  const [L, R, C] = mtypes(syntax, "i8", "i32", M, N, 32);
  const nib = (w, sh) => `((((${sh ? `(${w} >> 4u)` : w} & 0x0F0F0F0Fu) | 0x80808080u) - 0x08080808u) ^ 0x80808080u)`;
  const wstage = rng(uPer).map((j) => `
    { let li = t + ${j * T}u; if (li < ${units}u) {
      let r = li % ${BM}u; let b = li / ${BM}u; let g = min(row0 + r, dOut - 1u) * nb + kb + b;
      ${fmt === "q4"
        ? `let wv = gm_q[g];
      ${rng(4).map((jj) => `gm_W[r * ${WSTR}u + b * 8u + ${jj}u] = ${nib(`wv[${jj}]`, false)}; gm_W[r * ${WSTR}u + b * 8u + ${4 + jj}u] = ${nib(`wv[${jj}]`, true)};`).join("\n      ")}`
        : `let w0 = gm_q[g * 2u]; let w1 = gm_q[g * 2u + 1u];
      ${rng(8).map((jj) => `gm_W[r * ${WSTR}u + b * 8u + ${jj}u] = w${jj >> 2}[${jj & 3}];`).join(" ")}`}
      gm_Ws[r * ${KB}u + b] = unpack2x16float(gm_sc[g >> 1u])[g & 1u];
    } }`).join("");
  const xstage = rng(xsPer).map((j) => `
    { let li = t + ${j * T}u; if (li < ${xsU}u) { let c = li % ${BN}u; let b = li / ${BN}u; gm_Xs[c * ${KB}u + b] = bitcast<f32>(gm_xq[xdo + (col0 + c) * nb + kb + b]); } }`).join("");
  const mma = rng(KB).map((b) => `
    {
      ${rng(TI).map((ti) => `let l${ti} = ${ld(syntax, L, "gm_W", `(wr * ${SM}u + ${ti * M}u) * ${WSTR}u + ${b * 8}u`, false, `${WSTR}u`)};`).join("\n      ")}
      ${rng(TJ).map((tj) => `let r${tj} = ${ld(syntax, R, "gm_xq", `(cs + ${tj * N}u) * nb8 + (kb + ${b}u) * 8u`, true, "nb8")};`).join("\n      ")}
      ${rng(TI).map((ti) => rng(TJ).map((tj) => st(syntax, "gm_C", `cb + ${b * SM * SN + tj * N * SM + ti * M}u`, `subgroupMatrixMultiply<${C}>(l${ti}, r${tj})`, true, `${SM}u`) + ";").join("\n      ")).join("\n      ")}
    }`).join("");
  const accs = rng(RA).map((a) => rng(SN).map((c) => `f${a}_${c}`));
  const epi = rng(KB).map((b) => rng(RA).map((a) => `
    { let r = ${a * SG}u + lane; let ws = gm_Ws[(wr * ${SM}u + r) * ${KB}u + ${b}u]; let o = cb + ${b * SM * SN}u + r;
      ${rng(SN).map((c) => `${accs[a][c]} += f32(gm_C[o + ${c * SM}u]) * (ws * gm_Xs[(wc * ${SN}u + ${c}u) * ${KB}u + ${b}u]);`).join("\n      ")}
    }`).join("")).join("");
  const store = rng(RA).map((a) => `
  { let row = rs + ${a * SG}u + lane; if (row < dOut) {
    ${rng(SN).map((c) => `gm_y[(cs + ${c}u) * ys + row] ${acc ? "+=" : "="} ${accs[a][c]};`).join(" ")}
  } }`).join("");
  return `
@compute @workgroup_size(${T})
fn gemm_m_${fmt}${acc ? "_acc" : ""}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let S = gm_s; let t = lid.x; let lane = t % ${SG}u; let sgi = t / ${SG}u; let wr = sgi / ${WC}u; let wc = sgi % ${WC}u;
  let dOut = S.dOut; let nb = S.dIn / 32u; let nb8 = nb * 8u; let xdo = S.xs4; let ys = S.ys;
  let row0 = wg.x * ${BM}u; let col0 = wg.y * ${BN}u; let rs = row0 + wr * ${SM}u; let cs = col0 + wc * ${SN}u;
  let cb = sgi * ${KB * SM * SN}u;
  ${accs.flat().map((a) => `var ${a} = 0.0;`).join(" ")}
  for (var kb: u32 = 0u; kb < nb; kb += ${KB}u) {
    workgroupBarrier();${wstage}${xstage}
    workgroupBarrier();${mma}
    workgroupBarrier();${epi}
  }${store}
}`;
}

function i8Module(P, syntax) {
  const { BM, BN, SM, SN, KB, WSTR, NS } = P;
  return `
@group(1) @binding(0) var<storage, read> gm_q: array<vec4<u32>>;
@group(1) @binding(1) var<storage, read> gm_sc: array<u32>;
@group(1) @binding(2) var<storage, read> gm_xq: array<u32>;
@group(1) @binding(3) var<storage, read_write> gm_y: array<f32>;
@group(1) @binding(4) var<uniform> gm_s: BShape;
var<workgroup> gm_W: array<u32, ${BM * WSTR}>;
var<workgroup> gm_Ws: array<f32, ${BM * KB}>;
var<workgroup> gm_Xs: array<f32, ${BN * KB}>;
var<workgroup> gm_C: array<i32, ${NS * KB * SM * SN}>;
${["q4", "q8"].flatMap((f) => [false, true].map((a) => i8Kernel(f, a, P, syntax))).join("\n")}

// activations -> i8 per (column, 32-block): xq[(c * nb + b) * 8 + j] (4 i8 each), d at xq[S.ys + c * nb + b]
// (S = { dOut: columns, dIn, xs4: column stride of x in vec4, ys: offset of the scales in xq })
@group(1) @binding(0) var<storage, read> gq_x: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> gq_o: array<u32>;
@group(1) @binding(2) var<uniform> gq_s: BShape;
@compute @workgroup_size(64)
fn wmma_quant(@builtin(global_invocation_id) gid: vec3<u32>) {
  let S = gq_s; let nb = S.dIn / 32u; let b = gid.x; let c = gid.y;
  if (b >= nb || c >= S.dOut) { return; }
  var v: array<vec4<f32>, 8>; var m = 0.0;
  for (var j = 0u; j < 8u; j++) { let a = gq_x[c * S.xs4 + b * 8u + j]; v[j] = a; let q = abs(a); m = max(m, max(max(q.x, q.y), max(q.z, q.w))); }
  let d = m / 127.0; let id = select(0.0, 1.0 / d, d > 0.0);
  for (var j = 0u; j < 8u; j++) { let s = v[j] * id; gq_o[(c * nb + b) * 8u + j] = pack4xI8(vec4<i32>(sign(s) * floor(abs(s) + vec4<f32>(0.5)))); }
  gq_o[S.ys + c * nb + b] = bitcast<u32>(d);
}`;
}

// ---- f32 family ----
function f32Kernel(fmt, acc, P, syntax) {
  const { BM, BN, SM, SN, KB, WSTR4, T, M, N, K } = P, WC = BN / SN, TI = SM / M, TJ = SN / N, KS = 32 * KB;
  const [L, R, C] = mtypes(syntax, "f32", "f32", M, N, K);
  const units = BM * KB, uPer = Math.ceil(units / T);
  const q4lo = (w) => `(vec4<f32>(unpack4xU8(${w} & 0x0F0F0F0Fu)) - vec4<f32>(8.0))`, q4hi = (w) => `(vec4<f32>(unpack4xU8((${w} >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0))`;
  const wstage = rng(uPer).map((j) => `
    { let li = t + ${j * T}u; if (li < ${units}u) {
      let r = li % ${BM}u; let b = li / ${BM}u; let g = min(row0 + r, dOut - 1u) * nb + kb + b; let s = unpack2x16float(gm_sc[g >> 1u])[g & 1u]; let o = r * ${WSTR4}u + b * 8u;
      ${fmt === "q4"
        ? `let wv = gm_q[g];
      ${rng(4).map((jj) => `gm_Wf[o + ${jj}u] = ${q4lo(`wv[${jj}]`)} * s; gm_Wf[o + ${4 + jj}u] = ${q4hi(`wv[${jj}]`)} * s;`).join("\n      ")}`
        : `let w0 = gm_q[g * 2u]; let w1 = gm_q[g * 2u + 1u];
      ${rng(8).map((jj) => `gm_Wf[o + ${jj}u] = vec4<f32>(unpack4xI8(w${jj >> 2}[${jj & 3}])) * s;`).join(" ")}`}
    } }`).join("");
  const cs = rng(TI).map((ti) => rng(TJ).map((tj) => `c${ti}_${tj}`));
  const steps = rng(KS / K).map((kk) => `
    {
      ${rng(TI).map((ti) => `let l${ti} = ${ld(syntax, L, "gm_Wf", `(wr * ${SM}u + ${ti * M}u) * ${WSTR4}u + ${kk * K / 4}u`, false, `${WSTR4}u`)};`).join("\n      ")}
      ${rng(TJ).map((tj) => `let r${tj} = ${ld(syntax, R, "gm_x", `(cs + ${tj * N}u) * xs4 + kb * 8u + ${kk * K / 4}u`, true, "xs4")};`).join("\n      ")}
      ${rng(TI).map((ti) => rng(TJ).map((tj) => `${cs[ti][tj]} = subgroupMatrixMultiplyAccumulate(l${ti}, r${tj}, ${cs[ti][tj]});`).join(" ")).join("\n      ")}
    }`).join("");
  const init = rng(TI).map((ti) => rng(TJ).map((tj) => acc
    ? `var ${cs[ti][tj]} = ${ld(syntax, C, "gm_y", `(cs + ${tj * N}u) * ys + min(rs + ${ti * M}u, dOut - ${M}u)`, true, "ys")};`
    : `var ${cs[ti][tj]}: ${C};`).join(" ")).join("\n  ");
  const store = rng(TI).map((ti) => `
  if (rs + ${ti * M}u < dOut) { ${rng(TJ).map((tj) => st(syntax, "gm_y", `(cs + ${tj * N}u) * ys + rs + ${ti * M}u`, cs[ti][tj], true, "ys") + ";").join(" ")} }`).join("");
  return `
@compute @workgroup_size(${T})
fn gemm_m_${fmt}${acc ? "_acc" : ""}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let S = gm_s; let t = lid.x; let sgi = t / ${P.SG}u; let wr = sgi / ${WC}u; let wc = sgi % ${WC}u;
  let dOut = S.dOut; let nb = S.dIn / 32u; let xs4 = S.xs4; let ys = S.ys;
  let row0 = wg.x * ${BM}u; let col0 = wg.y * ${BN}u; let rs = row0 + wr * ${SM}u; let cs = col0 + wc * ${SN}u;
  ${init}
  for (var kb: u32 = 0u; kb < nb; kb += ${KB}u) {
    workgroupBarrier();${wstage}
    workgroupBarrier();${steps}
  }${store}
}`;
}

function f32Module(P, syntax) {
  return `
@group(1) @binding(0) var<storage, read> gm_q: array<vec4<u32>>;
@group(1) @binding(1) var<storage, read> gm_sc: array<u32>;
@group(1) @binding(2) var<storage, read> gm_x: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> gm_y: array<f32>;
@group(1) @binding(4) var<uniform> gm_s: BShape;
var<workgroup> gm_Wf: array<vec4<f32>, ${P.BM * P.WSTR4}>;
${["q4", "q8"].flatMap((f) => [false, true].map((a) => f32Kernel(f, a, P, syntax))).join("\n")}`;
}

// A standalone shader module (own enable directives). P: wmmaPlan(...). Entry points:
// gemm_m_q4, gemm_m_q4_acc, gemm_m_q8, gemm_m_q8_acc (+ wmma_quant for "i8").
export function gemmWmmaWGSL(P, syntax = "template") {
  return `enable chromium_experimental_subgroup_matrix;
diagnostic(off, chromium.subgroup_matrix_uniformity);
// ---- subgroup-matrix wide prefill GEMM (${P.kind} ${P.M}x${P.N}x${P.K}, BM=${P.BM} BN=${P.BN} SM=${P.SM} SN=${P.SN} KB=${P.KB}; ${P.T} threads, ${P.smem} B workgroup memory) ----
struct BShape { dOut: u32, dIn: u32, xs4: u32, ys: u32 };
${P.kind === "i8" ? i8Module(P, syntax) : f32Module(P, syntax)}
`;
}

// Size in u32 of the i8 activation buffer for `cols` columns of width dIn: [i8 data | f32 scales].
export const wmmaXqWords = (cols, dIn) => cols * dIn / 4 + cols * dIn / 32;

// CPU reference of the i8 family's arithmetic (float64 accumulation of the same quantized operands):
// w: Float32Array dOut x dIn of dequantization-free integer weights (q - 8 or i8), ws: per-(row, block)
// scales, x: cols x dIn activations. Returns y[col * dOut + row].
export function wmmaI8Ref({ wq, ws, x, dOut, dIn, cols }) {
  const nb = dIn / 32, y = new Float64Array(cols * dOut), xq = new Int8Array(cols * dIn), xd = new Float32Array(cols * nb);
  for (let c = 0; c < cols; c++) for (let b = 0; b < nb; b++) {
    let m = 0; for (let k = 0; k < 32; k++) m = Math.max(m, Math.abs(x[c * dIn + b * 32 + k]));
    const d = Math.fround(m / 127), id = d > 0 ? Math.fround(1 / d) : 0; xd[c * nb + b] = d;
    for (let k = 0; k < 32; k++) { const s = Math.fround(x[c * dIn + b * 32 + k] * id); xq[c * dIn + b * 32 + k] = Math.sign(s) * Math.floor(Math.abs(s) + 0.5); }
  }
  for (let c = 0; c < cols; c++) for (let r = 0; r < dOut; r++) {
    let a = 0;
    for (let b = 0; b < nb; b++) { let s = 0; for (let k = 0; k < 32; k++) s += wq[r * dIn + b * 32 + k] * xq[c * dIn + b * 32 + k]; a += s * ws[r * nb + b] * xd[c * nb + b]; }
    y[c * dOut + r] = a;
  }
  return y;
}
