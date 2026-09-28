// Generated WGSL: cooperative GEMV, batched/twin variants, fused gate/up, accumulate variants.
// See docs/kernels.md for the rules these kernels follow.
//
// One workgroup of WG threads computes ROWS output rows together (the shape
// llama.cpp's WebGPU backend, web-llm's generated kernels and zero-tvm all
// converge on for decode GEMV). Thread t = (block-lane bl = t/4) x (quarter
// qt = t%4, an 8-element slice of a 32-element quant block): consecutive
// threads read consecutive words of the same row (coalesced) and each
// thread's activation slice is loaded once and reused across all ROWS rows.
// Scalar accumulators only: a dynamically-indexed local array spills to
// scratch memory and ran 3x slower. Reduction is a portable shared-memory
// halving tree (no subgroups: absent from shipping Safari 26).

export async function probeUnpack(device) {
  if (device.__unpackOk !== undefined) return device.__unpackOk;
  device.pushErrorScope("validation");
  const m = device.createShaderModule({ code: `@compute @workgroup_size(1) fn p() { let v = vec4<f32>(unpack4xU8(0x0F0F0F0Fu)) + vec4<f32>(unpack4xI8(1u)); }` });
  const info = await m.getCompilationInfo();
  const err = await device.popErrorScope();
  device.__unpackOk = !err && !info.messages.some((x) => x.type === "error");
  return device.__unpackOk;
}

export function coopWGSL(WG = 256, ROWS = 4, WGB = 64, COLS = 4, ROWSB = ROWS, UNPACK = true, WIDE = null) {
  // Rows per workgroup for a C-column batched kernel: hold accumulators/thread
  // constant, so the 8- and 4-column twins are not starved of rows when COLS=16.
  const rowsFor = (C) => Math.max(1, Math.min(8, Math.round(ROWSB * COLS / C)));
  // dequant snippets: unpack4xU8/unpack4xI8 turn 4 packed bytes into a vec4 in
  // one instruction (Q4: two masked unpacks per word; Q8: one), vs 3 ALU ops
  // per element. Same values bit-for-bit; the GEMVs are instruction-bound.
  const q4lo = (w) => UNPACK ? `vec4<f32>(unpack4xU8(${w} & 0x0F0F0F0Fu)) - vec4<f32>(8.0)`
    : `vec4<f32>(f32(${w} & 0xFu), f32((${w} >> 8u) & 0xFu), f32((${w} >> 16u) & 0xFu), f32((${w} >> 24u) & 0xFu)) - vec4<f32>(8.0)`;
  const q4hi = (w) => UNPACK ? `vec4<f32>(unpack4xU8((${w} >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0)`
    : `vec4<f32>(f32((${w} >> 4u) & 0xFu), f32((${w} >> 12u) & 0xFu), f32((${w} >> 20u) & 0xFu), f32((${w} >> 28u) & 0xFu)) - vec4<f32>(8.0)`;
  const i8x4 = (w) => UNPACK ? `vec4<f32>(unpack4xI8(bitcast<u32>(${w})))`
    : `vec4<f32>(f32((${w} << 24u) >> 24u), f32((${w} << 16u) >> 24u), f32((${w} << 8u) >> 24u), f32(${w} >> 24u))`;
  const LANES = WG / 4;
  const LANESB = WGB / 4;   // batched kernels: smaller workgroups amortize the reductions
  const accDecl = Array.from({ length: ROWS }, (_, r) => `var acc${r} = 0.0;`).join(" ");
  const fullBody = (term) => Array.from({ length: ROWS }, (_, r) => `      acc${r} += ${term(r)};`).join("\n");
  const tailBody = (term, dOut) => Array.from({ length: ROWS - 1 }, (_, r) =>
    `      if (row0 + ${r}u < ${dOut}) { acc${r} += ${term(r)}; }`).join("\n");
  const store = Array.from({ length: ROWS }, (_, r) => `  mvc_part[${r * WG}u + t] = acc${r};`).join("\n");
  const treeAdd = Array.from({ length: ROWS }, (_, r) =>
    `      mvc_part[${r * WG}u + t] += mvc_part[${r * WG}u + t + stride];`).join("\n");
  const reduce = `
${store}
  workgroupBarrier();
  var stride: u32 = ${WG / 2}u;
  while (stride > 0u) {
    if (t < stride) {
${treeAdd}
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }`;
  // fused gate/up: two GEMVs sharing x, SiLU(g)*u epilogue in-kernel.
  // Saves one full-size dispatch + the silu dispatch + the u round-trip per
  // layer. Helpers are generated per concrete buffer (no pointer params:
  // unrestricted_pointer_parameters is not guaranteed in Safari 26).
  const guKernel = (kind, batched) => {
    const P = kind === "f32" ? "guf" : kind === "q8" ? "gu8" : "gu4";
    const shape = `${P}_shape`;
    const decl = kind === "f32" ? `
@group(1) @binding(0) var<storage, read> ${P}_gw: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> ${P}_uw: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> ${P}_x: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> ${P}_y: array<f32>;
@group(1) @binding(4) var<uniform> ${shape}: BShape;` : `
@group(1) @binding(0) var<storage, read> ${P}_gqs: array<u32>;
@group(1) @binding(1) var<storage, read> ${P}_gsc: array<u32>;
@group(1) @binding(2) var<storage, read> ${P}_uqs: array<u32>;
@group(1) @binding(3) var<storage, read> ${P}_usc: array<u32>;
@group(1) @binding(4) var<storage, read> ${P}_x: array<vec4<f32>>;
@group(1) @binding(5) var<storage, read_write> ${P}_y: array<f32>;
@group(1) @binding(6) var<uniform> ${shape}: BShape;`;
    const helpers = kind === "f32" ? "" : ["g", "u"].map((w) => kind === "q8" ? `
fn ${P}_${w}row(wBase: u32, si: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
  let sc = unpack2x16float(${P}_${w}sc[si >> 1u])[si & 1u];
  let w0 = bitcast<i32>(${P}_${w}qs[wBase]);
  let w1 = bitcast<i32>(${P}_${w}qs[wBase + 1u]);
  let d0 = ${i8x4("w0")};
  let d1 = ${i8x4("w1")};
  return sc * (dot(d0, xa) + dot(d1, xb));
}` : `
fn ${P}_${w}row(wIdx: u32, si: u32, xlo: vec4<f32>, xhi: vec4<f32>) -> f32 {
  let sc = unpack2x16float(${P}_${w}sc[si >> 1u])[si & 1u];
  let word = ${P}_${w}qs[wIdx];
  let lo = ${q4lo("word")};
  let hi = ${q4hi("word")};
  return sc * (dot(lo, xlo) + dot(hi, xhi));
}`).join("");
    const rowTerm = (which, r) => kind === "f32"
      ? `dot(${P}_${which}w[off4 + ${r}u * dIn4], xa) + dot(${P}_${which}w[off4 + ${r}u * dIn4 + 1u], xb)`
      : `${P}_${which}row(${kind === "q8" ? "wBase" : "wIdx"} + ${r}u * rowWords, scBase + ${r}u * nb, xa, xb)`;
    const loads = kind === "q4" ? `
      let xa = ${P}_x[xcol + b * 8u + qt];
      let xb = ${P}_x[xcol + b * 8u + qt + 4u];
      let wIdx = row0 * rowWords + b * 4u + qt;
      let scBase = row0 * nb + b;` : kind === "q8" ? `
      let xa = ${P}_x[xcol + b * 8u + qt * 2u];
      let xb = ${P}_x[xcol + b * 8u + qt * 2u + 1u];
      let wBase = row0 * rowWords + b * 8u + qt * 2u;
      let scBase = row0 * nb + b;` : `
      let xa = ${P}_x[xcol + b * 8u + qt * 2u];
      let xb = ${P}_x[xcol + b * 8u + qt * 2u + 1u];
      let off4 = row0 * dIn4 + b * 8u + qt * 2u;`;
    const accs = (w) => Array.from({ length: ROWS }, (_, r) => `var ${w}${r} = 0.0;`).join(" ");
    const body = (w, which) => `      if (full) {
${Array.from({ length: ROWS }, (_, r) => `        ${w}${r} += ${rowTerm(which, r)};`).join("\n")}
      } else {
${Array.from({ length: ROWS - 1 }, (_, r) => `        if (row0 + ${r}u < dOut) { ${w}${r} += ${rowTerm(which, r)}; }`).join("\n")}
      }`;
    const reduceTo = (w, dest) => `
${Array.from({ length: ROWS }, (_, r) => `  mvc_part[${r * WG}u + t] = ${w}${r};`).join("\n")}
  workgroupBarrier();
  {
    var stride: u32 = ${WG / 2}u;
    while (stride > 0u) {
      if (t < stride) {
${Array.from({ length: ROWS }, (_, r) => `        mvc_part[${r * WG}u + t] += mvc_part[${r * WG}u + t + stride];`).join("\n")}
      }
      workgroupBarrier();
      stride = stride >> 1u;
    }
  }
  ${dest}
  workgroupBarrier();`;
    const colBody = (m) => `
  {
    let xcol = ${batched ? `${m}u * ${shape}.xs4 * 0u + ${m}u * ${shape}.xs4` : "0u"};
    ${accs("ag")}
    ${accs("au")}
    for (var b: u32 = bl; b < nb; b += ${LANES}u) {${loads}
${body("ag", "g")}
${body("au", "u")}
    }
${reduceTo("ag", `if (t < ${ROWS}u) { gu_res[t] = mvc_part[t * ${WG}u]; }`)}
${reduceTo("au", `if (t < ${ROWS}u) {
    let row = row0 + t;
    if (row < dOut) {
      let g = gu_res[t];
      ${P}_y[${batched ? `${m}u * ${shape}.ys + row` : "row"}] = (g / (1.0 + exp(-g))) * mvc_part[t * ${WG}u];
    }
  }`)}
  }`;
    return `${batched ? "" : decl + helpers}
@compute @workgroup_size(${WG})
fn matvec${kind === "f32" ? "" : "_" + kind}_gu${batched ? "_b" : ""}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let qt = t & 3u;
  let bl = t >> 2u;
  let dIn = ${shape}.dIn;
  let dOut = ${shape}.dOut;
  let nb = dIn / 32u;
  let dIn4 = dIn / 4u;
  let rowWords = ${kind === "q4" ? "dIn / 8u" : "dIn / 4u"};
  let row0 = (wg.y * 32768u + wg.x) * ${ROWS}u;
  let full = row0 + ${ROWS - 1}u < dOut;
${batched ? [0, 1, 2, 3].map(colBody).join("\n") : colBody(0)}
}`;
  };
  // batched gate/up: ONE pass over the weights; each g/u word is loaded and
  // decoded once and applied to all 4 columns (2*ROWS*4 accumulators), then
  // one reduction round per column.
  const guKernelB = (kind, C = COLS) => {
    const P = kind === "f32" ? "guf" : kind === "q8" ? "gu8" : "gu4";
    const shape = `${P}_shape`;
    const RB = rowsFor(C), rows = Array.from({ length: RB }, (_, r) => r);
    const cols = Array.from({ length: C }, (_, m) => m);
    const xLoads = kind === "q4"
      ? cols.map((m) => `let xa${m} = ${P}_x[${m}u * xs4 + b * 8u + qt]; let xb${m} = ${P}_x[${m}u * xs4 + b * 8u + qt + 4u];`).join("\n      ")
      : cols.map((m) => `let xa${m} = ${P}_x[${m}u * xs4 + b * 8u + qt * 2u]; let xb${m} = ${P}_x[${m}u * xs4 + b * 8u + qt * 2u + 1u];`).join("\n      ");
    const idx = kind === "q4" ? `let wIdx = row0 * rowWords + b * 4u + qt; let scBase = row0 * nb + b;`
      : kind === "q8" ? `let wBase = row0 * rowWords + b * 8u + qt * 2u; let scBase = row0 * nb + b;`
      : `let off4 = row0 * dIn4 + b * 8u + qt * 2u;`;
    const nib = (w, v) => `let ${v}lo = ${q4lo(w)};
        let ${v}hi = ${q4hi(w)};`;
    const sx = (w, v) => `let ${v} = ${i8x4(w)};`;
    const rowBlock = (r) => {
      const acc = (ga, gb, ua, ub, gs, us) => cols.map((m) =>
        `ag${r}_${m} += ${gs}(dot(${ga}, xa${m}) + dot(${gb}, xb${m})); au${r}_${m} += ${us}(dot(${ua}, xa${m}) + dot(${ub}, xb${m}));`).join(" ");
      if (kind === "f32") return `if (full || row0 + ${r}u < dOut) {
        let ga = ${P}_gw[off4 + ${r}u * dIn4]; let gb = ${P}_gw[off4 + ${r}u * dIn4 + 1u];
        let ua = ${P}_uw[off4 + ${r}u * dIn4]; let ub = ${P}_uw[off4 + ${r}u * dIn4 + 1u];
        ${acc("ga", "gb", "ua", "ub", "", "")}
      }`;
      if (kind === "q8") return `if (full || row0 + ${r}u < dOut) {
        let si = scBase + ${r}u * nb;
        let gs = unpack2x16float(${P}_gsc[si >> 1u])[si & 1u]; let us = unpack2x16float(${P}_usc[si >> 1u])[si & 1u];
        let g0 = bitcast<i32>(${P}_gqs[wBase + ${r}u * rowWords]); let g1 = bitcast<i32>(${P}_gqs[wBase + ${r}u * rowWords + 1u]);
        let u0 = bitcast<i32>(${P}_uqs[wBase + ${r}u * rowWords]); let u1 = bitcast<i32>(${P}_uqs[wBase + ${r}u * rowWords + 1u]);
        ${sx("g0", "gd0")} ${sx("g1", "gd1")} ${sx("u0", "ud0")} ${sx("u1", "ud1")}
        ${acc("gd0", "gd1", "ud0", "ud1", "gs * ", "us * ")}
      }`;
      return `if (full || row0 + ${r}u < dOut) {
        let si = scBase + ${r}u * nb;
        let gs = unpack2x16float(${P}_gsc[si >> 1u])[si & 1u]; let us = unpack2x16float(${P}_usc[si >> 1u])[si & 1u];
        let gw = ${P}_gqs[wIdx + ${r}u * rowWords]; let uw = ${P}_uqs[wIdx + ${r}u * rowWords];
        ${nib("gw", "g")}
        ${nib("uw", "u")}
        ${acc("glo", "ghi", "ulo", "uhi", "gs * ", "us * ")}
      }`;
    };
    const K2 = 2 * RB;
    return `
@compute @workgroup_size(${WGB})
fn matvec${kind === "f32" ? "" : "_" + kind}_gu_b${C === COLS ? "" : C}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let qt = t & 3u;
  let bl = t >> 2u;
  let dIn = ${shape}.dIn;
  let dOut = ${shape}.dOut;
  let xs4 = ${shape}.xs4;
  let ys = ${shape}.ys;
  let nb = dIn / 32u;
  let dIn4 = dIn / 4u;
  let rowWords = ${kind === "q4" ? "dIn / 8u" : "dIn / 4u"};
  let row0 = (wg.y * 32768u + wg.x) * ${RB}u;
  let full = row0 + ${RB - 1}u < dOut;
  ${rows.map((r) => cols.map((m) => `var ag${r}_${m} = 0.0; var au${r}_${m} = 0.0;`).join(" ")).join("\n  ")}
  for (var b: u32 = bl; b < nb; b += ${LANESB}u) {
      ${xLoads}
      ${idx}
      ${rows.map(rowBlock).join("\n      ")}
  }
${cols.map((m) => `
  ${rows.map((r) => `gub_part[${r * WGB}u + t] = ag${r}_${m}; gub_part[${(RB + r) * WGB}u + t] = au${r}_${m};`).join(" ")}
  workgroupBarrier();
  {
    var stride: u32 = ${WGB / 2}u;
    while (stride > 0u) {
      if (t < stride) { ${Array.from({ length: K2 }, (_, k) => `gub_part[${k * WGB}u + t] += gub_part[${k * WGB}u + t + stride];`).join(" ")} }
      workgroupBarrier();
      stride = stride >> 1u;
    }
  }
  if (t < ${RB}u) {
    let row = row0 + t;
    if (row < dOut) {
      let g = gub_part[t * ${WGB}u];
      ${P}_y[${m}u * ys + row] = (g / (1.0 + exp(-g))) * gub_part[(${RB}u + t) * ${WGB}u];
    }
  }
  workgroupBarrier();`).join("\n")}
}`;
  };
  const guAll = `
var<workgroup> gu_res: array<f32, ${ROWS}>;
var<workgroup> gub_part: array<f32, ${2 * rowsFor(4) * WGB}>;
` + ["f32", "q8", "q4"].map((k) => guKernel(k, false)).join("\n")
    + ["f32", "q8", "q4"].map((k) => guKernelB(k)).join("\n")
    + (COLS > 8 ? ["f32", "q8", "q4"].map((k) => guKernelB(k, 8)).join("\n") : "")
    + (COLS > 4 ? ["f32", "q8", "q4"].map((k) => guKernelB(k, 4)).join("\n") : "");
  const singleCoop = `@compute @workgroup_size(${WG})
fn matvec_coop(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let qt = t & 3u;
  let bl = t >> 2u;
  let dIn = mv_shape.dIn;
  let nb = dIn / 32u;
  let dIn4 = dIn / 4u;
  let row0 = (wg.y * 32768u + wg.x) * ${ROWS}u;
  let full = row0 + ${ROWS - 1}u < mv_shape.dOut;
  ${accDecl}
  for (var b: u32 = bl; b < nb; b += ${LANES}u) {
    let c4 = b * 8u + qt * 2u;
    let xa = mv_x4[c4];
    let xb = mv_x4[c4 + 1u];
    let off4 = row0 * dIn4 + c4;
    if (full) {
${fullBody((r) => `mvf_row(off4 + ${r}u * dIn4, xa, xb)`)}
    } else {
${tailBody((r) => `mvf_row(off4 + ${r}u * dIn4, xa, xb)`, "mv_shape.dOut")}
    }
  }
${reduce}
  if (t < ${ROWS}u) {
    let row = row0 + t;
    if (row < mv_shape.dOut) { mv_y[row] = mvc_part[t * ${WG}u]; }
  }
}

fn q8_row(wBase: u32, sc: f32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
  let w0 = bitcast<i32>(q8_qs[wBase]);
  let w1 = bitcast<i32>(q8_qs[wBase + 1u]);
  // shift-based sign extension (extractBits goes through slow polyfill paths)
  let d0 = ${i8x4("w0")};
  let d1 = ${i8x4("w1")};
  return sc * (dot(d0, xa) + dot(d1, xb));
}
@compute @workgroup_size(${WG})
fn matvec_q8_coop(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let qt = t & 3u;
  let bl = t >> 2u;
  let dIn = q8_shape.dIn;
  let nb = dIn / 32u;
  let rowWords = dIn / 4u;
  let row0 = (wg.y * 32768u + wg.x) * ${ROWS}u;
  let full = row0 + ${ROWS - 1}u < q8_shape.dOut;
  ${accDecl}
  for (var b: u32 = bl; b < nb; b += ${LANES}u) {
    let x4 = b * 8u + qt * 2u;
    let xa = q8_x4[x4];
    let xb = q8_x4[x4 + 1u];
    let wBase = row0 * rowWords + b * 8u + qt * 2u;
    let scBase = row0 * nb + b;
    if (full) {
${fullBody((r) => `q8_row(wBase + ${r}u * rowWords, q8s(scBase + ${r}u * nb), xa, xb)`)}
    } else {
${tailBody((r) => `q8_row(wBase + ${r}u * rowWords, q8s(scBase + ${r}u * nb), xa, xb)`, "q8_shape.dOut")}
    }
  }
${reduce}
  if (t < ${ROWS}u) {
    let row = row0 + t;
    if (row < q8_shape.dOut) { q8_y[row] = mvc_part[t * ${WG}u]; }
  }
}

fn q4_row(wIdx: u32, sc: f32, xlo: vec4<f32>, xhi: vec4<f32>) -> f32 {
  let word = q4_qs[wIdx];
  let lo = ${q4lo("word")};
  let hi = ${q4hi("word")};
  return sc * (dot(lo, xlo) + dot(hi, xhi));
}
@compute @workgroup_size(${WG})
fn matvec_q4_coop(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let qt = t & 3u;
  let bl = t >> 2u;
  let dIn = q4_shape.dIn;
  let nb = dIn / 32u;
  let rowWords = dIn / 8u;   // 4 bits/weight -> dIn/8 u32 words per row
  let row0 = (wg.y * 32768u + wg.x) * ${ROWS}u;
  let full = row0 + ${ROWS - 1}u < q4_shape.dOut;
  ${accDecl}
  for (var b: u32 = bl; b < nb; b += ${LANES}u) {
    // word qt of block b covers x[j..j+3] (low nibbles) and x[j+16..j+19] (high)
    let xlo = q4_x4[b * 8u + qt];
    let xhi = q4_x4[b * 8u + qt + 4u];
    let wIdx = row0 * rowWords + b * 4u + qt;
    let scBase = row0 * nb + b;
    if (full) {
${fullBody((r) => `q4_row(wIdx + ${r}u * rowWords, q4s(scBase + ${r}u * nb), xlo, xhi)`)}
    } else {
${tailBody((r) => `q4_row(wIdx + ${r}u * rowWords, q4s(scBase + ${r}u * nb), xlo, xhi)`, "q4_shape.dOut")}
    }
  }
${reduce}
  if (t < ${ROWS}u) {
    let row = row0 + t;
    if (row < q4_shape.dOut) { q4_y[row] = mvc_part[t * ${WG}u]; }
  }
}
`;
  // accumulate variants: y[row] += W x (residual add folded into the matvec)
  const singleCoopAcc = singleCoop.replace(/fn (q8|q4)_row\([\s\S]*?\n}\n/g, "")   // helpers already defined once
    .replace(/_coop\(/g, "_coop_acc(").replace(/_y\[row\] = /g, "_y[row] += ");
  return /* wgsl */ `
var<workgroup> mvc_part: array<f32, ${WG * ROWS}>;   // [${ROWS} rows][${WG} threads]

// vec4 views of the same buffers the scalar kernels bind (same @group/@binding
// is legal as long as no single entry point references both views). All x/w
// buffers are multiples of 16 bytes (dims divisible by 4).
@group(1) @binding(0) var<storage, read> mv_w4: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> mv_x4: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> q8_x4: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> q4_x4: array<vec4<f32>>;

fn mvf_row(off4: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
  return dot(mv_w4[off4], xa) + dot(mv_w4[off4 + 1u], xb);
}
${WIDE ? wideGEMV(WIDE, [COLS, 8, 4].filter((c, i) => i === 0 || c < COLS), { q4lo, q4hi, i8x4 }).single : singleCoop + "\n" + singleCoopAcc}

// ---- batched (COLS-column) variants for prefill / verify: each weight word is
// loaded and decoded once and applied to C token columns (rowsFor(C) rows per WG). x is [C][xs4] vec4s,
// y is [C][ys] f32s (strides in BShape; slices are 256-byte aligned by the
// engine). Workgroup ${WGB}: more loop work per thread, cheaper reductions.
struct BShape { dOut: u32, dIn: u32, xs4: u32, ys: u32 };
@group(1) @binding(3) var<uniform> mvb_shape: BShape;
@group(1) @binding(4) var<uniform> qb_shape: BShape;
var<workgroup> mvb_part: array<f32, ${2 * rowsFor(4) * WGB}>;
${WIDE ? wideGEMV(WIDE, [COLS, 8, 4].filter((c, i) => i === 0 || c < COLS), { q4lo, q4hi, i8x4 }).batched : [COLS, 8, 4].filter((c, i) => i === 0 || c < COLS).map((C) => [false, true].map((ACC) => ["", "_q8", "_q4"].map((kind) => {
  const shp = kind === "" ? "mvb_shape" : "qb_shape";
  const xbuf = kind === "" ? "mv_x4" : kind === "_q8" ? "q8_x4" : "q4_x4";
  const ybuf = kind === "" ? "mv_y" : kind === "_q8" ? "q8_y" : "q4_y";
  const RB = rowsFor(C), rows = Array.from({ length: RB }, (_, r) => r);
  const cols = Array.from({ length: C }, (_, m) => m);
  const xLoads = kind === "_q4"
    ? cols.map((m) => `let xa${m} = ${xbuf}[${m}u * xs4 + b * 8u + qt]; let xb${m} = ${xbuf}[${m}u * xs4 + b * 8u + qt + 4u];`).join("\n    ")
    : cols.map((m) => `let xa${m} = ${xbuf}[${m}u * xs4 + b * 8u + qt * 2u]; let xb${m} = ${xbuf}[${m}u * xs4 + b * 8u + qt * 2u + 1u];`).join("\n    ");
  const idx = kind === "_q4" ? "let wIdx = row0 * rowWords + b * 4u + qt; let scBase = row0 * nb + b;"
    : kind === "_q8" ? "let wBase = row0 * rowWords + b * 8u + qt * 2u; let scBase = row0 * nb + b;"
    : "let off4 = row0 * rowWords + b * 8u + qt * 2u;";
  const rowBlock = (r) => kind === "_q4" ? `if (full || row0 + ${r}u < dOut) {
      let word = q4_qs[wIdx + ${r}u * rowWords];
      let lo = ${q4lo("word")};
      let hi = ${q4hi("word")};
      let s = q4s(scBase + ${r}u * nb);
      ${cols.map((m) => `a${r}_${m} += s * (dot(lo, xa${m}) + dot(hi, xb${m}));`).join(" ")}
    }` : kind === "_q8" ? `if (full || row0 + ${r}u < dOut) {
      let w0 = bitcast<i32>(q8_qs[wBase + ${r}u * rowWords]);
      let w1 = bitcast<i32>(q8_qs[wBase + ${r}u * rowWords + 1u]);
      let d0 = ${i8x4("w0")};
      let d1 = ${i8x4("w1")};
      let s = q8s(scBase + ${r}u * nb);
      ${cols.map((m) => `a${r}_${m} += s * (dot(d0, xa${m}) + dot(d1, xb${m}));`).join(" ")}
    }` : `if (full || row0 + ${r}u < dOut) {
      let wa = mv_w4[off4 + ${r}u * rowWords]; let wb = mv_w4[off4 + ${r}u * rowWords + 1u];
      ${cols.map((m) => `a${r}_${m} += dot(wa, xa${m}) + dot(wb, xb${m});`).join(" ")}
    }`;
  return `
@compute @workgroup_size(${WGB})
fn matvec${kind}_coop_b${C === COLS ? "" : C}${ACC ? "_acc" : ""}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let qt = t & 3u;
  let bl = t >> 2u;
  let dIn = ${shp}.dIn;
  let dOut = ${shp}.dOut;
  let xs4 = ${shp}.xs4;
  let ys = ${shp}.ys;
  let nb = dIn / 32u;
  let rowWords = ${kind === "_q4" ? "dIn / 8u" : "dIn / 4u"};
  let row0 = (wg.y * 32768u + wg.x) * ${RB}u;
  let full = row0 + ${RB - 1}u < dOut;
  ${rows.map((r) => cols.map((m) => `var a${r}_${m} = 0.0;`).join(" ")).join("\n  ")}
  for (var b: u32 = bl; b < nb; b += ${LANESB}u) {
    ${xLoads}
    ${idx}
    ${rows.map(rowBlock).join("\n    ")}
  }
${Array.from({ length: C / 2 }, (_, h) => h).map((h) => `
  ${rows.map((r) => `mvb_part[${r * WGB}u + t] = a${r}_${2 * h}; mvb_part[${(RB + r) * WGB}u + t] = a${r}_${2 * h + 1};`).join(" ")}
  workgroupBarrier();
  {
    var stride: u32 = ${WGB / 2}u;
    while (stride > 0u) {
      if (t < stride) { ${Array.from({ length: 2 * RB }, (_, k) => `mvb_part[${k * WGB}u + t] += mvb_part[${k * WGB}u + t + stride];`).join(" ")} }
      workgroupBarrier();
      stride = stride >> 1u;
    }
  }
  if (t < ${2 * RB}u) {
    let r = t % ${RB}u;
    let m = ${2 * h}u + t / ${RB}u;
    let row = row0 + r;
    if (row < dOut) { ${ybuf}[m * ys + row] ${ACC ? "+=" : "="} mvb_part[t * ${WGB}u]; }
  }
  workgroupBarrier();`).join("")}
}`;
}).join("\n")).join("\n")).join("\n")}
${guAll}
`;
}


// ---- wide GEMV layout (engine option coopWide; for GPUs without subgroups where the coop layout above is
// load- and barrier-bound, e.g. Apple GPUs in Safari) ----
// The coop kernels give each thread an 8-weight quarter of a 32-weight block (two 4-byte Q8 loads or one Q4
// word a block row) and reduce ROWS arrays over the whole workgroup (log2 WG barrier levels). The wide layout
// is the fused MoE kernels' (engine/wgsl/moe.js, moeFusedLayout "wide") applied to the dense GEMVs: groups of
// TPR threads, each group owns R consecutive rows, each thread owns whole blocks (b = lane, lane + TPR, ...: one
// 16-byte vec4<u32> load per Q4 block, two per Q8 block, eight vec4<f32> of x), and each group reduces its rows
// with a log2(TPR)-level tree. Rows past dOut are clamped (computed, never stored), so no row has a tail branch.
// The batched kernels (b<C>: verify and short prefill passes) run the same per-row code for every column (same
// lanes, same block order, same term and the same tree), so a column's output has the same bits in a C-column
// pass as in the one-column kernel: decode and verify agree, spec == plain. Its sums are in another order
// than the coop layout's (as are coop's at another WG), so a device's GEMV bits depend on the layout.
// Entry points and bindings are the coop ones, so the engine only changes its grid (rows per workgroup:
// (WG / TPR) * R one-column, WG / TPR batched).
export const COOP_WIDE = Object.freeze({ WG: 64, TPR: 8, R: 1 });
export function coopWideConfig(opt) {
  if (!opt) return null;
  const c = { ...COOP_WIDE, ...(opt === true || opt === "wide" ? {} : opt) }, p2 = (n) => Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
  if (!p2(c.WG) || c.WG > 256 || !p2(c.TPR) || c.TPR > c.WG || ![1, 2, 4].includes(c.R)) throw new Error(`coopWide: ${JSON.stringify(opt)} (WG, TPR powers of two, TPR <= WG <= 256, R 1, 2 or 4)`);
  return Object.freeze({ ...c, rows: (c.WG / c.TPR) * c.R, rowsB: c.WG / c.TPR });
}
function wideGEMV({ WG, TPR, R }, widths, O) {
  const RPW = (WG / TPR) * R, rs = Array.from({ length: R }, (_, r) => r);
  const KINDS = [["", "mv", "mv_w4", "mv_x4", "mv_y"], ["_q8", "q8", "q8_qs4", "q8_x4", "q8_y"], ["_q4", "q4", "q4_qs4", "q4_x4", "q4_y"]];
  // one block's term for row index expression ri, x vec4s named x0..x7 (prefix px)
  const term = (k, Q, ri, px) => {
    if (k === "_q4") {
      const p = (i) => `(dot(${O.q4lo(`w[${i}]`)}, ${px}${i}) + dot(${O.q4hi(`w[${i}]`)}, ${px}${i + 4}))`;
      return { load: `let w = ${Q}[${ri} * nb + b];`, expr: `sc * ((${p(0)} + ${p(1)}) + (${p(2)} + ${p(3)}))`, sc: `let sc = q4s(${ri} * nb + b);` };
    }
    if (k === "_q8") {
      const d = (v, i, s) => `dot(${O.i8x4(`bitcast<i32>(${v}[${i}])`)}, ${px}${s})`;
      const h = (v, o) => `((${d(v, 0, o)} + ${d(v, 1, o + 1)}) + (${d(v, 2, o + 2)} + ${d(v, 3, o + 3)}))`;
      return { load: `let wa = ${Q}[(${ri} * nb + b) * 2u]; let wb = ${Q}[(${ri} * nb + b) * 2u + 1u];`, expr: `sc * (${h("wa", 0)} + ${h("wb", 4)})`, sc: `let sc = q8s(${ri} * nb + b);` };
    }
    const d = (i) => `dot(${Q}[wo + ${i}u], ${px}${i})`;
    return { load: `let wo = ${ri} * (dIn / 4u) + b * 8u;`, expr: `((${d(0)} + ${d(1)}) + (${d(2)} + ${d(3)})) + ((${d(4)} + ${d(5)}) + (${d(6)} + ${d(7)}))`, sc: "" };
  };
  const tree = (n, red, W) => `
  workgroupBarrier();${TPR === 1 ? "" : `
  for (var st: u32 = ${TPR / 2}u; st > 0u; st >>= 1u) {
    if (lane < st) {
${Array.from({ length: n }, (_, a) => `      ${red}[${a * W}u + t] += ${red}[${a * W}u + t + st];`).join("\n")}
    }
    workgroupBarrier();
  }`}`;
  const single = KINDS.map(([k, P, Q, X, Y]) => [false, true].map((ACC) => `
@compute @workgroup_size(${WG})
fn matvec${k}_coop${ACC ? "_acc" : ""}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x; let lane = t % ${TPR}u; let grp = t / ${TPR}u;
  let dIn = ${P}_shape.dIn; let dOut = ${P}_shape.dOut; let nb = dIn / 32u;
  let row0 = (wg.y * 32768u + wg.x) * ${RPW}u + grp * ${R}u;
${rs.map((r) => `  let rr${r} = min(row0 + ${r}u, dOut - 1u); var a${r}: f32 = 0.0;`).join("\n")}
  for (var b: u32 = lane; b < nb; b += ${TPR}u) {
    ${Array.from({ length: 8 }, (_, s) => `let x${s} = ${X}[b * 8u + ${s}u];`).join(" ")}
${rs.map((r) => { const T = term(k, Q, `rr${r}`, "x"); return `    { ${T.load} ${T.sc}\n      a${r} += ${T.expr}; }`; }).join("\n")}
  }
${rs.map((r) => `  mvw_red[${r * WG}u + t] = a${r};`).join("\n")}
${tree(R, "mvw_red", WG)}
  if (lane < ${R}u) {
    let row = row0 + lane;
    if (row < dOut) { ${Y}[row] ${ACC ? "+=" : "="} mvw_red[lane * ${WG}u + grp * ${TPR}u]; }
  }
}`).join("\n")).join("\n");
  const batched = widths.map((C) => KINDS.map(([k, P, Q, X, Y]) => [false, true].map((ACC) => {
    const shp = k === "" ? "mvb_shape" : "qb_shape", cs = Array.from({ length: C }, (_, m) => m), T = term(k, Q, "rr", "x");
    return `
@compute @workgroup_size(${WG})
fn matvec${k}_coop_b${C === widths[0] ? "" : C}${ACC ? "_acc" : ""}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x; let lane = t % ${TPR}u; let grp = t / ${TPR}u;
  let dIn = ${shp}.dIn; let dOut = ${shp}.dOut; let xs4 = ${shp}.xs4; let ys = ${shp}.ys; let nb = dIn / 32u;
  let row = (wg.y * 32768u + wg.x) * ${WG / TPR}u + grp;
  let rr = min(row, dOut - 1u);
  ${cs.map((m) => `var a${m}: f32 = 0.0;`).join(" ")}
  for (var b: u32 = lane; b < nb; b += ${TPR}u) {
    ${T.load} ${T.sc}
${cs.map((m) => `    { ${Array.from({ length: 8 }, (_, s) => `let x${s} = ${X}[${m}u * xs4 + b * 8u + ${s}u];`).join(" ")}\n      a${m} += ${T.expr}; }`).join("\n")}
  }
${cs.map((m) => `  mvwb_red[${m * WG}u + t] = a${m};`).join("\n")}
${tree(C, "mvwb_red", WG)}
  if (row < dOut) {
    for (var m: u32 = lane; m < ${C}u; m += ${TPR}u) { ${Y}[m * ys + row] ${ACC ? "+=" : "="} mvwb_red[m * ${WG}u + grp * ${TPR}u]; }
  }
}`;
  }).join("\n")).join("\n")).join("\n");
  return {
    single: `
@group(1) @binding(0) var<storage, read> q8_qs4: array<vec4<u32>>;
@group(1) @binding(0) var<storage, read> q4_qs4: array<vec4<u32>>;
var<workgroup> mvw_red: array<f32, ${R * WG}>;
${single}`,
    batched: `
var<workgroup> mvwb_red: array<f32, ${Math.max(...widths) * WG}>;
${batched}`,
  };
}
