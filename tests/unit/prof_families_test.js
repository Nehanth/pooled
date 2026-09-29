// The profilers' kernel families come from the engine's own pipeline list (tests/prof/families.js,
// issue #82): every pipeline a Qwen engine creates, over the room's settings and the A/B variants, must
// land in a named family. A new kernel with no family rule fails here instead of vanishing into
// "everything else" in a profile. CPU only: the engines are built on the recording mock device.
//   deno test --no-check --allow-read tests/unit/prof_families_test.js
import { mockDevice } from "./mock_gpu.js";
import { buildSynthGGUF, SYNTH_MOE } from "../e2e/synth.mjs";
import { parseGGUFHeader, qwen35Weights, GGML_EMBED } from "../../engine/gguf.js";
import { Qwen35Engine } from "../../engine/qwen35.js";
import { roomQwen35Options } from "../../engine/preset.js";
import { FAMILIES, pipeFamily, pipeNames, familiesOf, pipeRows } from "../prof/families.js";

const eq = (a, b, m) => { const x = JSON.stringify(a), y = JSON.stringify(b); if (x !== y) throw new Error(`${m}: ${x} != ${y}`); };
const models = { dense: buildSynthGGUF({ layers: 8 }).bytes, moe: buildSynthGGUF({ layers: 8, moe: SYNTH_MOE }).bytes };

async function engine(model, flags = "", extra = {}, { lo = 0, head = true } = {}) {
  const buf = models[model];
  const G = parseGGUFHeader(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const bytesOf = async (i) => buf.slice(i.byteOffset, i.byteOffset + i.byteLength);
  const L = G.meta["qwen35.block_count"] - 1;
  const origWarn = console.warn; console.warn = () => {};
  try {
    return await Qwen35Engine.create({ device: mockDevice({ wgMem: 32768 }), meta: G.meta, layerRange: [lo, L], hasEmbed: lo === 0, hasHead: head,
      vocab: G.tensors[GGML_EMBED].shape[0], maxSeq: 512, ...roomQwen35Options(flags), ...extra,
      weights: await qwen35Weights(G, bytesOf, { lo, hi: L, hasEmbed: lo === 0, hasHead: head, mtp: head }) });
  } finally { console.warn = origWarn; }
}

Deno.test("families: known names", () => {
  const cases = { matvec_q4_coop_b8_acc: "GEMV (projections, LM head)", matvec_coop_h: "GEMV (projections, LM head)", gemm_q8_5120_s4_r16: "prefill GEMM",
    gemm_sgm_q4_5120_s2: "prefill GEMM", gemm_w_q4_acc: "prefill GEMM", moe_route: "MoE router", moe_router: "MoE router", moe_gsort: "MoE sort/combine",
    moe_combw: "MoE sort/combine", moe_gusg_4: "MoE experts", moe_dnc_q4: "MoE experts", moe_gu_q8: "MoE experts", dn_delta_gn: "DeltaNet core",
    dn_pre_mc: "DeltaNet core", attn_flash_tile: "attention", kv_store_q8: "attention", head_norm_mc: "attention", rmsnorm_mc: "norms + residual",
    add_res: "norms + residual", silu_mul_w: "SiLU gate", topk_b: "sampling (argmax, top-k)", argmax: "sampling (argmax, top-k)", emb_gather: "embedding gather",
    brand_new_kernel: "other" };
  for (const [n, f] of Object.entries(cases)) eq(pipeFamily(n), f, n);
  eq(new Set(FAMILIES.map(([f]) => f)).size, FAMILIES.length, "family names are distinct");
});

Deno.test("families: every pipeline of the room's engines and the A/B variants has a family", async () => {
  const seen = new Set();
  const variants = [
    ["dense host, room settings", "dense", ""], ["MoE host, room settings", "moe", ""],
    ["MoE worker (no embedding, no head)", "moe", "", {}, { lo: 4, head: false }],
    ["dense ?kv=q8", "dense", "kv=q8"], ["dense ?fuse=0", "dense", "fuse=0"], ["MoE ?moefuse=0", "moe", "moefuse=0"],
    ["MoE ?gpusample=0", "moe", "gpusample=0"], ["dense wide prefill", "dense", "", { prefillUbatch: 256 }],
    ["dense 4 batch columns", "dense", "", { batchCols: 4, coopRowsB: 4 }], ["dense head rows", "dense", "", { headRows: 8 }],
  ];
  for (const [what, model, flags, extra = {}, range = {}] of variants) {
    const eng = await engine(model, flags, extra, range);
    const names = pipeNames(eng);
    if (!names.length) throw new Error(`${what}: no pipelines`);
    const fams = familiesOf(names);
    if (fams.other) throw new Error(`${what}: pipelines with no family (add a rule in tests/prof/families.js): ${fams.other.join(" ")}`);
    // a partition: every pipeline exactly once
    eq(Object.values(fams).flat().sort(), names, `${what}: families cover the pipeline list`);
    names.forEach((n) => seen.add(n));
  }
  // the variants reach the kernels a family list used to miss
  for (const n of ["attn_flash", "dn_pre", "matvec_q4_coop_acc", "kv_store_q8", "moe_route", "moe_router", "topk_a", "gemm_w_q4", "silu_mul_w", "matvec_q8_coop_h"]) {
    if (!seen.has(n)) throw new Error(`no variant created ${n}; the check above never saw it`);
  }
});

Deno.test("pipeRows: every pipeline listed, undispatched at 0, unknown dispatches flagged", () => {
  const rows = pipeRows(["b", "a", "c"], [{ name: "a", ms: 2 }, { name: "a", ms: 4 }, { name: "c", ms: 1 }, { name: "zz", ms: 3 }], 2);
  eq(rows.map((r) => r.pipe), ["a", "zz", "c", "b"], "order: by ms");
  eq(rows[0], { pipe: "a", family: "other", ms: 3, n: 1, usEach: 3000 }, "a per unit");
  eq(rows.find((r) => r.pipe === "b"), { pipe: "b", family: "other", ms: 0, n: 0, usEach: 0 }, "never dispatched");
  eq(rows.find((r) => r.pipe === "zz").unlisted, true, "not in the engine's list");
  eq(pipeRows(["matvec_coop"], [], 1)[0].family, "GEMV (projections, LM head)", "family column");
});
