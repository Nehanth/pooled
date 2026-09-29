// The room's engine settings (engine/preset.js): the defaults, every ?flag, and that the room and the
// benchmarks / profilers all build their engines from it (issue #82).
//   deno test --no-check --allow-read tests/unit/preset_test.js
import { roomQwen35Options, roomEngineFlags, applyRoomFlags, ROOM_BATCH_COLS, ROOM_DRAFT_VOCAB } from "../../engine/preset.js";

const eq = (a, b, m) => { const x = JSON.stringify(a), y = JSON.stringify(b); if (x !== y) throw new Error(`${m}: ${x} != ${y}`); };
const read = (p) => Deno.readTextFileSync(new URL("../../" + p, import.meta.url));

const DEFAULTS = { batchCols: 16, coopRowsB: 1, draftVocab: 65536, draftVocabAuto: true, draftChain: true, specFuse: true,
  kvQ8: false, moeFuse: true, moeDnRows: 1, gpuSample: true, argmaxWide: true };

Deno.test("preset: the room's defaults with no flags, from every kind of flag source", () => {
  eq(ROOM_BATCH_COLS, 16, "batch columns"); eq(ROOM_DRAFT_VOCAB, 65536, "draft vocab");
  for (const src of [undefined, null, "", "?", new URLSearchParams(), {}, "?model=q36moe&room=abc"]) eq(roomQwen35Options(src), DEFAULTS, `options from ${JSON.stringify(src)}`);
  eq(roomEngineFlags(), { mtpBatchFill: true, mtpBatchRefill: true, mtpPreDraft: true }, "engine flags");
});

Deno.test("preset: every room ?flag changes its option, the same from a string, URLSearchParams or an object", () => {
  const cases = [
    ["draftvocab=0", { draftVocab: 0 }], ["draftvocab=32768", { draftVocab: 32768 }], ["draftvocab=junk", { draftVocab: 0 }],
    ["dvauto=0", { draftVocabAuto: false }], ["draftchain=0", { draftChain: false }], ["specfuse=0", { specFuse: false }],
    ["kv=q8", { kvQ8: true }], ["kv=f16", {}], ["moefuse=0", { moeFuse: false }], ["moednrows=4", { moeDnRows: 4 }], ["moednrows=x", {}],
    ["gpusample=0", { gpuSample: false, argmaxWide: false }], ["gpusample=0&argmaxwide=1", { gpuSample: false, argmaxWide: true }],
    ["argmaxwide=0", { argmaxWide: false }], ["gpusample=1", {}], ["draftchain=1", {}],
  ];
  for (const [qs, diff] of cases) {
    const want = { ...DEFAULTS, ...diff };
    eq(roomQwen35Options(qs), want, qs);
    eq(roomQwen35Options("?" + qs), want, "?" + qs);
    eq(roomQwen35Options(new URLSearchParams(qs)), want, "URLSearchParams " + qs);
    eq(roomQwen35Options(Object.fromEntries(new URLSearchParams(qs))), want, "object " + qs);
  }
  const unfused = roomQwen35Options("fuse=0");
  eq([unfused.attnGlue, unfused.dnFuse, unfused.attnMC], [false, false, false], "fuse=0");
  eq("attnGlue" in roomQwen35Options(""), false, "fused kernels are left to the engine's defaults");
  // the prefill options stay the engine's defaults, so host and workers agree
  for (const k of ["attnPrefillTile", "prefillUbatch", "moeGroupPrefill"]) eq(k in roomQwen35Options("kv=q8&moefuse=0"), false, k);
});

Deno.test("preset: engine flags and applyRoomFlags", () => {
  eq(roomEngineFlags("mtpbatch=0"), { mtpBatchFill: false, mtpBatchRefill: true, mtpPreDraft: true }, "mtpbatch=0");
  eq(roomEngineFlags("mtprefill=0&predraft=0"), { mtpBatchFill: true, mtpBatchRefill: false, mtpPreDraft: false }, "mtprefill=0&predraft=0");
  const e = { mtpBatchFill: false, other: 1 };
  eq(applyRoomFlags(e, "predraft=0"), { mtpBatchFill: true, other: 1, mtpBatchRefill: true, mtpPreDraft: false }, "applied");
  eq(applyRoomFlags(null, ""), null, "no engine");
});

Deno.test("room.js builds its Qwen engine from the preset and reads no engine flag itself", () => {
  const src = read("room.js");
  if (!/import \{[^}]*roomQwen35Options[^}]*\} from "\.\/engine\/preset\.js"/.test(src)) throw new Error("room.js does not import the preset");
  const create = src.slice(src.indexOf("ai.engine = await Qwen35Engine.create({"), src.indexOf("} else if (M.kind === \"gguf\")"));
  if (!create.includes("...roomQwen35Options(location.search)")) throw new Error("Qwen35Engine.create in room.js does not spread the preset");
  for (const k of ["draftVocab", "draftChain", "specFuse", "kvQ8", "moeFuse", "gpuSample", "argmaxWide", "batchCols"]) {
    if (new RegExp(`\\b${k}:`).test(create)) throw new Error(`room.js sets ${k} itself instead of taking it from the preset`);
  }
  if (!src.includes("applyRoomFlags(ai.engine, location.search)")) throw new Error("room.js does not apply the preset's engine flags");
  for (const f of ["draftvocab", "dvauto", "draftchain", "specfuse", "kv", "moefuse", "moednrows", "gpusample", "argmaxwide", "mtprefill", "predraft", "mtpbatch"]) {
    if (src.includes(`get("${f}")`)) throw new Error(`room.js reads ?${f} itself; it belongs in engine/preset.js`);
  }
});

Deno.test("benchmarks and profilers build their Qwen engines from the same preset", () => {
  for (const p of ["benchmarks/bench.js", "benchmarks/bench_breakdown.js", "benchmarks/bench_enc_probe.js", "benchmarks/bench_pipe_ab2.js", "tests/prof/prof_moe_decode.js", "tests/bench/prof.html", "tests/bench/bench.html", "tests/bench/layer_prof.html", "tests/test_moe_split.js"]) {
    const src = read(p);
    if (!/import \{[^}]*roomQwen35Options[^}]*\} from "[./]*engine\/preset\.js"/.test(src)) throw new Error(`${p} does not import the preset`);
    if (!src.includes("...roomQwen35Options(")) throw new Error(`${p} does not pass the preset to Qwen35Engine.create`);
    if (!src.includes("applyRoomFlags(")) throw new Error(`${p} does not apply the preset's engine flags`);
  }
});

Deno.test("every benchmark and profiler that builds a Qwen engine takes the preset", () => {
  const dirs = ["benchmarks", "tests/prof", "tests/bench"];
  for (const d of dirs) {
    for (const e of Deno.readDirSync(new URL("../../" + d, import.meta.url))) {
      if (!/\.(js|html)$/.test(e.name)) continue;
      const src = read(`${d}/${e.name}`);
      if (src.includes("Qwen35Engine.create(") && !src.includes("...roomQwen35Options(")) throw new Error(`${d}/${e.name} builds a Qwen engine without the room's preset (engine/preset.js)`);
    }
  }
});

// the Deno profilers and benchmarks at the top of tests/ (prof_*.js, bench_*.js) and the memory probe page.
// Exempt, each on purpose: prefill tile sweeps and calibrations that pin their own prefill options, and
// bench_ctx.js (long-context runs; its engine options are being reworked in PR #252, move it then).
const PRESET_EXEMPT = new Set(["tests/prof_prefill.js", "tests/bench_wide_tiles.js", "tests/calib_attn_tile.js", "tests/bench_ctx.js"]);
Deno.test("the tests/ profilers and benchmarks and memprobe.html take the preset too", () => {
  const files = ["memprobe.html"];
  for (const e of Deno.readDirSync(new URL("../../tests", import.meta.url))) if (/^(prof|bench)_.*\.js$/.test(e.name)) files.push(`tests/${e.name}`);
  const missing = files.filter((p) => !PRESET_EXEMPT.has(p) && read(p).includes("Qwen35Engine.create(") && !read(p).includes("...roomQwen35Options("));
  if (missing.length) throw new Error(`build a Qwen engine without the room's preset (engine/preset.js): ${missing.join(", ")}`);
  for (const p of ["tests/prof_ts.js", "tests/prof_moe.js", "tests/prof_overhead.js", "memprobe.html"]) {
    if (!read(p).includes("applyRoomFlags(")) throw new Error(`${p} does not apply the preset's engine flags`);
  }
});
