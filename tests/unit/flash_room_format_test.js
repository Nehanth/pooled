// Qwen3.8-Flash-Next with QSA in rooms (room/models.js): what a build says it loads (meta.flashLong), the format the host
// picks for the room (ai-load.flash), a worker's reading of it, what its ai-ready reports (ctx, fsig) and the host's
// check of that, and the planner's QSA constant against the engine's own accounting (engine/qwen4exp_mem.js).
// No GPU.   deno test --allow-read tests/unit/flash_room_format_test.js
import { flashLongMeta, flashRoomFormat, flashLoadOpts, flashReadyFields, readyCtxShort, flashQsaIn, roomBytes, SHAPE,
  FLASH_QSA_RING, FLASH_QSA_LAYER_BYTES, flashLayerCost } from "../../room/models.js";
import { qwen4expMemory, QSA_RAW_RING } from "../../engine/qwen4exp_mem.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const throws = (f, re, m) => { try { f(); } catch (e) { if (re && !re.test(e.message)) throw new Error(`${m || "wrong error"}: ${e.message}`); return; } throw new Error(m || "expected a throw"); };
const F = "qwen3.8-flash";

Deno.test("flash format: meta.flashLong says the longest context one f32 K / V buffer binds, at most QSA's", () => {
  eq(flashLongMeta(128), { ctx: 65536, kv: ["f32"], batch: 1 });
  eq(flashLongMeta(128, { batch: false }).batch, 0, "a node with POOLED_FLASH_BATCH=0");
  eq(flashLongMeta(512).ctx, 262144);
  eq(flashLongMeta(1e9).ctx, 411000, "QSA_MAX_SEQ");
  eq(flashLongMeta(undefined).ctx, 65536, "no maxBindMB: WebGPU's default 128 MiB");
});

Deno.test("flash format: QSA only when every dealt device loads it; one format for the room", () => {
  const nw = { flashLong: flashLongMeta(128) }, old = { webgpu: true };
  eq(flashRoomFormat(F, [nw, nw]), { qsa: 1, kv: "f32", ring: FLASH_QSA_RING });
  eq(flashRoomFormat(F, [nw]), { qsa: 1, kv: "f32", ring: 1024 }, "solo");
  eq(flashRoomFormat(F, [nw, old]), null, "an older worker: the dense engine everywhere");
  eq(flashRoomFormat(F, [old, nw]), null, "an older host device");
  eq(flashRoomFormat(F, [nw, undefined]), null, "a device whose meta is unknown");
  eq(flashRoomFormat("qwen3.8-27b", [nw, nw]), null, "Flash only");
  eq(flashRoomFormat(F, []), null);
  // a peer's meta is its own word: anything but a sane flashLong counts as an older build
  for (const fl of [1, "yes", { ctx: 65536 }, { ctx: 1024, kv: ["f32"] }, { ctx: 65536, kv: ["f16"] }, { ctx: 65536, kv: "f32" }])
    eq(flashRoomFormat(F, [nw, { flashLong: fl }]), null, JSON.stringify(fl));
  eq(FLASH_QSA_RING, QSA_RAW_RING, "the engine's default ring");
});

Deno.test("flash format: a worker's reading of ai-load.flash", () => {
  eq(flashLoadOpts(undefined), null, "an older host: dense");
  eq(flashLoadOpts(null), null);
  eq(flashLoadOpts({ qsa: 0 }), null);
  eq(flashLoadOpts({ qsa: 1, kv: "f32", ring: 1024 }), { kv: "f32", ring: 1024 });
  eq(flashLoadOpts({ qsa: 1 }), { kv: "f32", ring: 1024 }, "defaults");
  throws(() => flashLoadOpts({ qsa: 1, kv: "f16" }), /f32 only/, "f16 is not loaded by this build yet");
  for (const ring of [1000, 0, -1024, 32, 2 ** 20, "1024"]) throws(() => flashLoadOpts({ qsa: 1, ring }), /bad QSA ring/, String(ring));
});

Deno.test("flash format: ai-ready's ctx / fsig, and the host's check of them", () => {
  const sig = { v: 1, arch: "qwen4exp", lo: 24, hi: 48, kv: "f32", qsa: true, qsaRing: 1024, dim: 2560 };
  const eng = { maxSeq: 2048, stateSignature: () => sig };
  eq(flashReadyFields(eng, F), { ctx: 2048, fsig: sig });
  eq(flashReadyFields({ maxSeq: 2048 }, F), { ctx: 2048 }, "no signature");
  eq(flashReadyFields({ maxSeq: 2048, stateSignature: () => { throw new Error("x"); } }, F), { ctx: 2048 });
  eq(flashReadyFields(eng, "qwen3.8-27b"), {}, "Flash only");
  eq(flashReadyFields(null, F), {});
  const qsa = { qsa: 1, kv: "f32", ring: 1024 };
  eq(readyCtxShort(F, 2048, { ctx: 2048, fsig: sig }, qsa), null);
  eq(readyCtxShort(F, 2048, {}, null), null, "an older worker (no ctx) holds 2048: enough for a 2048 room");
  ok(/holds 2048 positions and the room runs 8192/.test(readyCtxShort(F, 8192, {}, qsa)), "...and not for a longer one");
  ok(/holds 4096/.test(readyCtxShort(F, 8192, { ctx: 4096 }, qsa)));
  ok(/K \/ V in f16/.test(readyCtxShort(F, 2048, { ctx: 2048, fsig: { ...sig, kv: "f16" } }, qsa)), "a format the host did not ask for");
  ok(/512-row ring/.test(readyCtxShort(F, 2048, { ctx: 2048, fsig: { ...sig, qsaRing: 512 } }, qsa)));
  eq(readyCtxShort(F, 2048, { ctx: 2048, fsig: { ...sig, lo: 24, hi: 27, qsa: false, qsaRing: 0 } }, qsa), null, "a range with no QSA layer reports qsa false");
  ok(/runs dense attention; the room asked for QSA/.test(readyCtxShort(F, 2048, { ctx: 2048, fsig: { ...sig, qsa: false, qsaRing: 0 } }, qsa)),
    "a dense engine on layers with QSA layers, in a QSA room");
  ok(/runs QSA; the room asked for dense/.test(readyCtxShort(F, 2048, { ctx: 2048, fsig: sig }, null)), "a QSA engine in a dense room");
  eq(readyCtxShort(F, 2048, { ctx: 2048, fsig: { ...sig, qsa: false, qsaRing: 0 } }, null), null, "dense in a dense room");
  eq([flashQsaIn(F, 0, 3), flashQsaIn(F, 0, 4), flashQsaIn(F, 44, 47), flashQsaIn(F, 47, 48), flashQsaIn(F, undefined, 4)], [false, true, false, true, true]);
  eq(readyCtxShort("qwen3.8-27b", 65536, { ctx: 2048 }, null), null, "Flash only");
  eq(readyCtxShort(F, undefined, { ctx: 1 }, null), null, "no room context known");
});

// the model's shape (qwen4expConfig of the bartowski Q4_0 header; tests/unit/q38flash_mem_test.js)
const cfg = { L: 48, dim: 2560, hc: 4, hDim: 10240, nHeads: 24, nKVHeads: 2, headDim: 256,
  qsa: Array.from({ length: 48 }, (_, i) => (i + 1) % 4 === 0),
  ssm: { convK: 4, stateSize: 128, nKHeads: 16, nVHeads: 48, inner: 6144 },
  ple: { layers: [1], ngram: 3, convK: 4 } };

Deno.test("flash format: the planner's QSA bytes are the engine's QSA state at 2048 plus the indexer weights", () => {
  const a = qwen4expMemory(cfg, { ctx: 2048, qsa: true }), b = qwen4expMemory(cfg, { ctx: 2048, qsa: false });
  eq(FLASH_QSA_LAYER_BYTES * 12, (a.state - b.state) + 12 * 3277824, "ring + pooled keys + indexer, 12 QSA layers");
  // the planner's per-layer costs (roomBytes -> flashLayerCost) count QSA at 2048, since the rooms load it there
  const on = flashLayerCost(F, 2048, { qsa: true }), off = flashLayerCost(F, 2048, { qsa: false });
  eq(roomBytes(F, 2048, "f16").layerCost.total, on.total, "roomBytes counts it for Flash");
  ok(Math.abs(on.total - off.total - (FLASH_QSA_LAYER_BYTES * 12 + on.once)) < 1, "ring + pooled keys + indexer, 12 QSA layers, and the score scratch once");
  ok((on.total - off.total) / 48 < 2 ** 20, "under 1 MiB a layer");
  eq(roomBytes("qwen3.8-27b", 2048, "f16").layerBytes, SHAPE["qwen3.8-27b"].layer + 2048 * SHAPE["qwen3.8-27b"].kvPos.f16, "and only for Flash");
});
