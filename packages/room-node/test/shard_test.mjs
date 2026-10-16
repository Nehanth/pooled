// shard.js without a GPU: how many layers a model has, and the layer-range check loadShard runs first
import test from "node:test";
import assert from "node:assert/strict";
import { modelLayers, checkRange, loadShard } from "../shard.js";
import { MODELS } from "../../../room/models.js";

// a source stub: the dense models' config.json, a qwen35 GGUF index (its meta only)
const stub = (key, { cfg = null, meta = {} } = {}) => ({ M: MODELS[key], cfg: async () => cfg, header: async () => ({ meta }) });

test("modelLayers: a dense GGUF's count comes from config.json, whatever its name", async () => {
  // qwen3-0.6b's GGUF has qwen3.* keys, not qwen35.*: solo.mjs read qwen35.block_count for every
  // model but qwen3-1.7b, got NaN, and loaded [0, NaN]: no layers, token 271 ("\n\n") forever
  for (const [key, n] of [["qwen3-0.6b", 28], ["qwen3-1.7b", 28], ["qwen3-4b", 36]])
    assert.equal(await modelLayers(stub(key, { cfg: { num_hidden_layers: n }, meta: { "qwen3.block_count": n } })), n);
});

test("modelLayers: a qwen35 GGUF's count is block_count less the MTP layers", async () => {
  assert.equal(await modelLayers(stub("qwen3.8-27b", { meta: { "qwen35.block_count": 65, "qwen35.nextn_predict_layers": 1 } })), 64);
  assert.equal(await modelLayers(stub("qwen3.6-35b-moe", { meta: { "qwen35.block_count": 40 } })), 40);
});

test("checkRange: whole layer numbers, 0 <= lo <= hi", () => {
  for (const r of [[0, 28], [10, 20], [0, 0]]) assert.doesNotThrow(() => checkRange(r));
  for (const r of [[0, NaN], [NaN, 4], [0, 2.5], [-1, 4], [5, 4], undefined, [0]]) assert.throws(() => checkRange(r), /bad layer range/);
});

test("loadShard refuses a NaN range before it asks for a GPU", async () => {
  const src = stub("qwen3-0.6b");
  await assert.rejects(loadShard({ modelKey: "qwen3-0.6b", range: [0, NaN], hasEmbed: true, hasHead: true, src, selfTest: false }), /bad layer range \[0, NaN\]/);
});

test("loadShard refuses a Flash format it can't load before it asks for a GPU or streams", async () => {
  const src = { M: { kind: "qwen4exp" }, header: () => { throw new Error("streamed"); } };
  await assert.rejects(loadShard({ modelKey: "qwen3.8-flash", range: [0, 4], hasEmbed: true, hasHead: false, src, selfTest: false,
    flash: { qsa: 1, kv: "q8", ring: 1024 } }), /f32 or f16/);
});

// ---------------- Qwen3.8-Flash-Next (qwen4exp): the real files' headers (tests/fixtures/q38flash), no weights ----------------
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { shardTensors, qwen4expTensorNames, flashShardBytes, flashCpuBytes, flashMemoryShort, openNgramTable } from "../shard.js";
import { parseGGUFHeader, qwen4expShardBytes, qwen4expConfig, GGML_PLE_TABLE } from "../../../engine/gguf.js";
import { ngramConfig, NgramState, dequantRow, rowAddress } from "../../../engine/ngram.js";
import { mergeSplitHeaders, flashRoomMem, flashLayerCost, ckptSlotsAt, FLASH_QSA_LAYER_BYTES } from "../../../room/models.js";
import { ggufFromFixture } from "../../../tests/helpers/gguf_fixture.js";

const FIX = JSON.parse(fs.readFileSync(new URL("../../../tests/fixtures/q38flash/headers.json", import.meta.url), "utf8"));
const flashHeader = () => mergeSplitHeaders(FIX.files.slice(0, 3).map((f) => parseGGUFHeader(ggufFromFixture(f), { skipTokenizer: true })));

test("qwen4exp modelLayers: 48 from the header (the MTP file is separate)", async () => {
  const G = flashHeader();
  assert.equal(await modelLayers({ M: MODELS["qwen3.8-flash"], header: async () => G }), 48);
});

test("qwen4exp shardTensors: what qwen4expWeights reads (no QSA indexer, no n-gram table), each file's tensors with their shard", () => {
  const G = flashHeader(), sum = (ts) => ts.reduce((a, t) => a + t.byteLength, 0);
  for (const o of [{ lo: 0, hi: 24, hasEmbed: true, hasHead: true }, { lo: 24, hi: 48, hasEmbed: false, hasHead: false }, { lo: 1, hi: 2, hasEmbed: false, hasHead: false }]) {
    const ts = shardTensors(G, "qwen4exp", o);
    assert.equal(sum(ts), qwen4expShardBytes(G, { ...o, qsa: false }), JSON.stringify(o));
  }
  // the whole model: every tensor of the three files except the n-gram table and the indexer
  const all = shardTensors(G, "qwen4exp", { lo: 0, hi: 48, hasEmbed: true, hasHead: true });
  const skip = Object.keys(G.tensors).filter((n) => n === GGML_PLE_TABLE || /\.indexer\./.test(n));
  assert.equal(all.length, Object.keys(G.tensors).length - skip.length);
  assert.ok(skip.length > 1 && skip.includes(GGML_PLE_TABLE));
  assert.ok(all.some((t) => t.shard === 2) && all.some((t) => !t.shard));
  assert.ok(!qwen4expTensorNames(G, 3).some((n) => n.includes("indexer")) && qwen4expTensorNames(G, 3).some((n) => n.includes("attn_q.")));
  assert.ok(qwen4expTensorNames(G, 1).some((n) => n.includes("ple_key")), "layer 1 carries the PLE block");
});

test("qwen4exp flashShardBytes / flashMemoryShort: a device short of memory is told before the load", () => {
  const G = flashHeader(), c = qwen4expConfig(G.meta);
  const half = flashShardBytes(G, { lo: 0, hi: 24, hasHead: true, ctx: 2048 });
  const w = qwen4expShardBytes(G, { lo: 0, hi: 24, hasEmbed: false, hasHead: true, qsa: false });
  const layer = (i) => qwen4expShardBytes(G, { lo: i, hi: i + 1, hasEmbed: false, hasHead: false, qsa: false });
  const staged = Math.max(...Array.from({ length: 24 }, (_, i) => layer(i)), w - qwen4expShardBytes(G, { lo: 0, hi: 24, hasEmbed: false, hasHead: false, qsa: false }));
  const cpu = flashCpuBytes(G, { hasHead: true });
  // the CPU embedding (~0.36 GB) and the biggest part's upload staging come on top of the weights
  assert.ok(cpu / 2 ** 30 > 0.3 && cpu / 2 ** 30 < 0.5, (cpu / 2 ** 30).toFixed(2));
  assert.equal(flashCpuBytes(G, {}), 0, "a middle device keeps no embedding");
  const extra = half - w - staged - cpu;
  // f32 K / V for its 6 attention layers at 2048 (4 MiB each), DeltaNet state for the other 18, the PLE ring, scratch
  assert.ok(extra > 6 * 2 * 2048 * c.nKVHeads * c.headDim * 4 && extra < 2 ** 30, String(extra));
  assert.ok(half / 2 ** 30 > 35 && half / 2 ** 30 < 42, (half / 2 ** 30).toFixed(1));
  assert.ok(flashShardBytes(G, { lo: 0, hi: 24, hasEmbed: true, hasHead: false, ctx: 2048 }) > flashShardBytes(G, { lo: 0, hi: 24, ctx: 2048 }) + 0.3 * 2 ** 30, "the host's embedding counts");
  assert.equal(flashMemoryShort(half, { kind: "unified", totalGB: 121, availGB: 100 }), null);
  assert.match(flashMemoryShort(half, { kind: "unified", totalGB: 121, availGB: 30 }), /needs 3\d\.\d GB and this computer has 30\.0 GB of unified memory free/);
  assert.match(flashMemoryShort(half, { kind: "discrete", name: "RTX 5070", freeGB: 11.5 }), /GPU memory and RTX 5070 has 11\.5 GB free/);
  // on a discrete GPU the embedding is in RAM: only the GPU part is weighed against its free memory
  const gpu = half - cpu, fits = { kind: "discrete", name: "big", freeGB: (gpu + cpu / 2) / 2 ** 30 + 0.5 };
  assert.equal(flashMemoryShort(half, fits, { cpuBytes: cpu }), null);
  assert.ok(flashMemoryShort(half, fits), "without cpuBytes it counts all of it");
  assert.equal(flashMemoryShort(half, { kind: "unknown" }), null, "nothing known: the load's out-of-memory scope decides");
  // a discrete GPU with expert offload: what it parks (and the disk tier's RAM cache) counts against the system's RAM
  const parked = 30 * 2 ** 30, gpuPart = 8 * 2 ** 30, rtx = { kind: "discrete", name: "RTX 5070", freeGB: 11.5 };
  assert.equal(flashMemoryShort(gpuPart + parked, rtx, { cpuBytes: parked, sysFreeGB: 40 }), null, "30 GB parked in 40 GB of free RAM");
  assert.match(flashMemoryShort(gpuPart + parked, rtx, { cpuBytes: parked, sysFreeGB: 24 }), /needs 30\.0 GB of system RAM .* 24\.0 GB free/);
  assert.equal(flashMemoryShort(gpuPart + parked, rtx, { cpuBytes: parked }), null, "system RAM not known: unchecked");
  assert.equal(flashMemoryShort(half, { kind: "unified", totalGB: 36 }), null, "a Mac (no free figure): unchecked");
});

// the load-time check and the planner size a range's state with one model (room/models.js flashRoomMem /
// flashLayerCost), so a deal the planner made is never refused at load for a different formula
test("qwen4exp flashShardBytes: the state and checkpoint reserve are the planner's, at 2048 and 32K, QSA off and on", () => {
  const G = flashHeader();
  const fixed = (o) => flashShardBytes(G, { ...o, ctx: 2048, qsa: false, slots: 0 }) - flashRoomMem(G, 2048, { qsa: false, range: [o.lo, o.hi] }).need;
  for (const [lo, hi] of [[0, 24], [24, 48], [0, 3], [3, 4], [0, 48]]) for (const qsa of [false, true]) for (const ctx of [2048, 32768]) {
    const o = { lo, hi, hasHead: lo === 0, ctx, qsa };
    const b = flashShardBytes(G, o);
    const m = flashRoomMem(G, ctx, { qsa, range: [lo, hi], ckptSlots: ckptSlotsAt(ctx) });
    // the weights (with the indexer when qsa), the upload staging, the CPU part and scratch don't depend on ctx
    const w = qwen4expShardBytes(G, { lo, hi, hasEmbed: false, hasHead: lo === 0, qsa }) - qwen4expShardBytes(G, { lo, hi, hasEmbed: false, hasHead: lo === 0, qsa: false });
    const staged = (q) => Math.max(...Array.from({ length: hi - lo }, (_, k) => qwen4expShardBytes(G, { lo: lo + k, hi: lo + k + 1, hasEmbed: false, hasHead: false, qsa: q })),
      lo === 0 ? qwen4expShardBytes(G, { lo: hi, hi, hasEmbed: false, hasHead: true, qsa: q }) - G.tensors["token_embd.weight"].byteLength : 0);
    assert.equal(b - m.need, fixed({ lo, hi, hasHead: lo === 0 }) + w + staged(qsa) - staged(false), `[${lo}, ${hi}) ctx ${ctx} qsa ${qsa}`);
    // and the planner's per-layer costs hold exactly the same state and reserve for those layers
    const lc = flashLayerCost(G, ctx, { qsa });
    let st = 0; for (let i = lo; i < hi; i++) st += lc.state[i];
    assert.equal(st, m.state);
    assert.equal(st * lc.slots, m.reserve);
  }
  // 2048 dense: what it was before the shared model (no reserve at 2048); 32K: the 2-slot reserve makes [0, 24) ~2.4 GB more
  const d = flashShardBytes(G, { lo: 0, hi: 24, hasHead: true, ctx: 2048 }), l = flashShardBytes(G, { lo: 0, hi: 24, hasHead: true, ctx: 32768, qsa: true });
  assert.ok(d / 2 ** 30 > 35 && d / 2 ** 30 < 36, (d / 2 ** 30).toFixed(2));
  assert.ok((l - d) / 2 ** 30 > 2.3 && (l - d) / 2 ** 30 < 2.6, ((l - d) / 2 ** 30).toFixed(2));
  // [0, 3) has no QSA layer, [3, 4) one: at 2048 the one layer holds less than the three (8.0 vs 9.7 MiB), at 32K
  // over 13x more (130 vs 9.7 MiB: its K / V grow with ctx, the DeltaNet state does not)
  const s03 = flashRoomMem(G, 32768, { range: [0, 3] }).state, s34 = flashRoomMem(G, 32768, { range: [3, 4] }).state;
  assert.ok(s34 > 13 * s03 && flashRoomMem(G, 2048, { range: [3, 4] }).state < s03, `${s03} ${s34}`);
  assert.equal(s03, flashRoomMem(G, 2048, { range: [0, 3] }).state, "DeltaNet + the PLE ring: fixed");
});

test("qwen4exp with QSA (ai-load.flash): the indexer is read and counted, with its raw-key ring and pooled keys", () => {
  const G = flashHeader(), sum = (ts) => ts.reduce((a, t) => a + t.byteLength, 0);
  for (const o of [{ lo: 0, hi: 24, hasEmbed: true, hasHead: true }, { lo: 24, hi: 48, hasEmbed: false, hasHead: false }, { lo: 0, hi: 3, hasEmbed: false, hasHead: false }])
    assert.equal(sum(shardTensors(G, "qwen4exp", { ...o, qsa: true })), qwen4expShardBytes(G, { ...o, qsa: true }), JSON.stringify(o));
  assert.ok(qwen4expTensorNames(G, 3, true).some((n) => n.includes("indexer")) && !qwen4expTensorNames(G, 2, true).some((n) => n.includes("indexer")));
  // 12 QSA layers, 6 in [0, 24): each adds its indexer weights (3.13 MiB) and, at 2048 positions, the 1024-row ring and
  // 512 pooled blocks of 128 f16 (FLASH_QSA_LAYER_BYTES), plus the QSA score scratch once (room/models.js flashRoomMem)
  const d = flashShardBytes(G, { lo: 0, hi: 24, hasHead: true, ctx: 2048, qsa: true }) - flashShardBytes(G, { lo: 0, hi: 24, hasHead: true, ctx: 2048 });
  const sc = flashRoomMem(G, 2048, { qsa: true, range: [0, 24] }).bytes.scores;
  assert.equal(d, 6 * FLASH_QSA_LAYER_BYTES + sc);
  assert.equal(flashShardBytes(G, { lo: 0, hi: 3, ctx: 2048, qsa: true }), flashShardBytes(G, { lo: 0, hi: 3, ctx: 2048 }), "no QSA layer in [0, 3)");
  // the planner's per-layer costs (room/models.js flashLayerCost) count the same QSA bytes for the whole model
  assert.equal(flashLayerCost(G, 2048, { qsa: true }).total - flashLayerCost(G, 2048, { qsa: false }).total,
    flashShardBytes(G, { lo: 0, hi: 48, ctx: 2048, qsa: true }) - flashShardBytes(G, { lo: 0, hi: 48, ctx: 2048 }));
  assert.equal(qwen4expShardBytes(G, { lo: 3, hi: 4, hasEmbed: false, hasHead: false }) - qwen4expShardBytes(G, { lo: 3, hi: 4, hasEmbed: false, hasHead: false, qsa: false }), 3277824);
});

test("qwen4exp openNgramTable: rows from the local shard at the table's offset (its own file handle), close() releases it", async () => {
  const G = flashHeader(), cfg = ngramConfig(G);
  assert.equal(cfg.shard, 0);
  // a sparse stand-in for shard 1: only the 16 rows of one token written, at their real offsets
  const ids = new NgramState(cfg).push([9707]);   // the token's 16 global rows
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "flash-ngram-")), file = path.join(dir, "shard1.gguf");
  const fd = fs.openSync(file, "w");
  const raw = [];
  for (let h = 0; h < ids.length; h++) {
    const b = new Uint8Array(cfg.rowBytes);
    for (let i = 0; i < b.length; i++) b[i] = (h * 31 + i * 7) & 255;
    // a 1.0 f16 scale at the start of each Q4_0 block, the nibbles arbitrary
    for (let blk = 0; blk < cfg.width / 32; blk++) { b[blk * 18] = 0; b[blk * 18 + 1] = 0x3c; }
    const [, off] = rowAddress(cfg, "tensor", ids[h]);
    fs.writeSync(fd, b, 0, b.length, off);
    raw.push(b);
  }
  fs.closeSync(fd);
  const logs = [];
  const t = await openNgramTable({ localPaths: [file, null, null], urls: [] }, G, (m) => logs.push(m));
  try {
    assert.equal(t.where, file);
    assert.match(logs[0], /n-gram table: 26\.8 GB, rows read from /);
    const got = await t.rows(ids), want = new Float32Array(ids.length * cfg.width);
    raw.forEach((b, h) => dequantRow(cfg, b, 0, want, h * cfg.width));
    assert.deepEqual([...got], [...want]);
  } finally { await t.close(); fs.rmSync(dir, { recursive: true, force: true }); }
  assert.equal(t.close, null, "closed once");
  // no local copy: HTTP range requests to the shard's URL
  const hl = [];
  const h = await openNgramTable({ localPaths: null, urls: MODELS["qwen3.8-flash"].shards }, G, (m) => hl.push(m));
  assert.equal(h.where, MODELS["qwen3.8-flash"].shards[0]);
  assert.match(hl[0], /NOT on this disk .* read by HTTP range from https:\/\/huggingface\.co\/.* slow/, "the fallback says so");
});
