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

// ---------------- Qwen3.8-Flash-Next (qwen4exp): the real files' headers (tests/fixtures/q38flash), no weights ----------------
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { shardTensors, qwen4expTensorNames, flashShardBytes, flashMemoryShort, openNgramTable } from "../shard.js";
import { parseGGUFHeader, qwen4expShardBytes, qwen4expConfig, GGML_PLE_TABLE } from "../../../engine/gguf.js";
import { ngramConfig, NgramState, dequantRow, rowAddress } from "../../../engine/ngram.js";
import { mergeSplitHeaders } from "../../../room/models.js";
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
  // f32 K / V for its 6 attention layers at 2048 (4 MiB each), DeltaNet state for the other 18, the PLE ring, scratch
  assert.ok(half > w + 6 * 2 * 2048 * c.nKVHeads * c.headDim * 4 && half < w + 2 ** 30, String(half - w));
  assert.ok(half / 2 ** 30 > 34 && half / 2 ** 30 < 40, (half / 2 ** 30).toFixed(1));
  assert.equal(flashMemoryShort(half, { kind: "unified", totalGB: 121, availGB: 100 }), null);
  assert.match(flashMemoryShort(half, { kind: "unified", totalGB: 121, availGB: 30 }), /needs 3\d\.\d GB and this computer has 30\.0 GB of unified memory free/);
  assert.match(flashMemoryShort(half, { kind: "discrete", name: "RTX 5070", freeGB: 11.5 }), /GPU memory and RTX 5070 has 11\.5 GB free/);
  assert.equal(flashMemoryShort(half, { kind: "unknown" }), null, "nothing known: the load's out-of-memory scope decides");
  assert.equal(flashMemoryShort(half, { kind: "unified", totalGB: 36 }), null, "a Mac (no free figure): unchecked");
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
  const h = await openNgramTable({ localPaths: null, urls: MODELS["qwen3.8-flash"].shards }, G);
  assert.equal(h.where, MODELS["qwen3.8-flash"].shards[0]);
});
