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
