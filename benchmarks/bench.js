// Benchmark harness: end-to-end tok/s for prefill and decode, plus output sanity.
// Usage (GB10 / Deno):
//   deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights benchmarks/bench.js
// Env:
//   MODEL=qwen|q38     (default qwen: 0.6B Q8_0; q38: 27B Q4_0, full 64 layers)
//   TOKENS=<n>         decode tokens to time (default 32)
//   VARIANT=coop|legacy  matvec kernel variant (default coop once it exists)
//   ROOM_FLAGS=<query>   q38: the room's switches (engine/preset.js), e.g. ROOM_FLAGS="kv=q8"; unset: the room's
//                        settings. BCOLS / ROWSB override its batch columns / batched rows per workgroup.
import { DenseEngine, makeTokenizer, argmax } from "../engine/engine.js";
import { Qwen35Engine } from "../engine/qwen35.js";
import { parseGGUFHeader, ggufWeights } from "../engine/gguf.js";
import { openGGUF, roomFlags } from "../tests/load_model.js";
import { roomQwen35Options, applyRoomFlags } from "../engine/preset.js";

const MODEL = Deno.env.get("MODEL") || "qwen";
const TOKENS = +(Deno.env.get("TOKENS") || 32);
const VARIANT = Deno.env.get("VARIANT") || "coop";
const WG = +(Deno.env.get("WG") || 256);
const ROWS = +(Deno.env.get("ROWS") || 4);

const openFile = async (path) => {
  const fh = await Deno.open(path);
  return async (off, len) => {
    await fh.seek(off, Deno.SeekMode.Start);
    const out = new Uint8Array(len);
    let got = 0;
    while (got < len) { const n = await fh.read(out.subarray(got)); if (n === null) break; got += n; }
    return out;
  };
};

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: {
  maxBufferSize: adapter.limits.maxBufferSize,
  maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });

let eng, tok, promptIds;
const t0 = performance.now();
if (MODEL === "q38") {
  // converted-weights cache (tests/weight_cache.js): needs --allow-write to fill, WEIGHT_CACHE=0 disables
  const model = openGGUF(Deno.env.get("GGUF") || new URL("../models/q38/model.gguf", import.meta.url).pathname);
  const G = model.G;
  tok = model.tokenizer();
  const L = +(Deno.env.get("LAYERS") || (G.meta["qwen35.block_count"] - (G.meta["qwen35.nextn_predict_layers"] || 0)));
  const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true });
  // the room's engine settings (engine/preset.js), so this number is the room's; the kernel tuning (WG, ROWS)
  // is per device in the room too (autotune), so it stays a knob here
  const flags = roomFlags(), BCOLS = Deno.env.get("BCOLS"), ROWSB = Deno.env.get("ROWSB");
  eng = applyRoomFlags(await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], vocab: G.tensors["token_embd.weight"]?.shape?.[0],
    hasEmbed: true, hasHead: true, maxSeq: 512, matvecVariant: VARIANT, coopWG: WG, coopRows: ROWS, ...roomQwen35Options(flags),
    ...(BCOLS ? { batchCols: +BCOLS, coopRowsB: +(ROWSB || (+BCOLS >= 16 ? 1 : ROWS)) } : ROWSB ? { coopRowsB: +ROWSB } : {}) }), flags);
} else {
  const readAt = await openFile(Deno.env.get("GGUF") || "../models/qwen/model.gguf");
  const G = parseGGUFHeader((await readAt(0, 64 << 20)).buffer, { skipTokenizer: true });
  tok = makeTokenizer(JSON.parse(await Deno.readTextFile("../models/qwen/tokenizer.json")));
  const cfg = JSON.parse(await Deno.readTextFile("../models/qwen/config.json"));
  const L = cfg.num_hidden_layers;
  const weights = await ggufWeights(G, (i) => readAt(i.byteOffset, i.byteLength),
    { lo: 0, hi: L, hasEmbed: true, hasHead: true });
  eng = await DenseEngine.create({ device, cfg, weights, layerRange: [0, L],
    hasEmbed: true, hasHead: true, maxSeq: 512, matvecVariant: VARIANT, coopWG: WG, coopRows: ROWS });
}
console.log(`load: ${((performance.now() - t0) / 1000).toFixed(1)}s  model=${MODEL} variant=${VARIANT}${MODEL === "q38" ? ` batchCols=${eng.NC} draftVocab=${eng.draftVocab || "off"} gpuSample=${eng.gpuSample} kvQ8=${eng.kvQ8}` : ""}`);

const REP = +(Deno.env.get("REP") || 1);
promptIds = tok.encode("The quick brown fox jumps over the lazy dog. ".repeat(REP) + "In a distant future, ");
// prefill: batched (batchCols tokens/pass) for all but the last prompt token
const tp0 = performance.now();
let logits = null;
await eng.prefillTokens(promptIds.slice(0, -1));
logits = await eng.forwardToken(promptIds[promptIds.length - 1]);
const tp1 = performance.now();
// decode
let next = argmax(logits), bad = 0, text = "";
const td0 = performance.now();
for (let i = 0; i < TOKENS; i++) {
  logits = await eng.forwardToken(next);
  next = argmax(logits);
  text += tok.decode([next]);
  if (!Number.isFinite(logits[0]) || !Number.isFinite(logits[next])) bad++;
}
const td1 = performance.now();
const prefillTs = promptIds.length / ((tp1 - tp0) / 1000);
const decodeTs = TOKENS / ((td1 - td0) / 1000);
console.log(`prefill: ${promptIds.length} tok in ${((tp1 - tp0) / 1000).toFixed(2)}s = ${prefillTs.toFixed(2)} tok/s`);
console.log(`decode:  ${TOKENS} tok in ${((td1 - td0) / 1000).toFixed(2)}s = ${decodeTs.toFixed(2)} tok/s`);
console.log(`sanity: nonfinite=${bad}  text: ${JSON.stringify(text.slice(0, 80))}`);
if (bad > 0) { console.error("FAIL: non-finite logits"); Deno.exit(1); }
