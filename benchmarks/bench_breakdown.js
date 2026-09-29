// Where does a pass spend its time? Times full passes with one kernel family skipped at a time (results
// are garbage; timing is what matters), for a single-token decode pass and a batched pass of the
// engine's batch width.
//   deno run --unstable-webgpu --allow-read --allow-env benchmarks/bench_breakdown.js
// env: LAYERS (default: every trunk layer), GGUF (default models/q38/model.gguf), REPS (passes per timing, 6),
//      ROOM_FLAGS: the room's switches (engine/preset.js), e.g. ROOM_FLAGS="fuse=0"; unset: the room's settings
// The engine is built with the room's settings, and the families come from the engine's own pipeline
// list (tests/prof/families.js): every pipeline it created is in exactly one family, so a new kernel is
// never silently counted as "everything else". The baseline is re-timed next to every skipped run
// (the mean of the runs before and after), so drift over the run does not turn into negative costs.
// eng.skip is checked on the GEMV op's own pipeline, so a full-width batched pass's prefill GEMM (which
// runs in place of that GEMV) is counted in the GEMV family; the "prefill GEMM" row then reads ~0.
import { Qwen35Engine } from "../engine/qwen35.js";
import { roomQwen35Options, applyRoomFlags } from "../engine/preset.js";
import { openGGUF, roomFlags, trunkLayers } from "../tests/load_model.js";
import { pipeNames, familiesOf } from "../tests/prof/families.js";

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
const model = openGGUF(Deno.env.get("GGUF") || new URL("../models/q38/model.gguf", import.meta.url).pathname, { skipTokenizer: true });
const G = model.G;
const L = +(Deno.env.get("LAYERS") || trunkLayers(G));
const REPS = +(Deno.env.get("REPS") || 6);
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true });
const flags = roomFlags();
const eng = applyRoomFlags(await Qwen35Engine.create({ device, meta: G.meta, weights, vocab: G.tensors["token_embd.weight"].shape[0], layerRange: [0, L],
  hasEmbed: true, hasHead: true, maxSeq: 512, ...roomQwen35Options(flags) }), flags);
eng._initBatch();
const NC = eng.NC, ids = Array.from({ length: NC }, (_, i) => 10 + i);
console.log(`${L} layers, batchCols ${NC}, room flags "${flags}"`);

async function timeBatch(n = REPS) {
  eng.reset(); eng.pos = 0;
  await eng.embedRunBatch(ids, 0); await device.queue.onSubmittedWorkDone();
  const t0 = performance.now();
  for (let i = 0; i < n; i++) await eng.embedRunBatch(ids, NC * (i + 1));
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - t0) / n;
}
async function timeSingle(n = REPS) {
  // drop the encode-ahead buffer: its key only says whether some skip set was on, not which one
  eng.reset(); eng.pos = 0; eng._fwdPre = null;
  await eng.forwardToken(10);
  const t0 = performance.now();
  for (let i = 0; i < n; i++) await eng.forwardToken(10);
  return (performance.now() - t0) / n;
}

// every pipeline the engine created, by family
const fams = familiesOf(pipeNames(eng));
console.log(`${pipeNames(eng).length} pipelines in ${Object.keys(fams).length} families:`);
for (const [fam, list] of Object.entries(fams)) console.log(`  ${fam.padEnd(28)} ${list.join(" ")}`);

for (const [what, time] of [["single-token pass", timeSingle], [`batched pass (${NC} columns)`, timeBatch]]) {
  eng.skip = null;
  let base = await time();
  console.log(`${what}: ${base.toFixed(1)} ms`);
  for (const [fam, list] of Object.entries(fams)) {
    eng.skip = new Set(list);
    const t = await time();
    eng.skip = null;
    const after = await time(), ref = (base + after) / 2;
    base = after;
    console.log(`  without ${fam.padEnd(28)} ${t.toFixed(1)} ms (baseline ${ref.toFixed(1)})  -> family costs ~${(ref - t).toFixed(1)} ms`);
  }
}
eng.skip = null;
model.close();
