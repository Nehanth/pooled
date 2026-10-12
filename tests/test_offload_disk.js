// GPU check of expert offload's disk tier (engine/expert_store.js disk layers) on the paths tests/test_q38_bits.js
// does not reach, against a resident engine in the same process (the same bits or it fails):
//   - a short prefill frame (an expert-grouped ubatch of fewer than 32 tokens: W * K < nExp, so the engine reads the
//     frame's routing back and fills the region with loadExpertsAsync, in runs)
//   - a batched pass wider than the slot pool (4 columns x 8 experts > 16 slots: prepare() writes the region in runs)
//   - decode with the speculative cut (cutSpec, cutSpecTheta 2: windows that roll back) and the pipelined cut (cutPipe):
//     both plan() on disk layers, so both must prepare() first
//   - a disk read that fails mid-token: the token throws, every later call refuses until reset(), and after reset()
//     the same prompt gives the resident bits again
// Every layer of the 35B MoE is offloaded with its experts on disk (nothing parked: ~3 GB besides the resident ~20).
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights test_offload_disk.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { ExpertStore } from "../engine/expert_store.js";
import { expertDisk } from "../engine/expert_disk.js";
import { argmax } from "../engine/engine.js";
import { openGGUF, MOE_PATH, gpuDevice } from "./load_model.js";

const model = openGGUF(MOE_PATH);
const { device } = await gpuDevice();
const L = model.trunkLayers, tok = model.tokenizer();
const opts = { device, meta: model.meta, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 512, batchCols: 8, coopRowsB: 1 };
const h = (arrs) => { let x = 0x811c9dc5; for (const a of arrs) { const u = new Uint32Array(a.buffer, a.byteOffset, a.length); for (let i = 0; i < u.length; i++) x = Math.imul(x ^ u[i], 0x01000193) >>> 0; } return x.toString(16); };
const ids = tok.encode(await Deno.readTextFile(new URL("./golden/q38_bits_prompt.txt", import.meta.url))).slice(0, 21);   // 20 prefilled: a 16-token grouped frame + a 4-column pass
let ok = true;
const check = (c, m) => { console.log(`${c ? "ok  " : "FAIL"} ${m}`); ok &&= !!c; };

// 20 prompt tokens, then 8 greedy decode tokens: every logit hashed
async function run(eng) {
  eng.reset();
  eng.prefillWide = false;   // the expert-grouped frames, not the wide chunks: a 16-token frame
  await eng.prefillTokens(ids.slice(0, -1));
  let lg = await eng.forwardToken(ids.at(-1));
  const all = [lg], out = [argmax(lg)];
  for (let i = 0; i < 8; i++) { lg = await eng.forwardToken(out.at(-1)); all.push(lg); out.push(argmax(lg)); }
  return { hash: h(all), toks: out.join(",") };
}

let ref;
{
  const eng = await Qwen35Engine.create({ ...opts, weights: await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: false }) });
  ref = await run(eng);
  console.log(`resident: ${ref.hash} tokens ${ref.toks}`);
  eng.destroy?.();
}

// every layer offloaded, experts on disk, a 16-slot pool, no RAM cache; the reader fails while `fail` is set
const base = expertDisk({ readAt: async (shard, off, len) => model.readShard(shard, off, len) });
const R = { fail: false, frames: 0, async readPart(info, e) { if (R.fail) throw new Error("EIO: test disk read failure"); return base.readPart(info, e); } };
const all = Array.from({ length: L }, (_, i) => i);
const st = new ExpertStore(device, { layers: all, slots: 16, disk: { layers: all, reader: R } });
const eng = await Qwen35Engine.create({ ...opts, weights: await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: false, experts: st }) });
const lea = st.loadExpertsAsync.bind(st);
st.loadExpertsAsync = (...a) => { R.frames++; return lea(...a); };

for (const [name, flags] of [["plain", {}], ["cutSpec always", { cutSpec: true, cutSpecTheta: 2 }], ["cutPipe", { cutPipe: true }]]) {
  Object.assign(eng, { cutSpec: false, cutPipe: false, cutSpecTheta: undefined }, flags);
  st.clear(); st.resetStats(); R.frames = 0;
  const r = await run(eng);
  check(r.hash === ref.hash && r.toks === ref.toks, `disk, ${name}: ${r.hash} ${r.hash === ref.hash ? "==" : "!="} resident`);
  check(R.frames > 0, `disk, ${name}: ${R.frames} short frames through loadExpertsAsync`);
  check(st.stats.regionCuts > R.frames && st.stats.diskReads > 0, `disk, ${name}: ${st.stats.regionCuts} region fills, ${st.stats.diskReads} expert reads`);
  if (flags.cutSpec) check(st.stats.specFails > 0, `disk, ${name}: ${st.stats.specFails} speculated layers rolled back`);
}

// a failed read mid-token, in each decode mode
for (const [name, flags] of [["plain", {}], ["cutSpec always", { cutSpec: true, cutSpecTheta: 2 }], ["cutPipe", { cutPipe: true }]]) {
  Object.assign(eng, { cutSpec: false, cutPipe: false, cutSpecTheta: undefined }, flags);
  eng.reset(); eng.prefillWide = false;
  await eng.prefillTokens(ids.slice(0, -1));
  const t = argmax(await eng.forwardToken(ids.at(-1)));
  st.clear();   // a cold pool: the next token misses on the first layer
  R.fail = true;
  let e1 = "", e2 = "", e3 = "";
  try { await eng.forwardToken(t); } catch (e) { e1 = e.message; }
  R.fail = false;
  try { await eng.forwardToken(t); } catch (e) { e2 = e.message; }
  try { await eng.prefillTokens(ids.slice(0, 4)); } catch (e) { e3 = e.message; }
  check(/EIO/.test(e1), `${name}: the token with the failed read throws (${e1.slice(0, 80)})`);
  check(/reset\(\)/.test(e2) && /reset\(\)/.test(e3), `${name}: later calls refuse until reset() (${e2.slice(0, 100)})`);
  st.clear();
  const r = await run(eng);   // reset() first
  check(r.hash === ref.hash, `${name}: after reset() the same bits as resident (${r.hash})`);
}
console.log(st.summary());
Deno.exit(ok ? 0 : 1);
