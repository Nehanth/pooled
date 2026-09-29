// A/B: pipelined (encode-ahead) vs plain decode submits on the 27B, interleaved blocks.
//   deno run --unstable-webgpu --allow-read --allow-env benchmarks/bench_pipe_ab2.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { parseGGUFHeader, qwen35Weights } from "../engine/gguf.js";
import { roomQwen35Options, applyRoomFlags } from "../engine/preset.js";
import { roomFlags } from "../tests/load_model.js";
const openFile = async (p) => { const fh = await Deno.open(p); return async (off, len) => { await fh.seek(off, Deno.SeekMode.Start); const o = new Uint8Array(len); let g = 0; while (g < len) { const n = await fh.read(o.subarray(g)); if (n === null) break; g += n; } return o; }; };
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
const readAt = await openFile(new URL("../models/q38/model.gguf", import.meta.url).pathname);
const G = parseGGUFHeader((await readAt(0, 64 << 20)).buffer, { skipTokenizer: true });
const weights = await qwen35Weights(G, (i) => readAt(i.byteOffset, i.byteLength), { lo: 0, hi: 64, hasEmbed: true, hasHead: true });
// the room's settings (engine/preset.js); ROOM_FLAGS takes the room's ?flags, e.g. ROOM_FLAGS="fuse=0"
const flags = roomFlags();
const eng = applyRoomFlags(await Qwen35Engine.create({ device, meta: G.meta, weights, vocab: 248320, layerRange: [0, 64], hasEmbed: true, hasHead: true, maxSeq: 512, ...roomQwen35Options(flags) }), flags);
const q = device.queue, V = eng.dims.vocab;
const build = (pos) => { const e = device.createCommandEncoder(); for (const L of eng.layers) eng._encodeLayerR(e, L, pos);
  const p = e.beginComputePass(); eng._d(p, "rmsnorm", eng.bgFinalNorm, 256, 256); eng._dop(p, eng.headOp); p.end(); return e.finish(); };
const rb = () => { const e = device.createCommandEncoder(); e.copyBufferToBuffer(eng.logits, 0, eng.stageLogits, 0, V*4); q.submit([e.finish()]); };
const drain = async () => { await eng.stageLogits.mapAsync(GPUMapMode.READ); const o = new Float32Array(eng.stageLogits.getMappedRange(),0,V)[0]; eng.stageLogits.unmap(); return o; };
let pos = 0;
function serialStep() { const t0=performance.now(); eng._setFrame(pos,pos+1); q.writeBuffer(eng.x,0,eng._embedRowF32(10)); const cb=build(pos); q.submit([cb]); rb(); pos++; const cpu=performance.now()-t0; return {cpu, p:drain().then(()=>performance.now()-t0)}; }
let held = null;
function pipeStep() { const t0=performance.now(); eng._setFrame(pos,pos+1); q.writeBuffer(eng.x,0,eng._embedRowF32(10)); q.submit([held]); rb(); pos++; const cpu=performance.now()-t0; const nx=build(pos); const cpu2=performance.now()-t0; return {cpu,cpu2,p:drain().then(()=>{held=nx; return performance.now()-t0;})}; }
async function runSerial(n){ const a=[],c=[]; for(let i=0;i<n;i++){const s=serialStep(); a.push(await s.p); c.push(s.cpu);} return {t:a,c}; }
async function runPipe(n){ const a=[],c=[]; if(!held) held=build(pos); for(let i=0;i<n;i++){const s=pipeStep(); a.push(await s.p); c.push(s.cpu);} return {t:a,c}; }
const med=a=>{a=a.slice().sort((x,y)=>x-y);return a[a.length>>1];};
eng.reset(); pos=0; await runSerial(6); held=null; await runPipe(6);   // warm
const S=[],P=[],SC=[],PC=[];
for (let r=0;r<8;r++){ const s=await runSerial(6); S.push(...s.t); SC.push(...s.c); const p=await runPipe(6); P.push(...p.t); PC.push(...p.c); }
console.log(`serial    n=${S.length} median ${med(S).toFixed(2)} ms  min ${Math.min(...S).toFixed(2)}  | CPU-before-await median ${med(SC).toFixed(2)} ms`);
console.log(`pipelined n=${P.length} median ${med(P).toFixed(2)} ms  min ${Math.min(...P).toFixed(2)}  | CPU-before-await median ${med(PC).toFixed(2)} ms`);
console.log(`delta median ${(med(S)-med(P)).toFixed(2)} ms/token   delta min ${(Math.min(...S)-Math.min(...P)).toFixed(2)} ms`);
