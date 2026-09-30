// dp4a wide prefill GEMM (engine/wgsl/gemm_wide.js gemmDp4aWGSL) vs the f32 wide GEMM on random Q4_0 / Q8_0
// weights: accuracy against a float64 CPU reference (sampled outputs) and GPU time per GEMM.
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env bench_dp4a_gemm.js
//   SHAPES="17408x5120,5120x17408"  COLS=256  ITERS=20  TILE='{"BM":64,"BN":64,"TM":4,"TN":4,"KB":2}'
import { gemmWideWGSL, wideTileConfig, gemmDp4aWGSL, dp4aTileConfig, probeDp4a } from "../engine/wgsl/gemm_wide.js";
import { f16ToF32, f32ToF16 } from "../engine/gguf.js";
const env = (k, d) => Deno.env.get(k) ?? d;
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
let errs = 0; device.addEventListener("uncapturederror", (e) => { if (errs++ < 3) console.error("GPU ERROR:", e.error.message.slice(0, 400)); });
if (!(await probeDp4a(device))) { console.log("SKIP: no dot4I8Packed on this device"); Deno.exit(0); }
const COLS = +env("COLS", 256), ITERS = +env("ITERS", 20);
const wc = wideTileConfig({}, 16384), dc = dp4aTileConfig(JSON.parse(env("TILE", "{}")), 16384);
const pre = `struct BShape { dOut: u32, dIn: u32, xs4: u32, ys: u32 };\nstruct MC { n: u32, s0: u32, s1: u32, s2: u32 };\n`;
const mod = device.createShaderModule({ code: pre + gemmWideWGSL(wc) + gemmDp4aWGSL(dc) });
const info = await mod.getCompilationInfo(); for (const m of info.messages) if (m.type === "error") { console.log("WGSL:", m.lineNum, m.message); Deno.exit(1); }
const pipe = (name) => device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: name } });
const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const buf = (a, usage = S) => { const b = device.createBuffer({ size: Math.max(16, Math.ceil(a.byteLength / 16) * 16), usage }); device.queue.writeBuffer(b, 0, a); return b; };
const bg = (p, bs) => device.createBindGroup({ layout: p.getBindGroupLayout(1), entries: bs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
let seed = 7; const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
async function read(b, n) {
  const st = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const e = device.createCommandEncoder(); e.copyBufferToBuffer(b, 0, st, 0, n * 4); device.queue.submit([e.finish()]);
  await st.mapAsync(GPUMapMode.READ); const r = new Float32Array(st.getMappedRange().slice(0)); st.destroy(); return r;
}
let fail = 0;
for (const fmt of ["q4", "q8"]) for (const sh of env("SHAPES", "17408x5120,5120x17408").split(",")) {
  const [dOut, dIn] = sh.split("x").map(Number), nb = dIn / 32, W = fmt === "q4" ? 4 : 8;
  const qs = new Uint32Array(dOut * nb * W); for (let i = 0; i < qs.length; i++) qs[i] = (rnd() * 4294967296) >>> 0;
  const sch = new Uint16Array(dOut * nb + (dOut * nb) % 2); for (let i = 0; i < dOut * nb; i++) sch[i] = f32ToF16((0.5 + rnd()) * 0.01);
  const x = new Float32Array(COLS * dIn); for (let i = 0; i < x.length; i++) x[i] = gauss() * (i % 97 === 0 ? 8 : 1);   // a few outliers
  const bq = buf(qs), bs = buf(new Uint32Array(sch.buffer)), bx = buf(x), by = device.createBuffer({ size: COLS * dOut * 4, usage: S });
  const bxq = device.createBuffer({ size: COLS * dIn, usage: S }), bxd = device.createBuffer({ size: COLS * nb * 4, usage: S });
  const shp = buf(new Uint32Array([dOut, dIn, dIn / 4, dOut]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const mc = buf(new Uint32Array([dIn, dIn / 4, 0, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const pW = pipe(`gemm_w_${fmt}`), pD = pipe(`gemm_d_${fmt}`), pQ = pipe("quant_q8_w");
  const bgW = bg(pW, [bq, bs, bx, by, shp]), bgD = bg(pD, [bq, bs, bxq, bxd, by, shp]), bgQ = bg(pQ, [bx, bxq, bxd, mc]);
  const runW = (p) => { p.setPipeline(pW); p.setBindGroup(1, bgW); p.dispatchWorkgroups(Math.ceil(dOut / wc.BM), COLS / wc.BN); };
  const runQ = (p) => { p.setPipeline(pQ); p.setBindGroup(1, bgQ); p.dispatchWorkgroups(Math.ceil(nb / 32), COLS); };
  const runD = (p) => { p.setPipeline(pD); p.setBindGroup(1, bgD); p.dispatchWorkgroups(Math.ceil(dOut / dc.BM), COLS / dc.BN); };
  const time = async (fns) => {
    let best = 1e9;
    for (let rep = 0; rep < 3; rep++) {
      await device.queue.onSubmittedWorkDone();
      const t0 = performance.now(); const e = device.createCommandEncoder(); const p = e.beginComputePass();
      for (let i = 0; i < ITERS; i++) for (const f of fns) f(p);
      p.end(); device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone();
      best = Math.min(best, (performance.now() - t0) / ITERS);
    }
    return best;
  };
  // accuracy: sampled outputs vs float64
  const ref = (row, col) => {
    let s = 0;
    for (let b = 0; b < nb; b++) {
      const g = row * nb + b, d = f16ToF32(sch[g]);
      for (let w = 0; w < W; w++) {
        const word = qs[g * W + w];
        for (let i = 0; i < 4; i++) {
          if (fmt === "q4") {
            s += d * (((word >>> (8 * i)) & 15) - 8) * x[col * dIn + b * 32 + 4 * w + i];
            s += d * (((word >>> (8 * i + 4)) & 15) - 8) * x[col * dIn + b * 32 + 16 + 4 * w + i];
          } else s += d * ((((word >>> (8 * i)) & 255) << 24) >> 24) * x[col * dIn + b * 32 + 4 * w + i];
        }
      }
    }
    return s;
  };
  const samp = Array.from({ length: 64 }, (_, i) => [(i * 7919 + 13) % dOut, (i * 131 + 5) % COLS]);
  const R = samp.map(([r, c]) => ref(r, c)), sc = Math.max(...R.map(Math.abs));
  const e1 = device.createCommandEncoder(); let p = e1.beginComputePass(); runW(p); p.end(); device.queue.submit([e1.finish()]);
  const yW = await read(by, COLS * dOut);
  const e2 = device.createCommandEncoder(); p = e2.beginComputePass(); runQ(p); runD(p); p.end(); device.queue.submit([e2.finish()]);
  const yD = await read(by, COLS * dOut);
  const err = (y) => Math.max(...samp.map(([r, c], i) => Math.abs(y[c * dOut + r] - R[i]))) / sc;
  const eW = err(yW), eD = err(yD);
  // full-output relDiff dp4a vs f32 wide
  let md = 0, mx = 0; for (let i = 0; i < yW.length; i++) { md = Math.max(md, Math.abs(yW[i] - yD[i])); mx = Math.max(mx, Math.abs(yW[i])); }
  const tW = await time([runW]), tQ = await time([runQ]), tD = await time([runD]);
  const fl = 2 * dOut * dIn * COLS / 1e9;
  console.log(`${fmt} ${dOut}x${dIn} x${COLS}: f32 wide ${tW.toFixed(3)} ms (${(fl / tW).toFixed(2)} TFLOPS) | dp4a ${tD.toFixed(3)} ms (${(fl / tD).toFixed(2)} TOPS) + quant ${tQ.toFixed(3)} ms => ${(tW / (tD + tQ)).toFixed(2)}x | relErr vs f64: f32 ${eW.toExponential(2)} dp4a ${eD.toExponential(2)} | dp4a vs f32 ${(md / mx).toExponential(2)}`);
  if (!(eD < 2e-2) || !(eW < 1e-4)) fail++;
  for (const b of [bq, bs, bx, by, bxq, bxd]) b.destroy();
}
console.log(fail || errs ? `DP4A GEMM FAIL (errors ${errs})` : "DP4A GEMM PASS");
Deno.exit(fail || errs ? 1 : 0);
