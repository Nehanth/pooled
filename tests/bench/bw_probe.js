// Achievable memory bandwidth and dispatch / pass / blit floors through WebGPU, no model.
// Answers roadmap/24 gate 2 ("is there a gap at all?") and sizes the per-dispatch overhead (step 3 of
// docs/archive/research/mac-metal-plan.md). Same module for Deno and the browser:
//   Deno:   cd tests && deno run --unstable-webgpu --allow-read --allow-env bench/bw_probe.js
//   Chrome: node tests/bench/bw_probe.mjs  (tests/bench/bw_probe.html)
// env / query: MB (streamed buffer size, 1024), REPS (8)
//
// Bandwidth: one buffer of MB MiB (far beyond any GPU cache) streamed by a grid-stride read kernel,
//   (a) as array<vec4<u32>> (one 16 B load per thread per iteration), (b) as array<u32> (4 B loads, the
//   shape of the coop Q4 GEMV), at several workgroup sizes and grids; (c) copyBufferToBuffer. Best of REPS
//   interleaved rounds; each figure is the slope between a submit of 2 and a submit of 6 back-to-back passes
//   (timed submit -> onSubmittedWorkDone), so the submit / completion round trip cancels (it is ~13 ms in
//   Deno on Metal, which would otherwise halve the figure).
// Floors (each is the slope over N = 256 vs 1024 ops, so the submit / wait round trip cancels):
//   dispatch: N dispatches of a 1-workgroup kernel in ONE compute pass            -> µs per dispatch
//   dispatch (48 WG), the dn_delta shape, and 4352 WG x 256 threads (a GEMV-sized grid, tiny work)
//   pass:     N compute passes of one 1-workgroup dispatch each                   -> µs per pass
//   blit:     N x (compute pass + copyBufferToBuffer 256 B)                       -> µs per pass + blit
//   submit:   an empty command buffer + onSubmittedWorkDone                       -> the round-trip floor

export async function bwProbe(device, { MB = 1024, REPS = 8, log = console.log } = {}) {
  const now = () => performance.now();
  const bytes = MB * 2 ** 20;
  const src = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const dst = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const out = device.createBuffer({ size: 1 << 20, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  // non-zero contents, so no driver shortcut on cleared memory
  { const chunk = new Uint32Array(16 << 20 >> 2); for (let i = 0; i < chunk.length; i++) chunk[i] = (i * 2654435761) >>> 0;
    for (let o = 0; o < bytes; o += chunk.byteLength) device.queue.writeBuffer(src, o, chunk, 0, Math.min(chunk.length, (bytes - o) >> 2)); }
  await device.queue.onSubmittedWorkDone();

  const readWGSL = (vec, WG) => `
@group(0) @binding(0) var<storage, read> src: array<${vec ? "vec4<u32>" : "u32"}>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let n = arrayLength(&src); let stride = nw.x * ${WG}u;
  var acc = ${vec ? "vec4<u32>(0u)" : "0u"};
  for (var i = g.x; i < n; i += stride) { acc ^= src[i]; }
  let s = ${vec ? "acc.x ^ acc.y ^ acc.z ^ acc.w" : "acc"};
  if (s == 0x9e3779b9u) { out[g.x & 262143u] = s; }
}`;
  const pipe = (code) => device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  const bind = (p, a, b) => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: a } }, { binding: 1, resource: { buffer: b } }] });

  const time = async (encode) => { const e = device.createCommandEncoder(); encode(e); const t = now(); device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone(); return now() - t; };
  const LO = 2, HI = 6;
  const res = {};
  const variants = [];
  for (const vec of [true, false]) for (const WG of [64, 256, 1024]) for (const grid of [1024, 4096, 16384]) {
    if (WG > device.limits.maxComputeInvocationsPerWorkgroup) continue;
    const p = pipe(readWGSL(vec, WG)); variants.push({ name: `${vec ? "vec4" : "u32 "} WG ${String(WG).padStart(4)} grid ${String(grid).padStart(5)}`, vec, p, bg: bind(p, src, out), grid, best: Infinity });
  }
  const copy = { name: "copyBufferToBuffer", best: Infinity };
  for (const v of variants) await time((e) => { const c = e.beginComputePass(); c.setPipeline(v.p); c.setBindGroup(0, v.bg); c.dispatchWorkgroups(v.grid); c.end(); });   // compile + warm
  for (let r = 0; r < REPS; r++) {
    for (const v of variants) {
      const run = (n) => time((e) => { for (let k = 0; k < n; k++) { const c = e.beginComputePass(); c.setPipeline(v.p); c.setBindGroup(0, v.bg); c.dispatchWorkgroups(v.grid); c.end(); } });
      v.lo = Math.min(v.lo ?? Infinity, await run(LO)); v.hi = Math.min(v.hi ?? Infinity, await run(HI)); v.best = v.hi - v.lo;
    }
    const run = (n) => time((e) => { for (let k = 0; k < n; k++) e.copyBufferToBuffer(src, 0, dst, 0, bytes); });
    copy.lo = Math.min(copy.lo ?? Infinity, await run(LO)); copy.hi = Math.min(copy.hi ?? Infinity, await run(HI)); copy.best = copy.hi - copy.lo;
  }
  const gbs = (ms, b) => b / (ms / 1e3) / 1e9;
  variants.sort((a, b) => a.best - b.best);
  log(`bandwidth, ${MB} MiB buffer, slope ${LO} -> ${HI} passes per submit, best of ${REPS}:`);
  for (const v of variants) log(`  read ${v.name}: ${gbs(v.best, (HI - LO) * bytes).toFixed(1)} GB/s`);
  log(`  ${copy.name}: ${gbs(copy.best, (HI - LO) * bytes).toFixed(1)} GB/s read (+ the same written)`);
  const bestOf = (vec) => variants.filter((v) => v.vec === vec)[0];
  res.bw = { vec4: +gbs(bestOf(true).best, (HI - LO) * bytes).toFixed(1), vec4Cfg: bestOf(true).name, u32: +gbs(bestOf(false).best, (HI - LO) * bytes).toFixed(1), u32Cfg: bestOf(false).name, copyRead: +gbs(copy.best, (HI - LO) * bytes).toFixed(1) };

  // ---- floors ----
  const tiny = pipe(`@group(0) @binding(0) var<storage, read> src: array<u32>; @group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < 256u) { out[g.x] = out[g.x] + src[g.x]; } }`);
  const small = device.createBuffer({ size: 4096, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const tbg = bind(tiny, src, small);
  const slope = async (fn) => {
    const r = {}; for (const n of [256, 1024]) { let b = Infinity; await time((e) => fn(e, n)); for (let i = 0; i < REPS; i++) b = Math.min(b, await time((e) => fn(e, n))); r[n] = b; }
    return (r[1024] - r[256]) / 768 * 1e3;   // µs per op
  };
  const disp = (wg) => (e, n) => { const c = e.beginComputePass(); c.setPipeline(tiny); c.setBindGroup(0, tbg); for (let i = 0; i < n; i++) c.dispatchWorkgroups(wg); c.end(); };
  res.floor = {
    dispatch1: await slope(disp(1)),
    dispatch48: await slope(disp(48)),
    dispatch4352: await slope(disp(4352)),
    pass: await slope((e, n) => { for (let i = 0; i < n; i++) { const c = e.beginComputePass(); c.setPipeline(tiny); c.setBindGroup(0, tbg); c.dispatchWorkgroups(1); c.end(); } }),
    passBlit: await slope((e, n) => { for (let i = 0; i < n; i++) { const c = e.beginComputePass(); c.setPipeline(tiny); c.setBindGroup(0, tbg); c.dispatchWorkgroups(1); c.end(); e.copyBufferToBuffer(small, 0, out, (i & 1023) * 256, 256); } }),
  };
  { let b = Infinity; for (let i = 0; i < 4 * REPS; i++) b = Math.min(b, await time(() => {})); res.floor.submit = b * 1e3; }
  for (const k in res.floor) res.floor[k] = +res.floor[k].toFixed(2);
  log(`floors (µs): dispatch 1 WG ${res.floor.dispatch1}, 48 WG ${res.floor.dispatch48}, 4352 WG ${res.floor.dispatch4352}; pass ${res.floor.pass}; pass + blit ${res.floor.passBlit}; empty submit round trip ${res.floor.submit}`);
  src.destroy(); dst.destroy(); out.destroy(); small.destroy();
  return res;
}

if (import.meta.main) {
  const env = (k, d) => Deno.env.get(k) ?? d;
  const ad = await navigator.gpu.requestAdapter();
  const device = await ad.requestDevice({ requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize, maxComputeInvocationsPerWorkgroup: ad.limits.maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX: ad.limits.maxComputeWorkgroupSizeX } });
  let errs = 0; device.addEventListener?.("uncapturederror", (e) => { if (errs++ < 4) console.error("GPU ERROR:", e.error?.message); });
  console.log(`adapter ${ad.info?.vendor} ${ad.info?.architecture} ${ad.info?.description ?? ""}`);
  const r = await bwProbe(device, { MB: +env("MB", 1024), REPS: +env("REPS", 8) });
  console.log(JSON.stringify(r), `GPU errors ${errs}`);
}
