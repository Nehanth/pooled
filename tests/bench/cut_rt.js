// The floor of one GPU -> CPU round trip (expert offload's per-layer cut, roadmap/32): a tiny dispatch, a copy into
// a MAP_READ buffer, submit, mapAsync, read 128 bytes. Runs on any navigator.gpu: Chrome (tests/bench/cut_rt.mjs
// opens tests/bench/cut_rt.html) and Node on Dawn (node tests/dawn_run.mjs tests/bench/cut_rt.js).
// Variants (each N round trips, median and p90 in ms):
//   idle     mapAsync of a buffer with no work pending (the IPC / event-loop floor)
//   rt       submit(dispatch + copy); await mapAsync                 (what _cutResolve does)
//   owd      submit; await onSubmittedWorkDone(); await mapAsync    (two waits)
//   ping     rt, plus an empty queue.submit([]) every macrotask while the map is pending (wakes Chrome's GPU process)
//   rtW      rt with a kernel of ~W us of GPU work in front (W = ?work / WORK env, about one layer's attention)
//   pingW    ping with that work
//   chain    W us of work in 4 dispatches, no readback, per iteration (the GPU time alone, for rtW - chainW)
export async function cutRoundTrip({ device, N = 300, workIters = 0, log = console.log } = {}) {
  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const buf = device.createBuffer({ size: 1 << 20, usage: S });
  const stage = device.createBuffer({ size: 128, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const tiny = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: `
    @group(0) @binding(0) var<storage, read_write> b: array<u32>;
    @compute @workgroup_size(32) fn main(@builtin(global_invocation_id) g: vec3u) { b[g.x] = b[g.x] + 1u; }` }), entryPoint: "main" } });
  // a busy kernel: 64 workgroups x 256 threads, each ITERS dependent FMAs over a 1 MiB buffer
  const busy = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: `
    @group(0) @binding(0) var<storage, read_write> b: array<f32>;
    override ITERS: u32 = 1000u;
    @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
      var x = b[g.x]; for (var i = 0u; i < ITERS; i++) { x = fma(x, 1.0000001, 0.0000001); } b[g.x] = x; }` }), entryPoint: "main", constants: { ITERS: Math.max(1, workIters) } } });
  const bgT = device.createBindGroup({ layout: tiny.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
  const bgB = device.createBindGroup({ layout: busy.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
  const q = device.queue;
  const kick = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = (work, rb = true) => {
    const e = device.createCommandEncoder(), p = e.beginComputePass();
    if (work) for (let i = 0; i < 4; i++) { p.setPipeline(busy); p.setBindGroup(0, bgB); p.dispatchWorkgroups(64); }
    p.setPipeline(tiny); p.setBindGroup(0, bgT); p.dispatchWorkgroups(1); p.end();
    if (rb) e.copyBufferToBuffer(buf, 0, stage, 0, 128);
    return e.finish();
  };
  const now = () => performance.now();
  const read = () => { const v = new Uint32Array(stage.getMappedRange(0, 128))[0]; stage.unmap(); return v; };
  let pinging = false, kickP = null;
  const mc = typeof MessageChannel !== "undefined" ? new MessageChannel() : null;
  const tick = (f) => (mc ? ((mc.port1.onmessage = f), mc.port2.postMessage(0)) : setImmediate(f));
  const ping = () => { if (!pinging) return; q.submit([]); tick(ping); };
  const variants = {
    idle: async () => { await stage.mapAsync(GPUMapMode.READ, 0, 128); read(); },
    rt: async () => { q.submit([enc(false)]); await stage.mapAsync(GPUMapMode.READ, 0, 128); read(); },
    owd: async () => { q.submit([enc(false)]); await q.onSubmittedWorkDone(); await stage.mapAsync(GPUMapMode.READ, 0, 128); read(); },
    ping: async () => { q.submit([enc(false)]); const m = stage.mapAsync(GPUMapMode.READ, 0, 128); pinging = true; tick(ping); await m; pinging = false; read(); },
  };
  if (workIters) Object.assign(variants, {
    rtW: async () => { q.submit([enc(true)]); await stage.mapAsync(GPUMapMode.READ, 0, 128); read(); },
    pingW: async () => { q.submit([enc(true)]); const m = stage.mapAsync(GPUMapMode.READ, 0, 128); pinging = true; tick(ping); await m; pinging = false; read(); },
    owdW: async () => { q.submit([enc(true)]); await q.onSubmittedWorkDone(); await stage.mapAsync(GPUMapMode.READ, 0, 128); read(); },
    // chained pings: while the map is pending, keep one cheap request in flight (each a GPU-process wake-up)
    kickMapW: async () => { q.submit([enc(true)]); let done = false; const m = stage.mapAsync(GPUMapMode.READ, 0, 128).then(() => { done = true; });
      while (!done) { kickP ||= kick.mapAsync(GPUMapMode.READ, 0, 16).then(() => { kick.unmap(); kickP = null; }); await Promise.race([m, kickP]); } read(); },
    kickOwdW: async () => { q.submit([enc(true)]); let done = false; const m = stage.mapAsync(GPUMapMode.READ, 0, 128).then(() => { done = true; });
      while (!done) await Promise.race([m, q.onSubmittedWorkDone()]); read(); },
    chainW: null,
  });
  const out = {};
  const stats = (a) => { a.sort((x, y) => x - y); return { med: a[a.length >> 1], p10: a[Math.floor(a.length * 0.1)], p90: a[Math.floor(a.length * 0.9)], mean: a.reduce((x, y) => x + y, 0) / a.length }; };
  for (const [name, f] of Object.entries(variants)) {
    const ts = [];
    if (name === "chainW") {   // GPU time of the work alone: N iterations back to back, one wait
      for (let r = 0; r < 5; r++) { const t0 = now(); for (let i = 0; i < 50; i++) q.submit([enc(true, false)]); await q.onSubmittedWorkDone(); ts.push((now() - t0) / 50); }
    } else {
      for (let i = 0; i < 20; i++) await f();
      for (let i = 0; i < N; i++) { const t0 = now(); await f(); ts.push(now() - t0); }
    }
    out[name] = stats(ts);
    log(`${name.padEnd(6)} median ${out[name].med.toFixed(3)} ms  p10 ${out[name].p10.toFixed(3)}  p90 ${out[name].p90.toFixed(3)}  mean ${out[name].mean.toFixed(3)}`);
  }
  buf.destroy(); stage.destroy(); kick.destroy();
  return out;
}

// Node / Deno entry (node tests/dawn_run.mjs tests/bench/cut_rt.js; WORK=iters, N=round trips)
if (typeof Deno !== "undefined" && typeof window === "undefined") {
  const ad = await navigator.gpu.requestAdapter();
  const device = await ad.requestDevice();
  const env = (k, d) => Deno.env.get(k) ?? d;
  console.log(`cut round trip, ${ad.info?.vendor || ""} ${ad.info?.architecture || ""} ${ad.info?.description || ""}`);
  await cutRoundTrip({ device, N: +env("N", 300), workIters: +env("WORK", 0) });
  if (+env("WORK", 0) === 0) { console.log("with work:"); await cutRoundTrip({ device, N: +env("N", 300), workIters: 3000 }); }
  Deno.exit(0);
}
