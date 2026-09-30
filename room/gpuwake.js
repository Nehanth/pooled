// Keep a worker's GPU clocked up while it waits for the next hidden state.
//
// Why: a phone GPU clocks down within a few ms of going idle. Back to back, one Qwen3.6 MoE layer
// takes 3-5 ms on an iPhone 14 Pro Max; in a room, where the phone idles 35-40 ms of every 47 ms
// lap waiting for the host, the same layer takes 7-12 ms (tests/bench/layer_prof.html ?gaps shows
// it without a room; docs/bench-log.md "device matrix"). So while the frame is on its way the
// device runs a small dummy kernel, and the frame's real work finds the GPU already awake.
//
// When: the host sends `ai-wake {pos}` to a worker that asked for it (hello meta `wake`) at the
// start of each decode lap (a plain token or a speculative verify), before its own layers; the
// worker spins from then until that frame arrives, capped at WAKE_MAX_MS. A wake that arrives
// after its frame (the two ride different channels) is ignored. Nothing here touches the model's
// buffers, so the answer cannot change; only when the GPU works.
//
// The dummy kernel streams a 4 MB buffer (memory clock as well as ALU: the real layer is mostly
// weight reads) in one short dispatch per submit, two submits in flight, so the frame's own work
// queues behind at most two of them. Plain WGSL, no subgroups or f16: runs on every WebGPU device.

const WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let n = arrayLength(&src);
  let stride = nw.x * 64u;
  var acc = vec4<f32>(0.0);
  for (var i = g.x; i < n; i += stride) { acc = acc * 0.5 + src[i]; }
  // never true (src is zeros): keeps the loads from being optimised away
  if (acc.x > 1.0) { dst[g.x & 63u] = acc.y; }
}`;

export const WAKE_BYTES = 4 << 20;
export const WAKE_GROUPS = 64;

export class GpuWaker {
  constructor(device, { bytes = WAKE_BYTES, groups = WAKE_GROUPS, inflight = 2 } = {}) {
    this.device = device;
    this.groups = groups;
    this.inflight = inflight;
    const src = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE });
    const dst = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE });
    this.bufs = [src, dst];
    this.pipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: WGSL }), entryPoint: "main" } });
    this.bind = device.createBindGroup({ layout: this.pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: src } }, { binding: 1, resource: { buffer: dst } }] });
    this.until = 0;
    this.running = false;
    this.stats = { wakes: 0, submits: 0, ms: 0 };
  }
  // spin until stop() or for at most `ms` from now; a wake while spinning extends the deadline
  wake(ms) {
    this.stats.wakes++;
    this.until = Math.max(this.until, performance.now() + ms);
    if (!this.running) this._run();
  }
  stop() { this.until = 0; }
  destroy() { this.stop(); for (const b of this.bufs) try { b.destroy(); } catch {} }
  async _run() {
    this.running = true;
    const t0 = performance.now(), q = this.device.queue, pending = [];
    try {
      while (performance.now() < this.until) {
        const enc = this.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipe); pass.setBindGroup(0, this.bind); pass.dispatchWorkgroups(this.groups);
        pass.end();
        q.submit([enc.finish()]);
        this.stats.submits++;
        pending.push(q.onSubmittedWorkDone());
        if (pending.length >= this.inflight) await pending.shift();
      }
    } catch {} finally {
      this.stats.ms += performance.now() - t0;
      this.running = false;
    }
  }
}
