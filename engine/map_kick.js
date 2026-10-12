// mapRead: buffer.mapAsync(READ) that resolves as soon as the GPU is done, in Chrome too.
//
// Chrome's GPU process resolves a pending map when it next checks the device for finished work. If the work behind
// the map was still running when the request arrived, that check is a timer about 2 ms away (tests/bench/cut_rt.js
// on the GB10, Chrome 131 / Vulkan: a readback behind 80 us of GPU work took 2.9 ms median, a ready one 0.1-0.5 ms).
// Every request the GPU process handles also checks the device, so mapRead keeps one cheap request in flight while
// the map is pending: a mapAsync of a 16-byte buffer that has no GPU work, which completes at once and is reissued.
// The same readback then took 0.4 ms. Expert offload reads the router's choice back once per offloaded layer and
// token (engine/qwen35.js _cutResolve), so this is most of a cut's cost in a tab. On Dawn in Node (and Deno) the
// map resolves within ~30 us anyway and the kick changes nothing measurable.
//
// The kick buffer is never read and holds no data: the mapped bytes, the order of GPU work and the results are
// exactly those of a plain mapAsync.
const KICK = new WeakMap();   // device -> { buf, p: the kick map in flight or null }
const LAST = new WeakMap();   // staging buffer -> ms its last readback took (kick "auto")

// kick: true (kick at once), false (a plain mapAsync), or "auto": in a browser, kick only if this buffer's last
// readback took under AUTO.ms. A readback behind short GPU work (a worker's hop over a few layers, a head) loses up to
// ~2 ms to the timer and the kick wins it back; one behind a whole token's work (> 15 ms) gains nothing, and a stream
// of kicks the whole time can slow it. Outside a browser (Node / Deno: Dawn, wgpu) "auto" never kicks: Dawn's map
// already resolves within ~30 us and the kick only costs (tests/bench/cut_core.js hopBench: +0.03-0.3 ms a readback).
// AUTO is exported so tests can change it.
export const AUTO = { ms: 6, browser: typeof navigator !== "undefined" && /Mozilla/.test(navigator.userAgent || "") };
export async function mapRead(device, buf, offset = 0, size = undefined, kick = true) {
  const m = size === undefined ? buf.mapAsync(GPUMapMode.READ, offset) : buf.mapAsync(GPUMapMode.READ, offset, size);
  if (kick === "auto" && !AUTO.browser) kick = false;
  if (kick === "auto") {
    const t0 = performance.now(), last = LAST.get(buf);
    m.then(() => LAST.set(buf, performance.now() - t0), () => {});
    kick = last === undefined || last < AUTO.ms;
  }
  if (!kick) return m;
  let done = false;
  m.then(() => { done = true; }, () => { done = true; });
  let k = KICK.get(device);
  if (!k) KICK.set(device, k = { buf: device.createBuffer({ label: "map kick", size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), p: null });
  while (!done) {
    // one kick at a time (a buffer takes one pending map), shared by every readback of the device
    k.p ||= k.buf.mapAsync(GPUMapMode.READ, 0, 16).then(() => { k.buf.unmap(); k.p = null; }, () => { k.p = null; });
    await Promise.race([m, k.p]);
  }
  return m;
}
