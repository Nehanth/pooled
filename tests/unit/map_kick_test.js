// engine/map_kick.js mapRead: the readback resolves with the buffer mapped, a kick map is in flight while it waits,
// never two on the kick buffer at once (a buffer takes one pending map), and kick = false is a plain mapAsync.
// No GPU.   deno test tests/unit/map_kick_test.js
import { mapRead } from "../../engine/map_kick.js";

globalThis.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
globalThis.GPUMapMode ??= { READ: 1, WRITE: 2 };
const assert = (c, m) => { if (!c) throw new Error(m); };
// a buffer whose map resolves after `ticks` kicks have completed (the GPU process noticing on a later request)
function fakeDevice(ticks) {
  const st = { kicks: 0, maxPending: 0, created: 0 };
  let pending = 0;
  const dev = {
    st,
    createBuffer({ size }) {
      st.created++;
      const b = { size, mapped: false, kick: true,
        mapAsync() {
          if (this.pend) return Promise.reject(new Error("buffer already has a pending map"));
          this.pend = true; pending++; st.maxPending = Math.max(st.maxPending, pending);
          return new Promise((r) => setTimeout(() => { st.kicks++; this.pend = false; pending--; this.mapped = true; r(); }, 0));
        },
        unmap() { this.mapped = false; } };
      return b;
    },
  };
  const target = { mapped: false, mapAsync() {
    return new Promise((r) => { const t = st.kicks + ticks, w = () => (st.kicks >= t ? ((target.mapped = true), r()) : setTimeout(w, 0)); w(); });
  } };
  return { dev, target };
}

Deno.test("mapRead: resolves once the map does, kicking one request at a time meanwhile", async () => {
  const { dev, target } = fakeDevice(5);
  await mapRead(dev, target, 0, 16);
  assert(target.mapped, "mapped");
  assert(dev.st.kicks >= 5, `kicks ${dev.st.kicks}`);
  assert(dev.st.created === 1, "one kick buffer per device");
  // back to back: the kick still in flight from the last call is reused, not mapped twice
  for (let i = 0; i < 5; i++) { target.mapped = false; await mapRead(dev, target, 0, 16); assert(target.mapped, "mapped " + i); }
  assert(dev.st.created === 1, "still one kick buffer");
});

Deno.test("mapRead: kick = false is the plain mapAsync", async () => {
  let called = 0;
  const dev = { createBuffer() { throw new Error("no kick buffer"); } };
  const b = { mapAsync: (m, o, s) => { called++; assert(m === GPUMapMode.READ && o === 8 && s === 32, "arguments"); return Promise.resolve(); } };
  await mapRead(dev, b, 8, 32, false);
  assert(called === 1, "mapAsync once");
});

Deno.test("mapRead: a failed map rejects", async () => {
  const { dev } = fakeDevice(0);
  const b = { mapAsync: () => Promise.reject(new Error("lost")) };
  let threw = false;
  try { await mapRead(dev, b, 0, 4); } catch (e) { threw = /lost/.test(e.message); }
  assert(threw, "rejects with the map's error");
});
