// How fast this GPU moves memory, for choosing the model host. Decode on every model the room runs
// is bound by memory bandwidth (each token reads every weight once), and the model host carries
// the embedding, the head, the draft block and the sampler on top of its layers, so the device that
// streams memory fastest should hold them. A buffer-to-buffer copy is the plainest measure of that.
//
// measureCopyGBps(adapter): GB/s (read + write) of a 64 MB copy, best of 5 timed passes of 8 copies
// each after a warm-up, on a device of its own that is destroyed afterwards. About 30-200 ms on a
// computer; 0 when it cannot tell (no device, a lost device, or a pass too short to time).
export async function measureCopyGBps(adapter, { mb = 64, copies = 8, passes = 5 } = {}) {
  let dev = null;
  try {
    dev = await adapter.requestDevice();
    const size = mb * 2 ** 20;
    const use = GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const a = dev.createBuffer({ size, usage: use }), b = dev.createBuffer({ size, usage: use });
    const pass = async (n) => {
      const enc = dev.createCommandEncoder();
      for (let i = 0; i < n; i++) enc.copyBufferToBuffer(i & 1 ? b : a, 0, i & 1 ? a : b, 0, size);
      const t0 = performance.now();
      dev.queue.submit([enc.finish()]);
      await dev.queue.onSubmittedWorkDone();
      return performance.now() - t0;
    };
    await pass(2);   // first touch commits the pages and wakes the GPU up
    let best = Infinity;
    for (let k = 0; k < passes; k++) best = Math.min(best, await pass(copies));
    a.destroy(); b.destroy();
    return best >= 0.5 ? Math.round((2 * size * copies) / (best / 1000) / 1e9) : 0;
  } catch { return 0; } finally { try { dev?.destroy(); } catch {} }
}
