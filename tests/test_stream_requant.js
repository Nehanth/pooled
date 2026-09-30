// Real GPU, real model files: K-quant / Q4_1 tensors requantized to Q8 on their way to the GPU
// (streamEntryToGPU -> streamRequantToGPU, #207) read back bit-identical to the CPU path
// (convertEntry). GGUF=path (default: the Qwen3.8 27B and Qwen3.6 MoE files under ../models).
// deno run --unstable-webgpu --allow-read --allow-env tests/test_stream_requant.js
import { parseGGUFHeader, convertEntry, streamEntryToGPU, streamable, GGML_Q4_0, GGML_Q8_0 } from "../engine/gguf.js";
const paths = Deno.env.get("GGUF") ? [Deno.env.get("GGUF")] : ["../models/q38/model.gguf", "../models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"];
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
const readBack = async (buf, n) => {
  const rb = device.createBuffer({ size: buf.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(buf, 0, rb, 0, buf.size); device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ); const out = new Uint8Array(rb.getMappedRange().slice(0, n)); rb.unmap(); rb.destroy(); return out;
};
let tested = 0;
for (const path of paths) {
  let fh; try { fh = await Deno.open(path); } catch { console.log("skip (missing)", path); continue; }
  const readAt = async (off, len) => { await fh.seek(off, Deno.SeekMode.Start); const out = new Uint8Array(len); let got = 0; while (got < len) { const n = await fh.read(out.subarray(got)); if (n === null) break; got += n; } return out; };
  const G = parseGGUFHeader((await readAt(0, 64 << 20)).buffer, { skipTokenizer: true });
  const seen = {};
  for (const [name, info0] of Object.entries(G.tensors)) {
    if (info0.ggmlType === GGML_Q4_0 || info0.ggmlType === GGML_Q8_0 || !streamable(info0.ggmlType)) continue;
    if ((seen[info0.ggmlType + "/" + info0.shape.length] || 0) >= 2 || info0.byteLength > 200 * 2 ** 20) continue;
    seen[info0.ggmlType + "/" + info0.shape.length] = (seen[info0.ggmlType + "/" + info0.shape.length] || 0) + 1;
    const info = info0.shape.length === 3 ? { ...info0, shape: [info0.shape[0] * info0.shape[1], info0.shape[2]] } : info0;
    const bytes = await readAt(info.byteOffset, info.byteLength);
    const ref = convertEntry(info, bytes);
    // network-like chunks of 64 KB
    const openRange = async () => new Response(new ReadableStream({ start(c) { for (let o = 0; o < bytes.length; o += 65536) c.enqueue(bytes.slice(o, o + 65536)); c.close(); } }), { status: 206 });
    const t0 = performance.now();
    const e = await streamEntryToGPU(device, info, openRange, { staging: 2 << 20 });
    const ms = performance.now() - t0;
    const qs = await readBack(e.gpu.qs, ref.qs.byteLength);
    const sc = await readBack(e.gpu.sc, ref.scales.byteLength);
    const rs = new Uint8Array(ref.scales.buffer, 0, ref.scales.byteLength);
    let bad = 0; for (let i = 0; i < ref.qs.length; i++) if (qs[i] !== ref.qs[i]) { bad++; break; }
    for (let i = 0; i < rs.length; i++) if (sc[i] !== rs[i]) { bad++; break; }
    console.log(`${bad ? "MISMATCH" : "ok"} ${name} type ${info.ggmlType} ${info.shape.join("x")} ${(info.byteLength / 2 ** 20).toFixed(0)} MB, ${ms.toFixed(0)} ms`);
    if (bad) Deno.exit(1);
    e.gpu.qs.destroy(); e.gpu.sc.destroy();
    tested++;
  }
  fh.close();
}
console.log("STREAM REQUANT PASS ✓", tested, "tensors bit-identical");
