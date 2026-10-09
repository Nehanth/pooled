// Validate cooperative-GEMV shapes on nonzero inputs, then time the passing shapes.
// Timing uses a representative layer and onSubmittedWorkDone, never timestamp-query.
// Correctness uses only 17 rows (full workgroups plus a tail), at the same input width,
// so the extra readbacks and CPU reference stay small even on a phone.
import { WGSL } from "./wgsl/base.js";
import { probeUnpack, coopWGSL } from "./wgsl/coop.js";
import { moduleSet } from "./wgsl/prune.js";
import { f16ToF32 } from "./gguf.js";

const CANDIDATES = [[256, 4], [128, 4], [256, 8], [128, 8], [64, 4]];
const CHECK_ROWS = 17, WRITE_WORDS = 65536;
const SCALE16 = [0x2c00, 0x3000, 0xb400, 0x3400, 0xb000];
const mix = (i) => { let h = Math.imul(i + 1, 0x9e3779b1); h ^= h >>> 16; h = Math.imul(h, 0x85ebca6b); return (h ^ (h >>> 13)) >>> 0; };
const scaleWord = (i) => SCALE16[(2 * i) % SCALE16.length] | (SCALE16[(2 * i + 1) % SCALE16.length] << 16);
function checkShape(dIn, dOut, kind) {
  if (!Number.isInteger(dIn) || dIn <= 0 || dIn > 0xffffffff || dIn % 32 || !Number.isInteger(dOut) || dOut <= 0 || dOut > 0xffffffff) throw new Error("autotune: dimensions must be positive integers, with dIn divisible by 32");
  if (kind !== "q4" && kind !== "q8") throw new Error("autotune: kind must be q4 or q8");
}
function activation(dIn) {
  return Float32Array.from({ length: dIn }, (_, i) => ((mix(i ^ 0x1234) & 2047) - 1024) / 1024 || 0.125);
}

// Small deterministic packed Q4_0 / Q8_0 fixture and an independent float64 dot product.
// Weight nibbles/bytes and f16 scales have the same layout the engine uploads.
export function coopFixture({ dIn = 96, dOut = CHECK_ROWS, kind = "q4" } = {}) {
  checkShape(dIn, dOut, kind);
  const nb = dIn / 32, words = kind === "q4" ? 4 : 8;
  const qs = Uint32Array.from({ length: dOut * nb * words }, (_, i) => mix(i));
  const scales = Uint32Array.from({ length: Math.ceil(dOut * nb / 2) }, (_, i) => scaleWord(i));
  const x = activation(dIn), expected = new Float64Array(dOut);
  for (let r = 0; r < dOut; r++) for (let b = 0; b < nb; b++) {
    const bi = r * nb + b, sc = f16ToF32((scales[bi >> 1] >>> ((bi & 1) * 16)) & 0xffff);
    for (let j = 0; j < 32; j++) {
      const k = kind === "q4" ? j & 15 : j;
      const byte = (qs[bi * words + (k >> 2)] >>> ((k & 3) * 8)) & 255;
      const q = kind === "q4" ? ((byte >>> (j < 16 ? 0 : 4)) & 15) - 8 : byte > 127 ? byte - 256 : byte;
      expected[r] += sc * q * x[b * 32 + j];
    }
  }
  return { qs, scales, x, expected };
}

// Per-row tolerance catches a missing/swapped row even when other rows are much larger.
// Different workgroup sizes sum in different orders, so this is not a bit-equality check.
export function checkCoopOutput(got, expected) {
  if (got.length !== expected.length) return { ok: false, reason: "wrong output length" };
  let maxAbsError = 0, maxRelError = 0;
  for (let row = 0; row < expected.length; row++) {
    const actual = got[row], want = expected[row];
    if (!Number.isFinite(actual) || !Number.isFinite(want)) return { ok: false, reason: "nonfinite output", row, actual, expected: want };
    const error = Math.abs(actual - want), limit = 1e-4 + 2e-3 * Math.abs(want);
    maxAbsError = Math.max(maxAbsError, error);
    maxRelError = Math.max(maxRelError, error / Math.max(1e-4, Math.abs(want)));
    if (error > limit) return { ok: false, reason: "incorrect output", row, actual, expected: want, maxAbsError, maxRelError };
  }
  return { ok: true, maxAbsError, maxRelError };
}

// Fill the full timing buffers in reusable 256 KiB chunks, rather than hold another
// complete matrix in JS memory while the device is loading a model.
function writeWords(device, buffer, n, valueAt) {
  const chunk = new Uint32Array(Math.min(WRITE_WORDS, n));
  for (let off = 0; off < n; off += chunk.length) {
    const count = Math.min(chunk.length, n - off);
    for (let j = 0; j < count; j++) chunk[j] = valueAt(off + j);
    device.queue.writeBuffer(buffer, off * 4, chunk, 0, count);
  }
}

export async function autotuneCoop(device, { dIn = 5120, dOut = 17408, kind = "q4", validateKinds = [kind] } = {}) {
  checkShape(dIn, dOut, kind);
  const kinds = [...new Set([kind, ...validateKinds])];
  for (const k of kinds) checkShape(dIn, CHECK_ROWS, k);
  const buffers = [], results = [], rejected = [], validated = [];
  const makeBuffer = (size, usage, label) => { const b = device.createBuffer({ size, usage, label }); buffers.push(b); return b; };
  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, U = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
  try {
    const nb = dIn / 32;
    const cfgB = makeBuffer(48, U, "autotune config"), frameB = makeBuffer(16, U, "autotune frame");
    const shape = makeBuffer(16, U, "autotune timing shape");
    const qs = makeBuffer(dOut * dIn / (kind === "q4" ? 2 : 1), S, "autotune timing weights");
    const sc = makeBuffer(Math.ceil(dOut * nb / 2) * 4, S, "autotune timing scales");
    const x = makeBuffer(dIn * 4, S, "autotune activations"), y = makeBuffer(dOut * 4, S, "autotune timing output");
    device.queue.writeBuffer(shape, 0, new Uint32Array([dOut, dIn, 0, 0]));
    writeWords(device, qs, qs.size / 4, mix);
    writeWords(device, sc, sc.size / 4, scaleWord);
    device.queue.writeBuffer(x, 0, activation(dIn));
    const checkShapeB = makeBuffer(16, U, "autotune check shape");
    device.queue.writeBuffer(checkShapeB, 0, new Uint32Array([CHECK_ROWS, dIn, 0, 0]));
    const stage = makeBuffer(CHECK_ROWS * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, "autotune readback");
    const checks = kinds.map((k) => {
      const fixture = coopFixture({ dIn, kind: k });
      const q = makeBuffer(fixture.qs.byteLength, S, `autotune ${k} check weights`), s = makeBuffer(fixture.scales.byteLength, S, `autotune ${k} check scales`);
      const out = makeBuffer(CHECK_ROWS * 4, S | GPUBufferUsage.COPY_SRC, `autotune ${k} check output`);
      device.queue.writeBuffer(q, 0, fixture.qs); device.queue.writeBuffer(s, 0, fixture.scales);
      return { kind: k, q, s, out, expected: fixture.expected };
    });
    const C = GPUShaderStage.COMPUTE;
    const l0 = device.createBindGroupLayout({ entries: [0, 1].map((binding) => ({ binding, visibility: C, buffer: { type: "uniform" } })) });
    const l1 = device.createBindGroupLayout({ entries: ["read-only-storage", "read-only-storage", "read-only-storage", "storage", "uniform"].map((type, binding) => ({ binding, visibility: C, buffer: { type } })) });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [l0, l1] });
    const bg0 = device.createBindGroup({ layout: l0, entries: [{ binding: 0, resource: { buffer: cfgB } }, { binding: 1, resource: { buffer: frameB } }] });
    const group = (bs) => device.createBindGroup({ layout: l1, entries: bs.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const bgTime = group([qs, sc, x, y, shape]);
    for (const c of checks) c.bg = group([c.q, c.s, x, c.out, checkShapeB]);
    const unpack = await probeUnpack(device);
    for (const [wg, rows] of CANDIDATES) {
      let scope = false;
      try {
        device.pushErrorScope("validation"); scope = true;
        const mod = moduleSet(device, WGSL + coopWGSL(wg, rows, 64, 4, 4, unpack));
        const pipes = new Map(), checked = [];
        let bad = false;
        for (const c of checks) {
          const entryPoint = `matvec_${c.kind}_coop`;
          const pipe = await device.createComputePipelineAsync({ layout, compute: { module: mod(entryPoint), entryPoint } });
          pipes.set(c.kind, pipe);
          // Poison output before dispatch: a validation failure or missing row must not
          // pass by leaving the preceding candidate's correct values in the buffer.
          device.queue.writeBuffer(c.out, 0, new Float32Array(CHECK_ROWS).fill(NaN));
          const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
          pass.setPipeline(pipe); pass.setBindGroup(0, bg0); pass.setBindGroup(1, c.bg);
          pass.dispatchWorkgroups(Math.ceil(CHECK_ROWS / rows)); pass.end();
          enc.copyBufferToBuffer(c.out, 0, stage, 0, CHECK_ROWS * 4);
          device.queue.submit([enc.finish()]);
          await stage.mapAsync(GPUMapMode.READ);
          let check;
          try { check = checkCoopOutput(new Float32Array(stage.getMappedRange()), c.expected); }
          finally { stage.unmap(); }
          if (!check.ok) { rejected.push({ wg, rows, kind: c.kind, ...check }); bad = true; break; }
          checked.push({ kind: c.kind, maxAbsError: check.maxAbsError, maxRelError: check.maxRelError });
        }
        const error = await device.popErrorScope(); scope = false;
        if (error) throw error;
        if (bad) continue;
        const entry = { wg, rows, checks: checked };
        validated.push(entry);
        const run = (n) => {
          const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
          pass.setPipeline(pipes.get(kind)); pass.setBindGroup(0, bg0); pass.setBindGroup(1, bgTime);
          for (let i = 0; i < n; i++) pass.dispatchWorkgroups(Math.ceil(dOut / rows));
          pass.end(); device.queue.submit([enc.finish()]);
          return device.queue.onSubmittedWorkDone();
        };
        // Software rasterizers can take minutes for the sweep. Keep the first
        // validated shape on a slow device instead of selecting an unchecked default.
        if (!results.length) {
          const t1 = performance.now(); await run(1);
          if (performance.now() - t1 > 20) return { wg, rows, results, rejected, validated, skipped: "slow device" };
        }
        const tw = performance.now();
        while (performance.now() - tw < (results.length ? 40 : 250)) await run(20);
        const t0 = performance.now(); await run(100);
        results.push({ ...entry, ms: (performance.now() - t0) / 100 });
      } catch (e) {
        rejected.push({ wg, rows, reason: "unsupported", error: String(e.message || e).slice(0, 300) });
      } finally {
        if (scope) await device.popErrorScope();
      }
    }
    if (!results.length) {
      if (validated.length) return { wg: validated[0].wg, rows: validated[0].rows, results, rejected, validated, skipped: "timing failed" };
      const numerical = rejected.some((r) => r.reason !== "unsupported");
      const error = new Error(numerical ? "GPU cooperative GEMV failed its correctness checks; no safe tuning shape is available" : "GPU supports none of the cooperative GEMV tuning shapes");
      error.code = numerical ? "autotune-correctness" : "autotune-unsupported";
      error.rejected = rejected;
      throw error;
    }
    results.sort((a, b) => a.ms - b.ms);
    const best = results[0], def = results.find((r) => r.wg === 256 && r.rows === 4);
    const pick = def && def.ms <= best.ms * 1.03 ? def : best;
    return { wg: pick.wg, rows: pick.rows, results, rejected, validated };
  } finally {
    for (const b of buffers) b.destroy();
  }
}
