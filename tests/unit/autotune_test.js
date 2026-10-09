// CPU references and a recording device exercise rejection, timing and teardown.
// tests/e2e/autotune.mjs runs the same checks on actual generated GPU kernels.
import { autotuneCoop, coopFixture, checkCoopOutput } from "../../engine/autotune.js";
import { dequantQ4, dequantQ8 } from "../../engine/quant.js";

const ok = (v, message = "assertion failed") => { if (!v) throw new Error(message); };
const eq = (a, b) => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${JSON.stringify(a)} != ${JSON.stringify(b)}`); };
const cpu = (kind, qs, scales, x, dOut) => {
  const dIn = x.length, q = { qs: new Uint8Array(qs.buffer, qs.byteOffset, qs.byteLength), scales };
  const w = (kind === "q4" ? dequantQ4 : dequantQ8)(q, dIn * dOut), out = new Float64Array(dOut);
  for (let r = 0; r < dOut; r++) for (let c = 0; c < dIn; c++) out[r] += w[r * dIn + c] * x[c];
  return out;
};

Deno.test("autotune fixtures use signed nonzero packed weights and odd scale counts", () => {
  for (const kind of ["q4", "q8"]) for (const dIn of [32, 96, 2240]) {
    const f = coopFixture({ dIn, kind }), other = coopFixture({ dIn, kind });
    eq([...f.qs], [...other.qs]); eq([...f.x], [...other.x]);
    eq([...f.expected], [...cpu(kind, f.qs, f.scales, f.x, 17)]);
    ok(f.x.every((v) => v !== 0)); ok(f.qs.some((v) => v !== 0));
    ok(f.expected.every(Number.isFinite)); ok(new Set(f.expected).size > 1);
    const q = { qs: new Uint8Array(f.qs.buffer), scales: f.scales };
    const w = (kind === "q4" ? dequantQ4 : dequantQ8)(q, dIn * 17);
    ok(w.some((v) => v < 0) && w.some((v) => v > 0));
    eq(f.scales.length, Math.ceil(17 * dIn / 32 / 2));
  }
});

Deno.test("autotune rejects invalid fixture formats and shapes", () => {
  for (const opts of [{ kind: "f32" }, { dIn: 33 }, { dIn: 0 }, { dIn: -32 }, { dIn: Infinity }, { dOut: 0 }, { dOut: 1.5 }]) {
    let caught = false; try { coopFixture(opts); } catch { caught = true; } ok(caught, JSON.stringify(opts));
  }
});

Deno.test("autotune output gate permits rounding but rejects wrong, missing and nonfinite rows", () => {
  ok(checkCoopOutput(new Float32Array([1.001, -2.001, 0.00001]), [1, -2, 0]).ok);
  eq(checkCoopOutput([1], [1, 2]).reason, "wrong output length");
  for (const value of [NaN, Infinity, -Infinity]) eq(checkCoopOutput([value], [1]).reason, "nonfinite output");
  // A large row cannot hide a missing low-amplitude row under a global norm.
  const bad = checkCoopOutput([100000, 0], [100000, 1]);
  eq(bad.reason, "incorrect output"); eq(bad.row, 1);
  eq(checkCoopOutput([0.1], [0]).reason, "incorrect output");
});

async function deviceTest(options, fn) {
  const saved = new Map(["GPUBufferUsage", "GPUShaderStage", "GPUMapMode", "performance"].map((n) => [n, Object.getOwnPropertyDescriptor(globalThis, n)]));
  let clock = 0, lastDispatches = 0, maps = 0, scopeCount = 0;
  for (const [name, value] of Object.entries({ GPUBufferUsage: { STORAGE: 128, COPY_DST: 8, COPY_SRC: 4, UNIFORM: 64, MAP_READ: 1 }, GPUShaderStage: { COMPUTE: 4 }, GPUMapMode: { READ: 1 }, performance: { now: () => clock } })) Object.defineProperty(globalThis, name, { configurable: true, value });
  const buffers = [], timed = [], checked = [], writes = [];
  const cache = new WeakMap();
  const device = {
    __unpackOk: true,
    createBuffer({ size, label, usage }) {
      if (options.failBuffer === buffers.length) throw new Error("allocation failed");
      const b = { size, label, usage, data: new ArrayBuffer(size), destroyed: false, mapped: false,
        destroy() { this.destroyed = true; },
        async mapAsync() { if (options.failMap === ++maps) throw new Error("mapping failed"); this.mapped = true; },
        getMappedRange() { return this.data; }, unmap() { this.mapped = false; } };
      buffers.push(b); return b;
    },
    createBindGroupLayout: (x) => x, createPipelineLayout: (x) => x,
    createBindGroup: ({ entries }) => entries.map((e) => e.resource.buffer),
    createShaderModule: (x) => x,
    async createComputePipelineAsync({ compute }) {
      if (options.failPipelines) throw new Error("unsupported pipeline");
      const wg = +compute.module.code.match(/@workgroup_size\((\d+)\)/)[1];
      const rows = +compute.module.code.match(/let row0 = .* \* (\d+)u;/)[1];
      return { wg, rows, kind: compute.entryPoint.includes("q8") ? "q8" : "q4" };
    },
    pushErrorScope() { scopeCount++; }, async popErrorScope() { scopeCount--; return null; },
    createCommandEncoder() {
      const ops = [];
      return {
        beginComputePass() {
          let pipe, inputs, n = 0;
          return { setPipeline(p) { pipe = p; }, setBindGroup(i, b) { if (i === 1) inputs = b; }, dispatchWorkgroups() { n++; },
            end() { ops.push(() => {
              lastDispatches = n;
              const [q, s, x, y, shape] = inputs;
              const [dOut, dIn] = new Uint32Array(shape.data), key = { wg: pipe.wg, rows: pipe.rows, kind: pipe.kind };
              if (y.label.includes("timing")) { timed.push(key); return; }
              checked.push(key);
              let want = cache.get(q);
              if (!want) { want = cpu(pipe.kind, new Uint32Array(q.data), new Uint32Array(s.data), new Float32Array(x.data, 0, dIn), dOut); cache.set(q, want); }
              const got = Float32Array.from(want);
              options.corrupt?.(key, got);
              // An unwritten output keeps the poison uploaded before this dispatch.
              if (!options.leaveOutput?.(key)) new Float32Array(y.data).set(got);
            }); } };
        },
        copyBufferToBuffer(src, srcOff, dest, destOff, n) { ops.push(() => new Uint8Array(dest.data, destOff, n).set(new Uint8Array(src.data, srcOff, n))); },
        finish() { return ops; },
      };
    },
    queue: {
      writeBuffer(b, off, data, dataOff = 0, count = data.length) {
        const bytes = data.BYTES_PER_ELEMENT, view = new Uint8Array(data.buffer, data.byteOffset + dataOff * bytes, count * bytes);
        new Uint8Array(b.data, off, view.length).set(view); writes.push({ label: b.label, off, size: view.length });
      },
      submit(commands) { for (const ops of commands) for (const op of ops) op(); },
      async onSubmittedWorkDone() { clock += options.slow ? 30 : lastDispatches === 1 ? 1 : lastDispatches === 100 ? (options.timeMs ?? 100) : 50; },
    },
  };
  try { await fn({ device, buffers, timed, checked, writes }); }
  finally {
    for (const [name, descriptor] of saved) { if (descriptor) Object.defineProperty(globalThis, name, descriptor); else delete globalThis[name]; }
    ok(buffers.every((b) => b.destroyed), "every buffer must be destroyed");
    ok(buffers.every((b) => !b.mapped), "readback must be unmapped"); eq(scopeCount, 0);
  }
}

Deno.test("autotune validates both formats, rejects wrong 8-row output, and times only passing candidates", async () => {
  await deviceTest({ corrupt: ({ rows, kind }, got) => { if (rows === 8 && kind === "q8") got[3] *= 20; } }, async ({ device, timed, checked, buffers }) => {
    const r = await autotuneCoop(device, { dIn: 96, dOut: 19, kind: "q4", validateKinds: ["q4", "q8", "q8"] });
    eq(r.results.length, 3); eq(r.rejected.length, 2); eq(r.rows, 4);
    ok(r.rejected.every((x) => x.kind === "q8" && x.row === 3 && x.reason === "incorrect output"));
    ok(timed.every((x) => x.rows === 4)); eq(checked.length, 10);
    ok(r.results.every((x) => x.checks.length === 2));
    // The full benchmark is nonzero too, including scales and the input vector.
    for (const label of ["autotune timing weights", "autotune timing scales", "autotune activations"]) ok(new Uint8Array(buffers.find((b) => b.label === label).data).some((b) => b !== 0));
  });
});

Deno.test("autotune poison catches a kernel that fails to write after another candidate passes", async () => {
  await deviceTest({ leaveOutput: ({ rows }) => rows === 8 }, async ({ device }) => {
    const r = await autotuneCoop(device, { dIn: 32, dOut: 17, kind: "q8" });
    eq(r.rejected.length, 2); ok(r.rejected.every((r) => r.reason === "nonfinite output")); eq(r.rows, 4);
  });
});

Deno.test("autotune refuses unchecked default when every candidate has incorrect output", async () => {
  await deviceTest({ corrupt: (_key, got) => got.fill(0) }, async ({ device, timed }) => {
    let error; try { await autotuneCoop(device, { dIn: 32, dOut: 17 }); } catch (e) { error = e; }
    eq(error?.code, "autotune-correctness"); eq(error.rejected.length, 5); eq(timed.length, 0);
  });
});

Deno.test("autotune slow-device fallback uses a checked alternative when default fails", async () => {
  await deviceTest({ slow: true, corrupt: ({ wg, rows }, got) => { if (wg === 256 && rows === 4) got[0] = Infinity; } }, async ({ device, timed }) => {
    const r = await autotuneCoop(device, { dIn: 32, dOut: 17 });
    eq(r.skipped, "slow device"); eq([r.wg, r.rows], [128, 4]); eq(r.validated.length, 1);
    eq(r.rejected[0].reason, "nonfinite output"); ok(timed.every((x) => x.wg === 128));
  });
});

Deno.test("autotune reports unsupported pipelines and cleans up all buffers", async () => {
  await deviceTest({ failPipelines: true }, async ({ device }) => {
    let error; try { await autotuneCoop(device, { dIn: 32, dOut: 17 }); } catch (e) { error = e; }
    eq(error?.code, "autotune-unsupported"); eq(error.rejected.length, 5);
    ok(error.rejected.every((r) => r.error.includes("unsupported pipeline")));
  });
});

Deno.test("autotune survives one failed readback and releases buffers on partial allocation", async () => {
  await deviceTest({ failMap: 1 }, async ({ device }) => {
    const r = await autotuneCoop(device, { dIn: 32, dOut: 17 });
    eq(r.rejected[0].reason, "unsupported"); ok(r.rejected[0].error.includes("mapping failed")); eq(r.results.length, 4);
  });
  await deviceTest({ failBuffer: 4 }, async ({ device }) => {
    let error; try { await autotuneCoop(device, { dIn: 32, dOut: 17 }); } catch (e) { error = e; }
    ok(error?.message.includes("allocation failed"));
  });
});

Deno.test("autotune initializes large timing buffers using bounded chunks", async () => {
  await deviceTest({}, async ({ device, writes }) => {
    await autotuneCoop(device, { dIn: 96, dOut: 6000, kind: "q8" });
    const chunks = writes.filter((w) => w.label === "autotune timing weights");
    ok(chunks.length > 1); ok(chunks.every((w) => w.size <= 256 * 1024));
    eq(chunks.reduce((sum, w) => sum + w.size, 0), 96 * 6000);
  });
});
