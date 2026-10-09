// Tiny real-kernel gate: Q4/Q8, every tuning shape, and deliberate eight-row corruption.
// CHROMIUM=/path/to/chrome E2E_GPU=real NODE_PATH=/path/to/node_modules node tests/e2e/autotune.mjs
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";
const PORT = +(process.env.AUTOTUNE_PORT || 18991);
const srv = serveRepo(PORT);
let browser;
try {
  const { chromium } = await loadPlaywright();
  const args = process.platform === "darwin" && process.env.E2E_GPU === "real"
    ? ["--no-sandbox", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--use-angle=metal"] : GPU_ARGS;
  browser = await chromium.launch({ executablePath: chromiumPath(), args });
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${PORT}/__blank.html`);
  const result = await page.evaluate(async () => {
    const { autotuneCoop } = await import("/engine/autotune.js");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("WebGPU adapter unavailable");
    const device = await adapter.requestDevice();
    const gpuErrors = [], checks = [];
    device.addEventListener("uncapturederror", (e) => gpuErrors.push(e.error.message));
    const assert = (name, ok, detail) => { checks.push({ name, ok, detail }); if (!ok) throw new Error(`${name}: ${JSON.stringify(detail)}`); };
    try {
      // nb=70 crosses both the 64- and 32-block strides; 17 check rows exercises tails.
      // Every GPU shader, packed format, and CPU known answer is real here.
      for (const kind of ["q4", "q8"]) {
        const r = await autotuneCoop(device, { dIn: 2240, dOut: 19, kind, validateKinds: ["q4", "q8"] });
        assert(`${kind}: every offered shape passes both known answers`, r.validated.length === 5 && r.rejected.length === 0 && r.validated.every((v) => v.checks.length === 2), r);
      }
      // Feed the same real WebGPU device shaders that compile but write a wrong result.
      // This proves bad candidates lose eligibility, even when the primary Q4 format passes.
      let corrupted = 0;
      const bad = new Proxy(device, { get(target, key) {
        if (key === "createShaderModule") return (desc) => {
          if (desc.code.includes("fn matvec_q8_coop(") && /let row0 = .* \* 8u;/.test(desc.code)) {
            const code = desc.code.replace(/q8_y\[row\] = mvc_part\[t \* \d+u\];/, "q8_y[row] = 0.0;");
            if (code !== desc.code) corrupted++;
            return target.createShaderModule({ ...desc, code });
          }
          return target.createShaderModule(desc);
        };
        const value = Reflect.get(target, key, target);
        return typeof value === "function" ? value.bind(target) : value;
      } });
      const r = await autotuneCoop(bad, { dIn: 2240, dOut: 19, kind: "q4", validateKinds: ["q4", "q8"] });
      assert("wrong eight-row Q8 kernels are rejected before timing", corrupted === 2 && r.rejected.length === 2 && r.rejected.every((v) => v.rows === 8 && v.kind === "q8" && v.reason === "incorrect output") && r.results.length === 3 && r.rows === 4, r);
      await device.queue.onSubmittedWorkDone();
      assert("no GPU validation errors", gpuErrors.length === 0, gpuErrors);
      return { adapter: { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture }, checks };
    } finally { device.destroy(); }
  });
  console.log("adapter", JSON.stringify(result.adapter));
  for (const c of result.checks) console.log(`PASS ${c.name}`);
} finally {
  if (browser) await browser.close();
  await new Promise((resolve) => srv.close(resolve));
}
