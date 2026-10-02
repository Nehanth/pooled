// Run a Deno GPU test under Node on Dawn (npm "webgpu", dawn.node: what packages/room-node runs on), for the
// numbers and the backends Deno (wgpu) does not give: Dawn's Vulkan on Linux, D3D12 + FXC on Windows. Deno's
// mapAsync takes ~12 ms per readback; Dawn's ~0.1-0.2 ms, which is what a per-layer readback (expert offload) needs.
//
//   node --no-maglev tests/dawn_run.mjs tests/test_moe.js      (env as for the Deno run: MODEL, OFFLOAD, ...)
//   WEBGPU=/path/to/node_modules/webgpu/index.js  where dawn.node lives (default: resolve "webgpu")
//   DAWN_OPTS="enable-dawn-features=..."          Dawn toggles, as room-node takes them
// --no-maglev: V8's Maglev compiler crashes Node 24 + dawn.node on Windows (0xC0000005; feasibility study 2.8).
//
// The shim gives the test the Deno APIs the GPU tests use (env, file reads, exit, build.os); it does not make
// navigator.gpu behave like Deno's: the engine sees Dawn's adapter info (vendor), as a room node does.
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const mod = process.env.WEBGPU ? await import(pathToFileURL(process.env.WEBGPU).href) : await import("webgpu");
const { create, globals } = mod.default?.create ? mod.default : mod;
Object.assign(globalThis, globals);
const dawn = create((process.env.DAWN_OPTS || "").split(" ").filter(Boolean));
// the discrete GPU, as room-node asks for it: without a preference Dawn may hand out the integrated one (the PC's
// Core Ultra iGPU instead of its RTX 5070)
const gpu = { requestAdapter: (o = {}) => dawn.requestAdapter({ powerPreference: "high-performance", ...o }),
  getPreferredCanvasFormat: () => dawn.getPreferredCanvasFormat?.(), get wgslLanguageFeatures() { return dawn.wgslLanguageFeatures; } };
Object.defineProperty(globalThis.navigator, "gpu", { value: gpu, configurable: true });
const toPath = (p) => (p instanceof URL ? p : String(p).startsWith("file:") ? new URL(p) : p);
globalThis.Deno = {
  env: { get: (k) => process.env[k], toObject: () => ({ ...process.env }) },
  readTextFileSync: (p) => fs.readFileSync(toPath(p), "utf8"),
  readTextFile: async (p) => fs.readFileSync(toPath(p), "utf8"),
  readFileSync: (p) => new Uint8Array(fs.readFileSync(toPath(p))),
  writeTextFileSync: (p, s) => fs.writeFileSync(toPath(p), s),
  exit: (c) => process.exit(c),
  build: { os: process.platform === "win32" ? "windows" : process.platform },
  args: process.argv.slice(3),
  permissions: { querySync: () => ({ state: "granted" }) },   // tests/weight_cache.js checks its --allow-write
};
await import(pathToFileURL(path.resolve(process.argv[2])).href);
