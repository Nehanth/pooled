// Node runtime for a headless room node: WebGPU from Dawn (npm "webgpu", dawn.node) and WebRTC from
// libdatachannel (node-datachannel/polyfill) under the same PeerJS client the room page loads. Both
// live in this one process and share its event loop: no browser, no child process.
//
// setupNode() is idempotent and must run before anything touches navigator.gpu or new Peer().

import { setFlagsFromString } from "node:v8";
import { measureCopyGBps } from "../../room/gpuspeed.js";
import { DENSE_SPEC_V } from "../../room/lookup.js";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import os from "node:os";
import { detectMemory } from "../../cli/lib/lend.js";

let ready = null;
export let Peer = null;
export const runtime = { gpu: null, rtc: null };

// Dawn's options: DAWN_OPTS (space separated, e.g. "enable-dawn-features=dump_shaders"), and on Windows
// d3d_skip_shader_optimizations. Dawn's D3D12 backend compiles HLSL with FXC there (npm webgpu ships
// no DXC), and FXC's optimizer is most of a slow kernel's compile: attn_flash_tile 13.3 s -> 2.0 s,
// attn_dec 11.2 s -> 2.0 s, a MoE shard's pipelines ~2x faster in all on an RTX 5070. The driver
// optimizes the DXBC again when it builds the GPU code: same outputs bit for bit, decode ~2-3% slower
// on that PC's 8-layer MoE shard (1.96 vs 1.91 ms). POOLED_FXC_OPTIMIZE=1 keeps FXC's optimizer.
export function dawnFlagsFor(env = process.env, platform = process.platform) {
  const flags = (env.DAWN_OPTS || "").split(" ").filter(Boolean);
  if (platform !== "win32" || env.POOLED_FXC_OPTIMIZE === "1" || flags.some((f) => f.includes("d3d_skip_shader_optimizations"))) return flags;
  const i = flags.findIndex((f) => f.startsWith("enable-dawn-features="));
  if (i < 0) return [...flags, "enable-dawn-features=d3d_skip_shader_optimizations"];
  return flags.map((f, j) => (j === i ? f + ",d3d_skip_shader_optimizations" : f));   // one list: Dawn reads the last flag of a name
}

// V8's flags for a process that holds a Dawn device. On Windows (Node 24, npm webgpu 0.6.1, RTX 5070)
// the process dies with an access violation (0xC0000005) when V8's Maglev compiler optimizes a hot
// function once a GPU device exists, even with no GPU work going on; --no-maglev (or --max-opt=1, or
// --no-concurrent-recompilation) avoids it. Hot functions then go from Sparkplug straight to TurboFan.
// The flag is set at run time (the OpenClaw plugin runs inside the gateway's process, which we can't
// re-exec): V8 compiles nothing with Maglev after it, and what Node had already compiled with Maglev
// before (path helpers while loading modules) predates the device.
export const v8FlagsFor = (platform = process.platform) => (platform === "win32" ? ["--no-maglev"] : []);

// webgpu: an async loader for Dawn ({ create, globals }), for a caller that resolves it from its own
// install (the pooled CLI: an optional package); default the npm "webgpu" package next to this one
export function setupNode({ dawnFlags = dawnFlagsFor(), webgpu = () => import("webgpu"), v8Flags = v8FlagsFor() } = {}) {
  return ready ||= (async () => {
    for (const f of v8Flags) setFlagsFromString(f);   // before Dawn makes a device
    // --- WebGPU (Dawn). Node 21+ already has a navigator object; add gpu to it.
    if (!globalThis.navigator?.gpu) {
      const { create, globals } = await webgpu();
      Object.assign(globalThis, globals);   // GPUBufferUsage, GPUMapMode, GPUShaderStage ...
      const gpu = highPerformance(create(dawnFlags));
      if (globalThis.navigator) Object.defineProperty(globalThis.navigator, "gpu", { value: gpu, configurable: true });
      else globalThis.navigator = { gpu, userAgent: "pooled-node" };
      runtime.gpu = "dawn.node";
    } else runtime.gpu = "native navigator.gpu";
    // --- WebRTC for PeerJS (as cli/lib/room.js does for the ask-only bridge)
    const rtc = await import("node-datachannel/polyfill");
    for (const [k, v] of Object.entries(rtc)) if (k !== "default" && globalThis[k] === undefined) globalThis[k] = v;
    globalThis.window ??= globalThis;
    globalThis.location ??= { protocol: "https:", search: "", hostname: "node" };   // peerjs util.isSecure() reads it
    const m = await import("peerjs");
    Peer = (m.default?.Peer ? m.default : m).Peer;
    runtime.rtc = "node-datachannel";
    return { Peer };
  })();
}

// Every adapter this process asks for is the high-performance one unless the caller says otherwise: without a
// preference Dawn may hand out the integrated GPU (the PC's Core Ultra iGPU instead of its RTX 5070), and a shard's
// self-test device would then test another GPU than the one that runs it (tests/dawn_run.mjs does the same).
export function highPerformance(gpu) {
  const ra = gpu.requestAdapter.bind(gpu);
  const req = (o = {}) => ra({ powerPreference: "high-performance", ...o });
  try { Object.defineProperty(gpu, "requestAdapter", { value: req, configurable: true }); } catch {}
  if (gpu.requestAdapter !== req) return new Proxy(gpu, { get: (t, k) => (k === "requestAdapter" ? req : typeof t[k] === "function" ? t[k].bind(t) : t[k]) });
  return gpu;
}

// What this device tells the room in its hello (the page's probeGPU, without the DOM): the room deals
// layers to devices with meta.webgpu and weighs them by meta.contribGB; meta.gbps (room/gpuspeed.js)
// lets a clearly faster GPU be the model host (room/plan.js pickModelHost). gbps: pin it (0 = unknown).
// ramGB: system RAM this device lets the room park a MoE model's experts in (expert offload: meta.offload and
// meta.ramGB, room/pledge.js ramGB); 0 or none: it doesn't offload. Browsers never do, and neither does a GPU on
// unified memory (Apple silicon, a GB10 / Jetson): its "RAM" is the memory its pledge already lends, so parking
// experts there frees nothing and spends the same memory twice. mem: cli/lib/lend.js detectMemory()'s answer
// (the CLI has it); without one, probeMeta asks the OS itself, only when ramGB asks for offload.
export async function probeMeta(pledgeGB, { gbps = null, ramGB = 0, mem = null } = {}) {
  const ua = process.platform === "darwin" ? "Mac" : "Device";
  // dspec: dense verify frames of any column count (the repo's engine; room/lookup.js chainDenseSpec)
  const meta = { ua, webgpu: false, gpu: "no WebGPU", maxBufGB: 0, native: "node-dawn", dspec: DENSE_SPEC_V };
  const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (a) {
    const info = a.info || {};
    meta.webgpu = true;
    meta.gpu = [...new Set([info.vendor, info.architecture || info.device].filter(Boolean))].join(" ") || "GPU";
    meta.maxBufGB = +(a.limits.maxBufferSize / 2 ** 30).toFixed(1);
    meta.budgetGB = meta.maxBufGB;
    // the largest buffer this device can bind (shard.js asks for the adapter's limit): the host keeps
    // the room's context within every device's limit (room/models.js ctxForBinding)
    meta.maxBindMB = Math.floor(Math.min(a.limits.maxStorageBufferBindingSize, a.limits.maxBufferSize) / 2 ** 20);
    meta.gbps = gbps != null ? Math.max(0, +gbps || 0) : await measureCopyGBps(a);
  }
  meta.phone = false;
  meta.contribGB = pledgeFor(pledgeGB, meta.maxBufGB);
  if (meta.webgpu && +ramGB > 0) {
    if (unifiedMemory(mem || hostMemory())) meta.noOffload = "unified memory";
    else { meta.offload = true; meta.ramGB = ramFor(ramGB); }
  }
  return meta;
}
// whether the GPU shares the system's memory (detectMemory's kind; Apple silicon reads as unified there too)
export const unifiedMemory = (mem) => mem?.kind === "unified";
// detectMemory on this machine (nvidia-smi, amdgpu sysfs, the platform), or { kind: "unknown" } when it fails
export function hostMemory() {
  try {
    return detectMemory({
      run: (cmd, args) => execFileSync(cmd, args, { encoding: "utf8", timeout: 5000, stdio: ["ignore", "pipe", "ignore"] }),
      read: (p) => readFileSync(p, "utf8"), totalmem: os.totalmem, freemem: os.freemem });
  } catch (e) { return { kind: "unknown", why: String(e?.message || e) }; }
}
// the RAM this device lets the room park experts in, GB: what it was told, at most 64 (as a pledge), whole tenths
export const ramFor = (gb) => Math.max(0, Math.min(64, Math.floor((+gb || 0) * 10) / 10));
// the GB this device lends: what it was told, else half its largest buffer; 1..64
export const pledgeFor = (pledgeGB, maxBufGB) => Math.min(64, Math.max(1, +pledgeGB || Math.round(maxBufGB * 0.5) || 1));
