// Node runtime for a headless room node: WebGPU from Dawn (npm "webgpu", dawn.node) and WebRTC from
// libdatachannel (node-datachannel/polyfill) under the same PeerJS client the room page loads. Both
// live in this one process and share its event loop: no browser, no child process.
//
// setupNode() is idempotent and must run before anything touches navigator.gpu or new Peer().

import { measureCopyGBps } from "../../room/gpuspeed.js";

let ready = null;
export let Peer = null;
export const runtime = { gpu: null, rtc: null };

export function setupNode({ dawnFlags = (process.env.DAWN_OPTS || "").split(" ").filter(Boolean) } = {}) {
  return ready ||= (async () => {
    // --- WebGPU (Dawn). Node 21+ already has a navigator object; add gpu to it.
    if (!globalThis.navigator?.gpu) {
      const { create, globals } = await import("webgpu");
      Object.assign(globalThis, globals);   // GPUBufferUsage, GPUMapMode, GPUShaderStage ...
      const gpu = create(dawnFlags);
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

// What this device tells the room in its hello (the page's probeGPU, without the DOM): the room deals
// layers to devices with meta.webgpu and weighs them by meta.contribGB; meta.gbps (room/gpuspeed.js)
// lets a clearly faster GPU be the model host (room/plan.js pickModelHost). gbps: pin it (0 = unknown).
export async function probeMeta(pledgeGB, { gbps = null } = {}) {
  const ua = process.platform === "darwin" ? "Mac" : "Device";
  const meta = { ua, webgpu: false, gpu: "no WebGPU", maxBufGB: 0, native: "node-dawn" };
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
  return meta;
}
// the GB this device lends: what it was told, else half its largest buffer; 1..64
export const pledgeFor = (pledgeGB, maxBufGB) => Math.min(64, Math.max(1, +pledgeGB || Math.round(maxBufGB * 0.5) || 1));
