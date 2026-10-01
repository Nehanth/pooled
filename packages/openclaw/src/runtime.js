// The Pooled runtime, loaded on first use so the plugin registers without touching the GPU or WebRTC:
// @pooled/room-node (Pooled's engine on Dawn + the room protocol over node-datachannel), the
// `pooled serve` bridge (an ask-only link to another host) and the CLI's Dawn loader. In the npm
// package all of it is bundled into dist/index.js (package.json build); in a checkout it is imported
// from the repo. Dawn (npm "webgpu") is an optional dependency: a machine without it can still join a
// room as an asker, and gets a plain install hint when it needs the GPU.
import { dawnLoader, DAWN_VERSION } from "../../../cli/lib/dawn.js";

export class PooledError extends Error {
  constructor(code, message, hint = null) { super(message); this.code = code; if (hint) this.hint = hint; }
}

// Dawn for this process: the CLI's loader (resolves "webgpu" from this package's install), with the
// errors worded for OpenClaw
export function dawnFor(loader = dawnLoader("openclaw")) {
  return async () => {
    try { return await loader(); }
    catch (err) {
      if (err.type === "dawn-missing")
        throw new PooledError("install", `this machine has no WebGPU for Node (npm "webgpu" ${DAWN_VERSION}, an optional dependency of @pooled/openclaw that did not install). ` +
          `Install it next to the plugin: cd ${pluginDirHint()} && npm install webgpu@${DAWN_VERSION}, then restart the gateway`);
      if (err.type === "dawn-broken") throw new PooledError("install", `${err.message}. ${err.hint || ""}`.trim());
      throw err;
    }
  };
}
const pluginDirHint = () => { try { return new URL("..", import.meta.url).pathname.replace(/\/$/, ""); } catch { return "<the plugin's folder>"; } };

let modP = null;
export async function pooledModules() {
  return modP ||= (async () => {
    try {
      const node = await import("../../room-node/index.js");
      const gate = await import("../../room-node/gate.js");
      const { Bridge } = await import("../../../cli/lib/room.js");
      return { ...node, gate, Bridge, dawn: dawnFor() };
    } catch (err) {
      modP = null;
      throw new PooledError("install", `the Pooled runtime did not load (${err.message}). Reinstall the plugin: openclaw plugins install @pooled/openclaw`);
    }
  })();
}
// tests: a fake runtime
export function setModules(m) { modP = m ? Promise.resolve(m) : null; }
