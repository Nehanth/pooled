// Dawn (WebGPU for Node) for pooled join / pooled host, loaded only when one of them runs.
//
// pooled serve needs no GPU, and Dawn is big (npm "webgpu" 0.6.1 is ~95 MB unpacked with all five
// OS builds), so @pooled/cli does not depend on it: it is an optional peer dependency. Where it
// comes from, first found wins:
//   1. @pooled/dawn-<platform>-<arch>: one OS's build (planned per-OS packages, exporting
//      { create, globals } as "webgpu" does);
//   2. webgpu: the upstream package, installed next to @pooled/cli
//      (npm i -g @pooled/cli webgpu@0.6.1, or npx -p @pooled/cli -p webgpu@0.6.1 pooled join CODE).
// Each is resolved from this package's own install, so a global install finds a global sibling.
// In a Pooled checkout it is also looked up from packages/room-node, which depends on webgpu:
// npm 11 won't add an optional peer dependency to cli/ itself (`npm install --no-save webgpu` in
// cli/ is "up to date" and installs nothing), so (cd packages/room-node && npm install) is the way there.
import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { pathToFileURL } from "node:url";

export const DAWN_VERSION = "0.6.1";   // the webgpu release the engine is tested on (packages/room-node)
const arch = () => (process.platform === "darwin" ? "universal" : process.arch);
export const dawnPackages = (platform = process.platform, a = arch()) => [`@pooled/dawn-${platform}-${platform === "darwin" ? "universal" : a}`, "webgpu"];

export function dawnMissing(cmd = "join", cause = null) {
  const e = new Error(`pooled ${cmd} needs Dawn, the WebGPU addon for Node (~95 MB), which pooled serve doesn't install.`);
  e.type = "dawn-missing";
  e.hint = `Install it next to pooled:\n    npm install -g @pooled/cli webgpu@${DAWN_VERSION}\n  or run it without installing:\n    npx -p @pooled/cli -p webgpu@${DAWN_VERSION} pooled ${cmd}${cmd === "join" ? " <CODE>" : ""}`;
  if (cause) e.cause = cause;
  return e;
}

// -> an async loader for setupNode({ webgpu }) that throws dawnMissing() when nothing is installed.
// name -> path, or throws: from this package's install, then from a checkout's packages/room-node
function defaultResolve(name) {
  try { return createRequire(import.meta.url).resolve(name); }
  catch (e) {
    const rn = new URL("../../packages/room-node/package.json", import.meta.url);
    if (!existsSync(rn)) throw e;
    return createRequire(rn).resolve(name);
  }
}

// resolve(name) -> path (or throws); injected for tests
export function dawnLoader(cmd, { resolve = defaultResolve, load = (p) => import(pathToFileURL(p).href) } = {}) {
  return async () => {
    let lastErr = null;
    for (const name of dawnPackages()) {
      let p;
      try { p = resolve(name); } catch (e) { lastErr = e; continue; }
      try {
        const m = await load(p);
        if (typeof m.create === "function" && m.globals) return { create: m.create, globals: m.globals };
      } catch (e) {
        // installed but it would not load (a missing system library, an OS older than its build)
        const err = new Error(`${name} is installed but did not load: ${String(e?.message || e).split("\n")[0]}`);
        err.type = "dawn-broken";
        err.hint = "Dawn's prebuilt addon needs macOS 26 or newer, Linux with glibc 2.38 or newer (Ubuntu 24.04, Debian 13, Fedora 39), or Windows 10/11.";
        throw err;
      }
    }
    throw dawnMissing(cmd, lastErr);
  };
}
