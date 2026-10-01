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
import fs, { existsSync } from "node:fs";
import os from "node:os";
import path from "node:path";
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

// ---------------- the GPU driver's own messages ----------------
// Bringing up Dawn makes the system's Vulkan drivers write to stderr from native code: on a GB10,
// "MESA: error: Opening /dev/dri/card0 failed: Permission denied", "TU: error: ... VK_ERROR_
// INCOMPATIBLE_DRIVER" (Mesa drivers for GPUs this computer does not have, probing) and Dawn's
// "Warning: maxDynamicUniformBuffersPerPipelineLayout artificially reduced ...". All harmless, and
// out of reach of process.stderr. So while Dawn runs its native setup (create(), requestAdapter(),
// requestDevice(): the messages come out synchronously inside those calls), fd 2 points at a file of
// our own; then it points back at the terminal. Node has no dup2, but close(2) + open() takes the
// lowest free descriptor, which is 2 again. What was caught is kept (driverLog()) for an error that
// needs it, and --verbose skips all of this.
let caught = "";
export const driverLog = () => caught;

// fn() with what native code writes to fd 2 caught -> fn's value. Linux and macOS; elsewhere, or when
// stderr can't be reopened (a socket: a Node parent's stdio "pipe", a systemd journal stream), it
// just runs fn.
export function quietStderr(fn, { platform = process.platform } = {}) {
  const self = platform === "linux" ? "/proc/self/fd/" : platform === "darwin" ? "/dev/fd/" : null;
  if (!self) return fn();
  let saved = -1, file = null;
  try {
    saved = fs.openSync(self + "2", "a");   // a second way to the same terminal / pipe / file
    file = path.join(os.tmpdir(), `pooled-gpu-${process.pid}.log`);
    fs.closeSync(fs.openSync(file, "w", 0o600));
  } catch { if (saved >= 0) try { fs.closeSync(saved); } catch {} return fn(); }
  let swapped = false;
  try {
    fs.closeSync(2);
    const fd = fs.openSync(file, "a");
    swapped = fd === 2;
    // (another thread opened a file in between and got 2: leave it to its owner)
    if (!swapped) fs.closeSync(fd);
  } catch {}
  try { return fn(); }
  finally {
    if (swapped) {
      try { fs.closeSync(2); fs.openSync(self + saved, "a"); } catch {}
      try { caught += fs.readFileSync(file, "utf8"); } catch {}
    }
    try { fs.closeSync(saved); } catch {}
    try { fs.unlinkSync(file); } catch {}
    if (caught.length > 64 * 1024) caught = caught.slice(-64 * 1024);
  }
}

// a Dawn loader (dawnLoader()) whose native setup runs under quietStderr: create(), and every
// requestAdapter() / requestDevice() after it
export function quietLoader(loader, { quiet = quietStderr } = {}) {
  return async () => {
    const m = await loader();
    const wrapAdapter = (a) => {
      if (!a || a.__pooledQuiet) return a;
      const rd = a.requestDevice;
      try { Object.defineProperty(a, "requestDevice", { value: (...x) => quiet(() => rd.apply(a, x)), configurable: true }); a.__pooledQuiet = true; } catch {}
      return a;
    };
    return { globals: m.globals, create: (flags) => {
      const gpu = quiet(() => m.create(flags));
      const ra = gpu.requestAdapter;
      try { Object.defineProperty(gpu, "requestAdapter", { value: (...x) => quiet(() => ra.apply(gpu, x)).then(wrapAdapter), configurable: true }); } catch {}
      return gpu;
    } };
  };
}
