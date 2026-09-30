// One download of a model at a time: onboarding's "Download now" and the gateway's background pull
// write the same <dir>/<model>/<file>.part (cli/lib/cache.js), so each takes <dir>/<model>/.pull.lock
// first ({ pid, host, at }). A lock whose process is gone (a crash, a kill) is taken over.
// (`pooled pull` from @pooled/cli does not take this lock yet.)
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export const lockPath = (dir, key) => path.join(dir, key, ".pull.lock");

const alive = (pid) => {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; } catch (e) { return e.code === "EPERM"; }
};

// -> a release() function, or null when another live process holds the lock
export function tryLock(dir, key, { pid = process.pid, isAlive = alive } = {}) {
  const p = lockPath(dir, key);
  fs.mkdirSync(path.dirname(p), { recursive: true });
  for (let i = 0; i < 2; i++) {
    try {
      const fd = fs.openSync(p, "wx");
      fs.writeSync(fd, JSON.stringify({ pid, host: os.hostname(), at: new Date().toISOString() }));
      fs.closeSync(fd);
      let done = false;
      return () => { if (done) return; done = true; try { if (JSON.parse(fs.readFileSync(p, "utf8")).pid === pid) fs.rmSync(p, { force: true }); } catch {} };
    } catch (e) {
      if (e.code !== "EEXIST") throw e;
      let holder = null;
      try { holder = JSON.parse(fs.readFileSync(p, "utf8")); } catch {}
      // another machine's lock on a shared folder: trust it for a day
      const foreign = holder?.host && holder.host !== os.hostname();
      const stale = !holder || (foreign ? Date.now() - Date.parse(holder.at || 0) > 864e5 : !isAlive(holder.pid));
      if (!stale) return null;
      try { fs.rmSync(p, { force: true }); } catch {}
    }
  }
  return null;
}

// wait for the lock (another process downloading the same model): polls until it is free or signal
// aborts -> release()
export async function lock(dir, key, { signal, pollMs = 2000, onWait = () => {}, ...o } = {}) {
  let said = false;
  for (;;) {
    const rel = tryLock(dir, key, o);
    if (rel) return rel;
    if (!said) { said = true; onWait(); }
    if (signal?.aborted) throw Object.assign(new Error("stopped"), { type: "aborted" });
    await new Promise((r) => setTimeout(r, pollMs));
  }
}
