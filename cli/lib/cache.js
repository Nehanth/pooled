// pooled pull / list / rm: the models this computer keeps, like `ollama pull`. A model's files go to
// <dir>/<model key>/ (the GGUF under its own name, and config.json / tokenizer.json for the dense
// models), where the room node looks first (packages/room-node/source.js), so `pooled host` and
// `pooled join` read them from disk instead of the network. The directory: --models, else
// POOLED_MODELS, else ~/.pooled/models.
//
// A download goes to <file>.part, resumes with a Range request after an interrupt (Ctrl-C, a lost
// network), is checked against the size (and the SHA-256, when room/models.js FILES has one) and is
// renamed when complete. No GPU here, and fetch is injected, so all of it is unit tested.
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { createHash } from "node:crypto";

export const defaultModelsDir = () => path.join(os.homedir(), ".pooled", "models");
export const modelsDir = ({ flag = null, env = process.env } = {}) => flag || env.POOLED_MODELS || defaultModelsDir();
// the models folder, made on first use (mkdir -p; only this user can write it) -> dir
export function ensureModelsDir(dir = modelsDir()) {
  fs.mkdirSync(dir, { recursive: true, mode: 0o755 });
  return dir;
}

// the files one model needs: [{ name, url, bytes?, sha256?, main }] (main: the GGUF)
export function modelFiles(key, MODELS, FILES = {}) {
  const M = MODELS[key];
  if (!M?.gguf) return [];
  const f = FILES[key] || {};
  // a split GGUF: every shard is a file, the first one is the main file (the loaders find the rest by name)
  const out = M.shards ? M.shards.map((url, i) => ({ name: url.split("/").pop(), url, bytes: f.shards?.[i]?.bytes || null, sha256: f.shards?.[i]?.sha256 || null, main: i === 0 }))
    : [{ name: M.gguf.split("/").pop(), url: M.gguf, bytes: f.bytes || null, sha256: f.sha256 || null, main: true }];
  // the dense engine reads config.json and tokenizer.json next to the GGUF (source.js sideFile)
  if (M.cfg) out.push({ name: "config.json", url: M.cfg, main: false });
  if (M.tok) out.push({ name: "tokenizer.json", url: M.tok, main: false });
  return out;
}

const size = (p) => { try { return fs.statSync(p).size; } catch { return null; } };
// the bytes a .part holds: a segmented download's .part is full size, its .json says how much is in
const partDone = (part) => {
  try { const m = JSON.parse(fs.readFileSync(part + ".json", "utf8")); return m.base + m.segs.reduce((a, g) => a + g.done, 0); }
  catch { return size(part) || 0; }
};

// what is on disk for a model: { pulled, bytes (complete files), partBytes (.part files), main (the
// GGUF's path when complete) }. local: the room node's older layout (packages/room-node/source.js
// LOCAL: file name -> path under dir, e.g. qwen17/model.gguf), which it also reads
export function modelState(dir, key, MODELS, FILES = {}, local = null) {
  const files = modelFiles(key, MODELS, FILES);
  const mf = files.find((f) => f.main);
  // where the files are: <dir>/<key>/ (pooled pull), else the older layout's folder
  let folder = path.join(dir, key), mainPath = mf ? path.join(folder, mf.name) : null;
  const alt = mf && local?.[mf.name] ? path.join(dir, local[mf.name]) : null;
  if (alt && size(mainPath) == null && size(alt) != null) { folder = path.dirname(alt); mainPath = alt; }
  let bytes = 0, partBytes = 0, pulled = files.length > 0, main = null;
  for (const f of files) {
    const p = f.main ? mainPath : path.join(folder, f.name);
    const s = size(p);
    if (s != null && (!f.bytes || s === f.bytes)) { bytes += s; if (f.main) main = p; }
    else { pulled = false; partBytes += partDone(p + ".part"); }
  }
  return { pulled, bytes, partBytes, main };
}

// every model that can be pulled (a GGUF in room/models.js), with its state; keys: the order to show
export function listModels(dir, MODELS, FILES = {}, keys = Object.keys(MODELS).filter((k) => MODELS[k].gguf), local = null) {
  return keys.map((key) => ({ key, label: MODELS[key].label, fileBytes: FILES[key]?.bytes || null, ...modelState(dir, key, MODELS, FILES, local) }));
}

// deletes a model's files (complete and partial) -> the bytes freed, or null when there were none
export function removeModel(dir, key) {
  const d = path.join(dir, key);
  if (!fs.existsSync(d)) return null;
  let freed = 0;
  for (const f of fs.readdirSync(d)) freed += size(path.join(d, f)) || 0;
  fs.rmSync(d, { recursive: true, force: true });
  return freed;
}

export class PullError extends Error {
  constructor(message, type) { super(message); this.type = type; }
}

// total size from a response: Content-Range's "/total", else Content-Length (+ the offset asked for)
function totalOf(res, from) {
  const cr = res.headers.get("content-range");
  const m = cr && /\/(\d+)\s*$/.exec(cr);
  if (m) return +m[1];
  const cl = res.headers.get("content-length");
  return cl != null && cl !== "" ? +cl + (res.status === 206 ? from : 0) : null;
}

async function sha256File(p, onProgress = () => {}) {
  const h = createHash("sha256");
  let done = 0;
  for await (const chunk of fs.createReadStream(p, { highWaterMark: 8 << 20 })) { h.update(chunk); done += chunk.length; onProgress(done); }
  return h.digest("hex");
}

// Download one file into dir (resuming its .part). onProgress({ name, done, total, resumed }) as bytes
// arrive; onVerify(done, total) while the SHA-256 is checked. -> { path, bytes, skipped, resumedFrom }
// segments: a file of known size (room/models.js FILES) and at least minSegmentedBytes downloads in
// this many ranges at once (one connection to Hugging Face gives ~35 MB/s on the Spark, four ~50+)
export async function pullFile(f, dir, { fetch = globalThis.fetch, signal, onProgress = () => {}, onVerify = () => {}, segments = 4, minSegmentedBytes = 64 * 2 ** 20 } = {}) {
  fs.mkdirSync(dir, { recursive: true });
  const dest = path.join(dir, f.name), part = dest + ".part";
  const have = size(dest);
  if (have != null && (!f.bytes || have === f.bytes)) return { path: dest, bytes: have, skipped: true, resumedFrom: 0 };
  if (have != null) fs.rmSync(dest, { force: true });   // a wrong size under the final name: fetch it again
  if (f.bytes && segments > 1 && f.bytes >= minSegmentedBytes) {
    const r = await pullSegmented(f, part, { fetch, signal, onProgress, segments });
    if (r) return finishPart(f, part, dest, r.from, { onVerify });
  }
  fs.rmSync(part + ".json", { force: true });
  let from = size(part) || 0;
  if (f.bytes && from > f.bytes) { fs.rmSync(part, { force: true }); from = 0; }
  let res = await fetch(f.url, { headers: from ? { range: `bytes=${from}-` } : {}, signal, redirect: "follow" });
  if (res.status === 416 && from) {
    // the .part may be whole already (Ctrl-C right after the last byte): the server says how big the file is
    const total = totalOf(res, 0) ?? f.bytes;
    try { await res.body?.cancel(); } catch {}
    if (total != null && from === total) res = null;
    else { fs.rmSync(part, { force: true }); from = 0; res = await fetch(f.url, { signal, redirect: "follow" }); }
  }
  let total = f.bytes || null;
  if (res) {
    if (!res.ok) throw new PullError(`${f.name}: the server answered ${res.status}${res.statusText ? ` ${res.statusText}` : ""}`, "http");
    // 200 to a range request: the server sends the whole file, so start over
    if (from && res.status !== 206) { fs.rmSync(part, { force: true }); from = 0; }
    const said = totalOf(res, from);
    if (f.bytes && said != null && said !== f.bytes)
      throw Object.assign(new PullError(`${f.name}: the server's file is ${said} bytes, not the ${f.bytes} this pooled expects (a different file?)`, "size"), { said });
    total = total || said;
    const out = fs.createWriteStream(part, { flags: from ? "a" : "w" });
    let done = from;
    const resumedFrom = from;
    onProgress({ name: f.name, done, total, resumed: resumedFrom });
    try {
      for await (const chunk of res.body) {
        if (!out.write(chunk)) await new Promise((r) => out.once("drain", r));
        done += chunk.length;
        if (total && done > total) throw new PullError(`${f.name}: got more than the ${total} bytes expected`, "size");
        onProgress({ name: f.name, done, total, resumed: resumedFrom });
      }
    } catch (e) {
      await new Promise((r) => out.end(r));
      if (e.type === "size") fs.rmSync(part, { force: true });
      if (signal?.aborted || e.name === "AbortError") throw new PullError("stopped", "aborted");
      throw e.type ? e : new PullError(`${f.name}: ${e.message}`, "network");
    }
    await new Promise((r, j) => out.end((e) => (e ? j(e) : r())));
  }
  return finishPart(f, part, dest, from, { total, onVerify });
}

// Several ranges at once into a .part of the file's full size, with <file>.part.json beside it saying
// how far each range got ({ total, base, segs: [{ start, end, done }] }), so Ctrl-C or a dropped network
// resumes each range where it stopped. A .part from a one-connection download (no .json) keeps its
// bytes: the ranges cover the rest. -> { from } once every byte is in; null when the server ignores
// ranges (the caller then downloads it in one piece); throws PullError otherwise.
async function pullSegmented(f, part, { fetch, signal, onProgress, segments }) {
  const total = f.bytes, metaPath = part + ".json";
  let meta = null;
  try { meta = JSON.parse(fs.readFileSync(metaPath, "utf8")); } catch {}
  if (!meta || meta.total !== total || !Array.isArray(meta.segs) || size(part) !== total) {
    // a .part without its .json: a one-connection download's first bytes
    let base = meta ? 0 : size(part) || 0;
    if (base > total) base = 0;
    if (!base) fs.rmSync(part, { force: true });
    const rest = total - base, n = Math.max(1, Math.min(segments, Math.ceil(rest / 2 ** 18)));
    const segs = [];
    for (let i = 0; i < n; i++) {
      const a = base + Math.floor((rest * i) / n), b = base + Math.floor((rest * (i + 1)) / n);
      if (b > a) segs.push({ start: a, end: b, done: 0 });
    }
    meta = { total, base, segs };
    const fd0 = fs.openSync(part, base ? "r+" : "w");
    fs.ftruncateSync(fd0, total);
    fs.closeSync(fd0);
    fs.writeFileSync(metaPath, JSON.stringify(meta));
  }
  const doneBytes = () => meta.base + meta.segs.reduce((a, g) => a + g.done, 0);
  const from = doneBytes();
  const save = () => { try { fs.writeFileSync(metaPath + ".tmp", JSON.stringify(meta)); fs.renameSync(metaPath + ".tmp", metaPath); } catch {} };
  const ac = new AbortController();
  const onAbort = () => ac.abort();
  signal?.addEventListener("abort", onAbort);
  if (signal?.aborted) ac.abort();
  const fh = await fs.promises.open(part, "r+");
  let noRanges = false, lastSave = Date.now();
  onProgress({ name: f.name, done: from, total, resumed: from });
  const one = async (g) => {
    if (g.done >= g.end - g.start) return;
    const res = await fetch(f.url, { headers: { range: `bytes=${g.start + g.done}-${g.end - 1}` }, signal: ac.signal, redirect: "follow" });
    if (res.status !== 206) {
      try { await res.body?.cancel(); } catch {}
      if (res.ok) { noRanges = true; throw new PullError("no ranges", "no-ranges"); }
      throw new PullError(`${f.name}: the server answered ${res.status}${res.statusText ? ` ${res.statusText}` : ""}`, "http");
    }
    const said = totalOf(res, 0);
    if (said != null && said !== total) throw Object.assign(new PullError(`${f.name}: the server's file is ${said} bytes, not the ${total} this pooled expects (a different file?)`, "size"), { said });
    for await (const chunk of res.body) {
      const want = g.end - g.start - g.done;
      const n = Math.min(chunk.length, want);
      if (n > 0) await fh.write(chunk, 0, n, g.start + g.done);
      g.done += n;
      onProgress({ name: f.name, done: doneBytes(), total, resumed: from });
      if (Date.now() - lastSave > 1000) { lastSave = Date.now(); save(); }
      if (g.done >= g.end - g.start) break;
    }
    if (g.done < g.end - g.start) throw new PullError(`${f.name}: a range stopped at ${g.start + g.done} of ${g.end}`, "short");
  };
  try {
    const rs = await Promise.allSettled(meta.segs.map((g) => one(g).catch((e) => { ac.abort(); throw e; })));
    await fh.close();
    save();
    if (noRanges) { fs.rmSync(metaPath, { force: true }); fs.rmSync(part, { force: true }); return null; }
    // the range that failed first, not the others it stopped (AbortError)
    const bad = rs.find((r) => r.status === "rejected" && r.reason?.name !== "AbortError")?.reason
      || rs.find((r) => r.status === "rejected")?.reason;
    if (bad) {
      if (signal?.aborted) throw new PullError("stopped", "aborted");
      if (bad.type === "size") { fs.rmSync(metaPath, { force: true }); fs.rmSync(part, { force: true }); }
      if (bad instanceof PullError) throw bad;
      if (bad.name === "AbortError") throw new PullError(`${f.name}: stopped`, "network");
      throw new PullError(`${f.name}: ${bad.message}`, "network");
    }
    if (doneBytes() !== total) throw new PullError(`${f.name}: the download stopped at ${doneBytes()} of ${total} bytes (run it again to resume)`, "short");
    fs.rmSync(metaPath, { force: true });
    return { from };
  } finally { signal?.removeEventListener("abort", onAbort); try { await fh.close(); } catch {} }
}

// a .part that is all there: its size, its SHA-256, then its final name
async function finishPart(f, part, dest, from, { total = f.bytes || null, onVerify = () => {} } = {}) {
  const got = size(part) || 0;
  if (total != null && got !== total) {
    if (got > total) { fs.rmSync(part, { force: true }); throw new PullError(`${f.name}: ${got} bytes, more than the ${total} expected`, "size"); }
    throw new PullError(`${f.name}: the download stopped at ${got} of ${total} bytes (run it again to resume)`, "short");
  }
  if (f.sha256) {
    const h = await sha256File(part, (d) => onVerify(d, got));
    if (h !== f.sha256) { fs.rmSync(part, { force: true }); throw new PullError(`${f.name}: the SHA-256 does not match (${h.slice(0, 12)}… instead of ${f.sha256.slice(0, 12)}…); removed it`, "hash"); }
  }
  fs.renameSync(part, dest);
  return { path: dest, bytes: got, skipped: false, resumedFrom: from };
}

// Pull every file of a model. onProgress({ file, done, total, allDone, allTotal }) over the whole
// model (sizes of the small side files count once known). -> { dir, bytes, skipped }
export async function pullModel(key, { dir, MODELS, FILES = {}, fetch = globalThis.fetch, signal, onProgress = () => {}, onVerify = () => {}, segments, minSegmentedBytes } = {}) {
  const files = modelFiles(key, MODELS, FILES);
  if (!files.length) throw new PullError(`${key} is not a model this pooled can download`, "unknown");
  const d = path.join(dir, key);
  // the side files first (small): the progress is then about the GGUF
  const order = [...files.filter((f) => !f.main), ...files.filter((f) => f.main)];
  let bytes = 0, skipped = true;
  for (const f of order) {
    const r = await pullFile(f, d, { fetch, signal, segments, minSegmentedBytes, onVerify: (a, b) => onVerify(a, b, f),
      onProgress: (p) => { if (f.main) onProgress({ file: f.name, done: p.done, total: p.total, resumed: p.resumed }); } });
    bytes += r.bytes; skipped &&= r.skipped;
    if (f.main && r.skipped) onProgress({ file: f.name, done: r.bytes, total: r.bytes, resumed: 0 });
  }
  return { dir: d, bytes, skipped };
}

// ---------------- showing it ----------------
export const fmtBytes = (b) => {
  if (b == null) return "?";
  if (b >= 2 ** 30) return `${(b / 2 ** 30).toFixed(b >= 100 * 2 ** 30 ? 0 : 1)} GB`;
  if (b >= 2 ** 20) return `${Math.round(b / 2 ** 20)} MB`;
  if (b >= 1024) return `${Math.round(b / 1024)} KB`;
  return `${b} B`;
};
export const fmtEta = (s) => {
  if (!Number.isFinite(s) || s < 0) return "--";
  s = Math.round(s);
  if (s >= 3600) return `${Math.floor(s / 3600)}h${String(Math.floor(s % 3600 / 60)).padStart(2, "0")}m`;
  if (s >= 60) return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
  return `${s}s`;
};

// the speed over the last few seconds: rate(done) -> bytes/s (null until there is enough to say)
export function rateMeter(windowMs = 5000, now = () => Date.now()) {
  const pts = [];
  return (done) => {
    const t = now();
    pts.push([t, done]);
    while (pts.length > 2 && t - pts[0][0] > windowMs) pts.shift();
    const [t0, d0] = pts[0];
    return t - t0 >= 250 ? ((done - d0) / (t - t0)) * 1000 : null;
  };
}

// one progress line: "▕████████░░░░▏ 42%  0.8 / 1.8 GB  56 MB/s  ETA 18s", to width columns
export function progressLine({ done, total, bps = null, width = 80, label = "" }) {
  const pct = total ? Math.min(100, Math.floor((done / total) * 100)) : null;
  const tail = [
    pct != null ? `${String(pct).padStart(3)}%` : "",
    total ? `${fmtBytes(done)} / ${fmtBytes(total)}` : fmtBytes(done),
    bps != null ? `${fmtBytes(bps)}/s` : "",
    total && bps ? `ETA ${fmtEta((total - done) / bps)}` : "",
  ].filter(Boolean).join("  ");
  const head = label ? `${label} ` : "";
  const barW = Math.max(0, Math.min(30, width - head.length - tail.length - 4));
  if (pct == null || barW < 6) return (head + tail).slice(0, Math.max(1, width));
  const fill = Math.round((pct / 100) * barW);
  return `${head}▕${"█".repeat(fill)}${"░".repeat(barW - fill)}▏ ${tail}`;
}
// the same without a terminal: a plain line (for logs)
export const progressPlain = ({ done, total, bps }) =>
  `${total ? `${Math.floor((done / total) * 100)}% ` : ""}${fmtBytes(done)}${total ? ` of ${fmtBytes(total)}` : ""}${bps ? ` at ${fmtBytes(bps)}/s` : ""}${total && bps ? `, ${fmtEta((total - done) / bps)} left` : ""}`;

// what was typed -> a model key: the key itself, any case, or a part of it that names one model
// ("35b", "moe", "1.7b"). -> { key } | { error, choices }
export function resolveModel(input, keys) {
  const t = String(input || "").trim().toLowerCase();
  if (!t) return { error: "give a model", choices: keys };
  if (keys.includes(t)) return { key: t };
  const hits = keys.filter((k) => k.includes(t));
  if (hits.length === 1) return { key: hits[0] };
  return { error: hits.length ? `"${input}" matches more than one model: ${hits.join(", ")}` : `unknown model "${input}"`, choices: hits.length ? hits : keys };
}

// ---------------- one writer per model: <dir>/<model>/.pull.lock ----------------
// Every download into the models folder takes this lock first: pooled pull, pooled host and pooled
// join (pullrun.js pullWithProgress), and the OpenClaw plugin's onboarding and background pull
// (packages/openclaw), which write the same <file>.part. The file holds { pid, host, at }. A lock whose
// process is gone (a crash, a kill) is stale; one from another machine (a shared folder) counts for a
// day. A stale lock is removed only if it is unchanged since it was read: another process may have
// taken it over meanwhile, and removing its fresh lock would let two writers into one .part.
export const pullLockPath = (dir, key) => path.join(dir, key, ".pull.lock");
const pidAlive = (pid) => {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; } catch (e) { return e.code === "EPERM"; }
};
// -> a release() function, or null when another live process holds the lock
export function tryPullLock(dir, key, { pid = process.pid, isAlive = pidAlive, hostname = os.hostname() } = {}) {
  const p = pullLockPath(dir, key);
  fs.mkdirSync(path.dirname(p), { recursive: true });
  for (let i = 0; i < 2; i++) {
    try {
      const fd = fs.openSync(p, "wx");
      fs.writeSync(fd, JSON.stringify({ pid, host: hostname, at: new Date().toISOString() }));
      fs.closeSync(fd);
      let done = false;
      return () => { if (done) return; done = true; try { if (JSON.parse(fs.readFileSync(p, "utf8")).pid === pid) fs.rmSync(p, { force: true }); } catch {} };
    } catch (e) {
      if (e.code !== "EEXIST") throw e;
      let holder = null, raw = null;
      try { raw = fs.readFileSync(p, "utf8"); holder = JSON.parse(raw); } catch {}
      const foreign = holder?.host && holder.host !== hostname;
      const stale = !holder || (foreign ? Date.now() - Date.parse(holder.at || 0) > 864e5 : !isAlive(holder.pid));
      if (!stale) return null;
      try { if (fs.readFileSync(p, "utf8") === raw) fs.rmSync(p, { force: true }); } catch {}
    }
  }
  return null;
}
// wait for the lock (another process downloading the same model) until it is free or signal aborts
// -> release(); onWait() once, when it has to wait
export async function pullLock(dir, key, { signal, pollMs = 2000, onWait = () => {}, ...o } = {}) {
  let said = false;
  for (;;) {
    const rel = tryPullLock(dir, key, o);
    if (rel) return rel;
    if (!said) { said = true; onWait(); }
    if (signal?.aborted) throw Object.assign(new Error("stopped"), { type: "aborted" });
    await new Promise((r) => setTimeout(r, pollMs));
  }
}
