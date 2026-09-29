// Code mode's Download: the project as a .zip, or its app as one HTML file.
//
//   zip(files) -> Blob                      files = [{ path, bytes: Uint8Array, mtime? }]; a plain
//                                           zip (deflate where it helps, else stored), UTF-8 names
//   projectFiles(ws) -> { files, skipped }  every file of a workspace, secrets and dot files left out
//   snapshotOf(files) -> snapshot           the shape buildPreviewDoc takes, entry index.html
//   appHtml(snapshot, { path, title })      one .html file: the preview document, still sandboxed
//   saveBlob(blob, name)                    a phone's share sheet when it takes files, else a download
//
// The exported app is the model's code, so it leaves the room in the box the preview gives it: the
// file is a page with no script of its own that holds the preview document (CSP, capture script,
// every file inlined as data: URLs; harness/preview-build.js) in an iframe with
// sandbox="allow-scripts", as "Open" does (harness/preview-frame.js openPreviewTab). Opened from
// disk it runs in an opaque origin: no access to the file:// folder, cookies or storage, and no
// network beyond what the preview's CSP lets through.
import { buildPreviewDoc, mimeFor } from "./preview-build.js";
import { normPath, secretPath, SKIP_DIRS } from "./workspace.js";

// ---------------------------------------------------------------- zip (PKWARE APPNOTE 4.3)
const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  return t;
})();
export function crc32(u8) {
  let c = 0xffffffff;
  for (let i = 0; i < u8.length; i++) c = CRC[(c ^ u8[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
// MS-DOS date and time (local, 2 s steps, 1980 at the earliest)
function dosTime(d) {
  const y = Math.max(1980, Math.min(2107, d.getFullYear()));
  return { time: (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1), date: ((y - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate() };
}
async function deflateRaw(u8) {
  if (typeof CompressionStream !== "function") return null;
  try {
    const s = new Blob([u8]).stream().pipeThrough(new CompressionStream("deflate-raw"));
    return new Uint8Array(await new Response(s).arrayBuffer());
  } catch { return null; }   // no deflate-raw (older Safari): stored
}
const enc = new TextEncoder();
export const ZIP_MAX_BYTES = 0xfffffffe, ZIP_MAX_FILES = 0xfffe;   // no zip64

export async function zip(files, { now = new Date(), compress = true } = {}) {
  if (files.length > ZIP_MAX_FILES) throw new Error(`too many files for a zip (${files.length})`);
  const parts = [], central = [], seen = new Set();
  let off = 0;
  for (const f of files) {
    const path = normPath(f.path);   // throws on ".."
    if (!path || seen.has(path)) throw new Error(`bad or repeated path in the zip: ${f.path}`);
    seen.add(path);
    const name = enc.encode(path), raw = f.bytes instanceof Uint8Array ? f.bytes : enc.encode(String(f.bytes ?? ""));
    const crc = crc32(raw), { time, date } = dosTime(f.mtime ? new Date(f.mtime) : now);
    let data = raw, method = 0;
    if (compress && raw.length > 64) { const z = await deflateRaw(raw); if (z && z.length < raw.length) { data = z; method = 8; } }
    const h = new DataView(new ArrayBuffer(30));
    h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x0800, true);   // version 2.0, bit 11: UTF-8 names
    h.setUint16(8, method, true); h.setUint16(10, time, true); h.setUint16(12, date, true);
    h.setUint32(14, crc, true); h.setUint32(18, data.length, true); h.setUint32(22, raw.length, true);
    h.setUint16(26, name.length, true); h.setUint16(28, 0, true);
    const c = new DataView(new ArrayBuffer(46));
    c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true);
    c.setUint16(10, method, true); c.setUint16(12, time, true); c.setUint16(14, date, true);
    c.setUint32(16, crc, true); c.setUint32(20, data.length, true); c.setUint32(24, raw.length, true);
    c.setUint16(28, name.length, true);   // extra, comment, disk, internal and external attributes: 0
    c.setUint32(42, off, true);
    parts.push(new Uint8Array(h.buffer), name, data);
    central.push(new Uint8Array(c.buffer), name);
    off += 30 + name.length + data.length;
    if (off > ZIP_MAX_BYTES) throw new Error("the project is too big for a zip (4 GB)");
  }
  let cdSize = 0;
  for (const p of central) cdSize += p.length;
  const e = new DataView(new ArrayBuffer(22));
  e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true); e.setUint16(10, files.length, true);
  e.setUint32(12, cdSize, true); e.setUint32(16, off, true);
  return new Blob([...parts, ...central, new Uint8Array(e.buffer)], { type: "application/zip" });
}

// ---------------------------------------------------------------- the files
const skippedDir = (p) => p.split("/").slice(0, -1).some((s) => SKIP_DIRS.has(s));
// every file of the project, as the agent sees it: dot files and key files (secretPath) stay out
// of a file meant to be passed around, as do dependency and build folders
export async function projectFiles(ws, { maxFiles = 5000, maxBytes = 256 << 20 } = {}) {
  const files = [], skipped = [];
  let total = 0;
  const paths = (await ws.walk(maxFiles + 1)).map((p) => normPath(p)).filter(Boolean).sort();
  if (paths.length > maxFiles) throw new Error(`the project has over ${maxFiles} files`);
  for (const p of paths) {
    if (secretPath(p) || skippedDir(p)) { skipped.push(p); continue; }
    const bytes = await ws.readBytes(p);
    total += bytes.length;
    if (total > maxBytes) throw new Error(`the project is over ${maxBytes >> 20} MB`);
    files.push({ path: p, bytes });
  }
  return { files, skipped };
}
// a preview snapshot's files (a member has these, not the project)
export const snapshotFiles = (snap) => [...snap.files].map(([path, f]) => ({ path, bytes: f.bytes }));

// the project's files as a preview snapshot: entry index.html, else the shallowest .html
export function snapshotOf(files) {
  const map = new Map(files.map((f) => [f.path, { type: mimeFor(f.path), bytes: f.bytes, hash: "" }]));
  const pages = [...map.keys()].filter((p) => /\.html?$/i.test(p)).sort((a, b) => a.split("/").length - b.split("/").length || a.localeCompare(b));
  return { entry: map.has("index.html") ? "index.html" : pages[0] || "index.html", files: map };
}
export const hasPage = (snap) => [...snap.files.keys()].some((p) => /\.html?$/i.test(p));

const escHtml = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
// -> { html, missing, warnings }; html is the whole file
export function appHtml(snap, { path = null, title = "" } = {}) {
  const page = path && snap.files.has(path) ? path : snap.entry;
  const doc = buildPreviewDoc(snap, { path: page, nonce: "" });
  const t = title || /<title[^>]*>([^<]*)<\/title>/i.exec(new TextDecoder().decode(snap.files.get(page)?.bytes || new Uint8Array()))?.[1]?.trim() || page;
  const html = `<!doctype html>\n<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">`
    + `<meta name="referrer" content="no-referrer"><title>${escHtml(t)}</title>`
    + `<style>html,body{margin:0;height:100%;background:#fff}iframe{border:0;width:100%;height:100%;display:block}</style></head>\n`
    + `<!-- Made in Pooled Code mode (https://pooled.run). The app runs in a sandboxed frame, as in its preview. -->\n`
    + `<body><iframe sandbox="allow-scripts" allow="" referrerpolicy="no-referrer" title="${escHtml(t)}" srcdoc="${escHtml(doc.html)}"></iframe></body></html>\n`;
  return { html, missing: doc.missing, warnings: doc.warnings };
}

// "Tetris game!" -> "tetris-game"
export const fileBase = (name) => String(name || "").toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "")
  .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40).replace(/-+$/, "") || "project";

// ---------------------------------------------------------------- saving
// Touch screens get the share sheet (Save to Files, AirDrop, a messenger) when it takes files;
// everything else, and a share sheet that fails, gets an ordinary download. Cancelling the sheet
// is not an error. A share sheet refused because the tap is too long ago (the zip took a while;
// Safari keeps a tap's permission for about a second) returns "blocked": the caller offers a
// button that saves the same blob on a fresh tap. -> "shared" | "downloaded" | "cancelled" | "blocked"
export async function saveBlob(blob, name, { nav = globalThis.navigator, doc = globalThis.document, share = null } = {}) {
  const touch = share ?? !!globalThis.matchMedia?.("(pointer: coarse)").matches;
  if (touch && typeof nav?.share === "function" && typeof File === "function") {
    const file = new File([blob], name, { type: blob.type || "application/octet-stream" });
    let can = false;
    try { can = !nav.canShare || nav.canShare({ files: [file] }); } catch {}
    if (can) {
      try { await nav.share({ files: [file], title: name }); return "shared"; } catch (e) { if (e?.name === "AbortError") return "cancelled"; if (e?.name === "NotAllowedError") return "blocked"; }
    }
  }
  const url = URL.createObjectURL(blob);
  const a = doc.createElement("a");
  a.href = url; a.download = name; a.rel = "noopener"; a.style.display = "none";
  doc.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
  return "downloaded";
}
