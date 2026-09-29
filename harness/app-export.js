// "Share with the room" -> Download (room/code.js): a built app as one .html file a person keeps.
// The file is built on each device from its own copy of the preview (on a peer, the snapshot
// PreviewSubscriber assembled from hash-checked blobs), with buildPreviewDoc, the same document the
// preview frame runs. It stays untrusted model output, so it is sandboxed exactly like a preview:
// the file is a wrapper page holding only a sandbox="allow-scripts" frame whose srcdoc is the app,
// so the app runs in an opaque origin wherever the file is opened (no storage, cookies or DOM of
// the page that opened it). The wrapper carries the preview's CSP too; a srcdoc document inherits
// its parent's policy, so the app gets the same network rules as in the room (none but the two
// script CDNs and Google Fonts). No script of our own and nothing about the room in the file.
//
//   appPage(snap, { title, path }) -> html      snap = { entry, files: Map<path, {type, bytes, hash}> }
//   appFileName(name, rev) -> "tetris-rev3.html"
//   downloadApp(snap, { name, rev }) -> the file name (saved through a download link)
import { buildPreviewDoc, CSP } from "./preview-build.js";

const escAttr = (t) => String(t).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

export function appPage(snap, { title = "app", path = null } = {}) {
  const at = path && snap.files.has(path) ? path : snap.entry;
  const { html } = buildPreviewDoc(snap, { path: at, nonce: "" });
  return `<!doctype html><html lang="en"><head><meta charset="utf-8">`
    + `<meta http-equiv="Content-Security-Policy" content="${escAttr(CSP)}">`
    + `<meta name="referrer" content="no-referrer">`
    + `<meta name="viewport" content="width=device-width, initial-scale=1"><title>${escAttr(String(title).slice(0, 80))}</title>`
    + `<style>html,body{margin:0;height:100%;background:#fff}iframe{border:0;width:100%;height:100%;display:block}</style></head>`
    + `<body><iframe sandbox="allow-scripts" allow="" referrerpolicy="no-referrer" title="${escAttr(String(title).slice(0, 80))}" srcdoc="${escAttr(html)}"></iframe></body></html>\n`;
}

export function appFileName(name, rev) {
  const base = String(name || "app").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) || "app";
  return `${base}${rev ? "-rev" + (rev >>> 0) : ""}.html`;
}

export function downloadApp(snap, { name = "app", rev = 0 } = {}, doc = globalThis.document) {
  const file = appFileName(name, rev);
  const url = URL.createObjectURL(new Blob([appPage(snap, { title: name })], { type: "text/html" }));
  const a = doc.createElement("a");
  a.href = url; a.download = file; a.rel = "noopener";
  doc.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
  return file;
}
