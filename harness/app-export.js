// "Share with the room" -> Download (room/code.js): a built app as one .html file a person keeps.
// The file is built on each device from its own copy of the preview (on a peer, the snapshot
// PreviewSubscriber assembled from hash-checked blobs). It is the same file as Download's "One HTML
// file" (harness/export.js appHtml): a wrapper page holding only a sandbox="allow-scripts" frame
// whose srcdoc is the preview document, under the preview's CSP, so the app runs in an opaque
// origin wherever the file is opened. No script of our own and nothing about the room in the file.
//
//   appPage(snap, { title, path }) -> html      snap = { entry, files: Map<path, {type, bytes, hash}> }
//   appFileName(name, rev) -> "tetris-rev3.html"
//   downloadApp(snap, { name, rev }) -> Promise<"shared" | "downloaded" | "cancelled" | "blocked">
import { appHtml, saveBlob, fileBase } from "./export.js";

export const appPage = (snap, { title = "app", path = null } = {}) => appHtml(snap, { title, path }).html;

export function appFileName(name, rev) {
  const base = fileBase(name).replace(/^project$/, "app");
  return `${base}${rev ? "-rev" + (rev >>> 0) : ""}.html`;
}

// phones get the share sheet (Save to Files...) where it takes files, else a download
export async function downloadApp(snap, { name = "app", rev = 0 } = {}, opts = {}) {
  return saveBlob(new Blob([appPage(snap, { title: name })], { type: "text/html" }), appFileName(name, rev), opts);
}
