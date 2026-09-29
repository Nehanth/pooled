// Code mode's Download button (on the Files heading): the project as a .zip, or the app as one HTML
// file (harness/export.js). What there is to download depends on the screen:
//   the host      - the project's files (secrets and dot files left out); the HTML is the page on
//                   screen in Preview, or the project's index.html when nothing is served
//   a member      - the files of the preview on screen (the host's project stays on the host)
// Phones get the share sheet (Save to Files...) where it takes files, else a download.
//
//   codeExport({ project: () => { name, ws } | null, preview: () => { snap, path, port } | null,
//                name: () => text })
import { zip, projectFiles, snapshotFiles, snapshotOf, hasPage, appHtml, saveBlob, fileBase } from "../harness/export.js";

const $ = (id) => document.getElementById(id);
const size = (n) => (n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`);

export function codeExport({ project, preview, name = () => "" }) {
  const btn = $("code-dl"), menu = $("code-dl-menu"), noteEl = $("code-dl-note");
  if (!btn || !menu) return null;
  let busy = false, ready = null;   // ready: a blob whose share sheet the browser refused, saved on the next tap
  const note = (text, err = false) => { noteEl.textContent = text || ""; noteEl.classList.toggle("err", !!err); };
  const items = () => [...menu.querySelectorAll("button[data-dl]")];

  function open(on) {
    menu.hidden = !on;
    btn.setAttribute("aria-expanded", String(on));
    if (!on) return;
    ready = null;
    const p = project(), v = preview();
    const zipBtn = menu.querySelector('[data-dl="zip"]'), htmlBtn = menu.querySelector('[data-dl="html"]');
    zipBtn.disabled = !p && !v;
    htmlBtn.disabled = !v && !p;
    zipBtn.querySelector("span").textContent = p ? ".zip · every file" : v ? `.zip · the files of :${v.port}` : ".zip";
    htmlBtn.querySelector("span").textContent = v ? `${v.path || v.snap.entry} from :${v.port}, ready to open` : "the app, ready to open";
    note(p || v ? "" : "Nothing to download yet: the files show here once the agent writes some.");
    (items().find((b) => !b.disabled) || btn).focus();
  }
  btn.addEventListener("click", () => open(menu.hidden));
  for (const el of [btn, menu]) el.addEventListener("keydown", (e) => { if (e.key === "Escape" && !menu.hidden) { e.stopPropagation(); open(false); btn.focus(); } });
  // a click elsewhere closes it (not the hidden download link's own click, which is untrusted)
  document.addEventListener("click", (e) => { if (e.isTrusted && !menu.hidden && !menu.contains(e.target) && !btn.contains(e.target)) open(false); });

  async function build(kind) {
    const p = project(), v = preview(), base = fileBase(p?.name || name() || (v ? `app-${v.port}` : "project"));
    if (kind === "zip") {
      if (p) {
        const { files, skipped } = await projectFiles(p.ws);
        if (!files.length) throw new Error("the project has no files yet");
        return { blob: await zip(files.map((f) => ({ ...f, path: base + "/" + f.path }))), name: base + ".zip", n: files.length, skipped };
      }
      const files = snapshotFiles(v.snap);
      return { blob: await zip(files.map((f) => ({ ...f, path: base + "/" + f.path }))), name: base + ".zip", n: files.length, skipped: [] };
    }
    let snap = v?.snap, path = v?.path || null;
    if (!snap) {
      snap = snapshotOf((await projectFiles(p.ws, { maxBytes: 8 << 20 })).files);
      if (!hasPage(snap)) throw new Error("no .html page in the project yet");
    }
    const { html, missing } = appHtml(snap, { path });
    return { blob: new Blob([html], { type: "text/html" }), name: base + ".html", missing };
  }

  async function save(b) {
    const how = await saveBlob(b.blob, b.name);
    if (how === "blocked") {   // the tap is too old for the share sheet: one more tap
      ready = b;
      note(`${b.name} is ready (${size(b.blob.size)}): tap Save`);
      showSave(true);
      return;
    }
    showSave(false);
    if (how === "cancelled") { note(""); return; }
    const extra = [b.skipped?.length ? `${b.skipped.length} hidden or secret file${b.skipped.length > 1 ? "s" : ""} left out` : "",
      b.missing?.length ? `missing: ${b.missing.slice(0, 3).join(", ")}${b.missing.length > 3 ? "…" : ""}` : ""].filter(Boolean).join(" · ");
    note(`${how === "shared" ? "Shared" : "Saved"} ${b.name} · ${size(b.blob.size)}${b.n ? ` · ${b.n} file${b.n > 1 ? "s" : ""}` : ""}${extra ? " · " + extra : ""}`);
  }
  let saveBtn = null;
  function showSave(on) {
    if (on && !saveBtn) {
      saveBtn = document.createElement("button");
      saveBtn.type = "button"; saveBtn.className = "primary"; saveBtn.textContent = "Save";
      saveBtn.onclick = () => { if (ready) { const b = ready; ready = null; save(b); } };
      noteEl.after(saveBtn);
    }
    if (saveBtn) saveBtn.hidden = !on;
  }

  menu.addEventListener("click", async (e) => {
    const b = e.target.closest("button[data-dl]");
    if (!b || b.disabled || busy) return;
    busy = true; showSave(false);
    for (const x of items()) x.disabled = true;
    note(b.dataset.dl === "zip" ? "Packing the files…" : "Building the page…");
    try { await save(await build(b.dataset.dl)); } catch (err) { note("Could not download: " + (err?.message || err), true); }
    busy = false;
    const p = project(), v = preview();
    for (const x of items()) x.disabled = !p && !v;
  });
  return { open };
}
