// harness/templates.js: the starter templates for a new Code project. Each one must run in the
// preview as written (every file it references there, its script parses, the elements the script
// looks up exist) and stay small enough for the agent to edit in one go.
import { TEMPLATES, templateById, applyTemplate } from "../../harness/templates.js";
import { buildPreviewDoc } from "../../harness/preview-build.js";
import { PreviewServer } from "../../harness/preview.js";
import { MemoryWorkspace, riskyPath, secretPath } from "../../harness/workspace.js";
import { snapshotOf } from "../../harness/export.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const enc = new TextEncoder();
const snapOf = (t) => snapshotOf(Object.entries(t.files).map(([path, text]) => ({ path, bytes: enc.encode(text) })));

Deno.test("templates: 3 to 4 of them, unique ids, a label, a blurb and a first change to ask for", () => {
  ok(TEMPLATES.length >= 3 && TEMPLATES.length <= 4, "3-4 templates");
  eq(new Set(TEMPLATES.map((t) => t.id)).size, TEMPLATES.length, "unique ids");
  for (const t of TEMPLATES) {
    ok(/^[a-z]+$/.test(t.id) && t.id !== "blank", t.id);
    ok(t.label && t.blurb && t.next, t.id + ": label, blurb, next");
    ok(t.next.length < 120, t.id + ": the suggested request is short");
  }
});

Deno.test("templates: plain, small files the agent may edit", () => {
  for (const t of TEMPLATES) {
    ok(t.files["index.html"], t.id + ": index.html");
    for (const [p, text] of Object.entries(t.files)) {
      ok(!riskyPath(p) && !secretPath(p), `${t.id}/${p}: not a risky or secret path`);
      ok(text.split("\n").length <= 100, `${t.id}/${p}: at most 100 lines (${text.split("\n").length})`);
      ok(!/https?:\/\//.test(text), `${t.id}/${p}: no network`);
      ok(!/<form[\s>]|["']submit["']/.test(text), `${t.id}/${p}: no form submit (the preview's sandbox blocks it)`);
    }
  }
});

Deno.test("templates: each builds into a preview with nothing missing", () => {
  for (const t of TEMPLATES) {
    const snap = snapOf(t);
    eq(snap.entry, "index.html", t.id);
    const { html, missing, warnings } = buildPreviewDoc(snap, { path: "index.html", nonce: "" });
    eq(missing, [], t.id + ": missing");
    eq(warnings, [], t.id + ": warnings");
    ok(!/src="app\.js"|href="style\.css"/.test(html), t.id + ": files inlined");
    // every file is used by the page
    const page = t.files["index.html"];
    for (const p of Object.keys(t.files)) if (p !== "index.html") ok(page.includes(`"${p}"`), `${t.id}: index.html references ${p}`);
  }
});

Deno.test("templates: the script parses and every id it looks up is in the page", () => {
  for (const t of TEMPLATES) {
    const js = t.files["app.js"], page = t.files["index.html"];
    ok(!/^\s*(import|export)\s/m.test(js), t.id + ": one module, no imports");
    try { new Function(js); } catch (e) { throw new Error(`${t.id}/app.js does not parse: ${e.message}`); }
    const ids = [...js.matchAll(/getElementById\("([^"]+)"\)|querySelector\("#([\w-]+)/g)].map((m) => m[1] || m[2]);
    ok(ids.length, t.id + ": looks up elements");
    for (const id of ids) ok(page.includes(`id="${id}"`), `${t.id}: #${id} in index.html`);
  }
});

Deno.test("templateById: blank and unknown ids are the empty project", () => {
  eq(templateById("game")?.id, "game");
  for (const id of ["", "blank", "nope", undefined, null, "__proto__", "constructor"]) eq(templateById(id), null, String(id));
});

Deno.test("applyTemplate: writes the files, and nothing for an empty project", async () => {
  const ws = new MemoryWorkspace();
  eq(await applyTemplate(ws, ""), []);
  eq(await applyTemplate(ws, "toString"), []);
  eq(await ws.walk(), []);
  const t = templateById("form");
  eq(await applyTemplate(ws, "form"), Object.keys(t.files));
  eq(await ws.walk(), Object.keys(t.files).sort());
  eq(await ws.read("app.js"), t.files["app.js"]);
});

Deno.test("applyTemplate, then serve: the project runs in Preview straight away", async () => {
  for (const t of TEMPLATES) {
    const ws = new MemoryWorkspace();
    await applyTemplate(ws, t.id);
    const server = new PreviewServer(ws);
    const snap = await server.serve({});
    eq(snap.entry, "index.html", t.id);
    eq([...snap.files.keys()].sort(), Object.keys(t.files).sort(), t.id);
    server.close();
  }
});
