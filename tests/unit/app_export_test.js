// harness/app-export.js: "Share with the room" -> Download. The file is a wrapper holding only a
// sandboxed frame (allow-scripts) whose srcdoc is the preview document, under the preview's CSP.
import { appPage, appFileName } from "../../harness/app-export.js";
import { buildPreviewDoc, mimeFor, CSP } from "../../harness/preview-build.js";

const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const enc = new TextEncoder();
const snap = (files, entry = "index.html") => ({
  entry, files: new Map(Object.entries(files).map(([p, v]) => [p, { type: mimeFor(p), bytes: enc.encode(v), hash: p }])),
});
const unesc = (t) => t.replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");

Deno.test("the app runs only inside a sandboxed srcdoc frame, under the preview's CSP", () => {
  const s = snap({ "index.html": `<!doctype html><title>x</title><script src="app.js"></script><p onclick="alert(&quot;hi&quot;)">"q" & <b>b</b></p>`, "app.js": "console.log('a</iframe>\"')" });
  const page = appPage(s, { title: 'tetris "</title><script>' });
  const outer = page.replace(/srcdoc="[^"]*"/, 'srcdoc=""');
  ok(!/<script/i.test(outer), "no script outside the frame: " + outer);
  ok((outer.match(/<iframe/g) || []).length === 1, "one frame");
  ok(/<iframe sandbox="allow-scripts" allow=""/.test(outer), "sandbox allow-scripts only");
  ok(!/allow-same-origin|allow-top-navigation|allow-popups|allow-forms/.test(page.slice(0, page.indexOf("srcdoc"))), "no other sandbox flags");
  const csp = /http-equiv="Content-Security-Policy" content="([^"]*)"/.exec(outer)?.[1];
  ok(csp && unesc(csp) === CSP, "the preview's CSP on the wrapper");
  ok(outer.includes("<title>tetris &quot;&lt;/title&gt;&lt;script&gt;</title>"), "title escaped: " + outer);
  const src = /srcdoc="([^"]*)"/.exec(page)[1];
  ok(unesc(src) === buildPreviewDoc(s, { path: "index.html", nonce: "" }).html, "srcdoc is the preview document, escaped losslessly");
});

Deno.test("a path that is not in the snapshot falls back to the entry", () => {
  const s = snap({ "index.html": "<p>home</p>", "b.html": "<p>b</p>" });
  ok(appPage(s, { path: "b.html" }).includes("&lt;p&gt;b&lt;/p&gt;"));
  ok(appPage(s, { path: "../nope.html" }).includes("&lt;p&gt;home&lt;/p&gt;"));
});

Deno.test("file names are plain", () => {
  ok(appFileName("Tetris Game", 3) === "tetris-game-rev3.html");
  ok(appFileName("../../etc/passwd", 1) === "etc-passwd-rev1.html");
  ok(appFileName("", 0) === "app.html");
  ok(appFileName("日本", 2) === "app-rev2.html");
});
