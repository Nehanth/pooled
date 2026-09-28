// harness/preview-build.js: one self-contained document from a snapshot (data: URLs all the way
// down), CSP and capture script first.
import { buildPreviewDoc, resolveRef, mimeFor, CSP, urlKey, CAPTURE_SOURCE } from "../../harness/preview-build.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const enc = new TextEncoder(), dec = new TextDecoder();

const snap = (files, entry = "index.html") => ({
  entry, files: new Map(Object.entries(files).map(([p, v]) => [p, { type: mimeFor(p), bytes: typeof v === "string" ? enc.encode(v) : v, hash: p }])),
});
// decode a data: URL back to text
const body = (u) => dec.decode(Uint8Array.from(atob(u.slice(u.indexOf(",") + 1)), (c) => c.charCodeAt(0)));
const urls = (s) => [...s.matchAll(/data:[\w/+.-]+(?:;[\w=.-]+)*,[A-Za-z0-9+/=]+/g)].map((m) => m[0]);
const PNG = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13]);
const png = (k) => new Uint8Array([...PNG, k]);   // distinct bytes: identical files share one data URL

Deno.test("resolveRef: relative, rooted, external and escaping refs", () => {
  eq(resolveRef("b.js", "src/a.js"), { path: "src/b.js" });
  eq(resolveRef("./b.js?v=2#x", "src/a.js"), { path: "src/b.js" });
  eq(resolveRef("../img/x.png", "src/a.js"), { path: "img/x.png" });
  eq(resolveRef("/style.css", "src/deep/a.html"), { path: "style.css" });
  eq(resolveRef("my%20file.js", "a.html"), { path: "my file.js" });
  for (const r of ["https://cdn.jsdelivr.net/npm/x", "//cdn/x", "data:,x", "#top", "", "mailto:a@b"]) eq(resolveRef(r, "a.html"), null, r);
  ok(resolveRef("../../etc/passwd", "a.html").outside);
});

Deno.test("HTML: script, stylesheet, img, srcset, inline style and <style> are rewritten", () => {
  const s = snap({
    "index.html": `<!DOCTYPE html><html><head><link rel="stylesheet" href="style.css"><link rel=icon href=icon.png></head>
<body style="background:url('bg.png')"><img src="img/a.png" srcset="img/a.png 1x, img/b.png 2x" alt=x>
<a href="page2.html">next</a><script src="classic.js"></script><style>.x{background:url(bg.png)}</style></body></html>`,
    "style.css": "body{color:red}", "icon.png": png(1), "bg.png": png(2), "img/a.png": png(3), "img/b.png": png(4), "classic.js": "console.log(1)", "page2.html": "<p>2",
  });
  const { html, missing, urlToPath } = buildPreviewDoc(s, { nonce: "n1" });
  eq(missing, []);
  ok(!/(src|href)="(style\.css|icon\.png|img\/a\.png|classic\.js)"/.test(html), "no relative refs left");
  ok(html.includes('href="page2.html"'), "links are left for the nav shim");
  ok(/srcset="data:image\/png;base64,[^ ]+ 1x, data:image\/png;base64,[^ ]+ 2x"/.test(html), "srcset");
  ok(/style="background:url\(&quot;data:image\/png/.test(html) || /style="background:url\(\"data:image\/png/.test(html.replace(/&quot;/g, '"')), "inline style url()");
  ok(/<style>\.x\{background:url\("data:image\/png/.test(html), "<style> block");
  const paths = new Set(Object.values(urlToPath));
  for (const p of ["style.css", "icon.png", "bg.png", "img/a.png", "classic.js"]) ok(paths.has(p), "urlToPath has " + p);
  for (const [u, p] of Object.entries(urlToPath)) if (p === "style.css") eq(body(u), "body{color:red}");
});

Deno.test("modules: a -> b -> c become nested data: URLs, dependencies first", () => {
  const s = snap({
    "index.html": `<script type="module" src="./src/a.js"></script>`,
    "src/a.js": `// import "./not-this.js"\nimport { b } from "./b.js";\nimport * as three from "three";\nimport x from "https://cdn.jsdelivr.net/npm/x/+esm";\nconst s = "import './nope.js'";\nexport const a = () => b() + 1;\nimport("./lazy.js");`,
    "src/b.js": `export { c as cc } from "../lib/c.js";\nimport { c } from '../lib/c.js';\nexport const b = () => c() * 2;`,
    "lib/c.js": "export const c = () => 20;",
    "src/lazy.js": "export default 1;",
  });
  const { html, missing, warnings, urlToPath } = buildPreviewDoc(s);
  eq(missing, []); eq(warnings, []);
  const entry = /<script type="module" src="(data:[^"]+)"/.exec(html)[1];
  eq(urlToPath[entry], "src/a.js");
  const a = body(entry);
  ok(a.includes('// import "./not-this.js"') && a.includes(`"import './nope.js'"`), "comments and strings untouched");
  ok(a.includes('from "three"') && a.includes('from "https://cdn.jsdelivr.net/npm/x/+esm"'), "bare and https specifiers untouched");
  ok(a.endsWith("\n//# sourceURL=src/a.js"), "sourceURL names the file");
  eq(a.split("\n").length, s.files.get("src/a.js").bytes.length && dec.decode(s.files.get("src/a.js").bytes).split("\n").length + 1, "line numbers kept");
  const [bUrl, lazyUrl] = urls(a);
  eq(urlToPath[bUrl], "src/b.js"); eq(urlToPath[lazyUrl], "src/lazy.js");
  const b = body(bUrl), cs = urls(b);
  eq(cs.length, 2); eq(cs[0], cs[1], "one data URL per module");
  eq(urlToPath[cs[0]], "lib/c.js");
  eq(body(cs[0]), "export const c = () => 20;\n//# sourceURL=lib/c.js");
});

Deno.test("modules: regex literals, templates and division do not confuse the scanner", () => {
  const s = snap({
    "index.html": `<script type=module>
const r = /import "x"/g, d = 4 / 2 / 1, t = \`\${"import('./no.js')"} import "./no.js"\`;
import { v } from "./v.js";
export * from "./w.js";
</script>`,
    "v.js": "export const v = 1;", "w.js": "export const w = 2;",
  });
  const { html, missing } = buildPreviewDoc(s);
  eq(missing, []);
  ok(html.includes(`/import "x"/g`) && html.includes(`import "./no.js"\``), "regex and template text kept");
  ok(/import \{ v \} from "data:text\/javascript/.test(html) && /export \* from "data:text\/javascript/.test(html));
});

Deno.test("CSS: url() and @import resolve relative to the CSS file", () => {
  const s = snap({
    "index.html": `<link rel="stylesheet" href="css/main.css">`,
    "css/main.css": `@import "parts/base.css";\n@import url('https://fonts.googleapis.com/css?family=X');\nbody{background:url(../img/bg.png)}`,
    "css/parts/base.css": `h1{background:url("../../img/bg.png")}`,
    "img/bg.png": PNG,
  });
  const { html, missing, urlToPath } = buildPreviewDoc(s);
  eq(missing, []);
  const main = body(/href="(data:text\/css[^"]+)"/.exec(html)[1]);
  ok(main.includes("fonts.googleapis.com"), "external @import untouched");
  const [baseUrl, bg] = urls(main);
  eq(urlToPath[baseUrl], "css/parts/base.css"); eq(urlToPath[bg], "img/bg.png");
  eq(urlToPath[urls(body(baseUrl))[0]], "img/bg.png");
});

Deno.test("missing files are listed and logged; a cycle is a warning", () => {
  const s = snap({
    "index.html": `<img src="sprites/block.png"><script type=module src="a.js"></script>`,
    "a.js": `import "./b.js"; import "./gone.js";`, "b.js": `import "./a.js";`,
  });
  const { html, missing, warnings } = buildPreviewDoc(s);
  eq(missing.sort(), ["gone.js", "sprites/block.png"]);
  eq(warnings.length, 1); ok(warnings[0].includes("a.js -> b.js -> a.js"), warnings[0]);
  ok(html.includes('"missing":["sprites/block.png","gone.js"]') || html.includes('"missing":["gone.js","sprites/block.png"]'), "the frame logs them");
  const miss = buildPreviewDoc(snap({ "a.html": "x" }), { path: "nope.html" });
  eq(miss.missing, ["nope.html"]); ok(miss.html.includes("404"));
});

Deno.test("CSP meta and capture script come before the agent's markup, whatever its shape", () => {
  const shapes = {
    "full": `<!doctype html><html lang="en"><head><script>window.AGENT=1</script></head><body></body></html>`,
    "no head": `<!DOCTYPE html><body><script>window.AGENT=1</script></body>`,
    "fragment": `<canvas id=c></canvas><script>window.AGENT=1</script>`,
    "script first": `<script>window.AGENT=1</script><!doctype html><head></head>`,
  };
  for (const [k, h] of Object.entries(shapes)) {
    const { html } = buildPreviewDoc(snap({ "index.html": h }), { nonce: "abc" });
    const rest = html.replace(/^\s*<!doctype[^>]*>/i, "");
    ok(rest.startsWith(`<meta http-equiv="Content-Security-Policy" content="${CSP}"><script>(function capture(`), k + ": " + rest.slice(0, 80));
    ok(html.indexOf("window.AGENT") > html.indexOf('"nonce":"abc"'), k + ": capture before agent script");
    if (/^<!doctype/i.test(h)) ok(/^<!doctype/i.test(html), k + ": doctype stays first");
  }
  ok(CSP.startsWith("default-src 'none'") && CSP.includes("connect-src data: blob:") && !CSP.includes("'self'"));
});

Deno.test("a page without a viewport meta gets one; one it has is kept, not doubled", () => {
  const VP = /<meta\b[^>]*name=["']?viewport/gi;
  eq(buildPreviewDoc(snap({ "index.html": "<!doctype html><canvas></canvas>" })).html.match(VP).length, 1);
  const own = buildPreviewDoc(snap({ "index.html": `<!doctype html><head><meta name="viewport" content="width=500"></head>` })).html;
  eq(own.match(VP).length, 1); ok(own.includes('content="width=500"'));
  ok(CAPTURE_SOURCE.includes('t: "size"'), "the frame reports its content size for the fit");
});

Deno.test("the capture config cannot close its <script> early", () => {
  const { html } = buildPreviewDoc(snap({ "index.html": "<p>", "x</script><b>.txt": "hi" }));
  eq(html.split("</script>").length, 2, "exactly one </script>, the capture's own");
});

Deno.test("urlToPath keys round-trip through the short keys the frame uses", () => {
  const { urlToPath } = buildPreviewDoc(snap({ "index.html": `<script type=module src=a.js></script>`, "a.js": "1", "b.png": PNG }));
  const keys = new Map(Object.entries(urlToPath).map(([u, p]) => [urlKey(u), p]));
  for (const [u, p] of Object.entries(urlToPath)) eq(keys.get(urlKey(u)), p);
});

Deno.test("importmap entries and new Worker literals are rewritten", () => {
  const s = snap({
    "index.html": `<script type="importmap">{"imports":{"lib":"./lib.js","cdn":"https://cdn.jsdelivr.net/npm/x"}}</script><script type=module>import "lib"; new Worker("./w.js", { type: "module" });</script>`,
    "lib.js": "export default 1", "w.js": "onmessage = () => {}",
  });
  const { html, missing } = buildPreviewDoc(s);
  eq(missing, []);
  ok(/"lib":"data:text\/javascript/.test(html) && html.includes('"cdn":"https://cdn.jsdelivr.net/npm/x"'));
  ok(/new Worker\("data:text\/javascript/.test(html));
});
