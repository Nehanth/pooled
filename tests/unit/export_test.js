// harness/export.js: Code mode's Download (a .zip of the project, the app as one HTML file).
// The zip is read back here by a small independent reader (central directory -> local headers,
// deflate-raw through DecompressionStream, CRC checked).
import { zip, crc32, projectFiles, snapshotOf, snapshotFiles, hasPage, appHtml, saveBlob, fileBase } from "../../harness/export.js";
import { MemoryWorkspace } from "../../harness/workspace.js";
import { CSP } from "../../harness/preview-build.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const enc = new TextEncoder(), dec = new TextDecoder();

async function unzip(blob) {
  const u8 = new Uint8Array(await blob.arrayBuffer()), dv = new DataView(u8.buffer);
  const e = u8.length - 22;
  eq(dv.getUint32(e, true), 0x06054b50, "end of central directory");
  const n = dv.getUint16(e + 10, true), cdSize = dv.getUint32(e + 12, true), cdOff = dv.getUint32(e + 16, true);
  eq(cdOff + cdSize, e, "central directory ends at the EOCD");
  const out = [];
  let o = cdOff;
  for (let k = 0; k < n; k++) {
    eq(dv.getUint32(o, true), 0x02014b50, "central header");
    const flags = dv.getUint16(o + 8, true), method = dv.getUint16(o + 10, true), crc = dv.getUint32(o + 16, true);
    const csize = dv.getUint32(o + 20, true), usize = dv.getUint32(o + 24, true), nl = dv.getUint16(o + 28, true);
    const xl = dv.getUint16(o + 30, true), cl = dv.getUint16(o + 32, true), lo = dv.getUint32(o + 42, true);
    const name = dec.decode(u8.subarray(o + 46, o + 46 + nl));
    eq(dv.getUint32(lo, true), 0x04034b50, "local header");
    eq(dv.getUint16(lo + 8, true), method, "local method");
    eq(dv.getUint32(lo + 14, true), crc, "local crc");
    eq(dec.decode(u8.subarray(lo + 30, lo + 30 + dv.getUint16(lo + 26, true))), name, "local name");
    const at = lo + 30 + dv.getUint16(lo + 26, true) + dv.getUint16(lo + 28, true), data = u8.subarray(at, at + csize);
    let bytes = data;
    if (method === 8) bytes = new Uint8Array(await new Response(new Blob([data]).stream().pipeThrough(new DecompressionStream("deflate-raw"))).arrayBuffer());
    else eq(method, 0, "stored or deflated");
    eq(bytes.length, usize, name + " size");
    eq(crc32(bytes), crc, name + " crc");
    out.push({ name, bytes, method, utf8: !!(flags & 0x800) });
    o += 46 + nl + xl + cl;
  }
  return out;
}

Deno.test("crc32: the standard check value", () => {
  eq(crc32(enc.encode("123456789")), 0xcbf43926);
  eq(crc32(new Uint8Array()), 0);
});

Deno.test("zip: text, binary, UTF-8 names, deflated where it helps, stored where not", async () => {
  const big = enc.encode("<p>hello</p>\n".repeat(500)), bin = crypto.getRandomValues(new Uint8Array(3000));
  const files = [
    { path: "tetris/index.html", bytes: big },
    { path: "tetris/img/noise.bin", bytes: bin },
    { path: "tetris/héllo wörld.txt", bytes: enc.encode("ü") },
    { path: "tetris/empty.txt", bytes: new Uint8Array() },
  ];
  const blob = await zip(files);
  eq(blob.type, "application/zip");
  const got = await unzip(blob);
  eq(got.map((f) => f.name), files.map((f) => f.path));
  for (let i = 0; i < files.length; i++) eq([...got[i].bytes], [...files[i].bytes], files[i].path);
  ok(got.every((f) => f.utf8), "UTF-8 flag");
  eq(got[0].method, 8, "repetitive text is deflated");
  eq(got[1].method, 0, "random bytes are stored");
  ok(blob.size < big.length, "smaller than the text alone");
  const stored = await unzip(await zip(files, { compress: false }));
  ok(stored.every((f) => f.method === 0));
});

Deno.test("zip: refuses paths that climb out, and repeated paths", async () => {
  let threw = false;
  try { await zip([{ path: "../evil.sh", bytes: enc.encode("x") }]); } catch { threw = true; }
  ok(threw, "..");
  threw = false;
  try { await zip([{ path: "a.txt", bytes: enc.encode("1") }, { path: "./a.txt", bytes: enc.encode("2") }]); } catch { threw = true; }
  ok(threw, "repeat");
});

Deno.test("projectFiles: every file, secrets, dot files and dependency folders left out", async () => {
  const ws = new MemoryWorkspace();
  for (const [p, t] of [["index.html", "<h1>x</h1>"], ["src/app.js", "1"], [".env", "KEY=1"], ["keys/id_rsa", "k"], ["node_modules/x/i.js", "x"], ["a/.git/config", "c"]]) await ws.write(p, t);
  await ws.writeBytes("img/a.png", new Uint8Array([137, 80, 78, 71]));
  const { files, skipped } = await projectFiles(ws);
  eq(files.map((f) => f.path), ["img/a.png", "index.html", "src/app.js"]);
  eq(skipped.sort(), [".env", "a/.git/config", "keys/id_rsa", "node_modules/x/i.js"]);
  eq([...files[0].bytes], [137, 80, 78, 71]);
  let threw = false;
  try { await projectFiles(ws, { maxBytes: 4 }); } catch { threw = true; }
  ok(threw, "over the byte cap");
});

Deno.test("snapshotOf: index.html, else the shallowest page", () => {
  const f = (...ps) => ps.map((p) => ({ path: p, bytes: enc.encode("x") }));
  eq(snapshotOf(f("a/b.html", "index.html")).entry, "index.html");
  eq(snapshotOf(f("deep/x/page.html", "game.html", "s.css")).entry, "game.html");
  ok(!hasPage(snapshotOf(f("a.js"))));
  eq(snapshotOf(f("s.css")).files.get("s.css").type, "text/css");
});

Deno.test("appHtml: the preview document, inlined, inside a sandboxed frame with no script of its own", () => {
  const snap = snapshotOf([
    { path: "index.html", bytes: enc.encode(`<!doctype html><html><head><title>Snake & "Co"</title><link rel="stylesheet" href="s.css"></head><body><script src="app.js"></script></body></html>`) },
    { path: "s.css", bytes: enc.encode("body{background:url(bg.png)}") },
    { path: "app.js", bytes: enc.encode(`console.log("</script><script>alert(1)</script>")`) },
    { path: "bg.png", bytes: new Uint8Array([137, 80, 78, 71]) },
    { path: "two.html", bytes: enc.encode("<p>two</p>") },
  ]);
  const { html, missing } = appHtml(snap);
  eq(missing, []);
  const outer = html.replace(/srcdoc="[^"]*"/, 'srcdoc=""');
  ok(!/<script/i.test(outer), "the outer page runs nothing");
  ok(/<iframe sandbox="allow-scripts" allow="" referrerpolicy="no-referrer"/.test(html), "sandboxed like the preview");
  ok(!/allow-same-origin|allow-top-navigation|allow-popups|allow-downloads|allow-forms/.test(html), "no other sandbox flags");
  ok(outer.includes("<title>Snake &amp; &quot;Co&quot;</title>"), "the app's title, escaped");
  const srcdoc = /srcdoc="([^"]*)"/.exec(html)[1];
  ok(!srcdoc.includes("<") && !srcdoc.includes('"'), "srcdoc is attribute-escaped");
  const inner = srcdoc.replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
  ok(inner.includes(`content="${CSP}"`), "the preview's CSP");
  ok(!/href="s\.css"|src="app\.js"/.test(inner), "files inlined");
  ok(/data:text\/css/.test(inner) && /data:text\/javascript/.test(inner), "as data: URLs");
  // another page of the project
  ok(appHtml(snap, { path: "two.html" }).html.includes("&lt;p&gt;two&lt;/p&gt;"));
  eq(appHtml(snap, { path: "nope.html" }).missing, [], "an unknown path falls back to the entry");
});

Deno.test("snapshotFiles: a member's preview snapshot as zip entries", () => {
  const snap = { files: new Map([["index.html", { bytes: enc.encode("a") }], ["js/x.js", { bytes: enc.encode("b") }]]) };
  eq(snapshotFiles(snap).map((f) => f.path), ["index.html", "js/x.js"]);
});

Deno.test("fileBase", () => {
  eq(fileBase("Tetris Game!"), "tetris-game");
  eq(fileBase("Café"), "cafe");
  eq(fileBase(""), "project");
  eq(fileBase("../../etc"), "etc");
});

Deno.test("saveBlob: share sheet on touch screens, download otherwise, and the fallbacks", async () => {
  const clicks = [];
  const doc = { body: { append() {} }, createElement: () => ({ style: {}, click() { clicks.push(this.download); }, remove() {} }) };
  const blob = new Blob(["x"], { type: "text/plain" });
  let shared = null;
  const nav = { canShare: () => true, share: async (d) => { shared = d; } };
  eq(await saveBlob(blob, "a.zip", { nav, doc, share: false }), "downloaded");
  eq(clicks, ["a.zip"]);
  eq(await saveBlob(blob, "b.zip", { nav, doc, share: true }), "shared");
  eq(shared.files[0].name, "b.zip");
  const fail = (name) => ({ canShare: () => true, share: async () => { throw Object.assign(new Error(name), { name }); } });
  eq(await saveBlob(blob, "c.zip", { nav: fail("AbortError"), doc, share: true }), "cancelled");
  eq(await saveBlob(blob, "d.zip", { nav: fail("NotAllowedError"), doc, share: true }), "blocked");
  eq(await saveBlob(blob, "e.zip", { nav: fail("DataError"), doc, share: true }), "downloaded");
  eq(await saveBlob(blob, "f.zip", { nav: { canShare: () => false, share: nav.share }, doc, share: true }), "downloaded");
  eq(await saveBlob(blob, "g.zip", { nav: {}, doc, share: true }), "downloaded");
  eq(clicks, ["a.zip", "e.zip", "f.zip", "g.zip"]);
});
