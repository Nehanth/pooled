// harness/preview.js + preview-tools.js: virtual ports over a MemoryWorkspace, live reload,
// the console ring, and the serve / preview_logs results, with a fake frame.
import { MemoryWorkspace, watch } from "../../harness/workspace.js";
import { PreviewServer } from "../../harness/preview.js";
import { previewTools, LOG_LINES } from "../../harness/preview-tools.js";
import { codingTools } from "../../harness/codetools.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const rejects = async (p, re) => { try { await p; } catch (e) { ok(re.test(e.message), e.message); return; } throw new Error("did not throw"); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// wait for a condition instead of a fixed sleep: a loaded CI runner can take longer than any guess
const until = async (cond, what, ms = 5000) => { for (const end = Date.now() + ms; !cond();) { if (Date.now() > end) throw new Error("timed out waiting for " + what); await sleep(2); } };

const app = () => watch(new MemoryWorkspace({
  "index.html": "<canvas></canvas><script type=module src=game.js></script>",
  "game.js": "console.log(1)\n",
  "style.css": "body{}",
  ".env": "SECRET=1",
  "node_modules/x/i.js": "x",
  "notes/plan.md": "# plan",
}));
const server = (ws, o = {}) => new PreviewServer(ws, { debounce: 10, frameWait: 30, ...o });
const nextUpdate = (s) => new Promise((r) => { const off = s.onUpdate((u) => { off(); r(u); }); });
// a frame that attaches on each rev, logs what `script` returns, then reports ready and idle
function fakeFrame(s, port, script = () => []) {
  let detach = () => {};
  return s.onUpdate((u) => {
    if (u.port !== port) return;
    detach();
    if (u.stopped) return;
    detach = s.attach(port);
    setTimeout(() => {
      for (const l of script(s.snapshot(port))) s.pushLog(port, { rev: u.rev, src: "", line: 0, col: 0, ms: 100, ...l });
      s.frameEvent(port, { t: "ready", rev: u.rev, ms: 12 });
      s.frameEvent(port, { t: "idle", rev: u.rev, ms: 512 });
    }, 1);
  });
}
const T = (s, ws) => Object.fromEntries([...previewTools(s), ...(ws ? codingTools(ws, { server: s }) : [])].map((t) => [t.name, t]));

Deno.test("serve snapshots the folder: skips dotfiles and SKIP_DIRS, hashes, ports", async () => {
  const s = server(app());
  const snap = await s.serve({});
  eq([...snap.files.keys()], ["game.js", "index.html", "notes/plan.md", "style.css"]);
  eq([snap.port, snap.dir, snap.entry, snap.rev], [5173, "", "index.html", 1]);
  eq(snap.files.get("game.js").type, "text/javascript");
  eq(snap.files.get("game.js").hash.length, 20);
  eq(snap.bytes, [...snap.files.values()].reduce((n, f) => n + f.bytes.length, 0));
  ok(Object.isFrozen(snap));
  const sub = await s.serve({ dir: "notes", port: 8080, entry: "plan.md" });
  eq([...sub.files.keys()], ["plan.md"]);
  eq(s.ports(), [{ port: 5173, dir: "", entry: "index.html", rev: 1 }, { port: 8080, dir: "notes", entry: "plan.md", rev: 1 }]);
  eq(s.servedPorts("notes/plan.md"), [5173, 8080]); eq(s.servedPorts("game.js"), [5173]);
  eq((await s.serve({})).rev, 2, "serving a port again replaces it with a new rev");
  ok(s.stop(8080)); ok(!s.stop(8080)); eq(s.snapshot(8080), null);
  s.close();
});

Deno.test("serve fails with messages the model can act on", async () => {
  const s = server(app());
  await rejects(s.serve({ port: 80 }), /port must be an integer from 1024 to 65535/);
  await rejects(s.serve({ entry: "main.html" }), /no main\.html in the project root; write it first/);
  await rejects(s.serve({ dir: "nope" }), /no folder nope/);
  await rejects(server(app(), { maxFiles: 3 }).serve({}), /too many files under the project root \(4, max 3\)/);
  await rejects(server(app(), { maxFile: 10 }).serve({}), /game\.js is 15 B \(max 10 B per file\)/);
  await rejects(server(app(), { maxBytes: 70 }).serve({}), /over 70 B; serve a smaller folder/);
  s.close();
});

Deno.test("live reload: a write under the dir bumps rev after the debounce; outside or same content does not", async () => {
  const ws = app(), s = server(ws, { debounce: 100 });   // wide enough that three quick writes always land in one update
  await s.serve({ dir: "", port: 5173 });
  await s.serve({ dir: "notes", port: 5174, entry: "plan.md" });
  const seen = [];
  s.onUpdate((u) => seen.push(u));
  await ws.write("game.js", "console.log(2)\n");
  await ws.write("game.js", "console.log(3)\n");   // debounced into one update
  await ws.write("new.js", "export {}");
  await until(() => seen.length, "the first update");
  eq(seen, [{ port: 5173, rev: 2, changed: ["game.js", "new.js"] }]);
  eq(new TextDecoder().decode(s.snapshot(5173).files.get("game.js").bytes), "console.log(3)\n");
  await ws.write("game.js", "console.log(3)\n");   // same bytes
  eq(await s.flush(5173), { rev: 2 }, "re-snapshotting now finds nothing new");
  eq(seen.length, 1, "no rev for an unchanged file");
  await ws.remove("notes");
  await until(() => seen.length >= 3, "an update on each port");
  seen.sort((a, b) => a.port - b.port);
  eq(seen.slice(1).map((u) => [u.port, u.rev, u.changed]), [[5173, 3, ["notes/plan.md"]], [5174, 2, ["plan.md"]]]);
  eq(s.logs(5174).lines.map((e) => e.text), ["plan.md was removed; the preview shows a 404 page"]);
  s.close();
});

Deno.test("refresh re-reads files changed behind the watcher's back", async () => {
  const mem = new MemoryWorkspace({ "index.html": "a" }), s = server(watch(mem));
  await s.serve({});
  await mem.write("index.html", "b");   // not through the watched view
  const u = nextUpdate(s);
  const snap = await s.refresh(5173);
  eq(snap.rev, 2); eq((await u).changed, ["index.html"]);
  s.close();
});

Deno.test("logs: cursor, ring of 500, dropped count", async () => {
  const s = server(app());
  await s.serve({});
  for (let i = 1; i <= 3; i++) s.pushLog(5173, { level: "log", text: "m" + i, ms: i });
  const a = s.logs(5173, 0);
  eq(a.lines.map((e) => [e.seq, e.text, e.rev]), [[1, "m1", 1], [2, "m2", 1], [3, "m3", 1]]);
  eq(a.next, 3); eq(s.logs(5173, 2).lines.map((e) => e.text), ["m3"]); eq(s.logs(5173, 3).lines, []);
  for (let i = 0; i < 600; i++) s.pushLog(5173, { level: "log", text: "x" });
  const b = s.logs(5173, 0);
  eq(b.lines.length, 500); eq(b.lines[0].seq, 104); eq(b.dropped, 103); eq(b.next, 603);
  s.stop(5173); await s.serve({});
  s.pushLog(5173, { level: "log", text: "after" });
  eq(s.logs(5173, 603).lines.map((e) => [e.seq, e.text]), [[604, "after"]], "a cursor survives stop + serve");
  eq(s.logs(5173, 9999).lines.length, 1, "a cursor from the future starts over");
  s.close();
});

Deno.test("whenIdle: resolved by the frame's idle for the current rev; null with no frame", async () => {
  const s = server(app());
  await s.serve({});
  eq(await s.whenIdle(5173, 200), null, "no frame attached within frameWait");
  const off = fakeFrame(s, 5173);
  await s.serve({});
  eq(await s.whenIdle(5173, 500), { loadedMs: 12 });
  s.frameEvent(5173, { t: "idle", rev: 1, ms: 1 });   // an old rev's frame is ignored
  eq(await s.whenIdle(5173, 10), { loadedMs: 12 });
  off(); s.close();
});

Deno.test("serve tool: first errors from the frame, missing files, no-frame message", async () => {
  const ws = app();
  await ws.write("index.html", `<img src="sprites/block.png"><script type=module src=game.js></script>`);
  const s = server(ws), t = T(s);
  eq(await t.serve.run({}), "serving . on :5173 (index.html, 4 files, 97 B) · no preview open, logs appear when it is\nmissing: sprites/block.png");
  const off = fakeFrame(s, 5173, (snap) => [
    { level: "log", text: "boot", ms: 5 },
    { level: "error", text: "ReferenceError: ctx is not defined\n    at draw (game.js:41:5)\n    at game.js:50:1", src: "game.js", line: 41, col: 5, ms: 100 },
    { level: "warn", text: "slow", ms: 200 },
  ]);
  const r = await t.serve.run({ port: 5173 });
  eq(r, "serving . on :5173 (index.html, 4 files, 97 B)\nloaded in 12 ms · 1 error, 1 warning:\n"
    + "[0.1s] error game.js:41:5 ReferenceError: ctx is not defined\n  at draw (game.js:41:5)\n  at game.js:50:1\n[0.2s] warn slow\n"
    + "more: preview_logs since=0\nmissing: sprites/block.png");
  off();
  const clean = fakeFrame(s, 5173);
  const r2 = await t.serve.run({});
  ok(r2.endsWith("loaded in 12 ms · no errors\nmissing: sprites/block.png"), r2);
  clean(); s.close();
});

Deno.test("preview_logs: newest last, folding, caps, revs, cursor; a stopped port", async () => {
  const s = server(app()), t = T(s);
  eq(await t.preview_logs.run({}), "error: nothing is served on :5173; call serve first");
  await s.serve({});
  eq(await t.preview_logs.run({}), "no new logs on :5173 (rev 1, no preview open)\nnext: since=0");
  for (let i = 0; i < 12; i++) s.pushLog(5173, { level: "error", text: "tick failed", src: "game.js", line: 9, col: 2, ms: 1000 });
  s.pushLog(5173, { level: "log", text: "done", ms: 1500 });
  eq(await t.preview_logs.run({ since: 0 }), "(rev 1, current)\n[1.0s] error game.js:9:2 tick failed ×12\n[1.5s] log done\nnext: since=13");
  eq(await t.preview_logs.run({ since: 13 }), "no new logs on :5173 (rev 1, no preview open)\nnext: since=13");
  for (let i = 0; i < 100; i++) s.pushLog(5173, { level: "log", text: "line " + i, ms: 2000 });
  const many = await t.preview_logs.run({ since: 13 });
  const rows = many.split("\n");
  eq(rows[0], `(${100 - (LOG_LINES - 1)} earlier lines, since=13; newest shown)`);
  eq(rows.length, LOG_LINES + 2); ok(rows.at(-2).endsWith("line 99") && rows.at(-1) === "next: since=113", many);
  s.pushLog(5173, { level: "log", text: "y".repeat(2000), ms: 1 });
  s.pushLog(5173, { level: "log", text: "z".repeat(2000), ms: 1 });
  ok((await t.preview_logs.run({ since: 113 })).length < 3000 + 200, "char cap");
  ok(s.stop(5173));
  eq(await t.preview_logs.run({}), "error: nothing is served on :5173; call serve first");
  s.close();
});

Deno.test("codingTools with a server: writes under a served dir say the preview reloads", async () => {
  const ws = app(), s = server(ws), t = T(s, ws);
  eq(await t.write_file.run({ path: "a.js", content: "x" }), "wrote a.js (1 lines, 1 B)");
  await s.serve({ dir: "notes", port: 5174, entry: "plan.md" });
  eq(await t.write_file.run({ path: "notes/b.md", content: "x\n" }), "wrote notes/b.md (1 lines, 2 B) · preview :5174 reloaded (rev 2)");
  eq(s.snapshot(5174).rev, 2, "the snapshot is taken before the tool answers");
  eq(await t.write_file.run({ path: "a.js", content: "y" }), "wrote a.js (1 lines, 1 B)");
  await sleep(30);
  s.close();
});

Deno.test("codingTools with a server: a snapshot that fails is reported, not 'reloaded'", async () => {
  const ws = app(), s = server(ws, { maxFile: 64 }), t = T(s, ws);
  await s.serve({ dir: "notes", port: 5174, entry: "plan.md" });
  const r = await t.write_file.run({ path: "notes/big.md", content: "x".repeat(100) });
  ok(/preview :5174 not updated: big\.md is 100 B/.test(r), r);
  s.close();
});

Deno.test("serve lists errors before warnings, repeats folded; preview_logs keeps the first error in sight", async () => {
  const s = server(app()), t = T(s);
  const off = fakeFrame(s, 5173, () => [
    { level: "warn", text: "AudioContext was not allowed to start", ms: 50 },
    { level: "error", text: "ReferenceError: grid is not defined", src: "game.js", line: 12, col: 3, ms: 100 },
    { level: "error", text: "TypeError: cannot read 'x'", src: "game.js", line: 80, col: 1, ms: 116 },
    { level: "error", text: "ReferenceError: grid is not defined", src: "game.js", line: 12, col: 3, ms: 132 },
  ]);
  const r = await t.serve.run({});
  const rows = r.split("\n");
  eq(rows[1], "loaded in 12 ms · 3 errors, 1 warning:");
  ok(/error game\.js:12:3 ReferenceError: grid is not defined ×2$/.test(rows[2]), r);
  ok(/error game\.js:80:1 TypeError/.test(rows[3]) && /warn AudioContext/.test(rows[4]), r);
  off();
  // the first error, then a flood of logs that pushes it out of preview_logs' window
  const since = s.cursor(5173);
  s.pushLog(5173, { level: "error", text: "SyntaxError: missing ) after argument list", src: "game.js", line: 7, col: 20, ms: 10, rev: s.snapshot(5173).rev });
  for (let i = 0; i < 80; i++) s.pushLog(5173, { level: "log", text: "frame " + i, ms: 20, rev: s.snapshot(5173).rev });
  const logs = await t.preview_logs.run({ since });
  ok(/\nfirst error: \[0\.0s\] error game\.js:7:20 SyntaxError: missing \) after argument list\n/.test(logs), logs);
  s.close();
});
