// serve / preview_logs: the agent's side of the preview server (docs/design/harness-app.md C.2).
// run_js is in run-js.js. Ports are stopped from the UI; the agent has no stop_serve
// (docs/design/harness-light.md A.2). serve answers with what the page did in its first 500 ms,
// so the common "write, serve, see the error" loop needs no separate preview_logs step.
import { DEFAULT_PORT } from "./preview.js";
import { buildPreviewDoc } from "./preview-build.js";
import { normPath, SKIP_DIRS } from "./workspace.js";

export const LOG_LINES = 40, LOG_CHARS = 3000, SERVE_LINES = 8;

const kb = (n) => (n < 1024 ? `${n} B` : `${(n / 1024).toFixed(1)} KB`);
const secs = (ms) => `${(ms / 1000).toFixed(1)}s`;
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
const where = (e) => (e.src ? `${e.src}${e.line ? `:${e.line}${e.col ? `:${e.col}` : ""}` : ""} ` : "");
// one log entry as the model reads it; stacks keep 3 frames
export function logLine(e) {
  const [first, ...rest] = e.text.split("\n");
  const more = rest.map((l) => l.trim()).filter(Boolean).slice(0, 3);
  const text = [first.slice(0, 300), ...more.map((l) => "  " + l.slice(0, 160))].join("\n");
  return `[${secs(e.t)}] ${e.level} ${where(e)}${text}`;
}
// consecutive identical entries fold into one line with ×N; with { all: true } every repeat folds
// into its first copy (a game loop that throws the same two errors each frame reads as two lines)
export function fold(lines, { all = false } = {}) {
  const out = [], at = new Map();
  for (const e of lines) {
    const k = e.level + "\u0000" + e.src + e.line + "\u0000" + e.text + "\u0000" + e.rev, p = all ? at.get(k) : out[out.length - 1];
    if (p && p.k === k) p.n++;
    else { const r = { k, e, n: 1 }; out.push(r); at.set(k, r); }
  }
  return out.map(({ e, n }) => ({ e, n, text: logLine(e) + (n > 1 ? ` ×${n}` : "") }));
}

const PAGE = /\.html?$/i, FILEISH = /\.[a-z0-9]{1,5}$/i;
const listed = (files, n = 8) => files.slice(0, n).join(", ") + (files.length > n ? ` (+${files.length - n} more)` : "");
// serve's arguments as small models write them, turned into what they meant (docs: harness/argfix.js):
//   dir names a file ("index.html", "site/index.html")  -> its folder, that file as the entry
//   entry repeats the dir ("site/index.html" with dir "site") -> the path inside the dir
//   a dir that does not exist, the pages are elsewhere -> the folder that has them
//   no index.html but one other page -> that page
//   a port that cannot be one (0, 80, "abc", 99999) -> the default
// -> { dir, entry, port, notes: [what was changed] } or { error } that says what to call instead.
// `files`: every file in the project (ws.walk()).
export function serveArgs({ dir = "", port, entry = "" } = {}, files = []) {
  const notes = [];
  const clean = (p) => { try { return normPath(String(p ?? "").trim()); } catch { return ""; } };
  files = files.filter((f) => !f.split("/").some((s) => s.startsWith(".") || SKIP_DIRS.has(s)));
  const isFile = (p) => files.includes(p), isDir = (p) => !p || files.some((f) => f.startsWith(p + "/"));
  let d = clean(dir), e = clean(entry);
  let p = port == null || port === "" ? DEFAULT_PORT : Number(port);
  if (!Number.isInteger(p) || p < 1024 || p > 65535) { notes.push(`port ${port} is not usable, so :${DEFAULT_PORT}`); p = DEFAULT_PORT; }
  // dir is a file
  if (d && (isFile(d) || (!isDir(d) && FILEISH.test(d)))) {
    const i = d.lastIndexOf("/"), f = d.slice(i + 1), folder = i < 0 ? "" : d.slice(0, i);
    if (!e || e === "index.html" || e === f || e === d) { e = f; notes.push(`dir ${d} is a file, so its folder with entry ${f}`); }
    else notes.push(`dir ${d} is a file, so its folder`);
    d = folder;
  }
  // the entry written with the dir in front of it
  if (d && e && e.startsWith(d + "/") && !isFile(d + "/" + e)) e = e.slice(d.length + 1);
  const pages = files.filter((f) => PAGE.test(f));
  if (!pages.length) {
    return { error: files.length ? `error: there is no HTML page to serve yet (the project has: ${listed(files)}). Write index.html with write_file first, then call serve with {}`
      : "error: the project is empty. Write index.html with write_file first, then call serve with {}" };
  }
  // a folder that does not exist: serve where the pages are
  if (d && !isDir(d)) {
    const home = e && isFile(e) ? "" : pages.find((f) => f.endsWith("/index.html") || f === "index.html") ?? pages[0];
    const folder = home === "" ? "" : home.includes("/") ? home.slice(0, home.lastIndexOf("/")) : "";
    notes.push(`there is no folder ${d}, so ${folder || "the project root"}`);
    d = folder;
  }
  const under = d ? pages.filter((f) => f.startsWith(d + "/")).map((f) => f.slice(d.length + 1)) : pages;
  if (!e) e = "index.html";
  if (!isFile(d ? d + "/" + e : e)) {
    const pick = under.find((f) => f === "index.html") ?? under.find((f) => f.endsWith("/index.html")) ?? (under.length === 1 ? under[0] : null);
    if (!pick) {
      return { error: `error: no ${e} in ${d || "the project root"}; the pages there are: ${listed(under.length ? under : pages)}. Call serve with {"entry": "${(under[0] ?? pages[0])}"}${d ? ` and {"dir": "${d}"}` : ""}` };
    }
    if (e !== "index.html" || pick !== "index.html") notes.push(`no ${e}, so entry ${pick}`);
    e = pick;
  }
  return { dir: d, entry: e, port: p, notes };
}

export function previewTools(server) {
  const portOf = (p) => (p == null || p === "" ? DEFAULT_PORT : Number(p));
  const nothing = (port) => {
    const others = server.ports().map((s) => ":" + s.port);
    return `error: nothing is served on :${port}${others.length ? ` (served: ${others.join(" ")})` : ""}; call serve first`;
  };
  return [
    {
      name: "serve", mutates: false,
      description: "Serve a folder on a preview port; returns the page's first errors.",
      parameters: { type: "object", properties: { dir: { type: "string" }, port: { type: "integer", description: `default ${DEFAULT_PORT}` }, entry: { type: "string", description: "default index.html" } } },
      async run(args = {}) {
        // (a server without a workspace, e.g. a test's: the arguments as given)
        const a = server.ws?.walk ? serveArgs(args, await server.ws.walk())
          : { dir: args.dir || "", entry: args.entry || "index.html", port: portOf(args.port), notes: [] };
        if (a.error) return a.error;
        const { dir, entry } = a, port = a.port;
        const since = server.cursor(port);
        const snap = await server.serve({ dir, port, entry });
        const { missing } = buildPreviewDoc(snap, {});
        const head = `serving ${snap.dir || "."} on :${port} (${snap.entry}, ${plural(snap.files.size, "file")}, ${kb(snap.bytes)})`;
        const fixed = a.notes.length ? `\n(${a.notes.join("; ")})` : "";
        const miss = fixed + (missing.length ? `\nmissing: ${missing.slice(0, 10).join(", ")}${missing.length > 10 ? ` (+${missing.length - 10} more)` : ""}` : "");
        const idle = await server.whenIdle(port, 2000);
        if (!idle && !server.hasFrame(port)) return `${head} · no preview open, logs appear when it is${miss}`;
        const lines = server.logs(port, since).lines.filter((e) => e.rev === snap.rev);
        const errs = lines.filter((e) => e.level === "error"), warns = lines.filter((e) => e.level === "warn");
        const counts = errs.length || warns.length ? [errs.length && plural(errs.length, "error"), warns.length && plural(warns.length, "warning")].filter(Boolean).join(", ") : "no errors";
        const state = idle ? `loaded in ${Math.round(idle.loadedMs)} ms` : "still loading after 2 s";
        // errors first, in the order they happened (the first one is usually the cause, the rest
        // its noise), then warnings
        const shown = [...fold(errs, { all: true }), ...fold(warns, { all: true })].slice(0, SERVE_LINES);
        const out = [`${head}`, `${state} · ${counts}${shown.length ? ":" : ""}`, ...shown.map((s) => s.text)];
        if (lines.length > shown.reduce((k, s) => k + s.n, 0)) out.push(`more: preview_logs since=${since}`);
        return out.join("\n") + miss;
      },
    },
    {
      name: "preview_logs", mutates: false,
      description: "Console output of a served page since a cursor.",
      parameters: { type: "object", properties: { port: { type: "integer" }, since: { type: "integer" } } },
      async run({ port, since = 0 } = {}) {
        port = portOf(port);
        const snap = server.snapshot(port);
        if (!snap) return nothing(port);
        const { lines, next, dropped } = server.logs(port, Number(since) || 0);
        if (!lines.length) {
          const at = server.loadedAt(port);
          return `no new logs on :${port} (rev ${snap.rev}, ${at ? `loaded ${secs(Date.now() - at)} ago` : server.hasFrame(port) ? "loading" : "no preview open"})\nnext: since=${next}`;
        }
        const rows = [];
        let rev = null;
        for (const r of fold(lines)) {
          if (r.e.rev !== rev) { rev = r.e.rev; rows.push({ e: r.e, text: `(rev ${rev}${rev === snap.rev ? ", current" : ""})` }); }
          rows.push(r);
        }
        // newest last; keep the tail within the caps and summarize what was cut
        let size = 0, keep = 0;
        for (let i = rows.length - 1; i >= 0 && keep < LOG_LINES; i--) {
          if (size + rows[i].text.length + 1 > LOG_CHARS && keep) break;
          size += rows[i].text.length + 1; keep++;
        }
        const cut = rows.length - keep, out = [];
        if (dropped) out.push(`(${dropped} older lines no longer kept)`);
        if (cut) out.push(`(${cut} earlier lines, since=${Number(since) || 0}; newest shown)`);
        // the current page's first error scrolled out of view: keep it in sight, it is the one to fix
        const first = rows.findIndex((r) => r.n && r.e.level === "error" && r.e.rev === snap.rev);
        if (first >= 0 && first < cut) out.push(`first error: ${rows[first].text}`);
        out.push(...rows.slice(cut).map((r) => r.text), `next: since=${next}`);
        return out.join("\n");
      },
    },
  ];
}
