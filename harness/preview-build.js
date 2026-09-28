// Builds one self-contained HTML document from a preview snapshot, for a sandboxed iframe (a blob:
// URL locally, srcdoc on the relay)
// (docs/design/harness-app.md B.1, B.3). The frame has an opaque origin and no service worker, so
// nothing can be fetched by relative URL: every relative reference (scripts, ES module imports,
// stylesheets, CSS url()/@import, images, media, fonts) becomes a data: URL, modules bottom-up
// over the import graph so each module's imports are already data: URLs when it is encoded.
// Pure string work: no DOM, runs in Deno for the unit tests.
//
//   buildPreviewDoc(snapshot, { path, nonce }) -> { html, missing: [path], warnings: [text], urlToPath }
//   snapshot = { entry, files: Map<path, { type, bytes: Uint8Array, hash }> }   (paths relative to the served dir)
//
// The document starts with a CSP meta (network off except the two script CDNs and Google Fonts)
// and the capture
// script (console, errors and shims, below), ahead of anything the agent wrote.
// Limits (B.5): computed import specifiers, new URL(x, import.meta.url) and location changes are
// not rewritten; they fail with a console error the agent can read.

export const CSP = "default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' data: blob: https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
  + "style-src 'unsafe-inline' data: https://fonts.googleapis.com; font-src data: https://fonts.gstatic.com; img-src data: blob:; "
  + "media-src data: blob:; connect-src data: blob:; worker-src blob: data:";

const MIME = {
  html: "text/html", htm: "text/html", js: "text/javascript", mjs: "text/javascript", css: "text/css", json: "application/json",
  svg: "image/svg+xml", png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", gif: "image/gif", webp: "image/webp", ico: "image/x-icon",
  wav: "audio/wav", mp3: "audio/mpeg", ogg: "audio/ogg", mp4: "video/mp4", webm: "video/webm",
  woff: "font/woff", woff2: "font/woff2", ttf: "font/ttf", otf: "font/otf", txt: "text/plain", md: "text/plain", csv: "text/csv", glsl: "text/plain",
};
export const mimeFor = (p) => MIME[/\.([^./]+)$/.exec(p)?.[1]?.toLowerCase()] || "application/octet-stream";

const dec = new TextDecoder(), enc = new TextEncoder();
export function b64(u8) {
  let s = "";
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
}
const textType = (t) => /^text\/|json|svg|javascript/.test(t);
export const dataUrl = (type, u8) => `data:${type}${textType(type) ? ";charset=utf-8" : ""};base64,${b64(u8)}`;
// The frame gets data URLs as error locations; it maps them back to file names by this short key
// (full data URLs would double the document).
export const urlKey = (u) => u.length + ":" + u.slice(-48);

const SCHEME = /^[a-zA-Z][a-zA-Z\d+.-]*:/;
const dirOf = (p) => (p.includes("/") ? p.slice(0, p.lastIndexOf("/") + 1) : "");

// A reference from file `from` to a file of the snapshot: { path } (may not exist), or null for
// things left alone (other schemes, //host, #fragment, empty). A path that climbs out of the
// served folder resolves to its clamped name, which does not exist, so it is reported missing.
export function resolveRef(ref, from) {
  ref = String(ref).trim();
  if (!ref || ref[0] === "#" || ref.startsWith("//") || SCHEME.test(ref)) return null;
  ref = ref.replace(/[?#].*$/, "");
  if (!ref) return null;
  try { ref = decodeURI(ref); } catch {}
  const parts = ref.startsWith("/") ? [] : dirOf(from).split("/").filter(Boolean);
  for (const s of ref.split("/")) {
    if (!s || s === ".") continue;
    if (s === "..") { if (!parts.length) return { path: "../" + ref.replace(/^(\.\.\/)+/, ""), outside: true }; parts.pop(); }
    else parts.push(s);
  }
  return { path: parts.join("/") };
}
const isRelSpec = (s) => /^(\.{1,2}\/|\/(?!\/))/.test(s);   // JS: only ./ ../ / are files; bare names go to the CDN or fail

// ---------------------------------------------------------------- JS: find import specifiers
// Masks comments, strings, templates and regex literals (so `import` inside them is not seen),
// records string literals by start offset, then matches import/export/dynamic import/new Worker
// on the masked text and returns the literal that follows each match.
const REGEX_AFTER_WORD = new Set(["return", "typeof", "instanceof", "in", "of", "new", "delete", "void", "throw", "case", "do", "else", "yield", "await"]);
function scanJs(src) {
  const n = src.length, m = src.split(""), lits = new Map();
  let i = 0, prev = "";   // prev: last significant token class ("a" operand, or a punctuator)
  const blank = (a, b) => { for (let k = a; k < b; k++) if (m[k] !== "\n") m[k] = " "; };
  const fill = (a, b) => { for (let k = a; k < b; k++) if (m[k] !== "\n") m[k] = "_"; };
  const skipString = (j) => {   // j at the quote; returns index after the closing quote
    const q = src[j++];
    while (j < n && src[j] !== q && src[j] !== "\n") j += src[j] === "\\" ? 2 : 1;
    return j + 1;
  };
  const skipTemplate = (j) => {   // j at the backtick; returns index after the closing one
    j++;
    while (j < n && src[j] !== "`") {
      if (src[j] === "\\") { j += 2; continue; }
      if (src[j] === "$" && src[j + 1] === "{") {
        let depth = 1; j += 2;
        while (j < n && depth) {
          const c = src[j];
          if (c === "'" || c === '"') j = skipString(j);
          else if (c === "`") j = skipTemplate(j);
          else if (c === "/" && src[j + 1] === "/") { while (j < n && src[j] !== "\n") j++; }
          else if (c === "/" && src[j + 1] === "*") { const e = src.indexOf("*/", j + 2); j = e < 0 ? n : e + 2; }
          else { if (c === "{") depth++; else if (c === "}") depth--; j++; }
        }
        continue;
      }
      j++;
    }
    return j + 1;
  };
  while (i < n) {
    const c = src[i], d = src[i + 1];
    if (c === "/" && d === "/") { const e = src.indexOf("\n", i); const j = e < 0 ? n : e; blank(i, j); i = j; continue; }
    if (c === "/" && d === "*") { const e = src.indexOf("*/", i + 2); const j = e < 0 ? n : e + 2; blank(i, j); i = j; continue; }
    if (c === '"' || c === "'" || c === "`") {
      const j = Math.min(n, c === "`" ? skipTemplate(i) : skipString(i));
      const raw = src.slice(i + 1, j - 1);
      if (!raw.includes("\\") && !(c === "`" && raw.includes("${"))) lits.set(i, { end: j, value: raw });
      fill(i + 1, j - 1); i = j; prev = "a"; continue;
    }
    if (c === "/" && (prev === "" || "(,=:[!&|?{};+-*%<>~^".includes(prev))) {   // a regex literal
      let j = i + 1, cls = false;
      while (j < n && src[j] !== "\n") {
        if (src[j] === "\\") { j += 2; continue; }
        if (src[j] === "[") cls = true; else if (src[j] === "]") cls = false; else if (src[j] === "/" && !cls) break;
        j++;
      }
      j++;
      while (j < n && /[a-z]/i.test(src[j])) j++;
      fill(i + 1, j - 1); i = j; prev = "a"; continue;
    }
    if (/\s/.test(c)) { i++; continue; }
    if (/[\w$]/.test(c)) {
      let j = i + 1;
      while (j < n && /[\w$]/.test(src[j])) j++;
      prev = REGEX_AFTER_WORD.has(src.slice(i, j)) ? "(" : "a"; i = j; continue;
    }
    prev = c; i++;
  }
  const masked = m.join(""), out = [];
  const RE = /(?<![.\w$])(?:import\s*(?:[\w$*{}\s,]+?\s*from\s*)?|export\s*(?:\*(?:\s*as\s+[\w$]+)?|\{[^}]*\})\s*from\s*|import\s*\(\s*|new\s+(?:Shared)?Worker\s*\(\s*)(?=["'`])/g;
  for (const mt of masked.matchAll(RE)) {
    const at = mt.index + mt[0].length, lit = lits.get(at);
    if (lit) out.push({ start: at, end: lit.end, value: lit.value, dynamic: /\($/.test(mt[0].trim()) });
  }
  return out;
}

// ---------------------------------------------------------------- the capture script
// Runs first in the frame. Serialized with toString(), so it must not close over anything.
function capture(C) {
  const P = window.parent, send = (d) => { try { P.postMessage(Object.assign({ pv: C.nonce }, d), "*"); } catch {} };
  const key = (u) => u.length + ":" + u.slice(-48);
  // the document is a blob: URL (or, in the isolated relay, a srcdoc under the relay page's URL):
  // name that URL as the page's path, and a ref left relative (the relay's folder, which the CSP
  // refuses) as the project path it was meant to be
  const SELF = location.href.replace(/#.*$/, "");
  const B = /^https?:/.test(document.baseURI) ? document.baseURI.replace(/[?#].*$/, "").replace(/[^/]*$/, "") : "\u0000";
  const OR = B.replace(/^(\w+:\/\/[^/]+).*$/, "$1/");
  const name = (u) => (!u ? "" : u.startsWith("data:") ? C.keys[key(u)] || "(inline)" : u === "about:srcdoc" || u === SELF ? C.path
    : u.startsWith(B) ? u.slice(B.length) : u.startsWith(OR) ? u.slice(OR.length) : u);
  const clean = (s) => String(s).replace(/data:[\w/+.-]+(?:;[\w=.-]+)*,[A-Za-z0-9+/=]+/g, name).replace(/about:srcdoc/g, C.path).split(SELF).join(C.path).split(B).join("");
  // WebRTC's STUN traffic is outside the CSP's connect-src: no peer connections from the app
  for (const k of ["RTCPeerConnection", "webkitRTCPeerConnection", "RTCDataChannel"]) { try { Object.defineProperty(window, k, { value: undefined, configurable: false }); } catch {} }
  const O = {};
  for (const l of ["log", "info", "warn", "error", "debug"]) O[l] = console[l].bind(console);
  let win = 0, n = 0, dropped = 0;
  const post = (level, text, src = "", line = 0, col = 0) => {
    const now = performance.now();
    if (now - win > 1000) {
      win = now; n = 0;
      if (dropped) { const k = dropped; dropped = 0; post("warn", `(${k} messages dropped)`); }
    }
    if (++n > 200) { dropped++; return; }
    text = clean(text);
    send({ t: "log", level, text: text.length > 1000 ? text.slice(0, 1000) + "…" : text, src: name(src), line, col, ms: Math.round(now) });
  };
  const fmt = (a) => a.map((x) => {
    if (typeof x === "string") return x;
    if (x instanceof Error) return x.stack && x.stack.includes(x.message) ? x.stack : `${x.name}: ${x.message}`;
    try { const j = JSON.stringify(x); if (j !== undefined) return j.length > 500 ? j.slice(0, 500) + "…" : j; } catch {}
    return String(x);
  }).join(" ");
  const where = () => {   // the caller of console.x: the third stack line (Error, this wrapper, caller)
    const l = (new Error().stack || "").split("\n")[3] || "", m = /\(?([^\s()]+?):(\d+):(\d+)\)?\s*$/.exec(l);
    return m ? [m[1], +m[2], +m[3]] : [];
  };
  for (const l of Object.keys(O)) console[l] = (...a) => { post(l === "debug" ? "log" : l, fmt(a), ...where()); O[l](...a); };
  addEventListener("error", (e) => {
    const t = e.target;
    if (t && t !== window && t.tagName) {   // a resource that failed to load (capture phase only)
      const n = name(t.src || t.href || "");
      if (!C.missing.includes(n)) post("error", `failed to load <${t.tagName.toLowerCase()}> ${n}`);   // missing ones are already a 404 line
      return;
    }
    const frames = e.error && e.error.stack ? e.error.stack.split("\n").filter((l) => /^\s+at /.test(l)) : [];
    const msg = (e.message || "error").replace(/^Uncaught /, "") + (frames.length > 1 ? "\n" + frames.slice(0, 4).join("\n") : "");   // one frame only repeats src:line
    post("error", msg, e.filename, e.lineno, e.colno);
  }, true);
  addEventListener("unhandledrejection", (e) => {
    const r = e.reason;
    post("error", "Unhandled promise rejection: " + (r instanceof Error ? r.stack || r.message : fmt([r])));
  });
  for (const p of C.missing) post("error", `404 ${p} (referenced but not in the project)`);
  for (const w of C.warnings) post("warn", w);

  // shims: storage throws in an opaque origin, modals are blocked in the sandbox
  const mem = () => {
    const m = new Map();
    return {
      getItem: (k) => (m.has(String(k)) ? m.get(String(k)) : null), setItem: (k, v) => { m.set(String(k), String(v)); },
      removeItem: (k) => { m.delete(String(k)); }, clear: () => m.clear(), key: (i) => [...m.keys()][i] ?? null, get length() { return m.size; },
    };
  };
  for (const k of ["localStorage", "sessionStorage"]) { try { Object.defineProperty(window, k, { value: mem(), configurable: true }); } catch {} }
  window.alert = (m) => { post("info", "alert: " + m); };
  window.confirm = (m) => { post("info", "confirm (answered false): " + m); return false; };
  window.prompt = (m) => { post("info", "prompt (answered null): " + m); return null; };

  // relative URLs used from code: resolve against the page's folder and serve from the table
  const base = C.path.includes("/") ? C.path.slice(0, C.path.lastIndexOf("/") + 1) : "";
  const local = (u) => {
    u = String(u == null ? "" : u).trim();
    if (!u || u[0] === "#" || u.startsWith("//") || /^[a-zA-Z][\w+.-]*:/.test(u)) return null;
    u = u.replace(/[?#].*$/, "");
    const parts = u[0] === "/" ? [] : base.split("/").filter(Boolean);
    for (const s of u.split("/")) { if (s === "..") parts.pop(); else if (s && s !== ".") parts.push(s); }
    try { return decodeURI(parts.join("/")); } catch { return parts.join("/"); }
  };
  const lookup = (u, what) => {
    const p = local(u);
    if (p == null) return u;
    if (C.assets[p]) return C.assets[p];
    post("error", `404 ${p} (${what})`);
    return "data:,";
  };
  const F = window.fetch;
  window.fetch = function (input, init) {
    const u = typeof input === "string" ? input : input instanceof URL ? input.href : input && input.url;
    const p = local(u);
    if (p == null) return F.call(this, input, init);
    if (C.assets[p]) return F.call(this, C.assets[p]);
    post("error", `404 ${p} (fetch)`);
    return Promise.resolve(new Response("not found", { status: 404, statusText: "Not Found" }));
  };
  const XO = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function (m, u, ...r) { return XO.call(this, m, lookup(u, "XMLHttpRequest"), ...r); };
  for (const K of [HTMLImageElement, HTMLMediaElement, HTMLSourceElement, HTMLScriptElement]) {
    const d = Object.getOwnPropertyDescriptor(K.prototype, "src");
    if (d && d.set) Object.defineProperty(K.prototype, "src", { ...d, set(v) { d.set.call(this, lookup(v, "src")); } });
  }
  const A = window.Audio;
  if (A) {
    window.Audio = function Audio(src) { const a = new A(); if (src !== undefined) a.src = src; return a; };
    window.Audio.prototype = A.prototype;
  }
  // relative links load another page of the project: the parent rebuilds the document for it
  document.addEventListener("click", (e) => {
    const a = e.target && e.target.closest && e.target.closest("a[href]");
    if (!a || e.defaultPrevented) return;
    const p = local(a.getAttribute("href"));
    if (p == null) return;
    e.preventDefault();
    send({ t: "nav", path: p });
  }, true);
  // run_js's loader (harness/run-js.js) reports that the snippet finished
  Object.defineProperty(window, "__pvDone", { value: (ok, ms, tok) => send({ t: "done", ok: !!ok, ms: +ms || 0, tok: String(tok ?? "").slice(0, 64) }) });
  addEventListener("load", () => {
    send({ t: "ready", ms: Math.round(performance.now()) });
    setTimeout(() => send({ t: "idle", ms: Math.round(performance.now()) }), 500);
  });

  // the page's content size against its viewport, for the parent to scale a fixed-size layout (a
  // 300x600 board and a side panel) down into the preview's box (preview-frame.js, fit). Measured
  // on load, resize and DOM changes, sent when it changes. Two overflows the document's scroll size
  // leaves out are added: content left of or above the page's origin (a centred flex row wider than
  // the viewport spills both ways, and only the right half scrolls), and what the body or a
  // full-page wrapper clips (overflow: hidden). A few levels deep; positioned overlays not counted.
  let first = true, last = "", queued = 0;
  const measure = () => {
    queued = 0;
    const d = document.documentElement, se = document.scrollingElement || d, b = document.body;
    if (!d) return;
    const iw = innerWidth, ih = innerHeight;
    let sw = Math.max(se.scrollWidth, d.scrollWidth), sh = Math.max(se.scrollHeight, d.scrollHeight), left = 0, top = 0, n = 0;
    const clips = (s) => s.overflowX !== "visible" || s.overflowY !== "visible";
    const clipped = (e, x, y) => { sw = Math.max(sw, x + e.scrollWidth); sh = Math.max(sh, y + e.scrollHeight); };
    const walk = (e, depth) => {
      for (const c of e.children) {
        if (++n > 400) return;
        const r = c.getBoundingClientRect(), x = r.left + scrollX, y = r.top + scrollY;
        if (r.width < 2 || r.height < 2) continue;
        let cs = null;
        const st = () => cs || (cs = getComputedStyle(c));
        if ((x < left || y < top) && !/fixed|absolute/.test(st().position)) { left = Math.min(left, x); top = Math.min(top, y); }
        if (depth >= 4 || !c.firstElementChild) continue;
        if (clips(st())) {   // a scroller or a clipped box of its own: only a full-page one counts
          if (r.width < iw * 0.9 || r.height < ih * 0.9) continue;
          clipped(c, x, y);
        }
        walk(c, depth + 1);
      }
    };
    if (b) {
      const r = b.getBoundingClientRect();
      if (clips(getComputedStyle(b))) clipped(b, r.left + scrollX, r.top + scrollY);
      walk(b, 1);
    }
    const m = { t: "size", sw: Math.ceil(sw - left), sh: Math.ceil(sh - top), cw: se.clientWidth, ch: se.clientHeight, iw, ih };
    const k = JSON.stringify(m);
    if (k === last) return;
    last = k; m.first = first; first = false;
    send(m);
  };
  const soon = () => { if (!queued) queued = setTimeout(measure, 50); };
  const watch = () => {
    measure();
    try { const ro = new ResizeObserver(soon); ro.observe(document.documentElement); if (document.body) ro.observe(document.body); } catch {}
    try { new MutationObserver(soon).observe(document.documentElement, { childList: true, subtree: true, attributes: true, characterData: true }); } catch {}
  };
  if (document.readyState === "loading") addEventListener("DOMContentLoaded", watch); else setTimeout(watch);
  addEventListener("load", soon);
  addEventListener("resize", soon);
}
export const CAPTURE_SOURCE = capture.toString();

// ---------------------------------------------------------------- the build
const escAttr = (s) => s.replace(/&/g, "&amp;").replace(/"/g, "&quot;");
const unescAttr = (s) => s.replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
const scriptJson = (o) => JSON.stringify(o).replace(/</g, "\\u003c").replace(/\u2028/g, "\\u2028").replace(/\u2029/g, "\\u2029");
const JS_TYPES = new Set(["", "text/javascript", "application/javascript", "module"]);

export function buildPreviewDoc(snapshot, { path, nonce = "" } = {}) {
  const files = snapshot.files, page = path || snapshot.entry || "index.html";
  const missing = new Set(), warnings = [], urlToPath = {}, memo = new Map();
  const text = (p) => dec.decode(files.get(p).bytes);
  const note = (url, p) => { urlToPath[url] = p; return url; };
  const find = (ref, from) => {   // -> path of an existing file, or null (missing is recorded)
    const r = resolveRef(ref, from);
    if (!r) return null;
    if (!files.has(r.path)) { missing.add(r.path); return null; }
    return r.path;
  };
  const asset = (p) => {
    if (!memo.has("a:" + p)) memo.set("a:" + p, note(dataUrl(files.get(p).type || mimeFor(p), files.get(p).bytes), p));
    return memo.get("a:" + p);
  };
  const stack = [];   // files being built, for cycle detection (JS and CSS)
  const cycle = (p) => {
    if (!stack.includes(p)) return false;
    warnings.push(`import cycle: ${[...stack.slice(stack.indexOf(p)), p].join(" -> ")} (that import is left unresolved)`);
    return true;
  };
  // JS file -> data URL, its relative imports rewritten first (depth-first)
  const js = (p) => {
    if (memo.has("j:" + p)) return memo.get("j:" + p);
    stack.push(p);
    const src = rewriteJs(text(p), p) + `\n//# sourceURL=${p}`;
    stack.pop();
    const url = note(dataUrl("text/javascript", enc.encode(src)), p);
    memo.set("j:" + p, url);
    return url;
  };
  const rewriteJs = (src, from) => {
    const refs = scanJs(src);
    for (let k = refs.length - 1; k >= 0; k--) {
      const r = refs[k];
      if (!isRelSpec(r.value)) continue;
      const p = find(r.value, from);
      if (p == null || cycle(p)) continue;
      const url = /\.m?js$/i.test(p) ? js(p) : asset(p);   // JSON / CSS module scripts: the raw file
      src = src.slice(0, r.start) + JSON.stringify(url) + src.slice(r.end);
    }
    return src;
  };
  const css = (p) => {
    if (memo.has("c:" + p)) return memo.get("c:" + p);
    stack.push(p);
    const out = rewriteCss(text(p), p);
    stack.pop();
    const url = note(dataUrl("text/css", enc.encode(out)), p);
    memo.set("c:" + p, url);
    return url;
  };
  const rewriteCss = (src, from) => src
    .replace(/@import\s+(?:url\(\s*(["']?)([^"')]*)\1\s*\)|(["'])([^"']*)\3)/g, (m, _q1, u1, _q2, u2) => {
      const p = find(u1 ?? u2, from);
      return p == null || cycle(p) ? m : `@import url("${css(p)}")`;
    })
    .replace(/url\(\s*(["']?)([^"')]*)\1\s*\)/g, (m, _q, u) => {
      const p = find(u, from);
      return p == null ? m : `url("${asset(p)}")`;
    });
  const srcset = (v, from) => v.split(",").map((c) => {
    const [u, ...d] = c.trim().split(/\s+/), p = find(u, from);
    return [p == null ? u : asset(p), ...d].join(" ");
  }).join(", ");
  const importmap = (body, from) => {
    try {
      const m = JSON.parse(body), fix = (o) => {
        for (const k of Object.keys(o || {})) {   // folder mappings ("./lib/") cannot become data URLs
          if (typeof o[k] !== "string" || o[k].endsWith("/")) continue;
          const p = find(o[k], from);
          if (p != null) o[k] = /\.m?js$/i.test(p) ? js(p) : asset(p);
        }
      };
      fix(m.imports);
      for (const s of Object.values(m.scopes || {})) fix(s);
      return JSON.stringify(m);
    } catch { return body; }
  };

  let html;
  if (!files.has(page) || !/\.html?$/i.test(page)) {
    missing.add(page);
    html = `<!doctype html><title>${page.replace(/[<&]/g, "")}</title><body style="font:14px system-ui;color:#888;padding:2em">404: ${page.replace(/[<&]/g, "")} is not in the project</body>`;
  } else html = rewriteHtml(text(page), page, { find, js, css, asset, rewriteJs, rewriteCss, srcset, importmap });

  // CSP first, then the capture script, before any of the agent's markup. Put before the agent's
  // <html>/<head>, they open an implied <head> that the agent's own <head> tag then continues.
  const keys = {}, assets = {};
  for (const [u, p] of Object.entries(urlToPath)) keys[urlKey(u)] = p;
  for (const p of files.keys()) if (!/\.html?$/i.test(p)) assets[p] = asset(p);
  const C = { nonce, path: page, keys, assets, missing: [...missing], warnings };
  // a page without a viewport meta gets one, so it lays out at the device's width when shown on
  // its own (open in a tab); inside the frame, the frame's size is the viewport either way
  const vp = /<meta\b[^>]*\bname\s*=\s*["']?viewport\b/i.test(html) ? "" : `<meta name="viewport" content="width=device-width, initial-scale=1">`;
  const head = `<meta http-equiv="Content-Security-Policy" content="${CSP}"><script>(${CAPTURE_SOURCE})(${scriptJson(C)});</script>${vp}`;
  const dt = /^\s*<!doctype[^>]*>/i.exec(html);
  html = dt ? dt[0] + head + html.slice(dt[0].length) : head + html;
  return { html, missing: [...missing], warnings, urlToPath };
}

// HTML: walk the tags, rewrite reference attributes, and the bodies of <script> and <style>.
function rewriteHtml(html, from, X) {
  const TAG = /<!--[\s\S]*?(?:-->|$)|<([a-zA-Z][\w:-]*)((?:[^>"']|"[^"]*"|'[^']*')*)>/g;
  const ATTR = /([^\s"'>\/=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
  let out = "", last = 0, mt;
  while ((mt = TAG.exec(html))) {
    out += html.slice(last, mt.index);
    last = TAG.lastIndex;
    if (!mt[1]) { out += mt[0]; continue; }   // comment
    const tag = mt[1].toLowerCase(), attrs = {}, spans = [];
    for (const a of mt[2].matchAll(ATTR)) {
      const name = a[1].toLowerCase(), raw = a[2] ?? a[3] ?? a[4];
      attrs[name] = raw == null ? "" : unescAttr(raw);
      if (raw != null) spans.push({ name, start: a.index, end: a.index + a[0].length });
    }
    const type = (attrs.type || "").toLowerCase().trim(), rel = (attrs.rel || "").toLowerCase();
    const set = {};   // attribute -> new value
    const ref = (name, fn) => { if (attrs[name] == null || attrs[name] === "") return; const p = X.find(attrs[name], from); if (p != null) set[name] = fn(p); };
    if (tag === "script") ref("src", X.js);
    else if (tag === "link") ref("href", rel.includes("stylesheet") ? X.css : rel.includes("modulepreload") ? X.js : X.asset);
    else if (!["a", "area", "base", "form", "iframe"].includes(tag)) {
      for (const n of ["src", "poster", "data", "href", "xlink:href"]) ref(n, X.asset);
      if (attrs.srcset) set.srcset = X.srcset(attrs.srcset, from);
    }
    if (attrs.style && /url\(/i.test(attrs.style)) set.style = X.rewriteCss(attrs.style, from);
    let t = mt[2];
    for (const s of spans.reverse()) if (set[s.name] != null) t = t.slice(0, s.start) + `${t.slice(s.start, s.start + s.name.length)}="${escAttr(set[s.name])}"` + t.slice(s.end);
    out += `<${mt[1]}${t}>`;
    if (tag === "script" || tag === "style") {   // raw text up to the closing tag
      const close = new RegExp(`</${tag}\\s*>`, "i"), rest = html.slice(last), c = close.exec(rest);
      const body = c ? rest.slice(0, c.index) : rest;
      if (tag === "style") out += X.rewriteCss(body, from);
      else if (type === "importmap") out += X.importmap(body, from);
      else if (!attrs.src && JS_TYPES.has(type)) out += X.rewriteJs(body, from);
      else out += body;
      out += c ? c[0] : "";
      last += body.length + (c ? c[0].length : 0);
      TAG.lastIndex = last;
    }
  }
  return out + html.slice(last);
}
