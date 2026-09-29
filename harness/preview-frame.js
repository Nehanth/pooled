// Mounts a port of a PreviewSource (PreviewServer on the host, PreviewSubscriber on a peer) in a
// sandboxed iframe (docs/design/harness-app.md B.1, B.4). The document comes from buildPreviewDoc.
// sandbox="allow-scripts" and nothing else, so the agent's code runs in an opaque origin: no
// storage, cookies, OPFS or DOM of the room page.
//
// Two ways to run it:
//   relay  - the frame is harness/preview-relay.html on another site (relayUrl()), which renders the
//            document in its own sandboxed child. Site isolation gives it its own process, so an
//            infinite loop or a memory bomb in the agent's code hangs that process, not the room:
//            the relay's heartbeat stops, and after HANG_MS the frame is removed and the hang logged.
//   local  - no other site configured: the document is a blob: URL in a frame of this page. Same
//            renderer process as the room (a hang freezes the tab), but still an opaque origin.
// In local mode the document is a blob: URL, not srcdoc: a srcdoc document inherits the room
// page's URL as its base, and that URL carries the room code. In relay mode the relay page on the
// other site loads it as srcdoc, where the inherited URL is the relay's, not the room's. A frame that navigates itself away
// (location.href = ..., meta refresh) is put back on the current rev, and stopped if it keeps doing it.
// Messages from the frame are accepted only from its own window and with this mount's nonce, and
// are treated as untrusted text (typed and capped here, shown with textContent by the UI).
//
//   mountPreview(el, source, port, { onLog, onStatus, autorun = true, relay = relayUrl(), onShow, onDone, run })
//     -> { reload(), destroy(), frame, rev }
// run: a hidden run_js frame (harness/run-js.js). It never falls back to local mode (no relay, or
// no hello: status "nohost"), and while it lives in the relay (from its hello) the other previews'
// watchdogs pause (the relay is one site, so one process: a snippet's loop would read as their
// hang). Not before its hello: a relay process that is hung already must still be seen as hung.
// When it goes while the relay is hung, the other relay previews get fresh frames, so the hung
// process has none left.
// After a hang: Chrome gives a new frame of the relay's site the process that already hosts that
// site, and a hung process goes away only some time after its last frame. A relay frame made in
// that window joins the hung process and never says hello. So relay frames made within QUIET_MS of
// a hang wait that long before loading, and on a page whose relay has answered before, a frame
// with no hello is not taken for "no relay": it and every other frame still waiting for a hello
// (so none of them keeps the old process alive) are made again after QUIET_MS, up to HELLO_RETRIES
// times, before local mode (a preview) or "nohost" (a run).
// Fit: the app's document reports its content size (preview-build.js); a page wider or taller than
// the box (a fixed 300x600 board and a side panel, on a phone) is scaled down to fit it, "contain",
// never up. The frame is laid out at the box size divided by the scale, then transform: scale()d
// back to the box, so the page sees a larger viewport and the browser maps pointer and touch
// input through the transform. Height is fitted only down to FIT_MIN_H (a long page scrolls
// instead); an axis whose overflow grows with the viewport (100vw, min-height: 100vh plus a margin)
// is left alone. A page that fits gets scale 1 and the frame is untouched. The handle's `scale`.
//   onLog({ level, text, src, line, col, ms, rev })
//   onStatus({ state: "idle"|"loading"|"ready"|"stopped"|"waiting"|"hung", rev, path })
// autorun false (a peer's first view) shows a "Run preview :port" button instead of running it.
import { buildPreviewDoc } from "./preview-build.js";

const LEVELS = new Set(["log", "info", "warn", "error"]);
const str = (v, n) => String(v ?? "").slice(0, n);
export const HANG_MS = 3000, HELLO_MS = 5000, MAX_NAV = 3, QUIET_MS = 1000, HELLO_RETRIES = 2;
// fit: the lowest scale for a tall page, for any page; steps per document; room around a scaled page (px)
const FIT_MIN_H = 0.5, FIT_MIN = 0.2, FIT_STEPS = 6, FIT_PAD = 24;
const views = new Set();   // the visible previews of this page (not run frames)
let runs = 0;              // run_js frames alive in the relay (said hello)
const waiting = new Set(); // mounts whose relay frame has not said hello yet
let relayOk = false;       // a relay frame of this page has said hello: the host is there
let quietUntil = 0;        // no new relay frame loads before this (a hung relay process is going away)
const quiet = () => { quietUntil = Date.now() + QUIET_MS; };

// The relay's address: <meta name="preview-origin" content="https://..."> on the page (a second
// deployment of this site on another registrable domain), else in development the other loopback
// name (localhost <-> 127.0.0.1 are different sites), else null (local mode).
// A meta the relay cannot isolate is ignored (local mode, and no run_js): not an http(s) URL, plain
// http from an https page (blocked as mixed content), or the page's own site (a subdomain such as
// preview.pooled.run shares the room's process, so a loop there would freeze the room).
export function relayUrl(doc = globalThis.document) {
  const loc = doc?.defaultView?.location;
  if (!loc) return null;
  const meta = doc.querySelector?.('meta[name="preview-origin"]')?.content?.trim();
  const path = "/harness/preview-relay.html";
  if (meta) {
    const why = relayProblem(meta, loc);
    if (!why) return meta.replace(/\/+$/, "") + path;
    console.warn(`preview-origin ignored: ${why}`);
    return null;
  }
  const port = loc.port ? ":" + loc.port : "";
  if (loc.hostname === "localhost") return `${loc.protocol}//127.0.0.1${port}${path}`;
  if (loc.hostname === "127.0.0.1") return `${loc.protocol}//localhost${port}${path}`;
  return null;
}

// Registrable domain ("site") of a host name, close enough for a relay check: the last two labels,
// or three under a known two-part suffix (co.uk, github.io, ...). IP addresses and single labels
// are their own site.
const SUFFIX2 = new Set(["co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "co.jp", "co.nz", "co.in",
  "com.br", "com.cn", "vercel.app", "github.io", "pages.dev", "netlify.app", "web.app", "workers.dev", "fly.dev"]);
export function siteOf(host) {
  const h = String(host || "").toLowerCase().replace(/\.$/, "");
  if (!h || /^[\d.]+$/.test(h) || h.includes(":") || !h.includes(".")) return h;
  const p = h.split(".");
  const n = SUFFIX2.has(p.slice(-2).join(".")) ? 3 : 2;
  return p.slice(-n).join(".");
}

// Why a preview-origin value cannot isolate previews from the page at `loc`, or "" when it can.
export function relayProblem(origin, loc) {
  let u;
  try { u = new URL(origin); } catch { return "not a URL"; }
  if (u.protocol !== "https:" && u.protocol !== "http:") return "not an http(s) URL";
  if (loc?.protocol === "https:" && u.protocol !== "https:") return "an https page needs an https preview origin";
  if (loc?.hostname && siteOf(u.hostname) === siteOf(loc.hostname)) return "same site as the page (it would share the room's process)";
  return "";
}

export function mountPreview(el, source, port, { onLog = () => {}, onStatus = () => {}, autorun = true, relay = undefined, onShow = null, onDone = null, run = false } = {}) {
  const doc = el.ownerDocument, win = doc.defaultView;
  if (relay === undefined) relay = relayUrl(doc);
  try { if (relay) new URL(relay); } catch { relay = null; }
  if (run && !relay) {   // a snippet in this tab's own process could freeze the room
    queueMicrotask(() => { try { onStatus({ state: "nohost", rev: 0, path: null }); } catch (e) { console.error(e); } });
    return { frame: null, mode: "none", rev: 0, path: null, loaded: false, run() {}, reload() {}, navigate() {}, destroy() {} };
  }
  let frame = null, mode = relay ? "relay" : "local";
  let nonce = "", rev = 0, path = null, detach = () => {}, running = autorun, gate = null;
  let url = null, expect = 0, navs = 0, html = null;
  let hello = false, beat = 0, dog = 0, helloTimer = 0, hung = false, tries = 0;
  // the in-frame capture script rate-limits itself, but the app's code can post directly
  let win0 = 0, count = 0;
  const flood = () => { const now = Date.now(); if (now - win0 > 1000) { win0 = now; count = 0; } return ++count > 300; };
  const status = (state) => { try { onStatus({ state, rev, path: path || source.snapshot(port)?.entry }); } catch (e) { console.error(e); } };
  const log = (level, text) => {
    const entry = { level, text, src: "(preview)", line: 0, col: 0, ms: 0, rev };
    source.pushLog?.(port, entry); onLog(entry);
  };

  // ---- fit (see the top): nat = per axis, the content size that needed a smaller scale
  let fit = 1, nat = { x: 0, y: 0 }, skip = { x: false, y: false }, step = null, steps = 0;
  const box = () => ({ w: el.clientWidth, h: el.clientHeight });
  const fitReset = () => { nat = { x: 0, y: 0 }; skip = { x: false, y: false }; step = null; steps = 0; };
  const fitTarget = (b) => {
    let t = nat.x ? Math.min(1, b.w / (nat.x + FIT_PAD)) : 1;   // a scaled page keeps a little room at its edges
    const th = nat.y ? b.h / (nat.y + FIT_PAD) : 1;
    if (th < t && th >= FIT_MIN_H) t = th;
    return Math.max(FIT_MIN, Math.min(1, t));
  };
  const applyFit = (t) => {
    fit = t >= 1 ? 1 : t;
    if (!frame) return;
    const st = frame.style;
    if (fit === 1) { st.position = st.left = st.top = st.transform = st.transformOrigin = ""; st.width = st.height = "100%"; return; }
    const b = box();
    if (win.getComputedStyle(el).position === "static") el.style.position = "relative";
    el.style.overflow = "hidden";
    st.position = "absolute"; st.left = st.top = "0"; st.transformOrigin = "0 0";
    st.width = b.w / t + "px"; st.height = b.h / t + "px"; st.transform = `scale(${t})`;
  };
  const onSize = (d) => {
    if (run || !frame) return;
    if (d.first) { fitReset(); if (fit !== 1) { applyFit(1); return; } }   // a new document starts at the box's size
    const b = box();
    if (!b.w || !b.h) return;   // hidden (another tab): measured again when it shows
    const num = (k) => Math.max(0, Number(d[k]) || 0);
    if (Math.abs(num("iw") - b.w / fit) > 2 || Math.abs(num("ih") - b.h / fit) > 2) return;   // measured before the last resize; another follows
    const o = { x: num("sw") - num("cw"), y: num("sh") - num("ch") }, v = { x: num("iw"), y: num("ih") }, s = { x: num("sw"), y: num("sh") };
    if (step && step.fit === fit) {   // the first look after a step: an overflow that did not shrink follows the viewport
      for (const a of ["x", "y"]) if (step.o[a] > 0 && o[a] >= step.o[a] * 0.8) { skip[a] = true; nat[a] = 0; }
      step = null;
    }
    for (const a of ["x", "y"]) if (!skip[a] && o[a] > Math.max(8, v[a] * 0.03)) nat[a] = Math.max(nat[a], s[a]);
    const t = fitTarget(b);
    if (Math.abs(t - fit) < 0.005 || steps >= FIT_STEPS) return;
    steps++;
    step = { fit: t >= 1 ? 1 : t, o: { x: nat.x ? o.x : 0, y: nat.y ? o.y : 0 } };
    applyFit(t);
  };
  const ro = !run && win.ResizeObserver ? new win.ResizeObserver(() => {
    if (frame && (fit !== 1 || nat.x || nat.y) && el.clientWidth && el.clientHeight) applyFit(fitTarget(box()));
  }) : null;
  ro?.observe(el);

  const makeFrame = () => {
    fitReset(); fit = 1;
    frame = doc.createElement("iframe");
    frame.setAttribute("sandbox", "allow-scripts");
    frame.setAttribute("allow", "");
    frame.setAttribute("referrerpolicy", "no-referrer");
    frame.setAttribute("title", `preview :${port}`);
    frame.className = "pv-frame";
    frame.style.cssText = "border:0;width:100%;height:100%;display:block;background:#fff";
    hello = false; expect = 0; navs = 0;
    if (!running) frame.style.display = "none";
    clearTimeout(helloTimer);
    if (mode === "relay") {
      expect = 1;
      waiting.add(me);
      const f = frame, wait = quietUntil - Date.now();
      const go = () => { if (frame === f) { f.src = relay; helloTimer = setTimeout(noHello, HELLO_MS); } };
      if (wait > 0) helloTimer = setTimeout(go, wait); else go();
    }
    el.appendChild(frame);
    frame.addEventListener("load", onFrameLoad);   // after inserting: the empty frame's about:blank load is not a navigation
  };
  const noHello = () => {
    if (hello || !frame) return;
    // the relay answered on this page before: its process is still going away after a hang
    if (relayOk && tries < HELLO_RETRIES) { quiet(); for (const m of [...waiting]) m.retry(); return; }
    waiting.delete(me);
    if (run) { status("nohost"); return; }
    // the relay never answered (not deployed, blocked): run here instead
    log("warn", "the isolated preview host did not answer; running the preview in this tab");
    mode = "local"; frame.remove(); makeFrame(); if (running) load();
  };
  const retry = () => { tries++; frame?.remove(); makeFrame(); };   // the document is sent on hello
  const blank = () => {
    html = null;
    if (!frame) return;
    if (mode === "relay") { if (hello) frame.contentWindow?.postMessage({ pvr: "doc", html: "" }, "*"); return; }
    expect = 1; frame.src = "about:blank";
    if (url) { URL.revokeObjectURL(url); url = null; }
  };
  const show = () => {
    if (!frame || html == null) return;
    if (mode === "relay") {
      if (!hello) return;   // sent on hello
      frame.contentWindow?.postMessage({ pvr: "doc", html }, "*");   // the relay is sandboxed too: an opaque origin
      onShow?.();
      return;
    }
    const old = url;
    url = URL.createObjectURL(new Blob([html], { type: "text/html" }));
    expect = 1; frame.src = url;
    onShow?.();
    if (old) setTimeout(() => URL.revokeObjectURL(old), 1000);
  };
  // local mode: every load we did not start is the page navigating itself somewhere
  function onFrameLoad() {
    if (mode !== "local") return;
    if (expect) { expect = 0; return; }
    navBlocked();
  }
  const navBlocked = (stopped = false) => {
    if (stopped || ++navs > MAX_NAV) {
      log("error", "preview stopped: the page keeps navigating away from itself");
      if (mode === "local") blank();
      status("stopped");
      return;
    }
    log("error", "navigation blocked: the preview tried to leave its page (location.href / meta refresh); reloaded rev " + rev);
    if (mode === "local") show();
  };

  const load = () => {
    const snap = source.snapshot(port);
    if (!snap) { blank(); rev = 0; status("stopped"); return; }
    if (!running) return;
    if (hung) { hung = false; gate?.remove(); gate = null; makeFrame(); }
    if (path && !snap.files.has(path)) path = null;   // the page went away: back to the entry
    nonce = crypto.getRandomValues(new Uint32Array(2)).join("-");
    rev = snap.rev; navs = 0;
    html = buildPreviewDoc(snap, { path: path || snap.entry, nonce }).html;
    show();
    status("loading");
  };

  // relay mode: no heartbeat for HANG_MS while the tab is visible means the agent's code hung the
  // preview's process. Background tabs throttle timers, so a hidden tab never counts as hung.
  const watchdog = () => {
    clearInterval(dog);
    beat = Date.now();
    dog = setInterval(() => {
      if (doc.visibilityState !== "visible" || (!run && runs)) { beat = Date.now(); return; }
      if (Date.now() - beat > HANG_MS) onHang();
    }, 500);
  };
  const onHang = () => {
    clearInterval(dog); dog = 0;
    hung = true; hello = false; quiet();
    frame?.remove(); frame = null;
    detach(); detach = () => {};
    log("error", `preview hung (infinite loop?): no answer for ${HANG_MS / 1000} s, so it was stopped; it runs again on the next edit`);
    status("hung");
    gate = doc.createElement("button");
    gate.type = "button"; gate.className = "pv-run"; gate.textContent = `preview hung · run :${port} again`;
    gate.addEventListener("click", () => { detach = source.attach?.(port) || (() => {}); load(); });
    el.appendChild(gate);
  };

  const onMessage = (e) => {
    if (!frame || e.source !== frame.contentWindow) return;
    const d = e.data;
    if (!d || typeof d !== "object") return;
    if (mode === "relay" && typeof d.pvr === "string") {
      if (d.pvr === "hello" && !hello) {
        hello = true; relayOk = true; tries = 0; waiting.delete(me); clearTimeout(helloTimer);
        if (run && !counted) { counted = true; runs++; }
        watchdog(); show();
      }
      else if (d.pvr === "beat") beat = Date.now();
      else if (d.pvr === "nav") navBlocked(!!d.stopped);
      return;
    }
    if (d.pv !== nonce || flood()) return;
    const ms = Math.max(0, Number(d.ms) || 0);
    if (d.t === "log") {
      const t = String(d.text ?? "");
      const entry = { level: LEVELS.has(d.level) ? d.level : "log", text: t.length > 1200 ? t.slice(0, 1194) + "…(cut)" : t, src: str(d.src, 200), line: d.line >>> 0, col: d.col >>> 0, ms, rev };
      source.pushLog?.(port, entry);
      onLog(entry);
    } else if (d.t === "ready" || d.t === "idle") {
      source.frameEvent?.(port, { t: d.t, rev, ms });
      if (d.t === "ready") status("ready");
    } else if (d.t === "size") onSize(d);
    else if (d.t === "done") onDone?.(d);   // run_js's snippet finished
    else if (d.t === "nav") {
      const p = str(d.path, 300);
      const snap = source.snapshot(port);
      if (snap?.files.has(p) && /\.html?$/i.test(p)) { path = p; load(); }
      else {
        const entry = { level: "error", text: `404 ${p} (link)`, src: "", line: 0, col: 0, ms, rev };
        source.pushLog?.(port, entry); onLog(entry);
      }
    }
  };
  win.addEventListener("message", onMessage);
  const off = source.onUpdate((u) => {
    if (u.port !== port) return;
    if (u.stopped) { detach(); detach = () => {}; blank(); rev = 0; status("stopped"); return; }
    if (running) { detach(); detach = source.attach?.(port) || (() => {}); }
    load();
  });

  const start = () => {
    running = true;
    gate?.remove(); gate = null;
    frame.style.display = "block";
    detach = source.attach?.(port) || (() => {});
    load();
  };
  // a run frame left while the relay was hung: move this preview to a fresh frame, quietly
  const refresh = () => {
    if (mode !== "relay" || !frame) return;
    clearInterval(dog); dog = 0; clearTimeout(helloTimer);
    frame.remove(); makeFrame(); if (running) load();
  };
  const me = { refresh, retry };
  let gone = false, counted = false;
  if (!run) views.add(me);
  makeFrame();
  if (autorun) start();
  else {
    gate = doc.createElement("button");
    gate.type = "button"; gate.className = "pv-run"; gate.textContent = `Run preview :${port}`;
    gate.addEventListener("click", start);
    el.appendChild(gate);
    status("waiting");
  }
  return {
    get frame() { return frame; },
    get mode() { return mode; },
    get rev() { return rev; },
    get path() { return path; },
    get loaded() { return html != null; },
    get scale() { return fit; },
    run: start,
    reload() { if (running) load(); },
    navigate(p) { path = p || null; load(); },
    destroy() {
      if (gone) return;
      gone = true;
      if (run) {
        if (counted) runs--;
        const stuck = mode === "relay" && (hung || (hello && Date.now() - beat > 1500));
        if (stuck) { quiet(); queueMicrotask(() => { for (const v of views) v.refresh(); }); }
      } else views.delete(me);
      waiting.delete(me);
      off(); detach(); clearInterval(dog); clearTimeout(helloTimer); ro?.disconnect();
      win.removeEventListener("message", onMessage);
      frame?.remove(); gate?.remove();
      if (url) URL.revokeObjectURL(url);
    },
  };
}

// "open ↗": the page in a tab of its own. The tab is a blob: page of this origin holding only a
// sandboxed frame (allow-scripts, as here), so the agent's code still runs in an opaque origin;
// its srcdoc's base is that blob: URL, which says nothing about the room. noopener: no way back.
export function openPreviewTab(source, port, path = null) {
  const snap = source.snapshot(port);
  if (!snap) return false;
  const { html } = buildPreviewDoc(snap, { path: path && snap.files.has(path) ? path : snap.entry, nonce: "" });
  const esc = (t) => t.replace(/&/g, "&amp;").replace(/"/g, "&quot;");
  const page = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>localhost:${port}</title>`
    + `<style>html,body{margin:0;height:100%;background:#fff}iframe{border:0;width:100%;height:100%;display:block}</style>`
    + `<iframe sandbox="allow-scripts" allow="" referrerpolicy="no-referrer" srcdoc="${esc(html)}"></iframe>`;
  const url = URL.createObjectURL(new Blob([page], { type: "text/html" }));
  globalThis.open?.(url, "_blank", "noopener");
  setTimeout(() => URL.revokeObjectURL(url), 60000);
  return true;
}
