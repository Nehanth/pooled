// Code mode's DOM (docs/design/harness-app.md E): the agent timeline, tool and diff cards, the
// file tree and the preview pane. No agent or room logic here: room/code.js drives it.
//
// The timeline is rendered only from ai-code-* messages, on the host as on the peers (the host
// renders its own messages before broadcasting them), so every screen shows the same thing.
// Everything that came from the model or the preview is untrusted text: it goes in with
// textContent, and model prose through mdChat (room/markdown.js), which escapes first.
import { mdChat } from "./markdown.js";
// a namespace import: a tab loaded before a deploy keeps the old working.js in memory, and a named
// import of something newer would fail to link (Safari: "Importing binding name ... is not found")
import * as W from "./working.js";

const $ = (id) => document.getElementById(id);
const h = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
const CON_MAX = 300;   // console rows kept per port

// rows: lineDiff output as [[op, text, skip?]] (the wire form). Too long to diff: head / tail
// (the first and last lines of the proposed file) and, on the host, full for "view full file".
// A tool call still being typed: { name, path, code } for write_file / edit_file, else null. Both
// the Qwen XML format and JSON are read loosely (the call is unfinished by definition).
function parseLive(raw) {
  const fx = /<function=([^>\s]+)>/.exec(raw);
  if (fx) {
    const name = fx[1];
    if (name !== "write_file" && name !== "edit_file") return null;
    const path = /<parameter=path>\n?([^\n<]*)/.exec(raw)?.[1]?.trim() || "";
    const m = /<parameter=(content|new)>\n?([\s\S]*)$/.exec(raw);
    const code = m ? m[2].replace(/\n?<\/parameter>[\s\S]*$/, "") : "";
    return { name, path, code };
  }
  const nm = /"name"\s*:\s*"(write_file|edit_file)"/.exec(raw);
  if (!nm) return null;
  const path = /"path"\s*:\s*"((?:[^"\\]|\\.)*)"/.exec(raw)?.[1] || "";
  const m = /"(content|new)"\s*:\s*"((?:[^"\\]|\\.)*)/.exec(raw);
  let code = m ? m[2] : "";
  code = code.replace(/\\$/, "").replace(/\\(u[0-9a-fA-F]{4}|.)/g, (_, c) => c[0] === "u" && c.length === 5 ? String.fromCharCode(parseInt(c.slice(1), 16)) : c === "n" ? "\n" : c === "t" ? "\t" : c === "r" ? "" : c);
  return { name: nm[1], path, code };
}

function diffBlock(d, onFull) {
  const box = h("div", "cm-diff" + (d.isNew ? " new" : ""));
  const head = h("div", "dh");
  head.append(h("b", null, d.path));
  if (d.isNew) head.append(h("span", null, `new file · ${plural(d.lines, "line")}`));
  else { head.append(h("span", "add", `+${d.add}`), h("span", "del", `-${d.del}`)); }
  if (d.full != null && onFull) {
    const b = h("button", "full", "view full file"); b.type = "button";
    b.onclick = () => onFull(d.path, d.full);
    head.append(b);
  }
  const rows = h("div", "rows");   // as wide as the longest line, so every row's tint spans it
  if (d.risky) box.append(h("div", "risk", "this file can run commands on your machine (a script, a hook, an npm script): read it before approving"));
  box.append(head, rows);
  if (!d.rows) {
    const all = [...(d.head || [])], tail = d.tail || [];
    const addRow = (t) => { const r = h("div", "r r-add"); r.innerHTML = t ? highlight(d.path, t) : " "; rows.append(r); };
    for (const t of all) addRow(t);
    const hidden = d.lines - all.length - tail.length;
    if (hidden > 0 || !all.length) rows.append(h("div", "r r-skip", all.length ? `… ${plural(hidden, "more line")} (view full file to read them)` : d.isNew ? `(${plural(d.lines, "line")}, too long to show)` : "(too many changes to show)"));
    for (const t of tail) addRow(t);
    return box;
  }
  if (!d.isNew && !d.add && !d.del) { rows.append(h("div", "r r-skip", "(no changes)")); return box; }
  for (const [op, text, skip] of d.rows) {
    if (skip) { rows.append(h("div", "r r-skip", `… ${plural(skip, "unchanged line")}`)); continue; }
    const r = h("div", "r" + (op === "+" ? " r-add" : op === "-" ? " r-del" : "")); r.innerHTML = text ? highlight(d.path, text) : " "; rows.append(r);
  }
  if (d.more) rows.append(h("div", "r r-skip", `… ${plural(d.more, "more line")}`));
  return box;
}

// ---------------- a small highlighter for the editor (JS/TS/JSON, CSS, HTML with its scripts and
// styles); anything else shows as plain text. It only wraps spans around escaped text.
const escH = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const KW = new Set(("const let var function return if else for while do break continue new class extends import export from default async await try catch "
  + "finally throw typeof instanceof in of switch case this null undefined true false yield delete void static get set super interface type enum as").split(" "));
const JS_RE = /(\/\/[^\n]*|\/\*[\s\S]*?(?:\*\/|$))|(`(?:\\[\s\S]|[^`\\])*`?|"(?:\\.|[^"\\\n])*"?|'(?:\\.|[^'\\\n])*'?)|\b(0[xX][\da-fA-F]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b|([A-Za-z_$][\w$]*)(\s*\()?/g;
function hlJS(src) {
  let out = "", at = 0;
  for (const m of src.matchAll(JS_RE)) {
    out += escH(src.slice(at, m.index)); at = m.index + m[0].length;
    if (m[1]) out += `<span class="t-c">${escH(m[1])}</span>`;
    else if (m[2]) out += `<span class="t-s">${escH(m[2])}</span>`;
    else if (m[3]) out += `<span class="t-n">${escH(m[3])}</span>`;
    else if (KW.has(m[4])) out += `<span class="t-k">${m[4]}</span>${escH(m[5] || "")}`;
    else if (m[5]) out += `<span class="t-f">${escH(m[4])}</span>${escH(m[5])}`;
    else out += escH(m[4]);
  }
  return out + escH(src.slice(at));
}
const CSS_RE = /(\/\*[\s\S]*?(?:\*\/|$))|("(?:\\.|[^"\\\n])*"?|'(?:\\.|[^'\\\n])*'?)|(#[\da-fA-F]{3,8}\b)|(@[\w-]+)|([\w-]+)(?=\s*:[^;{}]*[;}])|(-?\b\d+(?:\.\d+)?(?:px|em|rem|%|vh|vw|vmin|vmax|s|ms|deg|fr|ch)?)/g;
function hlCSS(src) {
  let out = "", at = 0;
  for (const m of src.matchAll(CSS_RE)) {
    out += escH(src.slice(at, m.index)); at = m.index + m[0].length;
    const cls = m[1] ? "t-c" : m[2] ? "t-s" : m[3] ? "t-n" : m[4] ? "t-k" : m[5] ? "t-p" : "t-n";
    out += `<span class="${cls}">${escH(m[0])}</span>`;
  }
  return out + escH(src.slice(at));
}
function hlHTML(src) {
  let out = "", i = 0;
  while (i < src.length) {
    const lt = src.indexOf("<", i);
    if (lt < 0) { out += escH(src.slice(i)); break; }
    out += escH(src.slice(i, lt));
    if (src.startsWith("<!--", lt)) {
      const e = src.indexOf("-->", lt + 4), end = e < 0 ? src.length : e + 3;
      out += `<span class="t-c">${escH(src.slice(lt, end))}</span>`; i = end; continue;
    }
    const tm = /^<\/?([A-Za-z][\w:-]*|!doctype)/i.exec(src.slice(lt, lt + 40));
    if (!tm) { out += "&lt;"; i = lt + 1; continue; }
    out += `<span class="t-t">${escH(tm[0])}</span>`;
    let j = lt + tm[0].length;
    // attributes, up to the tag's end
    const AR = /\s+|([^\s=>\/]+)(\s*=\s*("[^"]*"?|'[^']*'?|[^\s>]+))?|(\/?>)|([\s\S])/y;
    AR.lastIndex = j;
    let m;
    while (j < src.length && (m = AR.exec(src))) {
      j = AR.lastIndex;
      if (m[4]) { out += `<span class="t-t">${escH(m[4])}</span>`; break; }
      if (m[1]) out += `<span class="t-a">${escH(m[1])}</span>` + (m[2] ? escH(m[2].slice(0, m[2].length - m[3].length)) + `<span class="t-s">${escH(m[3])}</span>` : "");
      else out += escH(m[0]);
    }
    i = j;
    // a script or a style: its body in its own language
    const name = tm[1].toLowerCase();
    if (!tm[0].startsWith("</") && (name === "script" || name === "style")) {
      const close = src.toLowerCase().indexOf("</" + name, i), end = close < 0 ? src.length : close;
      out += (name === "script" ? hlJS : hlCSS)(src.slice(i, end));
      i = end;
    }
  }
  return out;
}
const LANG = { js: hlJS, mjs: hlJS, cjs: hlJS, ts: hlJS, jsx: hlJS, tsx: hlJS, json: hlJS, css: hlCSS, html: hlHTML, htm: hlHTML, svg: hlHTML, xml: hlHTML };
export function highlight(path, text) {
  const f = LANG[(/\.([\w]+)$/.exec(path || "")?.[1] || "").toLowerCase()];
  return f && text.length < 400000 ? f(text) : escH(text);
}

// the dots logo, inline (no image to fetch, so it never pops in late)
const DOTS = '<svg class="dl" viewBox="0 0 24 24" aria-hidden="true">' + [[3.4, 3.4, 1.8, 0], [10.2, 3.4, 1.99, 1], [18.5, 3.4, 2.38, 2], [3.4, 10.2, 1.99, 1], [10.2, 10.2, 2.38, 2], [18.5, 10.2, 2.94, 3], [3.4, 18.5, 2.38, 2], [10.2, 18.5, 2.94, 3], [18.5, 18.5, 3.9, 4]]
  .map(([x, y, r, k]) => `<circle cx="${x}" cy="${y}" r="${r}" style="--rc:${k}"/>`).join("") + "</svg>";

export function codeUI({ onMode = () => {} } = {}) {
  const pane = $("chatpane"), log = $("code-log");
  // host: this tab runs the agent (the model host). drive: this screen can send requests, stop
  // its own run and answer its own approvals (the host, and any member when the room shares Code)
  let host = false, drive = false, mode = "chat", empty = null, waiting = 0;
  let waitText = () => "waiting for approval";
  const title0 = document.title;
  // short announcements for screen readers (the streamed log itself is not live)
  const say = (text) => { const s = $("code-status"); if (s) s.textContent = text; };

  // ---------------- mode switch
  function show(m) {
    mode = m;
    pane.classList.toggle("code-mode", m === "code");
    $("mode-chat").setAttribute("aria-selected", String(m === "chat"));
    $("mode-code").setAttribute("aria-selected", String(m === "code"));
    if (m === "code" && !waiting) $("mode-code").classList.remove("fresh");
    onMode(m);
  }
  $("mode-chat").addEventListener("click", () => show("chat"));
  // Left / Right move between the tabs of a tablist
  const arrows = (list) => list.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    const tabs = [...list.querySelectorAll('[role="tab"]')].filter((t) => !t.hidden), i = tabs.indexOf(document.activeElement);
    if (i < 0) return;
    const t = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    e.preventDefault(); t.focus(); t.click();
  });
  arrows($("mode-bar")); arrows($("code-out-tabs")); arrows($("code-tabs"));
  // ---------------- phones (640px and narrower): one view at a time, Agent / Preview / Files, from a
  // tab bar at the bottom. The last tab is kept for the session; a dot on a tab says something
  // happened there (a new revision, an approval waiting) or, on Agent, that the agent is working.
  const phone = matchMedia("(max-width: 640px)");
  const cp = $("code-pane"), TABS = ["agent", "preview", "files"];
  let ptab = "agent";
  try { const t = sessionStorage.getItem("pooled-code-tab"); if (TABS.includes(t)) ptab = t; } catch {}
  const tabBtn = (t) => $("ctab-" + t);
  function badge(t, on) { tabBtn(t)?.classList.toggle("badge", !!on); }
  function setTab(t, { focus = false } = {}) {
    if (!TABS.includes(t)) return;
    ptab = t; cp.dataset.ptab = t;
    try { sessionStorage.setItem("pooled-code-tab", t); } catch {}
    for (const x of TABS) {
      const b = tabBtn(x), on = x === t;
      b.setAttribute("aria-selected", String(on)); b.tabIndex = on ? 0 : -1;
    }
    tabBtn("files").setAttribute("aria-controls", cp.classList.contains("ed-open") ? "code-out" : "code-files");
    if (t !== "agent" || !waiting) badge(t, false);
    if (t === "agent") { badge("agent", false); requestAnimationFrame(() => { log.scrollTop = log.scrollHeight; jump.hidden = true; }); }
    if (phone.matches) outTab(t === "files" ? "files" : "preview");
    if (focus) tabBtn(t).focus();
  }
  // the editor, full screen inside Files (Back returns to the tree)
  function edOpen(on) {
    cp.classList.toggle("ed-open", !!on);
    tabBtn("files").setAttribute("aria-controls", on ? "code-out" : "code-files");
    if (on && phone.matches && ptab !== "files") setTab("files");
  }
  $("ed-back").onclick = () => { edOpen(false); $("code-tree").querySelector(".f.on")?.focus(); };
  $("code-tabs").addEventListener("click", (e) => { const b = e.target.closest("button[data-ptab]"); if (b) setTab(b.dataset.ptab); });
  function served() {
    let first = true;
    try { first = !sessionStorage.getItem("pooled-code-served"); sessionStorage.setItem("pooled-code-served", "1"); } catch {}
    if (first && phone.matches && mode === "code") setTab("preview");
    else if (ptab !== "preview") badge("preview", true);
  }
  function newRev() { if (ptab !== "preview") badge("preview", true); }
  function busy(on) { tabBtn("agent").classList.toggle("busy", !!on); }
  // the panes' roles follow the layout
  const roles = () => {
    for (const id of ["code-agent", "code-out", "code-files"]) { if (phone.matches) $(id).setAttribute("role", "tabpanel"); else $(id).removeAttribute("role"); }
    for (const [id, t] of [["code-agent", "agent"], ["code-out", "preview"], ["code-files", "files"]]) { if (phone.matches) $(id).setAttribute("aria-labelledby", "ctab-" + t); else $(id).removeAttribute("aria-labelledby"); }
    if (phone.matches) setTab(ptab);
  };
  phone.addEventListener("change", roles);
  cp.dataset.ptab = ptab;

  const poke = () => { $("mode-bar").hidden = false; if (mode !== "code") $("mode-code").classList.add("fresh"); };

  // ---------------- timeline
  const near = () => log.scrollHeight - log.scrollTop - log.clientHeight < 80;
  // new output while the user reads further up: a pill offers the way down (nothing lands out of sight unannounced)
  const jump = h("button", "cm-jump"); jump.type = "button"; jump.hidden = true;
  jump.innerHTML = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 2v8M2.5 6.5L6 10l3.5-3.5"/></svg>New output';
  log.after(jump);
  jump.onclick = () => { log.scrollTo({ top: log.scrollHeight, behavior: "smooth" }); jump.hidden = true; };
  log.addEventListener("scroll", () => { if (near()) jump.hidden = true; }, { passive: true });
  const follow = (stick) => { if (stick) log.scrollTop = log.scrollHeight; else if (!empty) jump.hidden = false; };
  const add = (el, before = null) => { const stick = near(); empty?.remove(); empty = null; if (before?.parentNode === log) log.insertBefore(el, before); else log.append(el); follow(stick); return el; };
  const key = (...a) => a.join(":");
  const find = (k) => log.querySelector(`[data-k="${CSS.escape(k)}"]`);
  // the model's text for a step is one paragraph until a card comes after it; text after the card
  // is a new paragraph under the card (it used to be appended above it, out of sight)
  const closeText = () => { for (const t of log.querySelectorAll('.cm-text[data-k]')) { t.removeAttribute("aria-busy"); t.dataset.k0 = t.dataset.k; t.removeAttribute("data-k"); } };
  // the model picker lives in Chat now, not in a sidebar
  const words = (t) => String(t).replace(/in the sidebar/g, "in Chat");
  function placeholder(text) {
    log.replaceChildren();
    empty = h("div", "cm-empty");
    empty.innerHTML = DOTS + words(text);
    log.append(empty);
  }
  function clear() { log.replaceChildren(); empty = null; jump.hidden = true; viewFile(null); runAt = 0; editWin.close(); }

  // the wait before the model's first output (after a request, and again after each tool result):
  // the working line at the end of the timeline, gone as soon as text, code or a tool call arrives
  let runAt = 0;
  function wait(on) {
    log.querySelector(".cm-working")?.remove();
    if (!on || !runAt) return;
    const w = h("div", "cm-working");
    w.append(W.working({ since: runAt, label: "the agent is working" }));
    add(w);
  }

  function apply(d) {
    switch (d.t) {
      case "ai-code-start": {
        say("the agent started");
        const u = h("div", "cm-user");
        u.append(h("div", "who", d.name || "host"), h("div", "bubble", d.text));
        add(u);
        runAt = performance.now();
        wait(true);
        busy(true);
        break;
      }
      case "ai-code-tok": {
        const k = key("t", d.mid, d.step);
        let el = find(k);
        if (!el) { el = add(h("div", "cm-text")); el.dataset.k = k; el.dataset.raw = ""; el.setAttribute("aria-busy", "true"); }
        el.dataset.raw += d.text;
        const stick = near();
        el.innerHTML = mdChat(el.dataset.raw.replace(/^\s+/, ""));
        if (!el.dataset.raw.trim()) el.hidden = true;   // (kept, so the next piece of the same text finds it)
        else { el.hidden = false; wait(false); }
        follow(stick);
        break;
      }
      case "ai-code-live": liveCard(d); break;
      case "ai-code-tool": toolCard(d); break;
      case "ai-code-note": wait(false); closeText(); add(h("div", "cm-note" + (d.err ? " err" : ""), words(d.text))); break;
      case "ai-code-done": {
        runAt = 0; wait(false); editWin.close(); busy(false);
        for (const l of log.querySelectorAll(".cm-live")) l.remove();
        closeText();
        const text = d.stats || `${plural(d.steps || 0, "step")} · ${d.reason || "done"}`;
        add(h("div", "cm-stats", text));
        say("the agent finished: " + text);
        break;
      }
    }
  }

  // code being written: the model is still typing a write_file / edit_file call. Shown live, then
  // replaced by the finished call's card (with its diff) when the call completes.
  function liveCard(d) {
    const k = key("l", d.mid, d.step, d.n);
    let el = find(k);
    // the call is complete: the card stays (hidden) as the place its tool card goes, so text the
    // model writes after the call lands under it
    if (d.end) { if (el) { el.hidden = true; el.classList.add("ended"); } editWin.end(); return; }
    if (!el) {
      closeText();
      el = h("div", "cm-live"); el.dataset.k = k; el.dataset.raw = "";
      const head = h("div", "lh"); head.append(h("span", "nm", ""), h("b", "", ""), h("span", "n", ""));
      el.append(head, h("pre", "code"));
      el.hidden = true; add(el);
    }
    el.dataset.raw = (d.reset ? "" : el.dataset.raw) + (d.text || "");
    const p = parseLive(el.dataset.raw);
    if (!p || !p.code) return;
    const stick = near(), pre = el.querySelector("pre");
    if (el.hidden) wait(false);
    el.hidden = false;
    el.querySelector(".nm").textContent = p.name === "edit_file" ? "editing" : "writing";
    el.querySelector("b").textContent = p.path || "";
    el.querySelector(".n").textContent = `${p.code.split("\n").length} lines`;
    pre.innerHTML = highlight(p.path, p.code);   // syntax colours as it is written
    pre.scrollTop = pre.scrollHeight;
    follow(stick);
    editWin.show(p);
  }

  // ---------------- the edit overlay: once an app is served, while the agent edits a file the preview
  // frosts over with the Pooled dots in their wave and "Editing game.js" (no code: the code shows in the agent's card), then
  // "Reloading" once the call is complete, and it lifts when the preview has reloaded (or after a moment).
  // Host and peers alike (it is drawn from the same ai-code-live messages).
  const editWin = (() => {
    let el = null, raf = 0, last = null, closeT = 0, revAt = 0, shownAt = 0, doneT = 0;
    const served = () => { const P = ports.get(active); return P && P.rev > 0 ? P : null; };
    function build() {
      el = h("div", "ew"); el.setAttribute("role", "status");
      const box = h("div", "ew-box");
      box.innerHTML = W.markSVG ? W.markSVG(40) : "";
      box.append(h("span", "ew-t", ""));
      el.append(box);
    }
    function paint() {
      raf = 0;
      if (!el || !last) return;
      const name = (last.path || "").split("/").pop();
      el.querySelector(".ew-t").textContent = el.classList.contains("done") ? "Reloading\u2026" : `${last.name === "edit_file" ? "Editing" : "Writing"} ${name}`;
      el.setAttribute("aria-label", el.querySelector(".ew-t").textContent);
    }
    function show(p) {
      const P = served();
      if (!P && !el?.isConnected) return;
      clearTimeout(closeT); closeT = 0;
      if (!el) build();
      const wrap = $("pv-frame-wrap");
      if (el.parentNode !== wrap) wrap.append(el);
      if (!el.isConnected || el.classList.contains("out") || el.classList.contains("done")) shownAt = performance.now();
      clearTimeout(doneT); doneT = 0;
      el.classList.remove("out", "done");
      last = p; revAt = P?.rev || 0;
      paint();   // just a label now: set it at once (a fast edit would otherwise skip straight to Reloading)
    }
    function close() {
      clearTimeout(closeT); closeT = 0; clearTimeout(doneT); doneT = 0;
      if (!el?.isConnected || el.classList.contains("out")) return;
      el.classList.add("out");
      const gone = () => { if (el.classList.contains("out")) el.remove(); };
      if (matchMedia("(prefers-reduced-motion: reduce)").matches) gone(); else setTimeout(gone, 200);
    }
    // the call is complete: the file is written, the preview reloads; close once it has (or soon)
    // "Editing <file>" stays up at least a moment, even when the call completes at once
    function end() {
      if (!el?.isConnected) return;
      const wait = Math.max(0, 900 - (performance.now() - shownAt));
      clearTimeout(doneT);
      doneT = setTimeout(() => { doneT = 0; if (!el?.isConnected) return; el.classList.add("done"); paint(); clearTimeout(closeT); closeT = setTimeout(close, 2500); }, wait);
    }
    // the preview reloaded with the new file: a beat to see the last lines, then close
    function reloaded(port, rev) {
      if (el?.isConnected && port === active && rev > revAt) { const go = () => { if (!el?.isConnected || el.classList.contains("out")) return; if (el.classList.contains("done")) { clearTimeout(closeT); closeT = setTimeout(close, 450); } else setTimeout(go, 100); }; go(); }
    }
    return { show, end, close, reloaded };
  })();

  function toolCard(d) {
    const k = key("c", d.mid, d.i);
    let el = find(k);
    // a call came in: no more waiting. Its result goes back to the model, which thinks again
    if (d.state === "done" || d.state === "error" || d.state === "declined") queueMicrotask(() => wait(true));
    else if (!el || d.state) wait(false);
    if (!el) {
      // the model's text before a tool call is complete once the call starts; the card takes the
      // place of the live card that showed the call being typed
      closeText();
      const at = log.querySelector(".cm-live");
      el = add(h("div", "cm-tool"), at);
      at?.remove();
      el.dataset.k = k;
      el.dataset.name = d.name || "?";
      const det = h("details"), sum = h("summary");
      sum.append(h("span", "nm", el.dataset.name), h("span", "br", ""), h("span", "ms", ""), h("span", "chip", ""));
      det.append(sum);
      el.append(det);
    }
    const stick = near(), det = el.firstChild, sum = det.firstChild;
    if (d.brief != null) {
      sum.querySelector(".br").textContent = d.brief.startsWith(el.dataset.name) ? d.brief.slice(el.dataset.name.length).trim() : d.brief;
      sum.querySelector(".br").title = d.brief;
    }
    const chip = sum.querySelector(".chip");
    if (d.state) {
      chip.className = "chip " + d.state.replace(/[^a-z]/g, "");
      chip.textContent = d.state === "pending" ? "needs approval" : d.state;
      if (d.state === "running") say(`running ${el.dataset.name}`);
      else if (d.state === "pending") say(`${el.dataset.name} needs approval`);
      else if (d.state === "done" || d.state === "error") say(`${el.dataset.name} ${d.state}`);
    }
    if (d.ms != null) sum.querySelector(".ms").textContent = d.ms < 1000 ? `${d.ms} ms` : `${(d.ms / 1000).toFixed(1)} s`;
    if (d.result != null) {
      let pre = det.querySelector("pre.res");
      if (!pre) { pre = h("pre", "res"); det.append(pre); }
      pre.textContent = d.result;
    }
    if (d.diff && !el.querySelector(".cm-diff")) el.append(diffBlock(d.diff, host ? viewFull : null));
    // everyone sees the pending state; whoever may answer it (the asker, the host) gets buttons from ask()
    let ap = el.querySelector(".cm-approve");
    if (d.state === "pending" && !host && !ap) { ap = h("div", "cm-approve"); ap.append(h("span", "wait", waitText(d.mid))); el.append(ap); }
    if (d.state !== "pending") ap?.remove();
    if (d.state === "error") det.open = true;
    // declined: the diff is struck through, and the reason, if one was given, says why
    if (d.state) el.classList.toggle("declined", d.state === "declined");
    if (d.state === "declined" || (el.classList.contains("declined") && d.result != null)) {
      const why = /^declined by the user: (.+)$/s.exec(String(d.result ?? ""))?.[1]?.trim();
      let line = el.querySelector(".cm-declined");
      if (!line) { line = h("div", "cm-declined"); el.append(line); }
      line.textContent = why && why !== "stopped" ? `Declined: ${why}` : "Declined";
    }
    follow(stick);
    return el;
  }

  // Approve / Reject… / Allow edits for this task, on the card of call i, for the host and for the
  // member who asked. While it waits, the Code tab carries a dot and the page title says so.
  function waitMark(on) {
    waiting = Math.max(0, waiting + (on ? 1 : -1));
    if (waiting) { $("mode-code").classList.add("fresh"); document.title = "(needs approval) " + title0; }
    else { if (mode === "code") $("mode-code").classList.remove("fresh"); document.title = title0; }
    badge("agent", waiting > 0 && ptab !== "agent");
  }
  function ask(mid, i, { risky = false } = {}) {
    const el = find(key("c", mid, i));
    if (!el) return Promise.resolve({ ok: false, reason: "approval card missing" });
    el.querySelector(".cm-approve")?.remove();
    const ap = h("div", "cm-approve"); ap.dataset.k = key("c", mid, i);
    const yes = h("button", "ok", "Approve"), no = h("button", null, "Reject…"), all = h("button", null, "Allow edits for this task");
    yes.type = no.type = all.type = "button";
    ap.append(yes, no);
    if (!risky) ap.append(all);   // a risky file asks every time anyway
    // a phone: the question waits above the prompt (the card may be far up the log), with what it is about
    const docked = phone.matches, head = () => {
      if (!docked) return;
      const t = h("div", "cd-t", `${el.dataset.name === "edit_file" ? "Edit" : el.dataset.name === "write_file" ? "Write" : "Run"}? `);
      t.append(h("span", null, el.querySelector(".br")?.textContent || ""));
      ap.prepend(t);
    };
    head();
    if (docked) { $("code-dock").append(ap); follow(true); }
    else { el.append(ap); el.scrollIntoView({ block: "nearest" }); }
    if (mode === "code") yes.focus({ preventScroll: true });
    waitMark(true);
    let settled = false;
    return new Promise((res) => {
      const done = (v) => { ap.remove(); if (!settled) { settled = true; waitMark(false); } res(v); };
      ap.cancel = () => { if (!settled) { settled = true; waitMark(false); } ap.remove(); };
      yes.onclick = () => done(true);
      all.onclick = () => done("all");
      no.onclick = () => {
        ap.replaceChildren(); head();
        const why = h("input"); why.type = "text"; why.placeholder = matchMedia("(max-width: 640px)").matches ? "why? (optional)" : "why? (optional, the agent reads it)"; why.maxLength = 300;
        const send = h("button", null, "Reject"); send.type = "button";
        const back = h("button", null, "Cancel"); back.type = "button";
        ap.append(why, send, back);
        why.focus();
        const go = () => done({ ok: false, reason: why.value.trim() });
        send.onclick = go;
        const cancel = () => { if (!settled) { settled = true; waitMark(false); } ap.remove(); ask(mid, i, { risky }).then(res); };
        why.onkeydown = (e) => { if (e.key === "Enter") go(); else if (e.key === "Escape") { e.preventDefault(); cancel(); } };
        back.onclick = cancel;
      };
    });
  }

  // the run was stopped while the card waited: take the buttons away
  function cancelAsk(mid, i) { (find(key("c", mid, i))?.querySelector(".cm-approve") || [...$("code-dock").children].find((a) => a.dataset.k === key("c", mid, i)))?.cancel?.(); }

  // ---------------- files: the tree, and the editor (a highlighted layer under a transparent
  // textarea: native editing, undo and selection, with colours). The host edits and saves into the
  // project (the preview reloads, the agent is told); peers and proposed files open read-only.
  let fileClick = () => {}, saveFile = null;
  const drafts = new Map();   // path -> unsaved text, kept while other files are open
  let edPath = null, edBase = "", edRO = true, hlRaf = 0;
  const ta = $("ed-text"), hl = $("ed-hl"), gutter = $("ed-ln"), edBox = $("ed");
  const PHONE = matchMedia("(max-width: 640px)");
  if (PHONE.matches) $("code-prompt").placeholder = "";   // phones: an empty box (the Agent tab says what it is)
  $("ed-save").querySelector("kbd").textContent = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent) ? "\u2318S" : "Ctrl+S";
  function tree(paths) {
    const t = $("code-tree");
    t.replaceChildren();
    if (!paths?.length) { t.append(h("div", "none", "No files yet")); return; }
    let prev = [];
    for (const p of paths.slice(0, 500)) {
      const parts = p.split("/");
      for (let i = 0; i < parts.length - 1; i++) {
        if (prev[i] === parts[i]) continue;
        const d = h("div", "d", parts[i]); d.style.paddingLeft = 8 + i * 14 + "px"; t.append(d);
        prev = parts.slice(0, i + 1);
      }
      prev = parts.slice(0, -1);
      const f = h("button", "f", parts[parts.length - 1]);
      f.type = "button";
      f.style.paddingLeft = 8 + (parts.length - 1) * 14 + "px";
      f.dataset.path = p;
      f.dataset.ext = (/\.(\w+)$/.exec(p)?.[1] || "").toLowerCase();
      if (p === edPath) f.classList.add("on");
      if (drafts.has(p)) f.classList.add("dirty");
      f.onclick = () => { t.querySelectorAll(".f.on").forEach((x) => x.classList.remove("on")); f.classList.add("on"); outTab("files"); edOpen(true); fileClick(p); };
      t.append(f);
    }
    if (paths.length > 500) t.append(h("div", "none", `(+${paths.length - 500} more)`));
  }
  const dirty = () => edPath != null && !edRO && ta.value !== edBase;
  function paint() {
    hlRaf = 0;
    hl.innerHTML = highlight(edPath || "", ta.value) + "\n";
    const n = ta.value.split("\n").length;
    if (+gutter.dataset.n !== n) { gutter.dataset.n = n; gutter.textContent = Array.from({ length: n }, (_, i) => i + 1).join("\n"); }
  }
  function edState() {
    const d = dirty();
    if (edPath != null && !edRO) { if (d) drafts.set(edPath, ta.value); else drafts.delete(edPath); }
    $("code-tree").querySelector(`.f[data-path="${CSS.escape(edPath || "")}"]`)?.classList.toggle("dirty", d);
    $("ed-save").disabled = !d; $("ed-save").hidden = edRO;
    $("ed-revert").hidden = !d;
    const st = $("ed-state");
    if (st.dataset.flash && !d) return;
    st.textContent = edRO ? "read only" : d ? "unsaved" : "";
    st.className = d ? "dirty" : "";
  }
  // open a file in the editor: text, and whether it can be saved (label: why it is shown, e.g. a proposed file)
  function openFile(path, text, { readOnly = !host || !saveFile, label = null } = {}) {
    if (edPath != null && dirty()) drafts.set(edPath, ta.value);
    edPath = path; edRO = readOnly || path == null;
    edBase = text ?? "";
    ta.value = !edRO && drafts.has(path) ? drafts.get(path) : edBase;
    ta.readOnly = edRO;
    $("ed-path").textContent = label || path || "";
    $("ed-path").title = label || path || "";
    $("ed-empty").hidden = true; edBox.hidden = false; $("ed-bar").hidden = false;
    $("ed-state").removeAttribute("data-flash");
    paint(); edState();
    edBox.scrollTop = 0; edBox.scrollLeft = 0;
  }
  function viewFile(text, label = null) {
    if (text == null) {
      edPath = null; edBase = ""; ta.value = ""; edRO = true; cp.classList.remove("ed-open");
      edBox.hidden = true; $("ed-bar").hidden = true; $("ed-empty").hidden = false;
      return;
    }
    openFile(null, text, { readOnly: true, label: label || "" });
  }
  // the file changed underneath (the agent wrote it): take the new text unless there are unsaved edits
  function fileChanged(path, text) {
    if (path !== edPath || text == null) return;
    if (dirty()) { edBase = text; edState(); return; }
    if (ta.value === text) return;
    const top = edBox.scrollTop;
    edBase = text; ta.value = text; paint(); edState(); edBox.scrollTop = top;
  }
  async function save() {
    if (!dirty() || !saveFile) return;
    const path = edPath, text = ta.value, st = $("ed-state");
    $("ed-save").disabled = true; st.textContent = "saving…";
    try {
      await saveFile(path, text);
      if (edPath === path) { edBase = text; drafts.delete(path); st.dataset.flash = "1"; st.textContent = "saved"; st.className = "ok"; setTimeout(() => { st.removeAttribute("data-flash"); edState(); }, 1800); }
      edState();
      say(`saved ${path}`);
    } catch (e) { st.textContent = "not saved: " + e.message; st.className = "err"; $("ed-save").disabled = false; }
  }
  // keep the caret in view (the textarea is as big as its text; the box around it scrolls)
  function reveal() {
    const cs = getComputedStyle(ta), lh = parseFloat(cs.lineHeight) || 19, pt = parseFloat(cs.paddingTop) || 0;
    const before = ta.value.slice(0, ta.selectionEnd), line = before.split("\n").length - 1;
    const y = pt + line * lh;
    if (y < edBox.scrollTop) edBox.scrollTop = y - lh;
    else if (y + lh * 2 > edBox.scrollTop + edBox.clientHeight) edBox.scrollTop = y + lh * 2 - edBox.clientHeight;
  }
  const insert = (text) => { if (!document.execCommand("insertText", false, text)) { ta.setRangeText(text, ta.selectionStart, ta.selectionEnd, "end"); ta.dispatchEvent(new Event("input")); } };
  ta.addEventListener("input", () => { hlRaf ||= requestAnimationFrame(paint); edState(); reveal(); });
  ta.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") { e.preventDefault(); save(); return; }
    if (edRO) return;
    if (e.key === "Tab" && !e.shiftKey && !e.metaKey && !e.ctrlKey && !e.altKey) { e.preventDefault(); insert("  "); return; }
    if (e.key === "Enter" && !e.shiftKey && !e.metaKey && !e.ctrlKey && !e.isComposing) {
      const before = ta.value.slice(0, ta.selectionStart), line = before.slice(before.lastIndexOf("\n") + 1);
      const ind = /^[ \t]*/.exec(line)[0] + (/[{[(]\s*$/.test(line) ? "  " : "");
      e.preventDefault(); insert("\n" + ind); return;
    }
    if (e.key === "Escape") { ta.blur(); e.preventDefault(); }
  });
  $("ed-save").onclick = save;
  $("ed-revert").onclick = () => { if (edPath == null) return; drafts.delete(edPath); ta.value = edBase; paint(); edState(); ta.focus(); };
  // a long proposed file, from its approval card (host only): shown read-only in the editor
  function viewFull(path, text) {
    outTab("files");
    $("code-tree").querySelectorAll(".f.on").forEach((x) => x.classList.remove("on"));
    viewFile(text, `${path} · proposed, not written yet`);
    if (phone.matches) { edOpen(true); setTab("files"); } else $("files-panel").scrollIntoView({ block: "nearest" });
  }
  function outTab(name) {
    for (const b of $("code-out-tabs").querySelectorAll("button[data-tab]")) {
      b.classList.toggle("on", b.dataset.tab === name);
      b.setAttribute("aria-selected", String(b.dataset.tab === name));
      b.tabIndex = b.dataset.tab === name ? 0 : -1;
    }
    $("pv-panel").hidden = name !== "preview";
    $("files-panel").hidden = name !== "files";
    grow();
  }
  $("code-out-tabs").addEventListener("click", (e) => { const b = e.target.closest("button[data-tab]"); if (b) outTab(b.dataset.tab); });

  // ---------------- the output column's size: small until there is something to show (an app is
  // served, or a file is open), then it grows; dragging its edge sets it (kept per viewer)
  const cpane = $("code-pane"), grip = $("code-grip");
  function grow() { cpane.classList.toggle("out-big", ports.size > 0 || !$("files-panel").hidden); }
  try { const w = +localStorage.getItem("pooled-code-out"); if (w > 0) { cpane.style.setProperty("--out-w", w + "px"); cpane.classList.add("sized"); } } catch {}
  grip.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault(); grip.setPointerCapture(e.pointerId);
    const right = cpane.getBoundingClientRect().right, files = $("code-files").getBoundingClientRect().width;
    const max = cpane.getBoundingClientRect().width - files - 320;
    cpane.classList.add("sizing", "sized");
    const move = (ev) => cpane.style.setProperty("--out-w", Math.round(Math.max(300, Math.min(max, right - ev.clientX))) + "px");
    const up = () => {
      grip.removeEventListener("pointermove", move); grip.removeEventListener("pointerup", up); grip.removeEventListener("pointercancel", up);
      cpane.classList.remove("sizing");
      try { localStorage.setItem("pooled-code-out", parseInt(cpane.style.getPropertyValue("--out-w"), 10)); } catch {}
    };
    grip.addEventListener("pointermove", move); grip.addEventListener("pointerup", up); grip.addEventListener("pointercancel", up);
  });
  // double-click: back to the automatic size
  grip.addEventListener("dblclick", () => { cpane.classList.remove("sized"); cpane.style.removeProperty("--out-w"); try { localStorage.removeItem("pooled-code-out"); } catch {} });
  grip.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const cur = $("code-out").getBoundingClientRect().width + (e.key === "ArrowLeft" ? 40 : -40);
    cpane.classList.add("sized"); cpane.style.setProperty("--out-w", Math.max(300, Math.round(cur)) + "px");
    try { localStorage.setItem("pooled-code-out", Math.max(300, Math.round(cur))); } catch {}
  });
  // ---------------- preview: tabs per port, one mounted view each, the console strip
  const ports = new Map();   // port -> { tab, view (div), mount, rows: [], rev, path, state }
  let active = null, onClose = null, onReload = () => {}, onOpen = () => {};
  function portTab(port, { mount, closable }) {
    let P = ports.get(port);
    if (P) return P;
    const wrap = h("span", "pv-tabw");
    const tab = h("button", "pv-tab", ":" + port); tab.type = "button";
    tab.onclick = () => activate(port);
    wrap.append(tab);
    // its own button, so the keyboard reaches it and a click on the tab never stops the port
    if (closable) { const x = h("button", "pv-x"); x.innerHTML = '<svg width="10" height="10" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M3 3l8 8M11 3l-8 8"/></svg>'; x.type = "button"; x.title = `stop serving :${port}`; x.setAttribute("aria-label", x.title); x.onclick = () => onClose?.(port); wrap.append(x); }
    $("pv-tabs").append(wrap);
    const view = h("div", "pv-view"); view.hidden = true;
    $("pv-frame-wrap").append(view);
    P = { tab, wrap, view, rows: [], rev: 0, path: "", state: "", mount: null };
    ports.set(port, P);
    P.mount = mount(view, P);
    $("pv-empty").hidden = true;
    $("pv-console").hidden = false;
    grow();
    // a phone: the first app served this session opens Preview (the landing demo does the same); later ones badge it
    served();
    return P;
  }
  function dropPort(port) {
    const P = ports.get(port);
    if (!P) return;
    P.mount?.destroy(); P.wrap.remove(); P.view.remove();
    ports.delete(port);
    grow();
    if (active === port) { active = null; const next = ports.keys().next().value; if (next != null) activate(next); else { $("pv-empty").hidden = false; $("pv-console").hidden = true; bar(); renderConsole(); } }
  }
  function activate(port) {
    active = port;
    for (const [p, P] of ports) { P.view.hidden = p !== port; P.tab.classList.toggle("on", p === port); }
    bar(); renderConsole();
  }
  function bar() {
    const P = ports.get(active), a = $("pv-addr");
    a.replaceChildren();
    if (P) a.append(h("span", "host", "localhost"), `:${active}/${P.path || "index.html"}`);
    else a.textContent = "No port served";
    $("pv-open").hidden = !P || !P.rev;
    $("pv-state").textContent = P ? (P.state === "ready" ? `rev ${P.rev}` : P.state === "loading" ? "loading…" : P.state === "waiting" ? "Click to run" : P.state === "stopped" ? "stopped" : P.state === "hung" ? "hung" : "") : "";
  }
  function status(port, s) {
    const P = ports.get(port);
    if (!P) return;
    // a new rev, or the same one loading again (Reload, run again after a hang): the rows so far are the old load's
    if (s.rev && s.state === "loading" && (s.rev !== P.rev || P.state !== "loading")) {
      P.rows.forEach((r) => (r.old = true));
      P.rows.push({ sep: `rev ${s.rev}` + (P.rev ? " · reloaded" : "") });
    }
    if (s.rev && P.rev && s.rev > P.rev) newRev();
    if (s.rev) P.rev = s.rev;
    P.path = s.path || P.path; P.state = s.state;
    if (s.state === "ready") editWin.reloaded(port, P.rev);
    if (port === active) { bar(); renderConsole(); }
  }
  function logRow(port, e) {
    const P = ports.get(port);
    if (!P) return;
    P.rows.push(e);
    if (P.rows.length > CON_MAX) P.rows.splice(0, P.rows.length - CON_MAX);
    if (e.level === "error" && port === active) conOpen(true);
    if (port === active) renderConsole();
  }
  let conTimer = 0;
  function renderConsole() {
    if (conTimer) return;
    conTimer = requestAnimationFrame(() => {
      conTimer = 0;
      const P = ports.get(active), rows = $("pv-con-rows"), stick = rows.scrollHeight - rows.scrollTop - rows.clientHeight < 30;
      rows.replaceChildren();
      const cur = (P?.rows || []).filter((r) => !r.old && !r.sep);
      const errs = cur.filter((r) => r.level === "error").length, warns = cur.filter((r) => r.level === "warn").length, logs = cur.length - errs - warns;
      const c = $("pv-counts");
      c.replaceChildren();
      // no '0 logs' next to errors or warnings: one line in the header on a phone
      const sh = phone.matches;
      const parts = [errs && h("b", "e", plural(errs, sh ? "err" : "error")), warns && h("b", "w", plural(warns, sh ? "warn" : "warning")), (logs || !(errs || warns)) && plural(logs, "log")].filter(Boolean);
      parts.forEach((x, i) => c.append(...(i ? [" · ", x] : [x])));
      c.dataset.errors = String(errs);
      $("pv-to-agent").hidden = !drive || !errs;   // only when there is something to fix
      for (const r of P?.rows || []) {
        if (r.sep) { rows.append(h("div", "pv-row rev", r.sep)); continue; }
        const row = h("div", `pv-row ${r.level}${r.old ? " old" : ""}`);
        const at = r.src ? `${r.src}${r.line ? ":" + r.line : ""}` : "";
        if (at) row.append(h("span", "src", at));
        row.append(h("span", null, r.text));
        rows.append(row);
      }
      if (stick) rows.scrollTop = rows.scrollHeight;
    });
  }
  function conOpen(open) {
    $("pv-console").classList.toggle("closed", !open);
    $("pv-con-toggle").setAttribute("aria-expanded", String(open));
  }
  $("pv-con-toggle").onclick = () => conOpen($("pv-console").classList.contains("closed"));
  $("pv-clear").onclick = () => { const P = ports.get(active); if (P) { P.rows = []; renderConsole(); } };
  $("pv-reload").onclick = () => { if (active != null) onReload(active); };
  $("pv-open").onclick = () => { if (active != null) onOpen(active, ports.get(active)?.path || null); };

  // ---------------- host vs peer chrome. Anyone who can drive gets the project bar, the prompt
  // and the bar under the log; only the host opens a folder on disk or saves in the editor.
  function setHost(v, { canDrive = v } = {}) {
    host = !!v; drive = !!canDrive;
    $("code-project").hidden = !drive;
    $("code-open").hidden = !host || $("code-open").dataset.can !== "1";
    $("code-row").hidden = !drive;
    $("code-bar").hidden = !drive;
    $("pv-to-agent").hidden = !drive || !(+$("pv-counts").dataset.errors > 0);
    if (edPath != null) { edRO = !host || !saveFile; ta.readOnly = edRO; edState(); }
  }
  // a line above the log about where the agent runs (empty: hidden)
  function driverNote(text) { const el = $("code-driver"); el.textContent = text || ""; el.hidden = !text; }
  setHost(false);
  viewFile(null);
  roles();

  return {
    show, poke, apply, ask, cancelAsk, clear, placeholder, setHost, driverNote, tree, viewFile, openFile, fileChanged, outTab,
    get mode() { return mode; },
    get activePort() { return active; },
    get openPath() { return edPath; },
    onFile(fn) { fileClick = fn; },
    onWaitText(fn) { waitText = fn; },
    onSave(fn) { saveFile = fn; },
    onClosePort(fn) { onClose = fn; },
    onReload(fn) { onReload = fn; },
    onOpen(fn) { onOpen = fn; },
    portTab, dropPort, activate, status, logRow, ports,
    // phones: show one of the tabs (Agent, Preview, Files); wider screens show them all
    tab(t) { if (phone.matches) setTab(t); },
    // how much of the context window the agent's conversation uses, as a percentage and a ring
    ctx(used, max) {
      const el = $("code-ctx");
      if (!used || !max) { el.replaceChildren(); el.removeAttribute("title"); return; }
      const pct = Math.min(100, Math.max(1, Math.round(used / max * 100)));
      el.innerHTML = `<i style="--p:${pct}" aria-hidden="true"></i>${pct}% of context`;
      el.title = `${used.toLocaleString("en-US")} of ${max.toLocaleString("en-US")} tokens`;
      el.classList.toggle("warn", used > max * 0.8);
    },
    // while a run goes, Send queues the next request; Stop shows for whoever may stop it
    running(on, canStop = on) {
      busy(on);
      $("code-send").textContent = on ? "Queue" : "Send";
      $("code-send").title = on ? "Runs after the current request" : "";
      $("code-stop").hidden = !(on && canStop);
      const pr = $("code-prompt");
      pr.dataset.ph ||= pr.placeholder;
      pr.placeholder = on && !PHONE.matches ? "Queue another request" : pr.dataset.ph;
    },
  };
}
