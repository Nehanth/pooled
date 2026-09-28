/* A small syntax highlighter for the demo's code (HTML, CSS, JS), one line at a time: the same token
   classes and colours as the room's editor (room/code-ui.js highlight, p2p.html #ed-hl .t-*). It only
   wraps spans around escaped text. */
(() => {
  "use strict";
  const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const KW = new Set(("const let var function return if else for while do break continue new class extends import export from default async await try catch "
    + "finally throw typeof instanceof in of switch case this null undefined true false yield delete void static get set super").split(" "));
  const JS_RE = /(\/\/.*|\/\*.*?(?:\*\/|$))|(`(?:\\.|[^`\\])*`?|"(?:\\.|[^"\\])*"?|'(?:\\.|[^'\\])*'?)|\b(0[xX][\da-fA-F]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b|([A-Za-z_$][\w$]*)(\s*\()?/g;
  const js = src => {
    let out = "", at = 0;
    for (const m of src.matchAll(JS_RE)) {
      out += esc(src.slice(at, m.index)); at = m.index + m[0].length;
      if (m[1]) out += `<span class="t-c">${esc(m[1])}</span>`;
      else if (m[2]) out += `<span class="t-s">${esc(m[2])}</span>`;
      else if (m[3]) out += `<span class="t-n">${esc(m[3])}</span>`;
      else if (KW.has(m[4])) out += `<span class="t-k">${m[4]}</span>${esc(m[5] || "")}`;
      else if (m[5]) out += `<span class="t-f">${esc(m[4])}</span>${esc(m[5])}`;
      else out += esc(m[4]);
    }
    return out + esc(src.slice(at));
  };
  const CSS_RE = /(\/\*.*?(?:\*\/|$))|("(?:\\.|[^"\\])*"?|'(?:\\.|[^'\\])*'?)|(#[\da-fA-F]{3,8}\b)|(@[\w-]+)|([\w-]+)(?=\s*:[^;{}]*(?:[;}]|$))|(-?\b\d+(?:\.\d+)?(?:px|em|rem|%|vh|vw|s|ms|deg|fr|ch)?)/g;
  const css = src => {
    let out = "", at = 0;
    for (const m of src.matchAll(CSS_RE)) {
      out += esc(src.slice(at, m.index)); at = m.index + m[0].length;
      const cls = m[1] ? "t-c" : m[2] ? "t-s" : m[3] ? "t-n" : m[4] ? "t-k" : m[5] ? "t-p" : "t-n";
      out += `<span class="${cls}">${esc(m[0])}</span>`;
    }
    return out + esc(src.slice(at));
  };
  const html = src => {
    let out = "", i = 0;
    while (i < src.length) {
      const lt = src.indexOf("<", i);
      if (lt < 0) { out += esc(src.slice(i)); break; }
      out += esc(src.slice(i, lt));
      if (src.startsWith("<!--", lt)) { const e = src.indexOf("-->", lt + 4), end = e < 0 ? src.length : e + 3; out += `<span class="t-c">${esc(src.slice(lt, end))}</span>`; i = end; continue; }
      const tm = /^<\/?([A-Za-z][\w:-]*|!doctype)/i.exec(src.slice(lt, lt + 40));
      if (!tm) { out += "&lt;"; i = lt + 1; continue; }
      out += `<span class="t-t">${esc(tm[0])}</span>`;
      const AR = /\s+|([^\s=>\/]+)(\s*=\s*("[^"]*"?|'[^']*'?|[^\s>]+))?|(\/?>)|([\s\S])/y;
      let j = lt + tm[0].length, m;
      AR.lastIndex = j;
      while (j < src.length && (m = AR.exec(src))) {
        j = AR.lastIndex;
        if (m[4]) { out += `<span class="t-t">${esc(m[4])}</span>`; break; }
        if (m[1]) out += `<span class="t-a">${esc(m[1])}</span>` + (m[2] ? esc(m[2].slice(0, m[2].length - m[3].length)) + `<span class="t-s">${esc(m[3])}</span>` : "");
        else out += esc(m[0]);
      }
      i = j;
    }
    return out;
  };
  const LANG = { js, css, html };
  // one line of a file, as HTML
  window.PooledHL = (path, line) => { const f = LANG[(/\.(\w+)$/.exec(path || "")?.[1] || "").toLowerCase()]; return f ? f(String(line)) : esc(String(line)); };
})();
