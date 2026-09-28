// Minimal markdown renderer for the chat transcript (escapes first; no raw HTML).
import { splitThink } from "./conversation.js";

export function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;"); }   // safe in text and in quoted attributes (peer names land in title=/data-name=)

export function md(src) {
  // code fences first, set aside: their text shows as written (no inline or line rules inside), without
  // the language tag; each comes back as a line of its own
  const pres = [];
  let s = esc(src.replace(/\u0000/g, "")).replace(/```([\s\S]*?)```/g, (_, c) => {
    pres.push(`<pre>${c.replace(/^[\w+#.-]*[ \t]*\n/, "").replace(/^\n+|\s+$/g, "")}</pre>`);
    return `\n\u0000${pres.length - 1}\u0000\n`;
  });
  s = s.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>").replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<i>$2</i>");
  const lines = s.split("\n");
  let out = "", list = null, para = [];
  const flushP = () => { if (para.length) { out += `<p>${para.join("<br>")}</p>`; para = []; } };
  const flushL = () => { if (list) { out += `</${list}>`; list = null; } };
  for (const ln of lines) {
    const h = ln.match(/^(#{1,4})\s+(.*)$/);
    const ul = ln.match(/^\s*[-*]\s+(.*)$/);
    const ol = ln.match(/^\s*\d+[.)]\s+(.*)$/);
    const pre = /^\u0000(\d+)\u0000$/.exec(ln);
    if (pre) { flushP(); flushL(); out += pres[+pre[1]]; continue; }
    if (h) { flushP(); flushL(); const lv = Math.min(4, h[1].length + 1); out += `<h${lv}>${h[2]}</h${lv}>`; continue; }
    if (ul) { flushP(); if (list !== "ul") { flushL(); list = "ul"; out += "<ul>"; } out += `<li>${ul[1]}</li>`; continue; }
    if (ol) { flushP(); if (list !== "ol") { flushL(); list = "ol"; out += "<ol>"; } out += `<li>${ol[1]}</li>`; continue; }
    if (!ln.trim()) { flushP(); flushL(); continue; }
    flushL(); para.push(ln);
  }
  flushP(); flushL();
  return out;
}

// An answer as chat HTML: a think block (thinking mode) folds into a <details>, open while it
// is still streaming, the rest is markdown.
export function mdChat(raw) {
  const { think, answer, open } = splitThink(raw);
  if (think === null) return md(raw);
  const words = think ? think.split(/\s+/).length : 0;
  return `<details class="think"${open ? " open" : ""}><summary>${open ? "thinking\u2026" : `thought for ${words} words`}</summary>${md(think)}</details>` + md(answer);
}
