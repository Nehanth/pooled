// The room card: a 1200x630 PNG of what this room just did (model, devices and their layers,
// tok/s, draft acceptance) to download or share. Canvas only, no dependencies.

const C = { bg: "#F6F5F1", dot: "#E4E2DA", panel: "#FBFAF7", border: "#E4E2DA", text: "#14161D", muted: "#5E616B", accent: "#2A45E0" };
const SANS = '"Geist", -apple-system, BlinkMacSystemFont, sans-serif';
const MONO = '"Geist Mono", ui-monospace, monospace';

// the mark's nine dots (site/logo/mark.svg), in a 24-unit box; the last one is blue
const DOTS = [[3.4, 3.4, 1.8], [10.2, 3.4, 1.99], [18.5, 3.4, 2.38], [3.4, 10.2, 1.99], [10.2, 10.2, 2.38], [18.5, 10.2, 2.94], [3.4, 18.5, 2.38], [10.2, 18.5, 2.94], [18.5, 18.5, 3.9]];
function mark(g, x, y, size) {
  const k = size / 24;
  DOTS.forEach(([cx, cy, r], i) => { g.fillStyle = i === 8 ? C.accent : C.text; g.beginPath(); g.arc(x + cx * k, y + cy * k, r * k, 0, 7); g.fill(); });
}
function roundRect(g, x, y, w, h, r) { g.beginPath(); g.roundRect ? g.roundRect(x, y, w, h, r) : g.rect(x, y, w, h); }
function fit(g, text, max) { let t = String(text); while (t.length > 1 && g.measureText(t).width > max) t = t.slice(0, -2) + "…"; return t; }

// info: { model, code, nodes: [{ name, layers, host }], tps, acc, lap, date }
export function drawCard(canvas, info) {
  const W = 1200, H = 630;
  canvas.width = W; canvas.height = H;
  const g = canvas.getContext("2d");
  g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
  g.fillStyle = C.dot;
  for (let y = 13; y < H; y += 26) for (let x = 13; x < W; x += 26) { g.beginPath(); g.arc(x, y, 1.2, 0, 7); g.fill(); }
  // header: the dots mark and the wordmark, like the site's, then the room and the day
  mark(g, 64, 62, 36);
  g.fillStyle = C.text; g.font = `500 34px ${SANS}`; g.textBaseline = "alphabetic";
  if ("letterSpacing" in g) g.letterSpacing = "-1.2px";
  g.fillText("Duga", 114, 92);
  if ("letterSpacing" in g) g.letterSpacing = "0px";
  g.fillStyle = C.muted; g.font = `400 16px ${SANS}`;
  g.fillText("Room", 64, 130);
  let x = 64 + g.measureText("Room ").width;
  g.fillStyle = C.text; g.font = `500 16px ${MONO}`;
  if ("letterSpacing" in g) g.letterSpacing = "2px";
  g.fillText(info.code || "", x, 130);
  x += g.measureText(info.code || "").width;
  if ("letterSpacing" in g) g.letterSpacing = "0px";
  g.fillStyle = C.muted; g.font = `400 16px ${MONO}`;
  if (info.date) g.fillText(` · ${info.date}`, x, 130);
  // the number
  g.fillStyle = C.text; g.font = `700 150px ${SANS}`;
  const tps = info.tps ? info.tps.toFixed(1) : "-";
  g.fillText(tps, 60, 300);
  const tw = g.measureText(tps).width;
  g.fillStyle = C.accent; g.font = `500 34px ${MONO}`; g.fillText("tok/s", 76 + tw, 300);
  g.fillStyle = C.text; g.font = `500 30px ${SANS}`;
  g.fillText(fit(g, `${info.model || "a model"} on ${info.nodes.length} device${info.nodes.length > 1 ? "s" : ""}, in browser tabs`, W - 128), 64, 356);
  g.fillStyle = C.muted; g.font = `400 18px ${MONO}`;
  const bits = [];
  if (info.acc != null) bits.push(`${Math.round(info.acc * 100)}% of drafts accepted`);
  if (info.lap) bits.push(`${info.lap} ms per word, round the room`);
  bits.push("no server did any of the thinking");
  g.fillText(fit(g, bits.join(" · "), W - 128), 64, 394);
  // the chain
  const n = info.nodes.length, gap = 26, y = 440, h = 92;
  const w = Math.min(250, (W - 128 - gap * (n - 1)) / n);
  info.nodes.forEach((d, i) => {
    const x = 64 + i * (w + gap);
    g.fillStyle = C.panel; roundRect(g, x, y, w, h, 16); g.fill();
    g.strokeStyle = d.host ? C.text : C.border; g.lineWidth = 2; roundRect(g, x, y, w, h, 16); g.stroke();
    g.fillStyle = C.accent; g.beginPath(); g.arc(x + 22, y + 32, 6, 0, 7); g.fill();
    g.fillStyle = C.text; g.font = `500 20px ${SANS}`; g.fillText(fit(g, d.name, w - 50), x + 38, y + 39);
    g.fillStyle = C.muted; g.font = `400 14px ${MONO}`;
    g.fillText(fit(g, `${d.host ? "embed · " : ""}${d.layers ? "layers " + d.layers : ""}${d.host ? " · head" : ""}`, w - 36), x + 18, y + 72);
    if (i < n - 1) { g.strokeStyle = C.accent; g.lineWidth = 3; g.beginPath(); g.moveTo(x + w + 4, y + h / 2); g.lineTo(x + w + gap - 4, y + h / 2); g.stroke(); }
  });
  g.fillStyle = C.muted; g.font = `400 16px ${MONO}`;
  g.fillText("one AI model across the devices in a room, in browser tabs · duga.app", 64, H - 34);
  return canvas;
}
