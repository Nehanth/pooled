// room/card.js: the 1200x630 room card, drawn on a recording fake canvas (no DOM, no GPU).
// Text is measured as 10 px per character, so truncation is exact.
import { drawCard } from "../../room/card.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

// a 2d context that records every fillText / stroke; `round` false drops roundRect (older Safari)
function fakeCanvas({ round = true, spacing = true } = {}) {
  const log = [];
  const g = {
    fillStyle: "", strokeStyle: "", lineWidth: 1, font: "", textBaseline: "",
    measureText: (t) => ({ width: String(t).length * 10 }),
    fillText(text, x, y) { log.push({ op: "text", text, x, y, fill: this.fillStyle }); },
    fillRect() {}, beginPath() {}, arc() {}, fill() {}, moveTo(x, y) { log.push({ op: "moveTo", x, y }); }, lineTo() {},
    rect(x, y, w, h) { log.push({ op: "rect", x, y, w, h }); },
    stroke() { log.push({ op: "stroke", style: this.strokeStyle, width: this.lineWidth }); },
  };
  if (round) g.roundRect = (x, y, w, h) => log.push({ op: "roundRect", x, y, w, h });
  if (spacing) g.letterSpacing = "0px";
  const canvas = { width: 0, height: 0, getContext: (k) => (k === "2d" ? g : null) };
  return { canvas, g, log, texts: () => log.filter((e) => e.op === "text").map((e) => e.text) };
}
const node = (name, layers, host = false) => ({ name, layers, host });

Deno.test("card: size, header, and the tok/s number (one decimal, '-' when unknown)", () => {
  const f = fakeCanvas();
  const out = drawCard(f.canvas, { model: "Qwen3.8 27B", code: "ABCD", date: "Sep 29", tps: 12.345, nodes: [node("Mac", "1-40", true)] });
  ok(out === f.canvas, "returns the canvas");
  eq([f.canvas.width, f.canvas.height], [1200, 630]);
  const t = f.texts();
  for (const s of ["pooled", "Room", "ABCD", " · Sep 29", "12.3", "tok/s"]) ok(t.includes(s), `missing ${s}: ${t}`);
  eq(f.g.letterSpacing, "0px", "letter spacing is put back");
  const g2 = fakeCanvas();
  drawCard(g2.canvas, { nodes: [node("a", "")] });
  ok(g2.texts().includes("-"), "no tps -> -");
  ok(g2.texts().includes(""), "no code draws an empty code");
  ok(!g2.texts().some((s) => s.startsWith(" · ")), "no date, no separator");
  ok(g2.texts().includes("a model on 1 device, in browser tabs"), g2.texts().join("|"));
});

Deno.test("card: the stats line, table-driven", () => {
  const cases = [
    [{}, "no server did any of the thinking"],
    [{ acc: 0.734 }, "73% of drafts accepted · no server did any of the thinking"],
    [{ acc: 0 }, "0% of drafts accepted · no server did any of the thinking"],
    [{ lap: 180 }, "180 ms per word, round the room · no server did any of the thinking"],
    [{ acc: 1, lap: 95 }, "100% of drafts accepted · 95 ms per word, round the room · no server did any of the thinking"],
    [{ lap: 0 }, "no server did any of the thinking"],   // 0 ms is no measurement
  ];
  for (const [extra, want] of cases) {
    const f = fakeCanvas();
    drawCard(f.canvas, { model: "m", nodes: [node("a", "1-2")], ...extra });
    ok(f.texts().includes(want), `${JSON.stringify(extra)}: ${f.texts().join(" | ")}`);
  }
});

Deno.test("card: long text is cut with an ellipsis to fit the card", () => {
  const f = fakeCanvas();
  const model = "M".repeat(300);
  drawCard(f.canvas, { model, nodes: [node("a very long device name indeed", "1-20"), node("b", "21-40")] });
  const line = f.texts().find((s) => s.startsWith("MMM"));
  ok(line.endsWith("…"), line);
  ok(line.length * 10 <= 1200 - 128, `fits: ${line.length * 10}`);
  // two devices: each box is 250 wide, the name gets w - 50
  const name = f.texts().find((s) => s.startsWith("a very"));
  ok(name.endsWith("…") && name.length * 10 <= 200, name);
});

Deno.test("card: the chain, one box per device, the host outlined and labelled, arrows between", () => {
  for (const n of [1, 2, 5]) {
    const f = fakeCanvas();
    const nodes = Array.from({ length: n }, (_, i) => node("d" + i, i ? `${i * 10 + 1}-${i * 10 + 10}` : "1-10", i === 0));
    drawCard(f.canvas, { model: "m", nodes });
    const boxes = f.log.filter((e) => e.op === "roundRect" && e.y === 440);
    eq(boxes.length, 2 * n, "fill and outline per box");
    const w = Math.min(250, (1200 - 128 - 26 * (n - 1)) / n);
    eq(boxes[0].w, w);
    const arrows = f.log.filter((e) => e.op === "moveTo");
    eq(arrows.length, n - 1, "an arrow between neighbours");
    const outlines = f.log.filter((e) => e.op === "stroke" && e.width === 2).map((e) => e.style);
    eq(outlines[0], "#14161D", "the host box is outlined in text colour");
    ok(outlines.slice(1).every((s) => s === "#E4E2DA"), "the others in the border colour");
    const host = f.texts().find((s) => s.startsWith("embed · "));
    ok(host === "embed · layers 1-10 · head" || (host.endsWith("…") && host.length * 10 <= w - 36), host);
    if (n > 1) ok(f.texts().includes("layers 11-20"), "a worker lists its layers only");
    ok(f.texts().includes(`m on ${n} device${n > 1 ? "s" : ""}, in browser tabs`));
  }
});

Deno.test("card: without roundRect or letterSpacing (older browsers) it still draws", () => {
  const f = fakeCanvas({ round: false, spacing: false });
  drawCard(f.canvas, { model: "m", code: "WXYZ", tps: 3, nodes: [node("a", "1-4", true), node("b", "")] });
  eq(f.log.filter((e) => e.op === "rect").length, 4, "plain rects for the two boxes");
  ok(!("letterSpacing" in f.g), "never sets a property the context lacks");
  ok(f.texts().includes("3.0"));
  ok(f.texts().includes(""), "a device with no layers and not the host has an empty line");
});
