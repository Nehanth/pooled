// room/transport.js: slicing and reassembly are byte-exact, tolerate reordering and duplicates,
// and keep every send under SLICE_BYTES.
import { makeLink, sendFrame, SLICE_BYTES } from "../../room/transport.js";

function fakeChannels(link, n, sink) {
  for (let i = 0; i < n; i++) link.chans.push({ readyState: "open", send: (buf) => sink.push({ i, buf }) });
}
// receive() is module-private: attach a stub peer connection and drive its onmessage
import { attachWire } from "../../room/transport.js";
// one receiving channel per sending channel (each ordered, like the real ones); deliver(buf, i)
function receiver(onFrame) {
  const link = makeLink(), handlers = [];
  const chan = (i) => {
    while (handlers.length <= i) {
      const pc = { createDataChannel: () => ({ set onmessage(f) { handlers.push(f); }, set onclose(_) {}, readyState: "open" }) };
      attachWire(link, { peerConnection: pc }, onFrame);
    }
    return handlers[i];
  };
  return (buf, i = 0) => chan(i)({ data: buf });
}

const dim = 5120;
const shapes = [
  { t: "ai-hidden", pos: 17, n: 1, cols: 1 },
  { t: "ai-hidden-b", basePos: 240, n: 16, cols: 16 },
  { t: "ai-hiddenret-b", basePos: 5, n: 6, spec: 1, cols: 6 },
];
for (const sh of shapes) {
  const data = new Uint16Array(dim * sh.cols); for (let i = 0; i < data.length; i++) data[i] = (i * 2654435761) >>> 16;
  Deno.test(`transport round trip ${sh.t} x${sh.cols}`, () => {
    const link = makeLink({ dup: 0 }), out = []; fakeChannels(link, 3, out);
    if (!sendFrame(link, { ...sh, data })) throw new Error("send refused");
    for (const { buf } of out) if (buf.byteLength > SLICE_BYTES) throw new Error("slice too big: " + buf.byteLength);
    const expectSlices = Math.ceil(data.byteLength / (SLICE_BYTES - 32));
    if (out.length !== expectSlices) throw new Error(`expected ${expectSlices} slices, got ${out.length}`);
    // stripes: consecutive slices land on different channels
    if (out.length > 1 && out[0].i === out[1].i) throw new Error("slices not striped");
    let got = null; const deliver = receiver((m) => { got = m; });
    // deliver reversed, with a duplicate in the middle
    const order = [...out].reverse(); order.splice(1, 0, out[Math.floor(out.length / 2)]);
    for (const { buf, i } of order) deliver(buf, i);
    if (!got) throw new Error("frame not reassembled");
    if (got.t !== sh.t) throw new Error("kind mismatch " + got.t);
    if ((sh.pos ?? sh.basePos) !== (got.pos ?? got.basePos)) throw new Error("pos mismatch");
    if (got.n !== sh.n || !!got.spec !== !!sh.spec) throw new Error("meta mismatch");
    if (got.data.length !== data.length) throw new Error("length mismatch");
    for (let i = 0; i < data.length; i++) if (got.data[i] !== data[i]) throw new Error("byte mismatch at " + i);
  });
}
Deno.test("transport refuses when no channel is open", () => {
  const link = makeLink(); link.chans.push({ readyState: "connecting", send() {} });
  if (sendFrame(link, { t: "ai-hidden", pos: 0, data: new Uint16Array(8) })) throw new Error("should refuse");
});
Deno.test("transport: frames that complete out of order are delivered in send order", () => {
  const link = makeLink(), out = []; fakeChannels(link, 4, out);
  const mk = (pos) => { const d = new Uint16Array(5120 * 4); d.fill(pos); return { t: "ai-hidden-b", basePos: pos, n: 4, data: d }; };
  sendFrame(link, mk(0)); const first = out.length;
  sendFrame(link, mk(4));
  sendFrame(link, { t: "ai-hidden-b", basePos: 8, n: 4, rb: 2, reset: 1, data: new Uint16Array(5120 * 4) });
  const got = []; const deliver = receiver((m) => got.push(m));
  // every slice of frames 2 and 3 first, then frame 1's. A real channel is ordered, so frames 2
  // and 3 arrive here on four other channels (as if striped over other associations) while frame
  // 1 is late on the first four
  for (const { buf, i } of out.slice(first)) deliver(buf, i + 4);
  if (got.length) throw new Error("delivered before frame 1 completed");
  for (const { buf, i } of out.slice(0, first)) deliver(buf, i);
  const order = got.map((m) => m.basePos).join(",");
  if (order !== "0,4,8") throw new Error("order " + order);
  if (got[2].rb !== 2 || got[2].reset !== 1 || got[0].rb !== undefined) throw new Error("flags lost");
});
Deno.test("transport: a frame an ordered channel has passed is lost; later frames flow at once and it is never delivered late", () => {
  const link = makeLink(), out = []; fakeChannels(link, 1, out);
  const mk = (pos) => ({ t: "ai-hidden", pos, data: new Uint16Array(8) });
  sendFrame(link, mk(1)); sendFrame(link, mk(2)); sendFrame(link, mk(3));
  const got = []; const deliver = receiver((m) => got.push(m.pos));
  deliver(out[1].buf); deliver(out[2].buf);   // frame 1 missing, and the only channel already showed frame 2
  if (got.join(",") !== "2,3") throw new Error("not skipped at once: " + got.join(","));
  deliver(out[0].buf);                        // frame 1 turning up anyway (a misbehaving sender)
  if (got.join(",") !== "2,3") throw new Error("late frame delivered: " + got.join(","));
});
Deno.test("transport carries checkpoint control (save / load / drop) with the frame", () => {
  const cases = [
    { sv: 3 }, { ld: 65534 }, { sv: 7, dp: [5] }, { sv: 9, ld: 2, dp: [1, 4] }, { dp: [0xffff] }, { dp: [1, 2, 3, 0xffff] }, {},
  ];
  for (const c of cases) {
    const link = makeLink(), out = []; fakeChannels(link, 2, out);
    sendFrame(link, { t: "ai-hidden-b", basePos: 3, n: 2, reset: 1, ...c, data: new Uint16Array(5120 * 2) });
    let got = null; const deliver = receiver((m) => { got = m; });
    for (const { buf } of out) deliver(buf);
    const want = { sv: c.sv, ld: c.ld, dp: c.dp && (c.dp.includes(0xffff) ? [0xffff] : c.dp) };
    for (const k of ["sv", "ld", "dp"]) if (JSON.stringify(got[k]) !== JSON.stringify(want[k])) throw new Error(`${JSON.stringify(c)}: ${k} ${JSON.stringify(got[k])}`);
    if (!got.reset) throw new Error("flags lost");
  }
  for (const bad of [{ sv: 70000 }, { ld: -1 }, { dp: [1, 2, 3] }]) {
    const link = makeLink(); fakeChannels(link, 1, []);
    let threw = false; try { sendFrame(link, { t: "ai-hidden", pos: 0, ...bad, data: new Uint16Array(2) }); } catch { threw = true; }
    if (!threw) throw new Error("accepted " + JSON.stringify(bad));
  }
});
