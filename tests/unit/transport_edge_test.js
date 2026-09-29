// room/transport.js edge cases: the real 5 s gap timer (driven by a fake clock), re-arming on
// progress, header layout, control bytes mixed together, malformed and adversarial slices, the
// 30 s cleanup of half-received frames, wireReady and striping over channels that close.
// No GPU, no network: channels are stubs and the clock is fake.
import {
  makeLink, sendFrame, attachWire, wireReady, packFlags, unpackFlags, packCkpt, unpackCkpt,
  SLICE_BYTES, WIRE_ID, DROP_ALL,
} from "../../room/transport.js";

const HDR = 32, PER = SLICE_BYTES - HDR, GAP_MS = 5000;
function ok(c, m) { if (!c) throw new Error(m || "assertion failed"); }
function eq(a, b, m) { const x = JSON.stringify(a), y = JSON.stringify(b); if (x !== y) throw new Error(`${m ? m + ": " : ""}${x} !== ${y}`); }
function throws(f, m) { let t = false; try { f(); } catch { t = true; } if (!t) throw new Error("did not throw: " + m); }

// Fake setTimeout/clearTimeout/performance.now. transport.js looks them up at call time.
function withClock(fn) {
  const saved = { st: globalThis.setTimeout, ct: globalThis.clearTimeout, pn: performance.now };
  let now = 0, seq = 0; const timers = [];
  globalThis.setTimeout = (f, ms) => { const id = ++seq; timers.push({ id, at: now + (ms || 0), f }); return id; };
  globalThis.clearTimeout = (id) => { const i = timers.findIndex((t) => t.id === id); if (i >= 0) timers.splice(i, 1); };
  performance.now = () => now;
  const clock = {
    get now() { return now; },
    pending: () => timers.length,
    advance(ms) {
      const end = now + ms;
      for (;;) {
        timers.sort((a, b) => a.at - b.at || a.id - b.id);
        const t = timers[0];
        if (!t || t.at > end) break;
        timers.shift(); now = t.at; t.f();
      }
      now = end;
    },
  };
  try { return fn(clock); } finally {
    globalThis.setTimeout = saved.st; globalThis.clearTimeout = saved.ct; performance.now = saved.pn;
  }
}

function sender(n = 1) {
  const link = makeLink(), out = [];
  for (let i = 0; i < n; i++) link.chans.push({ readyState: "open", send: (buf) => out.push({ i, buf }) });
  return { link, out };
}
// a receiving link driven through attachWire's onmessage (receive() is module-private)
function receiver(onFrame, opts) {
  const link = makeLink(); let handler = null, onclose = null, cfg = null;
  const ch = { set onmessage(f) { handler = f; }, set onclose(f) { onclose = f; }, readyState: "open" };
  // attachWire may also open the keep-alive channel (swarm-ka); only the wire channel is under test here.
  const other = { readyState: "open", send() {} };
  attachWire(link, { peerConnection: { createDataChannel: (label, c) => { if (label !== "swarm-wire") return other; cfg = { label, ...c }; return ch; } } }, onFrame, opts);
  return { link, deliver: (buf) => handler({ data: buf }), close: () => onclose(), cfg: () => cfg, ch };
}
const frame = (pos, words = 8, extra = {}) => ({ t: "ai-hidden", pos, data: new Uint16Array(words).fill(pos & 0xffff), ...extra });
const hdr = (buf) => { const dv = new DataView(buf); return { magic: dv.getUint16(0), kind: dv.getUint8(2), flags: dv.getUint8(3), id: dv.getUint32(4), pos: dv.getUint32(8), n: dv.getUint16(12), k: dv.getUint16(14), nSlices: dv.getUint16(16), total: dv.getUint32(20) }; };
// forge a slice with arbitrary header fields
function forge({ id = 1, kind = 0, flags = 0, pos = 0, n = 1, k = 0, nSlices = 1, total, payload = new Uint8Array(0), magic = 0x5357 }) {
  const buf = new ArrayBuffer(HDR + payload.length), dv = new DataView(buf);
  dv.setUint16(0, magic); dv.setUint8(2, kind); dv.setUint8(3, flags); dv.setUint32(4, id); dv.setUint32(8, pos);
  dv.setUint16(12, n); dv.setUint16(14, k); dv.setUint16(16, nSlices); dv.setUint32(20, total ?? payload.length);
  new Uint8Array(buf, HDR).set(payload);
  return buf;
}

// ---------------------------------------------------------------- gap skip with the real timer

Deno.test("transport gap: a missing frame holds later ones for exactly 5 s, then they flow and the late one is dropped", () => withClock((clock) => {
  const { link, out } = sender(); for (let p = 1; p <= 4; p++) sendFrame(link, frame(p));
  const got = []; const r = receiver((m) => got.push(m.pos));
  r.deliver(out[1].buf); r.deliver(out[2].buf);            // frame 1 lost
  eq(got, [], "nothing before the gap fills");
  eq(clock.pending(), 1, "one gap timer armed");
  clock.advance(GAP_MS - 1); eq(got, [], "still held at 4999 ms");
  clock.advance(1); eq(got, [2, 3], "skipped at 5000 ms");
  eq(clock.pending(), 0, "nothing left to wait for, no timer");
  eq(r.link.expect, 4);
  r.deliver(out[0].buf); eq(got, [2, 3], "late frame 1 dropped");
  r.deliver(out[3].buf); eq(got, [2, 3, 4], "the frame after the skip is delivered at once");
  eq(clock.pending(), 0); eq(r.link.done.size, 0);
}));

Deno.test("transport gap: the timer re-arms when delivery made progress instead of skipping", () => withClock((clock) => {
  const { link, out } = sender(); for (let p = 1; p <= 4; p++) sendFrame(link, frame(p));
  const got = []; const r = receiver((m) => got.push(m.pos));
  r.deliver(out[1].buf); r.deliver(out[3].buf);            // waiting for 1 (and 3)
  clock.advance(3000); r.deliver(out[0].buf);              // 1 arrives: 1, 2 flow; now waiting for 3
  eq(got, [1, 2]);
  clock.advance(2000);                                      // first timer fires at t=5000: progress seen, re-arm
  eq(got, [1, 2], "no skip when the awaited frame arrived meanwhile");
  eq(clock.pending(), 1, "re-armed for the next gap");
  clock.advance(GAP_MS - 1); eq(got, [1, 2], "held until 5 s after the re-arm");
  clock.advance(1); eq(got, [1, 2, 4], "frame 3 skipped at t=10000");
  r.deliver(out[2].buf); eq(got, [1, 2, 4], "late 3 dropped");
}));

Deno.test("transport gap: a gap that fills in time leaves a harmless timer and no double delivery", () => withClock((clock) => {
  const { link, out } = sender(); for (let p = 1; p <= 3; p++) sendFrame(link, frame(p));
  const got = []; const r = receiver((m) => got.push(m.pos));
  r.deliver(out[2].buf); r.deliver(out[1].buf); r.deliver(out[0].buf);
  eq(got, [1, 2, 3]);
  clock.advance(60000); eq(got, [1, 2, 3]); eq(clock.pending(), 0);
}));

Deno.test("transport gap: several gaps are skipped one at a time, each after its own 5 s", () => withClock((clock) => {
  const { link, out } = sender(); for (let p = 1; p <= 6; p++) sendFrame(link, frame(p));
  const got = []; const r = receiver((m) => got.push(m.pos));
  for (const i of [5, 3, 1]) r.deliver(out[i].buf);        // 2, 4, 6 arrive; 1, 3, 5 lost
  const table = [[GAP_MS, [2]], [2 * GAP_MS, [2, 4]], [3 * GAP_MS, [2, 4, 6]]];
  for (const [t, want] of table) { clock.advance(t - clock.now - 1); ok(got.length === want.length - 1, "early at " + t); clock.advance(1); eq(got, want, "at " + t); }
  eq(clock.pending(), 0);
}));

Deno.test("transport gap: a skip followed by more multi-slice frames interleaved across stripes", () => withClock((clock) => {
  const { link, out } = sender(3);
  const words = (PER * 2 + 100) / 2;                        // 3 slices each
  const sizes = [];
  for (let p = 1; p <= 4; p++) { sendFrame(link, { t: "ai-hidden-b", basePos: p * 10, n: 2, data: new Uint16Array(words).fill(p) }); sizes.push(out.length); }
  const slicesOf = (f) => out.slice(f ? sizes[f - 1] : 0, sizes[f]);
  const got = []; const r = receiver((m) => { got.push(m.basePos); ok(m.data.every((v) => v === m.basePos / 10), "payload intact"); });
  // frame 1 loses its middle slice; frames 2..4 interleave slice by slice
  const f1 = slicesOf(0); r.deliver(f1[0].buf); r.deliver(f1[2].buf);
  for (let k = 0; k < 3; k++) for (const f of [3, 1, 2]) r.deliver(slicesOf(f)[k].buf);
  eq(got, []);
  clock.advance(GAP_MS); eq(got, [20, 30, 40]);
  r.deliver(f1[1].buf); eq(got, [20, 30, 40], "frame 1 completing late is dropped");
  eq(r.link.rx.size, 0);
}));

// ---------------------------------------------------------------- header layout and slicing

Deno.test("transport: header layout, kinds, pos vs basePos, n default, slice boundaries", () => {
  const kinds = [["ai-hidden", "pos", 0], ["ai-hidden-b", "basePos", 1], ["ai-hiddenret", "pos", 2], ["ai-hiddenret-b", "basePos", 3]];
  // payload bytes -> expected slice count
  const sizes = [[0, 1], [2, 1], [PER, 1], [PER + 2, 2], [PER * 2, 2], [PER * 2 + 2, 3]];
  for (const [t, key, kind] of kinds) for (const [bytes, nSlices] of sizes) {
    const { link, out } = sender(2);
    const data = new Uint16Array(bytes / 2); for (let i = 0; i < data.length; i++) data[i] = i * 7 + kind;
    ok(sendFrame(link, { t, [key]: 4000000000, data }), "sent");
    eq(out.length, nSlices, `${t} ${bytes} B`);
    out.forEach(({ buf }, k) => {
      const h = hdr(buf);
      eq([h.magic, h.kind, h.id, h.pos, h.n, h.k, h.nSlices, h.total], [0x5357, kind, 1, 4000000000, 1, k, nSlices, bytes]);
      ok(buf.byteLength <= SLICE_BYTES, "slice fits");
      eq(buf.byteLength - HDR, Math.min(PER, bytes - k * PER), "payload length of slice " + k);
    });
    let got = null; const r = receiver((m) => { got = m; });
    for (const { buf } of out) r.deliver(buf);
    ok(got && got.t === t && got[key] === 4000000000 && got.enc === "f16" && got.n === 1, `${t} ${bytes} meta`);
    ok(!((key === "pos" ? "basePos" : "pos") in got), `${t} carries only ${key}`);
    eq(Array.from(got.data), Array.from(data), `${t} ${bytes} bytes`);
  }
});

Deno.test("transport: a zero-length payload is one header-only slice and round trips", () => {
  const { link, out } = sender();
  ok(sendFrame(link, { t: "ai-hiddenret", pos: 9, data: new Uint16Array(0) }));
  eq(out.length, 1); eq(out[0].buf.byteLength, HDR);
  let got = null; const r = receiver((m) => { got = m; }); r.deliver(out[0].buf);
  ok(got && got.data.length === 0 && got.pos === 9);
});

Deno.test("transport: a view into a larger buffer sends only the view's bytes", () => {
  const big = new Uint16Array(100); for (let i = 0; i < 100; i++) big[i] = i;
  const view = big.subarray(10, 20);
  const { link, out } = sender(); sendFrame(link, { t: "ai-hidden", pos: 0, data: view });
  let got = null; receiver((m) => { got = m; }).deliver(out[0].buf);
  eq(Array.from(got.data), Array.from(view));
});

Deno.test("transport: sendFrame refuses unknown kinds and counts sends and ids", () => {
  const { link, out } = sender();
  for (const t of ["ai-hidden-x", "hello", undefined]) throws(() => sendFrame(link, { t, pos: 0, data: new Uint16Array(2) }), String(t));
  eq([link.sent, link.nextId, out.length], [0, 1, 0], "a refused kind consumes nothing");
  sendFrame(link, frame(1)); sendFrame(link, frame(2));
  eq([link.sent, link.nextId], [2, 3]); eq(out.map(({ buf }) => hdr(buf).id), [1, 2]);
});

Deno.test("transport: n is carried as u16; n 0 and a missing n both go out as 1", () => {
  for (const [n, want] of [[undefined, 1], [0, 1], [1, 1], [512, 512], [65535, 65535]]) {
    const { link, out } = sender(); sendFrame(link, { t: "ai-hidden-b", basePos: 0, n, data: new Uint16Array(4) });
    eq(hdr(out[0].buf).n, want, "n " + n);
  }
});

// ---------------------------------------------------------------- flags and checkpoint control

Deno.test("transport flags: rollback range, bit layout, and every combination round trips", () => {
  const table = [
    [{}, 0], [{ spec: 1 }, 1], [{ reset: 1 }, 2], [{ spec: 1, reset: 1 }, 3],
    [{ rb: 0 }, 4], [{ rb: 62 }, 252], [{ spec: 1, reset: 1, rb: 62 }, 255], [{ rb: null }, 0],
  ];
  for (const [f, byte] of table) eq(packFlags(f), byte, JSON.stringify(f));
  eq(packFlags(), 0);
  for (const rb of [63, -1, 64, 1000]) throws(() => packFlags({ rb }), "rb " + rb);
  eq(unpackFlags(0), { spec: 0 });
  for (let b = 0; b < 256; b++) eq(packFlags(unpackFlags(b)), b, "byte " + b);
});

Deno.test("transport ckpt: packCkpt/unpackCkpt table", () => {
  const dv = () => new DataView(new ArrayBuffer(HDR));
  const good = [
    [{}, {}], [{ sv: 1 }, { sv: 1 }], [{ ld: 65534 }, { ld: 65534 }], [{ dp: 5 }, { dp: [5] }], [{ dp: [] }, {}],
    [{ dp: [DROP_ALL, 5] }, { dp: [DROP_ALL] }], [{ dp: [1, 2, 3, DROP_ALL] }, { dp: [DROP_ALL] }],
    [{ sv: 2, ld: 3, dp: [4, 5] }, { sv: 2, ld: 3, dp: [4, 5] }],
    [{ sv: null, ld: undefined, dp: null }, {}],
  ];
  for (const [inp, want] of good) { const d = dv(); packCkpt(d, inp); eq(unpackCkpt(d), want, JSON.stringify(inp)); }
  const bad = [{ sv: 0 }, { ld: 0 }, { dp: 0 }, { dp: [0] }, { sv: -1 }, { ld: 65536 }, { sv: NaN }, { dp: [1, 2, 3] }, { dp: [1, 70000] }];
  for (const b of bad) throws(() => packCkpt(dv(), b), JSON.stringify(b));
  // packCkpt only touches bytes 24..31
  const d = new DataView(new ArrayBuffer(HDR)); new Uint8Array(d.buffer).fill(0xaa);
  packCkpt(d, { sv: 1 }); ok(new Uint8Array(d.buffer, 0, 24).every((v) => v === 0xaa), "frame header untouched");
});

Deno.test("transport ckpt: behaviours pinned as they are today (not produced by packCkpt)", () => {
  // only the second drop set: packCkpt never writes this, and unpackCkpt reports no drop at all
  const d = new DataView(new ArrayBuffer(HDR)); d.setUint16(30, 5); eq(unpackCkpt(d), {});
  // 65535 is accepted as a save/load slot although the documented range is 1..65534
  const e = new DataView(new ArrayBuffer(HDR)); packCkpt(e, { sv: DROP_ALL, ld: DROP_ALL }); eq(unpackCkpt(e), { sv: DROP_ALL, ld: DROP_ALL });
});

Deno.test("transport: frames mixing rb + reset + spec + sv + ld + dp carry everything on every slice", () => {
  const table = [
    { spec: 1, reset: 1, rb: 0, sv: 1, ld: 2, dp: [3, 4] },
    { reset: 1, rb: 62, sv: 65534, dp: DROP_ALL },
    { spec: 1, rb: 5, ld: 9 },
    { reset: 1, dp: [7] },
  ];
  for (const c of table) {
    const { link, out } = sender(3);
    sendFrame(link, { t: "ai-hiddenret-b", basePos: 11, n: 3, ...c, data: new Uint16Array((PER * 2 + 10) / 2) });
    eq(out.length, 3);
    for (const { buf } of out) eq(new Uint8Array(buf, 3, 1)[0], packFlags(c), "flags on every slice");
    let got = null; const r = receiver((m) => { got = m; });
    for (const { buf } of out.reverse()) r.deliver(buf);
    const want = { spec: c.spec ? 1 : 0, reset: c.reset, rb: c.rb, sv: c.sv, ld: c.ld, dp: c.dp == null ? undefined : [].concat(c.dp) };
    for (const k of Object.keys(want)) eq(got[k], want[k], `${JSON.stringify(c)} ${k}`);
  }
});

// ---------------------------------------------------------------- malformed and adversarial input

Deno.test("transport: non-binary, short and wrong-magic messages are ignored without state", () => withClock(() => {
  const got = []; const r = receiver((m) => got.push(m));
  const good = forge({ payload: new Uint8Array(4) });
  const inputs = [
    "a string", null, undefined, new Uint8Array(good),           // a typed-array view is not an ArrayBuffer
    new ArrayBuffer(0), new ArrayBuffer(HDR - 1), good.slice(0, HDR - 1),
    forge({ magic: 0x5358, payload: new Uint8Array(4) }), forge({ magic: 0x5753 }),
  ];
  for (const x of inputs) r.deliver(x);
  eq([got.length, r.link.rx.size, r.link.recv, r.link.expect], [0, 0, 0, 1]);
  r.deliver(good); eq(got.length, 1, "a well-formed slice still gets through afterwards");
}));

Deno.test("transport: a slice index past nSlices neither throws nor completes a frame that is missing a slice", () => withClock(() => {
  const got = []; const r = receiver((m) => got.push(m));
  const total = PER + 10;
  const s0 = forge({ id: 1, k: 0, nSlices: 2, total, payload: new Uint8Array(PER).fill(1) });
  // k = 2 of a 2-slice frame, empty payload: must not count toward completion
  const rogue = forge({ id: 1, k: 2, nSlices: 2, total, payload: new Uint8Array(0) });
  let threw = null; try { r.deliver(rogue); } catch (e) { threw = e; }
  ok(!threw, "rogue slice threw " + threw);
  r.deliver(s0);
  eq(got.length, 0, "frame completed with slice 1 missing");
  r.deliver(forge({ id: 1, k: 1, nSlices: 2, total, payload: new Uint8Array(10).fill(2) }));
  eq(got.length, 1, "the real slice 1 completes it");
  const bytes = new Uint8Array(got[0].data.buffer, 0, total);
  ok(bytes.subarray(0, PER).every((v) => v === 1) && bytes.subarray(PER).every((v) => v === 2), "bytes");
}));

Deno.test("transport: slices that disagree with the frame's first slice are ignored", () => withClock(() => {
  const total = PER * 2;
  const cases = [
    ["nSlices grows", { k: 1, nSlices: 3, total, payload: new Uint8Array(PER) }],
    ["nSlices shrinks", { k: 0, nSlices: 1, total, payload: new Uint8Array(PER) }],
    ["total differs", { k: 1, nSlices: 2, total: total + 2, payload: new Uint8Array(PER) }],
    ["payload too long", { k: 1, nSlices: 2, total, payload: new Uint8Array(PER + 2) }],
    ["payload too short", { k: 1, nSlices: 2, total, payload: new Uint8Array(PER - 2) }],
  ];
  for (const [name, bad] of cases) {
    const got = []; const r = receiver((m) => got.push(m));
    // the first slice to arrive is slice 1 of a 2-slice frame whose slice 0 is still missing
    const first = forge({ id: 1, k: name === "nSlices shrinks" ? 1 : 0, nSlices: 2, total, payload: new Uint8Array(PER) });
    r.deliver(first);
    let threw = null; try { r.deliver(forge({ id: 1, ...bad })); } catch (e) { threw = e; }
    ok(!threw, name + ": threw " + threw);
    eq(got.length, 0, name + ": completed from a mismatched slice");
  }
}));

Deno.test("transport: a duplicate slice after its frame completed does not open a new partial", () => withClock(() => {
  const { link, out } = sender(2); sendFrame(link, { t: "ai-hidden-b", basePos: 0, n: 1, data: new Uint16Array(PER) });   // 2 slices
  eq(out.length, 2);
  const got = []; const r = receiver((m) => got.push(m));
  for (const { buf } of out) r.deliver(buf);
  eq(got.length, 1);
  r.deliver(out[0].buf);
  eq(r.link.rx.size, 0, "a stale slice leaked a partial");
  r.deliver(out[1].buf);
  eq(got.length, 1, "delivered twice");
}));

Deno.test("transport: half-received frames older than 30 s are dropped once more than 64 are pending", () => withClock((clock) => {
  const got = []; const r = receiver((m) => got.push(m));
  const half = (id) => forge({ id, k: 0, nSlices: 2, total: PER * 2, payload: new Uint8Array(PER) });
  for (let id = 100; id < 140; id++) r.deliver(half(id));          // 40 old partials at t=0
  clock.advance(20000);
  for (let id = 200; id < 230; id++) r.deliver(half(id));          // 30 younger ones at t=20 s
  eq(r.link.rx.size, 70);
  clock.advance(10001);                                            // old ones are now 30.001 s old
  r.deliver(forge({ id: 1, payload: new Uint8Array(2) }));          // any completed frame runs the sweep
  eq(got.length, 1);
  eq(r.link.rx.size, 30, "only the stale partials were dropped");
  ok([...r.link.rx.keys()].every((id) => id >= 200));
  clock.advance(30000); r.deliver(forge({ id: 2, payload: new Uint8Array(2) }));
  eq(r.link.rx.size, 30, "no sweep at or below 64 partials");
}));

// ---------------------------------------------------------------- channels

Deno.test("transport: attachWire opens a negotiated channel; unordered means no retransmits; no pc means null", () => {
  const a = receiver(() => {}); eq(a.cfg(), { label: "swarm-wire", negotiated: true, id: WIRE_ID, ordered: true });
  eq(a.ch.binaryType, "arraybuffer");
  const b = receiver(() => {}, { ordered: false }); eq(b.cfg(), { label: "swarm-wire", negotiated: true, id: WIRE_ID, ordered: false, maxRetransmits: 0 });
  const link = makeLink(); eq(attachWire(link, { peerConnection: null }, () => {}), null); eq(link.chans.length, 0);
  a.close(); eq(a.link.chans.length, 0, "onclose removes the channel");
});

Deno.test("transport: wireReady is true only while some channel is open", () => {
  const table = [[[], false], [["connecting"], false], [["closed", "closing"], false], [["closed", "open"], true], [["open"], true]];
  for (const [states, want] of table) {
    const link = makeLink(); for (const s of states) link.chans.push({ readyState: s, send() {} });
    eq(wireReady(link), want, states.join(","));
  }
});

Deno.test("transport: round-robin striping skips non-open channels and continues across frames", () => {
  const link = makeLink(), out = [];
  const states = ["open", "connecting", "open", "open"];
  states.forEach((s, i) => link.chans.push({ readyState: s, send: (buf) => out.push(i) }));
  const three = new Uint16Array((PER * 2 + 2) / 2);
  sendFrame(link, { t: "ai-hidden-b", basePos: 0, n: 1, data: three });
  sendFrame(link, { t: "ai-hidden-b", basePos: 1, n: 1, data: three });
  eq(out, [0, 2, 3, 0, 2, 3], "open channels only, rr carried over");
  link.chans[0].readyState = "closed";
  sendFrame(link, frame(5)); sendFrame(link, frame(6));
  eq(out.slice(6), [2, 3], "a closed channel drops out of the rotation");
});

Deno.test("transport: a channel that closes mid-send leaves a gap the receiver skips after 5 s", () => withClock((clock) => {
  const link = makeLink(), wire = [];
  let sends = 0;
  const chans = [0, 1].map((i) => ({ readyState: "open", send(buf) {
    if (this.readyState !== "open") throw new Error("InvalidStateError");
    wire.push(buf);
    if (++sends === 4) chans[1].readyState = "closed";          // channel 1 dies after frame 2's first slice
  } }));
  link.chans.push(...chans);
  const three = (p) => ({ t: "ai-hidden-b", basePos: p, n: 1, data: new Uint16Array((PER * 2 + 2) / 2).fill(p) });
  sendFrame(link, three(1));                                     // slices on 0, 1, 0
  throws(() => sendFrame(link, three(2)), "send on a closed channel");   // 1 ok, 0 ok, 1 throws
  eq(wire.length, 5, "frame 2 got two of its three slices out");
  link.chans = link.chans.filter((c) => c !== chans[1]);         // what attachWire's onclose does
  ok(wireReady(link));
  ok(sendFrame(link, three(3)));
  const got = []; const r = receiver((m) => got.push(m.basePos));
  for (const b of wire) r.deliver(b);
  eq(got, [1], "frame 3 waits behind the dead frame 2");
  clock.advance(GAP_MS); eq(got, [1, 3]);
  eq(r.link.rx.size, 1, "frame 2's partial stays until the 30 s sweep");
}));
