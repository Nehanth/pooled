// Hidden-state transport: a dedicated data channel per peer link that sends activation frames
// as small slices, optionally striped over several peer connections.
//
// Why: Chrome's SCTP stack (dcSCTP) releases at most 4 packets per send opportunity and starts
// with a ~12 KB congestion window, so a single 10 KB message pays an extra round trip and a 50 KB
// speculative verify block pays three. Measured on a 100 ms link: 1 KB = 51 ms one-way,
// 5 KB = 153 ms, 20 KB = 254 ms (docs/bench-log.md). Slicing every send under four packets and
// spreading a block across several associations brings a hop back to one one-way trip.
//
// Exact by construction: only the packaging of the bytes changes.
//
// Ordering: a worker's recurrent state depends on the order it sees frames and on the control
// that goes with them (a speculative rollback, a new-conversation reset). Both used to be
// separate PeerJS messages on another channel, which a frame on the wire could overtake after a
// lost packet; now they ride in the frame header, and the receiver hands frames to the
// application strictly in send order even when slices of consecutive frames interleave across
// stripes. That is what lets the host keep several prefill rounds in flight.

export const WIRE_ID = 77;                 // negotiated channel id, same on both ends
export const SLICE_BYTES = 4600;           // ~4 packets of 1150 B payload
const HDR = 32;   // 24 bytes of frame + 8 of checkpoint control (protocol 4)
const MAGIC = 0x5357;                      // "SW"
const KINDS = ["ai-hidden", "ai-hidden-b", "ai-hiddenret", "ai-hiddenret-b"];
// Gaps. Every wire channel is reliable and ordered, so a frame that has not arrived yet is late
// (a freeze, a lost packet waiting out SCTP's retransmission timer), not lost, as long as the
// channels that carried it are open. A frame is only ever lost with a channel that closed.
const GAP_MS = 5000;        // a channel closed recently: a frame still missing this long went down with it
const CLOSE_WINDOW_MS = 15000;   // "recently": frames sent before a close are the ones it can have taken along
const GAP_MAX_MS = 60000;   // no channel closed: wait this long (a backstop; the host's own lap timeout is 30 s)
// Small frames (a decode token's hidden state) go out twice, on two different associations, when
// the link has more than one: a lost packet on one then costs nothing instead of a retransmission
// timeout (hundreds of ms to seconds, because a lone message has no later packet to trigger a
// fast retransmit). The receiver takes whichever copy completes first; protocol-4 receivers
// already drop the second copy as a duplicate.
export const DUP_SLICES = 3;

// Room protocol version: peers with a different one are refused at hello (docs/protocol.md).
export const PROTOCOL = 4;

// Header flags byte: bit 0 speculative verify, bit 1 reset before this frame, bits 2..7 roll the
// recurrent state back to after column k before this frame (stored as k + 1; 0 = none).
export function packFlags({ spec, reset, rb } = {}) {
  if (rb != null && (rb < 0 || rb > 62)) throw new Error("rollback column out of range: " + rb);
  return (spec ? 1 : 0) | (reset ? 2 : 0) | (rb != null ? (rb + 1) << 2 : 0);
}
// Checkpoint control (header bytes 24..31, u16 each, 0 = none): sv = save this device's state
// under a slot number before the frame, ld = load a slot, dp = up to two slots to drop, or
// DROP_ALL. Slot numbers are 1..65534.
export const DROP_ALL = 0xffff;
export function packCkpt(dv, { sv, ld, dp } = {}) {
  const drops = dp == null ? [] : [].concat(dp);
  if (drops.length > 2 && !drops.includes(DROP_ALL)) throw new Error("at most two checkpoint drops per frame");
  for (const v of [sv, ld, ...drops]) if (v != null && !(v >= 1 && v <= 0xffff)) throw new Error("checkpoint slot out of range: " + v);
  const d = drops.includes(DROP_ALL) ? [DROP_ALL] : drops;
  dv.setUint16(24, sv || 0); dv.setUint16(26, ld || 0); dv.setUint16(28, d[0] || 0); dv.setUint16(30, d[1] || 0);
}
export function unpackCkpt(dv) {
  const out = {}, sv = dv.getUint16(24), ld = dv.getUint16(26), d0 = dv.getUint16(28), d1 = dv.getUint16(30);
  if (sv) out.sv = sv;
  if (ld) out.ld = ld;
  if (d0) out.dp = d1 ? [d0, d1] : [d0];
  return out;
}
export function unpackFlags(f) {
  const out = { spec: f & 1 };
  if (f & 2) out.reset = 1;
  if (f >> 2) out.rb = (f >> 2) - 1;
  return out;
}

// Per-link state: { chans: [RTCDataChannel], rr, rx: Map<msgId, partial>, expect: next id to
// hand over, done: Map<msgId, completed frame waiting for an earlier one>, hi: Map<channel,
// highest frame id received on it>, closedAt: when a wire channel of this link last closed,
// unordered: some channel may drop messages (attachWire { ordered: false }), dup: frames of up to
// this many slices are sent twice (0 = never), rxAt: when anything last arrived on this link's
// wire (a slice of a frame or a keep-alive byte: a sign of life, room/liveness.js), ka*: keep-alive }
export function makeLink({ dup = DUP_SLICES } = {}) {
  return { chans: [], rr: 0, rx: new Map(), nextId: 1, sent: 0, recv: 0, expect: 1, done: new Map(), gapTimer: null, gapFor: 0,
    hi: new Map(), closedAt: null, unordered: false, dup, dups: 0, skipped: 0,
    ka: [], kaRr: 0, kaSent: 0, lastTx: 0, active: 0, rxAt: 0 };
}

// Open the wire channel on a PeerJS DataConnection's RTCPeerConnection. Both sides call this with
// the same id, so no ondatachannel event fires and PeerJS never sees the channel.
export function attachWire(link, conn, onFrame, { ordered = true } = {}) {
  const pc = conn.peerConnection;
  if (!pc) return null;
  const ch = pc.createDataChannel("swarm-wire", { negotiated: true, id: WIRE_ID, ordered, ...(ordered ? {} : { maxRetransmits: 0 }) });
  ch.binaryType = "arraybuffer";
  if (!ordered) link.unordered = true;
  ch.onmessage = (ev) => { link.rxAt = performance.now(); receive(link, ev.data, onFrame, ch); };
  ch.onclose = () => {
    link.chans = link.chans.filter((c) => c !== ch);
    link.hi.delete(ch);
    // whatever this channel still had in flight is gone: gaps may now be real
    link.closedAt = performance.now();
    if (link.done.size) { clearTimeout(link.gapTimer); link.gapTimer = null; checkGap(link, onFrame); }
  };
  link.chans.push(ch);
  attachKeepalive(link, pc);
  return ch;
}

export function wireReady(link) { return link.chans.some((c) => c.readyState === "open"); }

// msg: { t, pos|basePos, n?, spec?, reset?, rb?, data: Uint16Array (f16) }
export function sendFrame(link, msg) {
  const kind = KINDS.indexOf(msg.t);
  if (kind < 0) throw new Error("not a wire kind: " + msg.t);
  const open = link.chans.filter((c) => c.readyState === "open");
  if (!open.length) return false;
  link.sent++;
  kaNote(link, true);
  const u16 = msg.data;
  const bytes = new Uint8Array(u16.buffer, u16.byteOffset, u16.byteLength);
  const per = SLICE_BYTES - HDR;
  const nSlices = Math.max(1, Math.ceil(bytes.length / per));
  const id = link.nextId++ >>> 0;
  const pos = msg.t === "ai-hidden" || msg.t === "ai-hiddenret" ? msg.pos : msg.basePos;
  const flags = packFlags(msg);
  const dup = open.length > 1 && nSlices <= link.dup;
  for (let k = 0, off = 0; k < nSlices; k++) {
    const len = Math.min(per, bytes.length - off);
    const buf = new ArrayBuffer(HDR + len), dv = new DataView(buf);
    dv.setUint16(0, MAGIC); dv.setUint8(2, kind); dv.setUint8(3, flags);
    dv.setUint32(4, id); dv.setUint32(8, pos >>> 0); dv.setUint16(12, msg.n || 1);
    dv.setUint16(14, k); dv.setUint16(16, nSlices); dv.setUint32(20, bytes.length);
    packCkpt(dv, msg);
    new Uint8Array(buf, HDR).set(bytes.subarray(off, off + len));
    off += len;
    // spread over associations so a block never waits on one congestion window
    const ch = pick(link, open, null);
    put(link, open, ch, buf);
    if (dup) { put(link, open, pick(link, open, ch), buf); link.dups++; }
  }
  return true;
}
// The open channel with the least data queued, starting the search at the round-robin position so
// equal channels take turns. A channel whose association lost a packet holds its later messages
// back (ordered) and backs off its congestion window, so its queue grows; new slices go elsewhere.
function pick(link, open, avoid) {
  let best = null;
  for (let i = 0; i < open.length; i++) {
    const c = open[(link.rr + i) % open.length];
    if (c === avoid) continue;
    if (!best || (c.bufferedAmount || 0) < (best.bufferedAmount || 0)) best = c;
  }
  link.rr++;
  return best;
}
// send, falling over to the other open channels if this one refuses (it closed under us); a slice
// no channel takes is lost, and the receiver skips its frame once the close is seen
function put(link, open, ch, buf) {
  for (const c of [ch, ...open.filter((o) => o !== ch)]) {
    if (!c || c.readyState !== "open") continue;
    try { c.send(buf); return true; } catch {}
  }
  return false;
}

function receive(link, buf, onFrame, ch) {
  if (!(buf instanceof ArrayBuffer) || buf.byteLength < HDR) return;
  const dv = new DataView(buf);
  if (dv.getUint16(0) !== MAGIC) return;
  const kind = dv.getUint8(2), flags = dv.getUint8(3), id = dv.getUint32(4), pos = dv.getUint32(8), n = dv.getUint16(12);
  // the sender numbers frames in order and each channel is ordered: once a channel has shown id,
  // nothing older can still come on it
  if (ch && id > (link.hi.get(ch) || 0)) link.hi.set(ch, id);
  if (link.done.size && id !== link.expect) checkGap(link, onFrame);
  const ck = unpackCkpt(dv);   // every slice carries the same control
  const k = dv.getUint16(14), nSlices = dv.getUint16(16), total = dv.getUint32(20);
  // a slice of a frame already delivered or skipped, or the second copy of a small frame: ignore
  // it rather than open a partial that leaks (and drop what a skipped frame had gathered)
  if (id < link.expect || link.done.has(id)) { link.rx.delete(id); return; }
  const per = SLICE_BYTES - HDR;
  // a slice must fit the frame exactly (index in range, same slice count and size as the first
  // slice seen, the payload length sendFrame gives slice k); anything else would throw in set()
  // after counting, or count toward completion and deliver a frame with a slice missing
  if (k >= nSlices || buf.byteLength - HDR !== Math.min(per, total - k * per)) return;
  let r = link.rx.get(id);
  if (!r) { r = { parts: new Array(nSlices), got: 0, n: nSlices, buf: new Uint8Array(total), t: performance.now(), ck }; link.rx.set(id, r); }
  if (r.n !== nSlices || r.buf.length !== total) return;
  if (r.parts[k]) return;   // duplicate
  r.parts[k] = true; r.got++;
  r.buf.set(new Uint8Array(buf, HDR), k * per);
  if (r.got < r.n) return;
  link.rx.delete(id);
  link.recv++;
  kaNote(link, false);
  const data = new Uint16Array(r.buf.buffer, 0, total >> 1);
  const t = KINDS[kind];
  const msg = { t, enc: "f16", data, n, ...unpackFlags(flags), ...r.ck };
  if (t === "ai-hidden" || t === "ai-hiddenret") msg.pos = pos; else msg.basePos = pos;
  deliverInOrder(link, id, msg, onFrame);
  // drop half-received frames older than 30 s so a lost slice cannot leak memory
  if (link.rx.size > 64) for (const [i, v] of link.rx) if (performance.now() - v.t > 30000) link.rx.delete(i);
}

// Frames complete out of order when their slices interleave across stripes; hand them over in
// send order, holding later frames while an earlier one is still on its way.
function deliverInOrder(link, id, msg, onFrame) {
  // a frame whose gap was already skipped: dropping it is the only safe choice, delivering it now
  // would run it after frames sent later (the waiter for it times out and says so)
  if (id < link.expect) return;
  link.done.set(id, msg);
  flush(link, onFrame);
  checkGap(link, onFrame);
}
function flush(link, onFrame) {
  while (link.done.has(link.expect)) {
    const m = link.done.get(link.expect);
    link.done.delete(link.expect);
    link.expect++;
    onFrame(m);
  }
}
// Frame `id` can no longer arrive when every open channel has already delivered something newer
// (each is ordered, so nothing older is queued behind it): some of its slices went out on a
// channel that has since closed, or never went out at all. Not on links with a lossy channel.
function provablyLost(link, id) {
  if (link.unordered) return false;
  const open = link.chans.filter((c) => c.readyState === "open");
  return open.length > 0 && open.every((c) => (link.hi.get(c) || 0) > id);
}
function skipTo(link, id) {
  for (let i = link.expect; i < id; i++) link.rx.delete(i);
  link.skipped += id - link.expect;
  link.expect = id;
}
// A missing frame holds back the later ones until it arrives. It is skipped at once when it
// provably cannot arrive; otherwise after GAP_MS without progress if a channel closed recently
// (or the link may drop messages), else only after GAP_MAX_MS (a backstop: channels are reliable,
// and a freeze or a retransmission backoff can hold a frame for many seconds). The timer
// remembers the frame it was waiting for; if that one arrived meanwhile it re-arms instead.
function checkGap(link, onFrame) {
  if (!link.done.size) { if (link.gapTimer) { clearTimeout(link.gapTimer); link.gapTimer = null; } return; }
  for (;;) {
    let moved = false;
    while (!link.done.has(link.expect) && provablyLost(link, link.expect)) { skipTo(link, link.expect + 1); moved = true; }
    if (!moved) break;
    flush(link, onFrame);
    if (!link.done.size) { clearTimeout(link.gapTimer); link.gapTimer = null; return; }
  }
  if (link.gapTimer && link.gapFor === link.expect) return;
  clearTimeout(link.gapTimer);
  const waitingFor = link.gapFor = link.expect;
  const now = performance.now();
  const lossy = link.unordered || (link.closedAt != null && now - link.closedAt < CLOSE_WINDOW_MS);
  link.gapTimer = setTimeout(() => {
    link.gapTimer = null;
    if (!link.done.size) return;
    if (link.expect === waitingFor) {
      skipTo(link, Math.min(...link.done.keys()));
      flush(link, onFrame);
    }
    checkGap(link, onFrame);
  }, lossy ? GAP_MS : GAP_MAX_MS);
}

// Keep-alive while frames flow. A phone's Wi-Fi drops into power save between sparse frames: in a
// room a lap is ~50 ms and the phone's own slice is busy for ~10 of it, so the radio idles ~40 ms
// per lap, and a frame for a dozing phone waits at the access point for its next wake-up (the
// phone hop's wire time: the same median as a computer's, twice the p95, a 0.8 s stall). While a
// link has carried a frame in the last KA_ACTIVE_MS, each end sends a 1-byte message whenever it
// has sent nothing for KA_MS, which keeps both radios awake; it stops by itself when the room goes
// quiet. The bytes ride their own negotiated channel (KA_ID), unordered and never retransmitted,
// so a lost one holds up nothing; being on the same association as the frames, they also let the
// receiver report a lost frame slice at once instead of the sender waiting out a retransmission
// timeout. A peer without the channel drops them (the SCTP stream is unknown to it), so this needs
// no protocol bump. ?ka=ms in the room URL sets the period, ?ka=0 turns it off.
export const KA_ID = 78;
const KA_ACTIVE_MS = 1500;
const KA_BYTE = new Uint8Array([0]);
let kaMs = 10, kaTimer = null;
const kaLinks = new Set();
export function setKeepalive(ms) { kaMs = Math.max(0, +ms || 0); }
function attachKeepalive(link, pc) {
  if (!link.ka) return;
  const ch = pc.createDataChannel("swarm-ka", { negotiated: true, id: KA_ID, ordered: false, maxRetransmits: 0 });
  ch.addEventListener?.("message", () => { link.rxAt = performance.now(); });   // the peer's keep-alive bytes: it is alive
  ch.onclose = () => { link.ka = link.ka.filter((c) => c !== ch); };
  link.ka.push(ch);
}
function kaNote(link, tx) {
  const now = performance.now();
  link.active = now;
  if (tx) link.lastTx = now;
  if (!kaMs || !link.ka?.length) return;
  kaLinks.add(link);
  if (!kaTimer) kaTimer = setInterval(kaTick, kaMs);
}
function kaTick() {
  const now = performance.now();
  for (const link of kaLinks) {
    if (now - link.active > KA_ACTIVE_MS) { kaLinks.delete(link); continue; }
    if (now - link.lastTx < kaMs) continue;
    // one association per tick, in turn: one packet keeps the radio up, the rotation gives every
    // association a recent packet for loss reports
    const open = link.ka.filter((c) => c.readyState === "open" && c.bufferedAmount < 1024);
    if (!open.length) continue;
    try { open[(link.kaRr++) % open.length].send(KA_BYTE); link.kaSent++; link.lastTx = now; } catch {}
  }
  if (!kaLinks.size || !kaMs) { clearInterval(kaTimer); kaTimer = null; }
}
