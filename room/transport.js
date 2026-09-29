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
const GAP_MS = 5000;                       // a frame that never completes stops holding back later ones after this

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
// hand over, done: Map<msgId, completed frame waiting for an earlier one> }
export function makeLink() { return { chans: [], rr: 0, rx: new Map(), nextId: 1, sent: 0, recv: 0, expect: 1, done: new Map(), gapTimer: null }; }

// Open the wire channel on a PeerJS DataConnection's RTCPeerConnection. Both sides call this with
// the same id, so no ondatachannel event fires and PeerJS never sees the channel.
export function attachWire(link, conn, onFrame, { ordered = true } = {}) {
  const pc = conn.peerConnection;
  if (!pc) return null;
  const ch = pc.createDataChannel("swarm-wire", { negotiated: true, id: WIRE_ID, ordered, ...(ordered ? {} : { maxRetransmits: 0 }) });
  ch.binaryType = "arraybuffer";
  ch.onmessage = (ev) => receive(link, ev.data, onFrame);
  ch.onclose = () => { link.chans = link.chans.filter((c) => c !== ch); };
  link.chans.push(ch);
  return ch;
}

export function wireReady(link) { return link.chans.some((c) => c.readyState === "open"); }

// msg: { t, pos|basePos, n?, spec?, reset?, rb?, fseq?, data: Uint16Array (f16) }
// fseq: the host's frame number (u16, 1..65535, wraps; 0 = none, what an older sender writes). It
// rides in header bytes 18..19, which protocol 4 left zero, so older peers neither send nor miss it.
export function sendFrame(link, msg) {
  const kind = KINDS.indexOf(msg.t);
  if (kind < 0) throw new Error("not a wire kind: " + msg.t);
  const open = link.chans.filter((c) => c.readyState === "open");
  if (!open.length) return false;
  link.sent++;
  const u16 = msg.data;
  const bytes = new Uint8Array(u16.buffer, u16.byteOffset, u16.byteLength);
  const per = SLICE_BYTES - HDR;
  const nSlices = Math.max(1, Math.ceil(bytes.length / per));
  const id = link.nextId++ >>> 0;
  const pos = msg.t === "ai-hidden" || msg.t === "ai-hiddenret" ? msg.pos : msg.basePos;
  const flags = packFlags(msg);
  for (let k = 0, off = 0; k < nSlices; k++) {
    const len = Math.min(per, bytes.length - off);
    const buf = new ArrayBuffer(HDR + len), dv = new DataView(buf);
    dv.setUint16(0, MAGIC); dv.setUint8(2, kind); dv.setUint8(3, flags);
    dv.setUint32(4, id); dv.setUint32(8, pos >>> 0); dv.setUint16(12, msg.n || 1);
    dv.setUint16(14, k); dv.setUint16(16, nSlices); dv.setUint16(18, msg.fseq || 0); dv.setUint32(20, bytes.length);
    packCkpt(dv, msg);
    new Uint8Array(buf, HDR).set(bytes.subarray(off, off + len));
    off += len;
    // round-robin over associations so a block never waits on one congestion window
    const ch = open[(link.rr++) % open.length];
    ch.send(buf);
  }
  return true;
}

function receive(link, buf, onFrame) {
  if (!(buf instanceof ArrayBuffer) || buf.byteLength < HDR) return;
  const dv = new DataView(buf);
  if (dv.getUint16(0) !== MAGIC) return;
  const kind = dv.getUint8(2), flags = dv.getUint8(3), id = dv.getUint32(4), pos = dv.getUint32(8), n = dv.getUint16(12);
  const ck = unpackCkpt(dv);   // every slice carries the same control
  const k = dv.getUint16(14), nSlices = dv.getUint16(16), fseq = dv.getUint16(18), total = dv.getUint32(20);
  let r = link.rx.get(id);
  if (!r) { r = { parts: new Array(nSlices), got: 0, n: nSlices, buf: new Uint8Array(total), t: performance.now(), ck }; link.rx.set(id, r); }
  if (r.parts[k]) return;   // duplicate
  r.parts[k] = true; r.got++;
  const per = SLICE_BYTES - HDR;
  r.buf.set(new Uint8Array(buf, HDR), k * per);
  if (r.got < r.n) return;
  link.rx.delete(id);
  link.recv++;
  const data = new Uint16Array(r.buf.buffer, 0, total >> 1);
  const t = KINDS[kind];
  const msg = { t, enc: "f16", data, n, ...unpackFlags(flags), ...r.ck, ...(fseq ? { fseq } : {}) };
  if (t === "ai-hidden" || t === "ai-hiddenret") msg.pos = pos; else msg.basePos = pos;
  deliverInOrder(link, id, msg, onFrame);
  // drop half-received frames older than 30 s so a lost slice cannot leak memory
  if (link.rx.size > 64) for (const [i, v] of link.rx) if (performance.now() - v.t > 30000) link.rx.delete(i);
}

// Frames complete out of order when their slices interleave across stripes; hand them over in
// send order. A gap that never fills (a send that died halfway) is skipped after GAP_MS.
function deliverInOrder(link, id, msg, onFrame) {
  // a frame whose gap was already skipped: dropping it is the only safe choice, delivering it now
  // would run it after frames sent later (the waiter for it times out and says so)
  if (id < link.expect) return;
  link.done.set(id, msg);
  flush(link, onFrame);
  armGap(link, onFrame);
}
function flush(link, onFrame) {
  while (link.done.has(link.expect)) {
    const m = link.done.get(link.expect);
    link.done.delete(link.expect);
    link.expect++;
    onFrame(m);
  }
}
// Skip a gap only when delivery has made no progress for GAP_MS: the timer remembers the frame it
// was waiting for, and if that one arrived in the meantime it re-arms for the next gap instead.
// Channels are reliable, so a real gap means a send died halfway (a closed channel).
function armGap(link, onFrame) {
  if (!link.done.size || link.gapTimer) return;
  const waitingFor = link.expect;
  link.gapTimer = setTimeout(() => {
    link.gapTimer = null;
    if (!link.done.size) return;
    if (link.expect !== waitingFor) { armGap(link, onFrame); return; }
    link.expect = Math.min(...link.done.keys());
    flush(link, onFrame);
    armGap(link, onFrame);
  }, GAP_MS);
}
