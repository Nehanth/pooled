// A minimal TURN server (RFC 5766 over UDP, long-term credentials) for tests: enough of the
// protocol for Chrome's TURN client to allocate a relay, install permissions, bind channels and
// move data both ways. Loopback only, no TCP/TLS, no nonce expiry, no quotas. Not for real use.
//
//   import { startTurn } from "./turn_server.mjs";
//   const turn = await startTurn({ user: "u", pass: "p" });   // turn.port, turn.stats, turn.close()
import dgram from "dgram";
import crypto from "crypto";

const COOKIE = 0x2112a442;
const M = { binding: 0x001, allocate: 0x003, refresh: 0x004, send: 0x006, data: 0x007, perm: 0x008, chan: 0x009 };
const A = { username: 0x0006, integrity: 0x0008, error: 0x0009, channel: 0x000c, lifetime: 0x000d, peer: 0x0012, data: 0x0013,
  realm: 0x0014, nonce: 0x0015, relayed: 0x0016, transport: 0x0019, mapped: 0x0020, software: 0x8022, fingerprint: 0x8028 };
const cls = (method, c) => (method & 0xf) | ((method & 0x70) << 1) | ((method & 0xf80) << 2) | c;   // c: 0 req, 0x10 ind, 0x100 ok, 0x110 err

function parse(buf) {
  if (buf.length < 20 || (buf[0] & 0xc0) !== 0 || buf.readUInt32BE(4) !== COOKIE) return null;
  const type = buf.readUInt16BE(0), len = buf.readUInt16BE(2);
  if (20 + len > buf.length) return null;
  const method = (type & 0xf) | ((type >> 1) & 0x70) | ((type >> 2) & 0xf80), klass = type & 0x110;
  const attrs = new Map(); let miAt = -1;
  for (let o = 20; o + 4 <= 20 + len;) {
    const t = buf.readUInt16BE(o), l = buf.readUInt16BE(o + 2);
    if (t === A.integrity) miAt = o;
    if (!attrs.has(t)) attrs.set(t, buf.subarray(o + 4, o + 4 + l));
    o += 4 + ((l + 3) & ~3);
  }
  return { method, klass, tid: buf.subarray(8, 20), attrs, miAt, buf };
}
function xaddr(v, tid) {
  const port = v.readUInt16BE(2) ^ (COOKIE >>> 16);
  const ip = [0, 1, 2, 3].map((i) => v[4 + i] ^ ((COOKIE >>> (24 - 8 * i)) & 0xff)).join(".");
  return { ip, port };
}
function xaddrBuf(ip, port) {
  const b = Buffer.alloc(8); b[1] = 1; b.writeUInt16BE(port ^ (COOKIE >>> 16), 2);
  ip.split(".").forEach((x, i) => { b[4 + i] = +x ^ ((COOKIE >>> (24 - 8 * i)) & 0xff); });
  return b;
}
function build(type, tid, attrs, key) {
  const parts = [];
  for (const [t, v] of attrs) {
    const h = Buffer.alloc(4); h.writeUInt16BE(t, 0); h.writeUInt16BE(v.length, 2);
    parts.push(h, v, Buffer.alloc((4 - (v.length % 4)) % 4));
  }
  let body = Buffer.concat(parts);
  const head = Buffer.alloc(20); head.writeUInt16BE(type, 0); head.writeUInt32BE(COOKIE, 4); tid.copy(head, 8);
  if (key) {
    head.writeUInt16BE(body.length + 24, 2);
    const mac = crypto.createHmac("sha1", key).update(Buffer.concat([head, body])).digest();
    const h = Buffer.alloc(4); h.writeUInt16BE(A.integrity, 0); h.writeUInt16BE(20, 2);
    body = Buffer.concat([body, h, mac]);
  }
  head.writeUInt16BE(body.length, 2);
  return Buffer.concat([head, body]);
}
const u32 = (n) => { const b = Buffer.alloc(4); b.writeUInt32BE(n); return b; };

export async function startTurn({ user = "pooled", pass = "test", realm = "pooled.test", host = "127.0.0.1", port = 0 } = {}) {
  const key = crypto.createHash("md5").update(`${user}:${realm}:${pass}`).digest();
  const nonce = Buffer.from(crypto.randomBytes(8).toString("hex"));
  const stats = { allocations: 0, authFailures: 0, relayedToPeer: 0, relayedToClient: 0, permissions: 0, channels: 0 };
  const allocs = new Map();   // "ip:port" of the client -> allocation
  const sock = dgram.createSocket("udp4");
  await new Promise((r) => sock.bind(port, host, r));
  const reply = (rinfo, buf) => sock.send(buf, rinfo.port, rinfo.address);
  const err = (m, rinfo, code, reason, extra = []) => {
    const e = Buffer.concat([Buffer.from([0, 0, Math.floor(code / 100), code % 100]), Buffer.from(reason)]);
    reply(rinfo, build(cls(m.method, 0x110), m.tid, [[A.error, e], ...extra], null));
  };
  // long-term credential check; false (and an error sent) when the request is not authenticated
  const authed = (m, rinfo) => {
    const un = m.attrs.get(A.username), mi = m.attrs.get(A.integrity);
    if (!un || !mi || m.miAt < 0) { err(m, rinfo, 401, "Unauthorized", [[A.realm, Buffer.from(realm)], [A.nonce, nonce]]); return false; }
    const b = Buffer.from(m.buf.subarray(0, m.miAt)); b.writeUInt16BE(m.miAt + 24 - 20, 2);
    const mac = crypto.createHmac("sha1", key).update(b).digest();
    if (un.toString() !== user || !mac.equals(mi)) { stats.authFailures++; err(m, rinfo, 401, "Unauthorized", [[A.realm, Buffer.from(realm)], [A.nonce, nonce]]); return false; }
    return true;
  };
  sock.on("message", async (buf, rinfo) => {
    const ck = `${rinfo.address}:${rinfo.port}`;
    // ChannelData from a client
    if (buf.length >= 4 && buf[0] >= 0x40 && buf[0] <= 0x7f) {
      const a = allocs.get(ck); if (!a) return;
      const peer = a.chans.get(buf.readUInt16BE(0)); if (!peer) return;
      const len = buf.readUInt16BE(2);
      a.relay.send(buf.subarray(4, 4 + len), peer.port, peer.ip); stats.relayedToPeer++;
      return;
    }
    const m = parse(buf); if (!m) return;
    if (m.method === M.binding && m.klass === 0) {
      reply(rinfo, build(cls(M.binding, 0x100), m.tid, [[A.mapped, xaddrBuf(rinfo.address, rinfo.port)]], null));
      return;
    }
    if (m.method === M.send && m.klass === 0x10) {
      const a = allocs.get(ck), p = m.attrs.get(A.peer), d = m.attrs.get(A.data);
      if (!a || !p || !d) return;
      const peer = xaddr(p, m.tid);
      if (!a.perms.has(peer.ip)) return;
      a.relay.send(d, peer.port, peer.ip); stats.relayedToPeer++;
      return;
    }
    if (m.klass !== 0) return;
    if (!authed(m, rinfo)) return;
    const ok = (attrs) => reply(rinfo, build(cls(m.method, 0x100), m.tid, attrs, key));
    if (m.method === M.allocate) {
      let a = allocs.get(ck);
      if (!a) {
        const relay = dgram.createSocket("udp4");
        await new Promise((r) => relay.bind(0, host, r));
        a = { relay, perms: new Set(), chans: new Map(), byPeer: new Map() };
        relay.on("message", (d, from) => {
          if (!a.perms.has(from.address)) return;
          stats.relayedToClient++;
          const ch = a.byPeer.get(`${from.address}:${from.port}`);
          if (ch) { const h = Buffer.alloc(4); h.writeUInt16BE(ch, 0); h.writeUInt16BE(d.length, 2); reply(rinfo, Buffer.concat([h, d])); }
          else reply(rinfo, build(cls(M.data, 0x10), crypto.randomBytes(12), [[A.peer, xaddrBuf(from.address, from.port)], [A.data, d]], null));
        });
        relay.on("error", () => {});
        allocs.set(ck, a); stats.allocations++;
      }
      const ra = a.relay.address();
      ok([[A.relayed, xaddrBuf(ra.address, ra.port)], [A.mapped, xaddrBuf(rinfo.address, rinfo.port)], [A.lifetime, u32(600)]]);
    } else if (m.method === M.refresh) {
      const lt = m.attrs.get(A.lifetime);
      if (lt && lt.readUInt32BE(0) === 0) { const a = allocs.get(ck); if (a) { a.relay.close(); allocs.delete(ck); } }
      ok([[A.lifetime, u32(lt ? lt.readUInt32BE(0) : 600)]]);
    } else if (m.method === M.perm) {
      const a = allocs.get(ck); if (!a) return err(m, rinfo, 437, "Allocation Mismatch");
      const p = m.attrs.get(A.peer); if (p) { a.perms.add(xaddr(p, m.tid).ip); stats.permissions++; }
      ok([]);
    } else if (m.method === M.chan) {
      const a = allocs.get(ck); if (!a) return err(m, rinfo, 437, "Allocation Mismatch");
      const c = m.attrs.get(A.channel), p = m.attrs.get(A.peer);
      if (!c || !p) return err(m, rinfo, 400, "Bad Request");
      const num = c.readUInt16BE(0), peer = xaddr(p, m.tid);
      a.chans.set(num, peer); a.byPeer.set(`${peer.ip}:${peer.port}`, num); a.perms.add(peer.ip); stats.channels++;
      ok([]);
    }
  });
  sock.on("error", () => {});
  return {
    port: sock.address().port, stats,
    close: () => { for (const a of allocs.values()) try { a.relay.close(); } catch {} try { sock.close(); } catch {} },
  };
}
