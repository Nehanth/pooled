// A PeerJS-compatible signaling server for tests, which can play the part of a malicious one: it hands
// a link meant for `target` to `attacker` instead, and makes the attacker's answers look as if they came
// from `target`. The attacker then sits in the middle of the link (WebRTC checks the certificate
// against the fingerprint signaling carried, and signaling is what lies here). relay() is such an
// attacker: it takes the victim's link and opens its own to the real target, and passes every message
// both ways unchanged.
//
// The server speaks what the PeerJS client (peerjs 1.5.4) needs: GET <path>peerjs/id, and the WebSocket
// at <path>peerjs?id=&token=&key= with OPEN, HEARTBEAT and OFFER / ANSWER / CANDIDATE / LEAVE / EXPIRE
// passed to dst with src set by the server.
import http from "node:http";
import crypto from "node:crypto";
import { WebSocketServer } from "ws";

// -> { port, close(), hijack(rule) }; rule: { target, attacker, spare: [ids whose links reach target as asked] }
export async function startSignal({ port = 0 } = {}) {
  const clients = new Map();   // id -> ws
  let rule = null;
  const srv = http.createServer((req, res) => {
    const u = new URL(req.url, "http://x");
    res.setHeader("access-control-allow-origin", "*");
    if (u.pathname.endsWith("/peerjs/id")) { res.setHeader("content-type", "text/html"); res.end(crypto.randomUUID()); return; }
    res.statusCode = 404; res.end();
  });
  const wss = new WebSocketServer({ noServer: true });
  srv.on("upgrade", (req, sock, head) => {
    const u = new URL(req.url, "http://x");
    if (!u.pathname.endsWith("/peerjs")) { sock.destroy(); return; }
    wss.handleUpgrade(req, sock, head, (ws) => {
      const id = u.searchParams.get("id");
      if (!id || clients.has(id)) { ws.send(JSON.stringify({ type: "ID-TAKEN", payload: { msg: "ID is taken" } })); ws.close(); return; }
      clients.set(id, ws);
      ws.send(JSON.stringify({ type: "OPEN" }));
      ws.on("message", (raw) => {
        let m; try { m = JSON.parse(String(raw)); } catch { return; }
        if (!m || m.type === "HEARTBEAT") return;
        let dst = m.dst, src = id;
        if (rule) {
          // a link to the target from anyone but the attacker's own side: to the attacker instead
          if (dst === rule.target && !rule.spare.includes(id) && id !== rule.attacker) dst = rule.attacker;
          // the attacker's answers to a victim: as if from the target
          else if (id === rule.attacker && dst !== rule.target) src = rule.target;
        }
        const to = clients.get(dst);
        if (!to) { if (m.type === "OFFER") ws.send(JSON.stringify({ type: "EXPIRE", src: m.dst, payload: { msg: "Could not connect to peer " + m.dst } })); return; }
        to.send(JSON.stringify({ type: m.type, src, dst, payload: m.payload }));
      });
      ws.on("close", () => { if (clients.get(id) === ws) clients.delete(id); });
    });
  });
  await new Promise((r) => srv.listen(port, "127.0.0.1", r));
  return {
    port: srv.address().port,
    hijack(r) { rule = r ? { spare: [], ...r } : null; },
    close() { for (const ws of clients.values()) try { ws.terminate(); } catch {} wss.close(); srv.close(); },
  };
}

// The attacker in the middle: Peer is a PeerJS class (room-node's env.Peer after setupNode). Its peer
// `attackerId` takes the victim's link; its second peer dials the real target and relays. -> { stop(),
// seen: messages that crossed (victim -> target as "up", target -> victim as "down") }
export async function relay(Peer, { port, attackerId, target, signal }) {
  const opts = { host: "127.0.0.1", port, path: "/", secure: false, debug: 0, config: { iceServers: [] } };
  const a1 = new Peer(attackerId, opts);
  const a2 = new Peer(undefined, opts);
  await Promise.all([new Promise((r) => a1.on("open", r)), new Promise((r) => a2.on("open", r))]);
  signal.hijack({ target, attacker: attackerId, spare: [a2.id] });
  const seen = [];
  a1.on("connection", (victim) => {
    const up = a2.connect(target, { reliable: true, label: victim.label });
    const toUp = [], toDown = [];
    let upOpen = false, downOpen = false;
    victim.on("open", () => { downOpen = true; for (const m of toDown.splice(0)) victim.send(m); });
    up.on("open", () => { upOpen = true; for (const m of toUp.splice(0)) up.send(m); });
    victim.on("data", (m) => { seen.push({ dir: "up", m }); if (upOpen) up.send(m); else toUp.push(m); });
    up.on("data", (m) => { seen.push({ dir: "down", m }); if (downOpen) victim.send(m); else toDown.push(m); });
    victim.on("close", () => { try { up.close(); } catch {} });
    up.on("close", () => { try { victim.close(); } catch {} });
    victim.on("error", () => {}); up.on("error", () => {});
  });
  return { seen, a2id: a2.id, stop() { signal.hijack(null); try { a1.destroy(); } catch {} try { a2.destroy(); } catch {} } };
}
