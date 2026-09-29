// Wire lab across two machines: what one activation frame costs on the real link, by frame size and
// by how it is sent. CPU only: no model, no WebGPU. One headless browser per machine, PeerJS
// signaling on machine A, a direct WebRTC link. A sends a frame, B echoes it the moment it is
// delivered, A times the round trip; one-way = (round trip - B's turnaround) / 2, B's turnaround
// (delivery -> echo handed to the channel) riding back in the echo. Manual trigger only.
// (tests/e2e/wire_lab.mjs on exp/wire-rtt is the one-machine loopback version.)
//
//   B: node tests/e2e/xwire_lab.mjs --role b --signal <A's ip>:9000 [--id pooled-xwire-b]
//   A: node tests/e2e/xwire_lab.mjs --role a [--reps 200] [--gap 30] [--sizes 4,10,16,32,40,80]
//        [--variants base,s1,unord,raw,peerjs] [--out f.json]
// A runs the signaling server (port 9000, every interface) and ends B by closing its link.
//
// Variants (B follows A's choice, sent in the connection's metadata):
//   base    the room's wire: room/transport.js, 4 negotiated channels on 4 peer connections
//           (?wire=stripe4), 4600-byte slices, ordered + reliable
//   s1      the same on one peer connection
//   unord   ordered: false as attachWire builds it (on main that also sets maxRetransmits: 0,
//           i.e. unreliable: a lost slice stalls the frame for the transport's 5 s gap timer)
//   raw     no transport: one negotiated channel, one send per frame (no slicing, no header)
//   peerjs  PeerJS's own DataConnection.send({ t, pos, data }) (binary serialization, 16 KB
//           chunks): the room's ?wire=off path, and how every control message travels
//   ping    the room's ping: a small JSON object over PeerJS (size ignored)
// --gap: ms between frames (a lap's cadence; 0 = back to back). Also reported per variant: the
// selected ICE candidate pair (addresses, protocol) and Chrome's own STUN round trip on it.
import { chromium } from "playwright";
import http from "http";
import fs from "fs";
import os from "os";
import path from "path";
import { spawn } from "child_process";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const ROLE = arg("role", "a"), REPS = +arg("reps", 200), GAP = +arg("gap", 30);
const SIZES = arg("sizes", "4,10,16,32,40,80").split(",").map((x) => +x * 1024);
const VARIANTS = arg("variants", "base,s1,unord,raw,peerjs,ping").split(",");
const PORT = +arg("port", 8171), SIG_PORT = +arg("signal-port", 9000);
const SIGNAL = arg("signal", `127.0.0.1:${SIG_PORT}`), BID = arg("id", "pooled-xwire-b");
const ROOT = path.resolve(arg("root", path.join(path.dirname(new URL(import.meta.url).pathname), "../..")));
const V = { base: { stripes: 4 }, s1: { stripes: 1 }, unord: { stripes: 4, ordered: 0 }, raw: { stripes: 1, raw: 1 }, peerjs: { stripes: 1, pj: 1 }, ping: { stripes: 1, pj: 1, ping: 1 } };
for (const v of VARIANTS) if (!V[v]) throw new Error("unknown variant " + v);

const PAGE = `<!doctype html><meta charset="utf-8"><title>xwire lab</title>
<script src="https://cdn.jsdelivr.net/npm/peerjs@1.5.4/dist/peerjs.min.js" crossorigin="anonymous"></script>
<script type="module">
const T = await import("/room/transport.js");
const P = new URLSearchParams(location.search), role = P.get("role");
const [host, port] = P.get("signal").split(":");
const now = () => performance.now();
const ICE = { iceServers: [{ urls: ["stun:stun.l.google.com:19302"] }] };   // as the room
const S = window.lab = { iso: crossOriginIsolated };
// one variant's link: stripes (PeerJS connections), the room's wire on each or one raw channel
function setup(cfg, onMsg) {
  const L = { cfg, link: T.makeLink(), conns: [], raw: null };
  L.attach = (c) => {
    L.conns.push(c);
    if (cfg.pj) { c.on("data", (d) => onMsg(d, L, c)); return; }
    if (cfg.raw) {
      const ch = c.peerConnection.createDataChannel("xwire-raw", { negotiated: true, id: 79, ordered: true });
      ch.binaryType = "arraybuffer"; ch.onmessage = (ev) => onMsg(ev.data, L, c); L.raw = ch; return;
    }
    T.attachWire(L.link, c, (m) => onMsg(m, L, c), { ordered: cfg.ordered !== 0 });
  };
  L.ready = () => cfg.pj ? L.conns.length >= 1 && L.conns[0].open : cfg.raw ? L.raw?.readyState === "open" : L.link.chans.filter((c) => c.readyState === "open").length >= cfg.stripes;
  return L;
}
// B: echo whatever arrives on the link it came in on, with its turnaround in the first 4 bytes
function echo(m, L, c) {
  const td = now();
  if (L.cfg.pj) { c.send({ ...m, turn: now() - td }); return; }
  if (L.cfg.raw) { new Float32Array(m, 0, 1)[0] = now() - td; L.raw.send(m); return; }
  new Float32Array(m.data.buffer, m.data.byteOffset, 1)[0] = now() - td;
  T.sendFrame(L.link, { t: "ai-hiddenret", pos: m.pos, data: m.data });
}
if (role === "b") {
  const peer = new Peer(P.get("id"), { host, port: +port, path: "/", secure: false, config: ICE, debug: 0 });
  const links = new Map();
  peer.on("connection", (c) => {
    const md = c.metadata || {};
    if (md.bye) { S.done = true; return; }   // A is finished
    c.on("open", () => {
      if (!links.has(md.run)) links.set(md.run, setup(md.cfg, echo));
      links.get(md.run).attach(c);
    });
  });
  S.loaded = true;
} else {
  S.pending = null;
  const onMsg = (m) => { const p = S.pending; S.pending = null; p?.(m); };
  // connect (B may not be registered yet: retry every 5 s)
  const tryOpen = (run, cfg) => new Promise((res) => {
    const peer = new Peer(undefined, { host, port: +port, path: "/", secure: false, config: ICE, debug: 0 });
    const L = setup(cfg, onMsg);
    const t0 = now();
    peer.on("open", () => {
      for (let i = 0; i < cfg.stripes; i++) { const c = peer.connect(P.get("to"), { reliable: true, label: i ? "stripe" : undefined, metadata: { run, cfg } }); c.on("open", () => L.attach(c)); }
    });
    const t = setInterval(() => {
      if (L.ready()) { clearInterval(t); S.L = L; S.peer = peer; res(true); }
      else if (now() - t0 > 5000) { clearInterval(t); peer.destroy(); res(false); }
    }, 20);
  });
  S.open = async (run, cfg) => { for (let i = 0; i < 60; i++) if (await tryOpen(run + "-" + i, cfg)) return true; throw new Error("B never answered"); };
  S.close = async (last) => {
    if (last) { S.peer.connect(P.get("to"), { metadata: { bye: 1 } }); await new Promise((r) => setTimeout(r, 1000)); }
    for (const c of S.L.conns) c.close();
    S.peer.destroy();
  };
  // the selected candidate pair of the first connection, and Chrome's own STUN round trip on it
  S.path = async () => {
    const st = await S.L.conns[0].peerConnection.getStats(); let pair = null; const c = {};
    st.forEach((r) => { if (r.type === "candidate-pair" && r.nominated && r.state === "succeeded") pair = r; if (/candidate$/.test(r.type)) c[r.id] = r; });
    if (!pair) return null;
    const l = c[pair.localCandidateId], r = c[pair.remoteCandidateId];
    return { local: l && l.address + " " + l.candidateType + " " + l.protocol, remote: r && r.address + " " + r.candidateType, stunRttMs: pair.currentRoundTripTime * 1000, totalRtt: pair.totalRoundTripTime, responses: pair.responsesReceived };
  };
  S.once = (bytes, pos) => new Promise((res) => {
    const L = S.L, cfg = L.cfg;
    let t0, t1, buf = 0;
    S.pending = (m) => {
      const rt = now() - t0;
      const turn = cfg.pj ? m.turn : cfg.raw ? new Float32Array(m, 0, 1)[0] : new Float32Array(m.data.buffer, m.data.byteOffset, 1)[0];
      res([rt, turn, buf, t1 - t0]);
    };
    const data = new Uint16Array(bytes / 2); data[2] = pos & 0xffff;
    t0 = now();
    if (cfg.ping) L.conns[0].send({ t: "ping", ts: t0 });
    else if (cfg.pj) L.conns[0].send({ t: "ai-hidden", pos, data });
    else if (cfg.raw) L.raw.send(data.buffer);
    else T.sendFrame(L.link, { t: "ai-hidden", pos, data });
    t1 = now();
    buf = Math.max(...(cfg.raw ? [L.raw] : cfg.pj ? [L.conns[0].dataChannel] : L.link.chans).map((c) => c?.bufferedAmount || 0));
  });
  S.run = async (bytes, reps, gap) => {
    const out = [];
    for (let i = 0; i < reps; i++) {
      const r = await Promise.race([S.once(bytes, i + 1), new Promise((res) => setTimeout(() => res(null), 3000))]);
      if (!r) { S.pending = null; out.push(null); continue; }
      out.push(r);
      if (gap) await new Promise((res) => setTimeout(res, gap));
    }
    return out;
  };
  S.loaded = true;
}
</script>`;
// the page on 127.0.0.1 (a secure origin), cross-origin isolated so performance.now() is good to 5 us
const srv = http.createServer((q, r) => {
  const u = new URL(q.url, "http://x");
  r.setHeader("cross-origin-opener-policy", "same-origin");
  r.setHeader("cross-origin-embedder-policy", "require-corp");
  if (u.pathname === "/lab.html") { r.setHeader("content-type", "text/html; charset=utf-8"); r.end(PAGE); return; }
  const p = path.join(ROOT, u.pathname);
  if (u.pathname.startsWith("/room/") || u.pathname.startsWith("/engine/")) {
    if (p.startsWith(ROOT) && fs.existsSync(p)) { r.setHeader("content-type", "text/javascript; charset=utf-8"); fs.createReadStream(p).pipe(r); return; }
  }
  r.statusCode = 404; r.end();
}).listen(PORT, "127.0.0.1");
let peerServer = null;
if (ROLE === "a" && arg("signal-server", "1") !== "0") {
  peerServer = spawn(arg("peerjs", path.join(ROOT, "node_modules/.bin/peerjs")), ["--port", String(SIG_PORT), "--path", "/", "--host", "0.0.0.0"], { stdio: "ignore" });
  await new Promise((r) => setTimeout(r, 1500));
}
const mac = process.platform === "darwin";
const ARGS = [...(mac ? [] : ["--no-sandbox"]), "--headless=new", "--disable-gpu", "--disable-features=WebRtcHideLocalIpsWithMdns"];
const CHROME = arg("chrome", mac ? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" : "");
const prof = fs.mkdtempSync(path.join(os.tmpdir(), "xwire-"));
const ctx = await chromium.launchPersistentContext(prof, { headless: false, args: ARGS, ...(CHROME ? { executablePath: CHROME } : {}) });
const p = ctx.pages()[0];
p.on("pageerror", (e) => console.error("pageerror", String(e).slice(0, 200)));
p.on("console", (m) => { if (m.type() === "error") console.error("console", m.text().slice(0, 200)); });
const quant = (a, f) => { const s = a.filter(Number.isFinite).sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(f * s.length))] : null; };
const f2 = (x) => (x == null ? "-" : x.toFixed(2));
const out = { role: ROLE, reps: REPS, gap: GAP, results: [] };
const done = async (code) => {
  if (arg("out")) fs.writeFileSync(arg("out"), JSON.stringify(out, null, 1));
  peerServer?.kill();
  await Promise.race([ctx.close().catch(() => {}), new Promise((r) => setTimeout(r, 10000))]);
  fs.rmSync(prof, { recursive: true, force: true }); srv.close();
  process.exit(code);
};
try {
  await p.goto(`http://127.0.0.1:${PORT}/lab.html?role=${ROLE}&signal=${SIGNAL}&id=${BID}&to=${BID}`);
  await p.waitForFunction(() => window.lab?.loaded, null, { timeout: 30000 });
  if (ROLE === "b") {
    console.error("B ready as", BID);
    await p.waitForFunction(() => window.lab.done, null, { timeout: +arg("maxmin", 30) * 60e3, polling: 500 });
    await done(0);
  }
  for (let vi = 0; vi < VARIANTS.length; vi++) {
    const name = VARIANTS[vi], cfg = V[name];
    await p.evaluate(([run, c]) => window.lab.open(run, c), [name + "-" + Date.now(), cfg]);
    const pathInfo = await p.evaluate(() => window.lab.path());
    await p.evaluate(() => window.lab.run(4096, 30, 5));   // warm-up
    for (const bytes of cfg.ping ? [0] : SIZES) {
      const rows = await p.evaluate(([b, n, g]) => window.lab.run(b, n, g), [Math.max(bytes, 8), REPS, GAP]);
      const ok = rows.filter(Boolean);
      const rtt = ok.map((r) => r[0]), one = ok.map((r) => (r[0] - (r[1] || 0)) / 2);
      const res = { variant: name, cfg, bytes, n: ok.length, lost: rows.length - ok.length, path: pathInfo,
        oneway: { p10: quant(one, 0.1), p50: quant(one, 0.5), p90: quant(one, 0.9), p99: quant(one, 0.99) }, rtt: { p50: quant(rtt, 0.5), p90: quant(rtt, 0.9) },
        turn: quant(ok.map((r) => r[1]), 0.5), sendMs: quant(ok.map((r) => r[3]), 0.5), bufMax: Math.max(0, ...ok.map((r) => r[2])) };
      out.results.push(res);
      console.log(`${name.padEnd(7)} ${String(bytes / 1024).padStart(3)} KB  one-way p10 ${f2(res.oneway.p10)} p50 ${f2(res.oneway.p50)} p90 ${f2(res.oneway.p90)} p99 ${f2(res.oneway.p99)}  rtt p50 ${f2(res.rtt.p50)}  send ${f2(res.sendMs)}  buffered max ${res.bufMax}  lost ${res.lost}  stun rtt ${f2(pathInfo?.stunRttMs)}  ${pathInfo?.local} -> ${pathInfo?.remote}`);
    }
    await p.evaluate((last) => window.lab.close(last), vi === VARIANTS.length - 1);
    await p.waitForTimeout(500);
  }
} catch (e) { out.error = String(e).slice(0, 400); console.error("FAILED", out.error); }
await done(out.error ? 1 : 0);
