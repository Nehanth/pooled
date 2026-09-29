// Signaling servers: which PeerJS server(s) a room uses to find its devices, and trying them in order.
//
// Signaling only introduces devices (who is in room ABCD, the WebRTC offer/answer); once the links are
// open, the room runs without it. The default is the public PeerJS cloud (0.peerjs.com). A deployment
// can list fallbacks with window.POOLED_SIGNAL_SERVERS (set before room.js loads), and ?signal= in the
// page URL wins over both (a comma list; with ?signal= only the servers it names are tried).
//
// Each server is its own namespace: a room registered on one is invisible on another. So the host
// tries the list in order and keeps the first that answers; a joiner does the same and, when that server
// has no such room, moves on down the list (the host may have fallen back); and the host's invite link
// carries ?signal= whenever it ended up on something other than the list's first choice.
//
// Specs: "cloud" (or "0.peerjs.com"), "host", "host:port", "host:port/path", or a URL
// "wss://host:port/path" / "ws://…" / "https://…" / "http://…". Bare host[:port] is secure when the page
// is (the old ?signal=host:port behaviour); port defaults to 443 (80 for ws:/http:), path to "/".
// window.POOLED_SIGNAL_SERVERS may also hold objects: { host, port, path, secure, key }.

export const CLOUD_HOST = "0.peerjs.com";
export const DEFAULT_SERVERS = ["cloud"];
// errors that mean "this server is unreachable or broken", so the next one is worth a try. Anything
// else (unavailable-id: the code is taken; invalid-id; browser-incompatible) is an answer, not an outage:
// falling back on those would split a room across two servers.
export const FALLBACK_ERRORS = new Set(["network", "server-error", "socket-error", "socket-closed", "ssl-unavailable"]);

const HOST_RE = /^(?:[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*|\[[0-9A-Fa-f:.]+\])$/;

// one spec -> { spec, label, opts } (opts go to new Peer()), or null when it doesn't parse
export function parseServer(spec, pageSecure = true) {
  if (spec && typeof spec === "object") {
    const host = String(spec.host || "");
    if (!HOST_RE.test(host)) return null;
    const secure = spec.secure ?? pageSecure;
    const port = +spec.port || (secure ? 443 : 80);
    const path = normPath(spec.path);
    const opts = { host, port, path, secure };
    if (spec.key) opts.key = String(spec.key);
    return { spec: specString(opts), label: label(opts), opts };
  }
  const s = String(spec ?? "").trim();
  if (!s) return null;
  if (s === "cloud" || s === CLOUD_HOST) return { spec: "cloud", label: CLOUD_HOST, opts: {} };
  const m = /^(?:(wss?|https?):\/\/)?([^/:\[\]]+|\[[^\]]+\])(?::(\d{1,5}))?(\/.*)?$/.exec(s);
  if (!m || !HOST_RE.test(m[2])) return null;
  const secure = m[1] ? m[1] === "wss" || m[1] === "https" : pageSecure;
  const port = m[3] ? +m[3] : (m[1] ? (secure ? 443 : 80) : 443);
  if (!(port > 0 && port < 65536)) return null;
  const opts = { host: m[2], port, path: normPath(m[4]), secure };
  return { spec: m[1] ? specString(opts) : s, label: label(opts), opts };
}
// PeerJS appends "peerjs" to the path, so it must end in "/" ("/pooled" -> "/pooled/")
const normPath = (p) => { p = String(p || "/"); if (!p.startsWith("/")) p = "/" + p; if (!p.endsWith("/")) p += "/"; return p; };
const label = (o) => o.host + (o.port === (o.secure ? 443 : 80) ? "" : ":" + o.port);
const specString = (o) => `${o.secure ? "wss" : "ws"}://${o.host}:${o.port}${o.path === "/" ? "" : o.path.slice(0, -1)}`;

// the ordered list: ?signal= (comma list) wins outright; else the configured list; else the cloud.
// Unparseable entries are skipped, duplicates dropped; an all-bad ?signal= falls through to the rest.
export function serverList({ query = null, configured = null, pageSecure = true } = {}) {
  const pick = (arr) => {
    const out = [], seen = new Set();
    for (const s of arr) { const p = parseServer(s, pageSecure); if (p && !seen.has(p.spec)) { seen.add(p.spec); out.push(p); } }
    return out;
  };
  if (query) { const q = pick(String(query).split(",")); if (q.length) return q; }
  if (Array.isArray(configured)) { const c = pick(configured); if (c.length) return c; }
  return pick(DEFAULT_SERVERS);
}

// Open a PeerJS peer on the first server in `servers` that answers. Resolves { peer, server, index }
// once the peer is registered ("open"); the caller attaches its own handlers after that. Rejects with the
// server's own error for an answer that isn't an outage (err.server says which), or, when every server
// failed, an Error with type "signaling-down" and .tried (the labels). onTry(server, index, lastError)
// runs before each attempt (for a status line). timeoutMs: a server that neither opens nor errors by then
// (a black-holed port, a captive portal) counts as down; the last server in the list gets longer
// (lastTimeoutMs), since there is nothing to fall back to and a slow phone network may still get there.
export function openPeer(PeerCtor, id, baseOpts, servers, { timeoutMs = 8000, lastTimeoutMs = 20000, onTry = () => {}, from = 0 } = {}) {
  return new Promise((resolve, reject) => {
    const tried = [];
    let last = null;
    const attempt = (i) => {
      if (i >= servers.length) {
        const e = new Error(`no signaling server answered (${tried.join(", ")})`);
        e.type = "signaling-down"; e.tried = tried; e.cause = last;
        reject(e); return;
      }
      const server = servers[i];
      tried.push(server.label);
      try { onTry(server, i, last); } catch {}
      let peer, done = false;
      const finish = (fn) => { if (done) return; done = true; clearTimeout(timer); fn(); };
      const giveUp = (err) => finish(() => {
        last = err;
        try { peer?.destroy(); } catch {}
        attempt(i + 1);
      });
      const ms = i < servers.length - 1 ? timeoutMs : lastTimeoutMs;
      const timer = setTimeout(() => { const e = new Error(`${server.label} did not answer in ${Math.round(ms / 1000)} s`); e.type = "timeout"; giveUp(e); }, ms);
      try { peer = new PeerCtor(id, { ...baseOpts, ...server.opts }); }
      catch (e) { giveUp(e); return; }
      peer.on("open", () => finish(() => resolve({ peer, server, index: i })));
      peer.on("error", (err) => {
        if (done) return;
        if (FALLBACK_ERRORS.has(err?.type)) { giveUp(err); return; }
        finish(() => { try { peer.destroy(); } catch {} ; err.server = server; reject(err); });
      });
    };
    attempt(from);
  });
}

// reconnect backoff for a peer that lost its signaling server mid-room: 2, 4, 8, 16, 30, 30 … s
export const reconnectDelay = (n) => Math.min(30000, 2000 * 2 ** Math.max(0, n));
