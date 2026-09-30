// ICE configuration for the room's WebRTC links: public STUN always, plus a TURN relay for
// networks where a direct connection cannot be made (symmetric NAT / CGNAT on both sides, a work
// network that blocks UDP and every port but 443). No relay password ships in the page.
//
// Where a relay comes from, first match wins:
//   1. the page URL: ?turn=turn:relay.example.org:3478 (comma-separate several URLs), with
//      ?turnuser= and ?turncred= for its long-term credentials, ?relay=1 to use only the relay
//   2. this browser's saved setting (the "Network" box under the join form), localStorage TURN_KEY
//   3. window.TURN_SERVERS, an RTCIceServer array a self-hosted deployment can define before
//      room.js loads
//   4. the deployment's default relay: POST /api/turn (api/turn.mjs) hands out credentials that
//      expire (hours), when the site's owner configured a provider; 204 or no answer within
//      RELAY_FETCH_MS means none, and the room works as before (direct links only). ?relay=0 skips
//      it; on localhost it is only asked with ?turnapi=1 (a static dev server has no /api).
// ICE always prefers direct candidates (host, then STUN), so a relay only carries a link when no
// direct path works, unless ?relay=1 (or the Network box's "Always go through the relay") forces it.
// Join links never carry turn / turnuser / turncred (roomLink strips them), so a relay password
// does not end up in a QR code or a chat.

export const STUN = { urls: ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"] };
export const TURN_KEY = "pooled-turn";
export const SECRET_PARAMS = ["turn", "turnuser", "turncred"];

// turn:host[:port][?transport=udp|tcp] or turns:… ; no spaces, no userinfo (credentials go in their own fields)
const TURN_URL = /^turns?:(\[[0-9a-f:.]+\]|[a-z0-9.-]+)(:\d{1,5})?(\?transport=(udp|tcp))?$/i;
export function validTurnUrl(u) { return typeof u === "string" && TURN_URL.test(u.trim()); }

// { urls, username, credential, force } from loose input; null when there is no usable URL.
// Invalid URLs are dropped (and reported in .bad) instead of failing the whole configuration,
// because RTCPeerConnection throws on a malformed ICE server and the room would not open at all.
export function normTurn(t) {
  if (!t) return null;
  const list = (Array.isArray(t.urls) ? t.urls : String(t.urls || "").split(",")).map((u) => String(u).trim()).filter(Boolean);
  const urls = list.filter(validTurnUrl), bad = list.filter((u) => !validTurnUrl(u));
  if (!urls.length) return bad.length ? { urls: [], bad } : null;
  const out = { urls, username: String(t.username || ""), credential: String(t.credential || ""), force: !!t.force };
  if (bad.length) out.bad = bad;
  return out;
}

// params: URLSearchParams of the page; stored: the saved setting's JSON string (or null)
export function turnFrom(params, stored) {
  const q = params?.get("turn");
  if (q) return { ...normTurn({ urls: q, username: params.get("turnuser"), credential: params.get("turncred"), force: params.get("relay") === "1" }), from: "url" };
  let s = null;
  try { s = stored ? JSON.parse(stored) : null; } catch { s = null; }
  const n = normTurn(s);
  if (n && n.urls.length) {
    if (params?.get("relay") === "1") n.force = true;
    return { ...n, from: "saved" };
  }
  return n && n.bad ? { ...n, from: "saved" } : null;
}

// The RTCConfiguration PeerJS passes to every RTCPeerConnection. `extra`: window.TURN_SERVERS
// and/or the default relay's servers; opts.force: relay only (?relay=1) whatever the relay's source.
export function iceConfig(turn, extra = [], opts = {}) {
  const servers = [STUN];
  if (turn && turn.urls && turn.urls.length) {
    const s = { urls: turn.urls };
    if (turn.username) s.username = turn.username;
    if (turn.credential) s.credential = turn.credential;
    servers.push(s);
  }
  if (Array.isArray(extra)) servers.push(...extra);
  const relayOnly = !!((turn && turn.force) || opts.force) && servers.some(isRelayServer);
  return relayOnly ? { iceServers: servers, iceTransportPolicy: "relay" } : { iceServers: servers };
}

// a join link: the page's query minus the relay's credentials
export function shareQuery(search) {
  const q = new URLSearchParams(search);
  for (const k of SECRET_PARAMS) q.delete(k);
  q.delete("relay");
  return q;
}

// "relay" when the connection's selected candidate pair goes through a TURN server, "direct"
// when it does not, null before a pair is chosen. stats: an RTCStatsReport (or any iterable of
// stats objects / a Map).
export function linkPath(stats) {
  const p = selectedPair(stats);
  if (!p) return null;
  return (p.l?.candidateType === "relay" || p.r?.candidateType === "relay") ? "relay" : "direct";
}
// how this device reaches its relay on that link: "udp", "tcp" or "tls" (null: not relayed here)
export function linkRelayProtocol(stats) {
  const p = selectedPair(stats);
  return p?.l?.candidateType === "relay" ? (p.l.relayProtocol || null) : null;
}
function selectedPair(stats) {
  const all = new Map();
  for (const s of (stats?.values ? stats.values() : stats || [])) all.set(s.id, s);
  let pair = null;
  for (const s of all.values()) if (s.type === "transport" && s.selectedCandidatePairId) pair = all.get(s.selectedCandidatePairId) || pair;
  if (!pair) for (const s of all.values()) if (s.type === "candidate-pair" && (s.selected || s.nominated) && s.state === "succeeded") { pair = s; break; }
  if (!pair) return null;
  const l = all.get(pair.localCandidateId), r = all.get(pair.remoteCandidateId);
  return l || r ? { l, r } : null;
}

const ICE_URL = /^(stun|turns?):(\[[0-9a-f:.]+\]|[a-z0-9.-]+)(:\d{1,5})?(\?transport=(udp|tcp))?$/i;
const isRelayServer = (s) => [].concat(s?.urls || []).some((u) => /^turns?:/i.test(String(u)));
export { isRelayServer };

// --- the default relay (api/turn.mjs) ---
export const RELAY_FETCH_MS = 2500;
export const RELAY_ENDPOINT = "/api/turn";

// Ask the deployment's /api/turn for a relay? Only when nothing else supplies one (a relay the user
// set, or window.TURN_SERVERS, wins and nothing is fetched), not with ?relay=0, and not on a local
// dev server unless ?turnapi=1.
export function wantDefaultRelay(params, hostname, turn, extra) {
  if (turn && turn.urls && turn.urls.length) return false;
  if (Array.isArray(extra) && extra.length) return false;
  if (params?.get("relay") === "0") return false;
  if (params?.get("turnapi") === "1") return true;
  if (params?.get("turnapi") === "0") return false;
  return !/^(localhost|127\.\d+\.\d+\.\d+|\[::1\]|::1|)$/.test(String(hostname || ""));
}

// Validate what /api/turn sent before it reaches RTCPeerConnection (which throws on a bad server,
// and the room would not open at all): well-formed URLs only, a TURN entry only with its credential.
export function cleanRelayServers(list) {
  const out = [];
  for (const s of Array.isArray(list) ? list : []) {
    if (!s || typeof s !== "object") continue;
    const urls = [].concat(s.urls || []).map(String).filter((u) => ICE_URL.test(u));
    if (!urls.length) continue;
    if (urls.some((u) => /^turns?:/i.test(u))) {
      if (typeof s.username !== "string" || typeof s.credential !== "string" || !s.username || !s.credential) continue;
      out.push({ urls, username: s.username, credential: s.credential });
    } else out.push({ urls });
  }
  return out.some(isRelayServer) ? out : [];
}

// POST the endpoint; -> { iceServers, ttl, expiresAt } or null (not configured, refused, slow, junk).
// Never throws: a missing relay must not stop a room from opening.
export async function fetchRelay(fetchImpl, url = RELAY_ENDPOINT, timeoutMs = RELAY_FETCH_MS) {
  if (typeof fetchImpl !== "function") return null;
  const ac = typeof AbortController === "function" ? new AbortController() : null;
  let timer;
  try {
    const got = await Promise.race([
      fetchImpl(url, { method: "POST", cache: "no-store", credentials: "same-origin", signal: ac?.signal }),
      new Promise((_, rej) => { timer = setTimeout(() => { try { ac?.abort(); } catch {} rej(new Error("timeout")); }, timeoutMs); }),
    ]);
    if (!got || got.status !== 200) return null;
    const j = await got.json();
    const iceServers = cleanRelayServers(j?.iceServers);
    if (!iceServers.length) return null;
    const ttl = Number.isFinite(+j.ttl) && +j.ttl > 0 ? +j.ttl : 3600;
    return { iceServers, ttl, expiresAt: Number.isFinite(+j.expiresAt) ? +j.expiresAt : null, provider: typeof j.provider === "string" ? j.provider : "" };
  } catch { return null; }
  finally { clearTimeout(timer); }
}

// when to fetch fresh credentials: at 3/4 of their lifetime (new links made after that use them),
// never sooner than a minute
export const refreshInMs = (ttl) => Math.max(60000, Math.floor((+ttl || 3600) * 750));

// Put new relay servers into the RTCConfiguration object PeerJS holds (it keeps the same object
// and reads it for every new RTCPeerConnection): the old relay entries out, the new ones in.
// (the default relay's entries carry a hidden `auto` mark, so a refresh replaces only those)
export const markAuto = (list) => (list || []).map((s) => Object.defineProperty({ ...s }, "auto", { value: true, enumerable: false }));
export function swapRelayServers(cfg, fresh) {
  if (!cfg || !Array.isArray(cfg.iceServers) || !fresh?.length) return cfg;
  const keep = cfg.iceServers.filter((s) => !s?.auto);
  cfg.iceServers.length = 0;
  cfg.iceServers.push(...keep, ...markAuto(fresh));
  return cfg;
}

// --- relayed links and model weights ---
// Devices can take model layers from another device's cache instead of the network (peer weights:
// several GB per device). Relayed bytes cost the relay's owner money, so a link that goes through
// the relay never carries weights: the device fetches them from the model host instead, and only
// the token traffic (KB per hop) uses the relay. Direct links still share weights.
// path: linkPath()'s answer for that link (null: not known yet); relayPossible: this page has a
// relay configured at all. Unknown with a relay possible counts as relayed (the safe side).
export function weightsOverLink(path, relayPossible) {
  if (path === "direct") return true;
  if (path === "relay") return false;
  return !relayPossible;
}

// --- is UDP getting out? ---
// Candidate lines gathered with STUN only -> what they say about this network: srflx means a UDP
// packet reached the STUN server and came back (the internet takes UDP); host only means UDP out
// is blocked (or STUN is), and without a relay only devices on the same network can connect.
export function classifyCandidates(cands) {
  const types = new Set();
  for (const c of cands || []) {
    const line = typeof c === "string" ? c : c?.candidate || "";
    const m = / typ (host|srflx|prflx|relay)\b/.exec(line);
    if (m && / udp /i.test(line)) types.add(m[1]);
  }
  return { udpOut: types.has("srflx") || types.has("relay") || types.has("prflx"), host: types.has("host") };
}

// Gather candidates on a throwaway connection (STUN only) for up to timeoutMs. -> { udpOut, host }
// or null when it can't tell (no RTCPeerConnection, an error). Costs a few STUN packets.
export async function probeUdp(PC, stun = STUN, timeoutMs = 4000) {
  if (typeof PC !== "function") return null;
  let pc;
  try {
    pc = new PC({ iceServers: [stun] });
    const cands = [];
    pc.createDataChannel("probe");
    const done = new Promise((res) => {
      const t = setTimeout(res, timeoutMs);
      pc.onicecandidate = (e) => {
        if (!e.candidate) { clearTimeout(t); res(); return; }
        cands.push(e.candidate.candidate);
        if (/ typ srflx /.test(e.candidate.candidate)) { clearTimeout(t); res(); }
      };
    });
    await pc.setLocalDescription(await pc.createOffer());
    await done;
    return classifyCandidates(cands);
  } catch { return null; }
  finally { try { pc?.close(); } catch {} }
}

export const WORK_DOCS = "https://github.com/Nehanth/pooled/blob/main/docs/rooms-at-work.md";

// What to tell someone whose network looks closed. udp: probeUdp()'s answer (null: unknown);
// relay: this page has a relay. null when there is nothing to say.
export function networkAdvice(udp, relay) {
  if (!udp || udp.udpOut) return null;
  if (relay) return { level: "info", text: "This network blocks direct (UDP) connections, so links to devices on other networks go through the relay." };
  return { level: "warn", text: "This network seems to block direct (UDP) connections and no relay (TURN) server is set, so only devices on this same network can connect. Add a relay under Network, or see Rooms at work." };
}
