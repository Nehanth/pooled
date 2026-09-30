// ICE configuration for the room's WebRTC links: public STUN always, plus an optional TURN relay
// for networks where a direct connection cannot be made (symmetric NAT / CGNAT on both sides, a
// firewall that blocks UDP). Off by default: no relay server or credentials ship with Pooled.
//
// Where a relay comes from, first match wins:
//   1. the page URL: ?turn=turn:relay.example.org:3478 (comma-separate several URLs), with
//      ?turnuser= and ?turncred= for its long-term credentials, ?relay=1 to use only the relay
//   2. this browser's saved setting (the "Network" box under the join form), localStorage TURN_KEY
//   3. window.TURN_SERVERS, an RTCIceServer array a self-hosted deployment can define before
//      room.js loads
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

// The RTCConfiguration PeerJS passes to every RTCPeerConnection. `extra`: window.TURN_SERVERS.
export function iceConfig(turn, extra = []) {
  const servers = [STUN];
  if (turn && turn.urls && turn.urls.length) {
    const s = { urls: turn.urls };
    if (turn.username) s.username = turn.username;
    if (turn.credential) s.credential = turn.credential;
    servers.push(s);
  }
  if (Array.isArray(extra)) servers.push(...extra);
  const relayOnly = !!(turn && turn.force) && servers.length > 1;
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
  const all = new Map();
  for (const s of (stats?.values ? stats.values() : stats || [])) all.set(s.id, s);
  let pair = null;
  for (const s of all.values()) if (s.type === "transport" && s.selectedCandidatePairId) pair = all.get(s.selectedCandidatePairId) || pair;
  if (!pair) for (const s of all.values()) if (s.type === "candidate-pair" && (s.selected || s.nominated) && s.state === "succeeded") { pair = s; break; }
  if (!pair) return null;
  const l = all.get(pair.localCandidateId), r = all.get(pair.remoteCandidateId);
  if (!l && !r) return null;
  return (l?.candidateType === "relay" || r?.candidateType === "relay") ? "relay" : "direct";
}
