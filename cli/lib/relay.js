// ICE servers for the CLI's WebRTC links (pooled chat / serve, and pooled host / join's room node): public
// STUN always, plus a TURN relay for when no direct path exists (symmetric NAT or CGNAT on both sides, a
// network that blocks UDP). The room page does the same in room/ice.js; this is its Node side, kept free
// of room/ imports because the npm package ships lib/ as is.
//
// Where a relay comes from, first match wins:
//   1. POOLED_TURN=turn:host:3478 (comma-separate several URLs), with POOLED_TURN_USER and POOLED_TURN_CRED
//      for its long-term credentials: a relay you run yourself. Nothing is fetched.
//   2. pooled.run's relay (api/turn.mjs, or POOLED_TURN_API=<url> for another deployment's): short-lived
//      credentials, fetched once and refreshed at 3/4 of their lifetime. No answer within RELAY_FETCH_MS,
//      or none configured, means STUN only, as before.
// POOLED_RELAY=0 skips the relay; POOLED_RELAY=1 sends every link through it (to test it).
// ICE prefers direct candidates (host, then STUN), so the relay only carries a link no direct path works for.

export const STUN = { urls: ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"] };
export const RELAY_API = "https://pooled.run/api/turn";
export const RELAY_FETCH_MS = 2500;
// api/turn.mjs lets a request with no Origin through only with this header (a browser on another site
// always sends Origin, so this is not a way for other pages to mint credentials)
export const CLIENT_HEADER = "X-Pooled-Client";

const ICE_URL = /^(stun|turns?):(\[[0-9a-f:.]+\]|[a-z0-9.-]+)(:\d{1,5})?(\?transport=(udp|tcp))?$/i;
const isTurn = (u) => /^turns?:/i.test(String(u));
export const isRelayServer = (s) => [].concat(s?.urls || []).some(isTurn);

// What a relay endpoint sent, made safe for RTCPeerConnection (which throws on a malformed server and the
// link would never open): well-formed URLs only, a TURN entry only with its credentials. [] when no relay.
export function cleanServers(list) {
  const out = [];
  for (const s of Array.isArray(list) ? list : []) {
    if (!s || typeof s !== "object") continue;
    const urls = [].concat(s.urls || []).map(String).filter((u) => ICE_URL.test(u));
    if (!urls.length) continue;
    if (urls.some(isTurn)) {
      if (typeof s.username !== "string" || typeof s.credential !== "string" || !s.username || !s.credential) continue;
      out.push({ urls, username: s.username, credential: s.credential });
    } else out.push({ urls });
  }
  return out.some(isRelayServer) ? out : [];
}

// POOLED_TURN and its credentials -> [server] or [] (bad URLs dropped)
export function envServers(env = process.env) {
  const urls = String(env.POOLED_TURN || "").split(",").map((u) => u.trim()).filter((u) => ICE_URL.test(u) && isTurn(u));
  if (!urls.length) return [];
  const s = { urls };
  if (env.POOLED_TURN_USER) s.username = String(env.POOLED_TURN_USER);
  if (env.POOLED_TURN_CRED) s.credential = String(env.POOLED_TURN_CRED);
  return [s];
}

// POST the relay endpoint -> { iceServers, ttl, provider } or null (not configured, refused, slow, junk).
// Never throws: a missing relay must not stop a room from opening.
export async function fetchRelay({ fetchImpl = globalThis.fetch, url = RELAY_API, timeoutMs = RELAY_FETCH_MS, client = "pooled-cli" } = {}) {
  if (typeof fetchImpl !== "function") return null;
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);
  try {
    const got = await fetchImpl(url, { method: "POST", headers: { [CLIENT_HEADER]: client }, signal: ac.signal });
    if (!got || got.status !== 200) return null;
    const j = await got.json();
    const iceServers = cleanServers(j?.iceServers);
    if (!iceServers.length) return null;
    const ttl = Number.isFinite(+j.ttl) && +j.ttl > 0 ? +j.ttl : 3600;
    return { iceServers, ttl, provider: typeof j.provider === "string" ? j.provider : "" };
  } catch { return null; }
  finally { clearTimeout(timer); }
}

export const refreshInMs = (ttl) => Math.max(60000, Math.floor((+ttl || 3600) * 750));

// -> { config, relay, stop() }. config is the RTCConfiguration to give PeerJS (it keeps the object and reads
// it for every new link, so a refresh swaps the fetched entries in place); relay says where the relay came
// from ("env", the provider's name, or null for STUN only).
export async function iceSetup({ env = process.env, fetchImpl = globalThis.fetch, client = "pooled-cli", log = () => {} } = {}) {
  const config = { iceServers: [STUN] };
  const mode = String(env.POOLED_RELAY ?? "");
  let relay = null, timer = null;
  if (mode !== "0") {
    const mine = envServers(env);
    if (mine.length) { config.iceServers.push(...mine); relay = "env"; }
    else {
      const url = env.POOLED_TURN_API || RELAY_API;
      const got = await fetchRelay({ fetchImpl, url, client });
      if (got) {
        config.iceServers.push(...got.iceServers);
        relay = got.provider || "relay";
        const refresh = async () => {
          const fresh = await fetchRelay({ fetchImpl, url, client });
          if (fresh) config.iceServers.splice(1, config.iceServers.length - 1, ...fresh.iceServers);
          timer = setTimeout(refresh, refreshInMs(fresh?.ttl ?? 300));
          timer.unref?.();
        };
        timer = setTimeout(refresh, refreshInMs(got.ttl));
        timer.unref?.();
      }
    }
  }
  if (mode === "1") {
    if (relay) config.iceTransportPolicy = "relay";
    else log("POOLED_RELAY=1 but no relay is available: links go direct");
  }
  return { config, relay, stop: () => clearTimeout(timer) };
}
