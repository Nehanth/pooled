// POST /api/turn: short-lived TURN relay credentials for the room page (a Vercel Function).
//
// Rooms connect devices directly (WebRTC over UDP). Work and school networks often block UDP and
// every port but 443, and then no direct path exists. A TURN relay gets around that: the devices
// both reach the relay over TCP or TLS on 443 and it forwards between them. The page asks this
// endpoint for the relay's addresses and a credential that expires (TURN_TTL, default 6 h), so no
// long-lived secret ever ships in the page, a join link or a QR code.
//
// Providers, configured by environment variables in the Vercel project (first one set wins):
//   Cloudflare Realtime TURN   TURN_KEY_ID + TURN_KEY_API_TOKEN
//                              (https://developers.cloudflare.com/realtime/turn/)
//   coturn "use-auth-secret"   TURN_SECRET + TURN_URLS (comma list of turn:/turns: URLs) or TURN_HOST
//                              (then turn:HOST:3478 UDP + TCP and turns:HOST:443 TLS)
// With neither set the endpoint answers 204 and the page behaves as before (direct links only).
//
// Abuse: anyone who can call this gets a working relay credential until it expires, and relayed
// bytes cost money (Cloudflare: 1,000 GB a month free, then per GB). So:
//   - only pages on this deployment's own origin (or TURN_ALLOWED_ORIGINS) may ask: POST only, and
//     the browser's Origin header must match. That stops other websites from minting credentials
//     in their visitors' browsers; it does not stop a script (it can forge headers), which is what
//   - the per-IP limit is for: TURN_RATE per minute per IP (default 10), per function instance.
//     Instances are not shared, so this is a speed bump; add a Vercel Firewall rate-limit rule on
//     /api/turn for a hard limit (docs/rooms-at-work.md).
//   - credentials are short-lived (TURN_TTL seconds, 600..86400) and the page refreshes them.

const CF_API = "https://rtc.live.cloudflare.com/v1/turn/keys";
const DEFAULT_TTL = 6 * 3600;
const DEFAULT_RATE = 10;
const FETCH_MS = 4000;

const TURN_URL = /^(stun|turns?):(\[[0-9a-f:.]+\]|[a-z0-9.-]+)(:\d{1,5})?(\?transport=(udp|tcp))?$/i;

export function ttlFrom(env) {
  const t = parseInt(env.TURN_TTL, 10);
  return Number.isFinite(t) ? Math.min(86400, Math.max(600, t)) : DEFAULT_TTL;
}

// which provider the environment configures, or null
export function providerOf(env) {
  if (env.TURN_KEY_ID && env.TURN_KEY_API_TOKEN) return "cloudflare";
  if (env.TURN_SECRET && (env.TURN_URLS || env.TURN_HOST)) return "coturn";
  return null;
}

// The URLs a coturn relay is reached on: TURN_URLS as given, or the three a strict firewall needs
// from TURN_HOST: UDP 3478 (fastest), TCP 3478 (UDP blocked), TLS 443 (only 443 open, or a proxy
// that only passes TLS).
export function coturnUrls(env) {
  if (env.TURN_URLS) return String(env.TURN_URLS).split(",").map((u) => u.trim()).filter((u) => TURN_URL.test(u));
  const h = String(env.TURN_HOST || "").trim();
  if (!/^[a-z0-9.-]+$/i.test(h)) return [];
  return [`turn:${h}:3478?transport=udp`, `turn:${h}:3478?transport=tcp`, `turns:${h}:443?transport=tcp`];
}

// coturn's TURN REST API ("use-auth-secret"): username "<expiry unix time>:<tag>", password
// base64(HMAC-SHA1(secret, username)). coturn recomputes it and refuses one past its expiry.
export async function coturnCredential(secret, ttl, now = Date.now(), tag = "pooled") {
  const username = `${Math.floor(now / 1000) + ttl}:${tag}`;
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-1" }, false, ["sign"]);
  const mac = new Uint8Array(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(username)));
  let bin = ""; for (const b of mac) bin += String.fromCharCode(b);
  return { username, credential: btoa(bin) };
}

// Cloudflare's answer -> RTCIceServer list: keep well-formed stun:/turn:/turns: URLs, drop the
// port-53 ones (browsers block port 53; Cloudflare's docs say to leave them out).
export function cleanIceServers(list) {
  const out = [];
  for (const s of [].concat(list || [])) {
    if (!s || typeof s !== "object") continue;
    const urls = [].concat(s.urls || s.url || []).map(String).filter((u) => TURN_URL.test(u) && !/:53(\?|$)/.test(u));
    if (!urls.length) continue;
    const hasTurn = urls.some((u) => /^turns?:/i.test(u));
    if (hasTurn && !(s.username && s.credential)) continue;   // a relay without its credential is useless
    const e = { urls };
    if (hasTurn) { e.username = String(s.username); e.credential = String(s.credential); }
    out.push(e);
  }
  return out;
}

export async function cloudflareServers(env, ttl, fetchImpl = fetch) {
  const ac = new AbortController(), timer = setTimeout(() => ac.abort(), FETCH_MS);
  try {
    const r = await fetchImpl(`${CF_API}/${encodeURIComponent(env.TURN_KEY_ID)}/credentials/generate-ice-servers`, {
      method: "POST",
      headers: { Authorization: `Bearer ${env.TURN_KEY_API_TOKEN}`, "Content-Type": "application/json" },
      body: JSON.stringify({ ttl }),
      signal: ac.signal,
    });
    if (!r.ok) throw new Error(`cloudflare answered ${r.status}`);
    const j = await r.json();
    const servers = cleanIceServers(j.iceServers);
    if (!servers.some((s) => s.urls.some((u) => /^turns?:/i.test(u)))) throw new Error("cloudflare sent no relay");
    return servers;
  } finally { clearTimeout(timer); }
}

// -> { iceServers, ttl, provider } or null when no provider is configured
export async function mint(env, { fetchImpl = fetch, now = Date.now() } = {}) {
  const provider = providerOf(env), ttl = ttlFrom(env);
  if (provider === "cloudflare") return { iceServers: await cloudflareServers(env, ttl, fetchImpl), ttl, provider };
  if (provider === "coturn") {
    const urls = coturnUrls(env);
    if (!urls.length) throw new Error("TURN_URLS / TURN_HOST has no usable URL");
    return { iceServers: [{ urls, ...(await coturnCredential(env.TURN_SECRET, ttl, now)) }], ttl, provider };
  }
  return null;
}

// Origins allowed to ask: this deployment's own (any preview URL is its own origin) plus
// TURN_ALLOWED_ORIGINS (comma list; "*" matches within one host label: https://pooled-*.vercel.app).
export function originAllowed(origin, selfHost, env = {}) {
  if (!origin || origin === "null") return false;
  let o; try { o = new URL(origin); } catch { return false; }
  if (o.host === selfHost && (o.protocol === "https:" || /^(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/.test(o.host))) return true;
  for (const pat of String(env.TURN_ALLOWED_ORIGINS || "").split(",").map((s) => s.trim()).filter(Boolean)) {
    const re = new RegExp("^" + pat.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, "[a-z0-9-]*") + "$", "i");
    if (re.test(o.origin)) return true;
  }
  return false;
}

// per-IP fixed window, per instance (see the top of this file)
export function makeLimiter(limit = DEFAULT_RATE, windowMs = 60000) {
  const hits = new Map();
  return (ip, now = Date.now()) => {
    if (hits.size > 5000) for (const [k, v] of hits) if (now - v.t >= windowMs) hits.delete(k);
    const h = hits.get(ip);
    if (!h || now - h.t >= windowMs) { hits.set(ip, { t: now, n: 1 }); return true; }
    return ++h.n <= limit;
  };
}

const clientIp = (req) => (req.headers.get("x-real-ip") || req.headers.get("x-forwarded-for") || "").split(",")[0].trim() || "?";

export async function handle(req, { env = {}, fetchImpl = fetch, limiter = null, now = Date.now() } = {}) {
  const selfHost = new URL(req.url).host;
  const origin = req.headers.get("origin");
  const ok = originAllowed(origin, selfHost, env);
  const base = { "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", Vary: "Origin" };
  const cors = ok ? { "Access-Control-Allow-Origin": origin, "Access-Control-Allow-Methods": "POST", "Access-Control-Max-Age": "600" } : {};
  const json = (status, body) => new Response(body == null ? null : JSON.stringify(body), { status, headers: { ...base, ...cors, ...(body == null ? {} : { "Content-Type": "application/json" }) } });
  if (req.method === "OPTIONS") return ok ? json(204, null) : json(403, { error: "origin not allowed" });
  if (req.method !== "POST") return new Response(JSON.stringify({ error: "POST only" }), { status: 405, headers: { ...base, Allow: "POST, OPTIONS", "Content-Type": "application/json" } });
  if (!ok) return json(403, { error: "origin not allowed" });
  if (!providerOf(env)) return json(204, null);
  if (limiter && !limiter(clientIp(req), now)) return json(429, { error: "too many requests" });
  try {
    const got = await mint(env, { fetchImpl, now });
    return json(200, { ...got, expiresAt: Math.floor(now / 1000) + got.ttl });
  } catch (err) {
    console.error("turn:", err?.message || err);   // never the secret: messages above name no env value
    return json(502, { error: "relay credentials unavailable" });
  }
}

let limiter = null;
export default {
  fetch(request) {
    const env = globalThis.process?.env || {};
    limiter ||= makeLimiter(parseInt(env.TURN_RATE, 10) > 0 ? parseInt(env.TURN_RATE, 10) : DEFAULT_RATE);
    return handle(request, { env, limiter });
  },
};
