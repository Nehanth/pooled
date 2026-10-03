// Rooms on a home network, served from one computer (npm run lan, scripts/lan.mjs).
//
// The host opens the page as http://localhost:… : browsers treat localhost as secure, so WebGPU works.
// Other devices on the Wi-Fi can't reach "localhost"; they need the computer's network address. So the
// host's page is opened with ?invite=http://<that address>:<port>, and its invite links and QR codes
// use that origin instead of its own.
//
// It is honored only when the page itself is on loopback and only for a private-network address. The
// invite link carries the room's invite key (#k=), so a crafted link like pooled.run/room?invite=
// https://evil.example must not be able to send that key off to another site.

const LOOPBACK = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);

// 10/8, 172.16/12, 192.168/16, 100.64/10 (carrier-grade NAT, also Tailscale), and mDNS names (*.local)
function privateHost(h) {
  if (/^[a-z0-9-]+(\.[a-z0-9-]+)*\.local$/i.test(h)) return true;
  const m = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(h);
  if (!m) return false;
  const [a, b, c, d] = m.slice(1).map(Number);
  if ([a, b, c, d].some((x) => x > 255)) return false;
  return a === 10 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || (a === 100 && b >= 64 && b <= 127);
}

// ?invite= value + this page's hostname -> the origin for invite links ("http://192.168.1.20:8080"), or
// null to keep the page's own
export function inviteOrigin(param, pageHost) {
  if (!param || !LOOPBACK.has(String(pageHost))) return null;
  let u;
  try { u = new URL(String(param)); } catch { return null; }
  if (u.protocol !== "http:" && u.protocol !== "https:") return null;
  if (u.username || u.password || !privateHost(u.hostname)) return null;
  return u.origin;
}
