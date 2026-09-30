// api/turn.mjs: the Vercel Function that hands the room page short-lived TURN credentials.
// Both providers with mocked HTTP, the origin check, the per-IP limit, and "not configured".
import { handle, mint, providerOf, coturnUrls, coturnCredential, cleanIceServers, originAllowed, makeLimiter, ttlFrom } from "../../api/turn.mjs";

function eq(a, b, m) { const x = JSON.stringify(a), y = JSON.stringify(b); if (x !== y) throw new Error(`${m ? m + ": " : ""}${x} !== ${y}`); }
function ok(c, m) { if (!c) throw new Error(m || "assertion failed"); }

const CF = { TURN_KEY_ID: "key123", TURN_KEY_API_TOKEN: "tok-secret" };
const CT = { TURN_SECRET: "north", TURN_HOST: "relay.example.org" };
const req = (method = "POST", origin = "https://pooled.run", url = "https://pooled.run/api/turn", extra = {}) =>
  new Request(url, { method, headers: { ...(origin ? { origin } : {}), "x-forwarded-for": "203.0.113.9", ...extra } });

// Cloudflare's documented answer (developers.cloudflare.com/realtime/turn/generate-credentials/)
const CF_ANSWER = { iceServers: [
  { urls: ["stun:stun.cloudflare.com:3478", "stun:stun.cloudflare.com:53"] },
  { urls: ["turn:turn.cloudflare.com:3478?transport=udp", "turn:turn.cloudflare.com:53?transport=udp", "turn:turn.cloudflare.com:3478?transport=tcp",
    "turn:turn.cloudflare.com:80?transport=tcp", "turns:turn.cloudflare.com:5349?transport=tcp", "turns:turn.cloudflare.com:443?transport=tcp"],
    username: "cfuser", credential: "cfcred" },
] };
function cfFetch(answer = CF_ANSWER, status = 201, seen = []) {
  return async (url, init) => { seen.push({ url, init }); return new Response(JSON.stringify(answer), { status }); };
}

Deno.test("turn api: which provider the environment configures", () => {
  eq(providerOf({}), null);
  eq(providerOf({ TURN_KEY_ID: "x" }), null, "a key id without its token");
  eq(providerOf(CF), "cloudflare");
  eq(providerOf(CT), "coturn");
  eq(providerOf({ TURN_SECRET: "s" }), null, "a secret without URLs");
  eq(providerOf({ ...CF, ...CT }), "cloudflare", "Cloudflare wins when both are set");
  eq([ttlFrom({}), ttlFrom({ TURN_TTL: "60" }), ttlFrom({ TURN_TTL: "999999" }), ttlFrom({ TURN_TTL: "7200" })], [21600, 600, 86400, 7200]);
});

Deno.test("turn api: Cloudflare: the API call, port 53 dropped, TLS 443 kept", async () => {
  const seen = [];
  const got = await mint({ ...CF, TURN_TTL: "3600" }, { fetchImpl: cfFetch(CF_ANSWER, 201, seen) });
  eq(seen.length, 1);
  eq(seen[0].url, "https://rtc.live.cloudflare.com/v1/turn/keys/key123/credentials/generate-ice-servers");
  eq(seen[0].init.method, "POST");
  eq(seen[0].init.headers.Authorization, "Bearer tok-secret");
  eq(JSON.parse(seen[0].init.body), { ttl: 3600 });
  eq(got.provider, "cloudflare"); eq(got.ttl, 3600);
  eq(got.iceServers[0], { urls: ["stun:stun.cloudflare.com:3478"] });
  const t = got.iceServers[1];
  eq(t.username, "cfuser"); eq(t.credential, "cfcred");
  ok(!t.urls.some((u) => /:53(\?|$)/.test(u)), "port 53 is blocked by browsers");
  ok(t.urls.includes("turns:turn.cloudflare.com:5349?transport=tcp"), "5349 is not 53");
  for (const u of ["turn:turn.cloudflare.com:3478?transport=udp", "turn:turn.cloudflare.com:3478?transport=tcp", "turns:turn.cloudflare.com:443?transport=tcp"]) ok(t.urls.includes(u), "missing " + u);
});

Deno.test("turn api: Cloudflare failures are a 502 without the token in it", async () => {
  const errs = []; const orig = console.error; console.error = (...a) => errs.push(a.join(" "));
  try {
    for (const f of [cfFetch({}, 401), cfFetch({ iceServers: [{ urls: ["stun:stun.cloudflare.com:3478"] }] }), cfFetch({ iceServers: [{ urls: ["turn:t:1"] }] }), async () => { throw new Error("offline"); }]) {
      const r = await handle(req(), { env: CF, fetchImpl: f });
      eq(r.status, 502);
      const body = await r.text();
      ok(!body.includes("tok-secret") && !body.includes("key123"), body);
    }
    ok(errs.every((e) => !e.includes("tok-secret")), "logs never carry the token");
  } finally { console.error = orig; }
});

Deno.test("turn api: coturn use-auth-secret: expiry username, HMAC-SHA1 password, UDP/TCP/TLS URLs", async () => {
  eq(coturnUrls(CT), ["turn:relay.example.org:3478?transport=udp", "turn:relay.example.org:3478?transport=tcp", "turns:relay.example.org:443?transport=tcp"]);
  eq(coturnUrls({ TURN_URLS: "turn:a.b:3478, bogus, turns:a.b:443?transport=tcp" }), ["turn:a.b:3478", "turns:a.b:443?transport=tcp"]);
  eq(coturnUrls({ TURN_HOST: "bad host" }), []);
  // known answer: printf '1700003600:pooled' | openssl dgst -sha1 -hmac north -binary | base64
  const c = await coturnCredential("north", 3600, 1700000000 * 1000);
  eq(c.username, "1700003600:pooled");
  const want = "kZDDleexVEAG4pyL8thZV4ZZpWo=";
  eq(c.credential, want);
  const got = await mint({ ...CT, TURN_TTL: "3600" }, { now: 1700000000 * 1000 });
  eq(got, { iceServers: [{ urls: coturnUrls(CT), username: "1700003600:pooled", credential: want }], ttl: 3600, provider: "coturn" });
});

Deno.test("turn api: cleanIceServers drops junk and relays without credentials", () => {
  eq(cleanIceServers(null), []);
  eq(cleanIceServers({ urls: "turn:a:1", username: "u", credential: "c" }), [{ urls: ["turn:a:1"], username: "u", credential: "c" }], "a single object");
  eq(cleanIceServers([{ urls: ["turn:a:1"] }, 5, { urls: ["http://x"] }, { urls: "stun:s:3478" }]), [{ urls: ["stun:s:3478"] }]);
});

Deno.test("turn api: only this site's own pages (or listed origins) may ask", () => {
  eq(originAllowed("https://pooled.run", "pooled.run"), true);
  eq(originAllowed("https://pooled-git-x-nehanth.vercel.app", "pooled-git-x-nehanth.vercel.app"), true, "a preview is its own origin");
  eq(originAllowed("http://pooled.run", "pooled.run"), false, "plain http on a public host");
  eq(originAllowed("http://localhost:3000", "localhost:3000"), true, "vercel dev");
  eq(originAllowed("https://evil.example", "pooled.run"), false);
  eq(originAllowed("https://pooled.run.evil.example", "pooled.run"), false);
  eq(originAllowed(null, "pooled.run"), false); eq(originAllowed("null", "pooled.run"), false); eq(originAllowed("::", "pooled.run"), false);
  const env = { TURN_ALLOWED_ORIGINS: "https://pooled-*.vercel.app, https://docs.example.org" };
  eq(originAllowed("https://pooled-abc.vercel.app", "pooled.run", env), true);
  eq(originAllowed("https://pooled-a.b.vercel.app", "pooled.run", env), false, "* stays inside one label");
  eq(originAllowed("https://docs.example.org", "pooled.run", env), true);
  eq(originAllowed("https://xpooled-abc.vercel.app", "pooled.run", env), false);
});

Deno.test("turn api: the handler: methods, origins, not configured, success, CORS, no-store", async () => {
  eq((await handle(req("GET"), { env: CT })).status, 405, "GET: a browser always sends Origin on POST");
  eq((await handle(req("POST", "https://evil.example"), { env: CT })).status, 403);
  eq((await handle(req("POST", null), { env: CT })).status, 403, "no Origin (a plain script) is refused too");
  eq((await handle(req("OPTIONS", "https://evil.example"), { env: CT })).status, 403);
  const pre = await handle(req("OPTIONS"), { env: CT });
  eq(pre.status, 204); eq(pre.headers.get("access-control-allow-origin"), "https://pooled.run");
  const none = await handle(req(), { env: {} });
  eq(none.status, 204, "no provider: 204, the page works as before");
  const r = await handle(req(), { env: CT, now: 1700000000 * 1000 });
  eq(r.status, 200);
  eq(r.headers.get("cache-control"), "no-store");
  const j = await r.json();
  eq(j.provider, "coturn"); eq(j.ttl, 21600); eq(j.expiresAt, 1700000000 + 21600);
  eq(j.iceServers[0].urls.length, 3);
  ok(!JSON.stringify(j).includes("north"), "the secret never leaves");
  const c = await handle(req(), { env: CF, fetchImpl: cfFetch() });
  eq(c.status, 200); eq((await c.json()).provider, "cloudflare");
});

Deno.test("turn api: per-IP rate limit", async () => {
  const lim = makeLimiter(3, 60000);
  eq([lim("a", 0), lim("a", 1), lim("a", 2), lim("a", 3), lim("b", 3)], [true, true, true, false, true]);
  eq(lim("a", 60001), true, "a new window");
  const l2 = makeLimiter(1);
  eq((await handle(req(), { env: CT, limiter: l2 })).status, 200);
  eq((await handle(req(), { env: CT, limiter: l2 })).status, 429);
  eq((await handle(req("POST", "https://pooled.run", "https://pooled.run/api/turn", { "x-forwarded-for": "198.51.100.1" }), { env: CT, limiter: l2 })).status, 200, "another IP");
});

Deno.test("turn api: the default export is a Vercel fetch handler", async () => {
  const mod = await import("../../api/turn.mjs");
  eq(typeof mod.default.fetch, "function");
});
