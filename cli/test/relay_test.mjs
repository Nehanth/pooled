import { test } from "node:test";
import assert from "node:assert/strict";
import { cleanServers, envServers, fetchRelay, iceSetup, refreshInMs, STUN, RELAY_API, CLIENT_HEADER } from "../lib/relay.js";

const CF = {
  iceServers: [
    { urls: ["stun:stun.cloudflare.com:3478"] },
    { urls: ["turn:turn.cloudflare.com:3478?transport=udp", "turns:turn.cloudflare.com:443?transport=tcp"], username: "u1", credential: "c1" },
  ],
  ttl: 21600, provider: "cloudflare",
};
const answer = (body, status = 200) => async () => ({ status, json: async () => body });

test("relay: cleanServers keeps well-formed servers and needs a TURN entry with credentials", () => {
  assert.deepEqual(cleanServers(CF.iceServers), CF.iceServers);
  assert.deepEqual(cleanServers([{ urls: "turn:x.org:3478" }]), [], "a TURN server without credentials is dropped");
  assert.deepEqual(cleanServers([{ urls: ["stun:x.org"] }]), [], "STUN alone is no relay");
  assert.deepEqual(cleanServers([{ urls: ["turn:bad host"], username: "a", credential: "b" }, null, 3, "x"]), []);
  assert.deepEqual(cleanServers("nope"), []);
});

test("relay: POOLED_TURN is a relay you run yourself", () => {
  assert.deepEqual(envServers({ POOLED_TURN: "turn:relay.example.org:3478, turns:relay.example.org:443?transport=tcp", POOLED_TURN_USER: "me", POOLED_TURN_CRED: "pw" }),
    [{ urls: ["turn:relay.example.org:3478", "turns:relay.example.org:443?transport=tcp"], username: "me", credential: "pw" }]);
  assert.deepEqual(envServers({ POOLED_TURN: "stun:x.org,http://x" }), [], "only turn: / turns: URLs");
  assert.deepEqual(envServers({}), []);
});

test("relay: fetchRelay posts with the client tag and never throws", async () => {
  let seen = null;
  const got = await fetchRelay({ fetchImpl: async (url, o) => { seen = { url, o }; return { status: 200, json: async () => CF }; }, client: "pooled-cli/9" });
  assert.equal(seen.url, RELAY_API);
  assert.equal(seen.o.method, "POST");
  assert.equal(seen.o.headers[CLIENT_HEADER], "pooled-cli/9");
  assert.deepEqual(got, { iceServers: CF.iceServers, ttl: 21600, provider: "cloudflare" });
  assert.equal(await fetchRelay({ fetchImpl: answer(null, 204) }), null, "not configured");
  assert.equal(await fetchRelay({ fetchImpl: answer({ error: "x" }, 403) }), null, "refused");
  assert.equal(await fetchRelay({ fetchImpl: async () => { throw new Error("offline"); } }), null, "offline");
  assert.equal(await fetchRelay({ fetchImpl: answer({ iceServers: [{ urls: "stun:x.org" }] }) }), null, "no relay in it");
  const slow = (url, o) => new Promise((_, rej) => o.signal.addEventListener("abort", () => rej(new Error("aborted"))));
  const t0 = Date.now();
  assert.equal(await fetchRelay({ fetchImpl: slow, timeoutMs: 50 }), null, "too slow");
  assert.ok(Date.now() - t0 < 1000);
});

test("relay: iceSetup: STUN first, the fetched relay after it, refreshed in place", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let calls = 0;
  const fetchImpl = async () => { calls++; return { status: 200, json: async () => ({ ...CF, iceServers: [{ ...CF.iceServers[1], username: "u" + calls }] }) }; };
  const ice = await iceSetup({ env: {}, fetchImpl });
  assert.equal(ice.relay, "cloudflare");
  assert.deepEqual(ice.config.iceServers[0], STUN);
  assert.equal(ice.config.iceServers[1].username, "u1");
  assert.equal(ice.config.iceTransportPolicy, undefined, "direct links first");
  const cfg = ice.config;
  t.mock.timers.tick(refreshInMs(21600));
  await new Promise((r) => setImmediate(r));
  assert.equal(calls, 2);
  assert.equal(ice.config, cfg, "the same object PeerJS holds");
  assert.equal(cfg.iceServers.length, 2);
  assert.equal(cfg.iceServers[1].username, "u2", "new credentials swapped in");
  ice.stop();
});

test("relay: iceSetup modes: off, forced, your own relay, none available", async () => {
  let fetched = 0;
  const fetchImpl = async () => { fetched++; return { status: 200, json: async () => CF }; };
  const off = await iceSetup({ env: { POOLED_RELAY: "0" }, fetchImpl });
  assert.deepEqual(off.config, { iceServers: [STUN] }); assert.equal(off.relay, null); assert.equal(fetched, 0);
  const forced = await iceSetup({ env: { POOLED_RELAY: "1" }, fetchImpl });
  assert.equal(forced.config.iceTransportPolicy, "relay"); forced.stop();
  const own = await iceSetup({ env: { POOLED_TURN: "turn:relay.example.org:3478", POOLED_TURN_USER: "a", POOLED_TURN_CRED: "b" }, fetchImpl });
  assert.equal(own.relay, "env"); assert.equal(fetched, 1, "your own relay: nothing fetched");
  assert.deepEqual(own.config.iceServers[1], { urls: ["turn:relay.example.org:3478"], username: "a", credential: "b" });
  const logs = [];
  const none = await iceSetup({ env: { POOLED_RELAY: "1" }, fetchImpl: answer(null, 204), log: (s) => logs.push(s) });
  assert.deepEqual(none.config, { iceServers: [STUN] }, "no relay: never relay-only (no link could open)");
  assert.match(logs[0], /no relay/);
  let url = null;
  await iceSetup({ env: { POOLED_TURN_API: "https://my.example/api/turn" }, fetchImpl: async (u) => { url = u; return { status: 204 }; } });
  assert.equal(url, "https://my.example/api/turn");
});
