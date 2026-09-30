// room/ice.js: the optional TURN relay. Off unless configured; URL beats the saved setting;
// malformed URLs are dropped (RTCPeerConnection would throw on them); join links never carry
// the relay's credentials; linkPath reads direct vs relayed from getStats().
import { turnFrom, iceConfig, normTurn, validTurnUrl, shareQuery, linkPath, STUN, TURN_KEY } from "../../room/ice.js";

function eq(a, b, m) { const x = JSON.stringify(a), y = JSON.stringify(b); if (x !== y) throw new Error(`${m ? m + ": " : ""}${x} !== ${y}`); }
const P = (s) => new URLSearchParams(s);

Deno.test("ice: no relay configured means STUN only, no transport policy", () => {
  eq(turnFrom(P(""), null), null);
  eq(iceConfig(turnFrom(P(""), null)), { iceServers: [STUN] });
  eq(iceConfig(null, []), { iceServers: [STUN] });
  eq(TURN_KEY, "pooled-turn");
});

Deno.test("ice: turn URL validation", () => {
  const good = ["turn:relay.example.org:3478", "turns:relay.example.org:5349?transport=tcp", "turn:10.0.0.2", "TURN:Relay.Example.org:80?transport=udp", "turn:[2001:db8::1]:3478", " turn:a.b:1 "];
  const bad = ["", "stun:stun.l.google.com:19302", "http://relay", "turn:", "turn:user:pass@relay:3478", "turn:relay:3478?transport=sctp", "turn:re lay:3478", "turn:relay:3478\nx", null, 42];
  for (const u of good) if (!validTurnUrl(u)) throw new Error("rejected " + u);
  for (const u of bad) if (validTurnUrl(u)) throw new Error("accepted " + JSON.stringify(u));
});

Deno.test("ice: ?turn= with credentials, several URLs, bad ones dropped and reported", () => {
  const t = turnFrom(P("turn=turn:r.example:3478,turns:r.example:5349?transport=tcp,bogus&turnuser=ann&turncred=s3cret"), null);
  eq(t, { urls: ["turn:r.example:3478", "turns:r.example:5349?transport=tcp"], username: "ann", credential: "s3cret", force: false, bad: ["bogus"], from: "url" });
  eq(iceConfig(t), { iceServers: [STUN, { urls: t.urls, username: "ann", credential: "s3cret" }] });
});

Deno.test("ice: ?relay=1 forces the relay, but only when there is one", () => {
  const t = turnFrom(P("turn=turn:r:3478&relay=1"), null);
  eq(iceConfig(t).iceTransportPolicy, "relay");
  eq(iceConfig(turnFrom(P("relay=1"), null)), { iceServers: [STUN] }, "relay-only with no relay would never connect");
  // window.TURN_SERVERS counts as a relay for the policy
  eq(iceConfig({ urls: [], force: true }, [{ urls: "turn:x:1" }]).iceTransportPolicy, "relay");
});

Deno.test("ice: the saved setting is used when the URL has none; the URL wins when both exist", () => {
  const saved = JSON.stringify({ urls: ["turn:saved.example:3478"], username: "u", credential: "p", force: true });
  eq(turnFrom(P(""), saved), { urls: ["turn:saved.example:3478"], username: "u", credential: "p", force: true, from: "saved" });
  eq(turnFrom(P("turn=turn:url.example:1"), saved).urls, ["turn:url.example:1"]);
  eq(turnFrom(P("relay=1"), JSON.stringify({ urls: "turn:s:1" })).force, true);
  for (const junk of ["{", "null", "42", '"turn:x:1"', "[]", JSON.stringify({ urls: [] })]) eq(turnFrom(P(""), junk), null, junk);
  eq(turnFrom(P(""), JSON.stringify({ urls: ["nope"] })), { urls: [], bad: ["nope"], from: "saved" });
  eq(iceConfig(turnFrom(P(""), JSON.stringify({ urls: ["nope"] }))), { iceServers: [STUN] }, "a bad saved URL never reaches RTCPeerConnection");
});

Deno.test("ice: normTurn coerces loose input; empty credentials are left out of the ICE server", () => {
  eq(normTurn(null), null); eq(normTurn({}), null); eq(normTurn({ urls: " , " }), null);
  eq(normTurn({ urls: "turn:a:1", username: 5, credential: null, force: "yes" }), { urls: ["turn:a:1"], username: "5", credential: "", force: true });
  eq(iceConfig({ urls: ["turn:a:1"], username: "", credential: "" }).iceServers[1], { urls: ["turn:a:1"] });
});

Deno.test("ice: join links drop turn, turnuser, turncred and relay, keep the rest", () => {
  const q = shareQuery("?signal=127.0.0.1:9000&turn=turn:r:1&turnuser=a&turncred=b&relay=1&wire=stripe2&code=OLD");
  eq([...q.keys()].sort(), ["code", "signal", "wire"]);
  eq(q.get("signal"), "127.0.0.1:9000");
  eq(shareQuery("").toString(), "");
});

Deno.test("ice: linkPath reads the selected candidate pair", () => {
  const mk = (lt, rt, via = "transport") => {
    const s = [
      { id: "L", type: "local-candidate", candidateType: lt }, { id: "R", type: "remote-candidate", candidateType: rt },
      { id: "CP", type: "candidate-pair", localCandidateId: "L", remoteCandidateId: "R", state: "succeeded", nominated: via !== "transport" },
      { id: "CP0", type: "candidate-pair", localCandidateId: "X", remoteCandidateId: "Y", state: "failed" },
    ];
    if (via === "transport") s.push({ id: "T", type: "transport", selectedCandidatePairId: "CP" });
    return new Map(s.map((x) => [x.id, x]));
  };
  eq(linkPath(mk("host", "srflx")), "direct");
  eq(linkPath(mk("relay", "host")), "relay");
  eq(linkPath(mk("srflx", "relay")), "relay");
  eq(linkPath(mk("prflx", "host", "pair")), "direct", "no transport stats: the nominated succeeded pair");
  eq(linkPath(new Map()), null); eq(linkPath(null), null);
  eq(linkPath([{ id: "T", type: "transport" }]), null, "no pair selected yet");
});

// --- the default relay (/api/turn), relayed links and weights, the UDP probe ---
import { wantDefaultRelay, cleanRelayServers, fetchRelay, refreshInMs, swapRelayServers, markAuto, weightsOverLink, classifyCandidates, networkAdvice, probeUdp, isRelayServer } from "../../room/ice.js";

Deno.test("ice: the default relay is asked for only when nothing else supplies one", () => {
  eq(wantDefaultRelay(P(""), "pooled.run", null, []), true);
  eq(wantDefaultRelay(P(""), "pooled.run", { urls: ["turn:mine:3478"] }, []), false, "the user's own relay wins");
  eq(wantDefaultRelay(P(""), "pooled.run", null, [{ urls: "turn:site:1" }]), false, "window.TURN_SERVERS wins");
  eq(wantDefaultRelay(P(""), "pooled.run", { urls: [], bad: ["x"] }, []), true, "a bad saved URL is no relay");
  eq(wantDefaultRelay(P("relay=0"), "pooled.run", null, []), false);
  eq(wantDefaultRelay(P(""), "localhost", null, []), false, "a static dev server has no /api");
  eq(wantDefaultRelay(P(""), "127.0.0.1", null, []), false);
  eq(wantDefaultRelay(P("turnapi=1"), "127.0.0.1", null, []), true);
  eq(wantDefaultRelay(P("turnapi=0"), "pooled.run", null, []), false);
});

Deno.test("ice: relay=1 forces relay-only for the default relay too; STUN-only never is", () => {
  const auto = markAuto([{ urls: ["turns:r.example:443?transport=tcp"], username: "u", credential: "c" }]);
  eq(iceConfig(null, auto, { force: true }).iceTransportPolicy, "relay");
  eq(iceConfig(null, auto).iceTransportPolicy, undefined, "direct preferred by default");
  eq(iceConfig(null, [{ urls: "stun:s:1" }], { force: true }).iceTransportPolicy, undefined, "relay-only with no relay would never connect");
  eq(isRelayServer({ urls: "turns:a:443" }), true); eq(isRelayServer({ urls: ["stun:a:1"] }), false);
});

Deno.test("ice: cleanRelayServers keeps well-formed servers, TURN only with its credential", () => {
  eq(cleanRelayServers(null), []); eq(cleanRelayServers({}), []);
  eq(cleanRelayServers([{ urls: ["stun:s:3478"] }]), [], "STUN alone is no relay");
  eq(cleanRelayServers([{ urls: ["stun:s:3478"] }, { urls: ["turn:t:3478?transport=udp", "javascript:alert(1)", "turns:t:443?transport=tcp"], username: "u", credential: "c", extra: 1 }]),
    [{ urls: ["stun:s:3478"] }, { urls: ["turn:t:3478?transport=udp", "turns:t:443?transport=tcp"], username: "u", credential: "c" }]);
  eq(cleanRelayServers([{ urls: ["turn:t:1"], username: "u" }]), [], "no credential");
  eq(cleanRelayServers([{ urls: ["turn:t:1"], username: 5, credential: {} }]), [], "non-string credentials");
});

Deno.test("ice: fetchRelay: 200 with servers, else null, and it never throws or hangs", async () => {
  const body = { iceServers: [{ urls: ["turn:t:3478"], username: "u", credential: "c" }], ttl: 600, expiresAt: 1234, provider: "coturn" };
  const res = (status, b) => async (url, init) => { eq(url, "/api/turn"); eq(init.method, "POST"); return new Response(b == null ? null : JSON.stringify(b), { status }); };
  eq(await fetchRelay(res(200, body)), { iceServers: body.iceServers, ttl: 600, expiresAt: 1234, provider: "coturn" });
  eq(await fetchRelay(res(204, null)), null, "not configured");
  eq(await fetchRelay(res(404, { x: 1 })), null, "no function (a static host)");
  eq(await fetchRelay(res(403, { error: "origin" })), null);
  eq(await fetchRelay(res(200, { iceServers: "nope" })), null);
  eq(await fetchRelay(async () => new Response("<html>", { status: 200 })), null, "an SPA fallback page");
  eq(await fetchRelay(async () => { throw new TypeError("offline"); }), null);
  eq(await fetchRelay(undefined), null);
  const t0 = Date.now();
  eq(await fetchRelay(() => new Promise(() => {}), "/api/turn", 150), null, "a hung endpoint");
  ok(Date.now() - t0 < 1000, "timed out fast");
  eq((await fetchRelay(res(200, { ...body, ttl: "x" }))).ttl, 3600, "a missing ttl gets a default");
});
function ok(c, m) { if (!c) throw new Error(m || "assertion failed"); }

Deno.test("ice: credentials refresh at 3/4 of their life; the swap replaces only the default relay", () => {
  eq(refreshInMs(21600), 16200000); eq(refreshInMs(10), 60000); eq(refreshInMs(undefined), 2700000);
  const mine = { urls: ["turn:mine:1"], username: "a", credential: "b" };
  const cfg = iceConfig(null, [mine, ...markAuto([{ urls: ["turn:old:1"], username: "o", credential: "o" }])]);
  const same = cfg.iceServers;
  swapRelayServers(cfg, [{ urls: ["turn:new:1"], username: "n", credential: "n" }]);
  ok(cfg.iceServers === same, "the same array PeerJS holds");
  eq(cfg.iceServers.map((s) => s.urls), [STUN.urls, ["turn:mine:1"], ["turn:new:1"]]);
  eq(JSON.stringify(cfg.iceServers[2]), '{"urls":["turn:new:1"],"username":"n","credential":"n"}', "the mark is not serialized");
  swapRelayServers(cfg, []);
  eq(cfg.iceServers.length, 3, "an empty refresh keeps the old credentials");
});

Deno.test("ice: model weights never go over a relayed link", () => {
  eq(weightsOverLink("direct", true), true);
  eq(weightsOverLink("direct", false), true);
  eq(weightsOverLink("relay", true), false);
  eq(weightsOverLink("relay", false), false);
  eq(weightsOverLink(null, true), false, "unknown path with a relay configured: the safe side");
  eq(weightsOverLink(null, false), true, "no relay anywhere: it can only be direct");
});

Deno.test("ice: classifyCandidates and the advice for a closed network", () => {
  const host = "candidate:1 1 udp 2122260223 192.168.1.5 50000 typ host generation 0";
  const srflx = "candidate:2 1 udp 1686052607 203.0.113.7 50000 typ srflx raddr 192.168.1.5 rport 50000";
  const tcp = "candidate:3 1 tcp 1518280447 192.168.1.5 9 typ host tcptype active";
  eq(classifyCandidates([host, srflx]), { udpOut: true, host: true });
  eq(classifyCandidates([host, tcp]), { udpOut: false, host: true }, "UDP blocked: host only");
  eq(classifyCandidates([{ candidate: srflx }]), { udpOut: true, host: false });
  eq(classifyCandidates(null), { udpOut: false, host: false });
  eq(networkAdvice(null, false), null, "unknown: say nothing");
  eq(networkAdvice({ udpOut: true }, false), null);
  eq(networkAdvice({ udpOut: false }, true).level, "info");
  const w = networkAdvice({ udpOut: false }, false);
  eq(w.level, "warn"); ok(/Network/.test(w.text) && /Rooms at work/.test(w.text), w.text);
});

Deno.test("ice: probeUdp with a fake RTCPeerConnection", async () => {
  const fake = (lines) => class {
    constructor(cfg) { this.cfg = cfg; this.closed = false; }
    createDataChannel() {}
    async createOffer() { return { type: "offer", sdp: "" }; }
    async setLocalDescription() { setTimeout(() => { for (const l of lines) this.onicecandidate?.({ candidate: { candidate: l } }); this.onicecandidate?.({ candidate: null }); }, 5); }
    close() { this.closed = true; }
  };
  eq(await probeUdp(fake(["candidate:1 1 udp 1 10.0.0.2 5 typ host", "candidate:2 1 udp 1 1.2.3.4 5 typ srflx raddr 10.0.0.2 rport 5"])), { udpOut: true, host: true });
  eq(await probeUdp(fake(["candidate:1 1 udp 1 10.0.0.2 5 typ host"])), { udpOut: false, host: true });
  eq(await probeUdp(undefined), null);
  eq(await probeUdp(class { constructor() { throw new Error("no"); } }), null);
  const silent = class { createDataChannel() {} async createOffer() { return {}; } async setLocalDescription() {} close() {} };
  eq(await probeUdp(silent, STUN, 50), { udpOut: false, host: false }, "no candidates by the deadline");
});

import { roomFns } from "./room_src.js";
Deno.test("room: a join that found the room but no path says why, from what the tab knows", () => {
  const { pathFailText } = roomFns(["pathFailText"], {});
  ok(/Check its address and password under Network/.test(pathFailText(true, "yours", null)), "the user's own relay");
  ok(/may block it too/.test(pathFailText(true, "default", { udpOut: false })), "the site's relay was tried");
  ok(/blocks direct \(UDP\) connections and there is no relay/.test(pathFailText(false, null, { udpOut: false })), "UDP blocked, no relay");
  ok(/strict firewall or mobile network/.test(pathFailText(false, null, { udpOut: true })), "UDP fine, NAT on one side");
  ok(/strict firewall or mobile network/.test(pathFailText(false, null, null)), "unknown");
});

import { linkRelayProtocol } from "../../room/ice.js";
Deno.test("ice: linkRelayProtocol says how this side reaches the relay", () => {
  const mk = (lt, proto) => new Map([
    ["L", { id: "L", type: "local-candidate", candidateType: lt, relayProtocol: proto }], ["R", { id: "R", type: "remote-candidate", candidateType: "relay" }],
    ["CP", { id: "CP", type: "candidate-pair", localCandidateId: "L", remoteCandidateId: "R", state: "succeeded" }], ["T", { id: "T", type: "transport", selectedCandidatePairId: "CP" }]]);
  eq(linkRelayProtocol(mk("relay", "tls")), "tls");
  eq(linkRelayProtocol(mk("relay", "tcp")), "tcp");
  eq(linkRelayProtocol(mk("host", undefined)), null, "the other side is relayed, this one is not");
  eq(linkRelayProtocol(null), null);
});
