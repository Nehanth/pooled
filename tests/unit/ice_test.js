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
