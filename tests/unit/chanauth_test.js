// room/chanauth.js and the gate's use of it (room/joingate.js decideAuth): the invite key and passes
// are proved, never sent; proofs are bound to the link's DTLS fingerprints, so a device in the middle of
// a link (fingerprints swapped by the signaling server) gets nowhere; nothing it sees can be replayed;
// the host proves the secret back; the six digits (SAS) match only on a link with nobody in the middle;
// mesh links prove the room's key; devices from before the proofs (raw key in the hello) follow the
// legacy rule.
import { sdpFingerprints, linkFingerprints, linkBinding, joinerStart, joinerProof, joinerCheckAdmit, hostStart, hostVerify,
  hostWantsAuth, commitOf, passSecret, passIdOf, sasOf, meshProof, meshCheck, newMeshKey, validMeshKey, sameHex, proofOf, newNonce } from "../../room/chanauth.js";
import { makeGate, saveGate, restoreGate, decide, decideAuth, startAuth, newKey, digest } from "../../room/joingate.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

// a fake certificate fingerprint, and a link end that shows it (as RTCPeerConnection descriptions do)
const fp = (c) => "a=fingerprint:sha-256 " + Array.from({ length: 32 }, () => c).join(":");
const sdp = (c) => `v=0\r\no=- 1 2 IN IP4 127.0.0.1\r\ns=-\r\n${fp(c)}\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n`;
const end = (mine, theirs) => ({ peerConnection: { localDescription: { sdp: sdp(mine) }, remoteDescription: { sdp: sdp(theirs) } } });
// a direct link J <-> H, and one with M in the middle (J <-> M1, M2 <-> H)
const J = "joiner-id", H = "pooled-room-4TKG9P";

// one gate exchange over (joiner end, host end); mitm(msg) may tamper with what crosses
async function exchange({ g, key = null, pass = null, jEnd, hEnd, relay = (m) => m }) {
  const st = await joinerStart({ key, pass });
  const hello = relay({ t: "hello", name: "otter", join: 1, auth: 1, ...st.helloFields });
  ok(hostWantsAuth(hello), "a hello that speaks the proofs");
  const { pending, msg } = startAuth(g, hello);
  const chal = relay(msg);
  const proof = relay(await joinerProof(st, chal, { fps: linkFingerprints(jEnd), me: J, host: H }));
  const r = await decideAuth(g, J, hello, pending, proof, { fps: linkFingerprints(hEnd), me: H, peer: J });
  const admit = r.kind === "admit" ? relay({ t: "admit", ...(r.pass ? { pass: r.pass } : {}), ...(r.hp ? { via: r.via, hp: r.hp } : {}) }) : null;
  const check = admit ? await joinerCheckAdmit(st, admit) : null;
  return { st, hello, proof, r, admit, check };
}

Deno.test("fingerprints: read from SDP, the same set on both ends of a link", () => {
  eq(sdpFingerprints(sdp("AB")), ["sha-256 " + Array(32).fill("ab").join(":")]);
  eq(sdpFingerprints("a=fingerprint:SHA-256 0A:0B\r\na=fingerprint:sha-1 01:02\na=fingerprint:sha-256 0a:0b"), ["sha-1 01:02", "sha-256 0a:0b"], "lower case, once each, sorted");
  eq(sdpFingerprints(""), []); eq(sdpFingerprints(null), []);
  eq(linkFingerprints({}), null, "no peerConnection");
  eq(linkFingerprints({ peerConnection: { localDescription: { sdp: sdp("AA") }, remoteDescription: null } }), null, "no remote description yet");
  const a = linkFingerprints(end("AA", "BB")), b = linkFingerprints(end("BB", "AA"));
  ok(linkBinding(a) && linkBinding(a) === linkBinding(b), "both ends agree");
  ok(linkBinding(linkFingerprints(end("AA", "CC"))) !== linkBinding(a), "another certificate, another binding");
  eq(linkBinding(null), "");
});

Deno.test("gate: the invite key is proved, never sent, and the host proves it back", async () => {
  const g = makeGate({ ask: true });
  const x = await exchange({ g, key: g.key, jEnd: end("AA", "BB"), hEnd: end("BB", "AA") });
  ok(!JSON.stringify([x.hello, x.proof]).includes(g.key), "the key never crosses the link");
  eq(x.hello.kc, 1); ok(!("key" in x.hello) && !("pass" in x.hello));
  eq(x.r.kind, "admit"); eq(x.r.via, "key");
  ok(x.r.pass && x.r.pass.length >= 22, "a pass to come back with");
  eq(x.check, "ok", "the joiner checked the host's proof");
  ok(/^\d{3} \d{3}$/.test(x.r.sas), "six digits"); eq(x.r.sas, x.st.sas, "the same on both screens");
});

Deno.test("gate: a wrong key gets no admit by key; the device waits for Allow and the host is told", async () => {
  const g = makeGate({ ask: true });
  const x = await exchange({ g, key: newKey(), jEnd: end("AA", "BB"), hEnd: end("BB", "AA") });
  eq(x.r.kind, "ask"); ok(x.r.failed, "the host hears that the link's key didn't check out");
  // with Ask off it comes in as anyone would, without the host's proof: the joiner refuses that
  const o = makeGate({ ask: false });
  const y = await exchange({ g: o, key: newKey(), jEnd: end("AA", "BB"), hEnd: end("BB", "AA") });
  eq(y.r.kind, "admit"); eq(y.r.via, "open");
  eq(y.check, "tofu", "a device that proved a key and got no proof back goes in on trust (and says so)");
});

Deno.test("gate: someone in the middle (fingerprints swapped by signaling) gets refused both ways", async () => {
  const g = makeGate({ ask: true });
  // the joiner's link ends at M (certificate C0), the host's link starts at M (certificate D1); M relays
  // every message unchanged, including the real host's answers
  const x = await exchange({ g, key: g.key, jEnd: end("AA", "C0"), hEnd: end("BB", "D1") });
  eq(x.r.kind, "ask", "the host does not let the relayed proof in");
  ok(x.r.failed);
  ok(x.st.sas && x.r.sas && x.st.sas !== x.r.sas, "the two screens show different codes");
  // M's own guesses of the host's proof: the joiner checks against its own link
  const fake = await proofOf(g.key, { role: "host", cred: "key", binding: linkBinding(linkFingerprints(end("BB", "D1"))), joiner: J, host: H, jn: x.st.jn, hn: x.st.hn });
  eq(await joinerCheckAdmit(x.st, { t: "admit", via: "key", hp: fake }), "bad", "a proof made for the host's link is worth nothing on the joiner's");
  // the same for a pass
  const st0 = await joinerStart({ key: g.key });
  const r0 = await decide(g, J, { t: "hello", name: "p", join: 1, ...st0.helloFields, key: g.key });
  eq(r0.kind, "ask", "a raw key beside the proof fields is never read");
});

Deno.test("gate: proofs can't be replayed: a new challenge, a revealed nonce that doesn't match its commitment", async () => {
  const g = makeGate({ ask: true });
  const jEnd = end("AA", "BB"), hEnd = end("BB", "AA");
  const st = await joinerStart({ key: g.key });
  const hello = { t: "hello", name: "otter", join: 1, ...st.helloFields };
  const a = startAuth(g, hello);
  const proof = await joinerProof(st, a.msg, { fps: linkFingerprints(jEnd), me: J, host: H });
  eq((await decideAuth(g, J, hello, a.pending, proof, { fps: linkFingerprints(hEnd), me: H, peer: J })).via, "key");
  // the same hello and proof again: a fresh host nonce, so the old proof fails
  const b = startAuth(g, hello);
  ok(b.pending.hn !== a.pending.hn);
  const again = await decideAuth(g, J, hello, b.pending, proof, { fps: linkFingerprints(hEnd), me: H, peer: J });
  eq(again.kind, "ask", "a replayed proof does not let it in");
  // a nonce that isn't the committed one: refused outright
  const c = startAuth(g, hello);
  const bad = await decideAuth(g, J, hello, c.pending, { ...proof, jn: newNonce() }, { fps: linkFingerprints(hEnd), me: H, peer: J });
  eq(bad.kind, "refuse");
  // another peer id on the same certificates (a different link) does not take the proof either
  const d = startAuth(g, hello);
  const st2 = await joinerStart({ key: g.key });
  const p2 = await joinerProof(st2, d.msg, { fps: linkFingerprints(jEnd), me: "someone-else", host: H });
  const r2 = await decideAuth(g, J, { ...hello, jc: st2.helloFields.jc }, { ...d.pending, jc: st2.helloFields.jc }, p2, { fps: linkFingerprints(hEnd), me: H, peer: J });
  eq(r2.kind, "ask");
});

Deno.test("gate: passes: the host keeps only hashes and still proves them back", async () => {
  const g = makeGate({ ask: true });
  const first = await exchange({ g, key: g.key, jEnd: end("AA", "BB"), hEnd: end("BB", "AA") });
  const pass = first.r.pass;
  ok(!JSON.stringify(saveGate(g)).includes(pass), "no plaintext pass in what the host saves");
  const h = await passSecret(pass);
  eq(h, await digest(pass), "the HMAC key is the hash the host keeps");
  ok(g.passes.has(h));
  const back = await exchange({ g, pass, jEnd: end("CC", "BB"), hEnd: end("BB", "CC") });
  eq(back.r.via, "pass"); eq(back.r.pass, null, "it has one already"); eq(back.check, "ok");
  eq(back.hello.pid, await passIdOf(h)); ok(!JSON.stringify(back.hello).includes(pass) && !JSON.stringify(back.proof).includes(h));
  // a pass the host never gave: waits
  const no = await exchange({ g, pass: newKey(), jEnd: end("CC", "BB"), hEnd: end("BB", "CC") });
  eq(no.r.kind, "ask");
  // a host restarted from what it saved: the same pass still gets in, by its hash
  const g2 = restoreGate(JSON.parse(JSON.stringify(saveGate(g))));
  eq(g2.mk, g.mk, "the mesh key survives a restart");
  const again = await exchange({ g: g2, pass, jEnd: end("DD", "BB"), hEnd: end("BB", "DD") });
  eq(again.r.via, "pass"); eq(again.check, "ok");
  // pass and key together: the pass is tried first (no new pass handed out)
  const both = await exchange({ g, pass, key: g.key, jEnd: end("EE", "BB"), hEnd: end("BB", "EE") });
  eq(both.r.via, "pass"); eq(both.check, "ok");
});

Deno.test("gate: mutual: a host that doesn't hold the key can't fake its proof; the device refuses it", async () => {
  const st = await joinerStart({ key: newKey() });
  const chal = { t: "auth", hn: newNonce() };
  await joinerProof(st, chal, { fps: linkFingerprints(end("AA", "BB")), me: J, host: H });
  eq(await joinerCheckAdmit(st, { t: "admit" }), "tofu", "no proof at all: trust on first use, never a refusal (a restarted host)");
  eq(await joinerCheckAdmit(st, { t: "admit", via: "key", hp: "0".repeat(64) }), "bad", "a made-up proof");
  eq(await joinerCheckAdmit(st, { t: "admit", via: "pass", hp: "0".repeat(64) }), "bad", "a pass it never showed");
  // after waiting in the lobby, the host's Allow is trust on first use (the caller says the link didn't check out)
  st.lobbied = true;
  eq(await joinerCheckAdmit(st, { t: "admit" }), "tofu");
  eq(await joinerCheckAdmit(st, { t: "admit", via: "key", hp: "0".repeat(64) }), "bad", "a wrong proof is never trusted");
  // a device that proved nothing (a typed code) has nothing to check: Allow is trust on first use
  const none = await joinerStart({});
  eq(none.helloFields.kc, undefined); eq(none.helloFields.pid, undefined);
  await joinerProof(none, chal, { fps: linkFingerprints(end("AA", "BB")), me: J, host: H });
  eq(await joinerCheckAdmit(none, { t: "admit" }), "ok");
  ok(/^\d{3} \d{3}$/.test(none.sas), "it still shows the six digits");
});

Deno.test("gate: no fingerprints, no proof: a link that hides them binds nothing and is never let in by key", async () => {
  const g = makeGate({ ask: true });
  const x = await exchange({ g, key: g.key, jEnd: {}, hEnd: {} });
  eq(x.r.kind, "ask"); eq(x.r.sas, null, "and no six digits");
});

Deno.test("SAS: the joiner's nonce is committed before the host's is known", async () => {
  const jn = newNonce(), hn = newNonce();
  const b = linkBinding(linkFingerprints(end("AA", "BB")));
  eq(await sasOf(b, jn, hn), await sasOf(b, jn, hn));
  eq(await sasOf("", jn, hn), null);
  ok((await commitOf(jn)).length === 64 && (await commitOf(jn)) !== (await commitOf(hn)));
  // spread: 300 random nonces give nearly as many different codes
  const seen = new Set();
  for (let i = 0; i < 300; i++) seen.add(await sasOf(b, newNonce(), hn));
  ok(seen.size > 290, "codes repeat too often: " + seen.size);
});

Deno.test("mesh: both ends prove the room key on their link; reflections, other keys and the middle fail", async () => {
  const mk = newMeshKey();
  ok(validMeshKey(mk)); ok(!validMeshKey("x"));
  const A = "dev-a", B = "dev-b";
  const aEnd = linkFingerprints(end("AA", "BB")), bEnd = linkFingerprints(end("BB", "AA"));
  const pa = await meshProof(mk, { fps: aEnd, dialer: A, acceptor: B, role: "dial" });
  const pb = await meshProof(mk, { fps: bEnd, dialer: A, acceptor: B, role: "accept" });
  ok(await meshCheck(mk, { fps: bEnd, dialer: A, acceptor: B, role: "dial" }, pa), "the acceptor checks the dialer");
  ok(await meshCheck(mk, { fps: aEnd, dialer: A, acceptor: B, role: "accept" }, pb), "the dialer checks the acceptor");
  ok(!(await meshCheck(mk, { fps: aEnd, dialer: A, acceptor: B, role: "dial" }, pb)), "a proof sent back to its sender is not the other role's");
  ok(!(await meshCheck(newMeshKey(), { fps: bEnd, dialer: A, acceptor: B, role: "dial" }, pa)), "another room's key");
  ok(!(await meshCheck(mk, { fps: linkFingerprints(end("BB", "C0")), dialer: A, acceptor: B, role: "dial" }, pa)), "someone in the middle");
  eq(await meshProof(mk, { fps: null, dialer: A, acceptor: B, role: "dial" }), null, "no fingerprints, no proof");
});

Deno.test("legacy: a raw key in an old device's hello still works (marked), unless the host refuses those", async () => {
  const g = makeGate({ ask: true });
  const r = await decide(g, "old", { t: "hello", name: "old", join: 1, key: g.key });
  eq(r.kind, "admit"); eq(r.via, "key"); eq(r.legacy, true);
  const pass = r.pass;
  const back = await decide(g, "old", { t: "hello", name: "old", join: 1, pass });
  eq(back.via, "pass"); eq(back.legacy, true);
  const strict = makeGate({ ask: true, legacy: false, key: g.key });
  const sr = await decide(strict, "old", { t: "hello", name: "old", join: 1, key: g.key });
  eq([sr.kind, sr.oldRaw], ["ask", true], "strict: the raw key is ignored, the host is asked, and told why");
  const strictOpen = makeGate({ ask: false, legacy: false });
  eq((await decide(strictOpen, "old", { t: "hello", name: "old", join: 1, key: strictOpen.key })).via, "open");
  ok(sameHex("ab", "ab") && !sameHex("ab", "ac") && !sameHex("", ""));
});

// A signaling server in the middle can add a=fingerprint lines for the other link's certificates to the
// descriptions it forwards. Firefox keeps every line in remoteDescription and accepts a certificate that
// matches any of them, so both ends would read the same set {J, M1, H, M2} and the relayed proofs would
// check out. Only one fingerprint per end binds the link: more (or a line that doesn't parse) binds nothing.
Deno.test("fingerprints: extra a=fingerprint lines added by signaling bind nothing (both ends would agree on them)", async () => {
  const multi = (...cs) => `v=0\r\ns=-\r\n${cs.map(fp).join("\r\n")}\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n`;
  const pcEnd = (mine, theirs) => ({ peerConnection: { localDescription: { sdp: mine }, remoteDescription: { sdp: theirs } } });
  // J <-> M (M's cert M1), M <-> H (M's cert M2); signaling stuffed the other link's certs into each remote side
  const jEnd = pcEnd(sdp("AA"), multi("C1", "BB", "C2")), hEnd = pcEnd(sdp("BB"), multi("C2", "AA", "C1"));
  eq(linkFingerprints(jEnd), null, "more than one fingerprint on an end: no binding");
  eq(linkFingerprints(hEnd), null);
  const g = makeGate({ ask: true });
  const x = await exchange({ g, key: g.key, jEnd, hEnd });
  ok(x.r.kind !== "admit", "the relayed proof doesn't get in");
  // the same certificate twice (a session-level and a media-level line) is still one
  ok(linkFingerprints(pcEnd(multi("AA", "AA"), sdp("BB"))), "a repeated fingerprint is one");
  // a fingerprint line this parser can't read (but a WebRTC stack might): nothing is bound
  eq(linkFingerprints(pcEnd(sdp("AA"), sdp("BB") + "a=fingerprint:sha-256 GG:HH extra\r\n")), null, "an unreadable fingerprint line");
});
