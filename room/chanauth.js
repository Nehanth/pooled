// Channel-bound authentication for room links (docs/protocol.md, "Proving who is on a link").
// DOM-free, WebCrypto only, so the room page, the room node (packages/room-node) and the CLI share it.
//
// Signaling (the PeerJS server, the public cloud by default) carries each link's offer and answer,
// and with them the DTLS certificate fingerprints that WebRTC checks the link against. A signaling
// server that swaps those fingerprints can sit in the middle of a link it introduced: it ends DTLS on
// both sides and passes messages along. So no secret ever crosses a link as is. Each side proves it
// knows the secret with an HMAC over this exact link: both certificate fingerprints (read from the
// link's own local and remote descriptions), both peer ids and a fresh nonce from each side. A device
// in the middle sees two links with different fingerprints, so a proof made on one is worth nothing on
// the other, and it learns nothing it can replay.
//
// Three exchanges use it:
//   the gate (a device joining the host): the invite key or a pass the host gave, proved both ways;
//   a short authentication string (SAS) both screens can show when the host lets a device in by hand
//     (Allow), where there is no shared secret: six digits from both fingerprints and both nonces, with
//     the joiner's nonce committed before it sees the host's, so a device in the middle can't grind
//     its certificates or its nonce until the two screens agree;
//   mesh links (device to device, and the extra "stripe" links for the hidden-state wire): a room key
//     the host hands every device it lets in, proved on each link before anything else goes over it.

export const AUTH_V = 1;
const LABEL = "pooled-auth-v1", MESH_LABEL = "pooled-mesh-v1", SAS_LABEL = "pooled-sas-v1", COMMIT_LABEL = "pooled-commit-v1", PID_LABEL = "pooled-pass-id-v1";
const enc = new TextEncoder();
const NONCE_RE = /^[A-Za-z0-9_-]{22,64}$/;
const HEX64_RE = /^[0-9a-f]{64}$/;
export const validNonce = (n) => typeof n === "string" && NONCE_RE.test(n);
export const validHex64 = (h) => typeof h === "string" && HEX64_RE.test(h);

function b64url(bytes) {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
export const randomBytes = (n) => crypto.getRandomValues(new Uint8Array(n));
// 128 random bits, base64url (22 characters)
export const newNonce = (rng = randomBytes) => b64url(rng(16));
export const newMeshKey = (rng = randomBytes) => b64url(rng(32));
export const validMeshKey = (k) => typeof k === "string" && /^[A-Za-z0-9_-]{43}$/.test(k);

export async function sha256hex(s) { return hex(await crypto.subtle.digest("SHA-256", enc.encode(String(s)))); }
async function hmacHex(secret, msg) {
  const key = await crypto.subtle.importKey("raw", enc.encode(String(secret)), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  return hex(await crypto.subtle.sign("HMAC", key, enc.encode(msg)));
}
// equal hex strings, in time that depends only on the length
export function sameHex(a, b) {
  a = String(a ?? ""); b = String(b ?? "");
  let d = a.length ^ b.length;
  for (let i = 0; i < Math.max(a.length, b.length); i++) d |= (a.charCodeAt(i) || 0) ^ (b.charCodeAt(i) || 0);
  return d === 0 && a.length > 0;
}

// ---- the link's certificate fingerprints ----
// "a=fingerprint:sha-256 AB:CD:..." lines of an SDP -> ["sha-256 ab:cd:..."] (lower case, sorted, once each)
export function sdpFingerprints(sdp) {
  const out = new Set();
  for (const line of String(sdp || "").split(/\r?\n/)) {
    const m = /^a=fingerprint:\s*([A-Za-z0-9-]+)\s+([0-9A-Fa-f:]+)\s*$/.exec(line);
    if (m) out.add(`${m[1].toLowerCase()} ${m[2].toLowerCase()}`);
  }
  return [...out].sort();
}
// the one certificate an end of a link shows -> ["sha-256 ab:..."], or null when it shows none, more than
// one, or a fingerprint line this parser can't read. More than one would bind nothing: a signaling server
// in the middle can add lines for the other link's certificates to what it forwards, and Firefox keeps
// every line in remoteDescription and accepts a certificate that matches any of them, so both ends would
// read the same set. (Chrome and libdatachannel show only the line they use.) A repeated line is one.
function oneFingerprint(sdp) {
  const lines = String(sdp || "").split(/\r?\n/).filter((l) => /^\s*a=fingerprint\s*:/i.test(l));
  const fps = sdpFingerprints(sdp);
  return lines.length && fps.length === 1 && lines.every((l) => /^a=fingerprint:\s*[A-Za-z0-9-]+\s+[0-9A-Fa-f:]+\s*$/.test(l)) ? fps : null;
}
// a PeerJS DataConnection (conn.peerConnection) or an RTCPeerConnection -> { local, remote } fingerprint
// lists (one each), or null when either side has none (not negotiated yet, or a runtime that hides its
// SDP) or more than one (oneFingerprint). WebRTC itself refuses a DTLS certificate that doesn't match the
// remote description's fingerprint, so these are the certificates the link really runs on.
export function linkFingerprints(connOrPc) {
  const pc = connOrPc?.peerConnection || connOrPc;
  try {
    const local = oneFingerprint(pc?.localDescription?.sdp), remote = oneFingerprint(pc?.remoteDescription?.sdp);
    if (!local || !remote) return null;
    return { local, remote };
  } catch { return null; }
}
// what both ends of one link agree on: every fingerprint of the link, sorted (each end sees the same set)
export function linkBinding(fps) {
  if (!fps) return "";
  return [...new Set([...fps.local, ...fps.remote])].sort().join(",");
}

// ---- the gate: a device proving the invite key or a pass to the host, and the host proving it back ----
// the joiner's nonce is committed in its hello (jc) and shown only after the host sent its own (hn)
export const commitOf = (jn) => sha256hex(`${COMMIT_LABEL}\n${jn}`);
// the HMAC key for a pass: its SHA-256 (hex), which is all the host keeps of it (joingate.js passes)
export const passSecret = (pass) => sha256hex(String(pass));
// which pass a device holds, without showing it: lets the host pick the right hash out of hundreds
export const passIdOf = async (passHash) => (await sha256hex(`${PID_LABEL}\n${passHash}`)).slice(0, 32);
export const validPassId = (p) => typeof p === "string" && /^[0-9a-f]{32}$/.test(p);

// fields: { role: "join" | "host", cred: "key" | "pass", binding (linkBinding), joiner, host (peer ids), jn, hn }
export function transcript({ role, cred, binding, joiner, host, jn, hn }) {
  return [LABEL, `role=${role}`, `cred=${cred}`, `fp=${binding}`, `ids=${joiner},${host}`, `jn=${jn}`, `hn=${hn}`].join("\n");
}
export function proofOf(secret, fields) { return hmacHex(secret, transcript(fields)); }
export async function checkProof(secret, fields, got) {
  if (!validHex64(got)) return false;
  return sameHex(await proofOf(secret, fields), got);
}

// six digits both screens show for a device the host lets in by hand: "482 019". null without fingerprints
export async function sasOf(binding, jn, hn) {
  if (!binding || !validNonce(jn) || !validNonce(hn)) return null;
  const h = await sha256hex(`${SAS_LABEL}\n${binding}\n${jn}\n${hn}`);
  const n = parseInt(h.slice(0, 8), 16) % 1000000;
  const s = String(n).padStart(6, "0");
  return `${s.slice(0, 3)} ${s.slice(3)}`;
}

// ---- the joiner's side of the gate ----
// -> state for this link: what its hello carries (no secret), and what it needs later
//   key: the invite key from its link; pass: the pass the host gave it before
export async function joinerStart({ key = null, pass = null, rng } = {}) {
  const jn = newNonce(rng);
  const st = { jn, key: key || null, pass: pass || null, passHash: null, hn: null, sas: null, proved: false };
  const fields = { auth: AUTH_V, jc: await commitOf(jn) };
  if (st.key) fields.kc = 1;
  if (st.pass) { st.passHash = await passSecret(st.pass); fields.pid = await passIdOf(st.passHash); }
  st.helloFields = fields;
  return st;
}
// the host's challenge { t: "auth", hn } -> the proof message to send back ({ t: "auth-proof", ... })
//   link: { fps (linkFingerprints), me (this device's peer id), host (the host's peer id) }
export async function joinerProof(st, chal, link) {
  if (!validNonce(chal?.hn)) return null;
  st.hn = chal.hn;
  st.binding = linkBinding(link.fps);
  st.ids = { joiner: link.me, host: link.host };
  st.sas = await sasOf(st.binding, st.jn, st.hn);
  const msg = { t: "auth-proof", jn: st.jn };
  const base = { binding: st.binding, joiner: link.me, host: link.host, jn: st.jn, hn: st.hn, role: "join" };
  if (st.key) msg.kp = await proofOf(st.key, { ...base, cred: "key" });
  if (st.passHash) msg.pp = await proofOf(st.passHash, { ...base, cred: "pass" });
  st.proved = !!(msg.kp || msg.pp);
  return msg;
}
// the host's admit ->
//   "ok"    it proved the secret this device used (or this device used none)
//   "tofu"  it proved nothing back: this device's secret didn't check out (a link from an earlier room,
//           a host restarted without its saved passes, Allow after the lobby) or it doesn't know it.
//           Trust on first use, as for a typed code: nothing secret crossed, so the caller goes in
//           and says plainly that it couldn't verify the host
//   "bad"   a proof that is wrong: a host pretending to prove what it can't. The caller leaves
export async function joinerCheckAdmit(st, admit) {
  if (!st.proved) return "ok";
  if (admit?.via !== "key" && admit?.via !== "pass") return "tofu";
  const secret = admit.via === "key" ? st.key : st.passHash;
  if (!secret || !st.hn) return "bad";
  const ok = await checkProof(secret, { binding: st.binding, ...st.ids, jn: st.jn, hn: st.hn, role: "host", cred: admit.via }, admit.hp);
  return ok ? "ok" : "bad";
}

// ---- the host's side of the gate ----
// a hello that speaks this (auth >= 1, a commitment) -> the pending exchange and the challenge to send
export function hostWantsAuth(hello) { return +hello?.auth >= AUTH_V && validHex64(hello?.jc); }
export function hostStart(hello, rng) {
  const p = { jc: hello.jc, kc: hello.kc === 1, pid: validPassId(hello.pid) ? hello.pid : null, hn: newNonce(rng) };
  return { pending: p, msg: { t: "auth", v: AUTH_V, hn: p.hn } };
}
// the device's { t: "auth-proof" } on the link -> { cred: "pass" | "key" | null, secret, sas, hp, bad }
//   g: { key, passHash(pid) -> hash | null }; link: { fps, me (host id), peer (joiner id) }
//   bad: the proof message is broken (no valid nonce, or one that doesn't match its commitment)
export async function hostVerify(p, proof, { key, passHashFor }, link) {
  const jn = proof?.jn;
  if (!validNonce(jn) || !sameHex(await commitOf(jn), p.jc)) return { bad: true };
  const binding = linkBinding(link.fps);
  const base = { binding, joiner: link.peer, host: link.me, jn, hn: p.hn };
  const sas = await sasOf(binding, jn, p.hn);
  const out = { cred: null, sas, bound: !!binding };
  // a proof made without fingerprints would bind nothing: never accept one
  if (binding) {
    if (p.pid && proof.pp) {
      const h = await passHashFor(p.pid);
      if (h && await checkProof(h, { ...base, role: "join", cred: "pass" }, proof.pp)) {
        out.cred = "pass"; out.hp = await proofOf(h, { ...base, role: "host", cred: "pass" });
        return out;
      }
    }
    if (p.kc && proof.kp && key && await checkProof(key, { ...base, role: "join", cred: "key" }, proof.kp)) {
      out.cred = "key"; out.hp = await proofOf(key, { ...base, role: "host", cred: "key" });
    }
  }
  return out;
}

// ---- mesh links: both ends prove the room key the host gave them ----
// role: "dial" (this end opened the link) | "accept"; dialer / acceptor: the two peer ids
export function meshTranscript({ role, binding, dialer, acceptor }) {
  return [MESH_LABEL, `role=${role}`, `fp=${binding}`, `ids=${dialer},${acceptor}`].join("\n");
}
// -> the proof this end sends ({ t: "mesh", p }), or null when the link has no fingerprints to bind
export async function meshProof(mk, { fps, dialer, acceptor, role }) {
  const binding = linkBinding(fps);
  if (!binding || !validMeshKey(mk)) return null;
  return hmacHex(mk, meshTranscript({ role, binding, dialer, acceptor }));
}
export async function meshCheck(mk, { fps, dialer, acceptor, role }, got) {
  const binding = linkBinding(fps);
  if (!binding || !validMeshKey(mk) || !validHex64(got)) return false;
  return sameHex(await hmacHex(mk, meshTranscript({ role, binding, dialer, acceptor })), got);
}
