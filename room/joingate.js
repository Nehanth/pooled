// Who gets into a room: room codes, the invite key and the host's Allow / Deny. DOM-free so it can
// be unit tested; room.js wires it (docs/protocol.md, "Joining a room").
//
// A room code names the room (the host's PeerJS id is "pooled-room-<CODE>"), so anyone who has it,
// or guesses it, can knock. Knocking is all it buys: the host lets a new device in only when
//   - it presents the room's invite key (a random secret in the invite link's #k= fragment, which
//     browsers never send to a server), or
//   - it presents a pass the host gave it earlier (a device the host already let in, coming back
//     after a reload, a lock or a dropped link), or
//   - the host presses Allow, or
//   - the host turned "Ask before new devices join" off.
// Until then the device is held in the lobby: no layers, no chat, no roster.

// 30 letters and digits, no I, L, O, U, 0 or 1 (easy to read aloud and to type from a screen)
export const CODE_ALPHABET = "ABCDEFGHJKMNPQRSTVWXYZ23456789";
export const CODE_LEN = 6;   // 30^6 = 729 million codes; rooms before this had 4 (810,000)
const CODE_RE = /^[A-HJKMNP-TV-Z2-9]{4,6}$/;

// n random characters from the alphabet, without modulo bias (bytes >= 240 are drawn again)
export function randomCode(n = CODE_LEN, rng = (k) => crypto.getRandomValues(new Uint8Array(k))) {
  let out = "";
  while (out.length < n) {
    for (const b of rng(n * 2)) {
      if (b >= 240) continue;
      out += CODE_ALPHABET[b % 30];
      if (out.length === n) break;
    }
  }
  return out;
}

// what a person typed or pasted -> a room code, or "" ("4tk-g9p", "4TK G9P" -> "4TKG9P"). Four
// characters are still codes (rooms opened before six, and their links); five never were.
export function parseCode(s) {
  const c = String(s ?? "").toUpperCase().replace(/[\s\-_.]/g, "");
  return (c.length === 4 || c.length === 6) && CODE_RE.test(c) ? c : "";
}

// a code as people read it: six in two groups of three ("4TK-G9P"); four as they were
export function formatCode(code) {
  const c = String(code ?? "");
  return c.length === 6 ? `${c.slice(0, 3)}-${c.slice(3)}` : c;
}

// --- the invite key ---
// 16 random bytes (128 bits) as base64url: 22 characters in a link fragment
export function newKey(rng = (k) => crypto.getRandomValues(new Uint8Array(k))) { return b64url(rng(16)); }
export function newPass(rng) { return newKey(rng); }
const KEY_RE = /^[A-Za-z0-9_-]{22,64}$/;
export const validKey = (k) => typeof k === "string" && KEY_RE.test(k);
// "#k=abc…" (maybe among other fragment parts, "#k=abc&x=1") -> the key, or ""
export function keyFromHash(hash) {
  const h = String(hash ?? "").replace(/^#/, "");
  for (const part of h.split("&")) {
    const [k, v] = part.split("=");
    if (k === "k" && validKey(v)) return v;
  }
  return "";
}
// the invite link's fragment for a key
export const keyFragment = (key) => (validKey(key) ? `#k=${key}` : "");

function b64url(bytes) {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
// SHA-256 of a secret, hex. The host keeps passes only as hashes, and compares keys by hash, so a
// comparison's time says nothing about how much of a guess was right
export async function digest(secret) {
  return hex(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(String(secret))));
}
// equal strings, in time that depends only on the length (hex digests: always 64)
export function sameHash(a, b) {
  a = String(a ?? ""); b = String(b ?? "");
  let d = a.length ^ b.length;
  for (let i = 0; i < Math.max(a.length, b.length); i++) d |= (a.charCodeAt(i) || 0) ^ (b.charCodeAt(i) || 0);
  return d === 0 && a.length > 0;
}

// --- the host's gate ---
export const LOBBY_MAX = 8;   // requests waiting at once; more are told to try again later
export const DENIED_TEXT = "The host didn't let this device in.";
export const OLD_TAB_TEXT = "This room's host asks before new devices join, and this tab runs an older Pooled that can't wait for that. Reload the page, then join again.";
export const FULL_TEXT = "The host has several devices waiting to join already. Try again in a minute.";

// What the host keeps (and saves with the room, so a reload keeps its links and its devices):
//   ask     "Ask before new devices join" (default on)
//   key     the invite key (links made before a reload keep working)
//   passes  { hash: name } of the passes it gave out (hashes only)
// and, for this page only, the lobby: requests waiting for Allow / Deny, oldest first.
export function makeGate({ ask = true, key = null, passes = null, rng } = {}) {
  const g = {
    ask: ask !== false,
    key: validKey(key) ? key : newKey(rng),
    passes: new Map(passes && typeof passes === "object" ? Object.entries(passes).filter(([h]) => /^[0-9a-f]{64}$/.test(h)).slice(-500) : []),
    lobby: [],             // [{ id, name, meta, at }]
    denied: new Set(),     // peer ids the host said no to (this page): their knocks get the bye at once
    rng,
    keyHash: null,
  };
  return g;
}
export function saveGate(g) {
  return { ask: g.ask, key: g.key, passes: Object.fromEntries([...g.passes].slice(-500)) };
}
// a gate from a saved room (saveGate), or a new one when the room was saved by an older build
export function restoreGate(saved, opts = {}) {
  if (!saved || typeof saved !== "object") return makeGate(opts);
  return makeGate({ ...opts, ask: saved.ask, key: saved.key, passes: saved.passes });
}

// Is this hello's device let in? hello: { name, join, key, pass, back, meta }; id: its peer id.
// -> { kind: "admit", via: "pass" | "key" | "open", pass }   in now; `pass` is what to give it
//                                                            (null for "pass": it has one, and for
//                                                            an old tab that can't keep one)
//  | { kind: "ask" }                                         held: the host is asked
//  | { kind: "refuse", reason }                              a bye, and the link closes
// Doesn't change the gate except to remember a pass it gives out (via key or open).
export async function decide(g, id, hello) {
  const pass = typeof hello?.pass === "string" && validKey(hello.pass) ? hello.pass : null;
  if (pass) {
    const h = await digest(pass);
    for (const known of g.passes.keys()) if (sameHash(h, known)) return { kind: "admit", via: "pass", pass: null };
  }
  if (typeof hello?.key === "string" && validKey(hello.key)) {
    g.keyHash ||= await digest(g.key);
    if (sameHash(await digest(hello.key), g.keyHash)) return { kind: "admit", via: "key", pass: await issuePass(g, hello.name) };
  }
  if (!g.ask) return { kind: "admit", via: "open", pass: hello?.join ? await issuePass(g, hello.name) : null };
  // a tab from before this change doesn't know it may have to wait: tell it to reload
  if (!hello?.join) return { kind: "refuse", reason: OLD_TAB_TEXT };
  if (g.denied.has(id)) return { kind: "refuse", reason: DENIED_TEXT };
  if (!g.lobby.some((r) => r.id === id) && g.lobby.length >= LOBBY_MAX) return { kind: "refuse", reason: FULL_TEXT };
  return { kind: "ask" };
}
async function issuePass(g, name) {
  const p = newPass(g.rng);
  g.passes.set(await digest(p), String(name ?? "").slice(0, 40));
  while (g.passes.size > 500) g.passes.delete(g.passes.keys().next().value);
  return p;
}

// the lobby: a request waits here until the host answers it (or its device leaves)
export function enqueue(g, id, name, meta, now = Date.now()) {
  const i = g.lobby.findIndex((r) => r.id === id);
  const r = { id, name: String(name ?? "").slice(0, 40), meta: meta || {}, at: i >= 0 ? g.lobby[i].at : now };
  if (i >= 0) g.lobby[i] = r; else g.lobby.push(r);
  return r;
}
export const waiting = (g) => g.lobby.slice();
// the host pressed Allow: -> { req, pass } (the pass to hand the device), or null if it is gone
export async function allow(g, id) {
  const i = g.lobby.findIndex((r) => r.id === id);
  if (i < 0) return null;
  const [req] = g.lobby.splice(i, 1);
  return { req, pass: await issuePass(g, req.name) };
}
// the host pressed Deny: -> the request, or null
export function deny(g, id) {
  const i = g.lobby.findIndex((r) => r.id === id);
  if (i < 0) return null;
  g.denied.add(id);
  return g.lobby.splice(i, 1)[0];
}
// its device left (closed the tab, lost the link) before the host answered
export function withdraw(g, id) {
  const i = g.lobby.findIndex((r) => r.id === id);
  return i < 0 ? null : g.lobby.splice(i, 1)[0];
}

// What the host reads in the request: "otter wants to join (Mac, 8 GB)"
export function requestLine(name, meta = {}) {
  const who = String(name ?? "").trim() || "A device";
  return `${who} wants to join (${deviceLabel(meta)})`;
}
export function deviceLabel(meta = {}) {
  if (meta?.api) return "API client";
  const kind = { iPhone: "iPhone", iPad: "iPad", Android: "Android phone", "Android tablet": "Android tablet", Mac: "Mac" }[meta?.ua] || "computer";
  const gb = +meta?.contribGB;
  const mem = meta?.webgpu && gb > 0 ? `${+gb.toFixed(1)} GB` : "chat only";
  return `${kind}, ${mem}`;
}
