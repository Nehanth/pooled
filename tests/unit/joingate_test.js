// room/joingate.js: room codes (six characters, four still read), the invite key, and the host's gate
// (who gets in without asking, the lobby, Allow / Deny, and what a host reload keeps).
import { CODE_ALPHABET, randomCode, parseCode, formatCode, newKey, validKey, keyFromHash, keyFragment, digest, sameHash,
  makeGate, saveGate, restoreGate, decide, enqueue, waiting, allow, deny, withdraw, requestLine, deviceLabel,
  LOBBY_MAX, DENIED_TEXT, OLD_TAB_TEXT, FULL_TEXT } from "../../room/joingate.js";
import { codeFromLocation } from "../../room/plan.js";
import { guestResume } from "../../room/resume.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

Deno.test("codes: six characters from the room alphabet, no modulo bias", () => {
  eq(CODE_ALPHABET.length, 30);
  ok(!/[ILOU01]/.test(CODE_ALPHABET), "no look-alikes");
  const seen = new Set();
  for (let i = 0; i < 2000; i++) {
    const c = randomCode();
    ok(/^[A-HJKMNP-TV-Z2-9]{6}$/.test(c), c);
    seen.add(c);
  }
  ok(seen.size > 1990, "codes repeat far too often: " + seen.size);
  // bytes >= 240 are drawn again (240 = 8 * 30): with a fixed rng, 255 and 240 are skipped
  let calls = 0;
  const rng = (n) => { calls++; return calls === 1 ? new Uint8Array(n).fill(255) : Uint8Array.from({ length: n }, (_, i) => i === 0 ? 240 : i); };
  eq(randomCode(4, rng), "BCDE", "skips biased bytes");
  eq(randomCode(4).length, 4);
  // every letter turns up, about evenly
  const n = new Map();
  for (let i = 0; i < 3000; i++) for (const ch of randomCode()) n.set(ch, (n.get(ch) || 0) + 1);
  eq(n.size, 30);
  const mean = 3000 * 6 / 30;
  for (const [, v] of n) ok(v > mean * 0.8 && v < mean * 1.2, "uneven: " + v);
});

Deno.test("codes: what people type or paste, and how codes read", () => {
  eq(parseCode("4tk-g9p"), "4TKG9P");
  eq(parseCode(" 4TK G9P "), "4TKG9P");
  eq(parseCode("4TKG9P"), "4TKG9P");
  eq(parseCode("abcd"), "ABCD", "four characters: rooms from before");
  eq(parseCode("ABCDE"), "", "five never was a code");
  eq(parseCode("ABCDEFG"), "");
  eq(parseCode("4TK-G9O"), "", "O is not in the alphabet");
  eq(parseCode("I1L0"), "");
  eq(parseCode(""), ""); eq(parseCode(null), "");
  eq(formatCode("4TKG9P"), "4TK-G9P");
  eq(formatCode("ABCD"), "ABCD");
  eq(formatCode(""), "");
});

Deno.test("codes: join links with four and six characters, and the key beside them", () => {
  eq(codeFromLocation("/r/4TKG9P", "", "#k=AAAAAAAAAAAAAAAAAAAAAA"), "4TKG9P", "the key is not a code");
  eq(codeFromLocation("/r/abcd", "", ""), "ABCD");
  eq(codeFromLocation("/room", "?code=4TKG9P", "#k=AAAAAAAAAAAAAAAAAAAAAA"), "4TKG9P");
  eq(codeFromLocation("/room", "", "#ABCD"), "ABCD");
  eq(codeFromLocation("/room", "", "#k=AAAAAAAAAAAAAAAAAAAAAA"), "");
  eq(codeFromLocation("/r/4TKG9O", "", ""), "");
});

Deno.test("key: 128 random bits in a link fragment", () => {
  const k = newKey();
  ok(validKey(k), k);
  eq(k.length, 22, "16 bytes as base64url");
  ok(newKey() !== k);
  eq(keyFragment(k), "#k=" + k);
  eq(keyFragment("short"), "");
  eq(keyFromHash("#k=" + k), k);
  eq(keyFromHash("k=" + k + "&x=1"), k);
  eq(keyFromHash("#x=1&k=" + k), k);
  eq(keyFromHash("#k=bad key"), "");
  eq(keyFromHash("#ABCD"), "");
  eq(keyFromHash(""), "");
  eq(newKey(() => new Uint8Array(16).fill(0xfb)), "-_v7-_v7-_v7-_v7-_v7-w", "base64url alphabet");
});

Deno.test("key: compared by hash", async () => {
  const a = await digest("x"), b = await digest("x"), c = await digest("y");
  eq(a.length, 64);
  ok(sameHash(a, b)); ok(!sameHash(a, c)); ok(!sameHash(a, a.slice(0, 63))); ok(!sameHash("", ""));
});

const hello = (name, extra = {}) => ({ t: "hello", name, v: 4, join: 1, meta: { ua: "Mac", webgpu: true, contribGB: 8 }, ...extra });

Deno.test("gate: a new device is asked about; the key and a pass let one in", async () => {
  const g = makeGate();
  ok(g.ask, "asks by default");
  eq(await decide(g, "p1", hello("otter")), { kind: "ask" }, "typed the code");
  eq(await decide(g, "p1", hello("otter", { key: newKey() })), { kind: "ask" }, "a wrong key is no key");
  const r = await decide(g, "p2", hello("fox", { key: g.key }));
  eq([r.kind, r.via], ["admit", "key"], "the invite link");
  ok(validKey(r.pass), "it gets a pass");
  // the pass brings it back (a reload, a lock, a dropped link), without the key
  const back = await decide(g, "p9", hello("fox", { pass: r.pass, back: 1 }));
  eq([back.kind, back.via, back.pass], ["admit", "pass", null]);
  eq(await decide(g, "p3", hello("owl", { pass: newKey() })), { kind: "ask" }, "a made-up pass");
});

Deno.test("gate: Allow and Deny, several requests at once", async () => {
  const g = makeGate();
  for (const [id, name] of [["a", "otter"], ["b", "fox"], ["c", "owl"]]) {
    eq((await decide(g, id, hello(name))).kind, "ask");
    enqueue(g, id, name, { ua: "iPhone", webgpu: true, contribGB: 1 }, 100 + id.charCodeAt(0));
  }
  enqueue(g, "a", "otter", {}, 999);   // its hello again (a second link): one request, first in line still
  eq(waiting(g).map((r) => r.id), ["a", "b", "c"], "oldest first");
  eq(waiting(g)[0].at, 100 + 97, "keeps its place");
  const r = await allow(g, "b");
  eq(r.req.name, "fox");
  ok(validKey(r.pass));
  eq((await decide(g, "b2", hello("fox", { pass: r.pass }))).via, "pass", "an allowed device comes back without asking");
  eq(deny(g, "a").name, "otter");
  eq(await decide(g, "a", hello("otter")), { kind: "refuse", reason: DENIED_TEXT }, "a denied link knocking again");
  eq((await decide(g, "a-new-tab", hello("otter"))).kind, "ask", "a new link is a new request");
  eq(withdraw(g, "c")?.name, "owl", "closed its tab while waiting");
  eq(waiting(g), []);
  eq(await allow(g, "c"), null, "gone");
  eq(deny(g, "zz"), null);
});

Deno.test("gate: the lobby holds at most LOBBY_MAX", async () => {
  const g = makeGate();
  for (let i = 0; i < LOBBY_MAX; i++) { eq((await decide(g, "p" + i, hello("d" + i))).kind, "ask"); enqueue(g, "p" + i, "d" + i, {}); }
  eq(await decide(g, "late", hello("late")), { kind: "refuse", reason: FULL_TEXT });
  eq((await decide(g, "p0", hello("d0"))).kind, "ask", "one already waiting may say hello again");
  eq((await decide(g, "late", hello("late", { key: g.key }))).kind, "admit", "the key still works");
});

Deno.test("gate: a namesake is not let in by its name", async () => {
  const g = makeGate();
  const r = await decide(g, "p1", hello("otter", { key: g.key }));
  // the same name on a new link: the old device reconnecting (with its pass) or someone else (without)
  eq((await decide(g, "p2", hello("otter", { pass: r.pass }))).via, "pass");
  eq((await decide(g, "p3", hello("otter"))).kind, "ask", "a name proves nothing");
  eq((await decide(g, "p1", hello("otter", { back: 1 }))).kind, "ask", "nor does its old peer id with back: 1");
});

Deno.test("gate: Ask off, and tabs from before the gate", async () => {
  const g = makeGate({ ask: false });
  const r = await decide(g, "p1", hello("otter"));
  eq([r.kind, r.via], ["admit", "open"]);
  ok(validKey(r.pass), "a pass anyway: turning Ask on later doesn't ask about it");
  g.ask = true;
  eq((await decide(g, "p1b", hello("otter", { pass: r.pass }))).via, "pass");
  const old = { t: "hello", name: "old", v: 4, meta: {} };   // no join: 1
  eq(await decide(g, "o1", old), { kind: "refuse", reason: OLD_TAB_TEXT }, "an older tab is told to reload");
  ok(/Reload the page/.test(OLD_TAB_TEXT));
  g.ask = false;
  eq(await decide(g, "o1", old), { kind: "admit", via: "open", pass: null }, "with Ask off it joins as before, no pass");
  eq((await decide(g, "o2", { ...old, key: g.key })).via, "key", "an old tab with a key link");
});

Deno.test("gate: a host reload keeps its key, its passes and Ask", async () => {
  const g = makeGate();
  const r = await decide(g, "p1", hello("fox", { key: g.key }));
  g.ask = false;
  const saved = JSON.parse(JSON.stringify(saveGate(g)));   // through localStorage
  ok(!JSON.stringify(saved).includes(r.pass), "passes are kept as hashes only");
  const h = restoreGate(saved);
  eq(h.key, g.key, "links already shared keep working");
  eq(h.ask, false);
  eq((await decide(h, "p7", hello("fox", { pass: r.pass, back: 1 }))).via, "pass", "its devices come back without asking");
  eq(h.lobby, [], "the lobby is not saved");
  // a room saved by an older build (no gate) or garbage: a new gate, asking
  const fresh = restoreGate(undefined);
  ok(fresh.ask && validKey(fresh.key) && fresh.key !== g.key);
  const bad = restoreGate({ ask: "yes", key: "nope", passes: { zz: "x", ["a".repeat(64)]: "ok" } });
  ok(bad.ask && validKey(bad.key));
  eq([...bad.passes.keys()], ["a".repeat(64)]);
});

Deno.test("gate: what the host reads", () => {
  eq(requestLine("otter", { ua: "Mac", webgpu: true, contribGB: 8 }), "otter wants to join (Mac, 8 GB)");
  eq(requestLine("fox", { ua: "iPhone", webgpu: true, contribGB: 0.5 }), "fox wants to join (iPhone, 0.5 GB)");
  eq(requestLine("owl", { ua: "Device", webgpu: false }), "owl wants to join (computer, chat only)");
  eq(requestLine("pooled serve ab12", { api: 1, webgpu: false, ua: "API" }), "pooled serve ab12 wants to join (API client)");
  eq(requestLine("", {}), "A device wants to join (computer, chat only)");
  eq(deviceLabel({ ua: "Android", webgpu: true, contribGB: 2.25 }), "Android phone, 2.3 GB");
});

Deno.test("guest resume keeps the pass the host gave the tab", () => {
  const now = 1e12, pass = newKey();
  eq(guestResume({ code: "4TKG9P", name: "otter", pass, t: now }, { now })?.pass, pass);
  eq(guestResume({ code: "4TKG9P", name: "otter", pass: "x y", t: now }, { now })?.pass, undefined, "not a pass");
  eq(guestResume({ code: "4TKG9P", name: "otter", t: now }, { now }), { code: "4TKG9P", name: "otter", gb: null });
});
