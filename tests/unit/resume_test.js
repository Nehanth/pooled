// room/resume.js: a run that survives a device dropping out of the chain, and the room.js pieces
// that put a returning device back into its slot (#207).
import { recoverableError, resumableGenerate, waitForRoom, linkSilent, backFromAway, sameShard, guestResume, GUEST_TTL_MS } from "../../room/resume.js";
import { roomFns } from "./room_src.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const rejects = async (p, re, m) => { try { await p; } catch (e) { if (re && !re.test(e.message)) throw new Error(`${m || "wrong error"}: ${e.message}`); return e; } throw new Error(m || "expected a rejection"); };

Deno.test("resume: which failures a recovery can fix", () => {
  ok(recoverableError(new Error("pipeline timeout (token)")));
  ok(recoverableError(new Error("otter left (layers 10–19)")));
  ok(recoverableError(new Error("a device left: re-deal the layers first")));
  ok(recoverableError(new Error("re-dealing the layers")));
  ok(recoverableError(new Error("fox: could not connect to the next device in the chain")));
  ok(!recoverableError(new Error("NaN in hidden returned by peers (pos 12)")));
  ok(!recoverableError(new Error("NaN produced on worker layers 3–5 (a device left?)")), "a NaN never retries");
  ok(!recoverableError(new Error("the model is not loaded")));
  ok(!recoverableError(null));
});

// a fake generation: emits the next tokens of `script` (continuing from what the prompt already
// holds past the original ids), failing once after `failAt` tokens
function fakeGen(script, base, { failAt = [], err = "otter left (layers 4–7)" } = {}) {
  const calls = [];
  const fails = [...failAt];
  const gen = async (ids, { maxNew, onToken }) => {
    calls.push({ ids: ids.slice(), maxNew });
    const from = ids.length - base;   // tokens of the script already in the prompt
    let n = 0;
    for (let i = from; i < script.length && n < maxNew; i++) {
      if (fails.length && i === fails[0]) { fails.shift(); throw new Error(err); }
      onToken(script[i], 0); n++;
    }
    return { tokens: script.slice(from, from + n), count: n, reason: n >= maxNew ? "max" : "stop", tPre: 1, tDecode: 2, prefilled: ids.length, stats: "s" };
  };
  return { gen, calls };
}

Deno.test("resume: a drop mid-answer carries on from the last emitted token", async () => {
  const script = [11, 12, 13, 14, 15, 16];
  const { gen, calls } = fakeGen(script, 3, { failAt: [3] });
  const seen = [], recs = [];
  const r = await resumableGenerate(gen, [1, 2, 3], { maxNew: 10, onToken: (t) => seen.push(t) },
    { recover: async (x) => { recs.push(x.emitted); } });
  eq(seen, script, "every token shown once, in order");
  eq(r.tokens, script);
  eq(r.count, 6);
  eq(r.resumed, 1);
  eq(recs, [3], "recovered once, after 3 tokens");
  eq(calls[1].ids, [1, 2, 3, 11, 12, 13], "the retry's prompt is the prompt plus what was emitted");
  eq(calls[1].maxNew, 7, "the retry's budget is what is left");
  eq(r.tPre, 1, "times add up over successful attempts only");
});

Deno.test("resume: the answer cap counts tokens from every attempt", async () => {
  const script = [11, 12, 13, 14, 15, 16];
  const { gen } = fakeGen(script, 2, { failAt: [2, 3] });
  const r = await resumableGenerate(gen, [1, 2], { maxNew: 4 }, { recover: async () => {} });
  eq(r.tokens, [11, 12, 13, 14]);
  eq(r.reason, "max");
  eq(r.resumed, 2);
});

Deno.test("resume: Stop, a lasting failure and too many drops are not retried", async () => {
  let recs = 0;
  const rec = { recover: async () => { recs++; } };
  await rejects(resumableGenerate(fakeGen([1, 2, 3], 1, { failAt: [1] }).gen, [0], { maxNew: 5 }, { ...rec, aborted: () => true }), /left/);
  await rejects(resumableGenerate(fakeGen([1, 2, 3], 1, { failAt: [1], err: "NaN after HOST layers" }).gen, [0], { maxNew: 5 }, rec), /NaN/);
  eq(recs, 0);
  await rejects(resumableGenerate(fakeGen([1, 2, 3, 4, 5], 1, { failAt: [1, 2, 3] }).gen, [0], { maxNew: 5 }, { ...rec, maxTries: 3 }), /left/);
  eq(recs, 2, "maxTries attempts: two recoveries, then the error");
  // no recover function: fail as before
  await rejects(resumableGenerate(fakeGen([1, 2], 1, { failAt: [0] }).gen, [0], { maxNew: 5 }), /left/);
});

Deno.test("resume: a recovery that fails fails the run with its own error", async () => {
  const { gen } = fakeGen([1, 2, 3], 1, { failAt: [1] });
  await rejects(resumableGenerate(gen, [0], { maxNew: 5 }, { recover: async () => { throw new Error("the room did not recover in 360 s"); } }), /did not recover/);
});

// a fake clock for waitForRoom
function clock() {
  let t = 0;
  return { now: () => t, sleep: async (ms) => { t += ms; }, at: () => t };
}

Deno.test("resume: the device comes back within the grace period: no re-deal", async () => {
  const c = clock();
  let back = false, dealt = 0;
  const said = [];
  await waitForRoom({ ready: () => { if (c.at() >= 20000) back = true; return back; }, gone: () => (back ? [] : ["phone"]),
    redeal: async () => { dealt++; }, status: (s) => said.push(s), now: c.now, sleep: c.sleep });
  eq(dealt, 0);
  ok(said[0].startsWith("waiting for phone to come back"), said[0]);
  ok(c.at() >= 20000 && c.at() < 21000);
});

Deno.test("resume: the device does not come back: re-deal after the grace period, once", async () => {
  const c = clock();
  let dealtAt = null, calls = 0;
  await waitForRoom({ ready: () => dealtAt != null && c.at() > dealtAt + 5000, gone: () => (dealtAt == null ? ["phone"] : []),
    redeal: async () => { calls++; dealtAt = c.at(); }, graceMs: 60000, now: c.now, sleep: c.sleep });
  eq(calls, 1);
  ok(dealtAt >= 60000 && dealtAt < 61000, "re-dealt at the end of the grace period: " + dealtAt);
});

Deno.test("resume: auto re-deal off waits until the limit; Stop ends the wait", async () => {
  const c = clock();
  let calls = 0;
  await rejects(waitForRoom({ ready: () => false, gone: () => ["phone"], redeal: async () => { calls++; }, autoRedeal: () => false,
    maxMs: 120000, now: c.now, sleep: c.sleep }), /did not recover/);
  eq(calls, 0);
  const c2 = clock();
  await rejects(waitForRoom({ ready: () => false, gone: () => ["phone"], aborted: () => c2.at() > 1000, now: c2.now, sleep: c2.sleep }), /stopped/);
});

Deno.test("resume: link and away helpers", () => {
  ok(!linkSilent(1000, 5000, 12000));
  ok(linkSilent(1000, 14000, 12000));
  ok(!linkSilent(undefined, 1e9), "a link never heard from is not judged");
  ok(!backFromAway(null, 5000));
  ok(!backFromAway(4000, 5000), "a one-second blip is not a lock");
  ok(backFromAway(1000, 9000));
  ok(sameShard({ model: "m", range: [4, 8], ctx: 4096 }, { model: "m", range: [4, 8], ctx: 4096 }));
  ok(!sameShard({ model: "m", range: [4, 8], ctx: 4096 }, { model: "m", range: [4, 9], ctx: 4096 }));
  ok(!sameShard(null, { model: "m", range: [4, 8] }));
});

Deno.test("resume: a reloaded guest tab walks back into its room", () => {
  const now = 1e12;
  eq(guestResume({ code: "ABCD", name: "otter", gb: 0.5, t: now - 1000 }, { now }), { code: "ABCD", name: "otter", gb: 0.5 });
  eq(guestResume({ code: "ABCD", name: "otter", t: now - GUEST_TTL_MS - 1 }, { now }), null, "too old");
  eq(guestResume({ code: "ABCD", name: "otter", t: now }, { now, linkCode: "WXYZ" }), null, "a link to another room wins");
  eq(guestResume({ code: "ABCD", name: "otter", t: now }, { now, linkCode: "ABCD" })?.code, "ABCD");
  eq(guestResume(null, { now }), null);
  eq(guestResume({ code: "ABCD", t: now }, { now }), null, "no name, no slot");
});

// room.js aiRejoin over stubs: who gets told what when a device comes back
function rejoinRoom({ chain, names, ready }) {
  const sent = [], logs = [];
  const ai = { role: "host", chain: chain.slice(), chainNames: names.slice(), readyPeers: new Set(ready), plan: new Map(names.map((n, i) => [n, { msg: { t: "ai-load", model: "m", range: [i * 4, i * 4 + 4], next: "x" } }])), fed: [], idleRedeal: 0 };
  const el = { style: {} };
  const fns = roomFns(["aiRejoin"], {
    ai, peer: { id: "HOST" }, sendTo: (id, m) => sent.push([id, m]), log: (_, t) => logs.push(t), aiStatus: () => {},
    ckptClear: () => {}, $: () => el, clearTimeout: () => {},
  });
  return { ai, sent, fns };
}

Deno.test("resume: a device back under the same id (after a lock) is re-seated if it had left", () => {
  const r = rejoinRoom({ chain: ["A", "B", "C"], names: ["fox", "phone", "owl"], ready: ["A", "C"] });
  r.fns.aiRejoin("B", "phone");
  eq(r.sent.map(([id, m]) => [id, m.t, m.next, m.relink]), [["A", "ai-next", "B", 1], ["B", "ai-load", "C", undefined]]);
  eq(r.ai.fed, null, "the caches are re-prefilled");
  // still in the chain and ready (a blip the host never noticed): nothing to do
  const q = rejoinRoom({ chain: ["A", "B"], names: ["fox", "phone"], ready: ["A", "B"] });
  q.fns.aiRejoin("B", "phone");
  eq(q.sent, []);
});

Deno.test("resume: a reloaded device (new id) takes its old slot; the first slot needs no relink", () => {
  const r = rejoinRoom({ chain: ["A", "B"], names: ["phone", "owl"], ready: ["B"] });
  r.fns.aiRejoin("A2", "phone");
  eq(r.ai.chain, ["A2", "B"]);
  eq(r.sent.map(([id, m]) => [id, m.t, m.next]), [["A2", "ai-load", "B"]]);
  // the last slot's next is the host
  const s = rejoinRoom({ chain: ["A", "B"], names: ["fox", "phone"], ready: ["A"] });
  s.fns.aiRejoin("B2", "phone");
  eq(s.sent.at(-1)[1].next, "host");
  // an unknown name is not seated
  const u = rejoinRoom({ chain: ["A"], names: ["fox"], ready: [] });
  u.fns.aiRejoin("Z", "stranger");
  eq(u.sent, []);
});
