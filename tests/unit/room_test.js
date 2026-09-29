// room/conversation.js, room/plan.js and room/qr.js: DOM-free room logic.
import { buildIds, fitContext, reusablePrefix, splitThink, PERSONAS } from "../../room/conversation.js";
import { planSplit, ladder, bestFit, codeFromLocation } from "../../room/plan.js";
import { qrMatrix, qrSVG, rsEncode } from "../../room/qr.js";
import { packFlags, unpackFlags } from "../../room/transport.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

// a toy tokenizer: one id per character (1000 + code), plus the chat specials
const tok = {
  vocab: { "<|im_start|>": 1, "<|im_end|>": 2, "<|endoftext|>": 3, "<think>": 4, "</think>": 5 },
  encode: (s) => [...s].map((c) => 1000 + c.codePointAt(0)),
};
const E = (s) => tok.encode(s);

Deno.test("conversation: the first turn is the single-turn template it replaces", () => {
  const ids = buildIds(tok, { turns: [{ role: "user", text: "hi" }] });
  eq(ids, [1, ...E("user\nhi"), 2, ...E("\n"), 1, ...E("assistant\n"), 4, ...E("\n\n"), 5, ...E("\n\n")]);
});
Deno.test("conversation: thinking leaves the think block open; a system prompt comes first", () => {
  const ids = buildIds(tok, { system: "be brief", turns: [{ role: "user", text: "q" }], thinking: true });
  eq(ids, [1, ...E("system\nbe brief"), 2, ...E("\n"), 1, ...E("user\nq"), 2, ...E("\n"), 1, ...E("assistant\n")]);
});
Deno.test("conversation: a second turn extends the first turn's ids exactly (answer ids verbatim)", () => {
  const t1 = [{ role: "user", text: "a" }];
  const first = buildIds(tok, { turns: t1 });
  const answer = [77, 78, 79];
  const second = buildIds(tok, { turns: [...t1, { role: "assistant", ids: answer }, { role: "user", text: "b" }] });
  const fed = [...first, ...answer, 2];   // what the plain path writes when the answer ends on <|im_end|>
  eq(second.slice(0, fed.length), fed, "history prefix");
  eq(reusablePrefix(fed, second), fed.length);
  // the speculative path does not write the final <|im_end|>: still a prefix
  eq(reusablePrefix([...first, ...answer], second), first.length + answer.length);
});
Deno.test("conversation: reusablePrefix refuses anything but a strict prefix", () => {
  eq(reusablePrefix([], [1, 2]), 0);
  eq(reusablePrefix(null, [1, 2]), 0);
  eq(reusablePrefix([1, 2], [1, 2]), 0, "nothing new to prefill");
  eq(reusablePrefix([1, 2, 3], [1, 2]), 0);
  eq(reusablePrefix([1, 9], [1, 2, 3]), 0, "diverged");
  eq(reusablePrefix([1, 2], [1, 2, 3]), 2);
});
Deno.test("conversation: fitContext drops the oldest exchanges, never the question", () => {
  const turns = [{ role: "user", text: "x".repeat(40) }, { role: "assistant", ids: new Array(40).fill(9) },
    { role: "user", text: "y".repeat(10) }, { role: "assistant", ids: [9, 9] }, { role: "user", text: "z" }];
  const all = buildIds(tok, { turns }).length;
  const r0 = fitContext(tok, { turns }, all + 10, 10);
  eq(r0.dropped, 0);
  const r1 = fitContext(tok, { turns }, all, 10);
  eq(r1.dropped, 1); eq(r1.turns[0].text, "y".repeat(10));
  let threw = false;
  try { fitContext(tok, { turns: [{ role: "user", text: "w".repeat(100) }] }, 64, 16); } catch (e) { threw = /Shorten/.test(e.message); }
  ok(threw, "an over-long question is refused, not truncated");
});
Deno.test("conversation: splitThink", () => {
  eq(splitThink("plain"), { think: null, answer: "plain" });
  eq(splitThink("<think>\nhmm\n</think>\n\nyes"), { think: "hmm", answer: "yes" });
  eq(splitThink("<think>still going"), { think: "still going", answer: "", open: true });
  ok(PERSONAS.default.system === "", "the default persona adds no system prompt");
});

Deno.test("plan: planSplit gives every device a layer and covers every layer once", () => {
  const GB = 2 ** 30;
  for (const caps of [[10 * GB, 1 * GB], [0.1 * GB, 14 * GB, 0.5 * GB], [1, 1, 1, 1, 1], [5 * GB]]) {
    const { assigned, ranges } = planSplit(64, caps);
    eq(assigned.reduce((a, b) => a + b, 0), 64, "total");
    ok(assigned.every((a) => a >= 1), "at least one layer each");
    eq(ranges[0][0], 0); eq(ranges[ranges.length - 1][1], 64);
    for (let i = 1; i < ranges.length; i++) eq(ranges[i][0], ranges[i - 1][1], "contiguous");
  }
  eq(planSplit(64, [62, 2]).assigned, [62, 2], "the MacBook + iPhone demo split");
});
Deno.test("plan: ladder and bestFit", () => {
  const need = { big: 16.5, small: 0.6, mid: 4.6 };
  eq(ladder(need, 5).map((x) => [x.key, x.ok, x.short]), [["small", true, 0], ["mid", true, 0], ["big", false, 11.5]]);
  eq(bestFit(need, 5), "mid");
  eq(bestFit(need, 0.1), "small", "nothing fits: smallest");
  eq(bestFit(need, 99), "big");
});
Deno.test("plan: codeFromLocation reads /r/CODE, ?code= and #CODE", () => {
  eq(codeFromLocation("/r/ABCD"), "ABCD");
  eq(codeFromLocation("/r/abcd/"), "ABCD");
  eq(codeFromLocation("/room", "?code=xy23"), "XY23");
  eq(codeFromLocation("/room", "", "#K7MP"), "K7MP");
  eq(codeFromLocation("/room", "", "#debug"), "");
  eq(codeFromLocation("/r/IO01"), "", "letters the room never generates");
  eq(codeFromLocation("/room"), "");
});

Deno.test("qr: Reed-Solomon matches the ISO 18004 annex example", () => {
  // version 1-M "01234567" example codewords and their 10 ECC bytes
  eq(rsEncode([16, 32, 12, 86, 97, 128, 236, 17, 236, 17, 236, 17, 236, 17, 236, 17], 10), [165, 36, 212, 193, 237, 54, 199, 135, 44, 85]);
});
Deno.test("qr: a room link encodes to the matrix jsQR decoded (fingerprint)", async () => {
  const m = qrMatrix("https://pooled.run/r/ABCD");
  eq(m.length, 25, "version 2");
  const bytes = new TextEncoder().encode(m.map((r) => r.join("")).join("\n"));
  const h = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 16);
  eq(h, "5d70fab9088aad5b");   // jsQR decodes this matrix back to the link
  // finder patterns in three corners
  for (const [r, c] of [[0, 0], [0, 18], [18, 0]]) ok(m[r][c] && m[r + 6][c + 6] && m[r + 3][c + 3] && !m[r + 1][c + 1], "finder at " + r + "," + c);
  ok(qrSVG("x").startsWith("<svg"), "svg");
});
Deno.test("transport: header flags round trip", () => {
  for (const f of [{}, { spec: 1 }, { reset: 1 }, { rb: 0 }, { rb: 7, spec: 1 }, { reset: 1, rb: 62 }]) {
    const back = unpackFlags(packFlags(f));
    eq(!!back.spec, !!f.spec); eq(!!back.reset, !!f.reset); eq(back.rb, f.rb);
  }
});

import { verdict, deviceKind } from "../../room/preflight.js";
const UA = {
  mac: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
  ipad: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
  iphone: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
  firefox: "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0",
  linux: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
};
Deno.test("preflight: device kinds (an iPad is not a Mac)", () => {
  eq(deviceKind({ ua: UA.ipad, touchPoints: 5 }), "iPad");
  eq(deviceKind({ ua: UA.mac, touchPoints: 0 }), "Mac");
  eq(deviceKind({ ua: UA.iphone }), "iPhone");
});
Deno.test("preflight: every browser without WebGPU gets a specific remedy", () => {
  const no = (ua, extra = {}) => verdict({ ua, secure: true, hasGpuApi: false, adapter: null, ...extra });
  ok(/Chrome or Edge/.test(no(UA.firefox).detail), "firefox");
  ok(/enable-unsafe-webgpu/.test(no(UA.linux).detail), "linux chrome flag");
  ok(/Safari 26/.test(no(UA.iphone).detail), "ios update");
  ok(/https/.test(no(UA.mac, { secure: false }).detail), "insecure page");
  ok(/blocklisted/.test(no(UA.mac, { hasGpuApi: true }).detail), "api but no adapter");
  const yes = verdict({ ua: UA.mac, secure: true, hasGpuApi: true, adapter: { vendor: "apple", architecture: "metal-3" } });
  ok(yes.ok && /apple metal-3/.test(yes.line), "ok line names the GPU");
  ok(!no(UA.firefox).ok && /can chat/.test(no(UA.firefox).line), "guests are told they can still chat");
});
Deno.test("preflight: phones are checked first and told in phone words", () => {
  const no = (ua, extra = {}) => verdict({ ua, secure: true, hasGpuApi: false, adapter: null, ...extra });
  for (const v of [no(UA.iphone), no(UA.iphone, { hasGpuApi: true }), no(UA.ipad, { touchPoints: 5 })]) ok(/^This phone's GPU/.test(v.line) && /join and chat/.test(v.line), v.line);
  ok(/Chrome or Edge on a laptop/.test(no(UA.mac).line), "desktop line");
  ok(!/chrome:\/\//.test(no(UA.linux).line), "the flag hint stays in the details");
});
Deno.test("conversation: an open assistant turn (Continue) extends the caches without an end token", () => {
  const t1 = [{ role: "user", text: "a" }];
  const answer = [77, 78];
  const closed = buildIds(tok, { turns: [...t1, { role: "assistant", ids: answer }] });
  const open = buildIds(tok, { turns: [...t1, { role: "assistant", ids: answer, open: true }] });
  eq(open, closed.slice(0, open.length), "open is the closed turn minus <|im_end|>\\n");
  eq(closed.length - open.length, 2);
  // what a capped speculative answer leaves in the caches (all but the last emitted token) is a prefix
  eq(reusablePrefix(open.slice(0, -1), open), open.length - 1);
});

import { unpackF16 } from "../../room/wire.js";
import { f16ToF32 } from "../../engine/gguf.js";
Deno.test("wire: the f16 lookup table matches f16ToF32 for all 65,536 values (NaNs stay NaN)", () => {
  const all = new Uint16Array(65536); for (let i = 0; i < 65536; i++) all[i] = i;
  const got = unpackF16(all);
  for (let h = 0; h < 65536; h++) {
    const want = f16ToF32(h);
    if (Number.isNaN(want) ? !Number.isNaN(got[h]) : !Object.is(Math.fround(want), got[h])) throw new Error("mismatch at " + h);
  }
});

import { planForSpeed } from "../../room/plan.js";
Deno.test("plan: planForSpeed fills the fastest devices first and leaves out the ones not needed", () => {
  // host holds everything: nobody else is needed (fewest hops)
  eq(planForSpeed(64, [100, 30, 30]).used, [0]);
  // unmeasured: the biggest device after the host, not everyone
  eq(planForSpeed(64, [20, 10, 60]).assigned, [20, 0, 44]);
  // measured: the fast laptop fills before the slow one, whatever their pledges
  eq(planForSpeed(64, [20, 60, 60], [2, 9, 3]).assigned, [20, 0, 44]);
  // the slow phone only gets what nobody faster can hold
  eq(planForSpeed(64, [30, 30, 10], [3, 3, 12]).assigned, [30, 30, 4]);
  // the host keeps a layer even when it is the slowest
  eq(planForSpeed(8, [1, 20], [50, 1]).assigned, [1, 7]);
  // overflow: everyone full, the rest spread by capacity; every layer placed exactly once
  const o = planForSpeed(64, [10, 10, 10]);
  eq(o.assigned.reduce((a, b) => a + b, 0), 64);
  eq(o.ranges[o.ranges.length - 1][1], 64);
});

import { lookupDrafts } from "../../room/lookup.js";
Deno.test("lookup: drafts continue the most recent earlier copy of the last n-gram", () => {
  // ... 1 2 3 4 5 ... 1 2 3 -> 4 5
  eq(lookupDrafts([9, 1, 2, 3, 4, 5, 8, 7, 1, 2, 3], 2), [4, 5]);
  // the longest n wins, and the most recent copy
  eq(lookupDrafts([1, 2, 6, 0, 2, 7, 0, 1, 2], 3), [6, 0, 2]);
  eq(lookupDrafts([5, 2, 9, 9, 3, 5, 2], 4), [9, 9, 3, 5]);
  // nothing repeats: no drafts
  eq(lookupDrafts([1, 2, 3, 4, 5], 4), []);
  // never proposes past the end of the context
  eq(lookupDrafts([1, 2, 1, 2], 7), [1, 2]);
});

import { badF32, packF16, WireRangeError, wireStats } from "../../room/wire.js";
Deno.test("wire: badF32 finds a single Inf or NaN anywhere; packF16 refuses f16-overflowing frames", () => {
  const a = new Float32Array(5120).fill(1);
  ok(!badF32(a));
  for (const i of [0, 1, 96, 97, 2500, 5119]) { const b = a.slice(); b[i] = Infinity; ok(badF32(b), "inf at " + i); b[i] = NaN; ok(badF32(b), "nan at " + i); }
  const big = a.slice(); big[1234] = -278.7;
  packF16(big); ok(Math.abs(wireStats.lastMax - 278.7) < 1e-3, "max |x| recorded");
  big[1234] = 70000;
  let threw = false; try { packF16(big); } catch (e) { threw = e instanceof WireRangeError; }
  ok(threw, "a value beyond 65504 is refused, not sent as Inf");
});

import { pickModelHost } from "../../room/plan.js";
Deno.test("plan: the model host is the strongest device, a computer before a phone, whoever pressed Start", () => {
  const phone = { id: "p", meta: { webgpu: true, phone: true, contribGB: 6 } };
  const laptop = { id: "z", meta: { webgpu: true, ua: "Mac", contribGB: 4 } };
  const desk = { id: "d", meta: { webgpu: true, ua: "Device", contribGB: 12 } };
  // the phone lends more, but the laptop holds the head and samples
  eq(pickModelHost([phone, laptop]), "z");
  eq(pickModelHost([laptop, phone]), "z");
  // among computers, the most memory; ties go to the lowest id (every screen agrees)
  eq(pickModelHost([phone, laptop, desk]), "d");
  eq(pickModelHost([{ id: "b", meta: { webgpu: true, contribGB: 4 } }, { id: "a", meta: { webgpu: true, contribGB: 4 } }]), "a");
  // a computer without WebGPU cannot hold layers: the phone it is
  eq(pickModelHost([phone, { id: "a", meta: { webgpu: false, contribGB: 16 } }]), "p");
  // an older tab without the phone flag: its user agent says so
  eq(pickModelHost([{ id: "a", meta: { webgpu: true, ua: "Android", contribGB: 8 } }, laptop]), "z");
  eq(pickModelHost([]), null);
});
Deno.test("plan: a computer whose GPU copies memory clearly faster hosts, if it lends at least half as much", () => {
  const gb10 = { id: "g", meta: { webgpu: true, ua: "Device", contribGB: 13, gbps: 200 } };
  const mac = { id: "m", meta: { webgpu: true, ua: "Mac", contribGB: 12, gbps: 450 } };
  eq(pickModelHost([gb10, mac]), "m");
  eq(pickModelHost([mac, gb10]), "m");
  // not clearly faster (under 1.5x): memory decides as before
  eq(pickModelHost([gb10, { ...mac, meta: { ...mac.meta, gbps: 280 } }]), "g");
  // faster but lends under half the memory pick's: memory decides
  eq(pickModelHost([gb10, { ...mac, meta: { ...mac.meta, contribGB: 6 } }]), "g");
  eq(pickModelHost([gb10, { ...mac, meta: { ...mac.meta, contribGB: 6.5 } }]), "m");
  // either side unmeasured (an older tab, a failed probe, ?gbps=0): memory decides
  eq(pickModelHost([gb10, { ...mac, meta: { ...mac.meta, gbps: 0 } }]), "g");
  // the same GPU on both (two tabs on one machine, one probed while the GPU was busy): memory decides
  const tab = (id, gb, gbps) => ({ id, meta: { webgpu: true, ua: "Device", contribGB: gb, gbps, gpu: "nvidia blackwell" } });
  eq(pickModelHost([tab("a", 2, 110), tab("b", 1, 199)]), "a");
  eq(pickModelHost([tab("b", 1, 199), tab("a", 2, 110)]), "a");
  // a different GPU still takes it
  eq(pickModelHost([tab("a", 2, 110), { ...mac, meta: { ...mac.meta, contribGB: 2, gpu: "apple metal-3" } }]), "m");
  eq(pickModelHost([{ ...gb10, meta: { ...gb10.meta, gbps: undefined } }, mac]), "g");
  // a phone never wins on speed over a computer
  eq(pickModelHost([gb10, { id: "p", meta: { webgpu: true, phone: true, contribGB: 13, gbps: 900 } }]), "g");
  // several clearly faster: the fastest, then memory, then id
  const mac2 = { id: "n", meta: { webgpu: true, ua: "Mac", contribGB: 8, gbps: 500 } };
  eq(pickModelHost([gb10, mac, mac2]), "n");
  eq(pickModelHost([gb10, mac, { ...mac2, meta: { ...mac2.meta, gbps: 450 } }]), "m");
  eq(pickModelHost([gb10, { ...mac, id: "q" }, { ...mac, id: "k" }]), "k");
});
