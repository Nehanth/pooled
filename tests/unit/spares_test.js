// Spare copies: planReplicas / coverSegments (room/plan.js), frame numbers, fencing and the
// failover rewiring (room/spares.js), and fseq on the wire (room/transport.js).
import { planReplicas, coverSegments, deviceRisk, spareCapable } from "../../room/plan.js";
import { fseqNext, seqAfter, seqStale, failoverPlan, makeProber, lapDeadline } from "../../room/spares.js";
import { makeLink, sendFrame, attachWire } from "../../room/transport.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const desk = (gpu = "nvidia blackwell") => ({ webgpu: true, rep: 1, gpu, ua: "Linux" });
const phone = { webgpu: true, rep: 1, phone: true, ua: "iPhone", gpu: "apple" };

Deno.test("spares: device risk and who may be a spare", () => {
  eq(deviceRisk(phone), 4);
  eq(deviceRisk({ ua: "Android" }), 4);
  eq(deviceRisk({ onBattery: true }), 2);
  eq(deviceRisk(desk()), 1);
  ok(spareCapable(desk()));
  ok(!spareCapable(phone), "phones are never spares");
  ok(!spareCapable({ webgpu: true }), "a device without the spare protocol (older tab) is never a spare");
  ok(!spareCapable({ rep: 1 }), "no WebGPU, no spare");
});

Deno.test("spares: planReplicas packs the model and keeps a device free to copy a segment", () => {
  // 28 layers; host holds 18, three desktops of 12 each: one worker is enough, the other two can be spares
  const p = planReplicas(28, [18, 12, 12, 12], [desk(), desk(), desk(), desk()]);
  ok(p, "a plan");
  eq(p.assigned, [18, 10, 0, 0]);
  eq(p.ranges[1], [18, 28]);
  eq(p.used, [0, 1]);
  eq([...p.spareOf], [[1, 2]], "device 2 (smallest that fits, then lowest index) copies device 1");
  eq(p.idle, [3]);
});

Deno.test("spares: the device that takes the remainder is the smallest that holds it, so a bigger one can copy it", () => {
  // host 20 of 28; a 1 GB-ish computer (8 layers) and a 2 GB-ish one (16): the small one takes the
  // 8 left and the big one copies it (biggest-first would leave only the small one, too small)
  const p = planReplicas(28, [20, 8, 16], [desk(), desk(), desk()]);
  eq(p.assigned, [20, 8, 0]);
  eq([...p.spareOf], [[1, 2]]);
});

Deno.test("spares: planReplicas returns null when the model needs every device or no spare fits", () => {
  eq(planReplicas(28, [10, 10, 10], [desk(), desk(), desk()]), null, "every device needed");
  // worker holds 10, the only free device can hold 5
  eq(planReplicas(28, [18, 12, 5], [desk(), desk(), desk()]), null, "the free device is too small");
  // two devices of 10 for a 10-layer segment: no 10% headroom
  eq(planReplicas(28, [18, 10, 10], [desk(), desk(), desk()]), null, "no headroom");
  eq([...planReplicas(28, [18, 11, 10], [desk(), desk(), desk()]).spareOf], [[2, 1]], "the 10 takes the layers, the 11 copies them");
});

Deno.test("spares: phones hold layers only when needed, and are never spares", () => {
  // host 18, a phone that could hold 12, two desktops: the desktop takes the layers, not the phone
  const p = planReplicas(28, [18, 12, 12, 12], [desk(), phone, desk(), desk()]);
  eq(p.assigned, [18, 0, 10, 0]);
  eq([...p.spareOf], [[2, 3]]);
  eq(p.idle, [1], "the phone stays a guest");
  // a phone holds layers only once every computer does, so at deal time no computer is left to
  // copy it; a computer that joins later covers it first (coverSegments, risk 4)
  const q = planReplicas(33, [18, 6, 6, 7], [desk(), phone, desk(), desk()]);
  eq(q, null, "every device needed, the phone included");
  const r = planReplicas(30, [18, 6, 5, 7, 8], [desk(), phone, desk(), desk(), desk()]);
  eq(r.assigned, [18, 0, 4, 0, 8], "the biggest computer, then the smallest that holds what is left");
  eq([...r.spareOf], [[2, 3]], "the 7-layer computer copies the 4-layer segment");
  eq(r.idle, [1]);
});

Deno.test("spares: coverSegments covers the riskiest and biggest segment first and matches the GPU", () => {
  const segs = [
    { idx: 1, layers: 6, meta: desk("nvidia") },
    { idx: 2, layers: 4, meta: phone },
  ];
  // one spare that fits both: it goes to the phone's segment (risk 4 beats size)
  eq([...coverSegments(segs, [{ idx: 5, cap: 8, meta: desk("amd") }])], [[2, 5]]);
  // two spares: the same GPU is preferred for the desktop's segment, then the smallest cap
  const m = coverSegments(segs, [{ idx: 5, cap: 20, meta: desk("amd") }, { idx: 6, cap: 7, meta: desk("nvidia") }, { idx: 7, cap: 5, meta: desk("amd") }]);
  eq(m.get(2), 7, "smallest fitting spare for the phone's 4 layers");
  eq(m.get(1), 6, "same GPU class for the desktop's segment");
});

Deno.test("spares: a spare much slower than the device it copies is refused once measured", () => {
  const segs = [{ idx: 1, layers: 4, meta: desk(), ms: 1.0 }];
  eq(coverSegments(segs, [{ idx: 2, cap: 10, meta: desk(), ms: 1.6 }]).size, 0);
  eq(coverSegments(segs, [{ idx: 2, cap: 10, meta: desk(), ms: 1.4 }]).get(1), 2);
  eq(coverSegments(segs, [{ idx: 2, cap: 10, meta: desk() }]).get(1), 2, "not measured yet: accepted");
});

Deno.test("spares: frame numbers wrap and stale frames are recognised", () => {
  eq(fseqNext(1), 2);
  eq(fseqNext(65535), 1, "wraps to 1, never 0");
  ok(seqAfter(2, 1)); ok(!seqAfter(1, 2)); ok(seqAfter(1, 65535), "after the wrap");
  ok(seqAfter(5, 0), "unknown is always after");
  ok(seqStale(10, 10), "a duplicate");
  ok(seqStale(9, 10), "just behind");
  ok(!seqStale(11, 10), "the next one");
  ok(seqStale(65535, 3), "behind across the wrap");
  ok(!seqStale(3, 5000), "far behind: a new numbering (the host reloaded), taken");
  ok(!seqStale(0, 10) && !seqStale(10, 0), "an older sender never drops");
});

Deno.test("spares: failoverPlan fences the dead device and seats the spare", () => {
  const chain = ["a", "b", "c"];
  const mid = failoverPlan({ chain, dead: "b", spare: "s", hostId: "H" });
  eq(mid.chain, ["a", "s", "c"]);
  eq(mid.msgs, [
    { to: "a", msg: { t: "ai-next", next: "s" } },
    { to: "c", msg: { t: "ai-upstream", id: "s" } },
    { to: "s", msg: { t: "ai-promote", next: "c", prev: "a" } },
  ]);
  const first = failoverPlan({ chain, dead: "a", spare: "s", hostId: "H" });
  eq(first.msgs.map((m) => m.to), ["b", "s"], "the host is the upstream: only the new chain");
  eq(first.msgs[1].msg, { t: "ai-promote", next: "b", prev: "H" });
  const last = failoverPlan({ chain, dead: "c", spare: "s", hostId: "H" });
  eq(last.msgs.map((m) => m.to), ["b", "s"]);
  eq(last.msgs[1].msg, { t: "ai-promote", next: "host", prev: "b" });
  eq(failoverPlan({ chain, dead: "x", spare: "s", hostId: "H" }), null, "not in the chain");
  eq(failoverPlan({ chain, dead: "a", spare: "b", hostId: "H" }), null, "a spare already in the chain");
  eq(failoverPlan({ chain: ["a"], dead: "a", spare: "s", hostId: "H" }).msgs, [{ to: "s", msg: { t: "ai-promote", next: "host", prev: "H" } }]);
});

Deno.test("spares: the prober suspects only after consecutive missed probes", () => {
  const p = makeProber({ strikes: 2 });
  ok(!p.missed("a"));
  p.answered("a");
  ok(!p.missed("a"), "an answer resets the count");
  ok(p.missed("a"), "two in a row");
  eq(lapDeadline(0), 300);
  eq(lapDeadline(200, 20), 640);
});

Deno.test("spares: fseq rides the frame header and is absent (not 0) when unset", () => {
  const link = makeLink(), out = [];
  link.chans.push({ readyState: "open", send: (b) => out.push(b) });
  sendFrame(link, { t: "ai-hidden-b", basePos: 8, n: 4, fseq: 65535, data: new Uint16Array(5120 * 4) });
  sendFrame(link, { t: "ai-hidden", pos: 12, data: new Uint16Array(5120) });
  const rx = makeLink(); let h = null; const got = [];
  attachWire(rx, { peerConnection: { createDataChannel: () => ({ set onmessage(f) { h = f; }, set onclose(_) {}, readyState: "open" }) } }, (m) => got.push(m));
  for (const b of out) h({ data: b });
  eq(got.length, 2);
  eq(got[0].fseq, 65535);
  ok(!("fseq" in got[1]), "an unnumbered frame (older sender) carries no fseq");
  eq(new DataView(out[out.length - 1]).getUint16(18), 0, "bytes 18..19 stay zero when unset, as protocol 4 wrote them");
});
