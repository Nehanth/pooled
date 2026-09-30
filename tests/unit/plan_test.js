// room/plan.js layer dealing: planSplit, planForSpeed and pickModelHost. Table-driven edge cases
// plus property tests over seeded random rooms (1 to 6 devices, tiny to huge pledges). No GPU.
import { planSplit, planForSpeed, pickModelHost } from "../../room/plan.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const sum = (a) => a.reduce((s, x) => s + x, 0);
const GB = 2 ** 30;

// mulberry32: a small seeded PRNG so a failing case can be replayed from its seed
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const int = (r, lo, hi) => lo + Math.floor(r() * (hi - lo + 1));
const pick = (r, xs) => xs[Math.floor(r() * xs.length)];

// the shape every deal must have: one count per device, every layer placed exactly once, in
// contiguous ranges starting at 0 in device order
function checkShape(L, n, p, ctx) {
  eq(p.assigned.length, n, ctx + " one count per device");
  eq(p.ranges.length, n, ctx + " one range per device");
  ok(p.assigned.every((a) => Number.isInteger(a) && a >= 0), ctx + " whole, non-negative counts " + JSON.stringify(p.assigned));
  eq(sum(p.assigned), L, ctx + " every layer placed once");
  let acc = 0;
  p.ranges.forEach(([lo, hi], i) => {
    eq(lo, acc, ctx + " contiguous at device " + i);
    eq(hi - lo, p.assigned[i], ctx + " range matches count at device " + i);
    acc = hi;
  });
  eq(acc, L, ctx + " ranges end at L");
}

// ---------------------------------------------------------------- planSplit

Deno.test("planSplit: table of edge cases", () => {
  const cases = [
    // [name, L, caps, expected assigned]
    ["1 device takes everything", 64, [5 * GB], [64]],
    ["5 equal devices, leftover to the first", 64, [1, 1, 1, 1, 1], [13, 13, 13, 13, 12]],
    ["the MacBook + iPhone demo", 64, [62, 2], [62, 2]],
    ["the demo in bytes (only ratios matter)", 64, [62 * GB, 2 * GB], [62, 2]],
    ["uneven leftovers go to the largest remainders", 7, [3, 3, 3], [3, 2, 2]],
    // host whose pledge barely covers the embedding: its cap is layerBytes/2 in room.js
    ["tiny host cap still keeps a layer", 32, [0.5, 70, 70], [1, 15, 16]],
    ["tiny host next to one big peer", 64, [0.05 * GB, 14 * GB, 0.5 * GB], [1, 61, 2]],
    ["tiny peer still gets a layer", 10, [100, 1, 1, 1, 1], [6, 1, 1, 1, 1]],
    // fewer layers than devices: the host keeps its layer, the last devices wait
    ["L=1 over 3 devices", 1, [1, 1, 1], [1, 0, 0]],
    ["L=2 over 3 devices", 2, [1, 1, 1], [1, 1, 0]],
    ["L=3 over 5 devices", 3, [1, 1, 1, 1, 1], [1, 1, 1, 0, 0]],
    ["L=4 over 5 devices", 4, [1, 1, 1, 1, 1], [1, 1, 1, 1, 0]],
    ["L=5 over 5 devices", 5, [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
    // zero, malformed and huge pledges
    ["all caps 0: dealt evenly", 4, [0, 0], [2, 2]],
    ["host cap 0", 10, [0, 5, 5], [1, 4, 5]],
    ["NaN cap counts as 0", 4, [1, NaN], [3, 1]],
    ["undefined cap counts as 0", 4, [1, undefined], [3, 1]],
    ["negative cap counts as 0", 64, [-5, 10], [1, 63]],
    ["all caps NaN: dealt evenly", 6, [NaN, NaN, NaN], [2, 2, 2]],
    ["huge caps do not overflow", 64, [Number.MAX_VALUE, Number.MAX_VALUE], [32, 32]],
    ["huge next to tiny", 64, [1e15, 1, 1], [62, 1, 1]],
    ["L=0 places nothing", 0, [1, 1], [0, 0]],
  ];
  for (const [name, L, caps, want] of cases) {
    const p = planSplit(L, caps);
    eq(p.assigned, want, name);
    checkShape(L, caps.length, p, name);
  }
});

Deno.test("planSplit: does not mutate the caps it is given", () => {
  const caps = [NaN, 3, 0, 5];
  planSplit(8, caps);
  eq(caps.map(String), ["NaN", "3", "0", "5"]);
});

Deno.test("planSplit: properties over 3000 seeded random rooms", () => {
  const r = rng(0x5eed1);
  const capGen = [() => r() * 16 * GB, () => 1 + r() * 4, () => r() < 0.5 ? 0.001 : 1e12, () => int(r, 0, 3)];
  for (let t = 0; t < 3000; t++) {
    const n = int(r, 1, 6);
    const L = pick(r, [int(r, 1, n + 1), int(r, 1, 130), 24, 28, 32, 36, 48, 64, 80]);
    const gen = pick(r, capGen);
    const caps = Array.from({ length: n }, gen);
    const ctx = `seed case ${t} planSplit(${L}, ${JSON.stringify(caps)})`;
    const p = planSplit(L, caps);
    checkShape(L, n, p, ctx);
    ok(p.assigned[0] >= 1, ctx + " host keeps a layer: " + JSON.stringify(p.assigned));
    if (L >= n) ok(p.assigned.every((a) => a >= 1), ctx + " every device gets a layer: " + JSON.stringify(p.assigned));
    else {
      eq(p.assigned.filter((a) => a > 0).length, L, ctx + " L devices hold one layer each");
      ok(p.assigned.every((a) => a <= 1), ctx + " nobody holds two while others wait");
    }
    // proportional: off from the exact share by at most the leftover and the one-each fix-ups
    const tot = sum(caps);
    if (tot > 0 && L >= n)
      p.assigned.forEach((a, i) => ok(Math.abs(a - L * caps[i] / tot) <= n, ctx + ` device ${i} far from its share`));
    // deterministic, and scaling every pledge by a power of two changes nothing
    eq(planSplit(L, caps).assigned, p.assigned, ctx + " deterministic");
    eq(planSplit(L, caps.map((c) => c * 1024)).assigned, p.assigned, ctx + " scale invariant");
  }
});

// ---------------------------------------------------------------- planForSpeed

Deno.test("planForSpeed: table of edge cases", () => {
  const cases = [
    // [name, L, caps (layers each can hold), msPerLayer, expected assigned]
    ["1 device holds everything", 64, [64], [], [64]],
    ["1 device overflowing: short, nothing dealt", 64, [10], [], [0]],
    ["L=1: only the host", 1, [5, 5], [], [1, 0]],
    ["fractional caps are floored", 10, [2.7, 3.9, 8.5], [], [2, 0, 8]],
    ["host cap below 1: short, nothing dealt", 10, [0.5, 20], [], [0, 0]],
    ["host cap 0: short, nothing dealt", 10, [0, 20], [], [0, 0]],
    ["partial msPerLayer: unmeasured count as slower", 10, [5, 5, 5], [null, 2], [5, 5, 0]],
    ["msPerLayer null behaves like none", 10, [5, 5], null, [5, 5]],
    ["msPerLayer longer than caps is ignored past the end", 10, [5, 6], [1, 2, 3], [5, 5]],
    ["zero and negative ms count as unmeasured", 10, [5, 5, 5], [0, -3, 1], [5, 0, 5]],
    ["5 devices measured: fastest filled first, host keeps 1", 64, [20, 20, 20, 20, 20], [5, 1, 3, 2, 4], [1, 20, 20, 20, 3]],
    ["5 devices, some measured: measured first, then lowest index", 64, [20, 20, 20, 20, 20], [5, null, 3, null, 4], [20, 4, 20, 0, 20]],
    // overflow: nobody can hold the rest; nothing is spread past a pledge (it used to be, by capacity)
    ["short: everyone full", 20, [2, 3, 4], [], [0, 0, 0]],
    ["short with fractional caps", 20, [2.5, 3.5], [], [0, 0]],
    ["short with every cap 0", 20, [0, 0, 0], [], [0, 0, 0]],
    // more devices than layers left to place
    ["n > left: only as many devices as needed", 3, [1, 1, 1, 1, 1], [], [1, 1, 1, 0, 0]],
    ["n > left, all caps 0: short", 2, [0, 0, 0, 0, 0], [], [0, 0, 0, 0, 0]],
    // malformed caps hold nothing
    ["NaN cap holds nothing", 10, [5, NaN, 20], [], [5, 0, 5]],
    ["undefined cap holds nothing", 10, [5, undefined, 20], [], [5, 0, 5]],
    ["negative cap holds nothing", 10, [5, -3, 20], [], [5, 0, 5]],
    ["huge pledges", 64, [1e9, 1e9], [], [64, 0]],
  ];
  for (const [name, L, caps, ms, want] of cases) {
    const p = planForSpeed(L, caps, ms);
    eq(p.assigned, want, name);
    if (p.short) { ok(sum(want) === 0, name + " short deals nothing"); continue; }
    checkShape(L, caps.length, p, name);
    eq(p.used, want.map((a, i) => (a > 0 ? i : -1)).filter((i) => i >= 0), name + " used");
  }
});

Deno.test("planForSpeed: properties over 3000 seeded random rooms", () => {
  const r = rng(0xfa57);
  for (let t = 0; t < 3000; t++) {
    const n = int(r, 1, 6);
    const L = pick(r, [int(r, 1, 8), int(r, 1, 130), 32, 64]);
    const capGen = pick(r, [() => int(r, 0, 40), () => r() * 30, () => int(r, 0, 3), () => int(r, 20, 200)]);
    const caps = Array.from({ length: n }, capGen);
    const msGen = pick(r, ["none", "all", "some", "ties"]);
    const ms = msGen === "none" ? [] : caps.map(() =>
      msGen === "ties" ? pick(r, [2, 5]) : msGen === "some" && r() < 0.4 ? null : 0.5 + r() * 20);
    const ctx = `seed case ${t} planForSpeed(${L}, ${JSON.stringify(caps)}, ${JSON.stringify(ms)})`;
    const p = planForSpeed(L, caps, ms);
    eq(planForSpeed(L, caps, ms).assigned, p.assigned, ctx + " deterministic");
    // what each device may hold: its cap, never more (a pledge is a promise); the host needs one
    const lim = caps.map((c) => Math.floor(c));
    const room = sum(lim);
    if (L > room || lim[0] < 1) {
      ok(p.short >= Math.max(1, L - room), ctx + ` short ${p.short}`);
      eq(sum(p.assigned), 0, ctx + " a short room deals nothing");
      continue;
    }
    eq(p.short, 0, ctx + " fits");
    checkShape(L, n, p, ctx);
    ok(p.assigned[0] >= 1, ctx + " host keeps a layer");
    eq(p.used, p.assigned.map((a, i) => (a > 0 ? i : -1)).filter((i) => i >= 0), ctx + " used lists holders in order");
    p.assigned.forEach((a, i) => ok(a <= lim[i], ctx + ` device ${i} over its cap: ${a} > ${lim[i]}`));
    // speed order: a strictly faster device is never left with room while a slower one holds
    // layers beyond the host's mandatory one
    const known = ms.filter((x) => x > 0);
    const fb = known.length ? Math.max(...known) * 1.5 : 1;
    const cost = caps.map((_, i) => (ms[i] > 0 ? ms[i] : fb));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      if (!(cost[i] < cost[j])) continue;
      const slowerHolds = p.assigned[j] > (j === 0 ? 1 : 0);
      if (slowerHolds) eq(p.assigned[i], lim[i], ctx + ` faster device ${i} not full while slower ${j} holds layers`);
    }
  }
});

Deno.test("planForSpeed: when unmeasured, the host alone is used if it can hold every layer", () => {
  const r = rng(7);
  for (let t = 0; t < 500; t++) {
    const n = int(r, 1, 6);
    const L = int(r, 1, 100);
    const caps = [L + int(r, 0, 50), ...Array.from({ length: n - 1 }, () => int(r, 0, 200))];
    eq(planForSpeed(L, caps).used, [0], `planForSpeed(${L}, ${JSON.stringify(caps)})`);
  }
});

Deno.test("planForSpeed: unmeasured phones go after computers; a measured phone keeps its speed", () => {
  const cases = [
    // [name, L, caps, msPerLayer, phone, expected assigned]
    ["computers can hold it: the phone asks only", 10, [5, 8, 8], [], [false, true, false], [5, 0, 5]],
    ["same room without phone flags: lowest index first", 10, [5, 8, 8], [], [], [5, 5, 0]],
    ["a phone is needed: it takes only the rest", 20, [8, 8, 10], [], [false, false, true], [8, 8, 4]],
    ["a phone host still keeps its one layer", 10, [5, 20], [], [true, false], [1, 9]],
    ["measured phone faster than an unmeasured computer is used first", 10, [5, 8, 8], [1, 0.5, null], [false, true, false], [2, 8, 0]],
    ["measured computers still beat an unmeasured phone", 10, [5, 8, 8], [1, null, 2], [false, true, false], [5, 0, 5]],
  ];
  for (const [name, L, caps, ms, phone, want] of cases) {
    const p = planForSpeed(L, caps, ms, phone);
    eq(p.assigned, want, name);
    checkShape(L, caps.length, p, name);
  }
});

// ---------------------------------------------------------------- pickModelHost

Deno.test("pickModelHost: table of edge cases", () => {
  const cases = [
    ["empty list", [], null],
    ["only null entries", [null, undefined], null],
    ["entries without an id are skipped", [null, { id: "" }, { meta: { webgpu: true } }, { id: "x" }], "x"],
    ["no meta at all: the lowest id", [{ id: "b" }, { id: "a" }, { id: "c" }], "a"],
    ["meta null", [{ id: "b", meta: null }, { id: "a", meta: null }], "a"],
    ["ties go to the lowest id", [{ id: "b", meta: { webgpu: true, contribGB: 4 } }, { id: "a", meta: { webgpu: true, contribGB: 4 } }], "a"],
    ["webgpu beats no meta", [{ id: "a" }, { id: "z", meta: { webgpu: true } }], "z"],
    ["contribGB as a string still counts", [{ id: "a", meta: { webgpu: true, contribGB: "2" } }, { id: "b", meta: { webgpu: true, contribGB: "8" } }], "b"],
    ["garbage contribGB counts as 0", [{ id: "a", meta: { webgpu: true, contribGB: "lots" } }, { id: "b", meta: { webgpu: true, contribGB: 1 } }], "b"],
    ["contribGB without webgpu does not count", [{ id: "a", meta: { contribGB: 64 } }, { id: "b", meta: { contribGB: 1 } }], "a"],
    ["iPhone user agent is a phone", [{ id: "a", meta: { webgpu: true, ua: "iPhone", contribGB: 8 } }, { id: "b", meta: { webgpu: true, contribGB: 1 } }], "b"],
    ["a phone with webgpu beats a computer without", [{ id: "a", meta: { ua: "Mac" } }, { id: "b", meta: { webgpu: true, phone: true } }], "b"],
    ["only phones: the most memory", [{ id: "a", meta: { webgpu: true, phone: true, contribGB: 2 } }, { id: "b", meta: { webgpu: true, phone: true, contribGB: 6 } }], "b"],
  ];
  for (const [name, devices, want] of cases) eq(pickModelHost(devices), want, name);
});

Deno.test("pickModelHost: every screen picks the same host whatever the list order", () => {
  const r = rng(0xb0057);
  for (let t = 0; t < 2000; t++) {
    const n = int(r, 1, 6);
    const devices = Array.from({ length: n }, (_, i) => ({
      id: "p" + String.fromCharCode(97 + int(r, 0, 25)) + i,
      meta: r() < 0.1 ? undefined : {
        webgpu: r() < 0.8,
        phone: r() < 0.3,
        ua: pick(r, ["Mac", "Windows", "Android", "iPhone", ""]),
        contribGB: pick(r, [0, 1, 2, 4, 4, 8, 16]),
      },
    }));
    const want = pickModelHost(devices);
    ok(devices.some((d) => d.id === want), "picks one of the devices");
    for (let k = 0; k < 3; k++) {
      const shuffled = [...devices];
      for (let i = shuffled.length - 1; i > 0; i--) { const j = int(r, 0, i); [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]; }
      eq(pickModelHost(shuffled), want, `case ${t} order ${k}`);
    }
  }
});
