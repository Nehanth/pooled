// room/liveness.js: drop detection on the host. A fake clock drives it; the numbers mirror what
// the host sees in a room (pings every 500 ms, pongs, wire slices, a tab that stalls).
import { makeLiveness, heard, arm, disarm, forget, tick, deadAfter, lapTimeout,
  HB_BUSY_MS, DEAD_MIN_MS, DEAD_MAX_MS, STALL_MS, LAP_MIN_MS } from "../../room/liveness.js";

function ok(c, m) { if (!c) throw new Error(m || "assertion failed"); }
function eq(a, b, m) { const x = JSON.stringify(a), y = JSON.stringify(b); if (x !== y) throw new Error(`${m ? m + ": " : ""}${x} !== ${y}`); }

// drive one device through a run: hearFrom(t) says whether a message arrives at time t
function run(L, { ms, step = 250, ids = ["a"], hearFrom = () => true, rtt = () => 50, t0 = 0 }) {
  const dead = [];
  let pings = 0;
  for (let t = t0; t <= t0 + ms; t += step) {
    for (const id of ids) if (hearFrom(id, t)) heard(L, id, t);
    const r = tick(L, t, ids, rtt);
    if (r.ping) pings++;
    for (const d of r.dead) dead.push({ t, ...d });
  }
  return { dead, pings };
}

Deno.test("deadAfter: 3.5 s on a LAN, grows with the round trip, capped at 5 s", () => {
  eq(deadAfter(null), DEAD_MIN_MS);
  eq(deadAfter(0), DEAD_MIN_MS);
  eq(deadAfter(20), DEAD_MIN_MS);
  eq(deadAfter(300), 3900);
  eq(deadAfter(600), 4800);
  eq(deadAfter(5000), DEAD_MAX_MS);
  eq(deadAfter(NaN), DEAD_MIN_MS);
});

Deno.test("nothing is judged until armed, and a live device is never dead", () => {
  const L = makeLiveness();
  eq(tick(L, 10000, ["a"]).dead, [], "unarmed");
  arm(L, 10000);
  const r = run(L, { t0: 10000, ms: 60000, hearFrom: (_, t) => t % 500 === 0 });
  eq(r.dead, []);
  ok(r.pings >= 60000 / HB_BUSY_MS - 1, "pings every 500 ms: " + r.pings);
});

Deno.test("a device that goes silent is dead within its limit (+ one tick), and only it", () => {
  const L = makeLiveness();
  arm(L, 0);
  const dieAt = 5000;
  const r = run(L, { ms: 20000, ids: ["a", "b"], hearFrom: (id, t) => id === "a" || t < dieAt });
  ok(r.dead.length > 0, "b called dead");
  ok(r.dead.every((d) => d.id === "b"), "only b");
  const first = r.dead[0];
  ok(first.t - dieAt > DEAD_MIN_MS - 250 && first.t - dieAt <= DEAD_MIN_MS + 500, "detected at " + (first.t - dieAt));
  eq(first.limitMs, deadAfter(50));
});

Deno.test("silence before the answer began does not count (idle pings are 2.5 s apart)", () => {
  const L = makeLiveness();
  heard(L, "a", 0);            // the last idle pong
  arm(L, 2400);                // a question starts 2.4 s later
  const r = run(L, { t0: 2400, ms: 2500, hearFrom: () => false });
  eq(r.dead, [], "counted from arm time, not from the idle pong");
  const r2 = run(L, { t0: 5150, ms: 1000, hearFrom: () => false });
  ok(r2.dead.length > 0, "but a device silent since the start is dead ~3.5 s in");
});

Deno.test("a stalled host tab (late tick) resets the baseline instead of blaming everyone", () => {
  const L = makeLiveness();
  arm(L, 0);
  run(L, { ms: 1000 });                      // heard up to t=1000
  // the host's main thread blocks for 6 s; the pongs that came meanwhile are still queued
  const r = tick(L, 7000, ["a"]);
  eq(r.dead, [], "no verdict on the first tick after a stall");
  ok(7000 - L.lastTick === 0 && L.since === 7000);
  // the queued pong is handled next, and it goes on normally
  const r2 = run(L, { t0: 7250, ms: 5000, hearFrom: (_, t) => t % 500 === 0 });
  eq(r2.dead, []);
  ok(STALL_MS < DEAD_MIN_MS);
});

Deno.test("300 ms latency with retransmission stalls: gaps under the limit are tolerated", () => {
  // a double SCTP loss at RTT 600 ms: ~1 s + 2 s of head-of-line blocking on top of the ping period
  const L = makeLiveness();
  arm(L, 0);
  // the e2e run (tests/e2e/room_drop.mjs noise600) saw 3.65 s of silence at RTT 600 ms + 5% loss
  const holes = [[4000, 6500], [20000, 23700]];
  const r = run(L, { ms: 40000, rtt: () => 600, hearFrom: (_, t) => t % 500 === 0 && !holes.some(([a, b]) => t >= a && t < b) });
  eq(r.dead, []);
  ok(L.maxSilence.get("a") >= 3600, "the stall was seen: " + L.maxSilence.get("a"));
});

Deno.test("disarm stops judging; forget drops a device", () => {
  const L = makeLiveness();
  arm(L, 0);
  run(L, { ms: 1000 });
  disarm(L);
  eq(tick(L, 60000, ["a"]).dead, []);
  forget(L, "a");
  ok(!L.heard.has("a") && !L.maxSilence.has("a"));
  arm(L, 60000);
  eq(L.since, 60000, "re-armed from now");
  for (const t of [61000, 62000, 63000, 63400]) eq(tick(L, t, ["a"]).dead, [], "t=" + t);
  ok(tick(L, 63600, ["a"]).dead.length === 1, "silent 3.6 s since re-arming");
});

Deno.test("lapTimeout: fixed fallback until 4 laps are measured, then tied to the slowest lap", () => {
  eq(lapTimeout(null, 30000), 30000);
  eq(lapTimeout({ n: 3, lap: 200, max: 300 }, 30000), 30000, "too few laps");
  eq(lapTimeout({ n: 10, lap: 200, max: 300 }, 30000), LAP_MIN_MS, "fast laps: the floor");
  eq(lapTimeout({ n: 10, lap: 800, max: 1500 }, 90000, 600), LAP_MIN_MS, "a 1.5 s lap: still the floor");
  eq(lapTimeout({ n: 10, lap: 2000, max: 2500 }, 90000, 600), 6 * 2500 + 1200 + 2000);
  eq(lapTimeout({ n: 10, lap: 3000, max: 3300 }, 30000, 600), 6 * 3300 + 1200 + 2000, "RTT 600 ms + 5% loss: clear of the 14 s stalls seen there");
  eq(lapTimeout({ n: 10, lap: 9000, max: 12000 }, 30000), 30000, "never above the fallback");
  eq(lapTimeout({ n: 10, lap: 400, max: 0 }, 30000), 30000, "no max yet");
});
