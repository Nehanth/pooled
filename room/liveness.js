// Drop detection: noticing a device in the chain that went silent mid-answer (its network died,
// its tab froze or was killed) in seconds instead of waiting for ICE to give up (~30 s) or for
// the lap timeouts (30-90 s).
//
// The host pings every device in the chain every HB_BUSY_MS while an answer runs (the plain
// `ping`/`pong` every protocol version answers, so this needs no protocol change) and counts
// anything it receives from a device as a sign of life: a pong, any control message, any slice
// on any of its wire channels. A device silent for longer than deadAfter(rtt) is dead: the host
// closes its link, which takes the existing path (every lap in flight fails, `ai-degraded`, re-deal).
//
// Why these numbers: the control channel is reliable and ordered, so one lost packet holds
// everything behind it until SCTP retransmits it (an RTO, ~1 s at first, doubling on each loss of
// the same packet). The silence limit has to cover a double loss at the worst latency we support
// (300 ms one way, 5% loss) without a false alarm, and stay within ~3-5 s. Measured with
// tests/e2e/room_drop.mjs (docs/protocol.md, "Drop detection").
//
// Pure: no DOM, no timers; the caller passes the clock.

export const HB_BUSY_MS = 500;     // ping period to chain devices while answering
export const DEAD_MIN_MS = 3000;   // never call a device dead sooner than this
export const DEAD_MAX_MS = 5000;   // ... nor later
export const STALL_MS = 1200;      // a tick this late means this tab stalled: its inbox is stale, judge nothing

// How long a device may stay silent before it counts as dead, given its measured round trip.
export function deadAfter(rttMs) {
  const rtt = Number.isFinite(rttMs) && rttMs > 0 ? rttMs : 0;
  return Math.round(Math.min(DEAD_MAX_MS, Math.max(DEAD_MIN_MS, 2500 + 2 * rtt)));
}

// Tracks when each device was last heard from and who went silent while armed (an answer runs).
export function makeLiveness() {
  return { heard: new Map(), since: 0, armed: false, lastTick: 0, lastPing: 0, maxSilence: new Map() };
}
export function heard(L, id, now) { L.heard.set(id, now); }
// Start judging (an answer began): silence is counted from now at the earliest, so a device that
// was idle between two pings is not held to the busy-time limit for time before the answer.
export function arm(L, now) { if (!L.armed) { L.armed = true; L.since = now; L.lastTick = now; L.maxSilence.clear(); } }
export function disarm(L) { L.armed = false; }
export function forget(L, id) { L.heard.delete(id); L.maxSilence.delete(id); }

// One check. ids: the chain; rttOf(id): its last measured round trip (ms) or null.
// Returns { ping: bool (time to send the next heartbeat), dead: [{ id, silentMs, limitMs }] }.
export function tick(L, now, ids, rttOf = () => null) {
  const out = { ping: false, dead: [] };
  if (!L.armed) { L.lastTick = now; return out; }
  // this tab did not run for a while (a long GPU submit, a background-tab throttle): messages that
  // arrived meanwhile are still queued behind this timer, so silence measured now is not theirs
  if (now - L.lastTick > STALL_MS) L.since = now;
  L.lastTick = now;
  if (now - L.lastPing >= HB_BUSY_MS) { out.ping = true; L.lastPing = now; }
  for (const id of ids) {
    const silent = now - Math.max(L.heard.get(id) ?? 0, L.since);
    if (silent > (L.maxSilence.get(id) ?? 0)) L.maxSilence.set(id, silent);
    const limit = deadAfter(rttOf(id));
    if (silent > limit) out.dead.push({ id, silentMs: Math.round(silent), limitMs: limit });
  }
  return out;
}

// Lap timeout for a decode lap (one token, or a speculative verify) from the laps measured so
// far: generous against the slowest recent lap, but seconds rather than the 30-90 s fallbacks
// used before any lap is measured (the first laps compile pipelines and warm caches).
// stat: { n, lap (EMA ms), max (decaying max ms) }; fallback: the old fixed timeout.
export const LAP_MIN_MS = 8000;
export function lapTimeout(stat, fallback, rttMs = 0) {
  if (!stat || stat.n < 4 || !(stat.max > 0)) return fallback;
  const rtt = Number.isFinite(rttMs) && rttMs > 0 ? rttMs : 0;
  return Math.round(Math.min(fallback, Math.max(LAP_MIN_MS, 5 * Math.max(stat.max, stat.lap) + 2 * rtt + 2000)));
}
