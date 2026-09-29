// Drop detection: noticing a device in the chain that went silent mid-answer (its network died,
// its tab froze or was killed) in seconds instead of waiting for ICE to give up (~30 s) or for
// the lap timeouts (30-90 s).
//
// The host pings every device in the chain every HB_BUSY_MS while an answer runs (the plain
// `ping`/`pong` every protocol version answers, so this needs no protocol change) and counts
// anything it receives from a device as a sign of life: a pong, any control message, any slice
// on any of its wire channels. A device silent for longer than deadAfter(rtt) is held: every lap
// in flight fails now ("<name> stopped responding; ask again") and the next question waits for it.
// If it is heard again it simply carries on (a freeze); silent past EVICT_MS it is dropped, which
// takes the existing path (`ai-degraded`, re-deal).
//
// Why these numbers: the control channel is reliable and ordered, so one lost packet holds
// everything behind it until SCTP retransmits it (an RTO, ~1 s at first, doubling on each loss of
// the same packet). The silence limit has to cover a double loss at the worst latency we support
// (300 ms one way, 5% loss) without a false alarm, and stay within ~3-5 s. Measured with
// tests/e2e/room_drop.mjs on 3 devices, Qwen3 1.7B: the longest silence was 0.65 s on a LAN,
// 2.5 s at RTT 300 ms + 5% loss and 3.65 s at RTT 600 ms + 5% loss, against limits of 3.5, 3.9
// and 4.8 s (docs/protocol.md, "Drop detection").
//
// Pure: no DOM, no timers; the caller passes the clock.

export const HB_BUSY_MS = 500;     // ping period to chain devices while answering
export const DEAD_MIN_MS = 3500;   // never call a device dead sooner than this
export const DEAD_MAX_MS = 5000;   // ... nor later
export const STALL_MS = 1200;
// A device that went silent is held, not dropped: the answer in flight fails at once (its frames
// are late at best), but the device keeps its place and layers in case it was only frozen (a
// Wi-Fi stall, a laptop lid, a busy phone). Only past EVICT_MS of silence is it dropped (the
// re-deal path). 15 s is when ICE gives up on a link anyway (see watchLink in room.js).
export const EVICT_MS = 15000;      // a tick this late means this tab stalled: its inbox is stale, judge nothing

// How long a device may stay silent before it counts as dead, given its measured round trip.
export function deadAfter(rttMs) {
  const rtt = Number.isFinite(rttMs) && rttMs > 0 ? rttMs : 0;
  return Math.round(Math.min(DEAD_MAX_MS, Math.max(DEAD_MIN_MS, 3000 + 3 * rtt)));
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

// A held (suspect) device, checked on every loop whether or not an answer runs: "back" once
// anything arrived from it after it went silent, "evict" once it has been silent EVICT_MS,
// otherwise "wait". lastHeard: when it was last heard from; silentSince: when the silence began
// (lastHeard, or when the answer started if that was later).
export function suspectCheck(lastHeard, silentSince, now) {
  if (lastHeard > silentSince) return "back";
  return now - silentSince > EVICT_MS ? "evict" : "wait";
}

// Lap timeout for a decode lap (one token, or a speculative verify) from the laps measured so
// far: generous against the slowest recent lap, but seconds rather than the 30-90 s fallbacks
// used before any lap is measured (the first laps compile pipelines and warm caches). A dead
// device is caught by the silence check above; this is for a frame lost on a live chain, so it
// stays well clear of the stalls a lossy link causes (14 s between two tokens at RTT 600 ms and
// 5% loss, with laps averaging 3 s).
// stat: { n, lap (EMA ms), max (decaying max ms) }; fallback: the old fixed timeout.
export const LAP_MIN_MS = 15000;
export function lapTimeout(stat, fallback, rttMs = 0) {
  if (!stat || stat.n < 4 || !(stat.max > 0)) return fallback;
  const rtt = Number.isFinite(rttMs) && rttMs > 0 ? rttMs : 0;
  return Math.round(Math.min(fallback, Math.max(LAP_MIN_MS, 6 * Math.max(stat.max, stat.lap) + 2 * rtt + 2000)));
}
