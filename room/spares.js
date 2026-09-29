// Spare copies: frame numbering, fencing and the failover rewiring. DOM-free so it can be unit
// tested (tests/unit/spares_test.js); room.js does the sending. docs/protocol.md "Spare copies".

// ---- frame numbers (fseq) ----
// The host numbers every compute frame it sends (u16, 1..65535, then 1 again; 0 = none, what an
// older sender writes). Every device copies the number onto the frame it forwards, and the return
// frame carries it back, so the host can tell the lap it is waiting for from a stale one.
export const fseqNext = (f) => (f % 65535) + 1;
// a is after b, on the 16-bit circle (half the circle counts as ahead)
export function seqAfter(a, b) {
  if (!a || !b) return true;
  const d = (a - b + 65535) % 65535;
  return d > 0 && d < 32768;
}
// A frame to drop as already seen: at or up to `win` behind the last one queued. Anything further
// behind is a new numbering (the host page reloaded), so it is taken. 0 on either side (an older
// sender, or nothing queued yet) never drops.
export function seqStale(f, last, win = 64) {
  if (!f || !last) return false;
  const back = (last - f + 65535) % 65535;
  return back < win;
}

// ---- failover ----
// The chain is the host (index -1, "host"), then chain[0..n-1], then back to the host. When
// chain[i] (dead) has a ready spare, the spare takes slot i:
//   the device before it forwards to the spare from now on (ai-next): the dead one gets no new input,
//   the device after it takes frames only from the spare (ai-upstream): late frames from the dead
//     one, if it is still alive somewhere, are dropped,
//   the spare becomes a worker (ai-promote) with the dead one's neighbours.
// The host is its own neighbour at either end: it sends to chain[0] and takes returns only from
// the last device, so a change there is just the new chain. Returns null when nothing can be done.
export function failoverPlan({ chain, dead, spare, hostId }) {
  const i = chain.indexOf(dead);
  if (i < 0 || !spare || chain.includes(spare)) return null;
  const up = i > 0 ? chain[i - 1] : null;             // null = the host
  const down = i + 1 < chain.length ? chain[i + 1] : null;
  const msgs = [];
  if (up) msgs.push({ to: up, msg: { t: "ai-next", next: spare } });
  if (down) msgs.push({ to: down, msg: { t: "ai-upstream", id: spare } });
  msgs.push({ to: spare, msg: { t: "ai-promote", next: down || "host", prev: up || hostId } });
  const next = chain.slice();
  next[i] = spare;
  return { i, up, down, msgs, chain: next };
}

// ---- stall probes ----
// A lap that is late gets the chain probed (ai-probe, answered by the page's message loop, which
// stays responsive while its GPU works); a device that misses `strikes` probes in a row while the
// lap is still late is suspected. A frozen tab (a phone with its screen off, a hung renderer) stops
// answering; a slow one does not, so a slow phone never flaps. Returns the ids to suspect.
export function makeProber({ strikes = 2 } = {}) {
  const miss = new Map();   // id -> consecutive missed probes
  return {
    missed(id) { const k = (miss.get(id) || 0) + 1; miss.set(id, k); return k >= strikes; },
    answered(id) { miss.delete(id); },
    clear() { miss.clear(); },
    strikes: (id) => miss.get(id) || 0,
  };
}
// How late a lap may be before the chain is probed: three times the usual lap plus two hops, at
// least 300 ms.
export const lapDeadline = (lapEmaMs, hopMs = 20) => Math.max(300, 3 * (lapEmaMs || 0) + 2 * hopMs);
