// Who is still in the room: the ping loop's silent-link drop and the host's duplicate-name probe.
// DOM-free so it can be unit tested (tests/unit/liveness_test.js).
//
// Every device answers a ping at once, so a link that has said nothing for a while is a device that
// vanished without a clean close (a crashed or killed tab, a laptop asleep, Wi-Fi gone): no pagehide,
// and the data channel's own close can take minutes to surface (#124). The ping loop drops such a
// link, the same path as a close. How long "a while" is depends on who is on the other end, because
// the other side cannot tell a stalled tab from a gone one:
//   - a computer: SILENT_MS. Its tab answers pings from the message loop even when hidden.
//   - a phone: PHONE_SILENT_MS. Its screen locks and the browser freezes the page; a phone that
//     comes back within a minute keeps its link and its layers.
//   - the host, as a guest sees it: HOST_SILENT_MS. The host's main thread does the heaviest work
//     (its shard load, the first and last layers), and a guest that drops a live host tears the
//     room down for itself.
//   - a device mid-load (its shard still loading, on either end): never. A worker's main thread
//     can block for seconds on a shard, and dropping it mid-load throws away the start. A device
//     that really died there is still noticed when its data channel closes.
// Any sign of life counts: a message, a pong, a hidden-state frame or a keep-alive byte on the wire
// (room/transport.js stamps link.rxAt). A late tick means this tab was the one stalled (throttled,
// or a long task): its queue of pongs is not read yet, so it judges nobody on that tick.
export const PING_MS = 2500;
export const SILENT_MS = 15000;
export const PHONE_SILENT_MS = 60000;
export const HOST_SILENT_MS = 30000;

// the last time anything came in from a link (a message, a wire frame, a keep-alive byte)
export function lastHeard(e) {
  return Math.max(e?.seen ?? -Infinity, e?.link?.rxAt ?? -Infinity);
}

// how long a link may stay silent before it counts as gone; Infinity: never dropped for silence
export function silentLimit({ toHost = false, phone = false, loading = false } = {}) {
  if (loading) return Infinity;
  if (toHost) return HOST_SILENT_MS;
  return phone ? PHONE_SILENT_MS : SILENT_MS;
}

// one tick of the ping loop for one link: drop it?
//   heard: lastHeard(e); late: this tick fired more than 2 pings late (this tab stalled)
export function isSilentGone({ now, heard, late = false, toHost = false, phone = false, loading = false }) {
  if (late) return false;
  return now - heard > silentLimit({ toHost, phone, loading });
}

// the host: is this device still loading its layers for a start? (a device in the chain that has
// not reported ready while the start runs). Such a device is never dropped for silence.
export function midLoad({ starting, inChain, ready }) {
  return !!(starting && inChain && !ready);
}

// --- duplicate names ---
// Names are the room's keys (colours, layers, load progress, the plan): the host makes a taken one
// unique ("laptop 2"), and the device takes the name the roster gives it (renameTo).

// a name no other device in the room uses (another device's, or the host's own), with a number added
// if needed. roster: [[id, { name }]] of the other devices; id: the device asking.
export function uniqueName(name, id, hostName, roster) {
  const taken = new Set([hostName, ...[...roster].filter(([rid]) => rid !== id).map(([, m]) => m.name)]);
  if (!taken.has(name)) return name;
  const base = name.slice(0, 36);
  let n = 2;
  while (taken.has(`${base} ${n}`)) n++;
  return `${base} ${n}`;
}

// A device that asks for a name held by a device that has gone quiet is most likely that device back
// from a crash or a killed tab (its old link never closed, so the ping loop has not dropped it yet).
// The hello waits NAME_PROBE_MS while the quiet one is pinged; if it did not answer, it is dropped, so
// the newcomer keeps its name and gets its layers back. One that answers keeps its name and the
// newcomer gets a number (uniqueName).
export const NAME_PROBE_MS = 1500;
export const QUIET_MS = 1000;

// the id of a device holding `name` that has been quiet for QUIET_MS (worth a probe), or null.
// heardOf(id): lastHeard of that device's link (-Infinity when it has none).
export function quietNamesake(name, id, roster, heardOf, now) {
  for (const [rid, m] of roster) {
    if (rid === id || m.name !== name) continue;
    if (!(now - heardOf(rid) < QUIET_MS)) return rid;
  }
  return null;
}

// after the probe: the ids holding `name` that said nothing since the probe went out at `since`
export function staleNamesakes(name, id, roster, heardOf, since) {
  const out = [];
  for (const [rid, m] of roster) {
    if (rid === id || m.name !== name) continue;
    if (heardOf(rid) >= since) continue;   // it answered: alive
    out.push(rid);
  }
  return out;
}

// a device: the name the host's roster lists this device under, when it differs from its own
// (the one it asked for was taken); null otherwise. Only the host's roster counts.
export function renameTo(members, selfId, myName, fromHost) {
  if (!fromHost || !Array.isArray(members)) return null;
  const mine = members.find((m) => m.id === selfId);
  return mine && mine.name && mine.name !== myName ? mine.name : null;
}
