// Who runs the room's model (#17). The decision is decentralized; the facts come from the room's creator.
//
// The model host deals the layers (ai-load) and sends what FROM_HOST lists (ai-layers, ai-token,
// ai-ready-all ...). It is the strongest device of the room (room/plan.js pickModelHost: a computer before a
// phone, then the most memory lent, then GPU speed, then the lowest id), whoever pressed Start. A device used
// to take a new model host from anyone in the room, so any admitted member could make itself another device's
// host and write into its chat.
//
// Facts: the creator's roster is the one source of who is in the room, in what order they joined (`seq`, a
// number the creator gives each device once and keeps across relinks and a resume) and the metas the pick is
// made with. A device ignores metas it hears on its own links (hellos, pledges) for this. So the creator is
// trusted for those facts, and nothing else: it can't make the rule pick a device it doesn't pick.
//
// Decision: any device may press Start. ai-start-req {boss, ids} names the pick and the devices it was made
// over; it goes to the creator, which checks it and relays it, and every device runs the same pick over those
// devices with the creator's facts (verifyHost) and follows only when it lands on boss. A device takes a Start
// only from the creator: one a member sends it directly is ignored (the relayed copy follows). A device that was not there for the Start (it joined later, or reloaded) takes a
// host only from the creator: the creator itself, or the host the creator names in ai-modelhost after it
// checked that Start itself. Once a device follows a host it takes ai-load and FROM_HOST messages only from it,
// and refuses another Start while that host is in the room and dealing (startBusy).
//
// Pledges, the device kind (ua) and the GPU speed in a meta are what the device says about itself: unverified
// claims. saneMeta only holds them to the caps the room already has (room/pledge.js: the kind's cap, the
// device's own pledgeMax, 64 GB). A member that claims more memory, or a faster GPU, than it has can win the
// pick; it is demoted only if it then fails to load its share (room/pledge.js afterLoadDeath).

import { pickModelHost } from "./plan.js";
import { pledgeGB } from "./pledge.js";

export const JOIN_SLACK_MS = 10000;     // a device seen for less than this may be missing from a Start's ids
export const GBPS_MAX = 2000;           // the most GPU speed a meta may claim
export const START_PENDING_MS = 60000;  // how long a followed Start waits for its host to start dealing

// what the pick takes from a meta: its pledge held to the room's caps (a claim all the same)
export function saneMeta(meta) {
  if (!meta || typeof meta !== "object") return {};
  const m = { ...meta };
  m.contribGB = m.webgpu ? Math.min(64, pledgeGB(m)) : 0;
  m.gbps = Math.min(GBPS_MAX, Math.max(0, +m.gbps || 0));
  return m;
}

// The roster's join order: ids by the creator's seq (a device without one, last, by id)
export function joinOrder(entries) {
  return [...entries].filter((m) => m?.id).sort((a, b) => (Number.isFinite(+a.seq) ? +a.seq : Infinity) - (Number.isFinite(+b.seq) ? +b.seq : Infinity) || (a.id < b.id ? -1 : 1)).map((m) => m.id);
}

// A device's check of a Start.
// boss: the device that would run the model; ids: the devices the pick was made over (null: a request from an
// older Pooled); view: Map id -> meta from the creator's roster (this device included); order: the roster's join
// order; creator: the room's creator id; from: who sent the request (a relay, for the creator's copy);
// requester: whoever pressed Start (null: unknown); seenFor(id): ms this device has known that device.
// -> { ok: true } | { ok: false, why }
export function verifyHost({ boss, ids = null, view, order = [], creator = null, from = null, seenFor = null }) {
  const no = (why) => ({ ok: false, why });
  if (typeof boss !== "string" || !view.has(boss)) return no("it isn't in the room's roster");
  // pooled host runs its own model: in its room no other device is the pick
  if (creator && view.get(creator)?.fixedHost) return boss === creator ? { ok: true } : no("this room's host (pooled host) runs the model itself");
  if (!Array.isArray(ids)) {
    // an older Pooled: only the room's creator's request (older creators hosted themselves or the strongest device)
    if (from !== creator) return no("a Start from an older Pooled is taken only from the room's creator");
    ids = [...view.keys()];
  }
  const set = new Set(ids.filter((x) => typeof x === "string"));
  if (!set.has(boss)) return no("it isn't among the devices it names");
  const pos = new Map(order.map((id, i) => [id, i]));
  let last = -1;
  for (const id of set) if (pos.has(id)) last = Math.max(last, pos.get(id));
  for (const id of view.keys()) {
    if (set.has(id)) continue;
    if (pos.has(id) && pos.get(id) < last) return no(`it leaves out ${id}, which joined before ${order[last]}`);
    if (seenFor && seenFor(id) >= JOIN_SLACK_MS) return no(`it leaves out ${id}, in the room for ${Math.round(seenFor(id) / 1000)} s`);
  }
  const elect = [...view.keys()].filter((id) => set.has(id));
  const pick = pickModelHost(elect.map((id) => ({ id, meta: saneMeta(view.get(id)) })));
  return pick === boss ? { ok: true } : no(`the room's pick is ${pick}`);
}

// A host message or layers from `from`, on a device that follows `dealer` (null: none), whose host before it
// follows anyone is `named` (the creator, or the host the creator named in ai-modelhost).
// -> "take" | "refuse" (and why)
export function hostAllowed({ from, host = undefined, dealer, dealerHere, named }) {
  if (host != null && host !== from) return { take: false, why: "a model host deals as itself" };
  if (dealer && dealerHere) return from === dealer ? { take: true } : { take: false, why: `this device follows ${dealer}` };
  return from === named ? { take: true } : { take: false, why: "only the room's creator, or the host it named, can deal to a device that wasn't there for the Start" };
}

// Is this device busy with a Start already? It follows a host that is in the room and either deals (it has
// shown it: ai-load, ai-layers, ai-hostprog ...) or was named less than START_PENDING_MS ago. Then a Start naming
// that host again is a replay (nothing to do), and one naming another device is refused. A device that loads or
// holds layers itself refuses a Start naming another device. A Start naming this device itself is a replay only
// while it loads or runs: after a failed start it can start again.
// -> "replay" | null (go ahead) | why it is refused
export function startBusy({ boss, self = null, dealer, dealerHere, dealing, running, pendingSince = null, now = 0 }) {
  const live = !!dealer && dealerHere && (dealing || (pendingSince != null && now - pendingSince < START_PENDING_MS));
  if (boss === self) return running ? "replay" : live && dealer !== self ? "another Start is in progress" : null;
  if (live && boss === dealer) return "replay";
  if (running) return "a model is already loading or running";
  if (live) return "another Start is in progress";
  return null;
}

// Who may hand this device a Start: the room's creator (its own, or one it relayed after checking it; a device
// in the room only seconds has no memory of who was there, so it takes the creator's check), or anyone when this
// device is the creator.
export const startFrom = ({ from, creator, isCreator }) => isCreator || from === creator;
