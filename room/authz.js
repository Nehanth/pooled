// Who runs the room's model: every device checks it for itself (#17). Verify, don't trust.
//
// The model host deals the layers (ai-load) and sends what FROM_HOST lists (ai-layers, ai-token,
// ai-ready-all ...). It is the strongest device of the room (room/plan.js pickModelHost: a computer
// before a phone, then the most memory lent, then GPU speed, then the lowest id), whoever pressed Start.
// A device used to take a new model host from anyone (an ai-start-req naming a boss, an ai-load naming
// its sender), so any admitted member could make itself every device's host and write into its chat.
// Now each device runs the same pick over its own view of the room and follows a host only when the
// pick lands on it:
//   - ai-start-req {boss, ids} and ai-load / ai-ready-all {ids} carry the devices the pick was made
//     over (the electorate: whoever was in the room at Start). A device checks the pick over the
//     electorate as it knows those devices, with its own copy of their metas (verifyHost);
//   - an electorate can't leave out a device the requester must have known: the creator's roster lists
//     devices in the order they joined, so any device listed before the newest one in `ids` joined
//     before it; leaving it out is refused. On a Start a device was present for, it also refuses one
//     that leaves out a device it has seen for JOIN_SLACK_MS or more (freshStart);
//   - a device then takes ai-load and FROM_HOST messages only from the host it verified, until that
//     host leaves its view;
//   - pledges are taken only as far as they are self-consistent (saneMeta): held to the device kind's
//     cap, its own pledgeMax (room/pledge.js) and 64 GB; GPU speed
//     to GBPS_MAX. They are still claims: a member can claim 64 GB and win the pick until it fails to load. A device that pledged more than
//     it can load is demoted by the room anyway (room/pledge.js afterLoadDeath halves its share, then
//     drops it).
// Two exceptions: a `pooled host` creator (the room node, the OpenClaw gateway) runs the model itself by
// design (it serves the API), and says so with meta.fixedHost; only the room's creator can claim that.
// A host from before this check (no ids) is taken when the pick over this device's whole view lands on
// it, or when it is the room's creator (older creators always hosted themselves or the strongest device).

import { pickModelHost } from "./plan.js";
import { pledgeGB } from "./pledge.js";


export const JOIN_SLACK_MS = 10000;     // a device seen for less than this may be missing from a Start's ids
export const GBPS_MAX = 2000;           // no GPU copies faster than this (an M3 Ultra: ~800 GB/s)
export const START_PENDING_MS = 60000;  // a Start this device followed counts as in progress this long (until it fails or loads)

// what the pick may take from a meta: its pledge held to what the device itself says it can lend
export function saneMeta(meta) {
  if (!meta || typeof meta !== "object") return {};
  const m = { ...meta };
  m.contribGB = m.webgpu ? Math.min(64, pledgeGB(m)) : 0;
  // (not held to budgetGB: browsers hide GPU memory, so a tab's budget is a conservative guess its lend
  // control doesn't enforce, and a room node's is its largest buffer)
  m.gbps = Math.min(GBPS_MAX, Math.max(0, +m.gbps || 0));
  return m;
}

// The verifier's check of a model host.
// boss: the device that would run the model; ids: the electorate it names (null: a peer from before);
// view: Map id -> meta, this device's own view of the room (itself included); self: this device's id;
// order: the creator's roster ids in join order; creator: the room's creator id;
// seenFor(id): ms this device has known that device (null: don't check: not a fresh Start here).
// -> { ok: true } | { ok: false, why }
export function verifyHost({ boss, ids = null, view, self, order = [], creator = null, seenFor = null }) {
  const no = (why) => ({ ok: false, why });
  if (typeof boss !== "string" || !view.has(boss)) return no("it isn't in this device's room");
  // pooled host runs its own model: in its room no other device is the pick
  if (creator && view.get(creator)?.fixedHost) return boss === creator ? { ok: true } : no("this room's host (pooled host) runs the model itself");
  let elect;
  if (Array.isArray(ids)) {
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
    elect = [...view.keys()].filter((id) => set.has(id));
  } else {
    if (boss === creator) return { ok: true };   // an older creator
    elect = [...view.keys()];
  }
  const pick = pickModelHost(elect.map((id) => ({ id, meta: saneMeta(view.get(id)) })));
  return pick === boss ? { ok: true } : no(`the pick over this device's view is ${pick}`);
}

// What a device does with layers or a host message from `from` (ai-load, ai-ready-all), given the host
// it follows (dealer, null: none yet) and whether that host is still in its view.
// -> "take" | "verify" (check `from` with verifyHost, then follow it) | "refuse"
export function loadGate({ from, host = undefined, dealer, dealerHere }) {
  if (host != null && host !== from) return "refuse";        // a model host deals as itself
  if (dealer && from === dealer) return "take";
  if (dealer && dealerHere) return "refuse";                  // it follows another host, still in the room
  return "verify";
}

// Is this device busy with a Start already? A second Start (another device pressed it too, or anyone pressed
// it again) must not move the room to another host while one is pending, loading or running: it is
// refused. The same boss again is fine (two devices whose views agree pick the same one).
// running: a model loads or runs here (or this device holds or deals layers); pendingSince: when this
// device followed a Start (null: none, or it failed). -> null (go ahead) or why it is refused
export function startBusy({ boss, dealer, dealerHere, running, pendingSince = null, now = 0 }) {
  if (dealer && boss === dealer) return null;
  if (running) return "a model is already loading or running";
  if (dealer && dealerHere && pendingSince != null && now - pendingSince < START_PENDING_MS) return "another Start is in progress";
  return null;
}
