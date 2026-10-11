// Who runs the room's model: the room's creator decides (#17).
//
// The model host (room/plan.js pickModelHost: the strongest device, whoever pressed Start) deals the
// layers (ai-load) and sends what FROM_HOST lists (ai-layers, ai-token, ai-ready-all ...); a device
// takes those only from its model host. A device used to take a new model host from anyone in the
// room: an ai-start-req naming a boss, or an ai-load naming its sender as host. So any admitted member
// could make itself every device's host and write into their chat. Now:
//   - only the room's creator names the model host: its own Start, an ai-start-req it sends, or
//     ai-modelhost. A member's Start goes to the creator, which picks between the member and itself
//     (the same choice the member made before) and tells the room;
//   - a device takes ai-load only from the creator or from the model host the creator named, and only
//     with its sender as the host;
//   - a creator ignores a Start while a model is loading or running.
// A creator says it does this with hello.ha (HOST_AUTH_V). In a room whose creator runs an older Pooled
// (no ha) devices behave as before, since that creator never names the model host.

export const HOST_AUTH_V = 1;

// the room's creator said hello with ha: devices follow only the model host it names
export const strictRoom = (creatorHa) => +creatorHa >= HOST_AUTH_V;

// a device that is not the creator: follow this ai-start-req (its boss runs the model)?
export function followStart({ from, creator, strict }) {
  return !strict || from === creator;
}

// may `from` deal this device layers (ai-load with d.host = host)? dealer: the model host the creator
// named (null: none yet)
export function loadAllowed({ from, host, creator, dealer, strict }) {
  if (!strict) return true;
  if (host != null && host !== from) return false;   // a model host deals as itself
  return from === creator || (!!dealer && from === dealer);
}

// the creator, on a member's Start (ai-start-req from `from`): the id of the device that runs the model,
// or null to ignore it. busy: a model is loading or running here. pick: room/plan.js pickModelHost over
// the two devices the member could see (itself and the creator), with the metas the creator holds, so a
// member can't name a third device or claim a different pledge in the request.
export function creatorPicks({ from, creator, busy, isMember, pick }) {
  if (busy || from === creator || !isMember) return null;
  return pick() || creator;
}
