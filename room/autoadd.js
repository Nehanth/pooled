// A device that joins a room whose model is already running is added to it on its own: the host shares the model out
// again (a re-deal, room.js aiRedeal) once the room is idle, with the chat kept on screen (the load card's slim line).
// Who is added and when are decided here; room.js autoAddTick does it. Pure (no DOM, no clock: `now` is passed in):
// tests/unit/autoadd_test.js.
//
// - Only devices that can hold part of the model: WebGPU, not an app connected through the API, not a phone (a phone
//   only asks while computers hold the model; it can still be added with the button), a pledge of one layer or more.
// - A few seconds after the newest join (GRACE_MS), so devices that join together are added in one re-deal.
// - Never during an answer, a Code run, a load or with questions waiting: it looks again a moment later (POLL_MS).
// - At most one re-deal every GAP_MS, and the same device at most MAX_PER_NAME times (a device that drops and comes
//   back again and again does not reload the room again and again); "Not now" leaves a device to the button.

export const GRACE_MS = 4000, GAP_MS = 90000, POLL_MS = 2000, MAX_PER_NAME = 2;

// Would a device that joins get layers at all? With the "by memory" split every device holds some. "For speed"
// (room/plan.js planForSpeed) fills the measured devices first and leaves an unmeasured newcomer out while the room's
// pledges hold the model: it helps only when the room is short (a device parks experts in RAM or reads them from disk,
// offload: the Flash demo's Mac alone) or every device here already holds layers (unneeded: none left out) on more than
// one device. Otherwise adding it would reload the whole room for nothing; the button stays.
export function wouldHelp({ split = "speed", offload = false, unneeded = false, chain = 0 } = {}) {
  return split !== "speed" || !!offload || (!unneeded && chain > 0);
}

// devs [{ name, webgpu, api, phone, gb }] -> the names that may be added on their own. layerGB: one layer's GB in this
// deal (0: not known, no check); skip: names the host said "Not now" to; added: name -> times it was added on its own
export function addable(devs = [], { layerGB = 0, skip = new Set(), added = new Map() } = {}) {
  return devs.filter((d) => d && d.name && d.webgpu && !d.api && !d.phone && !(+layerGB > 0 && !(+d.gb >= +layerGB))
    && !skip.has(d.name) && (added.get(d.name) || 0) < MAX_PER_NAME).map((d) => d.name);
}

// The room now -> { go: true } to re-deal now, { go: false, wait: ms } to look again then, or { go: false, wait: null }
// for nothing to do. on: the Room setting; host: this device runs the model; online: the room is up (not loading);
// degraded: a device left (the drop handling re-deals, not this); busy: an answer or a Code run; queued: questions
// waiting; names: addable(); joinedAt: when the newest of them joined; lastAt: the last re-deal's start (0: none)
export function addWhen({ on = true, host = false, online = false, degraded = false, busy = false, loading = false, queued = 0, names = [], joinedAt = 0, lastAt = 0, now = 0 } = {}) {
  if (!on || !host || degraded || !names.length) return { go: false, wait: null };
  let wait = Math.max(0, joinedAt + GRACE_MS - now, lastAt > 0 ? lastAt + GAP_MS - now : 0);
  if (!online || busy || loading || queued > 0) wait = Math.max(wait, POLL_MS);
  return wait > 0 ? { go: false, wait } : { go: true, wait: 0 };
}
