// Stopping a start that cannot finish, and re-seating a worker. DOM-free so it can be unit tested
// (tests/unit/startstop_test.js).
//
// A start fails when a device in the chain could not load its layers (it sends ai-error with load).
// The host then stops the whole start: every screen goes back to the model picker (ai-start-failed),
// and a load still running anywhere stops at its next tensor instead of finishing for nothing.

// the host, on ai-error from `from`: does it stop the start? Only a load error, from a device in the
// chain, while the start is still running (not after the room went online: then it is a lost device,
// and the Re-deal path handles it).
export function stopsStart({ load, inChain, starting, online }) {
  return !!(load && inChain && starting && !online);
}

// the host, stopping a start: while its own shard still loads it only marks the start as failed
// ("defer": the load stops at its next tensor and its catch finishes the stop); otherwise at once.
export function stopWhen(loadingShard) {
  return loadingShard ? "defer" : "now";
}

// the reason a stopped start gives: the device that failed first, not what this load hit after
export function stopReason(startFailed, err) {
  return startFailed || err?.message || String(err ?? "the start was stopped");
}

export const loadKey = (model, range) => `${model || "smollm-135m"}:${range}`;

// a worker, on ai-load: "keep" when the host re-seats it (it dropped this tab and it reconnected)
// while its load of the same layers still runs: that load reports ai-ready when it is in. Otherwise
// "load" afresh.
export function onLoadRequest({ loadingShard, role, currentKey, key }) {
  return loadingShard && role === "worker" && currentKey === key ? "keep" : "load";
}

// a worker whose load threw: "stopped" when the host stopped the start (ai-start-failed arrived,
// the load stopped with it: free the layers quietly); "error" when this device failed on its own
// (tell the host, with load, so it stops the start).
export function onLoadError(startFailed) {
  return startFailed ? "stopped" : "error";
}

// a worker, on ai-start-failed: free its layers now, or let the load in flight stop at its next
// tensor (its catch frees them)
export function freeOnStartFailed(loadingShard) {
  return !loadingShard;
}
