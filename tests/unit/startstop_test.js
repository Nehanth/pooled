// room/startstop.js: stopping a start that cannot finish (aiStartStopped, aiLoadFailed) and
// re-seating a worker whose load still runs. No GPU.
import { stopsStart, stopWhen, stopReason, loadKey, onLoadRequest, onLoadError, freeOnStartFailed } from "../../room/startstop.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };

Deno.test("stopsStart: only a load error from a chain device while the start runs", () => {
  eq(stopsStart({ load: 1, inChain: true, starting: true, online: false }), true);
  eq(stopsStart({ load: 0, inChain: true, starting: true, online: false }), false, "an error that is not a load error");
  eq(stopsStart({ load: 1, inChain: false, starting: true, online: false }), false, "a device outside the chain");
  eq(stopsStart({ load: 1, inChain: true, starting: false, online: false }), false, "no start running");
  eq(stopsStart({ load: 1, inChain: true, starting: true, online: true }), false, "the room is already online");
});

Deno.test("stopWhen: the host defers while its own shard loads", () => {
  eq(stopWhen(true), "defer");
  eq(stopWhen(false), "now");
});

Deno.test("stopReason: the device that failed first is the reason", () => {
  eq(stopReason("phone couldn't load its layers (oom)", new Error("the start was stopped")), "phone couldn't load its layers (oom)");
  eq(stopReason(null, new Error("out of memory")), "out of memory");
  eq(stopReason(null, "plain"), "plain");
  eq(stopReason(null, undefined), "the start was stopped");
});

Deno.test("onLoadRequest: a re-seat of the same layers keeps the load in flight", () => {
  const key = loadKey("qwen", [4, 8]);
  eq(key, "qwen:4,8");
  eq(loadKey(undefined, [0, 2]), "smollm-135m:0,2");
  eq(onLoadRequest({ loadingShard: true, role: "worker", currentKey: key, key }), "keep");
  eq(onLoadRequest({ loadingShard: true, role: "worker", currentKey: loadKey("qwen", [4, 9]), key }), "load", "other layers");
  eq(onLoadRequest({ loadingShard: false, role: "worker", currentKey: key, key }), "load", "the load already finished");
  eq(onLoadRequest({ loadingShard: true, role: "host", currentKey: key, key }), "load", "not a worker");
});

Deno.test("onLoadError: a stopped start frees quietly, a failure tells the host", () => {
  eq(onLoadError("laptop couldn't load its layers"), "stopped");
  eq(onLoadError(null), "error");
});

Deno.test("freeOnStartFailed: a load in flight stops at its next tensor instead", () => {
  eq(freeOnStartFailed(true), false);
  eq(freeOnStartFailed(false), true);
});
