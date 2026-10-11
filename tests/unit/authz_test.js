// room/authz.js: every device checks who runs the model itself (#17). The pick (room/plan.js pickModelHost)
// over the devices a Start names, with this device's own metas; an electorate can't leave out a device that
// joined before one it names, nor (on a Start this device saw) one seen for a while; pledges count only as
// far as they are self-consistent; layers come only from the host this device checked.
import { verifyHost, loadGate, saneMeta, startBusy, JOIN_SLACK_MS, GBPS_MAX, START_PENDING_MS } from "../../room/authz.js";

function eq(a, b, m) { const x = JSON.stringify(a), y = JSON.stringify(b); if (x !== y) throw new Error(`${m ? m + ": " : ""}${x} !== ${y}`); }
const C = "pooled-room-4TKG9P";
const pc = { webgpu: true, contribGB: 8, ua: "Linux" }, mac = { webgpu: true, contribGB: 24, ua: "Mac" }, lap = { webgpu: true, contribGB: 4, ua: "Win" };
// the creator (8 GB), a laptop (4 GB, joined 2nd), a Mac (24 GB, joined 3rd); "me" is the laptop
const view = () => new Map([["lap", lap], [C, pc], ["mac", mac]]);
const order = [C, "lap", "mac"];
const V = (o) => verifyHost({ view: view(), self: "lap", order, creator: C, ...o });

Deno.test("authz: the top pledger runs the model; anyone else naming a host is refused", () => {
  eq(V({ boss: "mac", ids: [C, "lap", "mac"] }).ok, true);
  eq(V({ boss: C, ids: [C, "lap", "mac"] }).ok, false, "the creator is not special");
  eq(V({ boss: "lap", ids: [C, "lap", "mac"] }).ok, false);
  eq(V({ boss: "ghost", ids: [C, "ghost"] }).ok, false, "a device this one doesn't know");
  eq(V({ boss: "mac", ids: [C, "lap"] }).ok, false, "a boss outside its own electorate");
});

Deno.test("authz: an electorate can't leave out a device that joined before one it names", () => {
  const r = V({ boss: "lap", ids: ["lap", "mac"] });
  eq(r.ok, false, "leaving out the creator (joined first)"); eq(/leaves out/.test(r.why), true);
  eq(V({ boss: C, ids: [C, "mac"] }).ok, false, "leaving out this device (it joined before the Mac)");
  eq(V({ boss: C, ids: [C, "lap"] }).ok, true, "only the newest device left out: a Start before it joined");
  eq(V({ boss: C, ids: [C, "lap"], seenFor: () => JOIN_SLACK_MS + 1 }).ok, false, "but not on a Start this device saw it for a while");
  eq(V({ boss: C, ids: [C, "lap"], seenFor: (id) => (id === "mac" ? 2000 : 60000) }).ok, true, "a device that joined seconds ago");
});

Deno.test("authz: a late joiner checks the pick over the electorate it wasn't in", () => {
  const v = new Map([["new", { webgpu: true, contribGB: 60, ua: "Linux" }], [C, pc], ["lap", lap]]);
  eq(verifyHost({ boss: C, ids: [C, "lap"], view: v, self: "new", order: [C, "lap", "new"], creator: C }).ok, true, "bigger but not in the room at Start");
  eq(verifyHost({ boss: "lap", ids: ["lap"], view: v, self: "new", order: [C, "lap", "new"], creator: C }).ok, false, "the creator left out");
});

Deno.test("authz: pooled host (fixedHost) only as the room's creator; older peers", () => {
  const v = view(); v.set(C, { ...pc, fixedHost: 1 });
  eq(verifyHost({ boss: C, ids: [C, "lap", "mac"], view: v, self: "lap", order, creator: C }).ok, true);
  const w = view(); w.set("lap", { ...lap, fixedHost: 1 });
  eq(verifyHost({ boss: "lap", ids: [C, "lap", "mac"], view: w, self: "mac", order, creator: C }).ok, false, "only the creator can claim it");
  eq(verifyHost({ boss: "mac", ids: [C, "lap", "mac"], view: v, self: "lap", order, creator: C }).ok, false, "in a pooled host's room nobody else is the pick");
  eq(V({ boss: "mac", ids: null }).ok, true, "an older requester: the pick over this device's whole view");
  eq(V({ boss: "lap", ids: null }).ok, false);
  eq(V({ boss: C, ids: null }).ok, true, "an older creator (it hosted as before)");
});

Deno.test("authz: pledges count only as far as they are self-consistent", () => {
  eq(saneMeta({ webgpu: true, contribGB: 500, ua: "Linux" }).contribGB, 64, "more than one computer lends");
  eq(saneMeta({ webgpu: true, contribGB: 40, pledgeMax: 6, ua: "Mac" }).contribGB, 6, "more than its own cap");
  eq(saneMeta({ webgpu: true, contribGB: 8, ua: "iPhone" }).contribGB <= 1, true, "a phone's kind cap");
  eq(saneMeta({ webgpu: false, contribGB: 64 }).contribGB, 0, "no GPU, no pledge");
  eq(saneMeta({ webgpu: true, contribGB: 8, gbps: 1e9 }).gbps, GBPS_MAX);
  eq(saneMeta({ webgpu: true, contribGB: 12, budgetGB: 2, canMeasure: true, ua: "Win" }).contribGB, 12, "a tab's budget is a guess, not a cap");
  eq(saneMeta({ webgpu: true, contribGB: 40, budgetGB: 4, ua: "Linux" }).contribGB, 40, "a room node's budget is its largest buffer, not its memory");
  eq(saneMeta({ webgpu: true, contribGB: 1e6, ua: "Linux" }).contribGB, 64, "a pledge message of 1e6");
  const v = view(); v.set("lap", { webgpu: true, contribGB: 999, pledgeMax: 4, ua: "Win" });
  eq(verifyHost({ boss: "lap", ids: [C, "lap", "mac"], view: v, self: "mac", order, creator: C }).ok, false, "a padded pledge doesn't win");
});

Deno.test("authz: layers from the checked host, dealt as itself", () => {
  eq(loadGate({ from: "mac", host: "mac", dealer: "mac", dealerHere: true }), "take");
  eq(loadGate({ from: "lap", host: "lap", dealer: "mac", dealerHere: true }), "refuse", "another host while it's in the room");
  eq(loadGate({ from: "lap", host: "lap", dealer: "mac", dealerHere: false }), "verify", "its host left: check the new one");
  eq(loadGate({ from: "lap", host: "lap", dealer: null, dealerHere: false }), "verify");
  eq(loadGate({ from: "mac", host: "lap", dealer: "mac", dealerHere: true }), "refuse", "dealing for someone else");
  eq(loadGate({ from: "mac", host: undefined, dealer: "mac", dealerHere: true }), "take", "an older host's ai-load");
});

Deno.test("authz: a second Start while one is pending, loading or running is refused", () => {
  const base = { dealer: "mac", dealerHere: true, now: 1000 };
  eq(startBusy({ ...base, boss: "mac", running: true }), null, "the same host again: fine");
  eq(startBusy({ ...base, boss: "lap", running: true }) !== null, true, "a model loads or runs");
  eq(startBusy({ ...base, boss: "lap", running: false, pendingSince: 500 }) !== null, true, "a Start is pending");
  eq(startBusy({ ...base, boss: "lap", running: false, pendingSince: 1000 - START_PENDING_MS - 1 }), null, "a pending Start that went nowhere");
  eq(startBusy({ ...base, boss: "lap", running: false, pendingSince: null }), null, "the last Start failed");
  eq(startBusy({ ...base, dealerHere: false, boss: "lap", running: false, pendingSince: 500 }), null, "its host left");
  eq(startBusy({ boss: "lap", dealer: null, dealerHere: false, running: false }), null);
});
