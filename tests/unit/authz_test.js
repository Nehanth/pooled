// room/authz.js: who runs the model (#17). The decision is every device's own (the same pick over the creator's
// roster); the facts (membership, join order, metas) come from the creator. These cover the rule and the
// review's repros on 2a1cf4b (packages/room-node/test/authz_test.mjs has the room node's side).
import { verifyHost, hostAllowed, saneMeta, startBusy, joinOrder, JOIN_SLACK_MS, GBPS_MAX, START_PENDING_MS } from "../../room/authz.js";

function eq(a, b, m) { const x = JSON.stringify(a), y = JSON.stringify(b); if (x !== y) throw new Error(`${m ? m + ": " : ""}${x} !== ${y}`); }
const C = "pooled-room-4TKG9P";
const M = (gb, ua = "Linux") => ({ webgpu: true, contribGB: gb, ua });
// the creator (8 GB), a laptop (4 GB, joined 2nd), a Mac (24 GB, joined 3rd)
const view = () => new Map([["lap", M(4, "Win")], [C, M(8)], ["mac", M(24, "Mac")]]);
const order = [C, "lap", "mac"];
const V = (o) => verifyHost({ view: view(), order, creator: C, from: "lap", ...o });

Deno.test("authz: the room's pick runs the model, whoever pressed Start; the creator is not special", () => {
  eq(V({ boss: "mac", ids: [C, "lap", "mac"] }).ok, true);
  eq(V({ boss: C, ids: [C, "lap", "mac"], from: C }).ok, false, "the creator naming itself");
  eq(V({ boss: "lap", ids: [C, "lap", "mac"] }).ok, false);
  eq(V({ boss: "ghost", ids: [C, "ghost"] }).ok, false, "a device not in the roster");
  eq(V({ boss: "mac", ids: [C, "lap"] }).ok, false, "a boss outside its own ids");
});

Deno.test("authz: ids can't leave out a device that joined before one they name, or one seen for a while", () => {
  eq(V({ boss: "lap", ids: ["lap", "mac"] }).ok, false, "the creator left out");
  eq(V({ boss: C, ids: [C, "mac"] }).ok, false, "the laptop left out (it joined before the Mac)");
  eq(V({ boss: C, ids: [C, "lap"] }).ok, true, "only the newest left out: a Start from before it joined");
  eq(V({ boss: C, ids: [C, "lap"], seenFor: () => JOIN_SLACK_MS + 1 }).ok, false, "not when this device has seen it for a while");
  eq(V({ boss: C, ids: [C, "lap"], seenFor: (id) => (id === "mac" ? 2000 : 60000) }).ok, true, "one that joined seconds ago");
});

Deno.test("authz: join order is the creator's seq, stable across a relink and a resume (review repro2)", () => {
  // W relinks (the creator's roster deletes and re-adds it) and L1 reconnects first after a resume: seq keeps W 3rd
  const entries = [{ id: "C", seq: 0 }, { id: "B", seq: 1 }, { id: "L1", seq: 4 }, { id: "W", seq: 2 }, { id: "L2", seq: 5 }];
  eq(joinOrder(entries), ["C", "B", "W", "L1", "L2"]);
  const v = new Map([["C", M(8)], ["B", M(24)], ["W", M(6)], ["L1", M(4)], ["L2", M(4)]]);
  eq(verifyHost({ boss: "B", ids: ["C", "B", "W"], view: v, creator: "C", from: "C", order: joinOrder(entries) }).ok, true, "B, after W's relink");
  eq(joinOrder([{ id: "b" }, { id: "a", seq: 1 }]), ["a", "b"], "no seq: last");
});

Deno.test("authz: an older Pooled's Start (no ids) only from the creator (review B)", () => {
  eq(V({ boss: C, ids: null, from: "lap" }).ok, false, "any member naming the creator");
  eq(V({ boss: "mac", ids: null, from: C }).ok, true, "the creator: the pick over the whole roster");
  eq(V({ boss: C, ids: null, from: C }).ok, false, "even the creator only for the pick");
});

Deno.test("authz: pooled host (fixedHost) only as the room's creator", () => {
  const v = view(); v.set(C, { ...M(8), fixedHost: 1 });
  eq(verifyHost({ boss: C, ids: [C, "lap", "mac"], view: v, order, creator: C, from: "lap" }).ok, true);
  eq(verifyHost({ boss: "mac", ids: [C, "lap", "mac"], view: v, order, creator: C, from: "lap" }).ok, false, "nobody else is the pick there");
  const w = view(); w.set("lap", { ...M(4), fixedHost: 1 });
  eq(verifyHost({ boss: "lap", ids: [C, "lap", "mac"], view: w, order, creator: C, from: "lap" }).ok, false, "a member can't claim it");
});

Deno.test("authz: saneMeta holds pledges to the room's caps; pledge, kind and speed stay claims", () => {
  eq(saneMeta(M(500)).contribGB, 64);
  eq(saneMeta({ ...M(40, "Mac"), pledgeMax: 6 }).contribGB, 6);
  eq(saneMeta(M(8, "iPhone")).contribGB <= 1, true);
  eq(saneMeta({ webgpu: false, contribGB: 64 }).contribGB, 0);
  eq(saneMeta({ ...M(8), gbps: 1e9 }).gbps, GBPS_MAX);
  eq(saneMeta({ ...M(12), budgetGB: 2, canMeasure: true }).contribGB, 12, "a tab's budget is a guess, not a cap");
  // review repro2: the claims this can't check. A claimed 2000 GB/s beats twice the memory; a spoofed ua skips a phone's cap
  const v = new Map([["C", M(8)], ["B", { ...M(24, "Mac"), gbps: 400 }], ["A", { ...M(12, "Mac"), gbps: 1e9, gpu: "x" }]]);
  eq(verifyHost({ boss: "A", ids: ["C", "B", "A"], view: v, creator: "C", from: "C", order: ["C", "B", "A"] }).ok, true, "(a claim: documented, not prevented)");
  eq(verifyHost({ boss: "A", ids: ["C", "A"], view: new Map([["C", M(8)], ["A", M(64, "Mac")]]), creator: "C", from: "C", order: ["C", "A"] }).ok, true, "(a claim)");
});

Deno.test("authz: a hostless device takes a host only from the creator or the host it named (review A, C)", () => {
  // A: mallory (10 GB) sends a late joiner its own ai-ready-all; the joiner's named host is the creator
  eq(hostAllowed({ from: "mallory", dealer: null, dealerHere: false, named: C }).take, false);
  eq(hostAllowed({ from: C, dealer: null, dealerHere: false, named: C }).take, true);
  eq(hostAllowed({ from: "mac", dealer: null, dealerHere: false, named: "mac" }).take, true, "named in ai-modelhost");
  // C: the creator itself, idle: its named host is itself, so nobody else
  eq(hostAllowed({ from: "mallory", dealer: null, dealerHere: false, named: C }).take, false);
  eq(hostAllowed({ from: "mac", dealer: "mac", dealerHere: true, named: C }).take, true, "its host");
  eq(hostAllowed({ from: "lap", dealer: "mac", dealerHere: true, named: "lap" }).take, false, "another while its host is here");
  eq(hostAllowed({ from: "mac", host: "lap", dealer: "mac", dealerHere: true, named: C }).take, false, "dealing for someone else");
  eq(hostAllowed({ from: "lap", dealer: "mac", dealerHere: false, named: C }).take, false, "its host left: still only the creator's word");
});

Deno.test("authz: one Start at a time; a replay is a no-op (review D, E)", () => {
  const base = { dealer: "mac", dealerHere: true, now: 1000, running: false, dealing: false };
  eq(startBusy({ ...base, boss: "mac", running: true }), "replay", "D: a Start naming the current host");
  eq(startBusy({ ...base, boss: "lap", running: true }) !== null, true, "a model loads or runs here");
  eq(startBusy({ ...base, boss: "lap", pendingSince: 500 }) !== null, true, "its Start waits for the host");
  eq(startBusy({ ...base, boss: "lap", dealing: true, pendingSince: 1000 - 10 * START_PENDING_MS }) !== null, true, "E: its host deals, however long it takes");
  eq(startBusy({ ...base, boss: "lap", pendingSince: 1000 - START_PENDING_MS - 1 }), null, "a Start whose host never began");
  eq(startBusy({ ...base, dealerHere: false, boss: "lap", dealing: true, pendingSince: 500 }), null, "its host left");
  eq(startBusy({ boss: "lap", dealer: null, dealerHere: false, running: false }), null);
});
