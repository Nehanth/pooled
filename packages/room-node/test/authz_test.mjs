// Who runs the model on the room node (room/authz.js, #17): every node checks the pick itself. A Start is
// followed only when its boss is the pick over the devices it names (with this node's metas); layers come
// only from the host this node checked; the creator relays a Start it checked; a pooled host creator runs
// its own model (fixedHost). No GPU, no network: messages handed to aiOnData.
import { test } from "node:test";
import assert from "node:assert/strict";
import { RoomNode } from "../roomnode.js";

const C = "pooled-room-4TKG9P";
const META = { [C]: { webgpu: true, contribGB: 8, ua: "Linux" }, "peer-mac": { webgpu: true, contribGB: 24, ua: "Mac" }, "peer-mallory": { webgpu: true, contribGB: 4, ua: "Win" } };
function device({ creatorFixed = false } = {}) {
  const logs = [], sent = [], started = [];
  const n = new RoomNode({ name: "pc", pledgeGB: 6, log: (s) => logs.push(s), stripes: 0, ckpt: false });
  n.code = "4TKG9P"; n.meta = { webgpu: true, contribGB: 6, ua: "Linux" };
  n.peer = { id: "peer-pc" };
  n.sendTo = (id, m) => sent.push({ id, ...m });
  n.start = async (m) => { started.push(m); };
  for (const id of [C, "peer-mac", "peer-mallory"]) n.conns.set(id, { name: id.slice(5), meta: META[id] });
  // the creator's roster: join order C, pc, mac, mallory
  n.onData(C, { t: "roster", members: [{ id: C, name: "host", meta: { ...META[C], ...(creatorFixed ? { fixedHost: 1 } : {}) } }, { id: "peer-pc", name: "pc", meta: n.meta },
    { id: "peer-mac", name: "mac", meta: META["peer-mac"] }, { id: "peer-mallory", name: "mallory", meta: META["peer-mallory"] }] });
  return { n, logs, sent, started };
}
const ALL = [C, "peer-pc", "peer-mac", "peer-mallory"];

test("device: a Start is followed only when its boss is this node's own pick", async () => {
  const { n, logs } = device();
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mallory", by: "mallory", ids: ALL });
  assert.equal(n.ai.hostId, null, "the smallest pledger can't make itself host");
  assert.ok(logs.some((s) => /not following/.test(s)));
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mallory", by: "mallory", ids: ["peer-mallory", "peer-pc"] });
  assert.equal(n.ai.hostId, null, "nor by leaving the bigger devices out");
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mac", by: "mallory", ids: ALL });
  assert.equal(n.ai.hostId, "peer-mac", "anyone may press Start: the Mac lends the most");
  assert.equal(n.ai.dealer, "peer-mac");
});

test("device: layers only from the host it checked, dealt as itself", async () => {
  const { n, sent } = device();
  // an unknown model stops workerLoad right after the checks: an ai-error back means the load was taken
  await n.aiOnData("peer-mallory", { t: "ai-load", model: "nope", range: [0, 1], next: "host", host: "peer-mallory", ids: ALL });
  assert.equal(sent.length, 0, "not the pick");
  await n.aiOnData("peer-mac", { t: "ai-load", model: "nope", range: [0, 1], next: "host", host: "peer-mallory", ids: ALL });
  assert.equal(sent.length, 0, "dealing for someone else");
  await n.aiOnData("peer-mac", { t: "ai-load", model: "nope", range: [0, 1], next: "host", host: "peer-mac", ids: ALL });
  assert.equal(sent.at(-1)?.t, "ai-error", "the pick, checked on its first ai-load (a device that missed the Start)");
  assert.equal(n.ai.dealer, "peer-mac");
  const k = sent.length;
  await n.aiOnData("peer-mallory", { t: "ai-load", model: "nope", range: [0, 1], next: "host", host: "peer-mallory", ids: ["peer-mallory"] });
  assert.equal(sent.length, k, "another host while the checked one is in the room");
  await n.aiOnData("peer-mac", { t: "ai-load", model: "nope", range: [0, 2], next: "host", host: "peer-mac" });
  assert.equal(sent.length, k + 1, "a re-deal from the same host");
});

test("device: host messages only from the host it checked", async () => {
  const { n } = device();
  await n.aiOnData(C, { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mac", by: "host", ids: ALL });
  await n.aiOnData("peer-mallory", { t: "ai-ready-all", model: "qwen3-1.7b", ids: ["peer-mallory"] });
  assert.equal(n.ai.online, false);
  await n.aiOnData("peer-mac", { t: "ai-ready-all", model: "qwen3-1.7b", ids: ALL });
  assert.equal(n.ai.online, true);
});

test("device: a pooled host creator runs its own model (fixedHost), whoever lends more", async () => {
  const { n, sent } = device({ creatorFixed: true });
  await n.aiOnData(C, { t: "ai-load", model: "nope", range: [0, 1], next: "host", host: C, ids: ALL });
  assert.equal(sent.at(-1)?.t, "ai-error", "taken");
  const { n: m, sent: ms } = device();
  await m.aiOnData(C, { t: "ai-load", model: "nope", range: [0, 1], next: "host", host: C, ids: ALL });
  assert.equal(ms.length, 0, "a creator without it is held to the pick like anyone");
});

test("creator (no fixedHost, as a room page): checks a member's Start and relays it", async () => {
  const logs = [], sent = [], started = [];
  const n = new RoomNode({ name: "host", pledgeGB: 8, log: (s) => logs.push(s), stripes: 0, ckpt: false });
  n.isHost = true; n.code = "4TKG9P"; n.meta = META[C]; n.peer = { id: C };
  n.roomView = function () { const v = RoomNode.prototype.roomView.call(this); v.set(C, META[C]); return v; };   // as a page creator: no fixedHost
  n.sendTo = (id, m) => sent.push({ id, ...m }); n.start = async (m) => { started.push(m); };
  for (const id of ["peer-mac", "peer-mallory"]) { n.conns.set(id, { name: id.slice(5), meta: META[id] }); n.roster.set(id, { name: id.slice(5), meta: META[id] }); }
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mallory", ids: [C, "peer-mac", "peer-mallory"] });
  assert.deepEqual(sent.map((m) => m.t), ["ai-start-refused"], "not relayed");
  sent.length = 0;
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mac", ids: [C, "peer-mac", "peer-mallory"] });
  assert.deepEqual(sent.map((m) => m.id), ["peer-mac"], "relayed to the devices the requester has no link to");
  assert.equal(n.helloMsg().meta.fixedHost, 1);
});

test("device: a second Start while one is pending is refused, with a reply; a failed start frees it", async () => {
  const { n, sent } = device();
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mac", by: "mallory", ids: ALL });
  assert.equal(n.ai.dealer, "peer-mac");
  // mallory pads its pledge and presses Start again while the Mac deals (the review's repro)
  n.members.find((m) => m.id === "peer-mallory").meta = { webgpu: true, contribGB: 1e6, ua: "Win" };
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mallory", by: "mallory", ids: ALL });
  assert.equal(n.ai.dealer, "peer-mac", "no re-pick mid-start");
  assert.equal(sent.at(-1).t, "ai-start-refused"); assert.equal(sent.at(-1).id, "peer-mallory");
  assert.match(sent.at(-1).why, /in progress/);
  await n.aiOnData("peer-mac", { t: "ai-start-failed", why: "out of memory" });
  assert.equal(n.ai.dealer, null);
  // with no Start pending a padded pledge counts as 64 GB at most, and still wins over 24: the residual risk (docs)
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mallory", by: "mallory", ids: ALL });
  assert.equal(n.ai.dealer, "peer-mallory");
});

test("device: a refused Start gets a reply saying why", async () => {
  const { n, sent } = device();
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mallory", by: "mallory", ids: ALL });
  assert.equal(sent.at(-1).t, "ai-start-refused"); assert.match(sent.at(-1).why, /pick/);
});

test("device: joining while a member-hosted model loads: it checks the host's ai-ready-all, then its re-deal", async () => {
  const { n, sent } = device();
  // this device joined after the Start (not in ids, listed last), and only knows the creator as host so far
  n.onData(C, { t: "roster", members: [{ id: C, name: "host", meta: META[C] }, { id: "peer-mac", name: "mac", meta: META["peer-mac"] },
    { id: "peer-mallory", name: "mallory", meta: META["peer-mallory"] }, { id: "peer-pc", name: "pc", meta: { webgpu: true, contribGB: 40, ua: "Linux" } }] });
  const ids = [C, "peer-mac", "peer-mallory"];
  await n.aiOnData("peer-mallory", { t: "ai-ready-all", model: "qwen3-1.7b", ids: ["peer-mallory"] });
  assert.equal(n.ai.online, false, "not the pick of the room at Start");
  await n.aiOnData("peer-mac", { t: "ai-ready-all", model: "qwen3-1.7b", ids });
  assert.equal(n.ai.online, true, "the Mac: checked over the devices of the Start (this bigger newcomer isn't one)");
  assert.equal(n.ai.dealer, "peer-mac");
  await n.aiOnData("peer-mac", { t: "ai-load", model: "nope", range: [0, 1], next: "host", host: "peer-mac", ids });
  assert.equal(sent.at(-1)?.t, "ai-error", "its re-deal is taken");
});

test("creator (pooled host): a member's Start naming another device is refused, with a reply", async () => {
  const sent = [], started = [];
  const n = new RoomNode({ name: "host", pledgeGB: 8, log: () => {}, stripes: 0, ckpt: false });
  n.isHost = true; n.code = "4TKG9P"; n.meta = META[C]; n.peer = { id: C };
  n.sendTo = (id, m) => sent.push({ id, ...m }); n.start = async (m) => { started.push(m); };
  for (const id of ["peer-mac", "peer-mallory"]) { n.conns.set(id, { name: id.slice(5), meta: META[id] }); n.roster.set(id, { name: id.slice(5), meta: META[id] }); }
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mac", ids: [C, "peer-mac", "peer-mallory"] });
  assert.deepEqual(started, []); assert.equal(sent.length, 1);
  assert.equal(sent[0].t, "ai-start-refused"); assert.match(sent[0].why, /pooled host/);
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: C, ids: [C, "peer-mac", "peer-mallory"] });
  assert.deepEqual(started, ["qwen3-1.7b"], "naming the room's host starts it");
});
