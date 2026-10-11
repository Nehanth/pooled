// Who runs the model on the room node (room/authz.js, #17): only the room's creator names the model host.
// A device follows Start only from the creator, takes layers only from the host the creator named and
// dealt as that host; a creator picks between a member that pressed Start and itself. A creator from
// before (no ha) keeps the old rules. No GPU, no network: messages handed to aiOnData.
import { test } from "node:test";
import assert from "node:assert/strict";
import { RoomNode } from "../roomnode.js";
import { HOST_AUTH_V } from "../../../room/authz.js";

const CREATOR = "pooled-room-4TKG9P";
function node({ isHost = false, ha = HOST_AUTH_V, meta = { webgpu: true, contribGB: 8, ua: "Linux" } } = {}) {
  const logs = [], sent = [], started = [];
  const n = new RoomNode({ name: isHost ? "host" : "pc", pledgeGB: 8, log: (s) => logs.push(s), stripes: 0, ckpt: false });
  n.code = "4TKG9P"; n.isHost = isHost; n.meta = meta;
  n.peer = { id: isHost ? CREATOR : "peer-pc" };
  if (!isHost && ha) n.creatorHa = ha;
  n.sendTo = (id, m) => sent.push({ id, ...m });
  n.start = async (m) => { started.push(m); };
  for (const id of [CREATOR, "peer-laptop", "peer-mallory"]) if (id !== n.peer.id) n.conns.set(id, { name: id.slice(5), meta: { webgpu: true, contribGB: 4, ua: "Mac" } });
  return { n, logs, sent, started };
}

test("device: Start is followed only from the room's creator", async () => {
  const { n, started } = node();
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mallory", by: "mallory" });
  assert.equal(n.ai.hostId, null, "a member can't name the model host");
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-pc", by: "mallory" });
  assert.deepEqual(started, [], "nor start the model here");
  await n.aiOnData(CREATOR, { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-laptop", by: "laptop" });
  assert.equal(n.ai.hostId, "peer-laptop", "the creator named it");
  await n.aiOnData(CREATOR, { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-pc", by: "laptop" });
  assert.deepEqual(started, ["qwen3-1.7b"], "the creator named this device");
});

test("device: layers only from the named host, dealt as itself", async () => {
  const { n, sent, logs } = node();
  await n.aiOnData(CREATOR, { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-laptop", by: "laptop" });
  // an unknown model stops workerLoad right after the checks: an ai-error back means the load was taken
  await n.aiOnData("peer-mallory", { t: "ai-load", model: "nope", range: [0, 1], next: "host", host: "peer-mallory" });
  assert.equal(sent.length, 0, "a member that was not named is ignored (FROM_HOST)");
  await n.aiOnData("peer-laptop", { t: "ai-load", model: "nope", range: [0, 1], next: "host", host: "peer-mallory" });
  assert.equal(sent.length, 0, "the named host dealing for someone else is ignored");
  assert.ok(logs.some((s) => /ignored layers/.test(s)));
  await n.aiOnData("peer-laptop", { t: "ai-load", model: "nope", range: [0, 1], next: "host", host: "peer-laptop" });
  assert.equal(sent.at(-1)?.t, "ai-error", "the named host dealing as itself is taken");
});

test("device: a creator from before (no ha) keeps the old rules", async () => {
  const { n } = node({ ha: 0 });
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-laptop", by: "mallory" });
  assert.equal(n.ai.hostId, "peer-laptop");
});

test("creator: a member's Start: it picks between the member and itself, and tells the room", async () => {
  const { n, sent, started } = node({ isHost: true });
  n.roster.set("peer-mallory", { name: "mallory", meta: { webgpu: true, contribGB: 4 } });
  n.roster.set("peer-laptop", { name: "laptop", meta: { webgpu: true, contribGB: 4 } });
  // the member names a third device: the creator ignores the name and picks (itself: it lends more)
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-laptop", by: "someone else" });
  assert.deepEqual(started, ["qwen3-1.7b"]);
  assert.equal(sent.length, 0);
  // a stronger member runs it, and every device hears it from the creator, with the member's room name
  const { n: h, sent: hs, started: hst } = node({ isHost: true, meta: { webgpu: true, contribGB: 2, ua: "Linux" } });
  h.roster.set("peer-mallory", { name: "mallory", meta: { webgpu: true, contribGB: 4 } });
  await h.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-laptop", by: "the boss" });
  assert.deepEqual(hst, []);
  assert.ok(hs.length >= 2 && hs.every((m) => m.t === "ai-start-req" && m.boss === "peer-mallory" && m.by === "mallory"));
});

test("creator: Start is ignored while a model loads or runs, from an API client, and for an unknown model", async () => {
  const { n, sent, started } = node({ isHost: true });
  n.roster.set("peer-mallory", { name: "mallory", meta: { webgpu: true, contribGB: 4 } });
  n.ai.online = true;
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mallory" });
  n.ai.online = false;
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "no-such-model", boss: "peer-mallory" });
  n.conns.get("peer-mallory").meta = { api: 1 };
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mallory" });
  assert.deepEqual(started, []); assert.equal(sent.length, 0);
});

test("hello and roster from a creator say ha; a device records it from either", () => {
  const { n } = node({ isHost: true });
  assert.equal(n.helloMsg().ha, HOST_AUTH_V);
  const { n: d } = node({ ha: 0 });
  assert.equal(d.helloMsg().ha, undefined, "a device that is not the creator doesn't");
  assert.equal(d.strict(), false);
  d.members = []; d.conns.get(CREATOR).meta = {};
  d.onData(CREATOR, { t: "roster", members: [], ha: HOST_AUTH_V });
  assert.equal(d.strict(), true);
});
