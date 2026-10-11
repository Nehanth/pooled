// Who runs the model on the room node (room/authz.js, #17): every node checks the pick itself with the
// creator's facts (its roster: membership, join order, metas); a node that wasn't there for the Start takes a
// host only from the creator or the host it named; refused layers fail the start out loud; one Start at a time.
// Includes the review's repros on 2a1cf4b (A-E). No GPU, no network: messages handed to aiOnData.
import { test } from "node:test";
import assert from "node:assert/strict";
import { RoomNode } from "../roomnode.js";

const C = "pooled-room-4TKG9P";
// join order: the creator (8 GB), mallory (10 GB), the Mac (24 GB, the room's pick), this node (pc, 6 GB)
const META = { [C]: { webgpu: true, contribGB: 8, ua: "Linux" }, "peer-mallory": { webgpu: true, contribGB: 10, ua: "Win" }, "peer-mac": { webgpu: true, contribGB: 24, ua: "Mac" } };
const ALL = [C, "peer-mallory", "peer-mac", "peer-pc"];
function roster(n, metas = {}, extra = {}) {
  n.onData(C, { t: "roster", members: [{ id: C, name: "host", meta: { ...META[C], ...extra }, seq: 0 }, { id: "peer-mallory", name: "mallory", meta: metas["peer-mallory"] || META["peer-mallory"], seq: 1 },
    { id: "peer-mac", name: "mac", meta: META["peer-mac"], seq: 2 }, { id: "peer-pc", name: "pc", meta: n.meta, seq: 3 }] });
}
function device({ fixed = false } = {}) {
  const logs = [], sent = [], started = [];
  const n = new RoomNode({ name: "pc", pledgeGB: 6, log: (s) => logs.push(s), stripes: 0, ckpt: false });
  n.code = "4TKG9P"; n.meta = { webgpu: true, contribGB: 6, ua: "Linux" };
  n.peer = { id: "peer-pc" };
  n.sendTo = (id, m) => sent.push({ id, ...m });
  n.start = async (m) => { started.push(m); };
  for (const id of [C, "peer-mallory", "peer-mac"]) n.conns.set(id, { name: id.slice(5), meta: META[id] });
  roster(n, {}, fixed ? { fixedHost: 1 } : {});
  return { n, logs, sent, started };
}
const load = (host, extra = {}) => ({ t: "ai-load", model: "nope", range: [0, 1], next: "host", host, ...extra });
// an unknown model stops workerLoad right after the checks: an ai-error without "refused" means the load was taken
const took = (sent) => sent.at(-1)?.t === "ai-error" && !/refused/.test(sent.at(-1).message);
const refused = (sent) => sent.at(-1)?.t === "ai-error" && sent.at(-1).load === 1 && /refused/.test(sent.at(-1).message);

test("Start: followed only when its boss is the room's pick, whoever pressed it", async () => {
  const { n, logs, sent } = device();
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mallory", by: "mallory", ids: ALL });
  assert.equal(n.ai.dealer, null); assert.ok(logs.some((s) => /not following/.test(s)));
  assert.equal(sent.at(-1).t, "ai-start-refused");
  // leaving out the devices that joined after it: only a Start from seconds before they joined may (JOIN_SLACK_MS)
  for (const id of ALL) n.seenAt.set(id, performance.now() - 60000);
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mallory", by: "mallory", ids: [C, "peer-mallory"] });
  assert.equal(n.ai.dealer, null, "nor by leaving the bigger devices out");
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mac", by: "mallory", ids: ALL });
  assert.equal(n.ai.dealer, "peer-mac"); assert.equal(n.ai.hostId, "peer-mac");
});

test("review A: a member's forged ai-ready-all doesn't make it a late joiner's host; the real host still deals", async () => {
  const { n, sent } = device();
  await n.aiOnData("peer-mallory", { t: "ai-ready-all", model: "qwen3-1.7b", ids: [C, "peer-mallory"] });
  assert.equal(n.ai.dealer, null); assert.equal(n.ai.online, false);
  await n.aiOnData("peer-mallory", load("peer-mallory"));
  assert.ok(refused(sent), "its layers refused out loud (the forger's start fails)");
  // the creator names the host it checked; that one deals
  await n.aiOnData(C, { t: "ai-modelhost", id: "peer-mac" });
  await n.aiOnData("peer-mac", load("peer-mac"));
  assert.ok(took(sent)); assert.equal(n.ai.dealer, "peer-mac");
  await n.aiOnData("peer-mac", { t: "ai-ready-all", model: "qwen3-1.7b" });
  assert.equal(n.ai.online, true);
});

test("review B: an older Start (no ids) is taken only from the creator", async () => {
  const { n } = device();
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: C, by: "mallory" });
  assert.equal(n.ai.dealer, null);
  await n.aiOnData(C, { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mac", by: "host" });
  assert.equal(n.ai.dealer, "peer-mac", "the creator's, for the room's pick");
});

test("review C: ai-modelhost only from the creator", async () => {
  const { n, sent } = device();
  await n.aiOnData("peer-mallory", { t: "ai-modelhost", id: "peer-mallory" });
  await n.aiOnData("peer-mallory", load("peer-mallory"));
  assert.ok(refused(sent));
});

test("review D: a replayed Start naming the current host does nothing", async () => {
  const { n } = device();
  await n.aiOnData(C, { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mac", by: "host", ids: ALL });
  const at = n.ai.startAt;
  await new Promise((r) => setTimeout(r, 5));
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-0.6b", boss: "peer-mac", by: "mallory", ids: ALL });
  assert.equal(n.ai.startAt, at, "no refresh");
});

test("review E: while its host deals, a re-pledged member's Start is refused, however long the load takes", async () => {
  const { n, sent } = device();
  await n.aiOnData(C, { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mac", by: "host", ids: ALL });
  await n.aiOnData("peer-mac", { t: "ai-hostprog", pct: 10 });
  assert.equal(n.ai.dealerActive, true);
  n.ai.startAt = performance.now() - 10 * 60000;
  roster(n, { "peer-mallory": { webgpu: true, contribGB: 64, ua: "Win" } });
  for (const id of ALL) n.seenAt.set(id, performance.now() - 100000);
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mallory", by: "mallory", ids: ALL });
  assert.equal(n.ai.dealer, "peer-mac");
  assert.equal(sent.at(-1).t, "ai-start-refused"); assert.match(sent.at(-1).why, /in progress/);
  await n.aiOnData("peer-mac", { t: "ai-ready-all", model: "qwen3-1.7b" });
  assert.equal(n.ai.online, true, "the host's ai-ready-all still counts");
});

test("a failed start frees the node for the next Start", async () => {
  const { n } = device();
  await n.aiOnData(C, { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mac", by: "host", ids: ALL });
  await n.aiOnData("peer-mac", { t: "ai-start-failed", why: "out of memory" });
  assert.equal(n.ai.dealer, null);
});

test("metas on the node's own links don't count: only the creator's roster (review 5)", async () => {
  const { n } = device();
  // mallory tells this node directly it lends 64 GB (a hello / pledge on their link): ignored for the pick
  n.onData("peer-mallory", { t: "pledge", gb: 64 });
  await n.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mallory", by: "mallory", ids: ALL });
  assert.equal(n.ai.dealer, null);
});

test("layers from the followed host, dealt as itself", async () => {
  const { n, sent } = device();
  await n.aiOnData(C, { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mac", by: "host", ids: ALL });
  await n.aiOnData("peer-mac", load("peer-mallory"));
  assert.ok(refused(sent), "dealing for someone else");
  await n.aiOnData("peer-mac", load("peer-mac"));
  assert.ok(took(sent));
  await n.aiOnData("peer-mallory", load("peer-mallory"));
  assert.ok(refused(sent), "another host while it's here");
});

test("pooled host (fixedHost): its own deal is taken; a member's Start naming another device is refused at the creator", async () => {
  const { n, sent } = device({ fixed: true });
  await n.aiOnData(C, load(C));
  assert.ok(took(sent));
  const csent = [], started = [];
  const h = new RoomNode({ name: "host", pledgeGB: 8, log: () => {}, stripes: 0, ckpt: false });
  h.isHost = true; h.code = "4TKG9P"; h.meta = META[C]; h.peer = { id: C };
  h.sendTo = (id, m) => csent.push({ id, ...m }); h.start = async (m) => { started.push(m); };
  for (const id of ["peer-mac", "peer-mallory"]) { h.conns.set(id, { name: id.slice(5), meta: META[id] }); h.roster.set(id, { name: id.slice(5), meta: META[id], seq: id === "peer-mac" ? 2 : 1 }); }
  await h.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: "peer-mac", ids: [C, "peer-mallory", "peer-mac"] });
  assert.deepEqual(started, []); assert.equal(csent.length, 1); assert.match(csent[0].why, /pooled host/);
  await h.aiOnData("peer-mallory", { t: "ai-start-req", model: "qwen3-1.7b", boss: C, ids: [C, "peer-mallory", "peer-mac"] });
  assert.deepEqual(started, ["qwen3-1.7b"]);
  assert.equal(h.helloMsg().meta.fixedHost, 1);
});

test("creator: join numbers are given once and kept when a device relinks", () => {
  const h = new RoomNode({ name: "host", pledgeGB: 8, log: () => {}, stripes: 0, ckpt: false });
  h.isHost = true; h.code = "4TKG9P"; h.meta = META[C]; h.peer = { id: C };
  h.broadcast = () => {};
  const hi = (id, name) => { h.conns.set(id, { name, meta: {} }); h.onData(id, { t: "hello", name, v: 4, meta: { webgpu: true, contribGB: 4 } }); };
  hi("peer-a", "a"); hi("peer-b", "b");
  const a = h.roster.get("peer-a").seq;
  h.roster.delete("peer-a"); hi("peer-a", "a");
  assert.equal(h.roster.get("peer-a").seq, a);
  assert.ok(h.roster.get("peer-b").seq > a);
});
