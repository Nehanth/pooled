// cli/lib/room.js without a network: what the Bridge keeps from the host's messages.
import { test } from "node:test";
import assert from "node:assert/strict";
import { Bridge, guardChunks } from "../lib/room.js";
import { cleanText } from "../lib/common.js";

const ESC = "\x1b]52;c;ZXZpbA==\x07\x1b[2Jpwned\r\n\x9b31m";

test("text from the host reaches the terminal with no control characters", () => {
  const logs = [];
  const b = new Bridge({ code: "ABCD", name: "t", client: "c", log: (m) => logs.push(m) });
  b.onData({ t: "ai-ready-all", model: "qwen3-1.7b" + ESC, label: "Qwen3 1.7B" + ESC });
  assert.match(b.model, /^qwen3-1\.7b[\w.:+\-/=]*$/);
  assert.ok(b.modelLabel.startsWith("Qwen3 1.7B") && !/[\u0000-\u001f\u007f-\u009f]/.test(b.modelLabel), JSON.stringify(b.modelLabel));
  b.onData({ t: "bye", reason: "go away" + ESC + "x".repeat(1000) });
  assert.ok(!/[\u0000-\u001f\u007f-\u009f]/.test(b.kicked + logs.join("")), JSON.stringify(logs));
  assert.ok(b.kicked.length <= 300);
  assert.equal(cleanText("a\x1b[2Jb\u0085c"), "a[2Jbc");
});

test("chunked messages from the host are rebuilt only within limits", () => {
  const seen = [];
  const conn = { _chunkedData: {}, _handleChunk(d) { seen.push(d.n); this._chunkedData[d.__peerData] = 1; } };
  guardChunks(conn);
  conn._handleChunk({ __peerData: 1, total: 3, n: 0, data: [] });
  conn._handleChunk({ __peerData: 2, total: 1e9, n: 0, data: [] });
  conn._handleChunk({ __peerData: 3, total: 3, n: 7, data: [] });
  conn._handleChunk({ __peerData: 4, total: "3", n: 0, data: [] });
  assert.deepEqual(seen, [0]);
  for (let id = 10; id < 30; id++) conn._handleChunk({ __peerData: id, total: 2, n: 0, data: [] });
  assert.equal(Object.keys(conn._chunkedData).length, 8, "at most 8 messages rebuilt at once");
  conn._handleChunk({ __peerData: 1, total: 3, n: 1, data: [] });
  assert.equal(seen.at(-1), 1, "one already started goes on");
});

test("the host's API version and context size: hello meta, then ai-ready-all keeps ctx current", () => {
  const b = new Bridge({ code: "ABCD", name: "t", client: "c" });
  assert.equal(b.hostApi, 1, "no hello yet: plain asks only");
  b.hostMeta = { api: 2, ctx: 8192 };
  assert.equal(b.hostApi, 2);
  b.onData({ t: "ai-ready-all", model: "m", label: "M", ctx: 65536 });
  assert.equal(b.hostMeta.ctx, 65536);
  b.onData({ t: "ai-ready-all", model: "m", label: "M", ctx: "lots" });
  assert.equal(b.hostMeta.ctx, 65536, "only a positive integer");
  b.hostMeta = { api: true };
  assert.equal(b.hostApi, 1, "an old bridge's truthy api is 1");
});

test("ai-call messages reach the ask they belong to; a v2 body goes out as api 2", () => {
  const b = new Bridge({ code: "ABCD", name: "t", client: "c" });
  const sent = [], got = [];
  b.conn = { open: true, send: (m) => sent.push(m) };
  assert.ok(b.ask("r1", { api: 2, system: "", messages: [], params: {} }, (d) => got.push(d.t)));
  assert.equal(sent[0].api, 2, "the body's api wins over the default 1");
  assert.ok(b.ask("r2", { system: "", messages: [], params: {} }, () => {}));
  assert.equal(sent[1].api, 1);
  b.onData({ t: "ai-call", rid: "r1", i: 0, name: "f" });
  b.onData({ t: "ai-call", rid: "zz", i: 0, name: "f" });
  b.onData({ t: "ai-gendone", rid: "r1" });
  b.onData({ t: "ai-call", rid: "r1", i: 0, a: "late" });
  assert.deepEqual(got, ["ai-call", "ai-gendone"], "routed by rid; nothing after the end");
});

test("a Peer class passed in is used instead of loading a second WebRTC stack (@pooled/room-node's)", async () => {
  const made = [];
  class FakePeer {
    constructor(id, opts) { made.push(opts); this.h = {}; setTimeout(() => this.h.error?.({ type: "peer-unavailable" }), 0); }
    on(ev, f) { this.h[ev] = f; }
    destroy() {}
  }
  const b = new Bridge({ code: "ABCD", name: "t", client: "c", Peer: FakePeer });
  await assert.rejects(b.connect(), /no room ABCD/);
  assert.equal(made.length, 1);
});

import { roomCodeFrom, roomKeyFrom } from "../lib/room.js";
const KEY = "AbCdEfGhIjKlMnOpQrStUv";

test("room codes and invite links: six characters, four still, the key from the fragment", () => {
  assert.equal(roomCodeFrom("4tk-g9p"), "4TKG9P");
  assert.equal(roomCodeFrom("4TKG9P"), "4TKG9P");
  assert.equal(roomCodeFrom("ABCD"), "ABCD");
  assert.equal(roomCodeFrom("ABCDE"), null);
  assert.equal(roomCodeFrom(`https://pooled.run/r/4TKG9P#k=${KEY}`), "4TKG9P");
  assert.equal(roomCodeFrom(`http://127.0.0.1:8080/p2p.html?code=4TKG9P&signal=x#k=${KEY}`), "4TKG9P");
  assert.equal(roomCodeFrom("https://pooled.run/r/ABCD"), "ABCD");
  assert.equal(roomKeyFrom(`https://pooled.run/r/4TKG9P#k=${KEY}`), KEY);
  assert.equal(roomKeyFrom("https://pooled.run/r/4TKG9P"), null);
  assert.equal(roomKeyFrom("4TKG9P"), null);
  assert.equal(roomKeyFrom("https://pooled.run/r/4TKG9P#k=short"), null);
});

// a stand-in PeerJS link: what the bridge sends, and the host's messages delivered by hand
function fakeLink(b) {
  const handlers = {}, sent = [];
  const conn = { open: true, on: (ev, f) => { handlers[ev] = f; }, send: (m) => sent.push(m), close() {} };
  b.peer = { connect: () => conn };
  return { conn, sent, open: () => handlers.open(), host: (m) => handlers.data(m) };
}

test("the bridge waits in the host's lobby, then keeps the pass it is given", () => {
  const logs = [];
  const b = new Bridge({ code: "4TKG9P", name: "t", client: "c", log: (m) => logs.push(m) });
  const L = fakeLink(b);
  let greeted = null, lobby = 0;
  b.on("lobby", () => lobby++);
  b.dial(false, (h) => { greeted = h; });
  L.open();
  assert.equal(L.sent[0].join, 1, "it can wait");
  assert.equal(L.sent[0].key, undefined, "no key without a link");
  L.host({ t: "hello", name: "host", v: 4, gate: 1, ask: 1, meta: { api: 2 } });
  assert.equal(greeted, null, "a host with the gate: not in yet");
  assert.equal(b.connected, false);
  L.host({ t: "ai-ready-all", model: "m" });
  assert.equal(b.ready, false, "nothing from the room before admit");
  L.host({ t: "lobby" });
  assert.equal(b.waiting, true); assert.equal(lobby, 1);
  assert.match(logs.join("\n"), /waiting for the host of room 4TKG9P/);
  L.host({ t: "admit", pass: KEY });
  assert.equal(greeted?.name, "host");
  assert.equal(b.connected, true); assert.equal(b.waiting, false);
  assert.equal(b.pass, KEY);
  assert.equal(b.helloMsg(true).pass, KEY, "a reconnect shows the pass");
});

test("the bridge: an invite key goes in the hello; an older host lets it in at once; Deny is final", () => {
  const b = new Bridge({ code: "4TKG9P", key: KEY, name: "t", client: "c" });
  let L = fakeLink(b), greeted = null;
  b.dial(false, (h) => { greeted = h; });
  L.open();
  assert.equal(L.sent[0].key, KEY);
  L.host({ t: "hello", name: "old host", v: 4, meta: { api: 2 } });   // no gate: 1
  assert.equal(greeted?.name, "old host");
  assert.equal(b.connected, true);

  const c = new Bridge({ code: "4TKG9P", name: "t", client: "c" });
  L = fakeLink(c);
  const refused = [];
  c.on("refused", (why) => refused.push(why));
  c.dial(false, () => { throw new Error("must not get in"); });
  L.open();
  L.host({ t: "hello", name: "host", v: 4, gate: 1, ask: 1, meta: { api: 2 } });
  L.host({ t: "lobby" });
  L.host({ t: "bye", reason: "The host didn't let this device in." });
  assert.deepEqual(refused, ["The host didn't let this device in."]);
  assert.equal(c.kicked, "The host didn't let this device in.");
  assert.equal(c.connected, false);
});
