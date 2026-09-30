// room/signal.js: parsing signaling server specs, the ordered list (?signal= > configured > cloud) and
// openPeer's fallback against a fake Peer (down, black-holed, taken id, all down). No network.
import { parseServer, serverList, openPeer, reconnectDelay, CLOUD_HOST } from "../../room/signal.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

Deno.test("parseServer: cloud, bare host:port, URLs, junk", () => {
  eq(parseServer("cloud"), { spec: "cloud", label: CLOUD_HOST, opts: {} });
  eq(parseServer("0.peerjs.com").spec, "cloud");
  // the old ?signal=host:port: secure follows the page, port defaults to 443, spec kept as typed
  eq(parseServer("127.0.0.1:9000", false), { spec: "127.0.0.1:9000", label: "127.0.0.1:9000", opts: { host: "127.0.0.1", port: 9000, path: "/", secure: false } });
  eq(parseServer("sig.example.com", true).opts, { host: "sig.example.com", port: 443, path: "/", secure: true });
  eq(parseServer("wss://sig.example.com/pooled", false), { spec: "wss://sig.example.com:443/pooled", label: "sig.example.com", opts: { host: "sig.example.com", port: 443, path: "/pooled/", secure: true } });
  eq(parseServer("wss://sig.example.com/pooled/").spec, "wss://sig.example.com:443/pooled", "trailing slash: same server");
  eq(parseServer(parseServer("wss://sig.example.com:8443/a/b").spec).opts, { host: "sig.example.com", port: 8443, path: "/a/b/", secure: true }, "spec round-trips");
  eq(parseServer("ws://10.0.0.2:9000", true).opts, { host: "10.0.0.2", port: 9000, path: "/", secure: false });
  eq(parseServer("http://box").opts.port, 80);
  eq(parseServer({ host: "sig.example.com", port: 8443, secure: true, key: "k" }).opts, { host: "sig.example.com", port: 8443, path: "/", secure: true, key: "k" });
  for (const bad of ["", "  ", "a b", "<x>:1", "host:99999", "javascript:alert(1)", "ftp://x", null, { host: "a/b" }]) eq(parseServer(bad), null, "junk " + JSON.stringify(bad));
});

Deno.test("serverList: ?signal= wins, then configured, then the cloud; dedupe and skip junk", () => {
  eq(serverList().map((s) => s.spec), ["cloud"]);
  eq(serverList({ configured: ["cloud", "wss://b.example:443", "cloud", "%%"] }).map((s) => s.label), [CLOUD_HOST, "b.example"]);
  eq(serverList({ query: "127.0.0.1:9000", configured: ["cloud", "b.example"], pageSecure: false }).map((s) => s.spec), ["127.0.0.1:9000"]);
  eq(serverList({ query: "127.0.0.1:1,127.0.0.1:9000", pageSecure: false }).map((s) => s.opts.port), [1, 9000]);
  eq(serverList({ query: "%%%", configured: ["b.example"] }).map((s) => s.label), ["b.example"], "an unparseable ?signal= falls through");
  eq(serverList({ configured: "not-an-array" }).map((s) => s.spec), ["cloud"]);
});

// A fake PeerJS: behaviour per host. "up" opens, "down" errors (network), "hole" never answers,
// "taken" answers unavailable-id, "noid" fails the id fetch (server-error).
function fakePeer(plan, made) {
  return class {
    constructor(id, opts) {
      this.id = id; this.opts = opts; this.h = {}; this.destroyed = false;
      made.push(this);
      const how = plan[opts.host || "cloud"];
      setTimeout(() => {
        if (how === "up") this.emit("open", id || "rnd");
        else if (how === "down") this.emit("error", { type: "network" });
        else if (how === "noid") this.emit("error", { type: "server-error" });
        else if (how === "taken") this.emit("error", { type: "unavailable-id" });
      }, 1);
    }
    on(k, f) { (this.h[k] ||= []).push(f); }
    emit(k, v) { for (const f of this.h[k] || []) f(v); }
    destroy() { this.destroyed = true; }
  };
}
const list = (...hosts) => hosts.map((h) => h === "cloud" ? parseServer("cloud") : parseServer(`ws://${h}:9000`));

Deno.test("openPeer: first server up wins, keeps base options", async () => {
  const made = [];
  const r = await openPeer(fakePeer({ cloud: "up" }, made), "pooled-room-ABCD", { debug: 1, config: { x: 1 } }, list("cloud", "b"));
  eq(r.index, 0); eq(r.server.spec, "cloud"); eq(made.length, 1);
  eq(made[0].opts, { debug: 1, config: { x: 1 } }); eq(made[0].id, "pooled-room-ABCD");
});

Deno.test("openPeer: down and black-holed servers hand over, in order", async () => {
  const made = [], tries = [];
  const r = await openPeer(fakePeer({ cloud: "down", a: "hole", b: "noid", c: "up" }, made), undefined, {}, list("cloud", "a", "b", "c"),
    { timeoutMs: 30, onTry: (s, i, err) => tries.push([s.label, i, err?.type || null]) });
  eq(r.index, 3); eq(r.server.label, "c:9000");
  eq(tries, [[CLOUD_HOST, 0, null], ["a:9000", 1, "network"], ["b:9000", 2, "timeout"], ["c:9000", 3, "server-error"]]);
  ok(made.slice(0, 3).every((p) => p.destroyed), "failed peers destroyed");
  ok(!made[3].destroyed, "the winner stays");
});

Deno.test("openPeer: a taken id is an answer, not an outage (no fallback: that would split the room)", async () => {
  const made = [];
  let err = null;
  try { await openPeer(fakePeer({ cloud: "taken", b: "up" }, made), "pooled-room-ABCD", {}, list("cloud", "b")); } catch (e) { err = e; }
  eq(err?.type, "unavailable-id"); eq(err.server.spec, "cloud"); eq(made.length, 1);
});

Deno.test("openPeer: all down rejects signaling-down with what it tried; from skips ahead", async () => {
  const made = [];
  let err = null;
  try { await openPeer(fakePeer({ cloud: "down", b: "hole" }, made), undefined, {}, list("cloud", "b"), { timeoutMs: 20, lastTimeoutMs: 20 }); } catch (e) { err = e; }
  eq(err?.type, "signaling-down"); eq(err.tried, [CLOUD_HOST, "b:9000"]); eq(err.cause?.type, "timeout");
  const r = await openPeer(fakePeer({ cloud: "up", b: "up" }, []), undefined, {}, list("cloud", "b"), { from: 1 });
  eq(r.index, 1);
});

Deno.test("openPeer: the last server waits longer than the ones with a fallback after them", async () => {
  const t = Date.now();
  let err = null;
  try { await openPeer(fakePeer({ a: "hole", b: "hole" }, []), undefined, {}, list("a", "b"), { timeoutMs: 20, lastTimeoutMs: 120 }); } catch (e) { err = e; }
  eq(err?.type, "signaling-down");
  ok(Date.now() - t >= 135, "20 ms + 120 ms");
});

Deno.test("reconnectDelay: 2 s doubling, capped at 30 s", () => {
  eq([0, 1, 2, 3, 4, 5, 50].map(reconnectDelay), [2000, 4000, 8000, 16000, 30000, 30000, 30000]);
});
