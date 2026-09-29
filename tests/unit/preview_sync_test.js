// harness/preview-sync.js: a host's PreviewServer mirrored to a peer over an in-memory link.
import { MemoryWorkspace, watch } from "../../harness/workspace.js";
import { PreviewServer } from "../../harness/preview.js";
import { PreviewPublisher, PreviewSubscriber } from "../../harness/preview-sync.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const tick = (ms = 0) => new Promise((r) => setTimeout(r, ms));
const text = (s) => new TextDecoder().decode(s.bytes);

// host <-> peer "P"; every message goes through structuredClone like a data channel would copy it
function link({ tamper } = {}) {
  const ws = watch(new MemoryWorkspace({ "index.html": "<canvas></canvas><script type=module src=game.js></script>", "game.js": "let a = 1;\n", "big.bin": "x".repeat(150000) }));
  const server = new PreviewServer(ws, { debounce: 10 });
  const log = [];
  let sub, pub;
  const toPeer = (msg) => {
    log.push(msg.t);
    const m = structuredClone(msg);
    if (tamper) tamper(m);
    queueMicrotask(() => (m.t === "ai-pv" ? sub.onManifest("H", m) : m.t === "ai-pv-blob" ? sub.onBlob("H", m) : sub.onStop("H", m)));
  };
  pub = new PreviewPublisher(server, { send: (id, m) => toPeer(m), broadcast: toPeer, chunk: 64 << 10 });
  sub = new PreviewSubscriber({ hostId: "H", send: (m) => { log.push(m.t + ":" + m.hs.length); queueMicrotask(() => pub.onWant("P", structuredClone(m))); } });
  return { ws, server, pub, sub, log };
}
const settle = async (sub, port, rev) => { for (let i = 0; i < 2000 && sub.snapshot(port)?.rev !== rev; i++) await tick(5); return sub.snapshot(port); };

Deno.test("preview sync: the first rev fetches every blob, the next only what changed", async () => {
  const L = link();
  const ups = [];
  L.sub.onUpdate((u) => ups.push(u));
  const s1 = await L.server.serve({ port: 5173 });
  const p1 = await settle(L.sub, 5173, s1.rev);
  ok(p1, "peer has rev 1");
  eq([...p1.files.keys()], ["big.bin", "game.js", "index.html"]);
  eq(text(p1.files.get("game.js")), "let a = 1;\n");
  eq(p1.files.get("big.bin").bytes.length, 150000, "a 3-chunk file reassembled");
  ok(L.log.includes("ai-pv-want:3"), L.log.join(" "));
  eq(L.log.filter((t) => t === "ai-pv-blob").length, 5, "1 + 1 + 3 chunks");
  L.log.length = 0;
  await L.ws.write("game.js", "let a = 2;\n");
  const p2 = await settle(L.sub, 5173, s1.rev + 1);
  eq(text(p2.files.get("game.js")), "let a = 2;\n");
  ok(L.log.includes("ai-pv-want:1"), "only the changed file: " + L.log.join(" "));
  eq(ups[ups.length - 1].changed, ["game.js"]);
  L.server.stop(5173);
  await tick(5);
  eq(L.sub.snapshot(5173), null);
  ok(ups[ups.length - 1].stopped);
  L.server.close();
});

Deno.test("preview sync: bad hashes, oversized manifests and foreign senders are dropped", async () => {
  const L = link({ tamper: (m) => { if (m.t === "ai-pv-blob") new Uint8Array(m.b)[0] ^= 1; } });
  await L.server.serve({ port: 5173 });
  await tick(50);
  eq(L.sub.snapshot(5173), null, "a corrupted blob never completes the snapshot");
  L.server.close();

  const sub = new PreviewSubscriber({ hostId: "H", send: () => { throw new Error("must not ask"); }, maxFiles: 2 });
  const h = "0123456789abcdef0123";
  sub.onManifest("H", { t: "ai-pv", port: 5173, rev: 1, entry: "index.html", manifest: [["a", "", h, 1], ["b", "", h, 1], ["c", "", h, 1]] });
  sub.onManifest("H", { t: "ai-pv", port: 5173, rev: 1, entry: "index.html", manifest: [["../etc", "", h, 1]] });
  sub.onManifest("X", { t: "ai-pv", port: 5173, rev: 1, entry: "index.html", manifest: [["a", "", h, 1]] });
  eq(sub.wants.size, 0);
});

Deno.test("preview sync: want for a hash outside the manifest is ignored", async () => {
  const ws = watch(new MemoryWorkspace({ "index.html": "hi" }));
  const server = new PreviewServer(ws);
  const sent = [];
  const pub = new PreviewPublisher(server, { send: (id, m) => sent.push(m), broadcast: () => {} });
  await server.serve({ port: 5173 });
  pub.onWant("P", { port: 5173, rev: 1, hs: ["ffffffffffffffffffff"] });
  pub.onWant("P", { port: 4000, rev: 1, hs: [server.snapshot(5173).files.get("index.html").hash] });
  await tick(5);
  eq(sent.length, 0);
  pub.helloTo("P");
  eq(sent.map((m) => m.t), ["ai-pv"]);
  server.close();
});

Deno.test("preview sync: a copy corrupted once on the way is asked for again", async () => {
  let bad = 1;
  const L = link({ tamper: (m) => { if (m.t === "ai-pv-blob" && m.n === 1 && bad > 0) { bad--; new Uint8Array(m.b)[0] ^= 1; } } });
  const s1 = await L.server.serve({ port: 5173 });
  ok(await settle(L.sub, 5173, s1.rev), "completes after one re-ask: " + L.log.join(" "));
  ok(L.log.includes("ai-pv-want:1"), L.log.join(" "));
  L.server.close();
});

Deno.test("preview sync: a host cannot make a peer hold more than the limits", () => {
  const sent = [];
  const sub = new PreviewSubscriber({ hostId: "H", send: (m) => sent.push(m), maxPorts: 2 });
  const man = (port, h, size = 100) => ({ t: "ai-pv", port, rev: 1, entry: "index.html", manifest: [["index.html", "text/html", h, size]] });
  sub.onManifest("H", man(2000, "a".repeat(20)));
  sub.onManifest("H", man(2001, "b".repeat(20)));
  sub.onManifest("H", man(2002, "c".repeat(20)));
  eq(sent.map((m) => m.port), [2000, 2001], "a third port is refused");
  sub.onManifest("H", { ...man(2000, "d".repeat(20)), rev: "2" });
  eq(sent.length, 2, "a rev that is not an integer is refused");
  // a chunk bigger than the file it claims to be is dropped before it is kept
  sub.onBlob("H", { h: "a".repeat(20), i: 0, n: 1, b: new ArrayBuffer(4096) });
  eq(sub._held(), 0);
  eq(sent.length, 3, "and asked for again");
  // a replaced rev's parts are let go
  sub.onManifest("H", { ...man(2000, "e".repeat(20)), rev: 2 });
  ok(!sub.parts.has("a".repeat(20)) && sub.parts.has("e".repeat(20)), [...sub.parts.keys()].join(","));
});
