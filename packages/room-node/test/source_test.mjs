// No GPU, no network: how a streamed shard's bytes are fetched (source.js planChunks,
// rangePrefetcher, fetchRange, openModel with an injected fetch) and the node's loadstat.
//   node --test packages/room-node/test/*_test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";
import { planChunks, rangePrefetcher, fetchRange, openModel } from "../source.js";
import { RoomNode } from "../roomnode.js";

const MiB = 2 ** 20;

test("planChunks: neighbours merge up to the chunk size, a long range is cut, far ranges stay apart", () => {
  const c = planChunks([{ off: 100, len: 10 }, { off: 0, len: 50 }, { off: 60, len: 30 }, { off: 10 * MiB, len: 5 }], { maxChunk: 1000, gap: 20 });
  assert.deepEqual(c.map(({ start, end }) => [start, end]), [[0, 110], [10 * MiB, 10 * MiB + 5]]);
  assert.equal(c[0].order, 0);   // the first read that needs it (ranges in the loader's order)
  const big = planChunks([{ off: 0, len: 2500 }], { maxChunk: 1000 });
  assert.deepEqual(big.map(({ start, end }) => [start, end]), [[0, 1000], [1000, 2000], [2000, 2500]]);
  // a chunk never grows past maxChunk by merging
  const many = planChunks(Array.from({ length: 10 }, (_, i) => ({ off: i * 300, len: 300 })), { maxChunk: 1000, gap: 0 });
  assert.ok(many.every((x) => x.end - x.start <= 1000));
  assert.equal(many.reduce((a, x) => a + x.end - x.start, 0), 3000);
});

// a file in memory behind a fetchRange that records concurrency
function fakeFile(n) {
  const body = randomBytes(n);
  const st = { live: 0, max: 0, calls: [] };
  const get = async (a, b, onBytes) => {
    st.live++; st.max = Math.max(st.max, st.live); st.calls.push([a, b]);
    await new Promise((r) => setTimeout(r, 2 + Math.random() * 5));
    st.live--;
    onBytes(b - a);
    return new Uint8Array(body.subarray(a, b));
  };
  return { body, st, get };
}

test("rangePrefetcher: every read gets its bytes (also across chunks), in few requests, at most N at once, memory bounded", async () => {
  const { body, st, get } = fakeFile(4 * MiB);
  // "tensors" of random sizes, read in a shuffled-but-planned order
  const ranges = [];
  for (let off = 4096; off < body.length - 70000;) { const len = 1000 + Math.floor(Math.random() * 60000); ranges.push({ off, len }); off += len + Math.floor(Math.random() * 64); }
  const pre = rangePrefetcher({ fetchRange: get, concurrency: 3, capBytes: 256 * 1024, maxChunk: 64 * 1024, gap: 128 });
  const total = pre.plan(ranges);
  assert.ok(total >= ranges.reduce((a, r) => a + r.len, 0));
  let maxHeld = 0;
  for (const r of ranges) {
    const b = await pre.read(r.off, r.len);
    maxHeld = Math.max(maxHeld, pre.held);
    assert.deepEqual(Buffer.from(b), body.subarray(r.off, r.off + r.len));
  }
  assert.ok(st.max <= 3, `at most 3 in flight (${st.max})`);
  assert.ok(st.calls.length < ranges.length, `fewer requests (${st.calls.length}) than reads (${ranges.length})`);
  assert.ok(maxHeld <= 256 * 1024 + 2 * 64 * 1024, `held ${maxHeld}`);
  assert.equal(pre.held, 0, "everything read is let go");
  // outside the plan: null (the caller fetches it itself)
  assert.equal(await pre.read(0, 100), null);
  pre.close();
});

test("rangePrefetcher: a read the plan did not expect first still goes (no deadlock under a small cap)", async () => {
  const { body, get } = fakeFile(MiB);
  const ranges = Array.from({ length: 16 }, (_, i) => ({ off: i * 65536, len: 65536 }));
  const pre = rangePrefetcher({ fetchRange: get, concurrency: 2, capBytes: 65536, maxChunk: 65536, gap: 0 });
  pre.plan(ranges);
  for (const r of [...ranges].reverse()) assert.deepEqual(Buffer.from(await pre.read(r.off, r.len)), body.subarray(r.off, r.off + r.len));
  pre.close();
});

test("fetchRange: retries a 503 and a dropped body, gives up at once on a 404, and takes back what a failed try counted", async () => {
  const body = randomBytes(1000);
  let n = 0;
  const flaky = async (url, { headers }) => {
    const [, a, b] = /bytes=(\d+)-(\d+)/.exec(headers.range).map(Number);
    n++;
    if (n === 1) return new Response("busy", { status: 503 });
    if (n === 2) {   // half the body, then the connection drops
      const s = new ReadableStream({ start(c) { c.enqueue(body.subarray(a, a + 100)); c.error(new TypeError("fetch failed")); } });
      return new Response(s, { status: 206 });
    }
    return new Response(body.subarray(a, b + 1), { status: 206 });
  };
  let counted = 0;
  const out = await fetchRange(flaky, "u", 10, 510, (k) => { counted += k; }, { backoffMs: 1 });
  assert.deepEqual(Buffer.from(out), body.subarray(10, 510));
  assert.equal(counted, 500);
  assert.equal(n, 3);
  let m = 0;
  await assert.rejects(fetchRange(async () => { m++; return new Response("", { status: 404 }); }, "u", 0, 10, () => {}, { backoffMs: 1 }), /404/);
  assert.equal(m, 1);
});

test("openModel streaming: plan() fetches the shard ahead in big ranges and counts it in stat", async () => {
  const body = randomBytes(3 * MiB);
  const seen = [];
  const fetchImpl = async (url, { headers }) => {
    const [, a, b] = /bytes=(\d+)-(\d+)/.exec(headers.range).map(Number);
    seen.push([a, b]);
    return new Response(body.subarray(a, Math.min(b + 1, body.length)), { status: 206 });
  };
  const src = openModel("qwen3-1.7b", { modelDir: null, fetch: fetchImpl });
  assert.equal(src.stat.from, "Hugging Face");
  const infos = Array.from({ length: 40 }, (_, i) => ({ byteOffset: 1000 + i * 70000, byteLength: 69000 }));
  src.plan(infos);
  assert.ok(src.stat.total >= 40 * 69000);
  for (const t of infos) assert.deepEqual(Buffer.from(await src.bytesOf(t)), body.subarray(t.byteOffset, t.byteOffset + t.byteLength));
  assert.ok(seen.length <= 2, `one or two ranges (${seen.length})`);
  assert.equal(src.stat.fetched, src.stat.total);
  await src.close();
});

test("RoomNode: watchLoad emits loadstat { from, fetched, total, bps }, beforeLoad is kept", async () => {
  const hook = async () => {};
  const n = new RoomNode({ name: "t", pledgeGB: 4, log: () => {}, beforeLoad: hook });
  assert.equal(n.beforeLoad, hook);
  const src = { stat: { from: "Hugging Face", fetched: 0, total: 1000, planned: true } };
  const got = [];
  n.on("loadstat", (s) => got.push(s));
  const stop = n.watchLoad(src, { everyMs: 5 });
  await new Promise((r) => setTimeout(r, 20));
  src.stat.fetched = 600;
  await new Promise((r) => setTimeout(r, 20));
  stop();
  const last = got.at(-1);
  assert.deepEqual(Object.keys(last).sort(), ["bps", "fetched", "from", "total"]);
  assert.equal(last.fetched, 600); assert.equal(last.total, 1000); assert.equal(last.from, "Hugging Face");
  assert.ok(last.bps > 0);
  assert.deepEqual(n.loadStat, last);
});
