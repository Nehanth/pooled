import { validateWeightRange, WeightRangeError } from "../../room/weight-range.js";

const eq = (got, expected) => {
  if (JSON.stringify(got) !== JSON.stringify(expected)) throw new Error(`Expected ${JSON.stringify(expected)}, received ${JSON.stringify(got)}`);
};
const ok = (value, message) => { if (!value) throw new Error(message || "Assertion failed"); };
function throws(fn, kind) {
  try { fn(); } catch (error) { ok(error instanceof WeightRangeError); eq(error.kind, kind); return error; }
  throw new Error("Expected WeightRangeError");
}
async function rejects(fn, pattern) {
  try { await fn(); } catch (error) { ok(pattern.test(error.message), error.message); return error; }
  throw new Error("Expected rejection");
}
function streamed(parts, { onPull = () => {}, onCancel = () => {} } = {}) {
  let index = 0;
  return new ReadableStream({
    pull(controller) {
      onPull();
      if (index < parts.length) controller.enqueue(new Uint8Array(parts[index++]));
      else controller.close();
    },
    cancel: onCancel,
  }, { highWaterMark: 0 });
}
const response = (parts, headers = {}, options = {}) => new Response(streamed(parts, options), {
  status: 206, statusText: "Partial Content", headers: { "content-range": "bytes 10-13/20", ...headers },
});

Deno.test("HTTP weight range preserves valid response metadata and streamed bytes", async () => {
  const source = response([[1, 2], [3, 4]], { "content-length": "4", "content-type": "application/octet-stream", "etag": "stable" });
  const result = validateWeightRange(source, 10, 13, 20);
  eq(result.status, 206); eq(result.statusText, "Partial Content");
  eq(result.headers.get("content-range"), "bytes 10-13/20");
  eq(result.headers.get("content-type"), "application/octet-stream"); eq(result.headers.get("etag"), "stable");
  eq([...new Uint8Array(await result.arrayBuffer())], [1, 2, 3, 4]);
});

Deno.test("HTTP weight range accepts a concrete total without a catalogue size", async () => {
  eq([...new Uint8Array(await validateWeightRange(response([[0, 1, 2, 3]]), 10, 13).arrayBuffer())], [0, 1, 2, 3]);
});

Deno.test("HTTP weight range accepts only legitimate EOF clamping for an oversized header probe", async () => {
  for (const expectedBytes of [null, 14]) {
    const result = validateWeightRange(response([[1, 2], [3, 4]], {
      "content-range": "bytes 10-13/14", "content-length": "4",
    }), 10, 12 * 2 ** 20 - 1, expectedBytes);
    eq([...new Uint8Array(await result.arrayBuffer())], [1, 2, 3, 4]);
  }
});

Deno.test("HTTP weight range does not mistake a short non-EOF response for EOF clamping", () => {
  throws(() => validateWeightRange(response([], { "content-range": "bytes 10-13/20" }), 10, 19), "range");
  throws(() => validateWeightRange(response([], { "content-range": "bytes 11-13/14" }), 10, 30), "range");
  throws(() => validateWeightRange(response([], { "content-range": "bytes 10-13/14" }), 10, 30, 20), "range");
});

Deno.test("HTTP weight range checks the actual EOF-clamped body length", async () => {
  await rejects(() => validateWeightRange(response([[1, 2, 3]], {
    "content-range": "bytes 10-13/14", "content-length": "4",
  }), 10, 30).arrayBuffer(), /Short weight range: received 3 of 4 bytes/);
  await rejects(() => validateWeightRange(response([[1, 2, 3, 4, 5]], {
    "content-range": "bytes 10-13/14",
  }), 10, 30).arrayBuffer(), /Long weight range/);
});

Deno.test("HTTP weight range rejects a request beginning at or past EOF", () => {
  throws(() => validateWeightRange(response([], { "content-range": "bytes 10-13/10" }), 10, 13), "range");
  throws(() => validateWeightRange(response([]), 10, 13, 10), "range");
});

Deno.test("HTTP weight range rejects a different offset even at the same length", () => {
  const error = throws(() => validateWeightRange(response([[1, 2, 3, 4]]), 0, 3, 20), "range");
  eq(error.status, 206);
});

Deno.test("HTTP weight range rejects changed file totals and impossible ranges", () => {
  throws(() => validateWeightRange(response([], { "content-range": "bytes 10-13/21" }), 10, 13, 20), "range");
  throws(() => validateWeightRange(response([], { "content-range": "bytes 10-13/13" }), 10, 13), "range");
  throws(() => validateWeightRange(response([], { "content-range": "bytes 10-14/20" }), 10, 13), "range");
});

Deno.test("HTTP weight range rejects missing, malformed and unrepresentable Content-Range", () => {
  for (const header of ["", "bytes 10-13/*", "bytes */20", "bytes 10-13/9007199254740992", "bytes 10-13/20 trailing", "items 10-13/20"])
    throws(() => validateWeightRange(response([], { "content-range": header }), 10, 13), "range");
});

Deno.test("HTTP weight range rejects incompatible Content-Length", () => {
  for (const length of ["3", "5", "4x", "-4", "4.0"])
    throws(() => validateWeightRange(response([], { "content-length": length }), 10, 13), "length");
});

Deno.test("HTTP weight range does not accept synthetic cache or peer 200 responses", () => {
  const error = throws(() => validateWeightRange(new Response(new Uint8Array(4), {
    headers: { "content-range": "bytes 10-13/20", "x-swarm-len": "4" },
  }), 10, 13, 20), "status");
  eq(error.status, 200);
});

Deno.test("HTTP weight range cancels an invalid header's upstream body", async () => {
  let cancelled = false;
  throws(() => validateWeightRange(response([], { "content-range": "bytes 0-3/20" }, { onCancel: () => { cancelled = true; } }), 10, 13), "range");
  await Promise.resolve();
  ok(cancelled);
});

Deno.test("HTTP weight range rejects a missing body", () => {
  throws(() => validateWeightRange(new Response(null, { status: 206, headers: { "content-range": "bytes 10-13/20" } }), 10, 13), "length");
});

Deno.test("HTTP weight range checks actual length even when headers claim completeness", async () => {
  const error = await rejects(() => validateWeightRange(response([[1], [2, 3]], { "content-length": "4" }), 10, 13).arrayBuffer(), /Short weight range: received 3 of 4 bytes/);
  ok(error instanceof WeightRangeError); eq(error.kind, "length");
});

Deno.test("HTTP weight range rejects excess bytes before forwarding the excess chunk", async () => {
  let cancelled = false;
  const reader = validateWeightRange(response([[1, 2], [3, 4, 5]], {}, { onCancel: () => { cancelled = true; } }), 10, 13).body.getReader();
  eq([...((await reader.read()).value)], [1, 2]);
  await rejects(() => reader.read(), /Long weight range/);
  ok(cancelled);
});

Deno.test("HTTP weight range checks EOF after the last expected byte", async () => {
  await rejects(() => validateWeightRange(response([[1, 2, 3, 4], [5]]), 10, 13).arrayBuffer(), /Long weight range/);
});

Deno.test("HTTP weight range does not drain upstream ahead of the consumer", async () => {
  let pulls = 0;
  const result = validateWeightRange(response([[1], [2], [3], [4]], {}, { onPull: () => { pulls++; } }), 10, 13);
  await Promise.resolve(); await Promise.resolve();
  eq(pulls, 0);
  const reader = result.body.getReader();
  eq([...((await reader.read()).value)], [1]);
  await Promise.resolve(); await Promise.resolve();
  eq(pulls, 1);
  await reader.cancel();
});

Deno.test("HTTP weight range forwards consumer cancellation upstream", async () => {
  let reason;
  const result = validateWeightRange(response([[1], [2, 3, 4]], {}, { onCancel: (value) => { reason = value; } }), 10, 13);
  const reader = result.body.getReader();
  await reader.read();
  await reader.cancel("load stopped");
  eq(reason, "load stopped");
});

Deno.test("HTTP weight range preserves upstream errors", async () => {
  const failure = new Error("transport failed");
  const source = new Response(new ReadableStream({ pull(controller) { controller.error(failure); } }, { highWaterMark: 0 }), {
    status: 206, headers: { "content-range": "bytes 10-13/20" },
  });
  ok((await rejects(() => validateWeightRange(source, 10, 13).arrayBuffer(), /transport failed/)) === failure);
});

Deno.test("HTTP weight range supports large file offsets without 32-bit truncation", async () => {
  const lo = 2 ** 35, hi = lo + 3, total = hi + 1;
  eq((await validateWeightRange(response([[1, 2, 3, 4]], { "content-range": `bytes ${lo}-${hi}/${total}` }), lo, hi, total).arrayBuffer()).byteLength, 4);
});

Deno.test("HTTP weight range rejects unsafe or invalid requested bounds", () => {
  for (const [lo, hi] of [[-1, 3], [10, 9], [0.5, 3], [0, Number.MAX_SAFE_INTEGER], [NaN, 3]])
    throws(() => validateWeightRange(response([]), lo, hi), "range");
  for (const total of [13, -1, 20.5, Number.MAX_SAFE_INTEGER + 1])
    throws(() => validateWeightRange(response([]), 10, 13, total), "range");
});
