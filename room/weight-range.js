// Validate HTTP weight ranges before loaders or the raw cache consume them. This
// checks placement and byte count, not content hashes; synthetic cache/peer 200s
// have their own checks and must not pass through this HTTP-only helper.
export class WeightRangeError extends Error {
  constructor(message, kind, status) {
    super(message);
    this.name = "WeightRangeError";
    this.kind = kind;
    this.status = status;
  }
}

// Return a response with the original status and headers and a counted body.
// Only one upstream read runs per downstream pull, with no additional queued
// chunks. A request past EOF may end at total - 1; other truncated ranges fail.
// Completion is checked at EOF, including after the expected last byte.
export function validateWeightRange(response, lo, hi, expectedBytes = null) {
  const status = response.status;
  const reject = (message, kind) => {
    const error = new WeightRangeError(message, kind, status);
    try { response.body?.cancel(error).catch(() => {}); } catch { /* already consumed */ }
    throw error;
  };
  if (!Number.isSafeInteger(lo) || !Number.isSafeInteger(hi) || lo < 0 || hi < lo || !Number.isSafeInteger(hi - lo + 1))
    reject("Invalid requested weight range", "range");
  if (expectedBytes != null && (!Number.isSafeInteger(expectedBytes) || expectedBytes <= lo))
    reject("Invalid expected weight file size", "range");
  if (status !== 206) reject(`Weight range request returned HTTP ${status}, expected 206`, "status");

  const match = /^bytes (\d+)-(\d+)\/(\d+)$/i.exec((response.headers.get("content-range") || "").trim());
  if (!match) reject("Weight range response has no valid Content-Range", "range");
  const [start, end, total] = match.slice(1).map(Number);
  if (![start, end, total].every(Number.isSafeInteger) || start !== lo || total <= lo || end !== Math.min(hi, total - 1))
    reject(`Weight range response does not match bytes ${lo}-${hi}`, "range");
  if (expectedBytes != null && total !== expectedBytes)
    reject(`Weight file size changed: received ${total} bytes, expected ${expectedBytes}`, "range");
  const length = end - start + 1;
  const contentLength = response.headers.get("content-length");
  if (contentLength != null && (!/^\d+$/.test(contentLength) || Number(contentLength) !== length))
    reject(`Weight range Content-Length does not match ${length} bytes`, "length");
  if (!response.body) reject("Weight range response has no body", "length");

  const reader = response.body.getReader();
  let got = 0;
  const body = new ReadableStream({
    async pull(controller) {
      try {
        const { value, done } = await reader.read();
        if (done) {
          if (got !== length) throw new WeightRangeError(`Short weight range: received ${got} of ${length} bytes`, "length", status);
          controller.close();
          reader.releaseLock();
          return;
        }
        if (!(value instanceof Uint8Array)) throw new WeightRangeError("Weight range body contains invalid bytes", "length", status);
        got += value.byteLength;
        if (got > length) throw new WeightRangeError(`Long weight range: received more than ${length} bytes`, "length", status);
        controller.enqueue(value);
      } catch (error) {
        controller.error(error);
        try { reader.cancel(error).catch(() => {}); } catch { /* already closed */ }
        reader.releaseLock();
      }
    },
    async cancel(reason) {
      try { await reader.cancel(reason); } finally { reader.releaseLock(); }
    },
  }, { highWaterMark: 0 });
  return new Response(body, { status, statusText: response.statusText, headers: response.headers });
}
