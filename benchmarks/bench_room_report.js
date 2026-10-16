// Pure report calculations shared by the runner and its deterministic tests.
const identity = (r) => JSON.stringify([r.kind, r.pos, r.n, r.spec]);
const finite = (n) => typeof n === "number" && Number.isFinite(n) && n >= 0;
const operations = new Set(["host", "head", "head-host", "prefill", "sample", "verify"]);
const tokenId = (n) => Number.isSafeInteger(n) && n >= 0 && n <= 0xffffffff;

function checkGeneration(r) {
  const valid = r.event === "start" ? Array.isArray(r.ids) && r.ids.every(tokenId)
    : r.event === "token" ? tokenId(r.id) && typeof r.drafted === "boolean"
    : r.event === "end" ? Array.isArray(r.result?.tokens) && r.result.tokens.every(tokenId)
    : r.event === "error" && typeof r.error === "string";
  if (!valid) throw new Error("invalid generation observation");
}

// Require exactly one complete attempt; retries or partial token traces cannot
// establish which host output belongs to the requesting client's measurement.
export function generationSample(capture) {
  if (!capture || capture.dropped !== 0) throw new Error("missing or dropped generation evidence");
  const records = capture.records.filter((r) => r.phase === "generation");
  for (const r of records) checkGeneration(r);
  if (records.filter((r) => r.event === "start").length !== 1 || records[0]?.event !== "start" ||
      records.at(-1)?.event !== "end" || records.slice(1, -1).some((r) => r.event !== "token")) {
    throw new Error("generation is incomplete or restarted");
  }
  const [start] = records, end = records.at(-1), tokens = records.slice(1, -1);
  if (records.some((r, i) => !finite(r.at) || (i && r.at < records[i - 1].at))) throw new Error("invalid generation timestamps");
  if (JSON.stringify(tokens.map((r) => r.id)) !== JSON.stringify(end.result.tokens)) throw new Error("generation token evidence disagrees");
  return { inputIds: start.ids, result: end.result, observed: tokens.map((r) => r.id), drafted: tokens.map((r) => r.drafted),
    tokenTimes: tokens.map((r) => r.at - start.at), elapsedMs: end.at - start.at };
}

function checkOperation(r) {
  if (!operations.has(r.operation) || !finite(r.durationMs) || r.durationMs > r.at ||
      !Number.isSafeInteger(r.pos) || r.pos < 0 || !Number.isSafeInteger(r.n) || r.n < 1) {
    throw new Error("invalid host operation");
  }
}

// Inclusive operations may overlap (verify includes trunk and sampling). Never
// add these categories to a traversal RTT or claim their sum is request time.
export function operationSummary(capture) {
  const records = capture.records.filter((r) => r.phase === "operation");
  for (const r of records) checkOperation(r);
  return Object.fromEntries([...operations].map((name) => [name,
    summarize(records.filter((r) => r.operation === name).map((r) => r.durationMs))]));
}

export function correlateTiming(captures, host, workers) {
  const first = captures[host];
  if (!first?.attempt) throw new Error("missing host capture");
  if (!(typeof first.epoch === "string" && first.epoch) && !(Number.isSafeInteger(first.epoch) && first.epoch >= 0)) {
    throw new Error("missing chain epoch");
  }
  if (new Set([host, ...workers]).size !== workers.length + 1) throw new Error("duplicate device");
  for (const device of [host, ...workers]) {
    const c = captures[device];
    if (!c || c.attempt !== first.attempt || c.epoch !== first.epoch) throw new Error("capture identity mismatch");
    if (c.dropped !== 0) throw new Error("timing records were dropped");
    for (const [i, r] of c.records.entries()) {
      if (r.sequence !== i || !finite(r.at)) throw new Error("invalid timing sequence");
    }
  }
  const laps = [], pending = new Map();
  for (const r of first.records) {
    if (r.phase === "operation") { checkOperation(r); continue; }
    if (r.phase === "generation") { checkGeneration(r); continue; }
    const key = identity(r);
    if (r.phase === "dispatch") {
      if (pending.has(key)) throw new Error("ambiguous concurrent frame identity");
      const lap = { frame: { kind: r.kind, pos: r.pos, n: r.n, spec: r.spec }, dispatchAt: r.at,
        roundTripMs: null, workers: [] };
      pending.set(key, lap); laps.push(lap);
    } else if (r.phase === "return") {
      const lap = pending.get(key);
      if (!lap || r.at < lap.dispatchAt) throw new Error("unmatched frame return");
      lap.roundTripMs = r.at - lap.dispatchAt;
      pending.delete(key);
    } else throw new Error("unexpected host timing phase");
  }
  if (pending.size) throw new Error("incomplete frame traversal");
  if (!workers.length && laps.length) throw new Error("frames without workers");
  for (const device of workers) {
    const records = captures[device].records;
    if (records.length !== laps.length) throw new Error("worker frame count mismatch");
    records.forEach((r, i) => {
      if (r.phase !== "worker" || r.status !== "forwarded" || identity(r) !== identity(laps[i].frame)) {
        throw new Error("worker frame order or status mismatch");
      }
      if (![r.queueMs, r.serviceMs, r.engineMs].every(finite) || r.engineMs > r.serviceMs) {
        throw new Error("invalid worker durations");
      }
      laps[i].workers.push({ device, queueMs: r.queueMs, serviceMs: r.serviceMs, engineMs: r.engineMs });
    });
  }
  return laps.map(({ dispatchAt, ...lap }) => lap);
}

export function deliveryRate(times) {
  if (!times.every(finite) || times.some((t, i) => i && t < times[i - 1])) throw new Error("invalid token timestamps");
  const elapsed = times.at(-1) - times[0];
  return times.length >= 2 && elapsed > 0 ? 1000 * (times.length - 1) / elapsed : null;
}

export function summarize(values) {
  if (!values.every(finite)) throw new Error("invalid metric sample");
  if (!values.length) return { n: 0, median: null, min: null, max: null };
  const sorted = [...values].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
  return { n: sorted.length, median: sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2,
    min: sorted[0], max: sorted.at(-1) };
}
