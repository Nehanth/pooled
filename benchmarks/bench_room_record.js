import { correlateTiming, operationSummary } from "./bench_room_report.js";

export const RECORD_VERSION = 1;
const hash = /^[a-f0-9]{64}$/;
const named = (s) => typeof s === "string" && /^[A-Za-z0-9][A-Za-z0-9._+-]*$/.test(s);

export async function sha256(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((n) => n.toString(16).padStart(2, "0")).join("");
}

// Canonical token hash: unsigned 32-bit little-endian IDs, with no header.
export function tokenBytes(ids) {
  if (!Array.isArray(ids) || !ids.every((n) => Number.isSafeInteger(n) && n >= 0 && n <= 0xffffffff)) {
    throw new Error("invalid token IDs");
  }
  const bytes = new Uint8Array(ids.length * 4), view = new DataView(bytes.buffer);
  ids.forEach((n, i) => view.setUint32(i * 4, n, true));
  return bytes;
}

export function checkManifest(m) {
  if (!m || !/^[a-f0-9]{40}$/.test(m.source?.commit) || !hash.test(m.source?.treeSha256) || !hash.test(m.harnessSha256)) {
    throw new Error("missing source/harness identity");
  }
  if (!named(m.model?.name)) throw new Error("missing model name");
  for (const part of ["gguf", "config", "tokenizer"]) {
    if (!hash.test(m.model[part]?.sha256) || !m.model[part]?.revision) throw new Error(`missing ${part} identity`);
  }
  if (!Array.isArray(m.devices) || !m.devices.length || new Set(m.devices.map((d) => d.alias)).size !== m.devices.length) {
    throw new Error("missing or duplicate devices");
  }
  for (const d of m.devices) {
    if (!named(d.alias) || !d.os || !d.runtime || !d.backend || !d.adapter || !d.limits || !Array.isArray(d.features)) {
      throw new Error("incomplete device identity");
    }
  }
  if (!m.cache?.weights || !m.cache?.shaders || !m.cache?.source) throw new Error("missing cache classification");
  if (m.client && (!named(m.client.alias) || !m.client.os || !m.client.runtime ||
      m.client.transport !== "Bridge API v1" || !m.client.network)) throw new Error("incomplete requesting-client identity");
}

// Keeps raw samples even when timing cannot be correlated. A failed trace is not
// an inference failure, but cannot contribute a per-hop timing breakdown.
export async function makeRecord({ run, attempt, trial, promptName, host, manifest, sample = null, request = null,
  captures = null, observations = {}, errors = [], failure = null }) {
  checkManifest(manifest);
  if (![run, attempt, promptName, host].every(named) || !["first", "warmup", "measured", "correctness", "overhead"].includes(trial)) {
    throw new Error("invalid run identity");
  }
  if (sample && !["pass", "failed", "non-equivalent"].includes(sample.status)) throw new Error("invalid generation sample");
  if (failure !== null && !["blocked", "incomplete", "failed", "non-equivalent"].includes(failure)) throw new Error("invalid control failure");
  if (!sample && (!failure || !errors.length)) throw new Error("missing generation sample");
  request ||= sample && { prompt: sample.prompt, mode: sample.mode, maxNew: sample.maxNew };
  if (typeof request?.prompt !== "string") throw new Error("missing request");
  const aliases = new Set(manifest.devices.map((d) => d.alias)), workers = sample?.before.chain ?? observations.before?.[host]?.chain ?? [];
  if (!aliases.has(host) || !Array.isArray(workers) || workers.some((w) => !aliases.has(w))) throw new Error("unidentified room participant");
  let timing = { status: "disabled", reason: "collector disabled", laps: null };
  if (captures !== null) {
    try {
      if (captures[host]?.attempt !== attempt) throw new Error("capture does not belong to this attempt");
      timing = { status: "pass", reason: null, laps: correlateTiming(captures, host, workers),
        operations: operationSummary(captures[host]) };
    } catch (e) { timing = { status: "incomplete", reason: e.message, laps: null }; }
  }
  const output = sample?.result?.tokens ?? sample?.observed;
  const hashes = { encoding: "u32le", prompt: await sha256(new TextEncoder().encode(request.prompt)),
    input: sample ? await sha256(tokenBytes(sample.inputIds)) : null, output: sample ? await sha256(tokenBytes(output)) : null };
  const status = observations.gpuErrors?.length ? "failed" : failure || sample.status;
  return { version: RECORD_VERSION, run, attempt, trial, promptName, host, manifest, status,
    request, hashes, sample, captures, timing, observations, errors };
}

export function firstDifference(a, b) {
  const end = Math.max(a.length, b.length);
  for (let i = 0; i < end; i++) if (a[i] !== b[i]) return { index: i, expected: a[i] ?? null, actual: b[i] ?? null };
  return null;
}
