import { summarize } from "./bench_room_report.js";

function speculation(record) {
  const sample = record.sample, result = sample?.result;
  if (record.captures?.[record.host]?.records.some((r) => r.phase === "dispatch" && r.spec) ||
      result?.acc != null || result?.copied > 0 || sample?.drafted?.some(Boolean)) return "observed";
  return result && Object.hasOwn(result, "acc") && Object.hasOwn(result, "copied") && Array.isArray(sample.drafted)
    ? "inactive" : "unknown";
}

// Group conservatively: any change in source, model, devices, cache, assignment,
// sampling or observer mode starts a new row. First/warm-up runs stay in raw data.
export function benchmarkRows(records) {
  const groups = new Map(), attempts = new Set();
  for (const r of records) {
    if (r.version !== 1) throw new Error("unsupported record version");
    const id = JSON.stringify([r.run, r.attempt]);
    if (attempts.has(id)) throw new Error("duplicate attempt");
    attempts.add(id);
    if (r.trial !== "measured") continue;
    const s = r.sample, before = s?.before || r.observations.before?.[r.host], request = r.request || s;
    const manifest = { ...r.manifest, devices: r.manifest.devices.map((d) => ({ ...d, features: [...d.features].sort() })) };
    const condition = { manifest, host: r.host, chain: before?.chain ?? null, layers: before?.layers ?? null,
      context: before?.context ?? null, options: before?.options ?? null,
      sampling: s?.sampling ?? (request.mode === "defaults" ? before?.settings.sampling : "exact"),
      mode: request.mode, maxNew: request.maxNew, prompt: r.promptName, promptHash: r.hashes.prompt, timing: r.captures !== null };
    condition.requestPath = r.observations.client ? "API client" : "host invocation";
    condition.hostTtftBoundary = r.observations.ttftBoundary;
    condition.resourceSampling = r.observations.resources?.configuration ?? null;
    const key = JSON.stringify(condition);
    if (!groups.has(key)) groups.set(key, { condition, records: [] });
    groups.get(key).records.push(r);
  }
  return [...groups.values()].map(({ condition, records: group }) => {
    const passed = group.filter((r) => r.status === "pass"), metric = (fn) => summarize(passed.map(fn).filter((v) => v != null));
    return { ...condition, attempts: group.length, passed: passed.length, failed: group.length - passed.length,
      inputHashes: [...new Set(passed.map((r) => r.hashes.input))], outputHashes: [...new Set(passed.map((r) => r.hashes.output))],
      timingFailures: group.filter((r) => r.timing.status === "incomplete").length,
      resourceFailures: group.filter((r) => r.observations.resources && r.observations.resources.status !== "pass").length,
      speculation: Object.fromEntries(["observed", "inactive", "unknown"].map((state) =>
        [state, passed.filter((r) => speculation(r) === state).length])),
      ttftMs: metric((r) => r.sample.ttftMs), prefillMs: metric((r) => r.sample.result.tPre),
      clientTtftMs: metric((r) => r.sample.clientTtftMs), clientElapsedMs: metric((r) => r.sample.clientElapsedMs),
      clientDeliveryTps: metric((r) => r.sample.clientDelivery?.tps),
      upstreamTps: metric((r) => r.sample.result.tps), deliveryTps: metric((r) => r.sample.deliveryTps),
      elapsedMs: metric((r) => r.sample.elapsedMs), outputTokens: metric((r) => r.sample.result.tokens.length),
      stopReasons: Object.fromEntries([...new Set(group.map((r) => r.sample?.result?.reason || r.status))].map((reason) =>
        [reason, group.filter((r) => (r.sample?.result?.reason || r.status) === reason).length])) };
  });
}

const cell = (s) => String(s).replaceAll("|", "\\|").replaceAll("\n", " ");
const range = (m) => m.n ? `${m.median.toFixed(2)} (${m.min.toFixed(2)}–${m.max.toFixed(2)})` : "—";
export function markdownTable(rows) {
  const lines = ["| Devices / layers | Source | Model SHA-256 | Prompt / mode | Pass / fail | Host TTFT ms | Client TTFT ms | Host delivery tok/s | Client delivery tok/s | Upstream tok/s | Output tokens | Checks |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"];
  for (const r of rows) lines.push("| " + [JSON.stringify(r.layers), `${r.manifest.source.commit} / ${r.manifest.source.treeSha256}`,
    r.manifest.model.gguf.sha256, `${r.prompt} / ${r.mode} (${r.sampling ?? "unknown sampling"})${r.timing ? " / timing on" : " / timing off"} / ${r.requestPath}`,
    `${r.passed} / ${r.failed}`, range(r.ttftMs), range(r.clientTtftMs), range(r.deliveryTps), range(r.clientDeliveryTps), range(r.upstreamTps), range(r.outputTokens),
    [r.inputHashes.length > 1 ? "inputs differ" : "", r.mode !== "defaults" && r.outputHashes.length > 1 ? "greedy outputs differ" : "",
      r.timingFailures ? `${r.timingFailures} incomplete traces` : "",
      r.resourceFailures ? `${r.resourceFailures} incomplete resource captures` : "",
      r.speculation.observed ? `draft verification observed in ${r.speculation.observed}/${r.passed}` : "",
      r.mode !== "plain" && r.speculation.inactive ? `no draft verification in ${r.speculation.inactive}/${r.passed}` : "",
      r.speculation.unknown ? `speculation activity unknown in ${r.speculation.unknown}/${r.passed}` : ""].filter(Boolean).join("; ") || "—"].map(cell).join(" | ") + " |");
  return lines.join("\n") + "\n\nValues are median (min–max); counts are in the JSON report. Host TTFT excludes automation IPC; client TTFT includes queueing and transport. Delivery rates use exact token counts, with tokens sharing a text chunk sharing its arrival time.\n";
}
