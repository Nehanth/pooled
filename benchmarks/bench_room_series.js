import { checkManifest, makeRecord } from "./bench_room_record.js";
import { checkRequest } from "./bench_room_sample.js";
import { wireDifference } from "./bench_room_wire.js";
import { deliveryRate } from "./bench_room_report.js";

// Adapters may be local, browser/CDP, or a Node process. Their commands must be
// bounded by their transport; a timeout is recorded, never retried as a new run.
export async function collectAttempt({ devices, host, client = null, resources = null, manifest, run, attempt, trial, promptName, request, epoch, timing = true }) {
  checkManifest(manifest);
  request = checkRequest(request);
  if (client && (!timing || !manifest.client)) throw new Error("client measurements require timing and a client manifest");
  const aliases = Object.keys(devices), observations = { before: {}, after: {}, gpuErrors: [], wire: {},
    resources: null, resourcesReason: "external resource sampling not supplied",
    ttftBoundary: "model host after lock, before reset/tokenization; excludes automation IPC" };
  if (!devices[host] || aliases.length !== manifest.devices.length ||
      aliases.some((a) => !manifest.devices.some((d) => d.alias === a))) throw new Error("configured devices differ from manifest");
  let sample = null, failure = null, prepared = null, resourceStarted = false;
  const errors = [], begun = [], captures = timing ? {} : null;
  async function each(names, action, fn) {
    const results = await Promise.allSettled(names.map(fn));
    results.forEach((r, i) => { if (r.status === "rejected") errors.push({ device: names[i], action, message: String(r.reason?.message || r.reason) }); });
  }
  await each(aliases, "snapshot-before", async (a) => { observations.before[a] = await devices[a].snapshot(); });
  if (!errors.length && !observations.before[host].ready) {
    failure = "blocked"; errors.push({ device: host, action: "ready", message: "host is not ready" });
  }
  for (const a of aliases) {
    const before = observations.before[a];
    if (before?.gpuErrors?.length) errors.push({ device: a, action: "gpu", message: "GPU error reported before request" });
    if (before && (a === host || before.loaded || before.range?.[1] > before.range?.[0]) && before.model !== manifest.model.name) {
      errors.push({ device: a, action: "identity", message: "model differs from manifest" });
    }
  }
  if (observations.before[host]?.chain?.some((a) => !aliases.includes(a))) {
    throw new Error("worker missing from configured devices");
  }
  if (!errors.length && resources) {
    try { await resources.begin(attempt); resourceStarted = true; }
    catch (e) { errors.push({ action: "resource-begin", message: String(e?.message || e) }); }
  }
  if (!errors.length) {
    await each(aliases, "begin", async (a) => { await devices[a].begin(attempt, epoch); begun.push(a); });
    if (!errors.length) {
      try {
        if (client) {
          prepared = await devices[host].prepareClient(request);
          observations.client = await client.run(request);
          if (observations.client.status !== "pass") {
            failure = ["failed", "blocked", "non-equivalent"].includes(observations.client.status) ? observations.client.status : "incomplete";
            errors.push({ device: manifest.client.alias, action: "client-request", message: observations.client.error || "client request did not pass" });
          }
        } else sample = await devices[host].run(request);
      }
      catch (e) { errors.push({ device: host, action: "run", message: String(e?.message || e) }); }
    }
    // Every successful begin is ended, even when another device could not begin.
    await each(begun, "end", async (a) => { const c = await devices[a].end(); if (timing) captures[a] = c; });
    if (client && prepared && observations.client?.status === "pass") {
      try {
        const paired = await devices[host].pairClient(observations.client, captures[host], prepared);
        sample = { ...request, ...paired, status: "pass", error: null, invalid: null, thinking: false,
          before: prepared.before, after: await devices[host].snapshot(),
          sampling: request.mode === "defaults" ? prepared.before.settings.sampling : "exact",
          ttftMs: paired.tokenTimes[0] ?? null, deliveryTps: deliveryRate(paired.tokenTimes) };
        observations.ttftBoundary = "host generation entry to emitted token; separate client TTFT includes queueing and transport";
      } catch (e) { errors.push({ device: host, action: "pair-client", message: String(e?.message || e) }); }
    }
  }
  await each(aliases, "snapshot-after", async (a) => { observations.after[a] = await devices[a].snapshot(); });
  if (resourceStarted) {
    try { observations.resources = await resources.end(); observations.resourcesReason = null; }
    catch (e) { errors.push({ action: "resource-end", message: String(e?.message || e) }); }
  } else if (resources) observations.resourcesReason = "resource monitor did not start because setup failed";
  for (const a of aliases) {
    for (const message of new Set([...(observations.before[a]?.gpuErrors || []), ...(observations.after[a]?.gpuErrors || [])])) {
      observations.gpuErrors.push({ device: a, message });
    }
    const before = observations.before[a], after = observations.after[a];
    observations.wire[a] = wireDifference(before?.wire, after?.wire);
    if (observations.wire[a].status === "incomplete") errors.push({ device: a, action: "wire", message: observations.wire[a].reason });
    if (before && after && JSON.stringify([before.model, before.range, before.chain]) !== JSON.stringify([after.model, after.range, after.chain])) {
      errors.push({ device: a, action: "assignment", message: "device assignment changed during attempt" });
    }
  }
  if (errors.length) failure ||= "incomplete";
  return makeRecord({ run, attempt, trial, promptName, host, manifest, request, sample, captures, observations, errors, failure });
}

export function standardPrompts(markdown) {
  const prompts = [];
  for (const name of ["hello", "meaning", "japan"]) {
    const line = markdown.split("\n").find((l) => l.startsWith(`| \`${name}\` |`));
    const text = line?.split("|")[3]?.trim();
    if (!text?.startsWith("`") || !text.endsWith("`")) throw new Error(`missing standard prompt: ${name}`);
    prompts.push({ name, text: text.slice(1, -1) });
  }
  return prompts;
}

// Weight loading belongs to setup. This series keeps the model resident and
// resets inference-prefix state through the normal adapter for every request.
export async function runSeries({ prompts, repeats = 5, mode = "plain", maxNew = 400, write, onRecord = () => {}, ...options }) {
  const validId = (s) => typeof s === "string" && /^[A-Za-z0-9][A-Za-z0-9._+-]*$/.test(s);
  if (!validId(options.run) || !Number.isSafeInteger(repeats) || repeats < 1 || !prompts?.length ||
    prompts.some((p) => !validId(p.name)) || new Set(prompts.map((p) => p.name)).size !== prompts.length) {
    throw new Error("invalid repetition schedule");
  }
  for (const p of prompts) checkRequest({ prompt: p.text, mode, maxNew });
  const schedule = [{ ...prompts[0], trial: "first" }];
  for (const prompt of prompts) {
    schedule.push({ ...prompt, trial: "warmup" });
    for (let i = 0; i < repeats; i++) schedule.push({ ...prompt, trial: "measured" });
  }
  let count = 0;
  for (const p of schedule) {
    const attempt = `${options.run}-${String(++count).padStart(3, "0")}`;
    const record = await collectAttempt({ ...options, attempt, trial: p.trial, promptName: p.name, request: { prompt: p.text, mode, maxNew } });
    await write(record);   // durable evidence before another request
    await onRecord(record);
    // A broken room needs inspection; continuing could turn recovery into a speed result.
    if (record.status !== "pass" || (options.timing !== false && record.timing.status !== "pass")) {
      return { completed: count, planned: schedule.length, stopped: record.attempt };
    }
  }
  return { completed: count, planned: schedule.length, stopped: null };
}
