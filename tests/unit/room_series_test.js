import { collectAttempt, runSeries, standardPrompts } from "../../benchmarks/bench_room_series.js";
import { benchmarkRows, markdownTable } from "../../benchmarks/bench_room_table.js";
import { createTimingCollector } from "../../room/timing.js";

const eq = (a, b) => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${JSON.stringify(a)} != ${JSON.stringify(b)}`); };
const ok = (v) => { if (!v) throw new Error("assertion failed"); };
function fixture() {
  const capture = createTimingCollector(), calls = [], records = [];
  const snapshot = () => ({ model: "qwen3-1.7b", ready: true, role: "host", range: [0, 28], chain: [], layers: { R: [0, 28] },
    context: 2048, settings: { sampling: "focused" }, options: { lookup: false }, gpuErrors: [] });
  const device = { snapshot, begin: (id) => { calls.push("begin"); capture.begin(id); },
    end: () => { calls.push("end"); return { ...capture.end(), epoch: 1 }; },
    run: async (request) => { calls.push("run"); return { ...request, before: snapshot(), after: snapshot(), status: "pass", sampling: "exact",
      inputIds: [1, 2], observed: [7, 8], ttftMs: 30, elapsedMs: 50, deliveryTps: 100,
      result: { tokens: [7, 8], reason: "max", tPre: 20, tps: 80, reused: 0 } }; } };
  const manifest = { source: { commit: "a".repeat(40), treeSha256: "b".repeat(64) }, harnessSha256: "c".repeat(64),
    model: { name: "qwen3-1.7b", ...Object.fromEntries(["gguf", "config", "tokenizer"].map((p) => [p, { sha256: "d".repeat(64), revision: "pinned" }])) },
    devices: [{ alias: "R", os: "test", runtime: "test", backend: "test", adapter: {}, limits: {}, features: [] }],
    cache: { weights: "resident", shaders: "warm", source: "local" } };
  return { device, calls, records, options: { devices: { R: device }, host: "R", manifest, run: "test", epoch: 1,
    prompts: [{ name: "hello", text: "hello" }, { name: "meaning", text: "what is the meaning of life" }],
    write: (r) => { calls.push("write"); records.push(r); } } };
}

Deno.test("series: first request, one warm-up per prompt and measured repeats persist in order", async () => {
  const f = fixture(), result = await runSeries({ ...f.options, repeats: 2 });
  eq(result, { completed: 7, planned: 7, stopped: null });
  eq(f.records.map((r) => r.trial), ["first", "warmup", "measured", "measured", "warmup", "measured", "measured"]);
  eq(f.calls, Array(7).fill(["begin", "run", "end", "write"]).flat());
  eq(new Set(f.records.map((r) => r.attempt)).size, 7);
  const rows = benchmarkRows(f.records); eq(rows.length, 2); eq(rows.map((r) => r.passed), [2, 2]);
  eq(rows[0].ttftMs, { n: 2, median: 30, min: 30, max: 30 });
  ok(markdownTable(rows).includes("hello / plain (exact) / timing on"));
});

Deno.test("series: client request is prepared, captured and paired before writing", async () => {
  const f = fixture();
  f.options.manifest.client = { alias: "C", os: "test", runtime: "test", transport: "Bridge API v1", network: "fixture" };
  f.device.prepareClient = (request) => { f.calls.push("prepare"); return { before: f.device.snapshot(), inputIds: [1, 2], request }; };
  f.device.pairClient = (client, capture, prepared) => {
    f.calls.push("pair"); eq(capture.attempt, "test-001"); eq(client.status, "pass"); eq(prepared.inputIds, [1, 2]);
    return { inputIds: [1, 2], observed: [7, 8], tokenTimes: [10, 20], elapsedMs: 25,
      result: { tokens: [7, 8], reason: "stop", tPre: 5, tps: 90, reused: 0 }, clientTtftMs: 15, clientElapsedMs: 30,
      clientDelivery: { tokenTimes: [15, 25], tps: 100, reason: null } };
  };
  const client = { run() { f.calls.push("client"); return { status: "pass", ttftMs: 15 }; } };
  const r = await collectAttempt({ ...f.options, client, attempt: "test-001", trial: "measured", promptName: "hello",
    request: { prompt: "hello" } });
  eq(f.calls, ["begin", "prepare", "client", "end", "pair"]); eq(r.status, "pass");
  eq(r.sample.ttftMs, 10); eq(r.sample.clientTtftMs, 15); eq(r.sample.deliveryTps, 100);
  const row = benchmarkRows([r])[0]; eq(row.requestPath, "API client"); eq(row.clientTtftMs.median, 15);
  eq(row.clientDeliveryTps.median, 100); ok(markdownTable([row]).includes("Client TTFT"));
  const failed = await collectAttempt({ ...f.options, client: { run: () => ({ status: "failed", error: "link lost", events: [] }) },
    attempt: "test-002", trial: "measured", promptName: "hello", request: { prompt: "hello" } });
  eq(failed.status, "failed"); eq(failed.sample, null); eq(failed.hashes.output, null);
  eq(failed.observations.client.error, "link lost"); eq(f.calls.at(-1), "end");
});

Deno.test("series: control failure is retained, collectors end, no automatic retry", async () => {
  const f = fixture(); f.device.run = async () => { f.calls.push("run"); throw new Error("CDP disconnected"); };
  const result = await runSeries(f.options);
  eq(result.completed, 1); eq(result.stopped, "test-001"); eq(f.calls, ["begin", "run", "end", "write"]);
  const r = f.records[0]; eq(r.status, "incomplete"); eq(r.sample, null); eq(r.hashes.output, null);
  eq(r.request.prompt, "hello"); eq(r.errors[0].action, "run");
});

Deno.test("series: resource observations drain after inference failure without hiding it", async () => {
  const f = fixture();
  f.device.run = () => { f.calls.push("run"); throw new Error("client lost"); };
  const resources = { begin: (attempt) => { eq(attempt, "test-001"); f.calls.push("resource-begin"); },
    end: () => { f.calls.push("resource-end"); return { attempt: "test-001", status: "incomplete", dropped: 1, samples: [] }; } };
  await runSeries({ ...f.options, resources });
  eq(f.calls, ["resource-begin", "begin", "run", "end", "resource-end", "write"]);
  eq(f.records[0].status, "incomplete"); eq(f.records[0].sample, null);
  eq(f.records[0].observations.resources.dropped, 1); eq(f.records[0].errors[0].message, "client lost");
});

Deno.test("series: failed begin unwinds successful captures and a lost end preserves the sample", async () => {
  const f = fixture(), worker = { ...f.device, begin: () => { throw new Error("worker gone"); } };
  f.options.devices.G1 = worker; f.options.manifest.devices.push({ ...f.options.manifest.devices[0], alias: "G1" });
  await runSeries(f.options); eq(f.calls, ["begin", "end", "write"]); eq(f.records[0].status, "incomplete");
  const g = fixture(); g.device.end = () => { throw new Error("capture unavailable"); };
  await runSeries(g.options); eq(g.records[0].status, "incomplete"); eq(g.records[0].sample.result.tokens, [7, 8]);
  eq(g.records[0].timing.status, "incomplete");
});

Deno.test("series: GPU errors, room readiness and write failures stop the schedule", async () => {
  const f = fixture(), snap = f.device.snapshot;
  f.device.snapshot = () => ({ ...snap(), gpuErrors: ["GPU lost"] });
  await runSeries(f.options); eq(f.records[0].status, "failed"); eq(f.records[0].observations.gpuErrors.length, 1);
  const g = fixture(); g.device.snapshot = () => ({ ...snap(), ready: false });
  await runSeries(g.options); eq(g.records[0].status, "blocked"); eq(g.calls, ["write"]);
  const h = fixture(); let error;
  try { await runSeries({ ...h.options, write: () => { throw new Error("disk full"); } }); } catch (e) { error = e; }
  eq(error.message, "disk full"); eq(h.calls.filter((c) => c === "run").length, 1);
});

Deno.test("report: source/settings separation, failed counts, greedy divergence and duplicate rejection", async () => {
  const f = fixture(); await runSeries({ ...f.options, repeats: 2 });
  const a = f.records[2], b = structuredClone(a); b.attempt = "other"; b.hashes.output = "e".repeat(64);
  const failed = structuredClone(a); failed.attempt = "failed"; failed.status = "failed";
  const rows = benchmarkRows([a, b, failed]); eq(rows.length, 1); eq([rows[0].passed, rows[0].failed], [2, 1]);
  ok(markdownTable(rows).includes("greedy outputs differ"));
  b.manifest.source.treeSha256 = "f".repeat(64); eq(benchmarkRows([a, b]).length, 2);
  let error; try { benchmarkRows([a, a]); } catch (e) { error = e; } ok(error.message.includes("duplicate"));
});

Deno.test("standard prompts are read verbatim from the pinned bench-log", () => {
  const prompts = standardPrompts(Deno.readTextFileSync("docs/bench-log.md"));
  eq(prompts.map((p) => p.name), ["hello", "meaning", "japan"]);
  eq(prompts[1].text, "what is the meaning of life"); ok(prompts[2].text.endsWith("worth it at all."));
});

Deno.test("report: GPU feature order is immaterial but feature changes separate rows", async () => {
  const f = fixture(); await runSeries({ ...f.options, repeats: 1 });
  const a = f.records[2], b = structuredClone(a); b.attempt = "other";
  a.manifest.devices[0].features = ["subgroups", "shader-f16"];
  b.manifest.devices[0].features = ["shader-f16", "subgroups"];
  eq(benchmarkRows([a, b]).length, 1);
  eq(a.manifest.devices[0].features, ["subgroups", "shader-f16"]);
  b.manifest.devices[0].features = ["subgroups"];
  eq(benchmarkRows([a, b]).length, 2);
});

Deno.test("report: differing resource providers or timeout bounds separate rows", async () => {
  const f = fixture(); await runSeries({ ...f.options, repeats: 1 });
  const a = f.records[2], b = structuredClone(a); b.attempt = "other";
  a.observations.resources = { configuration: { intervalMs: 1000, capacity: 4096, devices: ["R"], sources: { R: { kind: "nvidia", timeoutMs: 5000 } } } };
  b.observations.resources = structuredClone(a.observations.resources);
  eq(benchmarkRows([a, b]).length, 1);
  b.observations.resources.configuration.sources.R.timeoutMs = 1000;
  eq(benchmarkRows([a, b]).length, 2);
  b.observations.resources.configuration.sources.R = { kind: "android", timeoutMs: 5000 };
  eq(benchmarkRows([a, b]).length, 2);
});

Deno.test("report: enabled speculation is distinct from observed verification", async () => {
  const f = fixture(); await runSeries({ ...f.options, repeats: 1 });
  const record = f.records[2]; record.request.mode = "defaults";
  record.sample.drafted = [false, false]; record.sample.result.acc = null; record.sample.result.copied = 0;
  let row = benchmarkRows([record])[0];
  eq(row.speculation, { observed: 0, inactive: 1, unknown: 0 });
  ok(markdownTable([row]).includes("no draft verification in 1/1"));
  record.sample.result.acc = 0; // rejected drafts still incurred a verification
  row = benchmarkRows([record])[0]; eq(row.speculation, { observed: 1, inactive: 0, unknown: 0 });
  delete record.sample.result.acc;
  row = benchmarkRows([record])[0]; eq(row.speculation, { observed: 0, inactive: 0, unknown: 1 });
});

Deno.test("series: mismatched host or loaded worker model prevents generation and retains evidence", async () => {
  for (const alias of ["R", "G1"]) {
    const f = fixture(), snapshot = f.device.snapshot;
    const wrong = () => ({ ...snapshot(), model: "different-model" });
    if (alias === "R") f.device.snapshot = wrong;
    else {
      f.options.devices.G1 = { ...f.device, snapshot: wrong };
      f.options.manifest.devices.push({ ...f.options.manifest.devices[0], alias: "G1" });
    }
    const result = await runSeries(f.options);
    eq(result.stopped, "test-001"); eq(f.calls, ["write"]);
    eq(f.records[0].status, "incomplete"); eq(f.records[0].sample, null);
    eq(f.records[0].observations.before[alias].model, "different-model");
    ok(f.records[0].errors.some((e) => e.device === alias && e.action === "identity"));
  }
});

Deno.test("series: unconfigured chain worker prevents generation even with timing disabled", async () => {
  const f = fixture(), snapshot = f.device.snapshot;
  f.device.snapshot = () => ({ ...snapshot(), chain: ["G1"] });
  let error;
  try { await runSeries({ ...f.options, timing: false }); } catch (e) { error = e; }
  eq(error.message, "worker missing from configured devices"); eq(f.calls, []); eq(f.records, []);
});

Deno.test("series: an unloaded guest does not claim a model identity", async () => {
  const f = fixture();
  f.options.devices.S = { snapshot: () => ({ model: null, loaded: false, range: null }), begin() {}, end: () => null };
  f.options.manifest.devices.push({ ...f.options.manifest.devices[0], alias: "S" });
  const result = await runSeries({ ...f.options, timing: false });
  eq(result.stopped, null); ok(f.records.every((r) => r.status === "pass"));
});
