import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawn } from "node:child_process";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const cli = path.join(root, "benchmarks/bench_room.js");
// This process exercises the CLI transport and evidence handling, not inference.
const fixture = `
import readline from 'node:readline';
import fs from 'node:fs';
const [mode, closed] = process.argv.slice(2);
if (mode === 'bad-protocol' || mode === 'log-failure') {
 fs.writeFileSync(closed + '.pid', String(process.pid));
 setTimeout(() => { fs.writeFileSync(closed, 'expired'); process.exit(2); }, 5000);
}
const reply = (id, result) => console.log('@pooled-bench ' + JSON.stringify({id, result}));
const snapshot = () => ({model:'qwen3-1.7b', ready:true, range:[0,28], layers:{R:[0,28]}, chain:[],
 context:2048, settings:{sampling:'exact'}, options:{}, gpuErrors:[]});
if (mode === 'startup-exit') process.exit(3);
if (mode === 'bad-protocol') {
 console.log('@pooled-bench {');
 await new Promise(() => {});
}
console.log('engine diagnostics before readiness');
reply('ready', {inventory: mode === 'client' ? {alias:'C',os:'test',runtime:'test',transport:'Bridge API v1',network:'fixture'} : {alias:'R',os:'test',runtime:'test',backend:'fixture',adapter:{},features:[],limits:{}}});
let attempt, epoch, prepared, ended=false;
for await (const line of readline.createInterface({input:process.stdin})) {
 const c=JSON.parse(line);
 if (c.action === 'close') { fs.writeFileSync(closed, 'closed'); reply(c.id, null); if (mode === 'close-exit') process.exitCode = 3; break; }
 if (c.action === 'snapshot') reply(c.id, snapshot());
 if (c.action === 'begin') { attempt=c.attempt; epoch=c.epoch; reply(c.id, null); }
 if (c.action === 'end') { ended=true; reply(c.id, {attempt, epoch, records:[], dropped:0}); }
 if (c.action === 'prepareClient') { prepared={before:snapshot(),inputIds:[1,2]}; ended=false; reply(c.id, prepared); }
 if (c.action === 'pairClient') {
  if (!ended || c.capture.attempt !== attempt || c.prepared.inputIds.length !== 2 || c.client.status !== 'pass') throw Error('bad pairing order');
  reply(c.id, {inputIds:[1,2],observed:[7,8],tokenTimes:[10,20],elapsedMs:25,
   result:{tokens:[7,8],reason:'max',tPre:5,tps:90,reused:0},clientTtftMs:15,clientElapsedMs:30,clientDelivery:{tps:100}});
 }
 if (c.action === 'run' && mode === 'client') reply(c.id, {status:'pass',ttftMs:15,events:[]});
 if (c.action === 'run' && mode !== 'timeout' && mode !== 'client') reply(c.id, {...c.request, status:'pass',
  before:snapshot(), after:snapshot(), sampling:'exact', inputIds:[1,2], observed:[7,8],
  ttftMs:10, elapsedMs:20, deliveryTps:100, result:{tokens:[7,8], reason:'max', tPre:5, tps:90, reused:0}});
}
`;

function execute(args, nodeArgs = []) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [...nodeArgs, cli, ...args], { cwd: root, stdio: ["ignore", "pipe", "pipe"] });
    let output = "";
    for (const stream of [child.stdout, child.stderr]) stream.on("data", (s) => { output += s; });
    const timeout = setTimeout(() => { child.kill(); reject(new Error("CLI test exceeded 15 seconds")); }, 15000);
    child.on("error", (e) => { clearTimeout(timeout); reject(e); });
    child.on("close", (code) => { clearTimeout(timeout); resolve({ code, output }); });
  });
}

async function setup(t, mode) {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "pooled-bench-cli-"));
  t.after(() => fs.rm(dir, { recursive: true, force: true }));
  const json = (file) => fs.readFile(path.join(dir, file), "utf8").then(JSON.parse);
  const save = (file, value) => fs.writeFile(path.join(dir, file), JSON.stringify(value));
  const manifest = { source: { commit: "a".repeat(40), treeSha256: "b".repeat(64) }, harnessSha256: "c".repeat(64),
    model: { name: "qwen3-1.7b", ...Object.fromEntries(["gguf", "config", "tokenizer"].map((p) => [p, { sha256: "d".repeat(64), revision: "test" }])) },
    devices: [{ alias: "R", os: "test", runtime: "test", backend: "fixture", adapter: {}, features: [], limits: {} }],
    cache: { weights: "test", shaders: "test", source: "test" } };
  await save("manifest.json", manifest);
  await fs.writeFile(path.join(dir, "device.mjs"), fixture);
  const config = { run: "cli", host: "R", manifest: path.join(dir, "manifest.json"), prompts: path.join(root, "docs/bench-log.md"),
    output: path.join(dir, "records"), repeats: 1, maxNew: 2, mode: "plain", epoch: 1, timeoutMs: 1000,
    devices: { R: { kind: "node", command: process.execPath, args: [path.join(dir, "device.mjs"), mode, path.join(dir, "closed")] } } };
  await save("config.json", config);
  return { dir, json, run: (nodeArgs) => execute(["run", path.join(dir, "config.json")], nodeArgs) };
}

test("benchmark CLI: malformed protocol terminates the live child", async (t) => {
  const f = await setup(t, "bad-protocol"), result = await f.run();
  assert.equal(result.code, 1, result.output);
  await assertChildTerminated(f.dir);
  assert.match((await f.json("records/runner-error.json")).error, /JSON|position|property/);
});

test("benchmark CLI: log write failure is handled and terminates the child", async (t) => {
  const f = await setup(t, "log-failure"), hook = path.join(f.dir, "fail-log.mjs");
  await fs.writeFile(hook, `import fs from 'node:fs/promises'; import path from 'node:path';
const open = fs.open;
fs.open = async (...args) => {
 const file = await open(...args);
 if (path.basename(args[0]) === 'R.log') file.write = async () => { throw Object.assign(Error('injected log failure'), {code:'ENOSPC'}); };
 return file;
};`);
  const result = await f.run(["--import", pathToFileURL(hook).href]);
  assert.equal(result.code, 1, result.output);
  await assertChildTerminated(f.dir);
  assert.doesNotMatch(result.output, /UnhandledPromiseRejection/);
  const error = await f.json("records/runner-error.json").catch(() => f.json("records/cleanup-errors.json"));
  assert.match(JSON.stringify(error), /injected log failure/);
});

async function assertChildTerminated(dir) {
  const pid = Number(await fs.readFile(path.join(dir, "closed.pid"), "utf8"));
  assert.throws(() => process.kill(pid, 0), { code: "ESRCH" });
  const marker = await fs.readFile(path.join(dir, "closed"), "utf8").catch((e) => {
    if (e.code !== "ENOENT") throw e;
    return null;
  });
  assert.notEqual(marker, "expired", "the child must not survive until the fixture watchdog");
}

test("benchmark CLI: real stdio transport persists a series, regenerates reports and closes its child", async (t) => {
  const f = await setup(t, "pass"), result = await f.run();
  assert.equal(result.code, 0, result.output);
  assert.deepEqual(await f.json("records/series.json"), { completed: 7, planned: 7, stopped: null });
  assert.equal((await f.json("records/report.json")).length, 3);
  assert.equal(await fs.readFile(path.join(f.dir, "closed"), "utf8"), "closed");
  assert.match(await fs.readFile(path.join(f.dir, "records/R.log"), "utf8"), /engine diagnostics/);
  const regenerated = await execute(["report", path.join(f.dir, "records"), path.join(f.dir, "again")]);
  assert.equal(regenerated.code, 0, regenerated.output);
  assert.deepEqual(await f.json("again.json"), await f.json("records/report.json"));
  const original = await fs.readFile(path.join(f.dir, "records/record-cli-001.json"), "utf8");
  assert.notEqual((await f.run()).code, 0); // an existing run must never be overwritten
  assert.equal(await fs.readFile(path.join(f.dir, "records/record-cli-001.json"), "utf8"), original);
});

test("benchmark CLI: request timeout preserves a failed attempt and closes the device", async (t) => {
  const f = await setup(t, "timeout"), result = await f.run();
  assert.equal(result.code, 1, result.output);
  const record = await f.json("records/record-cli-001.json");
  assert.equal(record.status, "incomplete");
  assert.equal(record.sample, null);
  assert.equal(record.hashes.output, null);
  assert.match(record.errors.find((e) => e.action === "run").message, /timed out/);
  assert.deepEqual(await f.json("records/series.json"), { completed: 1, planned: 7, stopped: "cli-001" });
  assert.equal(await fs.readFile(path.join(f.dir, "closed"), "utf8"), "closed");
});

test("benchmark CLI: API client series remains separate and closes both processes", async (t) => {
  const f = await setup(t, "pass"), config = await f.json("config.json"), manifest = await f.json("manifest.json");
  manifest.client = { alias: "C", os: "test", runtime: "test", transport: "Bridge API v1", network: "fixture" };
  config.client = { kind: "node", command: process.execPath,
    args: [path.join(f.dir, "device.mjs"), "client", path.join(f.dir, "client-closed")] };
  await fs.writeFile(path.join(f.dir, "manifest.json"), JSON.stringify(manifest));
  await fs.writeFile(path.join(f.dir, "config.json"), JSON.stringify(config));
  const result = await f.run(); assert.equal(result.code, 0, result.output);
  assert.deepEqual(await f.json("records/series.json"), { completed: 7, planned: 7, stopped: null });
  const report = await f.json("records/report.json"); assert.equal(report[0].requestPath, "API client");
  assert.equal(report[0].clientTtftMs.median, 15); assert.equal(report[0].ttftMs.median, 10);
  assert.equal(await fs.readFile(path.join(f.dir, "closed"), "utf8"), "closed");
  assert.equal(await fs.readFile(path.join(f.dir, "client-closed"), "utf8"), "closed");
  assert.deepEqual((await f.json("records/run.json")).client, { kind: "node" });
});

test("benchmark CLI: early device exit is a retained setup failure, not a measurement", async (t) => {
  const f = await setup(t, "startup-exit"), result = await f.run();
  assert.equal(result.code, 1, result.output);
  assert.match((await f.json("records/runner-error.json")).error, /device process exited \(3\)/);
  assert.equal((await fs.readdir(path.join(f.dir, "records"))).filter((s) => s.startsWith("record-")).length, 0);
});

test("benchmark CLI: nonzero device exit during cleanup fails an otherwise successful series", async (t) => {
  const f = await setup(t, "close-exit"), result = await f.run();
  assert.equal(result.code, 1, result.output);
  assert.deepEqual(await f.json("records/series.json"), { completed: 7, planned: 7, stopped: null });
  assert.match(JSON.stringify(await f.json("records/cleanup-errors.json")), /device cleanup exited \(3\)/);
});

test("benchmark CLI: external resources are attached to records without exposing command arguments", async (t) => {
  const f = await setup(t, "pass"), config = await f.json("config.json");
  config.resources = { R: { kind: "nvidia", command: process.execPath,
    args: ["-e", "console.log('fixture, test, 40, 5, 10, 100, 20')", "--"] } };
  await fs.writeFile(path.join(f.dir, "config.json"), JSON.stringify(config));
  const result = await f.run(); assert.equal(result.code, 0, result.output);
  const record = await f.json("records/record-cli-001.json"), resources = record.observations.resources;
  assert.equal(resources.attempt, record.attempt); assert.equal(resources.status, "pass");
  assert.equal(resources.samples[0].phase, "before"); assert.equal(resources.samples.at(-1).phase, "after");
  assert.equal(resources.samples[0].value.metrics.gpuTemperatureC, 40);
  assert.equal(record.observations.resourcesReason, null);
  assert.deepEqual(resources.configuration.sources, { R: { kind: "nvidia", timeoutMs: 5000 } });
  assert.ok(!JSON.stringify(resources.configuration).includes("console.log"));
  assert.deepEqual((await f.json("records/run.json")).resources, { R: { kind: "nvidia" } });
  assert.equal((await f.json("records/report.json"))[0].resourceSampling.intervalMs, 1000);
});

test("benchmark CLI: conflicting or missing Node inventory stops before requests and closes the child", async (t) => {
  for (const key of ["alias", "os", "runtime", "backend", "adapter", "limits", "features"]) {
    const f = await setup(t, "pass"), device = path.join(f.dir, "device.mjs");
    // Change the observed readiness, leaving the declared manifest intact.
    const source = await fs.readFile(device, "utf8");
    const inventory = { alias: "R", os: "test", runtime: "test", backend: "fixture", adapter: {}, features: [], limits: {} };
    inventory[key] = key === "features" ? ["unexpected"] : key === "adapter" || key === "limits" ? { unexpected: 1 } : "different";
    await fs.writeFile(device, source.replace("{alias:'R',os:'test',runtime:'test',backend:'fixture',adapter:{},features:[],limits:{}}", JSON.stringify(inventory)));
    const result = await f.run(); assert.equal(result.code, 1, result.output);
    assert.match((await f.json("records/runner-error.json")).error, new RegExp("identity differs from manifest: " + key));
    assert.equal((await fs.readdir(path.join(f.dir, "records"))).filter((s) => s.startsWith("record-")).length, 0);
    assert.deepEqual((await f.json("records/R-ready.json")).inventory, inventory);
    assert.equal(await fs.readFile(path.join(f.dir, "closed"), "utf8"), "closed");
  }
  const f = await setup(t, "pass"), device = path.join(f.dir, "device.mjs");
  await fs.writeFile(device, (await fs.readFile(device, "utf8")).replace("{alias:'R',os:'test',runtime:'test',backend:'fixture',adapter:{},features:[],limits:{}}", "{}"));
  assert.equal((await f.run()).code, 1);
  assert.match((await f.json("records/runner-error.json")).error, /identity differs from manifest: alias/);
});

test("benchmark CLI: feature and object key ordering do not change device identity", async (t) => {
  const f = await setup(t, "pass"), manifest = await f.json("manifest.json"), device = path.join(f.dir, "device.mjs");
  manifest.devices[0].features = ["b", "a"];
  manifest.devices[0].adapter = { vendor: "test", device: "fixture" };
  await fs.writeFile(path.join(f.dir, "manifest.json"), JSON.stringify(manifest));
  await fs.writeFile(device, (await fs.readFile(device, "utf8")).replace("adapter:{},features:[]", "adapter:{device:'fixture',vendor:'test'},features:['a','b']"));
  const result = await f.run(); assert.equal(result.code, 0, result.output);
  assert.deepEqual(await f.json("records/series.json"), { completed: 7, planned: 7, stopped: null });
});

test("benchmark CLI: undeclared or unconfigured devices fail before launching a child", async (t) => {
  for (const extra of [true, false]) {
    const f = await setup(t, "pass"), manifest = await f.json("manifest.json");
    if (extra) manifest.devices.push({ ...manifest.devices[0], alias: "G1" });
    else manifest.devices[0].alias = "G1";
    await fs.writeFile(path.join(f.dir, "manifest.json"), JSON.stringify(manifest));
    assert.equal((await f.run()).code, 1);
    assert.match((await f.json("records/runner-error.json")).error, /configured devices differ from manifest/);
    assert.ok(!(await fs.readdir(path.join(f.dir, "records"))).includes("R.log"));
  }
});

test("benchmark CLI: wrong model is retained as an incomplete attempt, never a passing row", async (t) => {
  const f = await setup(t, "pass"), manifest = await f.json("manifest.json");
  manifest.model.name = "different-model";
  await fs.writeFile(path.join(f.dir, "manifest.json"), JSON.stringify(manifest));
  const result = await f.run(); assert.equal(result.code, 1, result.output);
  const record = await f.json("records/record-cli-001.json");
  assert.equal(record.status, "incomplete"); assert.equal(record.sample, null);
  assert.equal(record.observations.before.R.model, "qwen3-1.7b");
  assert.ok(record.errors.some((e) => e.action === "identity" && e.message === "model differs from manifest"));
  assert.deepEqual(await f.json("records/report.json"), []);
  assert.deepEqual(await f.json("records/series.json"), { completed: 1, planned: 7, stopped: "cli-001" });
  assert.equal(await fs.readFile(path.join(f.dir, "closed"), "utf8"), "closed");
});

test("benchmark CLI: an unconfigured chain worker retains a setup error and closes the child", async (t) => {
  const f = await setup(t, "pass"), device = path.join(f.dir, "device.mjs");
  await fs.writeFile(device, (await fs.readFile(device, "utf8")).replace("chain:[]", "chain:['G1']"));
  const result = await f.run(); assert.equal(result.code, 1, result.output);
  assert.match((await f.json("records/runner-error.json")).error, /worker missing from configured devices/);
  assert.equal((await fs.readdir(path.join(f.dir, "records"))).filter((s) => s.startsWith("record-")).length, 0);
  assert.equal(await fs.readFile(path.join(f.dir, "closed"), "utf8"), "closed");
});
