// Prepared benchmark rooms only; connection settings and raw device logs are local.
// node benchmarks/bench_room.js run config.json
// node benchmarks/bench_room.js report records-directory output-prefix
import fs from "node:fs/promises";
import path from "node:path";
import readline from "node:readline";
import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { isDeepStrictEqual } from "node:util";
import { runSeries, standardPrompts } from "./bench_room_series.js";
import { benchmarkRows, markdownTable } from "./bench_room_table.js";
import { checkManifest } from "./bench_room_record.js";
import { browserReady, configureBrowser } from "./bench_room_browser.js";
import { resourceReader, resourceMonitor } from "./bench_room_resources.js";

const json = async (file) => JSON.parse(await fs.readFile(file, "utf8"));
const save = (file, value) => fs.writeFile(file, JSON.stringify(value, null, 2) + "\n", { flag: "wx" });
function bounded(promise, ms, label) {
  let timer;
  return Promise.race([promise, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`${label} timed out`)), ms); })])
    .finally(() => clearTimeout(timer));
}

async function browserDevice(spec, timeoutMs) {
  if (!spec.dedicated || !spec.url || new URL(spec.url).searchParams.get("bench") !== "1") throw new Error("explicit dedicated bench URL required");
  const require = createRequire(import.meta.url), { chromium } = require("playwright");
  const browser = await chromium.connectOverCDP(spec.cdp, { timeout: timeoutMs });
  const pages = browser.contexts().flatMap((c) => c.pages()).filter((p) => p.url() === spec.url);
  if (pages.length !== 1) { await browser.close(); throw new Error("benchmark URL must identify exactly one tab"); }
  const page = pages[0];
  const call = (method, args = []) => bounded(page.evaluate(async ({ method, args }) => {
    const b = window.__pooledBench;
    if (!b) throw new Error("benchmark adapter not loaded");
    const result = await b[method](...args);
    return method === "snapshot" ? { ...result, gpuErrors: window.__benchGpuErrors ?? null,
      gpuErrorCoverage: window.__benchGpuErrors ? "installed by launcher before navigation" : "unavailable" } : result;
  }, { method, args }), timeoutMs, method);
  return { snapshot: () => call("snapshot"), begin: (...a) => call("begin", a), end: () => call("end"), run: (r) => call("run", [r]),
    prepareClient: (r) => call("prepareClient", [r]), pairClient: (...a) => call("pairClient", a),
    async start({ model, split } = {}) {
      await page.evaluate(configureBrowser, { model, split });
      const loaded = (await call("snapshot")).loaded;
      await page.locator(loaded ? "#ai-redeal" : "#ai-start").click({ timeout: timeoutMs });
      await page.waitForFunction(browserReady, null, { timeout: timeoutMs });
    },
    close: () => browser.close() };   // CDP connection only; the caller owns the dedicated tab
}

async function nodeDevice(spec, timeoutMs, logFile) {
  const log = await fs.open(logFile, "wx");
  const child = spawn(spec.command, spec.args, { cwd: spec.cwd, stdio: ["pipe", "pipe", "pipe"], windowsHide: true });
  const pending = new Map(); let serial = 0, terminal = null, exited = false, logError = null;
  let writes = Promise.resolve();
  const append = (s) => { writes = writes.then(() => logError ? undefined : log.write(s)).catch((e) => { logError ??= e; fail(e); }); };
  child.stderr.on("data", (b) => append(b));
  const fail = (error) => { terminal ??= error; for (const p of pending.values()) p.reject(terminal); pending.clear(); };
  child.stdin.on("error", fail);
  child.on("error", fail); child.on("exit", (code, signal) => fail(new Error(`device process exited (${code ?? signal})`)));
  const closed = new Promise((resolve) => child.on("close", (code, signal) => { exited = true; resolve({ code, signal }); }));
  const lines = readline.createInterface({ input: child.stdout });
  lines.on("line", (line) => {
    if (!line.startsWith("@pooled-bench ")) { append(line + "\n"); return; }
    try {
      const r = JSON.parse(line.slice(14)), p = pending.get(r.id);
      if (!p) return;
      pending.delete(r.id); r.error ? p.reject(new Error(r.error)) : p.resolve(r.result);
    } catch (e) { fail(e); }
  });
  const wait = (id) => terminal ? Promise.reject(terminal) : bounded(new Promise((resolve, reject) => pending.set(id, { resolve, reject })), timeoutMs, id)
    .finally(() => pending.delete(id));
  const call = (action, fields = {}) => {
    const id = String(++serial), reply = wait(id);
    if (!terminal) child.stdin.write(JSON.stringify({ id, action, ...fields }) + "\n", (e) => { if (e) fail(e); });
    return reply;
  };
  async function close() {
    let failure = terminal;
    try {
      if (!exited) {
        if (!terminal) {
          try { await bounded(call("close"), 5000, "device close"); child.stdin.end(); }
          catch (e) { failure ??= e; child.kill(); }
        } else child.kill();
        try { await bounded(closed, 5000, "device exit"); }
        catch (e) { failure ??= e; child.kill("SIGKILL"); await bounded(closed, 5000, "device termination"); }
      }
      const { code, signal } = await closed;
      if (code !== 0) failure ??= new Error(`device cleanup exited (${code ?? signal})`);
      if (failure) throw failure;
    } finally {
      lines.close();
      try { await writes; if (logError) throw logError; }
      finally { await log.close(); }
    }
  }
  try {
    const ready = await wait("ready");
    return { ready, snapshot: () => call("snapshot"), begin: (attempt, epoch) => call("begin", { attempt, epoch }),
      end: () => call("end"), run: (request) => call("run", { request }), start: () => call("start"),
      prepareClient: (request) => call("prepareClient", { request }),
      pairClient: (client, capture, prepared) => call("pairClient", { client, capture, prepared }), close };
  } catch (e) { await close(); throw e; }
}

async function report(directory, output) {
  const files = (await fs.readdir(directory)).filter((f) => f.startsWith("record-") && f.endsWith(".json")).sort();
  if (!files.length) throw new Error("no benchmark records");
  const rows = benchmarkRows(await Promise.all(files.map((f) => json(path.join(directory, f)))));
  await save(output + ".json", rows);
  await fs.writeFile(output + ".md", markdownTable(rows), { flag: "wx" });
  console.log(`${rows.length} rows from ${files.length} raw records`);
}

async function run(file) {
  const config = await json(file), manifest = await json(config.manifest), devices = {};
  let client = null;
  checkManifest(manifest);
  const timeoutMs = config.timeoutMs ?? 600000;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1) throw new Error("invalid command timeout");
  await fs.mkdir(config.output, { recursive: true });
  await save(path.join(config.output, "run.json"), { ...config, devices: Object.keys(config.devices), client: config.client ? { kind: config.client.kind } : null,
    resources: config.resources ? Object.fromEntries(Object.entries(config.resources).map(([alias, spec]) => [alias, { kind: spec.kind }])) : null });
  try {
    const aliases = Object.keys(config.devices);
    if (!aliases.includes(config.host) || aliases.length !== manifest.devices.length ||
        manifest.devices.some((d) => !aliases.includes(d.alias))) throw new Error("configured devices differ from manifest");
    for (const [alias, spec] of Object.entries(config.devices)) {
      devices[alias] = spec.kind === "browser" ? await browserDevice(spec, timeoutMs)
        : spec.kind === "node" ? await nodeDevice(spec, timeoutMs, path.join(config.output, alias + ".log")) : null;
      if (!devices[alias]) throw new Error("unknown device kind");
      if (spec.kind === "node") {
        const ready = devices[alias].ready, declared = manifest.devices.find((d) => d.alias === alias);
        await save(path.join(config.output, alias + "-ready.json"), ready);
        for (const key of ["alias", "os", "runtime", "backend", "adapter", "limits", "features"]) {
          const observed = ready?.inventory?.[key], expected = declared[key];
          const same = key === "features" ? Array.isArray(observed) && isDeepStrictEqual([...observed].sort(), [...expected].sort())
            : isDeepStrictEqual(observed, expected);
          if (!same) throw new Error(`${alias} identity differs from manifest: ${key}`);
        }
      }
    }
    if (config.start) await devices[config.host].start(config.start);
    if (config.client) {
      if (config.client.kind !== "node" || config.timing === false || !manifest.client) throw new Error("client runs require a Node API client, timing and client manifest");
      client = await nodeDevice(config.client, timeoutMs, path.join(config.output, "client.log"));
      await save(path.join(config.output, "client-ready.json"), client.ready);
      for (const key of ["alias", "os", "runtime", "transport", "network"]) {
        if (client.ready.inventory[key] !== manifest.client[key]) throw new Error("client identity differs from manifest: " + key);
      }
    }
    let resources = null;
    if (config.resources) {
      const readers = Object.fromEntries(Object.entries(config.resources).map(([alias, spec]) => {
        if (!manifest.devices.some((d) => d.alias === alias)) throw new Error("resource source missing from device manifest");
        return [alias, resourceReader(spec)];
      }));
      const sources = Object.fromEntries(Object.entries(config.resources).map(([alias, spec]) =>
        [alias, { kind: spec.kind, timeoutMs: spec.timeoutMs ?? 5000 }]));
      resources = resourceMonitor(readers, { intervalMs: config.resourceIntervalMs, capacity: config.resourceCapacity, sources });
    }
    const result = await runSeries({ ...config, manifest, devices, client, resources, prompts: standardPrompts(await fs.readFile(config.prompts, "utf8")),
      write: (record) => save(path.join(config.output, "record-" + record.attempt + ".json"), record),
      onRecord: (r) => console.log(`${r.attempt} ${r.trial} ${r.promptName} ${r.status}`) });
    await save(path.join(config.output, "series.json"), result);
    await report(config.output, path.join(config.output, "report"));
    if (result.stopped) process.exitCode = 1;
  } catch (e) {
    await save(path.join(config.output, "runner-error.json"), { error: e.stack }); throw e;
  } finally {
    const closed = await Promise.allSettled([...Object.values(devices), ...(client ? [client] : [])].map((d) => d.close()));
    const errors = closed.filter((r) => r.status === "rejected").map((r) => String(r.reason));
    if (errors.length) { await save(path.join(config.output, "cleanup-errors.json"), errors); process.exitCode = 1; }
  }
}

const [command, ...args] = process.argv.slice(2);
if (command === "run" && args.length === 1) await run(args[0]);
else if (command === "report" && args.length === 2) await report(...args);
else throw new Error("usage: bench_room.js run config.json | report records-directory output-prefix");
