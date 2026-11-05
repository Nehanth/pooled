// Dedicated Node benchmark process. JSON-line commands on stdin; replies are
// prefixed because the engine may also write diagnostics to stdout.
// node benchmarks/bench_room_device.js /path/to/local-device.json
import fs from "node:fs/promises";
import readline from "node:readline";
import os from "node:os";
import { createRoom, joinRoom } from "../packages/room-node/index.js";
import { setupNode, runtime } from "../packages/room-node/env.js";
import { createTimingCollector } from "../room/timing.js";
import { nodeBenchmark } from "./bench_room_node.js";

const config = JSON.parse(await fs.readFile(process.argv[2], "utf8"));
const timing = config.timing ? createTimingCollector() : null, errors = [];
await setupNode();
const adapter = await navigator.gpu.requestAdapter();
if (!adapter || adapter.isFallbackAdapter) throw new Error("hardware WebGPU adapter required");
const proto = Object.getPrototypeOf(adapter), requestDevice = proto.requestDevice;
proto.requestDevice = async function (...args) {
  const device = await requestDevice.apply(this, args);
  device.addEventListener("uncapturederror", (e) => errors.push(e.error.message));
  device.lost.then((e) => { if (e.reason !== "destroyed") errors.push("device lost: " + e.message); });
  return device;
};
const info = adapter.info;
const inventory = { alias: config.options.name, os: `${os.platform()} ${os.release()}`, runtime: process.version, backend: runtime.gpu,
  adapter: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description },
  features: [...adapter.features], limits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } };
const options = { ...config.options, onTiming: timing?.record, log: (s) => process.stderr.write(s + "\n") };
const room = config.join ? await joinRoom(config.join, options) : await createRoom(options);
const bench = nodeBenchmark(room, { timing });
const reply = (id, result, error = null) => process.stdout.write("@pooled-bench " + JSON.stringify({ id, result, error }) + "\n");
reply("ready", { code: room.code, inventory });
const input = readline.createInterface({ input: process.stdin });
try {
  for await (const line of input) {
    let cmd;
    try {
      cmd = JSON.parse(line);
      let result;
      switch (cmd.action) {
        case "snapshot": result = { ...bench.snapshot(), gpuErrors: errors.slice() }; break;
        case "begin": bench.begin(cmd.attempt, cmd.epoch); result = null; break;
        case "end": result = await bench.end(); break;
        case "run": result = await bench.run(cmd.request); break;
        case "prepareClient": result = await bench.prepareClient(cmd.request); break;
        case "pairClient": result = bench.pairClient(cmd.client, cmd.capture, cmd.prepared); break;
        case "start": await room.start(); result = bench.snapshot(); break;
        case "close": result = null; break;
        default: throw new Error("unknown benchmark command");
      }
      reply(cmd.id, result);
      if (cmd.action === "close") break;
    } catch (e) { reply(cmd?.id ?? null, null, e.message); }
  }
} finally { input.close(); await room.close(); }
// Dawn/libdatachannel can retain native handles after the room releases its GPU.
// This dedicated process has finished; drain its reply before terminating it.
await new Promise((resolve) => process.stdout.write("", resolve));
process.exit(0);
