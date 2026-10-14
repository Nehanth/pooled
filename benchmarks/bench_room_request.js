// Ask-only benchmark client using the ordinary room API. Connection details stay
// in the caller's local config; replies contain no room code or invite key.
import fs from "node:fs/promises";
import os from "node:os";
import readline from "node:readline";
import { Bridge } from "../cli/lib/room.js";
import { measureRequest } from "./bench_room_client.js";

const config = JSON.parse(await fs.readFile(process.argv[2], "utf8"));
const bridge = new Bridge({ ...config.connection, log: (s) => process.stderr.write(s + "\n") });
const reply = (id, result, error = null) => process.stdout.write("@pooled-bench " + JSON.stringify({ id, result, error }) + "\n");
const input = readline.createInterface({ input: process.stdin });
let serial = 0;
try {
  await bridge.connect();
  if (!bridge.ready) await new Promise((resolve, reject) => {
    const cleanup = () => { clearTimeout(timer); bridge.off("state", change); };
    const change = () => { if (bridge.ready) { cleanup(); resolve(); } };
    const timer = setTimeout(() => { cleanup(); reject(new Error("client did not see the host ready")); }, 30000);
    bridge.on("state", change); change();
  });
  reply("ready", { inventory: { alias: config.alias, os: `${os.platform()} ${os.release()}`, runtime: process.version,
    transport: "Bridge API v1", network: config.network } });
  for await (const line of input) {
    let cmd;
    try {
      cmd = JSON.parse(line);
      if (cmd.action === "close") { reply(cmd.id, null); break; }
      if (cmd.action !== "run") throw new Error("unknown requesting-client command");
      reply(cmd.id, await measureRequest(bridge, cmd.request, { rid: "bench-" + ++serial, timeoutMs: config.timeoutMs ?? 600000 }));
    } catch (e) { reply(cmd?.id ?? null, null, String(e?.message || e)); }
  }
} finally { input.close(); await bridge.leave(); }
// The dedicated client owns this WebRTC stack; drain replies before native exit.
await new Promise((resolve) => process.stdout.write("", resolve));
process.exit(0);
