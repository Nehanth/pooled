#!/usr/bin/env node
// pooled serve <ROOM CODE>: a Pooled room as a local OpenAI and Anthropic compatible endpoint.
// docs/design/serve.md; cli/README.md for setting up tools.
import { parseArgs } from "node:util";
import { readFileSync } from "node:fs";
import os from "node:os";

const major = +process.versions.node.split(".")[0];
if (major < 22) { console.error("pooled needs Node 22 or newer"); process.exit(1); }

const VERSION = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")).version;
const HELP = `pooled ${VERSION}: use a Pooled room from your own tools

Usage
  pooled serve <ROOM CODE | room link> [options]

  Joins the room as an API client (no layers, no GPU needed here) and serves its model on
  127.0.0.1 as an OpenAI and an Anthropic compatible endpoint.

Options
  --port <n>        HTTP port (default 8080)
  --token <t>       require "Authorization: Bearer <t>" or "x-api-key: <t>" on every request
  --name <s>        how the room shows this client (default: "API · <hostname>")
  --signal <h:p>    PeerJS signaling server, as the room page's ?signal= (default: PeerJS cloud)
  --max-queue <n>   requests that may wait here before 429 / 529 (default 8)
  --quiet           print only errors
  --json-log        one JSON object per log line
  -v, --version     print the version
  -h, --help        this help

Then point a tool at it
  OpenAI     OPENAI_BASE_URL=http://127.0.0.1:8080/v1   (any API key)
  Anthropic  ANTHROPIC_BASE_URL=http://127.0.0.1:8080
`;

let opts;
try {
  opts = parseArgs({
    allowPositionals: true,
    options: {
      port: { type: "string", default: "8080" }, token: { type: "string" }, name: { type: "string" },
      signal: { type: "string" }, "max-queue": { type: "string", default: "8" },
      quiet: { type: "boolean" }, "json-log": { type: "boolean" },
      version: { type: "boolean", short: "v" }, help: { type: "boolean", short: "h" },
    },
  });
} catch (e) { console.error(`pooled: ${e.message}\n\n${HELP}`); process.exit(2); }
const { values: o, positionals: pos } = opts;
if (o.version) { console.log(VERSION); process.exit(0); }
if (o.help || !pos.length || pos[0] === "help") { console.log(HELP); process.exit(pos.length && pos[0] !== "help" ? 2 : 0); }
if (pos[0] !== "serve") { console.error(`pooled: unknown command ${pos[0]}\n\n${HELP}`); process.exit(2); }

const { Bridge, roomCodeFrom } = await import("../lib/room.js");
const { createServer } = await import("../lib/http.js");
const { cleanLabel } = await import("../lib/common.js");

const code = roomCodeFrom(pos[1]);
if (!code) { console.error(`pooled: give a room code (4 to 6 letters and digits) or a room link${pos[1] ? `, not "${pos[1]}"` : ""}`); process.exit(2); }
const port = +o.port;
if (!Number.isInteger(port) || port < 0 || port > 65535) { console.error("pooled: --port must be a port number"); process.exit(2); }
const maxQueue = Math.max(0, parseInt(o["max-queue"], 10));
if (!Number.isFinite(maxQueue)) { console.error("pooled: --max-queue must be a number"); process.exit(2); }
if (o.token === "") { console.error("pooled: --token must not be empty"); process.exit(2); }

const log = (msg, level = "info") => {
  if (o.quiet && level !== "error") return;
  if (o["json-log"]) console.error(JSON.stringify({ t: new Date().toISOString(), level, msg }));
  else console.error(`${new Date().toTimeString().slice(0, 8)} ${msg}`);
};
const host = os.hostname().split(".")[0].slice(0, 20) || "computer";
const bridge = new Bridge({ code, signal: o.signal || null, name: cleanLabel(o.name) || `API · ${host}`, client: `pooled-cli/${VERSION}`, log });
// POOLED_KEEPALIVE_MS: the queue keep-alive interval (tests; 10 s otherwise)
const api = createServer({ bridge, port, token: o.token || null, maxQueue, log, version: VERSION, keepAliveMs: +process.env.POOLED_KEEPALIVE_MS || undefined });

// the port first: fail fast, before joining a room
let bound;
try { bound = await api.listen(); }
catch (e) {
  if (e.code === "EADDRINUSE") console.error(`pooled: port ${port} is busy (the repo's \`npm run serve\` uses 8080 too): pass --port`);
  else console.error(`pooled: cannot listen on 127.0.0.1:${port}: ${e.message}`);
  process.exit(1);
}
try { await bridge.connect(); }
catch (e) { console.error(`pooled: ${e.message}`); api.server.close(); process.exit(1); }

const label = () => (bridge.ready ? `${bridge.modelLabel || bridge.model}` : "model not ready yet");
const print = (s) => { if (!o.quiet) console.log(s); };
print(`pooled serve · room ${code} · ${label()}
  OpenAI     http://127.0.0.1:${bound}/v1         (OPENAI_BASE_URL, any API key)
  Anthropic  http://127.0.0.1:${bound}            (ANTHROPIC_BASE_URL)
  bound to 127.0.0.1 only · ${o.token ? "token required" : "no token (add --token to require one)"}`);
if (o["json-log"]) log(JSON.stringify({ ready: bridge.ready, port: bound, room: code }));

let wasReady = bridge.ready;
bridge.on("state", () => {
  if (bridge.ready && !wasReady) log(`the room's model is ready: ${bridge.modelLabel || bridge.model}`);
  else if (!bridge.ready && wasReady && !bridge.kicked) log("the room's model is not ready (a device left or the host is re-dealing)");
  wasReady = bridge.ready;
  if (bridge.kicked) log(`disconnected by the host: ${bridge.kicked}; requests now get 503`, "error");
});

let stopping = false;
async function shutdown() {
  if (stopping) process.exit(0);
  stopping = true;
  api.closeAll("pooled serve is shutting down");
  await bridge.leave();
  api.server.close();
  setTimeout(() => process.exit(0), 200);
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
