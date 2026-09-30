#!/usr/bin/env node
// Join a Pooled room from the command line and lend this machine's GPU: the room node holds the
// layers the host deals it until Ctrl-C. No browser, no OpenClaw.
//   node packages/room-node/join.mjs <CODE> [--gb 12] [--name mac] [--signal host:port] [--models dir]
// --signal: a PeerServer host:port (default the PeerJS cloud server pooled.run uses)
// --models: local model files (source.js LOCAL layout) instead of HTTP range reads
// --state <file>: write the room status there every 5 s (JSON)
import fs from "node:fs";
import { joinRoom } from "./index.js";

const argv = process.argv.slice(2);
const opt = (k, d = null) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const code = argv.find((a, i) => !a.startsWith("--") && !(i > 0 && argv[i - 1].startsWith("--")));
if (!code) { console.error("usage: node join.mjs <CODE> [--gb N] [--name NAME] [--signal host:port] [--models DIR] [--state FILE]"); process.exit(2); }
const t0 = Date.now();
const log = (m) => console.log(`${new Date().toISOString()} +${((Date.now() - t0) / 1000).toFixed(1)}s ${m}`);
const node = await joinRoom(code.toUpperCase(), {
  pledgeGB: +opt("gb", 0) || undefined, name: opt("name") || undefined, signal: opt("signal"), modelDir: opt("models"), log,
});
log(`joined room ${node.code} as ${node.name}`);
node.on("loaded", (x) => log(`holding layers ${x.range[0]}-${x.range[1] - 1} of ${x.model}`));
node.on("hostgone", () => log("lost the link to the host"));
node.on("back", () => log("back in the room"));
const stateFile = opt("state");
if (stateFile) setInterval(() => { try { fs.writeFileSync(stateFile, JSON.stringify({ at: new Date().toISOString(), ...node.status() }, null, 1)); } catch {} }, 5000).unref();
const bye = async () => { log("leaving"); try { await node.close(); } catch {} process.exit(0); };
process.on("SIGINT", bye); process.on("SIGTERM", bye);
setInterval(() => {}, 1 << 30);
