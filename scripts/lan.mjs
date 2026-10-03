// A room on your home Wi-Fi, served from this computer: no pooled.run, no public signaling server.
//   npm run lan                         (then scan the room's QR code with your phone)
//   node scripts/lan.mjs [--port 8080] [--signal-port 9000] [--host 192.168.1.20] [--no-open]
//
// One process serves the site (with the /room and /r/:code rewrites from serve.json) and a PeerJS
// signaling server, both on this computer's network address. It opens the room on localhost here
// (a secure page, so WebGPU works) with ?invite= set to the network address, so the room's QR code and
// invite links point other devices at this computer (room/lan.js).
//
// Other devices load the page over plain http, which browsers don't treat as secure, so they get no
// WebGPU until it is allowed once per device, in the browser the QR code opens: chrome://flags (brave://flags
// in Brave) -> "Insecure origins treated as secure" -> add both addresses printed below -> Relaunch. Only do
// this on a network you trust: the page and signaling are unencrypted on the Wi-Fi.
import http from "node:http";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import { PeerServer } from "peer";

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const PORT = +arg("--port", 8080), SIGNAL_PORT = +arg("--signal-port", 9000);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// this computer's address on the local network: the first private IPv4 on an interface that is up
function lanAddress() {
  const all = Object.values(os.networkInterfaces()).flat().filter((a) => a && a.family === "IPv4" && !a.internal);
  const priv = all.find((a) => /^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/.test(a.address));
  return (priv || all[0])?.address;
}
const HOST = arg("--host", lanAddress());
if (!HOST) { console.error("No network address found. Connect to Wi-Fi, or pass --host <this computer's IP>."); process.exit(1); }

const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png",
  ".jpg": "image/jpeg", ".ico": "image/x-icon", ".webmanifest": "application/manifest+json", ".woff2": "font/woff2",
  ".wasm": "application/wasm", ".txt": "text/plain; charset=utf-8", ".md": "text/plain; charset=utf-8", ".mp4": "video/mp4",
};

// The whole network can reach this server, so it never serves dotfiles (.git, .env) or node_modules.
function resolve(urlPath) {
  let p = decodeURIComponent(urlPath);
  if (p === "/room" || /^\/r\/[^/]+\/?$/.test(p)) p = "/p2p.html";
  if (p.endsWith("/")) p += "index.html";
  const parts = p.split("/").filter(Boolean);
  if (parts.some((s) => s.startsWith(".") || s === "node_modules")) return null;
  const file = path.join(ROOT, ...parts);
  return file.startsWith(ROOT + path.sep) ? file : null;
}

const site = http.createServer((req, res) => {
  let file = null;
  try { file = resolve(new URL(req.url, "http://x").pathname); } catch {}
  if (req.method !== "GET" && req.method !== "HEAD") { res.writeHead(405).end(); return; }
  fs.stat(file || "", (err, st) => {
    if (!file || err || !st.isFile()) { res.writeHead(404, { "content-type": "text/plain" }).end("not found"); return; }
    res.writeHead(200, {
      "content-type": TYPES[path.extname(file).toLowerCase()] || "application/octet-stream",
      "content-length": st.size, "cache-control": "no-cache", "x-content-type-options": "nosniff",
    });
    if (req.method === "HEAD") res.end(); else fs.createReadStream(file).pipe(res);
  });
});

const listening = (srv, port, what) => new Promise((ok) => {
  srv.on("error", (e) => {
    console.error(e.code === "EADDRINUSE" ? `Port ${port} (${what}) is in use. Stop what's on it, or pass ${what === "site" ? "--port" : "--signal-port"} <other>.` : e.message);
    process.exit(1);
  });
  srv.listen(port, "0.0.0.0", ok);
});
await listening(site, PORT, "site");
// PeerServer listens by itself and doesn't hand back its server, so check the port is free first
const probe = http.createServer();
await listening(probe, SIGNAL_PORT, "signaling");
await new Promise((ok) => probe.close(ok));
await new Promise((ok) => PeerServer({ port: SIGNAL_PORT, path: "/" }, ok));

const siteOrigin = `http://${HOST}:${PORT}`, signalOrigin = `http://${HOST}:${SIGNAL_PORT}`;
const hostUrl = `http://localhost:${PORT}/room?signal=${HOST}:${SIGNAL_PORT}&invite=${encodeURIComponent(siteOrigin)}`;
console.log(`
pooled on your network · this computer is ${HOST}

  1. On this computer, open (it opens by itself):
       ${hostUrl}
     and press "Start a room".

  2. Once per phone or other device, in the browser that opens the QR code:
       chrome://flags (brave://flags in Brave)  ->  "Insecure origins treated as secure"  ->  paste this, Enabled, Relaunch:
       ${siteOrigin},${signalOrigin}

  3. Scan the room's QR code with that device. It joins the room, no typing.

Only on a network you trust: the page and signaling go over the Wi-Fi unencrypted.
Ctrl+C stops both servers.
`);
if (!process.argv.includes("--no-open")) {
  const opener = process.platform === "darwin" ? "open" : process.platform === "win32" ? "explorer" : "xdg-open";
  execFile(opener, [hostUrl], () => {});
}
