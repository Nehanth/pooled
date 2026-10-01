// pooled serve <ROOM CODE>: a Pooled room as a local OpenAI and Anthropic compatible endpoint.
// docs/design/serve.md; cli/README.md for setting up tools.
import { parseArgs } from "node:util";
import { readFileSync } from "node:fs";
import { argsError, needsRoom, notARoom } from "./cli.js";

export const HELP_SERVE = `Usage
  pooled serve <ROOM CODE | "room link"> [options]

  Joins the room as an API client (no layers, no GPU needed here) and serves its model on
  127.0.0.1 as an OpenAI (Chat Completions and Responses) and an Anthropic (Messages)
  compatible endpoint, tool calls included.

  With the room's invite link (quote it: "https://pooled.run/r/4TKG9P#k=..."), the host lets
  this client in at once. With the code alone, it waits until the host allows it (a host that
  asks before new devices join) and the host's "Allow API clients" must be on either way.

Options
  --port <n>        HTTP port (default 8080)
  --token-file <f>  require the token in this file as "Authorization: Bearer <t>" or
                    "x-api-key: <t>" on every request (or set POOLED_TOKEN)
  --token <t>       the same, given on the command line (other local users can read it with ps)
  --name <s>        how the room shows this client (default: "pooled serve" and 4 random letters)
  --signal <h:p>    PeerJS signaling server, as the room page's ?signal= (default: PeerJS cloud)
  --max-queue <n>   requests that may wait here behind the running one before 429 / 529
                    (default 8; 0 = only when idle)
  --quiet           print only errors
  --json-log        one JSON object per log line
  -h, --help        this help

Then point a tool at it
  OpenAI     OPENAI_BASE_URL=http://127.0.0.1:8080/v1   (any API key; Codex: wire_api = "responses")
  Anthropic  ANTHROPIC_BASE_URL=http://127.0.0.1:8080   (Claude Code too)
  opencode   an @ai-sdk/openai-compatible provider (the banner prints one for the room)
`;

const OPTIONS = {
  port: { type: "string", default: "8080" }, token: { type: "string" }, "token-file": { type: "string" }, name: { type: "string" },
  signal: { type: "string" }, "max-queue": { type: "string", default: "8" },
  quiet: { type: "boolean" }, "json-log": { type: "boolean" }, help: { type: "boolean", short: "h" },
};
export const SERVE_FLAGS = Object.keys(OPTIONS);

// argv after "serve" -> { help } | options | { error: [lines], code }
export function parseServeArgs(argv, { roomCodeFrom }) {
  let r;
  try { r = parseArgs({ args: argv, allowPositionals: true, strict: true, options: OPTIONS }); }
  catch (e) { return { error: argsError("serve", e, SERVE_FLAGS), code: 2 }; }
  const { values: o, positionals: pos } = r;
  if (o.help) return { help: true };
  if (!pos.length) return { error: [needsRoom("serve")], code: 2 };
  const code = roomCodeFrom(pos[0]);
  if (!code) return { error: [notARoom("serve", pos[0])], code: 2 };
  if (pos.length > 1) return { error: [`pooled serve takes one room, not ${pos.length}: ${pos.slice(0, 3).join(" ")}`], code: 2 };
  return { o, pos, code };
}

// -> exit status, or never returns (serves until Ctrl-C)
export async function serveMain(argv, { version = "" } = {}) {
  const { Bridge, roomCodeFrom, roomKeyFrom } = await import("./room.js");
  const p = parseServeArgs(argv, { roomCodeFrom });
  if (p.help) { process.stdout.write(HELP_SERVE); return 0; }
  if (p.error) { process.stderr.write(p.error.join("\n") + "\n"); return p.code; }
  const { o, pos, code } = p;
  const { createServer } = await import("./http.js");
  const { showRoom } = await import("./banner.js");
  const { cleanLabel, cleanText } = await import("./common.js");
  const die = (msg) => { process.stderr.write(`pooled serve: ${msg}\n`); return 2; };

  // the invite link's key (#k=…): the host lets this client in without asking. Quote the link in the shell
  const key = roomKeyFrom(pos[0]);
  const port = +o.port;
  if (!Number.isInteger(port) || port < 0 || port > 65535) return die("--port must be a port number");
  const maxQueue = Math.max(0, parseInt(o["max-queue"], 10));
  if (!Number.isFinite(maxQueue)) return die("--max-queue must be a number");
  // the token: POOLED_TOKEN or --token-file (not visible in ps), else --token with a warning
  let token = null;
  if (o["token-file"]) {
    try { token = readFileSync(o["token-file"], "utf8").trim(); }
    catch (e) { return die(`cannot read --token-file: ${e.message}`); }
    if (!token) return die("the --token-file is empty");
  } else if (process.env.POOLED_TOKEN != null) {
    token = process.env.POOLED_TOKEN.trim();
    if (!token) return die("POOLED_TOKEN is empty");
  } else if (o.token != null) {
    if (o.token === "") return die("--token must not be empty");
    token = o.token;
    if (!o.quiet) console.error("pooled: note: other users on this computer can read --token from the process list; POOLED_TOKEN or --token-file keeps it out of it");
  }

  const log = (msg, level = "info") => {
    if (o.quiet && level !== "error") return;
    // every line can carry text from the room (errors, reasons, the model's name): no control characters
    msg = String(msg).split("\n").map((l) => cleanText(l)).join("\n  ");
    if (o["json-log"]) console.error(JSON.stringify({ t: new Date().toISOString(), level, msg }));
    else console.error(`${new Date().toTimeString().slice(0, 8)} ${msg}`);
  };
  // never the hostname by default: every guest in the room sees this name
  const tag = Math.random().toString(36).slice(2, 6).padEnd(4, "0");
  const bridge = new Bridge({ code, key, signal: o.signal || null, name: cleanLabel(o.name) || `pooled serve ${tag}`, client: `pooled-cli/${version}`, log });
  // POOLED_KEEPALIVE_MS: the queue keep-alive interval (tests; 10 s otherwise)
  const api = createServer({ bridge, port, token, maxQueue, log, version, keepAliveMs: +process.env.POOLED_KEEPALIVE_MS || undefined });

  // the port first: fail fast, before joining a room
  let bound;
  try { bound = await api.listen(); }
  catch (e) {
    if (e.code === "EADDRINUSE") console.error(`pooled: port ${port} is busy (the repo's \`npm run serve\` uses 8080 too): pass --port`);
    else console.error(`pooled: cannot listen on 127.0.0.1:${port}: ${e.message}`);
    return 1;
  }
  try { await bridge.connect(); }
  catch (e) { console.error(`pooled: ${e.message}`); api.server.close(); return 1; }

  // the host's ai-ready-all follows its hello: give it a moment, so the banner can name the model
  if (!bridge.ready) await new Promise((r) => { const t = setTimeout(r, 1500); bridge.once("state", () => { if (bridge.ready) { clearTimeout(t); r(); } }); });
  const print = (s) => { if (!o.quiet) console.log(s); };
  showRoom(bridge, { code, port: bound, token, print, log });
  if (o["json-log"]) log(JSON.stringify({ ready: bridge.ready, port: bound, room: code }));

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
  return new Promise(() => {});   // until shutdown()
}
