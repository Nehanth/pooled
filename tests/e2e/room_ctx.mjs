// Qwen3 1.7B's context in a real browser room: one laptop-size tab lends --gb GB (default 4 and 8),
// picks the 1.7B and starts it. 4 GB (an 8 GB laptop's default) is short for 16K, so the room opens
// it at 8K and says so: the need line, the picker row ("4 GB · 8K fits"), the chat note and the model
// band; after one answer the context meter reads "/ 8k". 8 GB holds 16K: no note, "/ 16k".
// room/models.js pickCtx decides; the machine's own GPU runs it (Linux/NVIDIA flags as room.mjs).
//
//   node tests/e2e/room_ctx.mjs                 4 GB then 8 GB
//   node tests/e2e/room_ctx.mjs --gb 4          one of them
//
// Weights: models/qwen17/model.gguf when present (served as Hugging Face's file), else Hugging Face.
import { chromium } from "playwright";
import http from "http";
import https from "https";
import fs from "fs";
import os from "os";
import path from "path";
import { spawn, execSync } from "child_process";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const GBS = arg("gb", "4,8").split(",").map(Number);
const PORT = +arg("port", 8133), TLS_PORT = PORT + 1, SIGNAL_PORT = PORT + 2;
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const LOCAL = { "Qwen3-1.7B-Q8_0.gguf": "models/qwen17/model.gguf" };
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };
const UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
const t0 = Date.now();
const log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(1).padStart(6) + "s", ...a);
const results = [];
const check = (name, ok, extra = "") => { results.push({ name, ok: !!ok }); log(ok ? "PASS" : "FAIL", name, ok ? "" : String(extra).slice(0, 300)); };

const srv = http.createServer((q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream"); fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
const tlsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-ctx-"));
execSync(`openssl req -x509 -newkey rsa:2048 -nodes -keyout ${tlsDir}/k.pem -out ${tlsDir}/c.pem -days 2 -subj /CN=127.0.0.1 2>/dev/null`);
const wsrv = https.createServer({ key: fs.readFileSync(`${tlsDir}/k.pem`), cert: fs.readFileSync(`${tlsDir}/c.pem`) }, (q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p)) { r.statusCode = 404; r.end(); return; }
  const size = fs.statSync(p).size, m = /bytes=(\d+)-(\d*)/.exec(q.headers.range || "");
  const lo = m ? +m[1] : 0, hi = m && m[2] ? Math.min(+m[2], size - 1) : size - 1;
  r.writeHead(m ? 206 : 200, { "content-type": "application/octet-stream", "content-range": `bytes ${lo}-${hi}/${size}`, "accept-ranges": "bytes", "content-length": String(hi - lo + 1), "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" });
  fs.createReadStream(p, { start: lo, end: hi }).pipe(r);
}).listen(TLS_PORT, "127.0.0.1");
const peerServer = spawn(path.join(ROOT, "node_modules/.bin/peerjs"), ["--port", String(SIGNAL_PORT), "--path", "/"], { stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));
const BASE = `http://127.0.0.1:${PORT}/p2p.html?ask=0&signal=127.0.0.1:${SIGNAL_PORT}`;
const browser = await chromium.launch({ headless: false, args: ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan", "--ignore-gpu-blocklist"] });

async function roomAt(gb) {
  const ctx = await browser.newContext({ userAgent: UA, ignoreHTTPSErrors: true });
  await ctx.route("**/*.gguf", (route) => {
    const file = LOCAL[route.request().url().split("/").pop().split("?")[0]];
    return file && fs.existsSync(path.join(ROOT, file)) ? route.continue({ url: `https://127.0.0.1:${TLS_PORT}/${file}` }) : route.continue();
  });
  const p = await ctx.newPage();
  const errs = [];
  p.on("pageerror", (e) => errs.push(String(e).slice(0, 200)));
  try {
    await p.goto(BASE);
    await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
    await p.fill("#name-input", `laptop-${gb}`); await p.fill("#join-gb", String(gb));
    await p.click("#create-btn");
    await p.waitForFunction(() => document.querySelector("#ai-ladder .rung"), null, { timeout: 30000 });
    await p.selectOption("#ai-model", "qwen3-1.7b");
    await p.click('#ai-ladder .rung[data-k="qwen3-1.7b"]');
    await p.waitForTimeout(300);
    const before = await p.evaluate(() => ({ need: document.getElementById("need-text").textContent, row: document.querySelector('#ai-ladder .rung[data-k="qwen3-1.7b"]')?.textContent || "",
      title: document.querySelector('#ai-ladder .rung[data-k="qwen3-1.7b"]')?.title || "", start: !document.getElementById("ai-start").disabled }));
    log(`${gb} GB before Start:`, JSON.stringify(before));
    const short = gb < 5.6;
    check(`${gb} GB: Start is on`, before.start, JSON.stringify(before));
    check(`${gb} GB: the need line says both needs`, before.need.startsWith("Needs 5.6 GB (4 GB at 8K). The room has"), before.need);
    check(`${gb} GB: ${short ? "the 8K note" : "no 8K note"} before Start`, before.need.includes("Qwen3 1.7B · 8K context: the room's memory is short for 16K.") === short, before.need);
    check(`${gb} GB: the picker row`, short ? /4 GB · 8K\s*fits/.test(before.row) && /short for 16K/.test(before.title) : /5\.6 GB\s*fits/.test(before.row), before.row);
    await p.click("#ai-start");
    await p.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online") || /^failed:/.test(document.getElementById("ai-status").textContent), null, { timeout: 600000 });
    const st = await p.textContent("#ai-status");
    check(`${gb} GB: the model starts`, !/^failed:/.test(st), st);
    await p.fill("#ai-prompt", "Say hi in three words."); await p.click("#ai-send");
    await p.waitForFunction(() => /^ready — prefill|^generation failed/.test(document.getElementById("ai-status").textContent), null, { timeout: 300000 });
    const after = await p.evaluate(() => ({ ctx: document.getElementById("sm-ctx").textContent, model: document.querySelector("#swarm-map .sm-model").textContent,
      modelTitle: document.querySelector("#swarm-map .sm-model").title, chat: document.getElementById("ai-output")?.textContent || "",
      log: [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => /memory per device|short for 16K/.test(t)) }));
    log(`${gb} GB after an answer:`, JSON.stringify({ ctx: after.ctx, model: after.model, modelTitle: after.modelTitle, log: after.log }));
    check(`${gb} GB: the room's context is ${short ? "8k" : "16k"}`, after.ctx.endsWith(short ? "/ 8k" : "/ 16k"), after.ctx);
    check(`${gb} GB: the model band`, short ? after.model === "Qwen3 1.7B · 8K context" && after.modelTitle === "Qwen3 1.7B · 8K context: the room's memory is short for 16K" : after.model === "Qwen3 1.7B" && !after.modelTitle, after.model);
    check(`${gb} GB: the chat says it`, after.chat.includes("Qwen3 1.7B · 8K context: the room's memory is short for 16K") === short);
    check(`${gb} GB: no page errors`, !errs.length, errs.join(" | "));
  } catch (e) {
    check(`${gb} GB: ran`, false, String(e));
  } finally { await ctx.close(); }
}

try {
  for (const gb of GBS) await roomAt(gb);
} finally {
  await browser.close(); srv.close(); wsrv.close(); fs.rmSync(tlsDir, { recursive: true, force: true }); peerServer.kill();
}
const failed = results.filter((r) => !r.ok);
console.log(JSON.stringify({ ok: !failed.length, passed: results.length - failed.length, failed: failed.map((r) => r.name) }));
process.exitCode = failed.length ? 1 : 0;
