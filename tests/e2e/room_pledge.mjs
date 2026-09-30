// Pledges in the real room page (p2p.html + room.js): the model picker says how short a room is
// and who could give more, Start stays off, and (--start) the host's deal keeps every device within
// its pledge. The reported room: a laptop creates it and lends 3 GB, an iPhone-shaped tab lends 1 GB.
// Two headless Chromium tabs, a local PeerServer, WebGPU on SwiftShader (no GPU needed).
//
//   node tests/e2e/room_pledge.mjs                picker checks only (no model download)
//   node tests/e2e/room_pledge.mjs --start        also press Start on Qwen3 1.7B and check the deal
//                                                 (fetches the 1.7B's header, config and tokenizer from
//                                                 Hugging Face; stops once the layers are dealt)
//   options: --shots DIR (screenshots of the picker), --port 8151 --signal-port 9011
//
// Needs playwright + peer + peerjs: NODE_PATH=<node_modules dir> or a node_modules next to the repo.
import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const flag = (k) => argv.includes("--" + k);
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const PORT = +arg("port", 8151), SIGNAL_PORT = +arg("signal-port", 9011), SHOTS = arg("shots", "");
const PEERJS_URL = "https://cdn.jsdelivr.net/npm/peerjs@1.5.4/dist/peerjs.min.js";
const IPHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 26_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.6 Mobile/15E148 Safari/604.1";
function resolvePkg(name) {
  const dirs = [...(process.env.NODE_PATH || "").split(path.delimiter).filter(Boolean), path.join(ROOT, "node_modules")];
  for (const d of dirs) for (const base of [d, path.join(d, "node_modules")]) if (fs.existsSync(path.join(base, name, "package.json"))) return path.join(base, name);
  throw new Error(`${name} not found: set NODE_PATH to the node_modules dir that contains it`);
}
let fails = 0;
const check = (c, m) => { console.log(`${c ? "PASS" : "FAIL"} ${m}`); if (!c) fails++; };

const peerjsJs = fs.readFileSync(path.join(resolvePkg("peerjs"), "dist/peerjs.min.js"));
const peerServer = spawn(process.execPath, [path.join(resolvePkg("peer"), "dist/bin/peerjs.js"), "--port", String(SIGNAL_PORT), "--host", "127.0.0.1", "--path", "/"], { stdio: "ignore" });
const srv = serveRepo(PORT);
for (let i = 0; ; i++) {
  if (await fetch(`http://127.0.0.1:${SIGNAL_PORT}/peerjs/id`).then((r) => r.ok, () => false)) break;
  if (i > 50) { console.error("PeerServer did not start"); process.exit(2); }
  await new Promise((r) => setTimeout(r, 200));
}
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), headless: true, args: GPU_ARGS });
const ctxFor = async (ua) => {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, ...(ua ? { userAgent: ua } : {}) });
  await ctx.route("**/*", async (route) => {
    const url = route.request().url();
    if (url.startsWith(`http://127.0.0.1:`)) return route.continue();
    if (url.split("?")[0] === PEERJS_URL) return route.fulfill({ status: 200, contentType: "text/javascript", body: peerjsJs });
    if (url.startsWith("https://fonts.googleapis.com/")) return route.fulfill({ status: 200, contentType: "text/css", body: "" });
    // --start: Hugging Face through this process (its bot wall answers a headless Chrome with 405)
    if (flag("start") && url.startsWith("https://huggingface.co/")) {
      const req = route.request(), cors = { "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" };
      if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers: { ...cors, "access-control-allow-headers": "range", "access-control-allow-methods": "GET, HEAD" } });
      const range = (await req.allHeaders()).range;
      const r = await fetch(url, { headers: range ? { range } : {} });
      const body = Buffer.from(await r.arrayBuffer());
      const h = { ...cors, "content-type": r.headers.get("content-type") || "application/octet-stream" };
      for (const k of ["content-range", "accept-ranges", "etag"]) if (r.headers.get(k)) h[k] = r.headers.get(k);
      return route.fulfill({ status: r.status, headers: h, body });
    }
    return route.abort();
  });
  return ctx;
};
const base = `http://127.0.0.1:${PORT}/p2p.html?split=speed&signal=127.0.0.1:${SIGNAL_PORT}`;
const laptop = await (await ctxFor(null)).newPage(), phone = await (await ctxFor(IPHONE)).newPage();
const errs = [];
for (const [n, p] of [["laptop", laptop], ["phone", phone]]) p.on("pageerror", (e) => errs.push(`${n}: ${String(e).slice(0, 200)}`));
if (flag("verbose")) for (const p of [laptop, phone]) p.on("response", (r) => { if (!r.url().startsWith("http://127.0.0.1")) console.error(r.status(), r.url().slice(0, 140), r.headers()["content-type"]); });
const pick = (p, key) => p.evaluate((k) => { const s = document.getElementById("ai-model"); s.value = k; s.dispatchEvent(new Event("change", { bubbles: true })); }, key);
const view = (p) => p.evaluate(() => ({
  short: document.getElementById("ai-short").hidden ? "" : document.getElementById("ai-short").textContent,
  start: document.getElementById("ai-start").disabled ? "off" : "on",
  rows: [...document.querySelectorAll("#ai-ladder .rung")].map((b) => b.textContent.replace(/\s+/g, " ").trim()),
  need: document.getElementById("ap-need").textContent,
}));
try {
  for (const p of [laptop, phone]) { await p.goto(base); await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 }); await p.waitForTimeout(1500); }
  await laptop.fill("#name-input", "Laptop"); await laptop.fill("#join-gb", "3");
  await phone.fill("#name-input", "iPhone"); await phone.fill("#join-gb", "1");
  await laptop.click("#create-btn");
  await laptop.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("room-badge").textContent), null, { timeout: 30000 });
  const code = (await laptop.textContent("#room-badge")).trim().match(/[A-Z0-9]{4}/)[0];
  await phone.fill("#code-input", code); await phone.click("#join-btn");
  for (const p of [laptop, phone]) await p.waitForFunction(() => document.querySelectorAll(".peer-card").length >= 2, null, { timeout: 60000 });
  await laptop.waitForTimeout(1500);
  const pools = await laptop.evaluate(() => document.getElementById("ap-total").textContent);
  check(pools === "4 GB", `the room pools 3 + 1 GB (${pools})`);

  // the 1.7B fits: 21 layers on the laptop within 3 GB, 8 would fit on the phone
  await pick(laptop, "qwen3-1.7b");
  let v = await view(laptop);
  check(v.start === "on" && !v.short, `Qwen3 1.7B fits 3 + 1 GB, Start on (${JSON.stringify(v)})`);
  // the 27B is short: say by how much, keep Start off
  await pick(laptop, "qwen3.8-27b");
  v = await view(laptop);
  check(v.start === "off", "Qwen3.8 27B: Start off");
  check(/^This room is [\d.]+ GB short for Qwen3\.8 27B\. Add a device or raise a pledge/.test(v.short), `27B short message: "${v.short}"`);
  check(/GB short/.test(v.need), `pool card says short: "${v.need}"`);
  if (SHOTS) { fs.mkdirSync(SHOTS, { recursive: true }); await laptop.locator("#ai-panel").screenshot({ path: path.join(SHOTS, "short-27b.png") }); }

  // the laptop lends 2 GB: 1.7B no longer fits (it would have been overfilled before)
  await pick(laptop, "qwen3-1.7b");
  await laptop.evaluate(() => { const i = document.getElementById("ap-gb"); i.value = "2"; i.dispatchEvent(new Event("change", { bubbles: true })); i.dispatchEvent(new Event("blur")); });
  await laptop.waitForFunction(() => document.getElementById("ap-total").textContent === "3 GB", null, { timeout: 5000 }).catch(() => {});
  v = await view(laptop);
  check(v.start === "off", `Laptop 2 GB + iPhone 1 GB for Qwen3 1.7B: Start off (${JSON.stringify(v.rows)})`);
  check(/^This room is [\d.]+ GB short for Qwen3 1\.7B\. Add a device or raise a pledge/.test(v.short), `1.7B short message: "${v.short}"`);
  check(!/iPhone could give/.test(v.short), "the iPhone, at its 1 GB cap, is not asked for more");
  const pv = await (async () => { await phone.waitForTimeout(800); await pick(phone, "qwen3-1.7b"); return view(phone); })();
  check(/GB short for Qwen3 1\.7B/.test(pv.short) && pv.start === "off", `the phone's screen says the same: "${pv.short}"`);
  if (SHOTS) await laptop.locator("#ai-panel").screenshot({ path: path.join(SHOTS, "short-1.7b.png") });

  if (flag("start")) {
    // back to 3 GB and start: the deal must keep the laptop within 3 GB and give the phone layers
    await laptop.evaluate(() => { const i = document.getElementById("ap-gb"); i.value = "3"; i.dispatchEvent(new Event("change", { bubbles: true })); i.dispatchEvent(new Event("blur")); });
    await laptop.waitForFunction(() => !document.getElementById("ai-start").disabled, null, { timeout: 10000 });
    await laptop.click("#ai-start");
    await laptop.waitForFunction(() => [...document.querySelectorAll("#chat-log div")].some((d) => /memory per device/.test(d.textContent)) || /^failed/.test(document.getElementById("ai-status").textContent), null, { timeout: 120000 });
    const lines = await laptop.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => /memory per device|layer split/.test(t)));
    console.log(lines.join("\n") || `no deal: ${await laptop.textContent("#ai-status")} | ${await laptop.evaluate(() => [...document.querySelectorAll("#chat-log div")].slice(-4).map((d) => d.textContent).join(" / "))}`);
    const mem = lines.find((t) => /memory per device/.test(t)) || "";
    const m = /Laptop ([\d.]+) of 3\.0 GB.*iPhone ([\d.]+) of 1\.0 GB/.exec(mem);
    check(!!m && +m[1] <= 3 && +m[2] > 0 && +m[2] <= 1, `deal within pledges: ${mem}`);
    await phone.waitForFunction(() => document.querySelector(".peer-card.self .pheld:not([hidden])"), null, { timeout: 30000 }).catch(() => {});
    const held = await phone.evaluate(() => [...document.querySelectorAll(".peer-card .pheld")].map((e) => e.closest(".peer-card").dataset.name + ": " + e.textContent));
    check(held.some((t) => /^iPhone: uses [\d.]+ of 1 GB pledged/.test(t)), `the phone's card: ${JSON.stringify(held)}`);
  }
  check(!errs.length, `no page errors ${JSON.stringify(errs)}`);
} catch (e) {
  check(false, String(e).slice(0, 300));
} finally {
  await browser.close(); srv.close(); peerServer.kill();
}
console.log(fails ? `ROOM PLEDGE FAIL (${fails})` : "ROOM PLEDGE PASS");
process.exit(fails ? 1 : 0);
