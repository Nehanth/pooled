// Proved links with a real model (GPU): a gated room node host (Ask on) and two room node devices
// that come in with the invite link, so the chain host -> a -> b -> host runs over links that proved
// themselves: the devices' links to the host (the gate's proofs) with their stripes, and a -> b (the
// room's mesh key) with its stripes. Then `pooled chat` (cli/bin/pooled.js, its own process) asks once
// with the invite link (in at once) and once with the code alone (waits; allowJoin lets it in), and
// the answers match the host's own (greedy).
//   node packages/room-node/test/chanauth_gpu_e2e.mjs   (run it through the machine's GPU queue)
// env: MODELS (default ~/.pooled/models, with qwen3-1.7b), MODEL, GB (each device's pledge, 2.5)
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { createRoom, joinRoom } from "../index.js";

const ROOT = path.resolve(new URL("../../..", import.meta.url).pathname);
const NM = path.resolve(new URL("../node_modules", import.meta.url).pathname);
const MODELS = path.resolve(process.env.MODELS || path.join(os.homedir(), ".pooled", "models"));
const MODEL = process.env.MODEL || "qwen3-1.7b";
const GB = +(process.env.GB || 2.5);
const SIG = 18700 + Math.floor(Math.random() * 200), SIGNAL = `127.0.0.1:${SIG}`;
const PROMPT = "What is the capital of France? Answer in one word.";
const T0 = Date.now();
const log = (...a) => console.error(((Date.now() - T0) / 1000).toFixed(1) + "s", ...a);
const out = { checks: {} };
const check = (k, v, extra = "") => { out.checks[k] = !!v; log(v ? "PASS" : "FAIL", k, extra); };
const lines = [];
const nodeLog = (who) => (s) => { lines.push(`${who}: ${s}`); if (/prove|proof|check out|unverified|legacy|older Pooled/i.test(s)) log(`[${who}] ${s}`); };

const peerSrv = spawn(path.join(NM, ".bin/peerjs"), ["--port", String(SIG), "--path", "/", "--host", "127.0.0.1"], { stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));
const nodes = [];
function chat(room, args = []) {
  return new Promise((resolve) => {
    const p = spawn(process.execPath, [path.join(ROOT, "cli/bin/pooled.js"), "chat", room, PROMPT, "--signal", SIGNAL, "--max-tokens", "24", "--temperature", "0", ...args], { stdio: ["ignore", "pipe", "pipe"] });
    let o = "", e = "";
    p.stdout.on("data", (c) => { o += c; }); p.stderr.on("data", (c) => { e += c; });
    p.on("close", (code) => resolve({ code, out: o.trim(), err: e.trim() }));
  });
}
let code = 0;
try {
  const host = await createRoom({ model: MODEL, pledgeGB: GB, name: "host", signal: SIGNAL, modelDir: MODELS, gate: true, ask: true, log: nodeLog("host") });
  nodes.push(host);
  const link = `http://127.0.0.1/r/${host.code}${host.inviteFragment}`;
  const reqs = [];
  host.on("joinrequest", (q) => reqs.push(q));
  const key = host.gate.key;
  const a = await joinRoom(host.code, { key, pledgeGB: GB, name: "dev-a", signal: SIGNAL, modelDir: MODELS, log: nodeLog("a") });
  const b = await joinRoom(host.code, { key, pledgeGB: GB, name: "dev-b", signal: SIGNAL, modelDir: MODELS, log: nodeLog("b") });
  nodes.push(a, b);
  const verified = { a: null, b: null };
  a.on("admitted", (x) => { verified.a = x.verified; }); b.on("admitted", (x) => { verified.b = x.verified; });
  for (let i = 0; i < 300 && (a.admission !== "in" || b.admission !== "in"); i++) await new Promise((r) => setTimeout(r, 100));
  check("devices in with the invite key, the host's proof checked, mesh key in hand", a.admission === "in" && b.admission === "in" && a.mk === host.mk && b.mk === host.mk && reqs.length === 0);
  const t0 = Date.now();
  await host.start(MODEL, { minDevices: 3, waitMs: 60000 });
  out.onlineS = (Date.now() - t0) / 1000;
  out.chain = host.ai.chainNames;
  log("chain", JSON.stringify(out.chain), "online in", out.onlineS, "s");
  let text = "";
  for await (const ev of host.ask([{ role: "user", content: PROMPT }], { maxTokens: 24, temperature: 0 })) if (ev.type === "token" && !ev.think) text += ev.text;
  out.hostText = text.trim();
  const links = [a, b].map((n) => [...n.conns.entries()].filter(([id]) => id !== host.peer.id).map(([id, e]) => ({ id, held: !!e.hold, stripes: e.stripes.length })));
  out.meshLinks = links;
  check("the host answers over the proved chain", /paris/i.test(out.hostText), out.hostText);
  check("a link between the two devices, proved, with stripes", links.flat().some((l) => !l.held && l.stripes >= 1), JSON.stringify(links));
  check("frames crossed the devices", (a.frames || 0) > 0 && (b.frames || 0) > 0, `a ${a.frames} b ${b.frames}`);
  check("no link closed for a failed proof", !lines.some((l) => /didn't prove|no proof/.test(l)));
  // pooled chat with the invite link: in at once, the same answer
  const c1 = await chat(link);
  check("pooled chat with the invite link: in at once, same answer", c1.code === 0 && reqs.length === 0 && c1.out === out.hostText, `${c1.code} ${JSON.stringify(c1.out)} ${c1.err.slice(-200)}`);
  // pooled chat with the code alone: waits with the six digits, Allow lets it in
  host.once("joinrequest", (q) => setTimeout(() => host.allowJoin(q.id), 500));
  const c2 = await chat(host.code);
  const sas = reqs[0]?.sas;
  check("pooled chat with the code: asked (with the code both sides show), allowed, same answer", c2.code === 0 && reqs.length === 1 && !!sas && c2.err.includes(sas) && c2.out === out.hostText, `${c2.code} sas ${sas} ${JSON.stringify(c2.out)} ${c2.err.slice(-300)}`);
} catch (e) {
  log("ERROR", e?.stack || e); code = 2;
} finally {
  if (Object.values(out.checks).some((v) => !v) && !code) code = 1;
  console.log(JSON.stringify(out));
  for (const n of nodes.reverse()) try { await n.close(); } catch {}
  peerSrv.kill();
  setTimeout(() => process.exit(code), 300);
}
