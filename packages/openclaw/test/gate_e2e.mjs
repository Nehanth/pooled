// Two OpenClaw gateways on one machine, each with the plugin installed from its npm tarball (no Pooled
// checkout at run time), through the join gate:
//   A hosts room <code> (onboarded non-interactively: gate on, two devices, 3 GB each);
//   B joins it with the code alone, so A's owner is asked: B waits in the lobby, `/pooled` on A shows
//   the request, `/pooled allow` lets it in, and B keeps its pass;
//   the room deals the 1.7B over both; a question on B goes to A over the ask bridge (which shows A
//   B's pass: no second join request) and one on A is answered in process;
//   A restarts: same invite link (the saved gate), B comes back in with its pass, no new request,
//   and the room warms up OpenClaw's system prompt and tools before the next question (timed).
//   node test/gate_e2e.mjs   (GPU: run it through the machine's GPU queue)
// env: OPENCLAW (bin; default openclaw on PATH), TGZ (a packed @pooled/openclaw; default: npm pack
//      here), POOLED_MODELS (a models folder with qwen3-1.7b; default ~/.pooled/models), WORK, OUT, NN
//      (node_modules with the "peer" server; default packages/room-node's)
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn, execFileSync } from "node:child_process";

const HERE = path.resolve(new URL("..", import.meta.url).pathname);
const OC = process.env.OPENCLAW || "openclaw";
const W = process.env.WORK || fs.mkdtempSync(path.join(os.tmpdir(), "pooled-gate-e2e-"));
const NN = process.env.NN || path.resolve(HERE, "../room-node/node_modules");
const MODELS = process.env.POOLED_MODELS || path.join(os.homedir(), ".pooled", "models");
const SIG = 18940 + Math.floor(Math.random() * 40);
const T0 = Date.now();
const log = (...a) => console.error(((Date.now() - T0) / 1000).toFixed(1) + "s", ...a);
const out = { steps: {}, turns: [] };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- one isolated OpenClaw per side (its own HOME and state: the machine's ~/.openclaw is never touched)
function side(name, port) {
  const d = path.join(W, name);
  for (const x of ["home", "state", "run"]) fs.mkdirSync(path.join(d, x), { recursive: true });
  const env = { ...process.env, HOME: path.join(d, "home"), OPENCLAW_HOME: path.join(d, "home"), OPENCLAW_STATE_DIR: path.join(d, "state"),
    OPENCLAW_CONFIG_PATH: path.join(d, "state", "openclaw.json"), XDG_RUNTIME_DIR: path.join(d, "run"), POOLED_MODELS: MODELS, POOLED_SIGNAL: `127.0.0.1:${SIG}` };
  delete env.DBUS_SESSION_BUS_ADDRESS;
  for (const k of Object.keys(env)) if (k.startsWith("POOLED_") && !["POOLED_MODELS", "POOLED_SIGNAL"].includes(k)) delete env[k];
  const oc = (args, extra = {}, o = {}) => execFileSync(OC, args, { env: { ...env, ...extra }, encoding: "utf8", maxBuffer: 64 << 20, timeout: o.timeout || 300000, stdio: ["ignore", "pipe", "pipe"] });
  const file = (...p) => path.join(d, "state", ...p);
  const json = (f) => { try { return JSON.parse(fs.readFileSync(f, "utf8")); } catch { return null; } };
  let gw = null;
  return {
    name, port, env, oc, dir: d,
    status: () => json(file("pooled", "status.json")),
    room: () => json(file("pooled", "room.json")),
    start() {
      try { fs.rmSync(file("pooled", "status.json")); } catch {}
      const lf = fs.openSync(path.join(d, `gateway-${Date.now()}.log`), "w");
      gw = spawn(OC, ["gateway", "run", "--port", String(port), "--allow-unconfigured", "--verbose"], { env: { ...env, POOLED_DEBUG: "1" }, stdio: ["ignore", lf, lf] });
      gw.on("exit", (c, s) => log(`${name}: gateway exited ${c ?? s}`));
    },
    async stop() { if (!gw) return; gw.kill("SIGTERM"); for (let i = 0; i < 40 && gw.exitCode == null && gw.signalCode == null; i++) await sleep(250); try { gw.kill("SIGKILL"); } catch {} gw = null; },
    // one chat turn (or a /command) through this side's gateway
    agent(message, session) {
      const t = Date.now();
      let stdout = "", stderr = "";
      try { stdout = oc(["agent", "--agent", "main", "--session-id", session, "--message", message, "--json", "--timeout", "900"], {}, { timeout: 960000 }); }
      catch (e) { stdout = e.stdout || ""; stderr = String(e.stderr || e.message).slice(-1500); }
      let j = null; try { j = JSON.parse(stdout.slice(stdout.indexOf("{"))); } catch {}
      const text = JSON.stringify(j?.result?.payloads || j?.payloads || j || stdout).slice(0, 1500);
      return { message, seconds: (Date.now() - t) / 1000, text, stderr: stderr || undefined };
    },
  };
}
async function until(fn, ms, what) {
  const t = Date.now();
  for (;;) { const v = await fn(); if (v) return v; if (Date.now() - t > ms) throw new Error("timeout: " + what); await sleep(500); }
}

const A = side("host", 18901), B = side("join", 18902);
const peer = spawn(path.join(NN, ".bin/peerjs"), ["--port", String(SIG), "--path", "/", "--host", "127.0.0.1"], { stdio: "ignore" });
let code = 1;
try {
  await sleep(1500);
  let tgz = process.env.TGZ;
  if (!tgz) { execFileSync("npm", ["pack", "--silent", "--pack-destination", W], { cwd: HERE, encoding: "utf8" }); tgz = fs.readdirSync(W).filter((f) => f.endsWith(".tgz")).map((f) => path.join(W, f))[0]; }
  log("tarball", tgz);
  for (const s of [A, B]) s.oc(["plugins", "install", tgz, "--force", "--accept-capabilities"]);
  // A: host, gate on (the default), two devices of 3 GB (the 1.7B at 16k needs about 5.5 GB)
  A.oc(["onboard", "--non-interactive", "--accept-risk", "--auth-choice", "pooled", "--gateway-port", String(A.port), "--skip-channels", "--skip-skills", "--skip-health", "--skip-daemon"],
    { POOLED_MODE: "host", POOLED_MIN_DEVICES: "2", POOLED_PLEDGE_GB: "3", POOLED_NO_SERVICE: "1" });
  const cfgA = JSON.parse(fs.readFileSync(A.env.OPENCLAW_CONFIG_PATH, "utf8")).plugins.entries.pooled.config;
  const keyA = A.room()?.host?.key || A.room()?.host?.gate?.key;
  out.code = cfgA.code;
  log(`A: room ${cfgA.code}, key saved: ${!!keyA}`);
  A.start();
  const a0 = await until(() => { const s = A.status(); return s?.code ? s : null; }, 120000, "A opening the room");
  out.linkA = a0.link;
  if (!a0.link.includes(`/r/${cfgA.code}#k=${keyA}`)) throw new Error(`A's link ${a0.link} is not the onboarding one`);
  // B: joins with the code alone (no link, no key): A's owner is asked
  B.oc(["onboard", "--non-interactive", "--accept-risk", "--auth-choice", "pooled", "--gateway-port", String(B.port), "--skip-channels", "--skip-skills", "--skip-health", "--skip-daemon"],
    { POOLED_MODE: "join", POOLED_CODE: cfgA.code, POOLED_PLEDGE_GB: "3", POOLED_NO_SERVICE: "1" });
  B.start();
  const tKnock = Date.now();
  const w = await until(() => { const s = A.status(); return s?.waiting?.length ? s.waiting : null; }, 120000, "B's join request on A");
  out.steps.request = { line: w[0].line, s: (Date.now() - tKnock) / 1000 };
  log("A sees:", w[0].line);
  await until(() => B.status()?.admission === "lobby", 30000, "B in the lobby");
  // an ask on B while it waits: a clear message, not a hang
  const early = B.agent("hello", "gate-early");
  out.steps.askInLobby = { seconds: early.seconds, says: /let this device in/.test(early.text + early.stderr) };
  log("B ask in the lobby:", early.seconds, "s", out.steps.askInLobby.says ? "(says it waits for the host)" : early.text.slice(0, 300));
  const st = A.agent("/pooled", "gate-cmd");
  out.steps.pooledStatus = st.text.slice(0, 600);
  log("A /pooled:", st.text.slice(0, 400));
  const al = A.agent("/pooled allow", "gate-cmd");
  out.steps.allow = al.text.slice(0, 300);
  log("A /pooled allow:", al.text.slice(0, 300));
  await until(() => B.status()?.admission === "in", 60000, "B let in");
  await until(() => B.room()?.join?.[cfgA.code]?.pass, 30000, "B's pass saved");
  log("B is in; its pass is saved");
  const on = await until(() => { const s = A.status(); return s?.online ? s : null; }, 600000, "the room online");
  out.split = on.split;
  log("room online:", on.split?.join(" · "));
  // B asks through the bridge (shows A its pass: no second request); A asks in process
  const b1 = B.agent("In one short sentence: what is 2 + 2?", "gate-b1");
  out.turns.push({ side: "B", ...b1 });
  log("B turn:", b1.seconds, "s", b1.text.slice(0, 300));
  out.steps.secondRequest = (A.status()?.waiting || []).length;
  const a1 = A.agent("In one short sentence: what colour is the sky?", "gate-a1");
  out.turns.push({ side: "A", ...a1 });
  log("A turn (cold):", a1.seconds, "s", a1.text.slice(0, 300));
  // A restarts: same link; B back in with its pass; warm-up before the next question
  await A.stop();
  await sleep(2000);
  A.start();
  const a2 = await until(() => { const s = A.status(); return s?.code ? s : null; }, 120000, "A reopening the room");
  out.steps.sameLink = a2.link === a0.link;
  log("A reopened:", out.steps.sameLink ? "same invite link" : `NEW link ${a2.link}`);
  await until(() => { const s = A.status(); return s?.online ? s : null; }, 600000, "the room online again");
  out.steps.requestsAfterRestart = (A.status()?.waiting || []).length;
  const warm = await until(() => (A.status()?.events || []).find((e) => /warmed up|warm-up/.test(e.m)), 300000, "the warm-up").catch(() => null);
  out.steps.warmup = warm?.m || null;
  log("warm-up:", warm?.m);
  const a3 = A.agent("In one short sentence: what colour is grass?", "gate-a3");
  out.turns.push({ side: "A", warm: true, ...a3 });
  log("A turn (after warm-up):", a3.seconds, "s", a3.text.slice(0, 300));
  code = 0;
} catch (e) {
  out.error = String(e?.stack || e).slice(0, 1000); log("FAILED", out.error);
} finally {
  out.statusA = A.status(); out.statusB = B.status();
  out.elapsedS = (Date.now() - T0) / 1000;
  console.log(JSON.stringify(out, null, 1));
  if (process.env.OUT) fs.writeFileSync(process.env.OUT, JSON.stringify(out, null, 1));
  await B.stop(); await A.stop(); peer.kill();
  setTimeout(() => process.exit(code), 300);
}
