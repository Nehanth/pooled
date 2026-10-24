// How much of a Flash token the room node's expert disk tier takes (benchmarks/disk-tier-direct-io.md step 1b, the
// information row): two room nodes on one machine, a host (HOST_GB, SPLIT=speed) and a node that joins with
// NODE_GB / NODE_RAM / --disk, the golden prompts through the host's generate (greedy, no stop), and the node's expert
// store stats (diskWaitMs, diskReads) diffed around each case. Runs on feat/flash-tab-disk (#368): main refuses Flash offload.
//   node flash_disk_share.mjs   env: MODELS, CASES (two-sum,bash,eos-mid), HOST_GB (3), NODE_GB (8), NODE_RAM (24), NODE_DISK (1), PORT (8271)
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
const ROOT = process.env.ROOT;
const { createRoom, joinRoom } = await import(new URL(`file:///${ROOT.replace(/\\/g, "/")}/packages/room-node/index.js`));
const { greedy } = await import(new URL(`file:///${ROOT.replace(/\\/g, "/")}/room/sampling.js`));
const env = (k, d) => process.env[k] ?? d;
const MODEL = "qwen3.8-flash", MODELS = path.resolve(env("MODELS", path.join(os.homedir(), "llm/models")));
const SIG = +env("PORT", 8271), SIGNAL = `127.0.0.1:${SIG}`, T0 = Date.now();
const log = (...a) => console.error(((Date.now() - T0) / 1000).toFixed(1) + "s", ...a);
const gold = JSON.parse(fs.readFileSync(path.join(ROOT, "tests/golden/q38flash_llama_greedy.json"), "utf8"));
const CASES = env("CASES", "two-sum,bash,eos-mid").split(",");
const NM = path.join(ROOT, "packages/room-node/node_modules");
const peerjs = path.join(NM, "peer/dist/bin/peerjs.js");
const peerServer = spawn(process.execPath, [peerjs, "--port", String(SIG), "--path", "/", "--host", "127.0.0.1"], { stdio: "ignore" });
for (let i = 0; i < 40; i++) { try { await fetch(`http://${SIGNAL}/`); break; } catch { await new Promise((r) => setTimeout(r, 500)); } }
const out = { hostGB: +env("HOST_GB", 3), nodeGB: +env("NODE_GB", 8), nodeRam: +env("NODE_RAM", 24), disk: env("NODE_DISK", "1") === "1", cases: {} };
let host = null, worker = null, code = 1;
try {
  host = await createRoom({ model: MODEL, pledgeGB: out.hostGB, name: "share-a", signal: SIGNAL, modelDir: MODELS, log: (m) => log("[a] " + m), split: env("SPLIT", "speed") });
  worker = await joinRoom(host.code, { pledgeGB: out.nodeGB, ramGB: out.nodeRam, expertDisk: out.disk, name: "share-b", signal: SIGNAL, modelDir: MODELS, log: (m) => log("[b] " + m) });
  const t = Date.now();
  await host.start(MODEL, { minDevices: 2, waitMs: 120000 });
  out.loadS = (Date.now() - t) / 1000;
  out.split = host.split;
  log(`online: ${JSON.stringify(host.split)} in ${out.loadS.toFixed(0)} s`);
  let fail = 0;
  for (const name of CASES) {
    const g = gold[name];
    const st = () => ({ ...(worker.ai?.experts?.stats || {}) });
    const s0 = st(), t1 = performance.now();
    const r = await host.locked(() => host.generate(g.prompt, { maxNew: g.ids.length, sample: greedy, stop: new Set() }));
    const wallMs = performance.now() - t1, s1 = st();
    const d = (k) => (s1[k] || 0) - (s0[k] || 0);
    const ids = r.tokens, diff = ids.findIndex((x, i) => x !== g.ids[i]);
    const ties = (g.nearTies || []).map((x) => x[0]), tieOk = diff >= 0 && ties.some((i) => i <= diff);
    const verdict = diff < 0 ? (ids.length < g.ids.length ? "SHORT" : "MATCH") : tieOk ? "TIE" : "MISMATCH";
    if (verdict === "MISMATCH" || verdict === "SHORT") fail++;
    const toks = g.prompt.length + ids.length;
    const decodeMs = ids.length / r.tps * 1000;
    out.cases[name] = { verdict, n: ids.length, prompt: g.prompt.length, tps: +r.tps.toFixed(2), prefillS: +(r.tPre / 1000).toFixed(2), wallS: +(wallMs / 1000).toFixed(1),
      diskReads: d("diskReads"), diskMiB: +(d("diskBytes") / 2 ** 20).toFixed(0), diskWaitS: +(d("diskWaitMs") / 1000).toFixed(2), t1Copies: d("t1Copies"),
      lookups: d("lookups"), hits: d("hits"), misses: d("misses"),
      diskWaitPerTokMs: +(d("diskWaitMs") / toks).toFixed(2), wallPerTokMs: +(wallMs / toks).toFixed(1), diskShareOfWall: +(d("diskWaitMs") / wallMs).toFixed(3), decodeMs: +decodeMs.toFixed(0) };
    log(`${name}: ${verdict} ${JSON.stringify(out.cases[name])}`);
    log("[b] " + worker.ai?.experts?.summary?.());
  }
  out.pass = !fail; code = fail ? 1 : 0;
} catch (e) { out.error = String(e?.stack || e); log("FAILED: " + (e?.message || e)); }
finally {
  console.log(JSON.stringify(out));
  for (const n of [worker, host]) { try { await n?.close?.(); } catch {} }
  try { peerServer.kill(); } catch {}
  setTimeout(() => process.exit(code), 500);
}
