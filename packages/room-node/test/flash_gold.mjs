// Qwen3.8-Flash-Next in a room of room nodes, on one machine (GPU, the whole model: ~75 GB), against llama.cpp's greedy
// ids (tests/golden/q38flash_llama_greedy.json, the prompts that fit the phase-1 context of 2048). The prompts go in
// as the golden's token ids through the host's generate (the room's prefill and decode, frames over WebRTC between
// the nodes), greedy, no stop token, as many tokens as the golden has. Judged by tests/q38flash_rule.js, the same rule
// as the engine's own golden test (tests/test_q38flash.js): an exact match, or a difference at a backend near tie (our
// token within 0.5 nats of llama.cpp's top there, and our own top-2 gap under 0.5).
//   node packages/room-node/test/flash_gold.mjs solo   one room node holds every layer (createRoom + start)
//   node packages/room-node/test/flash_gold.mjs pair   two room nodes in this process, a local PeerJS server between
//        them: HOST_GB / WORKER_GB (default 40 / 40, memory split: layers 0-23 + embedding/head and the n-gram table
//        on the host, 24-47 on the worker). SPLIT=speed with HOST_GB=3 puts layers 1-47 (and the table) on the worker.
// env: MODELS (default ~/llm/models: models/q38flash/<file>, source.js LOCAL, or <dir>/qwen3.8-flash/<file>),
//      CASES=a,b (default every golden case under 2048 tokens), SOLO_GB (90), OUT (result JSON), PORT (8251)
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { createRoom, joinRoom } from "../index.js";
import { greedy } from "../../../room/sampling.js";
import { judgeGolden } from "../../../tests/q38flash_rule.js";

const MODE = process.argv[2] || "solo";
const MODEL = "qwen3.8-flash";
const NM = path.resolve(new URL("../node_modules", import.meta.url).pathname);
const MODELS = path.resolve(process.env.MODELS || path.join(os.homedir(), "llm/models"));
const SIG = +(process.env.PORT || 8251), SIGNAL = `127.0.0.1:${SIG}`;
const T0 = Date.now();
const log = (...a) => console.error(((Date.now() - T0) / 1000).toFixed(1) + "s", ...a);
const gold = JSON.parse(fs.readFileSync(new URL("../../../tests/golden/q38flash_llama_greedy.json", import.meta.url), "utf8"));
const pick = process.env.CASES ? process.env.CASES.split(",") : null;
const cases = Object.entries(gold).filter(([k, v]) => !k.startsWith("_") && (!pick || pick.includes(k)) && v.prompt.length + v.ids.length <= 2048);
const out = { mode: MODE, model: MODEL, cases: {} };

const peerServer = spawn(path.join(NM, ".bin/peerjs"), ["--port", String(SIG), "--path", "/", "--host", "127.0.0.1"], { stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));
let host = null, worker = null, code = 1;
const nodeLog = (who) => (m) => log(`[${who}] ${m}`);
try {
  if (MODE === "solo") {
    host = await createRoom({ model: MODEL, pledgeGB: +(process.env.SOLO_GB || 90), name: "flash-solo", signal: SIGNAL, modelDir: MODELS, log: nodeLog("host"), split: "speed" });
    const t = Date.now();
    await host.start(MODEL);
    out.loadS = (Date.now() - t) / 1000;
  } else if (MODE === "pair") {
    const split = process.env.SPLIT || "memory";
    host = await createRoom({ model: MODEL, pledgeGB: +(process.env.HOST_GB || 40), name: "flash-a", signal: SIGNAL, modelDir: MODELS, log: nodeLog("a"), split });
    worker = await joinRoom(host.code, { pledgeGB: +(process.env.WORKER_GB || 40), name: "flash-b", signal: SIGNAL, modelDir: MODELS, log: nodeLog("b") });
    const t = Date.now();
    await host.start(MODEL, { minDevices: 2, waitMs: 60000 });
    out.loadS = (Date.now() - t) / 1000;
  } else throw new Error("mode: solo | pair");
  out.split = host.split;
  log(`online: ${JSON.stringify(host.split)} in ${out.loadS.toFixed(0)} s`);
  let fail = 0;
  for (const [name, g] of cases) {
    const t = performance.now();
    // the room's head logits at every step (plain decoding: one sample call per answer token; Flash has no
    // speculative path), for the rule's top-2 gap and the top-5 at the first difference
    const gaps = [], tops = [];
    const sample = (lg) => { gaps.push(gap(lg)); tops.push(top5(lg)); return greedy(lg); };
    const r = await host.locked(() => host.generate(g.prompt, { maxNew: g.ids.length, sample, stop: new Set() }));
    const ids = r.tokens;
    if (gaps.length < ids.length) throw new Error(`${name}: ${gaps.length} sample calls for ${ids.length} tokens (a path that samples elsewhere: the gaps would not line up)`);
    const J = judgeGolden(g, ids, gaps, g.ids.length);
    let diff = J.diff;
    if (diff < 0 && ids.length < g.ids.length) diff = ids.length;
    const verdict = diff < 0 ? "MATCH" : J.excused ? "TIE" : "MISMATCH";
    if (verdict === "MISMATCH") fail++;
    out.cases[name] = { verdict, n: g.ids.length, diff, rule: J.msg, tps: +r.tps.toFixed(2), prefillS: +(r.tPre / 1000).toFixed(2), prompt: g.prompt.length, wallS: +((performance.now() - t) / 1000).toFixed(1),
      text: host.ai.tok.decode(ids), ...(diff >= 0 ? { got: ids.slice(Math.max(0, diff - 2), diff + 4), want: g.ids.slice(Math.max(0, diff - 2), diff + 4), ourTop: tops[diff], llamaTop: g.top?.[diff] } : {}) };
    log(`${name}: ${verdict}${diff >= 0 ? ` at ${diff}: ${diff < ids.length ? J.msg : `${ids.length} of ${g.ids.length} ids`}` : ` (${g.ids.length} ids)`} · prompt ${g.prompt.length} tok in ${(r.tPre / 1000).toFixed(1)} s · decode ${r.tps.toFixed(2)} tok/s`);
  }
  out.pass = !fail;
  code = fail ? 1 : 0;
} catch (e) {
  out.error = String(e?.stack || e);
  log("FAILED: " + (e?.message || e));
} finally {
  if (process.env.OUT) fs.writeFileSync(process.env.OUT, JSON.stringify(out, null, 1));
  console.log(JSON.stringify(out));
  console.log(out.pass ? "FLASH GOLD PASS" : "FLASH GOLD FAIL");
  for (const n of [worker, host]) { try { await n?.close?.(); } catch {} }
  try { peerServer.kill(); } catch {}
  setTimeout(() => process.exit(code), 500);
}

// top-2 gap of a step's logits (nats: the same as its log-softmax's), and its top 5 as [id, logprob]
function gap(lg) {
  let a = -Infinity, b = -Infinity;
  for (const v of lg) { if (v > a) { b = a; a = v; } else if (v > b) b = v; }
  return a - b;
}
function top5(lg) {
  let mx = -Infinity, z = 0;
  for (const v of lg) if (v > mx) mx = v;
  for (const v of lg) z += Math.exp(v - mx);
  const lz = mx + Math.log(z), top = [];
  for (let i = 0; i < lg.length; i++) {
    if (top.length < 5 || lg[i] > top[4][1]) { top.push([i, lg[i]]); top.sort((x, y) => y[1] - x[1]); if (top.length > 5) top.pop(); }
  }
  return top.map(([i, v]) => [i, +(v - lz).toFixed(3)]);
}
