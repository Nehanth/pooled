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
//        CODE_FILE=<path>: the worker is not started here: the room's code is written to that file and the host waits
//        (WAIT_S, default 600) for a node from another process to join, e.g. one from an older build (a mixed room).
// env: MODELS (default ~/llm/models: models/q38flash/<file>, source.js LOCAL, or <dir>/qwen3.8-flash/<file>),
//      CASES=a,b (default every golden case under 2048 tokens), SOLO_GB (90), OUT (result JSON), PORT (8251),
//      POOLED_QSA_ALWAYS=1 (QSA's gathered attention on every step, env.js qsaAlways). The result records the room's
//      Flash format (ai-load.flash: QSA, or null for dense) and what each worker's ai-ready said it loaded.
//      SAME_AS=<an earlier run's OUT>: also require every case's ids to equal that run's (a batched room against the
//        same room per token, POOLED_FLASH_BATCH=0, or against main's)
//      WORKER_RAM_GB=N (pair): the worker lends N GB of RAM for expert offload, so a WORKER_GB under its share parks
//        experts (shard.js loadFlash: batched passes with the whole-layer region when it fits, else per token)
//      POOLED_FLASH_BATCH=0 | exact (room-node env.js): batched prompt passes off / on the bit-exact kernels
//      FRAMES=128,256 (a node host's prompt frame widths, POOLED_PREFILL_FRAME): every case once per width, keyed
//        name@width, the caches cleared between (no checkpoint or live prefix carries over)
// Each case reports its prompt's prefill time and tok/s (prefillS, prefillTps), decode tok/s and its ids
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
    if (process.env.CODE_FILE) { fs.writeFileSync(process.env.CODE_FILE, host.code); log(`room ${host.code}: waiting for a worker from another process`); }
    else worker = await joinRoom(host.code, { pledgeGB: +(process.env.WORKER_GB || 40), name: "flash-b", signal: SIGNAL, modelDir: MODELS, log: nodeLog("b"),
      ...(+process.env.WORKER_RAM_GB > 0 ? { ramGB: +process.env.WORKER_RAM_GB } : {}) });
    const t = Date.now();
    await host.start(MODEL, { minDevices: 2, waitMs: process.env.CODE_FILE ? +(process.env.WAIT_S || 600) * 1000 : 60000 });
    out.loadS = (Date.now() - t) / 1000;
  } else throw new Error("mode: solo | pair");
  out.split = host.split;
  out.flash = host.ai.roomFlash ?? null;
  out.engines = [{ name: host.name, maxSeq: host.ai.engine?.maxSeq, sig: host.ai.engine?.stateSignature?.() || null },
    ...[...(host.ai.readyInfo || new Map())].map(([name, r]) => ({ name, maxSeq: r.ctx, sig: r.fsig }))];
  out.qsaAlways = process.env.POOLED_QSA_ALWAYS === "1";
  log(`online: ${JSON.stringify(host.split)} in ${out.loadS.toFixed(0)} s; Flash format ${JSON.stringify(out.flash)}; engines ${JSON.stringify(out.engines)}`);
  let fail = 0;
  const same = process.env.SAME_AS ? JSON.parse(fs.readFileSync(process.env.SAME_AS, "utf8")) : null;
  const widths = process.env.FRAMES ? process.env.FRAMES.split(",") : [null];
  for (const [name, g, w] of widths.flatMap((w) => cases.map(([k, v]) => [w ? `${k}@${w}` : k, v, w]))) {
    if (w) { process.env.POOLED_PREFILL_FRAME = w; host.ai.fed = null; host.ckptClear(true); }
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
    // the same ids as the reference run (SAME_AS), when given
    const ref = (same?.cases?.[name] || same?.cases?.[name.split("@")[0]])?.ids, sameIds = ref ? ref.length === ids.length && ref.every((x, i) => x === ids[i]) : null;
    if (sameIds === false) fail++;
    out.cases[name] = { verdict, n: g.ids.length, diff, rule: J.msg, tps: +r.tps.toFixed(2), prefillS: +(r.tPre / 1000).toFixed(2), prefillTps: +(g.prompt.length / (r.tPre / 1000)).toFixed(1),
      prompt: g.prompt.length, wallS: +((performance.now() - t) / 1000).toFixed(1), ...(sameIds !== null ? { sameIds } : {}),
      text: host.ai.tok.decode(ids), ids: Array.from(ids), ...(diff >= 0 ? { got: ids.slice(Math.max(0, diff - 2), diff + 4), want: g.ids.slice(Math.max(0, diff - 2), diff + 4), ourTop: tops[diff], llamaTop: g.top?.[diff] } : {}) };
    log(`${name}: ${verdict}${diff >= 0 ? ` at ${diff}: ${diff < ids.length ? J.msg : `${ids.length} of ${g.ids.length} ids`}` : ` (${g.ids.length} ids)`} · prompt ${g.prompt.length} tok in ${(r.tPre / 1000).toFixed(1)} s (${(g.prompt.length / (r.tPre / 1000)).toFixed(1)} tok/s) · decode ${r.tps.toFixed(2)} tok/s${sameIds === null ? "" : sameIds ? " · ids == SAME_AS" : " · ids DIFFER from SAME_AS"}`);
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
