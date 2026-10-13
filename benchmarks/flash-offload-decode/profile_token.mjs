// Profile and A/B of Flash offload decode (benchmarks/flash-offload-decode.md): two room nodes in one process on one
// machine, a host (HOST_GB, SPLIT=speed) and a node that joins with NODE_GB / NODE_RAM / --disk (NODE_DISK), the golden
// prompts through the host's generate (greedy, no stop), as benchmarks/disk-tier-direct-io/flash_disk_share.mjs.
// The node's engine is timed from outside (prototype patches, nothing in the repo changes): per decode token, each
// offloaded layer's cut (Qwen4ExpEngine._afterRoute) split into the readback (submit -> the ids mapped), the disk wait
// (ExpertStore.prepare) and the plan (ExpertStore.plan: the pool LRU and the copy encode, incl. the writeBuffer of
// read bytes), plus the time between cuts (encoding the rest, outside any cut) and from the last cut to the next token.
//   ROOT=<checkout> node profile_token.mjs     env: MODELS, CASES (two-sum,bash,eos-mid), HOST_GB (3), NODE_GB (8),
//   NODE_RAM (24), NODE_DISK (1), PORT (8271), SPLIT (speed), PROF (1: patch and time; 0: only the stats)
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
const ROOT = path.resolve(process.env.ROOT);
const url = (p) => new URL(`file:///${path.join(ROOT, p).replace(/\\/g, "/").replace(/^\/+/, "")}`).href;
const { createRoom, joinRoom } = await import(url("packages/room-node/index.js"));
const { greedy } = await import(url("room/sampling.js"));
const env = (k, d) => process.env[k] ?? d;
const MODEL = "qwen3.8-flash", MODELS = path.resolve(env("MODELS", path.join(os.homedir(), "llm/models")));
const SIG = +env("PORT", 8271), SIGNAL = `127.0.0.1:${SIG}`, T0 = Date.now();
const log = (...a) => console.error(((Date.now() - T0) / 1000).toFixed(1) + "s", ...a);
const gold = JSON.parse(fs.readFileSync(path.join(ROOT, "tests/golden/q38flash_llama_greedy.json"), "utf8"));
const CASES = env("CASES", "two-sum,bash,eos-mid").split(",");
const PROF = env("PROF", "1") === "1";

// ---- the timers (one engine in this process does cuts: the node's; the host's layer 0 is resident)
const rec = { tokens: [], cur: null };
let patched = 0;
if (PROF) {
  // the engine modules read WebGPU's globals at import: Dawn first, as a room node does
  await (await import(url("packages/room-node/env.js"))).setupNode();
  const { Qwen4ExpEngine } = await import(url("engine/qwen4exp.js"));
  const { ExpertStore } = await import(url("engine/expert_store.js"));
  const P = Qwen4ExpEngine.prototype, S = ExpertStore.prototype;
  const begin0 = P._begin, after0 = P._afterRoute, prep0 = S.prepare, plan0 = S.plan;
  P._begin = function (...a) {
    const t = performance.now();
    if (rec.cur) { rec.cur.end = t; rec.tokens.push(rec.cur); }
    rec.cur = this.experts ? { t0: t, cuts: [], prep: 0, plan: 0, reads: 0 } : null;
    return begin0.apply(this, a);
  };
  P._afterRoute = async function (enc, l) {
    const c = rec.cur;
    if (!l.off || !c) return after0.call(this, enc, l);
    const t = performance.now(), p0 = c.prep, q0 = c.plan;
    const r = await after0.call(this, enc, l);
    const t1 = performance.now();
    c.cuts.push({ l: l.i, t, t1, prep: c.prep - p0, plan: c.plan - q0 });
    return r;
  };
  S.prepare = async function (...a) {
    const t = performance.now(), n = await prep0.apply(this, a);
    if (rec.cur) { rec.cur.prep += performance.now() - t; rec.cur.reads += n; }
    return n;
  };
  S.plan = function (...a) {
    const t = performance.now(), r = plan0.apply(this, a);
    if (rec.cur) rec.cur.plan += performance.now() - t;
    return r;
  };
  patched = 1;
}
// one token's split (ms): cut = readback + disk + plan; between: from a cut's end to the next cut (encode and the
// non-cut work); head: _begin to the first cut; tail: the last cut's end to the next token's _begin (the rest of the
// layers, the readback of the residual, the hop to the host and back)
function split(tok) {
  const c = tok.cuts, s = { tok: tok.end - tok.t0, cuts: c.length, readback: 0, disk: 0, plan: 0, between: 0, head: 0, tail: 0, reads: tok.reads };
  for (let i = 0; i < c.length; i++) {
    s.readback += c[i].t1 - c[i].t - c[i].prep - c[i].plan; s.disk += c[i].prep; s.plan += c[i].plan;
    if (i) s.between += c[i].t - c[i - 1].t1;
  }
  if (c.length) { s.head = c[0].t - tok.t0; s.tail = tok.end - c.at(-1).t1; }
  return s;
}
const sum = (xs) => { const o = {}; for (const x of xs) for (const [k, v] of Object.entries(x)) o[k] = (o[k] || 0) + v; return o; };

const NM = path.join(ROOT, "packages/room-node/node_modules");
const peerjs = path.join(NM, "peer/dist/bin/peerjs.js");
const peerServer = spawn(process.execPath, [peerjs, "--port", String(SIG), "--path", "/", "--host", "127.0.0.1"], { stdio: "ignore" });
for (let i = 0; i < 40; i++) { try { await fetch(`http://${SIGNAL}/`); break; } catch { await new Promise((r) => setTimeout(r, 500)); } }
const out = { root: ROOT, hostGB: +env("HOST_GB", 3), nodeGB: +env("NODE_GB", 8), nodeRam: +env("NODE_RAM", 24), disk: env("NODE_DISK", "1") === "1", prof: patched, cases: {} };
let host = null, worker = null, code = 1;
try {
  host = await createRoom({ model: MODEL, pledgeGB: out.hostGB, name: "prof-a", signal: SIGNAL, modelDir: MODELS, log: (m) => log("[a] " + m), split: env("SPLIT", "speed") });
  worker = await joinRoom(host.code, { pledgeGB: out.nodeGB, ramGB: out.nodeRam, expertDisk: out.disk, name: "prof-b", signal: SIGNAL, modelDir: MODELS, log: (m) => log("[b] " + m) });
  const t = Date.now();
  await host.start(MODEL, { minDevices: 2, waitMs: 120000 });
  out.loadS = (Date.now() - t) / 1000;
  out.split = host.split;
  log(`online: ${JSON.stringify(host.split)} in ${out.loadS.toFixed(0)} s`);
  let fail = 0;
  for (const name of CASES) {
    const g = gold[name];
    const st = () => ({ ...(worker.ai?.experts?.stats || {}) });
    rec.tokens = []; rec.cur = null;
    const s0 = st(), t1 = performance.now();
    const r = await host.locked(() => host.generate(g.prompt, { maxNew: g.ids.length, sample: greedy, stop: new Set() }));
    const wallMs = performance.now() - t1, s1 = st();
    if (rec.cur) { rec.cur.end = performance.now(); rec.tokens.push(rec.cur); rec.cur = null; }
    const d = (k) => (s1[k] || 0) - (s0[k] || 0);
    const ids = r.tokens, diff = ids.findIndex((x, i) => x !== g.ids[i]);
    const ties = (g.nearTies || []).map((x) => x[0]), tieOk = diff >= 0 && ties.some((i) => i <= diff);
    const verdict = diff < 0 ? (ids.length < g.ids.length ? "SHORT" : "MATCH") : tieOk ? "TIE" : "MISMATCH";
    if (verdict !== "MATCH") fail++;
    const o = out.cases[name] = { verdict, n: ids.length, prompt: g.prompt.length, tps: +r.tps.toFixed(3), decodeS: +(r.tDecode / 1000).toFixed(2), prefillS: +(r.tPre / 1000).toFixed(2),
      wallS: +(wallMs / 1000).toFixed(1), diskReads: d("diskReads"), diskMiB: +(d("diskBytes") / 2 ** 20).toFixed(0), diskWaitS: +(d("diskWaitMs") / 1000).toFixed(2),
      t1Copies: d("t1Copies"), t1Writes: d("t1Writes"), lookups: d("lookups"), hits: d("hits"), misses: d("misses"), poolMiB: +(d("bytes") / 2 ** 20).toFixed(0) };
    if (patched) {
      // decode tokens: the steps after the prompt's (the prompt's go one at a time with offload); the first decode step
      // follows the last prompt step, so it is dropped too
      const dec = rec.tokens.slice(g.prompt.length + 1).map(split), n = dec.length, S = sum(dec);
      o.prof = { decodeSteps: n };
      for (const [k, v] of Object.entries(S)) o.prof[k + "Ms"] = +(v / n).toFixed(2);
      for (const k of ["readback", "disk", "plan", "between", "head", "tail"]) o.prof[k + "Share"] = +(S[k] / S.tok).toFixed(3);
      o.prof.perCutMs = { readback: +(S.readback / S.cuts).toFixed(3), disk: +(S.disk / S.cuts).toFixed(3), plan: +(S.plan / S.cuts).toFixed(3) };
    }
    log(`${name}: ${verdict} ${JSON.stringify(o)}`);
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
