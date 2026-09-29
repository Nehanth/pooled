// room_synth.mjs for the DENSE path: the synthetic Qwen3 dense model (tests/e2e/synth_dense.mjs)
// stands in for a dense MODELS key (--model-key qwen3-0.6b | qwen3-1.7b | qwen3-4b, default
// qwen3-0.6b): its gguf, config.json and tokenizer.json URLs (room/models.js) are routed to the
// synthetic files (smollm-135m: a BF16 Llama-style safetensors, synth_dense.mjs writeSmol).
// Extra options: --layers N --untied --layer-scale X --seed S (default 5) --shape 0.6b|1.7b|4b
// (the real model's dims, hidden/FFN/heads/head_dim, with --layers of them, default 4).
// Everything else as in room_synth.mjs (below):
//
// The REAL room (p2p.html + room.js) end to end on one machine with the synthetic Qwen 3.5/3.8
// model (tests/e2e/synth.mjs) standing in for "qwen3.8-27b": 1-N headless Chromium tabs, a local
// PeerServer, real WebRTC between the tabs, WebGPU (SwiftShader works: no GPU needed), no network.
//
//   node tests/e2e/room_synth.mjs                         host + 1 worker, 2 rounds
//   node tests/e2e/room_synth.mjs --devices 3 --greedy    3 tabs, argmax sampling
//   node tests/e2e/room_synth.mjs --solo --greedy         host only
//   node tests/e2e/room_synth.mjs --compare --devices 3   greedy solo run, then greedy split run:
//                                                         compares the host's answers per round
//   options: --prompt "..." (repeatable; round r uses prompt r mod count) --rounds R (the rounds
//            are turns of ONE conversation unless --new-chat, which clicks #new-chat between them)
//            --pledges 12,5,5  --model file.gguf  --mtp echo|random  --port 8141 --signal-port 9001
//            --stop [--stop-rounds 0,2] [--stop-after 5] [--stop-selector "#ai-send"]: press Stop
//            once the answer has >= stop-after tokens
//            --timeout-s 300 (per answer) --headed
//            --dense-js FILE: serve FILE as /engine/dense.js (a patch, without touching the tree)
//            --rows 4 (stub's coop rows; also --rows-per-tab 4,8 and --wg-per-tab 256,64, host first)
//            --wg 64: the cooperative-GEMV workgroup size every tab uses (engine/autotune.js is
//            replaced by a stub returning it); --real-autotune keeps the real autotune (> 10 min
//            on SwiftShader)
//
// How it is wired:
//   * http://127.0.0.1:<port>/ serves the repo (a secure origin, so navigator.gpu exists)
//   * the PeerJS client (cdn.jsdelivr.net) is fulfilled from node_modules/peerjs/dist/peerjs.min.js
//   * the 27B's Hugging Face URL (room/models.js) is fulfilled from the synthetic GGUF with HTTP
//     Range support (206 + content-range; the room refuses anything else)
//   * any other external request is aborted
//   * --greedy: the host picks the "exact" (argmax) sampling preset in #ai-sampling. For older room
//     builds without it, --greedy-hack makes Math.random() return 0 when called from
//     room/sampling.js (aiSample then picks the top logit); other callers keep real randomness
// Pledges go through #join-gb (the 27B needs 17 GB in the room; the tiny model fits anywhere).
// The creating tab pledges the most so it is the one that deals the layers and runs the head.
//
// Needs playwright + peer + peerjs importable: NODE_PATH=<node_modules dir> or a node_modules next
// to the repo. CHROMIUM=<path> overrides the browser binary.
import fs from "fs";
import os from "os";
import path from "path";
import { spawn } from "child_process";
import { writeDense, writeSmol } from "./synth_dense.mjs";
import { MODELS } from "../../room/models.js";
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";

const argv = process.argv.slice(2);
const stats = {};   // model server counters
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const args = (k) => argv.flatMap((a, i) => (a === "--" + k ? [argv[i + 1]] : []));
const flag = (k) => argv.includes("--" + k);
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const PORT = +arg("port", 8141), SIGNAL_PORT = +arg("signal-port", 9001);
const ROUNDS = +arg("rounds", 2);
const PROMPTS = args("prompt").length ? args("prompt") : ["Write three sentences about the ocean.", "Tell me a story about a robot who learns to paint."];
const COMPARE = flag("compare");
const GREEDY = flag("greedy") || COMPARE;
const DEVICES = flag("solo") ? 1 : Math.max(1, +arg("devices", 2));
const TIMEOUT = +arg("timeout-s", 300) * 1000;
// --stop: press Stop (the Send button while an answer streams) in these rounds, default round 0
const STOP_ROUNDS = new Set(flag("stop") ? (arg("stop-rounds", "0")).split(",").map(Number) : []);
const MODEL_KEY = arg("model-key", "qwen3-0.6b");
const MODEL_URL = MODELS[MODEL_KEY].gguf || MODELS[MODEL_KEY].st, CFG_URL = MODELS[MODEL_KEY].cfg, TOK_URL = MODELS[MODEL_KEY].tok;
let cfgBytes, tokBytes;   // set in main
const PEERJS_URL = "https://cdn.jsdelivr.net/npm/peerjs@1.5.4/dist/peerjs.min.js";

function resolvePkg(name) {
  const dirs = [...(process.env.NODE_PATH || "").split(path.delimiter).filter(Boolean), path.join(ROOT, "node_modules")];
  for (const d of dirs) for (const base of [d, path.join(d, "node_modules")]) {
    if (fs.existsSync(path.join(base, name, "package.json"))) return path.join(base, name);
  }
  throw new Error(`${name} not found: set NODE_PATH to the node_modules dir that contains it`);
}
const pledgesFor = (n) => {
  const p = arg("pledges");
  if (p) return p.split(",").map(String);
  if (n === 1) return ["8"];
  return ["6", ...Array(n - 1).fill("5")];   // the creator is the biggest
};

const t0 = Date.now();
const T = () => ((Date.now() - t0) / 1000).toFixed(1).padStart(6) + "s";
const log = (...a) => console.error(T(), ...a);

// ---------------------------------------------------------------- one room session
async function session(browser, modelBytes, peerjsJs, nDev, label) {
  const size = modelBytes.length;
  const blocked = new Set();
  // --isolate: one browser context per tab, so every "device" has its own Cache API storage (as
  // real devices do); without it the tabs share one context and one weight cache
  const ctxs = [];
  const makeCtx = async () => {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  ctxs.push(ctx);
  await ctx.route("**/*", async (route) => {
    const req = route.request(), url = req.url();
    if (url.startsWith(`http://127.0.0.1:${PORT}/`) || url.startsWith(`http://127.0.0.1:${SIGNAL_PORT}/`)) return route.continue();
    const cors = { "access-control-allow-origin": "*", "access-control-allow-headers": "range", "access-control-expose-headers": "content-range, content-length, accept-ranges" };
    if (url.split("?")[0] === PEERJS_URL) return route.fulfill({ status: 200, contentType: "text/javascript", headers: cors, body: peerjsJs });
    if (url.split("?")[0] === CFG_URL) return route.fulfill({ status: 200, headers: { ...cors, "content-type": "application/json" }, body: cfgBytes });
    if (url.split("?")[0] === TOK_URL) return route.fulfill({ status: 200, headers: { ...cors, "content-type": "application/json" }, body: tokBytes });
    if (url.split("?")[0] === MODEL_URL) {
      if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers: { ...cors, "access-control-allow-methods": "GET, HEAD" } });
      const m = /bytes=(\d+)-(\d*)/.exec((await req.allHeaders()).range || "");
      if (!m) return route.fulfill({ status: 200, headers: { ...cors, "content-type": "application/octet-stream", "accept-ranges": "bytes" }, body: modelBytes });
      const lo = +m[1], hi = m[2] ? Math.min(+m[2], size - 1) : size - 1;
      // --ttfb MS: every range request waits this long before its first byte, like a CDN would
      if (+arg("ttfb", 0)) await new Promise((r) => setTimeout(r, +arg("ttfb", 0)));
      stats.requests = (stats.requests || 0) + 1;
      if (lo >= size) return route.fulfill({ status: 416, headers: { ...cors, "content-range": `bytes */${size}` } });
      return route.fulfill({ status: 206, headers: { ...cors, "content-type": "application/octet-stream", "accept-ranges": "bytes", "content-range": `bytes ${lo}-${hi}/${size}` },
        body: modelBytes.subarray(lo, hi + 1) });
    }
    // web fonts: an empty stylesheet (no network, and no console error for the fallback)
    if (url.startsWith("https://fonts.googleapis.com/")) return route.fulfill({ status: 200, contentType: "text/css", body: "" });
    if (!blocked.has(url)) { blocked.add(url); log(`[${label}] blocked external request ${url.slice(0, 120)}`); }
    return route.abort();
  });
  // fallback for room builds without the "exact" sampling preset (see the header comment)
  if (GREEDY && flag("greedy-hack")) await ctx.addInitScript(() => {
    const rnd = Math.random;
    Math.random = function () { const s = new Error().stack || ""; return /\/room\/sampling\.js/.test(s) ? 0 : rnd(); };
  });
  return ctx;
  };
  const shared = flag("isolate") ? null : await makeCtx();
  const names = ["host", ...Array.from({ length: nDev - 1 }, (_, i) => (nDev === 2 ? "worker" : "worker" + (i + 1)))];
  const tabs = {};
  for (const n of names) tabs[n] = await (shared || await makeCtx()).newPage();
  // --rows-per-tab 4,8,8 / --wg-per-tab 256,64: what autotuneCoop returns in each tab (host first),
  // like a room of different GPUs (engine/autotune.js picks among [256|128|64] x [4|8] per device)
  const perTab = (k) => (arg(k) || "").split(",").filter(Boolean).map(Number);
  for (const [i, n] of names.entries()) {
    const r = perTab("rows-per-tab")[i], w = perTab("wg-per-tab")[i];
    if (r || w) await tabs[n].addInitScript(([r, w]) => { if (r) globalThis.__e2eRows = r; if (w) globalThis.__e2eWG = w; }, [r, w]);
  }
  const errs = Object.fromEntries(names.map((n) => [n, []]));
  for (const [n, p] of Object.entries(tabs)) {
    p.on("console", (m) => { if ((m.type() === "error" || m.type() === "warning") && !/Could not connect to peer/.test(m.text())) errs[n].push(`${m.type()}: ${m.text().slice(0, 240)}`); });
    p.on("pageerror", (e) => errs[n].push("pageerror: " + String(e).slice(0, 240)));
  }
  // Room settings are segmented controls over hidden selects: set the select and fire its change
  const setSelect = (p, id, value) => p.evaluate(([id, value]) => {
    const s = document.getElementById(id);
    if (![...s.options].some((o) => o.value === value)) throw new Error(`#${id} has no option ${value}`);
    s.value = value; s.dispatchEvent(new Event("change", { bubbles: true }));
  }, [id, value]);
  const status = (p) => p.evaluate(() => [document.getElementById("ai-status")?.textContent, document.getElementById("ldg-sub")?.textContent].join(" | "));
  const pledges = pledgesFor(nDev);
  const out = { label, devices: nDev, rounds: [], errors: errs };
  const base = `http://127.0.0.1:${PORT}/p2p.html?split=memory&signal=127.0.0.1:${SIGNAL_PORT}`;
  const maxNewQ = arg("max-new", ""); const baseQ = (maxNewQ ? `&maxnew=${maxNewQ}` : "") + (arg("netlag") ? `&netlag=${arg("netlag")}` : "") + (arg("query") ? `&${arg("query")}` : "");
  try {
    for (const p of Object.values(tabs)) await p.goto(base + baseQ);
    for (const p of Object.values(tabs)) await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
    for (const [i, n] of names.entries()) { await tabs[n].fill("#name-input", n + "-e2e"); await tabs[n].fill("#join-gb", pledges[i] || "5"); }
    await tabs.host.click("#create-btn");
    await tabs.host.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("room-badge").textContent), null, { timeout: 30000 });
    const code = (await tabs.host.textContent("#room-badge")).trim().match(/[A-Z0-9]{4}/)[0];
    log(`[${label}] room ${code}, pledges ${pledges.slice(0, nDev).join("+")} GB`);
    for (const n of names.slice(1)) { await tabs[n].fill("#code-input", code); await tabs[n].click("#join-btn"); await tabs[n].waitForTimeout(200); }
    for (const p of Object.values(tabs)) await p.waitForFunction((k) => document.querySelectorAll(".peer-card").length >= k, nDev, { timeout: 60000 });
    log(`[${label}] ${nDev} device(s) in the room`);
    // --virtual N: the host adds N virtual devices (iframes of the room on the same machine)
    if (+arg("virtual", 0)) {
      for (let i = 0; i < +arg("virtual", 0); i++) await tabs.host.click("#add-virtual");
      await tabs.host.waitForFunction((k) => document.querySelectorAll(".peer-card").length >= k, nDev + +arg("virtual", 0), { timeout: 90000 });
      await tabs.host.waitForTimeout(2000);
      log(`[${label}] ${arg("virtual")} virtual device(s) joined: ${await tabs.host.$$eval(".peer-card .pname", (e) => e.map((x) => x.textContent).join(", "))}`);
    }
    if (nDev > 1) await tabs.host.waitForTimeout(2000);   // stripe connections
    if (GREEDY) {   // the host's sampling preset "exact" = argmax (room/sampling.js)
      const has = await tabs.host.evaluate(() => [...(document.getElementById("ai-sampling")?.options || [])].some((o) => o.value === "exact"));
      if (has) await setSelect(tabs.host, "ai-sampling", "exact");
      else if (!flag("greedy-hack")) throw new Error("no #ai-sampling 'exact' preset in this room build: rerun with --greedy-hack");
    }
    await setSelect(tabs.host, "ai-model", MODEL_KEY);
    await tabs.host.waitForFunction(() => !document.getElementById("ai-start").disabled, null, { timeout: 20000 });
    const tLoad = Date.now();
    if (arg("split")) await setSelect(tabs.host, "ai-split", arg("split"));
    await tabs.host.click("#ai-start");
    log(`[${label}] start pressed`);
    const poll = setInterval(async () => { for (const [n, p] of Object.entries(tabs)) { try { log(`[${label}] ${n}: ${(await status(p)).slice(0, 140)}`); } catch {} } }, 15000);
    try {
      // online, or a tab says failed:
      await Promise.all(Object.entries(tabs).map(([n, p]) => p.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online")
        || /^failed:|FAILED|GPU error/.test(document.getElementById("ai-status").textContent), null, { timeout: 600000 })));
    } finally { clearInterval(poll); }
    for (const [n, p] of Object.entries(tabs)) {
      const st = await p.textContent("#ai-status");
      if (!(await p.evaluate(() => document.getElementById("ai-panel").classList.contains("online")))) throw new Error(`${n} did not come online: ${st}`);
    }
    out.loadS = +((Date.now() - tLoad) / 1000).toFixed(1);
    out.online = await tabs.host.textContent("#ai-status");
    out.split = await tabs.host.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => /layer split/.test(t)).slice(-1)[0] || "");
    log(`[${label}] online after ${out.loadS}s: ${out.online}`);
    if (out.split) log(`[${label}] ${out.split}`);

    for (let r = 0; r < ROUNDS; r++) {
      const prompt = PROMPTS[r % PROMPTS.length];
      const nStats = await tabs.host.evaluate(() => document.querySelectorAll(".m.bot .stats").length);
      const nBots = await tabs.host.evaluate(() => document.querySelectorAll(".m.bot").length);
      const tr = Date.now();
      await tabs.host.fill("#ai-prompt", prompt);
      await tabs.host.click("#ai-send");
      // --queue: a worker asks too while the host's question is being answered; it waits in the
      // host's queue and is answered next, so this round has two answers
      let expect = 1;
      const asker2 = tabs[names[1]];
      if (flag("queue") && asker2) {
        await tabs.host.waitForFunction(() => /generating|prefill/.test(document.getElementById("ai-status").textContent), null, { timeout: TIMEOUT });
        await asker2.fill("#ai-prompt", "And a second question from the worker?");
        await asker2.click("#ai-send");
        expect = 2;
        log(`[${label}] round ${r}: worker1 queued a question`);
      }
      if (STOP_ROUNDS.has(r)) {   // Send turns into Stop while an answer streams
        const after = +arg("stop-after", 5);
        await tabs.host.waitForFunction((k) => { const m = /generating… (\d+) tok/.exec(document.getElementById("ai-status").textContent); return m && +m[1] >= k; }, after, { timeout: TIMEOUT });
        await tabs.host.click(arg("stop-selector", "#ai-send"));
        log(`[${label}] round ${r}: pressed stop after >= ${after} tokens`);
      }
      await tabs.host.waitForFunction(([k, e]) => document.querySelectorAll(".m.bot .stats").length >= k + e
        || /^generation failed/.test(document.getElementById("ai-status").textContent), [nStats, expect], { timeout: TIMEOUT });
      // --continue: while the answer stopped at the length cap, press Continue (the round's answer is
      // then every bot bubble since the question, joined)
      for (let c = 0; flag("continue") && c < 20 && await tabs.host.isVisible("#continue-btn"); c++) {
        const k = await tabs.host.evaluate(() => document.querySelectorAll(".m.bot .stats").length);
        await tabs.host.click("#continue-btn");
        await tabs.host.waitForFunction((k) => document.querySelectorAll(".m.bot .stats").length > k, k, { timeout: TIMEOUT });
        log(`[${label}] round ${r}: continued (${c + 1})`);
      }
      const nStatsEnd = await tabs.host.evaluate(() => document.querySelectorAll(".m.bot .stats").length);
      // workers get their copy of the answer over the host link: give it a moment
      for (const p of Object.values(tabs).slice(1)) await p.waitForFunction((k) => document.querySelectorAll(".m.bot .stats").length >= k, nStatsEnd, { timeout: 20000 }).catch(() => {});
      const per = {};
      for (const [n, p] of Object.entries(tabs)) per[n] = await p.evaluate((nb) => {
        const bots = [...document.querySelectorAll(".m.bot")];
        const b = bots[bots.length - 1];
        return { answer: bots.slice(nb).map((x) => x.querySelector(".bubble")?.textContent || "").join(""), stats: b?.querySelector(".stats")?.textContent || "", status: document.getElementById("ai-status").textContent };
      }, nBots);
      const secs = ((Date.now() - tr) / 1000).toFixed(1);
      out.rounds.push({ prompt, secs: +secs, per });
      log(`[${label}] round ${r} (${secs}s) host status: ${per.host.status}`);
      for (const [n, v] of Object.entries(per)) console.log(`[${label}] round ${r} ${n}: stats=${JSON.stringify(v.stats)}\n    answer=${JSON.stringify(v.answer.slice(0, 400))}`);
      await tabs.host.waitForTimeout(500);
      // --redeal-after R [--redeal-split speed]: after round R, re-deal the layers (the conversation
      // carries on: the next question re-prefills it on the new split)
      if (arg("redeal-after") !== undefined && +arg("redeal-after") === r && r + 1 < ROUNDS) {
        const req0 = stats.requests || 0;
        if (arg("redeal-split")) await setSelect(tabs.host, "ai-split", arg("redeal-split"));
        await tabs.host.evaluate(() => { const b = document.getElementById("ai-redeal"); b.hidden = false; b.click(); });
        await tabs.host.waitForFunction(() => /cluster online/.test(document.getElementById("ai-status").textContent), null, { timeout: TIMEOUT });
        const split = await tabs.host.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => /layer split/.test(t)).pop());
        log(`[${label}] re-dealt after round ${r}: ${split} \u00b7 ${(stats.requests || 0) - req0} range requests to the model host during the re-deal`);
        for (const [n, p] of Object.entries(tabs)) { const w = await p.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => /came from devices/.test(t)).pop()); if (w) log(`[${label}] ${n}: ${w}`); }
      }
      // --reload-after R: reload the host's tab after round R and resume the room from the join
      // screen; the guests wait for it, it deals the layers again, the conversation continues
      if (arg("reload-after") !== undefined && +arg("reload-after") === r && r + 1 < ROUNDS) {
        const t0 = Date.now();
        await tabs.host.reload();
        // the join screen no longer shows a resume note (removed from the UI); --reload-after has no way back in
        throw new Error("--reload-after: the join screen's resume note was removed, so a reloaded host can't pick its room back up");
        await tabs.host.waitForFunction(() => /cluster online/.test(document.getElementById("ai-status").textContent), null, { timeout: TIMEOUT });
          for (const p of Object.values(tabs).slice(1)) await p.waitForFunction(() => document.getElementById("ai-row").style.display === "flex", null, { timeout: 60000 });
        log(`[${label}] host reloaded and resumed after round ${r} in ${((Date.now() - t0) / 1000).toFixed(1)}s: ${await tabs.host.textContent("#ai-status")}`);
      }
      if (flag("new-chat") && r + 1 < ROUNDS) {
        await tabs.host.click("#new-chat").catch((e) => log(`[${label}] new chat: ${String(e).slice(0, 100)}`));
        await tabs.host.waitForTimeout(500);
      }
    }
    // --social: a guest sees the copy button under the last answer, and types; the host sees the note
    if (flag("social") && names[1]) {
      const g = tabs[names[1]];
      if (!(await g.isVisible("#ai-output .m.bot:last-of-type .copy-ans"))) throw new Error("no copy button under the guest's last answer");
      await g.type("#ai-prompt", "hmm");
      await tabs.host.waitForFunction(() => /is typing/.test(document.getElementById("typing-note").textContent), null, { timeout: 10000 });
      const hostSees = await tabs.host.textContent("#typing-note");
      log(`[${label}] social: the guest has a copy button; host sees "${hostSees}"`);
      await g.fill("#ai-prompt", "");
    }
    // --screenshot PREFIX: the host's and the first guest's whole page after the last round
    if (arg("screenshot")) {
      await tabs.host.setViewportSize({ width: 1280, height: 800 });
      await tabs.host.screenshot({ path: arg("screenshot") + "-host.png" });
      if (names[1]) { await tabs[names[1]].setViewportSize({ width: 390, height: 844 }); await tabs[names[1]].screenshot({ path: arg("screenshot") + "-guest.png" }); }
      log(`[${label}] screenshots: ${arg("screenshot")}-host.png, -guest.png`);
    }
    // --card PATH: open the room card and save a screenshot of it
    if (arg("card")) {
      await tabs.host.click("#room-menu > summary");   // the room card lives in the room menu
      await tabs.host.click("#card-btn");
      await tabs.host.locator("#card-canvas").screenshot({ path: arg("card") });
      log(`[${label}] room card saved to ${arg("card")}`);
      await tabs.host.click("#card-close");
    }
    // --regen: press Regenerate after the last round; with greedy sampling the answer must repeat
    if (flag("regen")) {
      const k = await tabs.host.evaluate(() => document.querySelectorAll(".m.bot .stats").length);
      const prev = await tabs.host.evaluate(() => { const b = [...document.querySelectorAll(".m.bot")].pop(); return b.querySelector(".bubble").textContent; });
      await tabs.host.click("#regen-btn");
      await tabs.host.waitForFunction((k) => document.querySelectorAll(".m.bot .stats").length > k, k, { timeout: TIMEOUT });
      const again = await tabs.host.evaluate(() => { const b = [...document.querySelectorAll(".m.bot")].pop(); return b.querySelector(".bubble").textContent; });
      out.regenSame = again === prev;
      log(`[${label}] regenerate: ${out.regenSame ? "same answer (greedy)" : "DIFFERENT answer"} \u00b7 ${await tabs.host.textContent("#ai-status")}`);
      if (GREEDY && !out.regenSame) throw new Error("regenerate under greedy sampling gave a different answer");
    }
    out.roomLog = {};
    for (const [n, p] of Object.entries(tabs)) out.roomLog[n] = await p.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => t.includes("⚠")).map((t) => t.slice(0, 240)));
    out.crumb = {};
    for (const [n, p] of Object.entries(tabs)) out.crumb[n] = await p.evaluate(() => { try { return JSON.parse(localStorage.getItem("pooled-crumb") || "null")?.s; } catch { return null; } });
    // pass: every answer finished, every tab saw it, no console errors, no ⚠ lines in any room log
    out.ok = out.rounds.every((r) => /^ready/.test(r.per.host.status) && r.per.host.stats && !/failed/.test(r.per.host.stats)
      && Object.values(r.per).every((v) => v.answer === r.per.host.answer))
      && Object.values(errs).every((e) => !e.some((x) => x.startsWith("error") || x.startsWith("pageerror")))
      && Object.values(out.roomLog).every((w) => !w.length);
  } catch (e) {
    out.ok = false; out.failure = String(e).slice(0, 500);
    log(`[${label}] FAILED: ${out.failure}`);
    for (const [n, p] of Object.entries(tabs)) { try { log(`[${label}] ${n}: ${await status(p)}`); } catch {} }
    try { for (const [n, p] of Object.entries(tabs)) log(`[${label}] ${n} room log: ${JSON.stringify((await p.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent))).slice(-6))}`); } catch {}
  } finally {
    for (const c of ctxs) await c.close();
  }
  return out;
}

// ---------------------------------------------------------------- main
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-room-synth-"));
const SHAPES = { "0.6b": { dim: 1024, inter: 3072, nH: 16, nKV: 8, hd: 128 }, "1.7b": { dim: 2048, inter: 6144, nH: 16, nKV: 8, hd: 128 }, "4b": { dim: 2560, inter: 9728, nH: 32, nKV: 8, hd: 128 } };
const shape = arg("shape") ? { ...SHAPES[arg("shape")], layers: 4 } : {};
const dense = (MODELS[MODEL_KEY].kind === "safetensors" ? writeSmol : writeDense)(tmp, { ...shape, seed: +arg("seed", 5),   // seed 5: no greedy near-ties (< 0.1 logit) on the default prompts, so f16 wire rounding cannot flip a token
   layers: +arg("layers", shape.layers || 8), untied: flag("untied"), ...(arg("layer-scale") ? { layerScale: +arg("layer-scale") } : {}) });
const modelFile = dense.files.gguf || dense.files.st;
cfgBytes = fs.readFileSync(dense.files.cfg); tokBytes = fs.readFileSync(dense.files.tok);
const modelBytes = fs.readFileSync(modelFile);
const peerjsJs = fs.readFileSync(path.join(resolvePkg("peerjs"), "dist/peerjs.min.js"));
const peerBin = path.join(resolvePkg("peer"), "dist/bin/peerjs.js");
// --host: without it the server binds :: and dies on machines without IPv6
const peerServer = spawn(process.execPath, [peerBin, "--port", String(SIGNAL_PORT), "--host", "127.0.0.1", "--path", "/"], { stdio: ["ignore", "ignore", "pipe"] });
let peerErr = "";
peerServer.stderr.on("data", (d) => { peerErr += d; });
// engine/autotune.js is swapped for a fixed pick unless --real-autotune: the real one times a
// 17408x5120 GEMV (the 27B's FFN) for five configs, which takes more than 10 minutes on SwiftShader
const extra = {};
if (!flag("real-autotune")) {
  const stub = path.join(tmp, "autotune.js");
  fs.writeFileSync(stub, `// e2e stub (tests/e2e/room_synth.mjs): fixed cooperative-GEMV config, no timing\nexport async function autotuneCoop() { return { wg: globalThis.__e2eWG || ${+arg("wg", 64)}, rows: globalThis.__e2eRows || ${+arg("rows", 4)}, results: [], stub: true }; }\n`);
  extra["/engine/autotune.js"] = stub;
}
if (arg("dense-js")) extra["/engine/dense.js"] = path.resolve(arg("dense-js"));   // --dense-js FILE: try a patched engine
const srv = serveRepo(PORT, extra);
for (let i = 0; ; i++) {   // wait until the PeerServer answers
  const up = await fetch(`http://127.0.0.1:${SIGNAL_PORT}/peerjs/id`).then((r) => r.ok, () => false);
  if (up) break;
  if (i > 50 || peerServer.exitCode !== null) { console.error("PeerServer did not start:", peerErr.slice(0, 400)); process.exit(2); }
  await new Promise((r) => setTimeout(r, 200));
}
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), headless: !flag("headed"),
  args: [...GPU_ARGS.filter((a) => !flag("headed") || a !== "--headless=new"), "--allow-loopback-in-peer-connection", "--disable-features=WebRtcHideLocalIpsWithMdns"] });
let code = 0;
try {
  log(`model ${modelFile} (${(modelBytes.length / 2 ** 20).toFixed(1)} MB) as ${MODEL_KEY}; greedy=${GREEDY}`);
  const runs = COMPARE ? [[1, "solo"], [Math.max(2, DEVICES), `split${Math.max(2, DEVICES)}`]] : [[DEVICES, DEVICES === 1 ? "solo" : `split${DEVICES}`]];
  const results = [];
  for (const [n, label] of runs) results.push(await session(browser, modelBytes, peerjsJs, n, label));
  const summary = results.map((r) => ({ label: r.label, ok: r.ok, loadS: r.loadS, split: r.split, failure: r.failure,
    rounds: r.rounds.map((x) => ({ secs: x.secs, stats: x.per.host.stats, answerChars: x.per.host.answer.length,
      workersSeeSameAnswer: Object.values(x.per).every((v) => v.answer === x.per.host.answer) })),
    crumb: r.crumb, roomWarnings: r.roomLog, consoleErrors: Object.fromEntries(Object.entries(r.errors).filter(([, v]) => v.length).map(([k, v]) => [k, v.slice(0, 5)])) }));
  console.log(JSON.stringify(summary, null, 1));
  let ok = results.every((r) => r.ok);
  if (COMPARE && results.length === 2 && results.every((r) => r.ok)) {
    for (let i = 0; i < ROUNDS; i++) {
      const a = results[0].rounds[i]?.per.host.answer, b = results[1].rounds[i]?.per.host.answer;
      const same = a === b;
      let d = 0; while (d < Math.min(a.length, b.length) && a[d] === b[d]) d++;
      console.log(`${same ? "PASS" : "DIFF"} round ${i}: solo vs ${results[1].label} greedy answers ${same ? "identical" : `differ at char ${d}: solo ${JSON.stringify(a.slice(d, d + 40))} vs split ${JSON.stringify(b.slice(d, d + 40))}`}`);
      if (!same) ok = false;
    }
  }
  console.log(ok ? "ROOM SYNTH PASS" : "ROOM SYNTH FAIL");
  code = ok ? 0 : 1;
} catch (e) {
  console.error("FAILED:", e.stack || e);
  code = 2;
} finally {
  await browser.close(); srv.close(); peerServer.kill();
  fs.rmSync(tmp, { recursive: true, force: true });
}
process.exit(code);
