// Engine end to end on the synthetic Qwen 3.5/3.8 model (tests/e2e/synth.mjs), in headless
// Chromium's WebGPU (works on SwiftShader, no GPU needed). Greedy (argmax) everywhere:
//
//   solo  plain   prefillTokens + forwardToken loop                         (the reference stream)
//   solo  spec    specStep(argmax, K) for K in --ks                          == plain
//   split plain   2 and 3 engines chained with embedRun/runHidden/headFromHidden, batched
//                 prefill via embedRunBatch/runHiddenBatch the way room.js does it  == solo plain
//   split spec    same chain, specStep with runTrunk/onReject (room protocol)  == solo plain
//   f16 wire      split plain and split spec with every hop rounded to f16 (room/wire.js does
//                 this): spec == plain must still hold; divergence from solo is only reported
//   multi-turn    prefill A, decode, prefill continuation B without reset, decode
//                 == reset + prefill(A + answer + B) + decode   (plain, spec, split spec)
//
//   node tests/e2e/engine_synth.mjs [--tokens 40] [--cols 4,16] [--ks 1,3,7] [--model file.gguf]
//                                   [--mtp echo|random] [--wg 64] [--no-split] [--keep]
//
// --wg: cooperative-GEMV workgroup size (room.js gets it from autotuneCoop). 64 by default because
// SwiftShader runs 256-thread workgroups ~7x slower; any value must give the same stream.
//
// Needs playwright importable: NODE_PATH=<dir containing node_modules/playwright> or a
// node_modules next to the repo. CHROMIUM=<path> overrides the browser binary.
import http from "http";
import fs from "fs";
import os from "os";
import path from "path";
import { createRequire } from "module";
import { writeSynth, SYNTH_MOE } from "./synth.mjs";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const flag = (k) => argv.includes("--" + k);
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");

export async function loadPlaywright() {
  try { return await import("playwright"); } catch {}
  const dirs = [...(process.env.NODE_PATH || "").split(path.delimiter).filter(Boolean), path.join(ROOT, "node_modules")];
  for (const d of dirs) {
    for (const base of [d, path.join(d, "node_modules")]) {
      try { return createRequire(path.join(base, "noop.js"))("playwright"); } catch {}
    }
  }
  throw new Error("playwright not found: set NODE_PATH to the node_modules dir that contains it");
}
export function chromiumPath() {
  if (process.env.CHROMIUM) return process.env.CHROMIUM;
  for (const p of ["/opt/pw-browsers/chromium-1194/chrome-linux/chrome"]) if (fs.existsSync(p)) return p;
  return undefined;   // playwright's own download
}
// E2E_GPU=real: the machine's GPU (as tests/bench/chrome_bench.mjs launches it) instead of SwiftShader,
// for hosts where SwiftShader has no adapter (the GB10)
export const GPU_ARGS = process.env.E2E_GPU === "real"
  ? ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan", "--ignore-gpu-blocklist"]
  : ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"];
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };
// static server for the repo root; extra: { "/__x": "/abs/file" } serves files outside it
// headers(url): extra response headers for a path (preview_browser.mjs: the relay's, from vercel.json)
export function serveRepo(port, extra = {}, { headers = () => null } = {}) {
  return http.createServer((q, r) => {
    const url = decodeURIComponent(q.url.split("?")[0]);
    if (url === "/__blank.html") { r.setHeader("content-type", MIME[".html"]); r.end("<!doctype html><title>e2e</title><body>e2e</body>"); return; }
    if (url === "/favicon.ico" && !fs.existsSync(path.join(ROOT, url))) { r.statusCode = 204; r.end(); return; }
    const p = extra[url] || path.join(ROOT, url);
    if (!extra[url] && !p.startsWith(ROOT)) { r.statusCode = 404; r.end(); return; }
    if (!fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
    r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream");
    r.setHeader("cache-control", "no-store");
    for (const [k, v] of Object.entries(headers(url) || {})) r.setHeader(k, v);
    // Range requests (tests/eval/eval.html streams a model too big for a tab's heap tensor by tensor)
    const m = /^bytes=(\d+)-(\d*)$/.exec(q.headers.range || "");
    if (m) {
      const size = fs.statSync(p).size, a = +m[1], b = Math.min(m[2] ? +m[2] : size - 1, size - 1);
      r.statusCode = 206; r.setHeader("content-range", `bytes ${a}-${b}/${size}`); r.setHeader("content-length", b - a + 1);
      fs.createReadStream(p, { start: a, end: b }).pipe(r);
      return;
    }
    fs.createReadStream(p).pipe(r);
  }).listen(port, "127.0.0.1");
}

// ---------------------------------------------------------------- in-page test body
async function pageMain({ modelUrl, tokens: N, cols, ks, wg, noSplit, kvQ8 }) {
  const { Qwen35Engine } = await import("/engine/qwen35.js");
  const { parseGGUFHeader, qwen35Weights, tokenizerFromGGUF, f32ToF16, f16ToF32, GGML_EMBED } = await import("/engine/gguf.js");
  const { makeTokenizer, argmax } = await import("/engine/engine.js");
  const out = [];
  const say = (s) => { out.push(s); console.log(s); };
  const results = [];
  const check = (name, ok, detail = "") => { results.push({ name, ok }); say(`${ok ? "PASS" : "FAIL"} ${name}${detail ? "  " + detail : ""}`); };
  const info = (s) => say("     " + s);

  const buf = await (await fetch(modelUrl)).arrayBuffer();
  const G = parseGGUFHeader(buf);
  const bytesOf = async (i) => new Uint8Array(buf, i.byteOffset, i.byteLength).slice();
  const tok = makeTokenizer(tokenizerFromGGUF(G.meta));
  const V = tok.vocab;
  const L = G.meta["qwen35.block_count"] - (G.meta["qwen35.nextn_predict_layers"] || 0);
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  const gpuErrors = [];
  device.addEventListener("uncapturederror", (e) => { gpuErrors.push(e.error?.message); console.error("GPU ERROR: " + e.error?.message); });
  info(`adapter: ${JSON.stringify(adapter.info?.vendor)} ${JSON.stringify(adapter.info?.architecture)} subgroups=${adapter.features.has("subgroups")} · model: ${L} trunk layers + nextn, dim ${G.meta["qwen35.embedding_length"]}, vocab ${G.meta["tokenizer.ggml.tokens"].length}, ${(buf.byteLength / 2 ** 20).toFixed(1)} MB`);

  // the room's chat template (room.js aiGenerate)
  const chat = (text) => [V["<|im_start|>"], ...tok.encode("user\n" + text), V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n"),
    V["<think>"], ...tok.encode("\n\n"), V["</think>"], ...tok.encode("\n\n")];
  const promptA = chat("Write three sentences about the ocean.");
  // turn 2's prompt, padded so its prefill (all but the last token) leaves a single-token tail of
  // 3 after the 4-wide batches: that is the path whose draft fill has its own position arithmetic
  const contB = (last) => {
    for (let k = 0; ; k++) {
      const B = [last, V["<|im_end|>"], ...tok.encode("\n"), ...chat("Now tell me a story about a robot who learns to paint" + "!".repeat(k + 1))];
      if ((B.length - 1) % 4 === 3) return B;
    }
  };
  const eos = new Set([V["<|im_end|>"], V["<|endoftext|>"]]);
  const f16 = (h) => { const o = new Float32Array(h.length); for (let i = 0; i < h.length; i++) o[i] = f16ToF32(f32ToF16(h[i])); return o; };
  const same = (a, b) => a.length === b.length && a.every((t, i) => t === b[i]);
  const firstDiff = (a, b) => { const n = Math.min(a.length, b.length); for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i; return a.length === b.length ? -1 : n; };
  const show = (ids) => JSON.stringify(tok.decode(ids).slice(0, 90));

  for (const NC of cols) {
    const eopts = { maxSeq: 512, batchCols: NC, coopRowsB: NC >= 16 ? 1 : 4, coopWG: wg, kvQ8 };
    const mk = async (lo, hi, head) => Qwen35Engine.create({ device, meta: G.meta, layerRange: [lo, hi], hasEmbed: head, hasHead: head,
      vocab: G.tensors[GGML_EMBED].shape[0], ...eopts,
      weights: await qwen35Weights(G, bytesOf, { lo, hi, hasEmbed: head, hasHead: head, mtp: head }) });
    say(`--- batchCols ${NC} (${NC >= 16 ? "room config: batchCols 16, coopRowsB 1" : "test config"}), coopWG ${wg} ---`);
    const t0 = performance.now();
    const solo = await mk(0, L, true);
    info(`solo engine built in ${((performance.now() - t0) / 1000).toFixed(1)}s; mtp=${!!solo.mtp} gemm=${solo.gemmOn}`);

    // ---------- solo ----------
    const soloPlain = async (prompt, n, fresh = true) => {
      if (fresh) solo.reset();
      solo.mtpFill = false;
      await solo.prefillTokens(prompt.slice(0, -1));
      let next = argmax(await solo.forwardToken(prompt[prompt.length - 1]));
      const o = [next];
      while (o.length < n) { next = argmax(await solo.forwardToken(next)); o.push(next); }
      return o;
    };
    const soloSpec = async (prompt, n, K, fresh = true) => {
      if (fresh) solo.reset();
      solo.mtpFill = true; solo.mtp.stats = { drafts: 0, accepted: 0 };
      await solo.prefillTokens(prompt.slice(0, -1));
      let next = argmax(await solo.forwardToken(prompt[prompt.length - 1]));
      const o = [next];
      while (o.length < n) { const got = await solo.specStep(next, argmax, K); o.push(...got); next = o[o.length - 1]; }
      return { toks: o.slice(0, n), all: o, st: { ...solo.mtp.stats } };
    };
    let t = performance.now();
    const plain = await soloPlain(promptA, N);
    const plainMs = performance.now() - t;
    const eosAt = plain.findIndex((x) => eos.has(x));
    check(`[NC=${NC}] solo plain decode: ${N} tokens, no NaN, no early EOS`, plain.length === N && plain.every(Number.isInteger) && (eosAt < 0 || eosAt > 40),
      `${(N / (plainMs / 1000)).toFixed(1)} tok/s, eos at ${eosAt}, distinct ${new Set(plain).size}, text ${show(plain)}`);
    const plainAgain = await soloPlain(promptA, N);
    check(`[NC=${NC}] solo plain is deterministic across reset()`, same(plain, plainAgain), same(plain, plainAgain) ? "" : `first diff at ${firstDiff(plain, plainAgain)}`);
    for (const K of ks) {
      t = performance.now();
      const s = await soloSpec(promptA, N, K);
      const ok = same(plain, s.toks);
      check(`[NC=${NC}] solo spec K=${K} == plain`, ok, `accepted ${s.st.accepted}/${s.st.drafts}, ${(s.all.length / ((performance.now() - t) / 1000)).toFixed(1)} tok/s` + (ok ? "" : `, first diff at ${firstDiff(plain, s.toks)}: ${show(s.toks)}`));
    }

    // ---------- split (room protocol, emulated in one page) ----------
    const splits = noSplit ? [] : [[0, Math.floor(L / 2), L], [0, 3, 6, L]].filter((c) => c.every((x, i) => i === 0 || x > c[i - 1]));
    for (const cut of splits) for (const wire16 of [false, true]) {
      const hops = [];
      const host = await mk(cut[0], cut[1], true);
      for (let j = 1; j < cut.length - 1; j++) hops.push(await mk(cut[j], cut[j + 1], false));
      const tag = `[NC=${NC}] split ${cut.slice(0, -1).map((a, j) => `${a}-${cut[j + 1] - 1}`).join(" | ")}${wire16 ? " f16-wire" : ""}`;
      const W = wire16 ? f16 : (h) => h;
      const chainOne = async (id, pos) => { let h = W(await host.embedRun(id, pos)); for (const e of hops) h = W(await e.runHidden(h, pos)); return h; };
      const chainBatch = async (ids, pos, snap) => {   // host batches of NC, each hop the same (room.js runTrunk / ai-hidden-b)
        const n = ids.length, D = host.dims.dim, hb = new Float32Array(n * D);
        for (let c = 0; c < n; c += host.NC) { const m = Math.min(host.NC, n - c); hb.set(await host.embedRunBatch(ids.slice(c, c + m), pos + c, snap ? { base: c, total: n } : false), c * D); }
        let h = W(hb);
        for (const e of hops) {
          const o = new Float32Array(n * D);
          for (let c = 0; c < n; c += e.NC) { const m = Math.min(e.NC, n - c); o.set(await e.runHiddenBatch(h.subarray(c * D, (c + m) * D), pos + c, snap ? { base: c, total: n } : false), c * D); }
          h = W(o);
        }
        return h;
      };
      let lastHidden = null;
      // room.js split prefill: widths NC, 8, 4 with up to 16 tokens per round, then single tokens
      const prefill = async (ids, pos0) => {
        let i = 0, pos = pos0, logits = null;
        for (const Wd of [host.NC, ...[8, 4].filter((w) => w < host.NC)]) while (ids.length - 1 - i >= Wd) {
          const nChunks = Math.max(1, Math.min(Math.floor(16 / Wd), Math.floor((ids.length - 1 - i) / Wd)));
          // the room runs the host's chunks first, then ships all of them down the chain at once
          const n = nChunks * Wd, D = host.dims.dim, hb = new Float32Array(n * D);
          for (let c = 0; c < nChunks; c++) hb.set(await host.embedRunBatch(ids.slice(i + c * Wd, i + (c + 1) * Wd), pos + c * Wd), c * Wd * D);
          let h = W(hb);
          for (const e of hops) {
            const o = new Float32Array(n * D);
            for (let c = 0; c < n; c += e.NC) { const m = Math.min(e.NC, n - c); o.set(await e.runHiddenBatch(h.subarray(c * D, (c + m) * D), pos + c, false), c * D); }
            h = W(o);
          }
          pos += n; i += n;
        }
        for (; i < ids.length; i++, pos++) {
          const h = await chainOne(ids[i], pos);
          lastHidden = h;
          if (i === ids.length - 1) logits = await host.headFromHidden(h);
        }
        return { logits, pos };
      };
      const resetAll = () => { host.reset(); for (const e of hops) e.reset(); };
      const splitPlain = async (prompt, n, pos0 = 0) => {
        let { logits, pos } = await prefill(prompt, pos0);
        let next = argmax(logits); const o = [next];
        while (o.length < n) { const h = await chainOne(next, pos++); lastHidden = h; next = argmax(await host.headFromHidden(h)); o.push(next); }
        return { toks: o, pos };
      };
      const specOpts = {
        runTrunk: (tokens, p) => chainBatch(tokens, p, true),
        onReject: async (k) => { for (const e of hops) e.restoreDN(k); },
      };
      const splitSpec = async (prompt, n, K, pos0 = 0) => {
        host.mtp.stats = { drafts: 0, accepted: 0 };
        const { logits, pos } = await prefill(prompt, pos0);
        host.setHidden(lastHidden); host.pos = pos;
        let next = argmax(logits); const o = [next];
        while (o.length < n) { const got = await host.specStep(next, argmax, K, specOpts); o.push(...got); next = o[o.length - 1]; }
        return { toks: o.slice(0, n), all: o, pos: host.pos, st: { ...host.mtp.stats } };
      };
      resetAll();
      const sp = await splitPlain(promptA, N);
      if (!wire16) check(`${tag}: plain == solo plain`, same(plain, sp.toks), same(plain, sp.toks) ? "" : `first diff at ${firstDiff(plain, sp.toks)}: ${show(sp.toks)}`);
      else info(`${tag}: plain vs solo: ${same(plain, sp.toks) ? "identical" : `diverges at token ${firstDiff(plain, sp.toks)} (expected: the room ships f16 hiddens)`}`);
      for (const K of ks) {
        resetAll();
        const ss = await splitSpec(promptA, N, K);
        const ok = same(sp.toks, ss.toks);
        check(`${tag}: spec K=${K} == split plain`, ok, `accepted ${ss.st.accepted}/${ss.st.drafts}` + (ok ? "" : `, first diff at ${firstDiff(sp.toks, ss.toks)}: ${show(ss.toks)}`));
      }
      // multi-turn over the chain (spec, largest K): continuation without reset == reset + full prefill
      if (!wire16 && cut.length === 3) {
        const K = ks[ks.length - 1], n1 = 20, n2 = Math.min(N, 32);
        resetAll();
        const a = await splitSpec(promptA, n1, K);
        // specStep may have produced more than n1 tokens: continue from what was emitted
        const ans = a.all, B = contB(ans[ans.length - 1]);
        const b = await splitSpec(B, n2, K, host.pos);
        resetAll();
        const full = [...promptA, ...ans.slice(0, -1), ...B];
        const ref = await splitPlain(full, n2);
        const ok = same(b.toks, ref.toks);
        check(`${tag}: multi-turn spec continuation == reset + full prefill`, ok, `turn1 ${ans.length} tok, turn2 prefill ${B.length} tok at pos ${full.length - B.length}` + (ok ? "" : `, first diff at ${firstDiff(ref.toks, b.toks)}`));
      }
    }

    // ---------- multi-turn, solo ----------
    {
      const n1 = 20, n2 = Math.min(N, 32), K = ks[ks.length - 1];
      // plain
      solo.reset();
      const a = await soloPlain(promptA, n1);
      const B = contB(a[a.length - 1]);
      const b = await soloPlain(B, n2, false);
      const full = [...promptA, ...a.slice(0, -1), ...B];
      const ref = await soloPlain(full, n2);
      check(`[NC=${NC}] multi-turn plain: continuation == reset + full prefill`, same(b, ref), `turn2 prefill ${B.length} tok at pos ${full.length - B.length}` + (same(b, ref) ? "" : `, first diff at ${firstDiff(ref, b)}`));
      // spec
      const as = await soloSpec(promptA, n1, K);
      const Bs = contB(as.all[as.all.length - 1]);
      const bs = await soloSpec(Bs, n2, K, false);
      const fulls = [...promptA, ...as.all.slice(0, -1), ...Bs];
      const refs = await soloPlain(fulls, n2);
      check(`[NC=${NC}] multi-turn spec: continuation == reset + full prefill`, same(bs.toks, refs), `turn2 accepted ${bs.st.accepted}/${bs.st.drafts}` + (same(bs.toks, refs) ? "" : `, first diff at ${firstDiff(refs, bs.toks)}`));
      // draft quality after a continuation vs a fresh full prefill (same trunk stream either way)
      const fresh = await soloSpec(fulls, n2, K);
      info(`[NC=${NC}] draft acceptance turn 2: continued ${bs.st.accepted}/${bs.st.drafts} vs fresh full prefill ${fresh.st.accepted}/${fresh.st.drafts}` +
        (bs.st.accepted !== fresh.st.accepted ? "  <- draft (MTP) cache differs after a continuation prefill" : ""));
      // the draft block's own K cache after [turn 1 spec + continuation prefill] vs after one
      // fresh prefill of the same tokens: rows should agree up to batching noise. A row that
      // differs grossly was written at the wrong position or never written (draft-only damage:
      // the output stream is still exact, the acceptance rate is what suffers).
      const readRows = async (n) => {
        const kv = solo.dims.kvDim, bytes = n * kv * 4;
        const st = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(solo.mtpLayer.kCache, 0, st, 0, bytes); device.queue.submit([enc.finish()]);
        await st.mapAsync(GPUMapMode.READ); const a = new Float32Array(st.getMappedRange()).slice(); st.unmap(); st.destroy();
        return Array.from({ length: n }, (_, r) => a.subarray(r * kv, (r + 1) * kv));
      };
      const clearMtp = () => { const enc = device.createCommandEncoder(); enc.clearBuffer(solo.mtpLayer.kCache); device.queue.submit([enc.finish()]); };
      clearMtp(); solo.reset();
      const a2 = await soloSpec(promptA, n1, K, false);
      const B2 = contB(a2.all[a2.all.length - 1]);
      await solo.prefillTokens(B2.slice(0, -1));
      const P = solo.pos, rowsCont = await readRows(P);
      clearMtp(); solo.reset(); solo.mtpFill = true;
      await solo.prefillTokens([...promptA, ...a2.all.slice(0, -1), ...B2].slice(0, -1));
      const rowsFresh = await readRows(P);
      const bad = [];
      for (let r = 1; r < P; r++) {   // row 0 is never drafted
        let md = 0, sc = 1e-6;
        for (let i = 0; i < rowsFresh[r].length; i++) { md = Math.max(md, Math.abs(rowsCont[r][i] - rowsFresh[r][i])); sc = Math.max(sc, Math.abs(rowsFresh[r][i])); }
        if (md / sc > 1e-2) bad.push(`${r}${rowsCont[r].every((v) => v === 0) ? "(never written)" : ""}`);
      }
      info(`[NC=${NC}] draft K-cache rows 1..${P - 1} after continuation vs fresh prefill: ${bad.length ? "WARN differ at rows " + bad.join(", ") + ` (turn 1 prompt ${promptA.length} tok, turn 2 starts at pos ${P - B2.length + 1})` : "all agree"}`);
    }
  }
  check("no GPU validation errors", gpuErrors.length === 0, gpuErrors.slice(0, 2).join(" | "));
  return { results, out };
}

// ---------------------------------------------------------------- node side
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname)) {
  const PORT = +arg("port", 8131);
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-synth-"));
  const model = arg("model") || writeSynth(path.join(tmp, "qwen35-synth.gguf"), { mtp: arg("mtp", "echo"), seed: +arg("seed", 1), eosAt: +arg("eos-at", 90), ...(flag("moe") ? { moe: SYNTH_MOE } : {}) }).file;   // --moe: the qwen35moe variant
  const srv = serveRepo(PORT, { "/__synth.gguf": path.resolve(model) });
  const { chromium } = await loadPlaywright();
  const browser = await chromium.launch({ executablePath: chromiumPath(), args: GPU_ARGS });
  const t0 = Date.now();
  let code = 1;
  try {
    const page = await browser.newPage();
    page.on("console", (m) => { if (m.type() === "error") console.error("[page error]", m.text().slice(0, 300)); });
    page.on("pageerror", (e) => console.error("[pageerror]", String(e).slice(0, 300)));
    page.on("console", (m) => { if (m.type() === "log") console.log(m.text()); });
    await page.goto(`http://127.0.0.1:${PORT}/__blank.html`);
    const r = await page.evaluate(pageMain, {
      modelUrl: "/__synth.gguf", tokens: +arg("tokens", 40),
      cols: arg("cols", "4,16").split(",").map(Number), ks: arg("ks", "1,3,7").split(",").map(Number), wg: +arg("wg", 64), noSplit: flag("no-split"), kvQ8: flag("q8"),   // --q8: int8 KV cache
    });
    const fails = r.results.filter((x) => !x.ok);
    console.log(`\n${r.results.length - fails.length}/${r.results.length} checks passed in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    code = fails.length ? 1 : 0;
  } catch (e) {
    console.error("FAILED:", e.stack || e);
    code = 2;
  } finally {
    await browser.close(); srv.close();
    if (!flag("keep")) fs.rmSync(tmp, { recursive: true, force: true }); else console.log("model kept at", model);
  }
  process.exit(code);
}
