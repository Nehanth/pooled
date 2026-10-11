// The n-gram table's browser path (engine/ngram.js, engine/ngram_worker.js) in headless Chromium, on a synthetic
// Q4_0 table laid out like the real one (16 prime-sized heads, 160 values a row) inside a fake GGUF shard:
//   1. HTTP range reads of the shard (layout "tensor") give dequantF32's rows
//   2. splitHeads copies the table into 16 OPFS files (HTTP range -> createWritable)
//   3. NgramWorkers (sync access handles in 1, 4 and 16 workers) give the same rows, for hashed token rows and random
//      ones; timings for decode (16 rows a token) and a 2048-token prefill (32,768 rows, cold in the row cache)
//   4. while the handles are open a second open fails (one table user per origin), and works again after close()
//   5. a split file of the wrong size is refused at open
// The timings are on this synthetic table (16 x --rows rows: 5.8 MB at the default 4000, so it sits in the page cache)
// and say what the OPFS sync-read path costs per call, not what a cold 28.8 GB table on an SSD costs. --rows 1300000
// makes a 1.9 GB table. No WebGPU needed.   node tests/e2e/ngram_opfs.mjs [--rows 4000] [--workers 1,4,16] [--keep]
import fs from "fs";
import os from "os";
import path from "path";
import { loadPlaywright, chromiumPath, serveRepo } from "./engine_synth.mjs";
import { f32ToF16 } from "../../engine/gguf.js";
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const PORT = +arg("port", 18996);

// head sizes: primes near --rows, like the real 20,000,003 ... 20,000,171
const isPrime = (n) => { for (let d = 2; d * d <= n; d++) if (n % d === 0) return false; return n > 1; };
const vocab = [];
for (let n = +arg("rows", 4000); vocab.length < 16; n++) if (isPrime(n)) vocab.push(n);
const offsets = vocab.map((_, h) => vocab.slice(0, h).reduce((a, b) => a + b, 0));
const nRows = offsets[15] + vocab[15] + 90, RB = 90, byteOffset = 1 << 20;
const plainCfg = { offsets, vocab, mult: [23703573157769, 20109073645365, 8052911324071], eos: 248044, width: 160, ggmlType: 2, nRows, byteOffset, shard: 0 };

let seed = 99;
const rnd = () => { let t = (seed = (seed + 0x6d2b79f5) | 0); t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32; };
const shard = new Uint8Array(byteOffset + nRows * RB + 4096);
for (let i = 0; i < shard.length; i++) shard[i] = (rnd() * 256) | 0;
for (let b = 0; b < nRows * 5; b++) { const s = f32ToF16((rnd() - 0.5) * 0.05), o = byteOffset + b * 18; shard[o] = s & 255; shard[o + 1] = s >> 8; }
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-ngram-"));
fs.writeFileSync(path.join(tmp, "shard.gguf"), shard);

async function pageMain(args) {
  const res = { out: [], ok: false };
  try { return await pageBody(args, res); } catch (e) { res.out.push(`FAIL ${e.name}: ${e.message}`); return res; }
}
async function pageBody({ cfg: plain, workers }, res) {
  const { makeConfig, NgramState, NgramTable, httpReadAt, splitHeads, ngramFileName } = await import("/engine/ngram.js");
  const { NgramWorkers } = await import("/engine/ngram_worker.js");
  const { dequantF32 } = await import("/engine/gguf.js");
  const out = res.out, results = [];
  const check = (name, ok, detail = "") => { results.push(ok); out.push(`${ok ? "PASS" : "FAIL"} ${name}${detail ? "  " + detail : ""}`); };
  const cfg = makeConfig(plain);
  const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
  // the reference: the whole tensor through dequantF32
  const tb = new Uint8Array(await (await fetch("/__shard.gguf", { headers: { Range: `bytes=${cfg.byteOffset}-${cfg.byteOffset + cfg.nRows * cfg.rowBytes - 1}` } })).arrayBuffer());
  const all = dequantF32({ ggmlType: cfg.ggmlType, nElems: cfg.nRows * cfg.width, shape: [cfg.nRows, cfg.width] }, tb);
  const want = (ids) => { const o = new Float32Array(ids.length * cfg.width); ids.forEach((r, i) => o.set(all.subarray(r * cfg.width, (r + 1) * cfg.width), i * cfg.width)); return o; };
  const toks = Array.from({ length: 64 }, (_, i) => (i === 20 ? cfg.eos : (i * 7919 + 13) % 248000));
  const hashed = new NgramState(cfg).push(toks);
  const end = cfg.offsets[15] + cfg.vocab[15];
  const random = Uint32Array.from({ length: 2000 }, (_, i) => (i < 32 ? (i % 2 ? cfg.offsets[i >> 1] + cfg.vocab[i >> 1] - 1 : cfg.offsets[i >> 1]) : Math.floor(Math.random() * end)));

  // 1. HTTP, tensor layout
  const http = new NgramTable(cfg, { readAt: httpReadAt(["/__shard.gguf"]), layout: "tensor", inflight: 16 });
  check("HTTP range rows (tensor layout) == dequantF32", same(await http.rows(hashed), want(hashed)), `${hashed.length} rows, ${http.stats.reads} range requests`);

  // 2. split into OPFS
  const root = await navigator.storage.getDirectory();
  try { await root.removeEntry("ngram-e2e", { recursive: true }); } catch {}
  const dir = await root.getDirectoryHandle("ngram-e2e", { create: true });
  const t0 = performance.now();
  const writers = [];
  for (let h = 0; h < cfg.nHeads; h++) writers[h] = await (await dir.getFileHandle(ngramFileName(h), { create: true })).createWritable();
  const bytes = await splitHeads(cfg, httpReadAt(["/__shard.gguf"]), (h, b, off) => writers[h].write({ type: "write", position: off, data: b }), { chunkRows: 1000 });
  for (const w of writers) await w.close();
  let sizes = true;
  for (let h = 0; h < cfg.nHeads; h++) sizes &&= (await (await dir.getFileHandle(ngramFileName(h))).getFile()).size === cfg.vocab[h] * cfg.rowBytes;
  check("splitHeads into 16 OPFS files", sizes, `${(bytes / 2 ** 20).toFixed(1)} MiB in ${(performance.now() - t0).toFixed(0)} ms`);

  // 3. workers with sync access handles
  for (const W of workers) {
    const t = await NgramWorkers.open(cfg, { dir: ["ngram-e2e"], workers: W });
    const a = await t.rows(hashed), b = await t.rows(random), c = await t.rows(hashed);
    check(`OPFS workers=${W}: hashed rows == dequantF32`, same(a, want(hashed)));
    check(`OPFS workers=${W}: 2000 random rows (head edges included) == dequantF32`, same(b, want(random)));
    check(`OPFS workers=${W}: second pass (row cache) unchanged`, same(c, a));
    // decode-shaped timing: 16 rows of one token, cold rows each time (warm page cache; not a disk number)
    const st = new NgramState(cfg), T = 200, t1 = performance.now();
    for (let i = 0; i < T; i++) await t.rows(st.push([(i * 104729 + 7) % 248000]));
    const decodeMs = (performance.now() - t1) / T;
    // prefill-shaped: one rows() call for 2048 tokens (other token ids than above; on a small table rows repeat and
    // some are in the row cache already, so the distinct count is printed)
    const preRows = new NgramState(cfg).push(Array.from({ length: 2048 }, (_, i) => (i * 7907 + 100003) % 248000));
    const t2 = performance.now();
    const pre = await t.rows(preRows);
    const preMs = performance.now() - t2;
    check(`OPFS workers=${W}: 2048-token prefill rows == dequantF32`, same(pre, want(preRows)));
    out.push(`     workers=${W}: ${decodeMs.toFixed(3)} ms per token (16 rows); 2048-token prefill (${preRows.length} rows, ${new Set(preRows).size} distinct): ${preMs.toFixed(0)} ms`);
    // 4. the handles are exclusive
    if (W === Math.max(...workers)) {
      let err = null;
      try { const t2 = await NgramWorkers.open(cfg, { dir: ["ngram-e2e"], workers: 2 }); await t2.close(); } catch (e) { err = e; }
      check("a second open while the handles are held fails", !!err, err ? `${err.name}: ${err.message.slice(0, 80)}` : "it opened");
    }
    await t.close();
  }
  const again = await NgramWorkers.open(cfg, { dir: ["ngram-e2e"], workers: 2 });
  check("open again after close()", same(await again.rows(hashed), want(hashed)));
  await again.close();

  // 5. a truncated head file
  const w5 = await (await dir.getFileHandle(ngramFileName(5))).createWritable({ keepExistingData: true });
  await w5.truncate(cfg.rowBytes * 10); await w5.close();
  let err5 = null;
  try { const t5 = await NgramWorkers.open(cfg, { dir: ["ngram-e2e"], workers: 4 }); await t5.close(); } catch (e) { err5 = e; }
  check("a split file of the wrong size is refused", !!err5 && /ngram-05\.bin/.test(err5.message), err5?.message);
  await root.removeEntry("ngram-e2e", { recursive: true });
  return { out, ok: results.every(Boolean) && results.length > 0 };
}

const srv = serveRepo(PORT, { "/__shard.gguf": path.join(tmp, "shard.gguf") });
const { chromium } = await loadPlaywright();
// serialize the page functions: evaluate gets pageMain, which calls pageBody in the page
const browser = await chromium.launch({ executablePath: chromiumPath(), args: ["--no-sandbox", "--headless=new"] });
let res = { out: [], ok: false };
try {
  const page = await browser.newPage();
  page.on("pageerror", (e) => console.log("pageerror:", e.message));
  await page.goto(`http://127.0.0.1:${PORT}/__blank.html`);
  await page.addScriptTag({ content: pageBody.toString() });
  res = await page.evaluate(pageMain, { cfg: plainCfg, workers: arg("workers", "1,4,16").split(",").map(Number) });
} catch (e) { console.log("FAIL", e.message); }
for (const l of res.out) console.log(l);
console.log(res.ok ? "NGRAM OPFS PASS" : "NGRAM OPFS FAIL");
await browser.close(); srv.close();
if (!argv.includes("--keep")) fs.rmSync(tmp, { recursive: true, force: true });
process.exit(res.ok ? 0 : 1);
