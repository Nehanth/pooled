// The disk tier's browser path (engine/expert_opfs.js) in headless Chromium, on a synthetic MoE shard (stacked Q4_0
// gate / up and Q8_0 down expert tensors at odd offsets, like a GGUF's):
//   1. ingestExperts copies the experts from HTTP range reads into per-layer OPFS files (a worker's sync handles),
//      with a tab "closed" mid-layer (the GGUF read fails): the rerun resumes from the checkpoint
//   2. OpfsExpertReader (read-only shared handles, 1 / 4 / 8 workers, and the one-owner-per-file fallback) gives
//      exactly expertDisk's conversion over HTTP for every expert of every layer, read the way ExpertStore reads a
//      cut (all of its misses at once), and for lone misses striped over the workers
//   3. opening a copy that is not complete, or whose file was cut short, is refused
// No WebGPU needed (the GPU side is tests/bench/tab_disk.mjs).   node tests/e2e/expert_opfs.mjs [--experts 64] [--keep]
import fs from "fs";
import os from "os";
import path from "path";
import { loadPlaywright, chromiumPath, serveRepo } from "./engine_synth.mjs";
import { ggmlTypeBytes, GGML_Q4_0, GGML_Q8_0 } from "../../engine/gguf.js";
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const PORT = +arg("port", 18997), NEXP = +arg("experts", 64), NL = 4, ROWS = 64, COLS = 256;

// the shard and its tensor table (what parseGGUFHeader would give)
const tensors = {}, spans = [];
let off = 4096 + 18;
for (let l = 0; l < NL; l++) for (const [p, type, shape] of [["gate", GGML_Q4_0, [NEXP, ROWS, COLS]], ["up", GGML_Q4_0, [NEXP, ROWS, COLS]], ["down", GGML_Q8_0, [NEXP, COLS, ROWS]]]) {
  const len = ggmlTypeBytes(type, shape[0] * shape[1] * shape[2]);
  tensors[`blk.${l}.ffn_${p}_exps.weight`] = { shape, ggmlType: type, byteOffset: off, byteLength: len, shard: 0 };
  spans.push([off, len, type === GGML_Q4_0 ? 18 : 34]);
  off += len + 96;
}
const shard = new Uint8Array(off);
let seed = 4242;
for (let i = 0; i < shard.length; i++) shard[i] = (seed = (seed * 1103515245 + 12345) >>> 0) >>> 24;
for (const [o0, len, blk] of spans) for (let o = o0; o < o0 + len; o += blk) { shard[o] = 0x00; shard[o + 1] = 0x24; }
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-expert-opfs-"));
fs.writeFileSync(path.join(tmp, "shard.gguf"), shard);

async function pageMain(args) {
  const res = { out: [], ok: false };
  try { return await pageBody(args, res); } catch (e) { res.out.push(`FAIL ${e.name}: ${e.message}`); return res; }
}
async function pageBody({ G, workers }, res) {
  const X = await import("/engine/expert_opfs.js");
  const { expertDisk } = await import("/engine/expert_disk.js");
  const out = res.out, results = [];
  const check = (name, ok, detail = "") => { results.push(ok); out.push(`${ok ? "PASS" : "FAIL"} ${name}${detail ? "  " + detail : ""}`); };
  const same = (a, b) => a.byteLength === b.byteLength && a.every((v, i) => v === b[i]);
  const readAt = async (_s, o, n) => new Uint8Array(await (await fetch("/__shard.gguf", { headers: { Range: `bytes=${o}-${o + n - 1}` } })).arrayBuffer());
  // as a tab does it: the key carries a content signature of the experts, read over HTTP ranges
  const sig = await X.expertContentSig(G, readAt), layers = [0, 1, 2, 3], lay = X.expertLayout(G, layers, { contentSig: sig });
  check("the model key carries the content signature", lay.key === `${X.expertModelKey(G)}-${sig}` && /^[0-9a-f]{8}$/.test(sig), lay.key);
  await X.removeExperts(lay.key);
  // 1. ingest, with a failure in layer 2, then a resume
  let store = await X.opfsExpertStore(lay), reads = 0, err = null;
  const t0 = performance.now();
  try {
    await X.ingestExperts({ layout: lay, store, chunkExperts: 8, checkpointBytes: 16 * lay.layers[0].recBytes,
      readAt: (s, o, n) => { if (++reads > 2 * 3 * (lay.layers[0].nExp / 8) + 9) throw new Error("tab closed"); return readAt(s, o, n); } });
  } catch (e) { err = e; }
  await store.shutdown();
  const man = await X.readManifest([X.EXPERT_OPFS_DIR, lay.key]);
  check("ingest stopped mid-layer leaves a checkpoint", !!err && man.layers[0].done === G.nExp && man.layers[1].done === G.nExp && man.layers[2].done > 0 && man.layers[2].done < G.nExp,
    `layers done ${JSON.stringify(Object.values(man.layers).map((x) => x.done))}`);
  store = await X.opfsExpertStore(lay);
  const r = await X.ingestExperts({ layout: lay, store, readAt, chunkExperts: 8 });
  await store.shutdown();
  check("resumed ingest writes only the rest", r.skippedBytes === (2 * G.nExp + man.layers[2].done) * lay.layers[0].recBytes && r.skippedBytes + r.wroteBytes === lay.totalBytes,
    `${(lay.totalBytes / 2 ** 20).toFixed(1)} MiB in ${(performance.now() - t0).toFixed(0)} ms, persisted ${store.persisted}`);
  check("manifest complete", X.ingested(await X.readManifest([X.EXPERT_OPFS_DIR, lay.key]), lay));
  // 2. readers
  const ref = expertDisk({ readAt });
  const want = [];
  for (const l of layers) for (const p of ["gate", "up", "down"]) for (let e = 0; e < G.nExp; e++) want.push([l, p, e, await ref.readPart({ name: `blk.${l}.ffn_${p}_exps.weight`, ...G.tensors[`blk.${l}.ffn_${p}_exps.weight`] }, e)]);
  for (const [W, mode] of [...workers.map((w) => [w, "shared"]), [2, "owned"]]) {
    const R = await X.OpfsExpertReader.open(lay, { workers: W, mode });
    let ok = R.shared === (mode === "shared");
    const t1 = performance.now();
    for (const l of layers) {   // one "cut" per layer: every expert's three parts at once
      const got = await Promise.all(want.filter((w) => w[0] === l).map(([, p, e]) => R.readPart({ name: `blk.${l}.ffn_${p}_exps.weight` }, e)));
      want.filter((w) => w[0] === l).forEach(([, , , w], i) => { ok &&= same(got[i].qs, w.qs) && same(got[i].sc, w.sc); });
    }
    check(`reader ${mode} workers=${W}: every expert == expertDisk over HTTP`, ok && R.stats.reads === layers.length * G.nExp,
      `${R.stats.reads} record reads in ${R.stats.batches} batches (max ${R.stats.maxBatch}), ${(performance.now() - t1).toFixed(0)} ms`);
    if (mode === "shared" && W > 1) {   // decode-shaped: one miss at a time, each record striped over the workers
      R.stripeBytes = 4096;
      const r0 = R.stats.reads;
      let ok1 = true;
      for (const [l, p, e, w] of want.filter((x, i) => i % 37 === 0)) { const g = await R.readPart({ name: `blk.${l}.ffn_${p}_exps.weight` }, e); ok1 &&= same(g.qs, w.qs) && same(g.sc, w.sc); }
      check(`reader shared workers=${W}: lone misses striped over the workers == expertDisk`, ok1 && R.stats.reads > r0);
    }
    if (mode === "shared" && W === Math.max(...workers)) {
      // read-only handles do not lock: a second reader opens alongside
      let e2 = null;
      try { const R2 = await X.OpfsExpertReader.open(lay, { workers: 2, mode: "shared" }); await R2.close(); } catch (e) { e2 = e; }
      check("a second shared reader opens while the first holds its handles", !e2, e2?.message);
    }
    await R.close();
  }
  // 3. refusals
  const lay5 = X.expertLayout({ ...G, tensors: { ...G.tensors, ...Object.fromEntries(["gate", "up", "down"].map((p) => [`blk.4.ffn_${p}_exps.weight`, G.tensors[`blk.3.ffn_${p}_exps.weight`]])) } }, [0, 4]);
  let e3 = null;
  try { await X.OpfsExpertReader.open({ ...lay5, key: lay.key }); } catch (e) { e3 = e; }
  check("a layer that was never ingested is refused", !!e3 && /does not hold/.test(e3.message), e3?.message);
  const dir = await (await (await navigator.storage.getDirectory()).getDirectoryHandle(X.EXPERT_OPFS_DIR)).getDirectoryHandle(lay.key);
  const w = await (await dir.getFileHandle(lay.layers[1].file)).createWritable({ keepExistingData: true });
  await w.truncate(lay.layers[1].recBytes * 3); await w.close();
  let e4 = null;
  try { const R = await X.OpfsExpertReader.open(lay, { workers: 2 }); await R.close(); } catch (e) { e4 = e; }
  check("a file cut short is refused", !!e4 && /L001\.bin is/.test(e4.message), e4?.message);
  await X.removeExperts(lay.key);
  return { out, ok: results.every(Boolean) && results.length > 0 };
}

const srv = serveRepo(PORT, { "/__shard.gguf": path.join(tmp, "shard.gguf") });
const { chromium } = await loadPlaywright();
// a persistent profile on disk: an in-memory one keeps OPFS in RAM
const prof = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-expert-opfs-prof-"));
const ctx = await chromium.launchPersistentContext(prof, { executablePath: chromiumPath(), headless: true, args: ["--no-sandbox", "--headless=new"] });
let res = { out: [], ok: false };
try {
  const page = await ctx.newPage();
  page.on("pageerror", (e) => console.log("pageerror:", e.message));
  await page.goto(`http://127.0.0.1:${PORT}/__blank.html`);
  await page.addScriptTag({ content: pageBody.toString() });
  res = await page.evaluate(pageMain, { G: { meta: { "general.name": "synth-moe" }, tensors, nExp: NEXP }, workers: arg("workers", "1,4,8").split(",").map(Number) });
} catch (e) { console.log("FAIL", e.message); }
for (const l of res.out) console.log(l);
console.log(res.ok ? "EXPERT OPFS PASS" : "EXPERT OPFS FAIL");
await ctx.close(); srv.close();
if (!argv.includes("--keep")) { fs.rmSync(tmp, { recursive: true, force: true }); fs.rmSync(prof, { recursive: true, force: true }); }
process.exit(res.ok ? 0 : 1);
