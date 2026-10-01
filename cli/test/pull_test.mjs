// pooled pull / list / rm (cli/lib/cache.js) against a local HTTP server with Range support, and the
// room node reading a pulled model from disk (packages/room-node/source.js).
import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createHash, randomBytes } from "node:crypto";
import { pullModel, pullFile, modelState, listModels, removeModel, modelsDir, resolveModel, progressLine, progressPlain, rateMeter, fmtBytes, modelFiles } from "../lib/cache.js";
import { parseLendArgs } from "../lib/lend.js";
import { openModel } from "../../packages/room-node/source.js";

// a fake GGUF: the magic, then random bytes
const BODY = Buffer.concat([Buffer.from("GGUF"), randomBytes(3 * 2 ** 20 - 4)]);
const SHA = createHash("sha256").update(BODY).digest("hex");

// serves /Fake-Q8.gguf (Range), /config.json, /tokenizer.json; /wrong.gguf is a different size.
// opts.ranges: every Range header seen
function server() {
  const seen = { ranges: [], gets: 0 };
  const srv = http.createServer((req, res) => {
    const files = { "/Fake-Q8.gguf": BODY, "/wrong.gguf": BODY.subarray(0, 1000), "/config.json": Buffer.from('{"num_hidden_layers":2}'), "/tokenizer.json": Buffer.from("{}") };
    const body = files[req.url];
    if (!body) { res.writeHead(404); res.end(); return; }
    seen.gets++;
    const m = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range || "");
    if (m) {
      seen.ranges.push(req.headers.range);
      const a = +m[1], b = m[2] ? +m[2] : body.length - 1;
      if (a >= body.length) { res.writeHead(416, { "content-range": `bytes */${body.length}` }); res.end(); return; }
      res.writeHead(206, { "content-range": `bytes ${a}-${b}/${body.length}`, "content-length": b - a + 1 });
      res.end(body.subarray(a, b + 1));
      return;
    }
    res.writeHead(200, { "content-length": body.length });
    // in pieces, so a test can abort in the middle
    let o = 0;
    const next = () => { if (o >= body.length) { res.end(); return; } const n = Math.min(256 * 1024, body.length - o); res.write(body.subarray(o, o + n)); o += n; setImmediate(next); };
    next();
  });
  return new Promise((r) => srv.listen(0, "127.0.0.1", () => r({ srv, seen, base: `http://127.0.0.1:${srv.address().port}` })));
}
const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), "pooled-pull-"));
const fakeModels = (base) => ({
  "fake-1b": { label: "Fake 1B · Q8", kind: "gguf", gguf: `${base}/Fake-Q8.gguf`, cfg: `${base}/config.json`, tok: `${base}/tokenizer.json` },
  "fake-moe": { label: "Fake MoE · Q4", kind: "qwen35", gguf: `${base}/wrong.gguf` },
});

test("pull: the model's files land in <dir>/<model>, with progress; a second pull does nothing", async () => {
  const { srv, base } = await server();
  const dir = tmp();
  try {
    const MODELS = fakeModels(base), FILES = { "fake-1b": { bytes: BODY.length, sha256: SHA } };
    const seen = [];
    const r = await pullModel("fake-1b", { dir, MODELS, FILES, onProgress: (p) => seen.push(p) });
    assert.equal(r.skipped, false);
    assert.deepEqual(fs.readFileSync(path.join(dir, "fake-1b", "Fake-Q8.gguf")), BODY);
    assert.ok(fs.existsSync(path.join(dir, "fake-1b", "config.json")) && fs.existsSync(path.join(dir, "fake-1b", "tokenizer.json")));
    assert.ok(!fs.existsSync(path.join(dir, "fake-1b", "Fake-Q8.gguf.part")));
    assert.ok(seen.length > 2, "progress as the bytes come");
    assert.equal(seen.at(-1).done, BODY.length); assert.equal(seen.at(-1).total, BODY.length);
    assert.ok(seen.every((p, i) => i === 0 || p.done >= seen[i - 1].done));
    const st = modelState(dir, "fake-1b", MODELS, FILES);
    assert.equal(st.pulled, true); assert.equal(st.main, path.join(dir, "fake-1b", "Fake-Q8.gguf"));
    assert.equal((await pullModel("fake-1b", { dir, MODELS, FILES })).skipped, true);
  } finally { srv.close(); fs.rmSync(dir, { recursive: true, force: true }); }
});

test("pull: an abort leaves a .part, and the next pull resumes it with a Range request", async () => {
  const { srv, seen, base } = await server();
  const dir = tmp();
  try {
    const MODELS = fakeModels(base), FILES = { "fake-1b": { bytes: BODY.length, sha256: SHA } };
    const ac = new AbortController();
    await assert.rejects(pullModel("fake-1b", { dir, MODELS, FILES, signal: ac.signal, onProgress: (p) => { if (p.done > BODY.length / 3) ac.abort(); } }),
      (e) => e.type === "aborted");
    const part = path.join(dir, "fake-1b", "Fake-Q8.gguf.part");
    const had = fs.statSync(part).size;
    assert.ok(had > 0 && had < BODY.length, `a partial file (${had})`);
    assert.equal(modelState(dir, "fake-1b", MODELS, FILES).pulled, false);
    assert.equal(modelState(dir, "fake-1b", MODELS, FILES).partBytes, had);
    let first = null;
    const r = await pullModel("fake-1b", { dir, MODELS, FILES, onProgress: (p) => { first ??= p; } });
    assert.equal(first.resumed, had, "picks up where it stopped");
    assert.ok(seen.ranges.includes(`bytes=${had}-`));
    assert.deepEqual(fs.readFileSync(path.join(dir, "fake-1b", "Fake-Q8.gguf")), BODY);
    assert.equal(r.skipped, false);
  } finally { srv.close(); fs.rmSync(dir, { recursive: true, force: true }); }
});

test("pull: a whole .part (stopped right at the end) is kept: the server's 416 says it is complete", async () => {
  const { srv, base } = await server();
  const dir = tmp();
  try {
    const MODELS = fakeModels(base), FILES = { "fake-1b": { bytes: BODY.length, sha256: SHA } };
    fs.mkdirSync(path.join(dir, "fake-1b"), { recursive: true });
    fs.writeFileSync(path.join(dir, "fake-1b", "Fake-Q8.gguf.part"), BODY);
    const r = await pullFile({ name: "Fake-Q8.gguf", url: `${base}/Fake-Q8.gguf`, bytes: null, sha256: SHA, main: true }, path.join(dir, "fake-1b"));
    assert.equal(r.bytes, BODY.length);
    assert.deepEqual(fs.readFileSync(path.join(dir, "fake-1b", "Fake-Q8.gguf")), BODY);
  } finally { srv.close(); fs.rmSync(dir, { recursive: true, force: true }); }
});

test("pull: a file of the wrong size or hash is rejected and never becomes the model", async () => {
  const { srv, base } = await server();
  const dir = tmp();
  try {
    const MODELS = fakeModels(base);
    // the server's file is 1000 bytes, room/models.js FILES says otherwise
    await assert.rejects(pullModel("fake-moe", { dir, MODELS, FILES: { "fake-moe": { bytes: 5000 } } }), (e) => e.type === "size" && /1000 bytes, not the 5000/.test(e.message));
    assert.ok(!fs.existsSync(path.join(dir, "fake-moe", "wrong.gguf")));
    assert.equal(modelState(dir, "fake-moe", MODELS, { "fake-moe": { bytes: 5000 } }).pulled, false);
    // right size, wrong SHA-256: removed
    await assert.rejects(pullModel("fake-1b", { dir, MODELS, FILES: { "fake-1b": { bytes: BODY.length, sha256: "0".repeat(64) } } }), (e) => e.type === "hash");
    assert.ok(!fs.existsSync(path.join(dir, "fake-1b", "Fake-Q8.gguf")) && !fs.existsSync(path.join(dir, "fake-1b", "Fake-Q8.gguf.part")));
    // a file of the wrong size under the final name (copied by hand) is not taken as pulled
    fs.writeFileSync(path.join(dir, "fake-1b", "Fake-Q8.gguf"), "short");
    assert.equal(modelState(dir, "fake-1b", MODELS, { "fake-1b": { bytes: BODY.length } }).pulled, false);
    // an unknown host answer
    await assert.rejects(pullFile({ name: "x.gguf", url: `${base}/nothing.gguf`, main: true }, dir), (e) => e.type === "http" && /404/.test(e.message));
  } finally { srv.close(); fs.rmSync(dir, { recursive: true, force: true }); }
});

test("pull: a big file comes in several ranges at once, resumes each after an abort, and keeps a one-connection .part", async () => {
  const { srv, seen, base } = await server();
  const dir = tmp();
  try {
    const MODELS = fakeModels(base), FILES = { "fake-1b": { bytes: BODY.length, sha256: SHA } };
    const seg = { segments: 4, minSegmentedBytes: 0 };
    const gguf = path.join(dir, "fake-1b", "Fake-Q8.gguf"), part = gguf + ".part";
    // straight through: four ranges that cover the file
    const r = await pullModel("fake-1b", { dir, MODELS, FILES, ...seg });
    assert.equal(r.skipped, false);
    assert.deepEqual(fs.readFileSync(gguf), BODY);
    const q = BODY.length / 4;
    for (let i = 0; i < 4; i++) assert.ok(seen.ranges.includes(`bytes=${Math.floor(q * i)}-${Math.floor(q * (i + 1)) - 1}`), `range ${i}`);
    assert.ok(!fs.existsSync(part) && !fs.existsSync(part + ".json"));
    removeModel(dir, "fake-1b");
    // stopped halfway: the .part and its .json say how far each range got; the next pull asks only for the rest
    const ac = new AbortController();
    await assert.rejects(pullModel("fake-1b", { dir, MODELS, FILES, ...seg, signal: ac.signal, onProgress: (p) => { if (p.done > BODY.length / 3) ac.abort(); } }),
      (e) => e.type === "aborted");
    const meta = JSON.parse(fs.readFileSync(part + ".json", "utf8"));
    const had = meta.base + meta.segs.reduce((a, g) => a + g.done, 0);
    assert.ok(had > 0 && had < BODY.length, `partial (${had})`);
    assert.equal(modelState(dir, "fake-1b", MODELS, FILES).partBytes, had);
    seen.ranges.length = 0;
    let first = null;
    await pullModel("fake-1b", { dir, MODELS, FILES, ...seg, onProgress: (p) => { first ??= p; } });
    assert.equal(first.resumed, had);
    for (const g of meta.segs) if (g.done < g.end - g.start) assert.ok(seen.ranges.includes(`bytes=${g.start + g.done}-${g.end - 1}`));
    assert.deepEqual(fs.readFileSync(gguf), BODY);
    assert.ok(!fs.existsSync(part + ".json"));
    removeModel(dir, "fake-1b");
    // a .part from a one-connection download (0.3.0): its bytes stay, the ranges fetch the rest
    fs.mkdirSync(path.dirname(part), { recursive: true });
    fs.writeFileSync(part, BODY.subarray(0, 1000000));
    seen.ranges.length = 0;
    await pullModel("fake-1b", { dir, MODELS, FILES, ...seg });
    assert.deepEqual(fs.readFileSync(gguf), BODY);
    assert.ok(seen.ranges.some((r) => r.startsWith("bytes=1000000-")), "starts after the bytes it had");
  } finally { srv.close(); fs.rmSync(dir, { recursive: true, force: true }); }
});

test("pull: a server that ignores ranges still works (one connection), and a wrong size is caught in ranges too", async () => {
  const srv = http.createServer((req, res) => { res.writeHead(200, { "content-length": BODY.length }); res.end(BODY); });
  await new Promise((r) => srv.listen(0, "127.0.0.1", r));
  const base = `http://127.0.0.1:${srv.address().port}`;
  const dir = tmp();
  try {
    const f = { name: "Fake-Q8.gguf", url: `${base}/Fake-Q8.gguf`, bytes: BODY.length, sha256: SHA, main: true };
    const r = await pullFile(f, dir, { segments: 4, minSegmentedBytes: 0 });
    assert.deepEqual(fs.readFileSync(r.path), BODY);
    assert.ok(!fs.existsSync(r.path + ".part.json"));
  } finally { srv.close(); fs.rmSync(dir, { recursive: true, force: true }); }
  const { srv: s2, base: b2 } = await server();
  const d2 = tmp();
  try {
    await assert.rejects(pullFile({ name: "Fake-Q8.gguf", url: `${b2}/Fake-Q8.gguf`, bytes: BODY.length + 5, main: true }, d2, { segments: 4, minSegmentedBytes: 0 }),
      (e) => e.type === "size");
    assert.ok(!fs.existsSync(path.join(d2, "Fake-Q8.gguf.part")) && !fs.existsSync(path.join(d2, "Fake-Q8.gguf.part.json")));
  } finally { s2.close(); fs.rmSync(d2, { recursive: true, force: true }); }
});

test("list and rm: what is here, what is not, partial downloads; rm deletes one model", async () => {
  const { srv, base } = await server();
  const dir = tmp();
  try {
    const MODELS = fakeModels(base), FILES = { "fake-1b": { bytes: BODY.length }, "fake-moe": { bytes: 1000 } };
    await pullModel("fake-1b", { dir, MODELS, FILES });
    fs.mkdirSync(path.join(dir, "fake-moe"), { recursive: true });
    fs.writeFileSync(path.join(dir, "fake-moe", "wrong.gguf.part"), Buffer.alloc(300));
    const rows = listModels(dir, MODELS, FILES);
    assert.deepEqual(rows.map((r) => [r.key, r.pulled]), [["fake-1b", true], ["fake-moe", false]]);
    assert.equal(rows[0].bytes, BODY.length + 23 + 2);
    assert.equal(rows[1].partBytes, 300); assert.equal(rows[1].fileBytes, 1000);
    assert.equal(removeModel(dir, "fake-1b"), BODY.length + 25);
    assert.equal(modelState(dir, "fake-1b", MODELS, FILES).pulled, false);
    assert.equal(removeModel(dir, "fake-1b"), null, "nothing left to remove");
  } finally { srv.close(); fs.rmSync(dir, { recursive: true, force: true }); }
});

test("the models directory: --models, else POOLED_MODELS, else ~/.pooled/models; names by a part of them", () => {
  assert.equal(modelsDir({ flag: "/m", env: { POOLED_MODELS: "/e" } }), "/m");
  assert.equal(modelsDir({ env: { POOLED_MODELS: "/e" } }), "/e");
  assert.equal(modelsDir({ env: {} }), path.join(os.homedir(), ".pooled", "models"));
  const keys = ["qwen3-0.6b", "qwen3-1.7b", "qwen3.8-27b", "qwen3.6-35b-moe"];
  assert.deepEqual(resolveModel("35b", keys), { key: "qwen3.6-35b-moe" });
  assert.deepEqual(resolveModel("QWEN3-1.7B", keys), { key: "qwen3-1.7b" });
  assert.match(resolveModel("qwen3", keys).error, /more than one/);
  assert.match(resolveModel("gpt", keys).error, /unknown model/);
  assert.deepEqual(modelFiles("x", { x: { label: "X", kind: "qwen35", gguf: "https://h/a/X.gguf" } }).map((f) => f.name), ["X.gguf"]);
});

test("progress: one line with a bar, %, size, speed and ETA that fits the width; plain lines for logs", () => {
  const l = progressLine({ done: 512 * 2 ** 20, total: 2 * 2 ** 30, bps: 64 * 2 ** 20, width: 80 });
  assert.match(l, /▕█+░+▏\s+25%\s+512 MB \/ 2\.0 GB\s+64 MB\/s\s+ETA 24s/);
  assert.ok(l.length <= 80);
  assert.ok(progressLine({ done: 1, total: 2 ** 40, bps: 1, width: 40 }).length <= 40);
  assert.equal(progressPlain({ done: 2 ** 30, total: 4 * 2 ** 30, bps: 2 ** 30 / 10 }), "25% 1.0 GB of 4.0 GB at 102 MB/s, 30s left");
  let t = 0;
  const rate = rateMeter(5000, () => t);
  assert.equal(rate(0), null);
  t = 1000; assert.equal(rate(10 * 2 ** 20), 10 * 2 ** 20);
  assert.equal(fmtBytes(20836243072), "19.4 GB");
});

test("host and join read a pulled model from disk: the room node finds <dir>/<model>/<file>", async () => {
  const dir = tmp();
  try {
    fs.mkdirSync(path.join(dir, "qwen3-1.7b"), { recursive: true });
    fs.writeFileSync(path.join(dir, "qwen3-1.7b", "Qwen3-1.7B-Q8_0.gguf"), BODY);
    fs.writeFileSync(path.join(dir, "qwen3-1.7b", "config.json"), '{"num_hidden_layers":28}');
    const src = openModel("qwen3-1.7b", { modelDir: dir });
    assert.equal(src.local, path.join(dir, "qwen3-1.7b", "Qwen3-1.7B-Q8_0.gguf"));
    assert.deepEqual(Buffer.from(await src.readAt(0, 4)), Buffer.from("GGUF"));
    assert.equal((await src.cfg()).num_hidden_layers, 28, "the side files next to it");
    await src.close();
    // not pulled: the URL (range requests), as before
    assert.equal(openModel("qwen3.6-35b-moe", { modelDir: dir }).local, null);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test("pooled host <model>: the model as a positional (or a part of its name), --model still works", () => {
  const MODELS = { "qwen3-1.7b": { kind: "gguf" }, "qwen3.6-35b-moe": { kind: "qwen35" } };
  assert.equal(parseLendArgs("host", ["qwen3.6-35b-moe"], { models: MODELS }).model, "qwen3.6-35b-moe");
  assert.equal(parseLendArgs("host", ["35b"], { models: MODELS }).model, "qwen3.6-35b-moe");
  assert.equal(parseLendArgs("host", ["--model", "qwen3.6-35b-moe"], { models: MODELS }).model, "qwen3.6-35b-moe");
  assert.equal(parseLendArgs("host", ["qwen3.6-35b-moe", "--model", "qwen3.6-35b-moe"], { models: MODELS }).modelGiven, true);
  const d = parseLendArgs("host", [], { models: MODELS });
  assert.equal(d.model, "qwen3-1.7b"); assert.equal(d.modelGiven, false);
});
