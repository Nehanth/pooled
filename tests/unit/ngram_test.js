// The n-gram row source (engine/ngram.js) on synthetic tables: rows must equal engine/gguf.js dequantF32 of the same
// rows of the whole tensor, for both layouts (the GGUF's tensor, per-head split files), every reader, with and
// without the row cache; and the predecessor state must give a straight run's rows across batches, rollback and
// export / import. The hash's agreement with llama.cpp is checked in the reference track (roadmap/33 §5), not here;
// here it is only checked against the formula written out with BigInt. No GPU.
//   deno test --allow-read tests/unit/ngram_test.js
// The real file: tests/ngram_real.js. OPFS in Chrome: tests/e2e/ngram_opfs.mjs.
import { makeConfig, ngramRowsOf, NgramState, NgramTable, dequantRow, rowAddress, memoryReadAt, fileReadAt, opfsReadAt, httpReadAt,
  splitHeads } from "../../engine/ngram.js";
import { dequantF32, f32ToF16, ggmlTypeBytes, GGML_Q4_0, GGML_Q8_0, GGML_F16, GGML_BF16, GGML_F32 } from "../../engine/gguf.js";

const assert = (c, m) => { if (!c) throw new Error(m); };
let seed = 12345;
const rnd = () => { let t = (seed = (seed + 0x6d2b79f5) | 0); t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32; };

// the real model's hash constants (Qwen3.8-Flash-Next GGUF, qwen4exp.ple.*)
const REAL = {
  mult: [23703573157769, 20109073645365, 8052911324071],
  vocab: [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171],
  eos: 248044,
};
REAL.offsets = REAL.vocab.map((_, h) => REAL.vocab.slice(0, h).reduce((a, b) => a + b, 0));

// a small table with the real layout: 16 heads of prime sizes, 160 values a row
function synth(ggmlType = GGML_Q4_0, { width = 160 } = {}) {
  const vocab = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179];
  const offsets = vocab.map((_, h) => vocab.slice(0, h).reduce((a, b) => a + b, 0));
  const nRows = offsets[15] + vocab[15] + 7;   // padded, like the real tensor (320,001,446 rows used of 320,001,536)
  const cfg = makeConfig({ offsets, vocab, mult: REAL.mult, eos: 50, width, ggmlType, nRows, byteOffset: 4096 + 32, shard: 1 });
  const bytes = new Uint8Array(nRows * cfg.rowBytes);
  if (ggmlType === GGML_Q4_0 || ggmlType === GGML_Q8_0) {
    const BB = ggmlType === GGML_Q4_0 ? 18 : 34;
    for (let b = 0; b < bytes.length / BB; b++) {
      const s = f32ToF16((rnd() - 0.5) * 0.02 * (b % 13 === 0 ? 1e-4 : 1));   // some subnormal-ish scales too
      bytes[b * BB] = s & 255; bytes[b * BB + 1] = s >> 8;
      for (let i = 2; i < BB; i++) bytes[b * BB + i] = (rnd() * 256) | 0;
    }
  } else if (ggmlType === GGML_F32) {
    const f = new Float32Array(bytes.buffer); for (let i = 0; i < f.length; i++) f[i] = rnd() - 0.5;
  } else {
    for (let i = 0; i < bytes.length; i += 2) { const h = f32ToF16(rnd() - 0.5); bytes[i] = h & 255; bytes[i + 1] = h >> 8; }
  }
  // the shard: some other bytes before the tensor, the table, more after
  const shard = new Uint8Array(cfg.byteOffset + bytes.length + 100);
  shard.fill(0xAB); shard.set(bytes, cfg.byteOffset);
  const all = dequantF32({ ggmlType, nElems: nRows * width, shape: [nRows, width] }, bytes.slice());
  return { cfg, bytes, shard, all };
}
const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
const randRows = (cfg, n) => Uint32Array.from({ length: n }, () => (rnd() * cfg.nRows) | 0);
const expect = (all, cfg, ids) => { const o = new Float32Array(ids.length * cfg.width); ids.forEach((r, i) => o.set(all.subarray(r * cfg.width, (r + 1) * cfg.width), i * cfg.width)); return o; };
async function splitInMemory(cfg, shard) {
  const files = cfg.vocab.map((v) => new Uint8Array(v * cfg.rowBytes));
  await splitHeads(cfg, memoryReadAt([null, shard]), async (h, b, off) => files[h].set(b, off), { chunkRows: 37 });
  return files;
}

Deno.test("rows equal dequantF32 of the same rows: Q4_0, Q8_0, F16, BF16, F32, both layouts", async () => {
  for (const T of [GGML_Q4_0, GGML_Q8_0, GGML_F16, GGML_BF16, GGML_F32]) {
    const { cfg, shard, all } = synth(T);
    assert(cfg.rowBytes === ggmlTypeBytes(T, 160), "row bytes");
    const ids = randRows(cfg, 300);
    // every head's first and last row, and the padding rows past the last head
    for (let h = 0; h < 16; h++) ids[h * 2] = cfg.offsets[h], ids[h * 2 + 1] = cfg.offsets[h] + cfg.vocab[h] - 1;
    const tensorIds = Uint32Array.from([...ids, cfg.nRows - 1]);
    const t = new NgramTable(cfg, { readAt: memoryReadAt([null, shard]), layout: "tensor" });
    assert(same(await t.rows(tensorIds), expect(all, cfg, tensorIds)), `type ${T}: tensor layout differs from dequantF32`);
    const files = await splitInMemory(cfg, shard);
    const s = new NgramTable(cfg, { readAt: memoryReadAt(files), layout: "split", cacheRows: 0 });
    // padding rows belong to no head: the split copy has no place for them
    const inHeads = ids.filter((r) => r < cfg.offsets[15] + cfg.vocab[15]);
    assert(same(await s.rows(inHeads), expect(all, cfg, inHeads)), `type ${T}: split layout differs`);
  }
});

Deno.test("dequantRow is bit-equal to dequantF32 on every Q4_0 / Q8_0 nibble and scale", () => {
  for (const T of [GGML_Q4_0, GGML_Q8_0]) {
    const cfg = makeConfig({ offsets: [0, 1], vocab: [1, 1], mult: [3, 5], ngramSize: 2, perNgram: 2, eos: 0, width: 32, ggmlType: T });
    const BB = cfg.rowBytes, out = new Float32Array(32);
    // all 65536 f16 scales (NaN, Inf, subnormals) against a fixed spread of quants
    for (let s = 0; s < 65536; s++) {
      const b = new Uint8Array(BB); b[0] = s & 255; b[1] = s >> 8;
      for (let i = 2; i < BB; i++) b[i] = (i * 37 + s) & 255;
      dequantRow(cfg, b, 0, out, 0);
      const ref = dequantF32({ ggmlType: T, nElems: 32, shape: [1, 32] }, b);
      if (!same(out, ref)) throw new Error(`type ${T} scale 0x${s.toString(16)} differs`);
    }
  }
});

Deno.test("the row cache: repeats are hits, results unchanged, size bounded", async () => {
  const { cfg, shard, all } = synth();
  const files = await splitInMemory(cfg, shard);
  let reads = 0;
  const base = memoryReadAt(files);
  const t = new NgramTable(cfg, { readAt: (f, o, n) => { reads++; return base(f, o, n); }, cacheRows: 50 });
  const ids = randRows(cfg, 40).map((r) => Math.min(r, cfg.offsets[15]));
  const a = await t.rows(ids);
  const r0 = reads;
  const b = await t.rows(ids);
  assert(reads === r0, `second pass read ${reads - r0} rows, the cache should have had them`);
  assert(same(a, b) && same(a, expect(all, cfg, ids)), "cached rows differ");
  // duplicates in one call are read once
  reads = 0;
  const u = new NgramTable(cfg, { readAt: (f, o, n) => { reads++; return base(f, o, n); }, cacheRows: 0 });
  await u.rows(Uint32Array.from([5, 5, 5, 300, 5, 300]));
  assert(reads === 2, `6 ids of 2 rows took ${reads} reads`);
  await t.rows(randRows(cfg, 200).map((r) => Math.min(r, cfg.offsets[15])));
  assert(t.lru.size <= 50, `cache holds ${t.lru.size} rows`);
});

Deno.test("bad input: out-of-range rows, short reads", async () => {
  const { cfg, shard } = synth();
  const t = new NgramTable(cfg, { readAt: memoryReadAt([null, shard]), layout: "tensor" });
  let threw = false;
  try { await t.rows([cfg.nRows]); } catch { threw = true; }
  assert(threw, "row past the tensor accepted");
  const s = new NgramTable(cfg, { readAt: async () => new Uint8Array(3) });
  threw = false;
  try { await s.rows([1]); } catch (e) { threw = /short read/.test(e.message); }
  assert(threw, "short read accepted");
  threw = false;
  try { rowAddress(cfg, "split", cfg.offsets[15] + cfg.vocab[15]); } catch { threw = true; }
  assert(threw, "split address for a padding row");
});

// the hash written out separately, as roadmap/33 §1 states it
function refRows(cfg, c0, c1, c2) {
  const m = cfg.mult.map(BigInt), out = [];
  const mix2 = (BigInt(c0) * m[0]) ^ (BigInt(c1) * m[1]), mix3 = mix2 ^ (BigInt(c2) * m[2]);
  for (let g = 0; g < 8; g++) out.push(Number(mix2 % BigInt(cfg.vocab[g])) + cfg.offsets[g]);
  for (let g = 0; g < 8; g++) out.push(Number(mix3 % BigInt(cfg.vocab[8 + g])) + cfg.offsets[8 + g]);
  return out;
}

Deno.test("hash: the formula, rows inside their heads, and doubles would get it wrong", () => {
  const cfg = makeConfig({ ...REAL, nRows: 320001536 });
  let doubleWrong = 0;
  for (let i = 0; i < 2000; i++) {
    const c = [0, 0, 0].map(() => (rnd() * 248320) | 0);
    const got = [...ngramRowsOf(cfg, c[0], c[1], c[2])];
    const want = refRows(cfg, c[0], c[1], c[2]);
    assert(same(got, want), `tokens ${c}: ${got} vs ${want}`);
    for (let h = 0; h < 16; h++) assert(got[h] >= cfg.offsets[h] && got[h] < cfg.offsets[h] + cfg.vocab[h], `head ${h} row ${got[h]} outside its range`);
    // the same with Number products (what a careless port does): 63-bit products lose their low bits
    const dbl = Number(((c[0] * REAL.mult[0]) % REAL.vocab[0])) + cfg.offsets[0];
    if (dbl !== got[0]) doubleWrong++;
  }
  assert(doubleWrong > 1500, `a Number product matched ${2000 - doubleWrong} of 2000 times; the check is not sensitive`);
  // fixed values, so a change to the hash shows up as a diff (computed with the formula above)
  assert(same([...ngramRowsOf(cfg, 9707, 248044, 248044)], refRows(cfg, 9707, 248044, 248044)), "fixed sample");
});

Deno.test("state: sequence start and EOS cuts", () => {
  const cfg = makeConfig({ ...REAL });
  const E = REAL.eos, st = new NgramState(cfg);
  const toks = [11, 22, 33, E, 44, 55, E, E, 66];
  const rows = st.push(toks);
  // what each token's (c0, c1, c2) must be
  const ctx = [[11, E, E], [22, 11, E], [33, 22, 11], [E, 33, 22], [44, E, E], [55, 44, E], [E, 55, 44], [E, E, E], [66, E, E]];
  ctx.forEach((c, i) => assert(same([...rows.subarray(i * 16, i * 16 + 16)], refRows(cfg, ...c)), `token ${i} (${toks[i]}): wrong context`));
  assert(st.pos === toks.length, "pos");
});

Deno.test("state: batches, peek, rollback and export/import all equal one straight run", () => {
  const cfg = makeConfig({ ...REAL });
  const E = REAL.eos;
  const toks = Array.from({ length: 300 }, (_, i) => (i % 37 === 5 ? E : (rnd() * 248000) | 0));
  const straight = new NgramState(cfg).push(toks);
  // in random batches, with a speculative detour (push a wrong draft, roll it back) before each real batch
  const st = new NgramState(cfg);
  const got = [];
  for (let i = 0; i < toks.length;) {
    const n = 1 + ((rnd() * 9) | 0), batch = toks.slice(i, i + n);
    const peeked = st.peek(batch);
    const draft = Array.from({ length: 1 + ((rnd() * 5) | 0) }, () => (rnd() < 0.2 ? E : (rnd() * 1000) | 0));
    st.push(draft); st.rollback(draft.length);
    const r = st.push(batch);
    assert(same(peeked, r), "peek differs from push");
    got.push(...r); i += n;
    // a partial accept: push the batch plus 3 drafted tokens, keep the batch
    if (i % 2) { const s0 = st.export(); st.push([1, 2, 3]); st.rollback(3); assert(JSON.stringify(st.export()) === JSON.stringify(s0), "rollback did not restore"); }
  }
  assert(same(got, [...straight]), "batched + rolled-back run differs from the straight run");
  // export after 150 tokens, import into a fresh state (a resumed session), continue
  const a = new NgramState(cfg); a.push(toks.slice(0, 150));
  const b = new NgramState(cfg); b.import(JSON.parse(JSON.stringify(a.export())));
  assert(same(b.push(toks.slice(150)), straight.subarray(150 * 16)), "imported state differs");
  let threw = false;
  try { b.rollback(151); } catch { threw = true; }
  assert(threw, "rollback past the import accepted");
  const c = new NgramState(cfg, { history: 4 }); c.push([1, 2, 3, 4, 5, 6]);
  threw = false;
  try { c.rollback(5); } catch { threw = true; }
  assert(threw, "rollback deeper than the history accepted");
  c.rollback(4);
  assert(same(c.push([3, 4]), new NgramState(cfg).push([1, 2, 3, 4]).subarray(32)), "rollback at the history limit");
});

Deno.test("readers: positional file reads (Deno file, Node-style handle), OPFS sync handle, HTTP range", async () => {
  const path = new URL("../../engine/gguf.js", import.meta.url);
  const whole = await Deno.readFile(path);
  // Deno.FsFile: seek + read, serialized per file even when calls overlap
  const f = await Deno.open(path, { read: true });
  try {
    const r = fileReadAt([f]);
    const offs = [0, 1000, 37, 5000, 999, 12345];
    const got = await Promise.all(offs.map((o) => r(0, o, 90)));
    offs.forEach((o, i) => assert(same(got[i], whole.subarray(o, o + 90)), `Deno read at ${o}`));
    const tail = await r(0, whole.length - 10, 90);
    assert(tail.byteLength === 10, "a read past the end should come back short");
  } finally { f.close(); }
  // a Node FileHandle's read(buf, off, len, pos), returning short reads in pieces
  const node = { read: async (buf, o, n, pos) => { const k = Math.min(n, 7, whole.length - pos); buf.set(whole.subarray(pos, pos + k), o); return { bytesRead: Math.max(0, k) }; } };
  assert(same(await fileReadAt([node])(0, 300, 90), whole.subarray(300, 390)), "Node handle read");
  // OPFS: FileSystemSyncAccessHandle.read(buf, { at })
  const sync = { read: (buf, { at }) => { const k = Math.min(buf.length, whole.length - at); buf.set(whole.subarray(at, at + k)); return k; } };
  assert(same(await opfsReadAt([sync])(0, 77, 90), whole.subarray(77, 167)), "OPFS read");
  // HTTP: a range server, and one that ignores ranges (200 with the whole file)
  const seen = [];
  const fetch206 = async (url, { headers }) => {
    seen.push(headers.Range);
    const [, a, b] = /bytes=(\d+)-(\d+)/.exec(headers.Range).map(Number);
    return { status: 206, arrayBuffer: async () => whole.slice(a, b + 1).buffer };
  };
  assert(same(await httpReadAt(["u"], { fetch: fetch206 })(0, 500, 90), whole.subarray(500, 590)), "HTTP range read");
  assert(seen[0] === "bytes=500-589", `range header ${seen[0]}`);
  const fetch200 = async () => ({ status: 200, arrayBuffer: async () => whole.slice().buffer });
  assert(same(await httpReadAt(["u"], { fetch: fetch200 })(0, 0, 90), whole.subarray(0, 90)), "200 at offset 0");
  let threw = false;
  try { await httpReadAt(["u"], { fetch: fetch200 })(0, 90, 90); } catch { threw = true; }
  assert(threw, "a 200 for a range past 0 must not be taken as the range");
});

Deno.test("a table through the HTTP reader, tensor layout, equals dequantF32", async () => {
  const { cfg, shard, all } = synth(GGML_Q4_0);
  const fetch = async (url, { headers }) => {
    const [, a, b] = /bytes=(\d+)-(\d+)/.exec(headers.Range).map(Number);
    assert(url === "shard1", url);
    return { status: 206, arrayBuffer: async () => shard.slice(a, b + 1).buffer };
  };
  const t = new NgramTable(cfg, { readAt: httpReadAt([null, "shard1"], { fetch }), layout: "tensor", inflight: 4 });
  const st = new NgramState(cfg);
  // rows from real hashing on this small table
  const ids = st.push([1, 2, 3, 4, 50, 6, 7]);
  assert(same(await t.rows(ids), expect(all, cfg, ids)), "HTTP rows differ");
});
