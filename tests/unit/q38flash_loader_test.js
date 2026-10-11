// The qwen4exp (Qwen3.8-Flash-Next) loader against the real files' headers: bartowski's Q4_0 (3 shards) and
// ggml-org's MTP Q4_0, kept as tests/fixtures/q38flash/headers.json (metadata and tensor tables, no weights; rebuilt
// by tests/fixtures/q38flash/make.mjs). Checks the metadata (engine/gguf.js qwen4expConfig), that the name maps
// claim every tensor of the files exactly once, the shapes and types the engine will be written against, the byte
// sums, and qwen4expWeights' structure with a stub conversion. No GPU, no weights.
//   deno test --allow-read tests/unit/q38flash_loader_test.js
// With the real files on disk, one more test checks the fixture against them (ignored otherwise):
//   Q38FLASH_DIR=~/llm/models/q38flash deno test --allow-read --allow-env tests/unit/q38flash_loader_test.js
import { parseGGUFHeader, qwen4expConfig, qwen4expLayerNames, qwen4expNamesFor, qwen4expMtpNames, qwen4expPleTable,
  qwen4expWeights, qwen4expShardBytes, qwen4expMtpBytes, entryByteLength, ggufEntry, QWEN4EXP_HEAD,
  GGML_EMBED, GGML_OUTPUT, GGML_PLE_TABLE, GGML_F32, GGML_F16, GGML_BF16, GGML_Q4_0, GGML_Q4_1, GGML_Q5_0, GGML_Q6_K,
  GGML_Q8_0 } from "../../engine/gguf.js";
import { mergeSplitHeaders } from "../../room/models.js";
import { readFixture, ggufFromFixture } from "../helpers/gguf_fixture.js";

const eq = (a, b, m = "") => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${m} ${JSON.stringify(a)} != ${JSON.stringify(b)}`); };
const ok = (v, m = "assertion failed") => { if (!v) throw new Error(m); };
const throws = (fn, message) => {
  try { fn(); } catch (e) { ok(String(e).includes(message), String(e)); return; }
  throw new Error("expected a throw: " + message);
};
const test = Deno.test;

const FIX = readFixture("q38flash");
const parse = (f) => parseGGUFHeader(ggufFromFixture(f), { skipTokenizer: true });
// fresh headers per test (mergeSplitHeaders and the loaders' callers write into G)
function headers() {
  const hs = FIX.files.map(parse);
  return { G: mergeSplitHeaders(hs.slice(0, 3)), Gm: hs[3] };
}
const LAYERS = 48, QSA = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47];
const range = (n) => Array.from({ length: n }, (_, i) => i);

test("fixture is self-consistent: each header parses to its stored sizes, and its tensors fill the file to the byte", () => {
  eq(FIX.files.map((f) => f.size), [39853467552, 39647284192, 21098500928, 2199652640]);
  for (const f of FIX.files) {
    const G = parse(f);
    eq(G.headerBytes, f.headerBytes, f.name); eq(G.dataStart, f.dataStart, f.name);
    const end = Math.max(...Object.values(G.tensors).map((t) => t.byteOffset + t.byteLength));
    eq(end, f.size, f.name + " ends at its last tensor");
    ok(Object.values(G.tensors).every((t) => t.byteLength > 0), f.name + ": a type the loader can't size");
  }
});

// Q38FLASH_DIR: a folder with the four real files (the fixture's names). Unset, or no --allow-env: the test is ignored
function realDir() {
  try { return Deno.env.get("Q38FLASH_DIR") || null; } catch { return null; }
}
const REAL = realDir();
test({ name: "q38flash real files: the fixture matches the files in Q38FLASH_DIR (metadata, tensor tables, sizes)", ignore: !REAL, fn: () => {
  for (const f of FIX.files) {
    const path = `${REAL}/${f.name}`;
    eq(Deno.statSync(path).size, f.size, f.name + " size");
    const fd = Deno.openSync(path), buf = new Uint8Array(f.headerBytes);
    try { let n = 0; while (n < buf.length) { const r = fd.readSync(buf.subarray(n)); if (!r) break; n += r; } } finally { fd.close(); }
    const real = parseGGUFHeader(buf.buffer, { skipTokenizer: true }), fix = parse(f);
    eq([real.headerBytes, real.dataStart], [f.headerBytes, f.dataStart], f.name);
    const meta = (G) => Object.fromEntries(Object.entries(G.meta).filter(([k]) => !k.startsWith("tokenizer.")));
    eq(meta(fix), meta(real), f.name + " metadata");
    eq(fix.tensors, real.tensors, f.name + " tensors");
    // without a tokenizer to stand in for, the fixture writes the header back byte for byte
    if (!f.omitted.length) ok(new Uint8Array(ggufFromFixture(f)).every((b, i) => b === buf[i]), f.name + " header bytes");
  }
} });

test("header: the split model merges to its 1224 tensors; qwen4exp keys aliased under qwen35.", () => {
  const { G, Gm } = headers();
  eq(Object.keys(G.tensors).length, G.meta["split.tensors.count"]);
  eq(G.meta["split.tensors.count"], 1224);
  eq(Object.keys(Gm.tensors).length, 34);
  eq(G.tensors["blk.47.ffn_down_exps.weight"].shard, 2);
  eq(G.meta["general.architecture"], "qwen4exp");
  // the keys the shared code reads (room/models.js kvBytesPerLayerPos, stateBytesPerLayer) find the qwen35 names
  for (const k of ["block_count", "embedding_length", "expert_count", "ssm.inner_size", "attention.head_count_kv"])
    eq(G.meta["qwen35." + k], G.meta["qwen4exp." + k], k);
  eq(G.meta["qwen35.block_count"], 48); eq(Gm.meta["qwen35.block_count"], 49);
  ok(!("tokenizer.fixture_pad" in G.meta), "the padding key is skipped with the tokenizer");
});

test("qwen4expConfig: the trunk", () => {
  const { G } = headers();
  const c = qwen4expConfig(G.meta);
  eq([c.L, c.nextn, c.dim, c.hc, c.hcRank, c.hDim], [48, 0, 2560, 4, 320, 10240]);
  eq([c.nHeads, c.nKVHeads, c.headDim, c.ropeDims, c.ropeTheta, c.ropeSections], [24, 2, 256, 64, 1e7, [11, 11, 10, 0]]);
  eq(Math.fround(c.eps), Math.fround(1e-6));
  eq(range(LAYERS).filter((i) => c.qsa[i]), QSA);
  eq(c.ssm, { convK: 4, stateSize: 128, nKHeads: 16, nVHeads: 48, inner: 6144 });
  eq([c.nExp, c.K, c.expFF, c.shFF], [512, 10, 640, 640]);
  eq(c.indexer, { nHeads: 4, headDim: 128, topK: 2048, block: 4 });
  ok(!("qwen4exp.attention.recurrent_layers" in G.meta), "the trunk's layer types come from the interval");
  const p = c.ple;
  eq([p.layers, p.ngram, p.headsPerNgram, p.nHeads, p.headDim, p.convK, p.eos, p.image], [[1], 3, 8, 16, 160, 4, 248044, 248056]);
  eq(p.multipliers.map(String), ["23703573157769", "20109073645365", "8052911324071"]);
  // 16 contiguous head ranges of ~20M rows each, every size a prime; 320,001,446 rows in use
  for (let h = 1; h < 16; h++) eq(p.offsets[h], p.offsets[h - 1] + p.vocab[h - 1], "head " + h);
  eq([p.vocab[0], p.vocab[15]], [20000003, 20000171]);
  const prime = (n) => { for (let d = 2; d * d <= n; d++) if (n % d === 0) return false; return true; };
  ok(p.vocab.every(prime), "head vocab sizes are primes");
  eq(p.rows, 320001446);
});

test("qwen4expConfig: the MTP file and bad metadata", () => {
  const { G, Gm } = headers();
  const m = qwen4expConfig(Gm.meta);
  eq([m.L, m.nextn, m.qsa[48], m.ple], [48, 1, true, null]);
  throws(() => qwen4expConfig({ ...G.meta, "general.architecture": "qwen35moe" }), "not a qwen4exp GGUF");
  const { ["qwen4exp.hyper_connection.count"]: _, ...noHC } = G.meta;
  throws(() => qwen4expConfig(noHC), "missing key qwen4exp.hyper_connection.count");
  throws(() => qwen4expConfig({ ...G.meta, "qwen4exp.ple.layers": [3] }), "PLE layer 3 is not a DeltaNet layer");
  throws(() => qwen4expConfig({ ...G.meta, "qwen4exp.ple.layer_multipliers": [2 ** 60, 1, 1] }), "past 2^53");
  throws(() => qwen4expConfig({ ...G.meta, "qwen4exp.ple.head_offsets": [0] }), "PLE arrays");
  throws(() => qwen4expConfig({ ...G.meta, "qwen4exp.attention.compress_ratios": [0, 4] }), "2 compress ratios for 48 layers");
  // the MTP file's layer types come from recurrent_layers (block 48 is QSA there), as llama.cpp reads them
  eq(range(49).filter((i) => m.qsa[i]), [...QSA, 48]);
  eq(qwen4expConfig({ ...Gm.meta, "qwen4exp.attention.recurrent_layers": Array(49).fill(true) }).qsa.some(Boolean), false);
  throws(() => qwen4expConfig({ ...Gm.meta, "qwen4exp.attention.recurrent_layers": [true] }), "1 recurrent layer flags for 49 layers");
  // without recurrent_layers: every interval-th trunk layer, and every MTP block
  const { ["qwen4exp.attention.recurrent_layers"]: _r, ...noRecurrent } = Gm.meta;
  eq(range(49).filter((i) => qwen4expConfig({ ...noRecurrent, "qwen4exp.full_attention_interval": 6 }).qsa[i]), [5, 11, 17, 23, 29, 35, 41, 47, 48]);
  throws(() => qwen4expConfig({ ...G.meta, "qwen4exp.full_attention_interval": 0 }), "full_attention_interval is 0");
  // the indexer block is the QSA layers' one shared ratio: 0 without ratios, a single value for every layer, mixed refused
  const { ["qwen4exp.attention.compress_ratios"]: _c, ...noRatios } = G.meta;
  eq(qwen4expConfig(noRatios).indexer.block, 0);
  eq(qwen4expConfig({ ...G.meta, "qwen4exp.attention.compress_ratios": 8 }).indexer.block, 8);
  throws(() => qwen4expConfig({ ...G.meta, "qwen4exp.attention.compress_ratios": [...Array(47).fill(0), 8].map((r, i) => i === 3 ? 4 : r) }),
    "share one compress ratio, got 4 and 8");
  throws(() => qwen4expConfig({ ...G.meta, "qwen4exp.attention.compress_ratios": 3 }), "compress ratio 3 does not divide top_k 2048");
  throws(() => qwen4expConfig({ ...G.meta, "qwen4exp.attention.compress_ratios": 1 }), "compress ratio 1");
  // an older file without image_token_id (or with 0) hashes image positions as EOS, as llama.cpp does
  const { ["qwen4exp.ple.image_token_id"]: _i, ...noImage } = G.meta;
  eq(qwen4expConfig(noImage).ple.image, 248044);
  eq(qwen4expConfig({ ...G.meta, "qwen4exp.ple.image_token_id": 0 }).ple.image, 248044);
});

test("names: the layer, head, MTP and table names claim every tensor of both files exactly once", () => {
  const { G, Gm } = headers();
  const claims = (H, names) => {
    const seen = new Map();
    for (const n of names) {
      ok(H.tensors[n], "no tensor " + n);
      ok(!seen.has(n), "claimed twice: " + n); seen.set(n, true);
    }
    eq(Object.keys(H.tensors).filter((n) => !seen.has(n)), [], "unclaimed");
  };
  const strings = (N) => Object.values(N).filter((v) => typeof v === "string");
  claims(G, [...range(LAYERS).flatMap((i) => strings(qwen4expNamesFor(G, i))),
    GGML_EMBED, GGML_OUTPUT, ...Object.values(QWEN4EXP_HEAD), GGML_PLE_TABLE]);
  claims(Gm, [...strings(qwen4expNamesFor(Gm, 48)), ...Object.values(qwen4expMtpNames(48)), GGML_EMBED, GGML_OUTPUT]);
  // per layer: 8 HC + 8 MoE + 9 DeltaNet or 6 attention + 4 indexer, + 6 PLE on layer 1
  eq(strings(qwen4expNamesFor(G, 0)).length, 25); eq(strings(qwen4expNamesFor(G, 1)).length, 31);
  eq(strings(qwen4expNamesFor(G, 3)).length, 26);
  ok(!("attnNorm" in qwen4expNamesFor(G, 0)) && !G.tensors["output_norm.weight"], "no RMSNorm around the blocks, no output norm");
});

test("shapes and types: what the engine is written against", () => {
  const { G, Gm } = headers();
  const T = (H, n) => { const t = H.tensors[n]; ok(t, "no " + n); return [t.shape, t.ggmlType]; };
  const F32 = GGML_F32, Q4 = GGML_Q4_0;
  // vectors are f32 in GGUF order; a matrix is [rows (out), cols (in)]
  eq(T(G, GGML_EMBED), [[248320, 2560], Q4]);
  eq(T(G, GGML_OUTPUT), [[248320, 2560], GGML_Q6_K]);
  eq(T(G, QWEN4EXP_HEAD.hcNorm), [[10240], F32]);
  eq(T(G, QWEN4EXP_HEAD.hcDown), [[320, 10240], GGML_BF16]);
  eq(T(G, QWEN4EXP_HEAD.hcUp), [[10240, 320], GGML_BF16]);
  eq(T(G, GGML_PLE_TABLE), [[320001536, 160], Q4]);
  const types = {};
  const tally = (k, t) => { types[k] ??= {}; types[k][t] = (types[k][t] || 0) + 1; };
  for (const i of range(LAYERS)) {
    const N = qwen4expNamesFor(G, i);
    for (const s of ["Attn", "Ffn"]) {
      eq(T(G, N[`hc${s}Norm`]), [[10240], F32]);
      eq(T(G, N[`hc${s}Down`]), [[320, 10240], Q4]);
      eq(T(G, N[`hc${s}Up`]), [[10240, 320], Q4]);
      eq(T(G, N[`hc${s}Inject`]), [[4, 10240], GGML_BF16]);
    }
    eq(T(G, N.router), [[512, 2560], F32]);
    eq(T(G, N.shRouter), [[2560], F32]);
    eq(G.tensors[N.expGate].shape, [512, 640, 2560]); eq(G.tensors[N.expUp].shape, [512, 640, 2560]);
    eq(G.tensors[N.expDown].shape, [512, 2560, 640]);
    eq(G.tensors[N.shGate].shape, [640, 2560]); eq(G.tensors[N.shDown].shape, [2560, 640]);
    ok(G.tensors[N.expGate].ggmlType === Q4 && G.tensors[N.expUp].ggmlType === Q4, "gate/up experts Q4_0");
    tally("expDown", G.tensors[N.expDown].ggmlType + (i < 6 ? "@0-5" : ""));
    tally("shared", G.tensors[N.shGate].ggmlType);
    if (N.isFull) {
      eq(G.tensors[N.wq].shape, [12288, 2560]); tally("wq", G.tensors[N.wq].ggmlType);
      eq(T(G, N.wk), [[512, 2560], GGML_Q8_0]); eq(T(G, N.wv), [[512, 2560], GGML_Q8_0]);
      eq(G.tensors[N.wo].shape, [2560, 6144]);
      eq(T(G, N.qNorm), [[256], F32]); eq(T(G, N.kNorm), [[256], F32]);
      eq(T(G, N.idxQ), [[512, 2560], GGML_BF16]); eq(T(G, N.idxK), [[128, 2560], GGML_BF16]);
      eq(T(G, N.idxQNorm), [[128], F32]); eq(T(G, N.idxKNorm), [[128], F32]);
    } else {
      eq(G.tensors[N.wqkv].shape, [10240, 2560]); eq(G.tensors[N.wz].shape, [6144, 2560]);
      eq(T(G, N.wAlpha), [[48, 2560], F32]); eq(T(G, N.wBeta), [[48, 2560], F32]);
      eq(T(G, N.ssmA), [[48], F32]); eq(T(G, N.dtBias), [[48], F32]);
      eq(T(G, N.conv), [[10240, 4], F32]); eq(T(G, N.ssmNorm), [[128], F32]);
      eq(G.tensors[N.wOut].shape, [2560, 6144]); tally("ssmOut", G.tensors[N.wOut].ggmlType);
    }
  }
  // the bartowski "Q4_0" mix: Q4_1 down experts on layers 0-5 only, 6 of 12 attn_q Q6_K (the rest Q4_0), half of ssm_out Q8_0,
  // shared experts Q8_0 / Q5_0
  eq(types.expDown, { [`${GGML_Q4_1}@0-5`]: 6, [Q4]: 42 });
  eq(types.wq, { [Q4]: 6, [GGML_Q6_K]: 6 });
  eq(types.ssmOut, { [GGML_Q8_0]: 18, [Q4]: 18 });
  eq(Object.keys(types.shared).map(Number).sort((a, b) => a - b), [GGML_Q5_0, GGML_Q8_0]);
  const P = qwen4expNamesFor(G, 1);
  eq(T(G, P.pleKey), [[10240, 2560], Q4]); eq(T(G, P.pleValue), [[2560, 2560], Q4]);
  for (const k of ["pleNormKey", "pleNormQuery", "pleNormConv"]) eq(T(G, P[k]), [[10240], F32], k);
  eq(T(G, P.pleConv), [[10240, 4], GGML_F16]);
  // MTP: a QSA layer with its own HC mixers (inject in Q4_0 here), eh_proj [fc_embedding | fc_hidden]
  const M = qwen4expMtpNames(48), ML = qwen4expLayerNames(48, { qsa: true });
  eq(T(Gm, M.ehProj), [[2560, 5120], Q4]); eq(T(Gm, M.enorm), [[2560], F32]); eq(T(Gm, M.hnorm), [[10240], F32]);
  eq(T(Gm, M.hcHeadNorm), [[10240], F32]); eq(T(Gm, M.hcHeadDown), [[320, 10240], Q4]); eq(T(Gm, M.hcHeadUp), [[10240, 320], Q4]);
  eq(T(Gm, ML.hcAttnInject), [[4, 10240], Q4]); eq(T(Gm, ML.idxQ), [[512, 2560], GGML_BF16]);
  eq(T(Gm, GGML_OUTPUT), [[248320, 2560], Q4]);
});

test("bytes: file sizes, the n-gram table, experts and dense weights as the engine holds them", () => {
  const { G, Gm } = headers();
  const GB = 1e9;
  const data = FIX.files.slice(0, 3).reduce((n, f) => n + f.size - f.dataStart, 0);
  eq(FIX.files.slice(0, 3).reduce((n, f) => n + f.size, 0), 100599252672);   // 100.6 GB
  // everything but the table is a shard's load; the table is the rest, byte for byte
  const all = qwen4expShardBytes(G, { lo: 0, hi: LAYERS, hasEmbed: true, hasHead: true });
  eq(all + G.tensors[GGML_PLE_TABLE].byteLength, data);
  eq(G.tensors[GGML_PLE_TABLE].byteLength, 28800138240);   // 28.8 GB
  eq(qwen4expMtpBytes(Gm, { ownHead: true }) + FIX.files[3].dataStart, FIX.files[3].size);
  eq(qwen4expShardBytes(G, { lo: 0, hi: 0, hasEmbed: false, hasHead: true, mtp: Gm }) - qwen4expShardBytes(G, { lo: 0, hi: 0, hasEmbed: false, hasHead: true }),
    qwen4expMtpBytes(Gm));
  // split across devices, the layer ranges add up
  const part = (lo, hi, e = false, h = false) => qwen4expShardBytes(G, { lo, hi, hasEmbed: e, hasHead: h });
  eq(part(0, 14, true) + part(14, 48, false, true), all + G.tensors[GGML_EMBED].byteLength);   // both ends hold the embedding
  // experts as parked: one expert 2.77 MB at Q4_0, 3.59 MB on layers 0-5 (Q4_1 down widened to Q8); 70.5 GB in all
  const expertsOf = (i) => ["gate", "up", "down"].reduce((n, p) => n + entryByteLength(G.tensors[`blk.${i}.ffn_${p}_exps.weight`]), 0);
  eq(expertsOf(6) / 512, 2764800); eq(expertsOf(0) / 512, 3584000);
  eq([+(expertsOf(6) / GB).toFixed(2), +(expertsOf(0) / GB).toFixed(2)], [1.42, 1.84]);
  const experts = range(LAYERS).reduce((n, i) => n + expertsOf(i), 0);
  eq(+(experts / GB).toFixed(1), 70.5);
  // dense, read every token: ~3.4 GB with the LM head (0.68 GB, Q6_K widened), plus the QSA indexer (0.08 GB as
  // f32) once QSA reads it; the embedding stays in JS
  let dense = 0, indexer = 0;
  for (const t of Object.values(G.tensors)) {
    if (/_exps\./.test(t.name) || t.name === GGML_PLE_TABLE || t.name === GGML_EMBED) continue;
    if (t.name.includes(".indexer.")) indexer += entryByteLength(t); else dense += entryByteLength(t);
  }
  eq(+(entryByteLength(G.tensors[GGML_OUTPUT]) / GB).toFixed(2), 0.68);
  eq([+(dense / GB).toFixed(1), +(indexer / GB).toFixed(2)], [3.4, 0.08]);
  eq(+(G.tensors[GGML_EMBED].byteLength / GB).toFixed(2), 0.36);
});

test("entryByteLength: what convertEntry makes of each type", () => {
  const m = (ggmlType, shape) => entryByteLength({ ggmlType, shape });
  // per 32 weights: 16 bytes of nibbles or 32 int8, and a 2-byte scale
  eq(m(GGML_Q4_0, [64, 64]), 64 * 64 / 2 + 128 * 2);
  eq(m(GGML_Q8_0, [64, 64]), 64 * 64 + 128 * 2);
  eq(m(GGML_Q6_K, [256, 256]), 256 * 256 + 2048 * 2);   // requantized to Q8
  eq(m(GGML_BF16, [4, 64]), 4 * 64 * 4);
  eq(m(GGML_Q4_0, [256]), 256 * 4);   // vectors are dequantized
  eq(m(GGML_Q4_0, [2, 32, 64]), 2 * 32 * 64 / 2 + 128 * 2);   // stacked experts: one matrix
});

test("the n-gram table is never converted; qwen4expPleTable says where its rows are", async () => {
  const { G, Gm } = headers();
  let read = 0;
  try { await ggufEntry(G, async () => { read++; return new Uint8Array(0); }, GGML_PLE_TABLE); throw new Error("loaded"); }
  catch (e) { ok(String(e).includes("read by rows"), String(e)); }
  eq(read, 0);
  const P = qwen4expPleTable(G);
  eq([P.shard, P.rows, P.rowElems, P.rowBytes, P.ggmlType, P.heads.length], [0, 320001536, 160, 90, GGML_Q4_0, 16]);
  ok(!("url" in P), "the caller maps the shard to its URL");
  eq(P.byteOffset, G.tensors[GGML_PLE_TABLE].byteOffset); eq(P.byteLength, P.rows * P.rowBytes);
  eq(P.heads[15].firstRow + P.heads[15].rows, 320001446);
  ok(P.byteOffset + P.byteLength <= FIX.files[0].size, "inside the first shard");
  eq(qwen4expPleTable(Gm), null);
  const bad = { ...G, tensors: { ...G.tensors, [GGML_PLE_TABLE]: { ...G.tensors[GGML_PLE_TABLE], shape: [1000, 160] } } };
  throws(() => qwen4expPleTable(bad), "the heads need 320001446");
});

// qwen4expWeights with every tensor "converted" to its own name: the structure, the progress bytes, what is read
function stubbed(H, log) {
  H.convert = (info) => ({ name: info.name });
  return async (info) => { log.push(info.name); return new Uint8Array(0); };
}

test("qwen4expWeights: layers 0-3 with experts parked on 2-3, the n-gram layer and the table descriptor", async () => {
  const { G } = headers();
  const log = [], parked = [];
  const experts = { has: (i) => i >= 2, park: (i, part, e, nExp) => { parked.push([i, part, e.name, nExp]); return { parked: e.name }; } };
  let progress = 0;
  const W = await qwen4expWeights(G, stubbed(G, log), { lo: 0, hi: 4, hasEmbed: true, hasHead: false, experts }, (n) => { progress = n; });
  eq(progress, qwen4expShardBytes(G, { lo: 0, hi: 4, hasEmbed: true, hasHead: false }));
  ok(!log.includes(GGML_PLE_TABLE), "the table was read");
  eq(W.layers.map((L) => [L.isFull, !!L.indexer, !!L.ple, !!L.wqkv]), [[false, false, false, true], [false, false, true, true],
    [false, false, false, true], [true, true, false, false]]);
  const [, L1, L2, L3] = W.layers;
  eq(L1.ple.key.name, "blk.1.ple_key.weight"); eq(L1.ple.conv.name, "blk.1.ple_conv1d.weight");
  eq(L3.indexer.q.name, "blk.3.indexer.q_proj.weight"); eq(L3.wq.name, "blk.3.attn_q.weight");
  eq(L2.hcAttn.inject.name, "blk.2.hc_attn_inject.weight"); eq(L2.hcFfn.down.name, "blk.2.hc_ffn_down.weight");
  ok(!("attnNorm" in L2) && !("postNorm" in L2), "no norms");
  eq(W.layers[0].expDown.name, "blk.0.ffn_down_exps.weight");
  eq(L2.expDown, { parked: "blk.2.ffn_down_exps.weight" });
  eq(parked.map(([i, p, , n]) => `${i}${p}${n}`), ["2gate512", "2up512", "2down512", "3gate512", "3up512", "3down512"]);
  eq(W.embed.name, GGML_EMBED); ok(!W.head && !W.headHC && !W.mtp, "no head");
  eq(W.pleTable.rowBytes, 90);
  const W2 = await qwen4expWeights(G, stubbed(G, []), { lo: 2, hi: 4 });
  ok(!W2.pleTable && !W2.embed, "no table or embedding past the n-gram layer");
});

test("qwen4expWeights: the head and the MTP block from its own file", async () => {
  const { G, Gm } = headers();
  const log = [], mlog = [];
  let progress = 0;
  const W = await qwen4expWeights(G, stubbed(G, log), { lo: 47, hi: 48, hasEmbed: false, hasHead: true, mtp: Gm,
    mtpBytesOf: stubbed(Gm, mlog) }, (n) => { progress = n; });
  eq(progress, qwen4expShardBytes(G, { lo: 47, hi: 48, hasEmbed: false, hasHead: true, mtp: Gm }));
  eq([W.headHC.norm.name, W.headHC.down.name, W.headHC.up.name], Object.values(QWEN4EXP_HEAD));
  eq(W.head.name, GGML_OUTPUT); eq(W.embed.name, GGML_EMBED); ok(!("finalNorm" in W), "no output norm");
  const M = W.mtp;
  eq([M.ehProj.name, M.enorm.name, M.hnorm.name], ["blk.48.nextn.eh_proj.weight", "blk.48.nextn.enorm.weight", "blk.48.nextn.hnorm.weight"]);
  eq(M.hcHead.down.name, "blk.48.nextn.hc_head_down.weight");
  eq([M.layer.isFull, M.layer.indexer.k.name, M.layer.expGate.name], [true, "blk.48.indexer.k_proj.weight", "blk.48.ffn_gate_exps.weight"]);
  ok(mlog.every((n) => n.startsWith("blk.48.")) && log.every((n) => !n.startsWith("blk.48.")), "each file read through its own reader");
  ok(!("embed" in M), "the MTP file's own embedding and head only on request");
  const W2 = await qwen4expWeights(G, stubbed(G, []), { lo: 0, hi: 0, hasHead: true, mtp: Gm, mtpBytesOf: stubbed(Gm, []), mtpOwnHead: true });
  eq([W2.mtp.embed.name, W2.mtp.head.name], [GGML_EMBED, GGML_OUTPUT]);
  // a trunk header is not an MTP file
  await qwen4expWeights(G, stubbed(G, []), { lo: 0, hi: 0, hasHead: true, mtp: G }).then(() => { throw new Error("loaded"); },
    (e) => ok(String(e).includes("no nextn block"), String(e)));
});

test("qwen4expWeights: qsa: false skips the indexer, and the byte counts leave it out too", async () => {
  const { G, Gm } = headers();
  const log = [], mlog = [];
  const opts = { lo: 0, hi: LAYERS, hasEmbed: true, hasHead: true, mtp: Gm, qsa: false };
  let progress = 0;
  const W = await qwen4expWeights(G, stubbed(G, log), { ...opts, mtpBytesOf: stubbed(Gm, mlog) }, (n) => { progress = n; });
  ok(![...log, ...mlog].some((n) => n.includes(".indexer.")), "an indexer tensor was read");
  ok(W.layers[3].isFull && W.layers[3].indexer === null && W.mtp.layer.indexer === null, "indexer groups are null");
  eq(progress, qwen4expShardBytes(G, opts));
  const indexerBytes = (H) => Object.values(H.tensors).filter((t) => t.name.includes(".indexer.")).reduce((n, t) => n + t.byteLength, 0);
  eq(qwen4expShardBytes(G, { ...opts, qsa: true }) - progress, indexerBytes(G) + indexerBytes(Gm));
  eq(qwen4expMtpBytes(Gm) - qwen4expMtpBytes(Gm, { qsa: false }), indexerBytes(Gm));
});

test("qwen4expWeights: the MTP file uses its own hooks (G.convert by default), and parks its experts when the store has it", async () => {
  const { G, Gm } = headers();
  const cached = [], streamed = [], mlog = [];
  G.convert = (info) => ({ name: info.name, by: "G.convert" });
  G.entryCache = { get: async (info) => { cached.push(info.name); return null; }, put: async () => {} };
  G.streamEntry = async (info) => { streamed.push(info.name); return null; };
  const mread = async (info) => { mlog.push(info.name); return new Uint8Array(0); };
  const parked = [];
  const experts = { has: (i) => i === 48, park: (i, part, e) => { parked.push(`${i}${part}`); return { parked: e.name }; } };
  const W = await qwen4expWeights(G, async () => new Uint8Array(0), { lo: 0, hi: 0, hasHead: true, mtp: Gm, mtpBytesOf: mread, experts });
  ok(!cached.some((n) => n.startsWith("blk.48.")) && !streamed.some((n) => n.startsWith("blk.48.")), "G's cache or stream saw an MTP tensor");
  eq(W.mtp.ehProj, { name: "blk.48.nextn.eh_proj.weight", by: "G.convert" });
  eq(parked, ["48gate", "48up", "48down"]);
  eq(W.mtp.layer.expDown, { parked: "blk.48.ffn_down_exps.weight" });
  ok(!("convert" in Gm) && !("entryCache" in Gm), "the caller's MTP header was written to");
  // hooks the caller sets on the MTP header are the ones used for it
  const mcached = [];
  Gm.entryCache = { get: async (info) => { mcached.push(info.name); return null; }, put: async () => {} };
  await qwen4expWeights(G, async () => new Uint8Array(0), { lo: 0, hi: 0, hasHead: true, mtp: Gm, mtpBytesOf: mread });
  ok(mcached.includes("blk.48.nextn.eh_proj.weight"), "the MTP header's own cache");
});
