// Model catalogue for the room: URLs, layer counts, memory needs, context length.

// GB each model needs at its default context (the picker's "needs N GB"); NEED_MIN_GB: what a model
// with a fallback context (CTX[model].fallback) needs there, the least a room can start it with
export const NEED_GB = { "qwen3-0.6b": 0.8, "qwen3-1.7b": 5.6, "qwen3-4b": 4.6, "qwen3.8-27b": 17.0, "qwen3.6-35b-moe": 22.5, "qwen3.5-122b-moe": 72.0,
  // Qwen3.8-Flash-Next: its 48 layers, the embedding and the head (71.8 GB in the file); the 28.8 GB n-gram table is
  // not GPU memory (it stays on the disk of the device holding layer 1: SHAPE ngram, room/plan.js dealRoom)
  "qwen3.8-flash": 74.5, "smollm-135m": 0.6 };
export const NEED_MIN_GB = { "qwen3-1.7b": 4.0 };

// What the model host holds besides its layers, in bytes, from the file's own tensor sizes: the
// embedding stays in JS memory for row lookups, the output head goes to the GPU (the embedding again
// when the two are tied), and the Qwen 3.5/3.8 engines add the draft (MTP) block and, for the
// one-submit draft chain, a GPU copy of the embedding table (engine/qwen35.js). Qwen3.8-Flash-Next (qwen4exp) keeps
// its embedding in JS memory too; its `out` counts the head's HC mixer with output.weight, and its draft block (a
// separate MTP file) is not loaded in rooms yet (mtp 0). This is what the
// host's pledge pays for before its layers (room/plan.js layerCaps).
export function hostHeldBytes(kind, { embed = 0, out = 0, mtp = 0 } = {}) {
  const held = embed + (out || embed);
  return kind === "qwen35" ? held + mtp + (mtp ? embed : 0) : kind === "qwen4exp" ? held + mtp : held;
}
// K+V bytes per layer and position for a dense (Qwen3) room: the dense engine keeps its KV cache in
// f32 (engine/dense.js), one K and one V row of kvDim per position
export const denseKvBytesPerLayerPos = (kvDim) => 2 * kvDim * 4;

// ---- what expert offload parks (engine/expert_store.js ExpertStore.park) ----
// The loader converts each expert tensor before it is parked (engine/gguf.js convertEntry): Q4_0 stays Q4 (nibbles +
// an f16 scale per 32: 18/32 B per weight), Q8_0 and every other quant (Q4_1, Q5_0, Q5_K, Q6_K) become Q8 (int8 + an
// f16 scale per 32: 34/32 B per weight). So a Q4_1 tensor parks 34/32 B per weight, not the file's 20/32.
const Q4 = 2, Q41 = 3;   // engine/gguf.js GGML_Q4_0, GGML_Q4_1
const FLOAT_TYPES = [0, 1, 30];   // F32, F16, BF16: ExpertStore can't park these
// One expert's six parked slices (gate qs, gate scales, up qs, up scales, down qs, down scales: ExpertStore.SLICES),
// in bytes, for experts of rows x cols weights (gate / up: [ffn, hidden]; down: [hidden, ffn]) of these GGML types
export function expertSlices(ffn, hidden, types) {
  const n = ffn * hidden;
  return types.flatMap((t) => [t === Q4 ? n / 2 : n, n / 16]);
}
// A layer's parked slices from a GGUF header (G.tensors[name]: { shape: [nExp, rows, cols], ggmlType }), or null when
// it has no routed experts. -> { nExp, slices: [6] }
export function expertSlicesOf(G, i) {
  const ts = ["gate", "up", "down"].map((p) => G.tensors[`blk.${i}.ffn_${p}_exps.weight`]);
  if (ts.some((t) => !t)) return null;
  const slices = [];
  for (const t of ts) {
    if (FLOAT_TYPES.includes(t.ggmlType) || t.shape?.length !== 3) return null;
    const n = t.shape[1] * t.shape[2];
    slices.push(t.ggmlType === Q4 ? n / 2 : n, n / 16);
  }
  return { nExp: ts[0].shape[0], slices };
}
// The experts' offload profile (room/plan.js offloadNeed / offloadPlan) for layers 0..L-1 of a header, or null for a
// dense model: { E: one layer's routed expert bytes in the file (layers 0-3's average: the part of a layer's VRAM
// estimate that parking frees), nExp, slices: per layer, one expert's six parked slices (what ExpertStore parks) }
// ... and what a device that offloads needs past its layers and its expert cache (room/plan.js offloadSpare): kvPos
// (f16 K+V bytes per layer and position, averaged: a checkpoint's KV rows), state (a layer's recurrent state, averaged:
// DeltaNet S and conv window, which a checkpoint copies and speculation's rollback keeps once more) and work (the
// engine's working buffers: prefill frames, batched passes; offloadWork)
// Qwen3.8-Flash-Next (qwen4exp) has its own sizes: K / V in f32 on its 12 QSA layers (averaged per layer), the DeltaNet
// state from its own keys, no whole-layer region (flashOffload) and checkpoints of at most its 2048 positions.
export function expertsOf(G, L) {
  const per = [];
  for (let i = 0; i < L; i++) { const x = expertSlicesOf(G, i); if (!x) return null; per.push(x); }
  if (!per.length || per.some((x) => x.nExp !== per[0].nExp)) return null;
  const M = G.meta || {};
  const base = { E: expertBytesOf(G, 0, Math.min(4, L)) / Math.min(4, L), nExp: per[0].nExp, slices: per.map((x) => x.slices) };
  if (M["general.architecture"] === "qwen4exp") return { ...base, ...flashOffload(flashMeta(M)) };
  return { ...base, kvPos: kvBytesPerLayerPos(M, "f16"), state: stateBytesPerLayer(M), work: offloadWork(+M["qwen35.embedding_length"] || 0) };
}
// the fields flashOffload reads, from a qwen4exp header (per-layer arrays: the QSA flags as engine/gguf.js
// qwen4expConfig sets them, every interval-th layer when the file has no recurrent_layers key)
function flashMeta(M) {
  const k = (n) => M["qwen4exp." + n];
  const L = k("block_count") - (k("nextn_predict_layers") || 0), rec = k("attention.recurrent_layers"), iv = k("full_attention_interval") ?? 4;
  const qsa = Array.from({ length: L }, (_, i) => (Array.isArray(rec) ? !rec[i] : (i + 1) % iv === 0)).filter(Boolean).length;
  return { L, qsa, K: k("expert_used_count"), dim: k("embedding_length"), nKV: k("attention.head_count_kv"), headDim: k("attention.key_length"),
    ssm: { state: k("ssm.state_size"), nVH: k("ssm.time_step_rank"), nKH: k("ssm.group_count"), inner: k("ssm.inner_size"), convK: k("ssm.conv_kernel") } };
}
// Expert offload's extra sizes for Qwen3.8-Flash-Next (room/plan.js offloadSpare, expertSizes), from flashMeta's fields:
//   kvPos   K + V in f32 (the phase-1 engine), 2 x nKV x headDim x 4 per QSA layer and position, averaged over the layers
//   state   a DeltaNet layer's S and conv window, averaged (the QSA layers have none)
//   work    the engine's working buffers (offloadWork by the hidden size)
//   region  false: engine/qwen4exp.js builds its expert store without the whole-layer region (decode one token at a
//           time: a cut never holds more than K experts), so the deal must not set one aside
//   ckptTokens  the checkpoints' KV rows: 8 slots of at most 2048 positions, not the Qwen 3.5 rooms' 32k
//   K       experts a token picks (10 of 512): room/plan.js tabDiskCost
export const FLASH_CKPT_TOKENS = 8 * 2048;
export function flashOffload(f) {
  const s = f.ssm, dn = (f.L - f.qsa) / f.L;
  return { kvPos: 2 * f.nKV * f.headDim * 4 * f.qsa / f.L,
    state: (s.nVH * s.state * s.state * 4 + (2 * s.nKH * s.state + s.inner) * (s.convK - 1) * 4) * dn,
    work: offloadWork(f.dim), region: false, ckptTokens: FLASH_CKPT_TOKENS, K: f.K || 10 };
}
// A Qwen3.5 / 3.6 layer's recurrent state, averaged over the layers (all but every full_attention_interval-th are
// DeltaNet): S (nVH x dState x dState f32) and the conv window (3 x convDim f32), as engine/qwen35.js allocates them
export function stateBytesPerLayer(meta) {
  const dState = +meta["qwen35.ssm.state_size"] || 0, nVH = +meta["qwen35.ssm.time_step_rank"] || 0;
  const convDim = 2 * dState * (+meta["qwen35.ssm.group_count"] || 0) + (+meta["qwen35.ssm.inner_size"] || 0);
  const k = +meta["qwen35.full_attention_interval"] || 1;
  return (nVH * dState * dState * 4 + 3 * convDim * 4) * (k - 1) / k;
}
// The engine's working buffers on a device that offloads, beyond its layers and the cache: the prefill frames' and
// batched passes' activations, allocated on the first long prompt. Measured with every buffer counted (Dawn, a worker
// shard, a 12k-token prefill, decode and verifies): 0.16 GiB on the 35B (hidden 2048), 0.30 GiB on the 122B (3072);
// D3D12 (the RTX 5070 PC) adds about 70% to that (64 KiB placement, Dawn's own heaps). 192 KiB per hidden unit
// covers both with that margin: 0.38 / 0.56 GiB.
export const offloadWork = (dim) => dim * 192 * 1024;
// SHAPE's park (runs of layers that park alike) as expertsOf's profile
const parkProfile = (s) => s.park && { E: s.exp, nExp: s.park.nExp, slices: s.park.runs.flatMap(([n, sl]) => Array(n).fill(sl)),
  ...(s.flash ? flashOffload(s.flash) : { kvPos: s.kvPos.f16, state: stateBytesPerLayer(s.meta), work: offloadWork(s.meta["qwen35.embedding_length"]) }) };

// the header fields stateBytesPerLayer and offloadWork read (SHAPE's MoE models: their GGUFs' values)
function ssmMeta(dim, nVH, nKH, inner) {
  return { "qwen35.embedding_length": dim, "qwen35.ssm.state_size": 128, "qwen35.ssm.time_step_rank": nVH, "qwen35.ssm.group_count": nKH,
    "qwen35.ssm.inner_size": inner, "qwen35.full_attention_interval": 4 };
}
// Each picker model's shape, from its GGUF header (the host reads the same numbers from the file at
// Start, so these only let the picker say before Start whether the room's pledges hold the model):
// L layers; layer = one layer's weights (the largest group's average for the hybrids); kvPos = K+V
// bytes per layer and position by KV format; embed / out / mtp: the host-only tensors; exp: one layer's routed
// experts in the file (the MoE models; layers 0-3's average, as `layer`); park: what ExpertStore parks per expert, by
// runs of layers (expertSlices; the 35B's layers 0-4 and the 122B's 0-5 have Q4_1 down experts, parked as Q8).
export const SHAPE = {
  "qwen3-1.7b": { kind: "gguf", L: 28, layer: 53494784, kvPos: { f16: denseKvBytesPerLayerPos(1024) }, embed: 330612736, out: 0, mtp: 0 },
  "qwen3.8-27b": { kind: "qwen35", L: 64, layer: 223970464, kvPos: { f16: 1024, q8: 576 }, embed: 715161600, out: 1042944000, mtp: 265197568 },
  "qwen3.6-35b-moe": { kind: "qwen35", L: 40, layer: 498197568, kvPos: { f16: 512, q8: 288 }, embed: 286064640, out: 417177600, mtp: 897955840, exp: 469762048,
    park: { nExp: 256, runs: [[5, expertSlices(512, 2048, [Q4, Q4, Q41])], [35, expertSlices(512, 2048, [Q4, Q4, Q4])]] }, meta: ssmMeta(2048, 32, 16, 4096) },
  "qwen3.5-122b-moe": { kind: "qwen35", L: 48, layer: 1485116672, kvPos: { f16: 512, q8: 288 }, embed: 429096960, out: 625766400, mtp: 2682195968, exp: 1409286144,
    park: { nExp: 256, runs: [[6, expertSlices(1024, 3072, [Q4, Q4, Q41])], [42, expertSlices(1024, 3072, [Q4, Q4, Q4])]] }, meta: ssmMeta(3072, 64, 16, 8192) },
  // Qwen3.8-Flash-Next (llama.cpp arch qwen4exp; roadmap/33), bartowski Q4_0 (three files), read from its headers:
  // layer: layers 0-3's average without QSA's indexer, the size room.js aiStart measures from the header
  // (qwen4expShardBytes over [0, 4) / 4), so the picker's need is the one Start checks. It must go with exp (also
  // layers 0-3's): the planner keeps layer - exp on the GPU for every offloaded layer, and the file's whole average
  // (1.476 GB) less exp left ~10 MB a layer where the file has 50-63 MB, so the picker said Flash fit in 5.7 GB of
  // GPU memory and Start, at ~8 GB, refused it. Layers 0-5 carry Q4_1 down experts, so this is a little over the
  // average: dealRoom has one layerBytes for every layer, and the heavier front is what the host holds. kvPos: the 12 QSA layers'
  // K+V, 2 KV heads x 256, kept in f32 by the phase-1 engine (engine/qwen4exp.js, PR #346), so 4 KB per QSA layer and
  // position, 1 KB averaged over 4 (the room's KV format stays "f16": kvModeFor); out: output.weight and the head's
  // HC mixer (output_hc_*); exp / park: 512 experts of 2560 x 640, layers 0-5 with Q4_1 down experts (parked as Q8).
  // hDim: the 4-stream hyper-connection residual (4 x 2560) is what crosses devices (room/pipeline.js wireDim).
  // ngram: the n-gram (PLE) table, read row by row by the device holding its layer (engine/ngram.js); never on a GPU
  // and never part of a layer's bytes. pleState: that layer's extra state a checkpoint copies: the PLE conv history
  // (10 positions x hDim f32, engine/qwen4exp.js st.ring).
  "qwen3.8-flash": { kind: "qwen4exp", L: 48, layer: 1531208864, kvPos: { f16: 1024 }, embed: 357580800, out: 534620160, mtp: 0, exp: 1468006400,
    park: { nExp: 512, runs: [[6, expertSlices(640, 2560, [Q4, Q4, Q41])], [42, expertSlices(640, 2560, [Q4, Q4, Q4])]] }, meta: ssmMeta(2560, 48, 16, 6144),
    hDim: 10240, ngram: { layer: 1, bytes: 28800138240 }, pleState: 10 * 10240 * 4,
    // flashOffload's fields (expert offload: room/plan.js), from the header (flashMeta)
    flash: { L: 48, qsa: 12, K: 10, dim: 2560, nKV: 2, headDim: 256, ssm: { state: 128, nVH: 48, nKH: 16, inner: 6144, convK: 4 } } },
};
// The width of the hidden state that crosses devices, in floats: the model's hidden size, or for Qwen3.8-Flash-Next
// its 4-stream residual (SHAPE hDim). 0 when SHAPE does not say (the room reads it from the engine: wire.js wireDim).
export const wireWidth = (model) => SHAPE[model]?.hDim || +SHAPE[model]?.meta?.["qwen35.embedding_length"] || 0;
// Where a model's n-gram table must be: { layer, bytes } (the device holding that layer reads it from its own disk,
// room/plan.js dealRoom ngram), or null for every model without one
export const ngramOf = (model) => SHAPE[model]?.ngram || null;
// A split GGUF's headers (llama.cpp gguf-split: MODELS[key].shards, in order) as one: the first file's metadata and
// every file's tensors, each tensor with its file (shard: index; url: that file's URL when urls is given, for the
// loaders that fetch by URL). Offsets stay within their own file. -> the first header, merged
export function mergeSplitHeaders(headers, urls = null) {
  const G = headers[0], n = +G.meta?.["split.count"] || headers.length;
  if (n !== headers.length) throw new Error(`a split GGUF of ${n} files, ${headers.length} read`);
  for (let i = 1; i < headers.length; i++)
    for (const [name, t] of Object.entries(headers[i].tensors)) G.tensors[name] = { ...t, shard: i, ...(urls ? { url: urls[i] } : {}) };
  return G;
}
// A MoE GGUF's routed expert bytes over layers [lo, hi) (G: a parsed header): what a device that offloads parks per
// layer (room/plan.js offloadNeed); 0 for a dense model
export function expertBytesOf(G, lo, hi) {
  let n = 0;
  for (let i = lo; i < hi; i++) for (const p of ["gate", "up", "down"]) n += G.tensors[`blk.${i}.ffn_${p}_exps.weight`]?.byteLength || 0;
  return n;
}
// { L, layerBytes, hostBytes, expertBytes } for a room running `model` at `ctx` positions with `kv` format, or
// null for a model without a SHAPE (the picker then falls back to NEED_GB). expertBytes: 0 for a dense model;
// experts: the MoE models' offload profile (expertsOf), else null.
export function roomBytes(model, ctx, kv = "f16") {
  const s = SHAPE[model]; if (!s) return null;
  return { L: s.L, layerBytes: s.layer + ctx * (s.kvPos[kv] ?? s.kvPos.f16), hostBytes: hostHeldBytes(s.kind, s), expertBytes: s.exp || 0,
    experts: parkProfile(s) || null };
}

// The whole weights file per picker model, in GB (the GGUF's size on Hugging Face). A room splits it:
// each device downloads about its share of the layers, so the picker can say what this device will fetch.
// Qwen3.8-Flash-Next: 71.8 of its 100.6 GB (FILES), without the 28.8 GB n-gram table, which only the device holding
// layer 1 fetches (SHAPE ngram, ngramOf), on top of its share: counting it here would put ~29 GB on every device.
export const FILE_GB = { "qwen3-1.7b": 1.83, "qwen3.8-27b": 16.06, "qwen3.6-35b-moe": 20.84, "qwen3.5-122b-moe": 72.52, "qwen3.8-flash": 71.8 };

// Each GGUF's exact size and SHA-256 (Hugging Face's x-linked-size / x-linked-etag), so `pooled pull`
// can check what it downloaded (cli/lib/cache.js)
export const FILES = {
  "qwen3-0.6b": { bytes: 639446688, sha256: "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031" },
  "qwen3-1.7b": { bytes: 1834426016, sha256: "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a" },
  "qwen3-4b": { bytes: 4280404704, sha256: "8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300" },
  "qwen3.8-27b": { bytes: 16056478688, sha256: "ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d" },
  "qwen3.6-35b-moe": { bytes: 20836243072, sha256: "52312daa5b2190c1f5723d33c3315c01c55af4206f6c6e6eb63f3d8dd52bb85e" },
  // a split GGUF (llama.cpp gguf-split, two files): bytes is the whole model, shards each file's size and SHA-256
  "qwen3.5-122b-moe": { bytes: 72517482816, shards: [
    { bytes: 39917224704, sha256: "717ab3efe330cb8581e7233bb93ad0c965fdb351ce5ae61a6f7b1252ba6925d9" },
    { bytes: 32600258112, sha256: "28d9accacf99f2e87c6b2f4a199556067775af6ecccd76e31904f5bb0fd5867f" }] },
  // three files at the pinned revision (MODELS: resolve/928589f...), from the Hugging Face API (paths-info, lfs oid);
  // the first holds the 28.8 GB n-gram table. mtp: ggml-org's MTP file at its pinned revision (not loaded in rooms yet)
  "qwen3.8-flash": { bytes: 100599252672, shards: [
    { bytes: 39853467552, sha256: "d2dbb1e9e21fc2b214c78dfba33e2db7c6d36efe81e10bd329c3ca90314d9685" },
    { bytes: 39647284192, sha256: "d2b40751578e1e7a3725fa506410ba01dab544ca039d7da59ee9c8cfd10d4b5a" },
    { bytes: 21098500928, sha256: "1404069b7cd14cf79d1631b8cf5c5fdff1bb4edd45d82c443e93503569c6706f" }],
    mtp: { bytes: 2199652640, sha256: "d484e6e541a36b714976f72f7d1a86ae2cc450cff110b81e3656512784abd1c6" } },
};

// The models the room's picker offers, with the DISK_COPY models below. The others stay for tests and ?dev=1.
export const PICKER = ["qwen3-1.7b", "qwen3.8-27b", "qwen3.6-35b-moe"];

// Models that keep a copy on each device's disk and stream their experts from it, so they run on far less memory
// than their size. The picker's row and its two settings under Start (GPU memory, Disk) come from this entry and the
// room's own figures (room/diskpick.js), so another big model gets the same picker by adding one.
//   name: the row's name; params / arch: its chip ("125B · MoE · from disk"); active: parameters used per token (the
//   longer description: "125B (6B active)"; the model card's numbers, roadmap/33)
//   diskBytes: the most one tab stores (diskCopyBytes: every layer's experts and the n-gram table, room/flashdisk.js),
//   the same bytes room.js hostDiskBytes gives the host's disk question for a deal that puts all of them on its disk
//   ("92.4 GB")
export const diskCopyBytes = (key) => { const s = SHAPE[key]; return s ? s.exp * s.L + (s.ngram?.bytes || 0) : 0; };
export const DISK_COPY = {
  "qwen3.8-flash": { name: "Qwen3.8 Flash", params: "125B", active: "6B", arch: "MoE", diskBytes: diskCopyBytes("qwen3.8-flash") },
};

const FLASH_BASE = "https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/resolve/928589fdb66c6ff07f22ac561e3fbce76553548f/Qwen3.8-Flash-Next-Q4_0/";
export const MODELS = {
  "qwen3-0.6b": { label: "Qwen3 0.6B · Q8", kind: "gguf",
    gguf: "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json" },
  "qwen3-1.7b": { label: "Qwen3 1.7B · Q8", kind: "gguf",
    gguf: "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/tokenizer.json" },
  "qwen3-4b": { label: "Qwen3 4B · Q8", kind: "gguf",
    gguf: "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/tokenizer.json" },
  "qwen3.8-27b": { label: "Qwen3.8 27B \u00b7 Q4", kind: "qwen35",
    gguf: "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_0.gguf" },
  // mixture of experts: 256 experts, 8 active per token (~3B of 35B), so decode reads far less than the 27B
  "qwen3.6-35b-moe": { label: "Qwen3.6 35B MoE \u00b7 Q4", kind: "qwen35",
    gguf: "https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" },
  // 256 experts, 8 active (~10B of 122B), 48 layers, 16 query heads per kv head. Two files (shards: the loaders read
  // both, room/models.js mergeSplitHeaders). In the ?dev=1 list, not the picker: 72 GB of weights, a room of big
  // machines (or a PC that offloads its experts to RAM: room/plan.js), tests/test_moe.js MODEL=122b, `pooled pull`
  "qwen3.5-122b-moe": { label: "Qwen3.5 122B MoE \u00b7 Q4", kind: "qwen35",
    gguf: "https://huggingface.co/bartowski/Qwen_Qwen3.5-122B-A10B-GGUF/resolve/main/Qwen_Qwen3.5-122B-A10B-Q4_0/Qwen_Qwen3.5-122B-A10B-Q4_0-00001-of-00002.gguf",
    shards: ["00001", "00002"].map((n) => `https://huggingface.co/bartowski/Qwen_Qwen3.5-122B-A10B-GGUF/resolve/main/Qwen_Qwen3.5-122B-A10B-Q4_0/Qwen_Qwen3.5-122B-A10B-Q4_0-${n}-of-00002.gguf`) },
  // Qwen3.8-Flash-Next (roadmap/33): 512 experts, 10 active (~6B of 125B), a 4-stream residual and a 28.8 GB n-gram
  // table read from disk by the device holding layer 1. ?dev=1 only, like the 122B: its own engine (engine/qwen4exp.js,
  // PR #346; room/flash.js adapts it to the room), context <= 2048 until QSA (CTX). Pinned revisions: FILES' sizes and
  // hashes are these files'. mtp: the draft block's file (not loaded in rooms yet).
  "qwen3.8-flash": { label: "Qwen3.8 Flash Next \u00b7 Q4", kind: "qwen4exp",
    gguf: FLASH_BASE + "Qwen3.8-Flash-Next-Q4_0-00001-of-00003.gguf",
    shards: ["00001", "00002", "00003"].map((n) => `${FLASH_BASE}Qwen3.8-Flash-Next-Q4_0-${n}-of-00003.gguf`),
    mtp: "https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF/resolve/052beeaca7bec4a303e59cc7bc630c4f3a1b845d/mtp-Qwen3.8-Flash-Next-Q4_0.gguf" },
  "smollm-135m": { label: "SmolLM 135M · bf16", kind: "safetensors",
    st: "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/model.safetensors",
    cfg: "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/config.json",
    tok: "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json" },
};

// Context window per room, in tokens: prompt + answer. Each full-attention layer keeps K and V
// for this many positions (4 KB per position each for the 27B, so 16 MiB per attention layer at
// 2048); the kernels only use it as a stride. Generation stops before the cache would overflow.
export const MAX_SEQ = 2048;
// The 27B family keeps its KV cache in f16 with split-K flash attention (engine attnFlash), so its
// rooms get 8192 positions for the price of 4096 in f32: 32 MiB per attention layer, ~0.5 GB for
// the whole model, spread over the devices that hold the layers. The host reads the engine's
// maxSeq, so prompts, answer budgets and "context full" all follow it.
export const MAX_SEQ_LONG = 8192;
// Context per model, in tokens: the default a room opens with and the most ?ctx=N may ask for, and
// for some a fallback: the context a room opens with instead when its pledges can't hold the model
// at the default but can at the fallback (pickCtx), so a room short of memory still starts.
// These hybrids keep a KV cache only on their full-attention layers (1 in 4), and the DeltaNet
// layers hold a fixed-size state, so context is cheap in memory: f16 K+V is 64 KB per position
// for the 27B (16 attention layers, 4 KV heads x 256) and 20 KB for the 35B MoE (10 layers,
// 2 KV heads x 256). 32k on the MoE = 0.64 GB, 128k = 2.5 GB, spread over the devices holding
// those layers. One layer's K (or V) is one GPU buffer bound whole: 128 MiB for the MoE at 131072
// positions in f16, exactly the binding size every WebGPU device supports (so no device limits the
// MoE); the 27B needs 256 MiB at 131072 in f16 (128 MiB in int8), which the room only asks for
// when every device can bind it (ctxForBinding); its cap stays at 64K (128 MiB) until 128K is
// checked on it. The practical limit is prefill speed, not memory
// (docs/long-context-and-sessions.md: needle and speed at 32K..128K).
export const CTX = {
  "qwen3.8-27b": { def: 16384, max: 65536 },
  "qwen3.6-35b-moe": { def: 32768, max: 131072 },
  // the 122B MoE: 12 attention layers, 2 KV heads x 256 (the 35B's K/V row): 24 KB per position in f16, 128 MiB per
  // layer's K at 131072. Checked at short contexts so far (phase 1): its default stays at the 35B's
  "qwen3.5-122b-moe": { def: 32768, max: 131072 },
  // Qwen3.8-Flash-Next: dense attention is exact up to 2051 cached tokens and the phase-1 engine refuses more (QSA,
  // roadmap/33 PR plan item 9, lifts it). 12 QSA layers, 2 KV heads x 256 in f32: 4 MiB per layer's K at 2048
  "qwen3.8-flash": { def: 2048, max: 2048 },
  // the dense engine keeps an f32 KV cache (~224 KB per position on the 1.7B: 1.8 GB at 8k, 3.6 GB
  // at 16k: 5.6 GB for the room in all). 16k fits an agent's prompt (OpenClaw's alone is ~12k), so
  // the room page, `pooled host` and the OpenClaw plugin all open it at 16k; a room short of that
  // (one 8 GB laptop lends 4 GB) opens it at 8k (4 GB), so one laptop still runs it alone.
  "qwen3-1.7b": { def: 16384, max: 16384, fallback: 8192 },
};
// WebGPU's default maxStorageBufferBindingSize: what a device binds when it reports nothing
export const WEBGPU_MIN_BIND = 128 * 2 ** 20;
// bytes of one attention layer's K (or V) cache buffer at ctx positions: kvDim x 1 (int8) or 2 (f16)
export const kvLayerBufBytes = (meta, ctx, kv = "f16") =>
  ctx * (meta["qwen35.attention.head_count_kv"] || 0) * (meta["qwen35.attention.key_length"] || 0) * (kv === "q8" ? 1 : 2);
// The longest context (<= ctx, a multiple of 256) whose per-layer K/V buffer fits bindBytes, the
// smallest binding limit among a room's devices (a device that reports none counts as WebGPU's 128 MiB).
export const ctxForBinding = (meta, ctx, kv = "f16", bindBytes = WEBGPU_MIN_BIND) => {
  const per = kvLayerBufBytes(meta, 1, kv);
  if (!per) return ctx;
  return Math.min(ctx, Math.floor(Math.max(bindBytes, WEBGPU_MIN_BIND) / per / 256) * 256);
};
export const maxSeqFor = (model, ask = 0) => {
  const c = CTX[model];
  if (!c) return MODELS[model]?.kind === "qwen35" ? MAX_SEQ_LONG : MAX_SEQ;
  return ask > 0 ? Math.min(c.max, Math.max(2048, Math.round(ask / 256) * 256)) : c.def;
};
// The contexts a room tries for `model`, best first: `want` (the context it would open with), then the
// model's fallback when it has one below `want`. An explicit ask (?ctx=, --ctx) is tried alone.
export const ctxChoices = (model, want = maxSeqFor(model), ask = 0) => {
  const fb = CTX[model]?.fallback;
  return ask > 0 || !(fb > 0) || fb >= want ? [want] : [want, fb];
};
// The context a room opens with: the first of ctxChoices whose deal fits (fitsAt(ctx) -> bool), else
// the smallest of them (what a short room is short for: the least it could start with).
// -> { ctx, want, fits, fellBack } (fellBack: it fits, but only at a shorter context than `want`)
export function pickCtx(model, { want = maxSeqFor(model), ask = 0, fitsAt }) {
  const cs = ctxChoices(model, want, ask);
  const i = cs.findIndex((c) => !!fitsAt(c));
  const ctx = cs[i < 0 ? cs.length - 1 : i];
  return { ctx, want, fits: i >= 0, fellBack: i > 0 };
}
// "16K", "8K": a context in words
export const ctxK = (n) => `${Math.round(n / 1024)}K`;
// What the room says where it shows the model when it fell back: "Qwen3 1.7B · 8K context: the
// room's memory is short for 16K"
export const ctxShortNote = (label, ctx, want) => `${label} · ${ctxK(ctx)} context: the room's memory is short for ${ctxK(want)}`;
// "needs 5.6 GB (4 GB at 8K)": a model's need at its default, and at its fallback when it has one
export const needText = (model, gb = NEED_GB[model]) => {
  const fb = CTX[model]?.fallback, min = NEED_MIN_GB[model];
  return fb && min != null ? `${gb} GB (${min} GB at ${ctxK(fb)})` : `${gb} GB`;
};

// KV cache format per room: "f16" (the default) or "q8" (int8 values + one f32 scale per 32, engine
// kvQ8, ~56% of f16's memory; asked for with ?kv=q8). Only the qwen35 engine has the int8 kernels, so
// any other model, or any other ask, stays f16. The host decides and sends it with ai-load, so every
// device of a room keeps the same format and the layer deal counts the right bytes.
export const KV_MODES = ["f16", "q8"];
export const kvModeFor = (model, ask) => (ask === "q8" && MODELS[model]?.kind === "qwen35" ? "q8" : "f16");
// A device's format for an ai-load: the host's `kv` when it sent one (so a worker's own ?kv= cannot
// make it differ from the room), else this device's own ask (a host from before the field existed).
export const kvForLoad = (model, sent, ownAsk) => kvModeFor(model, sent ?? ownAsk);
// K+V bytes per position, averaged over a model's layers (for dealing layers by memory), f16 or int8
export const kvBytesPerLayerPos = (meta, kv = "f16") => {
  const kvDim = (meta["qwen35.attention.head_count_kv"] || 0) * (meta["qwen35.attention.key_length"] || 0);
  const perKV = kv === "q8" ? kvDim + kvDim / 32 * 4 : kvDim * 2;   // int8 values + f32 scales, or f16
  return perKV * 2 / (meta["qwen35.full_attention_interval"] || 1);
};
// The browser snapshots these loading options when it creates its loader. Null prefetch keeps
// the live device policy; engine presets are separate and read when an engine is constructed.
// ?wcache=0 disables converted weights; ?wcacheverify=1 verifies them; ?prefetch=N overrides lookahead.
export function modelLoadOptions(search) {
  const q = new URLSearchParams(search), prefetch = q.get("prefetch");
  // parload=0: a Flash tab's load parts one after the other, as before room/flashload.js; ngramworker=0: its n-gram
  // table copied on the main thread (after the experts')
  return { wcache: q.get("wcache") !== "0", wcacheVerify: q.get("wcacheverify") === "1",
    prefetch: prefetch === null ? null : Math.max(0, parseInt(prefetch, 10) || 0),
    parload: q.get("parload") !== "0", ngramWorker: q.get("ngramworker") !== "0" };
}

export const MAX_NEW = 400;    // longest answer, tokens
export const MAX_NEW_THINKING = 1200;   // with thinking on, the think block comes out of the same budget
export const MIN_ROOM = 32;    // a prompt must leave at least this many tokens for the answer
