// room/models.js: maxSeqFor decides every room's context window (host and devices build their
// engines with it) and kvBytesPerLayerPos decides how many layers each device is dealt at that
// context. Both are pure; nothing else asserts them.
import { MODELS, NEED_GB, PICKER, CTX, MAX_SEQ, MAX_SEQ_LONG, MAX_NEW, MAX_NEW_THINKING, MIN_ROOM, maxSeqFor, kvBytesPerLayerPos, kvModeFor, KV_MODES } from "../../room/models.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

Deno.test("maxSeqFor: table of models and ?ctx= asks", () => {
  const cases = [
    // [model, ask, expected, why]
    ["qwen3.8-27b", undefined, 16384, "no ask: the model's default"],
    ["qwen3.8-27b", 0, 16384, "ask 0 = no ask"],
    ["qwen3.8-27b", -5, 16384, "a negative ask is ignored"],
    ["qwen3.8-27b", NaN, 16384, "NaN (a garbled ?ctx=) is ignored"],
    ["qwen3.8-27b", 4096, 4096, "an exact multiple of 256 is kept"],
    ["qwen3.8-27b", 4100, 4096, "rounds to the nearest 256 (down)"],
    ["qwen3.8-27b", 4224, 4352, "a tie at 128 rounds up"],
    ["qwen3.8-27b", 4223, 4096, "just under the tie rounds down"],
    ["qwen3.8-27b", 1, 2048, "tiny asks are floored at 2048"],
    ["qwen3.8-27b", 2047, 2048, "floor"],
    ["qwen3.8-27b", 32768, 32768, "exactly the cap"],
    ["qwen3.8-27b", 32769, 32768, "just over the cap is clamped"],
    ["qwen3.8-27b", 1e9, 32768, "huge asks are clamped"],
    ["qwen3.8-27b", Infinity, 32768, "Infinity is clamped"],
    ["qwen3.6-35b-moe", undefined, 32768, "MoE default"],
    ["qwen3.6-35b-moe", 65536, 65536, "MoE cap"],
    ["qwen3.6-35b-moe", 100000, 65536, "MoE clamp"],
    ["qwen3.6-35b-moe", 8000, 7936, "MoE rounding (8000/256 = 31.25)"],
    ["qwen3-1.7b", undefined, 8192, "dense default"],
    ["qwen3-1.7b", 20000, 16384, "dense cap"],
    ["qwen3-1.7b", 3000, 3072, "dense rounding (3000/256 = 11.7)"],
    // models without a CTX entry: the ask is ignored, the kind decides
    ["qwen3-0.6b", undefined, MAX_SEQ, "gguf without CTX: MAX_SEQ"],
    ["qwen3-0.6b", 16384, MAX_SEQ, "no CTX entry: ?ctx= is ignored"],
    ["qwen3-4b", 8192, MAX_SEQ, "gguf without CTX"],
    ["smollm-135m", undefined, MAX_SEQ, "safetensors: MAX_SEQ"],
    ["no-such-model", undefined, MAX_SEQ, "unknown model: MAX_SEQ"],
    ["no-such-model", 99999, MAX_SEQ, "unknown model ignores the ask"],
    [undefined, undefined, MAX_SEQ, "no model at all"],
  ];
  for (const [model, ask, want, why] of cases) {
    const got = ask === undefined ? maxSeqFor(model) : maxSeqFor(model, ask);
    eq(got, want, `${model} ask=${ask}: ${why}`);
  }
});

Deno.test("maxSeqFor: a qwen35 model without a CTX entry gets MAX_SEQ_LONG", () => {
  // every qwen35 in the catalogue has a CTX entry today; add one without, and it must fall back to
  // the long f16 window rather than the dense 2048
  const key = "__test-qwen35-no-ctx";
  MODELS[key] = { label: "t", kind: "qwen35", gguf: "x" };
  try {
    eq(maxSeqFor(key), MAX_SEQ_LONG);
    eq(maxSeqFor(key, 4096), MAX_SEQ_LONG, "the ask is still ignored without a CTX entry");
  } finally { delete MODELS[key]; }
});

Deno.test("maxSeqFor: ?ctx= arrives as a string or number the way room.js reads it", () => {
  // room.js: maxSeqFor(modelKey, +new URLSearchParams(location.search).get("ctx") || 0)
  const ask = (q) => +new URLSearchParams(q).get("ctx") || 0;
  eq(maxSeqFor("qwen3.8-27b", ask("?ctx=8192")), 8192);
  eq(maxSeqFor("qwen3.8-27b", ask("?ctx=abc")), 16384, "junk -> default");
  eq(maxSeqFor("qwen3.8-27b", ask("")), 16384, "missing -> default");
  eq(maxSeqFor("qwen3.8-27b", ask("?ctx=1e5")), 32768, "exponent form is clamped");
});

Deno.test("maxSeqFor: every result is a multiple of 256, within [2048, max] (sweep)", () => {
  for (const [model, c] of Object.entries(CTX)) {
    for (let ask = 1; ask <= c.max * 2; ask += 97) {
      const n = maxSeqFor(model, ask);
      ok(n % 256 === 0, `${model} ask=${ask} -> ${n} not a multiple of 256`);
      ok(n >= 2048 && n <= c.max, `${model} ask=${ask} -> ${n} outside [2048, ${c.max}]`);
      ok(Math.abs(n - Math.min(c.max, Math.max(2048, ask))) <= 128, `${model} ask=${ask} -> ${n} further than half a step`);
    }
  }
});

Deno.test("CTX table: defaults within caps, 256-aligned, for real models", () => {
  for (const [model, c] of Object.entries(CTX)) {
    ok(MODELS[model], `CTX entry ${model} is not in MODELS`);
    ok(c.def <= c.max, `${model}: def ${c.def} > max ${c.max}`);
    ok(c.def >= 2048 && c.def % 256 === 0 && c.max % 256 === 0, `${model}: not 256-aligned or under 2048`);
  }
});

Deno.test("CTX caps keep one layer's K (or V) buffer at or under 64 MB", () => {
  // the comment in models.js: the caps keep one layer's K or V buffer <= 64 MB (WebGPU's default
  // binding limit is 128 MB). K per position per layer = kvHeads * headDim * bytes.
  const shape = {
    "qwen3.8-27b": { kvHeads: 4, headDim: 256, bytes: 2 },      // f16 KV
    "qwen3.6-35b-moe": { kvHeads: 2, headDim: 256, bytes: 2 },  // f16 KV
    "qwen3-1.7b": { kvHeads: 8, headDim: 128, bytes: 4 },       // dense engine keeps f32 KV
  };
  for (const [model, c] of Object.entries(CTX)) {
    const s = shape[model];
    ok(s, `add ${model}'s KV shape to this test`);
    const bytes = c.max * s.kvHeads * s.headDim * s.bytes;
    ok(bytes <= 64 * 2 ** 20, `${model}: K buffer at max ctx ${c.max} is ${bytes} bytes`);
  }
});

Deno.test("kvBytesPerLayerPos: table of GGUF metadata", () => {
  const m = (kv, key, interval) => {
    const o = {};
    if (kv !== undefined) o["qwen35.attention.head_count_kv"] = kv;
    if (key !== undefined) o["qwen35.attention.key_length"] = key;
    if (interval !== undefined) o["qwen35.full_attention_interval"] = interval;
    return o;
  };
  const cases = [
    // 27B: 4 KV heads x 256, f16 K+V = 4 KB per attention layer per position, 1 layer in 4 -> 1 KB avg
    [m(4, 256, 4), 1024, "27B"],
    // 35B MoE: 2 x 256 -> 2 KB per attention layer, 1 in 4 -> 512
    [m(2, 256, 4), 512, "35B MoE"],
    [m(4, 256), 4096, "no interval: every layer is attention"],
    [m(4, 256, 0), 4096, "interval 0 is treated as 1 (no division by zero)"],
    [m(4, 256, 1), 4096, "interval 1"],
    [m(undefined, 256, 4), 0, "no KV heads: no KV cache"],
    [m(4, undefined, 4), 0, "no key length: no KV cache"],
    [{}, 0, "empty metadata"],
    [m(1, 1, 3), 4 / 3, "non-integer averages are kept"],
  ];
  for (const [meta, want, why] of cases) {
    const got = kvBytesPerLayerPos(meta);
    ok(Number.isFinite(got), `${why}: not finite (${got})`);
    ok(Math.abs(got - want) < 1e-9, `${why}: ${got} != ${want}`);
  }
});

Deno.test("kvBytesPerLayerPos x maxSeqFor: the 27B's whole KV cache at 16k is 1 GB (the comment's number)", () => {
  const meta = { "qwen35.attention.head_count_kv": 4, "qwen35.attention.key_length": 256, "qwen35.full_attention_interval": 4 };
  const layers = 64;   // 16 of them are attention layers
  const total = layers * maxSeqFor("qwen3.8-27b") * kvBytesPerLayerPos(meta);
  eq(total, 2 ** 30);
});

Deno.test("kvModeFor: int8 KV only when asked for, and only on the qwen35 engine", () => {
  eq(KV_MODES, ["f16", "q8"]);
  const qwen35 = Object.keys(MODELS).filter((k) => MODELS[k].kind === "qwen35");
  const other = Object.keys(MODELS).filter((k) => MODELS[k].kind !== "qwen35");
  ok(qwen35.length && other.length, "need both kinds in the catalogue");
  for (const k of qwen35) {
    eq(kvModeFor(k, "q8"), "q8", `${k} ?kv=q8`);
    for (const ask of [null, undefined, "", "f16", "Q8", "int8", "q4", "1"]) eq(kvModeFor(k, ask), "f16", `${k} ask ${ask}: off by default`);
  }
  for (const k of other) eq(kvModeFor(k, "q8"), "f16", `${k}: no int8 kernels, stays f16`);
  eq(kvModeFor("no-such-model", "q8"), "f16");
  eq(kvModeFor(undefined, "q8"), "f16");
});

Deno.test("kvModeFor: ?kv= read the way room.js reads it", () => {
  const ask = (q) => new URLSearchParams(q).get("kv");
  eq(kvModeFor("qwen3.8-27b", ask("?kv=q8")), "q8");
  eq(kvModeFor("qwen3.8-27b", ask("?kv=q8&ctx=32768")), "q8");
  eq(kvModeFor("qwen3.8-27b", ask("")), "f16", "missing -> f16");
  eq(kvModeFor("qwen3.8-27b", ask("?kv=")), "f16", "empty -> f16");
});

Deno.test("kvBytesPerLayerPos: int8 is values + one f32 scale per 32, ~56% of f16", () => {
  const m = (kv, key, interval) => ({ "qwen35.attention.head_count_kv": kv, "qwen35.attention.key_length": key, "qwen35.full_attention_interval": interval });
  // 27B: kvDim 1024 -> 1024 int8 + 32 scales * 4 = 1152 per K (or V), 2304 per attention layer, 1 in 4 -> 576
  eq(kvBytesPerLayerPos(m(4, 256, 4), "q8"), 576);
  eq(kvBytesPerLayerPos(m(2, 256, 4), "q8"), 288, "35B MoE");
  eq(kvBytesPerLayerPos(m(4, 256, 4), "f16"), 1024, "explicit f16 = the default");
  eq(kvBytesPerLayerPos(m(4, 256, 4), "junk"), 1024, "unknown format counts as f16");
  eq(kvBytesPerLayerPos({}, "q8"), 0, "no KV cache");
  for (const [kv, key] of [[4, 256], [2, 256], [8, 128], [1, 32]]) {
    const r = kvBytesPerLayerPos(m(kv, key, 1), "q8") / kvBytesPerLayerPos(m(kv, key, 1));
    ok(Math.abs(r - 0.5625) < 1e-12, `${kv}x${key}: q8/f16 ${r}`);
  }
});

Deno.test("kvBytesPerLayerPos x maxSeqFor: the 27B's int8 KV cache at 32K is 1.125 GB (the changelog's 1.2 GB)", () => {
  const meta = { "qwen35.attention.head_count_kv": 4, "qwen35.attention.key_length": 256, "qwen35.full_attention_interval": 4 };
  eq(64 * maxSeqFor("qwen3.8-27b", 32768) * kvBytesPerLayerPos(meta, "q8"), 1.125 * 2 ** 30);
  eq(64 * maxSeqFor("qwen3.8-27b", 32768) * kvBytesPerLayerPos(meta), 2 * 2 ** 30, "f16: 2 GB");
});

Deno.test("catalogue: picker models exist, have memory needs and URLs by kind", () => {
  for (const k of PICKER) { ok(MODELS[k], `picker ${k} not in MODELS`); ok(NEED_GB[k] > 0, `picker ${k} has no NEED_GB`); }
  for (const k of Object.keys(NEED_GB)) ok(MODELS[k], `NEED_GB ${k} not in MODELS`);
  for (const [k, M] of Object.entries(MODELS)) {
    ok(["gguf", "qwen35", "safetensors"].includes(M.kind), `${k}: unknown kind ${M.kind}`);
    ok(M.label, `${k}: no label`);
    if (M.kind === "safetensors") ok(/^https:\/\/.*\.safetensors$/.test(M.st) && M.cfg && M.tok, `${k}: safetensors URLs`);
    else ok(/^https:\/\/.*\.gguf$/.test(M.gguf), `${k}: gguf URL`);
    if (M.kind === "gguf") ok(M.cfg && M.tok, `${k}: dense gguf needs cfg and tok`);
  }
});

Deno.test("answer budgets fit in the smallest context", () => {
  ok(MIN_ROOM > 0 && MAX_NEW > MIN_ROOM && MAX_NEW_THINKING > MAX_NEW);
  ok(MAX_NEW_THINKING + MIN_ROOM < MAX_SEQ, "a thinking answer must fit the 2048 window with room for a prompt");
  ok(MAX_SEQ_LONG > MAX_SEQ);
});
