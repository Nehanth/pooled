import { MODELS, FILES, coopTuneOptions, weightFileBytes } from "../../room/models.js";
import { cacheKey } from "../../room/weightcache.js";

const eq = (a, b) => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${JSON.stringify(a)} != ${JSON.stringify(b)}`); };
const ok = (x, message) => { if (!x) throw new Error(message || "assertion failed"); };

Deno.test("model URLs pin weights, configs and tokenizers to immutable revisions", () => {
  for (const M of Object.values(MODELS)) {
    for (const url of [M.gguf, M.st, M.cfg, M.tok, ...(M.shards || [])].filter(Boolean)) {
      ok(/^https:\/\/huggingface\.co\/[^/]+\/[^/]+\/resolve\/[a-f0-9]{40}\//.test(url), url);
      const mutable = url.replace(/\/resolve\/[^/]+\//, "/resolve/main/");
      ok(cacheKey(url, 0, 1023) !== cacheKey(mutable, 0, 1023), "pinned ranges must not reuse mutable cache entries");
    }
    if (M.shards) eq(M.shards[0], M.gguf);
  }
});

Deno.test("weight total identifies the exact revision and individual split file", () => {
  for (const [key, F] of Object.entries(FILES)) {
    const M = MODELS[key];
    if (M.shards) {
      eq(M.shards.map(weightFileBytes), F.shards.map((s) => s.bytes));
      eq(F.shards.reduce((n, s) => n + s.bytes, 0), F.bytes);
    } else eq(weightFileBytes(M.gguf), F.bytes);
    eq(weightFileBytes(M.gguf.replace(/\/resolve\/[^/]+\//, "/resolve/main/")), null);
  }
  eq(weightFileBytes("https://localhost/model.gguf"), null);
});

Deno.test("autotune chooses a real layer projection and checks both converted formats", () => {
  const G = { tensors: {
    "token_embd.weight": { shape: [151936, 2048], ggmlType: 8 },
    "blk.0.attn_norm.weight": { shape: [2048], ggmlType: 0 },
    "blk.0.ffn_up_exps.weight": { shape: [256, 512, 2048], ggmlType: 2 },
    "blk.0.ffn_down_exps.weight": { shape: [256, 2048, 512], ggmlType: 3 },
    "blk.0.ffn_gate.weight": { shape: [512, 2048], ggmlType: 1 },
  } };
  eq(coopTuneOptions(G), { dIn: 2048, dOut: 512, kind: "q4", validateKinds: ["q8", "q4"] });
  G.tensors["blk.0.ffn_up_exps.weight"].ggmlType = 8;
  eq(coopTuneOptions(G), { dIn: 2048, dOut: 512, kind: "q8", validateKinds: ["q8"] });
});

Deno.test("autotune handles dense and float-only models without assuming the 27B shape", () => {
  eq(coopTuneOptions({ tensors: { "blk.0.ffn_up.weight": { shape: [6144, 2048], ggmlType: 8 } } }),
    { dIn: 2048, dOut: 6144, kind: "q8", validateKinds: ["q8"] });
  for (const ggmlType of [0, 1, 30]) eq(coopTuneOptions({ tensors: { "blk.0.ffn_up.weight": { shape: [6144, 2048], ggmlType } } }), null);
  eq(coopTuneOptions({ tensors: { invalid: { shape: [128, 33], ggmlType: 2 } } }), null);
  eq(coopTuneOptions(null), null);
});
