// Where does a MoE decode token go? Time forwardToken with groups of kernels skipped (outputs are wrong
// while skipping; only the timing matters). Also counts dispatches per token by pipeline.
import { Qwen35Engine } from "../engine/qwen35.js";
import { makeTokenizer } from "../engine/engine.js";
import { qwen35Weights, tokenizerFromGGUF } from "../engine/gguf.js";
import { openGGUF, MOE_PATH, roomFlags } from "./load_model.js";
import { roomQwen35Options, applyRoomFlags } from "../engine/preset.js";
const PATH = Deno.env.get("MOE") || MOE_PATH;
const model = openGGUF(PATH);   // converted-weights cache (tests/weight_cache.js)
const readAt = model.readAt;
const ad = await navigator.gpu.requestAdapter(); const device = await ad.requestDevice({ requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
const G = model.G; const tok = makeTokenizer(tokenizerFromGGUF(G.meta));
const arch = G.meta["general.architecture"], L = G.meta[arch + ".block_count"] - (G.meta[arch + ".nextn_predict_layers"] || 0);
const weights = await qwen35Weights(G, (i) => readAt(i.byteOffset, i.byteLength), { lo: 0, hi: L, hasEmbed: true, hasHead: true });
// the room's settings (engine/preset.js); MOE_FUSE=0 / MOE_DN_ROWS=N are ?moefuse / ?moednrows, ROOM_FLAGS takes any other
const flags = roomFlags({ ...(Deno.env.get("MOE_FUSE") ? { moefuse: Deno.env.get("MOE_FUSE") } : {}), ...(Deno.env.get("MOE_DN_ROWS") ? { moednrows: Deno.env.get("MOE_DN_ROWS") } : {}) });
const eng = applyRoomFlags(await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 512,
  vocab: G.tensors["token_embd.weight"]?.shape?.[0], ...roomQwen35Options(flags) }), flags);   // MOE_FUSE=0: unfused MoE kernels (A/B)
// count dispatches per pipeline for one token
const count = {}; const wrap = (fn, nameOf) => function (...a) { const n = nameOf(a); count[n] = (count[n] || 0) + 1; return fn.apply(this, a); };
const o3 = eng._d3, oxyz = eng._dxyz, od = eng._d;
eng._d3 = wrap(o3, (a) => a[1]); eng._dxyz = wrap(oxyz, (a) => typeof a[1] === "string" ? a[1] : "?"); eng._d = wrap(od, (a) => a[1]);
const ids = tok.encode("The capital of France is"); for (const id of ids.slice(0, -1)) await eng.forwardToken(id);
for (const k in count) delete count[k]; await eng.forwardToken(ids.at(-1));
eng._d3 = o3; eng._dxyz = oxyz; eng._d = od;
const total = Object.values(count).reduce((a, b) => a + b, 0);
console.log(`dispatches per token: ${total}`); console.log(Object.entries(count).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k}:${v}`).join("  "));
const all = Object.keys(count);
const time = async (skip, n = 24) => { eng.skip = new Set(skip); await eng.forwardToken(1); const t0 = performance.now(); for (let i = 0; i < n; i++) await eng.forwardToken(1); eng.skip = null; return (performance.now() - t0) / n; };
const pick = (re) => all.filter((k) => re.test(k));
const groups = [["nothing skipped", []], ["everything (fixed cost)", all],
  ["MoE experts (gu+dn)", pick(/^moe_(gu|dn)/)], ["MoE router+combine", pick(/^moe_(route|combine)/)],
  ["router GEMV + shared expert", []], ["attention (full layers)", pick(/^(attn|kv|flash|rope|qk|head_norm|q_split|ks|fa)/i)],
  ["DeltaNet recurrence", pick(/^dn_/)], ["rmsnorm", pick(/^rmsnorm/)]];
for (const [name, sk] of groups) { if (name.startsWith("router GEMV")) continue; const ms = await time(sk); console.log(`${name.padEnd(28)} ${ms.toFixed(2)} ms/token  (skips ${sk.length} pipes)`); }
// matvec/coop pipes are shared by many ops: skip them all to see the dense-GEMV share
console.log(`${"all GEMV (matvec/coop)".padEnd(28)} ${(await time(pick(/^(mv|matvec|coop|gu_|gemv)/))).toFixed(2)} ms/token`);
