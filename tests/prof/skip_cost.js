// What each kernel family costs inside a real decode token (one compute pass, no per-dispatch timestamps): the
// back-to-back GPU time per token with that family skipped (engine.skip), against the full token, in alternating
// rounds. Per-dispatch timestamps (tests/prof_ts.js) put every kernel in its own pass and add ~5-10 us to each;
// this measures what removing the kernel saves. Outputs are garbage while a family is skipped (only timing).
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights prof/skip_cost.js
//   MODEL=moe|27b  N=24 (tokens per measure)  ROUNDS=3  FAMS='moe_route;moe_nrt' (default: every pipeline a token dispatches)
import { Qwen35Engine } from "../../engine/qwen35.js";
import { openGGUF, gpuDevice, trunkLayers, MOE_PATH, Q38_PATH } from "../load_model.js";
import { roomQwen35Options } from "../../engine/preset.js";
const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "moe"), N = +env("N", 24), ROUNDS = +env("ROUNDS", 3);
const model = openGGUF(MODEL === "moe" ? MOE_PATH : Q38_PATH);
const { device } = await gpuDevice();
const G = model.G, L = trunkLayers(G), tok = model.tokenizer();
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true });
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 2048, ...roomQwen35Options("") });
const ids = tok.encode("<|im_start|>user\nWrite a long, detailed essay about the history of computing.<|im_end|>\n<|im_start|>assistant\n");
eng.reset(); await eng.prefillTokens(ids.slice(0, -1)); await eng.forwardToken(ids.at(-1));
const pos0 = eng.pos, desc = { kind: "greedy" }, emb = eng._embedRowF32(ids.at(-1));
// which pipelines one token dispatches, and how often
const count = new Map();
{
  const enc0 = device.createCommandEncoder.bind(device);
  device.createCommandEncoder = (d) => { const e = enc0(d); const ob = e.beginComputePass.bind(e);
    e.beginComputePass = (pd) => { const p = ob(pd); const sp = p.setPipeline.bind(p), dw = p.dispatchWorkgroups.bind(p); let cur = null;
      p.setPipeline = (x) => { cur = x; return sp(x); }; p.dispatchWorkgroups = (...a) => { const n = Object.entries(eng.pipes).find(([, v]) => v === cur)?.[0] || "?"; count.set(n, (count.get(n) || 0) + 1); return dw(...a); }; return p; };
    return e; };
  eng._fwdPre = null; eng._encodeForward(pos0, desc);
  device.createCommandEncoder = enc0;
}
async function b2b(skip) {
  eng.skip = skip; eng._fwdPre = null;
  const jobs = []; for (let i = 0; i < N; i++) jobs.push(eng._encodeForward(pos0 + i, desc));
  await device.queue.onSubmittedWorkDone();
  const t0 = performance.now();
  for (let i = 0; i < N; i++) { eng._setFrame(pos0 + i, pos0 + i + 1); device.queue.writeBuffer(eng.x, 0, emb); device.queue.submit([jobs[i].cb]); }
  await device.queue.onSubmittedWorkDone();
  eng.skip = null;
  return (performance.now() - t0) / N;
}
const fams = env("FAMS", "") ? env("FAMS").split(";").map((f) => f.split(",")) : [...count.keys()].map((k) => [k]);
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
await b2b(null);
const rows = [];
for (const f of fams) {
  const full = [], cut = [];
  for (let r = 0; r < ROUNDS; r++) { full.push(await b2b(null)); cut.push(await b2b(new Set(f))); }
  const d = med(full) - med(cut), n = f.reduce((a, k) => a + (count.get(k) || 0), 0);
  rows.push({ fam: f.join("+"), n, ms: d, us: n ? d * 1000 / n : 0, full: med(full) });
  console.log(`${f.join("+").padEnd(36)} ${String(n).padStart(4)}x  saves ${d.toFixed(3)} ms/token (${(n ? d * 1000 / n : 0).toFixed(1)} us each), full ${med(full).toFixed(3)} ms`);
}
console.log(`sum of savings ${rows.reduce((a, r) => a + r.ms, 0).toFixed(2)} ms of ${med(rows.map((r) => r.full)).toFixed(2)} ms/token`);
Deno.exit(0);
