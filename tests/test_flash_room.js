// Qwen3.8-Flash-Next split across two engines the way a room splits it (room/flash.js over engine/qwen4exp.js, PR
// #346): the host runs the embedding and layer 0, a worker runs layers 1..N-1 with the n-gram (PLE) layer, and the
// residual crosses between them at hDim = 10240 floats with the token ids (room/wire.js). Checked against one engine
// holding layers 0..N-1:
//   exact:  the residual handed over in f32 -> the worker's output equals the single engine's, bit for bit
//   wire:   handed over in f16 (packWire / splitWire, as a room sends it) -> relative error at f16's rounding
//   head:   the host's headFromHidden (the head's HC mixer and output.weight, what a room host runs for every token)
//           on the worker's residual equals the single engine's logits (forwardToken wantLogits), and greedy decoding
//           through the split picks the same ids
//   slots:  saveSlot / loadSlot and exportSlot / importState on both ends restore the PLE conv ring and the n-gram
//           predecessors: the continuation after a restore is bit-identical
// Needs engine/qwen4exp.js (PR #346) and the model (bartowski Qwen3.8-Flash-Next-Q4_0 in models/q38flash, or
// Q38F_DIR=<dir>). Each engine loads its own layers (~1.5 GB each): about 3N layers on the GPU in all (LAYERS: N,
// default 3), plus the head twice.
//   deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights tests/test_flash_room.js
import fs from "node:fs";
import path from "node:path";
import { Qwen4ExpEngine } from "../engine/qwen4exp.js";
import { qwen4expWeights, qwen4expConfig, gpuUploadEntry } from "../engine/gguf.js";
import { ngramConfig, NgramTable } from "../engine/ngram.js";
import { FlashRoomEngine } from "../room/flash.js";
import { packWire, splitWire } from "../room/wire.js";
import { openGGUF, gpuDevice, watchGpuErrors } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const DIR = env("Q38F_DIR", new URL("../models/q38flash", import.meta.url).pathname);
const first = fs.readdirSync(DIR).filter((f) => /^Qwen3\.8-Flash-Next-.*-00001-of-\d{5}\.gguf$/.test(f))[0];
if (!first) throw new Error("no Qwen3.8-Flash-Next shards in " + DIR);
const N = +env("LAYERS", 3);

const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(path.join(DIR, first));
const G = model.G, cfg = qwen4expConfig(G.meta);
if (!(N >= 2 && N <= cfg.L)) throw new Error("LAYERS must be 2.." + cfg.L);
const t0 = performance.now();
const up = (e, name) => { if (name !== "token_embd.weight") gpuUploadEntry(device, e, false); };
const fds = model.paths.map((p) => fs.openSync(p, "r"));
const readAt = async (file, off, len) => { const b = new Uint8Array(len); fs.readSync(fds[file], b, 0, len, off); return b; };
const table = () => new NgramTable(ngramConfig(G), { readAt, layout: "tensor" });
// each engine loads its own weights: the engine owns them (it stacks some matrices into one and frees the parts)
const mk = async (lo, hi, hasEmbed, hasHead = false) => {
  const weights = await qwen4expWeights(G, model.bytesOf, { lo, hi, hasEmbed, hasHead, qsa: false }, undefined, up);
  device.queue.submit([device.createCommandEncoder().finish()]);
  await device.queue.onSubmittedWorkDone();
  return Qwen4ExpEngine.create({ device, meta: G.meta, hasEmbed, hasHead, layerRange: [lo, hi], weights,
    ngram: cfg.ple.layers.some((l) => l >= lo && l < hi) ? table() : null });
};
const ref = await mk(0, N, true), refHead = await mk(0, N, true, true);
const host = FlashRoomEngine.wrap(await mk(0, 1, true, true)), work = FlashRoomEngine.wrap(await mk(1, N, false));
console.log(`qwen4exp room split: layers 0..${N - 1} + head loaded in ${((performance.now() - t0) / 1000).toFixed(0)} s; host [0, 1) + head, worker [1, ${N}) with the n-gram layer`);

let fail = 0;
const check = (c, what) => { console.log(`  ${c ? "ok  " : "FAIL"} ${what}`); if (!c) fail++; };
const rel = (a, b) => { let n = 0, d = 0; for (let i = 0; i < b.length; i++) { const e = a[i] - b[i]; n += e * e; d += b[i] * b[i]; } return Math.sqrt(n / (d || 1)); };
const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
// a prompt with an EOS inside (the n-gram context restarts after it)
const ids = [151644, 872, 198, 9707, 11, 1246, 525, 498, 30, 248044, 13, 576, 3974, 13876, 38835, 34208];

// 1. exact (the residual handed over in f32), then the wire (f16 with the id, as a room sends it)
const want = [];
ref.reset();
for (let p = 0; p < ids.length; p++) want.push((await ref.forwardToken(ids[p])).hidden);
for (const wire of [false, true]) {
  host.reset(); work.reset();
  let worst = 0, exact = true, idsOk = true, width = true;
  for (let p = 0; p < ids.length; p++) {
    const h = await host.embedRun(ids[p], p);
    width &&= h.length === cfg.hDim;
    let got;
    if (!wire) got = await work.runHidden(h, p, { ids: [ids[p]] });
    else {
      const back = splitWire(packWire(h, [ids[p]]), 1, cfg.hDim);
      idsOk &&= back.ids?.[0] === ids[p];
      got = await work.runHidden(back.h, p, { ids: back.ids });
    }
    exact &&= same(got, want[p]);
    worst = Math.max(worst, rel(got, want[p]));
  }
  check(width, `the host's residual is hDim = ${cfg.hDim} floats`);
  if (!wire) check(exact, `f32 hand-over: the worker's residual equals the single engine's, bit for bit (${ids.length} tokens)`);
  else {
    check(idsOk, "the token ids cross the wire with the residual");
    check(worst < 2e-2, `f16 wire (20 KB + 4 B per token): worst relative error over ${ids.length} tokens ${worst.toExponential(2)} (< 2e-2)`);
  }
}

// 2. the head on the host (qwen4expCore.head: headOps into B.hl, then logitStage), on the residual the worker returns,
// against the single engine's own head: the logits, then greedy decoding fed back through the split
{
  const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
  host.reset(); work.reset(); refHead.reset();
  let exact = true, worst = 0, ids1 = true, next = 0;
  for (let p = 0; p < ids.length; p++) {
    const want = await refHead.forwardToken(ids[p], { wantLogits: true });
    const got = await host.headFromHidden(await work.runHidden(await host.embedRun(ids[p], p), p, { ids: [ids[p]] }));
    exact &&= same(got, want.logits);
    worst = Math.max(worst, rel(got, want.logits));
    ids1 &&= argmax(got) === want.id;
    next = want.id;
  }
  check(exact, `headFromHidden: the host's logits equal the single engine's, bit for bit (${ids.length} tokens, vocab ${host.dims.vocab}; worst rel. error ${worst.toExponential(2)})`);
  check(ids1, "and its argmax is the single engine's greedy id at every position");
  const G1 = [], G2 = [];
  let a = next, b = next;   // from the prompt's last greedy id, both at pos ids.length
  for (let k = 0, p = ids.length; k < 8; k++, p++) {
    a = (await refHead.forwardToken(a)).id; G1.push(a);
    const lg = await host.headFromHidden(await work.runHidden(await host.embedRun(b, p), p, { ids: [b] }));
    b = argmax(lg); G2.push(b);
  }
  check(same(G1, G2), `greedy through the split: the same 8 ids as one engine (${G2.join(" ")})`);
}

// 3. slots and disk copies on both ends (PR #346 must create its K/V caches with COPY_SRC for a worker with a QSA
// layer, LAYERS >= 5; without it that engine has no slot methods: room/flash.js ckpt)
check(host.ckpt && work.ckpt, "every state buffer can be checkpointed (COPY_SRC)");
if (!(host.ckpt && work.ckpt)) { console.log("FLASH ROOM FAIL"); Deno.exit(1); }
const run = async (toks, base) => { const out = []; for (let k = 0; k < toks.length; k++) out.push(await work.runHidden(await host.embedRun(toks[k], base + k), base + k, { ids: [toks[k]] })); return out; };
host.reset(); work.reset();
await run(ids.slice(0, 12), 0);   // saved two tokens past the EOS: both predecessors are real tokens
host.saveSlot(7); work.saveSlot(7);
const cont = await run(ids.slice(12), 12);
const dumped = [await host.exportSlot(7), await work.exportSlot(7)];
host.reset(); work.reset();
await run([9707, 9707, 9707], 0);
host.loadSlot(7); work.loadSlot(7);
const again = await run(ids.slice(12), 12);
check(again.every((h, k) => same(h, cont[k])), "loadSlot: the continuation is bit-identical (DeltaNet, K/V, PLE ring, n-gram predecessors)");
host.reset(); work.reset();
await run([13, 13], 0);
host.importState(dumped[0]); work.importState(dumped[1]);
const fromDisk = await run(ids.slice(12), 12);
check(fromDisk.every((h, k) => same(h, cont[k])), "exportSlot -> importState: bit-identical too");
const wl = dumped[1].parts.at(-1);
check(wl.byteLength === 12 && new Int32Array(wl)[2] === 12 && new Int32Array(wl)[0] === ids[11] && new Int32Array(wl)[1] === ids[10], "the worker's state ends with the n-gram predecessors [p1, p2, pos]");
// without the predecessors the PLE rows differ: the checks above would catch a lost one
work.loadSlot(7); work.core.ngs.reset(); host.loadSlot(7);
const lost = await run(ids.slice(12, 13), 12);
check(!same(lost[0], cont[0]), "(control) a reset n-gram state changes the next residual");

if (errors.count) { console.log(`${errors.count} GPU errors`); fail++; }
console.log(fail ? "FLASH ROOM FAIL" : "FLASH ROOM PASS ✓");
Deno.exit(fail ? 1 : 0);
