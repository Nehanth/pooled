// Real Qwen3-0.6B Q8 golden through the extracted runtime, solo and across two shards.
// No DOM or PeerJS: only activation delivery is in-process. The production pipeline still
// packs/unpacks its f16 frames, queues worker compute, carries resets and resolves token laps.
// Run: deno run --unstable-webgpu --allow-read tests/test_runtime.js
// 27B Qwen35Engine split: npm run test:runtime:q38 (requires models/q38, ~16 GB GPU weights).
import { DenseEngine, argmax, makeTokenizer } from "../engine/engine.js";
import { parseGGUFHeader, ggufWeights } from "../engine/gguf.js";
import { createPipeline } from "../room/pipeline.js";
import { createGenerator } from "../engine/generate.js";

const assert = (ok, message) => { if (!ok) throw new Error(message); };
const equal = (a, b, message) => assert(JSON.stringify(a) === JSON.stringify(b),
  `${message}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`);
const dir = new URL(".", import.meta.url).pathname;
let cfg, golden, tok, raw, G, bytesOf;
async function loadDense() {
  cfg = JSON.parse(await Deno.readTextFile(dir + "../models/qwen/config.json"));
  golden = JSON.parse(await Deno.readTextFile(dir + "golden/golden_qwen.json"));
  tok = makeTokenizer(JSON.parse(await Deno.readTextFile(dir + "../models/qwen/tokenizer.json")));
  raw = await Deno.readFile(dir + "../models/qwen/model.gguf");
  G = parseGGUFHeader(raw.buffer);
  bytesOf = (info) => new Uint8Array(raw.buffer, info.byteOffset, info.byteLength);
}
async function checkMode(split) {
  const label = split ? "split" : "solo";
  // Browser GPUAdapters can be consumed by requestDevice; each mode requests a fresh one.
  const adapter = await navigator.gpu.requestAdapter();
  assert(adapter, `${label}: no WebGPU adapter`);
  console.log(`runtime ${label} adapter:`, JSON.stringify(adapter.info));
  // One device per mode, destroyed before the next mode allocates anything. No forced GC,
  // duplicated full-model engines, or changes to model/kernel configuration are needed.
  const device = await adapter.requestDevice({ requiredLimits: {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  } });
  const gpuErrors = [], peers = new Map(), frames = [];
  let unexpectedLoss = null;
  device.addEventListener("uncapturederror", (event) => gpuErrors.push(event.error?.message || "GPU error"));
  device.lost.then((info) => { if (info.reason !== "destroyed") unexpectedLoss = info.message; });
  async function cleanGPU() {
    await device.queue.onSubmittedWorkDone();
    assert(!gpuErrors.length, `${label}: ${gpuErrors.join("; ")}`);
    assert(unexpectedLoss === null, `${label}: device lost: ${unexpectedLoss}`);
  }
  try {
    const L = cfg.num_hidden_layers, mid = Math.floor(L / 2);
    const assignments = split ? [["host", 0, mid], ["worker", mid, L]] : [["host", 0, L]];
    for (const [id, lo, hi] of assignments) {
      const host = id === "host";
      const weights = await ggufWeights(G, bytesOf, { lo, hi, hasEmbed: host, hasHead: host });
      const engine = await DenseEngine.create({ device, cfg, weights, layerRange: [lo, hi], hasEmbed: host, hasHead: host });
      const state = { engine, device, role: host ? "host" : "worker", chain: host && split ? ["worker"] : [],
        hostId: "host", next: host && split ? "worker" : "host", range: [lo, hi], pos: 0, fed: [],
        settings: { sampling: "exact" }, abort: false, degraded: false, waiters: new Map(), q: Promise.resolve(),
        pendingCtl: {}, pending: null, ckpt: null, ckptN: 0 };
      const pipeline = createPipeline({ state, transport: {
        sendHidden(to, message) {
          frames.push({ from: id, to, t: message.t });
          // Like an asynchronous network delivery, return before the peer performs its work.
          queueMicrotask(() => peers.get(to).pipeline.handleFrame(id, message));
        },
        sendTo(to, message) {
          assert(message.t === "ai-error", "unexpected runtime control message");
          const target = peers.get(to);
          target.state.abort = true; target.pipeline.failWaiters(new Error(message.message));
        },
      } });
      peers.set(id, { state, pipeline, generate: createGenerator({ state, pipeline }) });
    }
    const host = peers.get("host"), request = { stop: new Set(), sample: argmax };
    const emitted = [];
    const full = await host.generate(golden.ids, { ...request, maxNew: golden.generated.length,
      onToken: (id) => emitted.push(id) });
    equal(full.tokens, golden.generated, `${label}: CPU golden tokens`);
    equal(emitted, golden.generated, `${label}: streamed CPU golden tokens`);
    equal(full.reason, "max", `${label}: length cap`);
    await cleanGPU();

    // Continue receives the complete displayed answer, as the browser conversation does.
    // Plain solo decode has written it all; split dense decode leaves its last emitted
    // token unpiped, so that one token must be prefilled on Continue.
    const half = Math.floor(golden.generated.length / 2);
    const first = await host.generate(golden.ids, { ...request, maxNew: half });
    equal(first.tokens, golden.generated.slice(0, half), `${label}: first half`);
    const openAnswer = [...golden.ids, ...first.tokens];
    const cached = openAnswer.length - (split ? 1 : 0);
    equal(host.state.fed.length, cached, `${label}: cached answer prefix`);
    const continued = await host.generate(openAnswer, { ...request, maxNew: golden.generated.length - half });
    equal([...first.tokens, ...continued.tokens], golden.generated, `${label}: capped continuation golden`);
    equal(continued.prefilled, split ? 1 : 0, `${label}: Continue prefills only the unpiped token`);
    equal(continued.reused, cached, `${label}: Continue reuse length`);
    await cleanGPU();

    // A strict-prefix follow-up appends one known golden token to the complete answer.
    // Prefill only its uncached suffix; compare the response with the independent CPU golden.
    await host.generate(golden.ids, { ...request, maxNew: half });
    const followup = [...golden.ids, ...golden.generated.slice(0, half + 1)];
    const reused = await host.generate(followup, { ...request, maxNew: golden.generated.length - half - 1 });
    equal(reused.tokens, golden.generated.slice(half + 1), `${label}: prefix reuse golden`);
    equal(reused.reused, cached, `${label}: cached prefix length`);
    equal(reused.prefilled, split ? 2 : 1, `${label}: suffix-only prefill`);

    const stopped = await host.generate(golden.ids, {
      sample: argmax, stop: new Set([golden.generated[0]]), maxNew: golden.generated.length,
    });
    equal(stopped.tokens, [], `${label}: stop token must not be emitted`);
    equal(stopped.reason, "stop", `${label}: stop reason`);
    equal(host.state.fed, golden.ids, `${label}: stop token must not enter caches`);
    await cleanGPU();
    if (split) {
      assert(frames.some((f) => f.from === "host" && f.t === "ai-hidden"), "split sent no activation frames");
      assert(frames.some((f) => f.from === "worker" && f.t === "ai-hiddenret"), "split returned no activation frames");
    }
    console.log(`RUNTIME ${label.toUpperCase()} PASS`, JSON.stringify(tok.decode(full.tokens)), `(${frames.length} frames)`);
    return full.tokens;
  } finally {
    for (const p of peers.values()) p.pipeline.failWaiters(new Error("test finished"));
    await Promise.all([...peers.values()].map((p) => p.state.q));
    device.destroy();
    await device.lost;
  }
}

// Explicit large-model case: shares the real runtime but keeps 27B weights out of quick.
async function checkQwen35() {
  const { Qwen35Engine } = await import("../engine/qwen35.js");
  const { openGGUF, Q38_PATH, gpuDevice } = await import("./load_model.js");
  const { packWire, unpackWire } = await import("../room/wire.js");
  const model = openGGUF(Q38_PATH), { device } = await gpuDevice();
  const peers = new Map(), frames = [], restored = [], gpuErrors = [];
  device.addEventListener("uncapturederror", (e) => gpuErrors.push(e.error.message));
  let lost = null;
  device.lost.then((info) => { if (info.reason !== "destroyed") lost = info.message; });
  try {
    const L = model.trunkLayers, split = Math.floor(L / 2), tokenizer = model.tokenizer();
    console.log("runtime Qwen35 model:", model.meta["general.name"], `layers ${L}, split ${split}/${L - split}`);
    for (const [id, lo, hi] of [["host", 0, split], ["worker", split, L]]) {
      const host = id === "host";
      const engine = await Qwen35Engine.create({ device, meta: model.meta, maxSeq: 512,
        layerRange: [lo, hi], hasEmbed: host, hasHead: host,
        vocab: model.G.tensors["token_embd.weight"].shape[0],
        weights: await model.weights({ lo, hi, hasEmbed: host, hasHead: host, mtp: host }),
      });
      const state = { engine, device, role: host ? "host" : "worker", chain: host ? ["worker"] : [],
        hostId: "host", next: host ? "worker" : "host", range: [lo, hi], pos: 0, fed: [],
        settings: { sampling: "exact" }, abort: false, degraded: false,
        waiters: new Map(), q: Promise.resolve(), pendingCtl: {}, pending: null };
      const options = { checkpointMax: 0, lookup: false, denseSpec: false };
      const pipeline = createPipeline({ state, options, transport: {
        sendHidden(to, frame) {
          frames.push({ from: id, to, t: frame.t, spec: frame.spec, rb: frame.rb });
          queueMicrotask(() => peers.get(to).pipeline.handleFrame(id, frame));
        },
        sendTo(to, frame) { peers.get(to).pipeline.failWaiters(new Error(frame.message)); },
      } });
      peers.set(id, { state, pipeline, generate: createGenerator({ state, pipeline, options }) });
    }
    const host = peers.get("host"), worker = peers.get("worker");
    const E = host.state.engine, W = worker.state.engine, specStep = E.specStep;
    assert(E.mtp && specStep, "27B host has no MTP head");
    const restore = W.restoreDN.bind(W);
    W.restoreDN = (pos) => { restored.push(pos); return restore(pos); };
    const wire = (h) => unpackWire(packWire(h));

    // Independent engine-level reference: token-at-a-time through the two shards,
    // with the same f16 wire conversion, without the extracted pipeline or generator.
    async function reference(ids, count) {
      E.reset(); W.reset(); E.mtpFill = false;
      let pos = 0, logits;
      const step = async (id) => E.headFromHidden(wire(await W.runHidden(wire(await E.embedRun(id, pos)), pos++)));
      for (const id of ids) logits = await step(id);
      const out = [];
      for (let i = 0; i < count; i++) { const id = argmax(logits); out.push(id); logits = await step(id); }
      return out;
    }
    const capital = await reference(tokenizer.encode("The capital of France is"), 12);
    equal(tokenizer.decode(capital), " Paris.\nThe capital of Germany is Berlin.\nThe", "27B reference golden");
    const V = tokenizer.vocab;
    const ids = [V["<|im_start|>"], ...tokenizer.encode("user\nWrite the Python code for two sum. Code only."),
      V["<|im_end|>"], ...tokenizer.encode("\n"), V["<|im_start|>"], ...tokenizer.encode("assistant\n"),
      V["<think>"], ...tokenizer.encode("\n\n"), V["</think>"], ...tokenizer.encode("\n\n")];
    const expected = await reference(ids, 40);
    const followIds = [...ids, ...expected, ...tokenizer.encode("\nContinue:")];
    const followExpected = await reference(followIds, 8);
    function reset() {
      E.reset(); W.reset(); E.mtpFill = true;
      Object.assign(host.state, { pos: 0, fed: [], pending: null, pendingCtl: {}, xAt: null, lastHidden: null });
      E.mtp.stats = { drafts: 0, accepted: 0 };
      frames.length = 0; restored.length = 0;
    }
    async function generate(prompt, count, spec) {
      const emitted = [];
      // This test switch selects the existing plain branch without duplicating it.
      E.specStep = spec ? specStep : undefined;
      try {
        const result = await host.generate(prompt, { stop: new Set(), sample: argmax, maxNew: count,
          onToken: (id) => emitted.push(id) });
        equal(emitted, result.tokens, "27B ordered stream");
        equal(result.reason, "max", "27B length cap");
        equal(host.state.pos, host.state.fed.length, "27B cached position");
        return result.tokens;
      } finally { E.specStep = specStep; }
    }
    reset();
    equal(await generate(ids, 40, false), expected, "27B shared plain versus reference");
    reset();
    equal(await generate(ids, 40, true), expected, "27B shared MTP versus reference");
    assert(frames.some((f) => f.from === "host" && f.spec), "27B did not send a verify frame");
    assert(E.mtp.stats.drafts > E.mtp.stats.accepted, "27B prompt exercised no draft rejection");
    const stats = { ...E.mtp.stats };
    // The next request must apply any final pending rollback and reuse a correct prefix.
    equal(await generate(followIds, 8, false), followExpected, "27B request after rejection");
    const rollbacks = frames.filter((f) => f.from === "host" && f.rb != null).map((f) => f.rb);
    assert(rollbacks.length > 0, "27B did not transmit rollback control");
    equal(restored, rollbacks, "27B worker applied each rollback");
    equal(host.state.fed, [...followIds, ...followExpected], "27B follow-up cache contents");
    equal(W.pos, host.state.pos, "27B worker position after rollback and follow-up");
    await device.queue.onSubmittedWorkDone();
    assert(!gpuErrors.length && lost === null, `27B GPU failure: ${gpuErrors.join("; ")} ${lost || ""}`);
    console.log("RUNTIME QWEN35 SPLIT PASS", JSON.stringify({ tokens: expected.length, followup: followExpected.length,
      frames: frames.length, rollbacks: rollbacks.length, ...stats }));
  } finally {
    for (const p of peers.values()) p.pipeline.failWaiters(new Error("test finished"));
    await Promise.all([...peers.values()].map((p) => p.state.q));
    model.close(); device.destroy(); await device.lost;
  }
}

try {
  if (Deno.args.includes("--q38")) {
    await checkQwen35();
  } else {
    await loadDense();
    const solo = await checkMode(false);
    const split = await checkMode(true);
    equal(split, solo, "solo/split runtime parity");
    console.log("RUNTIME GPU PASS");
  }
} catch (error) {
  console.error("RUNTIME GPU FAIL:", error);
  Deno.exit(1);
}
