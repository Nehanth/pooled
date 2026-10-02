// Real Qwen3-0.6B Q8 golden through the extracted runtime, solo and across two shards.
// No DOM or PeerJS: only activation delivery is in-process. The production pipeline still
// packs/unpacks its f16 frames, queues worker compute, carries resets and resolves token laps.
// Run: deno run --unstable-webgpu --allow-read tests/test_runtime.js
import { DenseEngine, argmax, makeTokenizer } from "../engine/engine.js";
import { parseGGUFHeader, ggufWeights } from "../engine/gguf.js";
import { createPipeline } from "../room/pipeline.js";
import { createGenerator } from "../engine/generate.js";

const assert = (ok, message) => { if (!ok) throw new Error(message); };
const equal = (a, b, message) => assert(JSON.stringify(a) === JSON.stringify(b),
  `${message}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`);
const dir = new URL(".", import.meta.url).pathname;
const cfg = JSON.parse(await Deno.readTextFile(dir + "../models/qwen/config.json"));
const golden = JSON.parse(await Deno.readTextFile(dir + "golden/golden_qwen.json"));
const tok = makeTokenizer(JSON.parse(await Deno.readTextFile(dir + "../models/qwen/tokenizer.json")));
const raw = await Deno.readFile(dir + "../models/qwen/model.gguf");
const G = parseGGUFHeader(raw.buffer);
const bytesOf = (info) => new Uint8Array(raw.buffer, info.byteOffset, info.byteLength);
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

try {
  const solo = await checkMode(false);
  const split = await checkMode(true);
  equal(split, solo, "solo/split runtime parity");
  console.log("RUNTIME GPU PASS");
} catch (error) {
  console.error("RUNTIME GPU FAIL:", error);
  Deno.exit(1);
}
