// DOM-free pipeline example and deterministic contract checks. No PeerJS, model or GPU is needed.
// Run: node tests/headless/runtime.mjs (or deno run tests/headless/runtime.mjs).
import { createPipeline } from "../../room/pipeline.js";
import { createGenerator } from "../../engine/generate.js";
import { attachWire, makeLink, sendFrame, PROTOCOL } from "../../room/transport.js";

// This harness is the caller: it owns serialization, failures and teardown.
function createRuntime({ state, transport, options }) {
  const pipeline = createPipeline({ state, transport, options });
  return {
    pipeline, generate: createGenerator({ state, pipeline, options }),
    handleFrame(from, frame) { pipeline.handleFrame(from, frame); },
    fail(err) { state.abort = true; pipeline.failWaiters(err); },
    async drain() { await state.q; },
    async dispose() { pipeline.failWaiters(new Error("test room closed")); await state.q; },
  };
}

export const assert = (condition, message) => { if (!condition) throw new Error(message); };
export const equal = (a, b, message) => assert(JSON.stringify(a) === JSON.stringify(b), message);
const sample = (logits) => logits.indexOf(Math.max(...logits));

// Integer additions are exactly representable on the f16 wire. This models stateful shard
// ordering, not neural inference: generation and framing use the production modules.
class CountingEngine {
  constructor(layers) {
    this.layers = layers; this.dims = { dim: 1 }; this.maxSeq = 256; this.NC = 4;
    this.pos = 0; this.slots = new Map(); this.calls = []; this.failNext = false;
  }
  reset() { this.pos = 0; }
  saveSlot(key) { this.slots.set(key, this.pos); }
  loadSlot(key) {
    if (!this.slots.has(key)) throw new Error("missing test checkpoint");
    this.pos = this.slots.get(key);
  }
  dropSlot(key) { this.slots.delete(key); }
  dropAllSlots() { this.slots.clear(); }
  async run(values, pos) {
    if (this.failNext) { this.failNext = false; throw new Error("injected worker failure"); }
    assert(pos === this.pos, `shard out of order: got ${pos}, expected ${this.pos}`);
    this.calls.push({ pos, n: values.length });
    // Yield so an unqueued worker implementation would overlap and fail the position check.
    await Promise.resolve();
    this.pos += values.length;
    return Float32Array.from(values, (x) => x + this.layers);
  }
  embedRun(token, pos) { return this.run([token], pos); }
  embedRunBatch(tokens, pos) { return this.run(tokens, pos); }
  runHidden(hidden, pos) { return this.run(hidden, pos); }
  runHiddenBatch(hidden, pos) { return this.run(hidden, pos); }
  async prefillToken(token) { await this.run([token], this.pos); }
  async headFromHidden(hidden) {
    const logits = new Float32Array(16).fill(-10);
    logits[hidden[0] % logits.length] = 10;
    return logits;
  }
}

/** One or three peers with an asynchronous binary link and injectable engine shards. */
export function makeRoom(devices = 3, { engineFactory = (layers) => new CountingEngine(layers) } = {}) {
  assert(devices === 1 || devices === 3, "example supports solo or three peers");
  const names = Array.from({ length: devices }, (_, i) => i ? `worker${i}` : "host");
  const peers = new Map(), links = new Map(), frames = [], errors = [];
  let stopped = false, reordered = 0;
  function link(from, to) {
    const key = `${from}:${to}`;
    if (links.has(key)) return links.get(key);
    const tx = makeLink(), rx = makeLink();
    let receive;
    attachWire(rx, { peerConnection: { createDataChannel: () => ({
      readyState: "open", set onmessage(fn) { receive = fn; }, set onclose(_) {},
    }) } }, (frame) => peers.get(to).runtime.handleFrame(from, frame), { ordered: false });
    const pending = []; let deliveryTimer = null;
    tx.chans.push({ readyState: "open", send(buf) {
      pending.push(buf);
      if (deliveryTimer !== null) return;
      deliveryTimer = setTimeout(() => {
        deliveryTimer = null;
        // Model an unordered channel: the transport must reassemble frames in send order.
        if (pending.length > 1) reordered++;
        for (const data of pending.splice(0).reverse()) if (!stopped) receive({ data });
      }, 0);
    } });
    const pair = { tx, rx, close() { clearTimeout(deliveryTimer); pending.length = 0; } };
    links.set(key, pair); return pair;
  }
  for (const [i, id] of names.entries()) {
    const engine = engineFactory(devices === 1 ? 3 : 1, i);
    const state = {
      engine, device: { queue: { async onSubmittedWorkDone() {} } },
      role: i ? "worker" : "host", chain: i ? [] : names.slice(1), hostId: "host",
      next: names[i + 1] || "host", range: [i, i + 1], pos: 0, fed: [],
      abort: false, degraded: false, settings: { sampling: "exact" }, q: Promise.resolve(),
      waiters: new Map(), pendingCtl: {}, pending: null, ckpt: null, ckptN: 0,
    };
    const runtime = createRuntime({ state, options: { maxSeq: 256, maxNew: 6, lookup: false }, transport: {
      sendHidden(to, message) {
        if (stopped) throw new Error("test transport closed");
        frames.push({ from: id, to, t: message.t, pos: message.pos ?? message.basePos });
        assert(sendFrame(link(id, to).tx, message), "transport refused frame");
      },
      sendTo(to, message) {
        if (message.t === "ai-error") {
          const error = new Error(message.error || message.msg || message.message || "worker error");
          errors.push(error.message); peers.get(to).runtime.fail(error);
        }
      },
    } });
    peers.set(id, { state, runtime, engine });
  }
  return {
    peers, frames, errors, host: peers.get("host"), get reordered() { return reordered; },
    async drain() { for (const peer of peers.values()) await peer.runtime.drain(); },
    async dispose() {
      stopped = true;
      await Promise.all([...peers.values()].map((p) => p.runtime.dispose()));
      for (const { tx, rx, close } of links.values()) {
        close();
        clearTimeout(tx.gapTimer); clearTimeout(rx.gapTimer);
        tx.chans.length = 0; rx.chans.length = 0;
      }
    },
  };
}

export async function proveRuntime() {
  const solo = makeRoom(1), split = makeRoom(3);
  const prompt = Array.from({ length: 35 }, (_, i) => i % 16);
  const request = { stop: new Set(), maxNew: 6, sample };
  try {
    const streamed = [];
    const a = await solo.host.runtime.generate(prompt, request);
    const b = await split.host.runtime.generate(prompt, { ...request, onToken: (id) => streamed.push(id) });
    equal(a.tokens, b.tokens, "solo/split token mismatch");
    equal(streamed, b.tokens, "stream differs from returned tokens");
    assert(split.frames.some((f) => f.t === "ai-hidden-b"), "batched prefill was not exercised");
    assert(split.reordered > 0, "test did not deliver any frames out of order");
    assert(split.frames.some((f) => f.from === "worker2" && f.to === "host"), "last worker did not return to host");
    equal(split.host.state.fed, solo.host.state.fed, "solo/split prefix mismatch");

    // Continue from caches, then branch away and return to the original checkpoint.
    const followup = [...prompt, ...a.tokens, 7];
    const c = await solo.host.runtime.generate(followup, request);
    const d = await split.host.runtime.generate(followup, request);
    equal(c.tokens, d.tokens, "follow-up mismatch");
    assert(d.reused > 0, "follow-up did not reuse a prefix");
    await split.host.runtime.generate([12, 13], request);
    const branch = await split.host.runtime.generate(followup, request);
    equal(branch.tokens, d.tokens, "checkpoint branch mismatch");

    const controller = new AbortController();
    const aborted = await split.host.runtime.generate([2, 4, 6], {
      ...request, signal: controller.signal, onToken: () => controller.abort(),
    });
    assert(aborted.reason === "abort" && aborted.tokens.length === 1, "abort contract mismatch");
    const recovered = await split.host.runtime.generate([3, 5], request);
    const fresh = makeRoom(1);
    try { equal(recovered.tokens, (await fresh.host.runtime.generate([3, 5], request)).tokens, "abort recovery mismatch"); }
    finally { await fresh.dispose(); }

    split.peers.get("worker1").engine.failNext = true;
    let failed = false;
    try { await split.host.runtime.generate([9, 8], request); } catch { failed = true; }
    assert(failed && split.errors.length === 1, "worker failure did not reject host");
    assert(split.host.state.fed === null && split.host.state.waiters.size === 0, "failed request left live state");
    split.host.state.abort = false; // explicit caller decision after handling the failure
    const afterError = await split.host.runtime.generate([1, 2], request);
    const reference = makeRoom(1);
    try { equal(afterError.tokens, (await reference.host.runtime.generate([1, 2], request)).tokens, "error recovery mismatch"); }
    finally { await reference.dispose(); }
    await split.drain();
    return { protocol: PROTOCOL, devices: 3, tokens: b.tokens, frames: split.frames.length,
      checks: ["solo/split parity", "ordered binary frames", "prefix/checkpoints", "abort/recovery", "worker error/recovery", "lifecycle"] };
  } finally { await solo.dispose(); await split.dispose(); }
}

if (import.meta.main || (typeof process !== "undefined" && process.argv[1] &&
    new URL(import.meta.url).pathname === process.argv[1])) {
  console.log(JSON.stringify(await proveRuntime(), null, 2));
}
