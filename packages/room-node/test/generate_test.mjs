// The node's single-attempt generator: scripted engine output, no GPU or network.
import { test } from "node:test";
import assert from "node:assert/strict";
import { RoomNode } from "../roomnode.js";
import { unpackWire } from "../../../room/wire.js";

class Engine {
  constructor({ mtp = false, dense = false, reject = false, answer = [7, 8, 9, 0], maxSeq = 128 } = {}) {
    this.answer = answer; this.at = 0; this.pos = 0; this.maxSeq = maxSeq;
    this.dims = { dim: 1 }; this.calls = [];
    if (dense) {
      this.specStep = undefined;
      this.specStats = { drafts: 0, accepted: 0 };
      this.embedRunBatch = async (ids, pos) => {
        this.calls.push(["batch", ids, pos]);
        this.pos = pos + ids.length;
        return new Float32Array(ids);
      };
      this.specStepDrafts = async (next, sample, drafts, spec) => {
        this.calls.push(["dense", next, drafts]);
        const pos = this.pos, accepted = reject ? drafts.slice(0, 1) : drafts;
        this.lastHidden = await spec.runTrunk([next, ...drafts], pos);
        this.pos = pos + accepted.length + 1;
        if (reject) await spec.onReject(this.pos);
        this.specStats.drafts += drafts.length;
        this.specStats.accepted += accepted.length;
        return [...accepted, (accepted.at(-1) || next) + 1];
      };
    }
    if (mtp) this.mtp = { stats: { drafts: 0, accepted: 0 } };
  }
  reset() { this.pos = 0; this.calls.push(["reset"]); }
  async prefillTokens(ids) { this.calls.push(["prefill", ...ids]); this.pos += ids.length; }
  async embedRun(id, pos) { this.calls.push(["token", id, pos]); this.pos = pos + 1; return new Float32Array([id]); }
  async headFromHidden() { this.calls.push(["head"]); return new Float32Array([this.answer[this.at++] ?? 0]); }
  async specStep(next, sample, k) {
    this.calls.push(["spec", next, k]);
    const out = this.answer.slice(this.at, this.at + k + 1);
    if (!out.length) out.push(0);
    this.at += out.length; this.pos += out.length;
    this.mtp.stats.drafts += out.length - 1; this.mtp.stats.accepted += out.length - 1;
    return out;
  }
}

function host(options = {}) {
  const n = new RoomNode({ name: "test", ckpt: false, log() {} });
  n.isHost = true;
  Object.assign(n.ai, { role: "host", online: true, engine: new Engine(options),
    device: { queue: { onSubmittedWorkDone: async () => {} } } });
  if (options.dense) {
    n.ai.chain = ["worker"];
    n.conns.set("worker", { name: "worker", meta: { dspec: options.dspec ?? 1 } });
    n.wakeChain = () => {};
    n.sendHidden = (id, frame) => {
      n.ai.engine.calls.push(["frame", frame.t, frame.spec || 0, frame.rb ?? null]);
      n.lapDone(frame.t === "ai-hidden-b" ? "b" + frame.basePos : frame.pos, unpackWire(frame));
    };
  }
  return n;
}
const opts = { stop: new Set([0]), sample: (logits) => logits[0], maxNew: 20 };

test("node generation emits plain tokens in order with numeric draft flags", async () => {
  const n = host(), shown = [], events = [];
  n.on("prefill", (event) => events.push(event));
  const r = await n.generateOnce([1, 2, 3], { ...opts, onToken: (...args) => shown.push(args) });
  assert.deepEqual(shown, [[7, 0], [8, 0], [9, 0]]);
  assert.deepEqual(r.tokens, [7, 8, 9]);
  assert.equal(r.reason, "stop");
  assert.deepEqual(n.ai.fed, [1, 2, 3, 7, 8, 9]);
  assert.equal(events.length, 1);
  assert.equal(events[0].count, 3);
  assert.equal(events[0].prefilled, 3);
  assert.equal(r.from, null);
  assert.equal(r.pinned, 0);
});

test("node speculation reports accepted drafts before the next sampled token", async () => {
  const n = host({ mtp: true, answer: [7, 8, 9, 10, 11, 0] }), shown = [];
  const r = await n.generateOnce([1, 2, 3], { ...opts, onToken: (...args) => shown.push(args) });
  assert.deepEqual(shown, [[7, 0], [8, 1], [9, 1], [10, 1], [11, 0]]);
  assert.deepEqual(r.tokens, [7, 8, 9, 10, 11]);
  assert.equal(r.reason, "stop");
  assert.equal(r.acc, 1);
  assert.equal(n.ai.pending, undefined);
});

test("node spec:false bypasses the draft head without changing the answer", async () => {
  const n = host({ mtp: true }), shown = [];
  const r = await n.generateOnce([1, 2, 3], { ...opts, spec: false, onToken: (...args) => shown.push(args) });
  assert.deepEqual(shown, [[7, 0], [8, 0], [9, 0]]);
  assert.deepEqual(r.tokens, [7, 8, 9]);
  assert.equal(n.ai.engine.calls.some(([op]) => op === "spec"), false);
  assert.equal(r.acc, null);
});

test("node cap does not sample or save a browser Continue token", async () => {
  const n = host(); let sampled = 0;
  const r = await n.generateOnce([1, 2, 3], { ...opts, maxNew: 2, sample: (lg) => { sampled++; return lg[0]; } });
  assert.deepEqual(r.tokens, [7, 8]);
  assert.equal(r.reason, "max");
  assert.equal(sampled, 2);
  assert.equal(n.ai.pending, undefined);
  assert.deepEqual(n.ai.fed, [1, 2, 3, 7, 8]);
});

test("node speculative cap keeps only emitted cache tokens and no Continue token", async () => {
  const n = host({ mtp: true, answer: [7, 8, 9, 10, 11, 12] });
  const r = await n.generateOnce([1, 2, 3], { ...opts, maxNew: 3 });
  assert.deepEqual(r.tokens, [7, 8, 9]);
  assert.equal(r.reason, "max");
  assert.deepEqual(n.ai.fed, [1, 2, 3, 7, 8, 9]);
  assert.equal(n.ai.pending, undefined);
});

test("node cancellation uses the request signal, not browser ai.abort", async () => {
  const n = host(), ac = new AbortController();
  n.ai.abort = true;
  const r = await n.generateOnce([1, 2, 3], { ...opts, signal: ac.signal, onToken: () => ac.abort() });
  assert.deepEqual(r.tokens, [7]);
  assert.equal(r.reason, "abort");
  assert.deepEqual(n.ai.fed, [1, 2, 3, 7]);
  assert.equal(n.ai.pending, undefined);
});

test("an already aborted node request does not run the engine or emit tokens", async () => {
  const n = host(), ac = new AbortController(); ac.abort();
  const r = await n.generateOnce([1, 2, 3], { ...opts, signal: ac.signal });
  assert.deepEqual(r.tokens, []);
  assert.equal(r.reason, "abort");
  assert.deepEqual(n.ai.engine.calls, [["reset"]]);
});

test("node context cap emits the last token without writing beyond the cache", async () => {
  const n = host({ maxSeq: 5 });
  const r = await n.generateOnce([1, 2, 3], opts);
  assert.deepEqual(r.tokens, [7, 8]);
  assert.equal(r.reason, "ctx");
  assert.deepEqual(n.ai.fed, [1, 2, 3, 7]);
  assert.equal(n.ai.pos, 4);
});

test("node callback failure invalidates caches for the next request", async () => {
  const n = host();
  await assert.rejects(n.generateOnce([1, 2, 3], { ...opts, onToken() { throw new Error("consumer failed"); } }), /consumer failed/);
  assert.equal(n.ai.fed, null);
  assert.deepEqual(n.ai.pendingCtl, {});
});

test("node completion retains the attempt's context limit if its engine is replaced", async () => {
  const n = host(), original = n.ai.engine;
  const head = original.headFromHidden.bind(original);
  original.headFromHidden = async () => {
    const logits = await head();
    n.ai.engine = new Engine({ maxSeq: 5, answer: [21, 22, 0] });
    return logits;
  };
  const r = await n.generateOnce([1, 2, 3], { ...opts, maxNew: 2 });
  assert.deepEqual(r.tokens, [7, 21]);
  assert.equal(r.reason, "max");
});

test("node lookup depth starts afresh on each request", async () => {
  const n = host({ mtp: true, answer: [3] });
  const E = n.ai.engine, lengths = [];
  E.maxDrafts = 12;
  E.specStepDrafts = async (next, sample, drafts) => {
    lengths.push(drafts.length);
    E.pos += drafts.length + 1;
    E.mtp.stats.drafts += drafts.length; E.mtp.stats.accepted += drafts.length;
    return [...drafts, drafts.at(-1) % 20 + 1];
  };
  const prompt = [...Array.from({ length: 20 }, (_, i) => i + 1), 1, 2];
  await n.generateOnce(prompt, { ...opts, maxNew: 24 });
  assert.equal(lengths[0], 7);
  assert.ok(lengths.slice(1).some((n) => n > 7), "accepted lookup can grow within one request");
  lengths.length = 0; E.at = 0;
  await n.generateOnce(prompt, { ...opts, maxNew: 24 });
  assert.equal(lengths[0], 7, "a later request starts with the original lookup depth");
  assert.equal(n.ai.lkFull, undefined);
});

const densePrompt = [...Array.from({ length: 20 }, (_, i) => i + 1), 1, 2];

test("node dense lookup uses verify laps and resets lookup depth each request", async () => {
  const n = host({ dense: true, answer: [3] }), E = n.ai.engine, shown = [];
  for (let i = 0; i < 2; i++) {
    E.at = 0; E.calls.length = 0;
    const r = await n.generateOnce(densePrompt, { ...opts, maxNew: 12, onToken: (...args) => shown.push(args) });
    assert.deepEqual(r.tokens, Array.from({ length: 12 }, (_, j) => j + 3));
    assert.deepEqual(E.calls.filter(([op]) => op === "dense").map((c) => c[2].length), [3, 7]);
    assert.equal(r.copied, 10);
    assert.equal(r.acc, 1);
    assert.equal(n.ai.xAt, n.ai.pos);
    assert.equal(n.ai.pending, undefined);
    assert.equal(n.ai.lkFullD, undefined);
    assert.ok(E.calls.some(([op, , spec]) => op === "frame" && spec === 1));
  }
  assert.deepEqual(shown.slice(0, 5), [[3, 0], [4, 2], [5, 2], [6, 2], [7, 0]]);
});

test("node dense lookup falls back for old peers and spec:false", async () => {
  for (const [dspec, spec] of [[0, true], [1, false]]) {
    const n = host({ dense: true, dspec, answer: [3, 4, 0] });
    const r = await n.generateOnce(densePrompt, { ...opts, spec });
    assert.deepEqual(r.tokens, [3, 4]);
    assert.equal(n.ai.engine.calls.some(([op]) => op === "dense"), false);
  }
});

test("node dense lookup rechecks a peer's capability between verify laps", async () => {
  const n = host({ dense: true, answer: [3, 0] });
  const r = await n.generateOnce(densePrompt, { ...opts, onToken: (id) => {
    if (id === 4) n.conns.get("worker").meta.dspec = 0;
  } });
  assert.deepEqual(r.tokens, [3, 4, 5, 6, 7]);
  assert.equal(n.ai.engine.calls.filter(([op]) => op === "dense").length, 1);
});

test("node dense reject carries rollback on the next frame and respects abort", async () => {
  const n = host({ dense: true, reject: true, answer: [3] }), ac = new AbortController();
  const r = await n.generateOnce(densePrompt, { ...opts, signal: ac.signal,
    onToken: (id) => { if (id === 7) ac.abort(); } });
  assert.equal(r.reason, "abort");
  assert.ok(n.ai.engine.calls.some(([op, , , rb]) => op === "frame" && rb != null));
  assert.equal(n.ai.pending, undefined);
});
