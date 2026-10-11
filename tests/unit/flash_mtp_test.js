// engine/qwen4exp_mtp.js without a GPU: specStep's accept rule, the draft-row refill and the hidden it hands the next
// step, and DecodeVerifier's two modes (stop at the first rejection; a full pass rolled back), on a fake engine whose
// "model" is a fixed next-token function with a recurrent state and an n-gram history that a missed rollback would
// corrupt. The buffers and copies are the recording mock device's (tests/unit/mock_gpu.js), so the GPU copies move
// real bytes. The model test is tests/test_q38flash_mtp.js.
import { assertEquals, assert, assertRejects } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { mockDevice } from "./mock_gpu.js";
import { NgramState } from "../../engine/ngram.js";

const device = mockDevice();
const { Qwen4ExpMtp, DecodeVerifier } = await import("../../engine/qwen4exp_mtp.js");

// A fake trunk: next = (31 * tok + 7 * conv + p1 + 1) % V, where conv is a running sum kept in a "GPU" state buffer
// (l.st.conv) and p1 is the n-gram predecessor (engine.ngs). B.R gets [pos, tok] as its hidden.
const V = 97, HD = 8;
function fakeEngine() {
  const buf = (n) => device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const conv = buf(4);
  const eng = {
    device, cfg: { hDim: HD }, pos: 0, maxSeq: 512, layers: [{ st: { conv } }], B: { R: buf(HD) },
    ngs: new NgramState({ eos: 0 }),
    calls: 0,
    async forwardToken(t) {
      this.calls++;
      const c = new Int32Array(conv.data.buffer)[0], p1 = this.ngs.p1;
      // the n-gram history as NgramState.push moves it (without hashing rows: the predecessor is all this model reads)
      const g = this.ngs, s = (g.pos % g.H) * 2;
      g.hist[s] = g.p1; g.hist[s + 1] = g.p2; g.pos++; g.nHist = Math.min(g.nHist + 1, g.H);
      g.p2 = t === 0 ? 0 : g.p1; g.p1 = t;
      new Int32Array(conv.data.buffer)[0] = c + t;
      const R = new Float32Array(this.B.R.data.buffer);
      R[0] = this.pos; R[1] = t;
      this.pos++;
      return { id: (31 * t + 7 * c + p1 + 1) % V };
    },
    reset() { this.pos = 0; this.ngs.reset(); conv.data.fill(0); },
  };
  return eng;
}
// plain greedy on the fake trunk
async function plain(eng, prompt, n) {
  eng.reset();
  let r;
  for (const t of prompt) r = await eng.forwardToken(t);
  const out = [r.id];
  while (out.length < n) out.push((await eng.forwardToken(out.at(-1))).id);
  return out;
}
// a fake draft block: right with probability ~ `good`, by peeking at a copy of the trunk; records its runs
function fakeMtp(eng, good) {
  const m = Object.create(Qwen4ExpMtp.prototype);
  const hT = device.createBuffer({ size: HD * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  Object.assign(m, { eng, device, cfg: eng.cfg, maxDrafts: 7, hT, runs: [], stats: { steps: 0, drafts: 0, accepted: 0 } });
  let seed = 1;
  m.run = async function (tok, pos, { from = "trunk", hFrom = null, head = false } = {}) {
    let h = null;
    if (hFrom) h = Array.from(new Float32Array(hFrom.buffer.data.buffer, hFrom.offset, 2));
    this.runs.push({ tok, pos, from, h, head });
    if (!head) return null;
    seed = (seed * 1103515245 + 12345) % 2 ** 31;
    return seed / 2 ** 31 < good ? -1 : (tok * 13 + 5) % V;   // -1: filled in by the oracle below
  };
  return m;
}

Deno.test("flash mtp: spec ids == plain ids, both verifier modes, any acceptance", async () => {
  const prompt = [3, 14, 15, 92, 65], N = 60;
  for (const good of [0, 0.5, 0.9, 1]) for (const K of [1, 3, 5]) for (const earlyStop of [true, false]) {
    const eng = fakeEngine(), ref = await plain(eng, prompt, N);
    eng.reset();
    let r;
    for (const t of prompt) r = await eng.forwardToken(t);
    const m = fakeMtp(eng, good), ver = new DecodeVerifier(eng, { maxCols: K + 1, earlyStop });
    // the oracle: a "good" draft is the plain sequence's next token
    const run0 = m.run;
    let at = 0;
    m.run = async function (tok, pos, o) {
      const x = await run0.call(this, tok, pos, o);
      return x === -1 ? (ref[pos - prompt.length + 1] ?? 0) : x;
    };
    const out = [r.id];
    while (out.length < N) { at = out.length; out.push(...await m.specStep(out.at(-1), ver, K)); }
    assertEquals(out.slice(0, N), ref, `good ${good} K ${K} earlyStop ${earlyStop} (from ${at})`);
    assertEquals(eng.pos, prompt.length + out.length - 1);
    assertEquals(m.stats.drafts, m.stats.steps * K);
    if (good === 1) assertEquals(m.stats.accepted, m.stats.drafts);
    if (good === 0) assert(m.stats.accepted < m.stats.drafts / 4);
  }
});

Deno.test("flash mtp: draft rows, refill rows and the next step's hidden", async () => {
  const eng = fakeEngine(), prompt = [5, 6, 7];
  const ref = await plain(eng, prompt, 12);
  eng.reset();
  let r;
  for (const t of prompt) r = await eng.forwardToken(t);
  const m = fakeMtp(eng, 0), ver = new DecodeVerifier(eng, { maxCols: 4 });
  // drafts: right, right, wrong
  const pos = eng.pos;
  m.run = async function (tok, p, { from = "trunk", hFrom = null, head = false } = {}) {
    this.runs.push({ tok, p, from, h: hFrom ? Array.from(new Float32Array(hFrom.buffer.data.buffer, hFrom.offset, 2)) : null, head });
    return head ? (p - pos < 2 ? ref[p - pos + 1] : 0) : null;
  };
  const out = await m.specStep(r.id, ver, 3);
  assertEquals(out, ref.slice(1, 4));   // two accepted drafts and the trunk's correction
  assertEquals(m.stats.accepted, 2);
  // drafts: rows pos .. pos + 2, the first from the trunk hidden, the rest chained
  assertEquals(m.runs.filter((x) => x.head).map((x) => [x.tok, x.p, x.from]), [[r.id, pos, "trunk"], [ref[1], pos + 1, "chain"], [ref[2], pos + 2, "chain"]]);
  // refill: rows pos + 1, pos + 2 from verify columns 0, 1 (their hidden: [position, token])
  assertEquals(m.runs.filter((x) => !x.head).map((x) => [x.tok, x.p, x.h]), [[ref[1], pos + 1, [pos, r.id]], [ref[2], pos + 2, [pos + 1, ref[1]]]]);
  // the next step's first draft reads verify column 2's hidden (the last kept column)
  assertEquals(Array.from(new Float32Array(m.hT.data.buffer, 0, 2)), [pos + 2, ref[2]]);
  // no refill: no extra runs
  m.runs = [];
  await m.specStep(out.at(-1), ver, 2, { refill: false });
  assertEquals(m.runs.filter((x) => !x.head).length, 0);
});

Deno.test("flash mtp: the full-pass verifier rolls the state and the n-gram history back", async () => {
  const eng = fakeEngine();
  for (const t of [1, 2, 3]) await eng.forwardToken(t);
  const conv0 = new Int32Array(eng.layers[0].st.conv.data.buffer)[0], p10 = eng.ngs.p1, pos0 = eng.pos;
  const ver = new DecodeVerifier(eng, { maxCols: 4, earlyStop: false });
  const v = await ver.verify([10, 20, 30, 40]);
  assertEquals(v.ids.length, 4);   // every column ran
  await ver.accept(1);
  assertEquals(eng.pos, pos0 + 1);
  assertEquals(new Int32Array(eng.layers[0].st.conv.data.buffer)[0], conv0 + 10);
  assertEquals(eng.ngs.p1, 10);
  assertEquals(eng.ngs.pos, pos0 + 1);
  assert(p10 === 3);
  await assertRejects(() => ver.accept(1), Error, "without a verify");
  // an early-stopped pass keeps every column it ran
  const es = new DecodeVerifier(eng, { maxCols: 4 });
  const w = await es.verify([1, 2, 3, 4]);
  await assertRejects(() => es.accept(w.ids.length - 1), Error, "early-stopped");
});
