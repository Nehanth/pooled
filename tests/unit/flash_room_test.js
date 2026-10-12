// Qwen3.8-Flash-Next in rooms (roadmap/33 PR plan item 8): the catalogue entries, the 4-stream residual on the wire
// (hDim = 10240, 20 KB per token in f16) with the token ids the n-gram layer needs, the n-gram table's placement in
// the deal, and a synthetic two-device room (room/pipeline.js on both ends, room/flash.js over a fake engine core on
// a fake GPU) checked against the same model on one device, through checkpoints and their copies on disk.
import { setImmediate } from "node:timers";
import { MODELS, SHAPE, FILES, NEED_GB, CTX, FILE_GB, PICKER, maxSeqFor, kvModeFor, roomBytes, wireWidth, ngramOf, hostHeldBytes, expertSlices } from "../../room/models.js";
import { packWire, unpackWire, splitWire, wireDim } from "../../room/wire.js";
import { SLICE_BYTES } from "../../room/transport.js";
import { dealRoom, placeNgram, ngramHolder, shortNote } from "../../room/plan.js";
import { offloadFor, ngramFor, ngramDisk, pledgeGB } from "../../room/pledge.js";
import { createPipeline } from "../../room/pipeline.js";
import { FlashRoomEngine, qwen4expCore } from "../../room/flash.js";
import { NgramState, makeConfig } from "../../engine/ngram.js";
import { f16ToF32, f32ToF16 } from "../../engine/gguf.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const GB = 2 ** 30, KEY = "qwen3.8-flash";

// ---------------------------------------------------------------- catalogue

Deno.test("flash catalogue: dev-gated entry, pinned files, shape from the GGUF headers", () => {
  const M = MODELS[KEY], F = FILES[KEY], S = SHAPE[KEY];
  eq([M.kind, PICKER.includes(KEY)], ["qwen4exp", false], "its own engine kind, ?dev=1 only (like the 122B)");
  eq(M.shards.length, 3);
  eq(M.gguf, M.shards[0]);
  for (const u of [...M.shards, M.mtp]) ok(/\/resolve\/[0-9a-f]{40}\//.test(u), "pinned revision: " + u);
  ok(M.shards.every((u, i) => u.endsWith(`-0000${i + 1}-of-00003.gguf`)));
  // FILES: the HF API's sizes and SHA-256 at the pinned revisions; bytes is the whole model
  eq(F.shards.map((s) => s.bytes), [39853467552, 39647284192, 21098500928]);
  eq(F.bytes, F.shards.reduce((a, s) => a + s.bytes, 0));
  ok(F.shards.every((s) => /^[0-9a-f]{64}$/.test(s.sha256)) && /^[0-9a-f]{64}$/.test(F.mtp.sha256));
  eq(F.mtp.bytes, 2199652640);
  // FILE_GB: what a device downloads for its share, so the files without the n-gram table (only its holder fetches it)
  eq(Math.round((F.bytes - S.ngram.bytes) / 1e8) / 10, FILE_GB[KEY], "FILE_GB is the files' size in GB without the table");
  // context: dense attention is exact to 2051 cells and the phase-1 engine refuses more
  eq([CTX[KEY], maxSeqFor(KEY), maxSeqFor(KEY, 32768), kvModeFor(KEY, "q8")], [{ def: 2048, max: 2048 }, 2048, 2048, "f16"]);
  // shape: 48 layers, 512 experts with layers 0-5's Q4_1 down experts parked as Q8, the 4-stream residual, the table
  eq([S.L, S.park.nExp, S.park.runs.map(([n]) => n), S.hDim, wireWidth(KEY), S.ngram, ngramOf(KEY)],
    [48, 512, [6, 42], 10240, 10240, { layer: 1, bytes: 28800138240 }, { layer: 1, bytes: 28800138240 }]);
  eq(S.park.runs[0][1], [819200, 102400, 819200, 102400, 1638400, 102400], "layer 0's parked slices (expertSlicesOf on the file)");
  eq(S.park.runs[1][1], expertSlices(640, 2560, [2, 2, 2]));
  eq(S.pleState, 10 * 10240 * 4, "the PLE conv ring: 10 positions x hDim f32");
  // the other models keep their width; a model without a SHAPE says 0 (the room reads the engine then)
  eq([wireWidth("qwen3.6-35b-moe"), wireWidth("qwen3.5-122b-moe"), wireWidth("qwen3-1.7b"), ngramOf("qwen3.6-35b-moe")], [2048, 3072, 0, null]);
  // what the host holds past its layers: the embedding (JS memory), output.weight and the head's HC mixer
  eq(hostHeldBytes("qwen4exp", S), 357580800 + 534620160);
  const rb = roomBytes(KEY, 2048);
  eq([rb.L, rb.layerBytes, rb.expertBytes, rb.experts.nExp, rb.experts.slices.length], [48, 1478069803 + 2048 * 1024, 1468006400, 512, 48]);
  // NEED_GB: the layers, embedding and head at 2048 (the table is disk, not GPU memory)
  const need = (48 * rb.layerBytes + rb.hostBytes) / 1e9;
  ok(Math.abs(need - NEED_GB[KEY]) < 0.5, `NEED_GB ${NEED_GB[KEY]} vs ${need.toFixed(2)} GB`);
});

// ---------------------------------------------------------------- wire

Deno.test("flash wire: a token's frame is hDim f16 values (20 KB) plus its id; splitWire reads both back", () => {
  const W = 10240, h = new Float32Array(W).map((_, i) => Math.sin(i) * 3);
  const plain = packWire(h), withId = packWire(h, [248044]);
  eq(plain.data.byteLength, 20480, "10240 x f16 = 20 KB per token");
  eq(withId.data.byteLength, 20480 + 4, "the id adds 4 bytes");
  eq(Math.ceil(withId.data.byteLength / (SLICE_BYTES - 32)), 5, "5 transport slices per decode token (the 35B's 2048-wide token is 1)");
  const a = splitWire(withId, 1, W), b = splitWire(plain, 1, W);
  eq([...a.ids], [248044]);
  eq(b.ids, null);
  for (let i = 0; i < W; i++) if (a.h[i] !== f16ToF32(f32ToF16(h[i])) || b.h[i] !== a.h[i]) throw new Error("value " + i);
  // a batch: ids above 2^16 and 2^31 survive the two u16 words; the width tells values from ids
  const n = 6, hb = new Float32Array(n * W).fill(0.5), ids = [0, 1, 65535, 65536, 248320, 0xfffffffe];
  const d = splitWire(packWire(hb, ids), n, W);
  eq([d.h.length, [...d.ids]], [n * W, ids]);
  // the f32 wire (WIRE_F16 off) carries them the same way
  const f = splitWire({ data: (() => { const o = new Float32Array(n * W + n); o.set(hb); new Uint32Array(o.buffer, n * W * 4).set(ids); return o; })() }, n, W);
  eq([f.h.length, [...f.ids]], [n * W, ids]);
  // no width, or a length that is not n x width (+ ids): all of it is hidden state, as unpackWire reads it
  eq(splitWire(withId, 1, 0).h.length, W + 2);
  eq([splitWire(plain, 1, 2048).h.length, splitWire(plain, 1, 2048).ids], [W, null]);
  eq(unpackWire(withId).length, W + 2, "returns never carry ids: unpackWire is unchanged");
  // wireDim: hDim when the engine has one
  eq([wireDim({ dims: { dim: 2560, hDim: 10240 } }), wireDim({ dims: { dim: 2048 } }), wireDim({}), wireDim(null)], [10240, 2048, 0, 0]);
  let threw = false; try { packWire(hb, [1.5]); } catch { threw = true; }
  ok(threw, "a non-integer id is refused");
});

// ---------------------------------------------------------------- planner

const mac = (gb, diskGB) => ({ webgpu: true, contribGB: gb, ...(diskGB != null ? { diskGB } : {}) });
const pc = (gb, ram, diskGB) => ({ webgpu: true, contribGB: gb, offload: true, ramGB: ram, ...(diskGB != null ? { diskGB } : {}) });
function deal(metas, { mode = "memory", ctx = 2048, model = KEY } = {}) {
  const rb = roomBytes(model, ctx), pledges = metas.map((m) => pledgeGB(m) * GB);
  return dealRoom({ L: rb.L, layerBytes: rb.layerBytes, hostBytes: rb.hostBytes, pledges, mode, off: offloadFor(metas, rb.experts), ngram: ngramFor(metas, ngramOf(model)) });
}

Deno.test("flash planner: ngramDisk / ngramFor read what each device says", () => {
  eq([ngramDisk({ diskGB: 40 }), ngramDisk({ ngram: true }), ngramDisk({}), ngramDisk({ diskGB: "x" }), ngramDisk(null), ngramDisk({ diskGB: 0 })], [40 * GB, Infinity, null, null, null, 0]);
  eq(ngramFor([mac(10, 40), {}], { layer: 1, bytes: 5 }), { layer: 1, bytes: 5, disk: [40 * GB, null] });
  eq([ngramFor([mac(10)], null), ngramFor([mac(10)], { layer: 1, bytes: 0 })], [null, null]);
  eq([ngramHolder([[0, 1], [1, 48]], 1), ngramHolder([[0, 14], [14, 48]], 1), ngramHolder([[0, 3]], 7)], [1, 0, -1]);
});

Deno.test("flash planner: Mac (tab) + 5070 PC (offload) deal puts the table with layer 1 on the Mac", () => {
  // the demo room (roadmap/33 §4): the Mac lends ~26 GB in a tab with the table in OPFS, the PC 10 GB VRAM + 48 GB RAM
  const d = deal([mac(26, 60), pc(10, 48, 400)]);
  ok(d.fit.fits && d.fit.offload, "fits with the PC offloading");
  eq(d.ranges[0][0], 0);
  ok(d.assigned[0] >= 2, "the Mac holds layers 0 and 1");
  eq([d.ngram.device, d.ngram.at, d.ngram.layer, d.ngram.checked], [0, 0, 1, true]);
  ok(d.offload[1] && !d.offload[0], "the PC offloads, the tab does not");
  // the Mac short of disk for the table: an offload deal does not move its boundary; the room does not start
  const s = deal([mac(26, 10), pc(10, 48, 400)]);
  eq([s.fit.fits, s.fit.ngram, s.used.length], [false, { layer: 1, bytes: 28800138240 }, 0]);
  // what the room page and the room node say then (shortNote): the disk, not a pledge
  eq(shortNote("Qwen3.8 Flash Next", s.fit, ["mac", "pc"]), "Qwen3.8 Flash Next needs 26.9 GB of free disk for its n-gram table on the device that holds layer 1, and the devices that could hold that layer don't have it. Free disk space there or add a device that has the room.");
  // neither says (an older client): the deal is the one without the field, unchecked
  const u = deal([mac(26), pc(10, 48)]);
  eq([u.fit.fits, u.ngram.device, u.ngram.checked], [true, 0, false]);
  eq(u.ranges, d.ranges);
});

Deno.test("flash planner: the boundary moves so a device with the disk holds layer 1", () => {
  const rb = roomBytes(KEY, 2048), L = rb.L, LB = rb.layerBytes, HB = rb.hostBytes, T = 28800138240;
  // memory mode, two big devices: the host holds 0..k (k >= 2) but has no disk -> the next device takes 1..k-1
  const base = dealRoom({ L, layerBytes: LB, hostBytes: HB, pledges: [40 * GB, 80 * GB] });
  ok(base.assigned[0] >= 2);
  const moved = dealRoom({ L, layerBytes: LB, hostBytes: HB, pledges: [40 * GB, 80 * GB], ngram: { layer: 1, bytes: T, disk: [GB, 500 * GB] } });
  eq([moved.assigned[0], moved.assigned[1], moved.ngram.device, moved.ngram.at], [1, L - 1, 1, 1]);
  eq(moved.held, [LB + HB, (L - 1) * LB]);
  // the other device can't take the host's layers past its pledge: no deal
  const cant = dealRoom({ L, layerBytes: LB, hostBytes: HB, pledges: [40 * GB, 50 * GB], ngram: { layer: 1, bytes: T, disk: [GB, 500 * GB] } });
  const b2 = dealRoom({ L, layerBytes: LB, hostBytes: HB, pledges: [40 * GB, 50 * GB] });
  ok(b2.fit.fits && b2.assigned[1] + b2.assigned[0] - 1 > b2.fit.caps[1], "(the test's premise)");
  eq([cant.fit.fits, cant.fit.ngram], [false, { layer: 1, bytes: T }]);
  // the host holds only layer 0 and has the disk, the next device doesn't: the host takes layer 1
  const solo1 = placeNgram({ fit: { fits: true, caps: [3, 47, 47] }, used: [0, 1, 2], assigned: [1, 23, 24], ranges: [[0, 1], [1, 24], [24, 48]], held: [], out: {} },
    { layer: 1, bytes: T, disk: [100 * GB, 0, null] }, { layerBytes: LB, hostBytes: HB });
  eq([solo1.assigned, solo1.ranges[0], solo1.ngram.device, solo1.ngram.checked], [[2, 22, 24], [0, 2], 0, true]);
  // ... and a device left holding nothing leaves the chain (an ask-only guest)
  const gone = placeNgram({ fit: { fits: true, caps: [3, 47, 47] }, used: [0, 1, 2], assigned: [1, 1, 46], ranges: [[0, 1], [1, 2], [2, 48]], held: [], out: {} },
    { layer: 1, bytes: T, disk: [100 * GB, 0, 0] }, { layerBytes: LB, hostBytes: HB });
  eq([gone.used, gone.assigned, gone.out, gone.ngram.at], [[0, 2], [2, 46], { 1: "unneeded" }, 0]);
  // nobody has the disk: no deal, and fit says what is missing
  const none = dealRoom({ L, layerBytes: LB, hostBytes: HB, pledges: [40 * GB, 80 * GB], ngram: { layer: 1, bytes: T, disk: [GB, GB] } });
  eq([none.fit.fits, none.fit.ngram, none.used], [false, { layer: 1, bytes: T }, []]);
  // speed mode deals the same way around it
  const sp = dealRoom({ L, layerBytes: LB, hostBytes: HB, pledges: [40 * GB, 80 * GB], mode: "speed", ms: [1, 1], ngram: { layer: 1, bytes: T, disk: [GB, Infinity] } });
  eq([sp.ngram.device, sp.assigned[0]], [1, 1]);
  // a model without a table: dealRoom is unchanged
  eq(dealRoom({ L, layerBytes: LB, hostBytes: HB, pledges: [40 * GB, 80 * GB], ngram: null }), base);
});

// ---------------------------------------------------------------- a fake GPU and a fake Flash core

// Buffers are bytes; copies run on submit, in order; mapAsync reads them. Enough for FlashRoomEngine's slots.
function fakeDevice() {
  const live = new Set();
  const dev = {
    live,
    createBuffer({ size }) {
      const b = { size, bytes: new Uint8Array(size), destroyed: false,
        destroy() { b.destroyed = true; live.delete(b); },
        async mapAsync() { if (b.destroyed) throw new Error("mapAsync on a destroyed buffer"); },
        getMappedRange(off = 0, n = size - off) { return b.bytes.buffer.slice(off, off + n); }, unmap() {} };
      live.add(b);
      return b;
    },
    createCommandEncoder() {
      const ops = [];
      return { copyBufferToBuffer(s, so, d, dO, n) { ops.push(() => {
        if (s.destroyed || d.destroyed) throw new Error("copy with a destroyed buffer");
        d.bytes.set(s.bytes.subarray(so, so + n), dO);
      }); }, finish: () => ops };
    },
    queue: {
      submit(list) { for (const ops of list) for (const f of ops) f(); },
      writeBuffer(b, off, data) { b.bytes.set(new Uint8Array(data.buffer ? data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) : data), off); },
    },
  };
  return dev;
}

// A toy Flash: dim 4, 4 streams (hDim 16), L layers. Layer 1 is the n-gram layer: the token's rows (a real
// NgramState over a toy table) gated into the residual, plus a dilated causal conv over a 10-slot ring as
// engine/qwen4exp.js keeps it. Layers 3, 7, ... "attend" over their K rows [0, pos]; the others keep a recurrent
// state. Everything that carries over between tokens lives in the fake GPU's buffers (or ngs), so a wrong slot,
// a wrong reset, a missing id or a lost predecessor changes the outputs.
const T_DIM = 4, T_HC = 4, T_HD = 16, T_VOCAB = 40, T_EOS = 39;
const NG_CFG = makeConfig({ offsets: [0, 101, 204, 307, 410, 513], vocab: [101, 103, 103, 103, 103, 103], mult: [23703573157769, 20109073645365, 8052911324071],
  ngramSize: 3, perNgram: 3, width: 32, eos: T_EOS, dim: 192 });
function fakeCore(dev, { lo, hi, L, hasEmbed, hasHead, maxSeq = 64 }) {
  const f32 = (b) => new Float32Array(b.bytes.buffer);
  const layers = [];
  for (let i = lo; i < hi; i++) {
    const st = {};
    if ((i + 1) % 4 === 0) { st.kc = dev.createBuffer({ size: maxSeq * T_HD * 4 }); }
    else { st.S = dev.createBuffer({ size: T_HD * 4 }); }
    if (i === 1) st.ring = dev.createBuffer({ size: 10 * T_HD * 4 });
    layers.push({ i, st });
  }
  const ple = lo <= 1 && 1 < hi;
  const w = (i, j) => Math.sin(1.3 * i + 0.7 * j + 0.1);   // a fixed "weight"
  const core = {
    device: dev, dims: { dim: T_DIM, hDim: T_HD, vocab: hasHead ? T_VOCAB : 0 }, lo, hi, hasEmbed, hasHead, maxSeq, pos: 0,
    ngs: ple ? new NgramState(NG_CFG) : null, pleLayers: ple ? [1] : [], steps: 0,
    async step(id, hidden) {
      if (core.pos >= maxSeq) throw new Error("context full");
      const p = core.pos, R = new Float32Array(T_HD);
      if (hidden) R.set(hidden); else { if (!hasEmbed) throw new Error("no embedding"); for (let c = 0; c < T_HC; c++) for (let k = 0; k < T_DIM; k++) R[c * T_DIM + k] = 0.3 * w(id, k); }
      for (const { i, st } of layers) {
        if (i === 1) {
          const rows = core.ngs.push([id]), ring = f32(st.ring), slot = p % 10;
          for (let j = 0; j < T_HD; j++) {
            let e = 0; for (const r of rows) e += Math.cos(r * 0.37 + j);
            const g = 0.05 * e / rows.length * (1 / (1 + Math.exp(-R[j])));
            ring[slot * T_HD + j] = g;
            const back = p >= 3 ? ring[((p - 3) % 10) * T_HD + j] : 0, back9 = p >= 9 ? ring[((p - 9) % 10) * T_HD + j] : 0;
            R[j] += g + 0.5 * back - 0.25 * back9;
          }
        }
        if (st.kc) {
          const K = f32(st.kc); K.set(R, p * T_HD);
          const m = new Float32Array(T_HD); for (let q = 0; q <= p; q++) for (let j = 0; j < T_HD; j++) m[j] += K[q * T_HD + j] / (p + 1);
          for (let j = 0; j < T_HD; j++) R[j] = Math.tanh(R[j] + 0.3 * m[(j + 1) % T_HD] * w(i, j));
        } else {
          const S = f32(st.S);
          for (let j = 0; j < T_HD; j++) { S[j] = 0.8 * S[j] + 0.2 * R[(j + i) % T_HD]; R[j] = Math.tanh(R[j] + 0.5 * S[j] * w(i, j)); }
        }
      }
      core.pos++; core.steps++;
      return R;
    },
    async head(h) {
      const out = new Float32Array(T_VOCAB);
      for (let v = 0; v < T_VOCAB; v++) for (let j = 0; j < T_HD; j++) out[v] += h[j] * w(v + 50, j);
      return out;
    },
    reset() {
      core.pos = 0; core.ngs?.reset();
      for (const { st } of layers) for (const b of [st.S, st.ring]) if (b) b.bytes.fill(0);
    },
    stateParts(pos) {
      const parts = [];
      for (const { st } of layers) {
        if (st.kc) parts.push({ buf: st.kc, bytes: pos * T_HD * 4 });
        for (const b of [st.S, st.ring]) if (b) parts.push({ buf: b, bytes: b.size });
      }
      return parts;
    },
  };
  return core;
}

const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };

Deno.test("flash adapter: one device, every layer: decode, slots, export / import with the n-gram state", async () => {
  const dev = fakeDevice(), L = 8;
  const E = new FlashRoomEngine(fakeCore(dev, { lo: 0, hi: L, L, hasEmbed: true, hasHead: true }));
  eq([E.dims.hDim, E.frameIds, E.ckpt, E.mtp, E.maxSeq, E.stateSignature()], [16, true, true, null, 64, { v: 1, arch: "qwen4exp", lo: 0, hi: L, hDim: 16, kv: "f32", ple: [1] }]);
  const run = async (ids, n) => {
    const out = [];
    await E.prefillTokens(ids.slice(0, -1));
    let t = ids[ids.length - 1];
    for (let k = 0; k < n; k++) { const h = await E.embedRun(t, E.pos); t = argmax(await E.headFromHidden(h)); out.push(t); }
    return out;
  };
  const prompt = [5, 9, 9, 2, T_EOS, 7, 3, 11, 12, 1];
  const a = await run(prompt, 12);
  E.reset();
  eq(await run(prompt, 12), a, "a reset clears the ring, the recurrent state and the n-gram predecessors");
  E.reset();
  await E.prefillTokens(prompt);
  E.saveSlot(1);
  const after = await run([30], 6);
  E.loadSlot(1);
  eq(await run([30], 6), after, "a slot restores the PLE ring and the predecessors");
  const st = await E.exportSlot(1);
  eq([st.pos, st.parts.length, st.parts[st.parts.length - 1].byteLength], [prompt.length, E.core.stateParts(1).length + 1, 12]);
  eq([...new Int32Array(st.parts[st.parts.length - 1])], [prompt[9], prompt[8], prompt.length], "[p1, p2, pos]: the last two tokens");
  E.reset();
  await run([4, 4, 4], 3);   // the state moves on
  E.importState(st);
  eq(E.pos, prompt.length);
  eq(await run([30], 6), after, "importState (a checkpoint read from disk) restores both too");
  let bad = false; try { E.importState({ ...st, sig: { ...st.sig, lo: 1 } }); } catch { bad = true; }
  ok(bad, "another layer range is refused");
  bad = false; try { E.importState({ ...st, parts: st.parts.slice(0, -1) }); } catch { bad = true; }
  ok(bad, "a state without the n-gram part is refused");
  bad = false; try { await E.runHidden(new Float32Array(16), E.pos); } catch (e) { bad = /token ids/.test(e.message); }
  ok(bad, "the n-gram layer refuses a frame without its ids");
  bad = false; try { await E.embedRunBatch([1, 2], E.pos, { base: 0, total: 2 }); } catch (e) { bad = /speculative/.test(e.message); }
  ok(bad, "no speculative verify yet");
  E.dropAllSlots();
  eq(E.slots.size, 0);
  // a state buffer without COPY_SRC (PR #346's K/V caches today): this device has no checkpoints at all. The slot
  // methods are gone, so everything that tests for them leaves checkpoints off (room/pipeline.js ckptEngine, the
  // worker's `sv` in workerFrame, ai-ready's `ckpt`, the room node's disk copies) instead of a save throwing
  const core = fakeCore(dev, { lo: 0, hi: L, L, hasEmbed: true, hasHead: true }), parts = core.stateParts;
  core.stateParts = (pos) => parts(pos).map((p, i) => (i === 0 ? { ...p, buf: Object.assign(Object.create(p.buf), { usage: 0x80 | 8 }) } : p));
  const N = new FlashRoomEngine(core);
  eq([N.ckpt, N.saveSlot, N.loadSlot, N.exportSlot, N.exportState, N.importState, N.stateSignature, N.saveSlot ? 1 : 0], [false, undefined, undefined, undefined, undefined, undefined, undefined, 0]);
  N.dropSlot(3); N.dropAllSlots();   // (dropping is still safe: there is nothing to drop)
  eq(N.slots.size, 0);
  ok(typeof E.saveSlot === "function", "a device whose buffers can all be copied keeps them");
});

// A host and a worker, each a room/pipeline.js over its own FlashRoomEngine, frames through a wire that keeps only
// what room/transport.js carries (kind, position, column count, flags, checkpoint control and the payload).
function twoDeviceRoom(split, { L = 8 } = {}) {
  const dev = [fakeDevice(), fakeDevice()];
  const state = (o) => ({ waiters: new Map(), pendingCtl: {}, dropQ: [], pos: 0, fed: [], q: Promise.resolve(), ...o });
  const host = state({ role: "host", chain: ["w"], engine: new FlashRoomEngine(fakeCore(dev[0], { lo: 0, hi: split, L, hasEmbed: true, hasHead: true })) });
  const work = state({ role: "worker", chain: [], next: "host", hostId: "host", range: [split, L],
    engine: new FlashRoomEngine(fakeCore(dev[1], { lo: split, hi: L, L, hasEmbed: false, hasHead: false })) });
  const frames = [];
  const wire = (m) => {
    const keep = ["t", "pos", "basePos", "n", "spec", "reset", "rb", "sv", "ld", "dp", "enc"];
    const out = Object.fromEntries(keep.filter((k) => m[k] != null).map((k) => [k, m[k]]));
    out.data = m.data.slice();   // the payload's bytes, nothing else
    frames.push(out);
    return out;
  };
  let ph, pw;
  ph = createPipeline({ state: host, options: { profile: "node", checkpointMax: 2 },
    transport: { sendTo() {}, sendHidden: (id, m) => { const f = wire(m); setImmediate(() => pw.handleFrame("host", f)); } } });
  pw = createPipeline({ state: work, options: { profile: "node" },
    transport: { sendTo: (id, m) => { if (m.t === "ai-error") ph.failWaiters(new Error("worker: " + m.message)); }, sendHidden: (id, m) => { const f = wire(m); setImmediate(() => ph.handleFrame("w", f)); } } });
  return { host, work, ph, pw, frames };
}
// prompt -> n greedy tokens through the pipeline (generate.js's plain decode: aiPrefill, then aiPipeToken per token);
// seen: every logit vector, for the exact checks
async function greedy(p, ai, ids, n, seen = []) {
  let logits = await p.aiPrefill(ids, { aborted: () => false });
  const out = [];
  for (let k = 0; k < n; k++) { seen.push(...logits); const t = argmax(logits); out.push(t); logits = await p.aiPipeToken(t, true); }
  return out;
}

Deno.test("flash room: host + worker (layer 1 on either side) match one device, through prefill rounds and checkpoints", async () => {
  const L = 8, prompt = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, T_EOS, 6, 4, 3, 3, 8];   // 28: two prefill rounds and a tail
  // one device: the same pipeline with no chain
  const soloAi = { waiters: new Map(), pendingCtl: {}, dropQ: [], pos: 0, fed: [], q: Promise.resolve(), chain: [], role: "host",
    engine: new FlashRoomEngine(fakeCore(fakeDevice(), { lo: 0, hi: L, L, hasEmbed: true, hasHead: true })) };
  const solo = createPipeline({ state: soloAi, options: { profile: "node" }, transport: { sendTo() {}, sendHidden() { throw new Error("solo"); } } });
  const want = await greedy(solo, soloAi, prompt, 10);
  for (const split of [1, 2, 5]) {
    // split 1: the host holds only layer 0, so the n-gram layer is the worker's and every frame must carry the ids
    const R = twoDeviceRoom(split, { L });
    const got = await greedy(R.ph, R.host, prompt, 10);
    // the f16 wire rounds the residual at the boundary: the toy's greedy ids are stable under that
    eq(got, want, `split at ${split}`);
    eq(R.host.pos, prompt.length + 10);
    ok(R.frames.some((f) => f.t === "ai-hidden-b") && R.frames.some((f) => f.t === "ai-hidden"), "batched prefill rounds and single tokens both ran");
    for (const f of R.frames) {
      const n = f.n || 1, ret = f.t.includes("ret");
      eq(f.data.length, n * T_HD + (ret ? 0 : 2 * n), `${f.t}: ${n} columns of hDim f16 ${ret ? "(a return: no ids)" : "+ their ids"}`);
    }
    eq(R.work.engine.core.ngs ? R.work.engine.core.ngs.pos : R.host.engine.core.ngs.pos, prompt.length + 10, "the n-gram layer saw every token once");

    // checkpoint: saved with the next frame on both devices, then a resume from it reproduces the continuation
    const R2 = twoDeviceRoom(split, { L });
    await R2.ph.aiPrefill(prompt, { aborted: () => false });
    R2.ph.ckptSave(true);
    const key = R2.host.pendingCtl.sv;
    ok(key > 0, "a save is pending for the chain");
    const tail = [11, 12, 13];
    const firstLogits = [], first = await greedy(R2.ph, R2.host, tail, 6, firstLogits);
    ok(R2.work.engine.slots.has(key) && R2.host.engine.slots.has(key), "both devices saved the slot");
    // something else in between: a different conversation moves every state on
    R2.ph.resetState();
    await greedy(R2.ph, R2.host, [20, 21, 22, 23, 24], 4);
    R2.ph.resetState();
    const ids = [...prompt, ...tail];
    const reused = R2.ph.ckptResume(ids, 0);
    eq(reused, prompt.length, "the checkpoint covers the prompt");
    eq(R2.host.pendingCtl.ld, key, "the load rides the next frame");
    const againLogits = [], again = await greedy(R2.ph, R2.host, tail, 6, againLogits);
    eq(again, first, `split at ${split}: resumed from the checkpoint (PLE ring + predecessors restored on whichever device has layer 1)`);
    eq(againLogits, firstLogits, `split at ${split}: the same logits, bit for bit`);
  }
});

// ---------------------------------------------------------------- the core's failed-step guard

Deno.test("flash core: a step that throws part way marks the engine failed, as forwardToken does; reset() clears it", async () => {
  const HD = 8;
  let lose = false, submits = 0;
  const stage = { mapAsync: async () => { if (lose) throw Object.assign(new Error("device lost"), { name: "AbortError" }); }, getMappedRange: () => new ArrayBuffer(HD * 4), unmap() {} };
  const pass = { end() {} }, enc = { beginComputePass: () => pass, copyBufferToBuffer() {}, finish: () => ({}) };
  const e = {
    device: { queue: { submit: () => { submits++; }, writeBuffer() {} }, createCommandEncoder: () => enc },
    cfg: { hDim: HD, nKVHeads: 1, headDim: 2 }, dims: {}, lo: 0, hi: 1, hasEmbed: true, hasHead: false, maxSeq: 4, pos: 0,
    layers: [{ attnOps: [], ffnOps: [], st: {} }], B: { R: {} }, hStage: stage, _failed: null,
    // engine/qwen4exp.js's checks, as #346 has them
    _check(id) {
      if (this._failed) throw new Error("qwen4exp: an earlier step failed (" + this._failed.message + "); reset() first");
      if (this.pos >= this.maxSeq) throw new Error(`qwen4exp: context full (${this.maxSeq})`);
      if (!(Number.isInteger(id) && id >= 0 && id < 10)) throw new Error(`qwen4exp: token id ${id} outside the vocab (10)`);
    },
    async _begin(id) { this._check(id); },
    _encode() {},
    reset() { this.pos = 0; this._failed = null; },
  };
  const core = qwen4expCore(e);
  eq((await core.step(1)).length, HD);
  eq(e.pos, 1);
  // a check that changes nothing throws without marking the engine
  let msg = "";
  try { await core.step(99); } catch (err) { msg = err.message; }
  ok(/outside the vocab/.test(msg) && !e._failed, "a bad id is refused, the engine not failed");
  lose = true;
  msg = "";
  try { await core.step(2); } catch (err) { msg = err.message; }
  ok(msg === "device lost" && e._failed?.message === "device lost", "the readback's error marks the engine failed");
  lose = false;
  const before = submits;
  msg = "";
  try { await core.step(3); } catch (err) { msg = err.message; }
  ok(/an earlier step failed \(device lost\); reset\(\) first/.test(msg), msg);
  eq(submits, before, "a refused step submits nothing");
  core.reset();
  eq((await core.step(4)).length, HD, "after reset() the engine steps again");
});

Deno.test("flash room engine: a checkpoint restored (loadSlot / importState) clears a failed step, as Qwen35Engine's do", async () => {
  const HD = 8;
  let lose = false;
  const stage = { mapAsync: async () => { if (lose) throw new Error("device lost"); }, getMappedRange: () => new ArrayBuffer(HD * 4), unmap() {} };
  const enc = { beginComputePass: () => ({ end() {} }), copyBufferToBuffer() {}, finish: () => ({}) };
  const e = {
    device: { queue: { submit() {}, writeBuffer() {} }, createCommandEncoder: () => enc, createBuffer: () => ({ destroy() {} }) },
    cfg: { hDim: HD, nKVHeads: 1, headDim: 2 }, dims: { hDim: HD }, lo: 0, hi: 1, hasEmbed: true, hasHead: false, maxSeq: 8, pos: 0,
    layers: [{ attnOps: [], ffnOps: [], st: {} }], B: { R: {} }, hStage: stage, _failed: null,
    _check() { if (this._failed) throw new Error("qwen4exp: an earlier step failed (" + this._failed.message + "); reset() first"); },
    async _begin(id) { this._check(id); },
    _encode() {},
    reset() { this.pos = 0; this._failed = null; },
  };
  const fe = new FlashRoomEngine(qwen4expCore(e));
  ok(typeof fe.saveSlot === "function", "checkpoints on");
  await fe.prefillToken(1);
  fe.saveSlot(1);
  const fail = async () => { lose = true; try { await fe.prefillToken(2); } catch { /* the readback */ } lose = false; ok(e._failed, "the step failed"); };
  await fail();
  fe.loadSlot(1);
  ok(!e._failed, "loadSlot cleared the failed step");
  await fe.prefillToken(3);
  await fail();
  fe.importState({ sig: fe.stateSignature(), pos: 1, parts: [] });
  ok(!e._failed, "importState cleared it too");
  eq(fe.pos, 1);
  await fe.prefillToken(4);
});

Deno.test("flash core: an offloaded layer's ops split at its router and the engine's cut runs between them, in order", async () => {
  const HD = 4, seen = [];
  const stage = { mapAsync: async () => {}, getMappedRange: () => new ArrayBuffer(HD * 4), unmap() {} };
  const mkEnc = () => ({ beginComputePass: () => ({ end() {} }), copyBufferToBuffer() {}, finish: () => ({}) });
  const op = (n, cut = false) => ({ n, cut });
  const L = (i, off) => ({ i, off, st: {}, halves: [{ n: `a${i}` }, { n: `f${i}` }] });
  const e = {
    device: { queue: { submit() {}, writeBuffer() {} }, createCommandEncoder: mkEnc },
    cfg: { hDim: HD, nKVHeads: 1, headDim: 2 }, dims: {}, lo: 0, hi: 3, hasEmbed: true, hasHead: false, maxSeq: 4, pos: 0,
    layers: [L(0, false), L(1, true), L(2, false)], B: { R: {} }, hStage: stage, _failed: null,
    // fused lists, two per layer: attention, then the MoE half with its router op marked cut
    _halfOps: (halves) => halves.map((h) => (h.n[0] === "a" ? [op(h.n)] : [op(h.n + ".route", true), op(h.n + ".exp")])),
    _check() {}, async _begin() {},
    _encode(_pass, ops) { for (const o of ops) seen.push(o.n); },
    async _afterRoute(enc, l) { seen.push(`cut${l.i}`); return mkEnc(); },
  };
  await qwen4expCore(e).step(1);
  eq(seen, ["a0", "f0.route", "f0.exp", "a1", "f1.route", "cut1", "f1.exp", "a2", "f2.route", "f2.exp"]);
});
