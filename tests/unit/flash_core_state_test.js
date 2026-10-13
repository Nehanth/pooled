// room/flash.js checkpoints over the engine's own state (engine/qwen4exp.js _stateParts / stateSignature, #375): the
// parts are the engine's list in its order (K / V rows in its KV format, QSA's raw-key ring and pooled keys, DeltaNet
// S / conv, the PLE ring), the signature is the engine's with the adapter's v 2, hDim and ple, slots and disk copies
// round trip every part, and a state of another shape or an older (v1) signature is refused. An engine without
// _stateParts (before #375) keeps the dense f32 list, and is refused with f16 KV or QSA rather than checkpointed wrong.
// No GPU.   deno test --allow-read tests/unit/flash_core_state_test.js
import { FlashRoomEngine, qwen4expCore } from "../../room/flash.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const throws = (f, re) => { try { f(); } catch (err) { if (re.test(err.message)) return; throw err; } throw new Error("did not throw"); };
const COPY_SRC = 4, COPY_DST = 8, MAP_READ = 1;

// a fake GPU: buffers are bytes, copies run at submit
function fakeDevice() {
  return {
    createBuffer({ size, usage = COPY_SRC | COPY_DST }) {
      const b = { size, usage, bytes: new Uint8Array(size), destroyed: false, destroy() { b.destroyed = true; },
        async mapAsync() {}, getMappedRange(off = 0, n = size - off) { return b.bytes.buffer.slice(off, off + n); }, unmap() {} };
      return b;
    },
    createCommandEncoder() {
      const ops = [];
      return { copyBufferToBuffer(s, so, d, dO, n) { ops.push(() => d.bytes.set(s.bytes.subarray(so, so + n), dO)); }, finish: () => ops };
    },
    queue: {
      submit(list) { for (const ops of list) for (const f of ops) f(); },
      writeBuffer(b, off, data) { b.bytes.set(new Uint8Array(data.buffer ? data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) : data), off); },
    },
  };
}

// A fake engine with #375's state API: layers 0-2 DeltaNet (S, conv), layer 3 attention with QSA (K, V, rawK ring,
// pooled keys); kv f16 (2 B per value), ring 4 rows. _stateParts says the bytes that matter at pos, as the engine does.
function fakeEngine({ kv = "f16", qsa = true, own = true } = {}) {
  const dev = fakeDevice(), row = 2 * (kv === "f16" ? 2 : 4), maxSeq = 16, ID = 2, POOL = 4;
  const B = (n) => dev.createBuffer({ size: n });
  const layers = [0, 1, 2].map((i) => ({ i, attnOps: [], ffnOps: [], st: { S: B(32), conv: B(8) } }));
  layers.push({ i: 3, attnOps: [], ffnOps: [], st: { kc: B(maxSeq * row), vc: B(maxSeq * row), ...(qsa ? { qsa: { rawK: B(4 * ID * 2), pooled: B(maxSeq / POOL * ID * 2), rawRing: true } } : {}) } });
  const e = { device: dev, cfg: { hDim: 8, nKVHeads: 1, headDim: 2 }, dims: { dim: 2, hDim: 8, vocab: 0 }, lo: 0, hi: 4, hasEmbed: true, hasHead: false,
    maxSeq, pos: 0, kv, qsa, qsaRing: 4, layers, _failed: null, reset() { this.pos = 0; } };
  if (own) {
    e._stateParts = (pos = e.pos) => {
      const parts = [];
      for (const { st } of layers) {
        if (st.kc) {
          parts.push({ buf: st.kc, bytes: pos * row }, { buf: st.vc, bytes: pos * row });
          if (st.qsa) parts.push({ buf: st.qsa.rawK, bytes: st.qsa.rawK.size }, { buf: st.qsa.pooled, bytes: Math.floor(pos / POOL) * ID * 2 });
        }
        for (const b of [st.S, st.conv, st.ring]) if (b) parts.push({ buf: b, bytes: b.size });
      }
      return parts;
    };
    e.stateSignature = () => ({ v: 1, arch: "qwen4exp", lo: e.lo, hi: e.hi, kv: e.kv, qsa: !!e.qsa, qsaRing: e.qsaRing, dim: 2 });
  }
  return e;
}
// every state byte the engine holds (the whole buffers), to compare states
const dump = (e) => e.layers.flatMap(({ st }) => [st.kc, st.vc, st.qsa?.rawK, st.qsa?.pooled, st.S, st.conv].filter(Boolean).map((b) => [...b.bytes]));
const scribble = (e, seed) => { let k = seed; for (const { st } of e.layers) for (const b of [st.kc, st.vc, st.qsa?.rawK, st.qsa?.pooled, st.S, st.conv]) if (b) for (let i = 0; i < b.size; i++) b.bytes[i] = (k = (k * 1103515245 + 12345) >>> 0) >>> 24; };

Deno.test("flash core: the engine's _stateParts are the checkpoint, in its order (f16 KV with QSA is accepted)", () => {
  const e = fakeEngine(), core = qwen4expCore(e);
  e.pos = 9;
  eq(core.stateParts(9).map((p) => p.bytes), e._stateParts(9).map((p) => p.bytes));
  const L3 = e.layers[3].st;
  eq(core.stateParts(9).slice(-4).map((p) => p.buf === L3.kc ? "K" : p.buf === L3.vc ? "V" : p.buf === L3.qsa.rawK ? "rawK" : p.buf === L3.qsa.pooled ? "pooled" : "?"),
    ["K", "V", "rawK", "pooled"], "K, V, the raw-key ring (whole), then the complete blocks' pooled keys");
  eq(core.stateParts(9).slice(-4).map((p) => p.bytes), [9 * 4, 9 * 4, 16, 2 * 4]);
  const E = new FlashRoomEngine(core);
  eq(E.ckpt, true);
  const sig = E.stateSignature();
  eq(sig, { v: 2, arch: "qwen4exp", lo: 0, hi: 4, kv: "f16", qsa: true, qsaRing: 4, dim: 2, hDim: 8, ple: [] }, "the engine's signature, then v 2, hDim, ple");
  eq(sig.v, 2, "the adapter's version wins over the engine's v 1");
});

Deno.test("flash core: a slot and a disk copy bring back every part (QSA ring and pooled keys included)", async () => {
  for (const [kv, qsa] of [["f16", true], ["f32", true], ["f32", false]]) {
    const e = fakeEngine({ kv, qsa }), E = new FlashRoomEngine(qwen4expCore(e));
    scribble(e, 1); e.pos = 13;
    const want = dump(e);
    E.saveSlot(1);
    const st = await E.exportSlot(1);
    eq([st.pos, st.parts.length, st.sig.v], [13, e._stateParts(13).length, 2]);
    scribble(e, 2); e.pos = 3;
    E.loadSlot(1);
    eq(e.pos, 13);
    // the rows past pos are not part of the state: compare what _stateParts says matters
    const live = (x) => e._stateParts(13).map((p) => x[e.layers.flatMap(({ st }) => [st.kc, st.vc, st.qsa?.rawK, st.qsa?.pooled, st.S, st.conv].filter(Boolean)).indexOf(p.buf)].slice(0, p.bytes));
    eq(live(dump(e)), live(want), `${kv}${qsa ? " QSA" : ""}: loadSlot`);
    scribble(e, 3); e.pos = 5;
    E.importState(st);
    eq([e.pos, live(dump(e))], [13, live(want)], `${kv}${qsa ? " QSA" : ""}: importState`);
    // another shape: a part cut short, or one part missing
    throws(() => E.importState({ ...st, parts: st.parts.map((p, i) => (i === st.parts.length - 3 ? p.slice(0, p.byteLength - 2) : p)) }), /wrong shape/);
    throws(() => E.importState({ ...st, parts: st.parts.slice(1) }), /wrong shape/);
    // an older (v1) signature, or another KV format / QSA / ring
    throws(() => E.importState({ ...st, sig: { ...st.sig, v: 1 } }), /different model/);
    throws(() => E.importState({ ...st, sig: { v: 1, arch: "qwen4exp", lo: 0, hi: 4, hDim: 8, kv: "f32", ple: [] } }), /different model/);
    throws(() => E.importState({ ...st, sig: { ...st.sig, kv: kv === "f16" ? "f32" : "f16" } }), /different model/);
    throws(() => E.importState({ ...st, sig: { ...st.sig, qsa: !qsa } }), /different model/);
    throws(() => E.importState({ ...st, sig: { ...st.sig, qsaRing: 8 } }), /different model/);
  }
});

Deno.test("flash core: an engine without _stateParts keeps the dense f32 list; f16 KV or QSA is refused there", () => {
  const plain = fakeEngine({ kv: "f32", qsa: false, own: false });
  const E = new FlashRoomEngine(qwen4expCore(plain));
  eq(E.stateSignature(), { v: 2, arch: "qwen4exp", lo: 0, hi: 4, kv: "f32", hDim: 8, ple: [] });
  eq(E.core.stateParts(3).map((p) => p.bytes), [32, 8, 32, 8, 32, 8, 3 * 8, 3 * 8], "S, conv x3, then f32 K / V rows");
  throws(() => qwen4expCore(fakeEngine({ kv: "f16", qsa: false, own: false })), /not kv f16/);
  throws(() => qwen4expCore(fakeEngine({ kv: "f32", qsa: true, own: false })), /with QSA/);
});
