// room.js checkpoints (?ckpt) and the control that rides on the next frame (pendingCtl), host and
// worker side. room.js is DOM-bound, so room_src.js cuts the real functions out of its source and
// runs them here over a stub `ai` state; nothing is copied.
//
// Why this matters: every device keeps its own caches, and the host only tells the chain what to
// do (roll back, save, drop, reset, load) on the next frame it sends. If the host's view of what
// the workers hold drifts from what they actually hold, the room either fails a lap ("no saved
// slot") or, worse, runs a worker's layers on the wrong state and answers garbage without an error.
import { roomFns, fnSource } from "./room_src.js";
import { PrefixIndex, pinSplit } from "../../harness/prefix.js";
import { DROP_ALL, sendFrame, makeLink } from "../../room/transport.js";
import { reusablePrefix } from "../../room/conversation.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const throws = (f, re, m) => {
  let err = null; try { f(); } catch (e) { err = e; }
  if (!err) throw new Error((m || "expected a throw") + ": nothing thrown");
  if (re && !re.test(err.message)) throw new Error((m || "wrong error") + ": " + err.message);
};
const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);

// An engine whose whole state is the list of tokens it has run, in order (a stand-in for the KV
// rows and the DeltaNet state, which are a function of exactly that list). Slots behave like
// Qwen35Engine's: saveSlot overwrites, loadSlot of a missing slot throws, dropSlot of a missing one
// is a no-op. restoreDN(k) keeps the last batch's first k+1 columns, like a verify rollback.
class FakeEngine {
  constructor() { this.st = []; this.slots = new Map(); this.log = []; this.lastBase = 0; this.dims = { dim: 1 }; this.NC = 64; }
  reset() { this.log.push("reset"); this.st = []; }
  saveSlot(k) { this.log.push("sv" + k); this.slots.set(k, this.st.slice()); }
  loadSlot(k) {
    this.log.push("ld" + k);
    const s = this.slots.get(k);
    if (!s) throw new Error("no saved slot " + k);
    this.st = s.slice();
  }
  dropSlot(k) { this.log.push("dp" + k); this.slots.delete(k); }
  dropAllSlots() { this.log.push("dpAll"); this.slots.clear(); }
  restoreDN(k) { this.log.push("rb" + k); this.st.length = this.lastBase + k + 1; }
  run(toks, base) {
    if (this.st.length !== base) throw new Error(`desync: state holds ${this.st.length} tokens, frame is at ${base}`);
    this.lastBase = base; this.st.push(...toks);
  }
  async runHidden(x, pos) { this.run([x[0]], pos); return Float32Array.of(x[0]); }
  async runHiddenBatch(xs, base) { this.run(Array.from(xs), base); return Float32Array.from(xs); }
}

const HOST_FNS = ["sendChain", "resetState", "ckptClear", "ckptSave", "ckptResume", "ckptWhere", "ckptPersist", "ckptForget", "ckptRestore", "ckptRejoin"];

// the host's functions over a stub ai; sent frames land in `out`. disk: a CkptStore (null: GPU only)
function host({ chain = ["w0"], ckptMax = 2, fed = [], engine = new FakeEngine(), out = [], disk = null, room = "ROOM", model = "m" } = {}) {
  const ai = { role: "host", model, chain, pendingCtl: {}, fed, pos: fed ? fed.length : 0, engine, ckpt: null, ckptN: 0 };
  const fns = roomFns(HOST_FNS, {
    ai, CKPT_MAX: ckptMax, PrefixIndex, DROP_ALL, wireStats: { lastMax: 0 }, ckptDisk: disk, roomCode: room,
    sendHidden: (to, msg) => out.push({ to, msg }),
  });
  return { ai, out, engine, ...fns };
}

// a worker's workerFrame over a stub ai; what it forwards lands in `send`
function worker({ next = "host", engine = new FakeEngine(), send = () => {}, disk = null, room = "ROOM", model = "m", ckptMax = 2 } = {}) {
  const ai = { role: "worker", model, engine, next, hostId: "host", range: [0, 1] };
  const { workerFrame, ckptRestore } = roomFns(["workerFrame", "ckptWhere", "ckptPersist", "ckptForget", "ckptRestore"], {
    ai, DROP_ALL, performance, ckptDisk: disk, roomCode: room, CKPT_MAX: ckptMax, ckptClear: () => {},
    unpackWire: (d) => Float32Array.from(d.x),
    packWire: (h) => ({ x: Array.from(h) }),
    badF32: () => false,
    aiStatus: () => {}, sendTo: () => {}, teleNote: () => {}, compute: { pass() {} }, keepWarm: () => {},
    sendHidden: send,
  });
  return { ai, engine, workerFrame, ckptRestore };
}

// ---------------------------------------------------------------------------------------------
// the functions are really there (a rename in room.js should fail here, loudly)

Deno.test("room_src: the checkpoint and worker functions are cut out of room.js whole", () => {
  for (const n of [...HOST_FNS, "workerFrame"]) {
    const s = fnSource(n);
    ok(s.startsWith("function " + n) || s.startsWith("async function " + n), n);
    ok(s.trim().endsWith("}"), n);
    new Function(s);   // parses on its own
  }
});

// ---------------------------------------------------------------------------------------------
// resetState

Deno.test("resetState keeps a pending rollback, save and drop, adds reset, drops a pending load", () => {
  const cases = [
    { pend: {}, want: { reset: 1 } },
    { pend: { rb: 0 }, want: { rb: 0, reset: 1 } },                       // rb 0 is a real column, not "none"
    { pend: { rb: 3, sv: 7 }, want: { rb: 3, sv: 7, reset: 1 } },
    { pend: { sv: 7, dp: [5] }, want: { sv: 7, dp: [5], reset: 1 } },
    { pend: { dp: [DROP_ALL] }, want: { dp: [DROP_ALL], reset: 1 } },
    { pend: { ld: 4 }, want: { reset: 1 } },                              // the reset replaces the load
    { pend: { reset: 1, ld: 4, sv: 2 }, want: { sv: 2, reset: 1 } },
    { pend: undefined, want: { reset: 1 } },
  ];
  for (const c of cases) {
    const h = host({ fed: [1, 2, 3] });
    h.ai.pendingCtl = c.pend;
    h.resetState();
    eq(h.ai.pendingCtl, c.want, JSON.stringify(c.pend));
    eq(h.ai.fed, [], "fed"); eq(h.ai.pos, 0, "pos");
    eq(h.engine.log, ["reset"], "the host resets its own engine now");
  }
});

Deno.test("resetState without a chain (solo) leaves nothing pending", () => {
  const h = host({ chain: [], fed: [1, 2] });
  h.ai.pendingCtl = { sv: 1, rb: 2 };
  h.resetState();
  eq(h.ai.pendingCtl, {});
});

Deno.test("resetState survives an engine whose reset throws, or has none", () => {
  const h = host({ fed: [1] });
  h.engine.reset = () => { throw new Error("gpu lost"); };
  h.resetState();
  eq(h.ai.pendingCtl, { reset: 1 }); eq(h.ai.fed, []);
  const h2 = host({ fed: [1] });
  h2.ai.engine = {};
  h2.resetState();
  eq(h2.ai.pendingCtl, { reset: 1 });
});

// ---------------------------------------------------------------------------------------------
// ckptClear

Deno.test("ckptClear drops the host's slots, and tells the chain DROP_ALL only when asked and there was something", () => {
  const cases = [
    { saved: 2, tell: true, chain: ["w0"], want: { dp: [DROP_ALL] } },
    { saved: 2, tell: false, chain: ["w0"], want: {} },
    { saved: 0, tell: true, chain: ["w0"], want: {} },       // nothing saved: nothing to drop anywhere
    { saved: 2, tell: true, chain: [], want: {} },            // solo
  ];
  for (const c of cases) {
    const h = host({ chain: c.chain, fed: [1, 2, 3], ckptMax: 4 });
    for (let i = 0; i < c.saved; i++) { h.ai.fed.push(10 + i); h.ckptSave(); h.ai.pendingCtl = {}; }   // each save's frame went out
    h.ai.pendingCtl = {};
    h.engine.log = [];
    h.ckptClear(c.tell);
    eq(h.ai.pendingCtl, c.want, JSON.stringify(c));
    eq(h.ai.ckpt.items, [], "index emptied");
    eq(h.engine.log, range(1, c.saved + 1).map((k) => "dp" + k), "host slots dropped");
    eq(h.engine.slots.size, 0);
  }
});

Deno.test("ckptClear(true) keeps the rest of the pending control and does not restart slot numbers", () => {
  const h = host({ fed: [1, 2] });
  h.ckptSave();                                   // key 1, pending sv 1
  h.ai.pendingCtl = { ...h.ai.pendingCtl, rb: 2 };
  h.ckptClear(true);
  eq(h.ai.pendingCtl, { sv: 1, rb: 2, dp: [DROP_ALL] });
  // a worker runs sv before dp, so a save sharing a frame with DROP_ALL would be dropped at once
  // on every worker: it is not made at all
  h.ckptSave();
  eq(h.ai.pendingCtl, { sv: 1, rb: 2, dp: [DROP_ALL] }, "no save rides with DROP_ALL");
  eq(h.ai.ckpt.items, []);
  h.ai.pendingCtl = {};                           // the frame went out
  h.ckptSave();
  eq(h.ai.pendingCtl, { sv: 2 }, "a new slot number, never one a worker might still hold");
});

Deno.test("ckptSave: a second save before any frame went out supersedes the first everywhere", () => {
  const cases = [
    { name: "no eviction", ckptMax: 4, pre: 0, want: { sv: 2 }, keys: [2] },
    { name: "first save evicted one: its drop stays pending", ckptMax: 1, pre: 1, want: { sv: 3, dp: [1] }, keys: [3] },
    { name: "with a rollback and a load pending", ckptMax: 2, pre: 0, pend: { rb: 1, ld: 9 }, want: { rb: 1, ld: 9, sv: 2 }, keys: [2] },
  ];
  for (const c of cases) {
    const h = host({ fed: [1], ckptMax: c.ckptMax });
    for (let i = 0; i < c.pre; i++) { h.ckptSave(); h.ai.pendingCtl = {}; }
    h.ai.pendingCtl = { ...(c.pend || {}) };
    h.ai.fed.push(2); h.ckptSave();               // not sent...
    h.ckptSave();                                 // ...and saved again
    eq(h.ai.pendingCtl, c.want, c.name);
    eq(h.ai.ckpt.items.map((x) => x.key), c.keys, c.name + ": host index");
    eq([...h.engine.slots.keys()], c.keys, c.name + ": host slots");
  }
  // solo: nothing is pending, nothing superseded; a save of tokens already saved keeps that slot
  const s = host({ chain: [], fed: [1], ckptMax: 4 });
  s.ckptSave(); s.ckptSave();
  eq(s.ai.ckpt.items.map((x) => x.key), [1], "no second slot for the same state");
  s.ai.fed.push(2); s.ckptSave();
  eq(s.ai.ckpt.items.map((x) => x.key), [1, 2]);
});

Deno.test("ckptSave: the same tokens again never evict an answer checkpoint; a pin promotes the existing one", () => {
  const s = host({ chain: [], fed: [1, 2], ckptMax: 1 });
  s.ckptSave(true);                               // pinned at [1, 2]
  s.ckptSave();                                   // Stop right after it (solo): the end save
  eq(s.ai.ckpt.items.map((x) => [x.key, x.pin]), [[1, true]], "the pinned one stands, no duplicate");
  s.ai.fed.push(3); s.ckptSave();
  s.ai.fed.length = 2; s.ckptSave();              // back at the pinned tokens: still one slot there
  eq(s.ai.ckpt.items.map((x) => [x.key, x.pin]), [[1, true], [2, false]]);
  // an answer checkpoint at the tokens being pinned becomes the pinned one (no second slot)
  const t = host({ chain: [], fed: [5, 6], ckptMax: 2 });
  t.ckptSave(); t.ckptSave(true);
  eq(t.ai.ckpt.items.map((x) => x.pin), [true]);
  eq([...t.engine.slots.keys()].length, 1);
});

Deno.test("ckptSave: a superseded save does not go out when the next save keeps an existing slot", () => {
  const h = host({ fed: [1, 2, 3], ckptMax: 2 });
  h.ckptSave(); h.ai.pendingCtl = {};             // slot 1 went out
  h.ai.fed.push(4, 5); h.ckptSave();              // slot 2 pending
  h.ai.pendingCtl = { ...h.ai.pendingCtl, ld: 1 }; h.ai.fed = [1, 2, 3];   // a resume of slot 1, Stop before a frame
  h.ckptSave();
  eq(h.ai.pendingCtl, { ld: 1 }, "no sv of a slot the host forgot");
  eq(h.ai.ckpt.items.map((x) => x.key), [1]);
  eq([...h.engine.slots.keys()], [1]);
});

Deno.test("ckptClear survives an engine whose dropSlot throws", () => {
  const h = host({ fed: [1] });
  h.ckptSave();
  h.engine.dropSlot = () => { throw new Error("destroyed"); };
  h.ckptClear(true);
  eq(h.ai.ckpt.items, []);
});

// ---------------------------------------------------------------------------------------------
// ckptSave

Deno.test("ckptSave does nothing without ?ckpt, without fed tokens, or without engine slots", () => {
  const cases = [
    { ckptMax: 0, fed: [1] },
    { ckptMax: 2, fed: [] },
    { ckptMax: 2, fed: null },
    { ckptMax: 2, fed: [1], noSlots: true },
  ];
  for (const c of cases) {
    const h = host({ ckptMax: c.ckptMax, fed: c.fed });
    if (c.noSlots) h.ai.engine = {};
    h.ckptSave();
    eq(h.ai.pendingCtl, {}, JSON.stringify(c));
    eq(h.ai.ckptN, 0);
  }
});

Deno.test("ckptSave: slot numbers run 1..65534 and wrap to 1, never 0 or DROP_ALL", () => {
  const h = host({ fed: [1], ckptMax: 1 });
  h.ai.ckptN = 65532;
  const keys = [];
  for (let i = 0; i < 4; i++) { h.ai.pendingCtl = {}; h.ai.fed.push(2 + i); h.ckptSave(); keys.push(h.ai.pendingCtl.sv); }
  eq(keys, [65533, 65534, 1, 2]);
  ok(!keys.includes(0) && !keys.includes(DROP_ALL));
  // and every one of them fits the frame header
  for (const sv of keys) sendFrame(fakeLink(), { t: "ai-hidden", pos: 0, sv, data: new Uint16Array(2) });
});

function fakeLink() {
  const link = makeLink();
  link.chans.push({ readyState: "open", send() {} });
  return link;
}

Deno.test("ckptSave evicts the least recently used checkpoint (a resume counts as a use) and drops it everywhere", () => {
  const h = host({ fed: [1, 2], ckptMax: 2 });
  h.ckptSave(); h.ai.pendingCtl = {};            // 1: [1 2], sent
  h.ai.fed = [1, 2, 3, 4]; h.ckptSave();          // 2: [1 2 3 4]
  h.ai.pendingCtl = {};
  // a resume from 1 makes 2 the oldest
  eq(h.ckptResume([1, 2, 9], 0), 2);
  h.ai.pendingCtl = {};
  h.ai.fed = [1, 2, 9, 9]; h.engine.log = [];
  h.ckptSave();
  eq(h.ai.pendingCtl, { sv: 3, dp: [2] });
  eq(h.engine.log, ["dp2", "sv3"]);
  eq(h.ai.ckpt.items.map((x) => x.key).sort(), [1, 3]);
});

Deno.test("ckptSave with ckpt=1 keeps exactly one checkpoint", () => {
  const h = host({ fed: [1], ckptMax: 1 });
  for (let i = 0; i < 5; i++) { h.ai.fed.push(i); h.ckptSave(); h.ai.pendingCtl = {}; }
  eq(h.ai.ckpt.items.length, 1);
  eq([...h.engine.slots.keys()], [5]);
});

Deno.test("ckptSave keeps a pending rollback, reset or load in front of it", () => {
  for (const pend of [{ rb: 1 }, { reset: 1 }, { ld: 9 }, { rb: 0, reset: 1 }]) {
    const h = host({ fed: [1] });
    h.ai.pendingCtl = { ...pend };
    h.ckptSave();
    eq(h.ai.pendingCtl, { ...pend, sv: 1 }, JSON.stringify(pend));
  }
});

// ---------------------------------------------------------------------------------------------
// ckptResume

Deno.test("ckptResume loads the longest checkpoint only when it beats what the caches hold", () => {
  const mk = () => {
    const h = host({ fed: [1, 2, 3], ckptMax: 4 });
    h.ckptSave(); h.ai.pendingCtl = {};            // 1: [1 2 3], sent
    h.ai.fed = [1, 2, 3, 4, 5]; h.ckptSave();       // 2: [1 2 3 4 5]
    h.ai.pendingCtl = {}; h.engine.log = [];
    return h;
  };
  const cases = [
    { ids: [1, 2, 3, 4, 5, 6], reused: 0, want: 5, key: 2 },
    { ids: [1, 2, 3, 4, 5, 6], reused: 5, want: 5, key: null },       // a tie keeps the caches
    { ids: [1, 2, 3, 4, 5, 6], reused: 6, want: 6, key: null },
    { ids: [1, 2, 3, 9], reused: 0, want: 3, key: 1 },
    { ids: [1, 2, 3], reused: 0, want: 0, key: null },                 // an exact match leaves nothing to run
    { ids: [7, 1, 2, 3, 4], reused: 0, want: 0, key: null },
    { ids: [1, 2, 3, 4, 5], reused: 0, want: 3, key: 1 },
  ];
  for (const c of cases) {
    const h = mk();
    const got = h.ckptResume(c.ids, c.reused);
    eq(got, c.want, JSON.stringify(c));
    if (c.key == null) { eq(h.ai.pendingCtl, {}, "no load"); eq(h.engine.log, []); }
    else {
      eq(h.ai.pendingCtl, { ld: c.key });
      eq(h.engine.log, ["ld" + c.key]);
      eq(h.ai.pos, c.want); eq(h.ai.fed, c.ids.slice(0, c.want));
      ok(h.ai.fed !== c.ids, "fed is a copy");
    }
  }
});

Deno.test("ckptResume strips a pending reset, keeps rollback / save / drop, and replaces an older load", () => {
  const cases = [
    { pend: { reset: 1 }, want: { ld: 1 } },
    { pend: { rb: 2, sv: 3, reset: 1 }, want: { rb: 2, sv: 3, ld: 1 } },
    { pend: { dp: [DROP_ALL], reset: 1 }, want: { dp: [DROP_ALL], ld: 1 } },
    { pend: { ld: 7 }, want: { ld: 1 } },
    { pend: undefined, want: { ld: 1 } },
  ];
  for (const c of cases) {
    const h = host({ fed: [1, 2] });
    h.ckptSave();
    h.ai.pendingCtl = c.pend;
    eq(h.ckptResume([1, 2, 3], 0), 2);
    eq(h.ai.pendingCtl, c.want, JSON.stringify(c.pend));
  }
});

Deno.test("ckptResume without ?ckpt, before any save, or solo", () => {
  const h0 = host({ ckptMax: 0, fed: [1, 2] });
  eq(h0.ckptResume([1, 2, 3], 1), 1);
  const h1 = host({ fed: [1, 2] });
  eq(h1.ckptResume([1, 2, 3], 0), 0, "ai.ckpt not made yet");
  const h2 = host({ chain: [], fed: [1, 2] });
  h2.ckptSave(); h2.ai.fed = [5];
  eq(h2.ckptResume([1, 2, 3], 0), 2);
  eq(h2.ai.pendingCtl, {}, "solo: nothing to tell");
  eq(h2.engine.log, ["sv1", "ld1"]);
});

// ---------------------------------------------------------------------------------------------
// sendChain: pending control rides on exactly one frame

Deno.test("sendChain sends the pending control with the next frame, once", () => {
  const h = host({ chain: ["w0", "w1"] });
  h.ai.pendingCtl = { rb: 1, sv: 2, dp: [1], reset: 1, ld: 3 };
  h.sendChain({ t: "ai-hidden", pos: 4, x: [1] });
  h.sendChain({ t: "ai-hidden", pos: 5, x: [2] });
  eq(h.out.map((o) => o.to), ["w0", "w0"], "to the first device only");
  eq(h.out[0].msg, { t: "ai-hidden", pos: 4, x: [1], rb: 1, sv: 2, dp: [1], reset: 1, ld: 3 });
  eq(h.out[1].msg, { t: "ai-hidden", pos: 5, x: [2] });
  eq(h.ai.frames, 2);
});

// ---------------------------------------------------------------------------------------------
// workerFrame: rb > sv > dp > reset > ld, then the frame, and the control goes on down the chain

Deno.test("workerFrame applies control in the order rb, sv, dp, reset, ld, before its layers run", async () => {
  const cases = [
    { d: { rb: 1, sv: 2, dp: [3, 4], reset: 1, ld: 2 }, log: ["rb1", "sv2", "dp3", "dp4", "reset", "ld2"] },
    { d: { ld: 2, reset: 1, sv: 2 }, log: ["sv2", "reset", "ld2"] },      // key order in the message is irrelevant
    { d: { dp: DROP_ALL, sv: 5 }, log: ["sv5", "dpAll"] },                  // a scalar dp works too
    { d: { dp: [DROP_ALL] }, log: ["dpAll"] },
    { d: { rb: 0 }, log: ["rb0"] },                                         // column 0 is a rollback, not "none"
    { d: { reset: 0 }, log: [] },
    { d: {}, log: [] },
  ];
  for (const frame of ["ai-hidden", "ai-hidden-b"]) {
    for (const c of cases) {
      const w = worker();
      w.engine.st = [9, 9, 9]; w.engine.lastBase = 0;
      w.engine.slots.set(2, [7, 7]);
      const log = w.engine.log;
      // the frame's own layers log nothing; record the moment they run
      const run = w.engine.run.bind(w.engine);
      w.engine.run = (toks, base) => { log.push("run"); w.engine.st.length = base; run(toks, base); };
      const pos = w.engine.st.length;
      const msg = frame === "ai-hidden" ? { t: frame, pos, x: [1], ...c.d } : { t: frame, basePos: pos, n: 1, x: [1], ...c.d };
      await w.workerFrame(msg);
      eq(log, [...c.log, "run"], `${frame} ${JSON.stringify(c.d)}`);
    }
  }
});

Deno.test("workerFrame passes the control on to the next worker, never back to the host", async () => {
  const ctl = { rb: 1, sv: 2, dp: [3], reset: 1, ld: 2 };
  for (const frame of ["ai-hidden", "ai-hidden-b"]) {
    for (const next of ["w1", "host"]) {
      const sent = [];
      const w = worker({ next, send: (to, msg) => sent.push({ to, msg }) });
      w.engine.slots.set(2, []);
      w.engine.run = () => {};   // only the forwarding is under test here
      const msg = frame === "ai-hidden" ? { t: frame, pos: 0, x: [5], ...ctl } : { t: frame, basePos: 0, n: 1, x: [5], ...ctl };
      await w.workerFrame(msg);
      eq(sent.length, 1);
      eq(sent[0].to, next);
      const fwd = sent[0].msg;
      for (const k of Object.keys(ctl)) {
        if (next === "host") ok(!(k in fwd), `${frame}: ${k} leaked back to the host`);
        else eq(fwd[k], ctl[k], `${frame}: ${k} forwarded`);
      }
      eq(fwd.t, next === "host" ? (frame === "ai-hidden" ? "ai-hiddenret" : "ai-hiddenret-b") : frame);
    }
  }
});

Deno.test("workerFrame only forwards control it received (no stray keys)", async () => {
  const sent = [];
  const w = worker({ next: "w1", send: (to, msg) => sent.push(msg) });
  await w.workerFrame({ t: "ai-hidden", pos: 0, x: [5] });
  for (const k of ["rb", "sv", "dp", "reset", "ld"]) ok(!(k in sent[0]), k);
});

Deno.test("workerFrame with no engine yet does nothing", async () => {
  const w = worker();
  w.ai.engine = null;
  await w.workerFrame({ t: "ai-hidden", pos: 0, x: [1], reset: 1 });
});

// ---------------------------------------------------------------------------------------------
// the room as a whole: a host and 1..4 workers, token-level state on every device. After every
// frame round, every worker must hold exactly the host's state, and exactly the host's slots.

function makeRoom(nWorkers, ckptMax = 2) {
  const queue = [];
  const send = (to, msg) => queue.push({ to, msg });
  const workers = [];
  for (let i = 0; i < nWorkers; i++) workers.push(worker({ next: i + 1 < nWorkers ? "w" + (i + 1) : "host", send }));
  // every frame the host sends must fit the wire header (one sv, one ld, at most two drops)
  const wire = ({ to, msg }) => { sendFrame(fakeLink(), { ...msg, data: new Uint16Array(2) }); send(to, msg); };
  const h = host({ chain: workers.map((_, i) => "w" + i), ckptMax, out: { push: wire } });
  const pump = async () => {
    while (queue.length) {
      const { to, msg } = queue.shift();
      if (to === "host") { h.returned = msg; continue; }
      await workers[+to.slice(1)].workerFrame(msg);
    }
  };
  const room = {
    h, workers, strict: true,
    // the host runs its layers, sends one batch frame round the chain (carrying whatever control
    // is pending), and waits for it to come back
    async feed(toks) {
      if (!toks.length) return;
      const base = h.ai.pos;
      h.engine.run(toks, base);
      h.sendChain({ t: "ai-hidden-b", basePos: base, n: toks.length, x: toks });
      h.ai.pos += toks.length; h.ai.fed.push(...toks);
      await pump();
      if (room.strict) room.check("after frame at " + base);
    },
    // the last verify kept k+1 columns: the host rolls back now, the chain on the next frame
    // (what roomGenerate's onReject does)
    reject(k) {
      h.engine.restoreDN(k);
      h.ai.fed.length = h.ai.pos = h.engine.lastBase + k + 1;
      h.ai.pendingCtl = { rb: k };
    },
    // roomGenerate's checkpoint logic, around a prefill of what is new and an answer. pin: the
    // system prompt's length (Code mode): the prefill pauses there for a pinned save. stopInPin /
    // stopAfterPin: Stop during the first part, or after its save but before the rest's frame.
    // (roomGenerate pins the first part whenever all of it reached the caches, Stop or not.)
    async turn(ids, { answer = [], abort = false, reject = null, pin = 0, stopInPin = false, stopAfterPin = false } = {}) {
      let reused = h.ckptResume(ids, reusablePrefix(h.ai.fed, ids));
      if (!reused) h.resetState();
      const cut = pinSplit(reused, pin, ids.length);
      if (cut && !abort) {
        if (stopInPin) { await room.feed(ids.slice(reused, cut - 1)); h.ckptSave(); return reused; }
        await room.feed(ids.slice(reused, cut));
        h.ckptSave(true);
        if (stopAfterPin) { h.ckptSave(); return reused; }
      }
      if (!abort) {
        await room.feed(ids.slice(cut || reused));
        if (answer.length) await room.feed(answer);
        if (reject != null) room.reject(reject);
      }
      h.ckptSave();
      return reused;
    },
    check(when) {
      for (const [i, w] of workers.entries()) {
        eq(w.engine.st, h.engine.st, `${when}: worker ${i} state`);
        eq([...w.engine.slots.keys()].sort((a, b) => a - b), [...h.engine.slots.keys()].sort((a, b) => a - b), `${when}: worker ${i} slots`);
        for (const [k, v] of h.engine.slots) eq(w.engine.slots.get(k), v, `${when}: worker ${i} slot ${k}`);
      }
      eq(h.ai.ckpt ? h.ai.ckpt.items.map((x) => x.key).sort((a, b) => a - b) : [], [...h.engine.slots.keys()].sort((a, b) => a - b), `${when}: host index vs host slots`);
      ok(!h.ai.ckpt || h.ai.ckpt.pinned().length <= 1, `${when}: at most one pinned checkpoint`);
      ok(!h.ai.ckpt || h.ai.ckpt.unpinned().length <= ckptMax, `${when}: at most ckpt answer checkpoints`);
      eq(h.engine.st, h.ai.fed, `${when}: host fed`);
    },
  };
  return room;
}

const SYS = [1, 2, 3, 4];
const Q1 = [...SYS, 10, 11], A1 = [12, 13, 14];
const T1 = [...Q1, ...A1];

// Each scenario is a list of turns; the room must stay in sync after every one of them.
const SCENARIOS = [
  { name: "follow-ups extend the caches", turns: [
    { ids: Q1, answer: A1 },
    { ids: [...T1, 20, 21], answer: [22], wantReused: T1.length },
  ] },
  { name: "regenerate resumes from a checkpoint", turns: [
    { ids: Q1, answer: A1 },
    { ids: [...T1, 20, 21], answer: [22, 23] },
    { ids: [...T1, 30], answer: [31], wantReused: T1.length },    // branch off after turn 1
  ] },
  { name: "a new chat resets everywhere", turns: [
    { ids: Q1, answer: A1 },
    { ids: [5, 6, 7], answer: [8], wantReused: 0 },
  ] },
  { name: "a rolled-back answer is saved rolled back on every device", turns: [
    { ids: Q1, answer: A1, reject: 1 },                          // keeps 12, 13
    { ids: [...Q1, 12, 13, 40], answer: [41], wantReused: Q1.length + 2 },
  ] },
  { name: "a rollback, then a reset, then a resume of the rolled-back save", turns: [
    { ids: Q1, answer: A1, reject: 0 },                          // keeps 12
    { ids: [9, 9], answer: [9] },                                // reset: rb and sv must still go first
    { ids: [...Q1, 12, 50], answer: [51], wantReused: Q1.length + 1 },
  ] },
  { name: "Stop before the first frame, then a follow-up", turns: [
    { ids: Q1, answer: A1 },
    { ids: [...T1, 20], abort: true },
    { ids: [...T1, 20], answer: [21] },
  ] },
  { name: "Stop before the first frame, then a branch (second save before any frame)", turns: [
    { ids: Q1, answer: A1 },                                     // save 1, pending
    { ids: [...T1, 20, 21], abort: true },                       // save 2 of the same state, pending again
    { ids: [...T1, 30, 31], answer: [32] },                      // frame goes out
    { ids: [...T1, 40, 41], answer: [42], wantReused: T1.length },  // resume from T1
  ] },
  { name: "Stop twice in a row with eviction (ckpt=1)", ckptMax: 1, turns: [
    { ids: Q1, answer: A1 },
    { ids: [...T1, 20], answer: [21] },
    { ids: [...T1, 20, 21, 22], abort: true },
    { ids: [...T1, 20, 21, 22], abort: true },
    { ids: [...T1, 20, 21, 22], answer: [23] },
  ] },
  { name: "a device rejoins (ckptClear(true)) while a save is pending", turns: [
    { ids: Q1, answer: A1 },
    { rejoin: true },
    { ids: [...T1, 20], answer: [21], wantReused: 0 },
    { ids: [...T1, 30], answer: [31], wantReused: 0 },          // the old saves are gone everywhere
  ] },
  { name: "a device rejoins, then Stop before the first frame, then a turn", turns: [
    { ids: Q1, answer: A1 },
    { ids: [...T1, 20], answer: [21] },
    { rejoin: true },
    { ids: [...T1, 30], abort: true },
    { ids: [...T1, 40], answer: [41], wantReused: 0 },
    { ids: [...T1, 40, 41, 50], answer: [51], wantReused: T1.length + 2 },
  ] },
  { name: "Stop again and again (ckpt=2): drops never pile up past the header's two", turns: [
    { ids: Q1, answer: A1 },
    { ids: [...T1, 20], answer: [21] },
    { ids: [...T1, 20, 21, 30], answer: [31] },
    { ids: [...T1, 20, 21, 30, 31, 40], abort: true },
    { ids: [...T1, 20, 21, 30, 31, 40], abort: true },
    { ids: [...T1, 20, 21, 30, 31, 40], abort: true },
    { ids: [...T1, 20, 21, 60], answer: [61], wantReused: T1.length + 2 },   // resume from turn 2's save
  ] },
  // the pinned system prompt (Code mode, issue #73): SYS is the system prompt + tools
  { name: "pin: a compacted prompt resumes at the system prompt", turns: [
    { ids: Q1, answer: A1, pin: SYS.length },
    { ids: [...T1, 20], answer: [21], pin: SYS.length, wantReused: T1.length },
    { ids: [...SYS, 50, 51], answer: [52], pin: SYS.length, wantReused: SYS.length },   // middle rewritten
    { ids: [...SYS, 50, 51, 52, 53], answer: [54], pin: SYS.length, wantReused: SYS.length + 3 },
  ] },
  { name: "pin: answer saves never evict it (ckpt=1)", ckptMax: 1, turns: [
    { ids: Q1, answer: A1, pin: SYS.length },
    { ids: [...T1, 20], answer: [21], pin: SYS.length },
    { ids: [...T1, 20, 21, 22], answer: [23], pin: SYS.length },
    { ids: [...T1, 20, 21, 22, 23, 24], answer: [25], pin: SYS.length },
    { ids: [...SYS, 60], answer: [61], pin: SYS.length, wantReused: SYS.length },
  ] },
  { name: "pin: another system prompt replaces it on every device", turns: [
    { ids: Q1, answer: A1, pin: SYS.length },
    { ids: [5, 6, 7, 8, 9, 10], answer: [11], pin: 4, wantReused: 0 },
    { ids: [5, 6, 7, 8, 70], answer: [71], pin: 4, wantReused: 4 },
    { ids: [...SYS, 72], answer: [73], pin: SYS.length, wantReused: 0 },   // the old one is gone
  ] },
  { name: "pin: Stop during the system prompt, then again", turns: [
    { ids: Q1, pin: SYS.length, stopInPin: true },
    { ids: Q1, answer: A1, pin: SYS.length, wantReused: SYS.length - 1 },
    { ids: [...SYS, 80], answer: [81], pin: SYS.length, wantReused: SYS.length },
  ] },
  { name: "pin: Stop after the pinned save, before the rest's frame (the end save supersedes it)", turns: [
    { ids: Q1, answer: A1, pin: SYS.length },
    { ids: [9, 9, 9, 9, 1], pin: 4, stopAfterPin: true },
    { ids: [9, 9, 9, 9, 2], answer: [3], pin: 4, wantReused: 4 },
    { ids: [9, 9, 9, 9, 4], answer: [5], pin: 4, wantReused: 4 },
  ] },
  // the save that supersedes the pinned one holds the same tokens, so it stays pinned: answer
  // saves then never evict it, and a later rewrite of the middle still resumes there
  { name: "pin: the save that supersedes a pinned one stays pinned", turns: [
    { ids: [9, 9, 9, 9, 1], pin: 4, stopAfterPin: true },
    { ids: [9, 9, 9, 9, 2], answer: [3], pin: 4, wantReused: 4 },
    { ids: [9, 9, 9, 9, 2, 3, 5], answer: [6], pin: 4 },
    { ids: [9, 9, 9, 9, 2, 3, 5, 6, 7], answer: [8], pin: 4 },
    { ids: [9, 9, 9, 9, 4], answer: [5], pin: 4, wantReused: 4 },
  ] },
  { name: "Stop after a reset, before the first frame", turns: [
    { ids: Q1, answer: A1 },
    { ids: [7, 7, 7], abort: true },                             // reset pending, nothing fed: no save
    { ids: [...T1, 20], answer: [21], wantReused: T1.length },   // resume replaces the reset
  ] },
];

for (const n of [1, 2, 4]) {
  for (const sc of SCENARIOS) {
    Deno.test(`room (${n + 1} devices): ${sc.name}`, async () => {
      const room = makeRoom(n, sc.ckptMax ?? 2);
      for (const [i, t] of sc.turns.entries()) {
        if (t.rejoin) { room.h.ai.fed = null; room.h.ckptClear(true); continue; }
        const reused = await room.turn(t.ids, t);
        if (t.wantReused != null) eq(reused, t.wantReused, `turn ${i} reused`);
      }
      // one more frame delivers whatever is still pending, then everything must agree
      await room.feed([99]);
      room.check("end");
    });
  }
}

// The exact failure the second scenario above guards against, spelled out: the host must never
// ask a worker to load a slot the worker was never told to save.
Deno.test("a save superseded before its frame went out is never loaded later", async () => {
  const room = makeRoom(1, 2);
  room.strict = false;   // let it run on to the load, the way the room would
  await room.turn(Q1, { answer: A1 });
  await room.turn([...T1, 20, 21], { abort: true });
  await room.turn([...T1, 30, 31], { answer: [32] });
  // every key the host may resume from is a slot the worker has
  for (const x of room.h.ai.ckpt.items) {
    if (x.key === room.h.ai.pendingCtl.sv) continue;   // on its way with the next frame
    ok(room.workers[0].engine.slots.has(x.key), `host indexes slot ${x.key} the worker never saved`);
  }
  await room.turn([...T1, 40, 41], { answer: [42] });   // would throw "no saved slot" on the worker
});
