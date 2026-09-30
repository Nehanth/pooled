// No GPU: the room node's checkpoints (ckpt.js and the host side in roomnode.js) with a fake engine
// whose next token depends on everything in its caches, so a wrong resume changes the answer.
//   node --test packages/room-node/test/*_test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { RoomNode, nodeCtxFor } from "../roomnode.js";
import { CkptIndex, cacheBoundary, boundaryPin, pinPoints, cutPoints, commonPrefix, CACHE_MARKS } from "../ckpt.js";
import { DROP_ALL } from "../../../room/transport.js";
import { renderApi, templateProfile } from "../../../room/conversation.js";

// ---- a fake engine: token history as its "caches", slots copy it ----
class FakeEngine {
  constructor() { this.hist = []; this.pos = 0; this.maxSeq = 4096; this.slots = new Map(); this.dims = { dim: 4 }; this.prefilled = 0; this.log = []; }
  reset() { this.hist = []; this.pos = 0; }
  _put(id, pos) { this.hist.length = pos; this.hist.push(id); this.pos = pos + 1; this.prefilled++; }
  async prefillTokens(ids) { for (const id of ids) this._put(id, this.pos); }
  async prefillToken(id) { this._put(id, this.pos); }
  async embedRun(id, pos) { this._put(id, pos); return new Float32Array(4); }
  // the next token: a hash of the whole history (never 0, the stop token, before 6 tokens of answer)
  async headFromHidden() {
    let h = 7;
    for (const t of this.hist.slice(0, this.pos)) h = (Math.imul(h, 31) + t + 1) | 0;
    const lg = new Float32Array(50); lg[1 + ((h >>> 0) % 49)] = 1; return lg;
  }
  saveSlot(k) { this.log.push("sv" + k); this.slots.set(k, { hist: this.hist.slice(0, this.pos), pos: this.pos }); }
  loadSlot(k) { const s = this.slots.get(k); if (!s) throw new Error("no saved slot " + k); this.log.push("ld" + k); this.hist = s.hist.slice(); this.pos = s.pos; }
  dropSlot(k) { this.slots.delete(k); }
  dropAllSlots() { this.slots.clear(); }
}
const argmax = (lg) => { let b = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[b]) b = i; return b; };
function soloHost(ckpt = {}) {
  const n = new RoomNode({ name: "host", pledgeGB: 8, log: () => {}, ckpt });
  n.isHost = true; n.code = "TEST"; n.peer = { id: "pooled-room-TEST" };
  Object.assign(n.ai, { engine: new FakeEngine(), online: true, model: "qwen3.6-35b-moe", role: "host", device: { queue: { onSubmittedWorkDone: async () => {} } } });
  return n;
}
const range = (a, n) => Array.from({ length: n }, (_, i) => a + i);
const gen = (n, ids, pins = []) => n.generateOnce(ids, { stop: new Set([0]), maxNew: 6, sample: argmax, spec: false, pins });

test("CkptIndex: pinned prefixes are never evicted by answers; each kind goes by last use", () => {
  const ix = new CkptIndex({ answers: 2, pins: 2 });
  const save = (ids, pin = false) => { const p = ix.plan(ids, { pin }); if (p.skip) return p; for (const k of p.drop) ix.remove(k); ix.commit(p.key, ids, pin); return p; };
  const P = range(100, 40);
  save(P, true);
  const a1 = save([...P, 1, 2]), a2 = save([...P, 3, 4]);
  const a3 = save([...P, 5, 6]);
  assert.deepEqual(a3.drop, [a1.key], "the oldest answer goes, the pinned one stays");
  assert.ok(ix.items.some((x) => x.pin && x.ids.length === 40));
  // a resume touches what it used: a2 is now newer than a3, so a3 goes next
  assert.equal(ix.best([...P, 3, 4, 9]).key, a2.key);
  assert.deepEqual(save([...P, 7, 8]).drop, [a3.key]);
  // the same tokens again: kept, not saved twice; an answer with a pin's tokens becomes pinned
  assert.equal(save([...P, 7, 8]).skip, true);
  const Q = range(200, 30);
  const q = save(Q);
  const qp = save(Q, true);
  assert.ok(qp.drop.includes(q.key) && ix.find(qp.key).pin);
  // pins: least recently used out first
  save(range(300, 30), true);
  assert.ok(!ix.items.some((x) => x.pin && x.ids[0] === 100), "the oldest pin went");
  assert.equal(ix.best([...Q, 1]).key, qp.key);
  assert.deepEqual(ix.hits, { pin: 1, answer: 1, miss: 0 });
  // slot numbers wrap in 1..65534
  ix.n = 65534; assert.equal(ix.nextKey(), 1);
});

test("cut and pin points: past what the caches hold, before the last prompt token, long enough", () => {
  assert.deepEqual(cutPoints(0, [50, 20, 50], 100), [20, 50]);
  assert.deepEqual(cutPoints(20, [20, 50], 100), [50]);
  assert.deepEqual(cutPoints(0, [100], 100), []);
  assert.deepEqual(pinPoints({ ids: range(0, 3000), systemLen: 2500 }, { boundary: 1200, minPin: 1024 }), [1200, 2500]);
  assert.deepEqual(pinPoints({ ids: range(0, 3000), systemLen: 900 }, { boundary: 0, minPin: 1024 }), []);
  assert.equal(commonPrefix([1, 2, 3], [1, 2, 4]), 2);
});

test("cacheBoundary / boundaryPin: OpenClaw's STABLE mark, as a prefix of the real prompt's ids (tools first in the xml style)", () => {
  const stable = "You are a helpful agent. ".repeat(60);
  const system = `<!-- openclaw:attempt:STABLE -->\n${stable}\n${CACHE_MARKS[0]}\n<!-- openclaw:attempt:DYNAMIC -->\nCurrent date: 2026-09-29\n<!-- /openclaw:attempt:DYNAMIC -->`;
  assert.equal(cacheBoundary(system), system.indexOf(CACHE_MARKS[0]) + CACHE_MARKS[0].length);
  assert.equal(cacheBoundary("no marks"), 0);
  assert.equal(cacheBoundary("x" + CACHE_MARKS[0]), 0, "a mark at the very end is the whole system prompt");
  // a character tokenizer with the template's special tokens
  const vocab = {}; let nx = 0;
  for (const t of ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]) vocab[t] = nx++;
  const tok = { vocab, encode: (s) => [...s].map((c) => 1000 + c.codePointAt(0)), decode: (ids) => ids.map((i) => String.fromCodePoint(i - 1000)).join("") };
  const tools = [{ name: "read", description: "Read a file", parameters: { type: "object", properties: { path: { type: "string" } } } }];
  for (const style of ["json", "xml"]) {
    const profile = { ...templateProfile("", tok), style };
    const req = { system, tools, messages: [{ role: "user", text: "hi" }], params: {} };
    const r = renderApi(tok, req, profile, {});
    const prompt = { ids: r.ids, systemLen: r.systemLen, profile, thinking: false };
    const b = boundaryPin(tok, req, prompt, { minPin: 100 });
    assert.ok(b > 100 && b < r.systemLen, `${style}: ${b} inside the system turn (${r.systemLen})`);
    const text = tok.decode(r.ids.slice(0, b).filter((i) => i >= 1000));
    assert.ok(text.endsWith(CACHE_MARKS[0].slice(0, -1)) || text.endsWith(CACHE_MARKS[0]) || text.endsWith(CACHE_MARKS[0] + "\n"), `${style}: ends at the mark: …${JSON.stringify(text.slice(-40))}`);
    assert.equal(text.includes("# Tools"), style === "xml", `${style}: the tools come ${style === "xml" ? "before" : "after"} the system text`);
    // another day: the dynamic part changed, the boundary still is a prefix
    const r2 = renderApi(tok, { ...req, system: system.replace("2026-09-29", "2026-09-30") }, profile, {});
    assert.ok(commonPrefix(r2.ids, r.ids) >= b);
  }
});

test("host: a pinned system prompt, answer checkpoints and other sessions give the same answers as prefilling everything", async () => {
  const S = range(100, 40), T = range(200, 10);   // system prompt (pinned at 40) and a title request's
  const run = async (n) => {
    const out = [];
    const a1 = await gen(n, [...S, 1, 2, 3], [40]);
    out.push(a1);
    out.push(await gen(n, [...T, 4, 5]));                                   // a side request
    out.push(await gen(n, [...S, 1, 2, 3, ...a1.tokens, 0, 7, 8], [40]));   // the conversation's next turn
    out.push(await gen(n, [...S, 9, 9], [40]));                             // a new session, same system prompt
    return out;
  };
  const plain = await run(soloHost(false));
  const n = soloHost({ answers: 3, pins: 2 });
  const cached = await run(n);
  assert.deepEqual(cached.map((r) => r.tokens), plain.map((r) => r.tokens), "the same answers");
  assert.deepEqual(cached.map((r) => r.from), [null, null, "answer", "pin"]);
  assert.deepEqual(cached.map((r) => r.pinned), [1, 0, 0, 0]);
  assert.equal(cached[2].prefilled, 3, "only the new turn is read");
  assert.equal(cached[3].prefilled, 2, "only what follows the system prompt");
  assert.deepEqual(plain.map((r) => r.prefilled), [43, 12, 43 + a1len(plain) + 3, 42]);
  const st = n.status().ckpt;
  assert.deepEqual(st.pinned, [40]);
  assert.equal(st.answers.length, 3);
  // many side requests later, the conversation's last answer is still there (it was used last)
  for (let i = 0; i < 2; i++) await gen(n, [...T, 50 + i]);
  const next = await gen(n, [...S, 1, 2, 3, ...cached[0].tokens, 0, 7, 8, ...cached[2].tokens, 0, 5]);
  assert.equal(next.from, "answer");
});
const a1len = (rs) => rs[0].tokens.length;

test("host: no checkpoints with ckpt: false, or for a dense model when a chain device does not apply them", () => {
  const n = soloHost(false);
  assert.equal(n.ckptOn(), false);
  const d = soloHost();
  d.ai.model = "qwen3-1.7b"; d.ai.chain = ["tab"];
  assert.equal(d.ckptOn(), false, "a tab from before the dense engine had slots");
  d.ai.ckptCap.set("tab", true);
  assert.equal(d.ckptOn(), true);
  const q = soloHost(); q.ai.chain = ["tab"];
  assert.equal(q.ckptOn(), true, "every qwen35 engine applies checkpoint control");
});

test("host with a chain: saves and loads ride the next frame, drops go out two per frame, a failure drops them all", () => {
  const n = soloHost({ answers: 1, pins: 1 });
  const frames = [];
  n.ai.chain = ["w"];
  n.sendHidden = (id, m) => frames.push(m);
  const E = n.ai.engine;
  const feed = (ids) => { E.hist = ids.slice(); E.pos = ids.length; n.ai.fed = ids.slice(); n.ai.pos = ids.length; };
  feed(range(1, 20));
  const k1 = n.ckptSave(true);
  assert.equal(n.ai.pendingCtl.sv, k1);
  n.sendChain({ t: "ai-hidden", pos: 20 });
  assert.equal(frames.at(-1).sv, k1);
  // two answers with room for one: the first is evicted, its drop rides a later frame
  feed(range(1, 25)); const k2 = n.ckptSave(); n.sendChain({ t: "ai-hidden", pos: 25 });
  feed(range(1, 30)); const k3 = n.ckptSave();
  n.resetState();   // a reset keeps the pending save (it records the last answer) and the drops
  assert.equal(n.ai.pendingCtl.sv, k3); assert.equal(n.ai.pendingCtl.reset, 1);
  n.sendChain({ t: "ai-hidden", pos: 0 });
  assert.deepEqual(frames.at(-1).dp, [k2]); assert.equal(frames.at(-1).sv, k3);
  // resume from the pinned one: ld rides the next frame, no reset with it
  const r = n.ckptResume([...range(1, 20), 99], 0);
  assert.deepEqual(r, { reused: 20, from: "pin" });
  assert.equal(n.ai.pendingCtl.ld, k1); assert.equal(n.ai.pendingCtl.reset, undefined);
  n.sendChain({ t: "ai-hidden", pos: 20 });
  // many evictions at once still go two per frame
  n.ai.dropQ.push(101, 102, 103);
  n.sendChain({ t: "ai-hidden", pos: 21 }); n.sendChain({ t: "ai-hidden", pos: 22 });
  assert.deepEqual([frames.at(-2).dp, frames.at(-1).dp], [[101, 102], [103]]);
  // a device dropped: everything goes, here and on the chain
  n.ckptClear(true);
  assert.equal(n.ai.ckpt.size, 0); assert.deepEqual(n.ai.pendingCtl.dp, [DROP_ALL]);
  feed(range(1, 20));
  assert.equal(n.ckptSave(), null, "no save may ride with DROP_ALL");
});

test("nodeCtxFor: 64k for the MoE, 32k for the 27B, the room default for the dense 1.7B; an ask wins", () => {
  assert.equal(nodeCtxFor("qwen3.6-35b-moe"), 65536);
  assert.equal(nodeCtxFor("qwen3.8-27b"), 32768);
  assert.equal(nodeCtxFor("qwen3-1.7b"), 8192);
  assert.equal(nodeCtxFor("qwen3.6-35b-moe", 16384), 16384);
  assert.equal(nodeCtxFor("qwen3-1.7b", 16384), 16384);
});
